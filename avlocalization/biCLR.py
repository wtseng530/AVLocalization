import math
from argparse import ArgumentParser

import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.models.self_supervised import SimCLR
from pl_bolts.models.self_supervised.resnets import resnet18, resnet50


class biCLR(SimCLR):
    def __init__(
            self,
            gpus: int,
            num_samples: int,
            batch_size: int,
            num_nodes: int = 1,
            arch: str = 'resnet50',
            mode: str = 'dsm',
            hidden_mlp: int = 2048,
            feat_dim: int = 128,
            warmup_epochs: int = 10,
            max_epochs: int = 100,
            temperature: float = 0.5,
            first_conv: bool = True,
            maxpool1: bool = True,
            optimizer: str = 'adam',
            lars_wrapper: bool = True,
            exclude_bn_bias: bool = False,
            start_lr: float = 0.,
            learning_rate: float = 1e-3,
            final_lr: float = 0.,
            weight_decay: float = 1e-6,
            **kwargs):

        super().__init__(
            gpus,
            num_samples,
            batch_size,
            mode,
            num_nodes,
            arch,
            hidden_mlp,
            feat_dim,
            warmup_epochs,
            max_epochs,
            temperature,
            first_conv,
            maxpool1,
            optimizer,
            lars_wrapper,
            exclude_bn_bias,
            start_lr,
            learning_rate,
            final_lr,
            weight_decay)

        del self.encoder
        self.encoder1, self.encoder2 = self.init_model()

    def init_model(self):
        if self.dataset == 'vxl' and self.arch == 'resnet18':
            backbone = resnet18
            return backbone(first_conv=self.first_conv, maxpool1=self.maxpool1, return_all_feature_maps=False), \
                ResNet(BasicBlock, [1, 1, 1, 1], spatial_size=32, sample_duration=5, num_classes=self.hidden_mlp)
        elif self.dataset == 'dsm' and self.arch == 'resnet18':
            backbone = resnet18
        elif self.dataset == 'dsm' and self.arch == 'resnet50':
            backbone = resnet50

        return backbone(first_conv=self.first_conv, maxpool1=self.maxpool1, return_all_feature_maps=False), \
            backbone(first_conv=self.first_conv, maxpool1=self.maxpool1, return_all_feature_maps=False)

    def forward(self, x1, x2):
        return (self.encoder1(x1)[-1], self.encoder2(x2)[-1])

    def shared_step(self, batch):
        rgbb, dptb = batch
        h1, h2 = self(rgbb, dptb)
        z1 = self.projection(h1)
        z2 = self.projection(h2)
        loss = self.nt_xent_loss(z1, z2, self.temperature)

        return loss

    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # model params
        parser.add_argument("--arch", default="resnet18", type=str, help="convnet architecture")
        # specify flags to store false
        parser.add_argument("--first_conv", action='store_false')
        parser.add_argument("--maxpool1", action='store_false')
        parser.add_argument("--hidden_mlp", default=512, type=int, help="hidden layer dimension in projection head")
        parser.add_argument("--feat_dim", default=128, type=int, help="feature dimension")
        parser.add_argument("--online_ft", action='store_true')
        parser.add_argument("--fp32", action='store_true')

        # # transform params
        parser.add_argument("--rgb_dir", type=str, default="../data/train/rgb", help='path to rgb image')
        parser.add_argument("--depth_dir", type=str, default="../data/train/dsm", help='path to depth image')
        parser.add_argument("--patch_dim", type=int, default=32, help='image patch size')
        parser.add_argument('--mode', type=str, default='dsm', choices=['dsm', 'vxl'],
                            help='branch for local patch training')
        parser.add_argument("--res", type=int, default=5, help='resolution of aerial image and dsm')
        parser.add_argument("--val_split", type=float, default='0.3', help='test and validation data percentage')

        # training params
        parser.add_argument("--fast_dev_run", default=1, type=int)
        parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes for training")
        parser.add_argument("--gpus", default=1, type=int, help="number of gpus to train on")
        parser.add_argument("--num_workers", default=1, type=int, help="num of workers per GPU")
        parser.add_argument("--optimizer", default="adam", type=str, help="choose between adam/sgd")
        parser.add_argument("--lars_wrapper", action='store_true', help="apple lars wrapper over optimizer used")
        parser.add_argument('--exclude_bn_bias', action='store_true', help="exclude bn/bias from weight decay")
        parser.add_argument("--max_epochs", default=1000, type=int, help="number of total epochs to run")
        parser.add_argument("--max_steps", default=-1, type=int, help="max steps")
        parser.add_argument("--warmup_epochs", default=10, type=int, help="number of warmup epochs")
        parser.add_argument("--batch_size", default=512, type=int, help="batch size per gpu")

        parser.add_argument("--temperature", default=0.1, type=float, help="temperature parameter in training loss")
        parser.add_argument("--weight_decay", default=1e-6, type=float, help="weight decay")
        parser.add_argument("--learning_rate", default=1e-3, type=float, help="base learning rate")
        parser.add_argument("--start_lr", default=0, type=float, help="initial warmup learning rate")
        parser.add_argument("--final_lr", type=float, default=1e-6, help="final learning rate")

        return parser


class ResNet(nn.Module):
    """
    source: https://github.com/okankop/Efficient-3DCNNs/blob/master/models/c3d.py
    """

    def __init__(self,
                 block,
                 layers,
                 spatial_size,
                 sample_duration,
                 shortcut_type='B',
                 num_classes=512):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 64, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 64, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(
            block, 128, layers[3], shortcut_type, stride=2)
        last_duration = int(math.ceil(sample_duration / 16))
        last_size = int(math.ceil(spatial_size / 32))
        self.avgpool = nn.AvgPool3d(
            (last_duration, last_size, last_size), stride=1)
        self.fc = nn.Linear(128 * last_size * last_size * 8, 512)

        self.fc2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc3 = nn.Sequential(
            nn.Linear(1024, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        # x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return [x]


class BasicBlock(nn.Module):
    """
    source: https://github.com/okankop/Efficient-3DCNNs/blob/master/models/c3d.py
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out
