import torch
from torch import nn as nn
from pl_bolts.models.self_supervised.resnets import BasicBlock, Bottleneck, ResNet
#https://github.com/okankop/Efficient-3DCNNs/blob/master/models/c3d.py
import math
import torch.nn as nn

class C3D(nn.Module):
    def __init__(self,
                 sample_size,
                 sample_duration,
                 num_classes=600):
        super(C3D, self).__init__()
        self.group1 = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(1, 2, 2)))
        self.group2 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        # self.group3 = nn.Sequential(
        #     nn.Conv3d(128, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(256),
        #     nn.ReLU(),
        #     nn.Conv3d(256, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(256),
        #     nn.ReLU(),
        #     nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        # self.group4 = nn.Sequential(
        #     nn.Conv3d(256, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(512),
        #     nn.ReLU(),
        #     nn.Conv3d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm3d(512),
        #     nn.ReLU(),
        #     nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)))
        self.group5 = nn.Sequential(
            nn.Conv3d(128, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1)))

        last_duration = int(math.floor(sample_duration / 16))
        last_size = int(math.ceil(sample_size / 32))
        self.fc1 = nn.Sequential(
            #nn.Linear((512 * last_duration * last_size * last_size), 4096),
            nn.Linear((512*8*5*3),4096),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.fc = nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.group1(x)
        out = self.group2(out)
        # out = self.group3(out)
        # out = self.group4(out)
        out = self.group5(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc(out)
        return out


def get_fine_tuning_parameters(model, ft_portion):
    if ft_portion == "complete":
        return model.parameters()

    elif ft_portion == "last_layer":
        ft_module_names = []
        ft_module_names.append('fc')

        parameters = []
        for k, v in model.named_parameters():
            for ft_module in ft_module_names:
                if ft_module in k:
                    parameters.append({'params': v})
                    break
            else:
                parameters.append({'params': v, 'lr': 0.0})
        return parameters

    else:
        raise ValueError("Unsupported ft_portion: 'complete' or 'last_layer' expected")


def get_model(**kwargs):
    """
    Returns the model.
    """
    model = C3D(**kwargs)
    return model


