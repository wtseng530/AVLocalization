from argparse import ArgumentParser
from pl_bolts.models.self_supervised import SimCLR

class biCLR(SimCLR):
  def __init__(
        self,
        gpus: int,
        num_samples: int,
        batch_size: int,
        num_nodes: int = 1,
        arch: str = 'resnet50',
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
        'mydfcdataset',
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

    self.encoder1 = self.init_model()
    self.encoder2 = self.init_model()


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
      parser.add_argument("--rgb_dir", type= str, default="../data/rgb_5cm.tif", help='path to rgb image')
      parser.add_argument("--depth_dir", type= str, default="../data/depthmap_5cm.tif", help='path to depth image')
      parser.add_argument("--patch_dim", type=int, default=32, help= 'image patch size')
      parser.add_argument("--res", type=int, default=5, help='resolution of aerial image and dsm')
      parser.add_argument("--val_split", type = float, default='0.3', help='test and validation data percentage')

      # training params
      parser.add_argument("--fast_dev_run", default=1, type=int)
      parser.add_argument("--num_nodes", default=1, type=int, help="number of nodes for training")
      parser.add_argument("--gpus", default=1, type=int, help="number of gpus to train on")
      parser.add_argument("--num_workers", default=8, type=int, help="num of workers per GPU")
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