from pytorch_lightning.metrics import Accuracy
from typing import Optional, Sequence, Tuple, Union,Any

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import device, Tensor
from torch.nn import functional as F


class OnlineEvaluator(Callback):
    def __init__(self,):
        super().__init__()

    def get_accuracy(self,
                pl_module: LightningModule,
                batch: Sequence,
                ) ->None:
        with torch.no_grad():
            x1, x2 = batch
            pres = pl_module(x1.to(pl_module.device), x2.to(pl_module.device))
            feature_1, feature_2 = map(lambda x: F.normalize(pl_module.projection(x), dim=1),pres)
            features = torch.cat([feature_1, feature_2], dim=0)
            features.detach()

        similarity_matrix = torch.mm(features, features.T)
        torch.diagonal(similarity_matrix).fill_(0)

        similarity_matrix = F.softmax(similarity_matrix, dim=1).to(pl_module.device)
        bs, _, _, _, = batch[0].shape
        label = torch.tensor(list(range(bs, bs + bs)) + list(range(0, 0 + bs))).to(pl_module.device)

        accuracy = Accuracy().to(pl_module.device)
        accuracy_top5 = Accuracy(top_k=5).to(pl_module.device)

        train_acc = accuracy(similarity_matrix, label)
        train_acc_top5 = accuracy_top5(similarity_matrix, label)

        return train_acc, train_acc_top5

    def on_train_batch_end(
        self,
        trainer: Trainer, pl_module: LightningModule, outputs: Sequence, batch: Sequence, batch_idx: int, dataloader_idx: int
    ) -> None:

        acc, acc5 = self.get_accuracy(pl_module, batch)
        pl_module.log('online_train_acc',acc, on_step=True, on_epoch=False)
        pl_module.log('online_train_acc_top5',acc5, on_step=True, on_epoch=False)

    def on_validation_batch_end(
        self, trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:

        acc, acc5 = self.get_accuracy(pl_module, batch)
        pl_module.log('online_val_acc', acc, on_step=False, on_epoch=True,sync_dist=True)
        pl_module.log('online_val_acc_top5', acc5, on_step=False, on_epoch=True,sync_dist=True)
