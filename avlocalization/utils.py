from pytorch_lightning.metrics import Accuracy
from typing import Optional, Sequence, Tuple, Union,Any

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import device, Tensor
from torch.nn import functional as F


class OnlineEvaluator(Callback):
    def __init__(self,):
        super().__init__()

    def get_representations(self, pl_module:LightningModule, x1: Tensor ,x2: Tensor) ->Tensor:
        with torch.no_grad():
            pres1, pres2 = pl_module(x1,x2)
            feature_1, feature_2 = pl_module.projection(pres1), pl_module.projection(pres2)
            feature_1, feature_2 = F.normalize(feature_1, dim=1), F.normalize(feature_2, dim=1)
            features = torch.cat([feature_1, feature_2], dim=0)
        return features

    def make_prediction(self,
        pl_module: LightningModule,
        batch: Sequence,
      ):
        # generate the label
        x1, x2 = batch
        x1,x2 = x1.to(pl_module.device),x2.to(pl_module.device)

        #TODO check the distributed version##
        features = self.get_representations(pl_module,x1,x2)
        features.detach()

        similarity_matrix = torch.mm(features, features.T)
        torch.diagonal(similarity_matrix).fill_(0)

        return similarity_matrix

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int
    ) -> None:
        with torch.no_grad():
            pred = self.make_prediction(pl_module, batch)
            pred = F.softmax(pred, dim=1).to(pl_module.device)
        bs,_,_,_, = batch[0].shape
        label = torch.tensor(list(range(bs, bs + bs)) + list(range(0, 0 + bs))).to(pl_module.device)

        accuracy = Accuracy().to(pl_module.device)
        accuracy_top5 = Accuracy(top_k=5).to(pl_module.device)

        train_acc = accuracy(pred,label)
        train_acc_top5 = accuracy_top5(pred,label)

        pl_module.log('online_train_acc',train_acc, on_step=True, on_epoch=False)
        pl_module.log('online_train_acc_top5',train_acc_top5, on_step=True, on_epoch=False)

    def on_validation_batch_end(
        self, trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        with torch.no_grad():
            pred = self.make_prediction(pl_module, batch)
            pred = F.softmax(pred, dim=1).to(pl_module.device)
        bs,_,_,_, = batch[0].shape
        label = torch.tensor(list(range(bs, bs + bs)) + list(range(0, 0 + bs))).to(pl_module.device)

        accuracy = Accuracy().to(pl_module.device)
        accuracy_top5 = Accuracy(top_k=5).to(pl_module.device)

        val_acc = accuracy(pred, label)
        val_acc_top5 = accuracy_top5(pred, label)

        pl_module.log('online_val_acc', val_acc, on_step=False, on_epoch=True,sync_dist=True)
        pl_module.log('online_val_acc_top5', val_acc_top5, on_step=False, on_epoch=True,sync_dist=True)
