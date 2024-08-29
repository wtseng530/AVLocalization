import random

from matplotlib import patches as patches, pyplot as plt, gridspec as gridspec
from pytorch_lightning.metrics import Accuracy
from typing import Optional, Sequence, Tuple, Union,Any
import rasterio
from rasterio.enums import Resampling
import numpy as np
from pyntcloud import PyntCloud

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torch import device, Tensor
from torch.nn import functional as F


class OnlineEvaluator(Callback):
    def __init__(self,):
        super().__init__()

    def get_acc(self,
                pl_module: LightningModule,
                batch: Sequence,
                ):
        with torch.no_grad():
            x1, x2 = batch
            x1, x2 = x1.to(pl_module.device), x2.to(pl_module.device)

            pres1, pres2 = pl_module(x1,x2)
            feature_1, feature_2 = pl_module.projection(pres1), pl_module.projection(pres2)
            feature_1, feature_2 = F.normalize(feature_1, dim=1), F.normalize(feature_2, dim=1)
            features = torch.cat([feature_1, feature_2], dim=0)
            features.detach()

        similarity_matrix = torch.mm(features, features.T)
        torch.diagonal(similarity_matrix).fill_(0)

        pred = F.softmax(similarity_matrix, dim=1).to(pl_module.device)
        bs= batch[0].shape[0]
        label = torch.tensor(list(range(bs, bs + bs)) + list(range(0, 0 + bs))).to(pl_module.device)

        accuracy = Accuracy().to(pl_module.device)
        accuracy_top5 = Accuracy(top_k=5).to(pl_module.device)
        return accuracy(pred,label), accuracy_top5(pred,label)

    # def get_acc(self,
    #             pl_module: LightningModule,
    #             batch: Sequence,
    #             ):
    #     with torch.no_grad():
    #         x1, x2 = batch
    #         x1, x2 = x1.to(pl_module.device), x2.to(pl_module.device)
    #
    #         pres1, pres2 = pl_module(x1,x2)
    #         feature_1, feature_2 = pl_module.projection(pres1), pl_module.projection(pres2)
    #         feature_1, feature_2 = F.normalize(feature_1, dim=1), F.normalize(feature_2, dim=1)
    #         feature_1.detach(), feature_2.detach()
    #
    #     similarity_matrix = torch.mm(feature_2, feature_1.T)
    #     pred, pred_5 = torch.argmax(similarity_matrix, dim=1), torch.topk(similarity_matrix, k=5, dim=1, ).indices
    #     error, error_5 = [label for label, predict in enumerate(pred) if predict != label], \
    #                      [label for label, predict in enumerate(pred_5) if not label in predict]
    #
    #     accuracy, accuracy_5 = 1 - len(error)/len(pred), 1 - len(error_5) / len(pred)
    #     print(' accuracy_5 is {}'.format( accuracy_5))
    #     return accuracy, accuracy_5

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
            train_acc, train_acc_top5 = self.get_acc(pl_module, batch)

        pl_module.log('online_train_acc',train_acc, on_step=True, on_epoch=False)
        pl_module.log('online_train_acc_top5',train_acc_top5, on_step=True, on_epoch=False)

    def on_validation_batch_end(
        self, trainer, pl_module: LightningModule, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        with torch.no_grad():
            val_acc, val_acc_top5 = self.get_acc(pl_module, batch)
        pl_module.log('online_val_acc', val_acc, on_step=False, on_epoch=True,sync_dist=True)
        pl_module.log('online_val_acc_top5', val_acc_top5, on_step=False, on_epoch=True,sync_dist=True)



def resample(factor, dir):
    with rasterio.open(dir) as dataset:
        # resample data to target shape
        data = dataset.read(
            out_shape=(
                dataset.count,
                int(dataset.height * factor),
                int(dataset.width * factor)
            ),
            resampling=Resampling.bilinear
        )

        # scale image transform
        transform = dataset.transform * dataset.transform.scale(
            (dataset.width / data.shape[-1]),
            (dataset.height / data.shape[-2])
        )
    image = np.moveaxis(data, 0, -1)
    return image


def vxlize(dir, res):
    pc = PyntCloud.from_file(dir)
    pc.points = pc.points[(pc.points.z >= -20) & (pc.points.z < 30)]
    res_meter = 100/res
    voxelgrid_id = pc.add_structure('voxelgrid',
                                     n_x= int(596 * res_meter),
                                     n_y= int(601 * res_meter),
                                     n_z=20,
                                     regular_bounding_box = False)
    # deault the z axis range 20
    voxelgrid = pc.structures[voxelgrid_id]
    binary_feature_vector = voxelgrid.get_feature_vector(mode = 'binary')
    binary_feature_vector = np.swapaxes(binary_feature_vector, 0,1)
    return binary_feature_vector


def bbox(idx, ks, data_size):
    wl = np.floor(data_size[0] / ks).astype(int)
    wh = np.floor(data_size[1] / ks).astype(int)
    # wh = np.sqrt(data_size).astype(int)
    X, Y = np.unravel_index(idx, (wl, wh))
    bdbox = []
    for x, y in zip(X, Y):
        bdbox.append(
            patches.Rectangle(((y - 1) * ks + 1, (x) * ks), ks, ks, linewidth=1, edgecolor='r', facecolor='none'))
    return bdbox


def plot_top5(error_5, pred_5, dm, r):
    rs = random.sample(error_5, 10)
    wrong = pred_5[rs]
    right = torch.Tensor(rs)[..., None]
    all = torch.hstack((right, wrong)).flatten()

    figure1 = plt.figure(figsize=(80, 50))
    gs0 = gridspec.GridSpec(1, 2)
    gs00 = gridspec.GridSpecFromSubplotSpec(10, 6, subplot_spec=gs0[0])
    gs01 = gridspec.GridSpecFromSubplotSpec(10, 6, subplot_spec=gs0[1])
    count = 0
    for i in range(10):
        for j in range(6):
            rimg, img = dm[int(all[count])][0].transpose(0, -1), dm[int(all[count])][1].transpose(0, -1)
            # mean = torch.mean(img, axis=(1, 2), keepdims=True)
            # std = torch.std(img, axis=(1, 2), keepdims=True)
            # img = (img - mean) / std
            count += 1

            ax00 = figure1.add_subplot(gs00[i, j])
            ax00.imshow(rimg)
            ax00.set_xticklabels([])
            ax00.set_yticklabels([])
            ax00.set_aspect('equal')

            ax01 = figure1.add_subplot(gs01[i, j])
            ax01.imshow(img)
            ax01.set_xticklabels([])
            ax01.set_yticklabels([])
            ax01.set_aspect('equal')

    figure1.tight_layout(rect=[0, 0.03, 1, 0.95])
    figure1.savefig('../thesis_out/Top5_{}.png'.format(r))
