from argparse import ArgumentParser
import skimage.io as io
import numpy as np
import random
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import scipy.ndimage
import glob

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from biCLR import biCLR
from dataset import DFCdataset


def bbox(idx, ks, data_size):
    wl = np.floor(data_size[0]/ks).astype(int)
    wh = np.floor(data_size[1]/ks).astype(int)
    #wh = np.sqrt(data_size).astype(int)
    X,Y = np.unravel_index(idx, (wl, wh))
    bdbox = []
    for x, y in zip(X,Y):
        bdbox.append(patches.Rectangle(((y-1)*ks+1, (x)*ks), ks, ks,linewidth=1, edgecolor='r', facecolor='none'))
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
            mean = torch.mean(img, axis=(1, 2), keepdims=True)
            std = torch.std(img, axis=(1, 2), keepdims=True)
            img = (img - mean) / std
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
    figure1.savefig('../output/Top5_{}.png'.format(r))


def cli_main ():

    parser = ArgumentParser()
    parser.add_argument("--input1_dir", type=str, default="../data/val/rgb",help="direction to the input for first encoder")
    parser.add_argument("--input2_dir", type=str, default="../data/val/dsm", help="direction to the input for second encoder")
    parser.add_argument("--patch_dim", type=int, default=32, help='image patch size')
    parser.add_argument("--res", type=int, default=5, help="resolution of training image")
    parser.add_argument('--mode', type=str, default='dsm', choices=['dsm', 'vxl'],
                        help='branch for local patch training')
    parser.add_argument('--wgtid', type=str, help='job number/under which log and weight is stored')

    args = parser.parse_args()

    fig, ax = plt.subplots(nrows=1, ncols=2)
    for r, (file1, file2) in enumerate(zip(ip1,ip2)):
        gbimg = io.imread(file1)
        gbimg = scipy.ndimage.zoom(gbimg, (5 / args.res, 5 / args.res, 1), order=3)

        dm = DFCdataset(file1, file2, args.mode, args.res, None, args.patch_dim)
        datasize = len(dm)
        dataloader = DataLoader(dm, batch_size= datasize, shuffle=False, num_workers=0)

        wgt = './lightning_logs/version_{}/checkpoints/last.ckpt'.format(args.wgtid)
        biclr = biCLR.load_from_checkpoint(wgt, strict=False)
        biclr.eval()

        input1, input2 = next(iter(dataloader))
        pres1, pres2 = biclr(input1, input2)
        feature_gl, feature_lcl = biclr.projection(pres1), biclr.projection(pres2)
        feature_gl, feature_lcl = F.normalize(feature_gl, dim=1), F.normalize(feature_lcl, dim=1)

        similarity_matrix = torch.mm(feature_lcl, feature_gl.T)
        pred, pred_5 = torch.argmax(similarity_matrix, dim=1),torch.topk(similarity_matrix, k=5, dim=1,).indices
        error,error_5 = [label for label, predict in enumerate(pred) if predict != label], [label for label, predict in enumerate(pred_5) if not label in predict]
        accuracy,accuracy_5 = 1 - len(error)/ datasize , 1 - len(error_5)/ datasize

        # export the top5 error compare to the true pair
        plot_top5(error_5, pred_5, dm, r)

        ax[r].imshow(gbimg)
        ax[r].add_collection(PatchCollection(bbox(error, args.patch_dim, gbimg.shape), match_original=True))
        ax[r].title.set_text("Accuracy: {:.2f}\nTop5: {:.2f}".format(accuracy, accuracy_5))

    fig.savefig('../output/Val_{}_{}.png'.format(args.res, args.wgtid))

if __name__ == '__main__':
    cli_main()
