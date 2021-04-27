from argparse import ArgumentParser
import skimage.io as io
import numpy as np
import random
import glob
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import scipy.ndimage

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

    tiles = [8,12]
    fig, ax = plt.subplots(nrows=1, ncols=2)
    for r, t in enumerate(tiles):
        gbimg = io.imread(args.input1_dir + '/Tile{}.tif'.format(t))
        gbimg = scipy.ndimage.zoom(gbimg, (5 / args.res, 5 / args.res, 1), order=3)

        dm = DFCdataset(args.input1_dir+ '/Tile{}.tif'.format(t), args.input2_dir+'/Tile{}.tif'.format(t), args.mode, args.res, None, args.patch_dim)
        datasize = len(dm)
        dataloader = DataLoader(dm, batch_size= datasize, shuffle=False, num_workers=0)

        wgt = './lightning_logs/version_{}/checkpoints/last.ckpt'.format(args.wgtid)
        biclr = biCLR.load_from_checkpoint(wgt, strict=False)
        biclr.eval()

        for batch in dataloader:
            input1, input2 = batch

        pres1, pres2 = biclr(input1, input2)
        feature_gl, feature_lcl = biclr.projection(pres1), biclr.projection(pres2)
        feature_gl, feature_lcl = F.normalize(feature_gl, dim=1), F.normalize(feature_lcl, dim=1)

        similarity_matrix = torch.mm(feature_lcl, feature_gl.T)
        pred = torch.argmax(similarity_matrix, dim=1)
        pred_5 = torch.topk(similarity_matrix, k=5, dim=1,).indices
        error = [label for label, predict in enumerate(pred) if predict != label]
        error_5 = [label for label, predict in enumerate(pred_5) if not label in predict]
        accuracy = 1 - len(error)/ datasize
        accuracy_5 = 1 - len(error_5)/ datasize

        figure = plt.figure(figsize=(50,50))
        ax = [figure.add_subplot(10,6,i+1) for i in range(60)]
        rs = random.sample(error_5, 10)
        wrong = pred_5[rs]
        right = torch.Tensor(rs)[..., None]
        all= torch.hstack((right,wrong)).flatten()
        for i, a in enumerate(ax):
            a.imshow(dm[int(all[i])][1].transpose(0,-1), vmin=0, vmax=1)
            a.set_xticklabels([])
            a.set_yticklabels([])
            a.set_aspect('equal')

        figure.subplots_adjust(wspace=0,hspace=0)
        figure.savefig('../output/Top5_{}dsm.png'.format(r))

        ax[r].imshow(gbimg)
        ax[r].add_collection(PatchCollection(bbox(error, args.patch_dim, gbimg.shape), match_original=True))
        ax[r].title.set_text("Accuracy: {:.2f}\nTop5: {:.2f}".format(accuracy, accuracy_5))

    fig.savefig('../output/Val_{}_{}.png'.format(args.res, args.wgtid))

if __name__ == '__main__':
    cli_main()
