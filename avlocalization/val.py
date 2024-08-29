import glob
import skimage.io as io
from argparse import ArgumentParser

import scipy.ndimage
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from avlocalization.utils import bbox, plot_top5
from model import biCLR

from data_preprocessing import DFCDataset


def cli_main():
    parser = ArgumentParser()
    parser.add_argument("--input1_dir", type=str, default="../data/val/rgb",
                        help="direction to the input for first encoder")
    parser.add_argument("--input2_dir", type=str, default="../data/val/dsm",
                        help="direction to the input for second encoder")
    parser.add_argument("--patch_dim", type=int, default=32, help='image patch size')
    parser.add_argument("--res", type=int, default=5, help="resolution of training image")
    parser.add_argument('--mode', type=str, default='dsm', choices=['dsm', 'vxl'],
                        help='branch for local patch training')
    parser.add_argument('--wgtid', type=str, help='job number/under which log and weight is stored')
    parser.add_argument('--all', action='store_true', help='evaluate on all the tiles')
    parser.add_argument('--top_5', action='store_true', help='visualize top5 error')

    args = parser.parse_args()

    if args.all:
        ip1 = [args.input1_dir + '/cat.tif']
        ip2 = [args.input2_dir + '/cat.tif']
    else:
        ip1 = glob.glob(args.input1_dir + '/t*')
        ip2 = glob.glob(args.input2_dir + '/t*')

    wgt = './lightning_logs/version_{}/checkpoints/last.ckpt'.format(args.wgtid)
    biclr = biCLR.load_from_checkpoint(wgt, strict=False, **args.__dict__)
    biclr.eval()

    fig, ax = plt.subplots(nrows=1, ncols=len(ip1))
    for r, (file1, file2) in enumerate(zip(ip1, ip2)):
        gbimg = io.imread(file1)
        gbimg = scipy.ndimage.zoom(gbimg, (50 / args.res, 50 / args.res, 1), order=3)

        dm = DFCDataset(file1, file2, args.mode, args.res, None, args.patch_dim)
        datasize = len(dm)

        dataloader = DataLoader(dm, batch_size=datasize, shuffle=False, num_workers=0)
        input1, input2 = next(iter(dataloader))
        pres1, pres2 = biclr(input1, input2)
        feature_gl, feature_lcl = biclr.projection(pres1), biclr.projection(pres2)
        glFeature, lclFeature = F.normalize(feature_gl, dim=1), F.normalize(feature_lcl, dim=1)

        similarity_matrix = torch.mm(lclFeature, glFeature.T)
        pred, pred_5 = torch.argmax(similarity_matrix, dim=1), torch.topk(similarity_matrix, k=5, dim=1, ).indices
        error, error_5 = [label for label, predict in enumerate(pred) if predict != label], [label for label, predict in
                                                                                             enumerate(pred_5) if
                                                                                             not label in predict]
        accuracy, accuracy_5 = 1 - len(error) / datasize, 1 - len(error_5) / datasize

        # export the top5 error compare to the true pair
        if args.top_5:
            plot_top5(error_5, pred_5, dm, r)

        if args.all:
            ax.imshow(gbimg)
            ax.add_collection(PatchCollection(bbox(error_5, args.patch_dim, gbimg.shape), match_original=True))
            ax.title.set_text("Accuracy: {:.2f}\nTop5: {:.2f}".format(accuracy, accuracy_5))
        else:
            ax[r].imshow(gbimg)
            ax[r].add_collection(PatchCollection(bbox(error_5, args.patch_dim, gbimg.shape), match_original=True))
            ax[r].title.set_text("Accuracy: {:.2f}\nTop5: {:.2f}".format(accuracy, accuracy_5))

    fig.savefig('../thesis_out/Val_{}_{}_{}_top5.png'.format(args.res, args.wgtid, args.all))


if __name__ == '__main__':
    cli_main()
