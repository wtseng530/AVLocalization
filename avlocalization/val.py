from argparse import ArgumentParser
import skimage.io as io
import numpy as np
import glob
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
        bdbox.append(patches.Rectangle(((y)*ks, (x+1)*ks), ks, ks,linewidth=1, edgecolor='r', facecolor='none'))
    return bdbox

def cli_main():

    parser = ArgumentParser()
    parser.add_argument("--input1_dir", type=str, default="../data/val/rgb",help="direction to the input for first encoder")
    parser.add_argument("--input2_dir", type=str, default="../data/val/dsm", help="direction to the input for second encoder")
    parser.add_argument("--res", type=int, default=5, help="resolution of training image")
    parser.add_argument('--weight', type=str, help='pretrained weight of biclr model')
    parser.add_argument('--output_nm', type=str, help='output name to prediction map')
    args = parser.parse_args()

    allrgb = [io.imread(img) for img in glob.glob(args.input1_dir + '/*')]
    alldpt = [io.imread(dpt) for dpt in glob.glob(args.input2_dir + '/*')]
    gbimg = np.concatenate(allrgb, axis=1)
    lclimg = np.concatenate(alldpt, axis=1)
    factor = 5 / args.res
    if factor  ==1:
        gbimg= gbimg[:5600,:5600,:]
        lclimg = lclimg[:5600,:5600,:]
    else:
        gbimg = scipy.ndimage.zoom(gbimg, (factor, factor, 1), order=3)
        lclimg = scipy.ndimage.zoom(lclimg, (factor, factor, 1), order=3)

    dm = DFCdataset(gbimg, lclimg, None, ksize=32)
    datasize = len(dm)
    dataloader = DataLoader(dm, batch_size= datasize, shuffle=False, num_workers=0)

    biclr = biCLR.load_from_checkpoint(args.weight, strict=False)
    biclr.eval()

    for batch in dataloader:
        input1, input2 = batch

    pres1, pres2 = biclr(input1, input2)
    feature_gl, feature_lcl = biclr.projection(pres1), biclr.projection(pres2)
    feature_gl, feature_lcl = F.normalize(feature_gl, dim=1), F.normalize(feature_lcl, dim=1)

    similarity_matrix = torch.mm(feature_lcl, feature_gl.T)
    pred = torch.argmax(similarity_matrix, dim=1)
    error = [label for label, predict in enumerate(pred) if predict != label]
    accuracy = 1 - len(error)/ datasize

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(gbimg)
    if len(error) != 0:
        ax.add_collection(PatchCollection(bbox(error, 32, lclimg.shape), match_original=True))
    ax.title.set_text("Accuracy: {:.2f}".format(accuracy))
    fig.savefig('../output/'+args.output_nm)


def bbox(idx, ks, data_size):
    wl = np.floor(data_size[0]/ks).astype(int)
    wh = np.floor(data_size[1]/ks).astype(int)
    #wh = np.sqrt(data_size).astype(int)
    X,Y = np.unravel_index(idx, (wl, wh))
    bdbox = []
    for x, y in zip(X,Y):
        bdbox.append(patches.Rectangle(((y)*ks, (x+1)*ks), ks, ks,linewidth=1, edgecolor='r', facecolor='none'))
    return bdbox

# def cli_main():
#
#     parser = ArgumentParser()
#     parser.add_argument("--input1_dir", type=str, default="../data/val/rgb",help="direction to the input for first encoder")
#     parser.add_argument("--input2_dir", type=str, default="../data/val/dsm", help="direction to the input for second encoder")
#     parser.add_argument("--res", type=int, default=5, help="resolution of training image")
#     parser.add_argument('--weight', type=str, help='pretrained weight of biclr model')
#     parser.add_argument('--output_nm', type=str, help='output name to prediction map')
#     args = parser.parse_args()
#
#     allrgb = [io.imread(img) for img in glob.glob(args.input1_dir + '/*')]
#     alldpt = [io.imread(dpt) for dpt in glob.glob(args.input2_dir + '/*')]
#     gbimg = np.concatenate(allrgb, axis=1)[:9600,:9600,:]
#     lclimg = np.concatenate(alldpt, axis=1)[:9600,:9600,:]
#     factor = 5 / args.res
#
#     if factor != 1:
#         gbimg = scipy.ndimage.zoom(gbimg, (factor, factor, 1), order=3)
#         lclimg = scipy.ndimage.zoom(lclimg, (factor, factor, 1), order=3)
#
#     dm = DFCdataset(gbimg, lclimg, None, ksize=32)
#     datasize = len(dm)
#     dataloader = DataLoader(dm, batch_size= 1024, shuffle=False, num_workers=0)
#
#     biclr = biCLR.load_from_checkpoint(args.weight, strict=False)
#     biclr.eval()
#
#     del allrgb, alldpt, gbimg, dm
#     feature_gl = []
#     for batch in dataloader:
#         input1,_= batch
#         pres1 = biclr.encoder1(input1)[-1]
#         fg = biclr.projection(pres1)
#         fg = F.normalize(fg, dim=1)
#         feature_gl.extend(fg)
#     feature_gl = np.array(feature_gl)
#     np.save(feature_gl)
#
#     err = []
#     for ep, batch in enumerate(dataloader):
#         _, input2 = batch
#         pres2 = biclr.encoder2(input2)[-1]
#         feature_lcl= biclr.projection(pres2)
#         feature_lcl = F.normalize(feature_lcl, dim=1)
#
#         similarity_matrix = torch.mm(feature_lcl, feature_gl.T)
#         pred = torch.argmax(similarity_matrix, dim=1)
#         error = [label for label, predict in enumerate(pred) if predict != label]
#         err.extend(error)
#
#     accuracy = 1 - len(err)/ datasize
#
#     fig, ax = plt.subplots(nrows=1, ncols=1)
#     ax.imshow(gbimg)
#     if len(error) != 0:
#         ax.add_collection(PatchCollection(bbox(err, 32, lclimg.shape), match_original=True))
#     ax.title.set_text("Accuracy: {:.2f}".format(accuracy))
#     fig.savefig('../output/'+args.output_nm)

if __name__ == '__main__':
    cli_main()
