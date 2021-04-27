import numpy as np
import os
import torch
import glob
import scipy.ndimage
import skimage.io as io
from torchvision.transforms import functional as F
from utils import vxlize

class DFCdataset(torch.utils.data.Dataset):
  def __init__(self, input1_dir, input2_dir, mode, res,transform,  ksize=32):
    self.res = res
    self.ksize = ksize
    self.input1_dir = input1_dir
    self.input2_dir= input2_dir
    self.transform = transform
    self.patch1 = self.process(self.input1_dir)

    if mode == 'dsm':
        self.patch2 = self.process(self.input2_dir)
    if mode == 'vxl':
        self.patch2= self.process_vxl(self.input2_dir)

  def process_vxl(self, dir):
      if os.path.isdir(dir):
          allpc = [vxlize(pc, self.res) for pc in glob.glob(dir+ '/*')]
          pc = np.concatenate(allpc, axis=1)
      else:
          pc = vxlize(dir, self.res)
      x = torch.from_numpy(np.moveaxis(pc, -1,0).astype(np.float32))
      x = x[None, None, ...]

      kh, kw = self.ksize, self.ksize
      dh, dw = self.ksize, self.ksize

      patches = x.unfold(3, kh, dh).unfold(4, kw,dw) # shape: torch.Size([1, 1, 20, 37, 112, 32, 32])
      patches = patches.permute(0,3,4,1,5,6,2).contiguous() # shape: torch.Size([1, 37, 112, 1, 20, 32, 32])

      patches = patches.view(-1, *patches.size()[3:]) # shape: torch.Size([4144, 1, 20, 32, 32])
      p_len = len(patches)
      self.len = p_len
      return  patches

  def process(self, dir):
      factor = 5 / self.res
      if os.path.isdir(dir):
          allimg = [io.imread(img) for img in glob.glob(dir + '/*')]
          img = np.concatenate(allimg, axis=1)
      else:
          img = io.imread(dir)
      img = scipy.ndimage.zoom(img, (factor, factor, 1), order=3)
      img = (img - np.min(img)) / (np.max(img) - np.min(img) )
      x = torch.from_numpy(np.moveaxis(img, -1, 0).astype(np.float32))
      x = x[None, ...]  # shape:([1, 3, 11874, 11874])

      kh, kw = self.ksize,self.ksize
      dh, dw = self.ksize,self.ksize

      x = F.pad(x, (1, 1, 1, 1))
      patches = x.unfold(2, kh, dh).unfold(3, kw, dw)  # shape: torch.Size([1, 3, 37, 223, 32, 32])
      patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous() # shape: torch.Size([1, 37, 223, 3, 32, 32])
      patches = patches.view(-1, *patches.size()[3:])  # shape: torch.Size([8251, 3, 32, 32])
      p_len = len(patches)
      self.len = p_len
      return patches

  def __len__(self):
    return self.len

  def __getitem__(self, idx):
    patch_1 = torch.FloatTensor(self.patch1[idx])
    patch_2 = torch.FloatTensor(self.patch2[idx])
    if self.transform:
      patch_1 = self.transform(patch_1)
      patch_2 = self.transform(patch_2)

    return  [patch_1, patch_2]
