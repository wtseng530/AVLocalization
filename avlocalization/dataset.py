import numpy as np
import torch
import glob
import scipy.ndimage
import skimage.io as io
from torchvision.transforms import functional as F

class DFCdataset(torch.utils.data.Dataset):
  def __init__(self, input1_dir, input2_dir, mode, res,transform,  ksize=32):
    self.res = res
    self.ksize = ksize
    self.input1_dir = input1_dir
    self.input2_dir= input2_dir
    self.transform = transform
    if mode == 'dsm':
        self.patch1 = self.process(self.input1_dir)
        self.patch2 = self.process(self.input2_dir)
    if mode == 'vxl':
        self.patch1= self.process(self.input1_dir)
        self.patch2= self.process(self.input2_dir, False)

  def process(self, dir, norm=True):
      factor = 5 / self.res
      allimg = [io.imread(img) for img in glob.glob(dir + '/*')]
      img = np.concatenate(allimg, axis=1)
      img = scipy.ndimage.zoom(img, (factor, factor, 1), order=3)
      if norm:
          img = (img - np.min(img)) / (np.max(img) - np.min(img) )
      x = torch.from_numpy(np.moveaxis(img, -1, 0).astype(np.float32))
      x = x[None, ...]  # shape:([1, 3, 11874, 11874])

      kh, kw = self.ksize,self.ksize
      dh, dw = self.ksize, self.ksize

      x = F.pad(x, (1, 1, 1, 1))
      patches = x.unfold(2, kh, dh).unfold(3, kw, dw)  # shape:([1, 3, 53, 53, 224, 224])
      patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
      patches = patches.view(-1, *patches.size()[3:])  # shape:([2809, 3, 224, 224])
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
