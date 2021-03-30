import numpy as np
import torch
from torchvision.transforms import functional as F

#TODO change to 5cm resolution images

class DFCdataset(torch.utils.data.Dataset):
  def __init__(self, rgbimg, dptimg, transform,  ksize=32):
    self.ksize = ksize
    self.rgb = rgbimg
    self.depth = dptimg
    self.transform = transform
    self.rgbpatch= self.process(self.rgb)
    self.depthpatch= self.process(self.depth)

  def process(self, img):
      norm_img = (img - np.min(img)) / (np.max(img) - np.min(img) )
      x = torch.from_numpy(np.moveaxis(norm_img, -1, 0).astype(np.float32))
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
    #_ = self.process(self.rgb)
    return self.len

  def __getitem__(self, idx):
    rgb_patch = torch.FloatTensor(self.rgbpatch[idx])
    dpt_patch = torch.FloatTensor(self.rgbpatch[idx])
    if self.transform:
      rgb_patch = self.transform(rgb_patch)
      dpt_patch = self.transform(dpt_patch)

    return  [rgb_patch, dpt_patch]
