import matplotlib.pyplot as plt
from typing import Tuple, Union
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from skimage import feature
from PIL import Image

def open_image(filepath: str) -> Image:
  return Image.open(filepath)

def get_transforms(in_size: Tuple[int, int]) -> transforms.Compose:
  return transforms.Compose([
      transforms.Resize(in_size),
      transforms.ToTensor()
      ])

### Hou, Sujuan, et al. ###
# * SIFT or HOG descriptors work well for logo detection
# * Current settings reduce data by 72.15%
# * Important to note color information is lost by HOG (I think)

def extract_hog_features(data: torch.Tensor, visualize: bool = False):
  np_data = data.numpy()
  orientations=9
  pixels_per_cell=(12,12)
  cells_per_block=(4,4)
  block_norm='L2-Hys'

  if visualize:
    hog, hog_img = feature.hog(
      np_data,
      orientations=orientations,
      pixels_per_cell=pixels_per_cell,
      cells_per_block=cells_per_block,
      block_norm=block_norm,
      feature_vector=True,
      visualize=visualize,
      channel_axis=0
      )
    return torch.tensor(hog), hog_img
  else:
    hog = feature.hog(
      np_data,
      orientations=orientations,
      pixels_per_cell=pixels_per_cell,
      cells_per_block=cells_per_block,
      block_norm=block_norm,
      feature_vector=True,
      visualize=visualize,
      channel_axis=0
      )
    return torch.tensor(hog)

def extract_all_hog_features(data: torch.Tensor) -> torch.Tensor:
  ret = []
  for image in data:
    ret.append(extract_hog_features(image))
  return torch.stack(ret)