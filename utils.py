from numpy.random.mtrand import dirichlet
import os
from typing import Dict, List, Tuple
import numpy as np

# OS utils

def get_range(label: int, bin_size: int = 30, low_limit: int = 30, up_limit: int = 300) -> Tuple[str, int]:
  if label < low_limit:
    low = 0
    up = low_limit
  elif label > up_limit:
    low = up_limit
    up = "inf"
  else:
    low = (label // bin_size) * bin_size
    up = low + bin_size
  bin_num = low // bin_size
  return "[{},{})".format(low, up), bin_num

def load_imagepaths_with_labels(curr_folder: str) -> List[Tuple[str, int]]:

  img_paths = []

  for dir in os.listdir(curr_folder):

    label = get_range(int(dir))[1]

    for file in os.listdir(os.path.join(curr_folder, dir)):
      if file.lower().endswith('.jpg'):
        filepath = os.path.join(curr_folder, dir, file)
        img_paths.append((filepath, label))

  return img_paths

def load_imagepaths(curr_folder: str) -> List[str]:
  img_paths = []
  for root, dirs, files in os.walk(curr_folder):
    for file in files:
      path = os.path.join(root, file)
      img_paths.append(path)
  return img_paths

def load_random_imagepath_with_label(curr_folder: str) -> Tuple[str, int]:
  img_paths = load_imagepaths_with_labels("dataset")
  return np.random.choice(img_paths)

def load_random_imagepath(curr_folder: str) -> str:
  img_paths = load_imagepaths("dataset")
  return np.random.choice(img_paths)

def get_labels(curr_folder: str) -> List[str]:
  dirs = os.listdir(curr_folder)
  labels = []
  for dir in dirs:
    ran = get_range(int(dir))[0]
    if ran not in labels:
      labels.append(ran)
  return labels