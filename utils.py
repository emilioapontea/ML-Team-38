from numpy.random.mtrand import dirichlet
import os
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image

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
    img_paths = load_imagepaths_with_labels(curr_folder)
    return np.random.choice(img_paths)

def load_random_imagepath(curr_folder: str) -> str:
    img_paths = load_imagepaths(curr_folder)
    return np.random.choice(img_paths)

def get_labels(curr_folder: str) -> List[str]:
    dirs = os.listdir(curr_folder)
    labels = []
    for dir in dirs:
        ran = get_range(int(dir))[0]
        if ran not in labels:
            labels.append(ran)
    return labels

def split_images(src_folder: str, dst_folder: str, p_val: float = 0.2, p_test = 0.1) -> Tuple[int, int, int]:
    img_paths = load_imagepaths_with_labels(src_folder)  # This should return a list of tuples
    test_count = 0
    val_count = 0
    train_count = 0
    for (path, label) in img_paths:
            r = np.random.uniform()
            try:
                img = Image.open(path)
                # Choose the directory based on the random number
                if r < p_test:
                    subdir = "test"
                    test_count += 1
                elif r < p_test + p_val:
                    subdir = "val"
                    val_count += 1
                else:
                    subdir = "train"
                    train_count += 1
                # Use the label as the directory name
                label_dir = os.path.join(dst_folder, subdir, str(label))
                os.makedirs(label_dir, exist_ok=True)  # Create the directory if it doesn't exist
                # Extract the filename from the path
                filename = os.path.basename(path)
                # Save the image in the directory named after the label
                img.save(os.path.join(label_dir, filename))
            except:
                print(f"The following image might cause problems: {path}")

    return test_count, val_count, train_count