"""
dataset.py
----------
Generic 2D medical-image segmentation dataset loader, matching the
preprocessing described in the paper:

  * 3D volumes are converted to 2D images by taking slices.
  * Negative samples (slices with no segmentation object) are removed.
  * Data is split into train (70%) / val (10%) / test (20%).

Works for both datasets used in the paper:
  - Cardiac (King's College London, MRI, 320x320)      -- binary heart mask
  - Brain   (LGG segmentation dataset / TCIA, 256x256)  -- binary tumor mask

Expected on-disk layout (adjust `image_glob` / `mask_glob` if yours differs):

    root/
      images/*.png (or .tif/.npy)
      masks/*.png  (same filenames as images, binary masks)

If your data is still in NIfTI/3D form, use `slice_volume_to_2d()` below to
pre-generate the 2D slice dataset first.
"""

import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


def slice_volume_to_2d(volume, mask, axis=2, drop_empty=True):
    """
    Converts a 3D (H, W, D) volume + mask into a list of 2D (image, mask)
    slice pairs, removing empty (no-foreground) slices if drop_empty=True.
    Use this once, offline, to build a 2D dataset from raw 3D NIfTI data
    before instantiating `MedicalSegmentationDataset`.
    """
    slices = []
    n_slices = volume.shape[axis]
    for i in range(n_slices):
        img_slice = np.take(volume, i, axis=axis)
        mask_slice = np.take(mask, i, axis=axis)
        if drop_empty and mask_slice.sum() == 0:
            continue
        slices.append((img_slice.astype(np.float32), (mask_slice > 0).astype(np.float32)))
    return slices


class MedicalSegmentationDataset(Dataset):
    """
    Loads 2D image/mask pairs from disk and returns normalized tensors:
        image: (1, H, W) float32 in [0, 1]
        mask:  (1, H, W) float32 in {0, 1}
    """

    def __init__(self, root, split="train", image_size=None,
                 image_glob="images/*.png", mask_dir="masks",
                 seed=42, split_ratios=(0.7, 0.1, 0.2), k_shot=None):
        super().__init__()
        self.root = root
        self.image_size = image_size

        image_paths = sorted(glob.glob(os.path.join(root, image_glob)))
        if len(image_paths) == 0:
            raise FileNotFoundError(
                f"No images found with pattern {os.path.join(root, image_glob)}. "
                f"Check your dataset layout / glob pattern."
            )

        pairs = []
        for img_path in image_paths:
          fname = os.path.basename(img_path)

          # Convert image name -> corresponding mask name
          mask_name = fname.replace(".png", "_mask.png")
          mask_path = os.path.join(root, mask_dir, mask_name)

          if os.path.exists(mask_path):
              pairs.append((img_path, mask_path))

        # Deterministic shuffle + split, matching "70% / 10% / 20%" in the paper
        rng = random.Random(seed)
        rng.shuffle(pairs)

        n = len(pairs)
        n_train = int(split_ratios[0] * n)
        n_val = int(split_ratios[1] * n)

        if split == "train":
            self.pairs = pairs[:n_train]
        elif split == "val":
            self.pairs = pairs[n_train:n_train + n_val]
        elif split == "test":
            self.pairs = pairs[n_train + n_val:]
        else:
            raise ValueError(f"Unknown split {split}")

        # k-shot / extreme-data-constraint experiments (Table 4): randomly
        # select k images from the training set as a new, smaller training set.
        if k_shot is not None and split == "train":
            rng2 = random.Random(seed + 1)
            self.pairs = rng2.sample(self.pairs, min(k_shot, len(self.pairs)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.image_size is not None:
            image = image.resize(self.image_size, Image.BILINEAR)
            mask = mask.resize(self.image_size, Image.NEAREST)

        image = np.asarray(image, dtype=np.float32) / 255.0
        mask = (np.asarray(mask, dtype=np.float32) > 127).astype(np.float32)

        image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)     # (1, H, W)

        return image, mask
