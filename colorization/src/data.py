import os, json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from skimage.color import rgb2lab
from glob import glob
from sklearn.model_selection import train_test_split
from torchvision import transforms

from .config import cfg


class ImageDataset(Dataset):

    def __init__(self, paths, image_size, train=True, augment=True):
        super().__init__()
        self.image_size = image_size
        self.train = train

        tfms = [transforms.Resize((image_size, image_size))]
        if train and augment:
            tfms += [
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomAffine(degrees=5, translate=(0.02, 0.02)),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                ),
            ]
        self.transforms = transforms.Compose(tfms)

        valid_paths = []
        print(f"Checking valid RGB images ({len(paths)} total)...")
        total_skipped = 0
        for p in paths:
            try:
                im = Image.open(p)
                if im.mode == "RGB":
                    valid_paths.append(p)
                else:
                    total_skipped += 1
            except:
                total_skipped += 1
        self.paths = valid_paths
        print(f"Found {len(self.paths)} valid RGB images, skipped {total_skipped}.")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.transforms(img)
        img = np.array(img)
        lab = rgb2lab(img).astype("float32")
        lab = torch.from_numpy(lab).permute(2, 0, 1)
        L = (lab[[0], ...] / 50.0) - 1.0
        ab = lab[[1, 2], ...] / 128.0
        return L, ab


def get_data_loaders(
    folder_path,
    cfg,
    stage=None,
    train_aug=True,
    seed=42,
):
    # Check if folder_path is a list with separate train and validation paths
    if isinstance(folder_path, list) and len(folder_path) == 2:
        train_folder, val_folder = folder_path
        train_paths = sorted(glob.glob(train_folder + "/**/*", recursive=True))
        val_paths = sorted(glob.glob(val_folder + "/**/*", recursive=True))
        train_limit = min(cfg.data_amounts, len(train_paths))
        val_limit = int(min(cfg.data_amounts * cfg.val_pct, len(val_paths)))
        train_paths = train_paths[:train_limit]
        val_paths = val_paths[:val_limit]
        print(
            f"Got {len(train_paths)} training images and {len(val_paths)} validation images"
        )
    # use single path and split
    elif isinstance(folder_path, str):
        paths = sorted(glob.glob(folder_path + "/*.jp*g", recursive=True))[
            : cfg.data_amounts
        ]
        print(f"Got {len(paths)} in paths")
        train_paths, val_paths = train_test_split(
            paths, test_size=cfg.val_pct, random_state=seed
        )  # Split for training and validation
    else:
        raise ValueError(
            "folder_path must be a string or a list of two paths [train_path, val_path]"
        )

    if stage is not None:
        print(f"Preparing stage {stage+1} dataloaders...(for progressive training)")
        print(
            f"Image size: {cfg.prog_img_sizes[stage]}, Batchsize: {cfg.prog_batchsizes[stage]}"
        )
        train_set = ImageDataset(
            train_paths,
            image_size=cfg.prog_img_sizes[stage],
            train=True,
            augment=train_aug,
        )
        val_set = ImageDataset(
            val_paths, image_size=cfg.prog_img_sizes[stage], train=False, augment=False
        )

        train_loader = DataLoader(
            train_set,
            batch_size=cfg.prog_batchsizes[stage],
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=cfg.prog_batchsizes[stage],
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        print(f"Preparing dataloaders...")
        print(f"Image size: {cfg.image_size}, Batchsize: {cfg.bs}")
        train_set = ImageDataset(
            train_paths, image_size=cfg.image_size, train=True, augment=train_aug
        )
        val_set = ImageDataset(
            val_paths, image_size=cfg.image_size, train=False, augment=False
        )

        train_loader = DataLoader(
            train_set,
            batch_size=cfg.bs,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=cfg.bs,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader
