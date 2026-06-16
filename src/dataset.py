"""
PyTorch Dataset + transforms for ISIC-2018 skin-lesion segmentation.

Expects images and masks already on disk:
    images_dir/<id>.jpg
    masks_dir/<id>_segmentation.png

Split CSVs (train/val/test) are produced by `make_splits`.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .utils import resolve

# ImageNet statistics — required when using ImageNet-pretrained encoders.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# -----------------------------------------------------------------------------
# Transforms
# -----------------------------------------------------------------------------
def train_transforms(image_size: int) -> A.Compose:
    """Augmentations that mimic real dermoscopy variation without distorting
    the lesion identity. Mask is transformed identically (additional_targets)."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=20, border_mode=cv2.BORDER_REFLECT_101, p=0.5),
        A.ElasticTransform(alpha=1, sigma=50, p=0.2),
        A.GridDistortion(p=0.2),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CLAHE(clip_limit=2.0, p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def eval_transforms(image_size: int) -> A.Compose:
    """Deterministic transforms for validation / test / inference."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def preprocess_single(image_rgb: np.ndarray, image_size: int) -> torch.Tensor:
    """Preprocess one RGB numpy image for inference -> (1,3,H,W) tensor."""
    tf = eval_transforms(image_size)
    return tf(image=image_rgb)["image"].unsqueeze(0)


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class SkinLesionDataset(Dataset):
    """Reads (image, mask) pairs listed in a split CSV with columns
    `image_path` and `mask_path` (absolute or repo-relative)."""

    def __init__(self, csv_path, image_size: int = 256, train: bool = False):
        self.df = pd.read_csv(resolve(csv_path))
        self.image_size = image_size
        self.tf = train_transforms(image_size) if train else eval_transforms(image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        image = cv2.imread(str(resolve(row["image_path"])), cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(resolve(row["mask_path"])), cv2.IMREAD_GRAYSCALE)

        # Binarise mask to {0,1} (ISIC masks are 0/255).
        mask = (mask > 127).astype(np.float32)

        augmented = self.tf(image=image, mask=mask)
        image_t = augmented["image"]                       # (3,H,W) float
        mask_t = augmented["mask"].unsqueeze(0).float()    # (1,H,W) float
        return image_t, mask_t


# -----------------------------------------------------------------------------
# Split creation
# -----------------------------------------------------------------------------
def make_splits(images_dir, masks_dir, splits_dir, mask_suffix="_segmentation",
                val_fraction=0.15, test_fraction=0.15, seed=42) -> dict:
    """Scan the raw dataset, pair images with masks, and write
    train/val/test CSVs. Returns counts per split."""
    images_dir, masks_dir, splits_dir = map(resolve, (images_dir, masks_dir, splits_dir))
    splits_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in image_exts:
            continue
        stem = img_path.stem
        # Try common mask naming patterns.
        candidates = [
            masks_dir / f"{stem}{mask_suffix}.png",
            masks_dir / f"{stem}.png",
            masks_dir / f"{stem}_mask.png",
        ]
        mask_path = next((c for c in candidates if c.exists()), None)
        if mask_path is None:
            continue
        rows.append({"image_path": str(img_path), "mask_path": str(mask_path)})

    if not rows:
        raise RuntimeError(
            f"No image/mask pairs found.\n  images: {images_dir}\n  masks:  {masks_dir}\n"
            "Run `python -m src.download_data` first, or check your folders.")

    df = pd.DataFrame(rows)
    train_df, temp_df = train_test_split(
        df, test_size=val_fraction + test_fraction, random_state=seed)
    rel = test_fraction / (val_fraction + test_fraction)
    val_df, test_df = train_test_split(temp_df, test_size=rel, random_state=seed)

    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    counts = {"train": len(train_df), "val": len(val_df), "test": len(test_df)}
    print(f"[splits] wrote CSVs to {splits_dir}: {counts}")
    return counts


# -----------------------------------------------------------------------------
# DataLoader factory
# -----------------------------------------------------------------------------
def build_dataloaders(cfg) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test loaders from config."""
    d = cfg["data"]
    t = cfg["train"]
    size = d["image_size"]
    splits = resolve(d["splits_dir"])

    train_ds = SkinLesionDataset(splits / "train.csv", size, train=True)
    val_ds = SkinLesionDataset(splits / "val.csv", size, train=False)
    test_ds = SkinLesionDataset(splits / "test.csv", size, train=False)

    common = dict(batch_size=t["batch_size"], num_workers=t["num_workers"],
                  pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader
