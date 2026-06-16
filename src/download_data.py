r"""
Dataset acquisition for ISIC-2018 Task 1 (lesion boundary segmentation).

Three ways to get data, in order of preference:

1) Kaggle API (recommended, automatic):
     - Create a Kaggle account, go to Account -> Create New API Token.
     - Put the downloaded kaggle.json in  %USERPROFILE%\.kaggle\kaggle.json
     - Run:  python -m src.download_data --source kaggle
   Dataset used: "tschandl/isic2018-challenge-task1-data-segmentation"
   (mirrors the official ISIC-2018 Task1 images + ground-truth masks).

2) Manual download:
     - Official site: https://challenge.isic-archive.com/data/#2018
       Download "Task 1: Training Input" (images) and
       "Task 1: Training GroundTruth" (masks).
     - Unzip images  -> data/raw/ISIC2018/images
     - Unzip masks   -> data/raw/ISIC2018/masks
     - Then run:  python -m src.download_data --source local   (just builds splits)

3) Synthetic smoke-test data (no download, for verifying the pipeline):
     - python -m src.download_data --source synthetic
   Generates a handful of fake "lesion" images + masks so train.py runs end-to-end.

After data is in place this script writes train/val/test CSVs to data/splits/.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np

from .dataset import make_splits
from .utils import load_config, resolve

KAGGLE_DATASET = "tschandl/isic2018-challenge-task1-data-segmentation"


def _ensure_dirs(images_dir: Path, masks_dir: Path):
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)


def download_kaggle(images_dir: Path, masks_dir: Path):
    """Download + unzip the ISIC-2018 segmentation mirror via the Kaggle CLI."""
    raw_root = images_dir.parent          # data/raw/ISIC2018
    raw_root.mkdir(parents=True, exist_ok=True)
    print(f"[kaggle] downloading {KAGGLE_DATASET} -> {raw_root}")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
             "-p", str(raw_root), "--unzip"],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print("\n[ERROR] Kaggle download failed.", e)
        print("Make sure the `kaggle` package is installed and kaggle.json is in "
              "%USERPROFILE%\\.kaggle\\. Falling back to manual instructions "
              "(see this file's docstring).")
        sys.exit(1)

    print("[kaggle] download complete. Inspect", raw_root,
          "and adjust images_dir/masks_dir in configs/config.yaml if the "
          "extracted folder names differ.")


def make_synthetic(images_dir: Path, masks_dir: Path, n: int = 40, size: int = 256):
    """Generate fake dermoscopy-like images with elliptical 'lesions' so the
    whole pipeline can be smoke-tested without any real download."""
    _ensure_dirs(images_dir, masks_dir)
    rng = np.random.default_rng(0)
    print(f"[synthetic] generating {n} fake image/mask pairs in {images_dir.parent}")
    for i in range(n):
        # Skin-like background.
        base = rng.integers(150, 210)
        img = np.full((size, size, 3), base, np.uint8)
        img = cv2.add(img, rng.integers(0, 25, (size, size, 3), dtype=np.uint8))

        mask = np.zeros((size, size), np.uint8)
        cx, cy = rng.integers(70, size - 70, 2)
        ax, ay = rng.integers(25, 60, 2)
        angle = int(rng.integers(0, 180))
        # Darker lesion blob.
        cv2.ellipse(img, (int(cx), int(cy)), (int(ax), int(ay)), angle, 0, 360,
                    (int(base * 0.45), int(base * 0.4), int(base * 0.5)), -1)
        cv2.ellipse(mask, (int(cx), int(cy)), (int(ax), int(ay)), angle, 0, 360, 255, -1)
        img = cv2.GaussianBlur(img, (5, 5), 0)

        stem = f"SYNTH_{i:04d}"
        cv2.imwrite(str(images_dir / f"{stem}.jpg"), img)
        cv2.imwrite(str(masks_dir / f"{stem}_segmentation.png"), mask)
    print("[synthetic] done.")


def main():
    parser = argparse.ArgumentParser(description="Acquire ISIC-2018 data + build splits.")
    parser.add_argument("--source", choices=["kaggle", "local", "synthetic"],
                        default="kaggle")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    d = cfg["data"]
    images_dir = resolve(d["images_dir"])
    masks_dir = resolve(d["masks_dir"])

    if args.source == "kaggle":
        download_kaggle(images_dir, masks_dir)
    elif args.source == "synthetic":
        make_synthetic(images_dir, masks_dir)
    # "local" assumes the user already placed files.

    make_splits(
        images_dir=images_dir,
        masks_dir=masks_dir,
        splits_dir=resolve(d["splits_dir"]),
        mask_suffix=d["mask_suffix"],
        val_fraction=d["val_fraction"],
        test_fraction=d["test_fraction"],
        seed=cfg["project"]["seed"],
    )


if __name__ == "__main__":
    main()
