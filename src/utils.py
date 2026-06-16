"""
Shared helpers: config loading, reproducibility, device selection, overlays.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


# Project root = parent of this file's parent (src/.. -> repo root)
ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | os.PathLike = "configs/config.yaml") -> Dict[str, Any]:
    """Load the YAML config. Relative paths are resolved against repo root."""
    cfg_path = Path(path)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve(path: str | os.PathLike) -> Path:
    """Resolve a possibly-relative path against the repo root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def set_seed(seed: int = 42) -> None:
    """Make runs reproducible across python / numpy / torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic cudnn is slower but reproducible.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(preferred: str = "cuda") -> torch.device:
    """Return cuda if requested and available, else cpu."""
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def overlay_mask(image: np.ndarray, mask: np.ndarray,
                 color=(255, 0, 0), alpha: float = 0.45) -> np.ndarray:
    """
    Blend a binary mask onto an RGB image for visualisation.

    image: HxWx3 uint8 RGB
    mask:  HxW   {0,1} (or 0..255)
    returns HxWx3 uint8 with the lesion tinted `color`.
    """
    image = image.astype(np.float32)
    mask_bool = mask.astype(bool)
    tint = np.zeros_like(image)
    tint[..., 0], tint[..., 1], tint[..., 2] = color
    blended = image.copy()
    blended[mask_bool] = (1 - alpha) * image[mask_bool] + alpha * tint[mask_bool]
    return blended.clip(0, 255).astype(np.uint8)


def count_parameters(model: torch.nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
