"""
Loss functions for binary medical segmentation.

All losses operate on raw LOGITS (the model's final conv output, no sigmoid).
They apply sigmoid internally — this is numerically stabler than applying
sigmoid in the model and then BCELoss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss = 1 - Dice. Robust to foreground/background imbalance
    because it only rewards overlap of the positive (lesion) class."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        intersection = (probs * targets).sum(dim=1)
        union = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class TverskyLoss(nn.Module):
    """Generalised Dice. alpha weights false positives, beta false negatives.
    alpha<beta penalises missed lesion pixels harder — good for tiny tumours."""

    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1e-6):
        super().__init__()
        self.alpha, self.beta, self.smooth = alpha, beta, smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits).view(logits.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1 - targets)).sum(dim=1)
        fn = ((1 - probs) * targets).sum(dim=1)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1 - tversky.mean()


class BCEDiceLoss(nn.Module):
    """Industry-standard combined loss: weighted BCE (pixel-wise) + Dice (shape).
    BCE catches per-pixel errors; Dice catches global overlap errors."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight, self.dice_weight = bce_weight, dice_weight

    def forward(self, logits, targets):
        return self.bce_weight * self.bce(logits, targets) + \
               self.dice_weight * self.dice(logits, targets)


def build_loss(cfg) -> nn.Module:
    """Select loss from config `train.loss`."""
    t = cfg["train"]
    name = t["loss"].lower()
    if name == "dice":
        return DiceLoss()
    if name == "tversky":
        return TverskyLoss(alpha=t["tversky_alpha"], beta=t["tversky_beta"])
    if name in ("bce_dice", "bcedice"):
        return BCEDiceLoss(bce_weight=t["bce_weight"], dice_weight=t["dice_weight"])
    raise ValueError(f"Unknown loss '{name}'. Use bce_dice | dice | tversky.")
