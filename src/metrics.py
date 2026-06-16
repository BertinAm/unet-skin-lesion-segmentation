"""
Evaluation metrics: Dice, IoU, Precision, Recall, Hausdorff distance.

Metrics take raw LOGITS and apply a threshold to get a binary prediction,
matching how the model is used at inference time.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch

try:
    from medpy.metric.binary import hd95 as _medpy_hd95
    _HAS_MEDPY = True
except Exception:  # medpy optional
    _HAS_MEDPY = False
    from scipy.spatial.distance import directed_hausdorff


def _binarize(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    return (torch.sigmoid(logits) > threshold).float()


@torch.no_grad()
def dice_coef(logits, targets, threshold=0.5, smooth=1e-6) -> float:
    pred = _binarize(logits, threshold).view(-1)
    tgt = targets.view(-1)
    inter = (pred * tgt).sum()
    return ((2 * inter + smooth) / (pred.sum() + tgt.sum() + smooth)).item()


@torch.no_grad()
def iou_score(logits, targets, threshold=0.5, smooth=1e-6) -> float:
    pred = _binarize(logits, threshold).view(-1)
    tgt = targets.view(-1)
    inter = (pred * tgt).sum()
    union = pred.sum() + tgt.sum() - inter
    return ((inter + smooth) / (union + smooth)).item()


@torch.no_grad()
def precision_recall(logits, targets, threshold=0.5, eps=1e-6):
    pred = _binarize(logits, threshold).view(-1)
    tgt = targets.view(-1)
    tp = (pred * tgt).sum()
    fp = (pred * (1 - tgt)).sum()
    fn = ((1 - pred) * tgt).sum()
    precision = (tp / (tp + fp + eps)).item()
    recall = (tp / (tp + fn + eps)).item()
    return precision, recall


@torch.no_grad()
def hausdorff95(logits, targets, threshold=0.5) -> float:
    """95th-percentile Hausdorff distance (pixels). Lower = better boundary.
    Uses medpy if available (robust), else a scipy fallback. Returns NaN when
    either mask is empty (undefined)."""
    pred = _binarize(logits, threshold).squeeze().cpu().numpy().astype(bool)
    tgt = targets.squeeze().cpu().numpy().astype(bool)
    if pred.sum() == 0 or tgt.sum() == 0:
        return float("nan")
    if _HAS_MEDPY:
        return float(_medpy_hd95(pred, tgt))
    a = np.argwhere(pred)
    b = np.argwhere(tgt)
    return float(max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0]))


@torch.no_grad()
def compute_all(logits, targets, threshold=0.5) -> Dict[str, float]:
    """All metrics for one batch (averaged per-sample for HD95)."""
    p, r = precision_recall(logits, targets, threshold)
    # HD95 is per-image; average over the batch.
    hds = []
    for i in range(logits.size(0)):
        hd = hausdorff95(logits[i], targets[i], threshold)
        if not np.isnan(hd):
            hds.append(hd)
    return {
        "dice": dice_coef(logits, targets, threshold),
        "iou": iou_score(logits, targets, threshold),
        "precision": p,
        "recall": r,
        "hd95": float(np.mean(hds)) if hds else float("nan"),
    }


class MetricTracker:
    """Accumulate batch metrics and report epoch averages."""

    def __init__(self):
        self.reset()

    def reset(self):
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float], n: int = 1):
        for k, v in metrics.items():
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            self._sums[k] = self._sums.get(k, 0.0) + v * n
            self._counts[k] = self._counts.get(k, 0) + n

    def average(self) -> Dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums}
