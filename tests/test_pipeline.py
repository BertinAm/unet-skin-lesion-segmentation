"""
Fast smoke tests — verify every component runs without a GPU or real data.
    pytest -q      (or)      python -m tests.test_pipeline
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.losses import BCEDiceLoss, DiceLoss, TverskyLoss  # noqa: E402
from src.metrics import compute_all  # noqa: E402
from src.models import AttentionUNet, build_model  # noqa: E402


def _fake_batch(n=2, c=3, hw=64):
    x = torch.randn(n, c, hw, hw)
    y = (torch.rand(n, 1, hw, hw) > 0.5).float()
    return x, y


def test_attention_unet_forward():
    model = AttentionUNet(in_channels=3, num_classes=1)
    x, _ = _fake_batch()
    out = model(x)
    assert out.shape == (2, 1, 64, 64)


def test_build_model_smp():
    cfg = {"model": {"arch": "unet", "encoder": "resnet18",
                     "encoder_weights": None, "in_channels": 3, "num_classes": 1}}
    model = build_model(cfg)
    out = model(torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 1, 64, 64)


def test_losses_positive():
    x, y = _fake_batch()
    logits = torch.randn_like(y)
    for loss_fn in (DiceLoss(), TverskyLoss(), BCEDiceLoss()):
        val = loss_fn(logits, y)
        assert val.item() >= 0


def test_metrics_keys():
    _, y = _fake_batch()
    logits = torch.randn_like(y)
    m = compute_all(logits, y)
    assert {"dice", "iou", "precision", "recall"} <= set(m)
    assert 0 <= m["dice"] <= 1


if __name__ == "__main__":
    test_attention_unet_forward()
    test_build_model_smp()
    test_losses_positive()
    test_metrics_keys()
    print("All smoke tests passed [OK]")
