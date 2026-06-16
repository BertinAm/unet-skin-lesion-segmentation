"""
Evaluate a trained checkpoint on the held-out test set and report all metrics.
Also supports comparing several checkpoints side by side for the report table.

Usage:
    python -m src.evaluate --checkpoint checkpoints/best_unet.pt
    python -m src.evaluate --compare checkpoints/best_unet.pt checkpoints/best_attention_unet.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from .dataset import build_dataloaders
from .metrics import MetricTracker, compute_all
from .models import build_model
from .utils import get_device, load_config, resolve


@torch.no_grad()
def evaluate_checkpoint(ckpt_path, cfg, device, threshold) -> dict:
    ckpt = torch.load(resolve(ckpt_path), map_location=device)
    # Use the architecture the checkpoint was trained with.
    ckpt_cfg = ckpt.get("config", cfg)
    model = build_model(ckpt_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, _, test_loader = build_dataloaders(cfg)
    tracker = MetricTracker()
    for images, masks in tqdm(test_loader, desc=Path(ckpt_path).stem, leave=False):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        tracker.update(compute_all(logits, masks, threshold), n=images.size(0))
    result = tracker.average()
    result["arch"] = ckpt_cfg["model"]["arch"]
    result["checkpoint"] = Path(ckpt_path).name
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--compare", nargs="+", default=None,
                        help="multiple checkpoint paths to compare")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device(cfg["project"]["device"])
    threshold = cfg["inference"]["threshold"]

    paths = args.compare or [args.checkpoint or cfg["inference"]["checkpoint"]]
    rows = [evaluate_checkpoint(p, cfg, device, threshold) for p in paths]

    df = pd.DataFrame(rows)
    cols = ["arch", "dice", "iou", "precision", "recall", "hd95", "checkpoint"]
    df = df[[c for c in cols if c in df.columns]]
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print("\n===== Test-set results =====")
    print(df.to_string(index=False))

    out = resolve(cfg["logging"]["outputs_dir"]) / "comparison.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
