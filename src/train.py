"""
Training loop with mixed precision, cosine warm-restart scheduling,
best-checkpoint saving on val Dice, and early stopping.

Usage:
    python -m src.train                       # uses configs/config.yaml
    python -m src.train --arch attention_unet # override architecture
    python -m src.train --epochs 5            # quick run

Outputs:
    checkpoints/best_<arch>.pt                # best weights + config
    outputs/history_<arch>.csv                # per-epoch metrics
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

from .dataset import build_dataloaders
from .losses import build_loss
from .metrics import MetricTracker, compute_all
from .models import build_model
from .utils import count_parameters, get_device, load_config, resolve, set_seed


def _build_optimizer(model, cfg):
    t = cfg["train"]
    if t["optimizer"].lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=t["lr"],
                                 weight_decay=t["weight_decay"])
    return torch.optim.Adam(model.parameters(), lr=t["lr"],
                            weight_decay=t["weight_decay"])


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    model.train()
    running = 0.0
    for images, masks in tqdm(loader, desc="train", leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        running += loss.item() * images.size(0)
    return running / len(loader.dataset)


@torch.no_grad()
def validate(model, loader, criterion, device, threshold):
    model.eval()
    tracker = MetricTracker()
    running = 0.0
    for images, masks in tqdm(loader, desc="val", leave=False):
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        running += criterion(logits, masks).item() * images.size(0)
        tracker.update(compute_all(logits, masks, threshold), n=images.size(0))
    metrics = tracker.average()
    metrics["loss"] = running / len(loader.dataset)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--arch", default=None, help="override model.arch")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.arch:
        cfg["model"]["arch"] = args.arch
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs

    set_seed(cfg["project"]["seed"])
    device = get_device(cfg["project"]["device"])
    arch = cfg["model"]["arch"]
    print(f"[train] arch={arch} device={device}")

    # Data
    train_loader, val_loader, _ = build_dataloaders(cfg)
    print(f"[train] train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    # Model / loss / optim
    model = build_model(cfg).to(device)
    print(f"[train] trainable params: {count_parameters(model):,}")
    criterion = build_loss(cfg)
    optimizer = _build_optimizer(model, cfg)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

    use_amp = cfg["train"]["mixed_precision"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    threshold = cfg["inference"]["threshold"]

    # Optional Weights & Biases
    wandb_run = None
    if cfg["logging"]["use_wandb"]:
        try:
            import wandb
            wandb_run = wandb.init(project=cfg["logging"]["wandb_project"],
                                   name=arch, config=cfg)
        except Exception as e:
            print("[wandb] disabled:", e)

    ckpt_dir = resolve(cfg["logging"]["checkpoints_dir"]); ckpt_dir.mkdir(exist_ok=True)
    out_dir = resolve(cfg["logging"]["outputs_dir"]); out_dir.mkdir(exist_ok=True)
    best_path = ckpt_dir / f"best_{arch}.pt"
    history_path = out_dir / f"history_{arch}.csv"

    best_metric = -1.0
    patience = cfg["train"]["early_stopping_patience"]
    bad_epochs = 0
    monitor = cfg["train"]["monitor_metric"]
    history = []

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     scaler, device, use_amp)
        val_metrics = validate(model, val_loader, criterion, device, threshold)
        scheduler.step()

        row = {"epoch": epoch, "train_loss": round(train_loss, 4),
               **{f"val_{k}": round(v, 4) for k, v in val_metrics.items()},
               "lr": optimizer.param_groups[0]["lr"], "secs": round(time.time() - t0, 1)}
        history.append(row)
        print(f"[{epoch:03d}] train_loss={train_loss:.4f} "
              f"val_dice={val_metrics.get('dice', 0):.4f} "
              f"val_iou={val_metrics.get('iou', 0):.4f} "
              f"val_loss={val_metrics['loss']:.4f} ({row['secs']}s)")
        if wandb_run:
            wandb_run.log(row)

        # Checkpoint on best monitored metric
        current = val_metrics.get(monitor, -1.0)
        if current > best_metric:
            best_metric = current
            bad_epochs = 0
            torch.save({"model_state": model.state_dict(), "config": cfg,
                        "epoch": epoch, f"val_{monitor}": current}, best_path)
            print(f"      [BEST] new best val_{monitor}={current:.4f} -> {best_path.name}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[train] early stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    # Write history CSV
    if history:
        with open(history_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
        print(f"[train] history -> {history_path}")
    print(f"[train] done. best val_{monitor}={best_metric:.4f}  weights={best_path}")
    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
