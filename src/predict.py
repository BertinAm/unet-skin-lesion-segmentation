"""
Inference utilities: load a checkpoint once, predict a mask for any RGB image.
Used by both the CLI and the Gradio app.

CLI:
    python -m src.predict --image path/to/photo.jpg --checkpoint checkpoints/best_unet.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from .dataset import preprocess_single
from .models import build_model
from .utils import get_device, load_config, overlay_mask, resolve


class SegmentationPredictor:
    """Wraps a trained model for single-image inference."""

    def __init__(self, checkpoint, config="configs/config.yaml", device=None):
        self.cfg = load_config(config)
        self.device = device or get_device(self.cfg["project"]["device"])
        ckpt = torch.load(resolve(checkpoint), map_location=self.device)
        # Copy the checkpoint's config and drop pretrained-encoder weights: at
        # inference we immediately overwrite all weights with the trained
        # state_dict, so re-downloading ImageNet weights would be wasted work.
        self.ckpt_cfg = dict(ckpt.get("config", self.cfg))
        self.ckpt_cfg["model"] = dict(self.ckpt_cfg["model"])
        self.ckpt_cfg["model"]["encoder_weights"] = None
        self.model = build_model(self.ckpt_cfg).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.image_size = self.cfg["data"]["image_size"]
        self.threshold = self.cfg["inference"]["threshold"]

    @torch.no_grad()
    def predict(self, image_rgb: np.ndarray):
        """Return (binary_mask, probability_map) at the ORIGINAL image size."""
        h, w = image_rgb.shape[:2]
        x = preprocess_single(image_rgb, self.image_size).to(self.device)
        logits = self.model(x)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()        # (H,W) at model size
        prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = (prob > self.threshold).astype(np.uint8)
        return mask, prob

    def predict_overlay(self, image_rgb: np.ndarray):
        """Return (overlay_image, mask, lesion_area_fraction)."""
        mask, prob = self.predict(image_rgb)
        overlay = overlay_mask(image_rgb, mask)
        area_fraction = float(mask.mean())
        return overlay, mask, area_fraction


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="checkpoints/best_unet.pt")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--out", default="outputs/prediction.png")
    args = parser.parse_args()

    predictor = SegmentationPredictor(args.checkpoint, args.config)
    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(args.image)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    overlay, mask, area = predictor.predict_overlay(rgb)
    out_path = resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_path.with_name(out_path.stem + "_mask.png")), mask * 255)
    print(f"lesion area fraction: {area:.3f}")
    print(f"saved overlay -> {out_path}")


if __name__ == "__main__":
    main()
