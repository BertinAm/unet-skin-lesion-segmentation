"""
Explainability via Grad-CAM for segmentation.

For segmentation we define the CAM "score" as the sum of the predicted lesion
probabilities — so the heatmap highlights the image regions that most increased
the model's belief that a lesion is present. This answers the report's question:
"why did the model segment *here*?"

Works with SMP U-Net / U-Net++ (targets the encoder's last stage) and the
custom Attention U-Net (targets the bottleneck).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from .dataset import preprocess_single, IMAGENET_MEAN, IMAGENET_STD


class _SegmentationTarget:
    """Grad-CAM target: total foreground probability mass."""
    def __call__(self, model_output):
        return torch.sigmoid(model_output).sum()


def _pick_target_layer(model):
    """Choose a sensible conv layer to attribute against, per architecture."""
    # SMP models expose `.encoder`.
    if hasattr(model, "encoder"):
        enc = model.encoder
        # The last module with parameters in the encoder.
        convs = [m for m in enc.modules() if isinstance(m, torch.nn.Conv2d)]
        if convs:
            return convs[-1]
    # Custom Attention U-Net: use the bottleneck block.
    if hasattr(model, "bottleneck"):
        convs = [m for m in model.bottleneck.modules() if isinstance(m, torch.nn.Conv2d)]
        if convs:
            return convs[-1]
    raise ValueError("Could not locate a target layer for Grad-CAM.")


def gradcam_overlay(predictor, image_rgb: np.ndarray) -> np.ndarray:
    """Return an RGB uint8 image with the Grad-CAM heatmap overlaid."""
    model = predictor.model
    device = predictor.device
    size = predictor.image_size

    input_tensor = preprocess_single(image_rgb, size).to(device)
    target_layer = _pick_target_layer(model)

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=[_SegmentationTarget()])[0]   # (H,W) in [0,1]

    # Build a normalised float version of the (resized) input for blending.
    import cv2
    resized = cv2.resize(image_rgb, (size, size)).astype(np.float32) / 255.0
    overlay = show_cam_on_image(resized, grayscale_cam, use_rgb=True)

    # Resize heatmap overlay back to original resolution.
    h, w = image_rgb.shape[:2]
    return cv2.resize(overlay, (w, h))
