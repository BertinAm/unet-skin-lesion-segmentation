"""
Gradio prototype: upload a skin image -> see the predicted lesion mask,
an overlay, a Grad-CAM explanation, and quantitative summaries.

Run:
    python app/gradio_app.py
    python app/gradio_app.py --checkpoint checkpoints/best_attention_unet.pt

Then open the printed local URL in your browser.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running as a script: add repo root to path so `src` imports work.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gradio as gr  # noqa: E402

from src.predict import SegmentationPredictor  # noqa: E402
from src.utils import load_config  # noqa: E402


def build_interface(checkpoint: str, config: str):
    predictor = SegmentationPredictor(checkpoint, config)

    def segment(image: np.ndarray, show_explain: bool):
        if image is None:
            return None, None, None, "Please upload an image."
        overlay, mask, area = predictor.predict_overlay(image)
        mask_vis = (mask * 255).astype(np.uint8)

        cam_img = None
        if show_explain:
            try:
                from src.explain import gradcam_overlay
                cam_img = gradcam_overlay(predictor, image)
            except Exception as e:
                cam_img = None
                print("[gradcam] failed:", e)

        # Simple clinical-style summary (NOT a diagnosis).
        if area < 0.005:
            verdict = "No significant lesion region detected."
        else:
            verdict = (f"Lesion region detected.\n"
                       f"Estimated lesion area: {area * 100:.1f}% of image.\n"
                       f"(Decision-support only — not a diagnosis.)")
        return overlay, mask_vis, cam_img, verdict

    with gr.Blocks(title="Medical Image Segmentation — Skin Lesions") as demo:
        gr.Markdown(
            "# 🩺 Skin Lesion Segmentation\n"
            "Upload a dermoscopy / skin photo. The U-Net model outlines the "
            "lesion. Toggle **Explain** for a Grad-CAM heatmap showing where the "
            "model looked.\n\n"
            "> ⚠️ Research/education prototype. **Not** a medical device.")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="numpy", label="Input image")
                explain = gr.Checkbox(label="Explain (Grad-CAM)", value=True)
                btn = gr.Button("Segment", variant="primary")
            with gr.Column():
                out_overlay = gr.Image(label="Lesion overlay")
                out_mask = gr.Image(label="Predicted mask")
                out_cam = gr.Image(label="Grad-CAM explanation")
                out_text = gr.Textbox(label="Summary", lines=4)

        btn.click(segment, [inp, explain],
                  [out_overlay, out_mask, out_cam, out_text])

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--share", action="store_true",
                        help="create a public Gradio link")
    args = parser.parse_args()

    cfg = load_config(args.config)
    checkpoint = args.checkpoint or cfg["inference"]["checkpoint"]
    demo = build_interface(checkpoint, args.config)
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
