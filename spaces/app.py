"""
Hugging Face Space entry point — Skin Lesion Segmentation.

Downloads the trained checkpoints from the HF model repo at startup (so the
Space stays small) and serves a Gradio UI: upload an image -> predicted mask,
overlay, and Grad-CAM explanation.

Expects this file at the Space root, alongside the project's `src/` and
`configs/` folders (see DEPLOY.md).
"""
import os

import cv2
import gradio as gr
import numpy as np
from huggingface_hub import hf_hub_download

from src.predict import SegmentationPredictor
from src.utils import overlay_mask

# Your uploaded model repo.
MODEL_REPO = os.environ.get("MODEL_REPO", "unixio/unet-skin-lesion-segmentation")

# Display name -> checkpoint filename in the model repo.
CHECKPOINTS = {
    "U-Net++ (best, Dice 0.905)": "best_unetpp.pt",
    "U-Net (baseline, Dice 0.906)": "best_unet.pt",
    "Attention U-Net (Dice 0.882)": "best_attention_unet.pt",
}

# Lazy cache so each model is downloaded/loaded only once.
_predictors: dict[str, SegmentationPredictor] = {}


def get_predictor(model_name: str) -> SegmentationPredictor:
    if model_name not in _predictors:
        ckpt_path = hf_hub_download(repo_id=MODEL_REPO, filename=CHECKPOINTS[model_name])
        _predictors[model_name] = SegmentationPredictor(ckpt_path, "configs/config.yaml")
    return _predictors[model_name]


def _describe_shape(mask: np.ndarray) -> str:
    """Rough, honest shape description from the binary mask."""
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    n_regions = max(num - 1, 0)  # exclude background
    if n_regions == 0:
        return ""
    # Largest region (skip background row 0).
    areas = stats[1:, cv2.CC_STAT_AREA]
    big = 1 + int(np.argmax(areas))
    x, y, w, h, a = stats[big]
    aspect = max(w, h) / max(min(w, h), 1)
    # Compactness: region area vs its bounding box (1.0 = fills the box).
    fill = a / max(w * h, 1)
    if fill > 0.78:
        border = "fairly regular / round"
    elif fill > 0.55:
        border = "moderately irregular"
    else:
        border = "irregular / ragged"
    elong = " and elongated" if aspect > 2.2 else ""
    multi = (f" The model found {n_regions} separate regions."
             if n_regions > 1 else "")
    return (f"The main region has a **{border}** border{elong}.{multi}")


def build_summary(model_name: str, area: float, confidence: float,
                  mask: np.ndarray) -> str:
    pct = area * 100.0

    if area < 0.005:
        return (
            "### 🔍 Result: no lesion region found\n"
            "The model did not highlight any significant region. If you expected "
            "a lesion, try a sharper, well-lit **close-up** of the area.\n\n"
            f"*Model: {model_name}. Research/education only — not a diagnosis.*"
        )

    # Size category
    if pct < 5:
        size = "small"
    elif pct < 20:
        size = "moderate"
    elif pct < 50:
        size = "large"
    else:
        size = "very large (covers most of the image)"

    # Confidence wording
    if confidence >= 0.85:
        conf_word = "high"
    elif confidence >= 0.65:
        conf_word = "moderate"
    else:
        conf_word = "low"

    lines = [
        "### 🔍 Segmentation result",
        f"- **Lesion region detected**, covering **{pct:.1f}%** of the image ({size}).",
        f"- **Model confidence** in that region: **{confidence * 100:.0f}% ({conf_word})**.",
        f"- {_describe_shape(mask)}",
        f"- Model used: **{model_name}**.",
    ]

    # Large-area note — informational, not a verdict. The model is most precise on
    # dermoscopy close-ups; on wider phone photos a big region may be real OR may
    # include healthy skin, so we flag it as "treat as approximate".
    if pct > 45:
        lines += [
            "",
            "> ℹ️ **Note on a large highlighted area:** this model was trained on "
            "**close-up dermoscopy images of a single lesion**, so it is most precise "
            "on tight close-ups. On a wider photo (a full face or limb) a large "
            "region can be genuinely affected skin **or** may include some healthy "
            "skin — treat the exact boundary as approximate. For the sharpest "
            "boundary, try a **cropped close-up** of the area of concern.",
        ]

    lines += [
        "",
        "**🌡️ How to read the Grad-CAM heatmap:** warm colours (red/orange) mark the "
        "pixels that most pushed the model toward 'lesion'; cool colours (blue) had "
        "little influence. It shows *where* the model looked, helping you judge "
        "whether it focused on the actual lesion or got distracted by hair, "
        "shadows, or the image border.",
        "",
        "**🩺 Clinical note:** this localises *where* a lesion is, not *what* it is "
        "(benign vs. malignant). It is a research/education prototype trained on "
        "ISIC-2018 — **not** a medical device. Any real skin concern should be seen "
        "by a qualified dermatologist.",
    ]
    return "\n".join(lines)


def segment(image: np.ndarray, model_name: str, show_explain: bool):
    if image is None:
        return None, None, None, "Please upload an image first."

    predictor = get_predictor(model_name)
    # Predict once; reuse mask + probability map for overlay, area, confidence.
    mask, prob = predictor.predict(image)
    overlay = overlay_mask(image, mask)
    mask_vis = (mask * 255).astype(np.uint8)
    area = float(mask.mean())
    # Mean probability inside the predicted region = how confident the model is.
    confidence = float(prob[mask > 0].mean()) if mask.any() else 0.0

    cam_img = None
    if show_explain:
        try:
            from src.explain import gradcam_overlay
            cam_img = gradcam_overlay(predictor, image)
        except Exception as e:  # Grad-CAM is best-effort
            print("[gradcam] failed:", e)

    summary = build_summary(model_name, area, confidence, mask)
    return overlay, mask_vis, cam_img, summary


with gr.Blocks(title="Skin Lesion Segmentation") as demo:
    gr.Markdown(
        "# 🩺 Skin Lesion Segmentation (U-Net / Attention U-Net / U-Net++)\n"
        "Upload a dermoscopy or skin photo. The model outlines the lesion and "
        "(optionally) shows a **Grad-CAM** heatmap of where it looked.\n\n"
        "> ⚠️ Research/education prototype trained on ISIC-2018. **Not** a medical device."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Image(type="numpy", label="Input image")
            model_choice = gr.Dropdown(
                choices=list(CHECKPOINTS.keys()),
                value="U-Net++ (best, Dice 0.905)",
                label="Model",
            )
            explain = gr.Checkbox(label="Explain (Grad-CAM)", value=True)
            btn = gr.Button("Segment", variant="primary")
        with gr.Column():
            out_overlay = gr.Image(label="Lesion overlay")
            out_mask = gr.Image(label="Predicted mask")
            out_cam = gr.Image(label="Grad-CAM explanation")
            out_text = gr.Markdown(label="Summary")

    btn.click(segment, [inp, model_choice, explain],
              [out_overlay, out_mask, out_cam, out_text])

    gr.Markdown(
        "Models & code: "
        "[GitHub](https://github.com/BertinAm/unet-skin-lesion-segmentation) · "
        "[Hugging Face](https://huggingface.co/unixio/unet-skin-lesion-segmentation)"
    )


if __name__ == "__main__":
    demo.launch()
