# 🩺 Automated Medical Image Segmentation System

Deep-learning segmentation of **skin lesions** (ISIC-2018) using **U-Net** and
its advanced variants **Attention U-Net** and **U-Net++**, with a Gradio demo
and Grad-CAM explainability.

> Course project — *Deep Learning-Based Segmentation of Medical Images Using U-Net Variants.*
> Research/education prototype. **Not** a medical device.

---

## What this does

Given a skin photo, the system predicts a pixel-level **mask** outlining any
lesion (semantic segmentation). It implements and **compares three architectures**,
reports **Dice / IoU / Precision / Recall / Hausdorff-95**, and ships a
**web app** where you upload an image and see the predicted mask plus a
heatmap explaining *where the model looked*.

---

## 1. Setup

```bash
# from D:\medical-image-segmentation
python -m venv .venv
.venv\Scripts\activate          # Windows PowerShell
pip install -r requirements.txt
```

> GPU strongly recommended for training. CPU works for the app/inference and the
> smoke tests. The code auto-detects CUDA and falls back to CPU.

---

## 2. Get the data

**Recommended — Kaggle (automatic):**

1. Create a free Kaggle account → *Account → Create New API Token*.
2. Put `kaggle.json` in `%USERPROFILE%\.kaggle\kaggle.json`.
3. Run:

```bash
python -m src.download_data --source kaggle
```

This downloads the ISIC-2018 Task-1 images + ground-truth masks and writes
`data/splits/{train,val,test}.csv`.

**Manual alternative:** download Task-1 input/groundtruth from
<https://challenge.isic-archive.com/data/#2018>, unzip into
`data/raw/ISIC2018/images` and `data/raw/ISIC2018/masks`, then:

```bash
python -m src.download_data --source local
```

**No download (verify the pipeline end-to-end):**

```bash
python -m src.download_data --source synthetic   # makes fake lesion images
```

---

## 3. Verify everything works

```bash
python -m tests.test_pipeline      # forward passes, losses, metrics
```

---

## 4. Train

```bash
python -m src.train                          # vanilla U-Net (baseline)
python -m src.train --arch attention_unet    # Attention U-Net
python -m src.train --arch unetpp            # U-Net++
```

Best weights are saved to `checkpoints/best_<arch>.pt`; per-epoch metrics to
`outputs/history_<arch>.csv`. Edit `configs/config.yaml` to change anything
(image size, loss, epochs, encoder, etc.).

---

## 5. Evaluate & compare

```bash
python -m src.evaluate --compare ^
  checkpoints/best_unet.pt ^
  checkpoints/best_attention_unet.pt ^
  checkpoints/best_unetpp.pt
```

Prints the comparison table and saves `outputs/comparison.csv` — drop this
straight into your report.

---

## 6. Run the prototype app

```bash
python app/gradio_app.py --checkpoint checkpoints/best_unet.pt
```

Open the printed local URL, upload any skin photo, and view the mask, overlay,
and Grad-CAM explanation. Test images: any dermoscopy photo, an ISIC sample, or
a clear photo of a mole.

Single-image CLI:

```bash
python -m src.predict --image my_photo.jpg --checkpoint checkpoints/best_unet.pt
```

---

## Project structure

```
medical-image-segmentation/
├── configs/config.yaml      # all hyperparameters (single source of truth)
├── data/                    # raw / processed / splits (gitignored)
├── src/
│   ├── dataset.py           # Dataset, albumentations transforms, split CSVs
│   ├── models.py            # U-Net & U-Net++ (SMP) + custom Attention U-Net
│   ├── losses.py            # Dice, Tversky, BCE+Dice
│   ├── metrics.py           # Dice, IoU, Precision/Recall, Hausdorff-95
│   ├── train.py             # AMP training loop, cosine restarts, early stop
│   ├── evaluate.py          # test-set metrics + model comparison table
│   ├── predict.py           # single-image inference (used by app + CLI)
│   ├── explain.py           # Grad-CAM for segmentation
│   └── download_data.py     # Kaggle / local / synthetic data acquisition
├── app/gradio_app.py        # upload-image → mask web UI
├── notebooks/               # EDA, baseline, comparison
└── tests/test_pipeline.py   # CPU smoke tests
```

---

## How it maps to the report

| Report section | Where it lives |
|---|---|
| Semantic segmentation principles | `dataset.py`, this README |
| Encoder–decoder architectures | `models.py` (U-Net, Attention U-Net, U-Net++) |
| Loss functions | `losses.py` (BCE, Dice, Tversky) |
| Evaluation metrics (Dice, IoU, Hausdorff) | `metrics.py`, `evaluate.py` |
| Clinical relevance | README + app summary |
| Explainability (extension) | `explain.py`, app Grad-CAM toggle |

---

## Target results (ISIC-2018)

| Model | Reported Dice | Target |
|---|---|---|
| U-Net (baseline) | ~0.83 | 0.80–0.84 |
| Attention U-Net | ~0.86 | 0.83–0.87 |
| U-Net++ | ~0.87 | 0.85–0.88 |
