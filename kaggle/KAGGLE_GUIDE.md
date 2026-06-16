# Running this project on Kaggle (free GPU)

Kaggle gives you a free NVIDIA P100/T4 GPU (~30 hrs/week). On GPU each epoch
drops from minutes (CPU) to seconds, so all three models train in well under
an afternoon.

There are two ways to get your code onto Kaggle. **Method A (upload code as a
Dataset)** is the most reliable and needs no GitHub. Method B is faster if you
already pushed to GitHub.

---

## STEP 0 — One-time Kaggle setup

1. Create a free account at <https://www.kaggle.com>.
2. **Verify your phone number** (Settings → Phone). This is required to enable
   GPU and internet in notebooks.

---

## STEP 1 — Add the ISIC-2018 dataset to your notebook

You don't download it manually — you *attach* it.

1. Go to <https://www.kaggle.com/code> → **New Notebook**.
2. Right panel → **Add Input** → search:
   `isic2018-challenge-task1-data-segmentation`
   (publisher *tschandl*). Click **Add**.
   - It mounts read-only at:
     `/kaggle/input/isic2018-challenge-task1-data-segmentation/`
3. (Alternative dataset if that one is unavailable: search
   `skin-cancer-mnist-ham10000` or `isic-2018-task-1-2-data` and adjust paths.)

> The exact subfolder names vary by dataset. The notebook's **Cell 2** prints
> the folder tree so you can copy the right image/mask paths into the config.

---

## STEP 2 — Get your code onto Kaggle

### Method A — Upload code as a Kaggle Dataset (recommended)

1. On your PC, zip the project **code only** (not `.venv`, not `data/`):
   - Easiest: zip the whole `D:\medical-image-segmentation` folder, then
     delete `.venv`, `data`, `checkpoints`, `outputs` from the zip — or just
     zip the `src/`, `app/`, `configs/`, `tests/` folders + `requirements.txt`.
2. Go to <https://www.kaggle.com/datasets> → **New Dataset** → upload the zip →
   name it e.g. `medical-seg-code` → **Create**.
3. In your notebook: **Add Input** → search your dataset name → **Add**.
   - It mounts at `/kaggle/input/medical-seg-code/`.
4. Use **Cell 3A** below to copy it into the writable working dir.

> To update code later: edit the Dataset → **New Version** → re-upload zip.

### Method B — Clone from GitHub

1. Push `medical-image-segmentation` to a GitHub repo.
2. Enable notebook internet: right panel → **Settings → Internet → On**.
3. Use **Cell 3B** below.

---

## STEP 3 — Turn on the GPU

Right panel → **Settings → Accelerator → GPU T4 x2** (or P100).
Confirm with the first line of Cell 1 (`nvidia-smi`).

---

## STEP 4 — Run the notebook

Open `kaggle_notebook.ipynb` from this folder (upload it as your notebook, or
copy the cells below). Run cells top to bottom. Outline:

| Cell | What it does |
|---|---|
| 1 | Confirm GPU, install `segmentation-models-pytorch`, `albumentations`, etc. |
| 2 | Print the ISIC dataset folder tree (so you can set correct paths) |
| 3A/3B | Bring your code into `/kaggle/working/medseg` |
| 4 | Point `config.yaml` at the Kaggle ISIC paths + build train/val/test splits |
| 5 | Train U-Net (`device: cuda` auto-detected) |
| 6 | Train Attention U-Net and U-Net++ |
| 7 | Evaluate + produce the comparison table (`outputs/comparison.csv`) |
| 8 | Visualise predictions; download the best checkpoint |

---

## STEP 5 — Get your trained model back

After training, the best weights live in `/kaggle/working/medseg/checkpoints/`.
- Right panel → **Output** → download `best_unet.pt` etc., **or**
- "Save Version" the notebook so outputs persist, then download later.

Bring `best_*.pt` back to `D:\medical-image-segmentation\checkpoints\` and run
the Gradio app locally:

```powershell
python app/gradio_app.py --checkpoint checkpoints/best_attention_unet.pt
```

---

## Tips / gotchas

- **Session limit:** Kaggle GPU sessions run up to 12 h, but idle-disconnect
  after ~20–40 min of no interaction. For long runs keep the tab active or use
  "Save Version → Save & Run All (Commit)" to run headless in the background.
- **Reduce epochs first:** do a 3-epoch run to confirm Dice is climbing before
  committing to 60.
- **Batch size:** T4/P100 (16 GB) handle `batch_size: 16` at 256×256 fine. Bump
  to 24–32 if you have headroom.
- **Reproducibility:** the seed is fixed in `config.yaml` (`project.seed: 42`).
