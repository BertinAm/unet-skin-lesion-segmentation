# Deploying to Hugging Face Spaces

This serves your Gradio app for free on a public URL. The Space stays small
because it **downloads the trained checkpoints at runtime** from your model repo
(`unixio/unet-skin-lesion-segmentation`) instead of bundling them.

The Space repo needs these at its **root**:
```
app.py              (from spaces/app.py)
requirements.txt    (from spaces/requirements.txt)
README.md           (from spaces/README.md  — has the Space config header)
src/                (the project's src/ folder)
configs/            (the project's configs/ folder)
```

---

## Step 1 — Create the Space

1. Go to <https://huggingface.co/new-space>.
2. **Owner:** unixio · **Space name:** `skin-lesion-segmentation`
3. **SDK:** **Gradio**
4. **Hardware:** CPU basic (free)
5. **Visibility:** Public
6. Click **Create Space**. (It starts empty.)

---

## Step 2 — Assemble the files locally and push

Run these from your project root (`D:\medical-image-segmentation`) in PowerShell.
Replace the URL if your Space name differs.

```powershell
# 1. Clone the empty Space repo somewhere outside your project
cd D:\
git clone https://huggingface.co/spaces/unixio/skin-lesion-segmentation
cd skin-lesion-segmentation

# 2. Copy in the app files + the code the app imports
copy D:\medical-image-segmentation\spaces\app.py .
copy D:\medical-image-segmentation\spaces\requirements.txt .
copy D:\medical-image-segmentation\spaces\README.md .
xcopy /E /I D:\medical-image-segmentation\src src
xcopy /E /I D:\medical-image-segmentation\configs configs

# 3. Commit and push
git add -A
git commit -m "Deploy skin lesion segmentation Gradio app"
git push
```

If `git push` asks for a password, use a **Hugging Face access token** (Settings →
Access Tokens → a *write* token) as the password, with your HF username.

---

## Step 3 — Watch it build

- Go to your Space page. It shows **Building** → installs `requirements.txt`
  (takes a few minutes, mostly PyTorch).
- When it flips to **Running**, your app is live at:
  `https://huggingface.co/spaces/unixio/skin-lesion-segmentation`
- First prediction is slightly slow (it downloads the chosen checkpoint once),
  then it's cached.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| Build fails on torch | Ensure the `--extra-index-url ...cpu` line stays at the top of `requirements.txt`. |
| `ModuleNotFoundError: src` | You forgot to copy the `src/` folder into the Space root. |
| `config.yaml not found` | You forgot to copy the `configs/` folder. |
| App loads but errors on predict | Check the Space **Logs** tab; usually a missing dependency — add it to `requirements.txt`. |
| Want a GPU | Space Settings → Hardware → upgrade (paid). CPU is fine for a demo. |

---

## Updating later

Edit files in the cloned Space folder (or re-copy from your project), then:
```powershell
git add -A && git commit -m "update" && git push
```
The Space rebuilds automatically.
