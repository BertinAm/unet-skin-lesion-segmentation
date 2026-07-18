---
title: Skin Lesion Segmentation
emoji: 🩺
colorFrom: pink
colorTo: red
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
---

# 🩺 Skin Lesion Segmentation

Deep-learning segmentation of skin lesions with **U-Net**, **Attention U-Net**,
and **U-Net++**, trained on the ISIC-2018 dataset. Upload a skin/dermoscopy
image to see the predicted lesion mask, an overlay, and a Grad-CAM explanation.

| Model | Test Dice | IoU | HD95 |
|---|---|---|---|
| U-Net (ResNet34) | 0.906 | 0.829 | 12.46 |
| U-Net++ (ResNet34) | 0.905 | 0.829 | 12.80 |
| Attention U-Net | 0.882 | 0.792 | 16.71 |

Checkpoints are pulled at runtime from
[unixio/unet-skin-lesion-segmentation](https://huggingface.co/unixio/unet-skin-lesion-segmentation).

> ⚠️ Research/education prototype. **Not** a medical device.

Code: [github.com/BertinAm/unet-skin-lesion-segmentation](https://github.com/BertinAm/unet-skin-lesion-segmentation)
