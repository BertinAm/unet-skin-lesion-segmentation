"""Generate report/PPT figures from outputs/comparison.csv."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "outputs" / "comparison.csv")
labels = {"unet": "U-Net", "attention_unet": "Attention U-Net", "unetpp": "U-Net++"}
df["name"] = df["arch"].map(labels)
order = ["U-Net", "Attention U-Net", "U-Net++"]
df = df.set_index("name").loc[order].reset_index()
assets = ROOT / "report" / "assets"
assets.mkdir(parents=True, exist_ok=True)

COL = ["#4C72B0", "#DD8452", "#55A868"]

# 1. Grouped bar: Dice/IoU/Precision/Recall
metrics = ["dice", "iou", "precision", "recall"]
mlabels = ["Dice", "IoU", "Precision", "Recall"]
x = np.arange(len(metrics)); w = 0.25
fig, ax = plt.subplots(figsize=(9, 5))
for i, (_, row) in enumerate(df.iterrows()):
    vals = [row[m] for m in metrics]
    bars = ax.bar(x + (i - 1) * w, vals, w, label=row["name"], color=COL[i])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.005, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(mlabels)
ax.set_ylim(0.7, 1.0); ax.set_ylabel("Score")
ax.set_title("Model comparison on ISIC-2018 test set (390 images)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(assets / "metrics_comparison.png", dpi=200); plt.close()

# 2. HD95 (lower better)
fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(df["name"], df["hd95"], color=COL)
for b, v in zip(bars, df["hd95"]):
    ax.text(b.get_x()+b.get_width()/2, v+0.2, f"{v:.2f}", ha="center", fontsize=10)
ax.set_ylabel("HD95 (pixels) — lower is better")
ax.set_title("Boundary accuracy (95th-percentile Hausdorff)")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout(); plt.savefig(assets / "hd95_comparison.png", dpi=200); plt.close()

print("figures written to", assets)
for f in sorted(assets.glob("*.png")):
    print(" -", f.name)
