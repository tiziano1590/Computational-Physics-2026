#!/usr/bin/env python
"""Figures for the crater-method comparison.

Produces:
  fig_metrics.png       — F1 / IoU / MCC bars (fair band) + predict-all reference.
  fig_qualitative.png   — grayscale / GT / each method's binary crater mask on sample tiles.

Run after compare_models.py has written the CSVs:
  KMP_DUPLICATE_LIB_OK=TRUE python Moon-Recognition/comparison/make_figures.py
"""
import os, sys, glob, warnings, logging, numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "Moon-Recognition/lunar_segmentation"))
sys.path.insert(0, HERE)
import compare_models as C
from lunar_segmentation.models.unet import SmallUNet
from lunar_segmentation.inference.predictor import Predictor
from lunar_segmentation.data.preprocessing import CLASS_NAMES
import maskrcnn_loader
from ultralytics import YOLO

NAVY, ORANGE, BLUE, GREY = "#102A43", "#E87A2B", "#1F6FB2", "#9aa5b1"


def metrics_figure():
    fb = pd.read_csv(os.path.join(HERE, "comparison_results_fairband.csv"))
    fb = fb.set_index("method")
    order = ["Classical (thr+morph)", "YOLOv8 (detection)", "Mask R-CNN (instance)", "U-Net (semantic)"]
    order = [m for m in order if m in fb.index]
    base = fb.loc["Predict-all (baseline)"] if "Predict-all (baseline)" in fb.index else None
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, title in zip(axes, ["f1", "iou", "mcc"], ["Pixel F1", "Pixel IoU", "MCC (prevalence-robust)"]):
        vals = [fb.loc[m, metric] for m in order]
        ax.barh(order, vals, color=[GREY, BLUE, ORANGE, NAVY][:len(order)])
        if base is not None and metric != "mcc":
            ax.axvline(base[metric], color="red", ls="--", lw=1.2, label=f"predict-all={base[metric]:.2f}")
            ax.legend(fontsize=8, loc="lower right")
        ax.set_title(title); ax.set_xlim(0, 1 if metric != "mcc" else max(0.3, max(vals) * 1.2))
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=9)
    fig.suptitle("Crater detection on the fair band (crater coverage 1–20%, leakage-free spatial val)", fontsize=11)
    fig.tight_layout()
    p = os.path.join(HERE, "fig_metrics.png"); fig.savefig(p, dpi=140); print("wrote", p)


def qualitative_figure(n=3, seed=1):
    df = pd.read_csv(os.path.join(ROOT, "data/MR/tiles/index.csv"))
    from lunar_segmentation.data.splits import spatial_train_val_split
    _, val = spatial_train_val_split(df, 0.2, 256, 1024, 42)
    frac = val["positive_pixels"] / 65536.0
    val = val[(frac >= 0.03) & (frac < 0.20)].reset_index(drop=True)
    rows = val.iloc[np.random.default_rng(seed).permutation(len(val))[:n]]

    dev = C.DEV
    unet = Predictor(SmallUNet(3, len(CLASS_NAMES)), weights_path=os.path.join(ROOT, "data/MR/weights/best_trained.pth"), device=dev)
    mrcnn, names, _ = maskrcnn_loader.build_and_load(os.path.join(ROOT, "data/MR/weights/best_model.pth"), device=dev)
    clab = names.index("impact_crater")
    yolo = YOLO(sorted(glob.glob(os.path.join(ROOT, "Moon-Recognition/yolo/YOLO/runs/detect/train*/weights/best.pt")))[-1])
    fb = pd.read_csv(os.path.join(HERE, "comparison_results_fairband.csv")).set_index("method")["tau"]

    cols = ["Tile", "GT crater", "U-Net", "Mask R-CNN", "YOLOv8", "Classical"]
    fig, axes = plt.subplots(n, len(cols), figsize=(2.1 * len(cols), 2.1 * n))
    for r, (_, row) in enumerate(rows.iterrows()):
        d = np.load(os.path.join(ROOT, "data/MR", row["tile_path"]))
        img = d["image"].astype(np.float32); gt = d["mask"][0] > 0
        g = img[0]
        masks = [
            g,
            gt,
            C.binmask_at(C.unet_repr(unet, img)[0], fb["U-Net (semantic)"]),
            C.binmask_at(C.maskrcnn_repr(mrcnn, img, clab)[0], fb["Mask R-CNN (instance)"]),
            C.binmask_at(C.yolo_repr(yolo, img)[0], fb["YOLOv8 (detection)"]),
            C.binmask_at(C.classical_repr(img)[0], 0.5),
        ]
        for c, (title, m) in enumerate(zip(cols, masks)):
            ax = axes[r, c] if n > 1 else axes[c]
            ax.imshow(m, cmap="gray" if c == 0 else "viridis"); ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=10)
    fig.suptitle("Qualitative crater predictions (fair-band tiles)", fontsize=11)
    fig.tight_layout()
    p = os.path.join(HERE, "fig_qualitative.png"); fig.savefig(p, dpi=130); print("wrote", p)


if __name__ == "__main__":
    metrics_figure()
    qualitative_figure()
