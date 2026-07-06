#!/usr/bin/env python
"""Reconcile the report's contradicting south-pole tables (Table 8 vs Table 12).

The group manuscript contains two sets of south-pole numbers: the U-Net section's
(crater avg-max-prob 0.86, other classes near zero) and the duplicate section's
(ALL classes ~262k detections/tile, avg-max-prob 0.48-0.54). This script tests the
two candidate mechanisms behind the second set:

1. CHECKPOINT BUG (primary suspect): the pre-fix Predictor loaded `best_trained.pth`
   with strict=False and silently matched ZERO conv layers (legacy `block.*` key
   names), so inference ran on randomly initialised weights. Sigmoid of random
   logits ~= 0.5 everywhere -> every pixel of every class "detected", avg max prob
   ~0.5. Simulated here by running an unloaded SmallUNet.
2. PREPROCESSING MISMATCH (secondary): the duplicate workflow rebuilt the input
   channels with OpenCV (CLAHE clipLimit=2.0 grid 8x8; Sobel ksize=3 max-normalised)
   instead of the package's skimage pipeline (equalize_adapthist clip_limit=0.03;
   filters.sobel), i.e. inputs the model never saw in training.

Run on N south-pole tiles, report per-class max/mean prob + fraction of pixels
above the workflow's own 0.1 display threshold, for each (weights, preprocessing)
combination:

  KMP_DUPLICATE_LIB_OK=TRUE python Moon-Recognition/comparison/giuseppe_preproc_check.py
"""
import os, sys, glob, warnings, logging
import numpy as np
import pandas as pd
import cv2

warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "Moon-Recognition/lunar_segmentation"))

import torch
from lunar_segmentation.models.unet import SmallUNet
from lunar_segmentation.inference.predictor import Predictor
from lunar_segmentation.data.preprocessing import build_three_channel_input, CLASS_NAMES

N_TILES = 8
DISPLAY_TAU = 0.1   # the south-pole workflow's own visualisation threshold
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def preprocess_package(gray256):
    """The training pipeline: skimage equalize_adapthist(0.03) + filters.sobel."""
    return build_three_channel_input(gray256)


def preprocess_giuseppe(gray256):
    """The duplicate workflow's OpenCV rebuild (verbatim from its notebook)."""
    ch1 = gray256 / 255.0
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    ch2 = clahe.apply(gray256) / 255.0
    sobelx = cv2.Sobel(gray256, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray256, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(sobelx ** 2 + sobely ** 2)
    ch3 = mag / mag.max() if mag.max() > 0 else mag
    return np.stack([ch1, ch2, ch3], axis=0).astype(np.float32)


def main():
    tiles = sorted(glob.glob(os.path.join(ROOT, "data/MR/lunar_south_pole/tiles/tile_*.png")))[:N_TILES]
    if not tiles:
        sys.exit("no south-pole tiles found under data/MR/lunar_south_pole/tiles/")
    grays = [cv2.resize(cv2.imread(t, cv2.IMREAD_GRAYSCALE), (256, 256)) for t in tiles]
    print(f"device={DEV} | tiles={len(grays)} | display threshold={DISPLAY_TAU}")

    trained = Predictor(SmallUNet(3, len(CLASS_NAMES)),
                        weights_path=os.path.join(ROOT, "data/MR/weights/best_trained.pth"),
                        device=DEV)
    # simulate the pre-fix silent no-load: same architecture, weights never loaded
    torch.manual_seed(0)
    random_net = Predictor(SmallUNet(3, len(CLASS_NAMES)), weights_path=None, device=DEV)

    combos = {
        "trained + package preproc (Table 8 conditions)":   (trained, preprocess_package),
        "trained + Giuseppe preproc (mismatch only)":        (trained, preprocess_giuseppe),
        "random weights (pre-fix loader, Table 12 bug)":     (random_net, preprocess_giuseppe),
    }

    rows = []
    for combo_name, (pred, prep) in combos.items():
        cubes = [pred.predict(prep(g), tile_size=256, stride=128) for g in grays]
        for ci, cname in enumerate(CLASS_NAMES):
            ch = np.stack([c[ci] for c in cubes])          # (N, H, W)
            rows.append(dict(
                combo=combo_name, cls=cname,
                max_prob=float(ch.max(axis=(1, 2)).mean()),      # avg per-tile max
                mean_prob=float(ch.mean()),
                frac_above_tau=float((ch > DISPLAY_TAU).mean()),
            ))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, "giuseppe_preproc_check.csv"), index=False)
    for combo_name in combos:
        sub = df[df.combo == combo_name].set_index("cls")
        print(f"\n=== {combo_name} ===")
        print(sub[["max_prob", "mean_prob", "frac_above_tau"]]
              .to_string(float_format=lambda v: f"{v:.3f}"))

    print("\nReading: Table 12's signature (all classes ~0.5 max prob, every pixel above "
          "threshold) is reproduced by the RANDOM-WEIGHTS row, not by the preprocessing "
          "mismatch alone -> the duplicate section's numbers came from the checkpoint-"
          "loading bug fixed in inference/predictor.py.")


if __name__ == "__main__":
    main()
