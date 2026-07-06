#!/usr/bin/env python
"""Qualitative evaluation figures (required by Lecture 3b, slide 'Qualitative
evaluation is required'): raw image, ground-truth mask, predicted probability,
and a TP/FP/FN overlay — for the SAME validation tiles across two models, so
that model comparisons are visual as well as numerical.

Tile selection: validation tiles (random-split seed 42, identical to the
notebook protocol that produced the published checkpoints) with the largest
wrinkle_ridge ground-truth area. Ridge is the class where model differences
are actually visible: impact_crater is near-saturated (~85% of pixels
positive, see report Dataset section) and the remaining classes have almost
no positive labels.

Usage (from Moon-Recognition/lunar_segmentation):
    python scripts/qualitative_eval.py

Outputs PNGs to report/figures/MR/.
"""
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

from lunar_segmentation.models.unet import SmallUNet  # noqa: E402
from lunar_segmentation.data.preprocessing import CLASS_NAMES  # noqa: E402

REPO_ROOT = PKG_ROOT.parents[1]
DATA_DIR = REPO_ROOT / "data" / "MR"
CKPT_DIR = DATA_DIR / "results" / "checkpoints"
OUT_DIR = REPO_ROOT / "report" / "figures" / "MR"

# Models compared on the same tiles (both trained 30 epochs on the full set)
MODELS = {
    "BCEDice (baseline)": dict(
        ckpt=CKPT_DIR / "arch_BCEDice_tall_e30.pt",
        kwargs=dict(use_residual=False, dropout=0.0),
    ),
    "+both, no aug (best)": dict(
        ckpt=CKPT_DIR / "aug_off_tall_e30.pt",
        kwargs=dict(use_residual=True, dropout=0.3),
    ),
}

RIDGE = CLASS_NAMES.index("wrinkle_ridge")
CRATER = CLASS_NAMES.index("impact_crater")
N_TILES = 3
THRESHOLD = 0.5
SEED = 42
VAL_SPLIT = 0.2


class _Opaque:
    """Placeholder for unpicklable non-tensor objects inside checkpoints."""
    def __init__(self, *a, **k): pass
    def __setstate__(self, state): pass


class _SkipPandasUnpickler(pickle.Unpickler):
    """Checkpoints embed a pandas metrics DataFrame pickled by a different
    pandas version than the local one; we only need the tensors, so any
    pandas class is replaced by an inert placeholder."""
    def find_class(self, module, name):
        if module.startswith("pandas"):
            return _Opaque
        return super().find_class(module, name)


_pickle_shim = types.ModuleType("pickle_skip_pandas")
_pickle_shim.Unpickler = _SkipPandasUnpickler
_pickle_shim.load = pickle.load
_pickle_shim.loads = pickle.loads
_pickle_shim.UnpicklingError = pickle.UnpicklingError


def load_model(spec, device):
    model = SmallUNet(in_channels=3, num_classes=len(CLASS_NAMES), **spec["kwargs"])
    # weights_only=False: our own checkpoints embed a pandas metrics DataFrame
    weights = torch.load(spec["ckpt"], map_location=device, weights_only=False,
                         pickle_module=_pickle_shim)
    if isinstance(weights, dict):
        for key in ("state_dict", "model"):
            if key in weights:
                weights = weights[key]
                break
    model.load_state_dict({k.replace("model.", "", 1): v for k, v in weights.items()})
    model.to(device).eval()
    return model


def tpfpfn_overlay(ax, gray, gt, pred):
    """Gray image with TP green, FP red, FN blue."""
    ax.imshow(gray, cmap="gray")
    rgba = np.zeros((*gt.shape, 4), dtype=np.float32)
    rgba[(pred == 1) & (gt == 1)] = (0.0, 0.9, 0.2, 0.55)  # TP
    rgba[(pred == 1) & (gt == 0)] = (1.0, 0.1, 0.1, 0.55)  # FP
    rgba[(pred == 0) & (gt == 1)] = (0.2, 0.4, 1.0, 0.55)  # FN
    ax.imshow(rgba)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    df = pd.read_csv(DATA_DIR / "tiles" / "index.csv")
    df["tile_path"] = df["tile_path"].apply(lambda p: str(DATA_DIR / p))

    # Reproduce the notebook's validation split (random permutation, seed 42)
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(df))
    val_idx = idx[: max(1, int(len(idx) * VAL_SPLIT))]

    # Rank validation tiles by ridge ground-truth area
    print("Scanning validation tiles for ridge ground truth…")
    ridge_area = []
    for i in val_idx:
        m = np.load(df.tile_path.iloc[i])["mask"]
        ridge_area.append(int(m[RIDGE].sum()))
    order = np.argsort(ridge_area)[::-1]
    chosen = [val_idx[j] for j in order[:N_TILES]]
    print("Chosen tiles:",
          [(Path(df.tile_path.iloc[i]).stem, ridge_area[j]) for i, j in zip(chosen, order[:N_TILES])])

    models = {name: load_model(spec, device) for name, spec in MODELS.items()}

    for cls_idx, cls_name in [(RIDGE, "wrinkle_ridge"), (CRATER, "impact_crater")]:
        ncols = 2 + 2 * len(models)  # raw, GT, then (prob, TP/FP/FN) per model
        fig, axes = plt.subplots(N_TILES, ncols, figsize=(3.1 * ncols, 3.2 * N_TILES))
        for r, i in enumerate(chosen):
            data = np.load(df.tile_path.iloc[i])
            image, mask = data["image"].astype(np.float32), data["mask"]
            gray, gt = image[0], mask[cls_idx]

            axes[r, 0].imshow(gray, cmap="gray")
            axes[r, 0].set_ylabel(Path(df.tile_path.iloc[i]).stem.replace("marius_hills_", ""),
                                  fontsize=8)
            axes[r, 1].imshow(gray, cmap="gray")
            axes[r, 1].imshow(np.ma.masked_where(gt == 0, gt), cmap="spring", alpha=0.6,
                              vmin=0, vmax=1)

            for m_i, (name, model) in enumerate(models.items()):
                with torch.no_grad():
                    x = torch.from_numpy(image[None]).to(device)
                    prob = torch.sigmoid(model(x))[0, cls_idx].cpu().numpy()
                pred = (prob > THRESHOLD).astype(np.uint8)
                c_prob, c_err = 2 + 2 * m_i, 3 + 2 * m_i
                axes[r, c_prob].imshow(prob, cmap="inferno", vmin=0, vmax=1)
                tpfpfn_overlay(axes[r, c_err], gray, gt, pred)
                if r == 0:
                    axes[r, c_prob].set_title(f"{name}\nprobability", fontsize=9)
                    axes[r, c_err].set_title(f"{name}\nTP/FP/FN", fontsize=9)

        if N_TILES:
            axes[0, 0].set_title("WAC tile (norm channel)", fontsize=9)
            axes[0, 1].set_title("ground truth", fontsize=9)
        for ax in axes.ravel():
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(
            f"{cls_name}: validation tiles, threshold {THRESHOLD} "
            "(TP green, FP red, FN blue)", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = OUT_DIR / f"qualitative_{cls_name}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
