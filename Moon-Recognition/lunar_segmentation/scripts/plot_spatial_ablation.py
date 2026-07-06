#!/usr/bin/env python
"""Plot aug-vs-no-aug validation curves from the spatial-split ablation runs.

Reads the per-epoch history CSVs written by train_spatial_ablation.py and
produces, for each (tiles, epochs) protocol found:
  - spatial_<tag>_val_ap.png    mean AP and wrinkle_ridge AP vs epoch
  - spatial_<tag>_val_f1.png    mean F1 and wrinkle_ridge F1 vs epoch
plus a printed final-epoch summary table. Copies the figures to
report/figures/MR/ for direct inclusion in the report.
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RES_DIR = REPO_ROOT / "data" / "MR" / "results" / "spatial_ablation"
FIG_DIR = REPO_ROOT / "report" / "figures" / "MR"

ARM_STYLE = {"noaug": ("no augmentation", "#2c7fb8", "-"),
             "aug": ("with augmentation", "#d95f0e", "--")}


def main():
    hist = {}
    for f in sorted(RES_DIR.glob("*_history.csv")):
        m = re.match(r"(aug|noaug)_(w\d+_t\w+_e\d+)_history\.csv", f.name)
        if m:
            hist.setdefault(m.group(2), {})[m.group(1)] = pd.read_csv(f)

    if not hist:
        sys.exit(f"no history CSVs found in {RES_DIR}")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for tag, arms in hist.items():
        for metric in ("ap", "f1"):
            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            for arm, df in arms.items():
                label, color, ls = ARM_STYLE[arm]
                g = df.groupby("epoch")
                mean_m = g[metric].mean()
                ridge_m = df[df["class"] == "wrinkle_ridge"].set_index("epoch")[metric]
                axes[0].plot(mean_m.index, mean_m, color=color, ls=ls, lw=2, label=label)
                axes[1].plot(ridge_m.index, ridge_m, color=color, ls=ls, lw=2, label=label)
            axes[0].set_title(f"mean {metric.upper()} (macro, classes with support)")
            axes[1].set_title(f"wrinkle_ridge {metric.upper()}")
            for ax in axes:
                ax.set_xlabel("epoch"); ax.set_ylabel(metric.upper())
                ax.legend(); ax.grid(alpha=0.3)
            fig.suptitle(f"Spatial-split augmentation ablation — {tag}")
            fig.tight_layout()
            out = RES_DIR / f"spatial_{tag}_val_{metric}.png"
            fig.savefig(out, dpi=140); plt.close(fig)
            (FIG_DIR / out.name).write_bytes(out.read_bytes())
            print(f"saved {out} (+ copy in report/figures/MR)")

        print(f"\n=== final-epoch summary — {tag} ===")
        for arm, df in arms.items():
            last = df[df["epoch"] == df["epoch"].max()]
            ridge = last[last["class"] == "wrinkle_ridge"]
            print(f"{arm:>6}: meanAP={last['ap'].mean():.4f}  "
                  f"ridgeAP={float(ridge['ap'].iloc[0]):.4f}  "
                  f"meanF1={last['f1'].mean():.4f}  "
                  f"ridgeF1={float(ridge['f1'].iloc[0]):.4f}")


if __name__ == "__main__":
    main()
