#!/usr/bin/env python
"""Study 3 re-run (augmentation on/off) under the leakage-free spatial split.

Why this script exists: the published augmentation comparison used a random
tile split, which leaks (tiles overlap 50%; 100% of validation tiles overlap
a training tile). This script re-runs the decisive pair — '+both' config with
and without geometric augmentation — using spatial_train_val_split, so train
and validation are strictly disjoint in pixel space. Both arms share the
identical split, model init seed, and schedule: the ONLY difference is
augmentation.

Designed to run locally (Apple Silicon MPS) with a scaled protocol when the
CloudVeneto T4 is unavailable, and at full protocol on the T4 later:

    # local scaled run (e.g. overnight):
    python scripts/train_spatial_ablation.py --tiles 6000 --epochs 15

    # CloudVeneto definitive run:
    python scripts/train_spatial_ablation.py --epochs 30

Scaling note: a paired comparison remains internally valid at reduced
tiles/epochs (both arms see identical data and budget); only the absolute
metric values are expected to differ from the full protocol.

Outputs in data/MR/results/spatial_ablation/:
    <arm>_w<width>_t<tiles>_e<epochs>.pt        final checkpoint
    <arm>_w<width>_t<tiles>_e<epochs>_history.csv  per-epoch per-class metrics
    train.log                                    appended log of all runs
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

PKG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG_ROOT))

from lunar_segmentation.models.unet import SmallUNet  # noqa: E402
from lunar_segmentation.training.trainer import (  # noqa: E402
    Trainer, FocalDiceLoss)
from lunar_segmentation.data.splits import (  # noqa: E402
    spatial_train_val_split, assert_no_overlap)
from lunar_segmentation.data.preprocessing import CLASS_NAMES  # noqa: E402

REPO_ROOT = PKG_ROOT.parents[1]
DATA_DIR = REPO_ROOT / "data" / "MR"
OUT_DIR = DATA_DIR / "results" / "spatial_ablation"

CLASS_WEIGHTS = torch.tensor([1.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0])


def log(msg, fh=None):
    line = f"{time.strftime('%H:%M:%S')}  {msg}"
    print(line, flush=True)
    if fh:
        fh.write(line + "\n")
        fh.flush()


class CachedTileDataset(Dataset):
    """In-RAM tiles. Images cached as float16 (halves RAM, values are in
    [0,1] so the precision loss is negligible and identical for both arms),
    masks as uint8; converted to float32 per item."""

    def __init__(self, tiles, augment=False):
        self.tiles = tiles
        self.augment = augment

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, i):
        img, mask = self.tiles[i]
        img = img.astype(np.float32)
        mask = mask.astype(np.float32)
        if self.augment:
            if np.random.random() < 0.5:
                img, mask = img[:, :, ::-1].copy(), mask[:, :, ::-1].copy()
            if np.random.random() < 0.5:
                img, mask = img[:, ::-1, :].copy(), mask[:, ::-1, :].copy()
            k = np.random.randint(0, 4)
            if k:
                img = np.rot90(img, k=k, axes=(1, 2)).copy()
                mask = np.rot90(mask, k=k, axes=(1, 2)).copy()
        return torch.from_numpy(img), torch.from_numpy(mask)


def load_tiles(df, label, fh):
    log(f"Loading {len(df)} {label} tiles into RAM…", fh)
    tiles = []
    for _, row in df.iterrows():
        d = np.load(row["tile_path"])
        tiles.append((d["image"].astype(np.float16),
                      d["mask"].astype(np.uint8)))
    return tiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", type=int, default=None,
                    help="subsample to N tiles total (proportional across "
                         "train/val, AFTER the spatial split); None = all")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--base-width", type=int, default=32)
    ap.add_argument("--arms", default="noaug,aug",
                    help="comma list among {aug,noaug}")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fh = open(OUT_DIR / "train.log", "a")

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    log(f"=== spatial ablation | device={device} tiles={args.tiles} "
        f"epochs={args.epochs} bs={args.batch_size} w={args.base_width} "
        f"arms={args.arms} seed={args.seed} ===", fh)

    df = pd.read_csv(DATA_DIR / "tiles" / "index.csv")
    df["tile_path"] = df["tile_path"].apply(lambda p: str(DATA_DIR / p))

    # Leakage-free split on the FULL index, then (optionally) subsample each
    # side proportionally — subsampling cannot reintroduce overlap.
    train_df, val_df = spatial_train_val_split(
        df, val_fraction=args.val_fraction, seed=args.seed)
    assert_no_overlap(train_df, val_df)
    if args.tiles is not None:
        frac = min(1.0, args.tiles / (len(train_df) + len(val_df)))
        train_df = train_df.sample(frac=frac, random_state=args.seed)
        val_df = val_df.sample(frac=frac, random_state=args.seed)
    log(f"Split: {len(train_df)} train / {len(val_df)} val "
        f"(spatially disjoint, verified)", fh)

    train_tiles = load_tiles(train_df, "train", fh)
    val_tiles = load_tiles(val_df, "val", fh)
    val_loader = DataLoader(CachedTileDataset(val_tiles, augment=False),
                            batch_size=args.batch_size, shuffle=False)

    tag_sfx = (f"w{args.base_width}"
               f"_t{args.tiles if args.tiles else 'all'}_e{args.epochs}")

    for arm in [a.strip() for a in args.arms.split(",")]:
        augment = (arm == "aug")
        log(f"--- arm '{arm}' (augment={augment}) ---", fh)

        torch.manual_seed(args.seed)          # identical init for both arms
        np.random.seed(args.seed)
        model = SmallUNet(in_channels=3, num_classes=len(CLASS_NAMES),
                          base_width=args.base_width,
                          use_residual=True, dropout=0.3)   # '+both' config
        optim = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optim, T_max=args.epochs, eta_min=1e-5)
        trainer = Trainer(model, optim,
                          FocalDiceLoss(gamma=2.0, alpha=0.25,
                                        class_weights=CLASS_WEIGHTS),
                          device=device, scheduler=sched)

        loader = DataLoader(CachedTileDataset(train_tiles, augment=augment),
                            batch_size=args.batch_size, shuffle=True)

        history = []
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            loss = trainer.train_one_epoch(loader)
            metrics = trainer.evaluate(val_loader)
            metrics["epoch"] = epoch
            metrics["loss"] = loss
            history.append(metrics)
            ridge_ap = float(
                metrics.loc[metrics["class"] == "wrinkle_ridge", "ap"].iloc[0])
            log(f"[{arm}] epoch {epoch}/{args.epochs}  loss={loss:.4f}  "
                f"meanAP={metrics['ap'].mean():.4f}  ridgeAP={ridge_ap:.4f}  "
                f"lr={trainer.current_lr():.2e}  {time.time()-t0:.1f}s", fh)

        ckpt = OUT_DIR / f"{arm}_{tag_sfx}.pt"
        torch.save({"state_dict": model.state_dict(),
                    "config": vars(args), "arm": arm}, ckpt)
        pd.concat(history).to_csv(OUT_DIR / f"{arm}_{tag_sfx}_history.csv",
                                  index=False)
        log(f"[SAVED] {ckpt.name}", fh)

    log("Done.", fh)
    fh.close()


if __name__ == "__main__":
    main()
