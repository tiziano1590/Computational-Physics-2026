"""Train/validation splitting strategies for tiled rasters.

Why this module exists
----------------------
Tiles are cut with a sliding window (tile_size=256, stride=128), so adjacent
tiles share 50% of their pixels.  A *random* tile-level split therefore puts
tiles in the validation set that overlap training tiles almost everywhere:
validation partially measures memorisation of pixels seen during training,
not generalisation.  This inflates validation scores and, crucially, biases
any comparison between a memorisation-prone setup and a regularised one
(e.g. the augmentation ablation: a no-augmentation model can memorise tile
appearance and is rewarded for it on a leaky split).

`spatial_train_val_split` fixes this by assigning contiguous *spatial blocks*
to the validation set and then dropping every training tile whose footprint
overlaps any validation tile.  Train and validation are then strictly
disjoint in pixel space.
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def spatial_train_val_split(
    index_df: pd.DataFrame,
    val_fraction: float = 0.2,
    tile_size: int = 256,
    block_px: int = 1024,
    seed: int = 42,
):
    """Leakage-free split of a tile index into train/validation sets.

    Tiles are grouped into square spatial blocks of ``block_px`` pixels
    (per AOI, using the tile origin ``row``/``col`` columns).  Whole blocks
    are assigned to validation until ``val_fraction`` of the tiles is
    reached.  Finally, any *training* tile whose (tile_size x tile_size)
    footprint intersects a validation tile is dropped, so no pixel appears
    in both sets.

    Args:
        index_df:     tile index with columns ['aoi', 'row', 'col', ...]
                      (as produced by save_tiles_for_aoi).
        val_fraction: target fraction of tiles in the validation set.
        tile_size:    tile edge in pixels (must match the tiling step).
        block_px:     spatial block edge in pixels. Must be >= tile_size;
                      larger blocks = fewer boundary tiles dropped but a
                      coarser (less random) split. 1024 px = a 4x4 group
                      of 256-px tiles.
        seed:         RNG seed for block assignment (reproducibility).

    Returns:
        (train_df, val_df): disjoint subsets of index_df, index reset.
        The number of boundary tiles dropped from train is logged.

    Example (architecture_comparison notebook)::

        from lunar_segmentation.lunar_segmentation.data.splits import (
            spatial_train_val_split)
        train_df, val_df = spatial_train_val_split(df, val_fraction=0.2)
    """
    if block_px < tile_size:
        raise ValueError(f"block_px ({block_px}) must be >= tile_size ({tile_size})")

    df = index_df.reset_index(drop=True)
    rng = np.random.default_rng(seed)

    # Block id per tile: (aoi, row block, col block) of the tile origin
    block_keys = list(zip(df["aoi"], df["row"] // block_px, df["col"] // block_px))
    df = df.assign(_block=pd.Series(block_keys, index=df.index))

    blocks = df["_block"].drop_duplicates().tolist()
    rng.shuffle(blocks)

    # Greedily assign shuffled blocks to validation until the target is met
    counts = df["_block"].value_counts().to_dict()
    target = int(round(val_fraction * len(df)))
    val_blocks, n_val = set(), 0
    for b in blocks:
        if n_val >= target:
            break
        val_blocks.add(b)
        n_val += counts[b]

    is_val = df["_block"].isin(val_blocks)
    val_df = df[is_val]
    train_df = df[~is_val]

    # Drop training tiles whose footprint overlaps any validation tile.
    # Two tiles overlap iff same AOI and |dr| < tile_size and |dc| < tile_size.
    keep = np.ones(len(train_df), dtype=bool)
    for aoi, vgrp in val_df.groupby("aoi"):
        tmask = (train_df["aoi"] == aoi).to_numpy()
        if not tmask.any():
            continue
        tr = train_df.loc[tmask, "row"].to_numpy()
        tc = train_df.loc[tmask, "col"].to_numpy()
        vr = vgrp["row"].to_numpy()
        vc = vgrp["col"].to_numpy()
        # (n_train, n_val) boolean overlap matrix, chunked over train tiles
        overlap = np.zeros(len(tr), dtype=bool)
        chunk = 2048
        for s in range(0, len(tr), chunk):
            e = s + chunk
            dr = np.abs(tr[s:e, None] - vr[None, :]) < tile_size
            dc = np.abs(tc[s:e, None] - vc[None, :]) < tile_size
            overlap[s:e] = (dr & dc).any(axis=1)
        keep[tmask] &= ~overlap

    n_dropped = int((~keep).sum())
    train_df = train_df[keep]

    logger.info(
        f"Spatial split: {len(train_df)} train / {len(val_df)} val tiles "
        f"({len(val_df) / max(len(df), 1):.1%} val); dropped {n_dropped} "
        f"boundary tiles from train to guarantee zero pixel overlap."
    )

    return (
        train_df.drop(columns="_block").reset_index(drop=True),
        val_df.drop(columns="_block").reset_index(drop=True),
    )


def assert_no_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame, tile_size: int = 256):
    """Raise AssertionError if any train tile overlaps any val tile (sanity check)."""
    for aoi, vgrp in val_df.groupby("aoi"):
        tgrp = train_df[train_df["aoi"] == aoi]
        if tgrp.empty:
            continue
        tr, tc = tgrp["row"].to_numpy(), tgrp["col"].to_numpy()
        for vr, vc in zip(vgrp["row"].to_numpy(), vgrp["col"].to_numpy()):
            bad = (np.abs(tr - vr) < tile_size) & (np.abs(tc - vc) < tile_size)
            assert not bad.any(), f"overlap at aoi={aoi}, val tile ({vr},{vc})"
