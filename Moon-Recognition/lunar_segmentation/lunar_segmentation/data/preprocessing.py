import numpy as np
import pandas as pd
from pathlib import Path
from rasterio import features
from rasterio.transform import Affine
from skimage import exposure, filters, morphology
import torch
import logging

logger = logging.getLogger(__name__)

# Default classes as used in the notebook for multi-label segmentation
CLASS_NAMES = [
    'impact_crater',
    'pit_skylight',
    'wrinkle_ridge',
    'lobate_scarp',
    'irregular_mare_patch',
    'apollo_site',
    'candidate_rille'
]
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}

def build_three_channel_input(gray: np.ndarray) -> np.ndarray:
    """3-channel input: [normalised, CLAHE, Sobel edges].

    CLAHE (equalize_adapthist) is the expensive step (~3 s per large tile on CPU).
    Call precompute_tile_inputs() once to cache results and avoid paying this cost
    on every inference run.
    """
    gray = gray.astype(np.float32)
    if gray.size == 0:
        return np.zeros((3, gray.shape[0], gray.shape[1]), dtype=np.float32)

    gmin, gmax = float(gray.min()), float(gray.max())
    norm = (gray - gmin) / (gmax - gmin) if gmax > gmin else np.zeros_like(gray)

    clahe = exposure.equalize_adapthist(norm, clip_limit=0.03).astype(np.float32)
    sobel = filters.sobel(norm).astype(np.float32)

    return np.stack([norm, clahe, sobel], axis=0)


def precompute_tile_inputs(
    tiles_dir: Path,
    output_dir: Path,
    force: bool = False,
) -> int:
    """Run build_three_channel_input on every PNG in tiles_dir and save to output_dir.

    This is a one-time cost (~3 s/tile × 3 639 tiles ≈ 3 hours).  Once done,
    inference loads the saved .npz files directly and skips CLAHE entirely.

    Args:
        tiles_dir:  directory containing tile_*.png files.
        output_dir: where to write preprocessed tile_*.npz files.
        force:      reprocess even if the output file already exists.

    Returns:
        Number of tiles actually processed (skips existing unless force=True).
    """
    from PIL import Image

    output_dir.mkdir(parents=True, exist_ok=True)
    tile_files = sorted(tiles_dir.glob('tile_*.png'))
    processed  = 0

    for i, tile_path in enumerate(tile_files):
        out_path = output_dir / (tile_path.stem + '.npz')
        if not force and out_path.exists():
            continue

        img  = Image.open(tile_path).convert('L')
        gray = np.array(img).astype(np.float32)
        del img

        chw = build_three_channel_input(gray)
        del gray

        np.savez(out_path, input=chw.astype(np.float16))
        del chw
        processed += 1

        if processed % 100 == 0:
            print(f"  precomputed {processed} / {len(tile_files)} tiles…", flush=True)

    print(f"Done. {processed} tiles written to {output_dir}")
    return processed

def rasterize_multilabel(labels: dict, out_shape, transform, raster_crs):
    """
    Rasterize all label GeoDataFrames into a multi-channel uint8 mask.

    - impact_crater: Circular polygons — NO dilation needed (already area features)
    - candidate_rille: Polygon features — moderate dilation only
    - pit_skylight, apollo_site: Point features — disk(5) dilation
    - wrinkle_ridge, lobate_scarp: Line features — disk(3) dilation
    - irregular_mare_patch: Area or point features — disk(5) dilation
    """
    mask = np.zeros((len(CLASS_NAMES), out_shape[0], out_shape[1]), dtype=np.uint8)

    for class_name, gdf in labels.items():
        if class_name not in CLASS_TO_INDEX or gdf is None or len(gdf) == 0:
            continue
        idx = CLASS_TO_INDEX[class_name]

        # Reproject to raster CRS
        try:
            if gdf.crs is not None and raster_crs is not None:
                layer = gdf.to_crs(raster_crs)
            else:
                layer = gdf
        except Exception as e:
            logger.warning(f"CRS reprojection failed for {class_name}: {e}, using raw coordinates")
            layer = gdf

        shapes_iter = [(geom, 1) for geom in layer.geometry if geom is not None and not geom.is_empty]
        if not shapes_iter:
            logger.info(f"No valid geometries for {class_name} in this extent")
            continue

        channel = features.rasterize(
            shapes=shapes_iter,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            all_touched=True,
            dtype='uint8',
        )

        positive_before = int(channel.sum())

        # Apply class-specific dilation
        if class_name == 'impact_crater':
            # Craters are already circular polygons — no dilation needed.
            # The Polygon geometries already encode the full crater diameter.
            pass
        elif class_name == 'candidate_rille':
            # Rilles may be polygons already; apply light dilation to fill gaps
            channel = morphology.binary_dilation(channel.astype(bool), morphology.disk(2)).astype(np.uint8)
        elif class_name in {'pit_skylight', 'apollo_site'}:
            # Point features: expand to visible area
            channel = morphology.binary_dilation(channel.astype(bool), morphology.disk(5)).astype(np.uint8)
        elif class_name in {'wrinkle_ridge', 'lobate_scarp'}:
            # Line features: thicken traces
            channel = morphology.binary_dilation(channel.astype(bool), morphology.disk(3)).astype(np.uint8)
        elif class_name == 'irregular_mare_patch':
            # Area / point features: expand
            channel = morphology.binary_dilation(channel.astype(bool), morphology.disk(5)).astype(np.uint8)

        positive_after = int(channel.sum())
        logger.info(f"  {class_name}: {positive_before} → {positive_after} positive pixels "
                    f"(dilation applied: {positive_before != positive_after})")

        mask[idx] = channel
    return mask

def iter_tile_origins(height: int, width: int, tile_size: int, stride: int):
    """
    Generator for tiling coordinates.
    """
    rows = list(range(0, max(height - tile_size + 1, 1), stride))
    cols = list(range(0, max(width - tile_size + 1, 1), stride))

    if rows and rows[-1] != height - tile_size:
        rows.append(max(height - tile_size, 0))
    if cols and cols[-1] != width - tile_size:
        cols.append(max(width - tile_size, 0))

    for r in rows:
        for c in cols:
            yield r, c

def save_tiles_for_aoi(aoi_name: str, bounds, raster_path: Path, labels: dict,
                     tile_size: int = 256, stride: int = 128,
                     processed_dir: Path = Path("data/processed"),
                     keep_empty_fraction: float = 0.1, seed: int = 42):
    """
    Tiles a lunar raster and its labels, saving them as .npz files.
    """
    from ..utils.geo_utils import crop_singleband_raster  # Expected helper

    img_gray, transform, profile = crop_singleband_raster(raster_path, bounds)
    if img_gray.size == 0 or img_gray.shape[0] < tile_size or img_gray.shape[1] < tile_size:
        logger.warning(f"AOI {aoi_name}: raster crop too small {img_gray.shape}, skipping.")
        return pd.DataFrame(columns=['aoi', 'tile_path', 'row', 'col', 'positive_pixels'])

    raster_crs = profile.get('crs')
    image = build_three_channel_input(img_gray)
    mask = rasterize_multilabel(labels, img_gray.shape, transform, raster_crs)

    tile_dir = processed_dir / 'tiles' / aoi_name
    tile_dir.mkdir(parents=True, exist_ok=True)

    records = []
    keep_rng = np.random.default_rng(seed)

    for r, c in iter_tile_origins(img_gray.shape[0], img_gray.shape[1], tile_size, stride):
        x = image[:, r:r+tile_size, c:c+tile_size]
        y = mask[:, r:r+tile_size, c:c+tile_size]

        if x.shape[1] != tile_size or x.shape[2] != tile_size:
            continue

        positive = int(y.sum())
        if positive == 0 and keep_rng.random() > keep_empty_fraction:
            continue

        tile_path = tile_dir / f'{aoi_name}_r{r:05d}_c{c:05d}.npz'
        np.savez_compressed(tile_path, image=x.astype(np.float32), mask=y.astype(np.uint8),
                             row=r, col=c, transform=np.array(transform))

        records.append({
            'aoi': aoi_name,
            'tile_path': str(tile_path),
            'row': r,
            'col': c,
            'positive_pixels': positive,
        })

    return pd.DataFrame(records)
