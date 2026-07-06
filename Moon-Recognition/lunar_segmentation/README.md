# lunar_segmentation

Lunar geofeature segmentation pipeline for detecting lunar surface features (craters, pits, rilles, ridges, etc.) from LRO WAC imagery using a U-Net architecture.

## Feature Classes

- `impact_crater` — Circular depressions from impacts
- `pit_skylight` — Openings into lava tubes
- `wrinkle_ridge` — Compression folds
- `lobate_scarp` — Cliff-like fault scarps
- `irregular_mare_patch` — Inhomogeneous mare units
- `apollo_site` — Apollo mission landing sites
- `candidate_rille` — Linear depressions (lava tubes/faults)

## Package Structure

```
lunar_segmentation/
├── configs/unet_config.yaml — definitive T4 training protocol (+ local variant)
├── scripts/qualitative_eval.py — GT/prediction/FP-FN figures on validation tiles
└── lunar_segmentation/      — the package itself
    ├── models/unet.py        — SmallUNet (configurable width/residual/dropout)
    ├── data/
    │   ├── resolver.py      — Data downloading from LROC/USGS
    │   ├── label_loader.py  — Label loading (shp, csv, xlsx)
    │   ├── preprocessing.py — 3-channel input, mask rasterization, tiling
    │   ├── datasets.py      — MoonTileDataset with augmentation
    │   └── splits.py        — leakage-free spatial block train/val split
    ├── training/trainer.py  — Focal/BCE+Dice losses, training loop, metrics
    ├── inference/predictor.py — Batched sliding-window prediction
    ├── utils/geo_utils.py  — Lunar coordinate handling, raster cropping
    └── visualization/plotter.py — Probability overlays
```

Note: overlapping tiles (stride 128 / size 256) make random train/val splits
leak; prefer `data/splits.py:spatial_train_val_split` for any new run
(course lecture recommendation; see `../IMPROVEMENTS.md`).

## Setup

```bash
pip install -r requirements.txt
```

## Usage from Notebook

```python
import sys
sys.path.insert(0, 'lunar_segmentation')

from lunar_segmentation.models.unet import SmallUNet
from lunar_segmentation.data.datasets import MoonTileDataset
from lunar_segmentation.data.preprocessing import CLASS_NAMES, build_three_channel_input
from lunar_segmentation.training.trainer import Trainer, BCEDiceLoss
from lunar_segmentation.inference.predictor import Predictor
from lunar_segmentation.visualization.plotter import generate_pretty_image
```

## Training Config

See `configs/unet_config.yaml` for model hyperparameters.
