 Amir-Moon-Recognition(R-CNN)
# Moon Recognition

Mask R-CNN workflow for detecting and segmenting lunar surface features from Marius Hills and lunar south pole image tiles.

This branch focuses on the **R-CNN / Mask R-CNN contribution** of the Moon Recognition project. The main goal is instance segmentation: for each lunar tile, the model predicts separate objects with a class label, bounding box, confidence score, and pixel mask.

## Overview

The available Marius Hills dataset contains semantic masks, not clean instance labels. For Mask R-CNN training, these semantic masks are converted into object instances using connected components. The trained detector is then evaluated on a held-out spatial validation split and applied to separate PNG test tiles.

The implemented Mask R-CNN model is adapted for small lunar features by using:

- a COCO-pretrained ResNet-50 + FPN backbone,
- smaller anchors for small craters, ridges, pits, and scarps,
- a deeper box head for classifying candidate regions,
- a deeper mask head for drawing object outlines,
- spatial train/validation splitting to reduce leakage between neighboring tiles,
- crash-safe training with `last_model.pth`, `best_model.pth`, and training history exports.

## Feature Classes

The Mask R-CNN notebook trains on four lunar feature classes:

| Class | Meaning |
| --- | --- |
| `impact_crater` | Circular impact depressions |
| `pit_skylight` | Openings or pits, often related to lava tubes |
| `wrinkle_ridge` | Linear or curving compressional ridges |
| `lobate_scarp` | Scarp-like tectonic landforms |

The original semantic mask channels also include `irregular_mare_patch`, `apollo_site`, and `candidate_rille`, but these were too rare for the Mask R-CNN training setup used here.

## Repository Structure

```text
Moon-Recognition/
|-- notebooks/
|   |-- amir_rcnn.ipynb                  # Main Mask R-CNN training, validation, and PNG-tile inference notebook
|   `-- south_pole_complete_workflow.ipynb
|-- figures/                             # Exported figures used in reports and analysis
|-- predictions/                         # Saved validation prediction files (.npz)
|-- lunar_segmentation/                  # Additional semantic-segmentation utilities
|-- metrics.json                         # Best-checkpoint validation metrics
|-- training_history.csv                 # Per-epoch training and validation history
|-- training_history.json
`-- requirements.txt                     # Minimal requirements for the Mask R-CNN notebook
```

## Main Notebook

The central notebook is:

```text
notebooks/amir_rcnn.ipynb
```

It contains the full Mask R-CNN workflow:

1. Configuration and environment setup.
2. Dataset discovery for Marius Hills `.npz` tiles and `index.csv`.
3. Semantic-channel audit and class selection.
4. Semantic-mask to instance-mask conversion.
5. Spatial train/validation split.
6. Mask R-CNN model construction.
7. Training and checkpointing.
8. Best-checkpoint evaluation.
9. Confusion matrix generation.
10. Validation prediction export.
11. Inference on PNG test tiles.
12. Test-tile feature proportion plots.
13. Low-threshold crater search.

## Setup

Create a Python environment and install the project requirements:

```bash
cd Moon-Recognition
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Then open the notebook:

```bash
jupyter lab notebooks/amir_rcnn.ipynb
```

The notebook expects the Marius Hills processed tiles and metadata in the format used by the project:

```text
tiles/index.csv
data/processed/tiles/marius_hills/*.npz
```

For the PNG test-tile inference section, it also expects a folder of image tiles:

```text
tiles/*.png
```

Model weights such as `best_model.pth` and `last_model.pth` are large and are not included in this repository. They are produced by running the training section of the notebook.

## Training Summary

The model was trained with:

- Mask R-CNN with ResNet-50 + FPN backbone,
- AdamW optimizer,
- learning rate `2e-4`,
- cosine decay after warm-up,
- batch size `8`,
- 20 epochs,
- CPU-only training in the local run,
- per-epoch checkpointing and history export.

The training history is stored in:

```text
training_history.csv
training_history.json
```

## Validation Results

The best checkpoint metrics are stored in `metrics.json`.

| Metric | Value |
| --- | ---: |
| mAP@0.5 | 0.0102 |
| Precision | 0.1040 |
| Recall | 0.0559 |
| F1 | 0.0727 |
| Mean mask IoU | 0.6356 |
| Mean mask Dice | 0.7732 |
| Semantic Dice | 0.1132 |
| Semantic IoU | 0.0705 |
| Count MAE | 6.3036 |
| True positives | 461 |
| False positives | 3973 |
| False negatives | 7790 |

Per-class AP@0.5:

| Class | AP@0.5 |
| --- | ---: |
| `impact_crater` | 0.0233 |
| `pit_skylight` | 0.0003 |
| `wrinkle_ridge` | 0.0172 |
| `lobate_scarp` | 0.0000 |

The main result is that the mask branch performs better than the detector branch. When an object is matched, the predicted mask can overlap the target well, but the detector misses many objects, producing low recall and low mAP.

## Figures

Key exported figures are stored in `figures/`:

- `pipeline_walkthrough.png` - Mask R-CNN architecture and project pipeline.
- `training_curves.png` - Training loss and validation curves.
- `prediction_compact.png` - Validation prediction examples.
- `label_noise.pdf` - Example of semantic-to-instance label noise.
- `best_model_confusion_matrix.png` - Validation confusion matrix.
- `best_model_10_tile_samples.png` - Best-model samples on PNG test tiles.
- `test_tile_feature_proportions_pie.jpg` - Predicted feature proportions on PNG test tiles.
- `low_threshold_crater_candidate_samples.png` - Low-threshold crater candidate examples.

## PNG Test-Tile Inference

The best Mask R-CNN checkpoint was applied to 3,639 PNG test tiles. At the normal score threshold of 0.50, the model was very conservative:

- 3,592 tiles had no confident prediction,
- 47 tiles had at least one confident prediction,
- 52 total confident detections were produced,
- all confident detections were `wrinkle_ridge`.

Because no crater detections appeared at the normal threshold, a diagnostic low-threshold crater search was added. This showed that crater-like responses exist below the normal confidence threshold, but many are weak, duplicated, or poorly placed. The low-threshold run is therefore useful for inspection, not as the final prediction setting.

## Saved Predictions

The `predictions/` folder contains 500 saved validation prediction files in `.npz` format. These store predicted masks, boxes, labels, and scores for later inspection without rerunning the model.

## Notes and Limitations

- The training labels are automatically generated from semantic masks, so the instance targets are noisy.
- Touching craters can be merged into one object, while long ridges can be split into several objects.
- Class imbalance is severe: craters and wrinkle ridges dominate, while pit/skylights and lobate scarps are rare.
- The model has useful mask quality on matched objects, but the detection branch has low recall.
- GPU training and cleaner instance labels would likely improve detection performance.

## Related Folder

The `lunar_segmentation/` folder contains additional semantic-segmentation utilities and a separate README. That folder is useful for U-Net-style semantic segmentation experiments, while `notebooks/amir_rcnn.ipynb` is the main Mask R-CNN instance-segmentation workflow for this branch.
=======
# Computational Physics 2026 — Group Project

Two recognition tasks tackled with several methods each, one branch per student,
merged into `main` for a combined comparison.

## Sub-projects

### 1. Astrophysical Object Recognition (`Astrophysical-Objects-Recognition/`)
Classification of SDSS objects into **GALAXY / QSO / STAR** using traditional ML
(SVC, Decision Tree, Random Forest, CatBoost, LightGBM), a voting ensemble, and a
PyTorch neural net, with SHAP/permutation-importance interpretability.
Packaged as the installable `stellar_classification` module.
- Data: `data/AOR/star_classification.csv`

### 2. Moon Feature Recognition (`Moon-Recognition/`)
Segmenting/detecting lunar features (craters, ridges, pits, …) on Marius Hills
tiles. Four approaches are compared on shared tiles:

| Method | Location | Type |
|---|---|---|
| Classical (thresholding + morphology) | `Moon-Recognition/notebooks/` | baseline |
| YOLOv8 detection | `Moon-Recognition/yolo/` | object detection |
| U-Net | `Moon-Recognition/lunar_segmentation/` | semantic segmentation |
| Mask R-CNN | `Moon-Recognition/notebooks/amir_rcnn.ipynb` | instance segmentation |

- Data: `data/MR/` (tiles, weights, results) — not tracked in git (large).
- Trained weights >100 MB are attached to the relevant GitHub Release, not committed.

## Environment
The `stellar` conda env runs everything in this project:
`torch torchvision ultralytics opencv-python rasterio scikit-image scikit-learn imbalanced-learn shap lightgbm catboost`.
On macOS, set `KMP_DUPLICATE_LIB_OK=TRUE` to avoid the libomp double-init abort.

## Reproduce all models
```bash
KMP_DUPLICATE_LIB_OK=TRUE python verify_models.py
```
Smoke-tests every model (loads weights + runs a few samples). The Mask R-CNN
weights (194 MB) are not in git — fetch once with
`gh release download v1.0 -R Amir-SaiediSaber/Computational-Physics-2026 -p best_model.pth -D data/MR/weights/`.

## Report
See `report/` for the written comparison. main
