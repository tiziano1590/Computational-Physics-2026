#!/usr/bin/env python
"""End-to-end reproducibility check for all models in the repo.

Run from the repo root in the `stellar` conda env:
    KMP_DUPLICATE_LIB_OK=TRUE python verify_models.py

Each block is a lightweight smoke test (load weights / data, run a few samples),
not a full retrain. Prints a PASS/FAIL summary. Missing optional weights are
reported as SKIP rather than failing the whole run.
"""
import os, sys, glob, json, warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
results = {}

def section(name):
    print(f"\n{'='*60}\n{name}\n{'='*60}")

# ---------------------------------------------------------------- AOR stellar
def check_aor():
    section("AOR — stellar classification (GALAXY/QSO/STAR)")
    sys.path.insert(0, os.path.join(ROOT, "Astrophysical-Objects-Recognition/stellar_classification"))
    import pandas as pd
    from stellar_classification.data.preprocessing import prepare_splits
    from stellar_classification.trainer import train_traditional
    df = pd.read_csv(os.path.join(ROOT, "data/AOR/star_classification.csv")).sample(6000, random_state=0)
    Xtr, Xva, Xte, ytr, yva, yte, le, scaler, feats = prepare_splits(df)
    models = train_traditional(Xtr, ytr, Xva, yva)
    from sklearn.metrics import accuracy_score
    rf = models["Random Forest"]
    acc = accuracy_score(yva, rf.predict(Xva))
    print(f"  Random Forest val accuracy: {acc:.3f}")
    assert acc > 0.85
    return f"val acc {acc:.3f} ({len(models)} models)"

# ---------------------------------------------------------------- U-Net (marius)
def check_unet():
    section("Moon — U-Net semantic segmentation (Marius Hills)")
    sys.path.insert(0, os.path.join(ROOT, "Moon-Recognition/lunar_segmentation"))
    from lunar_segmentation.models.unet import SmallUNet
    from lunar_segmentation.inference.predictor import Predictor, _remap_doubleconv_key
    from lunar_segmentation.data.preprocessing import CLASS_NAMES
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    w = os.path.join(ROOT, "data/MR/weights/best_trained.pth")
    # confirm clean load (the bug we fixed)
    ck = torch.load(w, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck.get("model", ck)) if isinstance(ck, dict) else ck
    sd = {_remap_doubleconv_key(k.replace("model.", "", 1)): v for k, v in sd.items()}
    miss = SmallUNet(3, len(CLASS_NAMES)).load_state_dict(sd, strict=False).missing_keys
    assert not miss, f"checkpoint did not load cleanly ({len(miss)} missing)"
    p = Predictor(SmallUNet(3, len(CLASS_NAMES)), weights_path=w, device=dev)
    tiles = sorted(glob.glob(os.path.join(ROOT, "data/MR/data/processed/tiles/marius_hills/*.npz")))[:4]
    for t in tiles:
        p.predict(np.load(t)["image"], tile_size=256, stride=128)
    print(f"  clean load (missing=0); inference OK on {len(tiles)} tiles")
    return "clean load + inference"

# ---------------------------------------------------------------- U-Net (south pole)
def check_southpole():
    section("Moon — U-Net applied to South Pole tiles (Giuseppe)")
    import cv2
    sys.path.insert(0, os.path.join(ROOT, "Moon-Recognition/lunar_segmentation"))
    from lunar_segmentation.models.unet import SmallUNet
    from lunar_segmentation.data.preprocessing import build_three_channel_input, CLASS_NAMES
    from lunar_segmentation.inference.predictor import Predictor
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    p = Predictor(SmallUNet(3, len(CLASS_NAMES)),
                  weights_path=os.path.join(ROOT, "data/MR/weights/best_trained.pth"), device=dev)
    tiles = sorted(glob.glob(os.path.join(ROOT, "data/MR/lunar_south_pole/tiles/tile_*.png")))[:4]
    for tp in tiles:
        g = cv2.resize(cv2.imread(tp, cv2.IMREAD_GRAYSCALE), (256, 256))
        p.predict(build_three_channel_input(g), tile_size=256, stride=128)
    print(f"  inference OK on {len(tiles)} south-pole tiles")
    return f"{len(tiles)} tiles OK"

# ---------------------------------------------------------------- YOLO
def check_yolo():
    section("Moon — YOLOv8 detection (Alireza)")
    from ultralytics import YOLO
    w = sorted(glob.glob(os.path.join(ROOT, "Moon-Recognition/yolo/YOLO/runs/detect/train*/weights/best.pt")))
    if not w:
        return "SKIP (no best.pt on disk)"
    m = YOLO(w[-1])
    tiles = sorted(glob.glob(os.path.join(ROOT, "data/MR/data/processed/tiles/marius_hills/*.npz")))[:8]
    ndet = 0
    for t in tiles:
        g = np.load(t)["image"][0]; g = ((g - g.min()) / (np.ptp(g) + 1e-9) * 255).astype(np.uint8)
        ndet += len(m.predict(np.stack([g]*3, -1), verbose=False, conf=0.10)[0].boxes)
    print(f"  loaded {os.path.basename(w[-1])}; {ndet} detections on {len(tiles)} tiles")
    return f"{ndet} detections"

# ---------------------------------------------------------------- Mask R-CNN
def check_maskrcnn():
    section("Moon — Mask R-CNN instance segmentation (Amir)")
    import torch.nn as nn
    from torchvision.models.detection import maskrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    w = os.path.join(ROOT, "data/MR/weights/best_model.pth")
    if not os.path.exists(w):
        return "SKIP (best_model.pth not downloaded — gh release download v1.0)"

    class ResidualConvBlock(nn.Module):
        def __init__(s, ch=256, groups=8, dropout=0.05):
            super().__init__()
            s.block = nn.Sequential(nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.GroupNorm(groups, ch),
                                    nn.ReLU(inplace=True), nn.Dropout2d(dropout),
                                    nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.GroupNorm(groups, ch))
            s.act = nn.ReLU(inplace=True)
        def forward(s, x): return s.act(x + s.block(x))

    class DeepResidualMaskHead(nn.Module):
        def __init__(s, in_ch=256, hidden=256, num_blocks=4, groups=8, dropout=0.05):
            super().__init__()
            s.stem = nn.Sequential(nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False),
                                   nn.GroupNorm(groups, hidden), nn.ReLU(inplace=True))
            s.blocks = nn.Sequential(*[ResidualConvBlock(hidden, groups, dropout) for _ in range(num_blocks)])
            s.out_channels = hidden
        def forward(s, x): return s.blocks(s.stem(x))

    class DeepMaskPredictor(nn.Module):
        def __init__(s, in_ch=256, hidden=256, num_classes=3, groups=8, dropout=0.05):
            super().__init__()
            s.refine = nn.Sequential(ResidualConvBlock(in_ch, groups, dropout),
                                     nn.Conv2d(in_ch, hidden, 3, padding=1, bias=False),
                                     nn.GroupNorm(groups, hidden), nn.ReLU(inplace=True))
            s.up = nn.ConvTranspose2d(hidden, hidden, 2, stride=2)
            s.act = nn.ReLU(inplace=True); s.drop = nn.Dropout2d(dropout)
            s.mask_logits = nn.Conv2d(hidden, num_classes, 1)
        def forward(s, x): return s.mask_logits(s.drop(s.act(s.up(s.refine(x)))))

    class DeepBoxHead(nn.Module):
        def __init__(s, in_ch=256, roi=7, hidden=1024, depth=4, dropout=0.10):
            super().__init__()
            layers = [nn.Flatten()]; cur = in_ch * roi * roi
            for _ in range(depth):
                layers += [nn.Linear(cur, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout)]; cur = hidden
            s.layers = nn.Sequential(*layers); s.out_channels = hidden
        def forward(s, x): return s.layers(x)

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    ck = torch.load(w, map_location="cpu", weights_only=False)
    import types; cfg = types.SimpleNamespace(**ck["config"]); ncls = len(ck["class_names"]); TILE = 256
    m = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None,
                              trainable_backbone_layers=cfg.trainable_backbone_layers)
    out_ch = m.backbone.out_channels
    roi = m.roi_heads.box_roi_pool.output_size; roi = int(roi[0]) if isinstance(roi, (tuple, list)) else int(roi)
    m.roi_heads.box_head = DeepBoxHead(out_ch, roi, cfg.box_head_hidden, cfg.box_head_depth, cfg.box_head_dropout)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_head.out_channels, ncls)
    m.roi_heads.mask_head = DeepResidualMaskHead(out_ch, cfg.mask_head_hidden, cfg.mask_head_blocks, dropout=cfg.head_dropout)
    m.roi_heads.mask_predictor = DeepMaskPredictor(m.roi_heads.mask_head.out_channels, cfg.mask_head_hidden, ncls, dropout=cfg.head_dropout)
    sizes = tuple((s,) for s in cfg.anchor_sizes); m.rpn.anchor_generator = AnchorGenerator(sizes=sizes, aspect_ratios=(tuple(cfg.anchor_ratios),)*len(sizes))
    m.transform.min_size = (TILE,); m.transform.max_size = TILE*2
    m.roi_heads.score_thresh = cfg.box_score_thresh; m.roi_heads.nms_thresh = cfg.box_nms_thresh
    m.roi_heads.detections_per_img = cfg.detections_per_img
    res = m.load_state_dict(ck["model_state"], strict=False)
    assert not res.missing_keys and not res.unexpected_keys, "architecture/checkpoint mismatch"
    m.to(dev).eval()
    tiles = sorted(glob.glob(os.path.join(ROOT, "data/MR/data/processed/tiles/marius_hills/*.npz")))[:4]
    with torch.no_grad():
        n = sum(len(m([torch.from_numpy(np.load(t)["image"].astype(np.float32)).to(dev)])[0]["boxes"]) for t in tiles)
    print(f"  clean load (missing=0, unexpected=0); {n} detections on {len(tiles)} tiles")
    return f"clean load + {n} detections"

for name, fn in [("AOR stellar", check_aor), ("U-Net (Marius)", check_unet),
                 ("U-Net (South Pole)", check_southpole), ("YOLOv8", check_yolo),
                 ("Mask R-CNN", check_maskrcnn)]:
    try:
        results[name] = fn()
    except Exception as e:
        results[name] = f"FAIL: {type(e).__name__}: {e}"

section("SUMMARY")
for k, v in results.items():
    tag = "SKIP" if str(v).startswith("SKIP") else ("FAIL" if str(v).startswith("FAIL") else "PASS")
    print(f"  [{tag}] {k:22s} {v}")
