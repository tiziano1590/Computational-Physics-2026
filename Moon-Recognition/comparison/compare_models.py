#!/usr/bin/env python
"""Shared-protocol comparison of the lunar crater methods.

Task types differ (semantic seg / instance seg / detection), so we reduce every
method to the SAME thing: a binary impact-crater pixel mask on the SAME tiles
(the leakage-free spatial val split, seed=42), scored with the SAME metrics.

Protocol
--------
* Data/split: spatial_train_val_split(seed=42) on data/MR/tiles/index.csv (Marius Hills).
* GT: mask channel 0 (impact_crater) > 0.
* Methods -> binary crater mask:
    - U-Net      : crater-channel probability, thresholded.
    - Mask R-CNN : union of crater-class instance masks with score >= tau.
    - YOLO       : union of predicted boxes (model collapses to one class) with score >= tau.
    - Classical  : thresholding + morphology (fixed; no score to tune).
* Thresholds for the score-based methods are tuned for best micro-F1 on a CALIB
  subset, then reported on a DISJOINT TEST subset (no test-set tuning).
* Metrics: pixel precision/recall/F1/IoU (micro over test pixels), crater
  count MAE (connected components, area 30..6000), and inference ms/tile.

Run:  KMP_DUPLICATE_LIB_OK=TRUE python Moon-Recognition/comparison/compare_models.py
"""
import os, sys, glob, time, warnings, logging, argparse
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np, pandas as pd, cv2, torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "Moon-Recognition/lunar_segmentation"))
sys.path.insert(0, HERE)
from lunar_segmentation.data.splits import spatial_train_val_split
from lunar_segmentation.models.unet import SmallUNet
from lunar_segmentation.inference.predictor import Predictor
from lunar_segmentation.data.preprocessing import CLASS_NAMES
from classical_baseline import detect_craters_classical
import maskrcnn_loader

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
AREA = dict(min_area=30, max_area=6000)


def count_components(binmask):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(binmask.astype(np.uint8), connectivity=8)
    return sum(1 for i in range(1, n) if AREA["min_area"] <= stats[i, cv2.CC_STAT_AREA] <= AREA["max_area"])


def micro_prf(tp, fp, fn):
    p = tp / (tp + fp + 1e-9); r = tp / (tp + fn + 1e-9)
    f1 = 2 * p * r / (p + r + 1e-9); iou = tp / (tp + fp + fn + 1e-9)
    return p, r, f1, iou

def mcc(tp, fp, fn, tn):
    import math
    tp, fp, fn, tn = float(tp), float(fp), float(fn), float(tn)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-9
    return (tp * tn - fp * fn) / denom


# ---- per-tile prediction representations (computed once) -------------------
def unet_repr(predictor, img):
    t = time.perf_counter()
    prob = predictor.predict(img, tile_size=256, stride=128)[0]   # crater channel
    return {"scoremap": prob.astype(np.float32)}, time.perf_counter() - t

def maskrcnn_repr(model, img, crater_label=1):
    t = time.perf_counter()
    with torch.no_grad():
        out = model([torch.from_numpy(img.astype(np.float32)).to(DEV)])[0]
    dt = time.perf_counter() - t
    H, W = img.shape[1:]
    sm = np.zeros((H, W), np.float32); scores = []
    lbl = out["labels"].cpu().numpy(); scs = out["scores"].cpu().numpy(); msk = out["masks"].cpu().numpy()
    for i in range(len(lbl)):
        if lbl[i] != crater_label:
            continue
        m = msk[i, 0] > 0.5
        sm[m] = np.maximum(sm[m], scs[i]); scores.append(float(scs[i]))
    return {"scoremap": sm, "scores": np.array(sorted(scores, reverse=True))}, dt

def yolo_repr(model, img):
    g = img[0]; g = ((g - g.min()) / (np.ptp(g) + 1e-9) * 255).astype(np.uint8)
    rgb = np.stack([g] * 3, -1)
    t = time.perf_counter()
    r = model.predict(rgb, verbose=False, conf=0.001)[0]
    dt = time.perf_counter() - t
    H, W = g.shape; sm = np.zeros((H, W), np.float32); scores = []
    for b, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W, x2), min(H, y2)
        sm[y1:y2, x1:x2] = np.maximum(sm[y1:y2, x1:x2], s); scores.append(float(s))
    return {"scoremap": sm, "scores": np.array(sorted(scores, reverse=True))}, dt

def classical_repr(img):
    g = img[0]; g = ((g - g.min()) / (np.ptp(g) + 1e-9) * 255).astype(np.uint8)
    t = time.perf_counter()
    mask, _ = detect_craters_classical(g, **AREA)
    return {"binmask": mask.astype(bool)}, time.perf_counter() - t

def predictall_repr(img):
    return {"binmask": np.ones(img.shape[1:], dtype=bool)}, 0.0


def binmask_at(rep, tau):
    if "binmask" in rep:        # classical: fixed
        return rep["binmask"]
    return rep["scoremap"] >= tau

def count_at(rep, tau):
    if "binmask" in rep:
        return count_components(rep["binmask"])
    if "scores" in rep:         # detection/instance: count predictions above tau
        return int((rep["scores"] >= tau).sum())
    return count_components(rep["scoremap"] >= tau)   # unet: components of thresholded mask


def best_tau(reps, gts, grid):
    best, best_f1 = grid[0], -1
    for tau in grid:
        tp = fp = fn = 0
        for rep, gt in zip(reps, gts):
            pm = binmask_at(rep, tau)
            tp += np.logical_and(pm, gt).sum(); fp += np.logical_and(pm, ~gt).sum(); fn += np.logical_and(~pm, gt).sum()
        f1 = micro_prf(tp, fp, fn)[2]
        if f1 > best_f1:
            best_f1, best = f1, tau
    return best


def evaluate(reps, gts, gtcounts, tau, times):
    tp = fp = fn = tn = 0; cerr = []
    for rep, gt, gc in zip(reps, gts, gtcounts):
        pm = binmask_at(rep, tau)
        tp += np.logical_and(pm, gt).sum(); fp += np.logical_and(pm, ~gt).sum()
        fn += np.logical_and(~pm, gt).sum(); tn += np.logical_and(~pm, ~gt).sum()
        cerr.append(abs(count_at(rep, tau) - gc))
    p, r, f1, iou = micro_prf(tp, fp, fn)
    return dict(precision=p, recall=r, f1=f1, iou=iou, mcc=mcc(tp, fp, fn, tn),
                count_mae=float(np.mean(cerr)), ms_per_tile=1000 * float(np.mean(times)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calib", type=int, default=60)
    ap.add_argument("--test", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac_lo", type=float, default=0.0, help="min crater coverage of eval tiles")
    ap.add_argument("--frac_hi", type=float, default=1.01, help="max crater coverage of eval tiles")
    ap.add_argument("--tag", type=str, default="allval", help="label for the output CSV")
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(ROOT, "data/MR/tiles/index.csv"))
    _, val = spatial_train_val_split(df, val_fraction=0.2, tile_size=256, block_px=1024, seed=42)
    frac = val["positive_pixels"] / 65536.0
    val = val[(frac >= args.frac_lo) & (frac < args.frac_hi)].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(val))[: args.calib + args.test]
    calib_rows = val.iloc[idx[: args.calib]]; test_rows = val.iloc[idx[args.calib:]]
    print(f"device={DEV} | band=[{args.frac_lo},{args.frac_hi}) | val tiles={len(val)} | calib={len(calib_rows)} test={len(test_rows)}")

    def load_tile(row):
        p = os.path.join(ROOT, "data/MR", row["tile_path"])
        d = np.load(p); return d["image"].astype(np.float32), (d["mask"][0] > 0)

    # models
    unet = Predictor(SmallUNet(3, len(CLASS_NAMES)),
                     weights_path=os.path.join(ROOT, "data/MR/weights/best_trained.pth"), device=DEV)
    mrcnn, mr_names, _ = maskrcnn_loader.build_and_load(
        os.path.join(ROOT, "data/MR/weights/best_model.pth"), device=DEV)
    crater_label = mr_names.index("impact_crater")
    from ultralytics import YOLO
    yolo = YOLO(sorted(glob.glob(os.path.join(ROOT, "Moon-Recognition/yolo/YOLO/runs/detect/train*/weights/best.pt")))[-1])

    methods = {
        "U-Net (semantic)":    dict(fn=lambda im: unet_repr(unet, im),            grid=np.round(np.arange(0.30, 0.91, 0.05), 2)),
        "Mask R-CNN (instance)": dict(fn=lambda im: maskrcnn_repr(mrcnn, im, crater_label), grid=np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])),
        "YOLOv8 (detection)":  dict(fn=lambda im: yolo_repr(yolo, im),            grid=np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])),
        "Classical (thr+morph)": dict(fn=lambda im: classical_repr(im),           grid=np.array([0.5])),
        "Predict-all (baseline)": dict(fn=lambda im: predictall_repr(im),          grid=np.array([0.5])),
    }

    # precompute reps
    calib_imgs = [load_tile(r) for _, r in calib_rows.iterrows()]
    test_imgs = [load_tile(r) for _, r in test_rows.iterrows()]
    prev = float(np.mean([gt.mean() for _, gt in test_imgs]))
    print(f"test-set crater prevalence (mean pixel fraction): {prev:.3f}")
    rows = []
    for name, spec in methods.items():
        print(f"  running {name} ...", flush=True)
        creps = [spec["fn"](im)[0] for im, _ in calib_imgs]
        treps, ttimes = [], []
        for im, _ in test_imgs:
            rep, dt = spec["fn"](im); treps.append(rep); ttimes.append(dt)
        tau = best_tau(creps, [gt for _, gt in calib_imgs], spec["grid"])
        m = evaluate(treps, [gt for _, gt in test_imgs],
                     [count_components(gt) for _, gt in test_imgs], tau, ttimes)
        m["method"] = name; m["tau"] = float(tau); rows.append(m)
        print(f"    tau={tau:.2f}  F1={m['f1']:.3f}  IoU={m['iou']:.3f}  MCC={m['mcc']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  countMAE={m['count_mae']:.1f}  {m['ms_per_tile']:.0f}ms/tile")

    out = pd.DataFrame(rows)[["method", "tau", "precision", "recall", "f1", "iou", "mcc", "count_mae", "ms_per_tile"]]
    csv = os.path.join(HERE, f"comparison_results_{args.tag}.csv"); out.to_csv(csv, index=False)
    print("\n=== CRATER DETECTION — shared protocol (leakage-free spatial val, seed=42) ===")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nsaved -> {csv}")
    return out, test_imgs, methods


if __name__ == "__main__":
    main()
