#!/usr/bin/env python
"""Shared-protocol comparison on WRINKLE_RIDGE — the class with genuine signal.

Unlike impact_crater (82% prevalence, near-trivial), wrinkle_ridge is a sparse
(~0.6% global, ~2.6% on ridge-bearing tiles) linear feature — the report's
strongest real result. This is where a method comparison is actually meaningful.

Protocol: leakage-free spatial val (seed=42), restricted to the 524 tiles that
contain ridge ground truth. Every method -> binary ridge mask (channel 2 GT);
threshold calibrated on a disjoint 40/60 calib/test split; metrics = pixel
P/R/F1/IoU + MCC + ms/tile.

YOLO entry: the train-5 run was trained on `yolo_dataset_no_class0` (impact_crater
channel dropped, remaining channels re-indexed 0..5), so its class1 = original
channel 2 = wrinkle_ridge — and its confusion matrix shows class1 is the only
class it ever detects (recall 0.18). Its class1 boxes are therefore a legitimate
ridge detector; a positive MCC here also validates the re-index reading of
`all_boxes_no_class0.csv` empirically.

  KMP_DUPLICATE_LIB_OK=TRUE python Moon-Recognition/comparison/ridge_comparison.py
"""
import os, sys, glob, time, warnings, logging, numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore"); logging.disable(logging.WARNING)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "Moon-Recognition/lunar_segmentation"))
sys.path.insert(0, HERE)
import compare_models as C
from lunar_segmentation.data.splits import spatial_train_val_split
from lunar_segmentation.models.unet import SmallUNet
from lunar_segmentation.inference.predictor import Predictor
from lunar_segmentation.data.preprocessing import CLASS_NAMES
from classical_baseline import ridge_response_classical
import maskrcnn_loader
from ultralytics import YOLO  # noqa: F401  (imported for parity / availability check)

RIDGE_CH = CLASS_NAMES.index("wrinkle_ridge")   # 2
# Use the report's DEFINITIVE leakage-free model (spatial split, no-aug, 10k/30ep,
# ridge AP 0.385) rather than best_trained.pth, which is an earlier crater-oriented
# checkpoint whose ridge channel is essentially dead (max prob ~0 on ridge tiles).
UNET_CKPT = "data/MR/results/spatial_ablation/noaug_w32_t10000_e30.pt"
# train-5 = the 6-class no-crater run; its class1 (only detected class) = wrinkle_ridge
YOLO_T5_CKPT = "Moon-Recognition/yolo/YOLO/runs/detect/train-5/weights/best.pt"
YOLO_RIDGE_CLS = 1
CALIB_FRAC, SEED = 0.40, 0
COLORS = {"U-Net (semantic)": "#102A43", "Mask R-CNN (instance)": "#E87A2B",
          "YOLOv8 train-5 (class1)": "#1F6FB2", "Classical (Sato ridge)": "#9aa5b1"}


def unet_ridge(pred, img):
    t = time.perf_counter()
    prob = pred.predict(img, tile_size=256, stride=128)[RIDGE_CH]
    return {"scoremap": prob.astype(np.float32)}, time.perf_counter() - t

def classical_ridge(img):
    g = img[0]; g = ((g - g.min()) / (np.ptp(g) + 1e-9) * 255).astype(np.uint8)
    t = time.perf_counter()
    r = ridge_response_classical(g)
    return {"scoremap": r}, time.perf_counter() - t

def yolo_t5_ridge(model, img):
    # Feed the SAME input train-5 saw in training: Alireza's PNGs were the full
    # 3-channel preprocessed npz image (normalized/CLAHE/Sobel) as HWC uint8 —
    # not a replicated grayscale channel (cf. yolo_repr's crater-run convention).
    rgb = np.transpose(img, (1, 2, 0))
    rgb = (rgb * 255).clip(0, 255).astype(np.uint8) if rgb.max() <= 1.0 else rgb.clip(0, 255).astype(np.uint8)
    t = time.perf_counter()
    r = model.predict(rgb, verbose=False, conf=0.001)[0]
    dt = time.perf_counter() - t
    H, W = img.shape[1:]; sm = np.zeros((H, W), np.float32); scores = []
    boxes = r.boxes
    for b, s, c in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.cls.cpu().numpy()):
        if int(c) != YOLO_RIDGE_CLS:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in b]
        x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(W, x2), min(H, y2)
        sm[y1:y2, x1:x2] = np.maximum(sm[y1:y2, x1:x2], s); scores.append(float(s))
    return {"scoremap": sm, "scores": np.array(sorted(scores, reverse=True))}, dt


def main():
    df = pd.read_csv(os.path.join(ROOT, "data/MR/tiles/index.csv"))
    _, val = spatial_train_val_split(df, 0.2, 256, 1024, 42)
    # keep tiles that actually contain ridge GT
    rows = []
    for _, r in val.iterrows():
        d = np.load(os.path.join(ROOT, "data/MR", r["tile_path"]))
        if (d["mask"][RIDGE_CH] > 0).sum() > 0:
            rows.append(r)
    rv = pd.DataFrame(rows).reset_index(drop=True)
    rv = rv.iloc[np.random.default_rng(SEED).permutation(len(rv))].reset_index(drop=True)
    ncal = int(round(CALIB_FRAC * len(rv)))
    calib_df, test_df = rv.iloc[:ncal], rv.iloc[ncal:]

    def load(row):
        d = np.load(os.path.join(ROOT, "data/MR", row["tile_path"]))
        return d["image"].astype(np.float32), (d["mask"][RIDGE_CH] > 0)
    calib_imgs = [load(r) for _, r in calib_df.iterrows()]
    test_imgs = [load(r) for _, r in test_df.iterrows()]
    test_gtcnt = [C.count_components(gt) for _, gt in test_imgs]
    prev = float(np.mean([gt.mean() for _, gt in test_imgs]))
    print(f"device={C.DEV} | ridge-bearing tiles={len(rv)} | calib={len(calib_imgs)} test={len(test_imgs)} | test prevalence={prev:.4f}")

    # Architecture MUST match the checkpoint: the spatial ablation trained the
    # '+both' config (use_residual=True, dropout=0.3). Loading these weights into
    # the default non-residual SmallUNet silently scrambles the forward pass.
    unet = Predictor(SmallUNet(3, len(CLASS_NAMES), base_width=32, use_residual=True, dropout=0.3),
                     weights_path=os.path.join(ROOT, UNET_CKPT), device=C.DEV)
    mrcnn, names, _ = maskrcnn_loader.build_and_load(os.path.join(ROOT, "data/MR/weights/best_model.pth"), device=C.DEV)
    rlab = names.index("wrinkle_ridge")
    yolo5 = YOLO(os.path.join(ROOT, YOLO_T5_CKPT))
    methods = {
        "U-Net (semantic)":      dict(fn=lambda im: unet_ridge(unet, im),                 grid=np.round(np.arange(0.05, 0.91, 0.05), 2)),
        "Mask R-CNN (instance)": dict(fn=lambda im: C.maskrcnn_repr(mrcnn, im, rlab),       grid=np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])),
        "YOLOv8 train-5 (class1)": dict(fn=lambda im: yolo_t5_ridge(yolo5, im),            grid=np.array([0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5])),
        "Classical (Sato ridge)": dict(fn=lambda im: classical_ridge(im),                  grid=np.round(np.arange(0.05, 0.61, 0.05), 2)),
    }

    rows_out = []
    for name, spec in methods.items():
        print(f"  {name} ...", flush=True)
        creps = [spec["fn"](im)[0] for im, _ in calib_imgs]
        treps, ttimes = [], []
        for im, _ in test_imgs:
            rep, dt = spec["fn"](im); treps.append(rep); ttimes.append(dt)
        tau = C.best_tau(creps, [gt for _, gt in calib_imgs], spec["grid"])
        m = C.evaluate(treps, [gt for _, gt in test_imgs], test_gtcnt, tau, ttimes)
        m.update(method=name, tau=float(tau), n=len(treps)); rows_out.append(m)
        print(f"    tau={tau:.2f}  F1={m['f1']:.3f}  IoU={m['iou']:.3f}  MCC={m['mcc']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  {m['ms_per_tile']:.0f}ms")

    # predict-all reference
    ones = [{"binmask": np.ones((256, 256), bool)} for _ in test_imgs]
    pa = C.evaluate(ones, [gt for _, gt in test_imgs], test_gtcnt, 0.5, [0.0] * len(test_imgs))
    pa.update(method="Predict-all (baseline)", tau=0.5, n=len(test_imgs)); rows_out.append(pa)

    out = pd.DataFrame(rows_out)[["method", "tau", "n", "precision", "recall", "f1", "iou", "mcc", "ms_per_tile"]]
    out.to_csv(os.path.join(HERE, "ridge_results.csv"), index=False)
    print("\n=== WRINKLE_RIDGE — shared protocol (ridge-bearing spatial-val tiles) ===")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # figure: metric bars + qualitative
    mp = out.set_index("method")
    order = ["U-Net (semantic)", "Mask R-CNN (instance)", "YOLOv8 train-5 (class1)", "Classical (Sato ridge)"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, metric, ttl in zip(axes, ["mcc", "f1", "iou"], ["MCC (skill)", "Pixel F1", "Pixel IoU"]):
        ax.bar(range(len(order)), [mp.loc[m, metric] for m in order],
               color=[COLORS[m] for m in order])
        ax.set_xticks(range(len(order))); ax.set_xticklabels(["U-Net", "Mask R-CNN", "YOLO t5", "Classical"], fontsize=9)
        ax.set_title(ttl)
        if metric != "mcc":
            ax.axhline(mp.loc["Predict-all (baseline)", metric], color="red", ls="--", lw=1, label="predict-all")
            ax.legend(fontsize=8)
        for i, m in enumerate(order):
            ax.text(i, mp.loc[m, metric], f"{mp.loc[m, metric]:.2f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("Wrinkle-ridge detection (the class with genuine signal) — leakage-free spatial val", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_ridge_metrics.png"), dpi=140)
    print("\nwrote fig_ridge_metrics.png")

    # qualitative on 3 ridge-rich test tiles
    areas = [(i, gt.sum()) for i, (_, gt) in enumerate(test_imgs)]
    pick = [i for i, _ in sorted(areas, key=lambda x: -x[1])[:3]]
    taus = {r["method"]: r["tau"] for r in rows_out}
    cols = ["Tile", "GT ridge", "U-Net", "Mask R-CNN", "YOLO t5", "Classical"]
    fig, axes = plt.subplots(3, len(cols), figsize=(2.1 * len(cols), 6.3))
    for r, i in enumerate(pick):
        img, gt = test_imgs[i]
        panels = [img[0], gt,
                  C.binmask_at(unet_ridge(unet, img)[0], taus["U-Net (semantic)"]),
                  C.binmask_at(C.maskrcnn_repr(mrcnn, img, rlab)[0], taus["Mask R-CNN (instance)"]),
                  C.binmask_at(yolo_t5_ridge(yolo5, img)[0], taus["YOLOv8 train-5 (class1)"]),
                  C.binmask_at(classical_ridge(img)[0], taus["Classical (Sato ridge)"])]
        for c, (ttl, p) in enumerate(zip(cols, panels)):
            ax = axes[r, c]; ax.imshow(p, cmap="gray" if c == 0 else "viridis"); ax.set_xticks([]); ax.set_yticks([])
            if r == 0: ax.set_title(ttl, fontsize=10)
    fig.suptitle("Wrinkle-ridge predictions on ridge-rich validation tiles", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_ridge_qualitative.png"), dpi=130)
    print("wrote fig_ridge_qualitative.png")


if __name__ == "__main__":
    main()
