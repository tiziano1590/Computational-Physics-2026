# Lunar crater detection — head-to-head comparison of the four methods

*Generated from the local integration branch. Reproduce with the commands at the bottom.*

## 1. The problem: four methods, three different tasks

The group has four crater-finding methods, but they output different things:

| Method | Author | Task type | Native output |
|---|---|---|---|
| U-Net | Giancarlo | semantic segmentation | per-pixel class probabilities |
| Mask R-CNN | Amir | instance segmentation | per-instance box + mask + score |
| YOLOv8 | Alireza | object detection | per-object box + score |
| Classical | *(newly implemented here)* | thresholding + morphology | binary mask + components |

You cannot compare an mAP against a pixel-IoU against a thresholded mask. So the comparison only means something if every method is reduced to **the same prediction on the same data, scored with the same metric**.

## 2. Shared protocol

- **Data & split:** the leakage-free **spatial** train/val split (`spatial_train_val_split`, seed=42) on the Marius Hills tiles — the same split the U-Net study uses. Evaluation is on held-out validation tiles only.
- **Common prediction:** every method is reduced to a **binary impact-crater pixel mask** (256×256):
  - U-Net → crater-channel probability, thresholded;
  - Mask R-CNN → union of crater-class instance masks with score ≥ τ;
  - YOLOv8 → union of predicted boxes with score ≥ τ (the model collapses to a single class);
  - Classical → threshold + morphology output.
- **Ground truth:** mask channel 0 (`impact_crater`) > 0.
- **Thresholds (τ)** for the score-based methods are tuned for best micro-F1 on a **calibration** subset, then reported on a **disjoint test** subset — no tuning on the test data.
- **Metrics:** pixel precision / recall / F1 / IoU, **MCC** (Matthews correlation — robust to class imbalance), crater **count MAE** (connected components, area 30–6000 px), and inference **ms/tile**.

## 3. The base-rate trap (why the obvious comparison is wrong)

The `impact_crater` channel is **extremely prevalent**: across the validation tiles its mean pixel coverage is **0.81** (median 0.96) — most tiles are almost entirely labelled crater. **2,489 of 3,248 val tiles are >70 % crater.**

On such data the pixel metrics are dominated by base rate. A trivial **"predict every pixel is crater"** baseline scores **F1 ≈ 0.90, IoU ≈ 0.82**. Full all-val results (prevalence 0.82):

| (all val, prevalence 0.82) | F1 | IoU | **MCC** | Precision | Recall | ms/tile |
|---|---|---|---|---|---|---|
| YOLOv8 | 0.916 | 0.846 | **0.470** | 0.893 | 0.940 | 9 |
| U-Net | 0.903 | 0.823 | **0.013** | 0.823 | 1.000 | 12 |
| *Predict-all baseline* | *0.903* | *0.823* | *0.000* | *0.823* | *1.000* | *0* |
| Classical | 0.374 | 0.230 | **0.078** | 0.878 | 0.238 | 1 |
| Mask R-CNN | 0.116 | 0.061 | **−0.166** | 0.616 | 0.064 | 918 |

Two things this exposes:
1. **The U-Net's headline 0.90 F1 is not skill** — F1/IoU match the predict-all baseline and MCC ≈ 0. It is exploiting the base rate, nothing more.
2. **The ranking is regime-dependent.** On these dense tiles YOLOv8 has genuine skill while Mask R-CNN goes *negative* (−0.166) — out-of-distribution, since its instance prep skips >20 % coverage tiles. "Which model is best" depends entirely on crater density. The calibrated, report-grade per-regime numbers are in §4 (this all-val snapshot uses a separate threshold calibration, so YOLO's MCC here is higher than its §4 dense figure — see the operating-point caveat in §6).

## 4. Report-grade per-regime comparison

The best method depends on crater density, so the headline result is **per density band**. Each regime is calibrated **on its own**: within the regime, tiles are split into a disjoint calibration / test set (40/60), τ is chosen on calibration (max F1), and metrics are reported on held-out test tiles — every method shown at its own per-regime optimum, no test-set tuning. See **`fig_regime.png`** (MCC and F1 vs crater coverage) and `fig_qualitative.png`.

**Sparse / discriminative regime (coverage 1–20 %, n = 94 test, prevalence ≈ 0.09)** — also the regime Mask R-CNN was trained for (`max_channel_fraction=0.20`):

| Method | F1 | IoU | **MCC** | Precision | Recall | Count MAE | ms/tile | τ |
|---|---|---|---|---|---|---|---|---|
| **Mask R-CNN** (instance) | 0.279 | 0.162 | **0.187** | 0.214 | 0.401 | 83 | 924 | 0.20 |
| **YOLOv8** (detection) | 0.253 | 0.145 | **0.159** | 0.163 | 0.563 | 106 | 10 | 0.05 |
| **Classical** (thr+morph) | 0.142 | 0.076 | **0.069** | 0.173 | 0.120 | 16 | 1 | — |
| **U-Net** (semantic) | 0.176 | 0.097 | **−0.000** | 0.097 | 1.000 | 24 | 13 | 0.30 |

**Dense regime (coverage ≥ 20 %, n = 192 test)** — the bulk of the dataset (~77 % of tiles):

| Method | F1 | IoU | **MCC** | Precision | Recall | Count MAE | ms/tile | τ |
|---|---|---|---|---|---|---|---|---|
| **YOLOv8** (detection) | 0.947 | 0.899 | **0.240** | 0.905 | 0.992 | 13 | 9 | 0.10 |
| **U-Net** (semantic) | 0.945 | 0.895 | **−0.001** | 0.895 | 1.000 | 5 | 13 | 0.30 |
| **Classical** (thr+morph) | 0.387 | 0.240 | **0.031** | 0.912 | 0.245 | 49 | 1 | — |
| **Mask R-CNN** (instance) | 0.132 | 0.071 | **−0.111** | 0.781 | 0.072 | 41 | 921 | 0.05 |

The U-Net F1 of **0.945** on dense tiles next to its **MCC of −0.001** is the base-rate trap in one line: near-perfect-looking F1, zero actual skill (precision = prevalence, recall = 1.0 — it predicts crater everywhere).

The flip is unambiguous: **Mask R-CNN** leads on sparse tiles (the only method with real skill there) and goes to **negative MCC** on dense tiles (out-of-distribution); **YOLOv8** is best on dense tiles; on crater, **U-Net never beats chance** (MCC ≈ 0) at any density — even where its F1 reaches 0.75, that is base rate, not skill (`fig_regime.png`, right panel). *But crater is the saturated, near-trivial class; the U-Net's real ability shows on ridges — see §5.*

## 5. Wrinkle-ridge — the class that actually has signal

`impact_crater` is saturated (§3), so it cannot discriminate methods. `wrinkle_ridge` is the opposite — sparse (~0.6 % globally, ~2.6 % on the 524 ridge-bearing val tiles) and the report's strongest genuine result. **This is where the comparison is meaningful.** Evaluated on the ridge-bearing spatial-val tiles (210 calib / 314 test, own-threshold calibration). The U-Net here is the report's definitive spatial checkpoint (`noaug_w32_t10000_e30.pt`, the residual `+both` config). The YOLO entry is the **train-5** run: it was trained on `yolo_dataset_no_class0` (crater dropped, channels re-indexed), so its `class1` — the only class it ever detects — is `wrinkle_ridge`; its boxes are painted as a ridge score map, fed the same 3-channel input its training PNGs used.

| Method | F1 | IoU | **MCC** | Precision | Recall | ms/tile | τ |
|---|---|---|---|---|---|---|---|
| **U-Net** (semantic) | 0.460 | 0.299 | **0.452** | 0.527 | 0.409 | 20 | 0.25 |
| **Mask R-CNN** (instance) | 0.380 | 0.235 | **0.366** | 0.414 | 0.352 | 1130 | 0.50 |
| **YOLOv8 train-5** (detection) | 0.099 | 0.052 | **0.076** | 0.063 | 0.228 | 10 | 0.01 |
| **Classical** (Sato ridge) | 0.040 | 0.020 | **−0.028** | 0.021 | 0.392 | 29 | — |
| *Predict-all (baseline)* | 0.050 | 0.026 | 0.000 | 0.026 | 1.000 | 0 | — |

The conclusion **flips versus crater**:
- **The U-Net is the best method on ridges** (MCC 0.452, F1 0.460) and ~55× faster than Mask R-CNN. So the U-Net is *not* skill-less — on the saturated crater class nothing can be (§3), but on the class with genuine signal it wins. `fig_ridge_qualitative.png` shows it recovering ridge topology most completely.
- **Mask R-CNN is a close, genuine second** (MCC 0.366).
- **YOLO train-5 has real but weak ridge skill** (MCC 0.076 — positive, unlike classical). Its positive score doubles as empirical confirmation of the `no_class0` re-indexing (a wrong label mapping would score ≈0). The weakness is representational, not just training: axis-aligned box fill is a poor match for thin diagonal linear features (precision 0.063), whereas the pixel-level methods can trace them. A four-way task-type lesson in one row.
- **All three learned methods beat the classical Sato baseline** (MCC −0.028, no better than chance) and the predict-all baseline. Ridges are too subtle and variable for a fixed ridge filter — classical fires on crater rims and texture everywhere (`fig_ridge_qualitative.png`, right column). See also `fig_ridge_metrics.png`.

This is the headline of the whole comparison: **on the only class with real signal, the learned models clearly beat the classical baseline, and the simple semantic U-Net edges out the heavier instance model at 1/60th the cost.**

## 5b. South-pole reconciliation — why the manuscript's two tables disagree (SOLVED)

The June manuscript contained two contradicting south-pole result sets: the U-Net section's Table 8 (crater avg-max-prob 0.86, other classes near zero) and the duplicate section's Table 12 (ALL seven classes ≈262,144 detections — every pixel — avg-max-prob 0.48–0.54). Both cannot describe the same trained model. `giuseppe_preproc_check.py` tests the two candidate mechanisms on the same south-pole tiles (8 tiles; per-tile max prob averaged; fraction of pixels above the workflow's own 0.1 display threshold):

| Configuration | crater max-prob | other 6 classes max-prob | pixels > 0.1 |
|---|---|---|---|
| trained weights + package preprocessing (Table 8 conditions) | 0.996 | 0.00–0.14 | crater only |
| trained weights + the duplicate workflow's OpenCV preprocessing | 0.832 | 0.00–0.06 | crater only |
| **random weights (pre-fix loader behaviour)** | **0.485** | **0.48–0.54** | **100 % — every class** |

Only the third row reproduces Table 12's signature. **Table 12 was produced by the checkpoint-loading bug**: the pre-fix `Predictor` loaded `best_trained.pth` with `strict=False` against legacy `block.*` key names, silently matched zero conv layers, and ran inference on randomly initialised weights — sigmoid of random logits ≈ 0.5 everywhere, so every pixel of every class clears a 0.1 threshold. The workflow's OpenCV-vs-skimage preprocessing mismatch (CLAHE clipLimit 2.0/8×8 grid vs `equalize_adapthist` 0.03; max-normalised Sobel vs `filters.sobel`) is real and measurably degrades the crater response (0.996 → 0.832), but it does NOT produce the uniform-0.5 pattern. For the final manuscript: drop Table 12, keep one south-pole section, and cite the loader fix (`inference/predictor.py`) — the contradiction has a demonstrated root cause, not a modelling disagreement.

## 6. What we actually learn

- **U-Net has no skill in either regime.** MCC is −0.000 (sparse) and −0.001 (dense) — flat along zero at every crater density (`fig_regime.png`), even where its F1 reaches 0.945. It predicts crater over almost the whole tile (recall ≈ 1.0, precision ≈ prevalence); the qualitative panel shows an all-crater column. It learned the dataset's dominant base rate, not crater shape. This is the single most important finding and it directly explains the weak numbers everyone reported.
- **Mask R-CNN is the only method with clear skill on sparse tiles** (MCC 0.187) — it genuinely localizes craters — but **~90× slower** (≈920 ms/tile vs ~10) and out-of-distribution on dense tiles (MCC −0.111), where its instance prep was never meant to operate.
- **YOLOv8** is the strongest on the dense majority (MCC 0.240) and a solid speed/accuracy compromise on sparse tiles (MCC 0.159 at 10 ms/tile), though it over-detects at low confidence (count MAE 106 on sparse).
- **The classical baseline beats the U-Net on sparse tiles** (MCC 0.069 vs ≈0) at 1 ms/tile and fully interpretable. That a 30-line OpenCV function out-discriminates the trained U-Net there is a genuine result, not a throwaway.
- **The answer depends on the class, and that *is* the finding.** On the saturated crater channel no method beats the base rate (§3–4). On `wrinkle_ridge`, the class with genuine signal, the **U-Net wins (MCC 0.45) > Mask R-CNN (0.37) ≫ YOLO train-5 (0.08) > classical (≈0)** — every learned method beats the classical baseline, and pixel-level methods beat box-level ones on linear features (§5). Reporting one global "best model" would be wrong both ways.
- **Best method also depends on crater density (regime flip).** Sparse crater (1–20 %): Mask R-CNN > YOLO > Classical > U-Net ≈ chance. Dense (≥20 %): YOLO > Classical > U-Net ≈ chance > Mask R-CNN (negative). No method dominates everywhere — the right framing is per-class and per-regime, not one leaderboard.
- **Trade-off summary:** accuracy (MCC) is class/regime-dependent (above); speed → Classical (1 ms) ≈ YOLO ≈ U-Net (~10–15 ms) ≫ Mask R-CNN (~920 ms, ~60–90×); interpretability → Classical > the rest. On ridges specifically, the U-Net is both the most accurate *and* far cheaper than Mask R-CNN.

## 7. Limitations (be honest about these)

- The shared metric is **pixel coverage of the crater class**; it structurally favours area-covering methods (semantic/box-fill) over instance methods. The count metric and MCC partly offset this, but no single number is perfectly fair across task types.
- **YOLO's label map — RESOLVED (2 Jul).** The committed `preprocessing.ipynb` generates boxes with `class_id` = mask channel index, so in the 7-class `yolo_dataset` **`class0` = `impact_crater`**: the crater comparison's treatment of run-1 boxes is correct, not an assumption. The later runs (train-2…5) used `yolo_dataset_no_class0` — crater channel dropped, remaining channels re-indexed 0–5 — so *their* `class1` = `wrinkle_ridge`; train-5's confusion matrix shows `class1` is the only class it detects, and its positive ridge MCC (§5) validates the re-indexing empirically. One residual fragility: the harness picks weights via `sorted(glob("train*/weights/best.pt"))[-1]`, which selects the 7-class `train/` run only because ASCII `-` sorts before `/` — pin the path explicitly if more runs are added.
- Mask R-CNN GT "count" via connected components on a dense semantic mask is noisy; count MAE should be read as indicative, not definitive.
- **Operating-point sensitivity:** MCC/F1 depend on the threshold τ. Each regime is now calibrated on its *own* disjoint calib set (τ shown per row); rankings and the regime flip are robust, but absolute values shift with τ — always report it.
- Numbers are on **286 held-out test tiles** (94 sparse / 192 dense; 191 calibration tiles total) for crater, and 314 test / 210 calib ridge-bearing tiles for ridge. Aggregates are stable; within the dense regime the 40–70 % band is undersampled (n≈22, the noisy mid-density dip in `fig_regime.png`). Widen the caps for final-manuscript precision.
- **Architecture-matching footgun (resolved here, but a group reproducibility risk):** the U-Net checkpoints do not record `use_residual` in a way the loader auto-applies. The crater checkpoint (`best_trained.pth`) is non-residual; the ridge/spatial checkpoint (`noaug_w32_t10000_e30.pt`) is the residual `+both` config. Loading a residual checkpoint into the default non-residual `SmallUNet` runs without error but silently scrambles the forward pass (ridge AP collapses 0.38→0.03). The loader's "unexpected keys" warning flags it — *heed it*. Each checkpoint must be loaded with the architecture it was trained with.
- The classical ridge baseline is a single Sato filter; a stronger classical pipeline (directional morphology, Frangi, tuned thresholds) might do better, though ridges' low contrast makes a large gain unlikely.

## 8. Implications for the final report

1. The comparison framework here gives the report a single organic spine: one split, one reduction, one metric set, a trivial baseline to calibrate claims, and per-class/per-regime breakdowns.
2. **Lead with MCC / skill-over-baseline, not F1/IoU** — otherwise the report repeats the base-rate mistake.
3. **Structure the comparison by class.** Crater = saturated → nobody beats base rate (a finding, not a failure). `wrinkle_ridge` = real signal → U-Net (MCC 0.45) > Mask R-CNN (0.37) ≫ YOLO train-5 (0.08) > classical/baseline, at a fraction of Mask R-CNN's cost. This is the informative head-to-head — genuinely four-way, one learned method per author plus the classical anchor — and it shows the U-Net's real capability, which the saturated crater metrics obscure.
4. There is now a real **classical baseline** (crater: morphology; ridge: Sato) to anchor the comparison — and the learned models clearly beat it where it matters.

## 9. Reproduce
```bash
# wrinkle-ridge comparison — the class with real signal (writes ridge_results.csv + fig_ridge_*.png)
KMP_DUPLICATE_LIB_OK=TRUE python Moon-Recognition/comparison/ridge_comparison.py
# south-pole Table 8 vs Table 12 reconciliation (writes giuseppe_preproc_check.csv)
KMP_DUPLICATE_LIB_OK=TRUE python Moon-Recognition/comparison/giuseppe_preproc_check.py
# report-grade per-regime crater analysis (writes regime_*.csv + fig_regime.png)
KMP_DUPLICATE_LIB_OK=TRUE python Moon-Recognition/comparison/regime_analysis.py
# single-band crater views (optional)
KMP_DUPLICATE_LIB_OK=TRUE python Moon-Recognition/comparison/compare_models.py \
    --frac_lo 0.01 --frac_hi 0.20 --calib 40 --test 117 --tag fairband
KMP_DUPLICATE_LIB_OK=TRUE python Moon-Recognition/comparison/compare_models.py \
    --frac_lo 0.0 --frac_hi 1.01 --calib 30 --test 90 --tag allval
# bar + qualitative figures
KMP_DUPLICATE_LIB_OK=TRUE python Moon-Recognition/comparison/make_figures.py
```
Files: `ridge_comparison.py` (ridge head-to-head), `regime_analysis.py` (per-regime crater), `compare_models.py` (single-band harness), `classical_baseline.py` (classical crater + ridge), `maskrcnn_loader.py` (rebuilds Amir's net), `ridge_results.csv` / `regime_*.csv` / `comparison_results_*.csv`, `fig_ridge_metrics.png`, `fig_ridge_qualitative.png`, `fig_regime.png`, `fig_metrics.png`, `fig_qualitative.png`.
