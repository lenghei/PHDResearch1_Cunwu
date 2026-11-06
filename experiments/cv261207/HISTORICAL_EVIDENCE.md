# CV261207: historical cross-model interaction review

Status: COMPLETE_WITH_LIMITATIONS. Independent test admission: DATASET_BLOCKED.
Formal measured pairs: 0/500. No new GPU inference, training or archive copying.

## Evidence

Historical RunID: vm26_r001. Review: `cv261207_vm26review_20261106T212756_fdf80ed8`. Gate review: `cv261207_historyreview_20261106T212926_539e1709`.
The 54 cells comprise three checkpoints × three resolutions (640/960/1280) ×
two blur settings (B0/B2) × three extra-JPEG settings (None/Q85/Q50).
All use the same 300 ds04 development images, native10 labels and pr01 FP32 evaluation.
Illumination and noise are fixed at I0/N0. This is not the full degradation space.
JPEG None means no additional encode/decode; the source images are already JPEGs.

All 54 metric and raw-prediction SHA256 identities and 300-record memberships passed.
Original GT contains 21,756 effective instances. All 24 endpoint bootstrap arrays
(200 shared image draws each) and their paired index/source receipts passed.
Two configurations were independently replayed from raw predictions through ignore-union
filtering and COCO matching. Their full-data AP/AR and six selected aggregate bootstrap
draws matched exactly. The remaining 4,794 saved endpoint draws were integrity-checked,
not freshly rematched. No per-image AP averaging was used.

## AP50:95 resolution differences

Values below are percentage points, 1280 minus 640. AP itself remains in [0,1] in CSVs.

| Checkpoint | B0 None | B0 Q85 | B0 Q50 | B2 None | B2 Q85 | B2 Q50 |
|---|---:|---:|---:|---:|---:|---:|
| YOLO26s | +11.235 | +11.004 | +7.953 | -2.181 | -2.089 | -2.064 |
| YOLOv8s | +11.541 | +11.103 | +7.969 | -1.855 | -1.819 | -1.790 |
| YOLO11s | +12.057 | +11.633 | +8.383 | -1.612 | -1.607 | -1.463 |

Let G(b,j) = AP1280(b,j) − AP640(b,j). The three-way contrast is
[G(B2,Q50) − G(B2,None)] − [G(B0,Q50) − G(B0,None)].

| Checkpoint | Three-way contrast (pp) | 95% simultaneous interval (pp) |
|---|---:|---:|
| YOLO26s | +3.399 | [+2.537, +4.260] |
| YOLOv8s | +3.637 | [+2.754, +4.520] |
| YOLO11s | +3.823 | [+2.923, +4.724] |

The original max-standardized centered bootstrap family has 36 contrasts.
Its critical value was reproduced from the 200 paired draws. These are conditional
image-bootstrap intervals, not training-seed or independent-dataset intervals.
There is no reliable flight clustering. Q85 and 960 confidence intervals remain NOT_EVALUATED.

The no-blur JPEG effect on the resolution gain is negative for every checkpoint,
with its joint interval below zero. The severe-blur JPEG effect has an interval
containing zero for every checkpoint. The positive three-way contrast describes
this difference between conditions; it does not mean JPEG repairs blur.
The signs of the 1280−640 gains are positive in all nine B0 cases and negative
in all nine B2 cases. Only two blur levels and fixed I0/N0 were tested here.

## Interpretable rules and validation

A depth-2 decision tree fitted 18 measured resolution-gain rows using categorical
checkpoint, blur and JPEG indicators. Its source, predictions and fitted rules are saved.
It describes this finite observed grid; it is not a new sampling algorithm.
Held-out image-and-configuration validation: NOT_EVALUATED. Every aggregate response
shares the same 300 images. Random configuration splits, checkpoint splits and bootstrap
replicas cannot establish image-disjoint validation. No such score is reported.
Differences between three selected checkpoints do not establish architecture causation.

## Independent testing and C1–C4

| Contribution | Status | Remaining evidence |
|---|---|---|
| C1 | VALIDATED for existing UVL v1 logic/generator | Executed configurations do not cover the full formal space. |
| C2 | PARTIAL: development interaction evidence | Unseen image/scene and configuration groups; frozen-rule confirmation; exact model provenance. |
| C3 | PARTIAL: static definitions only | Measured AP/cost failures, fixed task thresholds and independent static-to-empirical prediction validation. |
| C4 | NOT_EVALUATED | Targeted augmentation remains untested. |

Current train/val/test-dev SHA and ID disjointness is verified in the prior sealed
admission. The bounded follow-up reconciles named Phase2/VM11/VM15/VM34 records.
It does not establish a completed historical test-dev evaluation, nor prove none occurred.
Creation-time training membership and complete manual/external selection/tuning records
remain missing. The independent gate is unchanged.
The supervisor plan proposes a separately admitted prospective native10 aerial cohort
if those records cannot be recovered. No dataset is substituted and no inference is started.

## Sources and reproduction

Local evidence: `cv261207_vm26review_20261106T212756_fdf80ed8` and `cv261207_historyreview_20261106T212926_539e1709`, each with SHA256SUMS.txt.
Source-to-cell identities are in provenance.json; metric and prediction files stay local.
Figures: resolution_blur_jpeg, checkpoint_resolution_effects, contrast_confidence_intervals,
rules_and_validation (PNG/SVG), with figure_sources.json. Open the review index.html.

```powershell
.\.venv\Scripts\python.exe -B main/experiments/cv261207/historical_review.py review
.\.venv\Scripts\python.exe -B main/experiments/cv261207/history_gate_review.py
.\.venv\Scripts\python.exe -B -m unittest discover -s main/experiments/cv261207/tests -v
```

Each review creates a new immutable RunID. Existing results are not overwritten.
