# Development evidence update

Analysis `cv261207_development_analysis_20261108T194456_6e430eb3`. New measured pairs: **12/12**.
Formal independent pairs remain **0/500 — DATASET_BLOCKED**.

The 248 validation IDs are outside the historical 300 fitting IDs and byte hashes. They include 247 unique image byte hashes.
Validation was historically available for checkpoint selection. These are development results, not independent test results.

Together, the historical 300 and new 248 IDs cover all 548 validation images. None is an untouched future test image; aggregate AP is not pooled across the different study protocols.
AP denotes AP50:95 over IoU 0.50:0.05:0.95 and 101 recall points. AP50 is reported separately. AR averages recall across the same IoUs at the custom 3,000 detection cap.
Original-coordinate area bins are tiny <256; small 256–<1024; medium 1024–<9216; large >=9216 square pixels. The custom bins are disjoint. Missing class metrics remain null in source JSON and blank in CSV.
The native10 ignore-union protocol and dense detection cap differ from official VisDrone leaderboard evaluation.

## Measured AP and cost

| Checkpoint | Condition | Input | AP50 | AP50:95 | Mean pipeline ms |
|---|---|---:|---:|---:|---:|
| ck01 | Downscale x1.5 | 1280 | 53.715% | 33.086% | 56.977 |
| ck01 | Extra JPEG Q25 | 1280 | 46.539% | 27.665% | 63.867 |
| ck01 | Downscale x1.5 | 640 | 40.605% | 23.605% | 50.543 |
| ck01 | Extra JPEG Q25 | 640 | 37.532% | 21.478% | 40.675 |
| ck05 | Downscale x1.5 | 1280 | 53.842% | 33.137% | 53.812 |
| ck05 | Extra JPEG Q25 | 1280 | 46.455% | 27.311% | 56.385 |
| ck05 | Downscale x1.5 | 640 | 38.231% | 22.073% | 38.264 |
| ck05 | Extra JPEG Q25 | 640 | 35.768% | 20.654% | 38.851 |
| ck06 | Downscale x1.5 | 1280 | 53.135% | 32.553% | 56.767 |
| ck06 | Extra JPEG Q25 | 1280 | 45.607% | 27.244% | 60.492 |
| ck06 | Downscale x1.5 | 640 | 38.381% | 22.135% | 38.757 |
| ck06 | Extra JPEG Q25 | 640 | 35.540% | 20.370% | 45.444 |

The timed pipeline reads, hashes and decodes a cached PNG, predicts/NMS, and writes compressed raw results. Generation, cold initialization, evaluation and bootstrap are separately recorded.
The repeated run reproduced predictions and AP exactly. Its damaged timing was not reconstructed.

Measured unique-pair worker wall: 2640.344 s; GPU forward profiler: 45.049 s. Worker wall is not GPU occupancy. Repeated-run timing and energy are unavailable.

## Same-checkpoint resolution contrasts

| Checkpoint | Condition | Size | AP50:95 difference, pp | Pointwise 95% interval, pp |
|---|---|---|---:|---|
| ck01 | Downscale x1.5 | all | +9.481 | [+8.635, +10.345] |
| ck01 | Downscale x1.5 | tiny | +9.124 | [+7.871, +10.755] |
| ck01 | Downscale x1.5 | small | +12.932 | [+11.491, +14.411] |
| ck01 | Extra JPEG Q25 | all | +6.187 | [+5.310, +6.896] |
| ck01 | Extra JPEG Q25 | tiny | +5.499 | [+4.580, +6.684] |
| ck01 | Extra JPEG Q25 | small | +8.204 | [+7.049, +9.498] |
| ck05 | Downscale x1.5 | all | +11.064 | [+10.101, +12.172] |
| ck05 | Downscale x1.5 | tiny | +9.603 | [+8.278, +10.893] |
| ck05 | Downscale x1.5 | small | +14.118 | [+12.351, +15.986] |
| ck05 | Extra JPEG Q25 | all | +6.656 | [+5.794, +7.636] |
| ck05 | Extra JPEG Q25 | tiny | +5.872 | [+4.941, +7.078] |
| ck05 | Extra JPEG Q25 | small | +8.339 | [+6.788, +10.478] |
| ck06 | Downscale x1.5 | all | +10.419 | [+9.548, +11.282] |
| ck06 | Downscale x1.5 | tiny | +8.877 | [+7.673, +10.133] |
| ck06 | Downscale x1.5 | small | +13.978 | [+12.228, +15.761] |
| ck06 | Extra JPEG Q25 | all | +6.874 | [+6.081, +7.880] |
| ck06 | Extra JPEG Q25 | tiny | +5.987 | [+5.143, +7.061] |
| ck06 | Extra JPEG Q25 | small | +8.915 | [+7.894, +10.645] |

Two conditions do not identify an isolated Blur×JPEG interaction. The historical VM26 interaction estimates remain separate.
The 200 paired image draws do not cover flight clustering, repeated scenes or detector training seeds. Intervals are pointwise.

## Learned responses

The fixed tree, rule-stump and forest definitions were fitted on the historical 279 responses only.
New configurations and image IDs were held out. The two test conditions are out of distribution; this is limited development regression.
Historical joint AP/latency labels are incomplete, so balanced accuracy, macro F1 and failure precision/recall remain unreported.

| Method | Fitting seed | AP MAE | AP RMSE | Resolution-effect MAE | Direction correct / total |
|---|---:|---:|---:|---:|---:|
| rule_stump | 42 | 0.043424 | 0.056172 | 0.084468 | 0/6 |
| tree | 42 | 0.018766 | 0.021573 | 0.018745 | 6/6 |
| forest | 42 | 0.024463 | 0.033264 | 0.021610 | 6/6 |
| rule_stump | 73 | 0.043424 | 0.056172 | 0.084468 | 0/6 |
| tree | 73 | 0.018766 | 0.021573 | 0.018745 | 6/6 |
| forest | 73 | 0.024688 | 0.033791 | 0.022932 | 6/6 |
| rule_stump | 101 | 0.043424 | 0.056172 | 0.084468 | 0/6 |
| tree | 101 | 0.018766 | 0.021573 | 0.018745 | 6/6 |
| forest | 101 | 0.024695 | 0.033717 | 0.022359 | 6/6 |

Errors use AP raw units (0–1). Seeds are fitting seeds, not independent detector training seeds.
All six measured resolution effects are positive. Direction accuracy cannot establish false-relation control without negative cases.

## RQ4

Clean-control canary: **CANARY_PASS**. Targeted augmentation efficacy and gap closure: **NOT_EVALUATED**.
A short canary cannot establish a reliable full-training budget or method efficacy.

## Remaining supervisor decisions

- Resolve checkpoint-specific historical test-dev use, or admit a genuinely new native10 image cohort.
- Decide whether the two-group OOD diagnostic supports expanding a development-only study.
- Approve a measured training budget and matched efficacy protocol separately from engineering readiness.

Source tables, raw prediction/GT checksums and bootstrap draws are indexed in the analysis run.

## Data and uncertainty limits

- One byte-identical image pair remains in the frozen 248-ID population.
- The duplicate pair has 41/40 valid boxes and 2/0 ignore regions; the annotation discrepancy is unresolved.
- Image resampling does not control scene/flight clustering or checkpoint training-seed variability.
- Intervals are pointwise exploratory intervals, without simultaneous familywise correction.

Canary: 64 network exposures, 32 batches, 1 optimizer and 1 EMA update; FP32 save/reload passed.
Supervised worker wall: 29.780 s. Torch peak allocated/reserved: 4031175680/5211422720 bytes.
The 16-image built-in engineering validator reported AP50=0.59943, AP50:95=0.38686. These are not pr01 research metrics or before/after efficacy estimates.

## Protocol identity

Protocol: `CV261207_DEV248_NATIVE10_DENSE_FP32_V1`. Render/bootstrap seed: `20261108`.
Native10; FP32; dense one-to-many path; confidence 0.001; class-aware multi-label NMS IoU 0.7; max_det=3000; max_nms=30000; square letterbox.
Class order: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor.

| ID | Checkpoint SHA256 |
|---|---|
| ck01 | `1b3861902e01416f199f2242cb43618f7680848cabf5f5df9546f2b404f0c488` |
| ck05 | `96c84ece325eccf07edefd231f088f758ba5fa6a75a4b577e8db32a94bfeeb8b` |
| ck06 | `839a3dcc962f2496777b87b7494b0584a74c940e03f75b7e6a7f699df798e0f1` |
