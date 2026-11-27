# Test-dev review

The snapshot contains 505 measured cells on 1,610 VisDrone Test-dev images. It combines 100 uniform FM_P draws with the targeted P18 check. This limited benchmark uses previously observed Test-dev data.

## Reproduce

From the Git root, run with Python, NumPy and Matplotlib.

```powershell
python -B experiments/uav_05_evaluation/report_testdev.py --out ../datasets/local_runs/uav_05_evaluation/testdev_101p_5d/review_brief
```

| Output | Inputs and interpretation |
|---|---|
| `report_tables.json` | Display-ready cells for the benchmark, Top 10 rankings, clean references, augmentation results, and pilot plan. AP is rounded to two decimals in percent and AP change is in percentage points. The Tiny-band selections are proposed. |
| `*_table.png` | The report tables rendered from the same display-ready data. Selection and timing tables are proposed. |
| `top_rankings.csv`, `testdev_top_rankings.png` | Sealed `testdev_metrics.csv`. Rank the best detector for each P by the target metric. All size APs and cached latency come from that same P–D cell. AP values in the CSV are fractions. |
| `detector_summary.csv`, `testdev_campaign.png` | The 100 uniform P conditions in `testdev_metrics.csv`. AP and AR are condition means in percent. Latency is the per-cell mean cached pipeline latency in ms/image. |
| `input_size_benefits.png`, `input_size_benefits.csv`, `resolution_effects.csv` | Same-checkpoint I1280 minus I640 contrasts. Clean Val and P18 use `clean_val_resolution_source.csv` and `p18_resolution_contrasts.csv`, respectively. Changes are percentage points. |
| `selected_input_size_cases.csv` | Four examples with same-checkpoint I640 and I1280 levels plus changes for both detectors. Each row identifies its two original runs. These are input-size changes. |
| `normal_baselines.png` | `normal_baselines.csv`. Additional clean I1280 models are listed in `normal_i1280_reference.csv`. |
| `augmentation_by_size.png` | `augmentation_reference.csv`. Measured, one training seed, paired image intervals. |
| `training_case_reference.csv` | Paired I1280 records from the completed VM37 ALL_FIVE and VM48 Motion-only pilots on VisDrone Val (548 images). The VM43 Gaussian pair uses Test-dev. Comparisons are within each run. |
| `clean_reference_comparison.csv` | Same-checkpoint clean Val AP at I640/I1280 for all eight models. `clean_val_resolution_source.csv` is the verified Week 3 excerpt. |
| `tiny_threshold_summary.csv`, `tiny_threshold_bins.csv` | Tiny AP attainment and disjoint bins for the 500 uniform cells. Best-D counts use unique P. P18 is separate. |
| `tiny_bands.csv`, `augmentation_pilot_plan.csv`, `review_summary.json` | The better Tiny AP of D2 and D4 (I1280) for each uniform P defines the bins. The pilot schedule is proposed. It records each reference detector and its same-cell size APs. |

The script checks source SHA256 values before processing. The frozen plan excerpts preserve P recipes, render seeds, D settings and checkpoint hashes. The manifest records original runs and hashes. Raw predictions, weights and images are archived locally.

## Augmentation scope

The seed-42 Gaussian pair is an exploratory reuse of existing T960 checkpoints. The first stage includes clean images and two conditions from each populated Tiny band. The empty >8% Test-dev band remains empty. P18 is an additional targeted check. D4/D5 runtime measurements plus cache estimates are scheduling proxies.

The research plan's RQ4 uses VisDrone T1280 baselines for YOLOv8s and YOLO26s, matched training schedules and seeds, five augmentation families, and one ablation per added family on YOLOv8s. Training uses parameter ranges that exclude the exact test values. Re-evaluate the same sampled P conditions to measure changes in empirical coverage and clean AP.

