# UAV-GAP

## Research Problem

Which UAV image degradations leave no measured detector setting that meets both tiny-object accuracy and latency requirements? Testing every degradation and detector combination is costly, and a training augmentation's presence alone cannot establish performance.

## Approach

FM_P describes degradations in UVL. FM_D describes detector, inference and training choices. FlamaPy checks logical validity. The experiments compare fixed settings on paired VisDrone images. Static training coverage is a mapping assumption. The empirical Gap requires evidence at a fixed Tiny AP50:95 threshold of 10% and a cached-pipeline latency limit of 100 ms.

## Main Contributions

- Two verified feature models: 46,368 legal degradation settings and 635,904 legal detector settings.
- A measured Gap assessment and a detector lookup that can abstain when the evidence is insufficient.
- Paired factorial comparisons and matched augmentation pilots with reported benefits, costs and limits.

## Key Results

| Study | Measured result |
| --- | --- |
| [VM36 local Gap](reports/results/vm36_supervisor_reproduction) | Of 21 purposefully selected conditions, 4 were covered and 17 had a Gap across the 4 measured detector settings. |
| [VM45 Motion × Haze](reports/results/vm45_motion_haze_gap_learning) | Overall AP50:95 interaction contrasts of +6.881 and +7.298 pp for YOLOv8s and YOLO26s. |
| [VM47 Gaussian × JPEG](reports/results/vm47_gaussian_jpeg_factorial) | Overall AP50:95 interaction contrasts of +2.787 and +3.901 pp for the same checkpoints. |
| [VM48 Motion-targeted pilot](reports/results/vm48_formal_rq4/motion_targeted_seed45_execution_20261118T210143_5ada2476) | Motion overall AP50:95 +4.250 pp. Motion Tiny AP rose from 0.109% to 0.581% (+0.471 pp, 95% paired-image interval +0.409 to +0.947 pp). **Zero confirmed Gap closures.** Normal AP −1.046 pp and Haze AP −1.684 pp. [10-class results](reports/results/vm48_formal_rq4/motion_targeted_seed45_execution_20261118T210143_5ada2476/per_class_all_conditions.csv). |

## Experiments

- [VM36](experiments/vm36_supervisor_reproduction) · Dual feature models and development measurements
- [VM39](experiments/vm39_multi_seed_confirmation) · Gaussian augmentation across three seeds
- [VM43](experiments/vm43_external_pilot) · Limited external exploration
- [VM44](experiments/vm44_tiny_error_variability_decision) · Tiny-object diagnostics and measured selection
- [VM45](experiments/vm45_motion_haze_gap_learning) · Motion × Haze and rule safety
- [VM47](experiments/vm47_gaussian_jpeg_factorial) · Gaussian × JPEG
- [VM48](experiments/vm48_formal_rq4) · Gaussian transfer and Motion-targeted training

Earlier workflows: [VM1 validation](experiments/vm1_flamapy_yolo26s_validation) and [VM2 analysis](experiments/vm2_degradation_gap_analysis). Progress: [weekly notes](reports/notes).

## Reproduction

From the repository root, use Python 3.10 or later to verify all eight public snapshots and replay the summary calculations:

```bash
python -B code/tools/replay_public_results.py
```

Each snapshot under [reports/results](reports/results) has a manifest with RunIDs and SHA256 digests. Raw AP regeneration requires access to the original archived predictions and evaluation inputs. Shared code and dependencies are in [code](code). See the [dashboard setup](code/tools/README.md) or the [online dashboard](https://uavgap.streamlit.app/).

## Limitations

The main evidence uses purposeful conditions on the VisDrone development split. VM43 is limited external exploration with unknown historical independence. The Motion pilot uses one training seed. Its intervals reflect image variability. The full-space empirical Gap and the formal 1280-pixel, 100-epoch training study remain open. On new development conditions, the VM45 tree passed all four actual failures, so unknown conditions still trigger abstention.
