# VM48: Targeted training and transfer

The Gaussian transfer summary covers 24 development cells from seeds 42–44. The Motion pilot compares two matched YOLO26s arms at seed 45, trained at 960 for 10 epochs and evaluated at 1280. Motion AP50:95 gained 4.250 percentage points and Tiny AP gained 0.471 points. The confirmed Gap count stayed at three. Normal went from PASS to UNCERTAIN. The separate formal 1280-pixel, 100-epoch study remains pending.

[Published result summaries](../../reports/results/vm48_formal_rq4)

The [10-class table](../../reports/results/vm48_formal_rq4/motion_targeted_seed45_execution_20261118T210143_5ada2476/per_class_all_conditions.csv) gives AP, AR, GT counts and pointwise paired-image intervals for all four development conditions. The [heatmap](../../reports/results/vm48_formal_rq4/motion_targeted_seed45_execution_20261118T210143_5ada2476/Figure_C_class_degradation_heatmap.png) shows AP50:95 changes. All ten classes improved under Motion, with the largest gains for car, van and bus. The intervals use 200 shared image resamples from a single training seed.

From the repository root, verify the snapshot hashes and replay the available summary calculations with Python 3.10 or later:

```bash
python -B code/tools/replay_public_results.py vm48_formal_rq4
```

The source RunIDs and SHA256 digests are in each manifest. Full regeneration requires the archived run inputs.
