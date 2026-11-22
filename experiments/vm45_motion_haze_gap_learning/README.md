# VM45: Motion and haze interaction

The four conditions are Normal, motion kernel 21, dense haze (beta 1.5) and their combination. YOLOv8s and YOLO26s use the same 548 development images at inference size 1280. The summary includes absolute AP50:95, four-corner contrasts, 200-draw image bootstrap intervals and observed false-PASS counts.

[Published result summaries](../../reports/results/vm45_motion_haze_gap_learning)

From the repository root, verify the snapshot hashes and replay the available summary calculations with Python 3.10 or later:

```bash
python -B code/tools/replay_public_results.py vm45_motion_haze_gap_learning
```

The source RunIDs and SHA256 digests are in each manifest. Full regeneration requires the archived run inputs.
