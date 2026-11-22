# VM48: Targeted training and transfer

The Gaussian transfer summary covers 24 development cells from seeds 42–44. The Motion pilot compares two matched YOLO26s arms at seed 45, trained at 960 for 10 epochs and evaluated at 1280. Motion AP50:95 gained 4.250 percentage points and Tiny AP gained 0.471 points. The confirmed Gap count stayed at three. Normal went from PASS to UNCERTAIN. The separate formal 1280-pixel, 100-epoch study remains pending.

[Published result summaries](../../reports/results/vm48_formal_rq4)

From the repository root, verify the snapshot hashes and replay the available summary calculations with Python 3.10 or later:

```bash
python -B code/tools/replay_public_results.py vm48_formal_rq4
```

The source RunIDs and SHA256 digests are in each manifest. Full regeneration requires the archived run inputs.
