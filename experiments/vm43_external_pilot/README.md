# VM43: Limited external exploration

Seed 42 CONTROL and Gaussian-guided YOLO26s were evaluated on 1,610 VisDrone test-dev images per cell, under Normal and added Gaussian noise. The four evaluations use inference size 1280. Historical split independence remains unknown. Use these results as exploratory evidence.

[Published result summaries](../../reports/results/vm43_external_pilot)

From the repository root, verify the snapshot hashes and replay the available summary calculations with Python 3.10 or later:

```bash
python -B code/tools/replay_public_results.py vm43_external_pilot
```

The source RunIDs and SHA256 digests are in each manifest. Full regeneration requires the archived run inputs.
