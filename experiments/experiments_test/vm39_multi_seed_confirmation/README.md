# VM39: Gaussian augmentation across seeds

Matched YOLO26s CONTROL and Gaussian-guided pilots used seeds 42–44, 960-pixel training and fixed epoch-10 EMA. At inference size 1280, Gaussian sigma 15 AP50:95 improved by 5.10–5.53 percentage points across the three seeds on the development split. The summary includes Normal, Gaussian and both JPEG settings.

[Published result summaries](../../../reports/results/vm39_multi_seed_confirmation)

From the repository root, verify the snapshot hashes and replay the available summary calculations with Python 3.10 or later:

```bash
python -B code/tools/replay_public_results.py vm39_multi_seed_confirmation
```

The source RunIDs and SHA256 digests are in each manifest. Full regeneration requires the archived run inputs.
