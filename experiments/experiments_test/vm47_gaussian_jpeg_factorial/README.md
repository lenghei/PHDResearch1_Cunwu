# VM47: Gaussian noise and JPEG interaction

YOLOv8s and YOLO26s were evaluated on the same 548 development images under Normal, Gaussian sigma 15, added JPEG Q25 and their combination. The four-corner AP50:95 contrasts were positive for both checkpoints. The new combined condition exposed another false PASS.

[Published result summaries](../../../reports/results/vm47_gaussian_jpeg_factorial)

From the repository root, verify the snapshot hashes and replay the available summary calculations with Python 3.10 or later:

```bash
python -B code/tools/replay_public_results.py vm47_gaussian_jpeg_factorial
```

The source RunIDs and SHA256 digests are in each manifest. Full regeneration requires the archived run inputs.
