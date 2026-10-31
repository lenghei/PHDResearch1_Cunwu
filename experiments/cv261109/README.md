# CV261109 — FM_P configuration generation

  
**Task:** Select a FlamaPy-valid degradation configuration and generate reproducible UAV images.

Run from project parent `uavgap/`:

```powershell
& .\.venv\Scripts\python.exe -B main/experiments/cv261109/run.py --seed 1031
```

Default: generate **all 548** VisDrone validation images under one selected configuration, with matching source annotations and SHA-verified source mapping. Only three images are used for local preview; this does not run YOLO.

Outputs: `datasets/local_runs/cv261109/<RunID>/`.

