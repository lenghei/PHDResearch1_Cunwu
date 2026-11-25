# Week 1 · YOLOv8s baseline

Replay the saved predictions and GT for 548 VisDrone validation images. Run from the outer `uavgap` directory:

```powershell
& ./.venv/Scripts/python.exe -B main/experiments/uav_01_baseline/run.py
```

Use `--archive` for a relocated run. The `evaluate` command runs the same checkpoint on GPU and compares its scores with the saved CSV. The archived 100-epoch settings are available through `preflight` and `train`.

[Metrics and timing](../../reports/results/uav_01_baseline/) · [Weights and selection](../../reports/results/uav_01_baseline/models/).
