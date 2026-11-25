# Week 2 · Image degradation

FM_P defines 25 degradation features and five constraints. `render_bindings.json` supplies additional rendering parameters. All 46,368 legal combinations passed the synthetic image test.

Run from the outer `uavgap` directory:

```powershell
& ./.venv/Scripts/python.exe -B main/experiments/uav_02_fmp/run.py --config-id p_7d560e75a98f5496 --image-index 0
```

Use `generate --config-id <id>` to render the 548-image set, or `audit-space` to replay the full configuration test. [Results and configuration CSV](../../reports/results/uav_02_fmp/).
