# Week 5 · Evaluation

Run from the outer `uavgap` directory. Set `$pair_folder` to the archived pair directory with predictions and GT.

```powershell
& ./.venv/Scripts/python.exe -B main/experiments/uav_05_evaluation/run.py --replay $pair_folder
```

The CPU replay checks the saved COCO metrics. [Exploratory measurements and formal status](../../reports/results/uav_05_evaluation/).
