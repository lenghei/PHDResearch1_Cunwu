# Week 1 · Baseline

Verify the CV261102 YOLOv8s, Train640/Infer1280, 100-epoch VisDrone val baseline. Reuse its raw prediction/GT evidence and compare the historical source hashes.

From the Git root, set `$plan` to the local original research plan PDF.

```powershell
& ../.venv/Scripts/python.exe -B code/tools/uav_week01_06.py --week 1 --plan $plan
```

Use `--baseline-source-zip` with the archived CV261102 naming patch to recover the exact historical evaluator.
[Verified summary](../../reports/results/uav_01_baseline/uav_01_baseline_20261124T154238_4a518050/snapshot.json).
