# CV261116 — FM_D and static-gap evidence

 
**Task:** Formalize detector variability with measured attributes and validate augmentation-coverage evidence.

Run from project parent `uavgap/`:

```powershell
& .\.venv\Scripts\python.exe -B main/experiments/cv261116/run.py
```

The current script audits the existing FM_D UVL (9 configurations), records its SHA and configuration count, and writes a truthful `PARTIAL` report to `datasets/local_runs/cv261116/<RunID>/`.

