# Week 6 · Empirical Gap

From the outer project root, set `$plan` to the original research plan PDF:

```powershell
& ./.venv/Scripts/python.exe -B main/code/tools/uav_week01_06.py --week 6 --plan $plan
```

The CPU audit reports the formal status and threshold sensitivity on the measured development subset.

```powershell
& ./.venv/Scripts/python.exe -B main/experiments/uav_06_gap_analysis/validate_static_gap.py
```

The second command checks 84 historical VM36 measurements, compares the static rules against confirmed performance on 21 development conditions, and verifies the earlier LEAF-N resolution comparison. [Results and scope](../../reports/results/uav_06_gap_analysis/).

After the two-checkpoint pilot finishes, pass its execution directory to the CPU summary:

```powershell
& ./.venv/Scripts/python.exe -B main/experiments/uav_06_gap_analysis/summarize_pilot1280.py $execution
```

The summary requires all ten sealed cells and matching image order, shared per-condition GT and pixel receipts, and common bootstrap draws. It reports overall and size-bin metrics, paired losses from Normal, and Gap status for the two evaluated detectors.
