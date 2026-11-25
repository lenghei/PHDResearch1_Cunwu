# Week 3 · Detector settings and static coverage

Run from the outer `uavgap` directory:

```powershell
& ./.venv/Scripts/python.exe -B main/experiments/uav_03_fmd_static_gap/run.py
& ./.venv/Scripts/python.exe -B main/experiments/uav_03_fmd_static_gap/measure_gpu.py
& ./.venv_flamapy/Scripts/python.exe -B main/experiments/uav_03_fmd_static_gap/check_sahi_uvl.py
```

The first command replays the original FM_D and static coverage on CPU. The second profiles the five settings on a shared development sample with the existing thermal supervisor. The third verifies the one-setting SAHI/TTA variant. `check_sahi_tta.py` tests the library path on CPU with SAHI 0.11.36 in a separate dependency folder.

[Measured results](../../reports/results/uav_03_fmd_static_gap/).
