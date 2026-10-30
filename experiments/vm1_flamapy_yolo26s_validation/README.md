# VM1 — FlamaPy–YOLO26s full validation

Primary model: ck01 YOLO26s fine-tuned on VisDrone; native10, 1280, FP32.

The default `run.py` command runs the entire VisDrone2019-DET-val split
(548 images) under two conditions: CLEAN and one FlamaPy-valid degraded
configuration. Both use the same checkpoint and evaluation protocol.

| Command | Operation |
|---|---|
| `run.py` or `run.py --full-paired` | 548 CLEAN + 548 DEGRADED; actual GPU inference |
| `run.py --full-clean` | 548 CLEAN only; actual GPU inference |
| `run.py --check` | CPU admission, FlamaPy and historical metric regression |
| `run.py --regenerate <run-directory>` | Recompute saved metrics, without inference |
| `run.py --refresh-index` | Rebuild run browser only |
| `run.py --demo` | Legacy five-image diagnostic; not part of the full study |

Run from the PyCharm project root `uavgap/` with `.venv`:

```powershell
& .\.venv\Scripts\python.exe -B main/experiments/vm1_flamapy_yolo26s_validation/run.py --full-paired
```

Set private `config.local.yaml` from `config.example.yaml` before running.
FlamaPy uses the separate `.venv_flamapy` interpreter. The supplied UVLs
currently describe a 27-configuration degradation baseline and a
9-configuration detector baseline, not the supervisor's full FM_P/FM_D.

Both conditions have saved raw predictions, COCO GT, AP/AR, provenance,
timing, environment and per-image hashes. The full-paired run also saves
`paired_comparison.csv` and `paired_comparison.json`. Five selected degraded
images are saved **only for visual preview**; metrics use all 548 images.

Full runs are **development-validation**, not independent test-dev results.
The current UVL-to-numeric mapping is implemented in Python. Gap coverage,
uncertainty estimates and targeted retraining are separate unfinished work.
Existing historical five-image runs and the ck05 YOLOv8s baseline are unchanged.
