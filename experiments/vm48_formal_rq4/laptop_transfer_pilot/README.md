# Saved-model cross-degradation pilot

Evaluate matched YOLO26s CONTROL/Gaussian-guided epoch10 EMA checkpoints from VM37–VM39. No training runs here. This is a separate960-resolution training pilot; the formal VM48 package stays unchanged.

- Seeds:42,43,44, in that order.
- Conditions: VM45 Normal, Motion21, Haze1.5, Motion+Haze.
- Data:548 development images, shared verified VM45 PNGs.
- Inference:1280, FP32, batch1, native10, accepted dense/NMS evaluator.
- Each seed:8 cells and200 paired pooled-image AP bootstrap draws.
- Admission: original allocation/expiry, complete seed-group cost,1200s shutdown reserve plus600s final analysis allowance, disk/AC/thermal checks. No retry.

The entrypoint is bound to the explicit2026-11-18 author decision and refuses a second attempt:

```powershell
.\.venv_rq4\Scripts\python.exe -B -X utf8 main/experiments/vm48_formal_rq4/laptop_transfer_pilot/run.py --execute
```

Outputs stay in datasets/local_runs/vm48_formal_rq4/laptop_transfer_pilot_execution_*/. Raw predictions, metrics, bootstrap draws, checkpoint/pixel hashes, telemetry, group costs and reports are preserved. Only complete supervised seed groups enter paired summaries. Inspect final_acceptance.json and resource_usage.json; a process exit code alone is insufficient.

Train960 is outside the original FM_D training-resolution domain. Record shared feature IDs without assigning Train1280. All results are DEVELOPMENT_ONLY and POST_HOC_POLICY_SELECTION. Unknown-P controllers abstain. This pilot cannot establish formal100-epoch RQ4, independent image generalization or training-seed significance.

The first execution used three warmups per model instead of the required three per cell. AP evidence remains usable; VM41 latency admission and PASS release are withdrawn. `review.py` preserves preliminary summaries and corrects those claims on CPU without rerunning inference. A parent module-name collision also affected final sealing, after all measured cells and CPU metrics were saved. Keep the executed sources for provenance; a future authorized timing run needs a corrected version.
