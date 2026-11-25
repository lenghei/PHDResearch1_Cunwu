# YOLO26-first RQ4 training

YOLO26s is primary; YOLOv8s is the comparison. The original VM36 feature models and 84 accepted evaluation pairs remain read-only.

This entry currently supports engineering admission and complete first-epoch cost studies. A cost study retains the intended 100-epoch schedule but stops after epoch 1. It is not a completed 100-epoch arm or an augmentation comparison.

## Local commands

Run from the Git root (`main/`) using the isolated local environment:

```powershell
& ..\.venv_rq4\Scripts\python.exe -B experiments/experiments_test/vm37_yolo26_rq4/cpu_tests.py
& ..\.venv_rq4\Scripts\python.exe -B experiments/experiments_test/vm37_yolo26_rq4/training.py --mode canary --architecture YOLO26s
& ..\.venv_rq4\Scripts\python.exe -B experiments/experiments_test/vm37_yolo26_rq4/training.py --mode full_epoch --architecture YOLO26s --arm CONTROL
```

GPU commands consume the existing finite allocation. They reject an active worker and an exhausted budget. Do not use them to repeat an accepted measurement without a new scientific reason.

`research_inference.py --recover <RunID>` evaluates a saved healthy checkpoint without repeating training. Source checksums must match the sealed training evidence.

## Files

- `training.py`: native10 FP32 training, real update counts and bounded worker.
- `research_inference.py`: dense one-to-many predictions, original NMS and ignore-union COCO evaluation.
- `study.py`: local paths, RunStore and the single 48-hour ledger.
- `TARGETED_POLICY.json`: source-backed five-family candidate, not an effectiveness claim.
- `TRAINING_MATRIX.csv`: the 12 planned 100-epoch arms and their actual status.
- `RESEARCH_PROTOCOL.md`: training and evaluation definitions.

Reusable dataset, effect, health and predictor adapters are in `code/main/`. Original data, checkpoints and detailed artifacts stay outside Git in the local project. The new environment shares unchanged Torch dependencies from the original environment; keep both environments available.

Internal Ultralytics validation and research COCO metrics are separate. All validation results are development evidence. Test-dev remains blocked. No exact training-resume claim is made.

## Measured status

Two complete first-epoch CONTROL cost studies and two engineering Canaries were run. The optimized full epoch took 24.600 minutes for 6471 train / 548 development images, 3236 forward/backward batches and 784 optimizer/EMA updates. Saved model, EMA, optimizer and scheduler states match the original implementation exactly.

The 100-epoch CONTROL forecast alone is 51.250 hours with the frozen 25% planning margin; the minimum single-resolution research-evaluation reserve is 8.705 hours. This is a forecast, not measured 100-epoch training. The complete pair failed resource admission. Zero formal 100-epoch arms and zero matched augmentation pairs are complete.

See results/engineering_summary.csv, results/research_metrics.csv and results/results_provenance.csv. All reported AP values are engineering/development evidence.

Release checksums use canonical Git blobs (LF text). Exact runtime-source bytes remain in each local Run's source snapshots.
