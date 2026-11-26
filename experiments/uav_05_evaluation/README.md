# Week 5 · Evaluation

Run from the outer `uavgap` directory. Set `$pair_folder` to the archived pair directory with predictions and GT.

```powershell
& ./.venv/Scripts/python.exe -B main/experiments/uav_05_evaluation/run.py --replay $pair_folder
```

The CPU replay checks the saved COCO metrics. [Exploratory measurements and formal status](../../reports/results/uav_05_evaluation/).

## Two-checkpoint baseline pilot

The pilot uses five purposeful degradation conditions and two fixed Train1280 checkpoints.
Both detectors use Infer1280, batch 1, FP32 and the same native-ten NMS and COCO settings.
Each condition shares its cached pixels and ground truth across both detectors.

```powershell
$pilot = 'main/experiments/uav_05_evaluation/pilot1280.py'
& ./.venv/Scripts/python.exe -B $pilot --plan
& ./.venv/Scripts/python.exe -B $pilot --check-checkpoints
```

The test-dev pilot requires its own scientific admission and GPU allocation.
After admission, use `--execute --science $science --gpu-authorization $gpu`.
Keep the emitted selection hash with the approval. The pilot uses 1,610 images per
cell and has zero credit toward the earlier frozen 500-pair batch. The pilot
measures the historical baselines. The matched retraining study is documented in [RQ4](../vm48_formal_rq4/README.md).
