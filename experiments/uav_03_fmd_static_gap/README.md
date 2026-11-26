# UAV-GAP · Week 3 · FM_D and Static Gap

Week 3 models YOLO detector settings, profiles their resource cost and estimates static augmentation coverage.

## Week 3 deliverables

- [Measured FM_D](fm_d_measured.uvl) binds 33 records to checkpoint SHA256 and inference settings. Each profile uses all 548 VisDrone validation images. Fields include parameters, GFLOPs, mean and P95 latency, and peak GPU memory. FlamaPy reads the attributes and verifies that the Boolean model is unchanged.
- [Static Gap rules](../../reports/results/uav_03_fmd_static_gap/static_gap_rules.csv) give the FM_P parameter values, source and rationale for all 25 leaves. Exact identity, brightness gain proxies and the broader plan proxy cover 1, 3 and 18 of 46,368 configurations. Historical augmentation activation is recorded separately.

## What is included

- `fm_d.uvl`: original FM_D with 34 leaf features and four cross-tree constraints. Its SHA256 is recorded in the results manifest.
- `export_configuration_validation.py`: standalone exhaustive enumerator for every valid configuration of the original FM_D.
- `validate_configuration_validation.py`: reads the exported CSV back and checks row count, 34-feature vectors, unique IDs, all four constraints and the 844 training-variant mapping.
- `run.py` checks FM_D with FlamaPy SAT, reproduces the count by exact factorization and verifies the static gap. `sample_fmd.py` checks eight seeded, complete assignments with PySAT.
- `measure_gpu.py` profiles detector settings. `summarize_profile.py` checks and exports full-validation timings. `reuse_checkpoints.py` verifies the local checkpoint pool and retrieves historical metrics.
- `check_sahi_tta.py`, `check_sahi_uvl.py`, `fm_d_sahi_tta.uvl`: separately scoped SAHI/TTA compatibility experiment. This is not merged into the original 635,904-configuration FM_D.
- `coverage_rules.json`, `source_audit.py`, `verify_static_gap.py`: provisional default-augmentation coverage policy and reproducible 1 / 3 / 18 sensitivity counts.
- `reference/`: read-only FM_P model and UVL snapshots used only to reproduce the Week 3 static-gap count. Do not copy these over `uav_02_fmp`.

## Results in `main/reports/results/uav_03_fmd_static_gap/`

- `configuration_validation.csv`: 635,904 valid FM_D configurations, one row per configuration, 67 columns including all 34 leaf-feature columns.
- `configuration_validation_summary.json`: export counts and SHA.
- `configuration_validation_check.json`: full CSV read-back validation.
- `measured_detector_profiles.csv`: the five pilot resource profiles. The checkpoint evidence supports the measured inference settings, with some historical training augmentation leaves unresolved.
- `static_gap_leaf_rules.csv` and `static_gap_policy_summary.csv`: per-leaf provisional mapping and strict / intermediate / broad sensitivity policies.
- `static_gap_recomputation.json`: recomputed 1 / 3 / 18 counts.
- `third_week_traceability.csv`: Week 3 source requirement-to-evidence matrix.
- `week3_experiment_report_CN.md`: Chinese experiment explanation and limitations.

## Reproduce on CPU

Run from the outer `uavgap` directory. The full matrix is optional and takes about 390 MiB.

```powershell
$w3 = 'main/experiments/uav_03_fmd_static_gap'
& ./.venv/Scripts/python.exe -B "$w3/run.py"
& ./.venv_flamapy/Scripts/python.exe -B "$w3/sample_fmd.py" --seed 20261125

$matrix = "datasets/local_runs/uav_03_fmd_static_gap/fmd_matrix_$(Get-Date -Format yyyyMMdd_HHmmss)"
New-Item -ItemType Directory $matrix | Out-Null
& ./.venv/Scripts/python.exe -B "$w3/export_configuration_validation.py" --output "$matrix/configuration_validation.csv"
& ./.venv/Scripts/python.exe -B "$w3/validate_configuration_validation.py" --csv "$matrix/configuration_validation.csv"
```

## Independently verify with FlamaPy (CPU only)

```powershell
& .\.venv_flamapy\Scripts\python.exe -B main/experiments/uav_03_fmd_static_gap/flamapy_count.py --model main/experiments/uav_03_fmd_static_gap/fm_d.uvl
```

The accepted `run.py` replay checks the original UVL with FlamaPy SAT and counts its valid configurations by exact factorization. The matrix export independently enumerates every row.

## Important interpretation

- 635,904 is the legal configuration count of the original FM_D. The separate SAHI+TTA extension has 635,905 and must not be mixed into the original count.
- 844 is the count of distinct training-time feature combinations; the remaining differences are inference-time choices.
- The 32-image pilot and the full 548-image profile use separate inputs and timing tables. Both measure resource use.
- The original plan's 18/46,368 static coverage is a provisional proxy assumption. This package also reports strict identity 1 and the intermediate brightness-only sensitivity 3. None of these counts is an empirical detector failure rate.
- Actual historical checkpoint coverage remains `UNKNOWN` until the saved training arguments, installed augmentation source, and Albumentations activation are fully bound to the checkpoint.
- The commands above run on CPU.

## Detector profiling

Run from the outer `uavgap` directory with the local VisDrone val ZIP and the Week 1 checkpoint.

```powershell
& ./.venv/Scripts/python.exe -B main/experiments/uav_03_fmd_static_gap/reuse_checkpoints.py
& ./.venv/Scripts/python.exe -B main/experiments/uav_03_fmd_static_gap/measure_gpu.py --full-validation
& ./.venv/Scripts/python.exe -B main/experiments/uav_03_fmd_static_gap/summarize_profile.py $run
```

The first command checks six local checkpoints and profiles operation counts at 640 and 1280 on CPU. The second profiles the five frozen YOLOv8s settings on all 548 Normal validation images, three passes per setting. Set `$run` to the printed GPU run path for CPU summary export.

[Week 3 results](../../reports/results/uav_03_fmd_static_gap/).

## Checkpoint validation

The local catalog under `code/models/` links six Ultralytics and two LEAF checkpoints. Both runners use the same 548 Normal images, FP32, batch 1 and native-ten COCO evaluation at 640 and 1280.

```powershell
& ./.venv/Scripts/python.exe -B main/experiments/uav_03_fmd_static_gap/validate_checkpoints.py --model ck01_yolo26s_ft1280
& ./.venv/Scripts/python.exe -B main/experiments/uav_03_fmd_static_gap/validate_leaf.py --check
& ./.venv/Scripts/python.exe -B main/experiments/uav_03_fmd_static_gap/validate_leaf.py
```

Use `--model <checkpoint>` or `--size 640` to select a subset. Use `--replay <run/model/infer640>` to recalculate a saved cell on CPU.

Run the Ultralytics checkpoints separately for bounded timing. The LEAF check profiles the archived checkpoint and tests CPU inference. The LEAF GPU entry uses the archived IDetect backend and shared evaluation. LEAF-S is an external control outside the original FM_D rule.

`sample_fmd.py --seed 20261125` uses `.venv_flamapy`. `export_week3_models.py --help` lists the accepted source runs needed to regenerate the comparison. Use `--cache` to select an equivalent verified Normal cache.

## Zero-shot and inference-option profiles

```powershell
$w3 = 'main/experiments/uav_03_fmd_static_gap'
& ./.venv/Scripts/python.exe -B "$w3/validate_zeroshot.py"
& ./.venv/Scripts/python.exe -B "$w3/profile_inference_options.py"
& ./.venv/Scripts/python.exe -B "$w3/profile_inference_options.py" --case v26_e2e640
& ./.venv_flamapy/Scripts/python.exe -B "$w3/verify_static_source.py"
```

The zero-shot run evaluates four local COCO weights on VisDrone val at Infer640 with a six-class mapping. The [inference profiles](../../reports/results/uav_03_fmd_static_gap/reproduce_20261125T142040_7a322c8d/inference_options.csv) cover ten settings on the 548-image VisDrone validation split: Full640, Full1536, SAHI, TTA, four preprocessing options and the two YOLO26s heads. They report latency, GFLOPs and peak GPU memory. These runs measure resource use. The archive contains SAHI 0.11.36.

Recheck the saved profiles and generate the matched extension plan on CPU:

```powershell
& ./.venv/Scripts/python.exe -B "$w3/replay_inference_options.py"
```

The extension adds 11 settings across YOLO11s T1280, YOLOv8n T640 and YOLO26n T640 Full20. It keeps the same 548 images, batch 1 and three warmups, and uses the shared GPU lock. Run when the device is available:

```powershell
foreach ($group in 'yolo11s', 'yolov8n', 'yolo26n-nms', 'yolo26n-e2e') {
    & ./.venv/Scripts/python.exe -B "$w3/profile_inference_options.py" --group $group
    if ($LASTEXITCODE -ne 0) { break }
}
```

The original FM_D keeps SAHI and TTA mutually exclusive. The NMS-free path uses a separate process. INT8 needs a calibrated engine for a selected VisDrone checkpoint, with training-only calibration images and the same validation timing boundary. The current profile uses PyTorch.

Use `export_week3_followup.py --help` to rebuild the result tables and `plan_inference_pairwise.py` to generate a logical candidate schedule from the roster.

## Rebuild the two deliverables

```powershell
$w3 = 'main/experiments/uav_03_fmd_static_gap'
$out = "datasets/local_runs/uav_03_fmd_static_gap/measured_$(Get-Date -Format yyyyMMdd_HHmmss)"
New-Item -ItemType Directory $out | Out-Null
& ./.venv_flamapy/Scripts/python.exe -B "$w3/export_measured_fmd.py" --uvl "$out/fm_d_measured.uvl" --rules "$out/static_gap_rules.csv"
```

The export verifies the SHA256 of the source tables. Use `--source` to supply the pinned [Ultralytics v8.4.33 augmentation source](https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.33/ultralytics/data/augment.py) on another machine.
