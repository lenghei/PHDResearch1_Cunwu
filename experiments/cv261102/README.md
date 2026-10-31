# CV261102: YOLOv8s training and reproducible VisDrone validation

## Purpose and scientific scope

This is the **2 November 2026 (week 1)** reproduction task from the supervisor's
research plan, *The Gap Between UAV Image Degradations and YOLO Configurations*
(22 October 2026). It starts from COCO-pretrained `yolov8s.pt`, performs a **new**
VisDrone2019-DET training run, evaluates all **548 annotated validation images**,
and produces CSV reports from the actual checkpoint and saved predictions.

The protocol is deliberately explicit:

| Item | Setting |
| --- | --- |
| Initialization | Local COCO-pretrained YOLOv8s weights, 80 categories |
| Training images | VisDrone2019-DET-train, 6,471 images |
| Training labels | Ten native VisDrone categories, in the documented order |
| Training | 100 epochs, 640 px, batch 4, seed 42, AMP enabled |
| Validation during training | VisDrone2019-DET-val, for checkpoint selection |
| Final evaluation | All 548 images from the same annotated val split |
| Final evaluation input | 1280 px, batch 2; shared project's COCO protocol |
| Metrics | AP50, AP50:95, AR50:95, per area and per class |
| Provenance | Input hashes, code hashes, checkpoint hash, timing, raw predictions, GT |

**Scientific limitation:** The validation split is also used for checkpoint
selection. These are **development-validation** results, NOT independent test-dev
results. Do not claim a held-out evaluation or new comparative improvement from
this run. A result is only confirmed after the real GPU run and a successful
prediction-based COCO replay. The previous ck05 CSV and previous RunIDs are
not used as training inputs or substituted for new evidence. A numerical match
to a historical CSV is not established by this package.

## Integration requirements

Extract the ZIP into the **outer** `uavgap/` project. It installs only
`main/experiments/cv261102/` and does **not** overwrite the earlier VM
experiments, checkpoints, or recorded runs. The complete shared implementation
is **not embedded** in this small overlay package. It must already exist at:

```text
uavgap/
  .venv/                           Existing YOLO environment, kept local
  main/                            GitHub sync root
    code/main/                     Existing labels and evaluation source (required)
    experiments/cv261102/         This reproducible experiment
  datasets/                        VisDrone archives and local experiment runs
  models/                          Local pretrained weights, if stored here
```

The shared evaluation modules must expose `main.labels.VIS_NAMES` and
`main.evaluation.evaluate` / `regenerate`, including saved COCO prediction and
GT output. A **preflight check verifies the environment and shared modules**;
this package cannot independently establish their scientific equivalence
without those original files. Do not replace them with fabricated stubs.

The repository/public code uses English file names, code comments, documentation,
and console messages. Never commit weights, data archives, local result folders,
private paths, or Python virtual environments. Use local Git commits for work
performed in the GitHub sync directory; the scripts never commit or push on
your behalf.

## Configuration

Copy `config.example.yaml` to `config.local.yaml` in this directory. Set
`train_zip`, `val_zip`, and `weights` to **actual local files**. The example
contains intentionally invalid `CHANGEME` placeholders so that missing setup
cannot silently start an experiment. Use forward slashes for Windows paths.

There are two mutually exclusive training-data options:

1. `train_zip`: annotated, unmodified original VisDrone train ZIP. The program
   converts 6,471 train and 548 val images to native-ten YOLO format under the
   private run folder without modifying source archives. The preflight scans
   every original annotation record; non-finite fields, negative dimensions,
   or illegal categories abort with source filename and line number. Original
   zero-area records are not YOLO training targets; they are skipped and counted
   in `data_preparation.json`, with example filenames and line numbers.
   VisDrone annotation records containing eight numeric fields followed by one
   trailing comma are accepted; genuine extra columns remain invalid.
2. `data_yaml`: an existing YOLO native-ten training YAML. Set `train_zip: null`;
   preflight validates class order, 6,471 train images, and that its 548 val
   images have exactly the original source ZIP bytes and image IDs.

The `val_zip` must be the original annotated VisDrone2019-DET-val archive in
both modes. The weights must be named `yolov8s.pt` and contain the 80-class
COCO initialization; renamed fine-tuned checkpoints are not valid starts.

Reduce `batch` to 2 only if the local GPU cannot fit batch 4. A changed
hyperparameter is recorded in the new run manifest. Defaults are *choices for
this experiment*, not claims about recommended or optimal values.

## Run on Windows from the outer project root

```powershell
& .\.venv\Scripts\python.exe -m unittest discover -s main/experiments/cv261102/tests -p 'test_*.py' -v
& .\.venv\Scripts\python.exe -B main/experiments/cv261102/run.py --preflight
& .\.venv\Scripts\python.exe -B main/experiments/cv261102/run.py
```

Alternatively, execute `preflight.ps1` or `run_full.ps1` from this experiment
directory. A clean preflight does **not** train; the full run performs actual
CUDA training and 548-image evaluation. The program does not download weights,
install packages, use historical results, or interact with GitHub.

## Output provenance

A new, never-overwritten run is stored at:

`uavgap/datasets/local_runs/cv261102/cv261102_full_YYYYMMDDTHHMMSS_<id>/`

```text
manifest.json                    Input, source and checkpoint hashes, status
data_preparation.json            Target conversion counts and excluded zero-area samples
training/epoch_times.csv         Real per-epoch callback timings
training/yolov8s/               Ultralytics best.pt, last.pt, args.yaml, results.csv
training/checkpoint.json         Checkpoint and original training log SHA256 values
validation/                     Source-linked GT, raw and COCO predictions, metrics
csv/metrics_overall.csv          Aggregate metrics for this fresh run
csv/metrics_by_area.csv          Metrics across original-image size bins
csv/metrics_by_class.csv         Metrics by native VisDrone category
csv/training_epoch_times.csv     Export of the recorded epoch timings
csv/evaluation_timing.csv        Measured evaluation and training wall clocks
coco_regeneration.json          COCO recomputation from saved GT and predictions
```

`manifest.json` reports `PASS` only when the training, evaluation, saved
prediction replay, and CSV export have completed. Failures remain `PARTIAL`
and retain a traceback. Do not count those runs as completed experiments.
Run results stay outside `main/` to prevent accidental publishing of large or
sensitive artifacts.

## Audit a completed run

```powershell
& .\.venv\Scripts\python.exe -B main/experiments/cv261102/validate_run.py --run 'C:/path/to/uavgap/datasets/local_runs/cv261102/EXACT_RUN_ID'
```

The **default audit regenerates COCO metrics** from saved predictions/GT and
compares those values with JSON and all five CSVs, alongside SHA256 checks and
training-log consistency. It does not retrain or run the detector.

`--integrity-only` skips the COCO recomputation and returns
`INTEGRITY_ONLY`, **not** scientific `PASS`. This is useful for machines without
the project's evaluator dependencies.

## Supervisor milestones and remaining gates

- Week 1, 2 November: A real, live full training/validation evaluation must be
  run and demonstrated from the committed code. Epoch times and evaluation
  time come from the actual run. This package provides the entry point; it
  cannot assert that the local run has already happened.
- Week 2, 9 November: FM_P must be a complete constrained UVL degradation model,
  and a supervisor-selected random valid configuration must generate images.
- Week 3, 16 November: FM_D needs validated UVL constraints, *measured* detector
  attributes, and reasoned static gap rules. These later stages are not
  implemented or claimed complete in CV261102.
- Later phases: 500 sampled pairs, held-out rule learning, gap estimation,
  targeted augmentation, and UAVDT transfer require their own evidence.

The supervisor's replication rule asks for raw outputs and scripts sufficient
to reconstruct every reported figure. Local policy currently keeps large raw
outputs outside GitHub. Before declaring the **paper's** replication package
complete, agree on a shareable artifact archive or repository data policy that
actually makes the raw evidence accessible. Source code plus hashes alone is
not full public numerical reproducibility.

VisDrone data: https://github.com/VisDrone/VisDrone-Dataset
