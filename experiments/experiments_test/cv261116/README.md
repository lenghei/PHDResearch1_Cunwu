# CV261116 — FM_D and static degradation-coverage gap

**Supervisor milestone:** Week 3, 16 November 2026. This experiment follows the
FM_P generator implemented in `experiments/cv261109` and does not change any
existing VM/CV run, checkpoint, dataset, or historical result.

## Scope and scientific interpretation

The supervisor's 22 October research plan defines a detector feature model with
34 leaf features and four constraints. The draft predicts 635,904 detector
configurations and 844 distinct training-time feature combinations. The
independent counter in `model.py` reproduces both figures. `--check` **also
requires independent FlamaPy counting and dead-feature checking** on the local Windows installation;
these precomputed quantities never substitute for a successful solver run.

The static coverage table implements the supervisor's *proxy* assumptions:
18 of 46,368 FM_P configurations can be approximately produced using a default
training augmentation family. This is not pixel-exact identity: HSV has no
additive offset; generic small blur is only a surrogate for defocus; geometric
scaling is only a surrogate for camera-altitude changes. The stricter
identity-only sensitivity comparison covers 1 of 46,368.

The static gap is **not** an empirical AP/latency gap. Coverage inferred from a
research-plan example is **provisional** until the actual training arguments,
installed source code, and exact checkpoint/augmentation provenance are
reviewed. The training-augmentation discussion in the plan refers to
Ultralytics **8.4.33**; this project previously used **8.4.61**. Do not assume
these implementations are equivalent. The `source_audit.py` tool records the
actual installed augmentation implementation's hash, class symbols and
relevant training arguments; it does not certify their scientific equivalence.

The supervisor's SAHI/TTA incompatibility is retained in the draft UVL, but its
API justification remains to be verified. Inference-time choices that do not
have an implemented runner remain unmeasured.

## Files and ownership

| File | Role |
| --- | --- |
| `fm_d.uvl` | 34 leaves, four constraints, full FM_D variability |
| `model.py` | Independent combinatorial cross-check and FM_P proxy rules |
| `flamapy_count.py` | FlamaPy SAT admissibility and exact count for guarded UVL v1 (no full #SAT enumeration) |
| `coverage_rules.json` | Nine explicit coverage families, rationales and evidence gates |
| `source_audit.py` | Installed Ultralytics source and saved `args.yaml` audit |
| `gpu_profile.py` | Real RTX GPU inference profiling, no training |
| `run.py` | Live demonstration, local output and sanitized publication |
| `tests/test_cv261116.py` | Fast CPU regression tests |

All full execution artifacts are written to:

`uavgap/datasets/local_runs/cv261116/<RunID>/`

and not into the public Git root. This package contains no data, weights,
virtual environments, machine-specific configuration, or caches.

## Installation and CPU checks

Unzip into the **outer** `E:\ForStudy\code\uavgap\` project, not into `main`.
It adds only `main/experiments/experiments_test/cv261116/` and one private Chinese guide under
`datasets/reportcn/`. Keep CV261109 and its verified `fm_p.uvl` unchanged.

From PowerShell:

```powershell
cd E:\ForStudy\code\uavgap
& .\.venv\Scripts\python.exe -B -m unittest discover -s main/experiments/experiments_test/cv261116/tests -p 'test_*.py' -v
& .\.venv\Scripts\python.exe -B main/experiments/experiments_test/cv261116/run.py --check
```

The second command requires the existing `.venv_flamapy/Scripts/python.exe`.
It uses FlamaPy UVL parsing and fast SAT decision operations, **not**
`PySATConfigurationsNumber`, which timed out on the large model. The count is
proven with an explicit symbolic factorization (260 regular training variants
per architecture, 64 LEAF_YOLO training variants, 1,104 postprocessing-weighted
branches, 576 inference branches per postprocessing choice: 635,904 total).
Run output contains the precise RunID, source hashes, model counts, training
source audit, two coverage policies, detailed tables, and a presentation-ready
HTML report. It does **not** invoke a detector or consume GPU training time.

Model success and static proxy verification do not imply the entire milestone
is finished; `result.json` deliberately uses `status=PARTIAL` while GPU
measurements and training-source review are outstanding.

## Real GPU measurements, no retraining

Choose a real checkpoint with known training provenance and the original
annotated VisDrone2019-DET-val ZIP. For example, substitute the actual path
to the fine-tuned YOLOv8s checkpoint from CV261102:

```powershell
& .\.venv\Scripts\python.exe -B main/experiments/experiments_test/cv261116/run.py --profile `
  --checkpoint 'E:\actual\path\to\best.pt' `
  --val-zip 'E:\actual\path\to\VisDrone2019-DET-val.zip' `
  --train-args 'E:\actual\path\to\args.yaml' `
  --imgsz 1280 --precision FP32 --sample-count 16 --repeat 2 --warmup 4
```

The profiler uses original validation images, evenly sampled in fixed filename
order, batch size 1. Latency is wall-clock per image, after CUDA synchronization;
CUDA-event time is recorded separately. GPU peak memory is PyTorch allocated
memory, not process-wide or device-wide consumption. Parameter count is read
from the loaded checkpoint. GFLOPs are an optional Ultralytics model estimate,
not a measured GPU instruction count. Missing GFLOPs remain null.

Supported measured pipeline: full-frame, NMS, no TTA, no preprocessing, FP32
or FP16. The configuration is tied to an explicit architecture, scale,
weights type and train-resolution declaration, checked against checkpoint
class count and (if supplied) the saved training resolution. Actual augmentation
coverage still requires a separate source audit. No false measurement is
generated for INT8, SAHI, TTA, or other unimplemented branches. The profiler does **not** calculate AP, run training,
or test the empirical gap. The sample of 16 images is a latency pilot, not the
complete 548-image evaluation.

## Public dashboard summary

After reviewing the completed run, publish only the sanitized snapshot:

```powershell
$run = 'E:\ForStudy\code\uavgap\datasets\local_runs\cv261116\<RunID>'
& .\.venv\Scripts\python.exe -B main/experiments/experiments_test/cv261116/run.py --publish-run $run
```

This creates `main/reports/results/cv261116/<RunID>/{manifest,snapshot}.json`,
compatible with automatic study discovery. The current generic dashboard can
list the run; the rich FM_D coverage report remains the local
`static_gap_report.html`. Do not claim the generic dashboard has a dedicated
FM_D visualization unless one is implemented and tested.

## Scientific acceptance criteria

1. FlamaPy independently confirms satisfiability, no dead features, 34 leaf
   features and four constraints. Two exact algebraic implementations agree on
   635,904 configurations and 844 training variants for the guarded UVL version.
   Independent FlamaPy BDD/#SAT counting has NOT been completed.
2. The unchanged CV261109 FM_P has 46,368 configurations. The nine proxy rule
   families are explicit and derive 18 covered configurations. Strict identity
   sensitivity derives one covered configuration.
3. Run metadata and source hashes trace all published numbers; no raw
   predictions, checkpoints, source datasets, local paths or Python caches are
   published.
4. A GPU measurement is marked `PASS` only if real CUDA inference completed
   for the documented checkpoint and original inputs.
5. Actual training coverage remains unverified until source and original saved
   training arguments are inspected and accepted. Static gap must be discussed
   as model/rule-dependent; no AP is claimed.

FlamaPy SAT references:
- https://docs.flamapy.org/framework/operations/satisfiable
- https://docs.flamapy.org/framework/operations/deaf_features
- https://docs.flamapy.org/framework/operations/number_of_configurations

Prior issue: `PySATConfigurationsNumber` exceeded 600 seconds (local Windows
run). This patch intentionally avoids high-cost #SAT enumeration, retains the
failed run record and does not modify existing FM_P/FM_D results. A separate
BDD solver can be installed and tested later for fully independent #SAT
verification, but the CPU acceptance here is explicitly SAT + factorization.
