# CV261207 dual-variability evaluation

`run.py` audits the original UVLs and freezes a uniform sample crossed with
five YOLOv8s inference configurations. The 500 planned pairs use one trained
checkpoint. Rolling-shutter configurations stay in FM_P but are excluded from
the executable cohort until GT and ignore-region geometry can be transformed
and validated.

From the outer project root in PowerShell:

```powershell
$py = '.\.venv\Scripts\python.exe'
$entry = 'main/experiments/cv261207/run.py'
& $py -B $entry preflight
& $py -B $entry admission --audit <audit-RunID>
& $py -B $entry publish-admission --run <admission-RunID>
& $py -B $entry smoke-grid --audit <audit-RunID> --budget-seconds 240
& $py -B $entry demo --audit <audit-RunID> --seed 20261104 --budget-seconds 120
& $py -B $entry regenerate --run <saved-pair-RunID>
& $py -B $entry figures --audit <audit-RunID> --run <saved-pair-RunID>
& $py -B $entry publish --audit <audit-RunID>
```

Smoke and demo use validation images for engineering checks. The formal branch
uses the original 1,610-image test-dev ZIP and requires a separate local
historical-use decision containing `status: AUTHOR_REVIEWED_ADMIT`, the audit
RunID, dataset gate SHA, test-dev ZIP SHA, selection SHA and protocol ID.
`checkpoint_reviews` is keyed by checkpoint SHA256. Each entry must contain
`training_membership`, `checkpoint_selection` and
`historical_evaluation_and_tuning` reviews with status `VERIFIED_INDEPENDENT`
and an `evidence` list of path and SHA256 records under the local datasets
directory. The program checks identity and file integrity. The reviewer
establishes scientific independence from the evidence.

The admission audit checks the three historical 1280 checkpoints separately
from CV261102. They are development candidates. The frozen 500 pairs still
bind CV261102; any multi-checkpoint design requires a new manifest.

`full` takes one frozen `--pair-id`, `--audit`, `--approval` and a bounded
`--budget-seconds`. `batch` also requires two matching full qualification runs
through repeated `--run` arguments. Their timings and evidence sizes establish
per-pair reserves. Resume skips only complete, hash-verified matching pairs. Each pair has its own RunID and lossless compressed
predictions. Run metadata and image/annotation hashes stay local. AP is custom
COCO AP50:95 in original image coordinates with the native-ten ignore-union
protocol. The separately declared FP16 conditions are measured as FP16.

The snapshot command exports the reviewed CPU audit to the dashboard. Raw
predictions, GT, preview images and private paths remain outside Git. Source
checksums identify the local evidence. They do not grant public access to it.

Dependencies: the existing YOLO and FlamaPy environments and Matplotlib for
figures. CUDA is required for GPU modes. No training or dependency installation
is performed by this entry point.
