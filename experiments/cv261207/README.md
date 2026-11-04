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
& $py -B $entry smoke-grid --audit <audit-RunID> --budget-seconds 240
& $py -B $entry demo --audit <audit-RunID> --seed 20261104 --budget-seconds 120
& $py -B $entry regenerate --run <saved-pair-RunID>
& $py -B $entry figures --audit <audit-RunID> --run <saved-pair-RunID>
& $py -B $entry publish --audit <audit-RunID>
```

Smoke and demo use validation images for engineering checks. The formal branch
uses the original 1,610-image test-dev ZIP and requires a separate local
historical-use decision containing `status: AUTHOR_REVIEWED_ADMIT`, the audit
RunID, dataset gate SHA, test-dev ZIP SHA, selection SHA, protocol ID and review
evidence. A complete archive alone does not establish historical isolation.

`full` takes one frozen `--pair-id`, `--audit`, `--approval` and a bounded
`--budget-seconds`. `batch` uses the same admission and skips only complete,
hash-verified matching pairs. Each pair has its own RunID and lossless compressed
predictions. Run metadata and image/annotation hashes stay local. AP is custom
COCO AP50:95 in original image coordinates with the native-ten ignore-union
protocol. The separately declared FP16 conditions are measured as FP16.

The snapshot command exports the reviewed CPU audit to the dashboard. Raw
predictions, GT, preview images and private paths remain outside Git. Source
checksums identify the local evidence. They do not grant public access to it.

Dependencies: the existing YOLO and FlamaPy environments and Matplotlib for
figures. CUDA is required for GPU modes. No training or dependency installation
is performed by this entry point.
