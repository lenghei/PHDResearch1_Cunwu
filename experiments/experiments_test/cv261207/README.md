# CV261207 dual-variability evaluation

`run.py` audits the original UVLs and freezes a uniform sample crossed with
five YOLOv8s inference configurations. The 500 planned pairs use one trained
checkpoint. Rolling-shutter configurations stay in FM_P but are excluded from
the executable cohort until GT and ignore-region geometry can be transformed
and validated.

From the outer project root in PowerShell:

```powershell
$py = '.\.venv\Scripts\python.exe'
$entry = 'main/experiments/experiments_test/cv261207/run.py'
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

## Historical interaction evidence

[Verified VM26 findings](HISTORICAL_EVIDENCE.md) cover 54 development configurations.
They do not admit test-dev or add formal pairs.

```powershell
& $py -B main/experiments/experiments_test/cv261207/historical_review.py review
& $py -B main/experiments/experiments_test/cv261207/history_gate_review.py
& $py -B main/experiments/experiments_test/cv261207/summarize_history.py --run <review-RunID> --gate <gate-RunID>
& $py -B main/experiments/experiments_test/cv261207/publish_history.py --run <review-RunID> --gate <gate-RunID>
```

The review verifies archived metrics and raw predictions, reproduces the saved
200-draw intervals, and independently rematches two raw configurations and six
bootstrap draws. A depth-2 tree describes observed gains; image-disjoint held-out
performance is not reported. Figures and full provenance stay in the local run.
The explicit publish command exports only small verified summaries to the
read-only dashboard. It does not commit or push.

## RQ1–RQ4 acceptance

`rq_evidence.py` verifies accepted inputs, exports the 34-feature compatibility
matrix, joins 279 unique historical development responses, evaluates conditional
subset gaps and prepares a separate cross-architecture v2 plan. It preserves the
frozen 500-pair v1 plan and the independent-test gate.

```powershell
& $py -B main/experiments/experiments_test/cv261207/rq_evidence.py
& $py -B main/experiments/experiments_test/cv261207/rq_evidence.py --live-demo
& $py -B main/experiments/experiments_test/cv261207/rq_figures.py --from-run <acceptance-RunID>
& $py -B main/experiments/experiments_test/cv261207/rq_publish.py --run <acceptance-RunID>
& $py -B main/experiments/experiments_test/cv261207/rq_publish.py --run <acceptance-RunID> --link-local --figures <figures-RunID>
```

The default reuses saved GPU evidence. `--live-demo` runs one development image
under a 120-second worker limit; it does not train or run a formal pair. Figures
export PNG, SVG, PDF and source CSV. A new figure run preserves earlier versions.
The local `rq_index.html` links the latest accepted report and figure book.

Held-out learning scores require both configuration and image separation.
Shared-image descriptive fits do not meet that gate. Unknown latency does not
establish PASS; a verified AP failure can establish FAIL independently.
The matched-augmentation protocol has a runnable CPU validator, not an accepted
training executor. RQ4 remains NOT_EVALUATED.

## New development measurements

`development_v1.py` freezes the 248 validation IDs outside the historical
300-image fitting set and two new degradation configurations. It measures
native10 FP32 predictions, replays COCO metrics, and saves 200 image-bootstrap
draws. Validation selection exposure remains; formal test-dev stays blocked.

`development_recovery.py` preserves interrupted files and recovers statistics
from complete saved predictions. A continuation skips verified measurements
and uses its own bounded execution record. Lost timing is not reconstructed.

`development_analysis.py` reports matched resolution contrasts, AP/latency
states, finite-subset diagnostics and fixed development regression baselines.
`development_figures.py` exports the source-linked figure update; PDF merging
uses `pypdf` in `UAVGAP_PDF_PYTHON`, or the current interpreter.
`development_delivery.py` publishes only small verified dashboard summaries.

The separate clean-control canary has a 600-second limit and cannot start
long training. Its preparation audits all 6,471 training annotations and
builds a derived label view without changing original ZIP files.

## Supervisor plan audit

The following commands write supporting records to the local run directory.

From the local project root:

```powershell
& $py -B main/experiments/experiments_test/cv261207/supervisor_compliance.py --pdf <plan.pdf> --pdf-python <python-with-pypdf>
& $py -B main/experiments/experiments_test/cv261207/supervisor_finish.py --run <new-audit-RunID> --pdf-python <python-with-pypdf>
& $py -B main/experiments/experiments_test/cv261207/supervisor_figures.py --run <new-audit-RunID> --pdf-python <python-with-pypdf>
```

Render and inspect all twelve PDF pages before final acceptance. Record the
reviewer, `status: PASS`, `PDF_sha256` and the twelve `rendered_pages` filename/SHA
pairs in the Run's `visual_review_input.json`. Rendered files belong in
`pdf_qa_final/`. Acceptance rejects missing or stale review evidence.

```powershell
& $py -B main/experiments/experiments_test/cv261207/supervisor_accept.py --run <new-audit-RunID>
& $py -B main/experiments/experiments_test/cv261207/supervisor_replay.py --pair <accepted-development-pair-RunID>
& $py -B main/experiments/experiments_test/cv261207/supervisor_replay.py --pair <accepted-development-pair-RunID> --metrics
```

The audit is CPU-only and has one 1,800-second wall budget across its stages.
CPU operation profiles use actual input shapes, with THOP's `2 × MAC` convention;
they exclude NMS and are not GPU latency measurements. Replay defaults to hash
verification. `--metrics` recomputes COCO AP from saved predictions and GT in a
new Run, without changing the source. Sealed audit Runs cannot be continued.

Local output: `datasets/local_runs/cv261207/<RunID>/index.html`. Demonstration
pixels: `datasets/cache/cv261207/<RunID>/`. Neither is a public-data fallback.
