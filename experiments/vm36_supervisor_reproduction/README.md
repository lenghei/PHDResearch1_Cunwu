# VM36 · Supervisor-aligned reproduction

Preparation freezes a reviewable protocol and audits source evidence. A separate,
bounded development worker measures admitted full-split conditions; preparation
does not authorize that worker or complete the research questions.
VM36 follows the archived VM00–VM35 identifiers; historical CV studies are reused.

The [FM_P](feature_models/fm_p.uvl) and [FM_D](feature_models/fm_d.uvl) replicas
match the supervisor PDF's numerical attributes. Their Boolean trees and
constraints are unchanged. The twelve historical rendering attributes are a
separate unapproved proposal. Motion conditions cannot silently inherit the old
orientation distribution. Draft model-size attributes are not measured counts.

From the local project root:

```powershell
$py = '.venv/Scripts/python.exe'
& $py -B main/experiments/vm36_supervisor_reproduction/run.py prepare
& $py -B main/experiments/vm36_supervisor_reproduction/run.py audit --run <new-RunID> --meeting-dir <original-meeting-directory> --source-run <verified-data-audit-RunID>
& $py -B main/experiments/vm36_supervisor_reproduction/benchmark.py --protocol-run <new-RunID>
& $py -B main/experiments/vm36_supervisor_reproduction/reporting.py --run <new-RunID> --benchmark-run <benchmark-RunID>
& $py -B -m unittest discover -s main/experiments/vm36_supervisor_reproduction/tests -v
& $py -B main/experiments/vm36_supervisor_reproduction/verify.py --run <new-RunID>
```

`audit` reuses complete verified dataset evidence by SHA when `--source-run` is
provided. Omitting it performs a new source audit and requires unused meeting
report filenames. Completed or sealed Runs must not be overwritten.

The CPU benchmark has a 600-second worker limit. It compares the installed COCO
accumulator with the existing validated vectorized implementation on identical
saved predictions and 200 paired image resamples. It does not measure GPU speed.

Private outputs: `datasets/local_runs/vm36_supervisor_reproduction/<RunID>/`.
The fixed `index.html` lists history. Chinese operating notes stay in
`datasets/reportcn/`. The existing Streamlit app discovers the saved local VM36
summary; it never executes the experiment. No unreviewed summary is exported to
cloud mode. Raw audio, datasets, predictions and weights remain outside Git.

Open the existing dashboard locally:

```powershell
$env:UAVGAP_DASHBOARD_MODE = 'local'
& .venv_dashboard/Scripts/python.exe -m streamlit run main/code/tools/dashboard.py
```

## Complete-validation development worker

`full_development.py` uses two primary checkpoints, two inference resolutions
(640/1280), and the original JPEG Q25/Q75 endpoints. Four matched corners per
checkpoint are required for a resolution × JPEG contrast: eight planned pairs,
each with all 548 validation images. Neither JPEG endpoint is CLEAN.

The worker has a 3,600-second total limit, measures the first complete pair and
checks remaining time before each subsequent pair. It may finish only a subset.
It preserves FP32, native10, the existing dense head/NMS/ignore rules, genuine
COCO evaluation, raw prediction replay and 200 paired image-bootstrap draws.
No training or independent test-dev evaluation is performed.

After a separately authorized worker has stopped, regenerate its summary:

```powershell
& .venv/Scripts/python.exe -B main/experiments/vm36_supervisor_reproduction/full_results.py --run <full548-RunID>
```

Only sealed, verified pairs enter summaries. Incomplete corners are not
interpolated. Completed pairs can be reused by identity; cross-reboot GPU
continuation is not yet validated. Regeneration does not restart inference.

`full_analysis.py` checks bootstrap pairing and incomplete-subset states.
`coverage_evidence.py` exports the 21-configuration FM_P core and its exact binary
coverage matrix. Sample-size tables are precision sensitivities, not selected
population-estimation budgets. The old 500/900 manifests remain planning
artifacts, not execution targets.

Motion orientation, rolling-shutter geometry and other missing runtime bindings
still need resolution before those conditions can be measured. Formal test-dev
remains blocked. Matched three-seed augmentation training is resource-blocked.
Software acceptance does not establish scientific completion of RQ1–RQ4.

## Completing missing corners

`complete_factorial.py` selects only the two missing ck01/1280 JPEG conditions
from the immutable six-pair source. It checks source hashes, checkpoint health,
disk/GPU memory and a conservative measured cost estimate. Its independent
worker limit is 1,800 seconds. The original run and its budget stay unchanged.
Do not rerun this GPU command to view or regenerate results.

After the explicitly authorized completion worker has stopped successfully:

```powershell
& .venv/Scripts/python.exe -B main/experiments/vm36_supervisor_reproduction/factorial_analysis.py --completion-run <missing-corners-RunID>
```

This CPU command creates a new joined analysis without copying old predictions.
It verifies all eight corners, bootstrap pairing and protocol identity; exports
AP/AR, size and per-class effects; and compares the two checkpoint DIDs.
Per-class intervals are marked unavailable because the source bootstrap did not
save per-class draws. The joined page reports six reused and two new conditions.

The follow-up audit separates the full logical population from a rendering-
parameter-admitted stratum. Its pairwise core is a CPU design, not measured AP.
Uniform sample-size/cost tables remain planning sensitivities. An eight-point
saturated fit is not accepted as held-out failure-rule learning. No learned UVL
constraint or augmentation efficacy is inferred from the factorial study.

`joint_design.py` prepares feature-only joint coverage, reusing accepted points.
`renderer_preflight.py --design-run <RunID>` checks its unmeasured recipes on six
fixed source images under a 300-second CPU cap. It saves pixel identities and
generation timings, without detector inference or AP evaluation.

## Frozen joint-coverage batch

`joint_batch.py` measures the eighteen frozen additions to the accepted eight
pairs. Its separate 14,400-second worker cap includes cache generation,
inference, COCO replay, bootstrap and reports. It checks the first complete-pair
cost before continuing. A single-use authorization ticket prevents repeating
the batch. This is a restricted 1,872 × 4 projection, not full FM_D coverage.

After the worker stops, run the CPU acceptance and summary:

```powershell
& .venv/Scripts/python.exe -B main/experiments/vm36_supervisor_reproduction/joint_analysis.py --batch-run <joint18-RunID>
& .venv/Scripts/python.exe -B main/experiments/vm36_supervisor_reproduction/joint_acceptance.py --analysis-run <joint-results-RunID>
```

Only sealed full-split results count. The summary separates accepted pairs,
feature-state coverage, measured outcomes and missing detector settings.
Unmatched recipes are not used as controlled interaction contrasts. The original
eight JPEG corners remain the source for their paired contrasts.

`continuation_audit.py` reviews checkpoint/test-dev provenance.
`uavdt_toolkit_source.py`, `uavdt_reconcile.py` and `uavdt_boundary.py` trace
official annotations, converted geometry and ignore semantics without editing
either source. Original UAVDT DET/MOT records and local SOT frames stay separate.
`training_readiness.py` reports matched-training requirements and historical
cost screening; it never starts training. These CPU audits do not admit an
independent test or authorize another GPU task.

## Matched follow-up and finite subset

`matched_followup.py` verifies the 26 unique measurements, treats the two summary
versions as aliases, and compares controlled contrasts, conditional subset
completion and probability sampling separately. It does not query a detector.

`training_data_readiness.py` audits the original 6,471-image ZIP and writes a
private candidate label view. `loader_seed_diagnostic.py` separates label validity
from small CPU augmentation-repeat checks. `post26_readiness.py` records
test-dev/UAVDT admission and the unexecuted matched-training matrix. None starts
training or changes original data.

The authorized full follow-up completes the same 20 contexts × four settings.
The original 26 points and the first three matched additions are reused, leaving
51 points in the second worker. `full20_completion.py` requires the frozen
manifest and a new author record. Its absolute day deadline retains the first
new-study start; a stage restart cannot reset it. Partial outputs remain partial.
Completed-pair resumption requires another explicit record, unchanged protocol
and verified sealed outputs; no exact mid-pair recovery is claimed.

After that worker stops:

```powershell
& .venv/Scripts/python.exe -B main/experiments/vm36_supervisor_reproduction/full20_results.py --worker-run <full20-remainder-RunID>
& .venv/Scripts/python.exe -B main/experiments/vm36_supervisor_reproduction/full20_acceptance.py --analysis-run <full20-results-RunID>
```

The CPU result contains actual metrics, within-context resolution differences,
matched context contrasts and finite subset gap states. A negative context DID
does not by itself imply a negative 1280-minus-640 difference. Pointwise intervals
do not establish training-seed effects.
Shared validation images do not constitute independent predictive validation.
Cloud mode continues to read committed summaries only.

## Scientific stage after the 80-point study

The `post80_*` tools never launch GPU inference or training:

- `post80_replay.py --request <task.txt>` recomputes all saved predictions on CPU
  under a measured-cost wall limit. It preserves the original evaluator.
- `post80_science.py --prepare <task.txt>` checks saved contrasts and freezes legal
  missing P configurations. FlamaPy runs in `.venv_flamapy`.
- `post80_loader.py` checks all 6,471 images with a private label-path adapter,
  followed by repeated CPU batches. Source images are neither copied nor edited.
- `post80_provenance.py --science-run <RunID> --loader-run <RunID>` records direct
  checkpoint, dataset and training-readiness evidence.
- `post80_finalize.py --science-run <RunID> --replay-run <RunID> --loader-run <RunID>`
  requires complete CPU replay before building the evidence report and figures.

Use the project `.venv` and the existing source RunIDs. Each preparation creates
a new private run. The four-point Normal-reference proposal needs a separate
finite GPU allocation. Its six comparisons remain exploratory. A majority-class
diagnostic is not an independently validated failure rule.
