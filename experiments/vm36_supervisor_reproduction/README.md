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
