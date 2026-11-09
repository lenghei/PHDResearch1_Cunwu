# VM36 · Supervisor-aligned reproduction

This stage freezes a reviewable protocol and audits source evidence. It does not
run detector inference or training, approve scientific choices, or claim completed full-split experiments.
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

Scientific review, runtime-binding approval and measured compute/disk admission
are still required before new detector measurements. Formal test-dev remains
blocked. Neither the candidate grid nor software tests complete RQ1–RQ4.
