# VM2: Degradation gap analysis

CPU replay of SHA-verified historical tables. It performs no inference or model fitting.

```text
python -B experiments/vm2_degradation_gap_analysis/run.py --replay
```

Copy `config.example.yaml` to ignored `config.local.yaml` and set the historical
artifact root and an output directory outside this repository.
The local archive uses `source_index.json` to resolve historical metric references
without opening the original workspace. The index and copied files are SHA-verified.
The replay checks summary hashes and the underlying detector metric files.
It retains the original RunIDs, checkpoints, dataset/protocol cohorts and source hashes.

Sources include the 243-point native10 grid, the separate 54-point JPEG subgrid,
fixed-original-area scale analyses, UVL timing, and frozen sampling/model comparisons.
The source index is in `results/source_index.json`. Historical 300-image data are
not pooled with new 548-image VM1 measurements.

Direct interactions require all four or eight matched corners.
Missing corners produce `NOT_EVALUATED / NOT_ESTIMABLE`.
Outputs include `configuration_metrics.csv`, `interaction_replay.json`, separate
historical summaries and `provenance.json`. Timing refers to recorded CUDA events
where available, not standardized deployment latency.
Each replay creates `index.html`, interaction plots and their source CSV in its
external run directory. These views use saved measurements and perform no inference.
The parent experiment `index.html` lists all runs, newest first, and the latest
successful replay. Use `--refresh-index` to rebuild it without replaying metrics.
Completion output includes start/finish times, total wall time, record counts and
clickable file URLs. The result page opens by default; use `--no-open` for batch runs.
Measured run timing is saved in `timing.json` and `result.json`.

Static Gap requires documented, checkpoint-specific training-coverage rules with
evidence hashes. Missing rules mean **NOT_READY**, not an inferred coverage gap.
Empirical Gap requires explicit AP/latency thresholds; defaults are unset.
Rule evaluation returns PASS, FAIL, UNCERTAIN or NOT_EVALUATED.

CPU replay passed. Scientific gap admission is **PARTIAL** until coverage evidence
and task thresholds are supplied. Sampling summaries remain retrospective diagnostics;
no new learned constraints or independent-validation claims are made.
