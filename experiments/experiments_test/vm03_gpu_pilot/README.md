# VM03: checkpoint evaluation

Run `run.py` without arguments for full validation using the adjacent private
`config.local.yaml`. A local HTML preview opens after evaluation.
Use `--smoke` for five images, `--no-open` to keep the browser closed, or
`--preview-run <saved-run>` to visualize saved outputs without inference.

Uses the shared `main.evaluation` module. From the repository root, copy `experiments/experiments_test/vm03_gpu_pilot/config.example.yaml`
to `experiments/experiments_test/vm03_gpu_pilot/config.local.yaml` and set external paths.

```text
python -B experiments/experiments_test/vm03_gpu_pilot/run.py --config experiments/experiments_test/vm03_gpu_pilot/config.local.yaml
```

Add `--full` for all 548 validation images; this was not run in GIT03.
Use `--regenerate <saved-run> --output <new-metrics.json>` to recompute metrics.

`results/smoke_summary.json` records the five-image engineering run, hashes and
timings. Complete predictions, GT and traces remain in the external artifact store.

The saved smoke command records the pre-cleanup layout; use the command above
for the current layout. Historical results are unchanged.
