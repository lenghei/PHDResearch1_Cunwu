# Dashboard

## Local command

From the Git root, using the dashboard interpreter:

```powershell
$env:UAVGAP_DASHBOARD_MODE = 'local'
python -m streamlit run code/tools/dashboard.py --server.address=127.0.0.1 --browser.gatherUsageStats=false
```

The PyCharm shortcut is `code/tools/run_dashboard.py`; it selects the separate
`.venv_dashboard` environment and local evidence mode automatically.

## Cloud entry point

Publish this repository root, then choose the repository and branch in Streamlit
Community Cloud. Set the entry point to `code/tools/dashboard.py` and Python to
3.11, matching the tested environment. No secrets are required. Cloud mode is the
safe default; do not set `UAVGAP_DASHBOARD_MODE=local` on Community Cloud.

Only the three reviewed snapshots in `reports/results/` are used. The manifest
pins each bundle SHA256; bundles retain RunIDs, model/protocol identity and source
artifact hashes. Exporting snapshots is an explicit local step, never an app action.

## Dependencies

Install `code/tools/requirements.txt`: Streamlit, Plotly and pandas only.
Community Cloud discovers this file beside the entry point. No Torch, CUDA, YOLO,
raw datasets, weights, private YAML, logs or inference environment are required.

## Limitations

Both modes are read-only. Cloud mode shows selected accepted snapshots, not the
complete local archive. VM1 currently reports CPU/UVL acceptance, not measured
YOLO26s AP. VM2 shows historical 300-image development results; dataset/protocol
cohorts remain separate. Plotted means are configuration means, not pooled-image
AP or significance tests. Missing or corrupted published summaries are reported
without falling back to private data. Actual Community Cloud deployment remains
an author action; local cloud-mode testing is not a hosted deployment test.
