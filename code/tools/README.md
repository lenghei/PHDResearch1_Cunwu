# UAV-GAP Dashboard

Both modes run the same Streamlit application, `code/tools/dashboard.py`.

- **Local:** run `code/tools/run_dashboard.py` from the outer project root `uavgap/`. This launcher starts `dashboard.py` with `UAVGAP_DASHBOARD_MODE=local`, using `.venv_dashboard`. Open `http://127.0.0.1:8501`.
- **Cloud:** Streamlit Community Cloud entry point is `code/tools/dashboard.py`, with `code/tools/requirements.txt` as the requirements file. Cloud mode only reads committed results.

## Automatic discovery

The experiment menu is constructed from saved run directories on every page rerun; there is no experiment name registry to edit.

In **local mode** the dashboard combines:

1. `uavgap/datasets/local_runs/<ExperimentID>/<RunID>/` for complete local run metadata, including native `selection.json` + `result.json` FM_P runs, and older `run.json`/`summary.json` formats.
2. `uavgap/main/reports/results/<ExperimentID>/<RunID>/` for SHA-verified published snapshots and supported CSV experiments when their private runs are not present.

If a RunID exists in both places, the local run is preferred. A local CV261109 run additionally displays the original/degraded JPEG previews under `preview/`. Published-only snapshots show saved configuration and generation evidence without source images.

In **cloud mode** only `main/reports/results/` is readable. Neither local runs, user data, checkpoints nor private configuration are accessed. New experiment pages appear after committing new compatible result folders and the Streamlit app reruns. No edit to `dashboard.py` or manual study registration is necessary.

Supported saved formats:

- Published schema-2 `manifest.json` + verified `snapshot.json`.
- Published experiment `manifest.json` with CSV evidence.
- Local `selection.json` + `result.json` (CV261109 and compatible future experiments).
- Legacy local `run.json` and `summary.json`.
- Local identity-valid `result.json` with `RunID` and `ExperimentID` for other simple experiments.

Select **Refresh** in the sidebar or reload the page to discover files written since the last run. This is a read-only viewer: it does not perform inference, image transformations or metrics recalculation. Image-generation runs display `AP: NOT_EVALUATED` when detection has not been run. Large outputs and raw predictions are not published automatically.

## Local setup

From `E:\\ForStudy\\code\\uavgap` in PowerShell:

```powershell
& .\\.venv\\Scripts\\python.exe -B main/code/tools/run_dashboard.py
```

If `.venv_dashboard` is not installed, create it once and install `main/code/tools/requirements.txt`. The launcher creates no project files or run results.

## Matched YOLO image comparisons

After a VM1 full-paired run is validated, use `code/tools/publish_image_comparison.py` with the `--run` argument for the original local evidence. Check redistribution rights before committing any selected source thumbnails.
