# UAV-GAP: Automatic Dashboard Study Discovery

This update enables the Streamlit dashboard to discover published experiments under `reports/results/` without maintaining a manual study list. Existing VM1 and VM2 visualizations are retained.

## Installation

Copy these files into the Git repository at `E:/ForStudy/code/uavgap/main/`:

- `code/tools/dashboard_autodiscovery.py`
- `code/tools/install_dashboard_autodiscovery.py`
- `code/tests/test_dashboard_autodiscovery.py`

From the Git repository root:

```powershell
& ..\.venv\Scripts\python.exe -B code/tools/install_dashboard_autodiscovery.py
& ..\.venv\Scripts\python.exe -m unittest discover -s code/tests -p 'test_dashboard_autodiscovery.py' -v
```

The installer modifies `code/tools/dashboard.py` and saves its original content outside Git as `../datasets/research_archive/dashboard_backups/dashboard_before_autodiscovery.py`. This backup is for local recovery. Running the installer a second time is safe.

## Publishing new results

Place each independent run at:

```text
reports/results/<experiment>/<RunID>/
    manifest.json
    csv/
        metrics_overall.csv
        metrics_by_area.csv
        training_epoch_times.csv
```

The CSV files depend on the experiment; the dashboard discovers additional CSV tables automatically. A manifest with matching `ExperimentID` and `RunID` and a `result` object is supported. SHA256 checksums in `result.csv_SHA256` are checked against the published CSV files. The reader also accepts Git-only CRLF-to-LF newline normalization while preserving exact data values. Existing `schema: 2` / `snapshot.json` releases continue to use the existing dashboard reader.

New experiment directories and RunIDs appear automatically after GitHub updates and Streamlit reruns. The **Refresh** button reruns the page. A separate code change is not needed for each new experiment or run.

Only committed, non-ignored files reach Streamlit Community Cloud. Large datasets, weights, raw predictions, and local configuration files are not included in this public viewer.

## Verification

```powershell
& ..\.venv\Scripts\python.exe -B code/tools/install_dashboard_autodiscovery.py
& ..\.venv\Scripts\python.exe -m unittest discover -s code/tests -p 'test_dashboard_autodiscovery.py' -v

git status --short

git add code/tools/dashboard.py code/tools/dashboard_autodiscovery.py code/tools/install_dashboard_autodiscovery.py code/tests/test_dashboard_autodiscovery.py

git commit -m "Discover dashboard studies from published results"
git push origin main
```

If the remote branch has diverged because it was edited online, follow the project's established local-first Git sync procedure rather than pulling changes blindly.
