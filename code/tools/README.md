# Dashboard

Local viewer: run `code/tools/run_dashboard.py` from PyCharm with the dashboard environment. Alternatively, from the Git root:

```powershell
$env:UAVGAP_DASHBOARD_MODE = 'local'
python -m streamlit run code/tools/dashboard.py --server.address=127.0.0.1
```

Cloud viewer: Streamlit Community Cloud entry point `code/tools/dashboard.py`; dependencies `code/tools/requirements.txt`. The cloud viewer reads only reviewed, versioned snapshots under `reports/results/<experiment>/<RunID>/`.

Each run has its own `manifest.json` with the snapshot SHA256. Add or delete an entire RunID directory to publish or remove that result; there is no global result index. See `reports/results/README.md` for the publishing command.

The dashboard is read-only; it does not execute inference, model training or metric recalculation. Complete predictions and local data remain outside Git. Historical development results are not independent test results.

## Matched image comparisons

From the project root (`uavgap`), after the VM1 full-paired Run has passed:

```powershell
$run = '.\datasets\local_runs\vm1_flamapy_yolo26s_validation\<RunID>'
& .\.venv\Scripts\python.exe -B main/code/tools/publish_image_comparison.py --run $run --ids 1 2 3
```

This reads the original dataset ZIP, saved transformed images and already evaluated predictions. It produces local `image_comparison/` previews. It does not run the detector or recalculate AP.

After checking redistribution rights for the source images and attribution, optionally publish the selected previews:

```powershell
& .\.venv\Scripts\python.exe -B main/code/tools/publish_image_comparison.py --run $run --ids 1 2 3 --publish --acknowledge-license
```

Review the generated files under `reports/results/vm1_flamapy_yolo26s_validation/<RunID>/image_comparison/` before committing. The Dashboard's **Image comparisons** tab reads the local evidence or these audited public thumbnails. Do not publish full datasets, raw predictions, or private configuration files.
