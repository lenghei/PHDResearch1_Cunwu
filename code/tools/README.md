# Dashboard

Local viewer: run `code/tools/run_dashboard.py` from PyCharm with the dashboard environment. Alternatively, from the Git root:

```powershell
$env:UAVGAP_DASHBOARD_MODE = 'local'
python -m streamlit run code/tools/dashboard.py --server.address=127.0.0.1
```

Cloud viewer: Streamlit Community Cloud entry point `code/tools/dashboard.py`; dependencies `code/tools/requirements.txt`. The cloud viewer reads only reviewed, versioned snapshots under `reports/results/<experiment>/<RunID>/`.

Each run has its own `manifest.json` with the snapshot SHA256. Add or delete an entire RunID directory to publish or remove that result; there is no global result index. See `reports/results/README.md` for the publishing command.

The dashboard is read-only; it does not execute inference, model training or metric recalculation. Complete predictions and local data remain outside Git. Historical development results are not independent test results.
