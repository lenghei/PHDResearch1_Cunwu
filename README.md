# UAV-GAP

Research project on the relationship between UAV image degradations and object detection configurations.

The project uses UVL feature models and FlamaPy to manage valid configurations, evaluate detector robustness, and study the gaps between training conditions and real-world image degradations.

Current experiments use YOLO and VisDrone, with support for extending the evaluation to other detectors and datasets.

**Online dashboard:** [https://uavgap.streamlit.app](https://uavgap.streamlit.app/)

## Project Structure

- `code/` — Shared implementations, tools and tests.
- `experiments/` — Experiment code, configurations and results.
- `reports/results/` — Summary results and figures.
- `reports/notes/` — Research notes and progress records.

## Experiments

| Experiment | Description |
|---|---|
| `vm1_flamapy_yolo26s_validation` | Configuration validation, image degradation and detector evaluation. |
| `vm2_degradation_gap_analysis` | Degradation interactions, configuration sampling and gap analysis. |

## Setup

Dependencies are listed in `code/requirements.txt` and `code/requirements-flamapy.txt`. The detector and FlamaPy use separate Python environments.

Copy the experiment's `config.example.yaml` to `config.local.yaml` and set the local dataset, checkpoint and output paths.

Datasets, model weights and full experiment outputs are not included in this repository.

## Run

From the repository root:

**VM1 — Full validation**

```bash
python -B experiments/vm1_flamapy_yolo26s_validation/run.py --full-paired
python -B experiments/vm1_flamapy_yolo26s_validation/run.py --full-clean
python -B experiments/vm1_flamapy_yolo26s_validation/run.py --check
```

**VM2 — Analysis**

```bash
python -B experiments/vm2_degradation_gap_analysis/run.py --replay
```

See the individual experiment directories for evaluation protocols, reproducibility details and results.
## Dashboard

The existing viewer supports local evidence and cloud-safe published summaries.
See [viewer setup and deployment](code/tools/README.md). Dashboard dependencies
are isolated in `code/tools/requirements.txt`.
