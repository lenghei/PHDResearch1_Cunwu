# VM46 held-out variability

CPU-only protocol freeze and admission review for eight degradation conditions, two historical detectors and 548 development images.

```powershell
.\.venv_rq4\Scripts\python.exe -B -X utf8 main/experiments/experiments_test/vm46_heldout_variability/run.py
```

Each invocation allocates a new local RunID under `datasets/local_runs/vm46_heldout_variability/`. It freezes configurations and the exported VM45 tree, checks SAT and deterministic pixels, records novelty and evaluates resource admission. Historical inputs are read-only.

This entry point cannot launch GPU work. A missing VM46 authorization or insufficient complete-design budget blocks that branch. Empty metric schemas and status records are explicitly marked as unmeasured. Configuration holdout uses the same development images and does not establish independent-image generalization.
