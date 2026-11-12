# VM37 post-cost continuation

The formal study remains 100 epochs at 1280: YOLO26s/YOLOv8s, seeds 42/43/44, matched CONTROL/ALL_FIVE arms. The separate 20-epoch pilot uses YOLO26s, seed 42 and its own scheduler horizon. Both require complete paired resource admission before training.

| Entry | Purpose |
|---|---|
| `admission.py` | Original allocation deadline, hardware observations and thermal history |
| `loader_cost.py` | CPU-only ALL_FIVE data-loading cost; no detector forward |
| `freeze.py` | Separate pilot and unchanged formal protocols |
| `gap_evidence.py` | Matched historical deficits from accepted metrics and saved bootstrap draws |
| `build_bundle.py` | Local-only portable sources, private assets and environment locks |
| `portable.py` | Relocated dry-run; gated matched training with the accepted trainer |
| `evaluate.py` | Raw predictions, native10 COCO metrics and paired image-bootstrap contrasts |
| `verify_portable.py` | Isolated CPU dry-runs, initialization health and pixel checks |
| `export_bundle.py` | Verified local ZIP containing frozen members only |
| `test_post_cost.py` | Budget, temperature, authorization and contrast tests |
| `finalize.py` | Source-linked final decision and preservation checks |

No entry installs dependencies or uploads data. The portable bundle stays under the private run store. The authorization template denies execution. A destination machine needs measured cost and thermal admission; CPU verification does not establish multi-epoch GPU reliability.

ALL_FIVE retains five independent 0.5 gates. It is broad stochastic augmentation, not proven Gap-guided training. Historical development deficits do not establish augmentation benefit. Exact training resume is not validated; partial checkpoints retain evidence but are not an exact continuation protocol.
