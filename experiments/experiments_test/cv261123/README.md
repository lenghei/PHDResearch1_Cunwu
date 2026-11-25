# CV261123 — Two variability models, research figures, and sampled pair design

This experiment turns the advisor's 26 October hand-drawn sketch and the 22 October research plan into reproducible, source-driven visualizations. It does not modify CV261109, CV261116, or historical results.

## Research boundary

The figures use **actual validated model and static-analysis files** (CV261109's `fm_p.uvl` and `model.py`; CV261116's `fm_d.uvl`, `result.json`, `model_count.json`, `static_gap.json`, and feature-coverage CSV). The 18/46,368 result is a **provisional augmentation-coverage proxy**, not a measured detector failure. The FM_D 635,904 count uses the documented exact factorization plus FlamaPy SAT checks; it is **not** an independent BDD/#SAT count. No detector AP, learned interaction, empirical gap, or targeted-retraining improvement is inferred.

## Commands (from project root)

```powershell
# Run from your local uavgap project root.
& .\.venv\Scripts\python.exe -B main/experiments/experiments_test/cv261123/run.py --check
& .\.venv\Scripts\python.exe -B main/experiments/experiments_test/cv261123/run.py
```

`--check` verifies the saved CV261116 file checksums and both original UVL hashes, enumerates the 46,368 valid FM_P configurations, and constructs the deterministic sampling design. It does not create a run or use the GPU. Full execution creates a **new** folder in `datasets/local_runs/cv261123/<RunID>/` with nine SVG and PNG figures, a combined figure PDF, a local `index.html`, a source manifest and sampling CSVs. Historical RunIDs are not overwritten.

When automatic evidence discovery cannot identify your static run, supply its folder explicitly:

```powershell
& .\.venv\Scripts\python.exe -B main/experiments/experiments_test/cv261123/run.py --static-run '.\datasets\local_runs\cv261116\<StaticRunID>'
```

## Generated figures

| Figure | Content | Evidence status |
| --- | --- | --- |
| 01 | FM_D + FM_P + sampling + oracle + two gaps | Conceptual research framework |
| 02 | UVL feature groups and checked counts | Saved model audit |
| 03 | 46,368 × 635,904 = 29,485,596,672 theoretical pairs | Exact model cardinality product |
| 04 | Selected-leaf histogram across all FM_P configurations | Full exact Python enumeration |
| 05 | Provisional proxy coverage vs exact identity | Saved CV261116 static result |
| 06 | Leaf-level static-coverage counts | Saved CSV with SHA256 validation |
| 07 | Pairwise + uniform sampling; first 500 planned pairs | Sampling design, not GPU results |
| 08 | RQ1–RQ4 evidence readiness | Saved evidence and supervisor plan |
| 09 | Blank cross-model influence matrix | Explicitly not measured |

`results/` contains a pre-rendered preview based on the audited CV261116 static RunID `cv261116_static_20261101T223010_016ba75c`. It is not a new measured experiment; regenerate to update figures from your current local records.

## Sampling protocol

- Enumerate all 46,368 valid FM_P leaf assignments from the existing CV261109 `model.py`; verify the current FM_P UVL hash against the saved CV261116 result.
- Compute all feasible **binary t=2 leaf-presence states** (not the same thing as exhaustive physical pairwise effects). The recorded model has 1,163 feasible states, covered by 21 greedy configurations. Pad to 80 unique pairwise configurations using a deterministic seed.
- Draw 100 configurations uniformly **without replacement from the full valid space**, using a seeded `random.sample`. This uniform subset should be kept distinct when later estimating population quantities. It is not a sample over equal-sized feature-count strata. **This implementation draws from an exhaustively enumerated, independently FlamaPy-validated space using Python's seeded `random.sample`; it does not claim to invoke FlamaPy's BDD sampling backend.** Check the official FlamaPy `Sampling` operation separately with the supervisors before replacing the method.
- Keep 20 active-learning slots **unselected** until measured pair outcomes are available.
- The first planned 500 pairs use 100 distinct FM_P configs × 5 unbound FM_D detector slots. The slots are **not checkpoints** and must be bound to validated trained weights and legal FM_D assignments before evaluation. The CSV includes no invented AP or latency.

For final paper claims, the advisor requires the held-out `VisDrone2019-DET-test-dev` split (1,610 images), not the 548-image development validation split. The initial 500-pair plan would therefore require 805,000 image-inference instances **if every pair is evaluated over the entire test-dev split**; run nothing until real time, disk and GPU budgets are checked. The source dataset and annotation hashes must be recorded. Do not tune decisions based on the held-out test split.

## Dependencies and privacy

Use the existing `.venv` (the project already requires `matplotlib`). Source files and comments are English. Private images, checkpoints, full logs and machine paths remain outside `main/`. All generated RunID folders remain outside the GitHub root. This package contains **no** `.pyc`, `__pycache__`, source images, checkpoints, or personal config.

The figure scripts provide reproducible visualizations and research-tool implementation. The student's final scientific interpretations, references and manuscript claims must be verified with the supervisors, following the research plan's working rules.

## Source links for follow-up verification

- FlamaPy sampling interface: https://docs.flamapy.org/framework/operations/sampling
- Official VisDrone data and availability of test-dev ground truth: https://github.com/VisDrone/VisDrone-Dataset
- Supervisor research plan (22 October 2026): source held privately, pp. 1-17.
