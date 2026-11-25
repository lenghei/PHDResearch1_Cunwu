# CV261116 — timeout hotfix verification (2026-11-01)

The original `PySATConfigurationsNumber` call exceeded 600 seconds on the
Windows `.venv_flamapy` interpreter. The number-of-configurations operation
is not used by this package anymore.

## Validation performed in this packaging environment

- 36 deterministic CPU tests: PASS (`python -B -m unittest discover`).
- Exact factorization: 635,904 FM_D configurations, 844 training variants.
- Structural contract: 34 declared leaves, exactly the original four UVL constraints.
- Mutation tests: changed UVL constraints and features are rejected.
- Stub-only SAT flow test: `PySATSatisfiable` and `PySATDeadFeatures` requested,
  never `PySATConfigurationsNumber`. **Stubs are engineering tests, not real SAT evidence.**
- Previous FM_P static proxy tests remain unchanged: 18 / 46,368; identity: 1 / 46,368.

## Still required on the user's Windows machine

- Real FlamaPy UVL parse and PySAT satisfiability/dead-feature validation.
- GPU profiling on the user's RTX 4060 and actual checkpoints.
- Review of saved training augmentation evidence and source.
- Independent BDD or #SAT model count is **NOT** claimed or performed.

The returned `backend` and `count_backend` in `model_count.json` distinguish
FlamaPy decision checks from the exact algebraic count. A successful check
retains the existing `PARTIAL` milestone status until measurements are collected.
An old failed/timeout RunID remains unchanged; retry creates a new RunID.
