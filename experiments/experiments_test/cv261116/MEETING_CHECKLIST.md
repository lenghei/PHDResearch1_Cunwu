# CV261116 — live supervisor meeting checklist

- [ ] `python -B -m unittest discover` finishes successfully.
- [ ] Run `run.py --check` live, without cached solver output.
- [ ] Explain the 34 leaf features and the four cross-tree constraints.
- [ ] Show independent count 635,904 and independent training variants 844.
- [ ] Show actual FlamaPy `PySATConfigurationsNumber` count 635,904.
- [ ] Open saved `static_gap_report.html` and `coverage_rules.csv`.
- [ ] Explain why the 18 covered configurations are provisional proxies, not exact pixel equivalence.
- [ ] Explain strict identity sensitivity: one covered configuration.
- [ ] Show saved actual training arguments and installed augment source SHA; explain if review is still pending.
- [ ] Run an original-VisDrone GPU latency pilot against a real checkpoint and show its CSV / hashes.
- [ ] State precisely which branches were not evaluated (SAHI, INT8, TTA, etc.).
- [ ] Do not claim AP, empirical gap or all 635,904 inference configurations were measured.

## Solver-method note (timeout repair)

The 600-second PySATConfigurationsNumber timeout was addressed by removing
#SAT enumeration. The recorded 635,904 count is an exact factorized proof for
the strictly guarded UVL model; FlamaPy independently validates UVL parsing,
satisfiability and the absence of dead features. Do not say FlamaPy itself
counted 635,904 unless a separate BDD/#SAT run is actually recorded.
