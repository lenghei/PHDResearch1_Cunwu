# Week 4 · Configuration samples

## Frozen first cohort

The 100 P × 5 D matrix uses seed 20261104. The original draw was uniform without replacement over 26,496 configurations eligible at the time. The population and IDs remain tied to that release.

- [100 degradation configuration IDs](review/p_configurations.csv)
- [Five detector configuration IDs](review/d_configurations.csv)
- [500 planned pairs](review/first500_pairs_PLAN_ONLY.csv)
- [Logical coverage](review/sampling_coverage.csv)
- [Regeneration and historical manifest match](review/sampling_reproduction.json)

## Full-space candidates

- [Pairwise and uniform P samples](reproduce_20261125T160144_3ba46079/full_space_p_samples.csv)
- [Uncovered P feature states](reproduce_20261125T160144_3ba46079/uncovered_p_states.csv)
- [Checkpoint-specific D settings](reproduce_20261125T160144_3ba46079/inference_pairwise_candidates.csv)
- [Inference coverage by checkpoint](reproduce_20261125T160144_3ba46079/inference_coverage.csv)
- [Seeds, populations and provenance](reproduce_20261125T160144_3ba46079/full_space_summary.json)

The pairwise cohort covers 1,163/1,163 feasible two-leaf states. The uniform cohort covers 1,162/1,163. FlamaPy checked all 143 checkpoint and inference bindings. INT8 calibration and several backend combinations remain pending. Native-ten and common-six have separate evaluation scopes. These tables were generated without detector responses.

## 548-image development results

The first run sealed 53 unique pairs and verified one same-pair repeat. At the 10% Tiny AP threshold and 100 ms limit, all five settings failed on ten complete degradation groups. The other 90 groups remain unresolved in this snapshot.

- [Verified cell metrics](reproduce_20261126T001041_5cf06595/verified_metrics.csv)
- [Gap threshold sensitivity](reproduce_20261126T001041_5cf06595/threshold_sensitivity.csv)
- [Matched detector differences](reproduce_20261126T001041_5cf06595/paired_detector_differences.csv)
- [Tiny AP by configuration](reproduce_20261126T001041_5cf06595/tiny_ap.png)

Formal completion remains 0/500. The 548-image results and the earlier exploratory test-dev repeat have separate evidence.
