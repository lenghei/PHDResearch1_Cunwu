<!-- development-update-20261108 -->
Latest update: [new development measurements](DEVELOPMENT_EVIDENCE.md), `cv261207_development_delivery_20261108T194939_4be5d3c4`. The earlier acceptance below is retained as a dated historical state.

# RQ1–RQ4 acceptance details

Source acceptance: `cv261207_rq_acceptance_20261107T220632_8271fd92`. Figure run: `cv261207_rq_figures_20261107T221015_cf1dcb63`.
Formal pairs: 0. Independent testing: DATASET_BLOCKED.

## RQ1

1. **research question**: Are the variability models correct and executable?
2. **tested hypothesis**: Preserved UVL logic, counts, deterministic generator and declared feature bindings agree.
3. **experiment design**: Hash-check original UVLs, verify exact counts and attribute IDs, reconcile solver/generator receipts and installed API source; run explicit constraint examples.
4. **actual configurations evaluated**: 46368 FM_P; 635904 FM_D; 844 training variants
5. **actual dataset and split**: Original UVL v1 and accepted synthetic generator tests; no AP dataset population.
6. **measured results**: Logical models validated; physical calibration and full detector executability remain incomplete.
7. **statistical uncertainty**: Exact guarded logical counts, not a statistical estimate. Physical realism and all-configuration executability remain unverified.
8. **reproduction commands**: python -B main/experiments/cv261207/rq_evidence.py
9. **evidence RunIDs**: cv261207_rq_acceptance_20261107T220632_8271fd92|cv261207_audit_20261104T193649_d1b63492|cv261207_vm26review_20261106T212756_fdf80ed8
10. **supported conclusion**: Logical models validated; physical calibration and full detector executability remain incomplete.
11. **unresolved limitations**: Unsupported features and uncalibrated scenario constraints.
12. **acceptance status**: PARTIAL

Source files and SHA256:

```json
{"SAT_demo_receipt.json": "bd6b908cefd84a6993e20081883baf7dfac4ca6733eef4ec8a0926a24b72a509", "fmd_compatibility.csv": "e66d84d32b622b51b5eeca42032215066a0675c8b87388bcd35cd4d536ecc803", "rq1_validation.json": "dd7a8dcf4c313a5108351eb6e5c377156a4dc46847b27efb1edbb84e25a8aebf", "uvl_constraints.csv": "9a4a2a4fa416c8beb42338f4aa6df51a2bd41440202133f1e891c863a5b9f978"}
```

## RQ2

1. **research question**: Which detector and degradation features interact?
2. **tested hypothesis**: Resolution gain depends on blur and additional JPEG under the fixed measured grid.
3. **experiment design**: Join VM09/VM26 with exact overlap removal; preserve measured numeric factors; reuse paired DID inference and fit descriptive tree/stump/forest baselines.
4. **actual configurations evaluated**: 243 VM09 +54 VM26 -18 reused =279
5. **actual dataset and split**: ds04, 300 VisDrone2019-DET-val images; native10; pr01; DEVELOPMENT_ONLY; checkpoint-selection exposure.
6. **measured results**: 279 unique historical responses; verified VM26 three-way effects; descriptive tree/stump/forest fits only.
7. **statistical uncertainty**: Original 200-draw paired image bootstrap with joint36 contrast family. No training-seed or flight-cluster inference. Held-out classifier metrics are not evaluated.
8. **reproduction commands**: python -B main/experiments/cv261207/rq_evidence.py
9. **evidence RunIDs**: cv261207_rq_acceptance_20261107T220632_8271fd92|cv261207_audit_20261104T193649_d1b63492|cv261207_vm26review_20261106T212756_fdf80ed8
10. **supported conclusion**: 279 unique historical responses; verified VM26 three-way effects; descriptive tree/stump/forest fits only.
11. **unresolved limitations**: Shared image population; no image/configuration-disjoint held-out scores.
12. **acceptance status**: PARTIAL

Source files and SHA256:

```json
{"grouped_learning_acceptance.json": "0c9e75532e391e43df288f533555d742a007b7a9c0a90f6302a9c813b3fb268d", "historical_response_union.json": "5ae2acd95ba09a011ac9720a8866ab6bf830822e06903c6eb82e27ca9000a761", "measured_learning_features.csv": "56df8668ea7d0c4e8cba1809e77168c69d3a801d49def39ca2e3ed1a2cf25faa"}
```

## RQ3

1. **research question**: How large are static and empirical gaps?
2. **tested hypothesis**: Static coverage is distinguishable from measured AP/latency failure.
3. **experiment design**: Separate three static policies; compute post-hoc joint72 endpoint/size intervals and four-state AP/latency outcomes on a finite measured subset.
4. **actual configurations evaluated**: 24 endpoints x3 size bins; 4 purposive contexts; 6 detector settings
5. **actual dataset and split**: ds04, 300 VisDrone2019-DET-val images; native10; pr01; DEVELOPMENT_ONLY; checkpoint-selection exposure.
6. **measured results**: 2/4 selected VM26 contexts have tiny-AP-supported failure across six measured endpoints at tau=0.10. Other contexts are unresolved.
7. **statistical uncertainty**: Post-hoc joint72 image-bootstrap intervals; four purposive contexts are not a probability sample of full FM_P. Missing latency prevents complete outcome ascertainment.
8. **reproduction commands**: python -B main/experiments/cv261207/rq_evidence.py
9. **evidence RunIDs**: cv261207_rq_acceptance_20261107T220632_8271fd92|cv261207_audit_20261104T193649_d1b63492|cv261207_vm26review_20261106T212756_fdf80ed8
10. **supported conclusion**: 2/4 selected VM26 contexts have tiny-AP-supported failure across six measured endpoints at tau=0.10. Other contexts are unresolved.
11. **unresolved limitations**: No compatible complete latency, random population sample or independently validated gap predictor.
12. **acceptance status**: PARTIAL

Source files and SHA256:

```json
{"development_subset_gaps.csv": "22de03ed89d40fba241387f35cab5b7421d9055cc55571bad0dcd96714ac93e9", "endpoint_confidence_intervals.csv": "3725853e3fa523fa005ebeea2acbd6e339db862ed16cfb3ca20ab7fd38f4a3d5", "static_empirical_diagnostics.csv": "4a8943cd538efb882504669ebd7d0f51b619a0970cc7b5503d9146a1fbd7ed02"}
```

## RQ4

1. **research question**: Does targeted augmentation close a measured gap?
2. **tested hypothesis**: Matched clean-control versus targeted continuation improves the frozen gap without unacceptable clean loss.
3. **experiment design**: Prepare matched clean-control and targeted continuation arms; run CPU protocol validator only.
4. **actual configurations evaluated**: 2 candidate architectures x6 candidate arms; all planned, zero executed
5. **actual dataset and split**: No training/evaluation data executed in RQ4; proposed inputs retain explicit gates.
6. **measured results**: Versioned matched protocol and CPU validator prepared; no training or gap-closure measurement.
7. **statistical uncertainty**: No before/after response, gap-reduction interval or training-seed variability measured.
8. **reproduction commands**: python -B main/experiments/cv261207/rq4_protocol.py --check datasets/local_runs/cv261207/cv261207_rq_acceptance_20261107T220632_8271fd92/rq4_matched_protocol.json
9. **evidence RunIDs**: cv261207_rq_acceptance_20261107T220632_8271fd92|cv261207_audit_20261104T193649_d1b63492|cv261207_vm26review_20261106T212756_fdf80ed8
10. **supported conclusion**: Versioned matched protocol and CPU validator prepared; no training or gap-closure measurement.
11. **unresolved limitations**: Representative baseline gap, independent admission, augmentation implementation/health canary and measured training budget.
12. **acceptance status**: NOT_EVALUATED

Source files and SHA256:

```json
{"rq4_matched_protocol.json": "40671c4466488ad5c0220b559948aa22fb22ec7a15bcf24b20c8930470a641f6"}
```
