<!-- development-update-20261108 -->
Latest update: [new development measurements](DEVELOPMENT_EVIDENCE.md), `cv261207_development_delivery_20261108T194939_4be5d3c4`. The earlier acceptance below is retained as a dated historical state.

# RQ1–RQ4 evidence acceptance

This is an evidence and execution report, not a manuscript results section.
Student and supervisor scientific review remains required.

RunID: `cv261207_rq_acceptance_20261107T220632_8271fd92`. Formal test-dev pairs: **0/500**. Admission remains **DATASET_BLOCKED**.
No historical experiment, original UVL, dataset or checkpoint was overwritten.

| RQ | Status | Actual evidence | Remaining blocker |
|---|---|---|---|
| RQ1 | PARTIAL | Logical models validated; physical calibration and full detector executability remain incomplete. | Unsupported features and uncalibrated scenario constraints. |
| RQ2 | PARTIAL | 279 unique historical responses; verified VM26 three-way effects; descriptive tree/stump/forest fits only. | Shared image population; no image/configuration-disjoint held-out scores. |
| RQ3 | PARTIAL | 2/4 selected VM26 contexts have tiny-AP-supported failure across six measured endpoints at tau=0.10. Other contexts are unresolved. | No compatible complete latency, random population sample or independently validated gap predictor. |
| RQ4 | NOT_EVALUATED | Versioned matched protocol and CPU validator prepared; no training or gap-closure measurement. | Representative baseline gap, independent admission, augmentation implementation/health canary and measured training budget. |

## RQ1 boundaries

FM_P: 25 leaves, five constraints, 46,368 valid configurations and no dead leaves.
FM_D: 34 leaves, four constraints, 635,904 valid configurations, 844 training variants, SAT and no dead leaves.
The FM_D count is guarded exact factorization checked against independent structural enumeration, not a BDD/#SAT count.
All numeric FM_P attributes and configuration IDs are checked. The original FM_D v1 has no numeric attribute blocks;
per-checkpoint measurements are external evidence, and generic parameter counts are not substituted.
The compatibility matrix covers every FM_D leaf. Feature support is not support for every combination.
The preserved SAHI/TTA exclusion and physical FM_P restrictions remain study assumptions.
RollingShutter is generated in renderer tests but excluded from evaluation until GT/ignore geometry is validated.

## RQ2 data and inference

VM09 has 243 configurations; VM26 has 54 with 18 reused None-JPEG configurations. Their union has 279 unique responses.
All share the same 300 ds04 development images. VM09 CSV and cell result hashes were checked; no new raw rematching of all 243 is claimed.
The accepted VM26 raw/GT replay and 200-draw bootstrap evidence is reused without rerunning it.
JPEG Q85 is outside the original FM_P; it remains a measured numeric factor. Historical noise variance 15/30 is not UVL sigma 15/30.
Feature analogues are not asserted to be pixel-identical to the newer UVL renderer.
Tree, rule-stump and forest fits are descriptive, in-sample diagnostics. Fitting seeds are not detector training seeds.
Balanced accuracy, macro F1, failure precision/recall and held-out confusion remain null: ML_INSUFFICIENT_EVIDENCE.
The learning admission now checks both degradation and exact image membership before accepting a fold.

## RQ3 uncertainty and missing outcomes

Three policies remain separate: exact identity, conservative source-supported proxy and supervisor provisional proxy.
For CV261102 the coverage counts are 1/3/18; the three older version-8.4.33 checkpoints have conservative coverage 1 under the stricter source-version evidence policy, not proof that brightness variation was absent.
The AP predicate uses the original-coordinate AP50:95 intervals; a verified AP failure is sufficient to fail AP AND latency even when latency is unavailable.
A PASS requires both a sufficient AP lower bound and compatible measured latency. Unknown outcomes are not failures.
The development subset analysis uses a newly declared post-hoc simultaneous family of 72 endpoint/size AP intervals from the saved 200 paired image draws.
It is separate from the original VM26 36-contrast DID family. There is no flight-cluster or training-seed uncertainty.
Only 640/1280 and JPEG None/Q50 have these intervals. Medium/large AP points exist, but their intervals are NOT_EVALUATED.
The four purposive contexts cannot estimate the 46,368-configuration population. No full-space gap or learned UVL constraint is claimed.
Confusion counts are shown with missing outcomes. Complete-case agreement is selected-failure diagnostics; specificity may be undefined.

## Sampling and future training

The frozen v1 500-pair manifest is unchanged. Cross-architecture v2 is a separate 900-pair plan:
the same uniform 100 of 26,496 geometry-eligible FM_P settings crossed with nine purposively bound checkpoint/resolution settings.
It is not a uniform sample of all 635,904 detector configurations. Albu activation, independent admission and full-pair cost are unresolved.
The RQ4 JSON freezes proposed matched continuation and ablation arms. Its validator runs on CPU.
A new augmentation training executor and health canary are not implemented or validated for this protocol; no training is marked complete.
Future execution requires one measured pair and identical repeat, a one-seed training feasibility pilot and an explicit bounded budget.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -B main/experiments/cv261207/rq_evidence.py
.\.venv\Scripts\python.exe -B main/experiments/cv261207/rq_evidence.py --live-demo
.\.venv\Scripts\python.exe -B main/experiments/cv261207/rq4_protocol.py --check <RunDir>/rq4_matched_protocol.json
.\.venv\Scripts\python.exe -B -m unittest discover -s main/experiments/cv261207/tests -v
```

The default reuses saved GPU evidence. --live-demo authorizes one bounded 120-second validation-image prediction, not a formal pair or training.
Both modes create a new RunID and verify inputs. See inputs.json, stage timings, per-figure CSVs and SHA256SUMS.txt.

The independent validation plan is linked through the accepted history review. No incompatible external dataset is substituted.
