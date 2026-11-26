# Week 3 · Detector settings and static coverage

- [FM_D with measured attributes](../../../experiments/uav_03_fmd_static_gap/fm_d_measured.uvl). The UVL includes 33 profiles and links each record to its source table and checkpoint SHA256.
- [Static Gap rule arguments](static_gap_rules.csv). One row per FM_P leaf with numeric levels, source references, coverage decisions and limitations.

- [11-checkpoint FM_D roster](reproduce_20261125T142040_7a322c8d/checkpoint_roster.csv), [COCO zero-shot common-six AP](reproduce_20261125T142040_7a322c8d/zeroshot_metrics.csv), and [10 additional inference profiles](reproduce_20261125T142040_7a322c8d/inference_options.csv). Each evaluation or profile uses all 548 development images. INT8 remains unmeasured.

- [Eight-checkpoint performance](reproduce_20261125T134035_61c7c6f2/validation_metrics.csv), [training records](reproduce_20261125T134035_61c7c6f2/checkpoint_training.csv), [per-class AP](reproduce_20261125T134035_61c7c6f2/validation_per_class.csv), and [comparison chart](reproduce_20261125T134035_61c7c6f2/checkpoint_validation.png). Six Ultralytics models and two LEAF models, 16 cells on the same 548 images. LEAF-S is an external control.

- [Full validation profiles](review/full_validation_profiles.csv), [per-pass timings](review/full_validation_passes.csv), and [latency and memory chart](review/full_validation_profiles.png). Five settings, 548 Normal images, three passes per setting on an RTX 4060 Laptop GPU.
- [32-image pilot](review/latency_memory.csv). Two passes per setting on a composite degradation.
- [Six-checkpoint comparison](reproduce_20261125T125315_5300a35e/validation_metrics.csv), [per-class AP](reproduce_20261125T125315_5300a35e/validation_per_class.csv), and [AP chart](reproduce_20261125T125315_5300a35e/checkpoint_validation.png). Native-ten FP32 on 548 Normal validation images at 640 and 1280. Ten cells come from the first run and two from the Full20 follow-up.
- [Checkpoint pool](review/checkpoint_pool.csv), [CPU parameters and GFLOPs](review/checkpoint_compute.csv), and [historical clean AP/AR](review/historical_clean.csv). The six verified checkpoints use the dataset and training conditions listed in the tables.
- [Static coverage rules by degradation](review/static_rules_by_leaf.csv) and [FM_D and SAHI/TTA results](review/model_and_policies.json).
- [Original FM_D](../../../experiments/uav_03_fmd_static_gap/fm_d.uvl) and [scoped SAHI/TTA variant](../../../experiments/uav_03_fmd_static_gap/fm_d_sahi_tta.uvl).

The original FM_D has 635,904 logical settings and 844 training variants. The scoped variant adds one tested SAHI/TTA combination. Static rules cover 1, 3 or 18 of the 46,368 FM_P settings under their respective assumptions.

Detector timing covers preprocessing, inference, NMS and synchronization. Cached pipeline timing includes PNG reads, integrity checks, decoding and box extraction. The full profile includes per-image mean, median and P95 latency plus peak allocated and reserved memory. The [earlier test-dev timing](review/frozen_detector_attributes.csv) uses a separate evaluation protocol.
