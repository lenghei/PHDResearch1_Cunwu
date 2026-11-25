# Week 3 · Detector settings and static coverage

- [Latency, memory, parameters and GFLOPs](review/latency_memory.csv) for five YOLOv8s settings on the same 32 VisDrone val images, two passes per setting on an RTX 4060 Laptop GPU.
- [Static coverage rules by degradation](review/static_rules_by_leaf.csv) and [FM_D and SAHI/TTA results](review/model_and_policies.json).
- [Original FM_D](../../../experiments/uav_03_fmd_static_gap/fm_d.uvl) and [scoped SAHI/TTA variant](../../../experiments/uav_03_fmd_static_gap/fm_d_sahi_tta.uvl).

The original FM_D has 635,904 logical settings and 844 training variants. The scoped variant adds one tested SAHI/TTA combination. Static rules cover 1, 3 or 18 of the 46,368 FM_P settings under their respective assumptions.

Timing starts either before prediction or before reading the verified cached PNG and ends after box extraction for the latter. Peak memory is PyTorch allocated memory. The sample is for resource profiling. The [earlier test-dev timing](review/frozen_detector_attributes.csv) uses a separate evaluation protocol.
