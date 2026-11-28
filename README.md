# 无人机-GAP

UAV-GAP studies how image degradation affects object detection in drone imagery. It uses two UVL feature models to describe image degradations (FM_P) and YOLO detector settings (FM_D). On VisDrone, the experiments compare detection accuracy and inference latency across these settings, with a focus on tiny objects. The goal is to identify conditions where the tested detector settings fail to meet both requirements and assess whether targeted training augmentation helps.

Online dashboard: [**https://uavgap.streamlit.app**](https://uavgap.streamlit.app) · [Results and downloads](reports/results/) · [Experiment code](experiments/)

| Experiment | Result |
| --- | --- |
| [YOLOv8s baseline](reports/results/uav_01_baseline/) | VisDrone val, 548 images. AP50 **50.42%**, AP50:95 **29.62%**. |
| [Image degradation](reports/results/uav_02_fmp/) | **46,368** legal configurations tested with a synthetic fixture. A selected configuration was rendered on all 548 val images. [Full CSV](reports/results/uav_02_fmp/configuration_validation.csv) |
| [Detector settings and static Gap](reports/results/uav_03_fmd_static_gap/) | **635,904** logical detector settings. The three training-coverage rules cover **1, 3 or 18** degradation configurations. |
| [Configuration sampling](reports/results/uav_04_sampling/) | Frozen design of **100 degradation settings × 5 detector settings**. |
| [Detector evaluation](reports/results/uav_05_evaluation/) | One exploratory test-dev pair and its repeat, each on **1,610 images**. |
| [Empirical Gap](reports/results/uav_06_gap_analysis/) | Formal evaluation **500 pairs**.  |
| [Targeted augmentation](reports/results/vm48_formal_rq4/) | Apply targeted training. |

Each experiment folder explains how to reproduce its results. The dashboard opens saved summaries. Full reproduction uses the local VisDrone inputs, model weights and prediction archives.

[Supporting experiments](experiments/experiments_test/) · [Progress notes](reports/notes/)
