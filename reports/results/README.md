# Verified results

| Experiment | Core evidence |
| --- | --- |
| [Week 1 · Baseline](uav_01_baseline/) | 548-image VisDrone val scores, 100 epoch timings and checkpoint selection. |
| [Week 2 · FM_P](uav_02_fmp/) | Original-plan UVL comparison, full catalog software test and one complete real-image render. |
| [Week 3 · FM_D](uav_03_fmd_static_gap/) | Static coverage assumptions, detector operation counts and measured resource limits. |
| [Week 4 · Sampling](uav_04_sampling/) | Frozen P and D IDs, planned pairs and coverage. |
| [Week 5 · Evaluation](uav_05_evaluation/) | Formal admission status and one separate exploratory test-dev replay. |
| [Week 6 · Gap](uav_06_gap_analysis/) | Formal status and development threshold sensitivity. |
| [VM48 · Targeted augmentation](vm48_formal_rq4/) | Transfer and matched Motion pilot results, 10-class metrics and figures. |

The [dashboard](https://uavgap.streamlit.app/) reads verified snapshots. For Weeks 1–5 select the `review` entry to see the latest verified tables. The CSV, JSON and figures are also available directly in each result directory. Use the SHA256 values in each `manifest.json` to verify the files. Full prediction replay uses the locally archived inputs.
