# Week 1 · YOLOv8s baseline

CV261102 trained YOLOv8s on 6,471 VisDrone images for 100 epochs at 640. Evaluation at 1280 used all 548 validation images and ten VisDrone classes. AP50 was 50.4169%, AP50:95 was 29.6244%, and AR50:95 was 53.1723%.

- [CPU replay and source hashes](review/replay.json)
- [Original best checkpoint selection](review/checkpoint_selection.json)
- [Metrics by class](review/metrics_by_class.csv) and [by area](review/metrics_by_area.csv)
- [Checkpoint files and usage](models/README.md)
- [Historical training and evaluation times](uav_01_baseline_20261124T154238_4a518050/snapshot.json)
- [Per-epoch timing plot](uav_01_baseline_20261124T154238_4a518050/baseline_epoch_timing.png)
- [File checksums](review/manifest.json)
