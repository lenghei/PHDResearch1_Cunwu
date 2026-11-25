# YOLOv8s checkpoints

| File in `local/` | Use |
| --- | --- |
| `yolov8s_coco_init.pt` | COCO initialization for the historical training run |
| `yolov8s_train640_best.pt` | Week 1 baseline and all five frozen detector configurations |
| `yolov8s_train640_last.pt` | Final checkpoint retained with the training run |

Training used VisDrone train (6,471 images), val (548 images), 640 input, 100 epochs, seed 42, batch 4 and Ultralytics 8.4.61. The trainer exported the best validation EMA. Its saved fitness (0.22795) matches the unique maximum logged at epoch 76. The final checkpoint is from epoch 100. The separate development baseline evaluates the selected checkpoint at 1280 FP32 with the repository COCO protocol. [Selection receipt](../review/checkpoint_selection.json) · [Reproduction commands](../../../../experiments/uav_01_baseline/README.md).

The weight files in `local/` are Git-ignored hard links to local artifacts. Treat them as read-only. [Checkpoint hashes and source run](checkpoints.csv).
