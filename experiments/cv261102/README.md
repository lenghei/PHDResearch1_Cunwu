# CV261102 — Fresh YOLOv8s training and validation

This is an entirely new experiment, not a comparison with old ck05 results or an earlier CSV.

The runner fine-tunes a local COCO-pretrained yolov8s.pt on full VisDrone native-ten training data, records each completed training epoch using real callbacks, evaluates the complete 548-image VisDrone2019-DET-val split with the repository's original-pixel COCO protocol, replays COCO metrics, and exports five CSV tables.

## Inputs
- Original VisDrone2019-DET-val annotated ZIP (548 images).
- Local pretrained yolov8s.pt.
- Exactly one of: native-ten Ultralytics YAML via --data, or original annotated VisDrone2019-DET-train ZIP via --train-zip. The latter converts images and labels locally, omitting ignore/other regions as training targets.

## Run (from the outer uavgap directory)

Preflight:

    & .\.venv\Scripts\python.exe -B main/experiments/cv261102/run.py --data "E:\path\to\visdrone10.yaml" --val-zip "E:\path\to\VisDrone2019-DET-val.zip" --weights "E:\path\to\yolov8s.pt" --preflight

Full fresh run (real GPU training and evaluation):

    & .\.venv\Scripts\python.exe -B main/experiments/cv261102/run.py --data "E:\path\to\visdrone10.yaml" --val-zip "E:\path\to\VisDrone2019-DET-val.zip" --weights "E:\path\to\yolov8s.pt" --epochs 100 --batch 4

Use --train-zip "E:\path\to\VisDrone2019-DET-train.zip" instead of --data if you want raw annotation conversion.

Defaults: 100 epochs, training 640px, eval 1280px, training batch 4, eval batch 2, seed 42, AMP training and FP32 evaluation. Reduce --batch to 2 if RTX 4060 VRAM is insufficient. Record all actual settings.

## New local Run layout

datasets/local_runs/cv261102/cv261102_full_<UTC>_<id>/
- prepared_dataset/ (if converted; private)
- training/yolov8s/weights/best.pt and last.pt, results.csv, args.yaml
- training/epoch_times.csv: measured epoch timing
- validation/: raw predictions, GT, COCO metrics, per-image trace, summary
- csv/: metrics_overall.csv, metrics_by_area.csv, metrics_by_class.csv, evaluation_timing.csv, training_epoch_times.csv
- manifest.json: checksums, status, Git commit, configuration, metrics

These outputs are local only. Do not commit datasets, raw predictions, or model weights.

Caution: VisDrone val is also used for training-time checkpoint selection. Thus the 548-image outcome is development validation, not an independent test-dev result.

Image source: [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset)
