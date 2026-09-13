# Research Summary

## Research objective
This study examines how lightweight object detectors behave in UAV aerial imagery when target size, input resolution, domain adaptation, and image degradation are considered together.

## Dataset and models
- Dataset: VisDrone2019-DET (6,471 training images; 548 validation images used for the controlled evaluation).
- Primary comparison: YOLOv8s vs YOLO26s.
- Supporting models: YOLO11s, YOLOv8n, YOLO11n, YOLO26n, and released LEAF-YOLO-N.
- Fine-tuning: 100 epochs for the six YOLO configurations under a shared high-level training schedule.

## Four experimental questions
1. **Compound degradation:** a fixed 3×3 matrix crosses illumination (normal / overexposed / underexposed) with motion/image-quality degradation (none / moderate motion blur / severe blur + Gaussian noise).
2. **Resolution and object size:** the same fine-tuned YOLOv8s and YOLO26s checkpoints are evaluated at 640 and 1280 without retraining, with fixed tiny/small/medium/large object groups.
3. **Domain adaptation:** COCO-pretrained zero-shot and VisDrone fine-tuned checkpoints are compared on the same common-six label space.
4. **Detector generation:** YOLOv8s and YOLO26s are compared under shared training and evaluation settings; other models are supporting evidence.

## Evaluation
- Genuine COCO-style AP50:95 is the main AP metric; AP50 is reported separately.
- Size-specific AP/AR, per-class AP, mean corrupted AP, minimum-condition AP, and robustness retention are recorded.
- Native VisDrone ten-class evaluation is kept separate from the common-six transfer evaluation.

## Main observations
- At 1280, YOLO26s reaches 35.24% clean AP and 17.49% mean corrupted AP; YOLOv8s reaches 34.80% and 17.36%.
- The worst cell for both primary 1280 models is underexposure combined with severe blur/noise (S3T3).
- For unchanged primary checkpoints, moving from 640 to 1280 increases clean AP and especially benefits the small-object AP / tiny-object AR measurements, while large-object AP decreases.
- Fine-tuning strongly improves absolute clean and corrupted AP on the matched common-six task, but proportional robustness retention does not consistently increase.
- The newer model shows a clearer gain on clean tiny-object AP than on mean corrupted AP, so clean accuracy and robustness should be reported separately.

## Scope
This is a controlled robustness and diagnostic study on VisDrone. Synthetic degradations are repeatable stress tests rather than a calibrated physical sensor model, and the reported scores are not an official VisDrone leaderboard submission.
