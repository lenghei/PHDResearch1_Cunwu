# Experiment Process

1. **Prepare VisDrone data**
   - Convert/validate VisDrone annotations and preserve ignored regions.
   - Keep original image coordinates for target-size grouping and final evaluation.

2. **Generate controlled degradation data**
   - Build nine S×T conditions from the same validation images.
   - S1/S2/S3: normal, overexposure, underexposure.
   - T1/T2/T3: no blur, motion blur k=7, motion blur k=21 + Gaussian noise.
   - Keep raw CLEAN images as a separate reference.

3. **Train detector checkpoints**
   - Small cohort: YOLOv8s / YOLO11s / YOLO26s at 1280.
   - Nano cohort: YOLOv8n / YOLO11n / YOLO26n at 640.
   - 100 epochs, one fixed seed, shared high-level schedule.

4. **Run inference and metric evaluation**
   - Evaluate clean and nine generated conditions.
   - Fine-tuned models: native VisDrone-10 plus common-six evaluation.
   - Zero-shot models: common-six evaluation only.
   - Compute genuine COCO AP50:95, AP50, AR, size and per-class results.

5. **Controlled resolution experiment**
   - Reuse the same YOLOv8s-FT and YOLO26s-FT checkpoint hashes.
   - Evaluate each at 640 and 1280 without retraining.
   - Compare all / tiny / small / medium / large metrics.

6. **Robustness analysis**
   - Mean corrupted AP = arithmetic mean of the eight non-control matrix cells.
   - Retention = mean corrupted AP / raw CLEAN AP.
   - Record the minimum-condition AP and condition identity.

7. **Uncertainty and consistency checks**
   - Use saved predictions/ground truth for image-bootstrap checks and paired comparisons.
   - Keep numerical provenance linking headline results to source metric files.

8. **Generate summary tables and figures**
   - Main model comparison, transfer comparison, full 3×3 matrix, size analysis, resolution analysis, per-class results, and supporting figures.
