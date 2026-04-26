# Month 2
## **Week 5: Data Augmentation Pipeline**
- **Runtime**: About 15 minutes
- **Code**: Data augmentation script
- **Task**: Generate 9 augmented datasets with spatial (brightness/contrast) and temporal (blur/noise) changes
- **Output**: 9 augmented dataset folders (S1T1–S3T3)
- **Purpose**: Prepare degraded image data for model robustness test
- **Personal View**: Parameters were chosen based on visual effect; no fixed standard, so I cannot fully confirm their rationality.

## **Week 6: Model Evaluation**
- **Runtime**: About 2 hours
- **Code**: week6.py
- **Task**: Evaluate 4 YOLOv8 models across 9 augmented conditions
- **Metrics**: mAP50, mAP50-95, inference time
- **Output**: week6_results.csv
- **Purpose**: Test model robustness and compare performance
- **Personal View**: The script runs stably with complete output, but mAP is low because pre-trained models were not fine-tuned on VisDrone.
