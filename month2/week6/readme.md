# Explanation of `week6.py`
**Code Runtime**: 2 hours

**Personal Statement**: This script was successfully debugged and revised **6 times** before full execution. The output results file was generated correctly. However, I have one limitation: the parameters for the 9 experimental conditions were determined based on visual image effects, since no official fixed standards were provided. I cannot fully confirm whether these parameter settings are fully optimal or reasonable.

## Overall Purpose
This script evaluates the **robustness** of four YOLOv8 object detection models on 9 augmented versions of the VisDrone validation dataset. The 9 conditions simulate real-world drone imaging challenges by combining **spatial variations (brightness/contrast)** and **temporal variations (blur/noise)**. The goal is to measure detection accuracy (mAP50, mAP50–95) and inference speed across all conditions, then export all results to a CSV file for analysis.

## Step 1 – Import Required Libraries
```python
import os
import pandas as pd
from ultralytics import YOLO
```
- Imports core tools: file system support, data processing, and the YOLOv8 model framework for validation and metric calculation.

## Step 2 – Define Models and 9 Evaluation Conditions
```python
MODELS = [
    ("YOLOv8n", "yolov8n.pt"),
    ("YOLOv8s", "yolov8s.pt"),
    ("YOLOv8m", "yolov8m.pt"),
    ("YOLOv8l", "yolov8l.pt"),
]

CONDITIONS = [
    "S1T1", "S1T2", "S1T3",
    "S2T1", "S2T2", "S2T3",
    "S3T1", "S3T2", "S3T3"
]
```
- **Models**: 4 YOLOv8 variants (nano, small, medium, large) for comprehensive performance comparison.
- **Conditions**: 9 augmented dataset folders corresponding to brightness/contrast and blur/noise combinations.

## Step 3 – Dynamic Dataset Configuration
For each condition, the script automatically generates a `visdrone.yaml` file to define:
- Path to the condition‑specific image folder
- Number of object classes (10 for VisDrone)
- Class names (pedestrian, car, van, bus, etc.)

This ensures the model validates on the **correct augmented dataset** for each test condition.

## Step 4 – Model Validation & Metric Extraction
```python
metrics = model.val(
    data="visdrone.yaml",
    imgsz=640,
    batch=1,
    device="cpu",
    verbose=False,
    plots=False,
)
```
- Runs evaluation on the current augmented dataset
- Uses fixed image size (640), batch size (1), and CPU
- Computes key performance indicators:
  - `mAP50`: detection accuracy at IoU = 0.5
  - `mAP50–95`: overall detection precision
  - `inference time`: speed per image

## Step 5 – Save Results to CSV
```python
df = pd.DataFrame(results)
df.to_csv("week6_results.csv", index=False)
```
- Stores all model‑condition performance data into a structured CSV file
- Enables easy comparison, visualization, and further analysis

## Summary
This script provides a complete, automated evaluation pipeline for testing model robustness under **9 real-world degraded imaging conditions**. It validates four YOLOv8 models, records accuracy and speed metrics, and outputs standardized results. 

