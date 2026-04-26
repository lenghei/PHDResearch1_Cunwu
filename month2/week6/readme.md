# Code Explanation for week6.py
**Code Runtime**: Approximately 2 hours

**Personal Statement**: This script was debugged and revised **6 times** before successful execution. The final output file was generated correctly. However, there is a limitation: the parameters for the 9 experimental conditions (S1T1–S3T3) were selected based on visual image quality, since no official or fixed standard parameters were provided. I cannot fully confirm whether these parameters are fully reasonable or optimal.

## 1. Library Import
```python
import os
import pandas as pd
from ultralytics import YOLO
```
- Import necessary tools: file system support, data table processing, and YOLOv8 model API.

## 2. Model & Condition Definition
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
- Define **4 YOLOv8 models** (nano, small, medium, large).
- Define **9 test conditions** for augmented VisDrone datasets.

## 3. Main Evaluation Logic
```python
results = []
for model_name, weight in MODELS:
    print("\n" + "=" * 50)
    print(f"Evaluating: {model_name}")
    print("=" * 50)
    model = YOLO(weight)
```
- Initialize each model and start evaluation.

```python
for cond in CONDITIONS:
    print(f"\n-> {model_name} | {cond}")
    yaml_content = f"""
path: .
train: visdrone_val_aug_9conditions/{cond}/images
val: visdrone_val_aug_9conditions/{cond}/images
nc: 10
names:
  0: pedestrian
  1: person
  2: car
  3: van
  4: bus
  5: truck
  6: motor
  7: bicycle
  8: awning-tricycle
  9: tricycle
"""
    with open("visdrone.yaml", "w", encoding="utf-8") as f:
        f.write(yaml_content)
```
- Dynamically generate a YAML dataset config for **each condition** to load the corresponding image folder.

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
- Run validation with fixed settings: image size 640, batch size 1, CPU inference, no extra logs or figures.

```python
mAP50 = round(metrics.box.map50, 4)
mAP5095 = round(metrics.box.map, 4)
speed = round(metrics.speed["inference"] / 1000, 4)

print(f"   mAP50: {mAP50} | mAP50-95: {mAP5095} | time: {speed}s")

results.append({
    "model": model_name,
    "condition": cond,
    "mAP50": mAP50,
    "mAP50_95": mAP5095,
    "time_per_image": speed
})
```
- Extract core indicators: **mAP50, mAP50–95, inference time** and store them in a list.

```python
df = pd.DataFrame(results)
df.to_csv("week6_results.csv", index=False)
print("\nWEEK 6 DONE!")
```
- Save all results to `week6_results.csv` and print completion message.

## 4. Summary
This script automatically evaluates 4 models across 9 dataset conditions, outputs complete performance metrics, and records all results in a CSV file.
