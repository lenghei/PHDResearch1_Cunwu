# Week 6 - Ultralytics Standard Path: datasets/xxx

import os
import pandas as pd
from ultralytics import YOLO

# ====================== Models ======================
MODELS = [
    ("YOLOv8n", "yolov8n.pt"),
    ("YOLOv8s", "yolov8s.pt"),
    ("YOLOv8m", "yolov8m.pt"),
    ("YOLOv8l", "yolov8l.pt"),
]

# ====================== 9 Conditions ======================
CONDITIONS = [
    "S1T1", "S1T2", "S1T3",
    "S2T1", "S2T2", "S2T3",
    "S3T1", "S3T2", "S3T3"
]

# ====================== Ultralytics Path Configuration ======================
yaml_content = """
path: .
train: VisDrone/VisDrone2019-DET-train/images
val: VisDrone/VisDrone2019-DET-val/images
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

# ====================== Validation ======================
results = []

for model_name, weight in MODELS:
    print("\n" + "=" * 50)
    print(f"Evaluating: {model_name}")
    print("=" * 50)

    model = YOLO(weight)

    for cond in CONDITIONS:
        print(f"\n-> {model_name} | {cond}")

        metrics = model.val(
            data="visdrone.yaml",
            imgsz=640,
            batch=1,
            device="cpu",
            verbose=False,
            plots=False,
        )

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

df = pd.DataFrame(results)
df.to_csv("week6_results.csv", index=False)
print("\nWEEK 6 DONE!")
