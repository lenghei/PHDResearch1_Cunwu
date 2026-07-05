# Week 12 Analysis Report: UAV Aerial Robustness & Cross-Domain Evaluation

## 📌 Project Overview
This repository contains the Week 8 benchmarking pipeline and robustness analysis for lightweight object detection architectures (YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l) evaluated on the **VisDrone dataset**. The core focus is evaluating model stability under compound environmental degradations across **9 distinct conditions** (Spatial-Temporal Matrix S1-S3, T1-T3).

* **Code Run Time**: 10 minutes
* **Execution Engine**: PyTorch + Ultralytics Pipeline

---

## 🗺️ Cross-Domain Class Mapping Logic
Directly evaluating standard COCO-pretrained weights on VisDrone creates severe label mismatches (e.g., the infamous car vs. van bounding box overlapping conflict). To build a scientifically valid zero-shot baseline without generating massive False Positives, a **Ground Truth Aggregation** approach was implemented:

### Ground Truth Translation (`VIS_TO_EVAL`)
VisDrone's 10 localized classes are aggregated into 6 unified macro-evaluation classes:
* `pedestrian` (0), `person` (1) $\rightarrow$ **person**
* `car` (2), `van` (3) $\rightarrow$ **car** (Resolves the spatial overlapping and copy-paste anomaly)
* `bus` (4) $\rightarrow$ **bus**
* `truck` (5) $\rightarrow$ **truck**
* `motorcycle` (6), `awning-tricycle` (8) $\rightarrow$ **motorcycle**
* `bicycle` (7), `tricycle` (9) $\rightarrow$ **bicycle**

### Prediction Translation (`COCO_TO_EVAL`)
Raw COCO outputs from the models are mapped to the identical 6 macro-classes (`person` $\rightarrow$ 0, `car` $\rightarrow$ 1, `motorcycle` $\rightarrow$ 4, etc.) to ensure a 1-to-1 comparison matrix.

---

## 👁️ Perspective Gap: UAV Aerial vs. COCO Flat Perspective
The baseline numbers exhibit a huge domain gap. Standard COCO models are trained on horizontal, eye-level perspectives where objects occupy substantial frame percentages. 
When deployed directly onto UAV aerial imagery:
1. **Geometric Transformation**: A top-down bird's-eye view changes standard profiles (e.g., cars look like flat rectangles; people look like small dots from above).
2. **Extreme Scale Downsampling**: Targets frequently drop below $16 \times 16$ pixels, causing critical spatial details to vanish during the backbone's multi-layer downsampling.

### 📉 The Bus & Bicycle Bottleneck
Among the aggregated classes, **bus** and **bicycle** exhibit dismal numerical performance:
* **Bicycles**: These represent the extreme end of tiny object degradation. At high altitudes, a bicycle's spatial area is almost completely diluted into the background noise, leaving zero distinct zero-shot geometric features for the pre-trained COCO detection head.
* **Buses**: While larger in physical size, buses suffer from massive aspect ratio distortions from a top-down perspective. Furthermore, they are highly sparse (long-tail distribution) compared to standard cars, leading to severe anchor mismatches.

---

## 📊 Personal Analysis: The "Negative Degradation" Phenomenon
An intriguing, counter-intuitive phenomenon was discovered during the numerical evaluation: **some severe degradation conditions resulted in performance gains rather than degradation.**

### 1. Condition Impact Ranking (From Strongest Drop to Highest Gain)
* **S1T2**: Positive degradation $\rightarrow$ **Largest performance drop** (Strongest negative influence on the model).
* **S1T3, S3T1, S2T1**: Negative degradation values $\rightarrow$ **Clear performance improvement**.
* **S3T3**: Largest negative degradation value $\rightarrow$ **Strongest performance improvement**.

### 2. Result Verification & Reasoning
$$\text{Negative Degradation Value} = \text{mAP higher than the clean baseline (S1T1)}$$

This patterns fully supports my physical reasoning: 
When images undergo specific illumination shifts (becoming significantly brighter or darker), **redundant background textures, environmental clutter, and pixel-level noise are heavily compressed or filtered out**. This creates a cleaner background context, making the silhouettes of micro-objects stand out more prominently. The simplified contrast stabilizes bounding box regression, leading to a higher mAP than the "clean" S1T1 baseline and validating unique model robustness behaviors.

---

## 📦 Completion Status & Deliverables
* **Metrics Computed**: Relative degradation percentages across all 9 matrix variations.
* **Model Benchmark**: Comprehensive comparison between YOLOv8n, v8s, v8m, and v8l.
* **Visualizations Saved**: 
  * 3x3 per-class isolated condition matrices.
  * Structural heatmaps highlighting degradation trends.
  * Comparative performance bar charts for each architecture.
