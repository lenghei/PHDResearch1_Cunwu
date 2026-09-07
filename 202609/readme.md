# Work report for September


I recently refined the research plan with more focus on the **research contribution and novelty**. The paper will now concentrate on four main aspects.

1. **3×3 Compound Degradation Evaluation**  
   Continue using the current 3×3 compound degradation experiment to simulate common UAV inspection conditions, including illumination changes, motion blur, and image noise.

2. **640 vs. 1280 Resolution Comparison**  
   Add a comparison between 640 and 1280 input resolutions, with particular attention to small-object detection. Since tiny and small objects are common in UAV aerial inspection, the experiment will investigate whether the performance improvement at higher resolution is mainly concentrated on tiny/small objects rather than only improving overall mAP.

3. **Zero-Shot vs. VisDrone Fine-Tuned Comparison**  
   Compare zero-shot models with VisDrone fine-tuned models to investigate whether fine-tuning improves not only normal detection accuracy, but also model robustness under difficult conditions.

4. **YOLOv8s vs. YOLO26s Comparison**  
   The main model comparison will focus on YOLOv8s and the newer YOLO26s. YOLO26s includes improvements related to small-object detection, so the experiment will verify whether these improvements are effective for UAV small-object detection and robustness under complex conditions. This also makes the paper more meaningful than simply comparing different model sizes within the same YOLO generation.

Currently, I am completing the updated experimental code, including unified training settings, 640/1280 evaluation, nine degradation conditions, genuine mAP50/mAP50:95 calculation, and small-object size statistics. After the code is completed, I will rerun the full set of experiments.



## 1. Research Objective

The current research focuses on the robustness of lightweight object detectors for **small-object detection in UAV aerial imagery**.

Based on practical UAV inspection scenarios, the main challenges considered in this study are:

- Small targets caused by high-altitude or long-distance image acquisition.
- Loss of small-object details at lower input resolutions.
- Overexposure and underexposure under different flight and lighting conditions.
- Motion blur and image noise caused by UAV movement.
- Performance differences between general pretrained models and models fine-tuned specifically on UAV imagery.

The target is to complete the main experiments and manuscript in **September 2026** and prepare the paper for submission to **IEEE Access**.

---

## 2. Main Improvements to the Research Plan

### 2.1 YOLOv8s vs. YOLO26s

The main detector comparison will focus on **YOLOv8s and YOLO26s**.

YOLOv8s is used as a stable and widely used baseline, while YOLO26s represents the latest YOLO generation.

YOLO26 also introduces improvements related to small-object learning. Therefore, the experiment is not simply comparing an old and a new YOLO model. The purpose is to verify whether the newer design actually provides better performance and robustness for small targets in UAV imagery.

---

### 2.2 Pretrained vs. VisDrone Fine-Tuned Models

Each detector will be evaluated in two forms:

- Pretrained model without VisDrone adaptation.
- The same model fine-tuned on VisDrone.

The main comparison will therefore include:

- YOLOv8s Pretrained
- YOLOv8s Fine-Tuned
- YOLO26s Pretrained
- YOLO26s Fine-Tuned

This experiment will show not only how much fine-tuning improves detection accuracy, but also whether domain adaptation improves robustness under difficult UAV conditions.

The analysis will distinguish between:

- **Absolute detection accuracy**
- **Relative robustness retention**

This is important because a fine-tuned model may achieve much higher clean accuracy without necessarily becoming equally more robust to image degradation.

---

### 2.3 Small-Target Resolution Analysis

Input resolutions of **640 and 1280** will be compared.

The main purpose is not only to compare overall mAP, but to determine whether the improvement from 640 to 1280 is mainly concentrated on **tiny and small objects**.

This is directly related to UAV inspection applications, where objects or defects may occupy only a small number of pixels in high-altitude aerial images.

The evaluation will therefore include object-size-based analysis in addition to overall detection performance.

---

### 2.4 Compound UAV Degradation Evaluation

The existing **3×3 degradation matrix** will remain as the main robustness experiment.

It combines three illumination conditions:

- Normal
- Overexposure
- Underexposure

with three motion/image-quality conditions:

- No blur
- Moderate motion blur
- Severe motion blur with noise

This produces nine test conditions.

The purpose is to evaluate detectors under combinations that are closer to practical UAV inspection conditions instead of testing only clean images or a single isolated corruption.

---

## 3. Core Contributions of the Paper

The paper is currently organized around four main contributions:

1. **Industry-Motivated Compound Robustness Protocol**  
   A 3×3 evaluation protocol is designed around illumination variation, motion blur, and image noise commonly encountered in UAV inspection.

2. **Small-Target Resolution Analysis**  
   The study investigates whether increasing the input resolution from 640 to 1280 mainly improves tiny and small target detection rather than only increasing overall mAP.

3. **Domain Adaptation Robustness Analysis**  
   Pretrained and VisDrone fine-tuned models are compared to determine whether domain-specific training improves both detection accuracy and robustness.

4. **Generational Detector Comparison**  
   YOLOv8s and YOLO26s are compared at a similar model scale to evaluate whether the latest small-target-aware detector design provides measurable advantages in UAV imagery.

---

## 4. Current Work

The current task is to reorganize and optimize the experimental code .

The main code improvements include:

- Unified training and evaluation configuration.
- YOLOv8s and YOLO26s support.
- Pretrained and fine-tuned model comparison.
- Automatic evaluation at 640 and 1280 resolutions.
- Automatic execution of all nine degradation conditions.
- Genuine mAP@0.5 and mAP@0.5:0.95 calculation.
- Tiny/small/medium/large object analysis.
- Per-class performance analysis.
- Robustness retention calculation.
- Standardized CSV/JSON result export.
- Reproducible experiment settings and logging.

After the code is finalized, the next step is to run the complete experiment matrix and update the Results and Discussion sections of the paper.

---

## 5. Target for September 2026

The planned work for September is:

1. Finalize the experimental code.
2. Fine-tune YOLOv8s and YOLO26s on VisDrone.
3. Complete the pretrained vs. fine-tuned comparison.
4. Complete the 640 vs. 1280 small-object analysis.
5. Run the full 3×3 robustness evaluation.
6. Generate final tables, figures, and statistical analysis.
7. Complete the manuscript.
8. Prepare the paper for **IEEE Access submission**.

The main goal is to keep the study focused and reproducible while strengthening the paper through **small-object analysis, resolution effects, domain adaptation, and evaluation of the latest detector generation**.

---


