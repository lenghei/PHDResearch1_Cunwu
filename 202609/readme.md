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

# 中文理解版

## 1. 当前研究目标

目前论文主要研究**无人机航拍图像中轻量目标检测模型的小目标检测和复杂环境鲁棒性**。

结合无人机巡检实际场景，重点考虑：

- 高空或远距离拍摄造成的小目标问题；
- 低分辨率输入导致小目标细节丢失；
- 过曝和欠曝；
- 无人机运动造成的运动模糊和噪声；
- 通用预训练模型与 VisDrone 专门训练模型之间的差异。

计划在 **2026 年 9 月完成主要实验和论文，并准备投稿 IEEE Access**。

---

## 2. 研究计划的主要改进

### YOLOv8s 与 YOLO26s 对比

论文主要选择 **YOLOv8s 和 YOLO26s**。

YOLOv8s 作为比较成熟的基线，YOLO26s 作为目前最新一代 YOLO。

选择 YOLO26s 不只是因为它比较新，还因为它增加了针对小目标学习的设计。因此实验主要想验证：

**这些新的设计在无人机小目标场景中是否真的能够带来提升。**

---

### 预训练模型与 Fine-Tuned 模型对比

两个模型都分别测试：

- 原始预训练模型；
- 在 VisDrone 上 Fine-Tuned 后的模型。

最终主要比较：

- YOLOv8s Pretrained
- YOLOv8s Fine-Tuned
- YOLO26s Pretrained
- YOLO26s Fine-Tuned

这样不仅可以看到 Fine-Tuning 后精度提升多少，还可以进一步研究：

**训练后的模型在复杂光照、运动模糊和噪声环境中是不是也更加鲁棒。**

论文会把：

**绝对检测精度**

和

**相对于正常环境的性能保持率**

分开分析。

---

### 640 与 1280 小目标分析

增加 **640 和 1280 输入分辨率对比**。

重点不是简单证明 1280 的整体 mAP 更高，而是分析：

**1280 的提升是不是主要来自 tiny/small objects。**

这与无人机巡检比较符合，因为航拍中的目标或者缺陷往往只占很少的像素。

---

### 保留 3×3 复合扰动实验

继续使用目前的 3×3 实验：

三种光照：

- 正常
- 过曝
- 欠曝

三种运动/图像质量：

- 无模糊
- 中等运动模糊
- 严重运动模糊 + 噪声

共 9 种情况。

主要目的是模拟更加接近真实无人机巡检的复杂情况，而不是只在正常图片或者单一扰动条件下测试。

---

## 3. 论文的四个核心贡献

1. **基于实际 UAV 巡检的复合鲁棒性评价方法**  
   使用光照、运动模糊和噪声建立 3×3 测试方案。

2. **小目标分辨率分析**  
   研究 640 提高到 1280 后，提升是否主要集中在 tiny/small objects。

3. **Domain Adaptation 鲁棒性分析**  
   对比预训练模型和 VisDrone Fine-Tuned 模型，分析 Fine-Tuning 对正常精度和复杂环境鲁棒性的影响。

4. **新旧两代检测模型对比**  
   对比 YOLOv8s 和 YOLO26s，验证新一代针对小目标的设计在 UAV 场景中是否真正有效。

---

## 4. 目前正在进行的工作

目前正在 **优化和重新整理实验代码**，主要包括：

- 统一训练和测试配置；
- 支持 YOLOv8s 和 YOLO26s；
- 支持 Pretrained / Fine-Tuned 对比；
- 自动测试 640 / 1280；
- 自动完成 9 种扰动条件；
- 使用真实 mAP50 和 mAP50:95；
- 增加 tiny/small/medium/large 分析；
- 增加各类别检测结果；
- 计算 robustness retention；
- 自动保存 CSV / JSON；
- 统一随机种子和实验日志，保证实验可重复。

代码完成以后，就开始重新跑完整实验，并根据最终数据更新论文的 Results 和 Discussion。

---

## 5. 9 月目标

9 月主要完成：

1. 完成实验代码；
2. Fine-Tune YOLOv8s 和 YOLO26s；
3. 完成 Pretrained 与 Fine-Tuned 对比；
4. 完成 640 与 1280 小目标实验；
5. 完成 3×3 鲁棒性测试；
6. 生成最终表格、图片和统计结果；
7. 完成论文；
8. 准备投稿 **IEEE Access**。

总体上不继续增加很多模型和复杂模块，而是集中把 **小目标、分辨率、Fine-Tuning、复杂环境鲁棒性和最新 YOLO 模型验证** 这几部分做好。
