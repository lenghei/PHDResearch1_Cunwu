# SOTA Comparative Analysis Report: VisDrone-DET Benchmark

**Dataset:** VisDrone-DET 2019 Benchmark

**Scope:** Literature Benchmark vs. Baseline Zero-Shot ($S_1T_1$) & Compound Environmental Degradation ($S_1$–$S_3$ Blur, $T_1$–$T_3$ Noise across YOLOv8 Scales)

---

## 1. Master Comparative Benchmark Matrix

The table below compiles empirical evaluation results alongside peer-reviewed SOTA literature published on the VisDrone benchmark:

| Model / Paper Name | Publication / Venue | Architecture / Key Modifications | Input Size | Overall $\text{mAP}_{50}$ | `Car` $\text{mAP}_{50}$ | `Person` $\text{mAP}_{50}$ | Direct Reference Link |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **DBNet** | ICCV 2021 Workshop | DCNv2 + Swin Transformer Backbone + Multi-scale TTA | $1536\text{px}$ | **65.34%** | **72.10%** | **44.50%** | [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2021W/VisDrone/html/Cao_VisDrone-DET2021_The_Vision_Meets_Drone_Object_Detection_Challenge_Results_ICCVW_2021_paper.html) |
| **TPH-YOLOv5** | ICCV 2021 Workshop | Transformer Prediction Heads + CBAM Attention + P2 Head | $1280\text{px}$ | **62.83%** | **68.50%** | **38.20%** | [CVF PDF Link](https://openaccess.thecvf.com/content/ICCV2021W/VisDrone/papers/Zhu_TPH-YOLOv5_Improved_YOLOv5_Based_on_Transformer_Prediction_Head_for_Object_ICCVW_2021_paper.pdf) |
| **MSC-YOLO** | CMC / SciOpen (2024) | Spatial-to-Depth (CSPDC) + Multi-Scale Spatial Context + P2 | $1280\text{px}$ | **62.40%** | **66.80%** | **37.10%** | [SciOpen Article](https://www.sciopen.com/article/10.32604/cmc.2024.047541) |
| **Drone-DETR** | MDPI Sensors (2024) | Enhanced RT-DETR + ESDNet + Query Selection + P2 Head | $1024\text{px}$ | **53.90%** | **65.40%** | **36.50%** | [MDPI Sensors](https://www.mdpi.com/1424-8220/24/17/5496) |
| **SOD-YOLO-l** | MDPI Remote Sensing (2024) | YOLOv8 + RFCBAM + BSSI-FPN Neck + P2 Layer | $1024\text{px}$ | **58.60%** | **63.10%** | **34.80%** | [MDPI Remote Sensing](https://www.mdpi.com/2072-4292/16/16/3057) |
| **LAF-YOLOv10** | arXiv (2026) | YOLOv10n + PC-C2f + AG-FPN + P2 Head + Wise-IoU | $640\text{px}$ | **41.20%** | **54.60%** | **30.10%** | [arXiv:2602.13378](https://arxiv.org/html/2602.13378v1) |
| **YOLOv8l (Ours)** | *Our Baseline ($S_1T_1$)* | Pre-trained COCO Weights (Zero-shot, Standard P3–P5 Head) | $640\text{px}$ | **13.14%** | **41.53%** | **22.18%** | *Internal Experiments* |
| **YOLOv8m (Ours)** | *Our Baseline ($S_1T_1$)* | Pre-trained COCO Weights (Zero-shot, Standard P3–P5 Head) | $640\text{px}$ | **12.99%** | **41.70%** | **22.04%** | *Internal Experiments* |
| **YOLOv8s (Ours)** | *Our Baseline ($S_1T_1$)* | Pre-trained COCO Weights (Zero-shot, Standard P3–P5 Head) | $640\text{px}$ | **10.84%** | **38.16%** | **17.45%** | *Internal Experiments* |
| **YOLOv8n (Ours)** | *Our Baseline ($S_1T_1$)* | Pre-trained COCO Weights (Zero-shot, Standard P3–P5 Head) | $640\text{px}$ | **7.34%** | **29.81%** | **10.68%** | *Internal Experiments* |
| **YOLOv8m (Ours)** | *Our 9-Condition Avg* | Degradation Average ($S_1$–$S_3$ Blur, $T_1$–$T_3$ Noise) | $640\text{px}$ | **8.91%** | **32.75%** | **12.99%** | *Internal Experiments* |

---

## 2. Detailed Profiles of Published SOTA Literature

### 1. DBNet (*VisDrone2021 Challenge Winner*)

* **Publication Information:** ICCV 2021 Workshops (*Vision Meets Drones Challenge*).
* **Direct Paper Link:** [VisDrone-DET2021: The Vision Meets Drone Object Detection Challenge Results](https://openaccess.thecvf.com/content/ICCV2021W/VisDrone/html/Cao_VisDrone-DET2021_The_Vision_Meets_Drone_Object_Detection_Challenge_Results_ICCVW_2021_paper.html)
* **Architecture Highlights:** DBNet combines Deformable Convolutional Networks (DCNv2) with Swin-Transformer backbones and dense multi-scale feature pyramids.
* **Key Innovations:** High input resolution ($1536 \times 1536$), Test-Time Augmentation (TTA), and ensemble heads optimized for dense UAV scenes.
* **Metrics:** Overall $\text{mAP}_{50} = 65.34\%$, `Car` $\text{mAP}_{50} \approx 72.10\%$, `Person` $\text{mAP}_{50} \approx 44.50\%$.

### 2. TPH-YOLOv5 (*ICCV 2021 Workshop*)

* **Publication Information:** ICCV 2021 Workshop on Vision Meets Drones.
* **Direct Paper Link:** [TPH-YOLOv5 Paper PDF (CVF Open Access)](https://openaccess.thecvf.com/content/ICCV2021W/VisDrone/papers/Zhu_TPH-YOLOv5_Improved_YOLOv5_Based_on_Transformer_Prediction_Head_for_Object_ICCVW_2021_paper.pdf)
* **Architecture Highlights:** Replaces standard convolutional prediction heads with Transformer Prediction Heads (TPH) to capture global context in cluttered aerial images.
* **Key Innovations:**
* Adds a dedicated **P2 prediction head** ($160 \times 160$ feature map) for tiny targets.
* Integrates Convolutional Block Attention Modules (CBAM) to reduce background noise.


* **Metrics:** Overall $\text{mAP}_{50} = 62.83\%$, `Car` $\text{mAP}_{50} = 68.50\%$, `Person` $\text{mAP}_{50} = 38.20\%$.

### 3. MSC-YOLO (*Computers, Materials & Continua, 2024*)

* **Publication Information:** *CMC / SciOpen*, Vol. 78, No. 3, 2024.
* **Direct Paper Link:** [MSC-YOLO Article on SciOpen](https://www.sciopen.com/article/10.32604/cmc.2024.047541)
* **Architecture Highlights:** Multi-Scale Spatial Context model based on YOLOv7.
* **Key Innovations:** Introduces Spatial-to-Depth Convolutional Combination (CSPDC) to prevent spatial detail loss and incorporates a P2 high-resolution head.
* **Metrics:** Overall $\text{mAP}_{50} = 62.40\%$, `Car` $\text{mAP}_{50} = 66.80\%$, `Person` $\text{mAP}_{50} = 37.10\%$.

### 4. Drone-DETR (*MDPI Sensors, 2024*)

* **Publication Information:** *Sensors*, 24(17), 5496, 2024.
* **Direct Paper Link:** [Drone-DETR MDPI Article](https://www.mdpi.com/1424-8220/24/17/5496)
* **Architecture Highlights:** Adapts Real-Time DEtection Transformer (RT-DETR) for drone aerial views via the Effective Small Object Detection Network (ESDNet).
* **Key Innovations:** Replaces anchor boxes with learnable object queries and adds multi-scale cross-attention feature fusion.
* **Metrics:** Overall $\text{mAP}_{50} = 53.90\%$, `Car` $\text{mAP}_{50} = 65.40\%$, `Person` $\text{mAP}_{50} = 36.50\%$.

### 5. SOD-YOLO (*MDPI Remote Sensing, 2024*)

* **Publication Information:** *Remote Sensing*, 16(16), 3057, 2024.
* **Direct Paper Link:** [SOD-YOLO MDPI Article](https://www.mdpi.com/2072-4292/16/16/3057)
* **Architecture Highlights:** Directly modifies YOLOv8 for small object recovery in UAV images.
* **Key Innovations:** Introduces Receptive Field Convolutional Block Attention Modules (RFCBAM) and BSSI-FPN neck for bidirectional feature integration.
* **Metrics:** Overall $\text{mAP}_{50} = 58.60\%$, `Car` $\text{mAP}_{50} = 63.10\%$, `Person` $\text{mAP}_{50} = 34.80\%$.

### 6. LAF-YOLOv10 (*arXiv Preprint, 2026*)

* **Publication Information:** arXiv:2602.13378 (UAV Object Detection Benchmark).
* **Direct Paper Link:** [LAF-YOLOv10 arXiv Paper](https://arxiv.org/html/2602.13378v1)
* **Architecture Highlights:** Lightweight architecture built on YOLOv10n incorporating PC-C2f, AG-FPN, and P2 micro-object heads.
* **Key Innovations:** Designed specifically for lightweight real-time edge deployment on NVIDIA Jetson devices while maintaining competitive accuracy.
* **Metrics:** Overall $\text{mAP}_{50} = 41.20\%$, `Car` $\text{mAP}_{50} = 54.60\%$, `Person` $\text{mAP}_{50} = 30.10\%$.

---

## 3. Comparative Gap Analysis: `Car` & `Person` Classes

### A. Focus on `Car` Category

* **Our Zero-Shot Baseline:** Achieved **41.70% $\text{mAP}_{50}$ (YOLOv8m)** and **41.53% $\text{mAP}_{50}$ (YOLOv8l)** under standard conditions ($S_1T_1$).
* **Published Fine-Tuned SOTA:** Fine-tuned models reach **63.10% – 72.10% $\text{mAP}_{50}$**.
* **Key Reasons for Performance Lead:**
1. **Bounding Box Scale:** Vehicles occupy larger pixel regions ($30\times30$ to $100\times100$ pixels) compared to pedestrians ($<15\times15$ pixels).
2. **Pre-trained Feature Transfer:** Features learned from COCO translate well to aerial top-down viewpoints for rigid geometric objects like cars.
3. **Environmental Robustness:** Across all 9 compound degradation conditions ($S_1$–$S_3$ spatial blur, $T_1$–$T_3$ texture noise), the `car` class maintained a mean of **32.75% $\text{mAP}_{50}$**, proving to be our most resilient target class.



### B. Focus on `Person` Category

* **Our Zero-Shot Baseline:** Reached **22.18% $\text{mAP}_{50}$ (YOLOv8l)** and **22.04% $\text{mAP}_{50}$ (YOLOv8m)**.
* **Published Fine-Tuned SOTA:** Fine-tuned models with P2 detection heads achieve **34.80% – 44.50% $\text{mAP}_{50}$**.
* **Key Bottleneck:** Standard YOLOv8 downsamples inputs by 32× at the P5 layer ($640 / 32 = 20 \times 20$), causing sub-15-pixel targets to lose critical spatial resolution before reaching the detection heads.

---

## 4. Summary & Action Plan for Domain Fine-Tuning

> ### Core Findings
> 
> 
> 1. **Baseline Capability:** Pre-trained COCO weights provide a reasonable zero-shot foundation for medium aerial targets (`car` at **41.70%**), but fail on smaller classes without domain adaptation.
> 2. **SOTA Gap Drivers:** The ~21.4% gap in `car` performance between our baseline (41.70%) and top SOTA models (~63%+) is primarily caused by:
> * Lack of VisDrone domain fine-tuning.
> * Standard input resolution ($640\text{px}$ vs. $1024\text{px}$–$1536\text{px}$).
> * Omission of a high-resolution P2 micro-head.
> 
> 
> 
> 

### Next Technical Steps:

1. **Domain Fine-Tuning Execution:** Run domain-specific training on the VisDrone training split using our custom pipeline to close the performance gap.
2. **Resolution Scaling:** Scale evaluation input resolution from $640\text{px}$ to $1024\text{px}$ to boost detection accuracy for small targets (`person`, `motorcycle`).
3. **Architecture Expansion:** Incorporate YOLOv10 and YOLOv11 models into the same 9-condition robustness matrix to assess state-of-the-art structural resilience under heavy environmental degradation ($S_3T_3$).
