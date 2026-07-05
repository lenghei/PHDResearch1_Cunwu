

#  Robustness Evaluation Data Pipeline and Evaluation Matrix

### 1. 3×3 Compound Stress Matrix and Data Pipeline Implementation (`run_data_pipeline.py`)

The primary objective of this module is to programmatically inject compound environmental and operational degradations into the VisDrone validation set without inducing artificial label space drift or coordinate misalignment, strictly following the parameters defined in our research plan.

* **Spatial Degradation (S - Illumination and Scene Factors):**
* **S1 (Normal):** Bypasses all modifications to preserve the clean, unmodified validation imagery as our experimental baseline.
* **S2 (Overexposure):** Simulates a high-altitude flight under direct, intense sunlight. In the Albumentations pipeline, the non-linear absolute pixel shift of $+80$ is mapped to a relative `brightness_limit` of `(0.31, 0.31)`, paired with a strict contrast constraint `contrast_limit=(0.3, 0.3)`.
* **S3 (Underexposure):** Simulates low-light conditions such as dawn, dusk, or heavy terrain shadowing. The parameters are constrained to a relative brightness drop of `(-0.20, -0.20)` (representing an absolute pixel drop of $-50$) and a contrast compression of `(-0.2, -0.2)`.


* **Temporal Degradation (T - Flight Dynamics and Hardware Artifacts):**
* **T1 (No Blur):** Simulates a stationary hover or extremely low cruising speed; no blur kernel is applied.
* **T2 (Moderate Blur):** Simulates standard autonomous cruising speed. This is implemented via an asymmetric `A.MotionBlur` kernel locked exactly at `(7, 7)`.
* **T3 (Heavy Blur):** Simulates high-speed flight maneuvers or severe wind-induced turbulence. The motion blur kernel is expanded to `(21, 21)` and sequentially combined with additive `A.GaussNoise` to accurately model sensor electronic noise under high-frequency mechanical vibration.



**Critical Bug Fix (Bounding Box Synchronization):**
In legacy preprocessing setups, spatial image manipulations often operated independently from the label space, leading to disjointed bounding boxes, compromised Intersection over Union (IoU) values, and artificial $mAP$ drops. This pipeline implements Albumentations' `BboxParams` protocol, locking the image array transformations and bounding box coordinate spaces together using the standard `'coco'` format `[x_min, y_min, width, height]`. When an image undergoes exposure shifts or blurring, its corresponding label boundaries scale pixel-for-pixel.

---

### 2. Robust Label Parsing and YOLO Relative Normalization

The raw VisDrone annotations are stored as absolute pixel coordinates. Standard lightweight object detection heads require localized relative center points and normalized dimensions.

* **Elimination of Hardcoded Resolution Bugs:**
The parser dynamically retrieves the resolution of each aerial frame via `img.shape[:2]`, completely replacing the previous hardcoded assumptions of a uniform $1920 \times 1080$ viewport. This guarantees mathematical correctness for mixed-resolution aerial datasets.
* **Boundary Truncation and Clipping:**
Due to edge-boundary targets in drone photography, certain bounding boxes contain pixels that overflow the image frame. These out-of-bounds coordinates trigger assertion crashes during geometric augmentation. The code actively clips these outliers to the valid physical dimensions of the image before computing the standard YOLO normalized formulas:

$$x_{center} = \frac{x_{min} + \frac{w}{2}}{w_{img}}, \quad y_{center} = \frac{y_{min} + \frac{h}{2}}{h_{img}}$$



The resulting coordinates are capped using `max(0.0, min(1.0, x))` to eliminate float rounding overflows. For background frames containing no objects, empty `.txt` files are written out to keep the validation set size perfectly aligned, preventing image-label count mismatch errors during evaluation.

---

### 3. Automated Cross-Validation Evaluation Engine (`run_evaluation_matrix.py`)

This engine automates the cross-testing loop across 4 distinct detection models and 9 stress configurations (totaling $4 \times 9 = 36$ distinct validation sessions), saving the finalized academic metrics into `week6_results.csv`.

* **Heterogeneous Model Benchmarking:**
The evaluation queue represents diverse architectural capacities and inductive biases. It pairs the standard lightweight anchors **YOLOv8n** (~3.2M params) and **YOLOv8s** (~11.2M params) with **LEAF-YOLO-N** (~1.2M params)—a custom 2025 state-of-the-art detector optimized for VisDrone—and **RT-DETR-R18** (~20M params), introducing a Transformer-based global attention mechanism for comparison.
* **Hardware-Constrained Inference Pipeline (`batch=4`):**
To accurately align our benchmarks with the real physical constraints of deploying models on embedded UAV edge hardware (such as the NVIDIA Jetson platform), the evaluation engine forces a serial inference scheme with `batch=4`, rather than utilizing standard large-batch acceleration configurations.
* **Environment Isolation and Cache Flushing:**
During sequential model evaluations, the underlying validation framework automatically generates an `images.cache` binary metadata file within the image directories. If a subsequent model architecture reads a legacy cache left by a previous run, the validation process misaligns, often driving $mAP$ scores down to zero. My implementation completely mitigates this issue through two mechanisms:
1. It programmatically writes a unique, isolated `temp_config_*.yaml` configuration file for each split, deleting it immediately after the session terminates to prevent directory pollution.
2. Prior to initializing each model call, it forces a filesystem check to delete any preexisting `.cache` files via `os.remove()`, ensuring that every architecture evaluates on uncontaminated, pristine data splits.
