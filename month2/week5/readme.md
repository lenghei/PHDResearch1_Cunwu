# Week 5 Code Explanation
**Code Runtime**: Approximately 0.2 hours

**Personal Statement**: This script was successfully debugged and revised **6 times** before full execution. The output results file was generated correctly. However, I have one limitation: the parameters for the 9 experimental conditions were determined based on visual image effects, since no official fixed standards were provided. I cannot fully confirm whether these parameter settings are fully optimal or reasonable.

## Overall Purpose
The script takes a validation image dataset from VisDrone (a benchmark dataset for object detection from drones) and generates 9 augmented versions of it by applying combinations of **spatial** (brightness/contrast) and **temporal** (blur/noise) perturbations. This is used to test how robust object detectors are under different imaging conditions.

## Step 1 – Setup & Imports
```python
import os, cv2, numpy as np, matplotlib.pyplot as plt
import albumentations as A
```
It imports standard libraries plus **OpenCV** for image loading/saving and **Albumentations** — a popular image augmentation library for computer vision.

It defines two paths: `IMG_DIR` points to the original VisDrone validation images, and `SAVE_ROOT` is where the 9 augmented datasets will be saved. It then loads a sorted list of all `.jpg` images.

## Step 2 – The 9 Conditions
```python
conditions = [
    ("S1T1", 1, 1), ("S1T2", 1, 2), ("S1T3", 1, 3),
    ("S2T1", 2, 1), ("S2T2", 2, 2), ("S2T3", 2, 3),
    ("S3T1", 3, 1), ("S3T2", 3, 2), ("S3T3", 3, 3)
]
```
This creates a 3×3 grid of conditions, combining:
- **Spatial (S)**: 3 levels — S1 = no brightness/contrast change, S2 = brighter/higher contrast, S3 = darker/lower contrast.
- **Temporal (T)**: 3 levels — T1 = no blur/noise, T2 = mild motion blur, T3 = strong motion blur + Gaussian noise.

Each combination represents a real-world scenario a drone camera might face (e.g., overexposed + blurry, underexposed + noisy).

## Step 3 – `get_pipeline(S, T)` Function
This builds an Albumentations augmentation pipeline based on the S and T level:
- S2 → applies `RandomBrightnessContrast` with positive values (brighter image, +25% brightness, +15% contrast).
- S3 → applies `RandomBrightnessContrast` with negative values (darker image, -22% brightness, -10% contrast).
- T2 → applies `MotionBlur` with a kernel of 5 (mild blur, simulating slight camera motion).
- T3 → applies `MotionBlur` with a kernel of 9 (strong blur) plus `GaussNoise` (simulates sensor noise).
- S1 / T1 → no transform is added for that axis, so the image is unaffected in that dimension.

## Step 4 – Generating and Saving the Augmented Images
For each of the 9 conditions, the script creates a subdirectory (e.g., `./visdrone_val_aug_9conditions/S2T3/`), then loops over every image:
1. Loads the image with OpenCV (BGR format).
2. Converts it to RGB (required by Albumentations).
3. Applies the augmentation pipeline.
4. Converts back to BGR and saves it to the condition's output folder.

The result is 9 complete copies of the dataset, each with a different combination of perturbations.

## Step 5 – Visual Validation via `show_comparison(index)`
This function picks one image by index and displays a **4×3 subplot grid** using Matplotlib:
- The **first panel** shows the original image.
- The **next 9 panels** show the image under each of the 9 conditions, labelled accordingly (e.g., "S2T3").

The script then calls `show_comparison()` four times (for images at indices 0, 10, 20, 30) to visually verify that the augmentations look correct before using the datasets for model evaluation.

## Summary
This script is a **data preparation pipeline** for a robustness experiment. It systematically creates degraded versions of a drone image dataset across a 3×3 grid of lighting and motion/noise conditions, then saves them for downstream evaluation of lightweight object detection models (like YOLO variants). The visual validation at the end is a sanity check to confirm the augmentations are working as expected.
