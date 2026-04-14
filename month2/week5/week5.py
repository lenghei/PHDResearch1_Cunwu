# Week 5: 9-Condition Image Perturbation for VisDrone
# Using Albumentations to generate spatial and temporal variations
# Save augmented dataset and perform visual validation

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A

# File paths
IMG_DIR = "./data/VisDrone2019-DET-val/images"
SAVE_ROOT = "./visdrone_val_aug_9conditions"

os.makedirs(SAVE_ROOT, exist_ok=True)
image_list = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])

# Define 9 combined conditions (S: spatial, T: temporal)
conditions = [
    ("S1T1", 1, 1), ("S1T2", 1, 2), ("S1T3", 1, 3),
    ("S2T1", 2, 1), ("S2T2", 2, 2), ("S2T3", 2, 3),
    ("S3T1", 3, 1), ("S3T2", 3, 2), ("S3T3", 3, 3),
]

# Create transformation for each condition
def get_pipeline(S, T):
    transforms = []

    # Spatial adjustments
    if S == 2:
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=(0.25, 0.25),
            contrast_limit=(0.15, 0.15),
            p=1.0))
    elif S == 3:
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=(-0.22, -0.22),
            contrast_limit=(-0.1, -0.1),
            p=1.0))

    # Temporal adjustments (blur + noise)
    if T == 2:
        transforms.append(A.MotionBlur(blur_limit=5, p=1.0))
    elif T == 3:
        transforms.append(A.MotionBlur(blur_limit=9, p=1.0))
        transforms.append(A.GaussNoise(var_limit=(10, 20), p=1.0))

    return A.Compose(transforms)

# Generate and save all augmented images
for cond_name, S, T in conditions:
    out_dir = os.path.join(SAVE_ROOT, cond_name)
    os.makedirs(out_dir, exist_ok=True)
    pipeline = get_pipeline(S, T)

    for img_name in image_list:
        img_path = os.path.join(IMG_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        aug = pipeline(image=img_rgb)["image"]
        aug_bgr = cv2.cvtColor(aug, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(out_dir, img_name), aug_bgr)

print("All 9 augmented datasets saved successfully.")

# Visual validation (FIXED subplot layout)
def show_comparison(index):
    img_path = os.path.join(IMG_DIR, image_list[index])
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(18, 12))

    # Show original
    plt.subplot(4, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original")
    plt.axis("off")

    # Show 9 conditions
    for i, (name, S, T) in enumerate(conditions):
        aug = get_pipeline(S, T)(image=img_rgb)["image"]
        plt.subplot(4, 3, i + 4)
        plt.imshow(aug)
        plt.title(name)
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Show sample results
show_comparison(0)
show_comparison(10)
show_comparison(20)
show_comparison(30)