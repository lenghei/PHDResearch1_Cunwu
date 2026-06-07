import os
import cv2
import numpy as np
import albumentations as A

# ==============================================================================
# 1. PATH CONFIGURATION
# ==============================================================================
# Raw VisDrone verification split paths
SRC_IMG_DIR = r"./data/VisDrone2019-DET-val/images"
SRC_LABEL_DIR = r"./data/VisDrone2019-DET-val/annotations"

# Destination directory for the generated 3x3 compound stress dataset
SAVE_ROOT = r"./visdrone_val_aug_9conditions"

# CRITICAL FIX: Map VisDrone 1-10 categories to standard YOLO 0-9 indices.
# This prevents class misalignment which caused the mAP to drop to ~0.02.
CLASS_MAP = {
    1: 0,  # pedestrian -> 0
    2: 1,  # person -> 1
    3: 2,  # car -> 2
    4: 3,  # van -> 3
    5: 4,  # bus -> 4
    6: 5,  # truck -> 5
    7: 6,  # motor -> 6
    8: 7,  # bicycle -> 7
    9: 8,  # awning-tricycle -> 8
    10: 9  # tricycle -> 9
}

# ==============================================================================
# 2. 3x3 MATRIX SETUP (MATCHING THE PHD RESEARCH PLAN)
# ==============================================================================
CONDITIONS = [
    ("S1T1", 1, 1), ("S1T2", 1, 2), ("S1T3", 1, 3),
    ("S2T1", 2, 1), ("S2T2", 2, 2), ("S2T3", 2, 3),
    ("S3T1", 3, 1), ("S3T2", 3, 2), ("S3T3", 3, 3),
]


def get_pipeline(S, T):
    """
    Builds the transformation pipeline using exact physical parameters
    defined in the research framework.
    """
    transforms = []

    # 2.1 Spatial Variability (Illumination modifications)
    if S == 2:
        # S2: Overexposure -> Brightness +80 (~0.31), Contrast +0.3
        transforms.append(A.RandomBrightnessContrast(brightness_limit=(0.31, 0.31), contrast_limit=(0.3, 0.3), p=1.0))
    elif S == 3:
        # S3: Underexposure -> Brightness -50 (~ -0.20), Contrast -0.2
        transforms.append(
            A.RandomBrightnessContrast(brightness_limit=(-0.20, -0.20), contrast_limit=(-0.2, -0.2), p=1.0))

    # 2.2 Temporal Variability (Motion blur & noise degradation)
    if T == 2:
        # T2: Moderate motion blur -> kernel size = 7
        transforms.append(A.MotionBlur(blur_limit=(7, 7), p=1.0))
    elif T == 3:
        # T3: Heavy motion blur -> kernel size = 21 paired with Gaussian Noise
        transforms.append(A.MotionBlur(blur_limit=(21, 21), p=1.0))
        transforms.append(A.GaussNoise(var_limit=(10.0, 50.0), p=1.0))

    # Synchronization fix: bounding boxes must transform pixel-by-pixel with the image.
    # VisDrone format matches COCO type perfectly: [x_min, y_min, width, height]
    return A.Compose(transforms, bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))


# ==============================================================================
# 3. VISDRONE LABEL PARSER (WITH DEGRADED DATA FILTERING)
# ==============================================================================
def read_visdrone_annotations(label_path):
    bboxes = []
    category_ids = []
    if not os.path.exists(label_path):
        return bboxes, category_ids

    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            if len(parts) < 8:
                continue

            try:
                x1 = float(parts[0])
                y1 = float(parts[1])
                w = float(parts[2])
                h = float(parts[3])
                # Index 4 is score (usually 1 in val set), Index 5 is the category id (1 to 11)
                category = int(parts[5])

                # Filter out invalid shapes and ignore classes like 0 (ignored regions) or 11 (others)
                if w <= 0 or h <= 0 or category not in CLASS_MAP:
                    continue

                bboxes.append([x1, y1, w, h])
                category_ids.append(CLASS_MAP[category])  # Map 1-10 to 0-9
            except ValueError:
                continue
    return bboxes, category_ids


# ==============================================================================
# 4. PIPELINE EXECUTION LOOP
# ==============================================================================
def run_pipeline():
    if not os.path.exists(SRC_IMG_DIR):
        raise FileNotFoundError(f"Source image path missing: {SRC_IMG_DIR}")

    image_list = sorted([f for f in os.listdir(SRC_IMG_DIR) if f.endswith((".jpg", ".png"))])
    print(f"Loaded baseline validation set: {len(image_list)} images found.")
    print("Starting data perturbation matrix generation with class alignment...")

    for cond_name, S, T in CONDITIONS:
        print(f"Processing condition split: {cond_name} (S{S}, T{T})")

        # Creating standard directory tree structure for YOLO validation setup
        img_out_dir = os.path.join(SAVE_ROOT, cond_name, "images")
        label_out_dir = os.path.join(SAVE_ROOT, cond_name, "labels")
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(label_out_dir, exist_ok=True)

        pipeline = get_pipeline(S, T)
        processed_count = 0

        for img_name in image_list:
            img_path = os.path.join(SRC_IMG_DIR, img_name)
            label_name = os.path.splitext(img_name)[0] + ".txt"
            label_path = os.path.join(SRC_LABEL_DIR, label_name)

            img = cv2.imread(img_path)
            if img is None:
                continue

            h_img, w_img = img.shape[:2]
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            bboxes, category_ids = read_visdrone_annotations(label_path)

            # If the image has zero objects, write an empty file to keep dataset aligned for YOLO
            if len(bboxes) == 0:
                cv2.imwrite(os.path.join(img_out_dir, img_name), img)
                with open(os.path.join(label_out_dir, label_name), 'w') as f:
                    pass
                continue

            # Clip outlier coordinates to avoid Albumentations out-of-bounds assert crashes
            valid_bboxes = []
            valid_cats = []
            for box, cat in zip(bboxes, category_ids):
                x, y, w, h = box
                if x < w_img and y < h_img:
                    w = min(w, w_img - x)
                    h = min(h, h_img - y)
                    if w > 0 and h > 0:
                        valid_bboxes.append([x, y, w, h])
                        valid_cats.append(cat)

            try:
                # Apply transformations simultaneously to image array and coordinate space
                transformed = pipeline(image=img_rgb, bboxes=valid_bboxes, category_ids=valid_cats)
                aug_img = transformed["image"]
                aug_bboxes = transformed["bboxes"]
                aug_cats = transformed["category_ids"]

                # Write back perturbed image to disk
                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(img_out_dir, img_name), aug_img_bgr)

                # Math conversion: Absolute bounding boxes to YOLO relative normalized values
                yolo_lines = []
                for box, cat in zip(aug_bboxes, aug_cats):
                    x1, y1, w, h = box
                    x_center = (x1 + w / 2.0) / w_img
                    y_center = (y1 + h / 2.0) / h_img
                    norm_w = w / w_img
                    norm_h = h / h_img

                    # Boundary protection: truncate rounding overflow to strictly fit [0.0, 1.0]
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    norm_w = max(0.0, min(1.0, norm_w))
                    norm_h = max(0.0, min(1.0, norm_h))

                    yolo_lines.append(f"{cat} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

                with open(os.path.join(label_out_dir, label_name), 'w', encoding='utf-8') as f:
                    f.writelines(yolo_lines)
                processed_count += 1

            except Exception as e:
                print(f"Skipping problematic sample {img_name}: {str(e)}")
                continue

        print(f"Finished {cond_name}. Outputted {processed_count} paired samples.")


if __name__ == "__main__":
    run_pipeline()
    print("\nData pipeline done. All 9 stress levels generated with accurate standard YOLO index mappings.")
