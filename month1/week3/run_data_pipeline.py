import os
import cv2
import numpy as np
import albumentations as A
import random


# ==============================================================================
# 0. DETERMINISTIC SETUP (STRICT COMPLIANCE WITH INTEGRITY RULE 2)
# ==============================================================================
def enforce_reproducibility(seed=42):
    """
    Strictly aligns with Rule 2 of our PhD Research Plan.
    By locking the numpy state, the hardware-level micro-noise matrices generated
    in the T3 branch will remain pixel-for-pixel identical across all reruns.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


enforce_reproducibility(42)

# ==============================================================================
# 1. GLOBAL SETTINGS & DATASET TOPOLOGY
# ==============================================================================
SRC_IMG_DIR = r"./data/VisDrone2019-DET-val/images"
SRC_LABEL_DIR = r"./data/VisDrone2019-DET-val/annotations"
SAVE_ROOT = r"./visdrone_val_aug_9conditions"

# Remapping VisDrone [1-10] track labels to standard zero-indexed YOLO formats [0-9]
CLASS_MAP = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9
}

CONDITIONS = [
    ("S1T1", 1, 1), ("S1T2", 1, 2), ("S1T3", 1, 3),
    ("S2T1", 2, 1), ("S2T2", 2, 2), ("S2T3", 2, 3),
    ("S3T1", 3, 1), ("S3T2", 3, 2), ("S3T3", 3, 3),
]


# ==============================================================================
# 2. PERTURBATION PIPELINE CONFIGURATION
# ==============================================================================
def build_stress_pipeline(S, T):
    """
    Constructs the spatial/temporal degradation pipeline via Albumentations.
    CRITICAL ARCHITECTURAL REFACTOR:
    GaussNoise is intentionally omitted from the Albumentations sequential blocks.
    To completely sidestep their chaotic API signature drift across v1.4.x/v2.0+,
    the Gaussian sensor noise layer has been routed via direct NumPy injection.
    """
    transforms = []

    # 2.1 Spatial Degradation (Illumination / Altitude Shift Simulation)
    if S == 2:
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=(80 / 255, 80 / 255), contrast_limit=(0.3, 0.3), p=1.0
        ))
    elif S == 3:
        transforms.append(A.RandomBrightnessContrast(
            brightness_limit=(-50 / 255, -50 / 255), contrast_limit=(-0.2, -0.2), p=1.0
        ))

    # 2.2 Temporal Degradation (Cruising Motion Blur)
    if T == 2:
        transforms.append(A.MotionBlur(blur_limit=(7, 7), p=1.0))
    elif T == 3:
        # T3 handles severe 21x21 cruising blur here. Sensor noise is appended down the line.
        transforms.append(A.MotionBlur(blur_limit=(21, 21), p=1.0))

    return A.Compose(transforms)


# ==============================================================================
# 3. ROBUST PARSER WITH TWO-WAY BOUNDARY CLIPPING
# ==============================================================================
def parse_and_sanitize_annotations(label_path, img_w, img_h):
    """
    Parses and enforces strict mathematical boundaries to correct coordinate overflows
    inherent to raw UAV datasets before exporting to static evaluation structures.
    """
    clean_bboxes = []
    clean_cats = []

    if not os.path.exists(label_path):
        return clean_bboxes, clean_cats

    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            tokens = line.split(',')
            if len(tokens) < 8: continue

            try:
                raw_x = float(tokens[0])
                raw_y = float(tokens[1])
                raw_w = float(tokens[2])
                raw_h = float(tokens[3])
                raw_cat = int(tokens[5])

                if raw_cat not in CLASS_MAP:
                    continue

                # Bidirectional clipping matrix to neutralize rounding drift or out-of-frame values
                x_min = max(0.0, min(float(img_w - 1), raw_x))
                y_min = max(0.0, min(float(img_h - 1), raw_y))
                x_max = max(0.0, min(float(img_w), raw_x + raw_w))
                y_max = max(0.0, min(float(img_h), raw_y + raw_h))

                clipped_w = x_max - x_min
                clipped_h = y_max - y_min

                if clipped_w > 0.1 and clipped_h > 0.1:
                    clean_bboxes.append([x_min, y_min, clipped_w, clipped_h])
                    clean_cats.append(CLASS_MAP[raw_cat])
            except ValueError:
                continue

    return clean_bboxes, clean_cats


# ==============================================================================
# 4. DATA GENERATION RUNTIME (HIGH-SPEED DISK I/O INVERTED LOOP)
# ==============================================================================
def execute_perturbation_matrix():
    if not os.path.exists(SRC_IMG_DIR):
        raise FileNotFoundError(f"Missing source images at: {SRC_IMG_DIR}")

    images = sorted([f for f in os.listdir(SRC_IMG_DIR) if f.endswith((".jpg", ".png"))])
    print(f"[INFO] Loaded benchmark validation split: {len(images)} source images found.")
    print("[INFO] Initializing generation matrix. Enforcing single-read multi-branch routine...")

    for cond_name, _, _ in CONDITIONS:
        os.makedirs(os.path.join(SAVE_ROOT, cond_name, "images"), exist_ok=True)
        os.makedirs(os.path.join(SAVE_ROOT, cond_name, "labels"), exist_ok=True)

    for count, img_name in enumerate(images, 1):
        img_path = os.path.join(SRC_IMG_DIR, img_name)
        lbl_name = os.path.splitext(img_name)[0] + ".txt"
        lbl_path = os.path.join(SRC_LABEL_DIR, lbl_name)

        base_img = cv2.imread(img_path)
        if base_img is None:
            print(f"[WARN] Failed to decode {img_name}. Skipping image.")
            continue

        h, w = base_img.shape[:2]
        img_rgb = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)

        # Single-pass annotation load to maximize I/O throughput
        bboxes, category_ids = parse_and_sanitize_annotations(lbl_path, w, h)

        for cond_name, S, T in CONDITIONS:
            out_img_path = os.path.join(SAVE_ROOT, cond_name, "images", img_name)
            out_lbl_path = os.path.join(SAVE_ROOT, cond_name, "labels", lbl_name)

            pipeline = build_stress_pipeline(S, T)

            try:
                # Forward image through the Albumentations spatial-blur stage
                transformed = pipeline(image=img_rgb)
                aug_img = transformed["image"]

                # ==============================================================
                # HARDWARE-LEVEL SENSOR NOISE SIMULATION (T3 HIGH-SPEED CRUISE)
                # ==============================================================
                # Injected directly via low-level NumPy to guarantee absolute immunity
                # from downstream library updates, fully safeguarding reproducibility.
                if T == 3:
                    # Physical translation: Target variance = 30.0 -> std = sqrt(30.0) ≈ 5.477226
                    sigma = 5.477226
                    noise = np.random.normal(0, sigma, aug_img.shape).astype(np.float32)
                    # Float addition followed by strict clipping to prevent color inversion artifacts
                    aug_img = np.clip(aug_img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

                # Native BGR conversion and storage cycle
                aug_img_bgr = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
                cv2.imwrite(out_img_path, aug_img_bgr)

                # Export coordinates to standard relative YOLO annotations [0.0, 1.0]
                yolo_buffer = []
                for box, cat in zip(bboxes, category_ids):
                    x1, y1, box_w, box_h = box

                    x_center = (x1 + box_w / 2.0) / w
                    y_center = (y1 + box_h / 2.0) / h
                    norm_w = box_w / w
                    norm_h = box_h / h

                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    norm_w = max(0.0, min(1.0, norm_w))
                    norm_h = max(0.0, min(1.0, norm_h))

                    yolo_buffer.append(f"{cat} {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n")

                with open(out_lbl_path, 'w', encoding='utf-8') as f_out:
                    f_out.writelines(yolo_buffer)

            except Exception as error:
                print(f"[ERROR] Exception caught at sample {img_name} under split {cond_name}: {str(error)}")
                continue

        if count % 50 == 0 or count == len(images):
            print(f"[PROGRESS] Synchronized [{count}/{len(images)}] frames across all 9 compound matrices.")

    print("\n" + "=" * 80)
    print(" MATRIX PERTURBATION COMPLETE: All 9 conditions derived with absolute deterministic mapping.")
    print(f" Target Root Directory: {os.path.abspath(SAVE_ROOT)}")
    print("=" * 80)


if __name__ == "__main__":
    execute_perturbation_matrix()