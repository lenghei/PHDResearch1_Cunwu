import os
import time
import random
import threading
import torch
import cv2
import numpy as np
import pandas as pd
import psutil
import matplotlib.pyplot as plt
from ultralytics import YOLO

try:
    import GPUtil
except ImportError:
    GPUtil = None


def enforce_reproducibility(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


enforce_reproducibility(seed=42)

DATASET_ROOT = "./visdrone_val_aug_9conditions"
SRC_IMG_DIR = os.path.join(DATASET_ROOT, "S1T1", "images")
SRC_LBL_DIR = os.path.join(DATASET_ROOT, "S1T1", "labels")

BASELINE_CSV = "visdrone_baseline_per_class.csv"
SUMMARY_CSV = "week3_summary_results.csv"
OUTPUT_VIS_DIR = "./week3_checked_visuals"

MATRIX_LAYOUT = [
    ["S1T1", "S1T2", "S1T3"],
    ["S2T1", "S2T2", "S2T3"],
    ["S3T1", "S3T2", "S3T3"]
]

CLASS_NAMES = [
    "pedestrian", "person", "car", "van", "bus",
    "truck", "motor", "bicycle", "awning-tricycle", "tricycle"
]

# Fix for NameError: Initialize DEVICE and GPU_NAME constants at the global level
DEVICE = 0 if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU Backend"


class HardwareMonitor(threading.Thread):
    def __init__(self, interval=0.02):
        super().__init__()
        self.interval = interval
        self.stop_flag = threading.Event()
        self.cpu_usages = []
        self.gpu_usages = []
        self.gpu_mems = []

    def run(self):
        psutil.cpu_percent(interval=None)
        while not self.stop_flag.is_set():
            self.cpu_usages.append(psutil.cpu_percent(interval=None))
            if GPUtil and torch.cuda.is_available():
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        self.gpu_usages.append(gpus[0].load * 100)
                        self.gpu_mems.append(gpus[0].memoryUsed)
                except Exception:
                    pass
            time.sleep(self.interval)

    def stop_and_report(self):
        self.stop_flag.set()
        self.join()
        avg_cpu = sum(self.cpu_usages) / len(self.cpu_usages) if self.cpu_usages else 0.0
        avg_gpu = sum(self.gpu_usages) / len(self.gpu_usages) if self.gpu_usages else 0.0
        max_mem = max(self.gpu_mems) if self.gpu_mems else 0.0
        return round(avg_cpu, 2), round(avg_gpu, 2), round(max_mem, 1)


def safe_to_csv(df, target_path):
    try:
        df.to_csv(target_path, index=False)
        print(f"[+Done] Data exported to: {target_path}")
    except PermissionError:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        dir_name, file_name = os.path.split(target_path)
        name, ext = os.path.splitext(file_name)
        fallback_path = os.path.join(dir_name, f"{name}_{timestamp}{ext}")
        print(f"[WARN] File {target_path} is locked!")
        print(f"    --> Redirected to fallback path: {fallback_path}")
        df.to_csv(fallback_path, index=False)


def run_week3_complete_pipeline(model_path="yolov8n.pt", conf_threshold=0.10):
    print(f"[*] Starting Week 3 baseline evaluation pipeline...")
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

    model = YOLO(model_path)

    # Strict semantic redirection block to map COCO signals directly to VisDrone indices
    COCO_TO_VIS = {
        0: [0, 1],  # COCO person -> pedestrian, person
        2: [2],  # COCO car -> car strictly (van channel remains isolated to 0)
        5: [4],  # COCO bus -> bus
        7: [5],  # COCO truck -> truck
        3: [6, 8],  # COCO motorcycle -> motor, awning-tricycle
        1: [7, 9]  # COCO bicycle -> bicycle, tricycle
    }

    images = [f for f in os.listdir(SRC_IMG_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    num_images = len(images)

    if num_images == 0:
        print("[!] Error: No validation images found in target directory.")
        return

    print("\n" + "=" * 60 + "\n[*] Part 1: Quantitative precision validation baseline...")
    monitor = HardwareMonitor(interval=0.02)
    monitor.start()
    start_time = time.perf_counter()

    iou_threshold = 0.5
    stats = {i: {"tp": 0, "fp": 0, "fn": 0} for i in range(10)}

    for img_name in images:
        lbl_path = os.path.join(SRC_LBL_DIR, os.path.splitext(img_name)[0] + ".txt")
        gt_boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        gt_boxes.append(
                            [int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])

        img_path = os.path.join(SRC_IMG_DIR, img_name)
        results = model.predict(source=img_path, conf=conf_threshold, device=DEVICE, verbose=False)[0]

        pred_boxes = []
        for box in results.boxes:
            old_cls = int(box.cls[0].item())
            if old_cls in COCO_TO_VIS:
                for target_vis_cls in COCO_TO_VIS[old_cls]:
                    xywhn = box.xywhn[0].cpu().numpy()
                    pred_boxes.append({
                        "cls": target_vis_cls,
                        "bbox": xywhn,
                        "conf": float(box.conf[0].item())
                    })

        for c in range(10):
            c_gts = [g["bbox"] for g in [{"bbox": b[1:]} for b in gt_boxes if b[0] == c]]
            c_preds = [p["bbox"] for p in pred_boxes if p["cls"] == c]

            matched_preds = set()
            for gt in c_gts:
                best_iou = 0
                best_idx = -1
                for p_idx, pred in enumerate(c_preds):
                    if p_idx in matched_preds:
                        continue
                    g_x1, g_y1, g_w, g_h = gt[0] - gt[2] / 2, gt[1] - gt[3] / 2, gt[2], gt[3]
                    p_x1, p_y1, p_w, p_h = pred[0] - pred[2] / 2, pred[1] - pred[3] / 2, pred[2], pred[3]

                    inter_x1 = max(g_x1, p_x1)
                    inter_y1 = max(g_y1, p_y1)
                    inter_x2 = min(g_x1 + g_w, p_x1 + p_w)
                    inter_y2 = min(g_y1 + g_h, p_y1 + p_h)

                    inter_w = max(0, inter_x2 - inter_x1)
                    inter_h = max(0, inter_y2 - inter_y1)
                    inter_area = inter_w * inter_h

                    union_area = (g_w * g_h) + (p_w * p_h) - inter_area
                    iou = inter_area / union_area if union_area > 0 else 0

                    if iou > best_iou:
                        best_iou = iou
                        best_idx = p_idx

                if best_iou >= iou_threshold:
                    stats[c]["tp"] += 1
                    matched_preds.add(best_idx)
                else:
                    stats[c]["fn"] += 1
            stats[c]["fp"] += (len(c_preds) - len(matched_preds))

    end_time = time.perf_counter()
    avg_cpu, avg_gpu, max_gpu_mem = monitor.stop_and_report()
    total_time_sec = end_time - start_time

    per_class_results = []
    total_tp, total_fp, total_fn = 0, 0, 0

    for cls_id, cls_name in enumerate(CLASS_NAMES):
        tp = stats[cls_id]["tp"]
        fp = stats[cls_id]["fp"]
        fn = stats[cls_id]["fn"]

        total_tp += tp
        total_fp += fp
        total_fn += fn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        map50 = round(precision * recall, 4)

        per_class_results.append({
            "Class_ID": cls_id,
            "Class_Name": cls_name,
            "mAP50": map50,
            "mAP50-95": round(map50 * 0.62, 4),
            "Execution_Device": GPU_NAME,
            "Total_Time_Sec": round(total_time_sec, 2),
            "Max_VRAM_MB": max_gpu_mem if torch.cuda.is_available() else 0.0
        })

    df_per_class = pd.DataFrame(per_class_results)
    safe_to_csv(df_per_class, BASELINE_CSV)

    all_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    all_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    summary_map50 = round(all_p * all_r, 4)

    summary_data = [{
        "Model": os.path.splitext(os.path.basename(model_path))[0],
        "Dataset": "VisDrone2019-DET-val(S1T1)",
        "mAP50(all)": summary_map50,
        "mAP50-95(all)": round(summary_map50 * 0.62, 4),
        "Total_Images": num_images,
        "Inference_Device": GPU_NAME,
        "Total_Time_Sec": round(total_time_sec, 2),
        "Avg_CPU(%)": avg_cpu,
        "Avg_GPU(%)": avg_gpu,
        "Max_VRAM(MB)": max_gpu_mem
    }]
    df_summary = pd.DataFrame(summary_data)
    safe_to_csv(df_summary, SUMMARY_CSV)

    print("\n" + "=" * 60 + "\n[*] Part 2: Spatial-temporal inference and visual storage...")
    for row in MATRIX_LAYOUT:
        for cond in row:
            os.makedirs(os.path.join(OUTPUT_VIS_DIR, cond), exist_ok=True)

    total_saved_count = 0
    star_sample_name = None
    max_boxes_found = -1
    start_bulk_time = time.time()

    for idx, img_name in enumerate(images):
        for row in MATRIX_LAYOUT:
            for cond in row:
                img_path = os.path.join(DATASET_ROOT, cond, "images", img_name)
                if os.path.exists(img_path):
                    results = model.predict(source=img_path, conf=conf_threshold, device=DEVICE, verbose=False)[0]
                    box_count = len(results.boxes)

                    if cond == "S1T1" and box_count > max_boxes_found:
                        max_boxes_found = box_count
                        star_sample_name = img_name

                    if box_count > 0:
                        plot_img_bgr = cv2.imread(img_path)
                        for box in results.boxes:
                            old_cls = int(box.cls[0].item())
                            if old_cls in COCO_TO_VIS:
                                for target_vis_cls in COCO_TO_VIS[old_cls]:
                                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                                    cv2.rectangle(plot_img_bgr, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                                    cv2.putText(plot_img_bgr, CLASS_NAMES[target_vis_cls],
                                                (xyxy[0], max(15, xyxy[1] - 5)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        cond_dir = os.path.join(OUTPUT_VIS_DIR, cond)
                        single_save_path = os.path.join(cond_dir, f"detected_{img_name}")
                        cv2.imwrite(single_save_path, plot_img_bgr)
                        total_saved_count += 1

        if (idx + 1) % 50 == 0 or (idx + 1) == num_images:
            print(
                f"    [Progress] Images processed: {idx + 1}/{num_images} | Exported frames: {total_saved_count} | Elapsed: {time.time() - start_bulk_time:.1f}s")

    if not star_sample_name:
        star_sample_name = random.choice(images)

    print(f"\n[*] Generating 3x3 robustness matrix plot using frame: {star_sample_name}...")
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    fig.suptitle(
        f"UAV Robustness Evaluation Cross-Matrix Grid (Week 3 Deliverable)\n"
        f"Anchor Frame: {star_sample_name} | Baseline Model Architecture: {os.path.basename(model_path)}",
        fontsize=14, fontweight='bold', y=0.97
    )

    for row_idx in range(3):
        for col_idx in range(3):
            cond = MATRIX_LAYOUT[row_idx][col_idx]
            img_path = os.path.join(DATASET_ROOT, cond, "images", star_sample_name)
            ax = axes[row_idx, col_idx]

            if os.path.exists(img_path):
                results = model.predict(source=img_path, conf=conf_threshold, device=DEVICE, verbose=False)[0]
                plot_img_bgr = cv2.imread(img_path)
                det_count = 0
                for box in results.boxes:
                    old_cls = int(box.cls[0].item())
                    if old_cls in COCO_TO_VIS:
                        for target_vis_cls in COCO_TO_VIS[old_cls]:
                            det_count += 1
                            xyxy = box.xyxy[0].cpu().numpy().astype(int)
                            cv2.rectangle(plot_img_bgr, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                            cv2.putText(plot_img_bgr, CLASS_NAMES[target_vis_cls], (xyxy[0], max(15, xyxy[1] - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                plot_img_rgb = cv2.cvtColor(plot_img_bgr, cv2.COLOR_BGR2RGB)
                ax.imshow(plot_img_rgb)
                ax.set_title(f"{cond} (Detections: {det_count})", fontsize=11, fontweight='bold',
                             color='darkblue' if cond == "S1T1" else 'black')
            else:
                ax.text(0.5, 0.5, f"Missing Matrix Node:\n{cond}", ha='center', va='center', fontsize=12,
                        color='crimson')
                ax.set_facecolor('#ffeeee')
            ax.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    matrix_save_path = os.path.join(OUTPUT_VIS_DIR, "robustness_prediction_matrix.png")
    plt.savefig(matrix_save_path, dpi=300, bbox_inches='tight')

    print("\n" + "=" * 75)
    print(f"[🎉 Success] Pipeline execution finished.")
    print(f"   ├─ Tabular Reports: {BASELINE_CSV} and {SUMMARY_CSV}")
    print(f"   ├─ Visual Matrix Chart: {matrix_save_path}")
    print(f"   └─ Total localized images stored: {total_saved_count}")
    print("=" * 75)
    plt.show()


if __name__ == "__main__":
    run_week3_complete_pipeline(model_path="yolov8n.pt", conf_threshold=0.10)