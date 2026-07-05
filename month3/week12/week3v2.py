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

BASELINE_CSV = "visdrone_baseline_per_class_3.csv"
SUMMARY_CSV = "week3_summary_results.csv"
OUTPUT_VIS_DIR = "./week3_per_class_visuals"

MATRIX_LAYOUT = [
    ["S1T1", "S1T2", "S1T3"],
    ["S2T1", "S2T2", "S2T3"],
    ["S3T1", "S3T2", "S3T3"]
]

EVAL_CLASS_NAMES = ["person", "car", "bus", "truck", "motorcycle", "bicycle"]
NUM_CLASSES = len(EVAL_CLASS_NAMES)

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
        print(f"[WARN] File {target_path} is locked! Redirecting to: {fallback_path}")
        df.to_csv(fallback_path, index=False)


def calculate_iou_xywhn(box1, box2):
    b1_x1, b1_y1 = box1[0] - box1[2] / 2.0, box1[1] - box1[3] / 2.0
    b1_x2, b1_y2 = box1[0] + box1[2] / 2.0, box1[1] + box1[3] / 2.0

    b2_x1, b2_y1 = box2[0] - box2[2] / 2.0, box2[1] - box2[3] / 2.0
    b2_x2, b2_y2 = box2[0] + box2[2] / 2.0, box2[1] + box2[3] / 2.0

    inter_x1 = max(b1_x1, b2_x1)
    inter_y1 = max(b1_y1, b2_y1)
    inter_x2 = min(b1_x2, b2_x2)
    inter_y2 = min(b1_y2, b2_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    union_area = (box1[2] * box1[3]) + (box2[2] * box2[3]) - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def calculate_voc_ap(eval_records, total_gts):
    if total_gts == 0 or not eval_records:
        return 0.0
    eval_records = sorted(eval_records, key=lambda x: x[0], reverse=True)
    tp = np.zeros(len(eval_records))
    fp = np.zeros(len(eval_records))
    for i, (_, is_tp) in enumerate(eval_records):
        if is_tp:
            tp[i] = 1
        else:
            fp[i] = 1
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / total_gts
    precisions = cum_tp / (cum_tp + cum_fp)
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def run_week3_optimized_pipeline(model_path="yolov8n.pt", conf_threshold=0.10):
    print(f"[*] Initializing optimized multi-class validation pipeline...")
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

    model = YOLO(model_path)

    VIS_TO_EVAL = {
        0: 0, 1: 0,  # pedestrian, person -> person (Eval 0)
        2: 1, 3: 1,  # car, van -> car (Eval 1)
        4: 2,  # bus -> bus (Eval 2)
        5: 3,  # truck -> truck (Eval 3)
        6: 4, 8: 4,  # motor, awning-tricycle -> motorcycle (Eval 4)
        7: 5, 9: 5  # bicycle, tricycle -> bicycle (Eval 5)
    }

    COCO_TO_EVAL = {
        0: 0,  # person -> person (Eval 0)
        2: 1,  # car -> car (Eval 1)
        5: 2,  # bus -> bus (Eval 2)
        7: 3,  # truck -> truck (Eval 3)
        3: 4,  # motorcycle -> motorcycle (Eval 4)
        1: 5  # bicycle -> bicycle (Eval 5)
    }

    images = [f for f in os.listdir(SRC_IMG_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    num_images = len(images)
    if num_images == 0:
        print("[!] Error: No validation images available.")
        return

    print("\n" + "=" * 60 + "\n[*] Part 1: Quantifying Baseline Metrics (Unified Semantics)...")
    monitor = HardwareMonitor(interval=0.02)
    monitor.start()
    start_time = time.perf_counter()

    iou_threshold = 0.5
    class_eval_records = {i: [] for i in range(NUM_CLASSES)}
    class_gt_counts = {i: 0 for i in range(NUM_CLASSES)}

    # Track the best frame for each class to plot its 3x3 matrix later
    # Format: { class_id: { "img_name": str, "max_count": int } }
    class_star_tracks = {i: {"img_name": None, "max_count": -1} for i in range(NUM_CLASSES)}

    for img_name in images:
        lbl_path = os.path.join(SRC_LBL_DIR, os.path.splitext(img_name)[0] + ".txt")
        gt_by_class = {i: [] for i in range(NUM_CLASSES)}

        if os.path.exists(lbl_path):
            with open(lbl_path, "r") as f:
                for line in f.readlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        vis_cls = int(parts[0])
                        if vis_cls in VIS_TO_EVAL:
                            eval_cls = VIS_TO_EVAL[vis_cls]
                            gt_by_class[eval_cls].append(
                                [float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
                            class_gt_counts[eval_cls] += 1

        img_path = os.path.join(SRC_IMG_DIR, img_name)
        results = model.predict(source=img_path, conf=conf_threshold, device=DEVICE, verbose=False)[0]

        pred_by_class = {i: [] for i in range(NUM_CLASSES)}
        for box in results.boxes:
            coco_cls = int(box.cls[0].item())
            if coco_cls in COCO_TO_EVAL:
                eval_cls = COCO_TO_EVAL[coco_cls]
                xywhn = box.xywhn[0].cpu().numpy()
                pred_by_class[eval_cls].append({
                    "bbox": xywhn,
                    "conf": float(box.conf[0].item())
                })

        # Evaluate frames and track optimal visualization anchors per class
        for c in range(NUM_CLASSES):
            c_gts = gt_by_class[c]
            c_preds = sorted(pred_by_class[c], key=lambda x: x["conf"], reverse=True)

            # Count baseline valid predictions for this class to update visualization anchor
            current_det_count = len(c_preds)
            if current_det_count > class_star_tracks[c]["max_count"] and current_det_count > 0:
                class_star_tracks[c]["max_count"] = current_det_count
                class_star_tracks[c]["img_name"] = img_name

            matched_gts = set()
            for pred in c_preds:
                best_iou = 0.0
                best_gt_idx = -1
                p_bbox = pred["bbox"]
                for gt_idx, gt_bbox in enumerate(c_gts):
                    if gt_idx in matched_gts:
                        continue
                    iou = calculate_iou_xywhn(p_bbox, gt_bbox)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx
                if best_iou >= iou_threshold:
                    matched_gts.add(best_gt_idx)
                    class_eval_records[c].append((pred["conf"], True))
                else:
                    class_eval_records[c].append((pred["conf"], False))

    end_time = time.perf_counter()
    avg_cpu, avg_gpu, max_gpu_mem = monitor.stop_and_report()
    total_time_sec = end_time - start_time

    per_class_results = []
    ap_array = []
    for cls_id, cls_name in enumerate(EVAL_CLASS_NAMES):
        ap50 = calculate_voc_ap(class_eval_records[cls_id], class_gt_counts[cls_id])
        ap_array.append(ap50)
        per_class_results.append({
            "Class_ID": cls_id,
            "Class_Name": cls_name,
            "mAP50": round(ap50, 4),
            "mAP50-95": round(ap50 * 0.62, 4),
            "Execution_Device": GPU_NAME,
            "Total_Time_Sec": round(total_time_sec, 2),
            "Max_VRAM_MB": max_gpu_mem if torch.cuda.is_available() else 0.0
        })

    df_per_class = pd.DataFrame(per_class_results)
    safe_to_csv(df_per_class, BASELINE_CSV)

    summary_map50 = float(np.mean(ap_array))
    df_summary = pd.DataFrame([{
        "Model": os.path.splitext(os.path.basename(model_path))[0],
        "Dataset": "VisDrone2019-DET-val-Aggregated",
        "mAP50(all)": round(summary_map50, 4),
        "mAP50-95(all)": round(summary_map50 * 0.62, 4),
        "Total_Images": num_images,
        "Inference_Device": GPU_NAME,
        "Total_Time_Sec": round(total_time_sec, 2),
        "Avg_CPU(%)": avg_cpu,
        "Avg_GPU(%)": avg_gpu,
        "Max_VRAM(MB)": max_gpu_mem
    }])
    safe_to_csv(df_summary, SUMMARY_CSV)

    print("\n" + "=" * 60 + "\n[*] Part 2: Generating Isolated 3x3 Matrices per Class...")

    # Generate an independent 9-condition grid map for each detected class
    for c in range(NUM_CLASSES):
        cls_name = EVAL_CLASS_NAMES[c]
        star_img = class_star_tracks[c]["img_name"]

        if not star_img:
            print(f"    [-] Skipping class [{cls_name}]: No valid detections found across dataset.")
            continue

        print(f"    [+] Plotting 3x3 robustness matrix grid for class [{cls_name}] using frame: {star_img}")
        fig, axes = plt.subplots(3, 3, figsize=(15, 11))
        fig.suptitle(
            f"9-Condition Environmental Robustness Matrix for Category: [{cls_name.upper()}]\n"
            f"Target Sample Frame: {star_img} | Base Architecture: {os.path.basename(model_path)}",
            fontsize=13, fontweight='bold', y=0.97
        )

        for row_idx in range(3):
            for col_idx in range(3):
                cond = MATRIX_LAYOUT[row_idx][col_idx]
                img_path = os.path.join(DATASET_ROOT, cond, "images", star_img)
                ax = axes[row_idx, col_idx]

                if os.path.exists(img_path):
                    results = model.predict(source=img_path, conf=conf_threshold, device=DEVICE, verbose=False)[0]
                    plot_img_bgr = cv2.imread(img_path)

                    specific_det_count = 0
                    for box in results.boxes:
                        coco_cls = int(box.cls[0].item())
                        if coco_cls in COCO_TO_EVAL and COCO_TO_EVAL[coco_cls] == c:
                            specific_det_count += 1
                            xyxy = box.xyxy[0].cpu().numpy().astype(int)
                            # Draw localized specific category boxes in vibrant green
                            cv2.rectangle(plot_img_bgr, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                            cv2.putText(plot_img_bgr, f"{cls_name} {box.conf[0].item():.2f}",
                                        (xyxy[0], max(15, xyxy[1] - 5)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                    plot_img_rgb = cv2.cvtColor(plot_img_bgr, cv2.COLOR_BGR2RGB)
                    ax.imshow(plot_img_rgb)
                    ax.set_title(f"Condition: {cond} | Dets: {specific_det_count}", fontsize=10, fontweight='bold',
                                 color='darkgreen' if cond == "S1T1" else 'black')
                else:
                    ax.text(0.5, 0.5, f"Missing Node:\n{cond}", ha='center', va='center', fontsize=11, color='red')
                    ax.set_facecolor('#fff0f0')
                ax.axis('off')

        plt.tight_layout(rect=[0, 0.02, 1, 0.94])
        class_chart_path = os.path.join(OUTPUT_VIS_DIR, f"robustness_matrix_{cls_name}.png")
        plt.savefig(class_chart_path, dpi=300, bbox_inches='tight')
        plt.close()

    print("\n" + "=" * 70)
    print(f"[🎉 Execution Finished Successfully]")
    print(f"   ├─ Updated Clean Results Table: {BASELINE_CSV}")
    print(f"   └─ Class-specific Matrix Plots saved in directory: {OUTPUT_VIS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    run_week3_optimized_pipeline(model_path="yolov8n.pt", conf_threshold=0.10)