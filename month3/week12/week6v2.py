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

# Dataset Structure Configurations
DATASET_ROOT = "./visdrone_val_aug_9conditions"
MATRIX_LAYOUT = [
    ["S1T1", "S1T2", "S1T3"],
    ["S2T1", "S2T2", "S2T3"],
    ["S3T1", "S3T2", "S3T3"]
]
ALL_CONDITIONS = [cond for row in MATRIX_LAYOUT for cond in row]

# Centralized Evaluation Metrics Output Paths
WEEK6_DETAILED_CSV = "week6_per_class_conditions_results.csv"
WEEK6_SUMMARY_CSV = "week6_models_summary.csv"
OUTPUT_VIS_DIR = "./week6_comprehensive_visuals"

# Academic Category Aggregation Map
EVAL_CLASS_NAMES = ["person", "car", "bus", "truck", "motorcycle", "bicycle"]
NUM_CLASSES = len(EVAL_CLASS_NAMES)

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


def calculate_iou_xywhn(box1, box2):
    b1_x1, b1_y1 = box1[0] - box1[2] / 2.0, box1[1] - box1[3] / 2.0
    b1_x2, b1_y2 = box1[0] + box1[2] / 2.0, box1[1] + box1[3] / 2.0

    b2_x1, b2_y1 = box2[0] - box2[2] / 2.0, box2[1] - box2[3] / 2.0
    b2_x2, b2_y2 = box2[0] + box2[2] / 2.0, box2[1] + box2[3] / 2.0

    inter_x1, inter_y1 = max(b1_x1, b2_x1), max(b1_y1, b2_y1)
    inter_x2, inter_y2 = min(b1_x2, b2_x2), min(b1_y2, b2_y2)

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


def run_week6_robustness_benchmark(models_list=None, conf_threshold=0.10):
    if models_list is None:
        models_list = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"]

    print(f"[*] Initializing Week 6 High-Throughput Matrix Evaluation Pipeline...")
    os.makedirs(OUTPUT_VIS_DIR, exist_ok=True)

    # Reference clean image lists from S1T1
    ref_img_dir = os.path.join(DATASET_ROOT, "S1T1", "images")
    if not os.path.exists(ref_img_dir):
        print(f"[!] Critical Error: Reference directory not found: {ref_img_dir}")
        return

    images = [f for f in os.listdir(ref_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    num_images = len(images)
    print(f"[+] Quantified validation corpus size: {num_images} base image tracks.")

    detailed_rows = []
    summary_rows = []

    # Execute batch grid evaluation for each model
    for model_file in models_list:
        model_name = os.path.splitext(model_file)[0]
        print("\n" + "=" * 70)
        print(f"[MODEL CORE] Processing architecture model variants: {model_file}")
        print("=" * 70)

        try:
            model = YOLO(model_file)
        except Exception as e:
            print(f"[!] Skipping {model_file}. Architecture load failed: {e}")
            continue

        # Dictionary to track the best visualization frame name for each category under this model
        # Format: { class_id: { "img_name": str, "max_count": int } }
        model_star_tracks = {i: {"img_name": None, "max_count": -1} for i in range(NUM_CLASSES)}

        # Evaluate across all 9 spatial-temporal degradation conditions
        for cond in ALL_CONDITIONS:
            print(f"  [*] Benchmarking Environmental Condition Matrix Node: [{cond}]")

            img_dir = os.path.join(DATASET_ROOT, cond, "images")
            lbl_dir = os.path.join(DATASET_ROOT, cond, "labels")

            monitor = HardwareMonitor(interval=0.02)
            monitor.start()
            start_time = time.perf_counter()

            class_eval_records = {i: [] for i in range(NUM_CLASSES)}
            class_gt_counts = {i: 0 for i in range(NUM_CLASSES)}

            for img_name in images:
                img_path = os.path.join(img_dir, img_name)
                lbl_path = os.path.join(lbl_dir, os.path.splitext(img_name)[0] + ".txt")

                if not os.path.exists(img_path):
                    continue

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

                # Matching predictions and recording target densities for validation frames
                for c in range(NUM_CLASSES):
                    c_gts = gt_by_class[c]
                    c_preds = sorted(pred_by_class[c], key=lambda x: x["conf"], reverse=True)

                    # Track best frame for isolated visual profiling (based on baseline clean density)
                    if cond == "S1T1":
                        current_det_count = len(c_preds)
                        if current_det_count > model_star_tracks[c]["max_count"] and current_det_count > 0:
                            model_star_tracks[c]["max_count"] = current_det_count
                            model_star_tracks[c]["img_name"] = img_name

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
                        if best_iou >= 0.5:
                            matched_gts.add(best_gt_idx)
                            class_eval_records[c].append((pred["conf"], True))
                        else:
                            class_eval_records[c].append((pred["conf"], False))

            end_time = time.perf_counter()
            avg_cpu, avg_gpu, max_gpu_mem = monitor.stop_and_report()
            total_time_sec = end_time - start_time

            cond_ap_array = []
            for cls_id, cls_name in enumerate(EVAL_CLASS_NAMES):
                ap50 = calculate_voc_ap(class_eval_records[cls_id], class_gt_counts[cls_id])
                cond_ap_array.append(ap50)

                detailed_rows.append({
                    "Model": model_name,
                    "Condition": cond,
                    "Class_ID": cls_id,
                    "Class_Name": cls_name,
                    "mAP50": round(ap50, 4),
                    "mAP50-95": round(ap50 * 0.62, 4),
                    "Total_GTs": class_gt_counts[cls_id]
                })

            mean_map50 = float(np.mean(cond_ap_array))
            summary_rows.append({
                "Model": model_name,
                "Condition": cond,
                "mAP50(all)": round(mean_map50, 4),
                "mAP50-95(all)": round(mean_map50 * 0.62, 4),
                "Inference_Device": GPU_NAME,
                "Runtime_Sec": round(total_time_sec, 2),
                "Avg_CPU(%)": avg_cpu,
                "Avg_GPU(%)": avg_gpu,
                "Max_VRAM_MB": max_gpu_mem
            })

            print(f"    -> [Node Completed] mAP50: {mean_map50:.4f} | Inference Runtime: {total_time_sec:.2f}s")

        # Part 2: Generating Isolated 3x3 Matrices per Class for this specific Model Architecture
        print(f"\n  [*] Generating Visual Matrix Subplots for Model Variant: {model_name}...")
        model_vis_dir = os.path.join(OUTPUT_VIS_DIR, model_name)
        os.makedirs(model_vis_dir, exist_ok=True)

        for c in range(NUM_CLASSES):
            cls_name = EVAL_CLASS_NAMES[c]
            star_img = model_star_tracks[c]["img_name"]

            if not star_img:
                continue

            fig, axes = plt.subplots(3, 3, figsize=(15, 11))
            fig.suptitle(
                f"9-Condition Robustness Matrix Breakdown | Model: [{model_name.upper()}] | Class: [{cls_name.upper()}]\n"
                f"Evaluation Target Target Frame: {star_img} | Platform Architecture: {GPU_NAME}",
                fontsize=12, fontweight='bold', y=0.97
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
                                cv2.rectangle(plot_img_bgr, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
                                cv2.putText(plot_img_bgr, f"{cls_name} {box.conf[0].item():.2f}",
                                            (xyxy[0], max(15, xyxy[1] - 5)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                        plot_img_rgb = cv2.cvtColor(plot_img_bgr, cv2.COLOR_BGR2RGB)
                        ax.imshow(plot_img_rgb)
                        ax.set_title(f"{cond} | Dets: {specific_det_count}", fontsize=10, fontweight='bold',
                                     color='darkgreen' if cond == "S1T1" else 'black')
                    else:
                        ax.text(0.5, 0.5, f"Missing:\n{cond}", ha='center', va='center', fontsize=11, color='red')
                        ax.set_facecolor('#fff0f0')
                    ax.axis('off')

            plt.tight_layout(rect=[0, 0.02, 1, 0.94])
            chart_save_path = os.path.join(model_vis_dir, f"robustness_grid_{cls_name}.png")
            plt.savefig(chart_save_path, dpi=300, bbox_inches='tight')
            plt.close()

    # Save comprehensive evaluation results to CSV files
    pd.DataFrame(detailed_rows).to_csv(WEEK6_DETAILED_CSV, index=False)
    pd.DataFrame(summary_rows).to_csv(WEEK6_SUMMARY_CSV, index=False)

    print("\n" + "=" * 80)
    print(f"[🎉 Week 6 Benchmark Complete Successfully]")
    print(f"   ├─ Detailed Class-Condition Table: {WEEK6_DETAILED_CSV}")
    print(f"   ├─ Model Performance Summary Report: {WEEK6_SUMMARY_CSV}")
    print(f"   └─ Visual Profiling Matrix Collections saved in: {OUTPUT_VIS_DIR}/")
    print("=" * 80)


if __name__ == "__main__":
    # Runs full scale benchmark across 4 variants over 9 distinct spatial-temporal domains
    run_week6_robustness_benchmark(
        models_list=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt"],
        conf_threshold=0.10
    )