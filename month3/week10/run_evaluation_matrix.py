import os
import time
import torch
import threading
import pandas as pd
import psutil

try:
    import GPUtil
except ImportError:
    GPUtil = None
from ultralytics import YOLO

# ==============================================================================
# 1. EVALUATION TARGET QUEUE (ALIGNED WITH PHD PLAN)
# ==============================================================================
MODELS = [
    ("YOLOv8n", "yolov8n.pt"),
    ("YOLOv8s", "yolov8s.pt"),
    ("LEAF-YOLO-N", "leaf_yolo_n.pt"),
    ("RT-DETR-R18", "rtdetr-r18x.pt"),
]

CONDITIONS = [
    "S1T1", "S1T2", "S1T3",
    "S2T1", "S2T2", "S2T3",
    "S3T1", "S3T2", "S3T3"
]

DATASET_ROOT = "./visdrone_val_aug_9conditions"
OUTPUT_CSV = "week6_results.csv"
VISUALS_DIR = "./evaluation_visuals"

# Automatic local platform hardware verification backend
DEVICE = 0 if torch.cuda.is_available() else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None (CPU Backend)"


# ==============================================================================
# 2. ASYNCHRONOUS HIGH-FREQUENCY HARDWARE MONITOR
# ==============================================================================
class HardwareMonitor(threading.Thread):
    """ Daemon thread capturing raw multi-core workloads to trace GPU bursts """
    def __init__(self, interval=0.02):
        super().__init__()
        self.interval = interval
        self.stop_flag = threading.Event()
        self.cpu_usages = []
        self.gpu_usages = []
        self.gpu_mems = []

    def run(self):
        psutil.cpu_percent(interval=None) # Warm up psutil state
        while not self.stop_flag.is_set():
            self.cpu_usages.append(psutil.cpu_percent(interval=None))
            if GPUtil and torch.cuda.is_available():
                try:
                    gpus = GPUtil.getGPUs()
                    if gpus:
                        load = gpus[0].load * 100
                        if load > 0.0: # Isolate active execution overhead from idle intervals
                            self.gpu_usages.append(load)
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


# ==============================================================================
# 3. ROBUSTNESS MATRIX CROSS-VALIDATION LOOP
# ==============================================================================
def run_evaluation_matrix():
    results_storage = []
    os.makedirs(VISUALS_DIR, exist_ok=True)

    print(f"[*] Robustness Matrix Benchmark Engine Engaged.")
    print(f"[*] Compute Target Device: {DEVICE} | Host GPU Architecture: {GPU_NAME}")

    for model_name, weight_path in MODELS:
        print("\n" + "="*95)
        print(f"Deploying Model Structure: {model_name} | Weights: {weight_path}")
        print("="*95)

        try:
            model = YOLO(weight_path)
        except Exception:
            print(f"[!] Target checkpoint {weight_path} missing locally. Trying cloud download fallback...")
            try:
                model = YOLO(f"{model_name.lower()}.pt")
            except Exception:
                continue

        for cond in CONDITIONS:
            temp_yaml_path = f"temp_config_{model_name}_{cond}.yaml"
            target_img_dir = os.path.abspath(os.path.join(DATASET_ROOT, cond, "images"))

            if not os.path.exists(target_img_dir):
                continue

            # CRITICAL PHD BUGFIX: FORCE CACHE PURGE
            # Ultralytics natively tracks an 'images.cache' file inside the directory.
            # If present, it completely ignores rewritten text labels, leading to broken mAP scores.
            # Explicitly removing old cache objects guarantees pristine labels parser pipeline initialization.
            for cache_extension in [".cache", ".matrix"]:
                suspected_cache_file = os.path.join(DATASET_ROOT, cond, f"images{cache_extension}")
                if os.path.exists(suspected_cache_file):
                    try:
                        os.remove(suspected_cache_file)
                        print(f"    [Cache Cleared] Extinguished stale tracking artifact: {cache_extension}")
                    except Exception as e:
                        print(f"    [Warning] Failed to delete cache file: {str(e)}")

            # Dynamically instantiate data profile sandbox configuration
            yaml_content = f"""
path: {os.path.abspath('.')}
train: {target_img_dir}
val: {target_img_dir}
nc: 10
names: {{0: pedestrian, 1: person, 2: car, 3: van, 4: bus, 5: truck, 6: motor, 7: bicycle, 8: awning-tricycle, 9: tricycle}}
"""
            with open(temp_yaml_path, "w", encoding="utf-8") as f:
                f.write(yaml_content)

            num_images = len([f for f in os.listdir(target_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])

            # Trigger asynchronous system monitor tracking
            monitor = HardwareMonitor(interval=0.02)
            monitor.start()
            start_time = time.perf_counter()

            try:
                # Primary forward execution block
                metrics = model.val(
                    data=temp_yaml_path,
                    imgsz=640,
                    batch=4,  # Elevate load bounds to saturate local RTX 4060 pipeline fully
                    device=DEVICE,
                    verbose=False,
                    plots=False
                )
                end_time = time.perf_counter()
                avg_cpu, avg_gpu, max_gpu_mem = monitor.stop_and_report()

                total_duration_sec = end_time - start_time
                avg_latency_ms = (total_duration_sec / num_images) * 1000 if num_images > 0 else 0.0
                mAP50 = float(metrics.box.map50)
                mAP50_95 = float(metrics.box.map)

                print(f"    [{cond}] Real mAP50: {mAP50:.4f} | Avg-GPU: {avg_gpu}% | Latency: {avg_latency_ms:.1f}ms")

                # Export single test proof illustration cases for report documentation 
                valid_frames = [f for f in os.listdir(target_img_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                if valid_frames:
                    model.predict(
                        source=os.path.join(target_img_dir, valid_frames[0]),
                        imgsz=640, device=DEVICE, save=True,
                        project=os.path.join(VISUALS_DIR, model_name, cond),
                        name="prediction", exist_ok=True, verbose=False
                    )

            except Exception as e:
                monitor.stop_and_report()
                print(f"    [!] Internal evaluation exception hit on condition {cond}: {str(e)}")
                mAP50, mAP50_95, total_duration_sec, avg_latency_ms, avg_cpu, avg_gpu, max_gpu_mem = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

            # Store records safely into the matrix storage framework
            results_storage.append({
                "Model": model_name, "Condition": cond,
                "mAP50": round(mAP50, 4), "mAP50-95": round(mAP50_95, 4),
                "GPU_Hardware": GPU_NAME, "Total_Time(s)": round(total_duration_sec, 2),
                "Avg_Latency(ms)": round(avg_latency_ms, 2), "Avg_CPU(%)": avg_cpu,
                "Avg_GPU(%)": avg_gpu, "Max_VRAM(MB)": max_gpu_mem
            })
            if os.path.exists(temp_yaml_path):
                os.remove(temp_yaml_path)

    # ==============================================================================
    # 4. CSV SUBMISSION GENERATION
    # ==============================================================================
    if results_storage:
        df = pd.DataFrame(results_storage)
        df.to_csv(OUTPUT_CSV, index=False)
        print("\n" + "="*95)
        print(f"[+Done] System analytics compiled successfully inside: {OUTPUT_CSV}")
        print(f"[+Done] High-resolution sample figures stored at: {VISUALS_DIR}")
        print("="*95)
        print(df.to_string(index=False))
    else:
        print("[!] Evaluation registry empty. Verify data paths structures.")


if __name__ == "__main__":
    run_evaluation_matrix()
