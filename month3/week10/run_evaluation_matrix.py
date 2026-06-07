import os
import pandas as pd
from ultralytics import YOLO

# ==============================================================================
# 1. EVALUATION TARGET QUEUE
# ==============================================================================
# Models matched exactly with the revised PhD Research Plan (Section 3.3)
MODELS = [
    ("YOLOv8n", "yolov8n.pt"),
    ("YOLOv8s", "yolov8s.pt"),
    ("LEAF-YOLO-N", "leaf_yolo_n.pt"),  # Custom 2025 SOTA checkpoint
    ("RT-DETR-R18", "rtdetr-r18x.pt"),  # Transformer-based model alternative
]

CONDITIONS = [
    "S1T1", "S1T2", "S1T3",
    "S2T1", "S2T2", "S2T3",
    "S3T1", "S3T2", "S3T3"
]

DATASET_ROOT = "./visdrone_val_aug_9conditions"
OUTPUT_CSV = "week6_results.csv"


# ==============================================================================
# 2. CROSS-VALIDATION MATRIX ENGINE
# ==============================================================================
def run_evaluation_matrix():
    results_storage = []

    for model_name, weight_path in MODELS:
        print("\n" + "=" * 70)
        print(f"Loading Model Architecture: {model_name} | Weights: {weight_path}")
        print("=" * 70)

        try:
            model = YOLO(weight_path)
        except Exception as e:
            # Fallback strategy in case local weights are missing during initial download testing
            print(f"Local checkpoint {weight_path} not found. Trying to fallback to default repo download...")
            try:
                model = YOLO(f"{model_name.lower()}.pt")
            except Exception as err:
                print(f"Failed to load {model_name}, skipping this benchmark lane. Error: {str(err)}")
                continue

        for cond in CONDITIONS:
            print(f"Evaluating Model: {model_name} on Condition Set: {cond}")

            # Isolated text config generation to prevent filesystem collision during runtime
            temp_yaml_path = f"temp_config_{model_name}_{cond}.yaml"
            target_img_dir = os.path.abspath(os.path.join(DATASET_ROOT, cond, "images"))

            yaml_content = f"""
path: {os.path.abspath('.')}
train: {target_img_dir}
val: {target_img_dir}
nc: 10
names:
  0: pedestrian
  1: person
  2: car
  3: van
  4: bus
  5: truck
  6: motor
  7: bicycle
  8: awning-tricycle
  9: tricycle
"""
            with open(temp_yaml_path, "w", encoding="utf-8") as f:
                f.write(yaml_content)

            # CRITICAL FIX: Delete any legacy label caches left behind by previous validation crashes.
            # If left unremoved, YOLO reads old binary cache files and metrics drop to zero.
            cache_file = os.path.join(DATASET_ROOT, cond, "images.cache")
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                except:
                    pass

            # Running validation loop
            try:
                metrics = model.val(
                    data=temp_yaml_path,
                    imgsz=640,
                    batch=1,  # Constrained to 1 to match real UAV edge computer deployment limits
                    device="cpu",  # Set to 0 or 'cuda' for desktop GPU acceleration
                    verbose=False,
                    plots=False,
                    save_json=False
                )

                # Extract academic standard evaluation metrics
                mAP50 = float(metrics.box.map50)
                mAP50_95 = float(metrics.box.map)

                print(f" Done -> mAP50: {mAP50:.4f} | mAP50-95: {mAP50_95:.4f}")

            except Exception as e:
                print(f" Validation pipeline crashed on condition {cond}: {str(e)}")
                mAP50 = 0.0
                mAP50_95 = 0.0

            # Store result data row wise
            results_storage.append({
                "model": model_name,
                "condition": cond,
                "mAP50": round(mAP50, 4),
                "mAP50-95": round(mAP50_95, 4)
            })

            # Clean up the runtime footprint
            if os.path.exists(temp_yaml_path):
                os.remove(temp_yaml_path)

    # ==============================================================================
    # 3. CSV EXPORT & COMPILATION
    # ==============================================================================
    if results_storage:
        df = pd.DataFrame(results_storage)
        df.to_csv(OUTPUT_CSV, index=False)
        print("\n" + "=" * 70)
        print(f"Matrix evaluation session completed successfully. Logs dumped to: {OUTPUT_CSV}")
        print("=" * 70)
        print(df.to_string())
    else:
        print("Error: No data rows collected. Check configuration flags.")


if __name__ == "__main__":
    run_evaluation_matrix()