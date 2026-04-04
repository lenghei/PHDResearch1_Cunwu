import os
import json
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Compatibility fix for numpy 2.x
import numpy
numpy.NPY_OWNDATA = 1

# VisDrone class names (0-9 as defined in the dataset)
CLASS_NAMES = [
    "pedestrian", "person", "car", "van", "bus",
    "truck", "motor", "bicycle", "awning-tricycle", "tricycle"
]

class VisDroneBaselineEvaluator:
    def __init__(self, data_dir="./data"):
        # Load official YOLOv8n pretrained model
        self.model = YOLO("yolov8n.pt")
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, "VisDrone2019-DET-val/images")
        self.ann_dir = os.path.join(data_dir, "VisDrone2019-DET-val/annotations")
        self.img_list = sorted([f for f in os.listdir(self.img_dir) if f.endswith(".jpg")])
        self.coco_gt = self.build_coco_format_annotations()

    def build_coco_format_annotations(self):
        # Convert VisDrone annotations to COCO format for evaluation
        coco = {
            "images": [], "annotations": [], "categories": [],
            "licenses": [], "info": {}
        }

        # Add class definitions
        for idx, name in enumerate(CLASS_NAMES):
            coco["categories"].append({
                "id": idx + 1,
                "name": name,
                "supercategory": "object"
            })

        ann_id = 1
        for img_id, img_name in enumerate(self.img_list, 1):
            # Read image to get real size
            img = cv2.imread(os.path.join(self.img_dir, img_name))
            h, w = img.shape[:2] if img is not None else (1080, 1920)

            coco["images"].append({
                "id": img_id,
                "file_name": img_name,
                "width": w,
                "height": h
            })

            # Read VisDrone annotation txt
            txt_path = os.path.join(self.ann_dir, img_name.replace(".jpg", ".txt"))
            if not os.path.exists(txt_path):
                continue

            with open(txt_path, "r") as f:
                lines = f.read().splitlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                if len(parts) < 8:
                    continue

                try:
                    x1 = int(parts[0])
                    y1 = int(parts[1])
                    bw = int(parts[2])
                    bh = int(parts[3])
                    score = int(parts[4])
                    cls = int(parts[5])

                    # Only use valid annotations
                    if score == 1:
                        coco["annotations"].append({
                            "id": ann_id,
                            "image_id": img_id,
                            "category_id": cls + 1,
                            "bbox": [x1, y1, bw, bh],
                            "area": bw * bh,
                            "iscrowd": 0
                        })
                        ann_id += 1
                except:
                    continue

        # Save temporary COCO json
        with open("visdrone_coco_gt.json", "w") as f:
            json.dump(coco, f, indent=2)

        return COCO("visdrone_coco_gt.json")

    def run_evaluation(self):
        # Run inference and collect predictions
        predictions = []
        print("Running YOLOv8n baseline evaluation on VisDrone val set...")

        for idx, img_name in enumerate(self.img_list):
            if (idx + 1) % 100 == 0:
                print(f"Processing image {idx+1}/{len(self.img_list)}")

            img = cv2.imread(os.path.join(self.img_dir, img_name))
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Inference with low conf to keep recall
            results = self.model(
                img_rgb,
                imgsz=640,
                conf=0.001,
                iou=0.65,
                verbose=False
            )

            for res in results:
                for box in res.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    w = x2 - x1
                    h = y2 - y1
                    conf = float(box.conf[0])
                    coco_cls = int(box.cls[0])

                    # Map COCO classes to VisDrone classes
                    mapping = {
                        0: [1, 2],
                        1: [8, 9, 10],
                        2: [3, 4],
                        3: [7],
                        5: [5],
                        7: [6]
                    }

                    if coco_cls in mapping:
                        for vis_cls in mapping[coco_cls]:
                            predictions.append({
                                "image_id": idx + 1,
                                "category_id": vis_cls,
                                "bbox": [x1, y1, w, h],
                                "score": conf
                            })

        # Evaluate using COCO API
        coco_dt = self.coco_gt.loadRes(predictions)
        evaluator = COCOeval(self.coco_gt, coco_dt, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

        # Extract precision matrix
        prec = evaluator.eval["precision"]

        # Calculate per-class metrics
        results = []
        for i, cls_name in enumerate(CLASS_NAMES):
            cid = i + 1
            if cid in evaluator.params.catIds:
                pos = evaluator.params.catIds.index(cid)
                ap50 = np.mean(prec[0, :, pos, 0, 2])
                ap50 = float(ap50) if not np.isnan(ap50) else 0.0
                ap5095 = np.mean(prec[:, :, pos, 0, 2])
                ap5095 = float(ap5095) if not np.isnan(ap5095) else 0.0
            else:
                ap50 = 0.0
                ap5095 = 0.0

            results.append({
                "class": cls_name,
                "AP@0.5": round(ap50, 4),
                "AP@0.5:0.95": round(ap5095, 4)
            })

        # Overall metrics
        mAP50 = round(float(evaluator.stats[1]), 4)
        mAP5095 = round(float(evaluator.stats[0]), 4)

        results.append({
            "class": "overall",
            "AP@0.5": mAP50,
            "AP@0.5:0.95": mAP5095
        })

        # Save to CSV
        df = pd.DataFrame(results)
        df.to_csv("yolov8n_visdrone_baseline.csv", index=False)

        # Print summary
        print("\n=== Baseline Evaluation Results ===")
        print(f"Overall mAP@0.5:      {mAP50}")
        print(f"Overall mAP@0.5:0.95: {mAP5095}")
        print("\nPer-class results saved to yolov8n_visdrone_baseline.csv")

if __name__ == "__main__":
    evaluator = VisDroneBaselineEvaluator(data_dir="./data")
    evaluator.run_evaluation()