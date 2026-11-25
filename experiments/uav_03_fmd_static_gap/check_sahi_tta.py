"""Check tiled inference with Ultralytics TTA on the baseline checkpoint."""
import argparse
from importlib import metadata
import os
from pathlib import Path
import sys
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "code/tools"))
from uav_reproduction import ROOT, VAL, digest, output_folder, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dependencies", type=Path, default=ROOT / "datasets/cache/uav03_sahi_01136")
    p.add_argument("--output", type=Path)
    args = p.parse_args()
    out = output_folder(HERE.name, args.output)
    print(out, flush=True)
    sys.path.insert(0, str(args.dependencies.resolve(strict=True)))
    os.environ.update(CUDA_VISIBLE_DEVICES="", YOLO_CONFIG_DIR=str(out / "runtime"),
                      YOLO_AUTOINSTALL="false", YOLO_OFFLINE="true")
    (out / "runtime/Ultralytics").mkdir(parents=True)

    import cv2
    from sahi.models.ultralytics import UltralyticsDetectionModel
    from sahi.predict import get_sliced_prediction
    import torch
    from ultralytics.nn.tasks import DetectionModel
    from main.dataset import VisDroneZIP
    from main.labels import VIS_NAMES

    checkpoint = ROOT / "models/checkpoints/yolov8s_cft640/best.pt"
    if digest(checkpoint) != "e564c819603bf36ac6d93e3acb0359e7d5a75503ce78458209e9a28113280667":
        raise ValueError("Checkpoint SHA differs.")
    if digest(VAL) != "abeea063037e5d20398837deb11084e652402a34ddf4f207bdf541a6f2a35ef9":
        raise ValueError("Validation archive SHA differs.")
    torch.set_num_threads(2)
    reader = VisDroneZIP(VAL, 548)
    try:
        image, _, member = reader.record(0)
    finally:
        reader.close()
    xywh = [600, 250, 480, 480]
    x, y, width, height = xywh
    rgb = cv2.cvtColor(image[y:y+height, x:x+width], cv2.COLOR_BGR2RGB)
    original = DetectionModel._predict_augment
    branches = []

    def counted(model, inputs):
        branches.append(list(inputs.shape))
        return original(model, inputs)

    outcomes = []
    for method in ("off", "config_path", "model_overrides"):
        config = out / f"{method}.yaml"
        config.write_text("augment: " + str(method != "off").lower() + "\n", encoding="utf-8")
        wrapper = UltralyticsDetectionModel(model_path=str(checkpoint), config_path=str(config),
                                           confidence_threshold=.03, device="cpu", image_size=640)
        if list(wrapper.model.names.values()) != VIS_NAMES or wrapper.model.model.end2end:
            raise ValueError("Expected the native ten-class dense checkpoint.")
        if method == "model_overrides":
            wrapper.model.overrides["augment"] = True
        branches.clear()
        with patch.object(DetectionModel, "_predict_augment", counted), torch.inference_mode():
            prediction = get_sliced_prediction(rgb, wrapper, slice_height=320, slice_width=320,
                        overlap_height_ratio=.25, overlap_width_ratio=.25, perform_standard_pred=False,
                        postprocess_type="NMS", postprocess_match_metric="IOU", postprocess_match_threshold=.7,
                        verbose=0)
        boxes = [item.bbox.to_xyxy() for item in prediction.object_prediction_list]
        if not boxes or any(not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height) for x1,y1,x2,y2 in boxes):
            raise ValueError("Returned detection boxes failed the image-coordinate check.")
        outcomes.append({"method": method, "effective_augment": bool(wrapper.model.predictor.args.augment),
                         "inference_augment_calls": len(branches), "tensor_shapes": branches[:],
                         "merged_boxes": len(boxes), "boxes_within_crop": True})
    supports_override = outcomes[2]["effective_augment"] and outcomes[2]["inference_augment_calls"] > 0
    if outcomes[0]["inference_augment_calls"] or not supports_override:
        raise ValueError("TTA controls did not produce the expected branches.")
    write_json(out / "sahi_tta.json", {"status": "COMPATIBLE_WITH_MODEL_OVERRIDE", "scope": "ONE_YOLOV8S_CPU_FIXTURE_NO_AP",
        "checkpoint_sha256": digest(checkpoint), "input_image_sha256": member["image_SHA256"],
        "crop_xywh": xywh, "sahi": metadata.version("sahi"), "ultralytics": metadata.version("ultralytics"),
        "image_size": 640, "tile_size": 320, "tile_overlap": .25, "results": outcomes})


if __name__ == "__main__":
    main()
