"""Profile the real frozen YOLOv8s checkpoint on CPU and join measured evidence."""
import argparse
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
import yaml

MAIN = Path(__file__).resolve().parents[2]
ROOT = MAIN.parent
AUDIT = ROOT / "datasets/local_runs/cv261207/cv261207_audit_20261104T193649_d1b63492"
BASE = ROOT / "datasets/local_runs/cv261102/cv261102_full_20261031T000013_d0be8587"
COMPLIANCE = ROOT / "datasets/local_runs/cv261207/cv261207_supervisor_compliance_20261109T121122_70d953f1"
PILOT = ROOT / "datasets/local_runs/cv261116/cv261116_gpu_20261101T225631_5bc4e6ce"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main(out):
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["YOLO_CONFIG_DIR"] = str(out / "runtime")
    os.environ.update(YOLO_AUTOINSTALL="false", YOLO_OFFLINE="true")
    (out / "runtime/Ultralytics").mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(MAIN / "experiments/experiments_test/cv261207"))
    from common import CHECKPOINT, sha256
    from supervisor_profile import profile
    from run import audit_plan

    _, plan = audit_plan(AUDIT.name)
    detectors = list({p["detector"]["detector_id"]: p["detector"] for p in plan["pairs"]}.values())
    if len(detectors) != 5 or len({d["checkpoint_sha256"] for d in detectors}) != 1:
        raise ValueError("Frozen detector membership differs.")
    checkpoint_sha = detectors[0]["checkpoint_sha256"]
    args_file = BASE / "training/yolov8s/args.yaml"
    if sha256(args_file) != read(BASE / "training/checkpoint.json")["ultralytics_args_SHA256"]:
        raise ValueError("Historical training arguments differ.")
    args = yaml.safe_load(args_file.read_text(encoding="utf-8-sig"))
    request = out / "frozen_profile_request.json"
    write(request, {"checkpoints": [{"id": detectors[0]["checkpoint"], "path": str(CHECKPOINT),
        "sha256": checkpoint_sha, "resolutions": sorted({d["resolution"] for d in detectors})}]})
    profile(request, out / "frozen_cpu_operation_profile.json")
    result = read(out / "frozen_cpu_operation_profile.json")
    by_size = {x["resolution"]: x for x in result["profiles"]}
    existing_path = COMPLIANCE / "tested_detector_attributes.csv"
    with existing_path.open(encoding="utf-8-sig", newline="") as stream:
        existing = list(csv.DictReader(stream))
    previous = read(PILOT / "result.json")
    reference = previous["gpu_measurements"]
    if sha256(PILOT / "measurements.csv") != previous["measurements_csv_SHA256"]:
        raise ValueError("Old latency receipt changed.")
    input_protocol = {"batch": 1, "warmup_calls": 3, "input": "RGB BCHW square LetterBox; /255",
        "conf": .001, "nms_iou": .7, "max_det": 3000, "max_nms": 30000, "multi_label": True,
        "class_aware": True, "nms_deadline": None, "head": "dense", "TF32": False,
        "strategy": "FullFrame", "SAHI": False, "TTA": False}
    queue, rows = [], []
    for detector in detectors:
        key = detector["detector_id"]
        dimensions = by_size[detector["resolution"]]
        exact_sha_matches = [r for r in existing if r["checkpoint_sha256"] == detector["checkpoint_sha256"]
                             and r["resolution"] == str(detector["resolution"])
                             and r["precision"] == detector["precision"]]
        # Development profiles and prediction-only clocks cannot supply formal pair measurements.
        queue.append({"D_ID": key, "measure": ["cached_latency_and_CI", "peak_VRAM", "wall_cost", "prediction_bytes"],
                      "same_checkpoint_development_rows": len(exact_sha_matches), "status": "AWAITING_ADMISSION"})
        rows.append({**detector, "family": "YOLOv8s", "train_resolution": args["imgsz"],
            "train_epochs_logged": read(BASE / "training/checkpoint.json")["actual_epochs"],
            "train_seed": args["seed"], "train_batch": args["batch"],
            "checkpoint_parameters": dimensions["checkpoint_parameters"], "GFLOPs_2xMAC": dimensions["GFLOPs_2xMAC"],
            "op_count_scope": "CPU FP32 topology; same operation count for FP16; not measured GPU throughput",
            "inference": input_protocol, "matched_gpu_attributes": None})
    archived = read(AUDIT / "static_gap.json")
    if len(archived["leaves"]) != 25 or {k: x["covered"] for k,x in archived["definitions"].items()} != {
        "exact_identity": 1, "conservative": 3, "supervisor_proxy": 18}:
        raise ValueError("Static rule evidence differs.")
    write(out / "five_detector_attributes.json", {"detectors": rows, "gpu_measurement_queue": queue,
        "GPU_seconds_this_profile": 0, "measured_scope": "CPU operation count only",
        "comparison_sources": {"tested_detector_attributes_sha256": sha256(existing_path),
            "prior_run": previous["RunID"], "prior_SHA256": sha256(PILOT / "result.json"),
            "prior_GPU": {k: reference[k] for k in ("scope", "imgsz", "precision", "source_image_count",
                    "latency_ms_per_image_mean", "CUDA_event_ms_mean", "peak_pytorch_allocated_bytes",
                    "peak_pytorch_reserved_bytes")},
            "prior_compatibility": "Same checkpoint at FP32 1280, 16x2 prediction-only measurements, distinct timing boundary"},
        "training_args_sha256": sha256(args_file), "training_logged_ultralytics": read(BASE / "manifest.json")["system"]["ultralytics"],
        "augment_basis": "Ultralytics 8.4.33", "installed_ultralytics": importlib.metadata.version("ultralytics"),
        "static_policies": {k: x["covered"] for k, x in archived["definitions"].items()},
        "static_leaves": archived["leaves"], "static_source_sha256": sha256(AUDIT / "static_gap.json"),
        "historical_albumentations_activation": "UNKNOWN",
        "profile_sha256": sha256(out / "frozen_cpu_operation_profile.json")})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output.resolve(strict=True))
