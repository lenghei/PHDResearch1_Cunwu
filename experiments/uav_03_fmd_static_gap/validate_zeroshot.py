"""Measure four COCO checkpoints on the VisDrone common-six class subset."""
import argparse
from copy import deepcopy
import csv
from pathlib import Path
import os
import sys

from validate_checkpoints import (CACHE, CACHE_SHA, HERE, MAIN, PROTOCOL, VAL_SHA, FILES,
                                  digest, disk_guard, healthy_state, input_set, load, output_folder,
                                  replay, require, run_cell, verify, write_json)
from main.dataset import VisDroneZIP
from main.labels import COCO_TO_COMMON, COMMON_NAMES, ground_truth
from uav_reproduction import VAL
from model import declared_profile_configuration

MODELS = {
    "yolov8n": ("YOLOv8", "Nano", "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36"),
    "yolov8s": ("YOLOv8", "Small", "1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a"),
    "yolo26n": ("YOLO26", "Nano", "9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef"),
    "yolo26s": ("YOLO26", "Small", "646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b"),
}
ZERO = {**PROTOCOL, "classes": "common6", "weights_state": "zero_shot", "detector_classes": sorted(COCO_TO_COMMON),
        "gt_matches": "strict common-six; unsupported VisDrone boxes join the ignore union"}
BUDGET = 1200


def weight(name):
    return MAIN / "code/models/zero_shot" / name / "local" / (name + ".pt")


def common_inputs(cache, val):
    require(digest(val) == VAL_SHA, "Original VisDrone val ZIP differs.")
    native, rows, _, _ = input_set(cache, cache / "ground_truth.json")
    records = []
    archive = VisDroneZIP(val, 548)
    try:
        for i, member in enumerate(rows):
            _, record, proof = archive.record(i)
            require(all(proof[k] == member[k] for k in proof)
                    and record["id"] == native["images"][i]["id"], "ZIP and Normal cache differ.")
            records.append(record)
    finally:
        archive.close()
    gt, regions = ground_truth(records, "common6")
    require(len(gt["annotations"]) == 27760 and len(gt["images"]) == 548
            and [c["name"] for c in gt["categories"]] == COMMON_NAMES, "Common-six GT differs.")
    return gt, rows, records, regions


def worker(out, cache, inputs):
    import torch
    import ultralytics
    from thop import profile
    from ultralytics import YOLO
    from main.predictor import IEEEDetectionPredictor

    class Traced(IEEEDetectionPredictor):
        def preprocess(self, images):
            tensor = super().preprocess(images)
            self.trace = (list(tensor.shape), str(tensor.dtype))
            return tensor

    require(ultralytics.__version__ == ZERO["ultralytics"] and torch.cuda.is_available()
            and torch.cuda.mem_get_info(0)[0] >= 3*2**30, "Runtime or GPU admission failed.")
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    write_json(out / "environment.json", {"gpu": torch.cuda.get_device_name(0),
               "torch": torch.__version__, "ultralytics": ultralytics.__version__, "cuda": torch.version.cuda})
    entries, summaries, classes = [], [], []
    for name, (architecture, scale, sha) in MODELS.items():
        disk_guard(out)
        path = weight(name)
        require(digest(path) == sha, f"Unexpected pretrained checkpoint: {name}")
        detector = YOLO(path)
        names = detector.names
        require(detector.task == "detect" and len(names) == 80
                and all(names[i] == COMMON_NAMES[k-1] for i,k in COCO_TO_COMMON.items()),
                "Expected COCO detection classes.")
        yaml = detector.model.yaml
        require(yaml.get("scale") == scale[0].lower()
                or Path(str(yaml.get("yaml_file", ""))).stem == name
                or (architecture == "YOLOv8" and (yaml.get("depth_multiple"), yaml.get("width_multiple")) ==
                    {"Nano": (.33, .25), "Small": (.33, .5)}[scale]), f"Checkpoint scale differs: {name}")
        require(bool(getattr(detector.model, "end2end", False)) == (architecture == "YOLO26"),
                "Native head family differs.")
        healthy_state(detector.model, torch)
        settings = detector.ckpt.get("train_args") or {}
        net = deepcopy(detector.model).cpu().float().eval()
        if hasattr(net,"end2end"):
            net.end2end = False
        net.fuse(verbose=False)
        with torch.inference_mode():
            macs, _ = profile(net, inputs=(torch.zeros(1,3,640,640),), verbose=False)
        del net
        require(macs > 0, f"Could not compute operation count: {name}")
        fm = declared_profile_configuration(architecture, scale, "COCO_ZeroShot", "Train640",
                                            640, "FP32", "FullFrame", "NMS")
        entries.append({"checkpoint": name, "sha256": sha, "architecture": architecture, "scale": scale,
                        "pretrained_data_argument": settings.get("data"),
                        "checkpoint_training_imgsz": settings.get("imgsz"),
                        "checkpoint_training_epochs_argument": settings.get("epochs"),
                        "checkpoint_optimizer_argument": settings.get("optimizer"),
                        "checkpoint_seed_argument": settings.get("seed"),
                        "saved_ultralytics": detector.ckpt.get("version"),
                        "VisDrone_finetuning": "NONE_IN_THIS_PRETRAINED_CHECKPOINT",
                        "FM_D_train_leaf": "Train640", "FM_D_configuration_id": fm["configuration_id"]})
        entry = {"checkpoint": name, "sha256": sha, "cohort": "COCO_ZeroShot",
                 "parameters": sum(p.numel() for p in detector.model.parameters())}
        torch.cuda.empty_cache()
        summary = run_cell(out, cache, inputs, entry, detector, 640, torch, Traced, ZERO)
        summaries.append({**summary, "classes": "common6", "GFLOPs_2xMAC": 2 * macs / 1e9,
                          "FM_D_configuration_id": fm["configuration_id"]})
        metrics = load(out / name / "infer640/metrics.json")["per_class"]
        classes.extend({"checkpoint": name, **r} for r in metrics if r["size"] in ("all","tiny"))
        del detector
    for filename, table in (("checkpoint_metadata.csv", entries), ("zeroshot_metrics.csv", summaries),
                            ("zeroshot_per_class.csv", classes)):
        with (out / filename).open("x", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, list(table[0]))
            writer.writeheader()
            writer.writerows(table)
    write_json(out / "result.json", {"status":"COMPLETE", "RunID":out.name, "models":4, "cells":4,
               "images_per_cell":548, "image_evaluations":2192, "scope":"DEVELOPMENT_VALIDATION_COMMON6"})


def main():
    from supervisor import guarded_run, sample
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--val", type=Path, default=VAL)
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--replay", type=Path)
    args = p.parse_args()
    cache = args.cache.resolve(strict=True)
    val = args.val.resolve(strict=True)
    inputs = common_inputs(cache, val)
    if args.replay:
        case = args.replay.resolve(strict=True)
        binding = load(case / "configuration.json")
        verify(case, FILES, {"checkpoint": binding["checkpoint"], "resolution": 640, "images":548})
        require(binding["protocol"] == ZERO and replay(case/"predictions.jsonl.gz", inputs, ZERO)
                == load(case/"metrics.json"), "CPU replay differs.")
        print("CPU replay matched the common-six metrics.")
        return
    if args.worker:
        out = args.worker.resolve(strict=True)
        require(os.environ.get("UAVGAP_ZERO") == str(out) and load(out/"ground_truth.json") == inputs[0],
                "Run must use the guarded entry and the recorded common-six GT.")
        worker(out,cache,inputs)
        return
    for name, (_,_,sha) in MODELS.items():
        require(digest(weight(name)) == sha, f"Verify local weights: {name}")
    out = output_folder(HERE.name)
    print(out, flush=True)
    write_json(out/"ground_truth.json", inputs[0])
    write_json(out/"inputs.json", {"protocol":ZERO, "cache":str(cache), "members_sha256":CACHE_SHA,
               "val_release_sha256":VAL_SHA, "ground_truth_sha256":digest(out/"ground_truth.json"),
               "checkpoints":{name: model[2] for name,model in MODELS.items()}})
    runtime = out/"runtime"
    (runtime/"Ultralytics").mkdir(parents=True)
    env = dict(os.environ, UAVGAP_ZERO=str(out), YOLO_AUTOINSTALL="false", YOLO_OFFLINE="true",
               YOLO_CONFIG_DIR=str(runtime), PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(out),
               "--cache", str(cache), "--val", str(val)]
    files = (Path(__file__), HERE/"validate_checkpoints.py", MAIN/"code/main/labels.py",
             MAIN/"code/main/statistics.py", MAIN/"code/main/predictor.py")
    write_json(out/"execution.json", {"command":command, "budget_seconds":BUDGET,
               "source_sha256":{f.relative_to(MAIN).as_posix():digest(f) for f in files}})
    guarded_run(command,out,BUDGET,env,sample()["uuid"])
    require(load(out/"result.json")["status"] == "COMPLETE", "Incomplete zero-shot batch.")


if __name__ == "__main__":
    main()
