"""Compare local native-ten checkpoints on the full VisDrone development split."""
import argparse
import csv
import gzip
from pathlib import Path
from statistics import mean, median
import os
import sys
from time import perf_counter

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[1]
sys.path[:0] = [str(MAIN / "code/tools"), str(MAIN / "code"),
                str(MAIN / "experiments/uav_05_evaluation")]
from uav_reproduction import RUNS, digest, output_folder, write_json
from pipeline import attempt, canonical, disk_guard, load, prediction_rows, require, seal, verify
from main.labels import VIS_NAMES, mapped_predictions
from main.statistics import evaluate_coco
from reuse_checkpoints import healthy_state
from summarize_profile import percentile95

CATALOG = MAIN / "code/models/catalog.csv"
CACHE = RUNS / HERE.name / "reproduce_20261125T122911_30a8b769/cache/p_cdbfeb94b010c58d"
CACHE_SHA = "0d17bbfd35fc28fece0857f2496f05389f1cc3c5480978cdd77a444af4a238d6"
GT_SHA = "fcc31d0ed104cd2e682898cff78bc94fd4e2df4e7ecfba786616004fc2327de3"
VAL_SHA = "abeea063037e5d20398837deb11084e652402a34ddf4f207bdf541a6f2a35ef9"
BUDGET = 1500
FILES = ("configuration.json", "predictions.jsonl.gz", "timing.json", "metrics.json", "summary.json")
PROTOCOL = {"dataset": "VisDrone2019-DET-val", "scope": "DEVELOPMENT_VALIDATION", "condition": "Normal",
            "images": 548, "precision": "FP32", "batch": 1, "warmups": 3, "classes": "native10",
            "letterbox": "square", "conf": .001, "iou": .7, "max_det": 3000, "max_nms": 30000,
            "multi_label": True, "class_aware": True, "NMS_deadline": "disabled",
            "head": "one-to-many", "tf32": False, "ignore_union": .5,
            "tiny_area": "original bbox area < 256 px2", "ultralytics": "8.4.61",
            "AP_iou": ".50:.05:.95", "recall_points": 101}


def input_set(cache, gt_file):
    import json

    receipt = load(cache / "set_receipt.json")
    require(receipt["leaves"] == ["Normal"] and receipt["status"] == "RENDER_VERIFIED_DEVELOPMENT_VAL"
            and receipt["source_archive_sha256"] == VAL_SHA and receipt["members_sha256"] == CACHE_SHA
            and receipt["ground_truth_sha256"] == GT_SHA
            and digest(cache / "members.jsonl") == CACHE_SHA and digest(gt_file) == GT_SHA,
            "Frozen Normal cache or GT identity differs.")
    rows = [json.loads(line) for line in (cache / "members.jsonl").read_bytes().splitlines()]
    gt = load(gt_file)
    images = gt["images"]
    require(len(rows) == len(images) == len({r["image_id"] for r in rows}) == PROTOCOL["images"]
            and list(gt["categories"]) == [{"id": i, "name": name} for i, name in enumerate(VIS_NAMES, 1)]
            and len(gt["annotations"]) == 38593, "Expected complete native-ten validation set.")
    records = []
    regions = {}
    for r, g in zip(rows, images):
        require((r["image_id"], r["output_png"], r["shape"]) ==
                (g["id"], g["file_name"], [g["height"], g["width"], 3]), "GT/cache identity mismatch.")
        records.append({"id": g["id"], "width": g["width"], "height": g["height"]})
        regions[g["id"]] = r["ignore_regions"] + r["other_regions"]
    return gt, rows, records, regions


def image_from_cache(cache, row):
    import cv2
    import numpy as np
    from hashlib import sha256

    path = (cache / row["output_png"]).resolve(strict=True)
    require(path.is_relative_to(cache), "Cache filename escapes the Normal directory.")
    payload = path.read_bytes()
    require(sha256(payload).hexdigest() == row["png_sha256"], "Cached PNG differs.")
    pixels = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    require(pixels is not None and list(pixels.shape) == row["shape"]
            and sha256(pixels.tobytes()).hexdigest() == row["pixel_sha256"], "Cached pixel data differs.")
    return pixels


def catalog():
    with CATALOG.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    require(len(rows) == len({r["checkpoint"] for r in rows}) == len({r["sha256"] for r in rows}) == 6,
            "Checkpoint catalog identities differ.")
    return rows


def source_files(entry):
    base = CATALOG.parent
    path = (base / entry["weight_file"]).resolve(strict=True)
    require(path.is_relative_to(base) and digest(path) == entry["sha256"], "Checkpoint SHA mismatch.")
    for name, key in (("args.yaml", "args_sha256"), ("training_results.csv", "training_log_sha256")):
        require(digest(path.with_name(name)) == entry[key], f"Training evidence differs: {name}")
    return path


def replay(path, inputs, protocol=PROTOCOL):
    import json

    gt, rows, records, regions = inputs
    raw = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for r in rows:
            line = json.loads(f.readline())
            require(line["image_id"] == r["image_id"]
                    and all(p["image_id"] == r["image_id"] for p in line["predictions"]),
                    "Prediction and GT image order differ.")
            raw.extend(line["predictions"])
        require(not f.read(1), "Prediction file contains extra image records.")
    detections = mapped_predictions(raw, records, regions, protocol["classes"],
                                    protocol.get("weights_state", "finetuned"), protocol["max_det"])
    per_area, per_class, _ = evaluate_coco(gt, detections, max_det=protocol["max_det"])
    return {"per_area": per_area, "per_class": per_class, "raw_predictions": len(raw),
            "evaluated_predictions": len(detections)}


def run_cell(out, cache, inputs, entry, detector, size, torch, predictor, protocol=PROTOCOL):
    def infer(image):
        with torch.inference_mode():
            results = detector.predict(image, predictor=predictor, imgsz=size,
                                       batch=1, device="0", half=False, conf=protocol["conf"],
                                       iou=protocol["iou"], max_det=protocol["max_det"],
                                       end2end=False, rect=False, agnostic_nms=False, classes=protocol.get('detector_classes'),
                                       augment=False, save=False, verbose=False)
        require(len(results) == 1 and results[0].orig_shape == image.shape[:2],
                "Prediction geometry differs.")
        return results[0].boxes.data, detector.predictor.trace

    return measure_cell(out, cache, inputs, entry, size, torch, infer, protocol)


def measure_cell(out, cache, inputs, entry, size, torch, infer, protocol):
    key = entry["checkpoint"]
    final = out / key / f"infer{size}"
    gt = inputs[0]
    gt_sha = digest(out / "ground_truth.json")
    with attempt(final) as folder:
        binding = {"run_id": out.name, "checkpoint": key, "checkpoint_sha256": entry["sha256"],
                   "resolution": size, "protocol": protocol, "gt_sha256": gt_sha,
                   "members_sha256": CACHE_SHA}
        write_json(folder / "configuration.json", binding)

        def checked(image):
            boxes, trace = infer(image)
            require(trace == ([1, 3, size, size], "torch.float32"), "Input tensor differs.")
            torch.cuda.synchronize()
            return boxes, trace

        rows = inputs[1]
        start = perf_counter()
        warm = image_from_cache(cache, rows[0])
        for _ in range(protocol["warmups"]):
            checked(warm)
        warm_seconds = perf_counter() - start
        torch.cuda.reset_peak_memory_stats()
        elapsed, raw_count = [], 0
        predict_started = perf_counter()
        with gzip.open(folder / "predictions.jsonl.gz", "xb", compresslevel=2) as stream:
            for i, row in enumerate(rows, 1):
                tick = perf_counter()
                pixels = image_from_cache(cache, row)
                ready = perf_counter()
                boxes, trace = checked(pixels)
                synchronized = perf_counter()
                raw = prediction_rows(row["image_id"], boxes.detach().cpu().numpy())
                extracted = perf_counter()
                stream.write(canonical({"image_id": row["image_id"], "predictions": raw}) + b"\n")
                elapsed.append({"image_id": row["image_id"], "pixel_sha256": row["pixel_sha256"],
                                "NCHW": trace[0], "dtype": trace[1],
                                "detector_ms": (synchronized-ready)*1000,
                                "pipeline_ms": (extracted-tick)*1000, "predictions": len(raw)})
                raw_count += len(raw)
                if i % 150 == 0 or i == len(rows):
                    print(f"{key} / {size}: {i}/{len(rows)} images", flush=True)
        inference_seconds = perf_counter() - predict_started
        latency = {}
        for field in ("detector", "pipeline"):
            values = [r[f"{field}_ms"] for r in elapsed]
            latency.update({f"{field}_mean_ms": mean(values), f"{field}_median_ms": median(values),
                            f"{field}_p95_ms": percentile95(values)})
        timing = {"samples": elapsed, **latency, "warm_seconds": warm_seconds,
                  "inference_write_seconds": inference_seconds,
                  "peak_allocated_MiB": torch.cuda.max_memory_allocated() / 2**20,
                  "peak_reserved_MiB": torch.cuda.max_memory_reserved() / 2**20}
        write_json(folder / "timing.json", timing)
        begin = perf_counter()
        metrics = replay(folder / "predictions.jsonl.gz", inputs, protocol)
        cpu_seconds = perf_counter() - begin
        areas = {r["size"]: r for r in metrics["per_area"]}
        require(areas["all"]["image_count"] == len(rows) and areas["all"]["gt_count"] == len(gt["annotations"])
                and raw_count == metrics["raw_predictions"], "COCO population differs.")
        write_json(folder / "metrics.json", metrics)
        summary = {"RunID": out.name, "checkpoint": key, "cohort": entry["cohort"],
                   "checkpoint_sha256": entry["sha256"], "resolution": size, "images": 548,
                   "parameters": int(entry["parameters"]),
                   "AP50": areas["all"]["map50"], "AP50_95": areas["all"]["map50_95"],
                   "AR50_95": areas["all"]["ar50_95"],
                   **{label + "_AP": areas[label.lower()]["map50_95"]
                      for label in ("Tiny", "Small", "Medium", "Large")},
                   "raw_predictions": raw_count, "evaluated_predictions": metrics["evaluated_predictions"],
                   **latency, "peak_allocated_MiB": timing["peak_allocated_MiB"],
                   "peak_reserved_MiB": timing["peak_reserved_MiB"],
                   "inference_write_seconds": inference_seconds, "evaluation_seconds": cpu_seconds}
        write_json(folder / "summary.json", summary)
        seal(folder, FILES, {"status": "COMPLETE", "checkpoint": key, "resolution": size, "images": 548})
    verify(final, FILES, {"checkpoint": key, "resolution": size, "images": 548})
    print(f"{key} / {size}: AP50:95 {100 * summary['AP50_95']:.2f}% | COCO {cpu_seconds:.1f}s", flush=True)
    return summary


def worker(out, cache, selection, sizes):
    import torch
    import ultralytics
    from ultralytics import YOLO
    from main.predictor import IEEEDetectionPredictor

    class TracedPredictor(IEEEDetectionPredictor):
        def preprocess(self, images):
            tensor = super().preprocess(images)
            self.trace = (list(tensor.shape), str(tensor.dtype))
            return tensor

    require(torch.cuda.is_available() and ultralytics.__version__ == PROTOCOL["ultralytics"],
            "CUDA or pinned Ultralytics is unavailable.")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    require(torch.cuda.mem_get_info(0)[0] >= 3 * 2**30, "Insufficient free VRAM.")
    write_json(out / "environment.json", {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "cuda": torch.version.cuda, "ultralytics": ultralytics.__version__, "python": sys.version})
    frozen = load(out / "inputs.json")
    require(digest(CATALOG) == frozen["catalog_sha256"] and selection == frozen["models"]
            and sizes == frozen["resolutions"], "Worker inputs differ from the recorded plan.")
    inputs = input_set(cache, out / "ground_truth.json")
    summaries = []
    for entry in catalog():
        if entry["checkpoint"] not in selection:
            continue
        disk_guard(out)
        detector = YOLO(str(source_files(entry)))
        names = detector.names
        head = detector.model.model[-1]
        require(detector.task == "detect" and [names[i] for i in range(len(names))] == VIS_NAMES
                and head.nc == 10 and all(hasattr(head, layer) for layer in ("cv2", "cv3"))
                and sum(p.numel() for p in detector.model.parameters()) == int(entry["parameters"]),
                "Expected a compatible native-ten detector.")
        healthy_state(detector.model, torch)
        torch.cuda.empty_cache()
        for size in sizes:
            summaries.append(run_cell(out, cache, inputs, entry, detector, size, torch, TracedPredictor))
        del detector
    require(len(summaries) == len(selection)*len(sizes), "Incomplete checkpoint matrix.")
    with (out / "validation_metrics.csv").open("x", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    write_json(out / "result.json", {"RunID": out.name, "status": "COMPLETE", "models": len(selection),
               "cells": len(summaries), "images_per_cell": 548,
               "image_evaluations": 548*len(summaries), "scope": PROTOCOL["scope"]})


def main():
    import shutil
    from supervisor import guarded_run, sample

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--model", action="append", choices=[r["checkpoint"] for r in catalog()])
    parser.add_argument("--size", type=int, action="append", choices=(640, 1280))
    parser.add_argument("--replay", type=Path, help="Recheck one completed model/resolution on CPU.")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    cache = args.cache.resolve(strict=True)
    selection = [r["checkpoint"] for r in catalog() if not args.model or r["checkpoint"] in args.model]
    sizes = sorted(set(args.size or (640, 1280)))
    if args.replay is not None:
        cell = args.replay.resolve(strict=True)
        binding = load(cell / "configuration.json")
        verify(cell, FILES, {"checkpoint": binding["checkpoint"], "resolution": binding["resolution"], "images": 548})
        inputs = input_set(cache, cell.parents[1] / "ground_truth.json")
        require(replay(cell / "predictions.jsonl.gz", inputs) == load(cell / "metrics.json"),
                "Saved COCO metrics do not reproduce.")
        print(f"CPU replay matched {binding['checkpoint']} / {binding['resolution']}.")
        return
    if args.worker:
        out = args.worker.resolve(strict=True)
        require(os.environ.get("UAVGAP_CHECKPOINT_VALIDATION") == str(out), "Launch the supervised entry point.")
        worker(out, cache, selection, sizes)
        return
    input_set(cache, cache / "ground_truth.json")
    for row in catalog():
        if row["checkpoint"] in selection:
            source_files(row)
    out = output_folder(HERE.name)
    print(out, flush=True)
    shutil.copyfile(cache / "ground_truth.json", out / "ground_truth.json")
    shutil.copyfile(CATALOG, out / "checkpoint_catalog.csv")
    runtime = out / "runtime"
    (runtime / "Ultralytics").mkdir(parents=True)
    env = dict(os.environ, UAVGAP_CHECKPOINT_VALIDATION=str(out), YOLO_CONFIG_DIR=str(runtime),
               PYTHONDONTWRITEBYTECODE="1", YOLO_AUTOINSTALL="false", YOLO_OFFLINE="true")
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(out), "--cache", str(cache)]
    for key, values in (("--model", selection), ("--size", sizes)):
        for val in values:
            command += [key, str(val)]
    write_json(out / "inputs.json", {"models": selection, "resolutions": sizes, "catalog_sha256": digest(CATALOG),
               "cache": str(cache), "members_sha256": CACHE_SHA, "gt_sha256": GT_SHA, "ZIP_sha256": VAL_SHA,
               "protocol": PROTOCOL})
    code = [Path(__file__), MAIN / "code/main/predictor.py", MAIN / "code/main/statistics.py",
            MAIN / "code/main/labels.py", MAIN / "experiments/uav_05_evaluation/supervisor.py",
            MAIN / "experiments/uav_05_evaluation/pipeline.py"]
    write_json(out / "execution.json", {"command": command, "budget_seconds": BUDGET,
                                       "source_sha256": {p.relative_to(MAIN).as_posix(): digest(p) for p in code}})
    guarded_run(command, out, BUDGET, env, sample()["uuid"])
    require(load(out / "result.json")["status"] == "COMPLETE", "Worker did not complete.")


if __name__ == "__main__":
    main()
