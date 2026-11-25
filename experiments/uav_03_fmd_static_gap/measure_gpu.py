"""Measure the five frozen detector settings on a shared development image sample."""
import argparse
import csv
import json
from math import ceil
import os
from pathlib import Path
import statistics
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[1]
sys.path.insert(0, str(MAIN / "code/tools"))
from uav_reproduction import EXPS, RESULTS, RUNS, digest, output_folder, write_json


CACHE = RUNS / "uav_02_fmp/reproduce_20261125T082824_61beb538/p_8f7e990050372f44"
TABLE = RESULTS / HERE.name / "review/frozen_detector_attributes.csv"
COUNT = 32
REPEATS = 2
WARMUPS = 3
BUDGET = 300


def worker(out, cache):
    sys.path.insert(0, str(MAIN / "experiments/uav_05_evaluation"))
    sys.path.insert(0, str(EXPS / "cv261207"))
    from evaluation import Device
    from pipeline import prediction_rows
    import cv2
    import numpy as np
    import ultralytics
    from hashlib import sha256

    receipt = json.loads((cache / "set_receipt.json").read_bytes())
    members = cache / "members.jsonl"
    if receipt["status"] != "RENDER_VERIFIED_DEVELOPMENT_VAL" or digest(members) != receipt["members_sha256"]:
        raise ValueError("The development image manifest differs.")
    rows = [json.loads(line) for line in members.read_bytes().splitlines()]
    if len(rows) != 548 or len({r["image_id"] for r in rows}) != 548:
        raise ValueError("Wrong development population.")
    selection = [rows[i * len(rows) // COUNT] for i in range(COUNT)]
    with TABLE.open(encoding="utf-8", newline="") as stream:
        settings = list(csv.DictReader(stream))
    checkpoints = {row["checkpoint_sha256"] for row in settings}
    if len(settings) != 5 or len(checkpoints) != 1:
        raise ValueError("Wrong detector matrix.")
    settings = [{**r, "resolution": int(r["resolution"])} for r in settings]
    device = Device(checkpoints.pop())
    torch = device.torch

    def load_image(row):
        asset = (cache / row["output_png"]).resolve(strict=True)
        if not asset.is_relative_to(cache.resolve()):
            raise ValueError("Image path escapes the cache.")
        data = asset.read_bytes()
        if sha256(data).hexdigest() != row["png_sha256"]:
            raise ValueError("PNG checksum differs.")
        image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if (image is None or list(image.shape) != row["shape"]
                or sha256(image.tobytes()).hexdigest() != row["pixel_sha256"]):
            raise ValueError("Decoded pixel evidence differs.")
        return image

    write_json(out / "sample.json", {"configuration_id": receipt["configuration_id"],
        "source_archive_sha256": receipt["source_archive_sha256"], "members_sha256": receipt["members_sha256"],
        "images": [{k: r[k] for k in ("image_id", "source_image_sha256", "pixel_sha256") if k in r} for r in selection]})
    summaries = []
    for spec in settings:
        if device.params != int(spec["parameters"]):
            raise ValueError("Checkpoint parameter count differs.")
        predict = device.detector(spec)
        warm_image = load_image(selection[0])
        for _ in range(WARMUPS):
            predict(warm_image)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        measured = []
        for rep in range(REPEATS):
            for row in selection:
                begin = time.perf_counter()
                pixels = load_image(row)
                infer = time.perf_counter()
                result, tensor = predict(pixels)
                torch.cuda.synchronize()
                end = time.perf_counter()
                boxes = result.boxes.data.detach().cpu().numpy()
                predictions = prediction_rows(row["image_id"], boxes)
                extracted = time.perf_counter()
                measured.append({"repeat": rep, "image_id": row["image_id"], "pixel_sha256": row["pixel_sha256"],
                    "batch_shape": tensor["NCHW"], "dtype": tensor["dtype"], "detections": len(predictions),
                    "predict_ms": (end-infer)*1000, "cached_to_boxes_ms": (extracted-begin)*1000})
        ident = spec["detector_id"]
        write_json(out / f"{ident}_timing.json", measured)
        predict_ms = [r["predict_ms"] for r in measured]
        summaries.append({k: spec[k] for k in ("detector_id", "resolution", "precision", "checkpoint_sha256",
                                               "parameters", "gflops_2xmac")} | {
            "images": COUNT, "repeats": REPEATS, "warmups": WARMUPS,
            "predict_mean_ms": statistics.mean(predict_ms), "predict_median_ms": statistics.median(predict_ms),
            "predict_p95_ms": sorted(predict_ms)[ceil(.95 * len(predict_ms))-1],
            "cached_to_boxes_mean_ms": statistics.mean(r["cached_to_boxes_ms"] for r in measured),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
        print(f"{ident}: {COUNT*REPEATS} samples", flush=True)
    write_json(out / "measurements.json", {"status": "COMPLETE", "scope": "VISDRONE_VAL_RESOURCE_PROFILE_NO_AP",
        "checkpoint_sha256": settings[0]["checkpoint_sha256"], "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "cuda": torch.version.cuda, "ultralytics": ultralytics.__version__,
        "configuration_id": receipt["configuration_id"], "results": summaries})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, default=CACHE)
    p.add_argument("--output", type=Path)
    p.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if args.worker is not None:
        if os.environ.get("UAVGAP_W3_PROFILE") != str(args.worker.resolve()):
            p.error("Start the supervised entry point.")
        worker(args.worker, args.cache)
        return
    sys.path.insert(0, str(MAIN / "experiments/uav_05_evaluation"))
    sys.path.insert(0, str(MAIN / "code"))
    from supervisor import guarded_run, sample
    cache = args.cache.resolve(strict=True)
    out = output_folder(HERE.name, args.output)
    print(out, flush=True)
    runtime = out / "runtime"
    (runtime / "Ultralytics").mkdir(parents=True)
    env = dict(os.environ, UAVGAP_W3_PROFILE=str(out), YOLO_CONFIG_DIR=str(runtime),
               PYTHONDONTWRITEBYTECODE="1", YOLO_AUTOINSTALL="false", YOLO_OFFLINE="true")
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(out), "--cache", str(cache)]
    write_json(out / "execution.json", {"command": command, "budget_seconds": BUDGET, "code_sha256": digest(Path(__file__))})
    guarded_run(command, out, BUDGET, env, sample()["uuid"])
    if not (out / "measurements.json").is_file():
        raise RuntimeError("The measurement receipt is missing.")


if __name__ == "__main__":
    main()
