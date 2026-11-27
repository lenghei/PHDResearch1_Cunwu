"""Validate archived LEAF-YOLO detectors on the Week 3 VisDrone population."""
import argparse
import csv
from importlib import metadata
import os
from pathlib import Path
import shutil
import sys

from validate_checkpoints import (CACHE, CACHE_SHA, GT_SHA, HERE, MAIN, PROTOCOL, VAL_SHA, VIS_NAMES,
                                  digest, disk_guard, healthy_state, image_from_cache, input_set, load,
                                  measure_cell, output_folder, require, write_json)

ASSETS = MAIN / "code/models/leaf"
RUNTIME = MAIN.parent / "datasets/research_archive/leaf_yolo_runtime"
MODELS = {
    "external_leaf_yolo_n": ("1b1e088578f45ae981d8f3f03900b964b97711ac67ff8088565647cd7fac89e5",
                             "FM_D_LEAF_NANO", 1201996, 1195822),
    "external_leaf_yolo_s": ("9bc03882ea4437e38e4da7574a0618a2dae0ab62fe09979503406d2217df99d6",
                             "EXTERNAL_LEAF_SMALL_OUTSIDE_FM_D", 4296136, 4284044),
}
LEAF_PROTOCOL = {**PROTOCOL, "backend": "archived LEAF-YOLO / YOLOv7 IDetect",
                 "head": "IDetect; objectness times class score; legacy multi-label NMS",
                 "ultralytics": "not_used"}
BUDGET = 1500


def check_runtime(runtime):
    lock = load(ASSETS / "runtime_sha256.json")
    for filename, checksum in lock.items():
        path = (runtime / filename).resolve(strict=True)
        require(path.is_relative_to(runtime) and digest(path) == checksum, f"LEAF runtime differs: {filename}")


def load_leaf(name, torch):
    from new_leaf_ieee_adapter import normalized_names

    sha, cohort, parameters, deployed = MODELS[name]
    path = ASSETS / name / "local/best.pt"
    require(digest(path) == sha, f"LEAF weight SHA mismatch: {name}")
    source = load(path.with_name("source.json"))
    require(source["sha256"] == sha and source["checkpoint"] == name, "LEAF weight source differs.")
    for filename, item in source["training_logs"].items():
        require(digest(path.with_name(filename)) == item["sha256"], f"LEAF training record differs: {name}/{filename}")
    packet = torch.load(path, map_location="cpu", weights_only=False)
    model = packet["ema"] if packet.get("ema") is not None else packet["model"]
    model = model.float().eval()
    require(normalized_names(model) == VIS_NAMES and type(model.model[-1]).__name__ == "IDetect"
            and model.model[-1].nc == 10 and sum(p.numel() for p in model.parameters()) == parameters,
            f"Expected native-ten LEAF architecture: {name}")
    healthy_state(model, torch)
    entry = {"checkpoint": name, "sha256": sha, "cohort": cohort, "parameters": parameters,
             "deployed_parameters": deployed}
    return model, entry


def predictor(model, size, torch, device):
    import numpy as np
    from new_leaf_ieee_adapter import leaf_nms
    from utils.datasets import letterbox
    from utils.general import scale_coords

    stride = max(int(model.stride.max()), 32)

    def infer(image):
        padded = letterbox(image, new_shape=(size, size), auto=False, stride=stride)[0]
        rgb = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1))
        tensor = torch.from_numpy(rgb).unsqueeze(0).to(device=device, dtype=torch.float32) / 255
        with torch.inference_mode():
            raw = model(tensor, augment=False)[0]
            require(raw.ndim == 3 and raw.shape[-1] == 15 and raw.dtype == torch.float32
                    and bool(torch.isfinite(raw).all()), "Invalid LEAF head output.")
            boxes = leaf_nms(raw, LEAF_PROTOCOL["conf"], LEAF_PROTOCOL["iou"],
                             LEAF_PROTOCOL["max_det"], max_nms=LEAF_PROTOCOL["max_nms"])[0]
            if len(boxes):
                scale_coords(tensor.shape[-2:], boxes[:, :4], image.shape)
        return boxes, (list(tensor.shape), str(tensor.dtype))

    return infer


def fused(model, entry, torch):
    model = model.fuse().eval()
    healthy_state(model, torch)
    require(sum(p.numel() for p in model.parameters()) == entry["deployed_parameters"],
            "Unexpected fused parameter count.")
    return model


def workload(out, cache, runtime, names, sizes, check):
    import torch
    import torchvision

    torch.set_num_threads(2)
    check_runtime(runtime)
    sys.path.insert(0, str(runtime))
    from models import yolo
    require(Path(yolo.__file__).resolve().is_relative_to(runtime), "LEAF model module comes from another runtime.")
    inputs = input_set(cache, out / "ground_truth.json")
    profile, rows = [], []
    if not check:
        require(torch.cuda.is_available() and torch.cuda.mem_get_info(0)[0] >= 3 * 2**30,
                "CUDA needs at least 3 GiB of free memory.")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    write_json(out / "environment.json", {"device": "CPU" if check else torch.cuda.get_device_name(0),
        "torch": torch.__version__, "torchvision": torchvision.__version__, "cuda": torch.version.cuda,
        "timm": metadata.version("timm"), "THOP": metadata.version("ultralytics-thop"),
        "python": sys.version, "runtime_sha256": digest(ASSETS / "runtime_sha256.json")})
    for name in names:
        model, entry = load_leaf(name, torch)
        if check:
            from copy import deepcopy
            from thop import profile as thop_profile

            for size in sizes:
                with torch.inference_mode():
                    macs, _ = thop_profile(deepcopy(model), inputs=(torch.zeros(1, 3, size, size),), verbose=False)
                require(macs > 0, "THOP returned no operations.")
                profile.append({**entry, "resolution": size, "GFLOPs_2xMAC": 2 * macs / 1e9})
        model = fused(model, entry, torch)
        if check:
            for size in sizes:
                boxes, trace = predictor(model, size, torch, "cpu")(image_from_cache(cache, inputs[1][0]))
                require(trace == ([1, 3, size, size], "torch.float32") and bool(torch.isfinite(boxes).all()),
                        "CPU canary failed.")
                print(f"{name} / {size}: CPU forward, geometry and NMS passed", flush=True)
        else:
            model = model.to("cuda:0")
            torch.cuda.empty_cache()
            for size in sizes:
                rows.append(measure_cell(out, cache, inputs, entry, size, torch,
                                         predictor(model, size, torch, "cuda:0"), LEAF_PROTOCOL))
            del model
    if check:
        write_json(out / "operation_profile.json", {"status": "PASS", "profiles": profile})
    else:
        require(len(rows) == len(names)*len(sizes), "Incomplete LEAF matrix.")
        with (out / "validation_metrics.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        write_json(out / "result.json", {"RunID": out.name, "status": "COMPLETE", "models": len(names),
                   "cells": len(rows), "images_per_cell": 548, "image_evaluations": 548*len(rows),
                   "scope": LEAF_PROTOCOL["scope"]})


def main():
    from supervisor import guarded_run, sample

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=RUNTIME)
    parser.add_argument("--cache", type=Path, default=CACHE)
    parser.add_argument("--model", choices=MODELS, action="append")
    parser.add_argument("--size", choices=(640, 1280), type=int, action="append")
    parser.add_argument("--check", action="store_true", help="Check only, using CPU and THOP.")
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    runtime = args.runtime.resolve(strict=True)
    cache = args.cache.resolve(strict=True)
    names = [name for name in MODELS if not args.model or name in args.model]
    sizes = sorted(set(args.size or (640, 1280)))
    if args.worker:
        out = args.worker.resolve(strict=True)
        require(os.environ.get("UAVGAP_LEAF_VALIDATION") == str(out), "Launch the supervised entry point.")
        recorded = load(out / "inputs.json")
        require(recorded["models"] == names and recorded["resolutions"] == sizes
                and recorded["runtime"] == str(runtime) and recorded["cache"] == str(cache),
                "Worker inputs differ from the recorded plan.")
        workload(out, cache, runtime, names, sizes, False)
        return

    check_runtime(runtime)
    input_set(cache, cache / "ground_truth.json")
    out = output_folder(HERE.name)
    disk_guard(out)
    print(out, flush=True)
    shutil.copyfile(cache / "ground_truth.json", out / "ground_truth.json")
    write_json(out / "inputs.json", {"models": names, "resolutions": sizes,
               "runtime": str(runtime), "cache": str(cache), "members_sha256": CACHE_SHA,
               "gt_sha256": GT_SHA, "ZIP_sha256": VAL_SHA, "protocol": LEAF_PROTOCOL})
    if args.check:
        workload(out, cache, runtime, names, sizes, True)
        return
    env = dict(os.environ, UAVGAP_LEAF_VALIDATION=str(out), PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(out),
               "--cache", str(cache), "--runtime", str(runtime)]
    for key, values in (("--model", names), ("--size", sizes)):
        for value in values:
            command.extend((key, str(value)))
    common = MAIN / "code/main"
    sources = [Path(__file__), HERE / "validate_checkpoints.py", common / "statistics.py",
               common / "labels.py", MAIN / "experiments/uav_05_evaluation/pipeline.py",
               MAIN / "experiments/uav_05_evaluation/supervisor.py", ASSETS / "runtime_sha256.json"]
    write_json(out / "execution.json", {"command": command, "budget_seconds": BUDGET,
               "source_sha256": {p.relative_to(MAIN).as_posix(): digest(p) for p in sources}})
    guarded_run(command, out, BUDGET, env, sample()["uuid"])
    require(load(out / "result.json")["status"] == "COMPLETE", "LEAF worker did not complete.")


if __name__ == "__main__":
    main()
