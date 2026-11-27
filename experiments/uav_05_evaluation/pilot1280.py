"""Plan and run the two-checkpoint, five-condition test-dev pilot."""

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
from math import isfinite
import os
from pathlib import Path
import re
import sys
from time import perf_counter
from uuid import uuid4

from run import (
    AUDIT, IMAGES, LEGACY, MAIN, ROOT, RUNS, TEST, Device, approved_evidence,
    bootstrap_coco, code_binding, evaluate_saved, generator, module, runtime, runtime_binding,
)
from admission import verify_review
from archives import OriginalZIP
from pipeline import disk_guard, identity, load, measure_pair, require, sha256, shared_cache, verify_pair, write
from sampling import models
from supervisor import guarded_run

PROTOCOL = "uav_05_testdev_two_checkpoints_infer1280_v1"
P = (
    ("Normal", ("Normal",), "Clean accuracy and cost reference"),
    ("Motion_k21", ("Motion_k21", "Normal"), "Single motion blur"),
    ("Haze_dense", ("Haze_dense", "Normal"), "Single haze"),
    ("JPEG_q25", ("JPEG_q25", "Normal"), "Single JPEG compression"),
    ("LowLight_Shot_low", ("LowLight", "Shot_low"), "Legal joint low-light and shot noise"),
)
D = (
    ("ck05", "YOLOv8s", "ck05_yolov8s_ft1280", 11139470,
     "96c84ece325eccf07edefd231f088f758ba5fa6a75a4b577e8db32a94bfeeb8b"),
    ("ck01", "YOLO26s", "ck01_yolo26s_ft1280", 9955604,
     "1b3861902e01416f199f2242cb43618f7680848cabf5f5df9546f2b404f0c488"),
)
SEED = 20261207
SCOPE = {"LIMITED_TEST_DEV_BENCHMARK", "INDEPENDENT_TEST_DEV"}


def weight(spec):
    path = ROOT / "models/checkpoints" / spec["catalog_directory"] / "best.pt"
    require(sha256(path) == spec["checkpoint_sha256"], "Checkpoint SHA differs.")
    return path


def plan():
    fmp, fmd, _ = models()
    fmp_file = MAIN / "experiments/uav_02_fmp/fm_p.uvl"
    require(sha256(fmp_file) == "e6eadd633d1c14e0d6bc425f5e54421d592d766c65a6e5f20c61bc34dd76fb5c",
            "Original FM_P source differs.")
    attrs = fmp.read_uvl_attributes(LEGACY.parent/"cv261109/fm_p.uvl")
    contexts = [{"name": name, "degradation_id": fmp.configuration_id(leaves),
                 "leaves": list(leaves), "recipe": fmp.recipe(leaves, attrs), "purpose": purpose}
                for name, leaves, purpose in P]
    require(len({row["degradation_id"] for row in contexts}) == 5, "Duplicate pilot degradation.")
    detectors = []
    for checkpoint, family, directory, params, sha in D:
        fm = fmd.declared_profile_configuration(family.rstrip("s"), "Small", "VisDrone_FT",
                                                "Train1280", 1280, "FP32", "FullFrame", "NMS")
        spec = {"detector_id": fm["configuration_id"], "architecture": family, "training_imgsz": 1280,
                "checkpoint": checkpoint, "catalog_directory": directory, "checkpoint_sha256": sha,
                "parameters": params, "resolution": 1280, "precision": "FP32", "leaves": fm["selected_features"]}
        weight(spec)
        detectors.append(spec)
    gate = load(AUDIT/"dataset_gate.json")
    archive = gate["archives"]["test-dev"]
    require(archive["images"] == IMAGES and sha256(AUDIT/"test_dev_members.jsonl")
            == gate["test_dev_member_manifest_sha256"], "Test-dev member ledger differs.")
    pairs = [{"pair_id": p["degradation_id"]+"__"+d["detector_id"], "degradation": p, "detector": d}
             for p in contexts for d in detectors]
    files = code_binding()
    files[Path(__file__).relative_to(MAIN).as_posix()] = sha256(Path(__file__))
    files[fmp_file.relative_to(MAIN).as_posix()] = sha256(fmp_file)
    files["experiments/experiments_test/cv261207/admission.py"] = sha256(LEGACY/"admission.py")
    return {"protocol": PROTOCOL, "split": "VisDrone2019-DET-test-dev", "images": IMAGES,
            "dataset_sha256": archive["sha256"], "members_sha256": gate["test_dev_member_manifest_sha256"],
            "render_seed": SEED, "bootstrap": {"samples": 200, "seed": SEED},
            "thresholds": {"Tiny_AP50_95": .10, "latency_ms": 100},
            "scope": "PURPOSEFUL_BASELINE_PILOT", "formal_pair_credit": 0,
            "image_evaluations": len(pairs)*IMAGES, "pairs": pairs,
            "code_sha256": files, "runtime": runtime_binding()}


class PilotDevice(Device):
    def __init__(self, spec, cuda=True):
        import torch
        from ultralytics import YOLO
        from main.labels import VIS_NAMES
        from main.training_health import model_health

        if cuda:
            require(torch.cuda.is_available() and torch.cuda.mem_get_info(0)[0] >= 3*2**30,
                    "CUDA or free GPU memory is unavailable.")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.torch = torch
        self.model = YOLO(str(weight(spec)))
        net = self.model.model
        params = sum(p.numel() for p in net.parameters())
        require(list(self.model.names.values()) == VIS_NAMES and params == spec["parameters"],
                "Checkpoint architecture or native-ten class order differs.")
        require(bool(getattr(net, "end2end", False)) == (spec["architecture"] == "YOLO26s"),
                "Checkpoint detector head differs.")
        model_health(net)
        self.pristine = deepcopy(net).cpu().float()
        self.predictor_key = None
        self.flops = {}
        self.params = params

    def release(self):
        self.model.predictor = None
        self.model.model = self.pristine
        self.predictor_key = None
        self.torch.cuda.empty_cache()


def available_seconds(gpu):
    return min(gpu["budget_seconds"],
               (datetime.fromisoformat(gpu["expires_utc"])-datetime.now(timezone.utc)).total_seconds())


def admission(spec, science_path, gpu_path):
    science, gpu = load(science_path), load(gpu_path)
    require(science.get("status") == "AUTHOR_AND_SUPERVISOR_APPROVED"
            and science.get("scope") in SCOPE and science.get("protocol") == PROTOCOL
            and science.get("selection_sha256") == identity(spec)
            and science.get("dataset_sha256") == spec["dataset_sha256"],
            "Scientific admission for the 10-cell test-dev pilot is missing or differs.")
    for party in ("author", "supervisor"):
        approved_evidence(science["approval_evidence"][party])
    if science["scope"] == "INDEPENDENT_TEST_DEV":
        verify_review(science, {p["detector"]["checkpoint_sha256"] for p in spec["pairs"]}, ROOT)
    else:
        require(science.get("acknowledged_prior_VM43_exposure") is True,
                "Prior test-dev use must be disclosed.")
    ticket = str(gpu.get("authorization_id",""))
    budget = gpu.get("budget_seconds")
    require(gpu.get("status") == "APPROVED" and gpu.get("scope") == "TWO_CHECKPOINT_TEST_DEV_PILOT"
            and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", ticket) is not None
            and gpu.get("selection_sha256") == identity(spec)
            and gpu.get("scientific_decision_sha256") == sha256(science_path)
            and type(budget) in (int, float) and isfinite(budget) and budget >= 60
            and available_seconds(gpu) >= 60, "Missing applicable finite GPU authorization.")
    approved_evidence(gpu["approval_evidence"])
    require(sha256(TEST) == spec["dataset_sha256"], "Test-dev archive SHA differs.")
    return science, gpu


def evaluate(out, spec, science, gpu):
    from itertools import groupby
    import cv2
    import numpy as np

    began = perf_counter()
    runtime(out)
    renderer = generator()
    members = [{k:v for k,v in json.loads(line).items() if k != "image_stem"}
               for line in (AUDIT/"test_dev_members.jsonl").read_text(encoding="utf-8-sig").splitlines()]
    require(len(members) == IMAGES, "Expected the complete test-dev ledger.")
    module("new_fast_coco_accumulate", LEGACY/"development_fast_coco.py")
    reader = OriginalZIP(TEST, IMAGES)
    pair_root = RUNS/"pilot1280_pairs"/identity({"plan":identity(spec),
                       "scope":science["scope"], "gpu_uuid":gpu["GPU_uuid"]})
    devices, results = {}, []
    try:
        for _, rows in groupby(spec["pairs"], key=lambda row:row["degradation"]["degradation_id"]):
            group = list(rows)
            p = group[0]["degradation"]
            key = identity({"source":spec["dataset_sha256"], "members":spec["members_sha256"],
                            "P":p, "seed":SEED, "code":spec["code_sha256"],
                            "opencv":cv2.__version__, "numpy":np.__version__})
            started = perf_counter()
            cache, cached, hit = shared_cache(RUNS/"cache", key, members, reader,
                    lambda image, stem: renderer.generate(image, p["leaves"], p["recipe"], stem, SEED))
            cache_time = perf_counter()-started
            for pair in group:
                d = pair["detector"]
                name = pair["pair_id"]
                target = pair_root/name
                binding = {"protocol":PROTOCOL, "scope":science["scope"], "spec":pair, "images":IMAGES,
                           "dataset_sha256":spec["dataset_sha256"], "members_sha256":spec["members_sha256"],
                           "selection_sha256":identity(spec), "gpu_uuid":gpu["GPU_uuid"],
                           "checkpoint_sha256":d["checkpoint_sha256"], "bootstrap":spec["bootstrap"],
                           "runtime":spec["runtime"], "code_sha256":spec["code_sha256"],
                           "cache_manifest_sha256":sha256(cache/"manifest.json"),
                           "gt_sha256":sha256(cache/"ground_truth.json")}
                if target.exists():
                    verify_pair(target, binding)
                    status = "VERIFIED_SKIP"
                else:
                    clock = perf_counter()
                    is_new = d["checkpoint"] not in devices
                    if is_new:
                        devices[d["checkpoint"]] = PilotDevice(d)
                    dev = devices[d["checkpoint"]]
                    torch = dev.torch
                    load_seconds = perf_counter()-clock if is_new else 0
                    prepare = perf_counter()
                    predict = dev.detector(d)
                    def wrapper(image):
                        result, trace = predict(image)
                        require(not bool(dev.model.predictor.model.end2end),
                                "Pilot requires the NMS path for both checkpoints.")
                        return result.boxes.data.detach().cpu().numpy(), trace, result.speed
                    try:
                        status = measure_pair(target, binding, cache, cached, wrapper, evaluate_saved,
                            bootstrap_coco, torch.cuda.synchronize, torch.cuda.reset_peak_memory_stats,
                            lambda:{"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                                    "peak_reserved_bytes":torch.cuda.max_memory_reserved()},
                            clock_origin=clock, prepare_seconds=perf_counter()-prepare,
                            model_load_seconds=load_seconds)
                    finally:
                        dev.release()
                row = {"pair_id":name, "artifact":target.relative_to(RUNS).as_posix(),
                       "cache_hit":hit, "cache_lookup_seconds":cache_time,
                       "status":status, "completion_sha256":sha256(target/"COMPLETE.json")}
                results.append(row)
                with (out/"progress.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps(row)+"\n")
                print(f"{name}: {status}", flush=True)
    finally:
        reader.close()
    write(out/"result.json", {"status":"COMPLETE", "scope":science["scope"], "cells":results,
                             "formal_pair_credit":0, "seconds":perf_counter()-began})


@contextmanager
def gpu_lock(out):
    lock = RUNS/".gpu.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(out))
    try:
        yield
    finally:
        lock.unlink()


def execute(spec, science_path, gpu_path):
    _, gpu = admission(spec, science_path, gpu_path)
    out = RUNS/"pilot1280_executions"/("uav05-pilot1280-"+uuid4().hex)
    with gpu_lock(out):
        out.mkdir(parents=True)
        ticket = RUNS/"authorizations_used"/(gpu["authorization_id"]+".json")
        ticket.parent.mkdir(exist_ok=True)
        write(ticket, {"execution":out.name, "selection_sha256":identity(spec)})
        token = uuid4().hex
        files = [sha256(path) for path in (science_path, gpu_path)]
        write(out/"plan.json", spec)
        write(out/"worker_lease.json", {"token":token, "authorizations":files})
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(out),
                   "--science", str(science_path), "--gpu-authorization", str(gpu_path)]
        seconds = available_seconds(gpu)
        require(seconds >= 60, "GPU allocation elapsed.")
        print(out, flush=True)
        guarded_run(command, out, seconds, dict(os.environ,UAV05_PILOT1280_TOKEN=token), gpu["GPU_uuid"])
        require(load(out/"result.json")["status"] == "COMPLETE", "Pilot did not seal all cells.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--check-checkpoints", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--science", type=Path)
    parser.add_argument("--gpu-authorization", type=Path)
    args = parser.parse_args()
    if (args.execute or args.worker) and not (args.science and args.gpu_authorization):
        parser.error("Execution requires approvals specific to the pilot and its GPU allocation.")
    spec = plan()
    if args.plan:
        print(json.dumps({"selection_sha256":identity(spec), **spec}, indent=2))
    elif args.check_checkpoints:
        import torch
        torch.set_num_threads(1)
        checks = [PilotDevice(pair["detector"], cuda=False).params for pair in spec["pairs"][:2]]
        print(json.dumps({"status":"CPU_CHECKPOINT_PASS", "parameters":checks,
                          "selection_sha256":identity(spec), "cells":len(spec["pairs"])}))
    else:
        science, gpu = (path.resolve(strict=True) for path in (args.science, args.gpu_authorization))
        if args.worker:
            out = args.worker.resolve(strict=True)
            lease = load(out/"worker_lease.json")
            require(lease["token"] == os.environ.get("UAV05_PILOT1280_TOKEN")
                    and lease["authorizations"] == [sha256(path) for path in (science,gpu)]
                    and spec == load(out/"plan.json"), "Invalid pilot worker lease.")
            evaluate(out, spec, *admission(spec, science, gpu))
        else:
            execute(spec, science, gpu)


if __name__ == "__main__":
    main()
