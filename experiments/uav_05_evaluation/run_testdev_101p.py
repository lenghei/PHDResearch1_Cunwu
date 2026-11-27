"""Validate or execute the frozen 101-P, five-D Test-dev campaign."""
import argparse
import csv
from datetime import datetime, timezone
from functools import partial
from itertools import groupby
import json
from math import isfinite
import os
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from uuid import uuid4

from run import (AUDIT, IMAGES, LEGACY, MAIN, ROOT, RUNS, TEST, approved_evidence,
                 bootstrap_coco, code_binding, evaluate_saved, generator, module,
                 runtime, runtime_binding)
from admission import verify_review
from archives import OriginalZIP
from pilot1280 import PilotDevice, gpu_lock
from pipeline import disk_guard, identity, load, measure_pair, require, sha256, verify_pair, write

HERE = Path(__file__).resolve().parent
W2 = HERE.parent / "uav_02_fmp"
W4 = HERE.parent / "uav_04_sampling"
sys.path.insert(0, str(W2))
sys.path.insert(0, str(W4))
from sparse_bootstrap import sparse_bootstrap
from sparse_matching import with_sparse_matching
from testdev_cache import build_cache
from testdev_supervisor import guarded_run

PROTOCOL = "uav_05_testdev_101p_5d_native10_v1"
PLAN = RUNS / "testdev_101p_5d_plan.csv"
PLAN_SHA = "4c8956316faba59267be3ede0b7b48b824f130cacfbacd916f16da07ec5da476"
CAMPAIGN = RUNS / "testdev_101p_5d"
MEMBERS = AUDIT / "test_dev_members.jsonl"
ENV_TOKEN = "UAVGAP_TESTDEV_101P_WORKER"
P18 = "p_7b49139cd7ab294c"
ROSTER = {
    "ck05_yolov8s_ft1280": ("YOLOv8s", 11139470),
    "ck01_yolo26s_ft1280": ("YOLO26s", 9955604),
    "yolov8s_cft640": ("YOLOv8s", 11139470),
}


def frozen_plan():
    require(sha256(PLAN) == PLAN_SHA, "The frozen evaluation CSV changed.")
    gate = load(AUDIT / "dataset_gate.json")
    test = gate["archives"]["test-dev"]
    require(test["images"] == IMAGES and sha256(TEST) == test["sha256"] and
            sha256(MEMBERS) == gate["test_dev_member_manifest_sha256"],
            "The Test-dev archive or member ledger changed.")
    with PLAN.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(len(rows) == len({r["pair_id"] for r in rows}) == 505, "Duplicate or missing pairs.")
    groups = [list(records) for _, records in groupby(rows, key=lambda r: r["p_id"])]
    require(len(groups) == 101 and all(len(group) == 5 for group in groups) and
            len({group[0]["p_id"] for group in groups}) == 101 and groups[-1][0]["p_id"] == P18,
            "Unexpected P/D matrix.")
    for i, group in enumerate(groups, 1):
        require(all(int(row["p_order"]) == i and row["analysis_group"] ==
                    ("PURPOSEFUL_P18" if i == 101 else "UNIFORM") for row in group),
                "The frozen sample order or cohort changed.")
        require([r["d_id"] for r in group] == [r["d_id"] for r in groups[0]],
                "Detector order differs between groups.")
    for row in groups[0]:
        require(sha256(ROOT / "models/checkpoints" / row["checkpoint"] / "best.pt") ==
                row["checkpoint_sha256"], "A selected checkpoint changed.")
    files = code_binding()
    for path in (Path(__file__), HERE / "testdev_cache.py", HERE / "testdev_supervisor.py", HERE / "pilot1280.py",
                 W2 / "geometry.py", W4 / "sparse_bootstrap.py", W4 / "sparse_matching.py",
                 W4 / "resume_development.py", W4 / "run_development.py",
                 LEGACY / "admission.py"):
        files[path.relative_to(MAIN).as_posix()] = sha256(path)
    spec = {
        "protocol": PROTOCOL, "selection_sha256": PLAN_SHA,
        "dataset_sha256": test["sha256"], "members_sha256": sha256(MEMBERS),
        "images": IMAGES, "code_sha256": files, "runtime": runtime_binding(),
    }
    return spec, groups


def degradation(row):
    return {"leaves": row["p_features"].split("|"), "recipe": json.loads(row["numeric_recipe_json"]),
            "seed": int(row["render_seed"]), "annotation_mode": row["annotation_mode"]}


def remaining(gpu):
    deadline = datetime.fromisoformat(gpu["expires_utc"])
    return min(gpu["budget_seconds"], (deadline - datetime.now(timezone.utc)).total_seconds())


def authorize(spec, groups, science_path, gpu_path):
    science, gpu = load(science_path), load(gpu_path)
    scope = science.get("scope")
    author_only = (scope == "LIMITED_TEST_DEV_BENCHMARK" and
                   science.get("status") == "AUTHOR_APPROVED_LIMITED_BENCHMARK")
    joint = (scope in {"LIMITED_TEST_DEV_BENCHMARK", "INDEPENDENT_TEST_DEV"} and
             science.get("status") == "AUTHOR_AND_SUPERVISOR_APPROVED")
    require((author_only or joint) and
            all(science.get(k) == spec[k] for k in ("protocol", "dataset_sha256", "selection_sha256")),
            "Scientific approval for this exact Test-dev plan is missing.")
    for role in (("author",) if author_only else ("author", "supervisor")):
        approved_evidence(science["approval_evidence"][role])
    if scope == "INDEPENDENT_TEST_DEV":
        verify_review(science, {r["checkpoint_sha256"] for r in groups[0]}, ROOT)
    else:
        require(science.get("acknowledged_prior_VM43_exposure") is True,
                "The limited benchmark must disclose prior Test-dev use.")
    seconds = gpu.get("budget_seconds")
    group_limit = gpu.get("max_new_p")
    require(gpu.get("status") == "APPROVED" and gpu.get("scope") == PROTOCOL and
            re.fullmatch(r"[A-Za-z0-9_-]{1,100}", str(gpu.get("authorization_id", ""))) and
            gpu.get("scientific_decision_sha256") == sha256(science_path) and
            gpu.get("implementation_sha256") == identity(spec) and
            type(seconds) in (int, float) and isfinite(seconds) and seconds >= 60 and
            type(group_limit) is int and 1 <= group_limit <= 101 and
            isinstance(gpu.get("GPU_uuid"), str) and remaining(gpu) >= 60,
            "A finite GPU allocation for this implementation is required.")
    approved_evidence(gpu["approval_evidence"])
    return science, gpu


def base_binding(study, row):
    return {**study, "pair": row, "bootstrap": {
        "samples": int(row["bootstrap_draws"]), "seed": int(row["bootstrap_seed"]),
    }}


def completed_pairs(root, groups, study):
    completed = set()
    for group in groups:
        shared = set()
        for row in group:
            target = root / row["pair_id"]
            if not target.exists():
                continue
            expected = base_binding(study, row)
            binding = load(target / "configuration.json")
            require(all(binding.get(k) == v for k, v in expected.items()), "An existing pair belongs to a different study.")
            verify_pair(target, binding)
            shared.add((binding["cache_manifest_sha256"], binding["gt_sha256"]))
            completed.add(row["pair_id"])
        require(len(shared) <= 1, "Sealed settings do not share their P inputs.")
    return completed


def detector_spec(row):
    architecture, parameters = ROSTER[row["checkpoint"]]
    return {
        "architecture": architecture, "parameters": parameters,
        "catalog_directory": row["checkpoint"], "checkpoint_sha256": row["checkpoint_sha256"],
        "resolution": int(row["infer_px"]), "precision": row["precision"],
    }


def gpu_pair(target, binding, cache, cached, row, devices):
    clock = perf_counter()
    checkpoint = row["checkpoint"]
    new = checkpoint not in devices
    if new:
        devices[checkpoint] = PilotDevice(detector_spec(row))
    device = devices[checkpoint]
    torch = device.torch
    prepare = perf_counter()
    predictor = device.detector(detector_spec(row))

    def predict(image):
        result, trace = predictor(image)
        require(not bool(device.model.predictor.model.end2end), "Expected the common NMS path.")
        return result.boxes.data.detach().cpu().numpy(), trace, result.speed

    try:
        return measure_pair(
            target, binding, cache, cached, predict,
            partial(with_sparse_matching, evaluate_saved),
            partial(sparse_bootstrap, fallback=bootstrap_coco),
            torch.cuda.synchronize, torch.cuda.reset_peak_memory_stats,
            lambda: {"peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                     "peak_reserved_bytes": torch.cuda.max_memory_reserved()},
            samples=binding["bootstrap"]["samples"], seed=binding["bootstrap"]["seed"],
            clock_origin=clock, prepare_seconds=perf_counter() - prepare,
            model_load_seconds=prepare - clock if new else 0,
        )
    finally:
        device.release()


def run_groups(out, root, groups, study, finished, members, reader, generate, measure, group_limit,
               time_left=None):
    completed = set(finished)
    pending = [g for g in groups if any(r["pair_id"] not in finished for r in g)]
    durations = []
    for group in pending[:group_limit]:
        first = group[0]
        if time_left is not None and durations and time_left() < 1.5 * max(durations) + 120:
            break
        began = perf_counter()
        print(f"P {first['p_order']} | {first['p_id']} | {first['analysis_group']}", flush=True)
        with TemporaryDirectory(prefix="pixels-", dir=out) as scratch:
            path = Path(scratch) / "images"
            cached = build_cache(path, identity({"study": study, "P": first["p_id"]}),
                                 degradation(first), members, reader, generate)
            source = {"cache_manifest_sha256": sha256(path / "manifest.json"),
                      "gt_sha256": sha256(path / "ground_truth.json")}
            for row in group:
                target = root / row["pair_id"]
                binding = {**base_binding(study, row), **source}
                if row["pair_id"] in completed:
                    verify_pair(target, binding)
                    continue
                print(f"D {row['d_order']} | {row['checkpoint']} | Infer{row['infer_px']}", flush=True)
                status = measure(target, binding, path, cached, row)
                completed.add(row["pair_id"])
                with (out / "progress.jsonl").open("a", encoding="utf-8") as log:
                    log.write(json.dumps({"pair_id": row["pair_id"], "status": status,
                                         "completion_sha256": sha256(target / "COMPLETE.json")}) + "\n")
        duration = perf_counter() - began
        durations.append(duration)
        with (out / "group_progress.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps({"p_id": first["p_id"], "p_order": int(first["p_order"]),
                                  "wall_seconds": duration, "png_bytes": cached["cache_png_bytes"]}) + "\n")
    return completed


def worker(out, spec, groups, science, gpu, deadline_monotonic):
    runtime(out)
    module("new_fast_coco_accumulate", LEGACY / "development_fast_coco.py")
    study = {**spec, "scope": science["scope"], "gpu_uuid": gpu["GPU_uuid"]}
    record = CAMPAIGN / "study.json"
    if record.exists():
        require(load(record) == study, "The campaign implementation, scope or device changed.")
    else:
        write(record, study)
    root = CAMPAIGN / "pairs"
    finished = completed_pairs(root, groups, study)
    members = [{k: v for k, v in json.loads(line).items() if k != "image_stem"}
               for line in MEMBERS.read_text(encoding="utf-8-sig").splitlines()]
    require(len(members) == IMAGES, "The campaign requires all 1,610 images.")
    reader = OriginalZIP(TEST, IMAGES)
    try:
        measure = partial(gpu_pair, devices={})
        done = run_groups(out, root, groups, study, finished, members, reader,
                          generator().generate, measure, gpu["max_new_p"],
                          lambda: deadline_monotonic - perf_counter())
    finally:
        reader.close()
    selected = {r["pair_id"] for g in groups for r in g if r["analysis_group"] == "UNIFORM"}
    write(out / "result.json", {
        "status": "COMPLETE" if len(done) == 505 else "BATCH_COMPLETE",
        "new_pairs": len(done - finished), "completed_pairs": len(done),
        "uniform_pairs": len(done & selected), "p18_pairs": len(done - selected),
        "scope": science["scope"], "selection_sha256": PLAN_SHA,
    })


def launch(spec, groups, science_path, gpu_path):
    _, gpu = authorize(spec, groups, science_path, gpu_path)
    out = CAMPAIGN / "batches" / ("uav05-testdev-" + uuid4().hex)
    with gpu_lock(out):
        disk_guard(RUNS)
        out.mkdir(parents=True)
        ticket = RUNS / "authorizations_used" / (gpu["authorization_id"] + ".json")
        ticket.parent.mkdir(exist_ok=True)
        token = uuid4().hex
        sources = [sha256(path) for path in (science_path, gpu_path)]
        write(ticket, {"execution": out.name, "selection_sha256": PLAN_SHA})
        seconds = remaining(gpu)
        require(seconds >= 60, "GPU allocation elapsed.")
        write(out / "lease.json", {"token": token, "authorization_sha256": sources,
                                   "deadline_monotonic": perf_counter() + seconds - 15})
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(out),
                   "--science", str(science_path), "--gpu-authorization", str(gpu_path)]
        write(out / "command.json", {"worker": command})
        print(out, flush=True)
        guarded_run(command, out, seconds, dict(os.environ, **{ENV_TOKEN: token}), gpu["GPU_uuid"])
        require((out / "result.json").is_file(), "Worker result is missing.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--identity", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--science", type=Path)
    parser.add_argument("--gpu-authorization", type=Path)
    args = parser.parse_args()
    if not args.identity and not (args.science and args.gpu_authorization):
        parser.error("Execution requires Test-dev science approval and a separate finite GPU allocation.")
    spec, groups = frozen_plan()
    if args.identity:
        print(json.dumps({"status": "PLAN_VERIFIED", "P": 101, "D": 5, "pairs": 505,
                          "uniform_pairs": 500, "P18_pairs": 5, "images_per_pair": IMAGES,
                          "implementation_sha256": identity(spec),
                          **{k: spec[k] for k in ("protocol", "selection_sha256", "dataset_sha256", "members_sha256")}}))
        return
    science_path, gpu_path = (path.resolve(strict=True) for path in (args.science, args.gpu_authorization))
    if args.execute:
        launch(spec, groups, science_path, gpu_path)
    else:
        out = args.worker.resolve(strict=True)
        require(out.parent == (CAMPAIGN / "batches").resolve(), "Unexpected worker output directory.")
        lease = load(out / "lease.json")
        require(os.environ.get(ENV_TOKEN) == lease["token"] and
                lease["authorization_sha256"] == [sha256(path) for path in (science_path, gpu_path)],
                "Worker authorization changed.")
        worker(out, spec, groups, *authorize(spec, groups, science_path, gpu_path),
               deadline_monotonic=lease["deadline_monotonic"])


if __name__ == "__main__":
    main()
