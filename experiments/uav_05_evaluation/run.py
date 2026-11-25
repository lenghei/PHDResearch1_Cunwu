"""Execute or replay the first frozen P group after separate scientific and GPU approval."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
from itertools import groupby
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from uuid import uuid4

MAIN = Path(__file__).resolve().parents[2]
ROOT = MAIN.parent
LEGACY = MAIN / "experiments/experiments_test/cv261207"
RUNS = ROOT / "datasets/local_runs/uav_05_evaluation"
AUDIT_ID = "cv261207_audit_20261104T193649_d1b63492"
AUDIT = ROOT / "datasets/local_runs/cv261207" / AUDIT_ID
LIMIT = 14400
IMAGES = 1610

sys.path.insert(0, str(MAIN / "code"))
sys.path.insert(0, str(LEGACY))

from common import TEST, CHECKPOINT, FMP, file_id
from evaluation import Device, evaluate_saved, runtime
from development_v1 import generator
from main.statistics import bootstrap_coco
from pipeline import (PROTOCOL, attempt, disk_guard, identity, load, measure_pair, require,
                      sha256, shared_cache, verify_pair, verify_repeat, write)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def frozen():
    legacy = module("uav_legacy_control", LEGACY/"run.py")
    audit, manifest = legacy.audit_plan(AUDIT_ID)
    specs = manifest["pairs"]
    groups = [list(g) for _,g in groupby(specs, key=lambda x:x["degradation"]["degradation_id"])]
    require(len(groups)==100 and len({x["pair_id"] for x in specs})==500 and all(len(g)==5 for g in groups),
            "Unexpected frozen P/D matrix.")
    detector_order = [p["detector"] for p in groups[0]]
    for group in groups:
        require([x["detector"] for x in group] == detector_order, "Frozen D settings differ.")
        for spec in group:
            p, d = spec["degradation"], spec["detector"]
            require("RollingShutter" not in p["leaves"] and p["degradation_id"] == file_id(p["leaves"], "p_")
                and d["detector_id"] == file_id(d["leaves"], "d_")
                and spec["pair_id"] == p["degradation_id"]+"__"+d["detector_id"],
                "P/D identity does not match its features.")
    return audit, manifest, groups[0]


def code_binding():
    dependencies = [Path(__file__), Path(__file__).with_name("pipeline.py"),
        Path(__file__).with_name("supervisor.py"), MAIN/"experiments/experiments_test/vm37_yolo26_rq4/post_cost/safety.py", FMP,
        LEGACY/"evaluation.py", LEGACY/"archives.py", LEGACY/"development_fast_coco.py",
        LEGACY/"development_v1.py", MAIN/"experiments/experiments_test/cv261109/generator.py", MAIN/"experiments/experiments_test/cv261109/model.py"]
    dependencies += [MAIN/"code/main"/f"{name}.py" for name in ("statistics", "dataset", "labels", "predictor")]
    return {p.relative_to(MAIN).as_posix(): sha256(p) for p in dependencies}


def runtime_binding():
    names = ("torch", "ultralytics", "numpy", "opencv-python", "pycocotools", "ultralytics-thop")
    return {"python": sys.version, "packages": {name: importlib.metadata.version(name) for name in names}}


def approved_evidence(row):
    path = (ROOT / row["path"]).resolve(strict=True)
    require(path.is_relative_to((ROOT/"datasets").resolve()) and path.is_file() and path.stat().st_size > 0
            and sha256(path)==row["sha256"], "Approval evidence does not match its private archive.")


def admission(science_path, gpu_path):
    science, gpu = load(science_path), load(gpu_path)
    exploratory = science.get("scope") == "EXPLORATORY_FIRST_PAIR_ONLY"
    if exploratory:
        require(science.get("status") == "AUTHOR_APPROVED_EXPLORATORY_ONLY"
                and science.get("formal_pair_credit") == 0,
                "Exploratory admission must have explicit author approval and zero formal credit.")
    else:
        require(science.get("scope") in ("LIMITED_TEST_DEV_BENCHMARK", "INDEPENDENT_TEST_DEV")
                and science.get("status") == "AUTHOR_AND_SUPERVISOR_APPROVED",
                "Scientific admission remains blocked.")
    for party in (("author",) if exploratory else ("author", "supervisor")):
        approved_evidence(science["approval_evidence"][party])
    approved_evidence(gpu["approval_evidence"])
    audit, plan, group = frozen()
    gate = load(audit/"dataset_gate.json")
    test = gate["archives"]["test-dev"]
    expected_plan = sha256(audit/"selection_manifest.json")
    require(science["dataset_sha256"]==test["sha256"] and science["selection_sha256"]==expected_plan
            and science["protocol"]==PROTOCOL, "Scientific approval covers a different design.")
    if exploratory:
        require(science.get("pair_id") == group[0]["pair_id"] and science.get("acknowledged_prior_VM43_exposure") is True,
                "Only the fixed first pair is authorized; prior exposure must be disclosed.")
    if science["scope"]=="INDEPENDENT_TEST_DEV":
        from admission import verify_review
        verify_review(science, {x["detector"]["checkpoint_sha256"] for x in group}, ROOT)
    else:
        require(science.get("acknowledged_prior_VM43_exposure") is True, "Prior test-dev use must be disclosed.")
    files = code_binding()
    ticket_id = gpu["authorization_id"]
    require(isinstance(ticket_id, str) and len(ticket_id) <= 100 and
            all(c.isascii() and (c.isalnum() or c in "-_") for c in ticket_id) and ticket_id,
            "Invalid authorization ID.")
    cap = gpu["budget_seconds"]
    require(gpu["status"]=="APPROVED" and type(cap) is int and 60 <= cap <= LIMIT
            and gpu["scientific_decision_sha256"]==sha256(science_path)
            and gpu["selection_sha256"]==expected_plan and gpu["code_sha256"]==identity(files)
            and gpu["runtime_sha256"]==identity(runtime_binding())
            and gpu["scope"]==("EXPLORATORY_FIRST_PAIR_AND_REPEAT" if exploratory
                              else "FIRST_PAIR_REPEAT_AND_FIRST_P_REMAINDER"),
            "There is no applicable GPU authorization.")
    deadline = datetime.fromisoformat(gpu["expires_utc"])
    seconds = min(cap, (deadline-datetime.now(timezone.utc)).total_seconds())
    require(seconds >= 60, "GPU authorization expired or insufficient.")
    require(test["images"]==IMAGES and sha256(TEST)==test["sha256"], "Frozen test-dev archive differs.")
    require(sha256(CHECKPOINT)==group[0]["detector"]["checkpoint_sha256"], "Frozen checkpoint differs.")
    members_path = audit/"test_dev_members.jsonl"
    require(sha256(members_path)==gate["test_dev_member_manifest_sha256"], "Frozen image/GT ledger differs.")
    return science, gpu, plan, group, files, members_path, test


def run_worker(out, science_path, gpu_path):
    # The authorization is consumed before the child is started.
    lease = load(out/"worker_lease.json")
    require(os.environ.get("UAV05_WORKER_TOKEN")==lease["token"]
            and sha256(science_path)==lease["science_sha256"]
            and sha256(gpu_path)==lease["authorization_sha256"], "Worker lease differs.")
    from archives import OriginalZIP
    import cv2
    import numpy as np

    began = time.perf_counter()
    science, gpu, plan, group, files, members_path, test = admission(science_path, gpu_path)
    runtime(out)
    det_sha = group[0]["detector"]["checkpoint_sha256"]
    start = time.perf_counter()
    device = Device(det_sha)
    model_seconds = time.perf_counter()-start
    torch = device.torch
    write(out/"environment.json", {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "cuda": torch.version.cuda, "ultralytics": importlib.metadata.version("ultralytics"),
        "numpy": np.__version__, "opencv": cv2.__version__, "model_load_seconds": model_seconds})
    from sampling import models
    p_model, _, _ = models()
    attributes = p_model.read_uvl_attributes(FMP)
    p = group[0]["degradation"]
    renderer = generator()
    member_rows = [json.loads(s) for s in members_path.read_text(encoding="utf-8-sig").splitlines()]
    require(len(member_rows)==IMAGES, "Frozen study requires 1,610 images.")
    # The legacy ledger adds image_stem to the archive's record proof.
    members = [{k:v for k,v in row.items() if k!="image_stem"} for row in member_rows]
    cache_binding = {"source_zip": test["sha256"], "members_sha256": sha256(members_path), "P": p,
        "parameters": {name: attributes[name] for name in p["leaves"]},
        "render_seed": plan["render_seed"], "software": {"numpy": np.__version__, "opencv": cv2.__version__},
        "sources": {k:v for k,v in files.items() if k.endswith(("generator.py","model.py","fm_p.uvl","pipeline.py","dataset.py","labels.py"))}}
    reader = OriginalZIP(TEST, IMAGES)
    cache_started = time.perf_counter()
    try:
        cache, cache_rows, hit = shared_cache(RUNS/"cache", identity(cache_binding), members, reader,
            lambda image, stem: renderer.generate(image, p["leaves"], attributes, stem, plan["render_seed"]))
    finally:
        reader.close()
    cache_access = time.perf_counter()-cache_started
    write(out/"cache_receipt.json", {**cache_binding, "cache_hit": hit, "cache_access_seconds": cache_access,
                                     "stored_png_bytes":cache_rows["cache_png_bytes"]})
    module("new_fast_coco_accumulate", LEGACY/"development_fast_coco.py")
    exploratory = science["scope"] == "EXPLORATORY_FIRST_PAIR_ONLY"
    result_root = RUNS/("exploratory_pairs" if exploratory else "pairs")/identity({"cache":cache.name, "engine":files, "scope":science["scope"],
                                        "runtime":runtime_binding(), "gpu_uuid":gpu["GPU_uuid"],
                                        "selection_sha256":sha256(AUDIT/"selection_manifest.json")})
    results = []
    schedule = [(group[0], False), (group[0], True)]
    if not exploratory:
        schedule += [(s, False) for s in group[1:]]
    reserve_seconds, reserve_bytes = None, 0
    for index, (spec, repeated) in enumerate(schedule):
        pair_start = time.perf_counter()
        binding = {"protocol":PROTOCOL, "spec": spec, "images":IMAGES, "dataset_sha256":test["sha256"],
                   "scope":science["scope"], "code_sha256":files,
                   "runtime":runtime_binding(), "gpu_uuid":gpu["GPU_uuid"],
                   "p_config_sha256":identity({"selected":p,"recipe":cache_binding["parameters"]}),
                   "d_config_sha256":identity(spec["detector"]), "checkpoint_sha256":det_sha,
                   "cache_manifest_sha256":sha256(cache/"manifest.json"),
                   "gt_sha256":sha256(cache/"ground_truth.json"), "bootstrap":{"samples":200,"seed":20261207}}
        suffix = "__repeat" if repeated else ""
        print(f"Starting {spec['pair_id']}{suffix} ({IMAGES} images)", flush=True)
        target = result_root/(spec["pair_id"]+suffix)
        if target.exists():
            verify_pair(target, binding)
            status = "VERIFIED_SKIP"
        else:
            elapsed = time.perf_counter()-began
            if reserve_seconds is not None and lease["seconds"]-elapsed < reserve_seconds+30:
                break
            disk_guard(cache, reserve_bytes)
            prepare_start = time.perf_counter()
            predictor = device.detector(spec["detector"])
            def predict(image):
                result, tensor = predictor(image)
                return result.boxes.data.detach().cpu().numpy(), tensor, result.speed

            status = measure_pair(target, binding, cache, cache_rows, predict, evaluate_saved,
                bootstrap_coco, torch.cuda.synchronize, torch.cuda.reset_peak_memory_stats,
                lambda: {"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
                         "peak_reserved_bytes":torch.cuda.max_memory_reserved()},
                clock_origin=pair_start, model_load_seconds=model_seconds if index==0 else 0,
                prepare_seconds=time.perf_counter()-prepare_start)
        results.append({"pair_id":spec["pair_id"], "repeat":repeated,
                        "status":status, "artifact":str(target.relative_to(RUNS))})
        if index==0:
            timing = load(target/"timing.json")
            reserve_seconds = 1.25*(timing["wall_seconds_to_sealing_including_setup"]+timing["model_load_seconds"])
            reserve_bytes = 2*sum(f.stat().st_size for f in target.iterdir() if f.is_file())
        if repeated:
            verify_repeat(result_root/spec["pair_id"], target, binding)
            write(out/"reproduced.json", {"status":"PASS", "identical_raw_content_and_bootstrap":True,
                 "first":results[0], "repeat":results[1]})
    write(out/"result.json", {"status":"COMPLETE" if len(results)==len(schedule) else "PARTIAL", "results":results,
        "scope":science["scope"], "formal_pair_credit":0 if exploratory else sum(not r["repeat"] for r in results),
        "verified_unique_pairs":sum(not row["repeat"] for row in results),
        "qualification_repeat_verified":(out/"reproduced.json").is_file(),
        "full_worker_wall_seconds":time.perf_counter()-began})


def supervise(science_path, gpu_path):
    science, gpu, _, _, _, _, _ = admission(science_path, gpu_path)
    # Keep the historical GPU budget separate. An authorization is consumed once.
    ticket_id = gpu["authorization_id"]
    kind = "exploratory_executions" if science["scope"] == "EXPLORATORY_FIRST_PAIR_ONLY" else "executions"
    out = RUNS/kind/("uav05-"+uuid4().hex)
    out.parent.mkdir(parents=True, exist_ok=True)
    (RUNS/"authorizations_used").mkdir(exist_ok=True)
    lock = RUNS/".gpu.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(out))
    try:
        write(RUNS/"authorizations_used"/(ticket_id+".json"), {"execution":out.name})
        out.mkdir()
        print(out, flush=True)
        from supervisor import guarded_run
        seconds = min(gpu["budget_seconds"],
            (datetime.fromisoformat(gpu["expires_utc"])-datetime.now(timezone.utc)).total_seconds())
        require(seconds >= 60, "GPU authorization elapsed.")
        secret = uuid4().hex
        write(out/"worker_lease.json", {"token":secret, "seconds":seconds-15,
            "science_sha256":sha256(science_path), "authorization_sha256":sha256(gpu_path)})
        command = [sys.executable, "-B", str(Path(__file__).resolve()), "--worker", str(out),
                   "--science", str(science_path), "--gpu-authorization", str(gpu_path)]
        guarded_run(command, out, seconds, dict(os.environ, UAV05_WORKER_TOKEN=secret),
                    expected_uuid=gpu["GPU_uuid"])
        require((out/"result.json").is_file(), "The worker did not seal an execution result.")
    finally:
        lock.unlink()


def replay(path):
    binding = load(path/"configuration.json")
    verify_pair(path, binding)
    metrics, _ = evaluate_saved(path)
    require(metrics == load(path/"metrics.json"), "Raw prediction replay differs.")
    print(json.dumps({"status":"PASS", "prediction_sha256":sha256(path/"predictions.jsonl.gz"),
                      "gt_sha256":sha256(path/"ground_truth.json"), "metrics":metrics}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--identity", action="store_true")
    mode.add_argument("--replay", type=Path)
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--science", type=Path)
    parser.add_argument("--gpu-authorization", type=Path)
    args = parser.parse_args()
    if args.identity:
        audit, _, group = frozen()
        print(json.dumps({"protocol":PROTOCOL, "dataset_sha256":load(audit/"dataset_gate.json")["archives"]["test-dev"]["sha256"],
            "selection_sha256":sha256(audit/"selection_manifest.json"), "code_sha256":identity(code_binding()),
            "runtime_sha256":identity(runtime_binding()),
            "first_group":[row["pair_id"] for row in group], "formal_images_per_pair":IMAGES}))
        return
    if args.replay:
        replay(args.replay.resolve(strict=True))
        return
    if not args.science or not args.gpu_authorization:
        parser.error("Execution requires separate science and GPU approval files.")
    if args.worker:
        run_worker(args.worker, args.science.resolve(strict=True), args.gpu_authorization.resolve(strict=True))
    else:
        supervise(args.science.resolve(strict=True), args.gpu_authorization.resolve(strict=True))


if __name__ == "__main__":
    main()
