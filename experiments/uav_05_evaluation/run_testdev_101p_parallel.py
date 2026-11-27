"""Resume the frozen Test-dev study with a separately qualified cache builder."""
import argparse
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from uuid import uuid4

import run_testdev_101p as core
from testdev_cache_parallel import build_cache


def qualification(study, groups, path):
    proof = core.load(path)
    cache_code = Path(__file__).with_name("testdev_cache_parallel.py")
    current = core.load(core.CAMPAIGN / "study.json")
    core.require(all(current[k] == v for k, v in study.items()) and
                 core.identity(current) == proof["study_identity"] and
                 core.sha256(cache_code) == proof["cache_source_sha256"],
                 "Cache qualification differs from the campaign or implementation.")
    results = proof["results"]
    core.require(len(results) == 2, "Qualification requires the first two full P caches.")
    for group, row in zip(groups[:2], results):
        first = group[0]
        folder = core.CAMPAIGN / "pairs" / first["pair_id"]
        binding = core.load(folder / "configuration.json")
        core.verify_pair(folder, binding)
        core.require(row["pair_id"] == first["pair_id"] and row["p_id"] == first["p_id"] and
                     row["images"] == core.IMAGES and row["exact"] is True and
                     all(binding[k] == row[k.replace("cache_", "")] for k in
                         ("cache_manifest_sha256", "gt_sha256")),
                     "Cache qualification differs from the sealed P.")
    return {
        "study_implementation_sha256": core.identity(study),
        "cache_source_sha256": core.sha256(cache_code),
        "cache_qualification_sha256": core.sha256(path),
        "continuation_source_sha256": core.sha256(Path(__file__)),
    }


def authorize(study, groups, science_path, gpu_path, cache):
    science, gpu = core.authorize(study, groups, science_path, gpu_path)
    core.require(gpu.get("cache_execution_sha256") == core.identity(cache),
                 "This GPU allocation does not cover the qualified cache implementation.")
    core.require(gpu.get("priority_p") in (None, core.P18), "Only the prespecified P18 case may be moved forward.")
    return science, gpu


def launch(study, groups, science_path, gpu_path, proof, cache):
    _, gpu = authorize(study, groups, science_path, gpu_path, cache)
    out = core.CAMPAIGN / "batches" / ("uav05-testdev-" + uuid4().hex)
    with core.gpu_lock(out):
        core.disk_guard(core.RUNS)
        out.mkdir(parents=True)
        ticket = core.RUNS / "authorizations_used" / (gpu["authorization_id"] + ".json")
        token = uuid4().hex
        sources = [core.sha256(p) for p in (science_path, gpu_path, proof)]
        seconds = core.remaining(gpu)
        core.require(seconds >= 60, "GPU allocation elapsed.")
        core.write(ticket, {"execution": out.name, "selection_sha256": core.PLAN_SHA,
                            "cache_execution_sha256": core.identity(cache)})
        core.write(out / "cache_implementation.json", cache)
        core.write(out / "lease.json", {"token": token, "authorization_sha256": sources,
                                      "deadline_monotonic": perf_counter() + seconds - 15})
        command = [sys.executable, "-B", "-X", "faulthandler", str(Path(__file__).resolve()),
                   "--worker", str(out), "--science", str(science_path), "--gpu-authorization",
                   str(gpu_path), "--cache-proof", str(proof)]
        core.write(out / "command.json", {"worker": command})
        print(out, flush=True)
        core.guarded_run(command, out, seconds, dict(os.environ, **{core.ENV_TOKEN: token}), gpu["GPU_uuid"])
        core.require((out / "result.json").is_file(), "Worker result is missing.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--identity", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--science", type=Path)
    parser.add_argument("--gpu-authorization", type=Path)
    parser.add_argument("--cache-proof", type=Path, required=True)
    args = parser.parse_args()
    if not args.identity and not (args.science and args.gpu_authorization):
        parser.error("Execution requires science and GPU authorization.")
    study, groups = core.frozen_plan()
    proof = args.cache_proof.resolve(strict=True)
    cache = qualification(study, groups, proof)
    if args.identity:
        print(json.dumps({**cache, "cache_execution_sha256": core.identity(cache)}))
        return
    science_path, gpu_path = (p.resolve(strict=True) for p in (args.science, args.gpu_authorization))
    if args.execute:
        launch(study, groups, science_path, gpu_path, proof, cache)
        return
    out = args.worker.resolve(strict=True)
    core.require(out.parent == (core.CAMPAIGN / "batches").resolve(), "Unexpected worker output directory.")
    lease = core.load(out / "lease.json")
    core.require(os.environ.get(core.ENV_TOKEN) == lease["token"] and
                 [core.sha256(p) for p in (science_path, gpu_path, proof)] == lease["authorization_sha256"] and
                 core.load(out / "cache_implementation.json") == cache,
                 "Worker authorization or cache qualification changed.")
    science, gpu = authorize(study, groups, science_path, gpu_path, cache)
    # Keep the frozen pair protocol and replace only the qualified cache builder.
    core.build_cache = build_cache
    queue = [groups[-1], *groups[:-1]] if gpu.get("priority_p") == core.P18 else groups
    core.worker(out, study, queue, science, gpu, deadline_monotonic=lease["deadline_monotonic"])


if __name__ == "__main__":
    main()
