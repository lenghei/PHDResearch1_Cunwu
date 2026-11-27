"""Summarize a complete two-checkpoint pilot from sealed prediction artifacts."""

import argparse
import gzip
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/"uav_05_evaluation"))
from pilot1280 import IMAGES, MAIN, PROTOCOL, RUNS
from pipeline import identity, load, require, sha256, verify_pair, write

sys.path.insert(0, str(MAIN/"code/tools"))
from uav_reproduction import output_folder
from uav_week01_06 import gap_status, index, pair_status, table


def read_cell(path, binding):
    verify_pair(path, binding)
    timing = load(path/"timing.json")
    boot = load(path/"bootstrap.json")
    require(boot["samples"] == 200 and boot["image_ids"] == [r["image_id"] for r in timing["samples"]],
            "Bootstrap and latency populations differ.")
    with gzip.open(path/"bootstrap_indices.npy.gz", "rb") as file:
        draws = np.load(file, allow_pickle=False)
    expected = np.random.default_rng(boot["seed"]).integers(0, binding["images"], (200, binding["images"]))
    require(np.array_equal(draws, expected), "Bootstrap index matrix differs.")
    latency = [r["pipeline_ms"] for r in timing["samples"]]
    latency_ci = np.quantile(np.asarray(latency)[draws].mean(axis=1), [.025,.975]).tolist()
    require(np.allclose(latency_ci, timing["cached_pipeline_mean_ci95_ms"]), "Latency interval differs.")
    point = {r["size"]:r for r in load(path/"metrics.json")["per_area"]}
    metrics = {}
    for area in ("all", "tiny", "small", "medium", "large"):
        draws_ap = np.asarray(boot["draws"]["values"][area]["map50_95"])
        require(draws_ap.shape == (200,) and np.isfinite(draws_ap).all(), "Invalid AP resamples.")
        metrics[area] = {"AP50":point[area]["map50"], "AP50_95":point[area]["map50_95"],
                         "AR50_95":point[area]["ar50_95"], "AP50_95_ci":np.quantile(draws_ap,[.025,.975]).tolist(),
                         "draws":draws_ap}
    return metrics, timing, boot, latency_ci


def summarize(execution):
    began = perf_counter()
    plan, result = (load(execution/name) for name in ("plan.json", "result.json"))
    require(plan["protocol"] == PROTOCOL and result["status"] == "COMPLETE"
            and plan["images"] == IMAGES and plan["formal_pair_credit"] == result["formal_pair_credit"] == 0,
            "Expected a completed baseline pilot.")
    specs = {r["pair_id"]:r for r in plan["pairs"]}
    rows = {r["pair_id"]:r for r in result["cells"]}
    require(len(specs) == len(result["cells"]) == len(rows) == 10 and specs.keys() == rows.keys(),
            "Missing or duplicate pilot cells.")
    cells, metrics_rows, population = {}, [], None
    for pair in plan["pairs"]:
        row = rows[pair["pair_id"]]
        path = (RUNS/row["artifact"]).resolve(strict=True)
        require(path.is_relative_to((RUNS/"pilot1280_pairs").resolve()) and path.name == pair["pair_id"]
                and sha256(path/"COMPLETE.json") == row["completion_sha256"], "Pilot artifact identity differs.")
        binding = load(path/"configuration.json")
        require(binding["spec"] == pair and binding["images"] == IMAGES
                and binding["runtime"] == plan["runtime"] and binding["code_sha256"] == plan["code_sha256"]
                and binding["selection_sha256"] == identity(plan) and binding["scope"] == result["scope"],
                "Pilot binding differs.")
        metrics, timing, boot, latency_ci = read_cell(path, binding)
        boot_identity = (boot["image_ids"], boot["seed"])
        if population is None:
            population = boot_identity
        require(boot_identity == population, "Detector cells do not share image and seed order.")
        checkpoint = pair["detector"]["checkpoint"]
        name = pair["degradation"]["name"]
        state = pair_status(metrics["tiny"]["AP50_95_ci"], latency_ci, .10, 100)
        cells[name,checkpoint] = (metrics, state, binding)
        for area, values in metrics.items():
            metrics_rows.append({"P":name, "D":checkpoint, "area":area, "images":IMAGES,
                                 **{k:v for k,v in values.items() if k != "draws"},
                                 "latency_ms":timing["cached_pipeline_mean_ms"], "state":state})
    losses, gap = [], []
    for name in dict.fromkeys(pair["degradation"]["name"] for pair in plan["pairs"]):
        a, b = (cells[name, checkpoint] for checkpoint in ("ck05","ck01"))
        for key in ("cache_manifest_sha256","gt_sha256"):
            require(a[2][key] == b[2][key], "Detectors did not share pixels and GT.")
        gap.append({"P":name, "ck05":a[1], "ck01":b[1], "status":gap_status((a[1],b[1]), expected=2)})
        if name == "Normal":
            continue
        for checkpoint in ("ck05","ck01"):
            for area in ("all","tiny"):
                baseline = cells["Normal",checkpoint][0][area]
                degraded = cells[name,checkpoint][0][area]
                lo, hi = np.quantile(100*(baseline["draws"]-degraded["draws"]), [.025,.975])
                losses.append({"P":name, "D":checkpoint, "area":area,
                               "AP50_95_loss_pp":100*(baseline["AP50_95"]-degraded["AP50_95"]),
                               "low_pp":lo, "high_pp":hi})
    out = output_folder(HERE.name)
    for filename, data in (("cell_metrics.csv", metrics_rows), ("candidate_gap.csv", gap), ("baseline_losses.csv", losses)):
        table(out/filename, data)
    write(out/"result.json", {"source":execution.name, "status":"COMPLETE", "scope":result["scope"],
                             "formal_pair_credit":0, "cells":len(rows)})
    index(out, {"ExperimentID":HERE.name, "status":"COMPLETE", "CPU_wall_seconds":perf_counter()-began})
    print(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("execution", type=Path)
    summarize(parser.parse_args().execution.resolve(strict=True))
