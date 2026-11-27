"""Verify the saved Week 3 GPU profiles and prepare a matched extension."""

import argparse
import csv
from importlib.util import find_spec
from math import isclose
from pathlib import Path
from shutil import which
from statistics import mean, median

from export_week3_followup import completed, table
from profile_inference_options import ADDITIONAL_CASES, CASES, PRECISION, case_spec, read_models, source_files
from validate_checkpoints import (
    CACHE, CACHE_SHA, HERE, MAIN, RUNS, digest, input_set, load, output_folder,
    percentile95, require, write_json,
)

PUBLISHED = MAIN / "reports/results" / HERE.name / "reproduce_20261125T142040_7a322c8d"
GPU_LOCK = RUNS / "uav_05_evaluation/.gpu.lock"


def verify_profiles(published, run_root, cache):
    manifest = load(published / "manifest.json")
    metrics = published / "inference_options.csv"
    require(digest(metrics) == manifest["files"][metrics.name], "Published table SHA differs.")
    require(digest(published / "snapshot.json") == manifest["SHA256"], "Snapshot SHA differs.")
    published_rows = table(metrics)
    require([row["case"] for row in published_rows] == list(CASES), "Profile roster differs.")
    _, members, _, _ = input_set(cache, cache / "ground_truth.json")
    image_ids = [row["image_id"] for row in members]
    raw = {}
    for source in manifest["sources"]:
        if source["kind"] != "inference_options":
            continue
        run = run_root / source["RunID"]
        plan = load(run / "plan.json")
        guard = load(run / "supervisor.json")
        require(plan["members_sha256"] == CACHE_SHA and plan["images"] == len(image_ids)
                and plan["warmups"] == 3 and not guard["stop_reason"], "Run protocol differs.")
        if guard["returncode"] == 0:
            completed(run, len(plan["order"]))
        else:
            require(guard["returncode"] == 1 and
                    "ValueError: Live tensors remained from the preceding case." in
                    (run / "worker.log").read_text(encoding="utf-8"), "Unknown partial-run cause.")
        for artifact in [source["plan"], source["supervisor"], *source["cells"]]:
            path = run / artifact["artifact"]
            require(digest(path) == artifact["SHA256"], f"Artifact SHA differs: {path.name}")
        for artifact in source["cells"]:
            path = run / artifact["artifact"]
            require(path.stem in CASES and path.stem not in raw, "Duplicate or unknown profile.")
            raw[path.stem] = (path, load(path), plan["cases"][path.stem])
    require(set(raw) == set(CASES), "Incomplete raw evidence.")
    models = read_models()
    for row in published_rows:
        name = row["case"]
        path, record, planned = raw[name]
        summary, samples = record["summary"], record["samples"]
        require([sample["image_id"] for sample in samples] == image_ids
                and summary["images"] == len(image_ids), "Validation identities differ.")
        require(summary["allocated_before_case_bytes"] == 0
                and summary["case"] == name
                and summary["checkpoint_sha256"] == models[CASES[name][0]]["sha256"]
                and summary["FM_D_configuration_id"] == planned["configuration_id"]
                == case_spec(name)["configuration_id"], "Profile binding differs.")
        require(all(str(value) == row[key] for key, value in summary.items()),
                f"Published values differ: {name}")
        for field in ("predict", "pipeline"):
            values = [s[field + "_ms"] for s in samples]
            for stat, fn in (("mean", mean), ("median", median), ("p95", percentile95)):
                require(isclose(fn(values), summary[f"{field}_{stat}_ms"]),
                        f"Timing did not reproduce: {name} {field} {stat}")
        row["source_run"] = path.parent.name
    return published_rows


def extension_plan(checkpoints, check_weights=True):
    admitted = set()
    rows = []
    for name, (model, size, strategy, post, pre, tta) in ADDITIONAL_CASES.items():
        entry = checkpoints[model]
        if check_weights and model not in admitted:
            source_files(entry)
            admitted.add(model)
        rows.append({"case": name, "checkpoint": model, "checkpoint_sha256": entry["sha256"],
                     "images": 548, "infer_px": size, "strategy": strategy, "postprocessing": post,
                     "precision": PRECISION.get(name, "FP32"), "preprocessing": pre, "TTA": tta,
                     "FM_D_configuration_id": case_spec(name)["configuration_id"],
                     "status": "PLANNED"})
    return rows


def write_table(path, rows):
    with path.open("x", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(rows):
    baselines = {"v8": "v8_full640", "v26": "v26_dense640"}
    cases = {row["case"]: row for row in rows}
    comparisons = []
    for name, row in cases.items():
        base_name = baselines[name.split("_")[0]]
        if name == base_name:
            continue
        baseline = cases[base_name]
        ms, ref = (float(item["pipeline_mean_ms"]) for item in (row, baseline))
        comparisons.append({"case": name, "baseline": base_name, "mean_ms_per_image": ms,
                            "baseline_mean_ms_per_image": ref, "change_ms_per_image": ms - ref,
                            "p95_ms_per_image": float(row["pipeline_p95_ms"]),
                            "GFLOPs_per_image": float(row["mean_GFLOPs_2xMAC"]),
                            "peak_allocated_MiB": float(row["peak_allocated_MiB"]),
                            "source_run": row["source_run"], "baseline_run": baseline["source_run"]})
    return comparisons


def plot_profiles(path, rows):
    from matplotlib.figure import Figure

    labels = ["v8s Full640", "v8s Full1536", "v8s SAHI640", "v8s TTA640", "v8s CLAHE",
              "v8s Gamma", "v8s Denoise", "v8s Deblur", "26s NMS", "26s NMS-free"]
    fig = Figure(figsize=(9.5, 4.8), layout="constrained")
    ax = fig.subplots()
    y = range(len(rows))
    for field, offset, color, label in (("pipeline_mean_ms", -.17, "#147d84", "Mean"),
                                         ("pipeline_p95_ms", .17, "#a79daf", "P95")):
        ax.barh([i + offset for i in y], [float(row[field]) for row in rows], .31,
                color=color, label=label)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Cached pipeline latency (ms/image)")
    ax.set_title("Inference options on VisDrone val (548 images)")
    ax.legend()
    ax.grid(axis="x", alpha=.2)
    fig.savefig(path, dpi=150)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--published", type=Path, default=PUBLISHED)
    parser.add_argument("--run-root", type=Path, default=RUNS / HERE.name)
    parser.add_argument("--cache", type=Path, default=CACHE)
    args = parser.parse_args()
    rows = verify_profiles(args.published, args.run_root, args.cache)
    plan = extension_plan(read_models())
    out = output_folder(HERE.name)
    write_table(out / "matched_resource_comparisons.csv", compare(rows))
    write_table(out / "extension_plan.csv", plan)
    plot_profiles(out / "inference_latency.png", rows)
    try:
        lock = Path(GPU_LOCK.read_text(encoding="utf-8").strip()).name
    except FileNotFoundError:
        lock = None
    result = {"RunID": out.name, "status": "COMPLETE_CPU_REPLAY", "verified_gpu_profiles": len(rows),
              "images_per_profile": 548, "original_GPU_image_calls": len(rows) * 548,
              "new_GPU_image_calls": 0, "planned_extension": len(plan), "blocking_gpu_run": lock,
              "reported_source_sha256": digest(args.published / "inference_options.csv"),
              "uvl_sha256": digest(HERE / "fm_d.uvl"), "INT8": {
                  "status": "NO_BOUND_CALIBRATED_BACKEND",
                  "tensorrt_importable": find_spec("tensorrt") is not None,
                  "trtexec_detected": which("trtexec") is not None,
                  "next_input": "Engine for a selected VisDrone checkpoint with train-only calibration provenance.",
              }}
    write_json(out / "result.json", result)
    (out / "index.html").write_text(
        f"<h1>Week 3 inference options</h1><p>{len(rows)} verified profiles, 548 images each.</p>"
        '<ul><li><a href="matched_resource_comparisons.csv">Matched resource comparisons</a></li>'
        '<li><a href="extension_plan.csv">Extension plan</a></li>'
        '<li><a href="result.json">Source and status</a></li></ul>'
        '<img src="inference_latency.png" alt="Mean and P95 latency">', encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
