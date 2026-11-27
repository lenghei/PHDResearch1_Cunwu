"""Verify completed checkpoint cells and export a small comparison snapshot."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from validate_checkpoints import FILES, GT_SHA, MAIN, PROTOCOL, digest, load, require, verify, write_json

LABELS = {"yolov8s_cft640": "YOLOv8s T640", "ck05_yolov8s_ft1280": "YOLOv8s T1280",
          "ck06_yolo11s_ft1280": "YOLO11s T1280", "ck01_yolo26s_ft1280": "YOLO26s T1280",
          "yolov8n_ft640": "YOLOv8n T640", "ck04_yolo26n_native10_full20": "YOLO26n Full20"}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def collect(run, continuations, expected=None):
    plan = load(run / "inputs.json")
    if expected is None:
        expected = [(name, size) for name in plan["models"] for size in plan["resolutions"]]
    catalog = {row["checkpoint"]: row for row in read_csv(run / "checkpoint_catalog.csv")}
    found, receipts = {}, []
    wall_seconds = 0
    for source in (run, *continuations):
        other = load(source / "inputs.json")
        require(all(other[key] == plan[key] for key in ("protocol", "catalog_sha256", "members_sha256", "gt_sha256"))
                and other["protocol"] == PROTOCOL and digest(source / "ground_truth.json") == GT_SHA
                and digest(source / "checkpoint_catalog.csv") == plan["catalog_sha256"], "Input binding differs.")
        supervisor = load(source / "supervisor.json")
        complete = supervisor["returncode"] == 0 and not supervisor["stop_reason"]
        require(complete or (continuations and supervisor["stop_reason"] == "WALL_CLOCK_HARD_STOP"),
                "Source did not pass supervision.")
        wall_seconds += supervisor["worker_wall_seconds"]
        for name in other["models"]:
            for size in other["resolutions"]:
                cell = source / name / f"infer{size}"
                if not cell.exists() and not complete:
                    continue
                key = name, size
                require(key in expected and key not in found, "Unexpected or duplicate checkpoint cell.")
                verify(cell, FILES, {"checkpoint": name, "resolution": size, "images": 548})
                receipts.append({"RunID": source.name, "artifact": f"{name}/infer{size}/COMPLETE.json",
                                 "SHA256": digest(cell / "COMPLETE.json")})
                row = load(cell / "summary.json")
                binding = load(cell / "configuration.json")
                require(row["RunID"] == source.name and row["checkpoint"] == name and row["resolution"] == size
                        and row["checkpoint_sha256"] == binding["checkpoint_sha256"] == catalog[name]["sha256"]
                        and binding["gt_sha256"] == GT_SHA and binding["members_sha256"] == plan["members_sha256"]
                        and binding["protocol"] == PROTOCOL, "Published row or cell binding differs.")
                found[key] = row, load(cell / "metrics.json")["per_class"]
        if complete:
            result = load(source / "result.json")
            expected_count = len(other["models"]) * len(other["resolutions"])
            require(result["RunID"] == source.name and result["status"] == "COMPLETE"
                    and result["cells"] == expected_count and result["image_evaluations"] == 548 * expected_count,
                    "Completion totals differ.")
        receipts.extend({"RunID": source.name, "artifact": name, "SHA256": digest(source / name)}
                        for name in ("inputs.json", "ground_truth.json", "execution.json", "supervisor.json"))
    require(set(found) == set(expected), "Checkpoint matrix is incomplete.")
    summary, classes = [], []
    for name, size in expected:
        row, metrics = found[name, size]
        summary.append(row)
        all_sizes = {r["category_id"]: r for r in metrics if r["size"] == "all"}
        tiny = {r["category_id"]: r for r in metrics if r["size"] == "tiny"}
        require(set(all_sizes) == set(tiny) == set(range(1, 11)), "Incomplete native-ten result.")
        classes.extend({"RunID": row["RunID"], "checkpoint": name, "resolution": size,
                        "category_id": i, "category_name": r["category_name"],
                        "AP50": r["map50"], "AP50_95": r["map50_95"], "AR50_95": r["ar50_95"],
                        "valid_GT": r["gt_count"], "Tiny_AP": tiny[i]["map50_95"],
                        "Tiny_GT": tiny[i]["gt_count"]} for i, r in all_sizes.items())
    return wall_seconds, summary, classes, receipts

def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(dict.fromkeys(r["checkpoint"] for r in summary))
    sizes = sorted({int(r["resolution"]) for r in summary})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, key, title in zip(axes, ("AP50_95", "Tiny_AP"), ("AP50:95 (%)", "Tiny AP50:95 (%)")):
        for j, (size, color) in enumerate(zip(sizes, ("#146c91", "#dc8127"))):
            matched = {r["checkpoint"]: float(r[key])*100 for r in summary if int(r["resolution"]) == size}
            shift = (j - (len(sizes)-1)/2)*.36
            ax.barh([i + shift for i in range(len(names))], [matched[name] for name in names], .33,
                    color=color, label=f"Infer{size}")
        ax.set(yticks=range(len(names)), yticklabels=[LABELS[name] for name in names], xlabel=title)
        ax.invert_yaxis()
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(loc='outside lower center', ncols=len(sizes), frameon=False)
    fig.suptitle(f"{len(names)} VisDrone checkpoints · Normal val · FP32")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def export(run, out, continuations=()):
    wall_seconds, rows, classes, sources = collect(run, continuations)
    out.parent.mkdir(parents=True, exist_ok=True)
    require(not out.exists(), "Output already exists.")
    with TemporaryDirectory(dir=out.parent) as temp:
        stage = Path(temp) / out.name
        stage.mkdir()
        for filename, table in (("validation_metrics.csv", rows), ("validation_per_class.csv", classes)):
            with (stage / filename).open("x", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, list(table[0]))
                writer.writeheader()
                writer.writerows(table)
        plot(rows, stage / "checkpoint_validation.png")
        identity = {"ExperimentID": "uav_03_fmd_static_gap", "RunID": run.name,
                    "status": "COMPLETE", "mode": "CHECKPOINT_VALIDATION",
                    "scope": PROTOCOL["scope"]}
        summary = {**identity, "images_per_condition": 548,
                   "cells": len(rows), "image_evaluations": 548 * len(rows),
                   "validation": [{"models": len({r["checkpoint"] for r in rows}), "precision": "FP32",
                                   "worker_wall_seconds": wall_seconds, "source_runs": 1 + len(continuations),
                                   "dataset": PROTOCOL["dataset"]}]}
        write_json(stage / "snapshot.json", {**identity, "files": {"summary.json": summary, "result.json": identity}})
        write_json(stage / "manifest.json", {**identity, "schema": 2, "bundle": "snapshot.json",
                   "started_at": datetime.fromtimestamp((run / "inputs.json").stat().st_mtime, timezone.utc).isoformat(),
                   "SHA256": digest(stage / "snapshot.json"), "sources": sources,
                   "files": {path.name: digest(path) for path in stage.iterdir()}})
        stage.rename(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--include", type=Path, action="append", default=[])
    args = parser.parse_args()
    run = args.run.resolve(strict=True)
    export(run, args.output or MAIN / "reports/results/uav_03_fmd_static_gap" / run.name,
           [path.resolve(strict=True) for path in args.include])
