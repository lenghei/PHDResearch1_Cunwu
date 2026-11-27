"""Export the matched Week 3 checkpoint portfolio from sealed validation runs."""
import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from summarize_checkpoint_validation import LABELS, collect, plot, read_csv
from validate_checkpoints import CACHE_SHA, FILES, GT_SHA, MAIN, PROTOCOL, digest, load, require, verify, write_json
from validate_leaf import ASSETS, LEAF_PROTOCOL, MODELS

LABELS.update(external_leaf_yolo_n="LEAF-YOLO-N", external_leaf_yolo_s="LEAF-YOLO-S")


def source(run, artifact):
    return {"RunID": run.name, "artifact": artifact, "SHA256": digest(run / artifact)}


def collect_leaf(run):
    plan, guard, result = [load(run / name) for name in ("inputs.json", "supervisor.json", "result.json")]
    require(plan["models"] == list(MODELS) and plan["resolutions"] == [640, 1280]
            and plan["protocol"] == LEAF_PROTOCOL and plan["members_sha256"] == CACHE_SHA
            and plan["gt_sha256"] == digest(run / "ground_truth.json") == GT_SHA
            and guard["returncode"] == 0 and guard["stop_reason"] is None
            and result["status"] == "COMPLETE" and result["cells"] == 4 and result["image_evaluations"] == 2192,
            "Full LEAF run did not pass.")
    merged, classes, sources = [], [], []
    for name, (sha, _, params, _) in MODELS.items():
        for size in (640, 1280):
            cell = run / name / f"infer{size}"
            verify(cell, FILES, {"checkpoint": name, "resolution": size, "images": 548})
            row, binding = load(cell / "summary.json"), load(cell / "configuration.json")
            require(row["RunID"] == binding["run_id"] == run.name
                    and row["resolution"] == binding["resolution"] == size
                    and row["checkpoint"] == binding["checkpoint"] == name
                    and row["checkpoint_sha256"] == binding["checkpoint_sha256"] == sha
                    and row["parameters"] == params and binding["gt_sha256"] == GT_SHA
                    and binding["members_sha256"] == CACHE_SHA and binding["protocol"] == LEAF_PROTOCOL,
                    "LEAF result binding differs.")
            merged.append(row)
            metrics = load(cell / "metrics.json")["per_class"]
            full = {r["category_id"]: r for r in metrics if r["size"] == "all"}
            tiny = {r["category_id"]: r for r in metrics if r["size"] == "tiny"}
            require(set(full) == set(tiny) == set(range(1, 11))
                    and sum(r["gt_count"] for r in full.values()) == 38593, "LEAF category cohort differs.")
            classes.extend({"RunID": run.name, "checkpoint": name, "resolution": size,
                            "category_id": i, "category_name": r["category_name"],
                            "AP50": r["map50"], "AP50_95": r["map50_95"], "AR50_95": r["ar50_95"],
                            "valid_GT": r["gt_count"], "Tiny_AP": tiny[i]["map50_95"],
                            "Tiny_GT": tiny[i]["gt_count"]} for i, r in sorted(full.items()))
            sources.append(source(run, f"{name}/infer{size}/COMPLETE.json"))
    return merged, classes, sources + [source(run, name) for name in (
        "ground_truth.json", "inputs.json", "supervisor.json", "execution.json", "environment.json")]


def training_rows(yolo):
    import yaml

    rows = []
    for cfg in read_csv(yolo / "checkpoint_catalog.csv"):
        name = cfg["checkpoint"]
        folder = (MAIN / "code/models" / cfg["weight_file"]).parent
        args = folder / "args.yaml"
        provenance = load(folder / "source.json")
        dataset_sha = digest(folder / "training_dataset.yaml")
        require(digest(args) == cfg["args_sha256"] and provenance["sha256"] == cfg["sha256"]
                and dataset_sha == provenance["training_dataset_sha256"]
                and digest(folder / "training_results.csv") == cfg["training_log_sha256"],
                f"Training evidence differs: {name}")
        settings = yaml.safe_load(args.read_text(encoding="utf-8"))
        rows.append({"checkpoint": name, "label": LABELS[name], "cohort": cfg["cohort"],
                     "FM_D_architecture_scale_eligible": "YES", "backend": "Ultralytics",
                     "training_data": cfg["data"], "train_imgsz": cfg["train_imgsz"],
                     "logged_epochs": cfg["recorded_epochs"], "batch": cfg["batch"], "seed": cfg["seed"],
                     "optimizer": cfg["optimizer"], "lr0": cfg["lr0"], "weights_role": cfg["weights_role"],
                     "training_hsv_v": settings.get("hsv_v"), "training_mosaic": settings.get("mosaic"),
                     "training_mixup": settings.get("mixup"), "training_amp_setting": settings.get("amp"),
                     "training_framework_version": cfg["saved_ultralytics"], "parameters": cfg["parameters"],
                     "fused_parameters": None, "checkpoint_sha256": cfg["sha256"],
                     "training_args_sha256": cfg["args_sha256"], "training_log_sha256": cfg["training_log_sha256"],
                     "training_dataset_yaml_sha256": dataset_sha, "training_hyp_sha256": None})
    for name, (sha, cohort, params, fused) in MODELS.items():
        folder = ASSETS / name / "local"
        provenance = load(folder / "source.json")
        config, hyp, curve = (folder / p for p in ("opt.yaml", "hyp.yaml", "results.txt"))
        dataset = folder / "training_dataset.yaml"
        require(all(digest(p) == provenance["training_logs"][p.name]["sha256"]
                    for p in (config, hyp, curve, dataset))
                and provenance["sha256"] == sha, "LEAF training evidence differs.")
        o, h = (yaml.safe_load(p.read_text(encoding="utf-8")) for p in (config, hyp))
        rows.append({"checkpoint": name, "label": LABELS[name], "cohort": cohort,
                     "FM_D_architecture_scale_eligible": "YES" if name.endswith("_n") else "NO",
                     "backend": "LEAF-YOLO IDetect", "training_data": "VisDrone DET train / val (external YAML)",
                     "train_imgsz": o["img_size"][0], "logged_epochs": provenance["observed_epoch_rows"],
                     "batch": o["batch_size"], "seed": "UNKNOWN",
                     "optimizer": "SGD (archived trainer and flag)", "lr0": h["lr0"],
                     "weights_role": "released best; exact epoch UNKNOWN",
                     "training_hsv_v": h.get("hsv_v"), "training_mosaic": h.get("mosaic"),
                     "training_mixup": h.get("mixup"), "training_amp_setting": "UNKNOWN",
                     "training_framework_version": "external LEAF-YOLO; exact versions UNKNOWN",
                     "parameters": params, "fused_parameters": fused, "checkpoint_sha256": sha,
                     "training_args_sha256": digest(config), "training_log_sha256": digest(curve),
                     "training_dataset_yaml_sha256": digest(dataset), "training_hyp_sha256": digest(hyp)})
    require(len(rows) == len({r["checkpoint_sha256"] for r in rows}) == 8, "Training roster differs.")
    return rows


def operation_rows(yolo_cpu, leaf_cpu):
    rows = read_csv(yolo_cpu / "checkpoint_compute.csv")
    profile = load(leaf_cpu / "operation_profile.json")
    require(profile["status"] == "PASS", "LEAF CPU profile did not pass.")
    rows.extend({"checkpoint": r["checkpoint"], "checkpoint_sha256": r["sha256"],
                 "resolution": r["resolution"], "checkpoint_parameters": r["parameters"],
                 "GFLOPs_2xMAC": r["GFLOPs_2xMAC"]} for r in profile["profiles"])
    profiles = {(r["checkpoint"], int(r["resolution"])): r for r in rows}
    require(len(rows) == len(profiles) == 16, "Operation profiles are incomplete.")
    return profiles


def export(yolo, supplement, leaf, yolo_cpu, leaf_cpu, out):
    yolo_plan = load(yolo / "inputs.json")
    require(yolo_plan["protocol"] == PROTOCOL and yolo_plan["members_sha256"] == CACHE_SHA
            and yolo_plan["gt_sha256"] == GT_SHA, "The original six models use another protocol or population.")
    expected = [(r['checkpoint'], size) for r in read_csv(yolo / 'checkpoint_catalog.csv') for size in (640, 1280)]
    _, native, native_classes, native_sources = collect(yolo, supplement, expected)
    legacy, legacy_classes, leaf_sources = collect_leaf(leaf)
    training = training_rows(yolo)
    by_name = {r["checkpoint"]: r for r in training}
    profiles = operation_rows(yolo_cpu, leaf_cpu)
    require(len(native) == 12 and len(legacy) == 4 and len(native_classes + legacy_classes) == 160,
            "Expected eight models and sixteen complete cells.")
    native = [{**r, "backend": "Ultralytics"} for r in native]
    legacy = [{**r, "backend": "LEAF-YOLO IDetect"} for r in legacy]
    comparison = []
    for row in native + legacy:
        name = row["checkpoint"]
        info = by_name[name]
        resource = profiles[name, row["resolution"]]
        require(row["checkpoint_sha256"] == info["checkpoint_sha256"] == resource["checkpoint_sha256"]
                and row["parameters"] == int(resource["checkpoint_parameters"]), "Resource profile binding differs.")
        comparison.append({**row, "GFLOPs_2xMAC": resource["GFLOPs_2xMAC"],
                           "FM_D_architecture_scale_eligible": info["FM_D_architecture_scale_eligible"]})

    out.parent.mkdir(parents=True, exist_ok=True)
    require(not out.exists(), "Output already exists.")
    with TemporaryDirectory(dir=out.parent) as temp:
        stage = Path(temp) / out.name
        stage.mkdir()
        for filename, table in (("checkpoint_training.csv", training), ("validation_metrics.csv", comparison),
                                ("validation_per_class.csv", native_classes + legacy_classes)):
            with (stage / filename).open("x", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, list(table[0]))
                writer.writeheader()
                writer.writerows(table)
        plot(comparison, stage / "checkpoint_validation.png")
        identity = {"ExperimentID": "uav_03_fmd_static_gap", "RunID": leaf.name,
                    "status": "COMPLETE", "mode": "MIXED_BACKEND_CHECKPOINT_VALIDATION",
                    "scope": "DEVELOPMENT_VALIDATION"}
        summary = {**identity, "images_per_condition": 548, "cells": 16, "image_evaluations": 8768,
                   "checkpoint_scope": [{"models": 8, "FM_D_architecture_scale_eligible": 7,
                       "external_control": "LEAF-YOLO-S", "dataset": "VisDrone2019-DET-val Normal"}]}
        write_json(stage / "snapshot.json", {**identity, "files": {"summary.json": summary, "result.json": identity}})
        write_json(stage / "manifest.json", {**identity, "schema": 2, "bundle": "snapshot.json",
            "started_at": datetime.fromtimestamp((leaf / "inputs.json").stat().st_mtime, timezone.utc).isoformat(),
            "SHA256": digest(stage / "snapshot.json"),
            "sources": native_sources + leaf_sources + [source(yolo_cpu, "checkpoint_compute.csv"),
                                                        source(leaf_cpu, "operation_profile.json")],
            "files": {path.name: digest(path) for path in stage.iterdir()}})
        stage.rename(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("yolo", "leaf", "yolo-cpu", "leaf-cpu"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--yolo-complement", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    export(args.yolo.resolve(strict=True), [p.resolve(strict=True) for p in args.yolo_complement],
           args.leaf.resolve(strict=True), args.yolo_cpu.resolve(strict=True), args.leaf_cpu.resolve(strict=True),
           args.output or MAIN / "reports/results/uav_03_fmd_static_gap" / args.leaf.name)
