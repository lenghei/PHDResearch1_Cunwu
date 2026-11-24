"""Publish selected, checksum-verified evidence from the six weekly runs."""
import argparse
import csv
from pathlib import Path
from tempfile import TemporaryDirectory
import sys

sys.dont_write_bytecode = True

from uav_week01_06 import (EXP_IDS, PUBLIC, RUNS, checked_copy, digest, load, require,
                           save, table)


EXPORTS = [
    ("BASELINE_DEVELOPMENT_VALIDATION", ("baseline_metrics.csv", "training_epoch_times.csv", "evaluation_timing.csv")),
    ("CPU_LOGICAL_AND_PIXEL_VALIDATION", ("sampling_population.json", "generator_tests.json")),
    ("LOGICAL_COUNTS_AND_STATIC_PROXY", ("static_gap_counts.csv", "measured_attributes.csv")),
    ("FROZEN_DESIGN_ONLY", ("sampling_coverage.csv",)),
    ("FORMAL_DATASET_BLOCKED", ("test_dev_admission.json", "historical_pair_eligibility.csv")),
    ("FORMAL_GAP_UNKNOWN", ("threshold_sensitivity.csv",)),
]


def validated_file(run, file):
    proofs = {name: value for value, name in
              (line.split("  ", 1) for line in (run / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines())}
    path = run / file
    require(file in proofs and path.is_file() and not path.is_symlink() and digest(path) == proofs[file],
            f"Invalid input checksum: {file}")
    return path, proofs[file]


def baseline_plot(out, times):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with times.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    epochs = [int(r["epoch"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.2, 3.4), layout="constrained")
    for field, label in (("train_plus_validation_seconds", "Training and validation"),
                         ("train_seconds", "Training")):
        ax.plot(epochs, [float(x[field]) for x in rows], label=label, linewidth=1.15)
    ax.set(xlabel="Epoch", ylabel="Wall time (s)", title="YOLOv8s baseline epoch times")
    ax.legend(frameon=False, loc="upper right")
    ax.grid(axis="y", alpha=.2)
    fig.savefig(out / "baseline_epoch_timing.png", dpi=150)
    plt.close(fig)


def export(experiment, run_id):
    require(run_id.startswith(experiment + "_") and Path(run_id).name == run_id, "Invalid RunID")
    run = RUNS / experiment / run_id
    require(run.resolve().parent == (RUNS / experiment).resolve(), "Run escapes result root")
    record = load(validated_file(run, "run.json")[0])
    require(record["ExperimentID"] == experiment and record["RunID"] == run_id
            and record["status"] != "FAIL", "Invalid weekly result identity")
    week = EXP_IDS.index(experiment) + 1
    scope, filenames = EXPORTS[week-1]
    paths = {name: validated_file(run, name)[0] for name in filenames}
    meta = {"ExperimentID": experiment, "RunID": run_id, "status": record["status"], "scope": scope}
    summary = {**meta, "cpu_wall_seconds": record["CPU_wall_seconds"],
               "supervisor_acceptance": record["supervisor_status"]}
    result = dict(summary)
    final = PUBLIC / "reports/results" / experiment / run_id
    require(not final.exists(), "Published run already exists")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = TemporaryDirectory(prefix=".staging-", dir=final.parent)
    out = Path(staging.name)
    public_files = []
    if week == 1:
        summary.update(training_images=record["membership"]["train"]["n"],
                       images=record["membership"]["val"]["n"], epochs=record["training_epochs"],
                       training_wall_seconds=record["training_wall_seconds"],
                       evaluation_wall_seconds=record["evaluation_wall_seconds"],
                       **record["metrics"])
        result.update(images=summary["images"], trained_epochs=summary["epochs"])
        for name in filenames:
            checked_copy(paths[name], out / name)
            public_files.append(name)
        baseline_plot(out, paths["training_epoch_times.csv"])
        public_files.append("baseline_epoch_timing.png")
    elif week == 2:
        count = load(paths["sampling_population.json"])
        tests = load(paths["generator_tests.json"])
        summary["population"] = [count]
        summary["generator"] = [{"deterministic_leaves": len(tests["fixtures"]),
            "rolling_shutter": tests["rolling_shutter_status"],
            "historical_rendered_images": tests["historical_generated_images"]}]
    elif week == 3:
        name = "static_gap_counts.csv"
        checked_copy(paths[name], out / name)
        public_files.append(name)
        fields = ("RunID", "checkpoint", "checkpoint_sha256", "resolution", "images", "precision", "scope",
                  "checkpoint_parameters", "GFLOPs_2xMAC", "mean_pipeline_ms", "peak_allocated_bytes", "peak_reserved_bytes")
        with paths["measured_attributes.csv"].open(encoding="utf-8-sig", newline="") as stream:
            profiles = [{k: row[k] for k in fields} for row in csv.DictReader(stream)]
        table(out / "development_detector_profiles.csv", profiles)
        public_files.append("development_detector_profiles.csv")
        logic = record["FlamaPy"]
        summary["logical_counts"] = [{"legal": logic["valid_count"], "training_variants": logic["training_variants"],
                                      "leaves": logic["leaves"], "constraints": logic["constraints"]}]
    elif week == 4:
        name = filenames[0]
        checked_copy(paths[name], out / name)
        public_files.append(name)
        summary["design"] = [{"planned_pairs": record["pairs"], "P": record["p"], "D": record["d"],
                              "seed": record["seed"], "measured_pairs": 0,
                              "safe_population": record["safe_population"]}]
    elif week == 5:
        summary["admission"] = [{"status": record["scientific_admission"], **record["archive_counts"],
            "planned_pairs": record["planned_pairs"], "formal_measured_pairs": record["measured_pairs"],
            "image_evaluations": record["measured_images"], "GPU_seconds": record["GPU_seconds"],
            "exploratory_VM43_cells": record["historical_exploratory_test_dev_cells"]}]
    elif week == 6:
        name = filenames[0]
        checked_copy(paths[name], out / name)
        public_files.append(name)
        summary["empirical_status"] = [{"formal_measured_pairs": record["formal_pairs"],
            "planned_pairs": record["planned_pairs"], "unknown_P": record["unknown_P"],
            "tiny_AP50_95_tau": record["primary"]["tiny_AP50_95"],
            "cached_latency_limit_ms": record["primary"]["cached_latency_ms"]}]
    provenance = [{"RunID": run_id, "artifact": n, "SHA256": validated_file(run, n)[1]}
                  for n in ("run.json", *filenames)]
    if week == 1:
        provenance.append({"RunID": record["source_RunID"], "artifact": "coco_regeneration.json",
                           "SHA256": digest(RUNS / "cv261102" / record["source_RunID"] / "coco_regeneration.json")})
    save(out / "snapshot.json", {**meta, "files": {"summary.json": summary, "result.json": result}})
    public_files.append("snapshot.json")
    save(out / "manifest.json", {**meta, "schema": 2, "started_at": record["started_utc"],
        "bundle": "snapshot.json", "SHA256": digest(out / "snapshot.json"),
        "files": {n: digest(out / n) for n in public_files}, "sources": provenance})
    out.rename(final)
    staging.cleanup()
    print(final)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_ids", nargs=len(EXP_IDS))
    args = parser.parse_args()
    for experiment, run in zip(EXP_IDS, args.run_ids, strict=True):
        export(experiment, run)


if __name__ == "__main__":
    main()
