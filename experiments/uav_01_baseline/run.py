"""Replay, evaluate, or retrain the frozen Week 1 baseline."""
import argparse
import csv
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "code/tools"))
from uav_reproduction import PUBLIC, RESULTS, ROOT, VAL, digest, output_folder, write_json
from uav_week01_06 import BASE


EXPERIMENT = HERE.name
BEST = "e564c819603bf36ac6d93e3acb0359e7d5a75503ce78458209e9a28113280667"
WEIGHTS = RESULTS / EXPERIMENT / "models/local"
TRAIN = VAL.with_name("VisDrone2019-DET-train.zip")
BASELINE_CSV = RESULTS / EXPERIMENT / "uav_01_baseline_20261124T154238_4a518050/baseline_metrics.csv"


def frozen_sources(destination):
    manifest = json.loads((HERE / "source_manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["code_SHA256"].items():
        pinned = HERE / "frozen_source" / relative
        source = pinned if pinned.is_file() else PUBLIC / relative
        data = source.read_bytes()
        if sha256(data).hexdigest() != expected:
            data = data.replace(b"\r\n", b"\n")
        if sha256(data).hexdigest() != expected:
            raise ValueError(f"Historical source hash mismatch: {relative}")
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    return manifest


def require_hash(path, expected):
    if digest(path) != expected:
        raise ValueError(f"Unexpected input SHA256: {path}")


def selection_evidence(archive):
    import torch

    training = archive / "training/yolov8s"
    checkpoint = training / "weights/best.pt"
    require_hash(checkpoint, BEST)
    with (training / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    field = "metrics/mAP50-95(B)"
    high = max(float(row[field]) for row in rows)
    selected = [row for row in rows if float(row[field]) == high]
    metadata = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if len(selected) != 1 or metadata["train_metrics"]["fitness"] != high:
        raise ValueError("Best checkpoint metadata does not identify a unique training log maximum.")
    return {"source_run": archive.name, "checkpoint_sha256": BEST,
            "logged_epochs": len(rows), "best_logged_epoch": int(selected[0]["epoch"]),
            "selection_metric": "Ultralytics validation box mAP50-95 fitness",
            "internal_selection_fitness": high, "final_epoch_fitness": float(rows[-1][field]),
            "exported_checkpoint_epoch_field": metadata["epoch"],
            "training_results_sha256": digest(training / "results.csv")}


def compare_live(evaluation, csv_path=BASELINE_CSV):
    summary = json.loads((evaluation / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((csv_path.parent / "manifest.json").read_text(encoding="utf-8"))
    require_hash(csv_path, manifest["files"][csv_path.name])
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 1 or rows[0]["size"] != "all":
        raise ValueError("Expected one overall baseline CSV record.")
    row = rows[0]
    aggregate = summary["aggregate"]
    if (summary["images"] != 548 or summary["checkpoint_SHA256"] != BEST
            or summary["full_validation_status"] != "FULL_VALIDATION_PASS"
            or summary["protocol"] != "pr01_native10_one2many_FP32_ignore_union_0.5_cap3000"
            or aggregate["gt_count"] != int(row["gt_count"])
            or aggregate["evaluated_categories"] != int(row["evaluated_categories"])
            or aggregate["max_det"] != int(row["max_det"])
            or aggregate["area_reference"] != row["area_reference"]):
        raise ValueError("Evaluation summary differs in images, checkpoint, classes, GT count or protocol.")
    require_hash(evaluation / "ground_truth.json", summary["GT_SHA256"])
    receipt = json.loads((RESULTS / EXPERIMENT / "review/replay.json").read_text(encoding="utf-8"))
    require_hash(evaluation / "ground_truth.json", receipt["original_ground_truth_sha256"])
    scores = {}
    for name in ("map50", "map50_95", "ar50_95"):
        expected, observed = float(row[name]), float(aggregate[name])
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (expected, observed)):
            raise ValueError(f"Invalid baseline metric: {name}")
        scores[name] = {"reference": expected, "observed": observed,
                        "raw_exact_match": expected == observed,
                        "match_at_four_percent_decimals": f"{100*expected:.4f}" == f"{100*observed:.4f}",
                        "difference_pp": 100*(observed-expected)}
    return {"status": "MATCH" if all(v["match_at_four_percent_decimals"] for v in scores.values()) else "MISMATCH",
            "source_run": row["RunID"], "evaluation_run": evaluation.name, "images": summary["images"],
            "checkpoint_sha256": BEST, "baseline_csv_sha256": digest(csv_path),
            "ground_truth_sha256": summary["GT_SHA256"], "evaluation_wall_seconds": summary["total_seconds"],
            "scores": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=("replay", "evaluate", "train", "preflight"), default="replay")
    parser.add_argument("--archive", type=Path, default=BASE)
    parser.add_argument("--train-zip", type=Path, default=TRAIN)
    parser.add_argument("--val", type=Path, default=VAL)
    parser.add_argument("--init", type=Path, default=ROOT / "models/checkpoints/yolov8s.pt")
    parser.add_argument("--best", type=Path, default=WEIGHTS / "yolov8s_train640_best.pt")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = output_folder(EXPERIMENT, args.output)
    runtime = out / "runtime"
    runtime.mkdir()
    config = out / "execution_config.json"
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", YOLO_AUTOINSTALL="false", YOLO_OFFLINE="true",
               YOLO_CONFIG_DIR=str(runtime / "ultralytics"), MPLCONFIGDIR=str(runtime / "matplotlib"),
               TORCH_HOME=str(runtime / "torch"))
    (runtime / "ultralytics/Ultralytics").mkdir(parents=True)

    with TemporaryDirectory(prefix="historical_source_", dir=runtime) as tmp:
        scratch = Path(tmp) / "main"
        source = frozen_sources(scratch)
        task = scratch / "experiments/cv261102"
        if args.mode == "replay":
            write_json(config, {"mode": "cpu_replay", "archive": str(args.archive.resolve()),
                                "source_sha256": source["code_SHA256"]})
            process = subprocess.run([sys.executable, "-B", str(task / "validate_run.py"),
                                      "--run", str(args.archive.resolve())],
                                     env=dict(env, CUDA_VISIBLE_DEVICES=""), cwd=out, check=True,
                                     capture_output=True, text=True, encoding="utf-8")
            receipt = json.loads(process.stdout)
            write_json(out / "replay.json", receipt)
            write_json(out / "checkpoint_selection.json", selection_evidence(args.archive))
            for filename in ("metrics_overall.csv", "metrics_by_area.csv", "metrics_by_class.csv",
                             "training_epoch_times.csv", "evaluation_timing.csv"):
                shutil.copyfile(args.archive / "csv" / filename, out / filename)
            print(json.dumps(receipt, indent=2))
        else:
            from importlib.metadata import version
            if (version("torch"), version("ultralytics")) != (source["torch"], source["ultralytics"]):
                raise ValueError("Use the recorded PyTorch and Ultralytics versions.")
            require_hash(args.val, source["val_zip_SHA256"])
            if args.mode == "evaluate":
                require_hash(args.best, BEST)
                write_json(config, {"dataset_zip": str(args.val.resolve()), "checkpoint": str(args.best.resolve()),
                            "output_root": str(out), "checkpoint_sha256": BEST, "expected_images": 548,
                            "smoke_images": 5, "imgsz": 1280, "batch": 2, "device": "0",
                            "allow_cpu_fallback": False, "run_prefix": "uav01_live_eval"})
                env["PYTHONPATH"] = str(scratch / "code")
                command = [sys.executable, "-B", "-c", "import sys; from main.evaluation import evaluate;"
                           " evaluate(sys.argv[1], full=True)", str(config)]
            else:
                require_hash(args.train_zip, source["train_zip_SHA256"])
                require_hash(args.init, source["initial_weights_SHA256"])
                if args.init.name != "yolov8s.pt":
                    raise ValueError("The historical trainer expects the COCO file named yolov8s.pt.")
                write_json(config, {"train_zip": str(args.train_zip.resolve()), "val_zip": str(args.val.resolve()),
                            "weights": str(args.init.resolve()), "output_root": str(out), "epochs": 100,
                            "train_imgsz": 640, "eval_imgsz": 1280, "batch": 4, "eval_batch": 2,
                            "workers": 0, "seed": 42})
                command = [sys.executable, "-B", str(task / "run.py"), "--config", str(config)]
                if args.mode == "preflight":
                    command.append("--preflight")
            subprocess.run(command, env=env, cwd=out, check=True)
            if args.mode == "evaluate":
                evaluation, = out.glob("uav01_live_eval_*")
                comparison = compare_live(evaluation)
                write_json(out / "live_csv_comparison.json", comparison)
                if comparison["status"] != "MATCH":
                    raise ValueError(f"Live evaluation differs from the CSV. See {out / 'live_csv_comparison.json'}")
                print("Validation metrics match the baseline CSV to four decimal places in percent.")
    print(out)


if __name__ == "__main__":
    main()
