"""Read-only metrics summary for an existing VM1 RunID."""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def elapsed_string(seconds):
    if seconds is None:
        return "NOT_RECORDED"
    return f"{seconds:.2f}s ({int(seconds // 60)}m {seconds % 60:.2f}s)"


def generate_run_report(root, *, started_at=None, ended_at=None, wall_seconds=None):
    """Create a run-level summary without touching measured metrics or raw data."""
    root = Path(root).resolve()
    record = read_json(root / "run.json") or {}
    result = read_json(root / "result.json") or {}
    dataset = read_json(root / "dataset_gate.json") or {}
    checkpoint = read_json(root / "checkpoint_gate.json") or {}
    config = read_json(root / "configuration.json") or {}
    environment = read_json(root / "environment.json") or {}
    uvl = read_json(root / "bridge.json") or {}
    manifest = dataset.get("manifest") or []
    before_ignore = (sum(row.get("object_count_before_ignore", 0) for row in manifest)
                     if manifest and all("object_count_before_ignore" in row for row in manifest) else None)
    # Prefer the actual source-condition summaries to avoid stale root-level copies.
    conditions = []
    filenames = ["CLEAN", "DEGRADED"]
    if (root / "conditions.json").is_file():
        filenames = [row["directory"] for row in read_json(root / "conditions.json")]
    for folder_name in filenames:
        summary = read_json(root / folder_name / "summary.json")
        if summary is None:
            continue
        agg = summary.get("aggregate", {})
        conditions.append({
            "name": summary.get("configuration", {}).get("name", folder_name),
            "configuration_id": summary.get("configuration", {}).get("configuration_id"),
            "features": summary.get("configuration", {}).get("features"),
            "degradation_parameters": summary.get("configuration", {}).get("parameters"),
            "images": summary.get("images"),
            "GT_evaluated": agg.get("gt_count"),
            "AP50": agg.get("map50"),
            "AP50_95": agg.get("map50_95"),
            "AR50": agg.get("ar50"),
            "AR50_95": agg.get("ar50_95"),
            "predictions_at_conf_0001": summary.get("raw_predictions"),
            "inference_call_seconds": summary.get("network_seconds"),
            "mean_predict_call_ms_per_image": (1000 * summary["network_seconds"] / summary["images"]
                                                if summary.get("images") and summary.get("network_seconds") is not None else None),
            "CUDA_event_seconds": summary.get("CUDA_event_seconds"),
            "COCO_evaluation_seconds": summary.get("evaluation_seconds"),
            "condition_wall_seconds": summary.get("condition_wall_seconds"),
            "peak_GPU_allocated_bytes": summary.get("GPU_peak_allocated_bytes"),
            "peak_GPU_reserved_bytes": summary.get("GPU_peak_reserved_bytes"),
            "condition_summary": folder_name + "/summary.json",
        })
    pair = read_json(root / "paired_comparison.json")
    first_condition = conditions[0]["name"] if conditions else None
    metrics_detail = read_json(root / first_condition / "metrics.json") if first_condition else None
    size_counts = ({x["size"]: x["gt_count"] for x in metrics_detail.get("per_area", [])}
                   if metrics_detail else None)
    class_counts = ({x.get("category_name", str(x.get("category_id"))): x["gt_count"]
                     for x in metrics_detail.get("per_class", []) if x.get("size") == "all"}
                    if metrics_detail else None)
    training = checkpoint.get("training_provenance", {})
    record_wall = record.get("wall_seconds") if wall_seconds is None else wall_seconds
    # Historical result.total_seconds starts inside the inference pipeline,
    # not at process launch; never label it as total process duration.
    timing = {
        "whole_run_wall_seconds": record_wall,
        "pipeline_seconds": result.get("total_seconds"),
        "start_utc": started_at or record.get("started_at"),
        "end_utc": ended_at or record.get("ended_at"),
        "clock_note": "whole_run_wall_seconds includes admission and reporting; pipeline_seconds excludes setup",
    }
    report = {
        "schema_version": 1,
        "RunID": record.get("RunID", root.name),
        "ExperimentID": record.get("ExperimentID"),
        "mode": record.get("mode"),
        "status": record.get("status", result.get("status", "UNKNOWN")),
        "scope": result.get("scope"),
        "dataset": {"name": "VisDrone2019-DET-val" if dataset else None,
                    "images_available": dataset.get("images"),
                    "images_per_condition": result.get("images_per_condition"),
                    "image_evaluations": sum(x["images"] or 0 for x in conditions),
                    "original_annotation_objects": before_ignore,
                    "GT_after_protocol": conditions[0]["GT_evaluated"] if conditions else None,
                    "GT_by_size": size_counts,
                    "GT_by_class": class_counts,
                    "classes": len(checkpoint.get("class_order", [])) or None,
                    "ZIP_SHA256": dataset.get("ZIP_SHA256") or config.get("dataset_zip_sha256")},
        "model": {"checkpoint_id": checkpoint.get("checkpoint_id") or result.get("checkpoint_id"),
                  "architecture": checkpoint.get("architecture"),
                  "parameter_count": checkpoint.get("parameters"),
                  "SHA256": checkpoint.get("SHA256"),
                  "class_order": checkpoint.get("class_order"),
                  "training_configured": training.get("configured"),
                  "training_args_SHA256": training.get("SHA256"),
                  "selected_best_epoch": training.get("selected_best_epoch")},
        "inference": {"imgsz": config.get("imgsz"), "batch": config.get("batch"),
                      "device": config.get("device"), "precision": "FP32" if conditions else None,
                      "metric_protocol": conditions and (read_json(root / conditions[0]["name"] / "summary.json") or {}).get("protocol"),
                      "predict_settings": {"conf": 0.001, "iou": 0.7, "max_det": 3000,
                                           "max_nms": 30000, "end2end": False, "half": False,
                                           "rect": False, "class_aware": True,
                                           "multi_label": True, "augment": False}
                      if conditions else None,
                      "seed": config.get("seed"), "sampling_seed": config.get("sampling_seed"),
                      "degradation_seed": config.get("degradation_seed")},
        "flamapy": {"dataset_valid_configs": (uvl.get("dataset") or {}).get("valid_count"),
                    "detector_valid_configs": (uvl.get("yolo") or {}).get("valid_count"),
                    "versions": uvl.get("versions")},
        "conditions": conditions,
        "paired_comparison": {"file": "paired_comparison.json" if pair else None,
                              "rows": pair.get("rows") if pair else None},
        "timing": timing,
        "environment": environment.get("packages"),
        "source_file": "source_SHA256.json" if (root / "source_SHA256.json").is_file() else None,
        "data_note": "Development validation. Original objects include ignored/filtered items."
    }
    from .config import save
    save(root / "summary.json", report)
    (root / "summary.txt").write_text(format_terminal(report) + "\n", encoding="utf-8")
    return report


def format_terminal(report):
    lines = ["=" * 65, "UAV-GAP VM1 | RUN REPORT", "RunID: " + str(report["RunID"]),
             "Status: " + str(report["status"]), "Mode: " + str(report["mode"])]
    d, m, inf = report["dataset"], report["model"], report["inference"]
    lines.extend([
        f"Images: {d['images_per_condition']} per condition; {d['image_evaluations']} evaluations",
        f"Annotations: {d['original_annotation_objects']} original; {d['GT_after_protocol']} in COCO evaluation",
        f"Checkpoint: {m['checkpoint_id']} | {m['architecture']} | parameters: {m['parameter_count']}",
        f"Inference: imgsz={inf['imgsz']} batch={inf['batch']} device={inf['device']} {inf['precision']}",
        "Predict settings: " + json.dumps(inf.get("predict_settings"), sort_keys=True),
        f"Seeds: evaluation={inf['seed']} sample={inf['sampling_seed']} degradation={inf['degradation_seed']}",
    ])
    if m.get("training_configured"):
        lines.append("Training args: " + json.dumps(m["training_configured"], ensure_ascii=False))
    for row in report["conditions"]:
        def pct(k):
            v = row.get(k)
            return f"{100*v:.2f}%" if v is not None else "N/A"
        lines.extend([
            f"[{row['name']}] config={row['configuration_id']} parameters={json.dumps(row['degradation_parameters'])}",
            f"  AP50={pct('AP50')} AP50:95={pct('AP50_95')} AR50:95={pct('AR50_95')}",
            f"  Predictions={row['predictions_at_conf_0001']} GT={row['GT_evaluated']} | infer={elapsed_string(row['inference_call_seconds'])} | COCO={elapsed_string(row['COCO_evaluation_seconds'])} | condition={elapsed_string(row['condition_wall_seconds'])}",
            f"  Mean predict-call time/image: {row['mean_predict_call_ms_per_image']:.2f} ms (not end-to-end latency)" if row.get('mean_predict_call_ms_per_image') is not None else "  Predict-call timing: NOT_RECORDED",
            f"  PyTorch peak GPU allocated: {row['peak_GPU_allocated_bytes'] / 1048576:.1f} MiB" if row.get('peak_GPU_allocated_bytes') is not None else "  GPU memory: NOT_RECORDED",
        ])
    rows = report["paired_comparison"]["rows"] or []
    for metric in ("map50", "map50_95", "ar50_95"):
        match = next((x for x in rows if x.get("size") == "all" and x.get("metric") == metric), None)
        if match:
            lines.append(f"Delta {metric} (degraded-clean): {100*match['delta_degraded_minus_clean']:+.2f} pp")
    t = report["timing"]
    lines.extend(["Run wall time: " + elapsed_string(t["whole_run_wall_seconds"]),
                  "Saved pipeline time: " + elapsed_string(t["pipeline_seconds"]),
                  "Saved report: summary.json + summary.txt", "=" * 65])
    return "\n".join(lines)
