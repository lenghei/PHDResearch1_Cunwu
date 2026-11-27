"""Stream-validate the exported FM_D configuration matrix and all four CTCs."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from export_configuration_validation import (
    ARCH, SCALE, WEIGHTS, TRAIN_RES, AUG_OPTIONAL, INFER_RES, STRATEGY, POST,
    PRECISION, PREPROCESS, MANDATORY_AUG, FEATURE_COLUMNS, EXPECTED_CONFIGS,
    EXPECTED_TRAIN_VARIANTS,
)

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = HERE.parents[1] / "reports" / "results" / HERE.name / "configuration_validation.csv"


def validate(path: Path) -> dict:
    seen_ids = set()
    seen_vectors = set()
    seen_train = set()
    rows = 0
    groups = (
        tuple(f"Architecture__{x}" for x in ARCH),
        tuple(f"ModelScale__{x}" for x in SCALE),
        tuple(f"Weights__{x}" for x in WEIGHTS),
        tuple(f"TrainResolution__{x}" for x in TRAIN_RES),
        tuple(f"InferResolution__{x}" for x in INFER_RES),
        tuple(f"InferenceStrategy__{x}" for x in STRATEGY),
        tuple(f"PostProcessing__{x}" for x in POST),
        tuple(f"Precision__{x}" for x in PRECISION),
    )
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(FEATURE_COLUMNS).difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing leaf columns: {sorted(missing)}")
        for row in reader:
            rows += 1
            selected = {name for name in FEATURE_COLUMNS if int(row[name]) == 1}
            vector = bytes(int(row[name]) for name in FEATURE_COLUMNS)
            vector_digest = hashlib.sha256(vector).hexdigest()
            cid = row["configuration_id"]
            if cid in seen_ids or vector_digest in seen_vectors:
                raise ValueError(f"Duplicate configuration at row {rows}: {cid}")
            seen_ids.add(cid)
            seen_vectors.add(vector_digest)
            seen_train.add(row["train_variant_id"])
            if row["is_valid"] != "1" or row["constraint_validation_status"] != "PASS_4_OF_4":
                raise ValueError(f"Invalid status at row {rows}")
            if any(sum(int(row[c]) for c in group) != 1 for group in groups):
                raise ValueError(f"Exactly-one group violation at row {rows}")
            if not set(MANDATORY_AUG).issubset(selected):
                raise ValueError(f"Mandatory augmentation missing at row {rows}")
            if "LEAF_YOLO" in selected and not {"Nano", "VisDrone_FT", "Train640", "NMS"}.issubset(selected):
                raise ValueError(f"LEAF_YOLO constraint violation at row {rows}")
            if "EndToEnd_NMSFree" in selected and "YOLO26" not in selected:
                raise ValueError(f"EndToEnd_NMSFree constraint violation at row {rows}")
            if "COCO_ZeroShot" in selected and not (
                "Train640" in selected and not set(AUG_OPTIONAL[1:]).intersection(selected)
            ):
                raise ValueError(f"COCO_ZeroShot constraint violation at row {rows}")
            if "SAHI_Slicing" in selected and "TTA" in selected:
                raise ValueError(f"SAHI/TTA constraint violation at row {rows}")
            # This matrix is a logic enumeration, not a per-row empirical evaluation.
            for col in ("params_count", "gflops_2xmac", "inference_latency_mean_ms",
                        "cached_png_to_box_mean_ms", "peak_allocated_mib", "peak_reserved_mib",
                        "AP50", "AP50_95", "AR50_95"):
                if row[col].strip():
                    raise ValueError(f"Unmatched measured value populated in matrix: {col}, row {rows}")
    if rows != EXPECTED_CONFIGS:
        raise ValueError(f"Expected {EXPECTED_CONFIGS:,} rows, got {rows:,}")
    if len(seen_ids) != rows or len(seen_vectors) != rows:
        raise ValueError("Configuration uniqueness check failed")
    if len(seen_train) != EXPECTED_TRAIN_VARIANTS:
        raise ValueError(f"Expected {EXPECTED_TRAIN_VARIANTS} train variants, got {len(seen_train)}")
    return {
        "status": "PASS_FULL_CSV_REVALIDATION",
        "csv": path.name,
        "data_rows_excluding_header": rows,
        "columns": len(reader.fieldnames or []),
        "configuration_id_unique": len(seen_ids) == rows,
        "34_leaf_vectors_unique": len(seen_vectors) == rows,
        "train_variant_count": len(seen_train),
        "all_rows_pass_all_four_constraints": True,
        "measured_values_falsely_joined": 0,
        "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    args = parser.parse_args()
    result = validate(args.csv.resolve(strict=True))
    out = args.csv.resolve().with_name("configuration_validation_check.json")
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
