"""Export every valid configuration of the original UAV-GAP FM_D model.

This enumerator is deliberately independent of FlamaPy so that the full CSV can
be generated without materialising a SAT model.  The four predicates are read
from the unchanged fm_d.uvl and mirrored below.  A separate FlamaPy check is
provided in flamapy_count.py and should be run in the project's .venv_flamapy.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE.parents[1] / "reports" / "results" / HERE.name / "configuration_validation.csv"

ARCH = ("YOLOv8", "YOLO11", "YOLO26", "LEAF_YOLO")
SCALE = ("Nano", "Small")
WEIGHTS = ("COCO_ZeroShot", "VisDrone_FT")
TRAIN_RES = ("Train640", "Train1280")
AUG_OPTIONAL = ("Albu_Default", "NoiseAug", "MotionBlurAug", "ExposureAug", "JPEGAug", "HazeAug")
INFER_RES = ("Infer640", "Infer960", "Infer1280", "Infer1536")
STRATEGY = ("FullFrame", "SAHI_Slicing")
POST = ("NMS", "EndToEnd_NMSFree")
PRECISION = ("FP32", "FP16", "INT8")
PREPROCESS = ("CLAHE_Pre", "Gamma_Pre", "Denoise_Pre", "Deblur_Pre")
MANDATORY_AUG = ("HSV_Default", "Geometric_Default")

FEATURE_COLUMNS = (
    *(f"Architecture__{x}" for x in ARCH),
    *(f"ModelScale__{x}" for x in SCALE),
    *(f"Weights__{x}" for x in WEIGHTS),
    *(f"TrainResolution__{x}" for x in TRAIN_RES),
    *MANDATORY_AUG,
    *AUG_OPTIONAL,
    *(f"InferResolution__{x}" for x in INFER_RES),
    *(f"InferenceStrategy__{x}" for x in STRATEGY),
    *(f"PostProcessing__{x}" for x in POST),
    *(f"Precision__{x}" for x in PRECISION),
    *PREPROCESS,
    "TTA",
)
assert len(FEATURE_COLUMNS) == 34, len(FEATURE_COLUMNS)

META_COLUMNS = (
    "configuration_id", "train_variant_id", "is_valid", "validation_method",
    "constraint_set_id", "constraint_validation_status", "train_variant_count_total",
    "architecture", "model_scale", "nominal_model_scale_params_m", "weights",
    "train_resolution", "inference_resolution", "inference_strategy", "post_processing",
    "precision", "preprocessing_selected", "tta", "selected_leaves",
    "measurement_status", "checkpoint_id", "measured_profile_id", "profile_join_status",
    "params_count", "gflops_2xmac", "inference_latency_mean_ms",
    "cached_png_to_box_mean_ms", "peak_allocated_mib", "peak_reserved_mib",
    "AP50", "AP50_95", "AR50_95", "evidence_ref",
)
FIELDNAMES = META_COLUMNS + FEATURE_COLUMNS
EXPECTED_CONFIGS = 635_904
EXPECTED_TRAIN_VARIANTS = 844
CONSTRAINT_SET_ID = "FM_D_ORIGINAL_4_CTC"


def sha_id(prefix: str, selected: set[str] | frozenset[str]) -> str:
    payload = json.dumps(sorted(selected), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return prefix + hashlib.sha256(payload).hexdigest()[:16]


def read_and_check_uvl(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    expected = (
        "LEAF_YOLO => Nano & VisDrone_FT & Train640 & NMS",
        "EndToEnd_NMSFree => YOLO26",
        "COCO_ZeroShot => Train640 & !NoiseAug & !MotionBlurAug & !ExposureAug & !JPEGAug & !HazeAug",
        "SAHI_Slicing => !TTA",
    )
    normalised = " ".join(text.split())
    for rule in expected:
        if " ".join(rule.split()) not in normalised:
            raise ValueError(f"The original FM_D UVL is missing or changed constraint: {rule}")
    if text.count("=>") != 4:
        raise ValueError("Expected exactly four FM_D cross-tree constraints.")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def training_variants():
    """Yield all valid training-time feature combinations (844 expected)."""
    for architecture, scale, weights, train_res, aug_bits in itertools.product(
        ARCH, SCALE, WEIGHTS, TRAIN_RES, range(1 << len(AUG_OPTIONAL))
    ):
        additions = frozenset(name for i, name in enumerate(AUG_OPTIONAL) if (aug_bits >> i) & 1)
        # CTC 1: LEAF_YOLO => Nano & VisDrone_FT & Train640 (NMS is inference-time).
        if architecture == "LEAF_YOLO" and not (
            scale == "Nano" and weights == "VisDrone_FT" and train_res == "Train640"
        ):
            continue
        # CTC 3: COCO_ZeroShot => Train640 and none of the five extra augmentations.
        if weights == "COCO_ZeroShot" and not (
            train_res == "Train640" and not additions.intersection(AUG_OPTIONAL[1:])
        ):
            continue
        train_selected = set((architecture, scale, weights, train_res, *MANDATORY_AUG, *additions))
        yield architecture, scale, weights, train_res, additions, train_selected


def valid_config(selected: set[str]) -> bool:
    # Exactly-one groups and mandatory augmentation leaves.
    groups = (ARCH, SCALE, WEIGHTS, TRAIN_RES, INFER_RES, STRATEGY, POST, PRECISION)
    if any(sum(name in selected for name in group) != 1 for group in groups):
        return False
    if not set(MANDATORY_AUG).issubset(selected):
        return False
    # Optional preprocessing is an OR group: none or a non-empty subset.
    # Optional augmentation and TTA are independent booleans.
    if "LEAF_YOLO" in selected and not {"Nano", "VisDrone_FT", "Train640", "NMS"}.issubset(selected):
        return False
    if "EndToEnd_NMSFree" in selected and "YOLO26" not in selected:
        return False
    if "COCO_ZeroShot" in selected and not (
        "Train640" in selected and not set(AUG_OPTIONAL[1:]).intersection(selected)
    ):
        return False
    if "SAHI_Slicing" in selected and "TTA" in selected:
        return False
    return True


def build_row(architecture, scale, weights, train_res, additions, train_selected,
              infer_res, strategy, post, precision, preprocess, tta, uvl_sha):
    selected = set(train_selected)
    selected.update((infer_res, strategy, post, precision))
    selected.update(preprocess)
    if tta:
        selected.add("TTA")
    if not valid_config(selected):
        raise AssertionError("Enumerator emitted an invalid FM_D configuration")
    config_id = sha_id("d_", selected)
    train_id = sha_id("tv_", train_selected)
    row = {
        "configuration_id": config_id,
        "train_variant_id": train_id,
        "is_valid": 1,
        "validation_method": "EXHAUSTIVE_INDEPENDENT_ENUMERATOR_MIRRORING_UVL_CTC",
        "constraint_set_id": CONSTRAINT_SET_ID,
        "constraint_validation_status": "PASS_4_OF_4",
        "train_variant_count_total": EXPECTED_TRAIN_VARIANTS,
        "architecture": architecture,
        "model_scale": scale,
        "nominal_model_scale_params_m": "2.5" if scale == "Nano" else "10.0",
        "weights": weights,
        "train_resolution": train_res,
        "inference_resolution": infer_res.removeprefix("Infer"),
        "inference_strategy": strategy,
        "post_processing": post,
        "precision": precision,
        "preprocessing_selected": "|".join(sorted(preprocess)) if preprocess else "None",
        "tta": int(tta),
        "selected_leaves": "|".join(sorted(selected)),
        "measurement_status": "NOT_MEASURED_FOR_EXACT_FULL_CONFIGURATION",
        "checkpoint_id": "",
        "measured_profile_id": "",
        "profile_join_status": "NOT_JOINED_TRAIN_AUGMENTATION_PROVENANCE_NOT_BOUND_TO_UNIQUE_PROFILE",
        "params_count": "",
        "gflops_2xmac": "",
        "inference_latency_mean_ms": "",
        "cached_png_to_box_mean_ms": "",
        "peak_allocated_mib": "",
        "peak_reserved_mib": "",
        "AP50": "",
        "AP50_95": "",
        "AR50_95": "",
        "evidence_ref": f"fm_d.uvl#sha256={uvl_sha}",
    }
    for col in FEATURE_COLUMNS:
        feature_name = col.split("__", 1)[1] if "__" in col else col
        row[col] = int(feature_name in selected)
    return row


def enumerate_rows(uvl_sha):
    for architecture, scale, weights, train_res, additions, train_selected in training_variants():
        for infer_res, strategy, post, precision, pre_bits, tta in itertools.product(
            INFER_RES, STRATEGY, POST, PRECISION, range(1 << len(PREPROCESS)), (False, True)
        ):
            # CTC 2: EndToEnd_NMSFree => YOLO26.
            if post == "EndToEnd_NMSFree" and architecture != "YOLO26":
                continue
            # CTC 4: SAHI_Slicing => !TTA.
            if strategy == "SAHI_Slicing" and tta:
                continue
            preprocess = frozenset(name for i, name in enumerate(PREPROCESS) if (pre_bits >> i) & 1)
            yield build_row(architecture, scale, weights, train_res, additions, train_selected,
                            infer_res, strategy, post, precision, preprocess, tta, uvl_sha)


def export(output: Path) -> dict:
    uvl = HERE / "fm_d.uvl"
    uvl_sha = read_and_check_uvl(uvl)
    output.parent.mkdir(parents=True, exist_ok=True)
    train_ids = set()
    config_ids = set()
    config_vectors = set()
    row_count = 0
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, extrasaction="raise")
        writer.writeheader()
        for row in enumerate_rows(uvl_sha):
            cid = row["configuration_id"]
            if cid in config_ids:
                raise ValueError(f"Duplicate configuration_id: {cid}")
            config_ids.add(cid)
            vector = tuple(row[c] for c in FEATURE_COLUMNS)
            if vector in config_vectors:
                raise ValueError("Duplicate 34-leaf configuration vector")
            config_vectors.add(vector)
            train_ids.add(row["train_variant_id"])
            writer.writerow(row)
            row_count += 1
    if row_count != EXPECTED_CONFIGS:
        raise ValueError(f"Expected {EXPECTED_CONFIGS:,} configurations; got {row_count:,}")
    if len(train_ids) != EXPECTED_TRAIN_VARIANTS:
        raise ValueError(f"Expected {EXPECTED_TRAIN_VARIANTS} training variants; got {len(train_ids)}")
    if len(config_ids) != row_count or len(config_vectors) != row_count:
        raise ValueError("Uniqueness validation failed")
    return {
        "status": "PASS_INDEPENDENT_EXHAUSTIVE_ENUMERATION",
        "source_model": str(uvl.name),
        "source_model_sha256": uvl_sha,
        "leaf_feature_count": len(FEATURE_COLUMNS),
        "cross_tree_constraint_count": 4,
        "valid_configuration_count": row_count,
        "training_variant_count": len(train_ids),
        "configuration_id_unique": True,
        "feature_vector_unique": True,
        "all_rows_pass_four_constraints": True,
        "measured_attribute_join": "NOT_JOINED; checkpoint training-augmentation leaves are not uniquely evidenced in supplied package",
        "measurement_values_in_matrix": 0,
        "measurement_status": "LOGIC_ONLY_NOT_MEASURED_FOR_EXACT_FULL_CONFIGURATION",
        "flamapy_independent_confirmation": "NOT_RUN_IN_THIS_CONTAINER; run flamapy_count.py in project .venv_flamapy",
        "csv_path": str(output),
        "csv_columns": len(FIELDNAMES),
        "data_rows_excluding_header": row_count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    summary = export(args.output.resolve())
    summary_path = args.output.resolve().with_name("configuration_validation_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
