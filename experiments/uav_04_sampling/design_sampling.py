"""Build the Week 4–6 sampling design and audit sealed development results."""

import argparse
from collections import Counter
from hashlib import sha256
from itertools import combinations, product
import json
from pathlib import Path
from random import Random
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import perf_counter

import run_development as dev
from full_space_plan import checked
import uav_week01_06 as audit
from uav_reproduction import check_uvl, flama_python

W2 = dev.HERE.parent / "uav_02_fmp"
W3 = dev.HERE.parent / "uav_03_fmd_static_gap"
ORIGINAL = "reproduce_20261125T173153_5b6d550c"
CONTINUATION = "reproduce_20261126T000929_14dcac74"
REFERENCE_54 = "reproduce_20261126T001648_debeed9b"
UNIFORM_SEED, COVERAGE_SEED = 20261126, 20261127
PAIRWISE_SIZE, UNIFORM_SIZE, ACTIVE_SLOTS = 80, 100, 20


def dimensional_pairs(selection, dimensions, level_sets, positions):
    values = [next((value for value in levels if value in selection), "None") for levels in level_sets]
    return tuple((dimensions[i], values[i], dimensions[j], values[j]) for i, j in positions)


def pairwise_design(catalog, fmp):
    dimensions = tuple(fmp.GROUPS)
    level_sets = tuple(fmp.GROUPS.values())
    positions = tuple(combinations(range(len(dimensions)), 2))
    levels = [names if name == "illumination" else ("None", *names)
              for name, names in fmp.GROUPS.items()]
    all_pairs = [(dimensions[i], left, dimensions[j], right)
                 for i, j in positions for left, right in product(levels[i], levels[j])]
    bits = {pair: 1 << i for i, pair in enumerate(all_pairs)}
    population_counts, masks = Counter(), []
    for row in catalog:
        pairs = dimensional_pairs(set(row), dimensions, level_sets, positions)
        population_counts.update(pairs)
        masks.append(sum(bits[pair] for pair in pairs))
    remaining = sum(bits[pair] for pair in population_counts)
    order = list(range(len(catalog)))
    rng = Random(COVERAGE_SEED)
    rng.shuffle(order)
    reference = catalog.index(("Normal",))
    chosen = {reference: "CLEAN_REFERENCE"}
    remaining &= ~masks[reference]
    while remaining:
        winner = max(order, key=lambda i: (remaining & masks[i]).bit_count())
        if not (remaining & masks[winner]):
            raise ValueError("Feasible pair could not be covered.")
        remaining &= ~masks[winner]
        chosen[winner] = "GREEDY_COVER"
    require_range = 1 <= len(chosen) <= PAIRWISE_SIZE and 60 <= PAIRWISE_SIZE <= 100
    dev.require(require_range, "Pairwise sample size is outside the design.")
    filled = rng.sample([i for i in order if i not in chosen], PAIRWISE_SIZE - len(chosen))
    chosen.update(dict.fromkeys(filled, "SEEDED_FILL"))
    selected_counts = Counter(pair for i in chosen
                              for pair in dimensional_pairs(set(catalog[i]), dimensions, level_sets, positions))
    matrix = [{"dimension_a": a, "level_a": x, "dimension_b": b, "level_b": y,
               "valid_population_count": population_counts[a, x, b, y],
               "pairwise_selected_count": selected_counts[a, x, b, y],
               "status": "COVERED" if population_counts[a, x, b, y] else "INFEASIBLE"}
              for a, x, b, y in all_pairs]
    dev.require(set(selected_counts) == set(population_counts), "Pairwise coverage is incomplete.")
    return chosen, matrix


def render_selected(selected, catalog, fmp, render, source_val):
    import cv2
    import numpy as np

    cv2.setNumThreads(1)
    dev.require(dev.sha256(source_val) == render.VAL_SHA, "VisDrone validation ZIP differs.")
    reader = render.VisDroneZIP(source_val, 548)
    try:
        image, record, source = reader.record(8)
    finally:
        reader.close()
    attrs = render.choose_parameters()
    receipts = {}
    for index in selected:
        leaves = catalog[index]
        dev.require(fmp.valid_leaves(leaves), "Renderer rejected a BDD selection.")
        params = fmp.recipe(leaves, attrs)
        first, trace = render.generate(image, leaves, attrs, record["stem"], render.SEED)
        second, again = render.generate(image, leaves, attrs, record["stem"], render.SEED)
        dev.require(trace == again and np.array_equal(first, second)
                    and first.shape == image.shape and first.dtype == np.uint8
                    and sorted(effect["feature"] for effect in trace) == list(leaves),
                    "Selected degradation failed repeat rendering.")
        rolling = next((effect for effect in trace if effect["feature"] == "RollingShutter"), None)
        if rolling is not None:
            before, gt, converted = render.transform_annotations(record, rolling)
            dev.require((before, gt, converted) == render.transform_annotations(record, rolling)
                        and all(ann["area"] == ann["source_bbox"][2]*ann["source_bbox"][3]
                                for ann in gt["annotations"]), "Rolling annotation replay differs.")
        receipts[index] = {"p_id": fmp.configuration_id(leaves), "selected_leaves": "|".join(leaves),
                           "numeric_recipe_json": json.dumps(params, sort_keys=True, separators=(",", ":")),
                           "recipe_sha256": dev.identity(params), "render_seed": render.SEED,
                           "render_image_id": record["stem"], "render_test_split": "VisDrone2019-DET-val",
                           "render_pixel_sha256": sha256(first.tobytes()).hexdigest(),
                           "annotation_mode": "WARP_GT_AND_IGNORE_ORIGINAL_AREA" if rolling else "ORIGINAL_GT",
                           "validation": "BDD_VALID_CPU_RENDER_REPEAT_PASS"}
    return receipts, {"archive_sha256": render.VAL_SHA, "fixture_member": source,
                      "opencv": cv2.__version__, "numpy": np.__version__, "seed": render.SEED}


def detector_design(fmd):
    library_path = dev.MAIN / "code/models/model_library.csv"
    library = {row["model_id"]:row for row in audit.rows(library_path)}
    spec = (
        ("ck01_yolo26s_ft1280", "YOLO26", "Small", (640, 1280, "e2e640")),
        ("ck05_yolov8s_ft1280", "YOLOv8", "Small", (640, 1280)),
        ("yolov8s_cft640", "YOLOv8", "Small", (640, 960, 1280, 1536, "fp16_640")),
        ("ck06_yolo11s_ft1280", "YOLO11", "Small", (640, "sahi640", "tta640", "clahe640")),
        ("yolov8n_ft640", "YOLOv8", "Nano", (640,)),
        ("ck04_yolo26n_native10_full20", "YOLO26", "Nano", (640,)),
        ("external_leaf_yolo_n", "LEAF_YOLO", "Nano", (640, 1280)),
        ("yolo26s_s45_control", "YOLO26", "Small", (1280,)),
        ("yolo26s_s45_motion_targeted", "YOLO26", "Small", (1280,)),
    )
    settings = {"e2e640": (640, "FP32", "FullFrame", "EndToEnd_NMSFree", "None", False),
                "fp16_640": (640, "FP16", "FullFrame", "NMS", "None", False),
                "sahi640": (640, "FP32", "SAHI_Slicing", "NMS", "None", False),
                "tta640": (640, "FP32", "FullFrame", "NMS", "None", True),
                "clahe640": (640, "FP32", "FullFrame", "NMS", "CLAHE_Pre", False)}
    rows = []
    for name, architecture, scale, variants in spec:
        model = library[name]
        weight = dev.ROOT / model["primary_weight"]
        record = dev.ROOT / model["training_record"]
        dev.require(dev.sha256(weight) == model["weight_sha256"], "Detector weight SHA differs.")
        train = "Train" + model["train_imgsz"]
        pilot = train == "Train960"
        for variant in variants:
            size, precision, strategy, post, pre, tta = settings.get(
                variant, (variant, "FP32", "FullFrame", "NMS", "None", False))
            if pilot:
                declared = {architecture, scale, "VisDrone_FT", train, "HSV_Default",
                            "Geometric_Default", "Infer1280", "FP32", "FullFrame", "NMS"}
                if name.endswith("motion_targeted"):
                    declared.add("MotionBlurAug")
                leaves = sorted(declared)
                d_id = "pilot_" + dev.identity(leaves)[:16]
            else:
                d = fmd.declared_profile_configuration(
                    architecture, scale, "VisDrone_FT", train, size, precision,
                    strategy, post, tta=tta, preprocessing=pre)
                d_id, leaves = d["configuration_id"], d["selected_features"]
            binding = dev.identity({"checkpoint_sha256": model["weight_sha256"], "leaves": leaves})
            rows.append({"D_ID": d_id, "binding_id": "binding_" + binding[:16],
                         "cohort": "TRAIN960_MATCHED_PILOT" if pilot else "ORIGINAL_FM_D",
                         "model_name": model["model_name"], "model_id": name,
                         "checkpoint_sha256": model["weight_sha256"],
                         "training_record_sha256": dev.sha256(record), "train_px": model["train_imgsz"],
                         "training_epochs": model["recorded_epochs"], "training_seed": model["seed"],
                         "infer_px": size, "precision": precision, "strategy": strategy, "postprocessing": post,
                         "preprocessing": pre, "TTA": tta, "declared_leaves": "|".join(leaves),
                         "classes": "native10", "scope": "DEVELOPMENT_ONLY" if pilot else "CANDIDATE_PENDING_ADMISSION",
                         "runtime_status": "OPTION_EVALUATOR_ADMISSION_REQUIRED" if isinstance(variant, str)
                         and variant != "fp16_640" else "PENDING_MATCHED_PROTOCOL_ADMISSION",
                         "remarks": model["remarks"]})
    dev.require(len(rows) == 20 and len({r["binding_id"] for r in rows}) == 20,
                "Detector roster is incomplete or contains duplicates.")
    return rows


def reuse_audit(dev_runs, verified_snapshot, selections, detectors, fmp):
    reference = {row["pair_id"]: row for row in verified_snapshot}
    dev.require(len(reference) == 54, "The 54-pair reference differs.")
    targets = {(row["checkpoint_sha256"], row["declared_leaves"]): row for row in detectors}
    matrix, protocol = dev.design(100)
    frozen = {spec["pair_id"]: spec for group in matrix for spec in group}
    attrs = fmp.read_uvl_attributes(dev.FMP)
    results, seen = [], set()
    for run in dev_runs:
        dev.require(dev.load(run/"protocol.json") == protocol, "The development protocols differ.")
        for folder in sorted((run/"pairs").iterdir()):
            if not folder.is_dir() or folder.name not in frozen:
                continue
            identity = folder.name
            binding = dev.load(folder/"configuration.json")
            spec = frozen[identity]
            p, d = spec["degradation"], spec["detector"]
            recipe_hash = dev.identity({"P": p, "recipe": fmp.recipe(p["leaves"], attrs)})
            dev.require(identity not in seen and binding["spec"] == spec
                        and all(binding[key] == value for key, value in protocol.items())
                        and binding["P_recipe_sha256"] == recipe_hash
                        and binding["d_config_sha256"] == dev.identity(d), "Invalid historical pair binding.")
            receipt = dev.verify_pair(folder, binding)
            hashes = receipt["sha256"]
            dev.require(hashes["ground_truth.json"] == binding["gt_sha256"], "Historical GT hash differs.")
            if identity in reference:
                row = reference[identity]
                dev.require(row["source_RunID"] == run.name and row["P"] == p["degradation_id"]
                            and row["D"] == d["detector_id"]
                            and row["predictions_sha256"] == hashes["predictions.jsonl.gz"]
                            and row["gt_sha256"] == binding["gt_sha256"], "54-pair reference binding differs.")
            seen.add(identity)
            p_id = p["degradation_id"]
            d_match = targets.get((d["checkpoint_sha256"], "|".join(d["leaves"])))
            results.append({"pair_id": identity, "p_id": p_id, "source_D_ID": d["detector_id"],
                            "source_RunID": run.name, "in_original_54_snapshot": identity in reference,
                            "p_sources": selections.get(p_id, ""),
                            "candidate_D_binding_id": d_match["binding_id"] if d_match else "",
                            "exact_P_and_D_overlap": p_id in selections and d_match is not None,
                            "source_split": binding["source_split"], "images": binding["images"],
                            "dataset_sha256": binding["dataset_sha256"], "checkpoint_sha256": d["checkpoint_sha256"],
                            "configuration_sha256": hashes["configuration.json"], "P_recipe_sha256": recipe_hash,
                            "D_configuration_sha256": binding["d_config_sha256"],
                            "evaluation_code_sha256": dev.identity(binding["code_sha256"]),
                            "protocol_sha256": dev.sha256(run/"protocol.json"),
                            "raw_predictions": "PRESENT_SHA_VERIFIED",
                            "predictions_sha256": hashes["predictions.jsonl.gz"],
                            "gt_sha256": binding["gt_sha256"], "metrics_sha256": hashes["metrics.json"],
                            "formal_reuse": "NO_DATASET_AND_PROTOCOL_DIFFER"})
    dev.require(set(reference).issubset(seen), "The 54-pair snapshot is incomplete.")
    return results


def main():
    started = perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-run", default=ORIGINAL)
    parser.add_argument("--continuation-run", default=CONTINUATION)
    parser.add_argument("--reference-54", default=REFERENCE_54)
    args = parser.parse_args()
    dev_runs = [(dev.RUNS/name).resolve(strict=True) for name in (args.original_run,args.continuation_run)]
    snapshot = (dev.RUNS/args.reference_54).resolve(strict=True)
    dev.require(all(p.parent == dev.RUNS.resolve() for p in (*dev_runs, snapshot)), "Expected local Week 4 runs.")
    fmp_uvl, fmd_uvl = check_uvl("FM_P"), check_uvl("FM_D")
    fmp, fmd, _ = dev.models()
    week2 = audit.PUBLIC/"reports/results/uav_02_fmp/review"
    receipt_path = checked(week2, "full_space_validation.json", audit)
    reference = audit.load(receipt_path)
    sys.path.insert(0, str(W2))
    import validate_geometry as render

    with TemporaryDirectory() as tmp:
        bdd_output = Path(tmp)/"catalog.json"
        subprocess.run([str(flama_python()), "-B", str(dev.HERE/"bdd_catalog.py"),
                        str(fmp_uvl), str(bdd_output), "--seed", str(UNIFORM_SEED),
                        "--size", str(UNIFORM_SIZE)], check=True)
        bdd = dev.load(bdd_output)
    catalog = [tuple(c) for c in bdd.pop("catalog")]
    dev.require(bdd["leaf_catalog_sha256"] == reference["sat_catalog"]["leaf_catalog_sha256"]
                and reference["passed"] == len(catalog) and reference["failed"] == 0
                and sorted(fmp.combinations()) == catalog, "BDD, SAT and renderer populations differ.")
    uniform = bdd["uniform_indices"]
    cover, matrix = pairwise_design(catalog, fmp)
    unique = list(dict.fromkeys((*cover, *uniform)))
    receipts, renderer = render_selected(unique, catalog, fmp, render, dev.VAL)
    rows = []
    for i in unique:
        sources = [label for label, group in (("PAIRWISE", cover), ("UNIFORM_BDD", uniform)) if i in group]
        rows.append({**receipts[i], "sampling_sources": "|".join(sources),
                     "pairwise_role": cover.get(i, ""), "bdd_catalog_index_0_based": i,
                     "formal_status": "PLAN_ONLY"})
    roster = detector_design(fmd)
    reference_rows = audit.rows(snapshot/"verified_metrics.csv")
    checksums = {filename: value for value, filename in
                 (line.split("  ",1) for line in (snapshot/"SHA256SUMS.txt").read_text().splitlines())}
    dev.require(checksums["verified_metrics.csv"] == dev.sha256(snapshot/"verified_metrics.csv"),
                "54-pair snapshot SHA differs.")
    source_map = {row["p_id"]: row["sampling_sources"] for row in rows}
    reuse = reuse_audit(dev_runs, reference_rows, source_map, roster, fmp)
    out = dev.output_folder(dev.HERE.name)
    original_d = [row for row in roster if row["cohort"] == "ORIGINAL_FM_D"]
    with TemporaryDirectory() as tmp:
        temp = Path(tmp)/"orig_d.csv"
        dev.table(temp, original_d)
        sat_out = subprocess.check_output([str(flama_python()), "-B", str(dev.HERE/"full_space_plan.py"),
                                           str(temp)], text=True, encoding="utf-8")
        dev.require(int(sat_out.strip()) == len(original_d), "Incomplete FM_D constraint validation.")
    for row in roster:
        row["FM_D_status"] = ("SAT_VALID_DECLARED_LEAVES" if row in original_d else "OUTSIDE_ORIGINAL_TRAIN_RESOLUTION")
    active = [{"reservation": i, "p_id": "", "uncertainty_score": "", "status": "WAITING_FOR_ADMITTED_MODEL",
               "selection_reason": "Reserve for boundary uncertainty; exclude frozen and held-out P"}
              for i in range(1, ACTIVE_SLOTS+1)]
    provenance = [
        {"source": "PAIRWISE", "seed": COVERAGE_SEED, "size": len(cover),
         "method": "Normal reference, seeded greedy feasible level-pair cover, seeded fill", "prevalence": "NO"},
        {"source": "UNIFORM_BDD", "seed": UNIFORM_SEED, "size": len(uniform),
         "method": bdd["method"], "prevalence": "PRIMARY_IF_MEASURED"},
        {"source": "ACTIVE_RESERVED", "seed": "", "size": len(active),
         "method": "Rank unseen P near predicted pass/fail boundary after frozen model fit", "prevalence": "NO"},
    ]
    for name, data in (
        ("FM_P_200_sampling_manifest.csv", rows), ("pairwise_coverage_matrix.csv", matrix),
        ("uniform_random_sample.csv", [{**receipts[i], "draw_number": k, "bdd_catalog_index_0_based": i,
                                        "inclusion_probability": bdd["inclusion_probability"]}
                                        for k,i in enumerate(uniform,1)]),
        ("active_sampling_candidates.csv", active), ("sampling_provenance.csv", provenance),
        ("existing_54_pairs_reuse_audit.csv", [row for row in reuse if row["in_original_54_snapshot"]]),
        ("paused_development_reuse_audit.csv", reuse), ("detector_20_configurations.csv", roster),
    ):
        dev.table(out/name, data)
    pause = dev.load(dev_runs[-1]/"result.json")
    original = [r for r in reuse if r["in_original_54_snapshot"]]
    dev.require(pause["total_unique_pairs"] == len(reuse), "Paused run count differs from sealed evidence.")
    results = {"RunID": out.name, "status": "DESIGN_VALIDATED_FORMAL_EVALUATION_PENDING",
               "plan_pages": [7, 9, 11, 12, 16], "BDD": bdd, "renderer": renderer,
               "FM_D_UVL_sha256": dev.sha256(fmd_uvl), "pairwise_n": len(cover), "uniform_n": len(uniform),
               "pairwise_before_fill": sum(role != "SEEDED_FILL" for role in cover.values()),
               "pairwise_feasible": sum(row["status"] == "COVERED" for row in matrix),
               "pairwise_infeasible": sum(row["status"] == "INFEASIBLE" for row in matrix),
               "P_overlap": len(set(cover)&set(uniform)), "fixed_unique_P": len(rows),
               "active_reserved": len(active), "active_selected": 0,
               "max_unique_P_after_active": len(rows)+len(active),
               "detector_checkpoints": len({r["checkpoint_sha256"] for r in roster}),
               "D_count": len(roster), "D_original_FM_D": len(original_d),
               "D_train960_development": len(roster)-len(original_d),
               "audited_54": {"P": len({r["p_id"] for r in original}),
                              "D": len({r["source_D_ID"] for r in original}),
                              "overlapping_P": sorted({r["p_id"] for r in original if r["p_sources"]}),
                              "P_D_matches": sum(r["exact_P_and_D_overlap"] for r in original)},
               "paused_development": {"pairs": len(reuse), "P": len({r["p_id"] for r in reuse}),
                                      "complete_P": pause["completed_groups"], "D": len({r["source_D_ID"] for r in reuse}),
                                      "overlapping_P": sorted({r["p_id"] for r in reuse if r["p_sources"]}),
                                      "P_D_matches": sum(r["exact_P_and_D_overlap"] for r in reuse),
                                      "formal_reusable_pairs": 0, "source_RunIDs": [r.name for r in dev_runs]},
               "formal_500_pairs": 0, "formal_split_gate": "DATASET_BLOCKED", "GPU_started": False,
               "threshold": {"Tiny_AP50_95": .10, "cached_latency_ms": 100},
               "source_sha256": {str(p.relative_to(dev.MAIN)): dev.sha256(p) for p in
                                 (receipt_path, fmp_uvl, fmd_uvl, Path(__file__),
                                  dev.HERE/"bdd_catalog.py", W2/"render_bindings.json",
                                  W2/"validate_geometry.py", W2/"geometry.py",
                                  Path(render.generate.__code__.co_filename))}}
    dev.write(out/"sampling_design_summary.json", results)
    dev.index(out, {"ExperimentID": dev.HERE.name, "status": results["status"],
                    "CPU_wall_seconds": perf_counter()-started})
    dev.text_file(out/"SHA256SUMS.txt",
                  "\n".join(f"{dev.sha256(path)}  {path.name}" for path in sorted(out.iterdir()) if path.is_file()))
    print(out, flush=True)


if __name__ == "__main__":
    main()
