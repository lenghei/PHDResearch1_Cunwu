"""Verify the first six UAV-GAP deliverables using local evidence only."""
import argparse
import csv
from datetime import datetime, timezone
from hashlib import sha256
import importlib.util
from itertools import combinations
import json
import math
from pathlib import Path, PurePosixPath
import random
import shutil
import subprocess
import sys
import time
import traceback
from uuid import uuid4
import zipfile

sys.dont_write_bytecode = True
PUBLIC = Path(__file__).resolve().parents[2]
ROOT = PUBLIC.parent
EXPS = PUBLIC / "experiments/experiments_test"
RUNS = ROOT / "datasets/local_runs"
BASE = RUNS / "cv261102/cv261102_full_20261031T000013_d0be8587"
AUDIT = RUNS / "cv261207/cv261207_audit_20261104T193649_d1b63492"
ADMISSION = RUNS / "cv261207/cv261207_admission_20261104T211249_32d83def"
COMPLIANCE = RUNS / "cv261207/cv261207_supervisor_compliance_20261109T121122_70d953f1"
STATIC = RUNS / "cv261116/cv261116_static_20261101T223010_016ba75c"
VM41 = RUNS / "vm41_degradation_coverage/vm41_degradation_coverage_coverage_20261116T020508_dfa7879d"
PLAN = None
BASELINE_SOURCE_ZIP = None
PLAN_SHA = "2f7620a017843e6f638591e85d373ead99c6e8192ab58fa3153665ed22b124af"
ALIGN = EXPS / "vm36_supervisor_reproduction"
FMP = ALIGN / "feature_models/fm_p.uvl"
FMD = ALIGN / "feature_models/fm_d.uvl"
P_IMPL = EXPS / "cv261109/fm_p.uvl"
D_IMPL = EXPS / "cv261116/fm_d.uvl"
FLAMA = ROOT / ".venv_flamapy/Scripts/python.exe"
MODELS = ROOT / "models/checkpoints"
EXP_IDS = ("uav_01_baseline", "uav_02_fmp", "uav_03_fmd_static_gap",
           "uav_04_sampling", "uav_05_evaluation", "uav_06_gap_analysis")
SOURCE_HASHES = {}
_USED = {}
sys.path.insert(0, str(PUBLIC / "code"))


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def save(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def table(path, data, columns=None):
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns or list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def digest(path):
    with path.open("rb") as stream:
        return sha256_file(stream)


def sha256_file(stream):
    digest = sha256()
    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source(path, expected=None):
    if path not in SOURCE_HASHES:
        SOURCE_HASHES[path] = {"path": str(path), "sha256": digest(path), "bytes": path.stat().st_size}
    item = SOURCE_HASHES[path]
    require(expected is None or item["sha256"] == expected, f"SHA mismatch for {path.name}")
    _USED[path] = item
    return item


def checked_copy(src, dst, expected=None):
    proof = source(src, expected)
    with dst.open("xb") as stream, src.open("rb") as data:
        shutil.copyfileobj(data, stream)
    require(digest(dst) == proof["sha256"], f"Copy mismatch for {dst.name}")
    return proof


def text_file(path, text):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text.rstrip() + "\n")


def module(name, path):
    source(path)
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def execution_command(week):
    return f'& "{sys.executable}" -B "{Path(__file__).resolve()}" --week {week} --plan "{PLAN}"'


def git(*args):
    return subprocess.run(["git", "-C", str(PUBLIC), *args], capture_output=True)


def baseline(out):
    manifest = load(BASE / "manifest.json")
    result, cfg, system = (manifest[k] for k in ("result", "configuration", "system"))
    source(BASE / "manifest.json")
    verify_home = EXPS / "cv261102"
    sys.path.insert(0, str(verify_home))
    try:
        verify = module("uav_baseline_verify", verify_home / "validate_run.py")
        verify.audit(BASE, recompute=False)
    finally:
        sys.path.pop(0)
    csv_root = BASE / "csv"
    for filename, sha in result["csv_SHA256"].items():
        source(csv_root / filename, sha)
    for name, original in (("baseline_metrics.csv", "metrics_overall.csv"),
                           ("training_epoch_times.csv", "training_epoch_times.csv"),
                           ("evaluation_timing.csv", "evaluation_timing.csv")):
        checked_copy(csv_root / original, out / name)
    record = load(BASE / "coco_regeneration.json")
    gt = BASE / "validation/ground_truth.json"
    raw = BASE / "validation/predictions_raw.json"
    source(raw, result["predictions_raw_SHA256"])
    source(gt, result["GT_SHA256"])
    source(BASE / "validation/predictions_coco.json", record["predictions_COCO_SHA256"])
    source(BASE / "validation/metrics.json", record["metrics_SHA256"])
    source(BASE / "coco_regeneration.json")
    checkpoint = source(BASE / "training/yolov8s/weights/best.pt", manifest["source_checkpoint_SHA256"])
    catalog = source(MODELS / "yolov8s_cft640/best.pt", checkpoint["sha256"])
    init = source(MODELS / "yolov8s.pt", system["initial_weights_SHA256"])
    save(out / "checkpoint_manifest.json", {"source_RunID": BASE.name, "best": checkpoint,
        "catalog": catalog, "initialization": init, "epochs": result["trained_epochs"]})
    recovery = []
    for path, expected in manifest["code_SHA256"].items():
        current = PUBLIC / path
        live = source(current)["sha256"] if current.is_file() else None
        revisions = [manifest["git"]["commit"]]
        revisions.extend(git("rev-list", "--all", "--reflog", "--", path).stdout.decode().splitlines())
        origin = "WORKTREE" if live == expected else ""
        archived = ""
        for commit in dict.fromkeys(revisions):
            if origin:
                break
            historical = git("show", commit + ":" + path)
            if historical.returncode != 0:
                continue
            blob = historical.stdout
            for encoding, data in (("blob", blob), ("Git checkout CRLF", blob.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))):
                if sha256(data).hexdigest() == expected:
                    origin = f"{commit}:{path} [{encoding}]"
                    break
        if not origin and BASELINE_SOURCE_ZIP:
            member = "main/" + path
            with zipfile.ZipFile(BASELINE_SOURCE_ZIP) as archive:
                if member in archive.namelist():
                    data = archive.read(member)
                    if sha256(data).hexdigest() == expected:
                        source(BASELINE_SOURCE_ZIP)
                        local = out / "recovered_source" / path
                        local.parent.mkdir(parents=True, exist_ok=True)
                        with local.open("xb") as stream:
                            stream.write(data)
                        archived = local.relative_to(out).as_posix()
                        origin = "Verified historical ZIP member: " + member
        recovery.append({"file": path, "recorded_sha256": expected, "current_sha256": live,
                         "origin": origin, "archived_source": archived, "recovered": bool(origin)})
    table(out / "runtime_source_recovery.csv", recovery)
    members = {}
    original = ROOT / "datasets/VisDrone"
    for split, expected_count in (("train", 6471), ("val", 548)):
        zip_path = original / f"VisDrone2019-DET-{split}.zip"
        source(zip_path, system[f"{split}_zip_SHA256"])
        directory = BASE / "prepared_dataset/images" / split
        labels_dir = BASE / "prepared_dataset/labels" / split
        member_rows = []
        with zipfile.ZipFile(zip_path) as archive:
            image_names = zip_members(archive, "images", (".jpg", ".png", ".jpeg"))
            label_names = zip_members(archive, "annotations", (".txt",))
            require(len(image_names) == len(label_names) == expected_count and image_names.keys() == label_names.keys(),
                    f"Wrong number of {split} image/label pairs")
            for stem, name in sorted(image_names.items()):
                original_sha = sha256(archive.read(name)).hexdigest()
                prepared_image = directory / PurePosixPath(name).name
                label_path = labels_dir / (stem + ".txt")
                member_rows.append({"image_id": stem, "original_image_sha256": original_sha,
                                    "original_label_sha256": sha256(archive.read(label_names[stem])).hexdigest(),
                                    "prepared_image_sha256": digest(prepared_image) if prepared_image.is_file() else None,
                                    "prepared_label_sha256": digest(label_path) if label_path.is_file() else None})
        table(out / f"{split}_members.csv", member_rows)
        identical = sum(x["original_image_sha256"] == x["prepared_image_sha256"] for x in member_rows)
        require(identical == expected_count, f"Prepared {split} images differ from original archive")
        require(sum(bool(x["prepared_label_sha256"]) for x in member_rows) == expected_count,
                f"Missing prepared {split} labels")
        members[split] = {"n": expected_count, "matching_image_bytes": identical,
                          "prepared_labels": expected_count,
                          "training_time_member_hashes": "archive identity and surviving prepared view; no creation-time per-label SHA"}
    payload = load(gt)
    gt_images = {Path(x["file_name"]).stem for x in payload["images"]}
    require(gt_images == {x["image_id"] for x in member_rows}, "COCO GT and val ZIP IDs differ")
    trace = load(BASE / "validation/trace.json")
    source(BASE / "validation/trace.json")
    expected_trace = {x["image_id"]: x for x in member_rows}
    require(len(trace) == 548 and set(expected_trace) == {PurePosixPath(x["ZIP_member"]).stem for x in trace},
            "Runtime trace and validation membership differ")
    for item in trace:
        expected = expected_trace[PurePosixPath(item["ZIP_member"]).stem]
        require(item["image_SHA256"] == expected["original_image_sha256"]
                and item["annotation_SHA256"] == expected["original_label_sha256"]
                and item["dtype"] == "torch.float32"
                and item["NCHW"] == [cfg["eval_batch"], 3, cfg["eval_imgsz"], cfg["eval_imgsz"]],
                "Historical input hashes, FP32 or tensor shape differ")
    from main.dataset import VisDroneZIP
    from main.labels import ground_truth
    reader = VisDroneZIP(original / "VisDrone2019-DET-val.zip", expected=548)
    try:
        records = [reader.record(i)[1] for i in range(548)]
    finally:
        reader.close()
    regenerated, _ = ground_truth(records, "native10")
    require(regenerated == payload, "Original-coordinate ignore-union GT did not reproduce")
    save(out / "ignore_and_gt_replay.json", {"status": "TECHNICAL_PASS", "GT_SHA256": digest(gt),
        "images": len(records), "categories": [x["name"] for x in payload["categories"]],
        "original_coordinate_annotations_after_ignore": len(payload["annotations"]),
        "runtime_FP32_traces_verified": len(trace)})
    args = BASE / "training/yolov8s/args.yaml"
    checked_copy(args, out / "historical_training_args.yaml")
    checked_copy(BASE / "validation/configuration.json", out / "historical_evaluation_configuration.json")
    counts = {k: manifest[k] for k in ("training_wall_seconds", "evaluation_wall_seconds")}
    source(BASE / "training/checkpoint.json")
    source(BASE / "data_preparation.json")
    evidence = {"source_RunID": BASE.name, "status": "TECHNICAL_PASS" if all(x["recovered"] for x in recovery) else "PARTIAL",
                "metrics_status": "TECHNICAL_PASS_SAVED_CPU_REPLAY_VERIFIED",
                "meeting_status": "SUPERVISOR_ACCEPTANCE_PENDING", "author_status": "AUTHOR_REVIEW_PENDING",
                "scope": manifest["scope"], "platform": system, "parameters": cfg,
                "metrics": {k: result[k] for k in ("AP50", "AP50_95", "AR50_95")},
                **counts, "training_epochs": result["trained_epochs"], "membership": members,
                "raw_predictions_sha256": result["predictions_raw_SHA256"], "CPU_replay": record,
                "runtime_source_files_verified": sum(x["recovered"] for x in recovery),
                "runtime_source_files_total": len(recovery), "historical_git": manifest["git"],
                "evaluation_protocol": load(BASE / "validation/summary.json")["protocol"],
                "historical_trace_records": len(trace)}
    save(out / "baseline_evidence.json", evidence)
    text_file(out / "reproduction_command.md", f"""# Baseline replay

From the Git root, set `$run` to the local CV261102 directory and run:

```powershell
& ../.venv/Scripts/python.exe -B experiments/experiments_test/cv261102/validate_run.py --run $run
```

This replays saved predictions on CPU. The weekly audit instead verifies the
previous replay receipt and compares the archived evidence. Neither command trains.
A live GPU validation is a separate meeting check.""")
    return evidence


def zip_members(archive, folder, suffixes):
    selected = [name for name in archive.namelist() if PurePosixPath(name).parent.name == folder
                and name.lower().endswith(suffixes) and "__MACOSX" not in name
                and not PurePosixPath(name).name.startswith("._")]
    result = {PurePosixPath(name).stem: name for name in selected}
    require(len(result) == len(selected), f"Duplicate {folder} stems")
    return result

def flama(out, path, domain):
    validator = ALIGN / "validate_models.py"
    for file in (validator, path, EXPS / "cv261109/flamapy_select.py",
                 EXPS / "cv261116/flamapy_count.py"):
        source(file)
    cmd = [str(FLAMA), "-B", str(validator), "--model", str(path), "--domain", domain]
    process = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", check=True)
    result = json.loads(process.stdout.splitlines()[-1])
    save(out / "flamapy_verification.json", result)
    text_file(out / "flamapy_command.txt", subprocess.list2cmdline(cmd))
    return result


def get_fmp():
    return module("uav_fmp", EXPS / "cv261109/model.py")


def read_attributes(model, path):
    import re
    attributes = dict.fromkeys(model.ALL_LEAVES, {})
    for line in path.read_text(encoding="utf-8").splitlines():
        match = model.ATTRIBUTE_RE.match(line)
        if match and match[1] in attributes:
            attributes[match[1]] = {p[1]: float(p[2]) for p in
                (model.NUMERIC_RE.fullmatch(entry) for entry in match[2].split(","))}
    text = path.read_text(encoding="utf-8").split("constraints\n", 1)[0]
    require(all(re.search(r"^\s+" + leaf + r"(?:\s*\{|\s*$)", text, re.M) for leaf in attributes),
            "FM_P model leaf mismatch")
    return attributes


def rolling_geometry(out, cv2, np, gen, attrs, image, record, render_seed):
    recipe = attrs["RollingShutter"]
    cases = [
        ("synthetic_interior", (64, 96), [31, 3, 18, 43], "GT"),
        ("synthetic_left_edge", (64, 96), [0, 0, 3, 64], "ignore"),
        ("synthetic_right_edge", (64, 96), [94, 1, 2, 52], "ignore"),
    ]
    for kind, boxes in (("GT", [item["bbox"] for item in record["objects"]]),
                        ("ignore", record["ignore_regions"] + record["other_regions"])):
        require(bool(boxes), "Demo image lacks a geometry test region")
        cases.extend(("actual_val_box_" + str(i), image.shape[:2], box, kind)
                     for i, box in enumerate(boxes[:3]))
    results = []
    for name, shape, box, kind in cases:
        h, w = shape
        x, y, bw, bh = box
        x0, y0 = max(0, math.floor(x)), max(0, math.floor(y))
        x1, y1 = min(w, math.ceil(x + bw)), min(h, math.ceil(y + bh))
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y0:y1, x0:x1] = 255
        rng = np.random.default_rng(gen.effect_seed(render_seed, record["stem"], "RollingShutter"))
        transformed, trace = gen.apply_rolling_shutter(mask, rng=rng, **recipe)
        yy, xx = np.nonzero(transformed)
        old = [x0, y0, x1, y1]
        new = [int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1] if xx.size else None
        overflow = transformed.copy()
        overflow[y0:y1, x0:x1] = 0
        results.append({"fixture": name, "region_type": kind, "width": w, "height": h,
                        "original_xyxy": json.dumps(old), "rendered_support_xyxy": json.dumps(new),
                        "original_bbox_area": bw * bh, "moved_outside_original": int(np.count_nonzero(overflow)),
                        "phase_rad": trace["phase_rad"], "support_sha256": sha256(transformed.tobytes()).hexdigest(),
                        "scope": "DIAGNOSTIC_MASK_NOT_A_GT_TRANSFORM"})
    require(any(r["moved_outside_original"] for r in results), "Expected an unchanged-GT counterexample")
    table(out / "geometry_safety.csv", results)
    return len(results)


def week2(out):
    import cv2
    import numpy as np
    from main.dataset import VisDroneZIP

    model = get_fmp()
    flama_result = flama(out, FMP, "FM_P")
    source(P_IMPL)
    configs = list(model.combinations())
    safe = [c for c in configs if "RollingShutter" not in c]
    require((len(set(configs)), len(safe)) == (flama_result["valid_count"], 26496), "FM_P totals differ")
    proof = rows(
        COMPLIANCE / "physical_justification_matrix.csv")
    source(COMPLIANCE / "physical_justification_matrix.csv")
    source(VM41 / "RollingShutter_geometry_counterexample.json")
    constraints = FMP.read_text(encoding="utf-8").split("constraints\n", 1)[1].strip().splitlines()
    table(out / "constraint_evidence.csv", [
        {"constraint": c.strip(), "status": "PRESERVED_SAT_VALID_PHYSICAL_REVIEW_PENDING",
         "PDF_pages": "5-6", "physical_reference": matching["physical_reference"],
         "author_review": matching["next_action"], "source": COMPLIANCE.name}
        for c, matching in zip(constraints, proof, strict=True)])
    checked_copy(FMP, out / "FM_P.uvl")
    attrs = model.read_uvl_attributes(P_IMPL)
    pdf_attrs = read_attributes(model, FMP)
    params = []
    reasons = {
        "illumination": "Affine intensity and clipping; synthetic exposure proxy.",
        "motion": "Normalized linear-kernel blur; orientation not calibrated.",
        "defocus": "Normalized disk convolution; no camera calibration.",
        "gaussian": "Independent additive Gaussian samples in pixel units.",
        "shot": "Intensity-proportional Poisson approximation, not a sensor measurement.",
        "compression": "One additional JPEG encode/decode.",
        "atmosphere": "Constant-depth haze or overlaid rain; no scene/weather calibration.",
        "altitude": "Image resampling and restoration of canvas, not a change of camera pose.",
        "rolling": "Sinusoidal horizontal row shift, reflected edges; GT and ignore mapping blocked.",
    }
    for family, leaves in model.GROUPS.items():
        for leaf in leaves:
            require(all(attrs[leaf][k] == v for k, v in pdf_attrs[leaf].items()),
                    f"Original numeric value changed: {leaf}")
            params.append({"feature": leaf, "family": family, "original_parameters": json.dumps(pdf_attrs[leaf]),
                           "generator_parameters": json.dumps(attrs[leaf]), "source_code": f"generator.py:leaf_{leaf}",
                           "interpretation": reasons[family], "augmentation_scope": "SYNTHETIC",
                           "author_review": "Required for physical interpretation and engineering bindings",
                           "geometry": "BLOCKED_GT_AND_IGNORE" if leaf == "RollingShutter" else "ORIGINAL_CANVAS"})
    table(out / "feature_parameters.csv", params)
    population = {"legal": len(configs), "geometry_safe_population": len(safe),
                  "excluded_rolling_shutter": len(configs)-len(safe),
                  "UVL_SHA256": flama_result["UVL_SHA256"], "binding_UVL_SHA256": digest(P_IMPL)}
    save(out / "sampling_population.json", population)
    sys.path.insert(0, str(P_IMPL.parent))
    try:
        generator = module("uav_fmp_generator", P_IMPL.with_name("generator.py"))
    finally:
        sys.path.pop(0)
    archive = ROOT / "datasets/VisDrone/VisDrone2019-DET-val.zip"
    source(archive, load(AUDIT / "dataset_gate.json")["archives"]["val"]["sha256"])
    reader = VisDroneZIP(archive, expected=548)
    try:
        image, record, provenance = reader.record(0)
    finally:
        reader.close()
    seed = 20261109
    render_seed = 20261207
    chosen = random.Random(seed).choice(sorted(safe))
    generated, trace = generator.generate(image, chosen, attrs, record["stem"], render_seed)
    repeated, trace2 = generator.generate(image.copy(), chosen, attrs, record["stem"], render_seed)
    require(np.array_equal(generated, repeated) and trace == trace2, "Nondeterministic real image rendering")
    preview = np.concatenate([image, generated], axis=1)
    require(cv2.imwrite(str(out / "random_render.png"), preview), "Could not write comparison image")
    fixture = np.random.default_rng(seed).integers(0, 256, (79, 103, 3), dtype=np.uint8)
    cases = {}
    for leaf in sorted(model.ALL_LEAVES):
        leaves = min((c for c in configs if leaf in c), key=lambda c: (len(c), c))
        pixels, a = generator.generate(fixture, leaves, attrs, "fixture", render_seed)
        again, b = generator.generate(fixture.copy(), leaves, attrs, "fixture", render_seed)
        require(a == b and pixels.shape == fixture.shape and pixels.dtype == np.uint8
                and np.array_equal(pixels, again), f"Renderer mismatch: {leaf}")
        cases[leaf] = {"leaves": leaves, "pixel_sha256": sha256(pixels.tobytes()).hexdigest()}
    geometry_tests = rolling_geometry(out, cv2, np, generator, attrs, image, record, render_seed)
    prior = RUNS / "cv261109/cv261109_fmp_p_936a456fc700e8d8_20261101T173419_099c950c"
    prior_result = load(prior / "result.json")
    source(prior / "result.json")
    source(prior / "image_manifest.jsonl", prior_result["image_manifest_SHA256"])
    save(out / "generator_tests.json", {"status": "PASS_DETERMINISTIC_PIXELS", "fixtures": cases,
         "geometry_tests": geometry_tests, "rolling_shutter_status": "BLOCKED_GT_IGNORE_AND_AREA_MAPPING",
         "versions": {"opencv": cv2.__version__, "numpy": np.__version__},
         "transform_order": model.ORDER, "historical_generation_RunID": prior.name,
         "historical_generated_images": prior_result["generated_images"]})
    save(out / "random_render_manifest.json", {"selection_seed": seed, "render_seed": render_seed,
        "selection_method": "uniform from sorted geometry-safe legal configurations",
        "config_id": model.configuration_id(chosen), "leaves": chosen, "parameters": model.recipe(chosen, attrs),
        "source_member": provenance, "trace": trace, "source_pixel_sha256": sha256(image.tobytes()).hexdigest(),
        "output_pixel_sha256": sha256(generated.tobytes()).hexdigest(),
        "preview_sha256": digest(out / "random_render.png")})
    text_file(out / "live_render_command.md", "# Live CPU render\n\n```powershell\n" +
              execution_command(2) + "\n```\n\nOpen random_render.png in the new Week 2 run.")
    return {**population, "status": "PARTIAL", "logical_status": "TECHNICAL_PASS",
            "deterministic_leaves": len(cases), "rolling_diagnostics": geometry_tests,
            "rolling_shutter": "BLOCKED_GT_IGNORE_AND_AREA_MAPPING",
            "rendered_source_images": 1, "historical_rendered_source_images": prior_result["generated_images"]}


def safe_sources():
    return (
        ("cv261102", "yolov8s_cft640", "best.pt"),
        ("ck01", "ck01_yolo26s_ft1280", "best.pt"),
        ("ck04", "ck04_yolo26n_native10_full20", "best.pt"),
        ("ck05", "ck05_yolov8s_ft1280", "best.pt"),
        ("ck06", "ck06_yolo11s_ft1280", "best.pt"),
        ("leaf_n", "external_leaf_yolo_n", "best.pt"),
        ("leaf_s", "external_leaf_yolo_s", "best.pt"),
        ("yolov8n640", "yolov8n_ft640", "best.pt"),
        ("yolo11n640", "yolo11n_ft640", "best.pt"),
    )

def week3(out):
    from importlib import metadata
    import yaml

    model = get_fmp()
    fmd = module("uav_fmd", EXPS / "cv261116/model.py")
    result = flama(out, FMD, "FM_D")
    independent = fmd.enumerate_count()
    require(result["valid_count"] == independent["independent_configurations"]
            and result["training_variants"] == independent["independent_training_variants"], "FM_D count mismatch")
    checked_copy(FMD, out / "FM_D.uvl")
    compatibility = rows(COMPLIANCE / "detector_compatibility.csv")
    require({r["feature"] for r in compatibility} == fmd.LEAVES, "Missing detector compatibility row")
    source(COMPLIANCE / "detector_compatibility.csv")
    for row in compatibility:
        row.update(RESOURCE_FEASIBLE="NOT_ADMITTED_PER_CONFIGURATION",
                   CHECKPOINT_AVAILABLE="CHECK_SEPARATE_CHECKPOINT_INVENTORY",
                   RUNTIME_VERIFIED=row["GPU_TESTED_CURRENT_248_PROTOCOL"])
    table(out / "detector_compatibility.csv", compatibility)
    training = rows(COMPLIANCE / "checkpoint_readiness.csv")
    source(COMPLIANCE / "checkpoint_readiness.csv")
    checkpoint_rows = []
    for row in training:
        folder = MODELS / row["model_id"]
        args = folder / "args.yaml"
        weights = [x for x in sorted(folder.glob("*.pt")) if digest(x) == row["checkpoint_sha256"]]
        weight = weights[0] if weights else None
        if weight:
            source(weight, row["checkpoint_sha256"])
        train = yaml.safe_load(args.read_text(encoding="utf-8")) if args.is_file() else {}
        args_matches = bool(train and source(args)["sha256"] == row["training_args_sha256"])
        checkpoint_rows.append({**row, "local_weight": str(weight) if weight else None,
            "weight_identity": "VERIFIED" if weight else "NOT_LOCATED_AT_REGISTERED_PATH",
            "actual_training_args_sha_match": args_matches, "training_dataset_record": train.get("data"),
            "train_population_status": "SEE_BASELINE_MEMBER_EVIDENCE" if row["model_id"] == "yolov8s_cft640"
                                       else "NO_COMPLETE_CREATION_TIME_MEMBER_PROOF_IN_THIS_INVENTORY"})
    table(out / "checkpoint_inventory.csv", checkpoint_rows)
    profile_name = "tested_detector_attributes.csv"
    checked_copy(COMPLIANCE / profile_name, out / "measured_attributes.csv")
    measure = rows(out / "measured_attributes.csv")
    required = {"mean_pipeline_ms", "peak_allocated_bytes", "peak_reserved_bytes",
                "checkpoint_parameters", "GFLOPs_2xMAC", "scope", "clock", "hardware"}
    require(required.issubset(measure[0]), "Missing measured attribute definitions")
    checked_copy(COMPLIANCE / "leaf_checkpoint_coverage.csv", out / "static_coverage_rules.csv")
    stored_static = load(AUDIT / "static_gap.json")
    source(AUDIT / "static_gap.json")
    policies = stored_static["definitions"]
    population = list(model.combinations())
    counts, sensitivity = [], []
    for name, value in policies.items():
        covered = [x for x in population if set(x).issubset(value["covered_leaves"])]
        require(len(covered) == value["covered"], f"Static policy {name} differs from recorded counts")
        counts.append({"policy": name, "covered": len(covered), "gap": len(population)-len(covered),
                       "population": len(population), "scope": "BASELINE_TRAINING_STATIC_PROXY",
                       "covered_leaves": "|".join(value["covered_leaves"])})
        sensitivity.extend({"policy": name, "feature": leaf,
                            "configs_with_feature": sum(leaf in x for x in population),
                            "covered_with_feature": sum(leaf in x for x in covered)}
                           for leaf in sorted(model.ALL_LEAVES))
    require({x["policy"]: x["covered"] for x in counts} ==
            {"exact_identity": 1, "conservative": 3, "supervisor_proxy": 18}, "Unexpected static definitions")
    table(out / "static_gap_counts.csv", counts)
    table(out / "coverage_sensitivity.csv", sensitivity)
    pin = COMPLIANCE / "sources/ultralytics_8.4.33_augment.py"
    history = load(COMPLIANCE / "historical_training_source.json")
    source(pin, history["source"]["sha256"])
    source(COMPLIANCE / "historical_training_source.json")
    historic_runtime = stored_static["augmentation_source"]
    runtime = Path(historic_runtime["path"])
    if runtime.is_file():
        source(runtime)
    environment = metadata.distribution("ultralytics")
    current = Path(environment.locate_file("ultralytics/data/augment.py"))
    source(current)
    recorded = stored_static["training_args"]
    source(BASE / "training/yolov8s/args.yaml")
    current_text = current.read_text(encoding="utf-8")
    transforms = [
        ("HSV gain", "APPROXIMATE", f"hsv_v={recorded['hsv_v']}; brightness multiplier without offset"),
        ("Additive exposure", "NOT_COVERED", "RandomHSV does not implement the stated affine bias"),
        ("Blur/MedianBlur", "UNKNOWN", "Default p=.01; runtime activation was not recorded; kernel mismatch"),
        ("CLAHE", "UNKNOWN", "Default p=.01; exists in source but historical activation not established"),
        ("JPEG", "NOT_COVERED", "Default ImageCompression p=0"),
        ("Gaussian/shot noise", "NOT_COVERED", "No matching default transport"),
        ("Directional motion blur", "NOT_COVERED", "Generic blur is a different transform"),
        ("Haze", "NOT_COVERED", "No matching default constant-depth haze transform"),
        ("Scale/Mosaic", "PARTIAL", f"scale={recorded['scale']}; mosaic={recorded['mosaic']}; training geometry changes"),
        ("Shear/Perspective", "NOT_COVERED", f"shear={recorded['shear']}; perspective={recorded['perspective']}; no row warp"),
    ]
    table(out / "augmentation_source_audit.csv", [
        {"operation": name, "classification": state, "training_evidence": detail,
         "plan_source_version": "8.4.33", "plan_source_sha256": digest(pin),
         "historical_runtime_version": "8.4.61", "historical_runtime_source_sha256": historic_runtime["sha256"],
         "installed_version": environment.version, "installed_source_sha256": digest(current),
         "installed_hsv_multiplier_present": "lut_val = np.clip(x * (r[2] + 1)" in current_text,
         "historical_albumentations_activation": "UNKNOWN"} for name, state, detail in transforms])
    constraints = FMD.read_text(encoding="utf-8").split("constraints\n", 1)[1].strip().splitlines()
    table(out / "constraint_evidence.csv", [
        {"constraint": x.strip(), "status": "PRESERVED_IN_MODEL",
         "scope": "Protocol exclusion; no verification of API impossibility or combined runtime"
                 if "SAHI" in x else "Logical constraint, not a capability certification"} for x in constraints])
    nine = next((x for x in rows(COMPLIANCE / "static_coverage_policies.csv") if x["covered"] == "9"), None)
    source(COMPLIANCE / "static_coverage_policies.csv")
    save(out / "coverage_definition_review.json", {"historical_9_policy": nine,
        "status": "DEFINITION_FOUND" if nine else "UNRESOLVED_IN_BOUNDED_RULE_FILES",
        "note": "9 is not interchangeable with the three verified baseline policies."})
    return {"status": "PARTIAL", "logical_status": "TECHNICAL_PASS",
            "FlamaPy": result, "independent_count": independent, "counts": counts,
            "measured_rows": len(measure), "candidates": len(checkpoint_rows),
            "checkpoint_sha_matches": sum(r["weight_identity"] == "VERIFIED" for r in checkpoint_rows),
            "static_policies_verified": True, "historical_nine_policy": nine}


def design():
    manifest = load(AUDIT / "selection_manifest.json")
    reference = load(ADMISSION / "summary.json")["source_hashes"]["selection_manifest.json"]
    source(ADMISSION / "summary.json")
    source(AUDIT / "selection_manifest.json", reference)
    source(P_IMPL, manifest["FM_P_sha256"])
    source(D_IMPL, manifest["FM_D_sha256"])
    return manifest


def week4(out, regenerated=None):
    model = get_fmp()
    prior = module("uav_sampling", EXPS / "cv261123/sampling.py")
    manifest = design() if regenerated is None else load(regenerated)
    manifest_path = AUDIT / "selection_manifest.json" if regenerated is None else regenerated
    population = prior.enumerate_configs(model)
    safe = [x for x in population if "RollingShutter" not in x]
    p, d, pairs = {}, {}, []
    for spec in manifest["pairs"]:
        leaves = tuple(spec["degradation"]["leaves"])
        detector = spec["detector"]
        pid, did = spec["degradation"]["degradation_id"], detector["detector_id"]
        did_expected = "d_" + sha256(json.dumps(sorted(detector["leaves"]), separators=(",", ":")).encode()).hexdigest()[:16]
        require(pid == model.configuration_id(leaves) and did == did_expected
                and spec["pair_id"] == f"{pid}__{did}", "Frozen configuration identity mismatch")
        require(pid not in p or p[pid] == leaves, "Same P has different features")
        require(did not in d or d[did] == detector, "Same D has different attributes")
        p[pid], d[did] = leaves, detector
        pairs.append({"pair_id": spec["pair_id"], "p_id": pid, "d_id": did,
                      "checkpoint_sha256": detector["checkpoint_sha256"], "status": "PLANNED_NOT_MEASURED"})
    require((len(p), len(d), len(pairs), len({x["pair_id"] for x in pairs})) == (100, 5, 500, 500),
            "Wrong frozen design cardinality")
    require(list(p.values()) == random.Random(manifest["seed"]).sample(safe, 100),
            "The frozen sample does not reproduce from its seed and enumeration order")
    if regenerated is None:
        for detector in d.values():
            source(MODELS / "yolov8s_cft640/best.pt", detector["checkpoint_sha256"])
    table(out / "first500_pairs_PLAN_ONLY.csv", pairs)
    table(out / "p_configurations.csv", [{"p_id": key, "selected_leaves": "|".join(value)} for key, value in p.items()])
    table(out / "d_configurations.csv", [{**x, "leaves": "|".join(x["leaves"])} for x in d.values()])
    pairwise = [tuple(x["selected_leaves"]) for x in manifest["pairwise"]["cohort"]]
    uniform_full = [tuple(x["selected_leaves"]) for x in manifest["full_space_uniform_candidates"]]
    all_leaves = sorted(model.ALL_LEAVES)
    _, full_mask = prior.compute_binary_pair_masks(population, all_leaves)
    _, safe_mask = prior.compute_binary_pair_masks(safe, all_leaves)
    cooccurrences = sum(1 << (4*i+3) for i in range(len(list(combinations(all_leaves, 2)))))
    coverage = []
    for name, sample in (("FIRST500_UNIFORM_SAFE", list(p.values())),
                         ("FULL_SPACE_PAIRWISE_PROPOSAL", pairwise), ("FULL_SPACE_UNIFORM_PROPOSAL", uniform_full)):
        require(len(set(sample)) == len(sample) and all(model.valid_leaves(x) for x in sample), "Invalid sample")
        _, found = prior.compute_binary_pair_masks(sample, all_leaves)
        denominator = safe_mask if name == "FIRST500_UNIFORM_SAFE" else full_mask
        available_leaves = set().union(*(set(x) for x in (safe if name == "FIRST500_UNIFORM_SAFE" else population)))
        coverage.append({"cohort": name, "sample_n": len(sample), "leaves": len(set().union(*map(set, sample))),
                         "leaf_denominator": len(available_leaves), "binary_states": found.bit_count(),
                         "binary_denominator": denominator.bit_count(), "full_space_binary_denominator": full_mask.bit_count(),
                         "cooccurrences": (found & cooccurrences).bit_count(),
                         "cooccurrence_denominator": (denominator & cooccurrences).bit_count(),
                         "rolling_configurations": sum("RollingShutter" in x for x in sample),
                         "scope": "LOGICAL_COVERAGE_ONLY"})
    table(out / "sampling_coverage.csv", coverage)
    save(out / "research_sampling_design.json", {"v1_seed": manifest["seed"], "render_seed": manifest["render_seed"],
        "cohorts": coverage, "first500_source_SHA256": digest(manifest_path),
        "frozen_protocol_source": source(EXPS / "cv261207/evaluation.py"),
        "safe_population": len(safe), "pairwise_target": "60-100 P", "uniform_target": "about 100 P",
        "checkpoint_candidate_target": "8-12 candidate checkpoints; infer settings form separate D configurations",
        "full_plan_approx": {"P": 200, "D": 20, "pairs": 4000},
        "active_sampling": "DEFERRED_WEEK7_PLUS", "actual_measured_formal": 0})
    return {"status": "TECHNICAL_PASS", "scope": "FROZEN_V1_ONLY", "p": len(p), "d": len(d),
            "pairs": len(pairs), "coverage": coverage, "seed": manifest["seed"],
            "safe_population": len(safe), "detectors": [{"resolution": x["resolution"],
            "precision": x["precision"]} for x in d.values()]}

def week5(out):
    manifest = design()
    prior = load(ADMISSION / "admission.json")
    source(ADMISSION / "admission.json", load(ADMISSION / "summary.json")["source_hashes"]["admission.json"])
    gate = load(AUDIT / "dataset_gate.json")
    source(AUDIT / "dataset_gate.json")
    archive_dir = ROOT / "datasets/VisDrone"
    for record in gate["archives"].values():
        source(archive_dir / record["name"], record["sha256"])
    current_test_zip = archive_dir / gate["archives"]["test-dev"]["name"]
    member_path = AUDIT / "test_dev_members.jsonl"
    source(member_path, gate["test_dev_member_manifest_sha256"])
    expected = [json.loads(line) for line in member_path.read_text(encoding="utf-8").splitlines()]
    archive_module = module("uav_original_zip", EXPS / "cv261207/archives.py")
    test = archive_module.OriginalZIP(current_test_zip, 1610)
    counts = {"image_records": 0, "ignore_boxes": 0, "other_boxes": 0, "native10_boxes_before_filter": 0}
    try:
        for index, original in enumerate(expected):
            _, gt, observed = test.record(index)
            require(all(observed[k] == original[k] for k in observed) and gt["stem"] == original["image_stem"],
                    "test-dev parsed image or label SHA differs")
            counts["image_records"] += 1
            counts["ignore_boxes"] += len(gt["ignore_regions"])
            counts["other_boxes"] += len(gt["other_regions"])
            counts["native10_boxes_before_filter"] += len(gt["objects"])
    finally:
        test.close()
    require(counts["image_records"] == 1610, "Incomplete test-dev membership")
    test_ids = {x["image_stem"] for x in expected}
    test_sha = {x["image_SHA256"] for x in expected}
    overlap = {}
    for split in ("train", "val"):
        with zipfile.ZipFile(archive_dir / gate["archives"][split]["name"]) as archive:
            images = zip_members(archive, "images", (".jpg", ".jpeg", ".png"))
            match = sum(sha256(archive.read(path)).hexdigest() in test_sha for path in images.values())
        overlap[split] = {"images": len(images), "matching_ids": len(test_ids.intersection(images)),
                          "matching_image_sha256": match}
    require(overlap == prior["archive_overlap"], "Split identity differs from prior evidence")
    vm11 = ROOT / "datasets/research_archive/variability/experiments/vm11_holdout_confirmation/runs/vm11_r002"
    vm43 = RUNS / "vm43_external_pilot/vm43_external_pilot_execution_20261116T222929_46c7e0ba"
    records = [vm11 / "results/dataset_gate.json", vm11 / "results/history_search.json",
               vm11 / "results/checkpoint_exposure_audit.csv", vm43 / "scientific_acceptance.json",
               VM41 / "Scene_proxy_overlap.json", VM41 / "Historical_testdev_records.json"]
    for path in records:
        source(path)
    vm43_result = load(vm43 / "scientific_acceptance.json")
    require(vm43_result["verified_cells"] == 4 and vm43_result["formal_measured_images"] == 6440,
            "VM43 external exploratory evidence differs")
    dataset = {"status": "DATASET_BLOCKED", "archive_status": "TECHNICAL_PASS",
        "source_archive_hashes": {k: v["sha256"] for k, v in gate["archives"].items()},
        "counts": counts, "overlap": overlap, "candidates": [
            {k: item.get(k) for k in ("id", "model", "checkpoint_sha256", "training_membership",
             "checkpoint_selection", "historical_testdev_evaluation_or_tuning", "formal_admission")}
            for item in prior["candidates"]],
        "testdev_prior_use": {"VM11": "BLOCKED_INDEPENDENT_DATA",
            "VM43": {"scope": vm43_result["scope"], "verified_cells": 4, "image_evaluations": 6440,
                     "exact_checkpoint_identity_scope": "YOLO26s 960/10ep EMA; not the frozen v1 YOLOv8s"}},
        "scene_and_flight_independence": "NOT_ESTABLISHED",
        "historical_selection_and_tuning": "NOT_FULLY_ESTABLISHED",
        "admission_reason": "Complete separation of split IDs and image bytes does not establish study-level independence.",
        "current_special_gpu_authorization": "NOT_PROVIDED",
        "formal_measured_pairs": 0, "formal_measured_image_evaluations": 0}
    save(out / "test_dev_admission.json", dataset)
    historical = [
        ("CV261207 v1", ADMISSION, "admission.json", "0 formal", "Dataset gate blocked; v1 has no formal pair"),
        ("CV261207 development", RUNS / "cv261207/cv261207_development_analysis_20261108T194456_6e430eb3",
         "configuration_metrics.csv", "248 development images per new pair", "Wrong split and selected candidate protocol"),
        ("VM36", RUNS / "vm36_supervisor_reproduction/vm36_supervisor_reproduction_full20_results_20261110T161658_46e4aed4",
         "summary.json", "development", "Wrong image membership and study design"),
        ("VM43", vm43, "scientific_acceptance.json", "4 x 1610 test-dev", "Different D/checkpoints and exploratory admission"),
        ("VM45", RUNS / "vm45_motion_haze_gap_learning/vm45_motion_haze_gap_learning_execution_20261117T172830_0d9cb1ee",
         "final_acceptance.json", "development", "Wrong split and checkpoint/condition cohort"),
        ("VM47", RUNS / "vm47_gaussian_jpeg_factorial/vm47_gaussian_jpeg_factorial_execution_20261117T190442_7166b669",
         "final_acceptance.json", "development", "Wrong split and checkpoint/condition cohort"),
        ("VM48", RUNS / "vm48_formal_rq4/motion_targeted_seed45_execution_20261118T210143_5ada2476",
         "final_acceptance.json", "development", "Wrong split, trained arms and checkpoint identities"),
    ]
    evidence = []
    for name, run, filename, split, reason in historical:
        audit = source(run / filename)
        evidence.append({"lineage": name, "RunID": run.name, "record": filename,
                         "record_sha256": audit["sha256"], "scope": split, "eligible_pairs": 0,
                         "first_failed_condition": reason, "other_conditions": "Not pooled after exclusion"})
    table(out / "historical_pair_eligibility.csv", evidence)
    profile = rows(COMPLIANCE / "tested_detector_attributes.csv")
    source(COMPLIANCE / "tested_detector_attributes.csv")
    refs = [float(x["pair_wall_seconds"]) for x in profile if x["pair_wall_seconds"]]
    resource = {"local_volume_free_bytes": shutil.disk_usage(ROOT).free,
        "status": "NOT_MEASURED_FORMAL_COST", "GPU_active_seconds_this_run": 0,
        "hardware_from_prior_measurement": sorted({x["hardware"] for x in profile}),
        "historical_reference": {"measured_images_per_pair": 248, "rows": len(refs),
            "pair_wall_seconds_min": min(refs), "pair_wall_seconds_max": max(refs),
            "scope": "Distinct development cohort and checkpoint identities; not a formal budget estimate"},
        "missing_benchmarks": ["Matched 1610-image pair", "Repeat pair", "Full P group of five D settings",
            "GPU authorization", "Cache/prediction storage and actual bootstrap scaling"],
        "first_benchmark_sequence": ["Admit source and GPU", "Run one frozen 1610-image pair", "CPU replay and repeat",
            "Measure generation, preprocessing, GPU, postprocess, serialization, COCO, bootstrap and disk",
            "Gate complete five-D groups independently of observed AP"],
        "thermal_limits": {"hard_C": 90, "sustained_C": 87, "sustained_seconds": 30, "poll_max_seconds": 2}}
    save(out / "resource_plan.json", resource)
    protocol_sources = {
        str(path.relative_to(PUBLIC)): source(path)["sha256"] for path in
        (EXPS / "cv261207/evaluation.py", EXPS / "cv261109/generator.py",
         PUBLIC / "code/main/predictor.py", PUBLIC / "code/main/labels.py", PUBLIC / "code/main/statistics.py")}
    definition = {"status": "CPU_SPECIFICATION_FROZEN_FORMAL_RUNNER_ADMISSION_PENDING",
        "split": "VisDrone2019-DET-test-dev", "P": 100, "D": 5, "pairs": 500, "images_per_pair": 1610,
        "planned_image_evaluations": 805000, "render_seed": manifest["render_seed"],
        "same_P_rule": "Identical PNG pixels and original-coordinate GT for all five D settings",
        "detector": "YOLOv8s v1 settings; exact IDs and per-setting resolution/precision in original manifest",
        "predictor": {"batch": 1, "rect": False, "conf": .001, "nms_iou": .7, "max_det": 3000,
            "max_nms": 30000, "head": "dense", "class_aware": True, "multi_label": True,
            "TF32": False, "TTA": False, "SAHI": False, "warmups": 3},
        "evaluation": {"labels": "VisDrone native10", "coordinates": "original image pixels",
            "ignore_union_fraction_excluded": .5, "size_ranges": ["[0,256)", "[256,1024)", "[1024,9216)", "[9216,inf)"],
            "iou": [.50, .55, .60, .65, .70, .75, .80, .85, .90, .95],
            "recall_grid_points": 101, "AP_and_AR": "all, per class and per size"},
        "latency": {"summary": "mean per-image cached pipeline duration; image-bootstrap percentile interval",
            "inside": ["PNG read and hash", "decode and decoded pixel verification", "synchronized preprocess/forward/NMS",
                       "extraction and compressed record serialization/write/flush"],
            "separate_clocks": ["model load", "each cell warmup", "PNG generation", "COCO evaluation", "bootstrap",
                                "end-to-end wall clock"],
            "must_instrument_separately": True},
        "bootstrap": {"unit": "paired image index", "interval": "95% empirical percentiles",
            "initial_draws": 200, "seed": 20261207, "replicate_check": "Compare prefixes 100/200 with 1000-draw reference on admitted pair; report endpoint drift and freeze final count before main wave",
            "AP_method": "reaccumulate COCO matches within each image-index resample"},
        "source_hashes": protocol_sources, "legacy_plan_sha256": digest(AUDIT / "selection_manifest.json")}
    save(out / "formal_evaluation_protocol.json", definition)
    text_file(out / "author_decision.md", """# Scientific decision for author and supervisor

1. Confirm whether the existing VisDrone test-dev split can be used for a
   prespecified **limited benchmark** despite known VM43 use and incomplete
   experiment/scene history, or require a new native10 cohort with recorded
   acquisition-group isolation and permissions. The independent gate stays blocked
   under the current evidence. A limited benchmark does not certify independence.
2. Agree on a controlled, hash-addressed raw-prediction archive and access policy
   if repository publication is unavailable. Summary CSVs do not replace raw replay.
3. Once a scientific route is agreed, authorize a separately bounded full-size
   benchmark and repeat, before budgeting the full 500-pair wave.
""")
    return {"status": "BLOCKED", "scientific_admission": dataset["status"],
            "archive_counts": {"train": overlap["train"]["images"], "val": overlap["val"]["images"],
                               "test_dev": counts["image_records"]},
            "planned_pairs": 500, "measured_pairs": 0, "measured_images": 0, "GPU_seconds": 0,
            "disk_free_bytes": resource["local_volume_free_bytes"], "historical_exploratory_test_dev_cells": 4}


def pair_status(ap_ci, latency_ci, tau, limit):
    for interval, minimum, maximum in ((ap_ci, 0, 1), (latency_ci, 0, math.inf)):
        if interval is not None:
            low, high = interval
            require(math.isfinite(low) and math.isfinite(high) and minimum <= low <= high <= maximum,
                    "Invalid confidence interval")
    if (ap_ci is not None and ap_ci[1] < tau) or (latency_ci is not None and latency_ci[0] > limit):
        return "FAIL"
    if ap_ci is not None and latency_ci is not None and ap_ci[0] >= tau and latency_ci[1] <= limit:
        return "PASS"
    return "UNCERTAIN"


def gap_status(states, expected=5):
    require(len(states) == expected, "Expected one status per planned detector")
    if "PASS" in states:
        return "COVERED"
    if all(state == "FAIL" for state in states):
        return "TESTED_CANDIDATE_GAP"
    return "UNKNOWN"


def week6(out):
    model = get_fmp()
    manifest = design()
    policies_path = AUDIT / "static_gap.json"
    source(policies_path)
    policies = load(policies_path)["definitions"]
    trials = [
        ((.10, .15), (70, 100), "PASS"),
        ((.02, .099), (50, 90), "FAIL"),
        ((.11, .2), (100.1, 110), "FAIL"),
        ((.11, .2), (90, 110), "UNCERTAIN"),
        ((.05, .10), (60, 90), "UNCERTAIN"),
        ((.11, .2), None, "UNCERTAIN"),
        (None, None, "UNCERTAIN"),
    ]
    for ap, ms, expected in trials:
        require(pair_status(ap, ms, .10, 100) == expected, "Interval decision test failed")
    for sequence, expected in ((["FAIL"]*5, "TESTED_CANDIDATE_GAP"),
                               (["FAIL"]*4+["NOT_MEASURED"], "UNKNOWN"),
                               (["UNCERTAIN"]*4+["PASS"], "COVERED")):
        require(gap_status(sequence) == expected, "Gap aggregation test failed")
    measured_tables = {
        "pair_metrics.csv": "pair_id,source_RunID,images,AP50,AP50_95,AR50_95,GT_SHA256,prediction_SHA256",
        "per_class_metrics.csv": "pair_id,class_id,class_name,AP50,AP50_95,AR50_95,valid_GT",
        "size_metrics.csv": "pair_id,area_bin,AP50,AP50_95,AR50_95,valid_GT",
        "timing_metrics.csv": "pair_id,clock,component,seconds,image_count,source_SHA256",
        "pair_bootstrap.csv": "pair_id,metric,lower_95,upper_95,draws,seed,image_index_SHA256",
    }
    for filename, fields in measured_tables.items():
        table(out / filename, [], fields.split(","))
    pairs, p = [], {}
    for spec in manifest["pairs"]:
        selection = spec["degradation"]
        pid = selection["degradation_id"]
        require(pid == model.configuration_id(selection["leaves"]), "Unexpected P identity")
        p.setdefault(pid, []).append(spec)
        pairs.append({"pair_id": spec["pair_id"], "p_id": pid, "d_id": spec["detector"]["detector_id"],
                      "measured_images": 0, "status": "NOT_MEASURED"})
    require(len({x["pair_id"] for x in pairs}) == 500 and len(p) == 100, "Invalid Gap denominator")
    table(out / "pair_status.csv", pairs)
    gap = [{"p_id": pid, "planned_detectors": len(design_rows),
            "measured_detectors": 0, "gap_status": gap_status(["NOT_MEASURED"]*len(design_rows))}
           for pid, design_rows in p.items()]
    table(out / "degradation_gap_status.csv", gap)
    table(out / "static_empirical_comparison.csv", [
        {"p_id": x["p_id"], "static_policy": key,
         "static_covered": set(p[x["p_id"]][0]["degradation"]["leaves"]).issubset(policy["covered_leaves"]),
         "empirical_status": x["gap_status"], "measured_D": 0}
        for x in gap for key, policy in policies.items()])
    table(out / "threshold_sensitivity.csv", [
        {"tiny_ap_tau": tau, "latency_limit_ms": limit, "formal_pairs": 0,
         "planned_pairs": 500, "confirmed_P_covered": 0, "confirmed_P_gap": 0,
         "unknown_P": 100, "measured_gap_prevalence": "NA"}
        for tau in (.05, .10, .15) for limit in (50, 100)])
    save(out / "decision_tests.json", {"interval_tests_passed": len(trials), "aggregation_tests_passed": 3,
        "note": "Synthetic test inputs only; no measured AP or latency."})
    return {"status": "BLOCKED", "decision_implementation": "TECHNICAL_PASS", "formal_pairs": 0,
            "planned_pairs": 500, "measured_images": 0, "unknown_P": 100,
            "primary": {"tiny_AP50_95": .10, "cached_latency_ms": 100}, "decision_tests": len(trials)+3}

def requirements_report(out, bundle):
    w = {x["week"]: x for x in bundle}
    data = [
        ("RQ1-P", "4-6", "Build a degradation feature model and test each generator binding",
         "25 leaves and 5 draft constraints", "Frozen UVL plus separate implementation bindings", 2,
         "flamapy_verification.json; feature_parameters.csv; generator_tests.json",
         "Physical justification; rolling annotations", "PARTIAL", "Review author bindings and keep rolling excluded"),
        ("RQ1-P-GEOMETRY", "5-6", "Preserve image/annotation alignment for each distortion",
         "No numeric rolling-shutter strength in the PDF", "0.008×width, 2 waves; excluded from formal v1", 2,
         "geometry_safety.csv", "Validated GT and ignore transform with clipping and area contract", "BLOCKED",
         "Agree on geometry-aware annotation policy before changing the safe population"),
        ("RQ1-D", "4,7-9", "Build detector feature model with measured attributes",
         "34 leaves, 4 draft constraints; nano/small parameter numbers are examples", "Preserved logic; measured attributes keyed by exact checkpoint", 3,
         "FM_D.uvl; measured_attributes.csv; checkpoint_inventory.csv", "Full deployment evidence across configurations", "PARTIAL",
         "Measure uncovered settings only after resource and protocol admission"),
        ("RQ1-D-SAHI", "9", "Validate or revise the SAHI/TTA rule against the libraries",
         "SAHI_Slicing implies not TTA", "Preserved study exclusion", 3, "constraint_evidence.csv",
         "No same-backend proof of incompatibility", "AUTHOR_REVIEW_PENDING", "Keep exclusion pending source/API review"),
        ("RQ2", "4,11,16", "Oracle-labelled data and held-out rule validation",
         "Later learned rules and boundary searches", "No formal learned rule in weeks 1-6", 6,
         "decision_tests.json", "Admitted response matrix", "BLOCKED", "Defer rule fitting to weeks 7-8"),
        ("RQ3-STATIC", "9-10,16", "Check default training augmentation and argue per-leaf coverage",
         "18/46368 provisional example for baseline", "Strict 1; conservative proxy 3; plan proxy 18", 3,
         "static_gap_counts.csv; augmentation_source_audit.csv",
         "Historical Albumentations activation; author justification", "PARTIAL", "Review each proxy independently"),
        ("RQ3-EMPIRICAL", "9-11", "Measure empirical gap on evaluated detector candidates",
         "10% Tiny AP and 50 ms latency are example settings", "10% Tiny, 100ms cached; CI-based status", 6,
         "degradation_gap_status.csv; threshold_sensitivity.csv", "Formal metrics and uncertainty", "BLOCKED",
         "Use only admitted pairs and report incomplete detector panels as UNKNOWN"),
        ("RQ4", "4,12,16", "Targeted augmentation and external validation",
         "Weeks 9-11", "Outside the current six-week workflow", 6, "historical references from Week 5",
         "Formal future work", "AUTHOR_REVIEW_PENDING", "Revisit after formal gap evidence"),
        ("BASELINE", "11,15-16", "Live evaluation on VisDrone val with a reproducible score and epoch timing",
         "YOLOv8s fine-tuned; held meeting", "640 training, 1280 inference, 100 epochs, historical source", 1,
         "baseline_evidence.json; runtime_source_recovery.csv; evaluation_timing.csv",
         "Meeting demonstration and approval", "SUPERVISOR_ACCEPTANCE_PENDING", "Demonstrate replay and confirm meeting scope"),
        ("LIVE-RENDER", "15-16", "Render a randomly selected legal configuration at the meeting",
         "Live demonstration", "Fixed-seed uniform draw from annotation-safe scope", 2,
         "random_render_manifest.json; random_render.png", "Meeting demonstration", "SUPERVISOR_ACCEPTANCE_PENDING",
         "Run the CPU render, demonstrate seed and annotate limits"),
        ("SAMPLE-PAIRWISE", "11", "Sample feasible degradation-feature combinations",
         "About 60-100 pairwise P", "Historical 80 logical P, includes rolling candidates", 4,
         "sampling_coverage.csv", "Admitted measurements for the broader cohort", "PARTIAL",
         "Keep pairwise proposal separate from 500-pair plan"),
        ("SAMPLE-UNIFORM", "11,16", "Sample uniformly from the legal configuration space",
         "About 100 uniform P", "Frozen v1 100 P among 26,496 geometry-safe choices", 4,
         "p_configurations.csv; research_sampling_design.json", "Excluded rolling branch", "TECHNICAL_PASS",
         "Retain the v1 cohort and freeze any future change separately"),
        ("SAMPLE-BOUNDARY", "11,16", "Sample near a learned decision boundary",
         "Active cohort in later research", "Deferred", 4, "research_sampling_design.json",
         "Oracle and learned boundary", "BLOCKED", "Do not perform active sampling in this stage"),
        ("SAMPLE-DETECTOR", "11", "Sample trained checkpoints and inference variants",
         "8-12 checkpoint candidates; around 20 D settings", "Five inference settings from one checkpoint", 4,
         "d_configurations.csv", "Broader independently eligible checkpoint set", "PARTIAL",
         "Preserve frozen v1, separately admit the broader cohort"),
        ("SPLIT", "11", "Use val for selection and test-dev for reported detection evaluation",
         "Train / val / test-dev", "6471 / 548 / 1610; test-dev prior exposure audit", 5,
         "test_dev_admission.json", "Historical tuning and scene isolation", "BLOCKED",
         "Author and supervisor decide permitted use without claiming independence"),
        ("FORMAL-500", "16", "Measure first 500 P-D pairs and reproduce one at the weekly demonstration",
         "500 first pairs; full study approximately 4000", "100×5, 1610 images per cell", 5,
         "formal_evaluation_protocol.json; historical_pair_eligibility.csv",
         "0 formal pairs, no applicable GPU authorization", "BLOCKED",
         "Admit one complete full-size pair and repeat it before estimating batch cost"),
        ("METRICS", "10-11", "Evaluate detection and robustness at matched settings",
         "Native task metrics, per category and Tiny; confidence intervals", "Native10; AP50/AP50:95/AR; original pixel areas", 5,
         "formal_evaluation_protocol.json", "Formal metric/size/CI records", "BLOCKED",
         "Verify 200-draw stability on a real admitted pair"),
        ("ATTRIBUTES", "9,11", "Measure deployment latency and memory on fixed hardware",
         "Fixed device; reference time range from full study plan", "Fixed component clocks and matched cached pipeline", 5,
         "resource_plan.json", "Real 1610-image runtime and storage reference", "BLOCKED",
         "No conversion between historical latency protocols"),
        ("RAW", "15-16", "Trace numbers to raw predictions and committed reproduction code",
         "Raw predictions in research repository", "Raw historical predictions retained locally with SHA", 5,
         "author_decision.md; source_manifest.json", "Publication authorization and raw formal pairs", "AUTHOR_REVIEW_PENDING",
         "Agree on verification access and a hash-addressed archive"),
        ("AUTHORSHIP", "13-17", "Authors review papers and write the scientific manuscript",
         "AI limited to assistance, verification through meetings", "Technical records and private operating guides", 1,
         "reproduction_command.md", "Human readings, interpretation and supervisor check-ins", "AUTHOR_REVIEW_PENDING",
         "Author reviews the evidence and performs live demonstrations"),
    ]
    fields = ("requirement_id", "source_page", "original_requirement", "original_parameter", "implementation_parameter",
              "existing_source_run", "existing_evidence", "missing_evidence", "acceptance_status", "next_action")
    table(out / "requirements_traceability.csv", [dict(zip(fields, (
        ident, page, req, orig, impl, w[week]["RunID"], evidence, missing, status, next_action), strict=True))
        for ident, page, req, orig, impl, week, evidence, missing, status, next_action in data])


def finish(out):
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt")
    text_file(out / "SHA256SUMS.txt", "\n".join(f"{digest(p)}  {p.relative_to(out).as_posix()}" for p in files))


def index(out, record):
    from html import escape
    paths = sorted(p.name for p in out.iterdir() if p.is_file())
    links = "".join(f'<li><a href="{escape(name)}">{escape(name)}</a></li>' for name in paths)
    render = '<figure><img style="max-width:100%" src="random_render.png"><figcaption>Source and rendered image</figcaption></figure>' \
        if (out / "random_render.png").is_file() else ""
    text_file(out / "index.html", f"""<!doctype html><meta charset="utf-8">
<title>{record['ExperimentID']}</title>
<main style="font:17px system-ui;max-width:64rem;margin:2rem auto">
<h1>{record['ExperimentID']}</h1>
<p>{out.name} · {record['status']} · CPU {record['CPU_wall_seconds']:.2f} s</p>
{render}<ul>{links}</ul></main>""")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week", type=int, choices=range(1, 7))
    parser.add_argument("--plan", type=Path, required=True, help="Local 17-page research plan PDF")
    parser.add_argument("--previous-executor", type=Path, help="Retain the exact source used by prior completed weeks")
    parser.add_argument("--baseline-source-zip", type=Path, help="Historical evaluator source archive")
    parser.add_argument("--traceability-only", nargs=6, metavar="RUN_ID", help="Write the requirements table in a new Week 6 run")
    args = parser.parse_args()
    global PLAN, BASELINE_SOURCE_ZIP
    PLAN = args.plan.resolve(strict=True)
    BASELINE_SOURCE_ZIP = args.baseline_source_zip.resolve(strict=True) if args.baseline_source_zip else None
    source(PLAN, PLAN_SHA)
    jobs = (baseline, week2, week3, week4, week5, week6)
    weeks = [args.week] if args.week else list(range(1, len(jobs)+1))
    previous = []
    if args.traceability_only:
        require(args.week is None, "Use only one execution mode")
        weeks = [6]
        for experiment, run_id in zip(EXP_IDS, args.traceability_only, strict=True):
            require(run_id.startswith(experiment + "_") and Path(run_id).name == run_id, "Invalid RunID")
            path = RUNS / experiment / run_id / "run.json"
            saved = load(path)
            require(saved["RunID"] == run_id and saved["status"] != "FAIL", "Invalid prerequisite run")
            previous.append(saved)
    bundle = []
    for week in weeks:
        name = EXP_IDS[week-1]
        now = datetime.now(timezone.utc)
        out = RUNS / name / f"{name}_{now:%Y%m%dT%H%M%S}_{uuid4().hex[:8]}"
        out.mkdir(parents=True)
        _USED.clear()
        source(PLAN, PLAN_SHA)
        source(Path(__file__))
        started = time.perf_counter()
        try:
            details = jobs[week-1](out)
        except Exception:
            trace = traceback.format_exc()
            text_file(out / "error.txt", trace)
            details = {"status": "FAIL", "error": trace.splitlines()[-1]}
        record = {"ExperimentID": name, "RunID": out.name, "week": week,
                  "started_utc": now.isoformat(), "finished_utc": datetime.now(timezone.utc).isoformat(),
                  "CPU_wall_seconds": time.perf_counter()-started, "GPU_executions": 0,
                  "author_status": "AUTHOR_REVIEW_PENDING",
                  "supervisor_status": "SUPERVISOR_ACCEPTANCE_PENDING", **details}
        bundle.append(record)
        if week == 6 and (previous or (len(bundle) == 6 and all(x["status"] != "FAIL" for x in bundle))):
            if args.previous_executor:
                checked_copy(args.previous_executor.resolve(strict=True), out / "prior_executor.py")
            accepted = previous or bundle
            requirements_report(out, accepted)
            save(out / "week01_06_index.json", accepted)
        save(out / "run.json", record)
        source_list = list(_USED.values())
        save(out / "source_manifest.json", source_list)
        text_file(out / "command.txt", subprocess.list2cmdline([sys.executable, "-B", *sys.argv]))
        index(out, record)
        finish(out)
        print(json.dumps({"week": week, "status": record["status"], "run": str(out),
                          "seconds": round(record["CPU_wall_seconds"], 3),
                          "error": record.get("error")}, ensure_ascii=True), flush=True)
    return any(x["status"] == "FAIL" for x in bundle)


if __name__ == "__main__":
    raise SystemExit(main())
