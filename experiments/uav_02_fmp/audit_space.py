"""Exercise every legal FM_P configuration on a fixed annotated CPU fixture."""
from copy import deepcopy
import csv
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import cv2
import numpy as np

from validate_geometry import SEED, choose_parameters, generate, model, transform_annotations, write_new
from generator import ORDER, apply_rolling_shutter, effect_seed
from main.labels import ground_truth
from uav_reproduction import check_uvl, flama_python


def fixture():
    height, width = 67, 131
    pixels = np.random.default_rng(42).integers(0, 256, (height, width, 3), dtype=np.uint8)
    record = {"id": 1, "stem": "fmp_space_fixture_v1", "file_name": "fixture.png",
              "width": width, "height": height,
              "objects": [{"class_id": c, "bbox": box} for c, box in
                          [(0, [0, 3, 3, 22]), (3, [121, 10, 10, 37]),
                           (9, [43, 18, 15, 17]), (1, [68, 40, 21, 25]),
                           (2, [21, 9, 5, 5])]],
              "ignore_regions": [[19, 2, 6, 37], [25, 15, 9, 7]],
              "other_regions": [[100, 49, 31, 18]]}
    return pixels, record


def mask_for(shape, boxes):
    mask = np.zeros(shape, dtype=np.float32)
    for x, y, width, height in boxes:
        mask[y:y+height, x:x+width] = 1
    return mask


def expected_rolling(record, source_gt, params):
    shape = record["height"], record["width"]
    seed = effect_seed(SEED, record["stem"], "RollingShutter")

    def warp(boxes):
        pixels, trace = apply_rolling_shutter(mask_for(shape, boxes), **params, rng=np.random.default_rng(seed))
        return pixels > 0, trace["phase_rad"]

    boxes = []
    for ann in source_gt["annotations"]:
        mask, _ = warp([ann["bbox"]])
        y, x = np.nonzero(mask)
        boxes.append([int(x.min()), int(y.min()), int(np.ptp(x))+1, int(np.ptp(y))+1])
    ignored, phase = warp(record["ignore_regions"])
    other, _ = warp(record["other_regions"])
    return boxes, ignored, other, phase


def audit_space(out, python=None):
    started = perf_counter()
    uvl = check_uvl("FM_P")
    script = Path(__file__).with_name("sat_catalog.py")
    sat = json.loads(subprocess.check_output([str(flama_python(python)), "-B", str(script), str(uvl)],
                                             text=True, encoding="utf-8"))
    members = sorted(model.combinations())
    payload = "".join("|".join(row) + "\n" for row in members).encode("ascii")
    ids = [model.configuration_id(row) for row in members]
    if (len(members) != sat["configurations"] or len(set(ids)) != len(members)
            or sha256(payload).hexdigest() != sat["leaf_catalog_sha256"]):
        raise ValueError("Generator and FlamaPy enumerate different FM_P configurations.")

    cv2.setNumThreads(1)
    pixels, record = fixture()
    frozen_record = deepcopy(record)
    shape = pixels.shape[:2]
    original_gt, _ = ground_truth([record], "native10")
    attrs = choose_parameters()
    expected_boxes, expected_ignore, expected_other, phase = expected_rolling(
        record, original_gt, attrs["RollingShutter"])
    render_count = 0
    passed = 0
    failed = 0
    rolling_passed = 0
    fields = ("configuration_id", "selected_leaves", "status", "pixel_sha256", "gt_targets", "error")
    evidence = out / "configuration_validation.csv"
    with evidence.open("x", encoding="utf-8", newline="") as stream:
        table = csv.DictWriter(stream, fields)
        table.writeheader()
        for index, (ident, leaves) in enumerate(zip(ids, members, strict=True), 1):
            row = {"configuration_id": ident, "selected_leaves": "|".join(leaves)}
            try:
                image, effects = generate(pixels, leaves, attrs, record["stem"], SEED)
                repeated, repeated_effects = generate(pixels, leaves, attrs, record["stem"], SEED)
                render_count += 2
                used = [effect["feature"] for effect in effects]
                stages = [ORDER.index(effect["stage"]) for effect in effects]
                if sorted(used) != list(leaves) or stages != sorted(stages):
                    raise ValueError("Selected leaves or operation order differs.")
                if effects != repeated_effects or not np.array_equal(image, repeated):
                    raise ValueError("Pixels or parameters changed on repeat.")
                if image.shape != pixels.shape or image.dtype != np.uint8:
                    raise ValueError("Pixel shape or dtype changed.")
                rolling = next((effect for effect in effects if effect["feature"] == "RollingShutter"), None)
                if rolling:
                    before, gt, derived = transform_annotations(record, rolling)
                    source = gt["annotations"]
                    if (before != original_gt or rolling["phase_rad"] != phase
                            or [ann["bbox"] for ann in source] != expected_boxes
                            or [(ann["source_bbox"], ann["area"], ann["category_id"]) for ann in source]
                                != [(ann["bbox"], ann["area"], ann["category_id"]) for ann in before["annotations"]]
                            or not np.array_equal(mask_for(shape, derived["ignore_regions"]) > 0, expected_ignore)
                            or not np.array_equal(mask_for(shape, derived["other_regions"]) > 0, expected_other)):
                        raise ValueError("Warped boxes, ignore regions or original area differs.")
                else:
                    gt, _ = ground_truth([record], "native10")
                    if gt != original_gt:
                        raise ValueError("Original-canvas operations changed annotation policy.")
                if leaves == ("Normal",) and not np.array_equal(image, pixels):
                    raise ValueError("Normal changed source pixels.")
                if record != frozen_record:
                    raise ValueError("Renderer modified the source annotations.")
                row.update(status="PASS", pixel_sha256=sha256(image.tobytes()).hexdigest(),
                           gt_targets=len(gt["annotations"]))
                passed += 1
                rolling_passed += rolling is not None
            except (ValueError, TypeError, IndexError, KeyError, cv2.error) as exc:
                row.update(status="FAIL", error=str(exc))
                failed += 1
            table.writerow(row)
            if index % 5000 == 0 or index == len(members):
                stream.flush()
                print(f"FM_P {index}/{len(members)}: {passed} passed, {failed} failed", flush=True)

    receipt = {"status": "PASS" if failed == 0 else "FAIL",
               "scope": "EXHAUSTIVE_SINGLE_ANNOTATED_SYNTHETIC_FIXTURE",
               "sat_catalog": sat, "catalog_match": True, "checked": len(members), "passed": passed,
               "failed": failed, "rolling_passed": rolling_passed, "nonrolling_passed": passed-rolling_passed,
               "pixel_render_calls": render_count, "fixture_shape": list(pixels.shape),
               "fixture_seed": 42, "render_seed": SEED, "fixture_sha256": sha256(pixels.tobytes()).hexdigest(),
               "annotation_fixture": frozen_record,
               "configuration_csv_sha256": sha256(evidence.read_bytes()).hexdigest(),
               "source_sha256": {path.name: sha256(path.read_bytes()).hexdigest() for path in
                                 (Path(__file__), script, uvl, uvl.with_name("render_bindings.json"),
                                  uvl.with_name("geometry.py"), Path(generate.__code__.co_filename))},
               "opencv": cv2.__version__, "numpy": np.__version__, "seconds": perf_counter()-started}
    write_new(out / "space_validation.json", receipt)
    if failed:
        raise ValueError(f"{failed} legal configurations failed validation. See {evidence}.")
