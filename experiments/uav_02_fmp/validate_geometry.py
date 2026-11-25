"""Validate the rolling annotation path on the original 548-image val split."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import cv2
import numpy as np

MAIN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MAIN / "code"))
sys.path.insert(0, str(MAIN / "experiments/experiments_test/cv261109"))
from main.config import sha256
from main.dataset import VisDroneZIP
from generator import generate
import model

from geometry import AREA_REFERENCE, RollingRaster, transform_annotations

VAL_SHA = "abeea063037e5d20398837deb11084e652402a34ddf4f207bdf541a6f2a35ef9"
SEED = 20261207


def write_new(path, obj):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(obj, handle, allow_nan=False, separators=(",", ":"))
        handle.write("\n")


def choose_parameters():
    folder = MAIN / "experiments/experiments_test/cv261109"
    historical = model.read_uvl_attributes(folder / "fm_p.uvl")
    aligned = Path(__file__).with_name("fm_p.uvl")
    bindings = json.loads(Path(__file__).with_name("render_bindings.json").read_text(encoding="utf-8"))
    attrs = {}
    for line in aligned.read_text(encoding="utf-8").split("constraints")[0].splitlines():
        feature = line.strip().split(maxsplit=1)
        if not feature or feature[0] not in model.ALL_LEAVES:
            continue
        name = feature[0]
        attrs[name] = {}
        parsed = model.ATTRIBUTE_RE.fullmatch(line)
        if parsed:
            for entry in parsed[2].split(","):
                match = model.NUMERIC_RE.fullmatch(entry)
                key, value = match.groups()
                attrs[name][key] = float(value) if "." in value else int(value)
    for name, recipe in bindings.items():
        if attrs[name].keys() & recipe.keys():
            raise ValueError(f"Render binding overrides a UVL value for {name}.")
        attrs[name].update(recipe)
    if attrs != historical or attrs["Gauss_s5"]["sigma"] != 5.5:
        raise ValueError("FM_P parameters differ from the verified historical renderer.")
    return attrs


def audit_geometry(archive, out):
    started = time.perf_counter()
    if sha256(archive) != VAL_SHA:
        raise ValueError("Wrong validation archive.")
    attrs = choose_parameters()
    fixture = np.random.default_rng(42).integers(0, 256, size=(65, 107, 3), dtype=np.uint8)
    legal = list(model.combinations())
    fixtures = {}
    for leaf in sorted(attrs):
        leaves = next(c for c in legal if leaf in c)
        image, trace = generate(fixture, leaves, attrs, "fixture", SEED)
        again, trace_again = generate(fixture, leaves, attrs, "fixture", SEED)
        if trace != trace_again or not np.array_equal(image, again) or leaf not in {x["feature"] for x in trace}:
            raise ValueError("Leaf fixture is not reproducible.")
        fixtures[leaf] = hashlib.sha256(image.tobytes()).hexdigest()
    leaves = ("Normal", "Motion_k7", "RollingShutter")
    reader = VisDroneZIP(archive, 548)
    gt, rows = None, []
    source_instances, dropped, regions_checked = 0, 0, 0
    try:
        for i in range(548):
            image, record, provenance = reader.record(i)
            pixels, trace = generate(image, leaves, attrs, record["stem"], SEED)
            again, trace_again = generate(image, leaves, attrs, record["stem"], SEED)
            if trace != trace_again or not np.array_equal(pixels, again):
                raise ValueError("Rolling-shutter real image changed between renders.")
            effect = next(x for x in trace if x["feature"] == "RollingShutter")
            before, after, derived = transform_annotations(record, effect)
            if gt is None:
                gt = {**after, "images": [], "annotations": []}
            gt["images"].extend(after["images"])
            for ann in after["annotations"]:
                x, y, w, h = ann["bbox"]
                sx, sy, sw, sh = ann["source_bbox"]
                if min(x, y) < 0 or x+w > image.shape[1] or y+h > image.shape[0] or ann["area"] != sw*sh:
                    raise ValueError("Box clipping or source area failed.")
                gt["annotations"].append({**ann, "id": len(gt["annotations"])+1})
            source_instances += len(before["annotations"])
            dropped += len(before["annotations"]) - len(after["annotations"])

            raster = RollingRaster(image.shape, effect)
            if before["annotations"]:
                ann = before["annotations"][0]
                x, y, w, h = ann["bbox"]
                marker = np.zeros(image.shape[:2], dtype=np.float32)
                marker[int(y):int(np.ceil(y+h)), int(x):int(np.ceil(x+w))] = 1
                ys, xs = np.nonzero(raster.warp(marker))
                expected = None if not len(xs) else [int(xs.min()), int(ys.min()),
                                                     int(xs.max()-xs.min()+1), int(ys.max()-ys.min()+1)]
                if expected != raster.bbox(ann["bbox"]):
                    raise ValueError("Real target disagrees with remapped raster.")
            ignore = record["ignore_regions"] + record["other_regions"]
            if ignore:
                source_mask = np.zeros(image.shape[:2], dtype=np.float32)
                for x, y, w, h in ignore:
                    source_mask[int(y):int(np.ceil(y+h)), int(x):int(np.ceil(x+w))] = 1
                derived_mask = np.zeros(image.shape[:2], dtype=bool)
                for x, y, w, h in derived["ignore_regions"] + derived["other_regions"]:
                    derived_mask[y:y+h, x:x+w] = True
                if not np.array_equal(raster.warp(source_mask) > 0, derived_mask):
                    raise ValueError("Real ignore union differs from the nonlinear warp.")
                regions_checked += 1
            rows.append({**provenance, "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                "rolling": effect, "remaining_gt": len(after["annotations"]),
                "ignore_regions": derived["ignore_regions"], "other_regions": derived["other_regions"]})
    finally:
        reader.close()
    write_new(out / "rolling_validation_gt.json", gt)
    write_new(out / "rolling_validation_members.json", rows)
    write_new(out / "rolling_geometry.json", {
        "status": "PASS_SYNTHETIC_GEOMETRY_ON_VISDRONE_VAL", "images": len(rows),
        "source_bbox_area_reference": AREA_REFERENCE, "original_retained_targets": source_instances,
        "warped_targets": len(gt["annotations"]), "clipped_out_targets": dropped,
        "images_with_verified_ignore_union": regions_checked, "seed": SEED,
        "source_archive_sha256": VAL_SHA, "configuration": leaves, "leaves": attrs,
        "fixture_pixel_sha256": fixtures, "logical_configurations": len(legal),
        "geometry_supported_configurations": len(legal), "unsupported_geometry_configurations": 0,
        "applicability": "Synthetic row-shift model, separate derived GT; no change to frozen first 500.",
        "geometry_sha256": sha256(Path(__file__).with_name("geometry.py")),
        "source_renderer_sha256": sha256(MAIN / "experiments/experiments_test/cv261109/generator.py"),
        "derived_gt_sha256": sha256(out / "rolling_validation_gt.json"),
        "members_sha256": sha256(out / "rolling_validation_members.json"),
        "elapsed_seconds": time.perf_counter()-started,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit_geometry(args.val, args.output)
