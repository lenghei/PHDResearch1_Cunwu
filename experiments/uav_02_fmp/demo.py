"""Render a deterministic FM_P sample and preview its annotation geometry."""
import argparse
import hashlib
from pathlib import Path
import random
import sys

sys.dont_write_bytecode = True
import cv2
import numpy as np

from validate_geometry import (SEED, VAL_SHA, VisDroneZIP, choose_parameters, generate,
                               model, sha256, transform_annotations, write_new)
from main.labels import ground_truth


def show(image, gt, record):
    panel = image.copy()
    for x, y, w, h in record["ignore_regions"] + record["other_regions"]:
        cv2.rectangle(panel, (int(x), int(y)), (int(x+w)-1, int(y+h)-1), (36, 117, 214), 1)
    for ann in gt["annotations"]:
        x, y, w, h = ann["bbox"]
        cv2.rectangle(panel, (int(x), int(y)), (int(x+w)-1, int(y+h)-1), (67, 233, 96), 1)
    return panel


def render(archive, out, config_id, selection_seed, image_index):
    if sha256(archive) != VAL_SHA:
        raise ValueError("Wrong VisDrone val archive.")
    candidates = sorted(model.combinations())
    if config_id is None:
        leaves = random.Random(selection_seed).choice(candidates)
    else:
        leaves = next((c for c in candidates if model.configuration_id(c) == config_id), None)
        if leaves is None:
            raise ValueError("The requested configuration ID is outside FM_P.")
    attrs = choose_parameters()
    reader = VisDroneZIP(archive, 548)
    try:
        original, record, member = reader.record(image_index)
    finally:
        reader.close()
    pixels, effects = generate(original, leaves, attrs, record["stem"], SEED)
    repeated, second = generate(original, leaves, attrs, record["stem"], SEED)
    if effects != second or not np.array_equal(pixels, repeated):
        raise ValueError("The same input and seed produced different pixels.")
    rolling = next((e for e in effects if e["feature"]=="RollingShutter"), None)
    if rolling:
        source_gt, transformed, derived = transform_annotations(record, rolling)
    else:
        source_gt, _ = ground_truth([record], "native10")
        transformed, derived = source_gt, record
    panels = (("Source", show(original, source_gt, record)), ("Rendered", show(pixels, transformed, derived)))
    for title, panel in panels:
        cv2.putText(panel, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, .85, (250, 250, 250), 2)
    out.mkdir(parents=True)
    if not cv2.imwrite(str(out/"preview.png"), np.hstack([p for _,p in panels])):
        raise RuntimeError("Preview could not be written.")
    write_new(out/"ground_truth.json", {"source":source_gt,"rendered":transformed})
    write_new(out/"demo_receipt.json", {"configuration_id":model.configuration_id(leaves),
        "leaves":leaves,"selection_seed":selection_seed,"selection_mode":"fixed_id" if config_id else "seeded_uniform",
        "render_seed":SEED,"population":len(candidates),"source_archive_sha256":VAL_SHA,"image":member,
        "effects":effects,"pixel_sha256":hashlib.sha256(pixels.tobytes()).hexdigest(),
        "annotation_counts":{"source":len(source_gt["annotations"]),"rendered":len(transformed["annotations"])},
        "repeated_pixel_identity":True,"preview_sha256":sha256(out/"preview.png"),
        "ground_truth_sha256":sha256(out/"ground_truth.json")})
    print(out)


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--val", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--config-id")
    p.add_argument("--sample-seed", type=int, default=20261109)
    p.add_argument("--image-index", type=int, choices=range(548), default=0,
                   metavar="0..547")
    a = p.parse_args()
    render(a.val, a.output, a.config_id, a.sample_seed, a.image_index)
