"""Generate a complete annotated VisDrone val set for one legal FM_P configuration."""
import hashlib
import json
from pathlib import Path
import time

from validate_geometry import (SEED, VAL_SHA, VisDroneZIP, choose_parameters, generate,
                               model, sha256, transform_annotations, write_new)

import cv2
import numpy as np

from main.labels import ground_truth


def render_set(archive, out, config_id):
    started = time.perf_counter()
    if sha256(archive) != VAL_SHA:
        raise ValueError("Wrong VisDrone val archive.")
    leaves = next((c for c in model.combinations() if model.configuration_id(c) == config_id), None)
    if leaves is None:
        raise ValueError("The requested configuration ID is outside FM_P.")
    attrs = choose_parameters()
    folder = out / config_id
    images = folder / "images"
    images.mkdir(parents=True)
    receipt = {"configuration_id": config_id, "leaves": leaves, "render_seed": SEED,
               "source_archive_sha256": VAL_SHA, "generator_sha256": sha256(Path(generate.__code__.co_filename)),
               "uvl_sha256": sha256(Path(__file__).with_name("fm_p.uvl")),
               "bindings_sha256": sha256(Path(__file__).with_name("render_bindings.json")),
               "geometry_sha256": sha256(Path(__file__).with_name("geometry.py")),
               "set_renderer_sha256": sha256(Path(__file__)), "opencv": cv2.__version__,
               "numpy": np.__version__}
    gt = None
    reader = VisDroneZIP(archive, 548)
    source_targets = 0
    try:
        with (folder / "members.jsonl").open("x", encoding="utf-8", newline="\n") as manifest:
            for i in range(548):
                original, record, member = reader.record(i)
                pixels, trace = generate(original, leaves, attrs, record["stem"], SEED)
                if pixels.shape != original.shape or pixels.dtype != np.uint8:
                    raise ValueError(f"Image canvas changed for {record['stem']}.")
                rolling = next((item for item in trace if item["feature"] == "RollingShutter"), None)
                if rolling:
                    before, after, derived = transform_annotations(record, rolling)
                else:
                    before, _ = ground_truth([record], "native10")
                    after, derived = before, record
                source_targets += len(before["annotations"])
                if gt is None:
                    gt = {**after, "images": [], "annotations": []}
                filename = record["stem"] + ".png"
                ok, encoded = cv2.imencode(".png", pixels, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                if not ok or not np.array_equal(cv2.imdecode(encoded, cv2.IMREAD_COLOR), pixels):
                    raise ValueError(f"Lossless PNG verification failed for {filename}.")
                payload = encoded.tobytes()
                with (images / filename).open("xb") as stream:
                    stream.write(payload)
                info = {**after["images"][0], "file_name": "images/" + filename}
                gt["images"].append(info)
                for ann in after["annotations"]:
                    gt["annotations"].append({**ann, "id": len(gt["annotations"])+1})
                manifest.write(json.dumps({
                    **member, "output_png": "images/" + filename,
                    "pixel_sha256": hashlib.sha256(pixels.tobytes()).hexdigest(),
                    "png_sha256": hashlib.sha256(payload).hexdigest(), "effects": trace,
                    "ignore_regions": derived["ignore_regions"], "other_regions": derived["other_regions"],
                    "source_targets": len(before["annotations"]), "output_targets": len(after["annotations"]),
                }, allow_nan=False, separators=(",", ":")) + "\n")
    finally:
        reader.close()
    if gt is None or len({r["id"] for r in gt["images"]}) != 548:
        raise ValueError("Incomplete or duplicate image set.")
    write_new(folder / "ground_truth.json", gt)
    write_new(folder / "set_receipt.json", {**receipt, "status": "RENDER_VERIFIED_DEVELOPMENT_VAL",
              "images": len(gt["images"]), "source_targets": source_targets,
              "output_targets": len(gt["annotations"]), "area_reference": gt["info"].get("area_reference",
                                                                                         "original_image_pixels"),
              "members_sha256": sha256(folder / "members.jsonl"),
              "ground_truth_sha256": sha256(folder / "ground_truth.json"),
              "seconds": time.perf_counter()-started})
    return folder
