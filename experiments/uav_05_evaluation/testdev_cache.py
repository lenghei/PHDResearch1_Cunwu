"""Build a deterministic shared cache with native-ten GT in rendered coordinates."""
from hashlib import sha256 as hash_bytes
from time import perf_counter

import cv2
import numpy as np

from geometry import AREA_REFERENCE, transform_annotations
from main.labels import ground_truth
from pipeline import CACHE_FILES, disk_guard, read_cache, require, seal, write


def build_cache(folder, key, p, members, reader, generate, monitor_disk=True):
    folder.mkdir()
    started = perf_counter()
    rows, records, coco = [], [], None
    rolling = p["annotation_mode"] == "WARP_GT_AND_IGNORE_ORIGINAL_AREA"
    for i, member in enumerate(members):
        if monitor_disk and i % 16 == 0:
            disk_guard(folder)
        image, record, observed = reader.record(i)
        require(member == observed and record["id"] == member["image_id"],
                "Image or annotation membership differs.")
        pixels, effects = generate(image, p["leaves"], p["recipe"], record["stem"], p["seed"])
        require(pixels.dtype == np.uint8 and list(pixels.shape) == member["shape"],
                "The renderer changed the source canvas.")
        geometry = [effect for effect in effects if effect["feature"] == "RollingShutter"]
        require(len(geometry) == int(rolling), "Rendered geometry differs from the declared mode.")
        if rolling:
            _, transformed, record = transform_annotations(record, geometry[0])
            require(transformed["info"]["area_reference"] == AREA_REFERENCE,
                    "Transformed ground truth lost the source area definition.")
            if coco is None:
                coco = {**transformed, "images": [], "annotations": []}
            coco["images"].extend(transformed["images"])
            for ann in transformed["annotations"]:
                coco["annotations"].append({**ann, "id": len(coco["annotations"]) + 1})
        ok, png = cv2.imencode(".png", pixels)
        require(ok, "Cache PNG encoding failed.")
        payload = png.tobytes()
        (folder / f"{record['id']:04d}.png").write_bytes(payload)
        rows.append({
            "image_id": record["id"], "source": member, "shape": list(pixels.shape),
            "png_sha256": hash_bytes(payload).hexdigest(),
            "pixel_sha256": hash_bytes(pixels.tobytes()).hexdigest(),
            "effects": effects, "png_bytes": len(payload),
        })
        records.append(record)
    if not rolling:
        coco, _ = ground_truth(records, "native10")
    write(folder / "ground_truth.json", {"records": records, "coco": coco})
    write(folder / "manifest.json", {
        "rows": rows, "cache_png_bytes": sum(row["png_bytes"] for row in rows),
    })
    seal(folder, CACHE_FILES, {"status": "COMPLETE", "images": len(rows), "cache_key": key})
    cached = read_cache(folder, key, [member["image_id"] for member in members])
    return {**cached, "cache_generation_seconds": perf_counter() - started}
