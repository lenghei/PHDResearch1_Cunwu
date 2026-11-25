"""Shared-P cache and checksummed pair lifecycle for the frozen evaluation."""
from contextlib import contextmanager
import gzip
from hashlib import sha256 as hash_bytes
import json
import shutil
import time
from uuid import uuid4

import cv2
import numpy as np

from main.config import sha256
from main.labels import ground_truth


PROTOCOL = "uav_05_shared_png_native10_v1"
WARMUPS = 3
PAIR_FILES = ("configuration.json", "ground_truth.json", "image_manifest.jsonl",
              "predictions.jsonl.gz", "metrics.json", "bootstrap.json",
              "bootstrap_indices.npy.gz", "timing.json")
CACHE_FILES = ("manifest.json", "ground_truth.json")


def load(path):
    return json.loads(path.read_bytes())


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def identity(value):
    return hash_bytes(canonical(value)).hexdigest()


def write(path, value):
    with path.open("xb") as stream:
        stream.write(canonical(value)+b"\n")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def disk_guard(path, extra_bytes=0):
    if shutil.disk_usage(path).free < 8 * 1024**3 + extra_bytes:
        raise RuntimeError("Disk space is below the evaluation reserve.")


@contextmanager
def attempt(final):
    """Keep failed or interrupted attempts isolated from verified result paths."""
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        raise FileExistsError(final)
    folder = final.with_name(f".partial-{final.name}-{uuid4().hex}")
    folder.mkdir()
    yield folder
    if final.exists():
        raise FileExistsError(final)
    folder.rename(final)


def seal(folder, files, record):
    write(folder/"COMPLETE.json", {**record, "sha256": {name: sha256(folder/name) for name in files}})


def verify(folder, files, expected):
    receipt = load(folder/"COMPLETE.json")
    require(all(receipt.get(k) == v for k,v in expected.items())
            and receipt.get("status") == "COMPLETE"
            and set(receipt["sha256"]) == set(files), "Completion binding differs.")
    for filename, expected_sha in receipt["sha256"].items():
        require(sha256(folder/filename) == expected_sha, f"Corrupted artifact: {filename}")
    return receipt


def read_cache_image(folder, row):
    start = time.perf_counter()
    payload = (folder/f"{row['image_id']:04d}.png").read_bytes()
    read_end = time.perf_counter()
    require(hash_bytes(payload).hexdigest() == row["png_sha256"], "Cache PNG hash differs.")
    hash_end = time.perf_counter()
    pixels = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    decode_end = time.perf_counter()
    require(pixels is not None and list(pixels.shape) == row["shape"]
            and hash_bytes(pixels.tobytes()).hexdigest() == row["pixel_sha256"], "Cache pixel hash differs.")
    return pixels, {"read_ms": 1000*(read_end-start), "png_hash_ms": 1000*(hash_end-read_end),
                    "decode_ms": 1000*(decode_end-hash_end),
                    "pixel_hash_ms": 1000*(time.perf_counter()-decode_end)}


def read_cache(folder, cache_key, member_ids, verify_pixels=False):
    receipt = verify(folder, CACHE_FILES, {"cache_key": cache_key})
    manifest = load(folder/"manifest.json")
    records = load(folder/"ground_truth.json")["records"]
    rows = manifest["rows"]
    require(receipt["images"] == len(member_ids) and member_ids == [r["image_id"] for r in rows]
            == [r["id"] for r in records] and len(member_ids) == len(set(member_ids)), "Cache/GT members differ.")
    if verify_pixels:
        for row in rows:
            read_cache_image(folder, row)
    return manifest


def shared_cache(root, cache_key, full_members, reader, render, monitor_disk=True):
    """Materialize the VM36 PNG strategy once per exact P/source binding."""
    folder = root/cache_key
    ids = [r["image_id"] for r in full_members]
    if folder.exists():
        return folder, read_cache(folder, cache_key, ids), True
    began = time.perf_counter()
    with attempt(folder) as pending:
        rows, records = [], []
        for i, source_member in enumerate(full_members):
            if monitor_disk and i % 16 == 0:
                disk_guard(pending)
            image, record, observed = reader.record(i)
            require(observed == source_member and record["id"] == ids[i], "Frozen source membership or order differs.")
            pixels, effects = render(image, record["stem"])
            require(pixels.dtype == np.uint8 and list(pixels.shape) == source_member["shape"],
                    "v1 cache must preserve original image geometry.")
            ok, encoded = cv2.imencode(".png", pixels)
            require(ok, "PNG encoding failed.")
            payload = encoded.tobytes()
            with (pending/f"{ids[i]:04d}.png").open("xb") as stream:
                stream.write(payload)
            rows.append({"image_id": ids[i], "source": source_member, "shape": list(pixels.shape),
                         "png_sha256": hash_bytes(payload).hexdigest(),
                         "pixel_sha256": hash_bytes(pixels.tobytes()).hexdigest(), "effects": effects,
                         "png_bytes": len(payload)})
            records.append(record)
        gt, _ = ground_truth(records, "native10")
        write(pending/"ground_truth.json", {"records": records, "coco": gt})
        write(pending/"manifest.json", {"rows": rows, "cache_generation_seconds": time.perf_counter()-began,
                                       "cache_png_bytes": sum(x["png_bytes"] for x in rows)})
        seal(pending, CACHE_FILES, {"status": "COMPLETE", "cache_key": cache_key, "images": len(rows)})
    return folder, read_cache(folder, cache_key, ids), False


def prediction_rows(image_id, boxes):
    require(np.isfinite(boxes).all(), "Nonfinite prediction.")
    rows = []
    for x1, y1, x2, y2, score, cat in boxes:
        require(int(cat) == cat and 0 <= cat < 10 and 0 <= score <= 1, "Invalid prediction class/score.")
        if x2 > x1 and y2 > y1:
            rows.append({"image_id": image_id, "class_id": int(cat),
                         "bbox": list(map(float, (x1, y1, x2-x1, y2-y1))), "score": float(score)})
    return rows


def indices_for(images, samples, seed):
    return np.random.default_rng(seed).integers(0, images, size=(samples, images))


def verify_pair(folder, binding):
    receipt = verify(folder, PAIR_FILES, {"binding": identity(binding), "images": binding["images"]})
    require(load(folder/"configuration.json") == binding, "Pair configuration differs.")
    metrics = load(folder/"metrics.json")
    require(metrics["images"] == binding["images"], "Pair metrics incomplete.")
    return receipt


def verify_repeat(first, second, binding):
    a, b = (verify_pair(path, binding) for path in (first, second))
    comparable = set(PAIR_FILES) - {"predictions.jsonl.gz", "bootstrap_indices.npy.gz", "timing.json"}
    require(all(a["sha256"][name] == b["sha256"][name] for name in comparable), "Repeat outputs differ.")
    for name in ("predictions.jsonl.gz", "bootstrap_indices.npy.gz"):
        with gzip.open(first/name, "rb") as x, gzip.open(second/name, "rb") as y:
            while True:
                block = x.read(1024*1024)
                require(block == y.read(1024*1024), f"Repeat content differs: {name}")
                if not block:
                    break


def measure_pair(final, binding, cache, cache_rows, detector, replay, bootstrap,
                 synchronize=lambda: None, reset_memory=lambda: None, memory=lambda: {},
                 monitor_disk=True, samples=200, seed=20261207,
                 clock_origin=None, model_load_seconds=0, prepare_seconds=0):
    """Use the CV261207 detector, replay and COCO bootstrap with the shared PNG cache."""
    if final.exists():
        verify_pair(final, binding)
        return "VERIFIED_SKIP"
    began = time.perf_counter()
    wall_start = began if clock_origin is None else clock_origin
    rows = cache_rows["rows"]
    require(len(rows) == binding["images"] and sha256(cache/"manifest.json") == binding["cache_manifest_sha256"]
            and sha256(cache/"ground_truth.json") == binding["gt_sha256"], "Pair cache identity differs.")
    with attempt(final) as pending:
        write(pending/"configuration.json", binding)
        shutil.copyfile(cache/"ground_truth.json", pending/"ground_truth.json")
        warm_image, _ = read_cache_image(cache, rows[0])
        warm_start = time.perf_counter()
        for _ in range(WARMUPS):
            detector(warm_image)
        synchronize()
        warm_seconds = time.perf_counter()-warm_start
        reset_memory()
        times = []
        with gzip.open(pending/"predictions.jsonl.gz", "wt", encoding="utf-8", newline="\n") as raw, (
                pending/"image_manifest.jsonl").open("wb") as manifest:
            for i,row in enumerate(rows):
                if monitor_disk and i % 32 == 0:
                    disk_guard(pending)
                t = time.perf_counter()
                pixels, stages = read_cache_image(cache, row)
                synchronize()
                before = time.perf_counter()
                boxes, tensor, device_ms = detector(pixels)
                synchronize()
                predicted = time.perf_counter()
                preds = prediction_rows(row["image_id"], boxes)
                after_extract = time.perf_counter()
                manifest_line = canonical({**row, "tensor": tensor})+b"\n"
                raw_line = canonical({"image_id": row["image_id"], "predictions": preds}).decode()+"\n"
                after_serial = time.perf_counter()
                raw.write(raw_line)
                manifest.write(manifest_line)
                raw.flush()
                manifest.flush()
                flushed = time.perf_counter()
                times.append({"image_id": row["image_id"], **stages,
                    "synchronized_predict_ms": (predicted-before)*1000, "backend_stage_ms": device_ms,
                    "extract_ms": (after_extract-predicted)*1000, "serialize_ms": (after_serial-after_extract)*1000,
                    "write_flush_ms": (flushed-after_serial)*1000, "pipeline_ms": (flushed-t)*1000})
                if (i+1) % 100 == 0 or i+1 == len(rows):
                    print(f"Inference {i+1}/{len(rows)}", flush=True)
        prediction_memory = memory()
        print("Recomputing COCO metrics from saved predictions", flush=True)
        evaluated = time.perf_counter()
        metrics, evaluator = replay(pending)
        evaluation_seconds = time.perf_counter()-evaluated
        require(metrics["images"] == len(rows), "Pair did not evaluate all images.")
        write(pending/"metrics.json", metrics)
        image_ids = [int(v) for v in evaluator._paramsEval.imgIds]
        require(image_ids == [r["image_id"] for r in rows], "Bootstrap population order differs.")
        draws = indices_for(len(image_ids), samples, seed)
        boot_started = time.perf_counter()
        intervals, estimates = bootstrap(evaluator, samples=samples, seed=seed,
            progress_callback=lambda done, total: print(f"Bootstrap {done}/{total}", flush=True))
        with gzip.open(pending/"bootstrap_indices.npy.gz", "wb") as stream:
            np.save(stream, draws.astype(np.int32), allow_pickle=False)
        write(pending/"bootstrap.json", {"image_ids": image_ids, "seed": seed, "samples": samples,
                                        "metrics": intervals, "draws": estimates})
        latency = np.asarray([r["pipeline_ms"] for r in times])
        latency_ci = np.quantile(latency[draws].mean(axis=1), [.025,.975]).tolist()
        write(pending/"timing.json", {"images": len(rows), "samples": times, "warmups": WARMUPS,
            "model_load_seconds": model_load_seconds, "pair_prepare_seconds": prepare_seconds,
            "warmup_seconds": warm_seconds, "cache_generation_seconds_at_source": cache_rows["cache_generation_seconds"],
            "COCO_replay_evaluation_seconds": evaluation_seconds, "bootstrap_seconds": time.perf_counter()-boot_started,
            "worker_pair_seconds_to_sealing": time.perf_counter()-began,
            "wall_seconds_to_sealing_including_setup": time.perf_counter()-wall_start,
            "cached_pipeline_mean_ms": float(latency.mean()), "cached_pipeline_mean_ci95_ms": latency_ci,
            "raw_prediction_gzip_bytes": (pending/"predictions.jsonl.gz").stat().st_size,
            "cached_png_bytes": cache_rows["cache_png_bytes"], **prediction_memory})
        seal(pending, PAIR_FILES, {"status": "COMPLETE", "binding": identity(binding), "images": len(rows)})
    verify_pair(final, binding)
    return "COMPLETE"
