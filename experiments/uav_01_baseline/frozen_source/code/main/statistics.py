"""COCO AP/AR and image-bootstrap utilities.

Areas use original pixels. AP uses IoU .50:.05:.95 and 101 recall points.
AR is category-averaged maximum recall at the configured cap. The custom
cap is 3000 per image/category, rather than COCO's standard 100.
Custom size bins are disjoint; coco_small uses inclusive [0, 32**2]."""

from __future__ import annotations

import contextlib
import copy
import gzip
import io
import math
import pickle
from pathlib import Path
from typing import Any, Callable

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


METRIC_NAMES = ("map50", "map50_95", "ar50", "ar50_95")
AREA_LABELS = ("all", "tiny", "small", "medium", "large", "coco_small")
CACHE_FORMAT = "new_coco_statistics.matches.v1"


def _area_ranges() -> list[list[float]]:
    # COCOeval's comparison is <= upper bound, so nextafter implements <.
    below = lambda value: float(np.nextafter(float(value), -np.inf))
    return [
        [0.0, float("inf")],
        [0.0, below(16**2)],
        [float(16**2), below(32**2)],
        [float(32**2), below(96**2)],
        [float(96**2), float("inf")],
        [0.0, float(32**2)],
    ]


def _mean_valid(values: np.ndarray) -> float | None:
    valid = values[values > -1]
    return float(valid.mean()) if valid.size else None


def _bbox_area(record: dict[str, Any]) -> float:
    box = record.get("bbox")
    if box is None or len(box) != 4 or not all(math.isfinite(float(x)) for x in box):
        raise ValueError("Every bbox must contain four finite xywh values.")
    if box[2] <= 0 or box[3] <= 0:
        raise ValueError("Bounding-box width and height must be positive.")
    return float(box[2]) * float(box[3])


def _validated_records(gt: dict, predictions: list[dict]) -> tuple[dict, list[dict]]:
    source = copy.deepcopy(gt)
    source.setdefault("info", {})
    source.setdefault("annotations", [])
    if not source.get("images") or not source.get("categories"):
        raise ValueError("GT must contain at least one image and one category.")
    image_ids = [im["id"] for im in source["images"]]
    category_ids = [cat["id"] for cat in source["categories"]]
    if any(not isinstance(x, int) for x in image_ids + category_ids):
        raise ValueError("Image and category IDs must be integers.")
    if len(set(image_ids)) != len(image_ids) or len(set(category_ids)) != len(category_ids):
        raise ValueError("Image and category IDs must each be unique.")
    image_ids, category_ids = set(image_ids), set(category_ids)
    annotation_ids = set()
    for ann in source["annotations"]:
        if not isinstance(ann.get("id"), int) or ann["id"] <= 0 or ann["id"] in annotation_ids:
            raise ValueError("GT annotation IDs must be unique positive integers (0 means unmatched in COCO).")
        annotation_ids.add(ann["id"])
        if ann["image_id"] not in image_ids or ann["category_id"] not in category_ids:
            raise ValueError("GT annotation references an unknown image/category.")
        ann["area"] = _bbox_area(ann)
        ann.setdefault("iscrowd", 0)
        if ann.get("ignore", 0) and not ann["iscrowd"]:
            raise ValueError("COCOeval does not honor bbox ignore=1 without iscrowd. Apply the dataset's ignore policy before evaluation.")
    detections = copy.deepcopy(list(predictions))
    for pred in detections:
        if pred["image_id"] not in image_ids or pred["category_id"] not in category_ids:
            raise ValueError("Prediction references an unknown image/category; check the class mapping.")
        _bbox_area(pred)
        if not math.isfinite(float(pred["score"])) or not 0 <= pred["score"] <= 1:
            raise ValueError("Prediction scores must be finite and in [0, 1].")
    return source, detections


def _metric_rows(evaluator: COCOeval, include_per_class: bool = True,
                 include_gt_counts: bool = True) -> tuple[list[dict], list[dict]]:
    p = evaluator.params
    precision, recall = evaluator.eval["precision"], evaluator.eval["recall"]
    at50 = np.flatnonzero(np.isclose(p.iouThrs, 0.5))
    if len(at50) != 1:
        raise ValueError("Exactly one IoU=0.50 threshold is required.")
    metadata = getattr(evaluator, "_new_statistics_metadata", {})
    category_names = {cat["id"]: cat["name"] for cat in metadata.get("categories", [])}
    n_images, n_areas = len(p.imgIds), len(p.areaRng)
    per_area, per_class = [], []

    def one_row(area_index: int, category_index: int | None) -> dict:
        cat_slice = slice(None) if category_index is None else slice(category_index, category_index + 1)
        prec = precision[:, :, cat_slice, area_index, -1]
        rec = recall[:, cat_slice, area_index, -1]
        cats = range(len(p.catIds)) if category_index is None else [category_index]
        gt_count = 0 if include_gt_counts else None
        if include_gt_counts:
            for k in cats:
                offset = (k * n_areas + area_index) * n_images
                gt_count += sum(int(np.count_nonzero(np.asarray(e["gtIgnore"]) == 0))
                                for e in evaluator.evalImgs[offset:offset + n_images] if e is not None)
        row = {
            "size": p.areaRngLbl[area_index],
            "map50": _mean_valid(prec[at50]),
            "map50_95": _mean_valid(prec),
            "ar50": _mean_valid(rec[at50]),
            "ar50_95": _mean_valid(rec),
            "gt_count": gt_count,
            "evaluated_categories": int(np.count_nonzero(np.any(rec > -1, axis=0))),
            "image_count": n_images,
            "max_det": int(p.maxDets[-1]),
            "area_reference": "original_image_pixels",
        }
        if category_index is not None:
            category_id = p.catIds[category_index]
            row.update(category_id=int(category_id), category_name=category_names.get(category_id, str(category_id)))
        return row

    for a in range(n_areas):
        per_area.append(one_row(a, None))
        if include_per_class:
            per_class.extend(one_row(a, k) for k in range(len(p.catIds)))
    return per_area, per_class


def evaluate_coco(gt: dict, pred: list[dict], max_det: int = 3000) -> tuple[list[dict], list[dict], COCOeval]:
    """Return area metrics, per-class metrics and COCO matches.

    Missing populations return None; GT with no predictions returns zero.
    Apply class mapping and dataset ignore rules before calling."""
    if isinstance(max_det, bool) or not isinstance(max_det, int) or max_det < 1:
        raise ValueError("max_det must be a positive integer.")
    source, detections = _validated_records(gt, pred)
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = source
        coco_gt.createIndex()
        if detections:
            coco_dt = coco_gt.loadRes(detections)
        else:
            # COCO.loadRes([]) indexes the first prediction and raises IndexError.
            coco_dt = COCO()
            coco_dt.dataset = {"images": copy.deepcopy(source["images"]),
                               "categories": copy.deepcopy(source["categories"]), "annotations": []}
            coco_dt.createIndex()
        evaluator = COCOeval(coco_gt, coco_dt, "bbox")
        evaluator.params.maxDets = [max_det]
        evaluator.params.areaRng = _area_ranges()
        evaluator.params.areaRngLbl = list(AREA_LABELS)
        evaluator.evaluate()
        evaluator.accumulate()
    evaluator._new_statistics_metadata = {
        "format": CACHE_FORMAT,
        "categories": copy.deepcopy(source["categories"]),
        "image_ids": sorted(im["id"] for im in source["images"]),
        "area_reference": "original_image_pixels",
        "area_definition": "bbox_width_times_height",
        "size_bounds": "tiny<256;256<=small<1024;1024<=medium<9216;large>=9216;coco_small<=1024",
        "max_det": max_det,
        "max_det_scope": "per_image_per_category_COCOeval",
    }
    metrics, per_class = _metric_rows(evaluator)
    return metrics, per_class, evaluator


def resample_evaluator(evaluator: COCOeval, image_indices: list[int] | np.ndarray) -> COCOeval:
    """Accumulate sampled image positions, including repetitions.

    Reuse per-image matches with distinct synthetic IDs. Match IDs retain
    matched/unmatched status; stable score-tie ordering follows sample order."""
    indices = np.asarray(image_indices)
    n_original = len(evaluator._paramsEval.imgIds)
    if indices.ndim != 1 or not len(indices) or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError("Sample indices must be a nonempty one-dimensional integer array.")
    if np.any(indices < 0) or np.any(indices >= n_original):
        raise ValueError("Sample index out of range.")
    sample = COCOeval(iouType="bbox")
    sample.params = copy.deepcopy(evaluator._paramsEval)
    sample.params.imgIds = list(range(1, len(indices) + 1))
    sample._paramsEval = copy.deepcopy(sample.params)
    blocks = len(sample.params.catIds) * len(sample.params.areaRng)
    sample.evalImgs = [evaluator.evalImgs[block * n_original + int(i)]
                       for block in range(blocks) for i in indices]
    sample._new_statistics_metadata = copy.deepcopy(getattr(evaluator, "_new_statistics_metadata", {}))
    with contextlib.redirect_stdout(io.StringIO()):
        from new_fast_coco_accumulate import accumulate_fast
        accumulate_fast(sample)
    return sample


def bootstrap_coco(evaluator: COCOeval, samples: int = 200, seed: int = 42,
                   progress_callback: Callable[[int, int], None] | None = None) -> tuple[list[dict], dict]:
    """Return percentile image-bootstrap 95% CIs and paired draws.

    Pair runs with identical sorted image IDs, seed and sample count.
    These intervals condition on trained weights, not training seeds.
    Progress callbacks run every 25 draws and at completion."""
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer.")
    image_ids = [int(x) for x in evaluator._paramsEval.imgIds]
    if image_ids != sorted(set(image_ids)):
        raise ValueError("Bootstrap requires unique, sorted source image IDs.")
    estimates, _ = _metric_rows(evaluator, include_per_class=False)
    values = {row["size"]: {key: [] for key in METRIC_NAMES} for row in estimates}
    rng = np.random.default_rng(seed)
    for draw in range(samples):
        indices = rng.integers(0, len(image_ids), size=len(image_ids))
        boot = resample_evaluator(evaluator, indices)
        rows, _ = _metric_rows(boot, include_per_class=False, include_gt_counts=False)
        for row in rows:
            for key in METRIC_NAMES:
                values[row["size"]][key].append(row[key])
        completed = draw + 1
        if progress_callback is not None and (completed % 25 == 0 or completed == samples):
            progress_callback(completed, samples)
    intervals = []
    for estimate in estimates:
        for key in METRIC_NAMES:
            valid = np.asarray([x for x in values[estimate["size"]][key] if x is not None], dtype=float)
            intervals.append({
                "size": estimate["size"], "metric": key, "estimate": estimate[key],
                "bootstrap_mean": float(valid.mean()) if valid.size else None,
                "ci95": [float(x) for x in np.percentile(valid, [2.5, 97.5])] if valid.size else [None, None],
                "n_valid": int(valid.size), "n_samples": samples, "seed": seed,
                "sampling_unit": "image", "method": "paired_percentile_bootstrap",
            })
    sample_values = {"image_ids": image_ids, "samples": samples, "seed": seed,
                     "sampling": "numpy.default_rng.integers, sorted source image positions",
                     "values": values}
    return intervals, sample_values


def save_evaluation_cache(evaluator: COCOeval, path: str | Path) -> None:
    """Save COCO parameters, arrays and per-image matches."""
    payload = {"format": CACHE_FORMAT, "params": evaluator._paramsEval,
               "evalImgs": evaluator.evalImgs, "eval": evaluator.eval,
               "metadata": getattr(evaluator, "_new_statistics_metadata", {})}
    with gzip.open(path, "wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)


def load_evaluation_cache(path: str | Path, *, trusted: bool = False) -> COCOeval:
    """Load a trusted local cache; external pickle files are unsafe."""
    if not trusted:
        raise ValueError("Only load your own locally generated cache, with trusted=True. Never unpickle an external download.")
    with gzip.open(path, "rb") as stream:
        payload = pickle.load(stream)
    if payload.get("format") != CACHE_FORMAT:
        raise ValueError("Unsupported statistics cache format.")
    evaluator = COCOeval(iouType="bbox")
    evaluator.params = copy.deepcopy(payload["params"])
    evaluator._paramsEval = copy.deepcopy(payload["params"])
    evaluator.evalImgs, evaluator.eval = payload["evalImgs"], payload["eval"]
    evaluator._new_statistics_metadata = payload["metadata"]
    return evaluator
