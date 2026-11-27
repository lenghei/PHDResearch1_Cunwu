"""Exact COCO bootstrap accumulation for cached single-cap bbox evaluations."""
from copy import copy
from types import SimpleNamespace

import numpy as np
import psutil

from main.statistics import METRIC_NAMES, _metric_rows


def prepare(evaluator):
    params = evaluator._paramsEval
    if not params.useCats or len(params.maxDets) != 1:
        return None
    cap = params.maxDets[0]
    images, areas = len(params.imgIds), len(params.areaRng)
    ious = len(params.iouThrs)
    if images * cap > np.iinfo(np.int32).max:
        return None

    categories = []
    for k in range(len(params.catIds)):
        offset = k * areas * images
        by_area = evaluator.evalImgs[offset:offset + areas * images]
        image_scores = [np.asarray(entry["dtScores"][:cap]) if entry is not None else np.empty(0)
                        for entry in by_area[:images]]
        scores = np.concatenate(image_scores)
        bounds = np.cumsum([0, *(len(row) for row in image_scores)])
        image_detections = [np.arange(lo, hi, dtype=np.int32) for lo, hi in zip(bounds, bounds[1:])]

        area_data = []
        for a in range(areas):
            entries = by_area[a * images:(a + 1) * images]
            gt = np.array([np.count_nonzero(entry["gtIgnore"] == 0) if entry is not None else 0
                           for entry in entries])
            if not gt.any():
                area_data.append((gt, None, None))
                continue
            for entry, reference in zip(entries, image_scores):
                if not np.array_equal(entry["dtScores"][:cap] if entry is not None else [], reference):
                    return None
            empty = np.empty((ious, 0), dtype=bool)
            valid = np.concatenate([~entry["dtIgnore"][:, :cap] if entry is not None else empty
                                    for entry in entries], axis=1)
            true_positive = np.concatenate([entry["dtMatches"][:, :cap] != 0 if entry is not None else empty
                                            for entry in entries], axis=1)
            true_positive &= valid
            area_data.append((gt, valid, true_positive))
        categories.append((scores, image_detections, area_data))
    return categories


def accumulate_draw(params, categories, indices):
    t, r, k, a = (len(params.iouThrs), len(params.recThrs),
                  len(params.catIds), len(params.areaRng))
    precision = np.full((t, r, k, a, 1), -1.)
    recall = np.full((t, k, a, 1), -1.)
    scores = np.full_like(precision, -1.)
    zero_recall = params.recThrs == 0
    epsilon = np.spacing(1)
    draw = indices.tolist()
    for cat, (source_scores, image_detections, area_data) in enumerate(categories):
        detections = np.concatenate([image_detections[i] for i in draw])
        # Stable ties follow the order of the sampled images, including repeats.
        order = detections[np.argsort(-source_scores[detections], kind="mergesort")]
        sorted_scores = source_scores[order]
        for area, (gt, valid, true_positive) in enumerate(area_data):
            n_gt = gt[indices].sum()
            if not n_gt:
                continue
            precision[:, :, cat, area, 0] = 0
            recall[:, cat, area, 0] = 0
            scores[:, :, cat, area, 0] = 0
            if not len(order):
                continue
            scores[:, zero_recall, cat, area, 0] = sorted_scores[0]
            for iou in range(t):
                positions = np.flatnonzero(true_positive[iou, order])
                if not len(positions):
                    continue
                starts = np.empty_like(positions)
                starts[0] = 0
                starts[1:] = positions[:-1] + 1
                # Interpolated precision can peak only at a true positive.
                relevant = valid[iou, order[:positions[-1] + 1]]
                totals = np.cumsum(np.add.reduceat(relevant, starts, dtype=np.int32), dtype=np.int32)
                hits = np.arange(1, len(positions) + 1, dtype=float)
                rc = hits / n_gt
                pr = hits / (totals.astype(float) + epsilon)
                recall[iou, cat, area, 0] = rc[-1]
                envelope = np.maximum.accumulate(pr[::-1])[::-1]
                at = np.searchsorted(rc, params.recThrs, side="left")
                found = at < len(positions)
                precision[iou, found, cat, area, 0] = envelope[at[found]]
                positive = found & ~zero_recall
                scores[iou, positive, cat, area, 0] = sorted_scores[positions[at[positive]]]
    return {"precision": precision, "recall": recall, "scores": scores}


def sparse_bootstrap(evaluator, samples, seed, progress_callback, fallback):
    if psutil.virtual_memory().available < 2 * 2**30:
        return fallback(evaluator, samples, seed, progress_callback)
    if isinstance(samples, bool) or not isinstance(samples, int) or samples < 1:
        raise ValueError("samples must be a positive integer.")
    params = evaluator._paramsEval
    image_ids = [int(i) for i in params.imgIds]
    if image_ids != sorted(set(image_ids)):
        raise ValueError("Bootstrap requires unique, sorted source image IDs.")
    categories = prepare(evaluator)
    if categories is None:
        return fallback(evaluator, samples, seed, progress_callback)

    estimates, _ = _metric_rows(evaluator, include_per_class=False)
    values = {row["size"]: {key: [] for key in METRIC_NAMES} for row in estimates}
    resample = SimpleNamespace(params=copy(params))
    resample.params.imgIds = list(range(1, len(image_ids) + 1))
    rng = np.random.default_rng(seed)
    for completed in range(1, samples + 1):
        indices = rng.integers(0, len(image_ids), size=len(image_ids))
        resample.eval = accumulate_draw(params, categories, indices)
        rows, _ = _metric_rows(resample, include_per_class=False, include_gt_counts=False)
        for row in rows:
            for key in METRIC_NAMES:
                values[row["size"]][key].append(row[key])
        if progress_callback is not None and (completed % 25 == 0 or completed == samples):
            progress_callback(completed, samples)

    intervals = []
    for estimate in estimates:
        for key in METRIC_NAMES:
            valid = np.asarray([v for v in values[estimate["size"]][key] if v is not None], dtype=float)
            intervals.append({
                "size": estimate["size"], "metric": key, "estimate": estimate[key],
                "bootstrap_mean": float(valid.mean()) if valid.size else None,
                "ci95": np.percentile(valid, [2.5, 97.5]).tolist() if valid.size else [None, None],
                "n_valid": int(valid.size), "n_samples": samples, "seed": seed,
                "sampling_unit": "image", "method": "paired_percentile_bootstrap",
            })
    return intervals, {
        "image_ids": image_ids, "samples": samples, "seed": seed,
        "sampling": "numpy.default_rng.integers, sorted source image positions",
        "values": values,
    }

