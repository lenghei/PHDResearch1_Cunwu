"""Evaluate the frozen image-bootstrap draws concurrently in input order."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from functools import partial

import numpy as np
import psutil
from pycocotools.cocoeval import COCOeval

from main.statistics import METRIC_NAMES, _metric_rows, bootstrap_coco


def evaluate_draw(evaluator, indices):
    from new_fast_coco_accumulate import accumulate_fast

    sample = COCOeval(iouType="bbox")
    sample.params = deepcopy(evaluator._paramsEval)
    sample.params.imgIds = list(range(1,len(indices)+1))
    sample._paramsEval = deepcopy(sample.params)
    original_size = len(evaluator._paramsEval.imgIds)
    blocks = len(sample.params.catIds)*len(sample.params.areaRng)
    sample.evalImgs = [evaluator.evalImgs[b*original_size+int(i)] for b in range(blocks) for i in indices]
    sample._new_statistics_metadata = evaluator._new_statistics_metadata
    accumulate_fast(sample)
    rows, _ = _metric_rows(sample,include_per_class=False,include_gt_counts=False)
    return rows


def parallel_bootstrap(evaluator, samples=200, seed=42, progress_callback=None):
    if psutil.virtual_memory().available<2*2**30:
        return bootstrap_coco(evaluator,samples,seed,progress_callback)
    if isinstance(samples,bool) or not isinstance(samples,int) or samples<1:
        raise ValueError("samples must be a positive integer.")
    image_ids = [int(i) for i in evaluator._paramsEval.imgIds]
    if image_ids!=sorted(set(image_ids)):
        raise ValueError("Bootstrap requires unique, sorted source image IDs.")

    estimates, _ = _metric_rows(evaluator,include_per_class=False)
    values = {row["size"]:{k:[] for k in METRIC_NAMES} for row in estimates}
    rng = np.random.default_rng(seed)
    indices = [rng.integers(0,len(image_ids),size=len(image_ids)) for _ in range(samples)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for completed, rows in enumerate(pool.map(partial(evaluate_draw,evaluator),indices),1):
            for row in rows:
                for key in METRIC_NAMES:
                    values[row["size"]][key].append(row[key])
            if progress_callback is not None and (completed%25==0 or completed==samples):
                progress_callback(completed,samples)

    intervals = []
    for estimate in estimates:
        for key in METRIC_NAMES:
            valid = np.asarray([v for v in values[estimate["size"]][key] if v is not None],dtype=float)
            intervals.append({
                "size":estimate["size"], "metric":key, "estimate":estimate[key],
                "bootstrap_mean":float(valid.mean()) if valid.size else None,
                "ci95":np.percentile(valid,[2.5,97.5]).tolist() if valid.size else [None,None],
                "n_valid":int(valid.size), "n_samples":samples, "seed":seed,
                "sampling_unit":"image", "method":"paired_percentile_bootstrap",
            })
    return intervals, {
        "image_ids":image_ids, "samples":samples, "seed":seed,
        "sampling":"numpy.default_rng.integers, sorted source image positions",
        "values":values,
    }
