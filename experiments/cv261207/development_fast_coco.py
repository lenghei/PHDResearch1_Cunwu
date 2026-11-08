"""Exact, vectorized COCO accumulation for aligned cached bbox evaluations.

Uses the same mergesort, float64 sums, 101-point recall interpolation and
ignore masks as pycocotools.cocoeval.COCOeval.accumulate. Only the Python
precision-envelope loop is vectorized. Does not approximate AP or resample
individual detections. Unsupported parameter subsets use the official path.

Algorithm reference: cocodataset/cocoapi, PythonAPI/pycocotools/cocoeval.py
(BSD license, Piotr Dollar and Tsung-Yi Lin). No installed package is modified.
"""
from datetime import datetime
import numpy as np


def accumulate_fast(evaluator):
    p, original = evaluator.params, evaluator._paramsEval
    if (not p.useCats or p.catIds != original.catIds or
            p.imgIds != original.imgIds or p.areaRng != original.areaRng or
            p.maxDets != original.maxDets):
        evaluator.accumulate()
        return
    T, R, K, A, M = (len(p.iouThrs), len(p.recThrs), len(p.catIds),
                      len(p.areaRng), len(p.maxDets))
    precision = -np.ones((T, R, K, A, M))
    recall = -np.ones((T, K, A, M))
    scores = -np.ones((T, R, K, A, M))
    n_images = len(p.imgIds)
    for k in range(K):
        for a in range(A):
            offset = (k*A + a)*n_images
            entries = [e for e in evaluator.evalImgs[offset:offset+n_images] if e is not None]
            if not entries:
                continue
            n_gt = sum(int(np.count_nonzero(e['gtIgnore'] == 0)) for e in entries)
            if n_gt == 0:
                continue
            for m, cap in enumerate(p.maxDets):
                dt_scores = np.concatenate([e['dtScores'][:cap] for e in entries])
                order = np.argsort(-dt_scores, kind='mergesort')
                sorted_scores = dt_scores[order]
                # Convert to bool BEFORE concatenation to avoid large float64 match copies.
                matched = np.concatenate([e['dtMatches'][:, :cap] != 0 for e in entries], axis=1)[:, order]
                ignored = np.concatenate([e['dtIgnore'][:, :cap] for e in entries], axis=1)[:, order]
                tp_sum = np.cumsum(matched & ~ignored, axis=1).astype(float)
                fp_sum = np.cumsum(~matched & ~ignored, axis=1).astype(float)
                for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
                    nd = len(tp)
                    rc = tp / n_gt
                    pr = tp / (fp + tp + np.spacing(1))
                    recall[t, k, a, m] = rc[-1] if nd else 0
                    q, ss = np.zeros(R), np.zeros(R)
                    if nd:
                        pr = np.maximum.accumulate(pr[::-1])[::-1]
                        ix = np.searchsorted(rc, p.recThrs, side='left')
                        valid = ix < nd
                        q[valid] = pr[ix[valid]]
                        ss[valid] = sorted_scores[ix[valid]]
                    precision[t, :, k, a, m] = q
                    scores[t, :, k, a, m] = ss
    evaluator.eval = {'params': p, 'counts': [T, R, K, A, M],
                      'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                      'precision': precision, 'recall': recall, 'scores': scores}
