"""Prune detections that cannot match before calling the COCO matcher."""
import numpy as np
from pycocotools.cocoeval import COCOeval


class SparseMatchesCOCOeval(COCOeval):
    def evaluate(self):
        self._ranked = {}
        try:
            return super().evaluate()
        finally:
            self._ranked.clear()

    def evaluateImg(self, img_id, cat_id, area_range, max_det):
        if not self.params.useCats or self.params.iouType != "bbox":
            return super().evaluateImg(img_id, cat_id, area_range, max_det)
        key = (img_id, cat_id)
        ious = self.ious[key]
        if not len(ious):
            return super().evaluateImg(img_id, cat_id, area_range, max_det)
        cache_key = (*key, max_det)
        if cache_key not in self._ranked:
            threshold = min(self.params.iouThrs.min(), 1 - 1e-10)
            columns = ious[:max_det]
            possible = np.flatnonzero(np.any(columns >= threshold, axis=1))
            if len(possible) == len(columns) or not np.isfinite(columns).all():
                self._ranked[cache_key] = None
            else:
                dt = self._dts[key]
                order = np.argsort([-d["score"] for d in dt], kind="mergesort")[:max_det]
                ordered = [dt[i] for i in order]
                self._ranked[cache_key] = (
                    [ordered[i] for i in possible], possible,
                    [d["id"] for d in ordered], [d["score"] for d in ordered],
                    np.asarray([d["area"] for d in ordered]), columns[possible],
                )
        ranks = self._ranked[cache_key]
        if ranks is None:
            return super().evaluateImg(img_id, cat_id, area_range, max_det)
        candidates, positions, dt_ids, scores, areas, selected_ious = ranks
        original = self._dts[key]
        self._dts[key], self.ious[key] = candidates, selected_ious
        try:
            result = super().evaluateImg(img_id, cat_id, area_range, max_det)
        finally:
            self._dts[key], self.ious[key] = original, ious

        # Restore unmatched detections before COCO accumulates precision.
        n_iou = len(self.params.iouThrs)
        matches = np.zeros((n_iou, len(dt_ids)))
        matches[:, positions] = result["dtMatches"]
        ignores = np.broadcast_to(((areas < area_range[0]) | (areas > area_range[1])), matches.shape).copy()
        ignores[:, positions] = result["dtIgnore"]
        result.update(dtIds=dt_ids, dtScores=scores, dtMatches=matches, dtIgnore=ignores)
        return result


def with_sparse_matching(replay, pair):
    from main import statistics

    original = statistics.COCOeval
    statistics.COCOeval = SparseMatchesCOCOeval
    try:
        return replay(pair)
    finally:
        statistics.COCOeval = original

