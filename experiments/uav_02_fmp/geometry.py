"""Annotation adapter for the historical sinusoidal RollingShutter renderer."""
import math

import cv2
import numpy as np

from main.labels import ground_truth


AREA_REFERENCE = "rolling_shutter_source_bbox_pixels_v1"


class RollingRaster:
    def __init__(self, shape, effect):
        height, width = shape[:2]
        if not (1 <= height < 32767 and 1 <= width < 32767):
            raise ValueError("Image exceeds the supported OpenCV remap dimensions.")
        if effect["feature"] != "RollingShutter":
            raise ValueError("Expected the recorded RollingShutter effect.")
        params = effect["params"]
        phase = float(effect["phase_rad"])
        fraction, waves = float(params["max_shift_fraction"]), int(params["waves"])
        if not 0 <= fraction < 1 or waves < 1 or not math.isfinite(phase):
            raise ValueError("Invalid rolling-shutter parameters.")
        self.height, self.width = height, width
        rows = np.arange(height, dtype=np.float32)
        shift = width * fraction * np.sin((rows / max(1, height-1)) * 2 * np.pi * waves + phase)
        mx = (np.arange(width, dtype=np.float32)[None, :] - shift[:, None]).astype(np.float32)
        my = np.broadcast_to(rows[:, None], (height, width)).copy()
        self.maps, self.fractions = cv2.convertMaps(mx, my, cv2.CV_16SC2)
        self.left = self.reflect(self.maps[:, :, 0].astype(np.int32))
        self.right = self.reflect(self.maps[:, :, 0].astype(np.int32) + 1)
        self.has_right = self.fractions % cv2.INTER_TAB_SIZE != 0
        self.margin = math.ceil(width * fraction) + 2

    def reflect(self, columns):
        if self.width == 1:
            return np.zeros_like(columns)
        edge = self.width - 1
        return edge - np.abs((columns % (2*edge)) - edge)

    def support(self, box):
        x, y, w, h = map(float, box)
        if not all(map(math.isfinite, (x, y, w, h))) or min(w, h) <= 0:
            raise ValueError("Invalid xywh annotation.")
        x0, y0 = max(0, math.floor(x)), max(0, math.floor(y))
        x1, y1 = min(self.width, math.ceil(x+w)), min(self.height, math.ceil(y+h))
        if x0 >= x1 or y0 >= y1:
            return None
        start, end = max(0, x0-self.margin), min(self.width, x1+self.margin)
        roi = np.s_[y0:y1, start:end]
        left, right = self.left[roi], self.right[roi]
        mask = ((left >= x0) & (left < x1)) | (
            self.has_right[roi] & (right >= x0) & (right < x1))
        return mask, start, y0

    def bbox(self, box):
        support = self.support(box)
        if support is None:
            return None
        mask, start, top = support
        y, x = np.nonzero(mask)
        if not len(x):
            return None
        return [int(x.min())+start, int(y.min())+top,
                int(x.max()-x.min()+1), int(y.max()-y.min()+1)]

    def warp(self, image):
        return cv2.remap(image, self.maps, self.fractions, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    def regions(self, boxes):
        if not boxes:
            return []
        union = np.zeros((self.height, self.width), dtype=bool)
        for box in boxes:
            support = self.support(box)
            if support is not None:
                mask, x, y = support
                union[y:y+mask.shape[0], x:x+mask.shape[1]] |= mask
        padded = np.pad(union, ((0, 0), (1, 1)))
        change_y, change_x = np.nonzero(padded[:, :1+self.width] != padded[:, 1:])
        return [[int(x), int(y), int(end-x), 1] for y, x, end in
                zip(change_y[::2], change_x[::2], change_x[1::2], strict=True)]


def transform_annotations(record, effect):
    """Return original COCO GT, transformed GT and derived native10 records."""
    original, _ = ground_truth([record], "native10")
    raster = RollingRaster((record["height"], record["width"]), effect)
    transformed, objects = [], []
    # Apply the established GT ignore policy in source coordinates.
    for ann in original["annotations"]:
        warped = raster.bbox(ann["bbox"])
        if warped is None:
            continue
        transformed.append({**ann, "bbox": warped, "source_bbox": list(ann["bbox"])})
        objects.append({"class_id": ann["category_id"]-1, "bbox": warped})
    gt = {**original, "info": {**original["info"], "area_reference": AREA_REFERENCE},
          "annotations": transformed}
    derived = {**record, "objects": objects,
               "ignore_regions": raster.regions(record["ignore_regions"]),
               "other_regions": raster.regions(record["other_regions"])}
    return original, gt, derived
