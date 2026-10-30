"""Ultralytics predictor with fixed evaluation NMS."""
from __future__ import annotations

from ultralytics.cfg import DEFAULT_CFG
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.utils import nms, ops


class IEEEDetectionPredictor(DetectionPredictor):
    """Square Letterbox with class-aware, multi-label NMS.

    COCO common-six IDs: [0, 1, 2, 3, 5, 7]. Keep all native VisDrone IDs
    for native10 evaluation. Use end2end=False for YOLO26 one-to-many inference."""

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        overrides = dict(overrides or {})
        overrides["rect"] = False
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

    def setup_model(self, model, verbose: bool = True):
        super().setup_model(model=model, verbose=verbose)
        if self.args.end2end is False and getattr(self.model, "end2end", False):
            raise RuntimeError("Requested one-to-many inference but the loaded backend is still end-to-end")

    def pre_transform(self, im):
        # Preserve square input when predictor arguments are reused.
        self.args.rect = False
        return super().pre_transform(im)

    def postprocess(self, preds, img, orig_imgs, **kwargs):
        if self.args.task != "detect":
            raise ValueError("IEEEDetectionPredictor supports axis-aligned detection only")
        if getattr(self, "_feats", None) is not None:
            raise ValueError("Feature embedding output is not part of the evaluation protocol")
        detections = nms.non_max_suppression(
            preds,
            conf_thres=self.args.conf,
            iou_thres=self.args.iou,
            classes=self.args.classes,
            agnostic=self.args.agnostic_nms,
            multi_label=True,
            max_det=self.args.max_det,
            nc=len(self.model.names),
            max_nms=30000,
            # Disable the NMS deadline so machine load cannot truncate predictions.
            max_time_img=float("inf"),
            end2end=getattr(self.model, "end2end", False),
            rotated=False,
        )
        if not isinstance(orig_imgs, list):
            orig_imgs = ops.convert_torch2numpy_batch(orig_imgs)[..., ::-1]
        return self.construct_results(detections, img, orig_imgs, **kwargs)
