"""Stable-release API adapter with the unchanged VM36 NMS calculation."""
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from .predictor import IEEEDetectionPredictor


class StableNMSPredictor(IEEEDetectionPredictor):
    def setup_model(self,model,verbose=True):
        if self.args.nms is False:raise ValueError('NMS-free output is outside this protocol.')
        DetectionPredictor.setup_model(self,model=model,verbose=verbose)
        if getattr(self.model,'end2end',False):raise RuntimeError('Expected the one-to-many inference head.')
        if self.args.quantize not in (None,32):raise ValueError('Research evaluation requires FP32.')
