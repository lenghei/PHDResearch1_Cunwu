"""CPU-tested timing boundary; detector and synchronization are injected."""
import hashlib
import json
import time
from pathlib import Path


def cached_image(path, expected_file_sha, expected_pixel_sha, predict, extract,
                 synchronize, prediction_stream, manifest_stream, image_id):
    import cv2
    import numpy as np
    started=time.perf_counter();phases={}
    path=Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest()!=expected_file_sha:
        raise ValueError('Cached file SHA differs.')
    phases['cache_integrity_ms']=(time.perf_counter()-started)*1000
    tick=time.perf_counter()
    image=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
    if image is None or hashlib.sha256(image.tobytes()).hexdigest()!=expected_pixel_sha:
        raise ValueError('Decoded pixels differ.')
    phases['read_decode_pixel_integrity_ms']=(time.perf_counter()-tick)*1000
    synchronize();tick=time.perf_counter()
    result,tensor=predict(image);synchronize()
    phases['predict_ms']=(time.perf_counter()-tick)*1000
    tick=time.perf_counter();predictions=extract(result,image_id)
    for row in predictions:
        if not np.isfinite([*row['bbox'],row['score']]).all():raise ValueError('Nonfinite output.')
        if row['bbox'][2]<=0 or row['bbox'][3]<=0 or not 0<=row['score']<=1:
            raise ValueError('Invalid output box or score.')
        if not isinstance(row['class_id'],int) or not 0<=row['class_id']<10:raise ValueError('Invalid native10 class.')
    phases['extract_validate_ms']=(time.perf_counter()-tick)*1000
    tick=time.perf_counter()
    prediction_stream.write(json.dumps({'image_id':image_id,'predictions':predictions})+'\n');prediction_stream.flush()
    manifest_stream.write(json.dumps({'image_id':image_id,'file_SHA256':expected_file_sha,
        'pixel_SHA256':expected_pixel_sha,'tensor':tensor})+'\n');manifest_stream.flush()
    phases['serialize_write_flush_ms']=(time.perf_counter()-tick)*1000
    phases['cached_pipeline_ms']=(time.perf_counter()-started)*1000
    return phases
