"""Measure an existing checkpoint on original VisDrone images; no training."""
from __future__ import annotations
import csv
import hashlib
import math
import platform
import statistics
import time
import zipfile
from pathlib import Path
from source_audit import audit_training
from model import sha256


def load_original_images(archive_path: Path, count: int):
    import cv2
    import numpy as np
    with zipfile.ZipFile(archive_path) as archive:
        names = sorted(p for p in archive.namelist()
                       if '/images/' in p and p.lower().endswith(('.jpg', '.jpeg', '.png'))
                       and '__MACOSX/' not in p)
        if len(names) != 548:
            raise ValueError('Expected 548 VisDrone validation images.')
        if count > len(names) or count < 1:
            raise ValueError('Invalid image sample count.')
        # Spread the sample deterministically over the full validation split.
        positions = [i * len(names) // count for i in range(count)]
        decoded = []
        for index in positions:
            data = archive.read(names[index])
            frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8),
                                 cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if frame is None:
                raise ValueError('Cannot decode source frame: ' + names[index])
            decoded.append((names[index], hashlib.sha256(data).hexdigest(), frame))
    return decoded


def run_profile(checkpoint, validation_zip, out, opts):
    import torch
    import ultralytics
    from ultralytics import YOLO
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU required for physical measurements.')
    gpu_index = int(opts.device)
    if gpu_index < 0 or gpu_index >= torch.cuda.device_count():
        raise ValueError('Unavailable GPU index.')
    if opts.precision not in ('FP32', 'FP16'):
        raise ValueError('INT8 needs a calibrated engine. It is not emulated by FP16.')
    if opts.strategy != 'FullFrame' or opts.preprocessing != 'None' or opts.tta:
        raise ValueError('Profile only implemented pipeline combinations. Others are NOT_MEASURED.')
    if opts.postprocessing != 'NMS':
        raise ValueError('This profile measures only the saved NMS pipeline.')
    if opts.sample_count < 1 or opts.sample_count > 548 or opts.repeat < 1 or opts.warmup < 1:
        raise ValueError('Invalid measurement repetitions.')
    checkpoint = Path(checkpoint).resolve(strict=True)
    validation_zip = Path(validation_zip).resolve(strict=True)
    model = YOLO(str(checkpoint))
    parameters = sum(p.numel() for p in model.model.parameters())
    classes = getattr(model.model, 'names', None)
    class_count = len(classes) if isinstance(classes, (dict, list, tuple)) else None
    expected = 10 if opts.weights_kind == 'VisDrone_FT' else 80
    if class_count != expected:
        raise ValueError(f'Checkpoint has {class_count} classes; declared weights require {expected}.')
    if opts.train_args:
        import yaml
        training = yaml.safe_load(Path(opts.train_args).resolve(strict=True).read_text(encoding='utf-8'))
        expected_res = int(opts.train_resolution.removeprefix('Train'))
        if not isinstance(training, dict) or int(training.get('imgsz', -1)) != expected_res:
            raise ValueError('Saved training resolution conflicts with declared FM_D training feature.')
    if not parameters:
        raise ValueError('Loaded checkpoint has no parameters.')
    images = load_original_images(validation_zip, opts.sample_count)
    torch.cuda.set_device(gpu_index)
    kwargs = {'imgsz': opts.imgsz, 'device': opts.device, 'half': opts.precision == 'FP16',
              'batch': 1, 'conf': .001, 'iou': .7, 'max_det': 3000,
              'augment': False, 'save': False, 'verbose': False}
    def measured_predict(image):
        return model.predict(source=image, **kwargs)
    try:
        for _ in range(opts.warmup):
            measured_predict(images[0][2])
        torch.cuda.synchronize()
    except Exception as exc:
        raise RuntimeError('GPU warmup failed. Do not record partial latency.') from exc
    torch.cuda.reset_peak_memory_stats(gpu_index)
    rows = []
    for rep in range(opts.repeat):
        for member, source_hash, image in images:
            torch.cuda.synchronize()
            start = time.perf_counter()
            cuda_start, cuda_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            cuda_start.record()
            results = measured_predict(image)
            cuda_end.record()
            torch.cuda.synchronize()
            duration_ms = (time.perf_counter() - start) * 1000
            cuda_ms = cuda_start.elapsed_time(cuda_end)
            if not results or len(results) != 1:
                raise RuntimeError('Prediction did not return one result.')
            rows.append({'repeat': rep, 'member': member, 'source_image_SHA256': source_hash,
                         'wall_ms': round(duration_ms, 6), 'CUDA_event_ms': round(cuda_ms, 6),
                         'detections': len(results[0].boxes) if results[0].boxes is not None else 0})
    if len(rows) != opts.repeat * opts.sample_count:
        raise RuntimeError('Recorded timing sample count mismatch.')
    wall = [r['wall_ms'] for r in rows]
    cuda = [r['CUDA_event_ms'] for r in rows]
    peak_bytes = torch.cuda.max_memory_allocated(gpu_index)
    reserved_bytes = torch.cuda.max_memory_reserved(gpu_index)
    # GFLOPs uses a static model computation estimate, not a GPU measurement.
    flops = None
    flops_error = None
    try:
        from ultralytics.utils.torch_utils import get_flops
        flops = float(get_flops(model.model, imgsz=opts.imgsz))
        if not math.isfinite(flops) or flops <= 0:
            flops_error = 'Nonfinite or nonpositive GFLOPs estimate'
            flops = None
    except Exception as exc:
        flops_error = type(exc).__name__
    with (out / 'measurements.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {'status': 'PASS', 'scope': 'INFERENCE_LATENCY_PILOT_ORIGINAL_VISDRONE_VAL_NO_AP',
        'checkpoint_SHA256': sha256(checkpoint), 'source_ZIP_SHA256': sha256(validation_zip),
        'checkpoint_parameters': parameters, 'checkpoint_class_count': class_count,
        'checkpoint_model_yaml': {k: str(v) for k,v in getattr(model.model, 'yaml', {}).items() if k in ('yaml_file', 'scale')},
        'GFLOPs_estimate': flops,
        'GFLOPs_estimate_note': flops_error or 'Ultralytics estimate, not GPU measurement',
        'latency_ms_per_image_mean': statistics.mean(wall),
        'latency_ms_per_image_median': statistics.median(wall),
        'latency_ms_per_image_p95': sorted(wall)[min(len(wall)-1, math.ceil(.95*len(wall))-1)],
        'CUDA_event_ms_mean': statistics.mean(cuda),
        'peak_pytorch_allocated_bytes': peak_bytes,
        'peak_pytorch_reserved_bytes': reserved_bytes,
        'memory_definition': 'PyTorch CUDA allocator peak, not total GPU memory',
        'warmup_iterations': opts.warmup, 'measured_samples': len(rows),
        'source_image_count': len(images), 'repeat': opts.repeat,
        'imgsz': opts.imgsz, 'precision': opts.precision,
        'strategy': opts.strategy, 'postprocessing': opts.postprocessing,
        'augmentation': False, 'gpu_name': torch.cuda.get_device_name(gpu_index),
        'device_index': gpu_index, 'model_inference': 'RUN', 'AP_metrics': 'NOT_EVALUATED',
        'software': {'torch':torch.__version__, 'ultralytics':ultralytics.__version__,
                     'platform':platform.platform()}}
    return summary
