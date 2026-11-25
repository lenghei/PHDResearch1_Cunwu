"""Run a native-ten CUDA pair and preserve inputs for COCO replay."""
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from hashlib import sha256 as digest
import copy
import csv
import gzip
import importlib.metadata
import json
import math
import os
import statistics
import sys
import time

from common import (FMP, CHECKPOINT, VAL, TEST, fingerprint, file_id, import_file,
                    load, new_run, save, seal, sha256, source, verify_hashes)
from sampling import models, RENDER_SEED

PROTOCOL = 'cv261207_pr01_native10_nms_fp32_or_declared_fp16_v1'
REQUIRED = ('run.json', 'configuration.json', 'environment.json', 'predictions.jsonl.gz',
            'ground_truth.json', 'metrics.json', 'timing.json', 'image_manifest.jsonl',
            'verification.json', 'index.html')


def evaluate_saved(out):
    from main.labels import mapped_predictions
    from main.statistics import evaluate_coco

    payload = load(out / 'ground_truth.json')
    records = payload['records']
    regions = {r['id']: r['ignore_regions'] + r['other_regions'] for r in records}
    raw = []
    ids = []
    with gzip.open(out / 'predictions.jsonl.gz', 'rt', encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            ids.append(row['image_id'])
            if any(p['image_id'] != row['image_id'] for p in row['predictions']):
                raise ValueError('Prediction record image mismatch.')
            raw.extend(row['predictions'])
    if ids != [r['id'] for r in records] or len(set(ids)) != len(ids):
        raise ValueError('Prediction/image membership differs from GT.')
    pred = mapped_predictions(raw, records, regions, 'native10', 'finetuned', 3000)
    areas, classes, evaluator = evaluate_coco(payload['coco'], pred, max_det=3000)
    return {'per_area': areas, 'per_class': classes, 'raw_count': len(raw),
            'mapped_count': len(pred), 'images': len(ids)}, evaluator


def verify(out, regenerate=False):
    checks = verify_hashes(out, REQUIRED)
    run = load(out / 'run.json')
    configuration = load(out / 'configuration.json')
    metrics = load(out / 'metrics.json')
    if (run['status'] != 'PASS' or run['RunID'] != out.name
            or configuration['protocol'] != PROTOCOL or metrics['images'] != configuration['images']
            or load(out / 'verification.json')['metrics_sha256'] != checks['metrics.json']):
        raise ValueError('Run identity or replay evidence mismatch.')
    if regenerate and evaluate_saved(out)[0] != metrics:
        raise ValueError('Regenerated metrics differ.')
    return run


def bootstrap(evaluator, count=200, seed=20261207):
    import numpy as np
    from pycocotools.cocoeval import COCOeval
    from main.statistics import _metric_rows

    params = evaluator._paramsEval
    n = len(params.imgIds)
    groups = len(params.catIds) * len(params.areaRng)
    ids = np.random.default_rng(seed).integers(n, size=(count, n))
    draws = {name: [] for name in ('all', 'tiny', 'small', 'medium', 'large')}
    for sample in ids:
        resampled = COCOeval(iouType='bbox')
        resampled.params = copy.deepcopy(params)
        resampled.params.imgIds = list(range(n))
        resampled._paramsEval = copy.deepcopy(resampled.params)
        resampled.evalImgs = [evaluator.evalImgs[g*n+int(i)] for g in range(groups) for i in sample]
        with redirect_stdout(StringIO()):
            resampled.accumulate()
        for row in _metric_rows(resampled, include_per_class=False, include_gt_counts=False)[0]:
            if row['size'] in draws:
                draws[row['size']].append(row['map50_95'])
    intervals = {}
    for size, rows in draws.items():
        values = [v for v in rows if v is not None]
        intervals[size] = {'ci95': list(map(float, np.percentile(values, [2.5, 97.5]))) if values else None,
                           'n_valid': len(values), 'draws': rows}
    return {'unit': 'image', 'metric': 'AP50:95', 'seed': seed, 'resamples': count,
            'image_ids': list(params.imgIds), 'intervals': intervals}


def runtime(out):
    private = out / 'runtime'
    for key in ('YOLO_CONFIG_DIR', 'MPLCONFIGDIR', 'TORCH_HOME', 'TMP', 'TEMP'):
        directory = private / key
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(directory)
    (Path(os.environ['YOLO_CONFIG_DIR']) / 'Ultralytics').mkdir(exist_ok=True)
    os.environ.update(YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true', PYTHONDONTWRITEBYTECODE='1')


class Device:
    def __init__(self, expected_sha):
        import torch
        from ultralytics import YOLO
        from main.labels import VIS_NAMES
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required for GPU execution.')
        free, total = torch.cuda.mem_get_info(0)
        if free < 3 * 1024**3:
            raise RuntimeError('Less than 3 GiB of free GPU memory.')
        if sha256(CHECKPOINT) != expected_sha:
            raise ValueError('Checkpoint SHA mismatch.')
        self.torch, self.free, self.total = torch, free, total
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = YOLO(str(CHECKPOINT))
        net = self.model.model
        small = net.yaml.get('scale') == 's' or (net.yaml.get('depth_multiple'), net.yaml.get('width_multiple')) == (.33, .5)
        if list(self.model.names.values()) != VIS_NAMES or not small:
            raise ValueError('Expected YOLOv8s and native-ten class order.')
        if not any(type(m).__name__ == 'C2f' for m in net.modules()):
            raise ValueError('YOLOv8 architecture check failed.')
        for name, tensor in net.state_dict().items():
            if tensor.is_floating_point() and (not torch.isfinite(tensor).all()
                    or 'running_var' in name and (tensor < 0).any()):
                raise ValueError('Unhealthy checkpoint tensor: ' + name)
        self.params = sum(p.numel() for p in net.parameters())
        self.checkpoint = source(CHECKPOINT)
        self.predictor_key = None
        self.pristine = copy.deepcopy(net)
        self.flops = {}

    def detector(self, spec):
        from ultralytics.utils.torch_utils import get_flops
        from main.predictor import IEEEDetectionPredictor

        class Traced(IEEEDetectionPredictor):
            def preprocess(self, images):
                tensor = super().preprocess(images)
                self.trace = {'NCHW': list(tensor.shape), 'dtype': str(tensor.dtype)}
                return tensor

        size, precision = spec['resolution'], spec['precision']
        if size not in (640, 960, 1280) or precision not in ('FP32', 'FP16'):
            raise ValueError('Unsupported detector resolution or precision.')
        key = size, precision
        if self.predictor_key != key:
            self.model.predictor = None
            self.model.model = copy.deepcopy(self.pristine)
            self.predictor_key = key
        if size not in self.flops:
            try:
                flops = float(get_flops(self.pristine, imgsz=size))
                self.flops[size] = flops if math.isfinite(flops) and flops > 0 else None
            except Exception:
                self.flops[size] = None
        kwargs = dict(predictor=Traced, imgsz=size, batch=1, device='0', half=precision=='FP16',
                      conf=.001, iou=.7, max_det=3000, end2end=False, rect=False,
                      agnostic_nms=False, classes=None, augment=False, save=False, verbose=False)

        def predict(image):
            with self.torch.inference_mode():
                results = self.model.predict(source=image, **kwargs)
            trace = self.model.predictor.trace
            if (trace['NCHW'] != [1, 3, size, size] or
                    trace['dtype'] != ('torch.float16' if precision == 'FP16' else 'torch.float32')
                    or len(results) != 1 or results[0].orig_shape != image.shape[:2]):
                raise ValueError('Inference tensor/geometry mismatch.')
            return results[0], trace
        return predict


def pair(out, spec, archive_path, indices, device, formal=False):
    import numpy as np
    from archives import OriginalZIP
    from main.labels import ground_truth, mapped_predictions
    from main.statistics import evaluate_coco
    from visualization import write_preview

    start = time.perf_counter()
    p, d = spec['degradation'], spec['detector']
    fmp, _, _ = models()
    leaves = p['leaves']
    if (not fmp.valid_leaves(leaves) or 'RollingShutter' in leaves
            or p['degradation_id'] != file_id(leaves, 'p_')
            or d['detector_id'] != file_id(d['leaves'], 'd_')
            or d['checkpoint_sha256'] != device.checkpoint['sha256']):
        raise ValueError('Pair binding or geometry invalid.')
    sys.path.insert(0, str(FMP.parent))
    try:
        generator = import_file('cv261109_generator', FMP.with_name('generator.py'))
    finally:
        sys.path.remove(str(FMP.parent))
    attrs = fmp.read_uvl_attributes(FMP)
    torch = device.torch
    cfg = dict(spec, protocol=PROTOCOL, images=len(indices), split=archive_path.stem, indices=indices,
               dataset_sha256=sha256(archive_path), render_seed=RENDER_SEED, code_sha256=fingerprint(),
               batch=1, model_hash=d['checkpoint_sha256'])
    save(out / 'configuration.json', cfg)
    save(out / 'environment.json', {'gpu': torch.cuda.get_device_name(0), 'free_bytes_at_start': device.free,
                                     'total_bytes': device.total, 'torch': torch.__version__,
                                     'cuda': torch.version.cuda, 'python': sys.version,
                                     'packages': {p: importlib.metadata.version(p) for p in
                                                  ('ultralytics', 'numpy', 'opencv-python', 'pycocotools')},
                                     'parameters': device.params, 'checkpoint': device.checkpoint})
    predict = device.detector(d)
    records, raw, times = [], [], []
    archive = OriginalZIP(archive_path, 1610 if formal else 548)
    try:
        img, rec, _ = archive.record(indices[0])
        warm, _ = generator.generate(img, leaves, attrs, rec['stem'], RENDER_SEED)
        for _ in range(3):
            predict(warm)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        prediction_tmp = out / 'predictions.jsonl.gz.partial'
        manifest_tmp = out / 'image_manifest.jsonl.partial'
        with gzip.open(prediction_tmp, 'xt', encoding='utf-8') as stream, manifest_tmp.open('x', encoding='utf-8') as manifest:
            for i in indices:
                begin = time.perf_counter()
                img, rec, provenance = archive.record(i)
                rendered, effects = generator.generate(img, leaves, attrs, rec['stem'], RENDER_SEED)
                torch.cuda.synchronize()
                t = time.perf_counter()
                result, tensor = predict(rendered)
                torch.cuda.synchronize()
                predicted_at = time.perf_counter()
                data = result.boxes.data.detach().cpu().numpy()
                if not np.isfinite(data).all():
                    raise ValueError('Nonfinite prediction.')
                observed = []
                for x1, y1, x2, y2, score, cat in data:
                    if int(cat) != cat or not 0 <= cat < 10 or not 0 <= score <= 1:
                        raise ValueError('Invalid prediction class/score.')
                    if x2 > x1 and y2 > y1:
                        observed.append({'image_id': rec['id'], 'class_id': int(cat),
                                         'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                                         'score': float(score)})
                stream.write(json.dumps({'image_id': rec['id'], 'predictions': observed}, allow_nan=False) + '\n')
                stream.flush()
                trace = dict(provenance, stem=rec['stem'], degraded_sha256=digest(rendered.tobytes()).hexdigest(),
                             effects=effects, tensor=tensor)
                manifest.write(json.dumps(trace, allow_nan=False) + '\n')
                manifest.flush()
                records.append(rec)
                raw.extend(observed)
                times.append({'image_id': rec['id'], 'predict_pipeline_ms': (predicted_at-t)*1000,
                              'network_forward_ms': result.speed['inference'],
                              'decode_and_degrade_ms': (t-begin)*1000,
                              'end_to_end_ms': (time.perf_counter()-begin)*1000})
                if len(records) == 1:
                    write_preview(out, rendered, rec, observed)
        prediction_tmp.rename(out / 'predictions.jsonl.gz')
        manifest_tmp.rename(out / 'image_manifest.jsonl')
    finally:
        archive.close()
    allocated, reserved = torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()
    gt, regions = ground_truth(records, 'native10')
    mapped = mapped_predictions(raw, records, regions, 'native10', 'finetuned', 3000)
    save(out / 'ground_truth.json', {'coco': gt, 'records': records})
    start_eval = time.perf_counter()
    areas, classes, evaluator = evaluate_coco(gt, mapped, 3000)
    metrics = {'per_area': areas, 'per_class': classes, 'raw_count': len(raw),
               'mapped_count': len(mapped), 'images': len(records)}
    cpu_eval = time.perf_counter()-start_eval
    save(out / 'metrics.json', metrics)
    if evaluate_saved(out)[0] != metrics:
        raise ValueError('COCO replay mismatch.')
    if formal:
        save(out / 'bootstrap.json', bootstrap(evaluator))
    save(out / 'timing.json', {'samples': times, 'cpu_evaluation_seconds': cpu_eval,
                              'pair_wall_seconds': time.perf_counter()-start,
                              'mean_pipeline_ms': statistics.mean(r['end_to_end_ms'] for r in times),
                              'peak_allocated_bytes': allocated, 'peak_reserved_bytes': reserved,
                              'warmups': 3, 'gflops_estimate': device.flops[d['resolution']],
                              'forward_clock': 'Ultralytics synchronized inference profiler',
                              'pipeline_clock': 'perf_counter; includes decode, degradation, inference and raw output writes'})
    save(out / 'verification.json', {'status': 'PASS', 'COCO_recomputed_from_raw': True,
                                     'metrics_sha256': sha256(out / 'metrics.json')})
    run = {'RunID': out.name, 'ExperimentID': 'cv261207', 'status': 'PASS',
           'mode': 'FORMAL' if formal else 'SMOKE', 'pair_id': spec['pair_id'],
           'images': len(records), 'scope': 'TEST_DEV' if formal else 'VAL_ENGINEERING_ONLY'}
    save(out / 'run.json', run)
    seal(out)
    print(f"[CV261207] {run['scope']} PASS | {run['images']} images | {out.name}", flush=True)
    return run
