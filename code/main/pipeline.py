"""YOLO26s validation with frozen UVL selections and pr01 evaluation."""
import datetime
import importlib.metadata
import json
import logging
import os
import platform
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

from .config import load_config, save, sha256

EXPERIMENT = 'vm1_flamapy_yolo26s_validation'
PUBLIC_ROOT = Path(__file__).resolve().parents[2]


def settings(path):
    cfg = load_config(path)
    if 'flamapy_python' not in cfg and 'uvl_python' in cfg:
        cfg['flamapy_python'] = cfg.pop('uvl_python')
    for name in ['flamapy_python', 'training_args', 'legacy_baseline']:
        if name in cfg:
            value = os.path.expandvars(cfg[name])
            if '$' in value or '%' in value:
                raise ValueError('Set the environment variable for ' + name)
            cfg[name] = str(Path(value).expanduser().resolve())
            if not Path(cfg[name]).exists():
                raise FileNotFoundError(name + ': ' + cfg[name])
    if cfg['checkpoint_id'] != 'ck01' or cfg['imgsz'] != 1280:
        raise ValueError('VM1 is frozen to ck01 at 1280 input.')
    return cfg


def new_run(cfg, experiment, mode):
    output_root = Path(cfg['output_root']).resolve()
    if output_root.is_relative_to(PUBLIC_ROOT):
        raise ValueError('Full artifacts must stay outside the publication root.')
    run_id = f"{experiment}_{mode}_{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}"
    out = output_root / run_id
    out.mkdir(parents=True, exist_ok=False)
    save(out / 'run.json', {'RunID': run_id, 'ExperimentID': experiment, 'mode': mode, 'status': 'RUNNING'})
    save(out / 'configuration.json', cfg)
    (out / 'command.txt').write_text(subprocess.list2cmdline([sys.executable, *sys.argv]), encoding='utf-8')
    versions = {}
    for name in ['torch', 'ultralytics', 'numpy', 'opencv-python', 'pycocotools']:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    save(out / 'environment.json', {'python': sys.version, 'interpreter': sys.executable,
                                    'platform': platform.platform(), 'packages': versions})
    save(out / 'source_SHA256.json', {p.relative_to(PUBLIC_ROOT).as_posix(): sha256(p)
                                     for d in ['code', 'experiments'] for p in sorted((PUBLIC_ROOT / d).rglob('*.py'))})
    logger = logging.getLogger(run_id)
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.addHandler(logging.FileHandler(out / 'execution.log', encoding='utf-8'))
    try:
        from .visualization import build_run_index
        build_run_index(output_root, experiment)
    except (OSError, ValueError) as exc:
        logger.warning('Result index update failed: %s', exc)
    return out, logger


def bridge(cfg, out):
    request = {'seed': cfg['sampling_seed'], 'resolution': cfg['imgsz']}
    save(out / 'bridge_request.json', request)
    entry = PUBLIC_ROOT / 'experiments' / EXPERIMENT / 'run.py'
    command = [cfg['flamapy_python'], '-B', str(entry), '--bridge-request', str(out / 'bridge_request.json'),
               '--output', str(out / 'bridge.json')]
    result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', timeout=120)
    (out / 'bridge.log').write_text(result.stdout + '\n' + result.stderr, encoding='utf-8')
    if result.returncode:
        raise RuntimeError('FlamaPy bridge failed; see bridge.log.')
    value = json.loads((out / 'bridge.json').read_text(encoding='utf-8'))
    folder = entry.parent
    if any(sha256(folder / name) != digest for name, digest in value['UVL_SHA256'].items()):
        raise ValueError('UVL source changed after selection.')
    return value


def check(cfg, out, logger):
    from .admission import checkpoint_gate, dataset_gate, verify_relocated_extracted
    logger.info('Checking relocated dataset...')
    dataset = dataset_gate(cfg)
    save(out / 'dataset_gate.json', dataset)
    save(out / 'extracted_membership.json', verify_relocated_extracted(cfg, dataset['manifest']))
    logger.info('Checking ck01 on CPU; no detector forward...')
    save(out / 'checkpoint_gate.json', checkpoint_gate(cfg))
    selection = bridge(cfg, out)
    logger.info('FlamaPy valid counts: %s / %s', selection['dataset']['valid_count'], selection['yolo']['valid_count'])
    command = [sys.executable, '-B', str(PUBLIC_ROOT / 'code/tools/check_project.py'),
               '--config', cfg['_config_file'], '--run-dir', cfg['legacy_baseline'],
               '--flamapy-python', cfg['flamapy_python'], '--require-flamapy', '--timeout', '600',
               '--output-dir', str(out / 'regression')]
    logger.info('Running CPU tests and historical ck05 metric regeneration...')
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1', YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true')
    with (out / 'regression.log').open('w', encoding='utf-8') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env,
                                timeout=cfg.get('CPU_budget_seconds', 900))
    if result.returncode:
        raise RuntimeError('CPU regression failed; see regression.log.')
    return {'status': 'PASS', 'admission': 'READY_FOR_MANUAL_RUN', 'images_verified': dataset['images'],
            'checkpoint': 'ck01', 'UVL_counts': [27, 9], 'GPU_inference': 'NOT_RUN',
            'demo': 'PREPARED', 'full_clean': 'PREPARED',
            'regeneration': 'PASS_ON_HISTORICAL_CK05; CK01_OUTPUT_NOT_YET_AVAILABLE'}


def predict(cfg, out, logger, mode):
    import cv2
    import numpy as np
    import torch
    from .admission import checkpoint_gate, dataset_gate
    from .dataset import VisDroneZIP
    from .degradation import degrade, field_for
    from .labels import ground_truth, mapped_predictions
    from .predictor import IEEEDetectionPredictor
    from .statistics import evaluate_coco

    gate = dataset_gate(cfg)
    save(out / 'dataset_gate.json', gate)
    health, detector = checkpoint_gate(cfg, return_detector=True)
    save(out / 'checkpoint_gate.json', health)
    selection = bridge(cfg, out)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; no automatic inference fallback.')
    torch.manual_seed(cfg['seed'])
    save(out / 'GPU_environment.json', {'torch': torch.__version__, 'CUDA': torch.version.cuda,
                                       'device': cfg['device'],
                                       'name': torch.cuda.get_device_name(int(cfg['device'])),
                                       'FP32': True})
    conditions = selection['conditions'] if mode == 'demo' else selection['conditions'][:1]
    count = 5 if mode == 'demo' else 548
    archive = VisDroneZIP(cfg['dataset_zip'], 548)
    aggregate = []
    started = time.perf_counter()

    class TracedPredictor(IEEEDetectionPredictor):
        def preprocess(self, images):
            tensor = super().preprocess(images)
            self.input_trace = {'NCHW': list(tensor.shape), 'dtype': str(tensor.dtype)}
            return tensor

    try:
        for condition in conditions:
            folder = out / condition['name']
            folder.mkdir()
            if mode == 'demo':
                (folder / 'images').mkdir()
            records, raw, trace = [], [], []
            raw_file = folder / 'predictions_raw.jsonl'
            raw_file.touch(exist_ok=False)
            model_seconds = 0.
            cuda_seconds = 0.
            torch.cuda.reset_peak_memory_stats()
            batch = cfg['batch']
            for first in range(0, count, batch):
                if time.perf_counter() - started > cfg['inference_budget_seconds']:
                    raise TimeoutError('Inference budget exceeded; partial files retained.')
                loaded = [archive.record(i) for i in range(first, min(first + batch, count))]
                raw_start = len(raw)
                pixels, proofs = [], []
                for image, record, proof in loaded:
                    if condition['indices'] == [0, 0, 0]:
                        transformed, recipe = image, {'identity': True, **condition['parameters']}
                    else:
                        transformed, recipe = degrade(image, condition['indices'], field_for(image, record['stem'], cfg['degradation_seed']))
                    if transformed.shape != image.shape or transformed.dtype != image.dtype:
                        raise ValueError('Degradation changed image geometry or dtype.')
                    proof = {**proof, 'configuration_id': condition['configuration_id'], 'recipe': recipe,
                             'output_pixel_SHA256': __import__('hashlib').sha256(transformed.tobytes()).hexdigest()}
                    if mode == 'demo':
                        name = f"images/{record['stem']}.png"
                        ok, encoded = cv2.imencode('.png', transformed)
                        if not ok:
                            raise RuntimeError('PNG encoding failed.')
                        (folder / name).write_bytes(encoded.tobytes())
                        proof.update(transformed_file=name, transformed_file_SHA256=sha256(folder / name))
                    pixels.append(transformed)
                    proofs.append(proof)
                torch.cuda.synchronize()
                tick = time.perf_counter()
                gpu_start, gpu_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                gpu_start.record()
                with torch.inference_mode():
                    results = detector.predict(pixels, predictor=TracedPredictor, imgsz=1280, batch=batch,
                                               device=cfg['device'], half=False, conf=.001, iou=.7,
                                               max_det=3000, end2end=False, rect=False, agnostic_nms=False,
                                               classes=None, augment=False, save=False, verbose=False)
                gpu_end.record()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - tick
                model_seconds += elapsed
                cuda_elapsed = gpu_start.elapsed_time(gpu_end) / 1000
                cuda_seconds += cuda_elapsed
                actual = detector.predictor.input_trace
                if actual != {'NCHW': [len(pixels), 3, 1280, 1280], 'dtype': 'torch.float32'}:
                    raise ValueError('Inference tensor protocol mismatch.')
                if len(results) != len(loaded):
                    raise ValueError('Prediction count differs from batch size.')
                for (_, record, _), proof, result in zip(loaded, proofs, results):
                    records.append(record)
                    values = result.boxes.data.detach().cpu().numpy()
                    if not np.isfinite(values).all():
                        raise ValueError('Nonfinite prediction.')
                    for x1, y1, x2, y2, score, category in values:
                        if x2 <= x1 or y2 <= y1:
                            continue
                        if category != int(category) or not 0 <= category < 10 or not 0 <= score <= 1:
                            raise ValueError('Invalid prediction class or score.')
                        raw.append({'image_id': record['id'], 'class_id': int(category),
                                    'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)], 'score': float(score)})
                    trace.append({**proof, **actual, 'device': str(cfg['device']), 'batch_wall_seconds': elapsed,
                                  'batch_CUDA_event_seconds': cuda_elapsed,
                                  'GPU_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                                  'GPU_peak_reserved_bytes': torch.cuda.max_memory_reserved()})
                with raw_file.open('a', encoding='utf-8') as stream:
                    for prediction in raw[raw_start:]:
                        stream.write(json.dumps(prediction, allow_nan=False) + '\n')
                save(folder / 'trace.json', trace)
                save(folder / 'progress.json', {'images_completed': len(records), 'images_total': count})
                if first == 0 or len(records) == count or (first // batch + 1) % max(1, count // batch // 10) == 0:
                    logger.info('%s: %s/%s images', condition['name'], len(records), count)
            gt, regions = ground_truth(records, 'native10')
            pred = mapped_predictions(raw, records, regions, 'native10', 'finetuned', 3000)
            for name, value in [('configuration.json', cfg), ('ground_truth.json', gt),
                                ('predictions_coco.json', pred), ('trace.json', trace)]:
                save(folder / name, value)
            tick = time.perf_counter()
            areas, classes, _ = evaluate_coco(gt, pred, 3000)
            evaluation_seconds = time.perf_counter() - tick
            save(folder / 'metrics.json', {'per_area': areas, 'per_class': classes})
            summary = {'RunID': out.name + '/' + condition['name'], 'ExperimentID': EXPERIMENT, 'Status': 'PASS',
                       'scope': 'PILOT_ONLY; DEVELOPMENT_VALIDATION' if mode == 'demo' else 'FULL_DEVELOPMENT_VALIDATION',
                       'images': count, 'checkpoint_id': 'ck01', 'checkpoint_SHA256': cfg['checkpoint_sha256'],
                       'protocol': 'pr01_native10_one2many_FP32_ignore_union_0.5_cap3000',
                       'configuration': condition, 'dataset_ZIP_SHA256': cfg['dataset_zip_sha256'],
                       'network_seconds': model_seconds, 'evaluation_seconds': evaluation_seconds,
                       'CUDA_event_seconds': cuda_seconds,
                       'GPU_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                       'GPU_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                       'raw_predictions': len(raw), 'aggregate': next(r for r in areas if r['size'] == 'all'),
                       'GT_SHA256': sha256(folder/'ground_truth.json'), 'raw_prediction_file': raw_file.name,
                       'predictions_SHA256': sha256(raw_file),
                       'predictions_coco_SHA256': sha256(folder/'predictions_coco.json'),
                       'metrics_SHA256': sha256(folder/'metrics.json')}
            save(folder / 'summary.json', summary)
            aggregate.append(summary)
    finally:
        archive.close()
    save(out / 'conditions.json', [{'name': item['configuration']['name'], 'directory': item['configuration']['name']} for item in aggregate])
    return {'status': 'PASS', 'scope': aggregate[0]['scope'], 'images_per_condition': count,
            'conditions': len(aggregate), 'checkpoint_id': 'ck01', 'summaries': aggregate,
            'total_seconds': time.perf_counter() - started}


def regenerate_run(source, out):
    from .evaluation import regenerate
    source = Path(source).resolve()
    index = source / 'conditions.json'
    folders = [source / row['directory'] for row in json.loads(index.read_text())] if index.exists() else [source]
    checks = []
    for folder in folders:
        if not folder.resolve().is_relative_to(source):
            raise ValueError('Condition index escapes its source run.')
        summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
        if summary['Status'] != 'PASS':
            raise ValueError('Metric regeneration requires a completed condition.')
        files = [(summary.get('raw_prediction_file', 'predictions_raw.json'), 'predictions_SHA256'),
                 ('ground_truth.json', 'GT_SHA256'), ('metrics.json', 'metrics_SHA256')]
        if 'predictions_coco_SHA256' in summary:
            files.append(('predictions_coco.json', 'predictions_coco_SHA256'))
        for filename, key in files:
            if sha256(folder / filename) != summary[key]:
                raise ValueError('Source artifact SHA mismatch: ' + filename)
        value = regenerate(folder)
        expected = json.loads((folder / 'metrics.json').read_text())
        if value != expected:
            raise ValueError('Regenerated metrics differ: ' + folder.name)
        save(out / f'{folder.name}_metrics.json', value)
        checks.append({'source': str(folder), 'result': 'EXACT_MATCH', 'metrics_SHA256': sha256(folder/'metrics.json')})
    return {'status': 'PASS', 'conditions': checks, 'GPU_inference': 'NOT_RUN'}


def execute(config_file, mode, source=None):
    cfg = settings(config_file)
    cfg['_config_file'] = str(Path(config_file).resolve())
    out, logger = new_run(cfg, EXPERIMENT, mode)
    for name in ['YOLO_CONFIG_DIR', 'MPLCONFIGDIR', 'TORCH_HOME']:
        folder = out / 'runtime' / name
        folder.mkdir(parents=True)
        os.environ[name] = str(folder)
    (Path(os.environ['YOLO_CONFIG_DIR']) / 'Ultralytics').mkdir()
    os.environ.update(YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true')
    try:
        if mode == 'check':
            result = check(cfg, out, logger)
        elif mode == 'regenerate':
            result = regenerate_run(source, out)
        else:
            result = predict(cfg, out, logger, mode)
        save(out / 'result.json', result)
        save(out / 'run.json', {'RunID': out.name, 'ExperimentID': EXPERIMENT, 'mode': mode, 'status': 'PASS'})
        logger.info('PASS: %s', out)
        return out
    except Exception as exc:
        save(out / 'failure.json', {'status': 'FAILED', 'error': repr(exc), 'traceback': traceback.format_exc()})
        save(out / 'run.json', {'RunID': out.name, 'ExperimentID': EXPERIMENT, 'mode': mode, 'status': 'FAILED'})
        logger.exception('Failed; partial artifacts retained: %s', out)
        try:
            from .visualization import build_run_index
            build_run_index(out.parent, EXPERIMENT)
        except (OSError, ValueError) as index_error:
            logger.warning('Result index update failed: %s', index_error)
        raise
