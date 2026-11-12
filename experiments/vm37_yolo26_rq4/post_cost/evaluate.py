"""Saved-prediction evaluation for portable matched training arms."""
import argparse
import gc
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

from portable import ROOT, POST, load, save, digest, within, runtime, renderer, resolve_data, matrix, check_worker_authorization, install_accumulator


def paired_contrasts(values):
    import numpy as np
    output = []
    for architecture, seed in sorted({key[:2] for key in values}):
        for resolution in (640, 1280):
            contexts = sorted({key[4] for key in values if key[:2] == (architecture, seed) and key[3] == resolution})
            for context in contexts:
                control = values[(architecture, seed, 'CONTROL', resolution, context)]
                target = values[(architecture, seed, 'ALL_FIVE', resolution, context)]
                normal_c = values[(architecture, seed, 'CONTROL', resolution, 'Normal')]
                normal_t = values[(architecture, seed, 'ALL_FIVE', resolution, 'Normal')]
                for other in (target, normal_c, normal_t):
                    if any(control['draws'][key] != other['draws'][key] for key in ('image_ids', 'samples', 'seed', 'sampling')):
                        raise ValueError('Bootstrap pairing differs.')
                for size in ('all', 'tiny', 'small', 'medium', 'large'):
                    for metric in ('map50', 'map50_95', 'ar50_95'):
                        def point(value):
                            return next(row[metric] for row in value['metrics']['per_area'] if row['size'] == size)
                        arrays = [value['draws']['values'][size][metric] for value in (control, target, normal_c, normal_t)]
                        if any(point(v) is None for v in (control, target, normal_c, normal_t)) or any(x is None for a in arrays for x in a):
                            output.append({'architecture': architecture, 'seed': seed, 'resolution': resolution,
                                'context': context, 'size': size, 'metric': metric, 'status': 'NOT_ESTIMABLE'})
                            continue
                        c, t, nc, nt = map(np.asarray, arrays)
                        delta = t - c
                        loss_change = (nt - t) - (nc - c)
                        output.append({'architecture': architecture, 'seed': seed, 'resolution': resolution,
                            'context': context, 'size': size, 'metric': metric, 'status': 'ESTIMATED',
                            'ALL_FIVE_minus_CONTROL': point(target) - point(control),
                            'difference_CI95': np.percentile(delta, [2.5, 97.5]).tolist(),
                            'degradation_loss_change': (point(normal_t) - point(target)) - (point(normal_c) - point(control)),
                            'loss_change_CI95': np.percentile(loss_change, [2.5, 97.5]).tolist(),
                            'CI_scope': 'Pointwise paired image percentile interval; conditional on one training seed.'})
    return output


def evaluate(bundle, data_root, parent, protocol_name):
    out = parent / 'evaluation'
    check_worker_authorization(parent, bundle, protocol_name)
    save(out / 'WORKER_STARTED.json', {'scope': 'AUTHORIZED_MATCHED_EVALUATION'})
    runtime(out, gpu=True)
    import cv2
    import numpy as np
    import torch
    from unittest.mock import patch
    from ultralytics import YOLO
    from ultralytics.utils import callbacks
    from main.labels import VIS_NAMES, ground_truth, mapped_predictions
    from main.predictor_stable import StableNMSPredictor
    from main.statistics import evaluate_coco, bootstrap_coco
    from main.training_health import model_health
    from post80_loader import read_only_sources
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = load(bundle / (protocol_name + '_protocol.json'))
    spec = load(bundle / ('evaluation_manifest.json' if protocol_name == 'pilot20' else 'formal_evaluation_manifest.json'))
    population = resolve_data(bundle, data_root)
    gt_source = load(bundle / spec['GT_asset'])
    records = gt_source['records']
    by_stem = {row['stem']: row for row in records}
    if set(by_stem) != {r['image_id'] for r in population['val']}:
        raise ValueError('Frozen GT population differs.')
    gt, regions = ground_truth(records, 'native10')
    save(out / 'ground_truth.json', {'coco': gt, 'records': records, 'source_SHA256': spec['GT_SHA256']})
    ops = renderer()
    install_accumulator()
    values = {}
    index = []
    started = time.perf_counter()
    for architecture, seed, arm in matrix(protocol):
        train_out = parent / (architecture + '_seed' + str(seed) + '_' + arm)
        acceptance = load(train_out / 'training_acceptance.json')
        checkpoint = Path(acceptance['selected_checkpoint']['path'])
        if acceptance['completed_epochs'] != protocol['epochs'] or digest(checkpoint) != acceptance['selected_checkpoint']['sha256']:
            raise ValueError('Training arm is incomplete or changed.')
        with patch.object(callbacks, 'add_integration_callbacks', lambda instance: None):
            model = YOLO(str(checkpoint))
        model_health(model.model)
        if list(model.names.values()) != VIS_NAMES:
            raise ValueError('Native10 class order differs.')
        for case in spec['contexts']:
            expected = [json.loads(line) for line in within(bundle, case['pixel_manifest_asset']).read_text(encoding='utf-8').splitlines()]
            expected = {Path(row['source']['ZIP_member']).stem: row for row in expected}
            if len(expected) != 548:
                raise ValueError('Incomplete frozen pixel manifest.')
            for resolution in spec['resolutions']:
                config_identity = train_out.name + '_' + case['id'] + '_R' + str(resolution)
                folder = out / ('e_' + hashlib.sha256(config_identity.encode()).hexdigest()[:12])
                folder.mkdir()
                traces = []
                class Traced(StableNMSPredictor):
                    def preprocess(self, images):
                        value = super().preprocess(images)
                        if list(value.shape) != [1, 3, resolution, resolution] or value.dtype != torch.float32:
                            raise ValueError('Inference tensor differs.')
                        self.tensor = {'NCHW': list(value.shape), 'dtype': str(value.dtype)}
                        return value
                    def postprocess(self, preds, img, orig_imgs, **kwargs):
                        raw = preds[0] if isinstance(preds, (tuple, list)) else preds
                        if self.model.end2end or raw.ndim != 3 or raw.shape[1] != 14 or not torch.isfinite(raw).all():
                            raise ValueError('Dense native10 prediction is unhealthy or incompatible.')
                        return super().postprocess(preds, img, orig_imgs, **kwargs)
                torch.cuda.reset_peak_memory_stats()
                tick = time.perf_counter()
                raw_path = folder / 'predictions.jsonl.gz'
                partial = raw_path.with_suffix('.partial')
                # Pixel values are deterministically regenerated and hash-checked against the same accepted manifest.
                with gzip.open(partial, 'xt', encoding='utf-8') as stream, read_only_sources(out), \
                        patch.object(callbacks, 'add_integration_callbacks', lambda instance: None):
                    for reference in population['val']:
                        stem = reference['image_id']
                        image = cv2.imdecode(np.frombuffer(Path(reference['image_path']).read_bytes(), np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                        pixels, effect_trace = ops.generate(image, case['p']['leaves'], case['p']['attributes'], stem, spec['render_seed'])
                        pixel_sha = hashlib.sha256(pixels.tobytes()).hexdigest()
                        if pixel_sha != expected[stem]['pixel_sha256']:
                            raise ValueError('Evaluation pixels differ from the frozen condition.')
                        torch.cuda.synchronize()
                        began = time.perf_counter()
                        with torch.inference_mode():
                            result = model.predict(pixels, predictor=Traced, imgsz=resolution, batch=1, device='0',
                                quantize=None, nms=None, conf=.001, iou=.7, max_det=3000, rect=False,
                                agnostic_nms=False, classes=None, augment=False, save=False, verbose=False)[0]
                        torch.cuda.synchronize()
                        ms = 1000 * (time.perf_counter() - began)
                        boxes = result.boxes.data.detach().cpu().numpy()
                        if not np.isfinite(boxes).all() or result.orig_shape != pixels.shape[:2]:
                            raise ValueError('Prediction health or original coordinates differ.')
                        record = by_stem[stem]
                        predictions = [{'image_id': record['id'], 'class_id': int(c), 'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)], 'score': float(score)}
                                       for x1, y1, x2, y2, score, c in boxes if x2 > x1 and y2 > y1]
                        stream.write(json.dumps({'image_id': record['id'], 'image_stem': stem, 'source_SHA256': reference['image_SHA256'],
                                     'pixel_SHA256': pixel_sha, 'predictions': predictions}) + '\n')
                        traces.append({'image_id': record['id'], 'input': model.predictor.tensor, 'inference_ms': ms,
                                       'predictions': len(predictions), 'effects': effect_trace})
                partial.rename(raw_path)
                inference_wall = time.perf_counter() - tick
                # Calculate metrics from the committed raw file, not transient GPU outputs.
                with gzip.open(raw_path, 'rt', encoding='utf-8') as stream:
                    raw_records = [json.loads(line) for line in stream]
                if len(raw_records) != 548 or len({r['image_id'] for r in raw_records}) != 548:
                    raise ValueError('Raw prediction membership differs.')
                raw = [p for row in raw_records for p in row['predictions']]
                mapped = mapped_predictions(raw, records, regions, 'native10', 'finetuned', 3000)
                tick = time.perf_counter()
                areas, classes, evaluator = evaluate_coco(gt, mapped, 3000)
                evaluation_seconds = time.perf_counter() - tick
                metrics = {'per_area': areas, 'per_class': classes, 'raw_count': len(raw), 'mapped_count': len(mapped), 'images': 548}
                save(folder / 'metrics.json', metrics)
                bootstrap_start = time.perf_counter()
                intervals, draws = bootstrap_coco(evaluator, samples=spec['bootstrap_samples'], seed=spec['bootstrap_seed'])
                bootstrap_seconds = time.perf_counter() - bootstrap_start
                save(folder / 'bootstrap.json', {'intervals': intervals, 'draws': draws})
                save(folder / 'trace.json', {'per_image': traces, 'inference_wall_seconds': inference_wall,
                    'COCO_seconds': evaluation_seconds, 'bootstrap_seconds': bootstrap_seconds,
                    'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_reserved_bytes': torch.cuda.max_memory_reserved(), 'timing_scope': 'Prediction latency excludes source decoding/effect generation; wall time includes them.'})
                evidence = {'checkpoint_SHA256': digest(checkpoint), 'GT_source_SHA256': spec['GT_SHA256'],
                    'pixel_manifest_SHA256': case['pixel_manifest_SHA256'], 'predictions_SHA256': digest(raw_path),
                    'metrics_SHA256': digest(folder / 'metrics.json'), 'bootstrap_SHA256': digest(folder / 'bootstrap.json'),
                    'scope': 'DEVELOPMENT_ONLY', 'status': 'COMPLETE'}
                save(folder / 'acceptance.json', evidence)
                values[(architecture, seed, arm, resolution, case['id'])] = {'metrics': metrics, 'draws': draws}
                index.append({'folder': folder.name, 'configuration': config_identity, 'architecture': architecture,
                              'seed': seed, 'arm': arm, 'resolution': resolution, 'context': case['id'], **evidence})
                del evaluator, raw_records, raw, mapped
                gc.collect()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    save(out / 'paired_contrasts.json', paired_contrasts(values))
    save(out / 'result_index.json', index)
    save(out / 'evaluation_result.json', {'status': 'COMPLETE', 'configurations': len(index),
         'expected_configurations': len(matrix(protocol)) * 2 * len(spec['contexts']), 'wall_seconds': time.perf_counter() - started,
         'scope': 'DEVELOPMENT_ONLY', 'full21_context_evaluation': protocol_name == 'formal100',
         'full_quality_and_latency_gap_computed': False})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--training-parent', required=True, type=Path)
    parser.add_argument('--protocol', required=True, choices=('pilot20', 'formal100'))
    args = parser.parse_args()
    evaluate(args.bundle.resolve(), args.data_root.resolve(), args.training_parent.resolve(), args.protocol)
