"""CPU parity through the v3 trainer's real dataset factory."""
import argparse
import copy
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace

POST = Path(__file__).resolve().parent
sys.path.insert(0, str(POST))
import portable_v3 as v3
from verify_input import value_signature, rng_signature


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def golden_checks(bundle, population, out):
    import gc
    import cv2
    import numpy as np
    import torch
    from main.training_health import model_health
    from main.labels import VIS_NAMES, ground_truth
    health = []
    for row in v3.base.load(bundle / 'portable_initializations.json'):
        model = torch.load(v3.base.within(bundle, row['asset']), map_location='cpu', weights_only=False)['model']
        model_health(model)
        if list(model.names.values()) != VIS_NAMES or any(t.is_floating_point() and t.dtype != torch.float32
                                                       for t in model.state_dict().values()):
            raise ValueError('Initialization class order or FP32 state differs.')
        health.append({'architecture': row['architecture'], 'seed': row['seed'], 'SHA256': row['sha256']})
        del model
        gc.collect()
    records = v3.base.load(bundle / 'assets/ground_truth.json')['records']
    if {row['stem'] for row in records} != {row['image_id'] for row in population['val']}:
        raise ValueError('Evaluation GT/image membership differs.')
    gt, _ = ground_truth(records, 'native10')
    if len(gt['images']) != 548 or len(gt['categories']) != 10:
        raise ValueError('Native10 GT changed.')
    images = {}
    for row in population['val'][:6]:
        images[row['image_id']] = cv2.imdecode(np.frombuffer(Path(row['image_path']).read_bytes(), np.uint8),
            cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    checks = []
    ops = v3.base.renderer()
    for name in ('evaluation_manifest.json', 'formal_evaluation_manifest.json'):
        spec = v3.base.load(bundle / name)
        for case in spec['contexts']:
            expected = {Path(row['source']['ZIP_member']).stem: row for row in map(json.loads,
                v3.base.within(bundle, case['pixel_manifest_asset']).read_text().splitlines())}
            if len(expected) != 548:
                raise ValueError('Golden evaluation manifest is incomplete.')
            for identity, image in images.items():
                pixels, _ = ops.generate(image, case['p']['leaves'], case['p']['attributes'], identity, spec['render_seed'])
                digest = hashlib.sha256(pixels.tobytes()).hexdigest()
                if digest != expected[identity]['pixel_sha256'] or list(pixels.shape) != expected[identity]['shape']:
                    raise ValueError('Target-host evaluation pixels differ from the accepted golden record.')
                checks.append({'manifest': name, 'condition': case['id'], 'image': identity, 'pixel_SHA256': digest})
    v3.base.save(out / 'golden_checks.json', {'initializations': health, 'pixels': checks,
        'GT_images': len(gt['images']), 'GT_instances': len(gt['annotations']), 'GPU_seconds': 0})
    return len(health), len(checks)


def make_dataset(bundle, population, protocol, out, list_path, architecture, seed, arm, pipeline, cache):
    import torch
    from ultralytics.cfg import get_cfg
    plan = v3.make_plan(bundle, population, protocol, architecture, seed, arm, out, 'formal100', pipeline)
    Trainer = v3.trainer_factory(plan, out)
    trainer = Trainer.__new__(Trainer)
    trainer.args = get_cfg(overrides=plan['args'])
    trainer.model = SimpleNamespace(stride=torch.tensor([32]))
    trainer.data = {'names': dict(enumerate(__import__('main.labels', fromlist=['VIS_NAMES']).VIS_NAMES)),
                    'nc': 10, 'channels': 3}
    target = out / 'dataset_cache/train/labels.cache'
    target.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        shutil.copyfile(cache, target)
    data = trainer.build_dataset(str(list_path), mode='train', batch=2)
    if not cache.exists():
        shutil.copyfile(target, cache)
    return data, plan['args']


def execute(bundle, data_root, output_root, batches=8, repeats=2):
    from main.pipeline import new_run
    out = new_run({'output_root': str(output_root)}, 'vm37_yolo26_rq4', 'portable_v3_cpu_comparison')[0]
    v3.base.runtime(out, gpu=False)
    import cv2
    import numpy as np
    import psutil
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.torch_utils import init_seeds
    v3.base.verify_bundle(bundle)
    v3.base.environment_check(bundle)
    v3.base.renderer()
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    population = v3.base.resolve_data(bundle, data_root)
    healthy, pixels = golden_checks(bundle, population, out)
    protocol = v3.protocol_for(bundle, 'formal100')
    fixture = v3.base.load(bundle / 'cpu_comparison_manifest.json')
    anchors = fixture['image_ids'][:batches * 2]
    if len(anchors) != batches * 2:
        raise ValueError('Frozen CPU anchor list is incomplete.')
    v3.base.save(out / 'protocol.json', {'batches_per_state': batches, 'repetitions': repeats,
        'architectures': protocol['architectures'], 'seeds': protocol['seeds'], 'arms': protocol['arms'],
        'epoch_states': [0, 1, 90], 'closed_mosaic_epoch': 90, 'workers': 0,
        'timing_scope': 'CPU input only; source caches fresh per pass, OS cache not flushed.',
        'startup': 'Warm imports before both implementations; reset all RNGs to the same seed before each pass.',
        'image_selection': 'Frozen historical training anchor IDs, not selected using response AP.',
        'GPU_seconds': 0},)
    list_path = out / 'train_images.txt'
    list_path.write_text('\n'.join(row['image_path'] for row in population['train']) + '\n', encoding='utf-8')
    cache = out / 'shared_labels.cache'
    init_seeds(42, deterministic=True)
    warm = out / 'warm'
    warm.mkdir()
    data, _ = make_dataset(bundle, population, protocol, warm, list_path, 'YOLO26s', 42, 'CONTROL', 'baseline', cache)
    del data
    started = time.perf_counter()
    timings = []
    signatures = []
    peak = 0
    for case, (architecture, seed, arm) in enumerate(v3.base.matrix(protocol)):
        expected = None
        for repeat in range(repeats):
            order = ('baseline', 'optimized') if repeat % 2 == 0 else ('optimized', 'baseline')
            for pipeline in order:
                folder = out / ('c' + str(case) + '_' + str(repeat) + '_' + pipeline[0])
                folder.mkdir()
                init_seeds(seed, deterministic=True)
                data, args = make_dataset(bundle, population, protocol, folder, list_path,
                                           architecture, seed, arm, pipeline, cache)
                lookup = {Path(path).stem: i for i, path in enumerate(data.im_files)}
                actual = []
                try:
                    for epoch in (0, 1, 90):
                        data.set_epoch(epoch)
                        if epoch == 90:
                            data.close_mosaic(get_cfg(overrides=args))
                        for position in range(batches):
                            ids = anchors[position * 2:position * 2 + 2]
                            begin = time.perf_counter()
                            batch = data.collate_fn([data[lookup[name]] for name in ids])
                            elapsed = time.perf_counter() - begin
                            record = {'epoch': epoch, 'position': position, 'ids': ids,
                                      'batch': value_signature(batch), 'events': copy.deepcopy(data.effect_events),
                                      'RNG': rng_signature(data), 'buffer': list(data.buffer),
                                      'occurrences': dict(data.occurrences)}
                            actual.append(record)
                            peak = max(peak, psutil.Process().memory_info().rss)
                            timings.append({'architecture': architecture, 'seed': seed, 'arm': arm,
                                'pipeline': pipeline, 'repeat': repeat, 'epoch': epoch, 'batch': position,
                                'input_seconds': elapsed, 'native_calls': len(data.effect_events)})
                            data.effect_events.clear()
                            if getattr(data, 'pending', None):
                                raise ValueError('Unconsumed native prefetch.')
                finally:
                    if hasattr(data, 'close'):
                        data.close()
                if expected is None:
                    expected = actual
                    signatures.append({'architecture': architecture, 'seed': seed, 'arm': arm, 'batches': actual})
                if expected != actual:
                    v3.base.save(out / 'mismatch.json', {'architecture': architecture, 'seed': seed,
                        'arm': arm, 'pipeline': pipeline, 'repeat': repeat, 'actual': actual})
                    raise ValueError('Integrated dataset parity failed.')
        print('V3_CPU_EXACT', architecture, seed, arm, flush=True)
    write_csv(out / 'timing.csv', timings)
    summary = []
    for architecture in protocol['architectures']:
        for arm in protocol['arms']:
            original = sum(row['input_seconds'] for row in timings if row['architecture'] == architecture
                           and row['arm'] == arm and row['pipeline'] == 'baseline')
            optimized = sum(row['input_seconds'] for row in timings if row['architecture'] == architecture
                            and row['arm'] == arm and row['pipeline'] == 'optimized')
            summary.append({'architecture': architecture, 'arm': arm, 'baseline_seconds': original,
                            'optimized_seconds': optimized, 'speedup': original / optimized,
                            'input_time_reduction': 1 - optimized / original})
    write_csv(out / 'performance.csv', summary)
    v3.base.save(out / 'baseline_signatures.json', signatures)
    receipt = {'status': 'CPU_EXACT_PARITY_PASS', 'bundle_manifest_SHA256': v3.base.digest(bundle / 'bundle_manifest.json'),
        'host': v3.host_identity(bundle), 'implementation_SHA256': v3.implementation_files(bundle),
        'tested_architectures': protocol['architectures'], 'tested_seeds': protocol['seeds'],
        'tested_arms': protocol['arms'], 'tested_epoch_states': [0, 1, 90],
        'batches_per_state': batches, 'repetitions': repeats,
        'golden_pixel_checks': pixels, 'healthy_initializations': healthy,
        'exact_tensors_labels_traces_RNG_and_order': True, 'CPU_wall_seconds': time.perf_counter() - started,
        'sampled_peak_RSS_bytes': peak, 'GPU_seconds': 0,
        'GPU_orchestrator_validation': 'NOT_RUN', 'full_epoch_speedup': 'NOT_MEASURED'}
    v3.base.save(out / 'CPU_receipt.json', receipt)
    v3.base.save(out / 'comparison_result.json', {'receipt': receipt, 'performance': summary,
        'input_manifest_SHA256': v3.base.digest(bundle / 'cpu_comparison_manifest.json')})
    (out / 'diary.md').write_text('# Integrated CPU comparison\n\nCompared both input implementations through '
        'the real trainer dataset factory. Two architectures, three seeds, two arms and three epoch/Mosaic '
        'states; all pixels, labels, traces, RNG and ordering matched. No GPU work.\n', encoding='utf-8')
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Portable v3 CPU parity</h1>'
        '<p>Exact input parity passed; GPU execution remains unvalidated.</p>'
        '<a href="performance.csv">CPU timing</a> · <a href="CPU_receipt.json">Admission receipt</a>', encoding='utf-8')
    v3.finish_run(out, 'CPU_EXACT_PARITY_PASS')
    print(out, flush=True)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--batches', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    if args.batches != 8 or args.repeats != 2:
        parser.error('The v3 admission design is frozen at 8 batches and 2 repeats.')
    execute(args.bundle.resolve(strict=True), args.data_root.resolve(strict=True), args.output_root.resolve(),
            args.batches, args.repeats)
