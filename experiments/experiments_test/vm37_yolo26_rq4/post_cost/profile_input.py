"""Profile the accepted input pipeline and replay saved native-pixel events."""
import argparse
from collections import defaultdict
from contextlib import ExitStack, contextmanager
import csv
import functools
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

POST = Path(__file__).resolve().parent
sys.path.insert(0, str(POST.parent))
from study import *

REFERENCE = 'vm37_yolo26_rq4_all_five_full_epoch_cost_20261112T200430_2ab983f6'


def csv_write(path, rows):
    rows = list(rows)
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Timings:
    def __init__(self):
        self.stack = []
        self.rows = []

    @contextmanager
    def measure(self, name):
        frame = [name, time.perf_counter(), 0.]
        self.stack.append(frame)
        try:
            yield
        finally:
            elapsed = time.perf_counter() - frame[1]
            self.stack.pop()
            if self.stack:
                self.stack[-1][2] += elapsed
            self.rows.append((name, elapsed, max(0., elapsed - frame[2])))

    def wrap(self, function, name):
        @functools.wraps(function)
        def timed(*args, **kwargs):
            with self.measure(name):
                return function(*args, **kwargs)
        return timed

    def summary(self, scope, wall):
        import numpy as np
        grouped = defaultdict(list)
        for name, total, own in self.rows:
            grouped[name].append((total, own))
        return [{'scope': scope, 'operation': name, 'calls': len(values),
                 'inclusive_seconds': sum(v[0] for v in values),
                 'exclusive_seconds': sum(v[1] for v in values),
                 'mean_seconds': float(np.mean([v[0] for v in values])),
                 'p50_seconds': float(np.percentile([v[0] for v in values], 50)),
                 'p95_seconds': float(np.percentile([v[0] for v in values], 95)),
                 'exclusive_percent_of_scope': 100 * sum(v[1] for v in values) / wall}
                for name, values in sorted(grouped.items())]


def setup(out):
    runtime(out, cuda=False)
    import importlib.metadata
    if importlib.metadata.version('ultralytics') != '8.4.210':
        raise RuntimeError('Use the frozen .venv_rq4 environment (Ultralytics 8.4.210).')
    import cv2
    import torch
    from portable import renderer
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    renderer()


def dataset(out, plan, cls=None):
    from ultralytics.cfg import get_cfg
    from main.training_data import RQ4Dataset
    from main.labels import VIS_NAMES
    cls = cls or RQ4Dataset
    write_atomic(out / 'images.txt', '\n'.join(row['image_path'] for row in plan['train']) + '\n', out)
    return cls(img_path=str(out / 'images.txt'), imgsz=1280, batch_size=2, augment=True,
               hyp=get_cfg(overrides=plan['args']), rect=False, cache=False, stride=32, pad=0.,
               task='detect', data={'names': dict(enumerate(VIS_NAMES)), 'nc': 10, 'channels': 3},
               label_lookup=plan['train_labels'], evidence=out / 'label_cache',
               recipe=plan['targeted_recipe'], seed=42, engineering=False)


@contextmanager
def instrument(timings):
    import cv2
    import main.training_data as data
    import main.training_effects as effects
    from ultralytics.data import augment
    ops = effects.renderer()
    with ExitStack() as stack:
        for name in ('apply_haze', 'apply_exposure', 'apply_motion_blur', 'apply_gaussian_noise',
                     'apply_shot_noise', 'apply_jpeg'):
            stack.enter_context(patch.object(ops, name, timings.wrap(getattr(ops, name), name)))
        for name in ('imdecode', 'imencode', 'filter2D', 'resize'):
            stack.enter_context(patch.object(cv2, name, timings.wrap(getattr(cv2, name), 'cv2.' + name)))
        stack.enter_context(patch.object(Path, 'read_bytes', timings.wrap(Path.read_bytes, 'file_read')))
        for owner, name, label in ((data.RQ4Dataset, 'load_image', 'native_load_cache_and_hash'),
                                   (data.NativeEffects, 'apply', 'native_effect_sampling_and_copy'),
                                   (data, 'resize_native', 'native_resize')):
            stack.enter_context(patch.object(owner, name, timings.wrap(getattr(owner, name), label)))
        for name in ('Mosaic', 'RandomPerspective', 'RandomHSV', 'RandomFlip', 'Albumentations',
                     'CopyPaste', 'MixUp', 'CutMix', 'LetterBox', 'Format'):
            owner = getattr(augment, name, None)
            if owner is not None:
                stack.enter_context(patch.object(owner, '__call__', timings.wrap(owner.__call__, name)))
        stack.enter_context(patch.object(augment.Format, '_format_img',
                                        timings.wrap(augment.Format._format_img, 'tensor_conversion')))
        yield


def execute(native_calls=512, pipeline_batches=32, request_file=None):
    out = create('input_profile_cpu')
    setup(out)
    import numpy as np
    import psutil
    import torch
    from ultralytics.utils.torch_utils import init_seeds
    reference = run_path(REFERENCE)
    verify_hashes(reference, ('protocol.json', 'training_batches.jsonl', 'augmentation_events.jsonl'))
    plan = load(reference / 'protocol.json')
    begin = time.perf_counter()
    batch_rows = [json.loads(line) for line in (reference / 'training_batches.jsonl').read_text().splitlines()]
    saved_events = []
    with (reference / 'augmentation_events.jsonl').open() as stream:
        for line in stream:
            saved_events.append(json.loads(line))
            if len(saved_events) == native_calls:
                break
    sources = [Path(__file__), PUBLIC / 'code/main/training_data.py', PUBLIC / 'code/main/training_effects.py',
               HERE.parent / 'cv261109/generator.py', HERE / 'TARGETED_POLICY.json',
               VM36 / 'feature_models/fm_p.uvl', VM36 / 'feature_models/fm_d.uvl']
    snapshots = out / 'source_snapshots'
    snapshots.mkdir()
    for path in sources:
        shutil.copyfile(path, snapshots / path.name)
    if request_file is not None:
        shutil.copyfile(Path(request_file).resolve(strict=True), out / 'author_task.txt')
    inputs = {'reference_RunID': REFERENCE, 'reference': [source(reference / name) for name in
              ('protocol.json', 'training_batches.jsonl', 'augmentation_events.jsonl')],
              'sources': [source(path) for path in sources], 'resource_ledger': source(RUNS / 'compute_allocation.json'),
              'historical_GPU_or_metrics_recomputed': False}
    save(out / 'inputs.json', inputs, out)
    save(out / 'protocol.json', {'seed': 42, 'native_calls': native_calls, 'pipeline_batches': pipeline_batches,
         'workers': 0, 'scope': 'CPU_INPUT_PROFILE_ONLY', 'GPU_seconds': 0,
         'native_replay': 'Exact saved image/epoch/occurrence and effect/pixel trace.',
         'pipeline_replay': 'Recorded anchor order, fixed fresh RNG; historical geometry RNG was not saved.',
         'timing': 'Inclusive and nested-exclusive times recorded separately; do not sum inclusive parents and children.',
         'memory': 'Process RSS after each native call/batch; sampled peak, not per-operator allocation.'}, out)
    cpu_rows = []
    for key in ('loader_seconds', 'forward_backward_health_seconds', 'forward_loss_wall_seconds', 'backward_wall_seconds'):
        values = np.array([row[key] for row in batch_rows])
        cpu_rows.append({'scope': 'historical_full_epoch', 'operation': key, 'calls': len(values),
             'total_seconds': float(values.sum()), 'mean_seconds': float(values.mean()),
             'p50_seconds': float(np.percentile(values, 50)), 'p95_seconds': float(np.percentile(values, 95)),
             'note': 'Forward/backward are nested in forward_backward_health; not additive.'})
    csv_write(out / 'CPU_PROFILE.csv', cpu_rows)
    init_seeds(42, deterministic=True)
    data = dataset(out, plan)
    lookup = {Path(path).stem: i for i, path in enumerate(data.im_files)}
    process = psutil.Process()
    memory = []
    native_timing = Timings()
    seen = []
    with instrument(native_timing):
        start = time.perf_counter()
        for position, expected in enumerate(saved_events):
            data.set_epoch(expected['epoch']) if data.epoch_number != expected['epoch'] else None
            with native_timing.measure('input_root'):
                data.load_image(lookup[expected['id']])
            actual = data.effect_events.pop()
            if actual != expected:
                raise ValueError('Historical native pixels or parameter trace differs at ' + str(position))
            seen.append(actual['id'])
            memory.append({'scope': 'historical_native_replay', 'position': position,
                           'RSS_bytes': process.memory_info().rss, 'native_cache_entries': len(data.native_cache)})
        native_wall = time.perf_counter() - start
    summary = native_timing.summary('historical_native_replay', native_wall)
    del data
    init_seeds(42, deterministic=True)
    pipeline_folder = out / 'pipeline'
    pipeline_folder.mkdir()
    data = dataset(pipeline_folder, plan)
    lookup = {Path(path).stem: i for i, path in enumerate(data.im_files)}
    pipeline_timing = Timings()
    with instrument(pipeline_timing):
        start = time.perf_counter()
        for position, row in enumerate(batch_rows[:pipeline_batches]):
            with pipeline_timing.measure('input_root'):
                batch = data.collate_fn([data[lookup[name]] for name in row['image_ids']])
            if list(batch['img'].shape) != row['NCHW'] or batch['img'].dtype != torch.uint8:
                raise ValueError('Pipeline tensor shape differs.')
            memory.append({'scope': 'fresh_geometry_pipeline', 'position': position,
                           'RSS_bytes': process.memory_info().rss, 'native_cache_entries': len(data.native_cache)})
            data.effect_events.clear()
        pipeline_wall = time.perf_counter() - start
    summary += pipeline_timing.summary('fresh_geometry_pipeline', pipeline_wall)
    csv_write(out / 'OPERATOR_TIMINGS.csv', summary)
    csv_write(out / 'memory.csv', memory)
    result = {'status': 'PASS', 'native_events_exact': len(saved_events), 'native_unique_images': len(set(seen)),
              'pipeline_batches': pipeline_batches, 'historical_native_seconds': native_wall,
              'fresh_geometry_pipeline_seconds': pipeline_wall,
              'sampled_peak_RSS_bytes': max(row['RSS_bytes'] for row in memory),
              'CPU_wall_seconds': time.perf_counter() - begin, 'GPU_seconds': 0,
              'queue_wait': 'No multiprocessing queue with accepted workers0. Root residual includes Python/collation.',
              'OS_cache': 'Not flushed; measurements are not cold-disk benchmarks.',
              'per_operator_memory': 'NOT_MEASURED; RSS belongs to the whole process.'}
    save(out / 'profile_result.json', result, out)
    top = sorted((row for row in summary if row['scope'] == 'historical_native_replay'),
                 key=lambda row: row['exclusive_seconds'], reverse=True)
    write_atomic(out / 'LOADER_BOTTLENECKS.md', '# Input pipeline profile\n\n'
         + '\n'.join(f"- {row['operation']}: {row['exclusive_seconds']:.3f}s exclusive, {row['calls']} calls."
                     for row in top[:10]) + '\n\nInclusive parent times overlap their children. '
         'Native replay matches saved pixels and effect parameters. Fresh geometry uses a new fixed RNG state; '
         'historical full tensors cannot be reconstructed from the old timing logs alone. '
         'Process RSS is sampled, not assigned to individual operators.\n', out)
    write_atomic(out / 'diary.md', '# Input profiling\n\nReplayed saved native events exactly, then measured a fixed '
         'fresh-geometry pipeline on recorded anchor IDs. Preserved old code and policy. GPU=0.\n', out)
    save(out / 'run.json', {'RunID': out.name, 'ExperimentID': EXPERIMENT,
         'mode': 'INPUT_PROFILE_CPU', **result}, out, replace=True)
    seal(out)
    print(json.dumps({'RunID': out.name, **result}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--native-calls', type=int, default=512)
    parser.add_argument('--pipeline-batches', type=int, default=32)
    parser.add_argument('--request-file', type=Path)
    args = parser.parse_args()
    if not 64 <= args.native_calls <= 1024 or not 8 <= args.pipeline_batches <= 64:
        parser.error('Use 64..1024 native calls and 8..64 pipeline batches.')
    execute(args.native_calls, args.pipeline_batches, args.request_file)
