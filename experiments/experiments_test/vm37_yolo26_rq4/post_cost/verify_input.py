"""Compare original and prefetched inputs without training or new metrics."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import pickle
import random
import shutil
import sys
import time

POST = Path(__file__).resolve().parent
sys.path.insert(0, str(POST))
from profile_input import REFERENCE, csv_write, dataset, setup
from study import *

PROFILE = 'vm37_yolo26_rq4_input_profile_cpu_20261113T094531_12ba5298'


def value_signature(value):
    import torch
    if isinstance(value, torch.Tensor):
        return {'dtype': str(value.dtype), 'shape': list(value.shape),
                'SHA256': hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()}
    if isinstance(value, dict):
        return {key: value_signature(child) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [value_signature(child) for child in value]
    return value


def rng_signature(data):
    import numpy as np
    import torch
    states = [random.getstate(), np.random.get_state(), torch.get_rng_state().numpy().tobytes()]
    seen = set()
    def visit(transform):
        if id(transform) in seen:
            return
        seen.add(id(transform))
        if hasattr(transform, 'random_generator'):
            states.append(transform.random_generator.bit_generator.state)
        if hasattr(transform, 'py_random'):
            states.append(transform.py_random.getstate())
        for child in getattr(transform, 'transforms', ()):
            visit(child)
        for name in ('transform', 'pre_transform'):
            child = getattr(transform, name, None)
            if child is not None:
                visit(child)
    visit(data.transforms)
    return hashlib.sha256(pickle.dumps(states)).hexdigest()


def run_pass(out, plan, anchors, implementation, repetition):
    import psutil
    import torch
    from ultralytics.utils.torch_utils import init_seeds
    from fast_input import PrefetchDataset, ArithmeticPrefetchDataset
    folder = out / (implementation + '_' + str(repetition))
    folder.mkdir()
    (folder / 'label_cache').mkdir()
    shutil.copyfile(run_path(PROFILE) / 'label_cache/labels.cache', folder / 'label_cache/labels.cache')
    init_seeds(42, deterministic=True)
    cls = {'original': None, 'prefetch2': PrefetchDataset, 'arithmetic2': ArithmeticPrefetchDataset}[implementation]
    data = dataset(folder, plan, cls)
    lookup = {Path(path).stem: i for i, path in enumerate(data.im_files)}
    signatures, rows = [], []
    process = psutil.Process()
    started = time.perf_counter()
    try:
        for position, identities in enumerate(anchors):
            tick = time.perf_counter()
            batch = data.collate_fn([data[lookup[name]] for name in identities])
            seconds = time.perf_counter() - tick
            signatures.append({'batch': value_signature(batch), 'events': data.effect_events.copy(),
                               'rng': rng_signature(data), 'buffer': list(data.buffer),
                               'occurrences': dict(data.occurrences)})
            rows.append({'implementation': implementation, 'repetition': repetition, 'position': position,
                         'input_seconds': seconds, 'RSS_bytes': process.memory_info().rss,
                         'native_calls': len(data.effect_events), 'NCHW': json.dumps(list(batch['img'].shape))})
            data.effect_events.clear()
            if getattr(data, 'pending', None):
                raise ValueError('Unconsumed native inputs at a batch boundary.')
    finally:
        if hasattr(data, 'close'):
            data.close()
    wall = time.perf_counter() - started
    return signatures, rows, wall


def execute(batches, repeats, optimized='prefetch2'):
    out = create('input_optimization_cpu')
    setup(out)
    reference = run_path(REFERENCE)
    plan = load(reference / 'protocol.json')
    trace = [json.loads(line) for line in (reference / 'training_batches.jsonl').read_text().splitlines()]
    anchors = [row['image_ids'] for row in trace[:batches]]
    save(out / 'protocol.json', {'status': 'FROZEN_BEFORE_MEASUREMENT', 'reference_RunID': REFERENCE,
         'batches_per_pass': batches, 'repetitions': repeats, 'seed': 42, 'workers': 0,
         'native_prefetch_threads': 2, 'comparison': 'Exact tensors, labels, traces, RNG and cache/mosaic order.',
         'optimized_implementation': optimized, 'import_warmup_before_comparison': True,
         'timing': 'Paired alternating order; each pass has a fresh native cache. OS cache is not flushed.',
         'GPU_seconds': 0, 'historical_geometry': 'Not claimed; both implementations share fresh frozen RNG.'}, out)
    sources = [Path(__file__), POST / 'fast_input.py', POST / 'input_ops.py', POST / 'profile_input.py',
               PUBLIC / 'code/main/training_data.py', PUBLIC / 'code/main/training_effects.py',
               HERE / 'TARGETED_POLICY.json', HERE.parent / 'cv261109/generator.py']
    save(out / 'inputs.json', {'sources': [source(path) for path in sources],
         'reference': source(reference / 'protocol.json'), 'profile': source(run_path(PROFILE) / 'profile_result.json')}, out)
    (out / 'source_snapshots').mkdir()
    for path in sources:
        shutil.copyfile(path, out / 'source_snapshots' / path.name)
    warmup = out / 'import_warmup'
    warmup.mkdir()
    data = dataset(warmup, plan)
    del data
    rows, passes = [], []
    expected = None
    begin = time.perf_counter()
    for repeat in range(repeats):
        order = ['original', optimized] if repeat % 2 == 0 else [optimized, 'original']
        for implementation in order:
            signatures, measured, wall = run_pass(out, plan, anchors, implementation, repeat)
            if expected is None:
                expected = signatures
                save(out / 'reference_signatures.json', signatures, out)
            if signatures != expected:
                save(out / ('mismatch_' + implementation + '_' + str(repeat) + '.json'), signatures, out)
                raise ValueError('Input, label, native trace, RNG or buffer differs.')
            rows.extend(measured)
            passes.append({'implementation': implementation, 'repetition': repeat,
                           'input_seconds': sum(row['input_seconds'] for row in measured),
                           'pass_wall_seconds': wall, 'exact': True})
            print('INPUT_PASS', implementation, repeat, passes[-1]['input_seconds'], flush=True)
    csv_write(out / 'timing.csv', rows)
    csv_write(out / 'throughput.csv', passes)
    baseline = sum(row['input_seconds'] for row in passes if row['implementation'] == 'original')
    fast = sum(row['input_seconds'] for row in passes if row['implementation'] == optimized)
    result = {'status': 'CPU_EXACT_INPUT_PASS', 'batches_per_pass': batches, 'repetitions': repeats,
              'completed_passes': len(passes), 'native_events_per_pass': sum(len(row['events']) for row in expected),
              'tensor_label_trace_RNG_buffer_equal': True, 'original_seconds': baseline,
              'optimized_seconds': fast, 'speedup_observed': baseline / fast,
              'reduction_fraction': 1 - fast / baseline, 'sampled_peak_RSS_bytes': max(row['RSS_bytes'] for row in rows),
              'CPU_wall_seconds': time.perf_counter() - begin, 'GPU_seconds': 0,
              'GPU_numerical_equivalence': 'NOT_EVALUATED', 'full_epoch_cost': 'NOT_REMEASURED',
              'scope': 'Fixed CPU input replay; not whole-epoch training throughput.'}
    save(out / 'input_acceptance.json', result, out)
    save(out / 'run.json', {'RunID': out.name, 'ExperimentID': EXPERIMENT,
         'mode': 'INPUT_OPTIMIZATION_CPU', **result}, out, replace=True)
    write_atomic(out / 'diary.md', '# Lossless input comparison\n\nAlternated original and bounded native-prefetch '
         'passes on fixed recorded anchors. Compared tensors, labels, native traces, RNG, cache order and occurrences. '
         'No detector or GPU work.\n', out)
    seal(out)
    print(json.dumps({'RunID': out.name, **result}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batches', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--implementation', choices=('prefetch2', 'arithmetic2'), default='prefetch2')
    args = parser.parse_args()
    if not 8 <= args.batches <= 64 or not 2 <= args.repeats <= 4:
        parser.error('Use 8..64 batches and 2..4 repetitions.')
    execute(args.batches, args.repeats, args.implementation)
