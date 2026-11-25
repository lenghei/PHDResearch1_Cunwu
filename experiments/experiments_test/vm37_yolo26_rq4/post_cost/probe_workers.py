"""Test Windows loader startup and trace identity without changing workers0."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

POST = Path(__file__).resolve().parent
sys.path.insert(0, str(POST))
from profile_input import REFERENCE, csv_write, dataset, setup
from study import *


class ProbeDataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        import hashlib
        import torch
        value = self.data[i]
        events = self.data.effect_events.copy()
        self.data.effect_events.clear()
        worker = torch.utils.data.get_worker_info()
        return {'id': Path(value['im_file']).stem,
                'tensor_SHA256': hashlib.sha256(value['img'].numpy().tobytes()).hexdigest(),
                'events': events, 'worker': None if worker is None else worker.id,
                'torch_initial_seed': torch.initial_seed(), 'epoch': self.data.epoch_number}


def identity_collate(values):
    return values


def worker_init(identity):
    import cv2
    import torch
    from ultralytics.data.build import seed_worker
    from portable import renderer
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    renderer()
    seed_worker(identity)


def child(out, workers):
    setup(out)
    import psutil
    import torch
    from ultralytics.utils.torch_utils import init_seeds
    plan = load(run_path(REFERENCE) / 'protocol.json')
    init_seeds(42, deterministic=True)
    data = dataset(out, plan)
    lookup = {Path(path).stem: i for i, path in enumerate(data.im_files)}
    records = [json.loads(line) for line in (run_path(REFERENCE) / 'training_batches.jsonl').read_text().splitlines()[:4]]
    identities = [identity for row in records for identity in row['image_ids']] * 2
    indices = [lookup[identity] for identity in identities]
    probe = ProbeDataset(data)
    loader = torch.utils.data.DataLoader(probe, batch_size=2, sampler=indices, num_workers=workers,
        collate_fn=identity_collate, generator=torch.Generator().manual_seed(42),
        worker_init_fn=worker_init, pin_memory=False,
        **({'prefetch_factor': 2, 'multiprocessing_context': 'spawn'} if workers else {}))
    started = time.perf_counter()
    iterator = iter(loader)
    pids = [process.pid for process in getattr(iterator, '_workers', ())]
    result = []
    peak = 0
    try:
        for batch in iterator:
            result.extend(batch)
            processes = [psutil.Process()] + [psutil.Process(pid) for pid in pids if psutil.pid_exists(pid)]
            peak = max(peak, sum(process.memory_info().rss for process in processes))
    finally:
        if hasattr(iterator, '_shutdown_workers'):
            iterator._shutdown_workers()
    alive = [pid for pid in pids if psutil.pid_exists(pid)]
    if alive or [row['id'] for row in result] != identities:
        raise RuntimeError('Worker shutdown or anchor order failed.')
    save(out / 'worker_results.json', {'workers': workers, 'status': 'COMPLETED',
        'wall_seconds': time.perf_counter() - started, 'images': len(result), 'anchor_order_equal': True,
        'sampled_total_RSS_bytes': peak, 'worker_PIDs': pids, 'worker_processes_alive_after_shutdown': alive,
        'records': result, 'scientific_worker_admission': False, 'GPU_seconds': 0}, out)


def execute():
    out = create('loader_workers_cpu')
    runtime(out, cuda=False)
    import psutil
    memory = psutil.virtual_memory()
    save(out / 'protocol.json', {'CPU_only': True, 'worker_candidates': [0, 1, 2], 'prefetch_factor': 2,
         'fixed_anchors': 8, 'passes_in_sequence': 2, 'seed': 42, 'child_hard_seconds': 180,
         'RAM_available_bytes': memory.available, 'scientific_workers_unchanged': 0,
         'epoch_propagation': 'Production set_epoch mutates the parent dataset, not spawned worker copies.'}, out)
    if memory.available < 4 * 1024**3:
        raise RuntimeError('Not enough RAM for the bounded worker investigation.')
    rows = []
    for count in (0, 1, 2):
        folder = out / ('workers' + str(count))
        folder.mkdir()
        command = [sys.executable, '-B', '-X', 'utf8', str(Path(__file__)), '--child', str(folder), '--workers', str(count)]
        with (folder / 'execution.log').open('x', encoding='utf-8') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2'))
            try:
                code = process.wait(timeout=180)
            except subprocess.TimeoutExpired:
                from safety import stop_owned
                stop_owned(process)
                code = -1
        if code or not (folder / 'worker_results.json').exists():
            rows.append({'workers': count, 'status': 'FAILED', 'wall_seconds': None,
                         'sampled_total_RSS_bytes': None, 'anchor_order_equal': False,
                         'pixel_trace_equal_workers0': False})
            continue
        value = load(folder / 'worker_results.json')
        if count == 0:
            reference = value
        equal = all(a['tensor_SHA256'] == b['tensor_SHA256'] and a['events'] == b['events']
                    for a, b in zip(reference['records'], value['records']))
        rows.append({'workers': count, 'status': 'ENGINEERING_COMPLETED',
             'wall_seconds': value['wall_seconds'], 'sampled_total_RSS_bytes': value['sampled_total_RSS_bytes'],
             'anchor_order_equal': value['anchor_order_equal'], 'pixel_trace_equal_workers0': equal})
    csv_write(out / 'worker_comparison.csv', rows)
    save(out / 'worker_admission.json', {'status': 'RETAIN_WORKERS0', 'measurements': rows,
         'reason': 'Spawned workers receive different RNG seeds, independent occurrence/cache state, and no verified epoch propagation.',
         'protocol_changed': False, 'GPU_seconds': 0}, out)
    save(out / 'inputs.json', {'sources': [source(Path(__file__)), source(PUBLIC / 'code/main/training_data.py'),
         source(Path(sys.executable).parents[1] / 'Lib/site-packages/ultralytics/data/build.py')],
         'original_protocol': source(run_path(REFERENCE) / 'protocol.json')}, out)
    write_atomic(out / 'diary.md', '# Windows loader probe\n\nCompared fixed anchor ordering, actual native traces, '
         'tensor hashes, process seeds, RAM and child shutdown with workers0/1/2. No training. '
         'Retained the accepted workers0 protocol.\n', out)
    save(out / 'run.json', {'RunID': out.name, 'ExperimentID': EXPERIMENT,
         'status': 'COMPLETE_WITH_LIMITATIONS', 'GPU_seconds': 0}, out, replace=True)
    seal(out)
    print(json.dumps({'RunID': out.name, 'results': rows}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--child')
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()
    if args.child:
        child(Path(args.child), args.workers)
    else:
        execute()
