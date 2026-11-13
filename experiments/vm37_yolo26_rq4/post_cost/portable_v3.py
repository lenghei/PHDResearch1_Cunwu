"""Versioned optimized training with explicit server and resource admission."""
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback
from unittest.mock import patch

POST = Path(__file__).resolve().parent
sys.path.insert(0, str(POST))
import portable as base

VERSION = 'VM37_PORTABLE_V3.1'
KINDS = ('formal100', 'pilot20', 'benchmark26', 'benchmark8')


def protocol_for(bundle, kind):
    if kind not in KINDS:
        raise ValueError('Unknown execution protocol.')
    name = kind + '_protocol.json' if kind in ('formal100', 'pilot20') else kind + '.json'
    return base.load(bundle / name)


def host_identity(bundle):
    import cv2
    import importlib.metadata
    return {'hostname': platform.node(), 'system': platform.system(), 'machine': platform.machine(),
            'Python': platform.python_version(), 'OpenCV_build_SHA256': hashlib.sha256(cv2.getBuildInformation().encode()).hexdigest(),
            'environment': {name: importlib.metadata.version(name) for name in
                            base.load(bundle / 'environment_observed.json')['packages']}}


def implementation_files(bundle):
    return {name: base.digest(POST / name) for name in ('portable_v3.py', 'fast_input.py', 'input_ops.py')}


def finish_run(out, status, gpu_seconds=0):
    old = base.load(out / 'run.json')
    if old.get('RunID') != out.name or old.get('ExperimentID') != 'vm37_yolo26_rq4':
        raise ValueError('Run metadata does not identify the approved output directory.')
    temporary = out / 'run.json.partial'
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump({**old, 'status': status, 'GPU_seconds': gpu_seconds,
                   'completed_UTC': datetime.now(timezone.utc).isoformat()}, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(out / 'run.json')
    from common import seal
    seal(out)


@contextmanager
def output_scope(out):
    from study import save
    namespace = save.__globals__
    previous = namespace['ROOT']
    namespace['ROOT'] = out.resolve()
    try:
        yield
    finally:
        namespace['ROOT'] = previous


def verify_receipt(bundle, path):
    receipt = base.load(path)
    if receipt.get('status') != 'CPU_EXACT_PARITY_PASS' or receipt.get('GPU_seconds') != 0:
        raise PermissionError('A successful target-host CPU parity receipt is required.')
    if receipt.get('bundle_manifest_SHA256') != base.digest(bundle / 'bundle_manifest.json'):
        raise PermissionError('CPU receipt identifies another bundle.')
    if receipt.get('host') != host_identity(bundle) or receipt.get('implementation_SHA256') != implementation_files(bundle):
        raise PermissionError('Host or input implementation differs from the CPU receipt.')
    if set(receipt.get('tested_seeds', ())) != {42, 43, 44} or set(receipt.get('tested_arms', ())) != {'CONTROL', 'ALL_FIVE'}:
        raise PermissionError('CPU parity does not cover the frozen seeds and arms.')
    if not receipt.get('exact_tensors_labels_traces_RNG_and_order'):
        raise PermissionError('CPU parity is incomplete.')
    if set(receipt.get('tested_architectures', ())) != {'YOLO26s', 'YOLOv8s'} or receipt.get('tested_epoch_states') != [0, 1, 90]:
        raise PermissionError('CPU parity must cover both models and the frozen epoch/Mosaic states.')
    if receipt.get('batches_per_state') != 8 or receipt.get('repetitions') != 2:
        raise PermissionError('The target CPU comparison design differs.')
    if receipt.get('golden_pixel_checks') != 150 or receipt.get('healthy_initializations') != 6:
        raise PermissionError('Target initialization or golden evaluation pixels were not verified.')
    return receipt


def make_plan(bundle, population, protocol, architecture, seed, arm, out, kind, pipeline):
    if pipeline not in ('baseline', 'optimized'):
        raise ValueError('Unknown input implementation.')
    plan = base.plan_arm(bundle, population, protocol, architecture, seed, arm, out)
    completed = 1 if kind.startswith('benchmark') else protocol['epochs']
    from study import expected_steps
    plan.update(v3_protocol=kind, executor_version=VERSION, input_pipeline=pipeline,
                execution_epochs=completed, stop_after_completed_epochs=completed,
                expected=expected_steps(6471, protocol['epochs'], stop_epochs=completed),
                input_implementation_SHA256=implementation_files(bundle),
                benchmark_only=kind.startswith('benchmark'))
    if kind.startswith('benchmark'):
        plan.update(mode='TARGET_SERVER_EPOCH1_COST_ONLY', selection='EPOCH1_EMA_ENGINEERING_COST_ONLY')
    return plan


def trainer_factory(plan, out):
    from training import trainer_class
    Parent = trainer_class(plan, out)
    class Trainer(Parent):
        def build_dataset(self, *args, **kwargs):
            value = super().build_dataset(*args, **kwargs)
            if value.augment and plan['input_pipeline'] == 'optimized':
                from fast_input import ArithmeticPrefetchDataset
                value = ArithmeticPrefetchDataset.adopt(value, native_threads=2)
            return value

        def train(self):
            try:
                return super().train()
            finally:
                loader = getattr(self, 'train_loader', None)
                data = getattr(loader, 'dataset', None)
                if hasattr(data, 'close'):
                    data.close()
    return Trainer


def authorize(bundle, kind, auth, receipt, *, initial_admission=True):
    protocol = protocol_for(bundle, kind)
    if auth.get('approved') is not True or auth.get('protocol_id') != protocol['id']:
        raise PermissionError('New protocol-specific author approval is required.')
    if auth.get('authorization_scope') != 'NEW_TARGET_SERVER_ONLY':
        raise PermissionError('Historical local GPU allocations cannot authorize server execution.')
    if auth.get('bundle_manifest_SHA256') != base.digest(bundle / 'bundle_manifest.json'):
        raise PermissionError('Authorization identifies another bundle.')
    if auth.get('input_pipeline') not in ('baseline', 'optimized'):
        raise PermissionError('One input implementation must be frozen for the whole comparison.')
    if auth.get('arms') != [list(row) for row in base.matrix(protocol)]:
        raise PermissionError('The complete matched matrix must be authorized.')
    if not all(auth.get(key) is True for key in ('cooling_confirmed', 'power_confirmed', 'machine_admitted')):
        raise PermissionError('Machine readiness has not been confirmed.')
    if auth.get('CPU_receipt_SHA256') != base.digest(receipt):
        raise PermissionError('Authorization does not bind the target-host CPU receipt.')
    verify_receipt(bundle, receipt)
    remaining = base.execution_remaining_seconds(auth)
    if remaining < 60:
        raise PermissionError('The authorized deadline or budget has expired.')
    if kind.startswith('benchmark'):
        if auth.get('purpose') != 'TARGET_SERVER_FULL_EPOCH_COST_ONLY':
            raise PermissionError('Benchmark-only authorization required.')
        cap = auth.get('per_arm_hard_seconds')
        if isinstance(cap, bool) or not isinstance(cap, (int, float)) or not 60 <= cap <= 7200:
            raise PermissionError('Benchmark per-arm cap must be 60..7200 seconds.')
        if initial_admission and 2 * cap > remaining:
            raise PermissionError('Both benchmark arms do not fit the unchanged authorization.')
    else:
        base.authorize(bundle, protocol, auth, initial_admission=initial_admission)
        costs = auth.get('resource_admission', {}).get('measured_architectures', [])
        if not set(protocol['architectures']).issubset(costs):
            raise PermissionError('Target-server cost evidence is missing for an architecture.')
        if auth['resource_admission'].get('evaluation_and_bootstrap_included') is not True:
            raise PermissionError('Full research evaluation and bootstrap must fit the admitted budget.')
    return remaining


def remaining(auth, began):
    elapsed = time.perf_counter() - began
    return max(0., min(base.execution_remaining_seconds(auth),
                      auth['total_seconds'] - sum(row['charged_seconds'] for row in auth['completed_workers']) - elapsed))


def execution_admission(bundle, kind, auth, receipt, began):
    admitted = min(authorize(bundle, kind, auth, receipt), remaining(auth, began))
    required = (2 * auth['per_arm_hard_seconds'] if kind.startswith('benchmark')
                else auth['resource_admission']['required_complete_matrix_seconds'])
    if admitted < required:
        raise PermissionError('Preflight consumed the reserve needed for the complete matched matrix.')
    return admitted


def check_worker(out):
    plan = base.load(out / 'protocol.json')
    parent = out.parent
    bundle = Path(plan['bundle_path'])
    ticket = base.load(parent / 'execution_ticket.json')
    if ticket != base.load(bundle / ('EXECUTION_TICKET_V3_' + plan['v3_protocol'] + '.json')) or ticket['RunID'] != parent.name:
        raise PermissionError('Worker is not bound to the single authorized execution ticket.')
    if ticket['protocol_SHA256'][out.name] != base.digest(out / 'protocol.json'):
        raise PermissionError('Worker plan changed after admission.')
    if ticket['authorization_SHA256'] != base.digest(parent / 'authorization.json'):
        raise PermissionError('Worker authorization changed.')
    auth = base.load(parent / 'authorization.json')
    if ticket['bundle_manifest_SHA256'] != base.digest(bundle / 'bundle_manifest.json'):
        raise PermissionError('Worker bundle changed.')
    if base.execution_remaining_seconds(auth) <= 0:
        raise PermissionError('Expired allocation.')
    if datetime.now(timezone.utc) >= datetime.fromisoformat(base.load(out / 'WORKER_LEASE.json')['deadline_UTC']):
        raise PermissionError('Worker lease has expired.')
    if plan['input_pipeline'] != auth['input_pipeline']:
        raise PermissionError('Input implementation changed between arms.')
    authorize(bundle, plan['v3_protocol'], auth, parent / 'CPU_receipt.json', initial_admission=False)
    initial = ticket.get('initial_matrix_admission', {})
    if initial.get('status') != 'PASS' or initial.get('available_seconds', 0) < initial.get('required_seconds', float('inf')):
        raise PermissionError('The complete matrix was not admitted before execution.')
    return plan, bundle


def check_evaluation_authorization(parent, bundle, kind):
    if kind not in ('formal100', 'pilot20'):
        raise PermissionError('Engineering benchmarks do not authorize research evaluation.')
    ticket = base.load(parent / 'execution_ticket.json')
    if ticket != base.load(bundle / ('EXECUTION_TICKET_V3_' + kind + '.json')) or ticket['RunID'] != parent.name:
        raise PermissionError('Evaluation is not bound to the v3 execution ticket.')
    if ticket['authorization_SHA256'] != base.digest(parent / 'authorization.json'):
        raise PermissionError('Evaluation authorization changed.')
    if ticket['bundle_manifest_SHA256'] != base.digest(bundle / 'bundle_manifest.json'):
        raise PermissionError('Evaluation bundle changed.')
    for name, expected in ticket['protocol_SHA256'].items():
        if base.digest(parent / name / 'protocol.json') != expected:
            raise PermissionError('Training plan changed before evaluation.')
    authorize(bundle, kind, base.load(parent / 'authorization.json'), parent / 'CPU_receipt.json', initial_admission=False)
    initial = ticket.get('initial_matrix_admission', {})
    if initial.get('status') != 'PASS' or initial.get('available_seconds', 0) < initial.get('required_seconds', float('inf')):
        raise PermissionError('The complete study was not initially admitted.')
    lease = base.load(parent / 'evaluation/WORKER_LEASE.json')
    if datetime.now(timezone.utc) >= datetime.fromisoformat(lease['deadline_UTC']):
        raise PermissionError('Evaluation lease has expired.')


def training_worker(out):
    plan, bundle = check_worker(out)
    base.verify_bundle(bundle)
    base.environment_check(bundle)
    if base.digest(plan['initialization']['path']) != plan['initialization']['sha256']:
        raise ValueError('Initialization changed.')
    base.save(out / 'WORKER_STARTED.json', {'PID': os.getpid(), 'UTC': datetime.now(timezone.utc).isoformat()})
    base.runtime(out, gpu=True)
    import torch
    import cv2
    import yaml
    from ultralytics.utils import callbacks
    from post80_loader import read_only_sources
    from training import verify_input_preservation
    from main.labels import VIS_NAMES
    from main.training_health import model_health
    started = time.perf_counter()
    trainer = None
    status = 'FAILED'
    try:
        base.renderer()
        torch.set_num_threads(2)
        cv2.setNumThreads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable.')
        protected = [plan['initialization']]
        for rows, image_key, sha_key, labels, id_key in (
                (plan['train'], 'image_path', 'image_SHA256', plan['train_labels'], 'image_id'),
                (plan['development'], 'loader_path', 'source_image_SHA256', plan['val_labels'], 'stem')):
            for row in rows:
                for path, sha in ((row[image_key], row[sha_key]), (labels[row[id_key]], row['derived_label_SHA256'])):
                    if base.digest(path) != sha:
                        raise ValueError('Frozen input changed.')
                    protected.append({'path': path, 'sha256': sha})
        for name, rows, key in (('train', plan['train'], 'image_path'), ('val', plan['development'], 'loader_path')):
            (out / (name + '_images.txt')).write_text('\n'.join(row[key] for row in rows) + '\n', encoding='utf-8')
        (out / 'dataset.yaml').write_text(yaml.safe_dump({'train': str(out / 'train_images.txt'),
             'val': str(out / 'val_images.txt'), 'names': dict(enumerate(VIS_NAMES))}), encoding='utf-8')
        backward = torch.autograd.backward
        def timed_backward(*args, **kwargs):
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            tick = time.perf_counter()
            result = backward(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            trainer.backward_count += 1
            trainer.backward_seconds = time.perf_counter() - tick
            trainer.backward_CUDA_ms = begin.elapsed_time(end)
            return result
        with patch.object(callbacks, 'add_integration_callbacks', lambda instance: None), \
                patch.object(torch.autograd, 'backward', timed_backward), output_scope(out), read_only_sources(out):
            trainer = trainer_factory(plan, out)(overrides=plan['args'])
            trainer.train()
        for epoch in range(1, plan['execution_epochs'] + 1):
            identities = [identity for row in trainer.rows if row['epoch'] == epoch for identity in row['image_ids']]
            if Counter(identities) != Counter(row['image_id'] for row in plan['train']):
                raise ValueError('Incomplete epoch membership.')
        if not len(trainer.rows) == trainer.forward_count == trainer.backward_count == plan['expected']['batches']:
            raise ValueError('Forward/backward count mismatch.')
        if trainer.rq4_optimizer_updates != plan['expected']['updates'] or trainer.ema.updates != plan['expected']['updates']:
            raise ValueError('Optimizer/EMA count mismatch.')
        model_health(trainer.model, gradients=True)
        model_health(trainer.ema.ema)
        weights = sorted((out / 'weights').glob('epoch*_fp32.pt'))
        if len(weights) != plan['execution_epochs'] or trainer.epoch + 1 != plan['execution_epochs']:
            raise ValueError('Epoch/checkpoint count mismatch.')
        with output_scope(out):
            verify_input_preservation(out, protected)
        base.save(out / 'training_acceptance.json', {
            'completed_epochs': trainer.epoch + 1, 'anchor_exposures': 6471 * (trainer.epoch + 1),
            'forward_batches': trainer.forward_count, 'backward_batches': trainer.backward_count,
            'optimizer_updates': trainer.rq4_optimizer_updates, 'EMA_updates': trainer.ema.updates,
            'selected_checkpoint': {'path': str(trainer.rq4_selected), 'sha256': base.digest(trainer.rq4_selected)},
            'epoch_checkpoints': [{'path': str(path), 'sha256': base.digest(path)} for path in weights],
            'GPU_peak_bytes': trainer.peak, 'internal_validation': trainer.validations,
            'training_wall_seconds': time.perf_counter() - started,
            'loader_seconds': sum(row['loader_seconds'] for row in trainer.rows),
            'input_pipeline': plan['input_pipeline'], 'benchmark_only': plan['benchmark_only'],
            'research_AP': 'NOT_EVALUATED', 'exact_resume_validated': False, 'scope': 'DEVELOPMENT_ONLY'})
        status = 'TRAINING_COMPLETED'
    except BaseException:
        base.save(out / 'failure.json', {'traceback': traceback.format_exc(), 'automatic_retry': False})
    finally:
        trainer = None
        gc.collect()
    base.save(out / 'worker_result.json', {'status': status, 'worker_seconds': time.perf_counter() - started})
    return status


def execute(bundle, data_root, kind, output_root, pipeline='optimized', authorization=None, receipt=None):
    began = time.perf_counter()
    manifest = base.verify_bundle(bundle)
    protocol = protocol_for(bundle, kind)
    output_root = output_root.resolve()
    if any(output_root.is_relative_to(path.resolve()) for path in (data_root, bundle / 'main', bundle / 'assets')):
        raise ValueError('Run outputs cannot be written into source code, frozen assets or the raw dataset.')
    auth = base.load(authorization) if authorization else None
    if auth is not None:
        if receipt is None:
            raise PermissionError('A target-host CPU receipt is required.')
        authorize(bundle, kind, auth, receipt)
        if pipeline != auth['input_pipeline']:
            raise PermissionError('CLI implementation differs from authorization.')
    from main.pipeline import new_run
    parent = new_run({'output_root': str(output_root)}, 'vm37_yolo26_rq4', 'portable_v3_' + kind)[0]
    base.runtime(parent, gpu=False)
    base.environment_check(bundle)
    population = base.resolve_data(bundle, data_root)
    plans = []
    for architecture, seed, arm in base.matrix(protocol):
        out = parent / (architecture + '_seed' + str(seed) + '_' + arm)
        out.mkdir()
        plan = make_plan(bundle, population, protocol, architecture, seed, arm, out, kind, pipeline)
        base.save(out / 'protocol.json', plan)
        plans.append((out, plan))
    base.save(parent / 'plans.json', {'protocol': protocol['id'], 'matrix': base.matrix(protocol),
        'input_pipeline': pipeline, 'implementation_SHA256': implementation_files(bundle),
        'training_images': 6471, 'development_images': 548, 'expected_research_configs':
        0 if kind.startswith('benchmark') else len(plans) * 2 * len(base.load(bundle /
        ('formal_evaluation_manifest.json' if kind == 'formal100' else 'evaluation_manifest.json'))['contexts'])})
    if auth is None:
        base.save(parent / 'dry_run.json', {'status': 'CPU_DRY_RUN_PASS_GPU_NOT_RUN',
            'training_arms': len(plans), 'GPU_seconds': 0, 'execution': 'NEW_EXPLICIT_AUTHORIZATION_REQUIRED',
            'wall_seconds': time.perf_counter() - began})
        (parent / 'diary.md').write_text('# Portable v3 dry-run\n\nVerified all frozen inputs and prepared the complete '
            'matched plans. No GPU, training or metric evaluation.\n', encoding='utf-8')
        (parent / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>Portable v3 dry-run</h1>'
            '<p>CPU plans verified. GPU execution requires new authorization.</p><a href="plans.json">Plans</a>', encoding='utf-8')
        finish_run(parent, 'CPU_DRY_RUN_PASS_GPU_NOT_RUN')
        print(parent, flush=True)
        return parent
    admitted = execution_admission(bundle, kind, auth, receipt, began)
    ticket_path = bundle / ('EXECUTION_TICKET_V3_' + kind + '.json')
    lock = bundle / '.vm37_gpu.lock'
    handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(handle, json.dumps({'RunID': parent.name, 'PID': os.getpid()}).encode())
    os.close(handle)
    status = 'PARTIAL_WITH_LIMITATIONS'
    try:
        shutil.copyfile(authorization, parent / 'authorization.json')
        shutil.copyfile(receipt, parent / 'CPU_receipt.json')
        ticket = {'RunID': parent.name, 'authorization_SHA256': base.digest(parent / 'authorization.json'),
                  'bundle_manifest_SHA256': base.digest(bundle / 'bundle_manifest.json'),
                  'protocol_SHA256': {out.name: base.digest(out / 'protocol.json') for out, _ in plans},
                  'UTC': datetime.now(timezone.utc).isoformat(), 'automatic_retry': False,
                  'initial_matrix_admission': {'status': 'PASS', 'available_seconds': admitted,
                     'required_seconds': 2 * auth['per_arm_hard_seconds'] if kind.startswith('benchmark')
                     else auth['resource_admission']['required_complete_matrix_seconds']}}
        base.save(ticket_path, ticket)
        base.save(parent / 'execution_ticket.json', ticket)
        from safety import run_owned, query
        for out, plan in plans:
            left = remaining(auth, began)
            cap = min(left, auth['per_arm_hard_seconds']) if kind.startswith('benchmark') else left
            if cap < 60:
                raise TimeoutError('The unchanged execution budget is exhausted.')
            if query()['utilization_percent'] > 10:
                raise RuntimeError('GPU is busy; no competing worker is admitted.')
            from datetime import timedelta
            base.save(out / 'WORKER_LEASE.json', {'deadline_UTC':
                (datetime.now(timezone.utc) + timedelta(seconds=cap)).isoformat()})
            result = run_owned([sys.executable, '-B', str(POST / 'portable_v3.py'), '--worker', str(out)],
                out, cap, base.load(bundle / 'safety_policy.json'),
                dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2'))
            base.save(parent / ('charge_' + out.name + '.json'), result)
            if result['returncode'] or result['stop_reason'] or base.load(out / 'worker_result.json')['status'] != 'TRAINING_COMPLETED':
                raise RuntimeError('Worker failed; no retry or next arm.')
        if not kind.startswith('benchmark'):
            evaluation = parent / 'evaluation'
            evaluation.mkdir()
            left = remaining(auth, began)
            if left < 60:
                raise TimeoutError('No admitted research evaluation time remains.')
            base.save(evaluation / 'WORKER_LEASE.json', {'deadline_UTC':
                (datetime.now(timezone.utc) + timedelta(seconds=left)).isoformat()})
            result = run_owned([sys.executable, '-B', str(POST / 'evaluate_v3.py'), '--bundle', str(bundle),
                 '--data-root', str(data_root), '--training-parent', str(parent), '--protocol', kind],
                 evaluation, left, base.load(bundle / 'safety_policy.json'),
                 dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1'))
            if result['returncode'] or result['stop_reason'] or base.load(evaluation / 'evaluation_result.json')['status'] != 'COMPLETE':
                raise RuntimeError('Research evaluation is incomplete.')
        status = 'BENCHMARK_PAIR_COMPLETED_ENGINEERING_ONLY' if kind.startswith('benchmark') else 'MATCHED_MATRIX_COMPLETED_DEVELOPMENT_ONLY'
    except BaseException:
        base.save(parent / 'failure.json', {'traceback': traceback.format_exc(), 'automatic_retry': False})
    finally:
        if base.load(lock)['RunID'] == parent.name:
            lock.unlink()
    base.save(parent / 'execution_result.json', {'status': status, 'wall_seconds': time.perf_counter() - began,
        'budget_reset': False, 'exact_resume': 'NOT_VALIDATED'})
    finish_run(parent, status, gpu_seconds=None)
    print(parent, flush=True)
    return parent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--protocol', choices=KINDS, default='formal100')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--input-pipeline', choices=('baseline', 'optimized'), default='optimized')
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--cpu-receipt', type=Path)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(0 if training_worker(args.worker.resolve(strict=True)) == 'TRAINING_COMPLETED' else 1)
    if not args.bundle or not args.data_root:
        parser.error('Provide --bundle and --data-root. Execution also requires new authorization and a CPU receipt.')
    bundle = args.bundle.resolve(strict=True)
    execute(bundle, args.data_root.resolve(strict=True), args.protocol,
            args.output_root.resolve() if args.output_root else bundle / 'datasets/local_runs/vm37_yolo26_rq4',
            args.input_pipeline, args.authorization, args.cpu_receipt)
