"""Portable, gated execution of the frozen matched training protocols."""
import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from uuid import uuid4

POST = Path(__file__).resolve().parent
PUBLIC = POST.parents[2]
ROOT = PUBLIC.parent
sys.path.insert(0, str(POST.parent))
sys.path.insert(0, str(PUBLIC / 'code'))


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(path.suffix + '.partial')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    temporary.rename(path)


def within(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('Path escapes the declared root.')
    return path


def verify_bundle(bundle):
    manifest = load(bundle / 'bundle_manifest.json')
    for relative, expected in manifest['files'].items():
        if digest(within(bundle, relative)) != expected:
            raise ValueError('Bundle input changed: ' + relative)
    return manifest


def resolve_data(bundle, data_root, verify=True):
    population = load(bundle / 'portable_dataset_manifest.json')
    resolved = {'train': [], 'val': []}
    for split, expected in (('train', 6471), ('val', 548)):
        if len(population[split]) != expected:
            raise ValueError('Incomplete frozen population.')
        for row in population[split]:
            image = within(bundle, row['image_override']) if row['image_override'] else within(data_root, row['image_relative'])
            annotation = within(data_root, row['annotation_relative'])
            label = within(bundle, row['label_asset'])
            if verify:
                for path, sha in ((image, row['image_SHA256']), (annotation, row['annotation_SHA256']), (label, row['label_SHA256'])):
                    if digest(path) != sha:
                        raise ValueError('Frozen data changed: ' + str(path))
            resolved[split].append({**row, 'image_path': str(image), 'annotation_path': str(annotation), 'label_path': str(label)})
        if len({row['image_id'] for row in resolved[split]}) != expected:
            raise ValueError('Duplicate population identity.')
    return resolved


def renderer():
    folder = PUBLIC / 'experiments/cv261109'
    original = sys.modules.pop('model', None)
    sys.path.insert(0, str(folder))
    try:
        spec = importlib.util.spec_from_file_location('vm37_portable_renderer', folder / 'generator.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(folder))
        sys.modules.pop('model', None)
        if original is not None:
            sys.modules['model'] = original
    from main import training_effects
    training_effects._renderer = module
    return module


def install_accumulator():
    path = PUBLIC / 'experiments/cv261207/development_fast_coco.py'
    spec = importlib.util.spec_from_file_location('new_fast_coco_accumulate', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules['new_fast_coco_accumulate'] = module


def matrix(protocol):
    return [(architecture, seed, arm) for architecture in protocol['architectures']
            for seed in protocol['seeds'] for arm in protocol['arms']]


def plan_arm(bundle, population, protocol, architecture, seed, arm, out):
    from study import expected_steps
    if (architecture, seed, arm) not in matrix(protocol):
        raise ValueError('Arm is outside the frozen matrix.')
    initial = next(row for row in load(bundle / 'portable_initializations.json')
                   if row['architecture'] == architecture and row['seed'] == seed)
    initialization = {'path': str(within(bundle, initial['asset'])), 'sha256': initial['sha256']}
    if digest(initialization['path']) != initial['sha256']:
        raise ValueError('Initialization changed.')
    args = load(bundle / 'ordinary_training_args.json')[architecture]
    args.update(protocol['shared_training_args'])
    args.update(model=initialization['path'], data=str(out / 'dataset.yaml'), epochs=protocol['epochs'], seed=seed,
                project=str(out), name='trainer', plots=False, exist_ok=False, cache=False, save=True,
                save_period=-1, resume=False)
    train = [{'image_id': r['image_id'], 'image_path': r['image_path'], 'image_SHA256': r['image_SHA256'],
              'derived_label_SHA256': r['label_SHA256']} for r in population['train']]
    development = [{'stem': r['image_id'], 'loader_path': r['image_path'], 'source_image_SHA256': r['image_SHA256'],
                    'derived_label_SHA256': r['label_SHA256']} for r in population['val']]
    return {'mode': 'PILOT20' if protocol['epochs'] == 20 else 'FORMAL100', 'scope': 'DEVELOPMENT_ONLY',
            'architecture': architecture, 'arm': arm, 'initialization': initialization,
            'COCO_source_SHA256': initial['COCO_source_SHA256'], 'args': args,
            'targeted_recipe': load(bundle / 'assets/TARGETED_POLICY.json') if arm == 'ALL_FIVE' else None,
            'train': train, 'development': development,
            'train_labels': {r['image_id']: r['label_path'] for r in population['train']},
            'val_labels': {r['image_id']: r['label_path'] for r in population['val']},
            'expected': expected_steps(6471, protocol['epochs']),
            'stop_after_completed_epochs': protocol['epochs'], 'library': '8.4.210',
            'selection': 'FIXED_FINAL_EPOCH_EMA', 'exact_resume_validated': False, 'bundle_path': str(bundle),
            'protocol_name': 'pilot20' if protocol['epochs'] == 20 else 'formal100',
            'augmentation_label': 'Broad stochastic ALL_FIVE, not proven Gap-guided.'}


def runtime(out, gpu=False):
    os.environ.update(CUDA_VISIBLE_DEVICES='0' if gpu else '', YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true',
                      PYTHONDONTWRITEBYTECODE='1')
    for key in ('YOLO_CONFIG_DIR', 'MPLCONFIGDIR', 'TORCH_HOME', 'TORCH_EXTENSIONS_DIR', 'CUDA_CACHE_PATH', 'TEMP', 'TMP'):
        path = out / 'runtime' / key
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    import tempfile
    tempfile.tempdir = str(out / 'runtime/TEMP')


def environment_check(bundle):
    import importlib.metadata
    expected = load(bundle / 'environment_observed.json')
    for name, version in expected['packages'].items():
        if importlib.metadata.version(name) != version:
            raise ValueError('Environment version differs: ' + name)
    if sys.version_info[:2] != (3, 11):
        raise ValueError('Python3.11 is required; revalidate a different environment separately.')
    package = Path(importlib.util.find_spec('ultralytics').origin).parent
    for relative, expected_sha in load(bundle / 'ultralytics_sources.json').items():
        if digest(package / relative) != expected_sha:
            raise ValueError('Installed official source changed: ' + relative)


def execution_remaining_seconds(authorization, charged=0., now=None):
    if not math.isfinite(charged) or charged < 0:
        raise ValueError('Invalid execution charge.')
    now = now or datetime.now(timezone.utc)
    consumed = sum(row['charged_seconds'] for row in authorization['completed_workers']) + charged
    return max(0., min(authorization['total_seconds'] - consumed,
                      (datetime.fromisoformat(authorization['absolute_deadline_UTC']) - now).total_seconds()))


def authorize(bundle, protocol, authorization, *, initial_admission=True):
    from safety import available_seconds
    if not authorization.get('approved') or authorization.get('protocol_id') != protocol['id']:
        raise PermissionError('A new explicit protocol-specific authorization is required.')
    manifest_sha = digest(bundle / 'bundle_manifest.json')
    if authorization.get('bundle_manifest_SHA256') != manifest_sha:
        raise PermissionError('Authorization does not identify this bundle.')
    if authorization.get('arms') != [list(row) for row in matrix(protocol)]:
        raise PermissionError('Authorization must cover complete matched arms.')
    if not all(authorization.get(key) is True for key in ('cooling_confirmed', 'power_confirmed', 'machine_admitted')):
        raise PermissionError('Machine readiness is unverified.')
    admission = authorization.get('resource_admission', {})
    if admission.get('status') != 'PASS' or admission.get('ALL_FIVE_GPU_cost_measured') is not True:
        raise PermissionError('Both-arm measured resource admission is missing.')
    required = admission.get('required_complete_matrix_seconds')
    if isinstance(required, bool) or not isinstance(required, (int, float)) or not math.isfinite(required) or required <= 0:
        raise PermissionError('Missing complete-matrix cost, including evaluation and safety margin.')
    cost_file = within(bundle, admission['cost_evidence_relative'])
    if digest(cost_file) != admission['cost_evidence_SHA256']:
        raise PermissionError('Resource evidence changed.')
    remaining = available_seconds(authorization)
    if remaining <= 0 or initial_admission and required > remaining:
        raise PermissionError('Complete matched matrix does not fit the unchanged deadline and budget.')
    return remaining


def training_worker(out):
    plan = load(out / 'protocol.json')
    bundle = Path(plan['bundle_path'])
    check_worker_authorization(out.parent, bundle, plan['protocol_name'])
    verify_bundle(bundle)
    environment_check(bundle)
    if digest(plan['initialization']['path']) != plan['initialization']['sha256']:
        raise ValueError('Initialization changed after matrix preparation.')
    save(out / 'WORKER_STARTED.json', {'PID': os.getpid(), 'UTC': datetime.now(timezone.utc).isoformat()})
    runtime(out, gpu=True)
    import gc
    import torch
    import cv2
    import yaml
    from unittest.mock import patch
    from ultralytics.utils import callbacks
    from post80_loader import read_only_sources
    from training import trainer_class, verify_input_preservation
    from main.labels import VIS_NAMES
    from main.training_health import model_health
    started = time.perf_counter()
    status = 'FAILED'
    trainer = None
    try:
        renderer()
        torch.set_num_threads(2)
        cv2.setNumThreads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable.')
        protected = []
        for rows, image_key, sha_key, labels, id_key in (
                (plan['train'], 'image_path', 'image_SHA256', plan['train_labels'], 'image_id'),
                (plan['development'], 'loader_path', 'source_image_SHA256', plan['val_labels'], 'stem')):
            for row in rows:
                for path, sha in ((row[image_key], row[sha_key]), (labels[row[id_key]], row['derived_label_SHA256'])):
                    if digest(path) != sha:
                        raise ValueError('Frozen training input changed.')
                    protected.append({'path': path, 'sha256': sha})
        for name, rows, key in (('train', plan['train'], 'image_path'), ('val', plan['development'], 'loader_path')):
            (out / (name + '_images.txt')).write_text('\n'.join(r[key] for r in rows) + '\n', encoding='utf-8')
        (out / 'dataset.yaml').write_text(yaml.safe_dump({'train': str(out / 'train_images.txt'),
             'val': str(out / 'val_images.txt'), 'names': dict(enumerate(VIS_NAMES))}), encoding='utf-8')
        original_backward = torch.autograd.backward
        def timed_backward(*args, **kwargs):
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            tick = time.perf_counter()
            value = original_backward(*args, **kwargs)
            end.record()
            torch.cuda.synchronize()
            trainer.backward_count += 1
            trainer.backward_seconds = time.perf_counter() - tick
            trainer.backward_CUDA_ms = begin.elapsed_time(end)
            return value
        with patch.object(callbacks, 'add_integration_callbacks', lambda instance: None), \
                patch.object(torch.autograd, 'backward', timed_backward), read_only_sources(out):
            trainer = trainer_class(plan, out)(overrides=plan['args'])
            trainer.train()
        for epoch in range(1, plan['args']['epochs'] + 1):
            ids = [identity for row in trainer.rows if row['epoch'] == epoch for identity in row['image_ids']]
            if Counter(ids) != Counter(row['image_id'] for row in plan['train']):
                raise ValueError('Epoch membership is incomplete: ' + str(epoch))
        expected = plan['expected']
        if not (len(trainer.rows) == trainer.forward_count == trainer.backward_count == expected['batches']):
            raise ValueError('Forward/backward counters differ.')
        if trainer.rq4_optimizer_updates != expected['updates'] or trainer.ema.updates != expected['updates']:
            raise ValueError('Optimizer/EMA counters differ.')
        model_health(trainer.model, gradients=True)
        model_health(trainer.ema.ema)
        weights = sorted((out / 'weights').glob('epoch*_fp32.pt'))
        if len(weights) != plan['args']['epochs'] or trainer.epoch + 1 != plan['args']['epochs']:
            raise ValueError('Final epoch or healthy snapshot count differs.')
        accepted = {'completed_epochs': trainer.epoch + 1, 'anchor_exposures': 6471 * (trainer.epoch + 1),
                    'forward_batches': trainer.forward_count, 'backward_batches': trainer.backward_count,
                    'optimizer_updates': trainer.rq4_optimizer_updates, 'EMA_updates': trainer.ema.updates,
                    'selected_checkpoint': {'path': str(trainer.rq4_selected), 'sha256': digest(trainer.rq4_selected)},
                    'epoch_checkpoints': [{'path': str(p), 'sha256': digest(p)} for p in weights],
                    'GPU_peak_bytes': trainer.peak, 'internal_validation': trainer.validations,
                    'pending_microbatches': expected['pending_microbatches'], 'training_wall_seconds': time.perf_counter() - started,
                    'research_AP': 'NOT_EVALUATED', 'exact_resume_validated': False, 'scope': 'DEVELOPMENT_ONLY'}
        verify_input_preservation(out, protected)
        save(out / 'training_acceptance.json', accepted)
        status = 'TRAINING_COMPLETED'
    except BaseException:
        save(out / 'failure.json', {'traceback': traceback.format_exc(), 'automatic_retry': False})
    finally:
        trainer = None
        gc.collect()
    save(out / 'worker_result.json', {'status': status, 'worker_seconds': time.perf_counter() - started})


def check_worker_authorization(parent, bundle, protocol_name):
    authorization_path = parent / 'authorization.json'
    ticket = load(bundle / 'EXECUTION_TICKET.json')
    if ticket['RunID'] != parent.name or ticket['authorization_SHA256'] != digest(authorization_path):
        raise PermissionError('Worker does not belong to the authorized attempt.')
    authorization = load(authorization_path)
    initial = ticket.get('initial_matrix_admission', {})
    required = authorization['resource_admission']['required_complete_matrix_seconds']
    if initial.get('status') != 'PASS' or initial.get('required_seconds') != required or initial.get('available_seconds', 0) < required:
        raise PermissionError('The complete matrix was not admitted before execution.')
    authorize(bundle, load(bundle / (protocol_name + '_protocol.json')), authorization, initial_admission=False)
    charged = sum(load(path)['worker_seconds'] for path in parent.glob('charge_*.json'))
    if execution_remaining_seconds(authorization, charged) <= 0:
        raise PermissionError('The original authorized worker budget is exhausted.')


def execute(bundle, data_root, protocol_name, authorization_path=None):
    began = time.perf_counter()
    manifest = verify_bundle(bundle)
    protocol = load(bundle / (protocol_name + '_protocol.json'))
    authorization = load(authorization_path) if authorization_path else None
    if authorization is not None:
        authorize(bundle, protocol, authorization)
    # Short run paths also work inside a relocated Windows bundle.
    parent = ROOT / 'datasets/local_runs' / ('vm37_' + uuid4().hex[:12])
    parent.mkdir(parents=True, exist_ok=False)
    runtime(parent, gpu=False)
    environment_check(bundle)
    population = resolve_data(bundle, data_root)
    plans = []
    for architecture, seed, arm in matrix(protocol):
        out = parent / (architecture + '_seed' + str(seed) + '_' + arm)
        out.mkdir()
        plan = plan_arm(bundle, population, protocol, architecture, seed, arm, out)
        save(out / 'protocol.json', plan)
        plans.append((out, plan))
    save(parent / 'inputs.json', {'bundle_manifest_SHA256': digest(bundle / 'bundle_manifest.json'),
         'protocol_SHA256': digest(bundle / (protocol_name + '_protocol.json')),
         'data_images': {key: len(value) for key, value in population.items()}, 'bundle_source': manifest['source_commit']})
    if authorization is None:
        save(parent / 'dry_run.json', {'status': 'DRY_RUN_PASS_NOT_GPU_VALIDATED', 'training_arms': len(plans),
             'GPU_seconds': 0, 'forward_backward': 0, 'wall_seconds': time.perf_counter() - began,
             'execution': 'BLOCKED_UNTIL_NEW_EXPLICIT_AUTHORIZATION_AND_MEASURED_RESOURCE_ADMISSION'})
        print(parent, flush=True)
        return parent
    from safety import run_owned, available_seconds
    ticket = bundle / 'EXECUTION_TICKET.json'
    admitted_seconds = authorize(bundle, protocol, authorization)
    shutil.copyfile(authorization_path, parent / 'authorization.json')
    save(ticket, {'RunID': parent.name, 'authorization_SHA256': digest(authorization_path),
                  'started_UTC': datetime.now(timezone.utc).isoformat(), 'retry_allowed': False,
                  'initial_matrix_admission': {'status': 'PASS', 'available_seconds': admitted_seconds,
                    'required_seconds': authorization['resource_admission']['required_complete_matrix_seconds']}})
    charged = 0.
    status = 'PARTIAL_WITH_LIMITATIONS'
    try:
        for out, plan in plans:
            remaining = execution_remaining_seconds(authorization, charged)
            if remaining < 60:
                raise TimeoutError('The original authorized deadline or budget has expired.')
            save(out / 'WORKER_LEASE.json', {'deadline_UTC': (datetime.now(timezone.utc) + timedelta(seconds=remaining)).isoformat()})
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2')
            result = run_owned([sys.executable, '-B', str(POST / 'portable.py'), '--worker', str(out)], out,
                               remaining, load(bundle / 'safety_policy.json'), env)
            charged += result['wall_seconds']
            save(parent / ('charge_' + out.name + '.json'), {'worker_seconds': result['wall_seconds'], 'cumulative_seconds': charged})
            if result['returncode'] or result['stop_reason'] or load(out / 'worker_result.json')['status'] != 'TRAINING_COMPLETED':
                raise RuntimeError('Training failed or stopped; no automatic retry or next arm.')
        # Run evaluation under the same total deadline, after all matched arms finish.
        eval_out = parent / 'evaluation'
        eval_out.mkdir()
        remaining = execution_remaining_seconds(authorization, charged)
        result = run_owned([sys.executable, '-B', str(POST / 'evaluate.py'), '--bundle', str(bundle),
                           '--data-root', str(data_root), '--training-parent', str(parent), '--protocol', protocol_name],
                          eval_out, remaining, load(bundle / 'safety_policy.json'), dict(os.environ, CUDA_VISIBLE_DEVICES='0'))
        charged += result['wall_seconds']
        if result['returncode'] or result['stop_reason'] or load(eval_out / 'evaluation_result.json')['status'] != 'COMPLETE':
            raise RuntimeError('Evaluation is incomplete.')
        status = 'MATCHED_MATRIX_COMPLETED_DEVELOPMENT_ONLY'
    except BaseException:
        save(parent / 'failure.json', {'traceback': traceback.format_exc(), 'automatic_retry': False})
    save(parent / 'execution_result.json', {'status': status, 'charged_worker_seconds': charged,
         'wall_seconds': time.perf_counter() - began, 'budget_reset': False, 'exact_training_resume': 'NOT_VALIDATED'})
    print(parent, flush=True)
    return parent


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--protocol', choices=('pilot20', 'formal100'), default='formal100')
    parser.add_argument('--authorization', type=Path)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    if args.worker:
        training_worker(args.worker.resolve(strict=True))
    elif args.bundle and args.data_root:
        execute(args.bundle.resolve(strict=True), args.data_root.resolve(strict=True), args.protocol, args.authorization)
    else:
        parser.error('Provide --bundle and --data-root; execution requires --authorization.')
