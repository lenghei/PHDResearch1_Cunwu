"""A separately authorized local ten-epoch matched pilot."""
import argparse
from collections import Counter
from datetime import datetime, timezone, timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'post_cost'))
import study
import portable_v3 as v3
import evaluate as research
from safety import run_owned, query

BUNDLE_RUN = 'vm37_yolo26_rq4_portable_v3_build_cpu_20261113T115150_6e2bcebf'
ID = 'PILOT_YOLO26S_10E_SEED42'
ROOT = study.ROOT
SOURCE = Path(__file__).resolve()


def load(path):
    return v3.base.load(path)


def save(path, value):
    v3.base.save(path, value)


def bundle():
    return study.run_path(BUNDLE_RUN) / 'bundle'


def charge_files(authorization):
    return sorted((authorization / 'charges').glob('*.json'))


def available(authorization, now=None):
    spec = load(authorization / 'authorization.json')
    used = sum(load(p)['charged_seconds'] for p in charge_files(authorization))
    now = now or datetime.now(timezone.utc)
    return max(0., min(spec['max_cumulative_GPU_worker_wall_seconds'] - used,
                      (datetime.fromisoformat(spec['absolute_expiry_UTC']) - now).total_seconds()))


def verify_protocol(spec):
    expected = {'id': ID, 'epochs': 10, 'schedule_horizon_epochs': 10,
                'architectures': ['YOLO26s'], 'seeds': [42], 'arms': ['CONTROL', 'ALL_FIVE'],
                'training_images': 6471, 'development_images': 548}
    if any(spec.get(k) != value for k, value in expected.items()):
        raise ValueError('Pilot identity, population or ten-epoch horizon changed.')
    shared = spec['shared_training_args']
    for key, value in {'imgsz': 1280, 'batch': 2, 'nbs': 64, 'optimizer': 'AdamW', 'lr0': .01,
                       'warmup_epochs': 3., 'amp': False, 'workers': 0, 'cos_lr': True,
                       'close_mosaic': 10, 'resume': False}.items():
        if shared.get(key) != value:
            raise ValueError('Frozen pilot training argument changed: ' + key)
    if spec['planned_counters_per_arm'] != study.expected_steps(6471, 10):
        raise ValueError('Pilot exposure/update plan differs.')


def source_paths():
    paths = [SOURCE]
    names = ('portable_v3.py', 'portable.py', 'evaluate.py', 'fast_input.py', 'input_ops.py', 'safety.py')
    paths += [HERE / 'post_cost' / n for n in names]
    paths += [HERE / 'training.py', HERE / 'study.py']
    paths += [study.PUBLIC / 'code/main' / n for n in
              ('training_data.py', 'training_effects.py', 'training_health.py', 'statistics.py',
               'labels.py', 'predictor_stable.py')]
    return paths


def verify_freeze(authorization):
    frozen = load(authorization / 'execution_freeze.json')
    for path, digest in frozen['files'].items():
        if study.sha256(Path(path)) != digest:
            raise ValueError('Frozen source or authorization changed: ' + path)
    verify_protocol(load(authorization / 'pilot10_protocol.json'))
    if study.sha256(study.RUNS / 'compute_allocation.json') != load(authorization / 'authorization.json')['old_ledger_SHA256_READ_ONLY']:
        raise ValueError('Historical GPU ledger changed; inspect before proceeding.')
    return frozen


def plan(authorization, out, arm, population, smoke=False):
    spec = load(authorization / 'pilot10_protocol.json')
    value = v3.make_plan(bundle(), population, spec, 'YOLO26s', 42, arm, out, 'pilot10', 'optimized')
    value.update(mode='PILOT10_ENTRYPOINT_GPU_SMOKE' if smoke else ID,
                 scope='CANARY_ENGINEERING_ONLY / DEVELOPMENT_ONLY' if smoke else 'EXPLORATORY_SINGLE_SEED_DEVELOPMENT_ONLY',
                 protocol_name='pilot10', selection='FIXED_EPOCH10_EMA', pilot_authorization=str(authorization))
    if smoke:
        ids = load(bundle() / 'cpu_comparison_manifest.json')['image_ids']
        by_id = {r['image_id']: r for r in value['train']}
        value['train'] = [by_id[identity] for identity in ids]
        value['development'] = value['development'][:4]
        value['train_labels'] = {r['image_id']: value['train_labels'][r['image_id']] for r in value['train']}
        value['val_labels'] = {r['stem']: value['val_labels'][r['stem']] for r in value['development']}
        value.update(execution_epochs=1, stop_after_completed_epochs=1,
                     expected=study.expected_steps(len(ids), 10, stop_epochs=1),
                     selection='EPOCH1_ENGINEERING_ONLY_NEVER_INITIALIZE_RESEARCH')
    return value


def worker_context(out):
    ticket = load(out / 'worker_ticket.json')
    authorization = Path(ticket['authorization'])
    verify_freeze(authorization)
    if study.sha256(out / 'protocol.json') != ticket['protocol_SHA256'] or available(authorization) <= 0:
        raise PermissionError('Worker protocol changed or its new budget expired.')
    if ticket.get('evaluation_stage_SHA256') and study.sha256(out / 'evaluation_stage.json') != ticket['evaluation_stage_SHA256']:
        raise PermissionError('Frozen evaluation stage changed.')
    if datetime.now(timezone.utc) >= datetime.fromisoformat(load(out / 'WORKER_LEASE.json')['deadline_UTC']):
        raise PermissionError('Worker lease expired.')
    if not out.resolve().is_relative_to((ROOT / 'datasets/local_runs').resolve()):
        raise PermissionError('Pilot outputs must remain in the accepted local run root.')
    return load(out / 'protocol.json'), authorization


def train_worker(out):
    value, authorization = worker_context(out)
    study.runtime(out, cuda=True)
    v3.base.verify_bundle(bundle())
    v3.base.environment_check(bundle())
    import torch
    import cv2
    import yaml
    from ultralytics.utils import callbacks
    from post80_loader import read_only_sources
    from training import verify_input_preservation
    from main.labels import VIS_NAMES
    from main.training_health import model_health
    began = time.perf_counter()
    trainer = None
    status = 'FAILED'
    save(out / 'WORKER_STARTED.json', {'UTC': datetime.now(timezone.utc).isoformat(), 'PID': os.getpid()})
    try:
        v3.base.renderer()
        torch.set_num_threads(2)
        cv2.setNumThreads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable.')
        protected = [value['initialization']]
        for rows, key, digest_key, labels, identity in (
            (value['train'], 'image_path', 'image_SHA256', value['train_labels'], 'image_id'),
            (value['development'], 'loader_path', 'source_image_SHA256', value['val_labels'], 'stem')):
            for row in rows:
                for path, digest in ((row[key], row[digest_key]), (labels[row[identity]], row['derived_label_SHA256'])):
                    if study.sha256(Path(path)) != digest:
                        raise ValueError('Frozen training input changed.')
                    protected.append({'path': path, 'sha256': digest})
        for name, rows, key in (('train', value['train'], 'image_path'), ('val', value['development'], 'loader_path')):
            (out / (name + '_images.txt')).write_text('\n'.join(r[key] for r in rows) + '\n', encoding='utf-8')
        (out / 'dataset.yaml').write_text(yaml.safe_dump({'train': str(out / 'train_images.txt'),
            'val': str(out / 'val_images.txt'), 'names': dict(enumerate(VIS_NAMES))}), encoding='utf-8')
        backward = torch.autograd.backward
        def timed_backward(*args, **kwargs):
            first, last = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            first.record()
            tick = time.perf_counter()
            result = backward(*args, **kwargs)
            last.record()
            torch.cuda.synchronize()
            trainer.backward_count += 1
            trainer.backward_seconds = time.perf_counter() - tick
            trainer.backward_CUDA_ms = first.elapsed_time(last)
            return result
        def epoch_begin(instance):
            torch.cuda.synchronize()
            instance.pilot_epoch_start = time.perf_counter()
            instance.pilot_epoch_UTC = datetime.now(timezone.utc).isoformat()
            torch.cuda.reset_peak_memory_stats()
        def epoch_done(instance):
            torch.cuda.synchronize()
            row = {'epoch': instance.epoch + 1, 'started_UTC': instance.pilot_epoch_UTC,
                'finished_UTC': datetime.now(timezone.utc).isoformat(),
                'training_validation_save_wall_seconds': time.perf_counter() - instance.pilot_epoch_start,
                'optimizer_updates': instance.rq4_optimizer_updates, 'EMA_updates': instance.ema.updates,
                'learning_rates': [g['lr'] for g in instance.optimizer.param_groups],
                'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_reserved_bytes': torch.cuda.max_memory_reserved()}
            with (out / 'epoch_progress.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row) + '\n')
            print('PILOT10_EPOCH_COMPLETED', json.dumps(row), flush=True)
        with patch.object(callbacks, 'add_integration_callbacks', lambda instance: None), \
                patch.object(torch.autograd, 'backward', timed_backward), v3.output_scope(out), read_only_sources(out):
            trainer = v3.trainer_factory(value, out)(overrides=value['args'])
            trainer.add_callback('on_train_epoch_start', epoch_begin)
            trainer.add_callback('on_fit_epoch_end', epoch_done)
            trainer.train()
        for epoch in range(1, value['execution_epochs'] + 1):
            actual = [i for row in trainer.rows if row['epoch'] == epoch for i in row['image_ids']]
            if Counter(actual) != Counter(r['image_id'] for r in value['train']):
                raise ValueError('An epoch is missing or repeats anchor exposure.')
        if len(trainer.rows) != trainer.forward_count or trainer.backward_count != value['expected']['batches'] or len(trainer.rows) != value['expected']['batches']:
            raise ValueError('Forward/backward count mismatch.')
        if trainer.rq4_optimizer_updates != value['expected']['updates'] or trainer.ema.updates != value['expected']['updates']:
            raise ValueError('Optimizer/EMA count mismatch.')
        weights = sorted((out / 'weights').glob('epoch*_fp32.pt'))
        if len(weights) != value['execution_epochs'] or trainer.epoch + 1 != value['execution_epochs']:
            raise ValueError('Incomplete epoch/checkpoint population.')
        model_health(trainer.model, gradients=True)
        model_health(trainer.ema.ema)
        with v3.output_scope(out):
            verify_input_preservation(out, protected)
        save(out / 'training_acceptance.json', {'completed_epochs': trainer.epoch + 1,
            'anchor_exposures': len(value['train']) * (trainer.epoch + 1),
            'forward_batches': trainer.forward_count, 'backward_batches': trainer.backward_count,
            'optimizer_updates': trainer.rq4_optimizer_updates, 'EMA_updates': trainer.ema.updates,
            'selected_checkpoint': {'path': str(trainer.rq4_selected), 'sha256': study.sha256(trainer.rq4_selected)},
            'epoch_checkpoints': [{'path': str(p), 'sha256': study.sha256(p)} for p in weights],
            'initialization': value['initialization'], 'internal_validation': trainer.validations,
            'training_wall_seconds': time.perf_counter() - began, 'scope': value['scope'],
            'input_pipeline': 'optimized', 'research_AP': 'NOT_EVALUATED', 'exact_resume_validated': False})
        status = 'TRAINING_COMPLETED'
    except BaseException:
        save(out / 'failure.json', {'traceback': traceback.format_exc(), 'automatic_retry': False})
    finally:
        trainer = None
        gc.collect()
    save(out / 'worker_result.json', {'status': status, 'worker_seconds': time.perf_counter() - began})
    return status


def evaluate_worker(out):
    value, authorization = worker_context(out)
    parent = out.parent
    stage = load(out / 'evaluation_stage.json')
    spec = load(bundle() / 'evaluation_manifest.json')
    spec['contexts'] = stage['contexts']
    original_load = research.load
    original_contrasts = research.paired_contrasts
    def mapped_load(path):
        if Path(path) == bundle() / 'pilot10_protocol.json':
            return load(authorization / 'pilot10_protocol.json')
        if Path(path) == bundle() / 'formal_evaluation_manifest.json':
            return spec
        return original_load(path)
    def accepted(*args):
        worker_context(out)
        for arm in ('CONTROL', 'ALL_FIVE'):
            folder = parent / ('YOLO26s_seed42_' + arm)
            check = load(folder / 'training_acceptance.json')
            if check['completed_epochs'] != 10 or study.sha256(Path(check['selected_checkpoint']['path'])) != check['selected_checkpoint']['sha256']:
                raise ValueError('The matched final checkpoints are not complete and healthy.')
    def contrasts(values):
        if stage.get('normal_reference'):
            initial = Path(stage['normal_reference'])
            for row in load(initial / 'result_index.json'):
                if row['context'] == 'Normal':
                    f = initial / row['folder']
                    values[(row['architecture'], row['seed'], row['arm'], row['resolution'], 'Normal')] = {
                        'metrics': load(f / 'metrics.json'), 'draws': load(f / 'bootstrap.json')['draws']}
        return original_contrasts(values)
    with patch.object(research, 'load', mapped_load), patch.object(research, 'check_worker_authorization', accepted), \
            patch.object(research, 'paired_contrasts', contrasts):
        research.evaluate(bundle(), ROOT / 'datasets/VisDrone', parent, 'pilot10')
    return 'COMPLETE'


def resource_check():
    import psutil
    snap = query()
    battery = psutil.sensors_battery()
    if battery is not None and not battery.power_plugged:
        raise RuntimeError('AC power is required.')
    if psutil.virtual_memory().available < 8 * 1024**3 or shutil.disk_usage(ROOT).free < 20 * 1024**3:
        raise RuntimeError('Insufficient available RAM or disk reserve.')
    if snap['temperature_C'] > 65 or snap['free_MiB'] < 7168 or snap['utilization_percent'] > 10:
        raise RuntimeError('GPU temperature, free memory or contention gate failed.')
    return {'GPU': snap, 'available_RAM_bytes': psutil.virtual_memory().available,
            'disk_free_bytes': shutil.disk_usage(ROOT).free, 'AC': None if battery is None else battery.power_plugged}


def cooldown(out):
    began = time.perf_counter()
    with (out / 'cooldown.jsonl').open('x', encoding='utf-8') as stream:
        while True:
            snap = query()
            stream.write(json.dumps(snap) + '\n')
            stream.flush()
            if snap['temperature_C'] >= 87:
                raise RuntimeError('Unsafe idle temperature; no new GPU task.')
            if snap['temperature_C'] <= 65 and snap['free_MiB'] >= 7168 and snap['utilization_percent'] <= 10:
                break
            if time.perf_counter() - began >= 900:
                raise RuntimeError('The bounded cooldown did not restore stage admission.')
            time.sleep(5)
    save(out / 'cooldown_result.json', {'idle_wall_seconds': time.perf_counter() - began,
        'GPU_worker_seconds': 0, 'scope': 'Planned idle cooldown, not a training retry.'})


def run_stage(authorization, out, stage, cap, mode='train'):
    left = available(authorization)
    if cap > left:
        raise PermissionError('Stage cap does not fit the new cumulative authorization.')
    cooldown(out)
    save(out / 'resource_before.json', resource_check())
    save(out / 'WORKER_LEASE.json', {'deadline_UTC': min(datetime.now(timezone.utc) + timedelta(seconds=cap),
        datetime.fromisoformat(load(authorization / 'authorization.json')['absolute_expiry_UTC'])).isoformat()})
    save(out / 'worker_ticket.json', {'authorization': str(authorization),
        'protocol_SHA256': study.sha256(out / 'protocol.json'), 'mode': mode,
        'evaluation_stage_SHA256': study.sha256(out / 'evaluation_stage.json') if mode == 'evaluate' else None})
    began = time.perf_counter()
    result = None
    try:
        result = run_owned([sys.executable, '-B', '-X', 'utf8', str(SOURCE), '--worker', str(out), '--worker-mode', mode],
            out, cap, load(bundle() / 'safety_policy.json'),
            dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2'))
        valid = (load(out / 'worker_result.json')['status'] == 'TRAINING_COMPLETED' if mode == 'train'
                 else load(out / 'evaluation_result.json')['status'] == 'COMPLETE')
        if result['returncode'] or result['stop_reason'] or not valid:
            raise RuntimeError('Worker did not complete; stop this GPU branch without retry.')
    finally:
        save(authorization / 'charges' / (out.parent.name + '_' + stage + '.json'), {
            'RunID': load(authorization / 'execution_ticket.json')['RunID'], 'stage_path': str(out),
            'stage': stage, 'charged_seconds': time.perf_counter() - began,
            'finished_UTC': datetime.now(timezone.utc).isoformat(), 'hard_cap': cap, 'supervisor': result})
    print('PILOT10_STAGE_COMPLETED', stage, 'remaining_seconds', available(authorization), flush=True)


def prepare(authorization):
    study.runtime(authorization, cuda=False)
    v3.base.verify_bundle(bundle())
    v3.base.environment_check(bundle())
    population = v3.base.resolve_data(bundle(), ROOT / 'datasets/VisDrone')
    spec = load(authorization / 'pilot10_protocol.json')
    verify_protocol(spec)
    ordinary = load(bundle() / 'ordinary_training_args.json')['YOLO26s']
    from ultralytics.utils.torch_utils import one_cycle
    schedule = one_cycle(1, ordinary['lrf'], 10)
    save(authorization / 'learning_rate_schedule.json', {'epochs': 10, 'cos_lr': True,
        'lr0': .01, 'lrf': ordinary['lrf'], 'warmup_epochs': 3, 'warmup_batches': 9708,
        'cosine_multipliers': [schedule(i) for i in range(11)],
        'note': 'Optimizer group LR during warmup is recorded per actual update; this is the nominal cosine multiplier.'})
    cases = load(bundle() / 'evaluation_manifest.json')['contexts']
    seen = {c['p']['id'] for c in cases}
    extension = [c for c in load(bundle() / 'formal_evaluation_manifest.json')['contexts'] if c['p']['id'] not in seen]
    save(authorization / 'evaluation_extension_plan.json', {'initial': cases, 'remaining_in_frozen_order': extension,
        'selection_uses_new_AP': False, 'cap_per_additional_four_configuration_context_seconds': 4200})
    rows = []
    for arm in ('CONTROL', 'ALL_FIVE'):
        folder = authorization / ('preview_' + arm)
        folder.mkdir()
        value = plan(authorization, folder, arm, population)
        save(folder / 'protocol.json', value)
        rows.append({'arm': arm, 'initialization': value['initialization'], 'expected': value['expected'],
                     'train': len(value['train']), 'development': len(value['development'])})
    save(authorization / 'preparation.json', {'status': 'CPU_PREPARED_GPU_NOT_RUN', 'plans': rows,
        'resources': resource_check(), 'source_hashes': {str(p): study.sha256(p) for p in source_paths()},
        'formal100_unchanged': True})
    print(authorization, flush=True)


def execute(authorization):
    study.runtime(authorization, cuda=False)
    verify_freeze(authorization)
    test = load(authorization / 'pilot10_CPU_tests_final.json')
    if test['status'] != 'PASS':
        raise PermissionError('New pilot entrypoint CPU tests did not pass.')
    resource = load(authorization / 'resource_admission.json')
    if available(authorization) < resource['initial_required_seconds']:
        raise PermissionError('Both arms and initial evaluation do not conservatively fit the new allocation.')
    resource_check()
    if (study.RUNS / '.gpu.lock').exists() or (study.RUNS / '.pilot10_gpu.lock').exists():
        raise RuntimeError('An owned GPU lock already exists.')
    parent = study.create('pilot10_execution')
    study.runtime(parent, cuda=False)
    save(authorization / 'execution_ticket.json', {'RunID': parent.name, 'authorization': str(authorization),
        'freeze_SHA256': study.sha256(authorization / 'execution_freeze.json'), 'automatic_retry': False})
    lock = study.RUNS / '.pilot10_gpu.lock'
    save(lock, {'RunID': parent.name, 'PID': os.getpid()})
    caps = resource['stage_hard_caps_seconds']
    status = 'PARTIAL_WITH_LIMITATIONS'
    try:
        population = v3.base.resolve_data(bundle(), ROOT / 'datasets/VisDrone')
        smoke = parent / 'entrypoint_smoke'
        smoke.mkdir()
        save(smoke / 'protocol.json', plan(authorization, smoke, 'ALL_FIVE', population, smoke=True))
        run_stage(authorization, smoke, 'entrypoint_smoke', caps['entrypoint_smoke'])
        for arm in ('CONTROL', 'ALL_FIVE'):
            reserve = caps[arm] + caps['initial_evaluation'] + (caps['ALL_FIVE'] if arm == 'CONTROL' else 0)
            if available(authorization) < reserve:
                raise PermissionError('The remaining matched work no longer fits; no unpaired launch.')
            out = parent / ('YOLO26s_seed42_' + arm)
            out.mkdir()
            save(out / 'protocol.json', plan(authorization, out, arm, population))
            run_stage(authorization, out, arm, caps[arm])
        evaluation = parent / 'evaluation'
        evaluation.mkdir()
        save(evaluation / 'protocol.json', {'scope': ID, 'completed_arms_required': 2})
        extension = load(authorization / 'evaluation_extension_plan.json')
        save(evaluation / 'evaluation_stage.json', {'contexts': extension['initial']})
        run_stage(authorization, evaluation, 'initial_evaluation', caps['initial_evaluation'], 'evaluate')
        save(parent / 'initial_pair_acceptance.json', {'status': 'PILOT10_INITIAL_PAIR_COMPLETED',
            'configurations': len(load(evaluation / 'result_index.json')), 'epochs_per_arm': 10,
            'scope': 'Single-seed development pilot; no independent or full21-condition gap claim.'})
        for position, context in enumerate(extension['remaining_in_frozen_order']):
            cap = extension['cap_per_additional_four_configuration_context_seconds']
            if available(authorization) < cap + 600:
                break
            extra = parent / ('extra_context_' + str(position).zfill(2))
            extra.mkdir()
            for arm in ('CONTROL', 'ALL_FIVE'):
                target = extra / ('YOLO26s_seed42_' + arm)
                target.mkdir()
                shutil.copyfile(parent / target.name / 'training_acceptance.json', target / 'training_acceptance.json')
            out = extra / 'evaluation'
            out.mkdir()
            save(out / 'protocol.json', {'scope': ID, 'additional_frozen_context': context['id']})
            save(out / 'evaluation_stage.json', {'contexts': [context], 'normal_reference': str(evaluation)})
            run_stage(authorization, out, 'extra_' + str(position).zfill(2), cap, 'evaluate')
        status = 'PILOT10_COMPLETED_DEVELOPMENT_ONLY'
    except BaseException:
        save(parent / 'failure.json', {'traceback': traceback.format_exc(), 'automatic_retry': False})
    finally:
        if load(lock)['RunID'] == parent.name:
            lock.unlink()
    save(parent / 'execution_result.json', {'status': status, 'new_authorization': str(authorization),
        'charged_worker_seconds': sum(load(p)['charged_seconds'] for p in charge_files(authorization)),
        'remaining_seconds': available(authorization), 'formal100_modified': False})
    (parent / 'run.json').write_text(json.dumps({'RunID': parent.name, 'ExperimentID': study.EXPERIMENT,
        'status': status, 'authorization_RunID': authorization.name}, indent=2) + '\n', encoding='utf-8')
    study.seal(parent)
    print('PILOT10_FINISHED', parent, status, flush=True)
    return 0 if status == 'PILOT10_COMPLETED_DEVELOPMENT_ONLY' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run')
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--worker-mode', choices=('train', 'evaluate'), default='train')
    args = parser.parse_args()
    if args.worker:
        result = train_worker(args.worker.resolve(strict=True)) if args.worker_mode == 'train' else evaluate_worker(args.worker.resolve(strict=True))
        raise SystemExit(0 if result in ('TRAINING_COMPLETED', 'COMPLETE') else 1)
    if not args.run or args.prepare == args.execute:
        parser.error('Choose --prepare or --execute and provide the immutable authorization RunID.')
    folder = study.run_path(args.run)
    if args.prepare:
        prepare(folder)
    else:
        raise SystemExit(execute(folder))
