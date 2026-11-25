"""One temperature-bounded ALL_FIVE training-prefix measurement."""
import argparse
import copy
import ctypes
from datetime import datetime, timezone
import gc
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

POST = Path(__file__).resolve().parent
sys.path.insert(0, str(POST.parent))
from study import *
from safety import available_seconds, query, run_owned
from portable import environment_check, plan_arm, renderer, resolve_data, verify_bundle

BUNDLE_RUN = 'vm37_yolo26_rq4_portable_bundle_cpu_20261112T140353_793d9ad0'
BATCHES = 128
CAP = 600
CONFIRMATION = 'call_301d6d075b6f492aafbb61e867df9f2c'
TICKET = RUNS / 'all_five_short_cost_ticket.json'


class PrefixComplete(Exception):
    pass


def prefix_steps(batches):
    import numpy as np
    warmup = 3 * 3236
    last = -1
    updates = 0
    for position in range(batches):
        accumulation = max(1, int(np.interp(position, [0, warmup], [1, 32]).round()))
        if position - last >= accumulation:
            updates += 1
            last = position
    return {'batches': batches, 'updates': updates, 'warmup_batches': warmup,
            'last_update_position': last, 'pending_microbatches': batches - 1 - last}


def ac_connected():
    class Power(ctypes.Structure):
        _fields_ = [('AC', ctypes.c_ubyte), ('flags', ctypes.c_ubyte), ('percent', ctypes.c_ubyte),
                    ('status', ctypes.c_ubyte), ('life', ctypes.c_ulong), ('full', ctypes.c_ulong)]
    value = Power()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(value)) or value.AC != 1:
        raise RuntimeError('Reliable external power is not currently connected.')
    return {'AC_connected': True, 'battery_percent': value.percent}


def prepare():
    out = create('all_five_short_cost')
    runtime(out, cuda=False)
    bundle = run_path(BUNDLE_RUN) / 'portable_bundle'
    verify_bundle(bundle)
    environment_check(bundle)
    receipt = load(RUNS / 'policy_admission.json')
    if receipt['status'] != 'PASS' or sha256(HERE/'TARGETED_POLICY.json') != receipt['policy_SHA256']:
        raise ValueError('Accepted ALL_FIVE policy changed.')
    if sha256(PUBLIC/'code/main/training_effects.py') != receipt['effects_implementation_SHA256'] or sha256(HERE.parent/'cv261109/generator.py') != receipt['generator_SHA256']:
        raise ValueError('Accepted image-effect implementation changed.')
    protocol = load(bundle / 'pilot20_protocol.json')
    population = resolve_data(bundle, ROOT / 'datasets/VisDrone')
    plan = plan_arm(bundle, population, protocol, 'YOLO26s', 42, 'ALL_FIVE', out)
    plan.update(mode='BOUNDED_ALL_FIVE_TRAINING_PREFIX', scope='ENGINEERING_COST_ONLY',
                scientific_training_arm=False, stop_after_batches=BATCHES, expected=prefix_steps(BATCHES),
                validation='NOT_RUN', AP='NOT_EVALUATED', completed_epochs=0,
                worker_hard_seconds=CAP, full_epoch_cost='NOT_MEASURED',
                stop_rule='Controlled exception after128 complete batches; no epoch-completion callback.',
                snapshot_role='Partial engineering state only; not a YOLO checkpoint or exact-resume input.')
    plan['sources'] = {str(p.relative_to(PUBLIC)): sha256(p) for p in
        [Path(__file__), POST / 'safety.py', POST / 'portable.py', HERE / 'training.py', HERE / 'study.py',
         HERE/'TARGETED_POLICY.json', HERE.parent/'cv261109/generator.py', HERE.parent/'cv261109/model.py',
         VM36/'feature_models/fm_p.uvl', VM36/'feature_models/fm_d.uvl'] +
        [PUBLIC / 'code/main' / name for name in ('training_health.py', 'training_data.py', 'training_effects.py')]}
    policy = load(bundle / 'safety_policy.json')
    ledger = load(RUNS / 'compute_allocation.json')
    if available_seconds(ledger) < CAP + 60:
        raise RuntimeError('The original allocation cannot admit this bounded measurement.')
    gpu = query()
    if gpu['temperature_C'] > policy['preflight_maximum_temperature_C'] or gpu['free_MiB'] < policy['preflight_minimum_free_MiB'] or gpu['utilization_percent'] > 10:
        raise RuntimeError('GPU readiness failed.')
    physical = {'received_UTC': utc().isoformat(), 'question_reference': CONFIRMATION,
        'author_reply': '已检查，以上条件均正常', 'hard_ventilated_surface': 'AUTHOR_CONFIRMED',
        'clear_intake_exhaust': 'AUTHOR_CONFIRMED', 'fans_normal': 'AUTHOR_CONFIRMED',
        'reliable_adapter': 'AUTHOR_CONFIRMED', 'sustained_GPU_stability': 'NOT_VALIDATED', **ac_connected()}
    save(out / 'protocol.json', plan, out)
    save(out / 'safety_policy.json', policy, out)
    save(out / 'physical_confirmation.json', physical, out)
    save(out / 'admission.json', {'status': 'GO_FOR_ONE_BOUNDED_ENGINEERING_PREFIX_ONLY', 'GPU_before': gpu,
        'available_seconds': available_seconds(ledger), 'absolute_deadline_UTC': ledger['absolute_deadline_UTC'],
        'ledger_before': source(RUNS / 'compute_allocation.json'), 'disk_free_bytes': shutil.disk_usage(ROOT).free,
        'pilot20_admitted': False, 'formal100_admitted': False, 'budget_reset': False}, out)
    save(out / 'source_hashes.json', plan['sources'], out)
    snapshots = out / 'source_snapshots'
    snapshots.mkdir()
    for relative in plan['sources']:
        shutil.copyfile(PUBLIC / relative, snapshots / relative.replace('/', '__').replace('\\', '__'))
    write_atomic(out / 'diary.md', '# ALL_FIVE bounded cost prefix\n\nPhysical cooling and adapter conditions were confirmed by the author. Prepared one128-batch training-prefix measurement under the unchanged allocation, with a600-second cap and the frozen temperature stops. No epoch, AP or augmentation-benefit claim.\n', out)
    print(out, flush=True)
    return out


def cpu_tree(value):
    import torch
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(child) for key, child in value.items()}
    if isinstance(value, list):
        return [cpu_tree(child) for child in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(child) for child in value)
    return copy.deepcopy(value)


def worker(out):
    plan = load(out / 'protocol.json')
    ticket = load(TICKET)
    if ticket['RunID'] != out.name or ticket['protocol_SHA256'] != sha256(out / 'protocol.json'):
        raise PermissionError('No matching one-attempt ticket.')
    save(out / 'WORKER_STARTED.json', {'PID': __import__('os').getpid(), 'UTC': utc().isoformat()}, out)
    runtime(out, cuda=True)
    import cv2
    import torch
    import yaml
    from unittest.mock import patch
    from ultralytics.utils import callbacks
    from post80_loader import read_only_sources
    from training import trainer_class, verify_input_preservation
    from main.labels import VIS_NAMES
    from main.training_health import model_health, cpu_state, finite_tree, same_tree
    started = time.perf_counter()
    status = 'FAILED'
    trainer = None
    try:
        if importlib.metadata.version('ultralytics') != '8.4.210':
            raise RuntimeError('Unfrozen training version.')
        for relative, expected in plan['sources'].items():
            if sha256(PUBLIC / relative) != expected:
                raise ValueError('Frozen execution source changed.')
        if sha256(Path(plan['initialization']['path'])) != plan['initialization']['sha256']:
            raise ValueError('Native10 initialization changed.')
        torch.set_num_threads(2)
        cv2.setNumThreads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable.')
        renderer()
        protected = []
        for rows, key, sha_key, labels, id_key in (
            (plan['train'], 'image_path', 'image_SHA256', plan['train_labels'], 'image_id'),
            (plan['development'], 'loader_path', 'source_image_SHA256', plan['val_labels'], 'stem')):
            for row in rows:
                for path, expected in ((row[key], row[sha_key]), (labels[row[id_key]], row['derived_label_SHA256'])):
                    if sha256(Path(path)) != expected:
                        raise ValueError('Frozen data changed.')
                    protected.append({'path': path, 'sha256': expected})
        write_atomic(out / 'train_images.txt', '\n'.join(r['image_path'] for r in plan['train'])+'\n', out)
        write_atomic(out / 'val_images.txt', '\n'.join(r['loader_path'] for r in plan['development'])+'\n', out)
        write_atomic(out / 'dataset.yaml', yaml.safe_dump({'train': str(out/'train_images.txt'),
            'val': str(out/'val_images.txt'), 'names': dict(enumerate(VIS_NAMES))}), out)
        Parent = trainer_class(plan, out)
        class PrefixTrainer(Parent):
            def batch_end(self, instance):
                super().batch_end(instance)
                if len(self.rows) == plan['stop_after_batches']:
                    raise PrefixComplete()
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
        training_tick = time.perf_counter()
        with patch.object(callbacks, 'add_integration_callbacks', lambda instance: None), \
             patch.object(torch.autograd, 'backward', timed_backward), read_only_sources(out):
            trainer = PrefixTrainer(overrides=plan['args'])
            try:
                trainer.train()
            except PrefixComplete:
                pass
            else:
                raise RuntimeError('The training-prefix boundary was not reached as planned.')
            torch.cuda.synchronize()
            prefix_seconds = time.perf_counter() - training_tick
            if not (len(trainer.rows) == trainer.forward_count == trainer.backward_count == BATCHES):
                raise ValueError('Forward/backward counts differ.')
            if trainer.rq4_optimizer_updates != plan['expected']['updates'] or trainer.ema.updates != plan['expected']['updates']:
                raise ValueError('Optimizer/EMA counts differ.')
            ids = [identity for row in trainer.rows for identity in row['image_ids']]
            if len(ids) != BATCHES*2 or len(set(ids)) != len(ids) or not set(ids).issubset({r['image_id'] for r in plan['train']}):
                raise ValueError('Training-prefix membership differs.')
            model_health(trainer.model, gradients=True)
            model_health(trainer.ema.ema)
            if not finite_tree(trainer.optimizer.state_dict()):
                raise FloatingPointError('Nonfinite optimizer state.')
            save_tick = time.perf_counter()
            payload = {'scope': 'PARTIAL_ENGINEERING_STATE_NOT_A_YOLO_CHECKPOINT', 'completed_epochs': 0,
                'completed_batches': BATCHES, 'online_state': cpu_state(trainer.model), 'EMA_state': cpu_state(trainer.ema.ema),
                'optimizer': cpu_tree(trainer.optimizer.state_dict()), 'scheduler': cpu_tree(trainer.scheduler.state_dict()),
                'scaler': cpu_tree(trainer.scaler.state_dict()), 'protocol_SHA256': sha256(out/'protocol.json'),
                'initialization_SHA256': plan['initialization']['sha256'], 'exact_resume_validated': False,
                'reuse_for_pilot_or_formal_training': False}
            partial = out/'engineering_state.partial'
            torch.save(payload, partial)
            reloaded = torch.load(partial, map_location='cpu', weights_only=False)
            if not finite_tree(reloaded) or not same_tree(payload, reloaded):
                raise ValueError('Partial engineering state differs after reload.')
            target = out/'partial_engineering_state.pt'
            partial.rename(target)
            state_seconds = time.perf_counter()-save_tick
        verify_input_preservation(out, protected)
        save(out/'short_measurement.json', {'status':'SHORT_PREFIX_COMPLETED_ENGINEERING_ONLY',
            'completed_epochs':0,'completed_batches':len(trainer.rows),'anchor_exposures':len(ids),
            'optimizer_updates':trainer.rq4_optimizer_updates,'EMA_updates':trainer.ema.updates,
            'forward_batches':trainer.forward_count,'backward_batches':trainer.backward_count,
            'loader_seconds':sum(r['loader_seconds'] for r in trainer.rows),
            'forward_backward_health_seconds':sum(r['forward_backward_health_seconds'] for r in trainer.rows),
            'prefix_setup_and_training_seconds':prefix_seconds,'state_save_reload_seconds':state_seconds,
            'GPU_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'GPU_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
            'state':source(target),'loss_first':trainer.rows[0]['loss'],'loss_last':trainer.rows[-1]['loss'],
            'AP':'NOT_EVALUATED','full_epoch_cost':'NOT_MEASURED','pilot20_admission':False,
            'limits':'Warmup prefix only; no internal validation, research evaluation or sustained-load proof.'}, out)
        status='SHORT_PREFIX_COMPLETED_ENGINEERING_ONLY'
    except BaseException:
        save(out/'failure.json', {'traceback':traceback.format_exc(),'automatic_retry':False,
            'completed_batches':len(trainer.rows) if trainer is not None else 0}, out)
    save(out/'worker_result.json', {'status':status,'worker_seconds':time.perf_counter()-started,
        'completed_epochs':0,'AP':'NOT_EVALUATED'}, out)


def execute(out):
    if TICKET.exists():
        raise RuntimeError('The one-attempt ticket is already occupied; no automatic retry.')
    plan=load(out/'protocol.json')
    if plan['stop_after_batches']!=BATCHES or plan['worker_hard_seconds']!=CAP:
        raise ValueError('Bounded protocol changed.')
    ac_connected()
    current = query()
    if current['utilization_percent'] > 10:
        raise RuntimeError('GPU is busy; this attempt is not admitted.')
    save(TICKET, {'RunID':out.name,'protocol_SHA256':sha256(out/'protocol.json'),
        'physical_confirmation_reference':CONFIRMATION,'created_UTC':utc().isoformat(),'retry_allowed':False}, RUNS)
    cap=budget_start(out,CAP,'FULL_DATA_COST')
    started=time.perf_counter()
    supervisor=None
    error=None
    try:
        supervisor=run_owned([sys.executable,'-B','-X','utf8',str(Path(__file__)),'--worker',out.name],out,cap,
            load(out/'safety_policy.json'),dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',
                YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',OMP_NUM_THREADS='2',PYTHONIOENCODING='utf-8'))
    except BaseException:
        error=traceback.format_exc()
    worker_result=load(out/'worker_result.json') if (out/'worker_result.json').exists() else {'status':'INTERRUPTED'}
    status=worker_result['status']
    if error or supervisor is None or supervisor['returncode'] or supervisor['stop_reason']:
        status='STOPPED_WITH_LIMITATIONS'
    charged=time.perf_counter()-started
    budget_finish(out,charged,status)
    traces=[json.loads(line) for line in (out/'gpu_telemetry.jsonl').read_text().splitlines()] if (out/'gpu_telemetry.jsonl').exists() else []
    save(out/'final_result.json', {'status':status,'worker':worker_result,'supervisor':supervisor,'error':error,
        'charged_seconds':charged,'available_seconds_after':available_seconds(load(RUNS/'compute_allocation.json')),
        'absolute_deadline_UTC':load(RUNS/'compute_allocation.json')['absolute_deadline_UTC'],'budget_reset':False,
        'maximum_sampled_temperature_C':max((r['temperature_C'] for r in traces),default=None),
        'maximum_sampled_memory_used_MiB':max((r['used_MiB'] for r in traces),default=None),
        'pilot20':'NOT_ADMITTED','formal100':'RESOURCE_BLOCKED','AP':'NOT_EVALUATED',
        'sustained_GPU_stability':'NOT_VALIDATED'}, out)
    save(out/'nested_SHA256.json', {str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and
        p.parent!=out and 'runtime' not in p.parts}, out)
    save(out/'run.json', {'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'ALL_FIVE_SHORT_COST',
        'status':status,'completed_epochs':0,'matched_pairs':0}, out,replace=True)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>ALL_FIVE bounded engineering prefix</h1>'
        '<p>'+status+'</p><p>No completed epoch, AP or augmentation-benefit result.</p><a href="final_result.json">Final evidence</a>'
        '<br><a href="gpu_telemetry.jsonl">Temperature and memory</a><br><a href="training_batches.jsonl">Actual batches</a>',out)
    seal(out)
    from main.visualization import build_run_index
    build_run_index(RUNS,EXPERIMENT)
    print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--execute')
    parser.add_argument('--worker')
    args=parser.parse_args()
    if args.prepare:
        prepare()
    elif args.execute:
        execute(run_path(args.execute))
    elif args.worker:
        worker(run_path(args.worker))
    else:
        parser.error('Use --prepare, or execute an explicitly prepared single-attempt RunID.')
