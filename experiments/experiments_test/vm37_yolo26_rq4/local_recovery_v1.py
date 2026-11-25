"""CPU preparation for independent laptop recovery protocols. No GPU launcher."""
import argparse
import ast
import copy
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import pickle
import random
from pathlib import Path
import sys
import textwrap
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'post_cost'))
import pilot10 as baseline

study = baseline.study
v3 = baseline.v3
VERSION = 'VM37_LOCAL_RECOVERY_V1'
AUTH = 'vm37_yolo26_rq4_pilot10_authorization_cpu_20261113T212036_50286ea7'
ASSESSMENT = 'vm37_yolo26_rq4_pilot10_thermal_recovery_cpu_20261113T223846_b610e4da'
LEGACY_PREPROCESS_AST = '6fa37cfecbabcbe4149d00289ff30ca6a4cdad0bc7026f66cd59cdef25f1bdd8'
OPTIONS = {'A': (1280, 1), 'B': (960, 10)}


def write_csv(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def candidate_protocol(option):
    size, completed = OPTIONS[option]
    spec = copy.deepcopy(baseline.load(study.run_path(AUTH) / 'pilot10_protocol.json'))
    spec.update(id=f'{VERSION}_{option}_{size}', version=VERSION,
                status='CPU_CANDIDATE_NOT_GPU_ADMITTED', option=option,
                execution_epochs=completed, engineering_only=option == 'A',
                selection='EPOCH1_ENGINEERING_ONLY' if option == 'A' else 'FIXED_EPOCH10_EMA',
                GPU_execution_authorized=False, exact_resume_validated=False,
                authorization_reference=AUTH, new_allocation=False,
                planned_counters_per_arm=study.expected_steps(6471, 10, stop_epochs=completed))
    spec['shared_training_args']['imgsz'] = size
    spec['optional_extension'] = 'Not admitted by this preparation stage.'
    spec['candidate_worker_cap_seconds_per_arm'] = 3600 if option == 'A' else None
    spec['resource_admission'] = 'PENDING; 960 cost and thermal stability are not measured.'
    return spec


def verify_candidate(spec):
    option = spec.get('option')
    if option not in OPTIONS or spec != candidate_protocol(option):
        raise ValueError('Candidate differs from the versioned matched protocol.')
    return OPTIONS[option][0]


def make_plan(spec, out, arm, population, pipeline='optimized'):
    verify_candidate(spec)
    plan = v3.make_plan(baseline.bundle(), population, spec, 'YOLO26s', 42, arm,
                        out, 'local_recovery_v1', pipeline)
    plan.update(mode=spec['id'], scope='CANARY_ENGINEERING_ONLY / DEVELOPMENT_ONLY'
                if spec['engineering_only'] else 'EXPLORATORY_SINGLE_SEED_DEVELOPMENT_ONLY',
                executor_version=VERSION, protocol_name='local_recovery_v1',
                recovery_protocol=copy.deepcopy(spec), execution_epochs=spec['execution_epochs'],
                stop_after_completed_epochs=spec['execution_epochs'],
                expected=spec['planned_counters_per_arm'], selection=spec['selection'],
                GPU_execution_authorized=False, benchmark_only=spec['engineering_only'])
    return plan


def verify_pair(control, target):
    keys = ('architecture', 'initialization', 'COCO_source_SHA256', 'train', 'development',
            'train_labels', 'val_labels', 'expected', 'execution_epochs', 'selection', 'recovery_protocol')
    if control['arm'] != 'CONTROL' or target['arm'] != 'ALL_FIVE':
        raise ValueError('The matched pair requires both original arms.')
    if any(control[key] != target[key] for key in keys):
        raise ValueError('Matched source, population or schedule differs.')
    args = [{k: v for k, v in plan['args'].items() if k not in ('data', 'project')} for plan in (control, target)]
    if args[0] != args[1] or control['targeted_recipe'] is not None:
        raise ValueError('Matched optimizer, geometry or ordinary augmentation differs.')
    if target['targeted_recipe'] != baseline.load(baseline.bundle() / 'assets/TARGETED_POLICY.json'):
        raise ValueError('The original ALL_FIVE policy changed.')


def assert_tensor(batch, size):
    import torch
    tensor = batch['img']
    if tensor.ndim != 4 or tuple(tensor.shape[1:]) != (3, size, size):
        raise ValueError('Unexpected training tensor shape.')
    if tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
        raise ValueError('Training input must be finite FP32.')


def trainer_factory(plan, out):
    """Replace only the audited size assertion; inherit health and save hooks."""
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    Parent = v3.trainer_factory(plan, out)
    method = ast.parse(textwrap.dedent(inspect.getsource(Parent.preprocess_batch))).body[0]
    if hashlib.sha256(ast.dump(method, include_attributes=False).encode()).hexdigest() != LEGACY_PREPROCESS_AST:
        raise ValueError('Legacy preprocessing changed; review the adapter before use.')
    size = verify_candidate(plan['recovery_protocol'])
    if plan['args']['imgsz'] != size:
        raise ValueError('Plan resolution differs from its candidate protocol.')

    class ResolutionTrainer(Parent):
        def preprocess_batch(self, batch):
            batch = DetectionTrainer.preprocess_batch(self, batch)
            assert_tensor(batch, size)
            self.tensor_shape = list(batch['img'].shape)
            return batch

    return ResolutionTrainer


def stage_diagnosis(out, supervisor, mode='train'):
    """Preserve the supervisor reason before inspecting worker output."""
    if supervisor.get('stop_reason'):
        return {'status': 'SUPERVISOR_STOPPED', 'primary_reason': supervisor['stop_reason'],
                'returncode': supervisor.get('returncode')}
    if supervisor.get('returncode') != 0:
        return {'status': 'WORKER_FAILED', 'primary_reason': 'WORKER_EXIT',
                'returncode': supervisor.get('returncode')}
    name, expected = ('worker_result.json', 'TRAINING_COMPLETED') if mode == 'train' else ('evaluation_result.json', 'COMPLETE')
    try:
        result = baseline.load(out / name)
    except FileNotFoundError:
        return {'status': 'INCOMPLETE', 'primary_reason': 'MISSING_WORKER_RESULT', 'file': name}
    except (ValueError, UnicodeError):
        return {'status': 'INCOMPLETE', 'primary_reason': 'INVALID_WORKER_RESULT', 'file': name}
    if result.get('status') != expected:
        return {'status': 'INCOMPLETE', 'primary_reason': 'WORKER_REPORTED_INCOMPLETE', 'worker': result}
    return {'status': 'STAGE_COMPLETED', 'primary_reason': None, 'worker': result}


def thermal_clearance(record, failure_utc, now=None):
    now = now or datetime.now(timezone.utc)
    try:
        checked = datetime.fromisoformat(record['checked_UTC'])
        if checked <= datetime.fromisoformat(failure_utc) or checked > now:
            return False
        fields = ('hard_ventilated_surface', 'unobstructed_intake_exhaust',
                  'fans_normal', 'reliable_AC_adapter', 'no_visible_airflow_problem')
        return all(record.get(key) is True for key in fields) and bool(record.get('OEM_mode')) and bool(record.get('room_observation'))
    except (KeyError, TypeError, ValueError):
        return False


def budget_snapshot(now=None):
    authorization = study.run_path(AUTH)
    spec = baseline.load(authorization / 'authorization.json')
    charges = baseline.charge_files(authorization)
    used = sum(baseline.load(path)['charged_seconds'] for path in charges)
    return {'authorization_RunID': AUTH, 'authorization_SHA256': study.sha256(authorization / 'authorization.json'),
            'maximum_seconds': spec['max_cumulative_GPU_worker_wall_seconds'], 'charged_seconds': used,
            'remaining_seconds': baseline.available(authorization, now),
            'absolute_expiry_UTC': spec['absolute_expiry_UTC'],
            'charges': [study.source(path) for path in charges], 'new_allocation': False,
            'GPU_seconds_this_run': 0}


def resource_admission(budget, caps, clearance=False, measured=False):
    reasons = []
    if not clearance:
        reasons.append('PHYSICAL_CLEARANCE_PENDING')
    if not measured:
        reasons.append('MATCHED_COST_AND_THERMAL_ADMISSION_PENDING')
    if not caps or any(not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0 for v in caps.values()):
        reasons.append('STAGE_CAPS_NOT_FROZEN')
    elif sum(caps.values()) > budget['remaining_seconds']:
        reasons.append('CAPS_EXCEED_REMAINING_AUTHORIZATION')
    if budget['remaining_seconds'] <= 0:
        reasons.append('AUTHORIZATION_EXPIRED_OR_EXHAUSTED')
    return {'status': 'BLOCKED' if reasons else 'RESOURCE_READY_ONLY', 'reasons': reasons,
            'GPU_execution_allowed_this_stage': False}


def resolution_audit(out):
    sources = sorted(set(baseline.source_paths() + [HERE / 'post_cost/profile_input.py',
        HERE / 'post_cost/loader_cost.py', HERE / 'post_cost/compare_v3.py']))
    rows = []
    for path in sources:
        for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
            if '1280' not in line:
                continue
            if path.name == 'training.py' and 'shape[1:]' in line:
                disposition = 'Bypassed only by the versioned ResolutionTrainer assertion.'
            elif path.name == 'pilot10.py' and "'imgsz': 1280" in line:
                disposition = 'Old validator retained; new candidate validator accepts only its frozen size.'
            elif path.name in ('profile_input.py', 'loader_cost.py'):
                disposition = 'Historical CPU benchmark only; not called by the new input replay.'
            elif path.name == 'evaluate.py':
                disposition = 'Intentional inference resolutions 640/1280; independent of training size.'
            else:
                disposition = 'Historical preparation/provenance; not a new training-size override.'
            rows.append({'source': str(path), 'SHA256': study.sha256(path), 'line': number,
                         'text': line.strip(), 'disposition': disposition})
    write_csv(out / 'resolution_assumptions.csv', rows)
    return rows


def cpu_inputs(out, population):
    """Exercise the real dataset factory at both sizes without model execution."""
    import torch
    import cv2
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.torch_utils import init_seeds
    from main.labels import VIS_NAMES
    from verify_input import value_signature, rng_signature
    torch.set_num_threads(2)
    cv2.setNumThreads(2)
    ids = baseline.load(baseline.bundle() / 'cpu_comparison_manifest.json')['image_ids'][:4]
    lookup = {r['image_id']: r for r in population['train']}
    subset = {'train': [lookup[i] for i in ids], 'val': population['val'][:2]}

    def global_state():
        import numpy as np
        return hashlib.sha256(pickle.dumps([random.getstate(), np.random.get_state(),
                                           torch.get_rng_state().numpy().tobytes()])).hexdigest()

    def build(folder, spec, arm, pipeline):
        folder.mkdir()
        plan = make_plan(spec, folder, arm, subset, pipeline)
        baseline.save(folder / 'protocol.json', plan)
        listing = folder / 'images.txt'
        listing.write_text('\n'.join(r['image_path'] for r in subset['train']) + '\n', encoding='utf-8')
        Trainer = trainer_factory(plan, folder)
        trainer = Trainer.__new__(Trainer)
        trainer.args = get_cfg(overrides=plan['args'])
        trainer.device = torch.device('cpu')
        trainer.model = SimpleNamespace(stride=torch.tensor([32]))
        trainer.data = {'names': dict(enumerate(VIS_NAMES)), 'nc': 10, 'channels': 3}
        data = trainer.build_dataset(str(listing), mode='train', batch=2)
        return trainer, data, plan

    # Match the accepted comparison's warm-import boundary before resetting seeds.
    init_seeds(42, deterministic=True)
    before_warm = global_state()
    _, warm, _ = build(out / 'import_warmup', candidate_protocol('A'), 'CONTROL', 'baseline')
    after_warm = global_state()
    del warm
    v3.base.renderer()
    baseline.save(out / 'cold_setup_rng.json', {'before_dataset_constructor': before_warm,
        'after_dataset_constructor': after_warm, 'seed_reset_before_each_measured_pass': True,
        'scope': 'Input comparison only; training seed placement is unchanged.'})
    rows = []
    for option in OPTIONS:
        spec = candidate_protocol(option)
        for arm in ('CONTROL', 'ALL_FIVE'):
            expected = None
            for pipeline in ('baseline', 'optimized'):
                folder = out / f'cpu_{option}_{arm}_{pipeline}'
                init_seeds(42, deterministic=True)
                before_build = global_state()
                trainer, data, plan = build(folder, spec, arm, pipeline)
                baseline.save(folder / 'construction_rng.json', {'before': before_build, 'after': global_state()})
                observations = []
                began = time.perf_counter()
                try:
                    for epoch in (0, 1):
                        data.set_epoch(epoch)
                        if epoch == 0:
                            data.close_mosaic(get_cfg(overrides=plan['args']))
                        by_id = {Path(p).stem: i for i, p in enumerate(data.im_files)}
                        for start in range(0, len(ids), 2):
                            batch = data.collate_fn([data[by_id[i]] for i in ids[start:start + 2]])
                            trainer.preprocess_batch(batch)
                            observations.append({'epoch': epoch, 'ids': ids[start:start + 2],
                                'batch': value_signature(batch), 'events': copy.deepcopy(data.effect_events),
                                'RNG': rng_signature(data), 'buffer': list(data.buffer),
                                'occurrences': dict(data.occurrences), 'NCHW': trainer.tensor_shape})
                            data.effect_events.clear()
                finally:
                    if hasattr(data, 'close'):
                        data.close()
                baseline.save(folder / 'signatures.json', observations)
                if expected is None:
                    expected = observations
                if observations != expected:
                    baseline.save(folder / 'mismatch.json', {
                        'differing_keys_per_batch': [[key for key in reference if reference[key] != actual[key]]
                                                     for reference, actual in zip(expected, observations)]})
                    raise ValueError('Resolution-aware baseline/optimized CPU input mismatch.')
                rows.append({'option': option, 'training_imgsz': spec['shared_training_args']['imgsz'],
                             'arm': arm, 'pipeline': pipeline, 'batches': len(observations),
                             'anchor_exposures': len(ids) * 2, 'epochs_observed': '0,1',
                             'input_wall_seconds': time.perf_counter() - began,
                             'exact_input_label_trace_RNG_parity': True,
                             'NCHW': json.dumps(trainer.tensor_shape), 'GPU_seconds': 0})
    write_csv(out / 'cpu_input_readiness.csv', rows)
    return rows


def prepare():
    out = study.create('local_recovery_v1_cpu')
    study.runtime(out, cuda=False)
    started = time.perf_counter()
    baseline.save(out / 'preparation_scope.json', {'version': VERSION, 'GPU_execution_allowed': False,
        'authorization': AUTH, 'assessment': ASSESSMENT, 'repeat_accepted_GPU_tests': False})
    assessment = study.run_path(ASSESSMENT)
    protected = baseline.load(assessment / 'source_inputs.json')
    for row in protected:
        if study.sha256(Path(row['path'])) != row['sha256']:
            raise ValueError('Historical evidence changed: ' + row['path'])
    baseline.verify_freeze(study.run_path(AUTH))
    v3.base.environment_check(baseline.bundle())
    population = v3.base.resolve_data(baseline.bundle(), study.ROOT / 'datasets/VisDrone')
    baseline.save(out / 'data_verification.json', {'train': len(population['train']),
        'development': len(population['val']), 'image_annotation_label_SHA256': 'ALL_VERIFIED',
        'manifest': study.source(baseline.bundle() / 'portable_dataset_manifest.json')})
    plans = []
    for option in OPTIONS:
        spec = candidate_protocol(option)
        folder = out / ('candidate_' + option)
        folder.mkdir()
        baseline.save(folder / 'candidate_protocol.json', spec)
        pair = []
        for arm in spec['arms']:
            target = folder / arm
            target.mkdir()
            plan = make_plan(spec, target, arm, population)
            baseline.save(target / 'protocol.json', plan)
            pair.append(plan)
            plans.append({'option': option, 'arm': arm, 'resolution': plan['args']['imgsz'],
                          'initialization': plan['initialization'], 'expected': plan['expected']})
        verify_pair(*pair)
    baseline.save(out / 'budget_read_only.json', budget_snapshot())
    resolution_audit(out)
    inputs = cpu_inputs(out, population)
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    baseline.save(out / 'adapter_source_audit.json', {'legacy_method_AST_SHA256': LEGACY_PREPROCESS_AST,
        'installed_preprocess_source': inspect.getsource(DetectionTrainer.preprocess_batch),
        'installed_preprocess_SHA256': hashlib.sha256(inspect.getsource(DetectionTrainer.preprocess_batch).encode()).hexdigest(),
        'only_override': 'preprocess_batch; all optimizer, health, dataset, save and epoch hooks inherited.',
        'CPU_model_forward_backward': False})
    baseline.save(out / 'preparation_result.json', {'status': 'CPU_PREPARED_GPU_BLOCKED',
        'RunID': out.name, 'version': VERSION, 'plans': plans, 'CPU_input_passes': len(inputs),
        'CPU_wall_seconds': time.perf_counter() - started, 'GPU_seconds': 0,
        'physical_clearance': 'PENDING_RECORDED_DETAILS', 'resource_admission': 'PENDING',
        'formal100_modified': False})
    print(out, flush=True)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true', required=True)
    parser.parse_args()
    prepare()
