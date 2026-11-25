"""One development-selected policy using the accepted VM37 training executor."""
import argparse
import copy
import csv
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import random
import sys
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'vm37_yolo26_rq4'))
import local_recovery_v1 as recovery

old, study = recovery.baseline, recovery.study
load, save = old.load, old.save
SOURCE = Path(__file__).resolve()
EXPERIMENT = HERE.name
RUNS = study.ROOT / 'datasets/local_runs' / EXPERIMENT
REFERENCE_ID = 'vm37_yolo26_rq4_pilot960_execution_20261114T110947_fe2f01b3'
PHYSICAL_ID = 'vm37_yolo26_rq4_pilot960_authorization_record_cpu_20261114T110935_8e5d6543'
ARM = 'VARIABILITY_GUIDED'
POLICY_ID = 'VM38_GAUSSIAN_P025_V1'
CAPS = {'training': 18000, 'evaluation': 10800, 'stop_reserve': 1800}


def run_path(identity):
    if Path(identity).name != identity or not identity.startswith(EXPERIMENT + '_'):
        raise ValueError('Invalid VM38 RunID.')
    return (RUNS / identity).resolve(strict=True)


def rows_csv(path, rows):
    if not rows:
        raise ValueError('Cannot publish an empty result table.')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def source(path):
    return {'path': str(path), 'sha256': study.sha256(path)}


def recipe():
    return {
        'version': POLICY_ID, 'status': 'APPROVED_TRAINING_RECIPE',
        'scientific_approval': {
            'basis': 'Author-delegated one-policy design; frozen before VM38 training.',
            'performance_validation': 'NOT_EVALUATED_AT_FREEZE',
            'selection_scope': 'POST_HOC_DEVELOPMENT_POLICY_SELECTION'},
        'order': ['HazeAug', 'ExposureAug', 'MotionBlurAug', 'NoiseAug', 'JPEGAug'],
        'families': {'NoiseAug': {
            'probability': .25, 'gaussian_probability': 1.,
            'sigma': {'distribution': 'uniform', 'low': 5.5, 'high': 30.},
            'photons': {'distribution': 'uniform', 'low': 20., 'high': 100.}}},
        'decision_rule': 'Select the family with the largest CONTROL mean AP95 loss over the two frozen resolutions.',
        'probability_rule': 'Fixed 0.25 prior: retain 75% of inputs before ordinary augmentation; no probability search.',
        'range_rule': 'Reuse the original FM_P Gaussian sigma hull; reject exact test levels using the accepted sampler.',
        'unused_schema_field': 'photons is required by the accepted noise schema but never sampled: gaussian_probability=1.',
        'constraints': 'NoiseAug is optional in FM_D. Normal plus GaussianNoise is legal in FM_P; no ShotNoise or LowLight is selected.',
        'training_resolution_scope': '960 is the accepted VM37 exploratory extension, not a TrainResolution leaf in the original FM_D.',
        'geometry': 'Native BGR photometry only, before the unchanged shared resizing and ordinary augmentation.',
        'not_claimed': 'An online controller, an optimal probability, independent confirmation, or full RQ4 completion.'}


def read_values(folder, verify=False):
    index = load(folder / 'result_index.json')
    values = {}
    for row in index:
        config = folder / row['folder']
        if verify:
            for name, field in (('metrics.json', 'metrics_SHA256'), ('bootstrap.json', 'bootstrap_SHA256'),
                                ('predictions.jsonl.gz', 'predictions_SHA256')):
                if study.sha256(config / name) != row[field]:
                    raise ValueError('Accepted evidence changed: ' + str(config / name))
        key = (row['architecture'], row['seed'], row['arm'], row['resolution'], row['context'])
        if key in values or row['status'] != 'COMPLETE':
            raise ValueError('Duplicate or incomplete reference configuration.')
        values[key] = {'metrics': load(config / 'metrics.json'), 'draws': load(config / 'bootstrap.json')['draws']}
    return index, values


def baseline_losses(values):
    import numpy as np
    rows = []
    for resolution in (640, 1280):
        normal = values[('YOLO26s', 42, 'CONTROL', resolution, 'Normal')]
        for context in ('JPEG_Q25', 'JPEG_Q75', 'Gaussian_s15'):
            degraded = values[('YOLO26s', 42, 'CONTROL', resolution, context)]
            for field in ('image_ids', 'samples', 'seed', 'sampling'):
                if normal['draws'][field] != degraded['draws'][field]:
                    raise ValueError('Reference bootstrap indices differ.')
            def point(v):
                return next(r['map50_95'] for r in v['metrics']['per_area'] if r['size'] == 'all')
            delta = np.asarray(normal['draws']['values']['all']['map50_95']) - np.asarray(degraded['draws']['values']['all']['map50_95'])
            interval = np.percentile(delta, [2.5, 97.5])
            rows.append({'resolution': resolution, 'context': context, 'Normal_AP95': point(normal),
                'degraded_AP95': point(degraded), 'absolute_loss': point(normal) - point(degraded),
                'loss_CI95_low': float(interval[0]), 'loss_CI95_high': float(interval[1]),
                'scope': 'DEVELOPMENT_ONLY_POINTWISE_SINGLE_SEED'})
    means = {context: sum(r['absolute_loss'] for r in rows if r['context'] == context) / 2
             for context in ('JPEG_Q25', 'JPEG_Q75', 'Gaussian_s15')}
    if max(means, key=means.get) != 'Gaussian_s15':
        raise ValueError('Frozen one-policy rationale does not match accepted evidence.')
    return rows, means


def verify_match(plan, reference):
    keys = ('architecture', 'initialization', 'COCO_source_SHA256', 'train', 'development',
            'train_labels', 'val_labels', 'expected', 'execution_epochs', 'selection', 'recovery_protocol')
    for arm in ('CONTROL', 'ALL_FIVE'):
        previous = load(reference / ('YOLO26s_seed42_' + arm) / 'protocol.json')
        if any(plan[key] != previous[key] for key in keys):
            raise ValueError('Matched population, source or schedule differs.')
        args = [{k: v for k, v in p['args'].items() if k not in ('data', 'project')} for p in (plan, previous)]
        if args[0] != args[1]:
            raise ValueError('Ordinary training settings differ.')
    if plan['targeted_recipe'] != recipe() or plan['arm'] != ARM:
        raise ValueError('The one frozen VM38 policy changed.')
    return {'status': 'MATCHED_SETTINGS_VERIFIED', 'comparators': ['CONTROL', 'ALL_FIVE'],
            'limits': ['One training seed', 'Development-selected policy', 'Sequential runs and thermal/host conditions differ',
                      'Augmentation streams differ by design; trained trajectories need not be equal',
                      '960/10 epochs is not the original 1280/100 protocol']}


def cpu_checks(out, plan):
    import numpy as np
    from main.training_effects import NativeEffects, validate_recipe
    from input_ops import ArithmeticEffects
    old.v3.base.renderer()
    validate_recipe(recipe())
    image = np.random.default_rng(6).integers(0, 256, (257, 321, 3), dtype=np.uint8)
    image.setflags(write=False)
    before = study.sha256(out / 'policy.json')
    native, fast = NativeEffects(recipe(), 42), ArithmeticEffects(recipe(), 42)
    active = 0
    for number in range(64):
        a, ta = native.apply(image, 'fixture_' + str(number), number % 2, 0)
        b, tb = fast.apply(image, 'fixture_' + str(number), number % 2, 0)
        if not np.array_equal(a, b) or ta != tb or a.shape != image.shape:
            raise ValueError('Targeted native/optimized pixels or RNG differ.')
        if len(ta) != 1 or ta[0]['family'] != 'NoiseAug':
            raise ValueError('An unselected family was activated.')
        if ta[0]['active']:
            active += 1
            if ta[0]['variant'] != 'gaussian' or not 5.5 < ta[0]['parameters']['sigma'] < 30 or ta[0]['parameters']['sigma'] == 15:
                raise ValueError('Noise variant or test-level exclusion differs.')
        elif not np.array_equal(a, image):
            raise ValueError('The no-op path changed pixels.')
    if not 0 < active < 64 or study.sha256(out / 'policy.json') != before:
        raise ValueError('The gate did not exercise both branches or policy bytes changed.')
    # Exercise the real dataset hook with the new recipe; no model forward.
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.torch_utils import init_seeds
    from main.labels import VIS_NAMES
    torch.set_num_threads(2)
    folder = out / 'cpu_dataset'
    folder.mkdir()
    ids = load(old.bundle() / 'cpu_comparison_manifest.json')['image_ids'][:4]
    lookup = {r['image_id']: r for r in plan['train']}
    small = copy.deepcopy(plan)
    small['train'] = [lookup[i] for i in ids]
    listing = folder / 'images.txt'
    listing.write_text('\n'.join(r['image_path'] for r in small['train']) + '\n', encoding='utf-8')
    Factory = recovery.trainer_factory(small, folder)
    trainer = Factory.__new__(Factory)
    trainer.args = get_cfg(overrides=small['args'])
    trainer.device = torch.device('cpu')
    trainer.model = SimpleNamespace(stride=torch.tensor([32]))
    trainer.data = {'names': dict(enumerate(VIS_NAMES)), 'nc': 10, 'channels': 3}
    init_seeds(42, deterministic=True)
    data = trainer.build_dataset(str(listing), mode='train', batch=2)
    observed = []
    try:
        data.set_epoch(0)
        data.close_mosaic(trainer.args)
        by_id = {Path(p).stem: i for i, p in enumerate(data.im_files)}
        for identity in ids:
            result = data[by_id[identity]]
            batch = data.collate_fn([result])
            trainer.preprocess_batch(batch)
            observed.append({'id': identity, 'NCHW': trainer.tensor_shape,
                             'classes': result['cls'].flatten().tolist()})
        events = copy.deepcopy(data.effect_events)
        if data.effects.recipe != recipe() or data.effects.seed != 42:
            raise ValueError('The production dataset did not receive the new policy.')
        if any(any(e['family'] != 'NoiseAug' for e in event['effects']) for event in events):
            raise ValueError('The dataset activated an unselected family.')
    finally:
        if hasattr(data, 'close'):
            data.close()
    result = {'status': 'PASS', 'synthetic_pixel_RNG_parity_cases': 64, 'active_cases': active,
              'production_dataset_images': observed, 'production_events': events,
              'workers': 0, 'GPU_seconds': 0, 'model_forward_backward': False,
              'reuse': 'Accepted VM37 health, FP32 saving, numerical and full-population tests are not repeated.'}
    save(out / 'cpu_checks.json', result)
    return result


def freeze_sources(out, reference, index):
    paths = [SOURCE, Path(recovery.__file__), *old.source_paths(),
             study.VM36 / 'feature_models/fm_p.uvl', study.VM36 / 'feature_models/fm_d.uvl',
             out / 'policy.json', out / 'scientific_protocol.json', out / 'resource_admission.json',
             out / 'cpu_checks.json', out / 'YOLO26s_seed42_VARIABILITY_GUIDED/protocol.json',
             out / 'evaluation/protocol.json', out / 'evaluation/evaluation_stage.json',
             reference / 'execution_freeze.json', reference / 'execution_result.json', reference / 'evaluation/result_index.json']
    for arm in ('CONTROL', 'ALL_FIVE'):
        folder = reference / ('YOLO26s_seed42_' + arm)
        paths += [folder / 'protocol.json', folder / 'training_acceptance.json']
        paths.append(Path(load(folder / 'training_acceptance.json')['selected_checkpoint']['path']))
    for row in index:
        folder = reference / 'evaluation' / row['folder']
        paths += [folder / n for n in ('acceptance.json', 'metrics.json', 'bootstrap.json')]
    return {str(path): study.sha256(path) for path in paths}


def prepare():
    out = study.new_run({'output_root': str(RUNS)}, EXPERIMENT, 'targeted960')[0]
    study.runtime(out, cuda=False)
    started = time.perf_counter()
    try:
        authorization = study.run_path(recovery.AUTH)
        old.verify_freeze(authorization)
        old.v3.base.verify_bundle(old.bundle())
        old.v3.base.environment_check(old.bundle())
        reference = study.run_path(REFERENCE_ID)
        if load(reference / 'execution_result.json')['status'] != 'PILOT960_MATCHED_COMPLETE_DEVELOPMENT_ONLY':
            raise ValueError('The previous matched pilot is not complete.')
        for path, digest in load(reference / 'execution_freeze.json')['files'].items():
            if study.sha256(Path(path)) != digest:
                raise ValueError('Accepted VM37 source changed.')
        index, values = read_values(reference / 'evaluation', verify=True)
        if len(index) != 16:
            raise ValueError('The accepted evaluation is incomplete.')
        losses, means = baseline_losses(values)
        rows_csv(out / 'policy_selection_evidence.csv', losses)
        save(out / 'policy.json', recipe())
        physical_path = study.run_path(PHYSICAL_ID) / 'physical_and_execution_authorization.json'
        physical = load(physical_path)
        if not all(physical.get(k) is True for k in ('hard_ventilated_surface', 'unobstructed_intake_exhaust', 'fans_normal', 'reliable_AC_adapter')):
            raise PermissionError('The latest physical confirmation is incomplete.')
        save(out / 'author_authorization.json', {
            'source': 'Current VM38 user instruction: autoexecute if original safety and complete-cost gates pass.',
            'original_budget': recovery.AUTH, 'new_allocation': False, 'no_deadline_extension': True,
            'one_training_attempt': True, 'one_frozen_policy': True, 'physical_confirmation': source(physical_path),
            'remaining_unknowns': ['Room temperature', 'OEM mode', 'Fan RPM'],
            'scope_extension': 'VM38 targeted arm and eight evaluations under the existing remaining authorization.'})
        protocol = {'id': 'VM38_TARGETED960_10E_SEED42_V1', 'architectures': ['YOLO26s'], 'seeds': [42],
            'arms': [ARM], 'epochs': 10, 'training_images': 6471, 'development_images': 548,
            'training_imgsz': 960, 'evaluation_resolutions': [640, 1280],
            'evaluation_contexts': ['Normal', 'JPEG_Q25', 'JPEG_Q75', 'Gaussian_s15'],
            'policy': source(out / 'policy.json'), 'reference_RunID': REFERENCE_ID,
            'selection_evidence_mean_loss': means, 'bootstrap_samples': 200, 'bootstrap_seed': 20261112,
            'primary': 'Absolute AP50:95, both Normal and degraded; reduced degradation loss alone is insufficient.',
            'secondary': ['Tiny AP', 'Clean accuracy cost', 'Degradation loss', 'Paired pointwise image intervals'],
            'policy_search': 'One predefined policy only. No AP-dependent retry or tuning.',
            'scope': 'POST_HOC_DEVELOPMENT_POLICY_SELECTION; NOT_INDEPENDENT_CONFIRMATION',
            'constraints': {'FM_D_augmentation_projection': ['NoiseAug'], 'FM_P_reference': ['Normal', 'Gauss_s15'],
                'training_strengths': 'Continuous native Gaussian hull, not an enumerated FM_P leaf assignment.',
                'Train960': 'VM37 exploratory extension; not expressible in the unchanged original FM_D.'},
            'formal1280_100_epochs': 'UNCHANGED_NOT_EXECUTED'}
        save(out / 'scientific_protocol.json', protocol)
        train = out / ('YOLO26s_seed42_' + ARM)
        train.mkdir()
        population = old.v3.base.resolve_data(old.bundle(), study.ROOT / 'datasets/VisDrone', verify=False)
        if len(population['train']) != 6471 or len(population['val']) != 548:
            raise ValueError('The accepted population differs.')
        plan = recovery.make_plan(recovery.candidate_protocol('B'), train, 'CONTROL', population)
        plan.update(arm=ARM, targeted_recipe=recipe(), mode=protocol['id'],
                    scope=protocol['scope'], augmentation_label='Development-selected Gaussian-only p=0.25',
                    GPU_execution_authorized=True)
        save(train / 'protocol.json', plan)
        save(out / 'comparability.json', verify_match(plan, reference))
        cpu_checks(out, plan)
        evaluation = out / 'evaluation'
        evaluation.mkdir()
        save(evaluation / 'protocol.json', {'scope': protocol['id'], 'completed_arms_required': 1})
        spec = load(old.bundle() / 'evaluation_manifest.json')
        if [r['id'] for r in spec['contexts']] != protocol['evaluation_contexts']:
            raise ValueError('The four accepted evaluation contexts differ.')
        save(evaluation / 'evaluation_stage.json', {'contexts': spec['contexts']})
        budget = recovery.budget_snapshot()
        budget.pop('GPU_seconds_this_run', None)
        if sum(CAPS.values()) > budget['remaining_seconds']:
            raise PermissionError('RESOURCE_BLOCKED_COMPLETE_WORK_DOES_NOT_FIT')
        save(out / 'resource_admission.json', {'status': 'COMPLETE_COST_FITS_ORIGINAL_BALANCE',
            'caps_seconds': CAPS, 'total_reserved_seconds': sum(CAPS.values()), 'budget_at_freeze': budget,
            'measured_reference_seconds': {'CONTROL_training': 8131.553, 'ALL_FIVE_training': 13627.289,
                                           'sixteen_configuration_evaluation': 10808.362},
            'extrapolation': '18,000s new training cap exceeds measured ALL_FIVE by 32%; 10,800s evaluation cap reserves nearly the prior 16-config cost for eight. This is a ceiling, not measured VM38 speed.',
            'thermal_safety': 'Reuse successful 960 admission, retain 90C immediate and 87C/30s sustained stops. Fewer effects do not establish thermal safety.',
            'no_budget_reset': True, 'no_retry': True})
        save(out / 'inputs.json', {'reference_RunID': REFERENCE_ID,
            'reference_index': source(reference / 'evaluation/result_index.json'),
            'initialization': plan['initialization'], 'dataset': source(old.bundle() / 'portable_dataset_manifest.json'),
            'bundle_manifest': source(old.bundle() / 'bundle_manifest.json'),
            'FM_P': source(study.VM36 / 'feature_models/fm_p.uvl'),
            'FM_D': source(study.VM36 / 'feature_models/fm_d.uvl'),
            'accepted_source_reuse': 'VM36 accepted 84-pair evidence and VM37 V3.1/960 results remain unchanged; no old experiment repeated.'})
        save(out / 'execution_freeze.json', {'frozen_UTC': datetime.now(timezone.utc).isoformat(),
            'files': freeze_sources(out, reference, index)})
        save(out / 'preparation_result.json', {'status': 'CPU_PREPARED', 'CPU_seconds': time.perf_counter() - started,
             'GPU_seconds': 0, 'new_training_configs': 1, 'new_evaluation_configs': 8, 'reused_configs': 16})
        (out / 'diary.md').write_text('# VM38 targeted augmentation\n\nOne Gaussian-only policy was selected from accepted development losses and frozen before training. The 0.25 probability is a fixed prior. Prior runs remain read-only.\n', encoding='utf-8')
        print('VM38_PREPARED', out, flush=True)
        return out
    except BaseException:
        save(out / 'preparation_failure.json', {'traceback': traceback.format_exc(), 'GPU_started': False})
        raise


def check(parent):
    for path, digest in load(parent / 'execution_freeze.json')['files'].items():
        if study.sha256(Path(path)) != digest:
            raise ValueError('Frozen evidence changed: ' + path)
    old.verify_freeze(study.run_path(recovery.AUTH))
    if old.available(study.run_path(recovery.AUTH)) <= 0:
        raise PermissionError('Original authorization expired or exhausted.')


def context(out):
    ticket = load(out / 'worker_ticket.json')
    parent = run_path(ticket['parent_RunID'])
    check(parent)
    if out.parent != parent or study.sha256(out / 'protocol.json') != ticket['protocol_SHA256']:
        raise PermissionError('Worker boundary or plan changed.')
    if ticket.get('evaluation_stage_SHA256') and study.sha256(out / 'evaluation_stage.json') != ticket['evaluation_stage_SHA256']:
        raise ValueError('The evaluation stage changed.')
    if datetime.now(timezone.utc) >= datetime.fromisoformat(load(out / 'WORKER_LEASE.json')['deadline_UTC']):
        raise PermissionError('Worker lease expired.')
    return load(out / 'protocol.json'), parent


def worker(out, mode):
    study.runtime(out, cuda=True)
    plan, parent = context(out)
    if mode == 'train':
        Factory = recovery.trainer_factory(plan, out)
        with patch.object(old, 'worker_context', context), patch.object(old.v3, 'trainer_factory', lambda *_: Factory):
            return old.train_worker(out)
    research = old.research
    original_load = research.load
    spec = load(old.bundle() / 'evaluation_manifest.json')
    spec['contexts'] = load(out / 'evaluation_stage.json')['contexts']
    def mapped_load(path):
        if Path(path) == old.bundle() / 'vm38_protocol.json':
            return load(parent / 'scientific_protocol.json')
        if Path(path) == old.bundle() / 'formal_evaluation_manifest.json':
            return spec
        return original_load(path)
    def admitted(*_):
        context(out)
        accepted = load(parent / ('YOLO26s_seed42_' + ARM) / 'training_acceptance.json')
        if accepted['completed_epochs'] != 10 or accepted['optimizer_updates'] != 1816 or accepted['EMA_updates'] != 1816:
            raise ValueError('New training did not complete its frozen budget.')
    with patch.object(research, 'load', mapped_load), patch.object(research, 'check_worker_authorization', admitted), \
            patch.object(research, 'paired_contrasts', lambda _: []):
        research.evaluate(old.bundle(), study.ROOT / 'datasets/VisDrone', parent, 'vm38')
    return 'COMPLETE'


def run_stage(parent, out, stage, cap, mode):
    check(parent)
    authorization = study.run_path(recovery.AUTH)
    if old.available(authorization) < cap + CAPS['stop_reserve']:
        raise PermissionError('The stage and safety reserve exceed the original balance.')
    old.cooldown(out)
    save(out / 'resource_before.json', old.resource_check())
    deadline = min(datetime.now(timezone.utc) + timedelta(seconds=cap),
                   datetime.fromisoformat(load(authorization / 'authorization.json')['absolute_expiry_UTC']))
    save(out / 'WORKER_LEASE.json', {'deadline_UTC': deadline.isoformat(), 'cap_seconds': cap})
    save(out / 'worker_ticket.json', {'parent_RunID': parent.name, 'protocol_SHA256': study.sha256(out / 'protocol.json'),
        'evaluation_stage_SHA256': study.sha256(out / 'evaluation_stage.json') if mode == 'evaluate' else None})
    command = [sys.executable, '-B', '-X', 'utf8', str(SOURCE), '--worker', str(out), '--mode', mode]
    save(out / 'command.json', command)
    started, result = time.perf_counter(), None
    try:
        result = old.run_owned(command, out, cap, load(old.bundle() / 'safety_policy.json'),
            dict(os.environ, CUDA_VISIBLE_DEVICES='0', PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='2'))
        diagnosis = recovery.stage_diagnosis(out, result, mode)
        save(out / 'stage_diagnosis.json', diagnosis)
        if diagnosis['status'] != 'STAGE_COMPLETED':
            raise RuntimeError(diagnosis['primary_reason'])
    finally:
        save(authorization / 'charges' / (parent.name + '_' + stage + '.json'), {
            'RunID': parent.name, 'stage': stage, 'stage_path': str(out), 'hard_cap': cap,
            'charged_seconds': time.perf_counter() - started, 'supervisor': result,
            'finished_UTC': datetime.now(timezone.utc).isoformat()})


def summarize(parent, status, error=None):
    import numpy as np
    reference = study.run_path(REFERENCE_ID)
    all_index, values = read_values(reference / 'evaluation')
    if (parent / 'evaluation/result_index.json').exists():
        new_index, added = read_values(parent / 'evaluation', verify=True)
        if len(new_index) != 8:
            raise ValueError('Eight new configurations were not completed.')
        values.update(added)
        all_index += new_index
    metrics, losses, comparisons = [], [], []
    for key, value in values.items():
        architecture, seed, arm, resolution, condition = key
        for row in value['metrics']['per_area']:
            metrics.append({'architecture': architecture, 'seed': seed, 'arm': arm,
                            'resolution': resolution, 'condition': condition, **row})
        normal = values[(architecture, seed, arm, resolution, 'Normal')]
        for size in ('all', 'tiny'):
            point = lambda v: next(r['map50_95'] for r in v['metrics']['per_area'] if r['size'] == size)
            a, b = normal['draws']['values'][size]['map50_95'], value['draws']['values'][size]['map50_95']
            interval = np.percentile(np.asarray(a) - np.asarray(b), [2.5, 97.5]).tolist()
            losses.append({'arm': arm, 'resolution': resolution, 'condition': condition, 'size': size,
                           'degradation_loss': point(normal) - point(value), 'CI95_low': interval[0], 'CI95_high': interval[1]})
    if any(k[2] == ARM for k in values):
        for comparator in ('CONTROL', 'ALL_FIVE'):
            selected = {}
            for key, value in values.items():
                if key[2] in (comparator, ARM):
                    alias = 'CONTROL' if key[2] == comparator else 'ALL_FIVE'
                    selected[(key[0], key[1], alias, key[3], key[4])] = value
            for row in old.research.paired_contrasts(selected):
                row['comparator'] = comparator
                row['strategy'] = ARM
                if 'ALL_FIVE_minus_CONTROL' in row:
                    row['strategy_minus_comparator'] = row.pop('ALL_FIVE_minus_CONTROL')
                comparisons.append(row)
    rows_csv(parent / 'configuration_metrics.csv', metrics)
    rows_csv(parent / 'degradation_losses.csv', losses)
    if comparisons:
        save(parent / 'paired_comparisons.json', comparisons)
        rows_csv(parent / 'paired_comparisons.csv', comparisons)
    save(parent / 'joint_result_index.json', {'reference_RunID': REFERENCE_ID, 'new_RunID': parent.name,
         'entries': all_index, 'original_results_copied_or_modified': False})
    stages = []
    for folder in (parent / ('YOLO26s_seed42_' + ARM), parent / 'evaluation'):
        item = {'stage': folder.name}
        for name in ('training_acceptance.json', 'safety_supervisor.json', 'stage_diagnosis.json', 'evaluation_result.json'):
            if (folder / name).exists():
                item[name] = load(folder / name)
        if (folder / 'gpu_telemetry.jsonl').exists():
            telemetry = [json.loads(line) for line in (folder / 'gpu_telemetry.jsonl').read_text().splitlines()]
            item['maximum_temperature_C'] = max((r['temperature_C'] for r in telemetry), default=None)
            item['peak_NVML_MiB'] = max((r['used_MiB'] for r in telemetry), default=None)
        stages.append(item)
    check(parent)
    budget = recovery.budget_snapshot()
    budget['GPU_seconds_this_run'] = sum(load(p)['charged_seconds'] for p in old.charge_files(study.run_path(recovery.AUTH))
                                        if load(p)['RunID'] == parent.name)
    result = {'status': status, 'error': error, 'stages': stages, 'budget': budget,
              'new_evaluation_configurations': len(all_index) - 16, 'original_VM37_preserved': True,
              'scope': 'POST_HOC_DEVELOPMENT_POLICY_SELECTION / SINGLE_TRAINING_SEED / NOT_FULL_RQ4'}
    save(parent / 'execution_result.json', result)
    report = study.ROOT / 'datasets/reportcn' / ('vm38_variability_guided_augmentation_科学比较_' + parent.name[-8:] + '.md')
    lines = ['# VM38 Variability引导增强科学比较', '', '**' + status + '**', '',
        'RunID：`' + parent.name + '`；对照来源：`' + REFERENCE_ID + '`。',
        '范围：DEVELOPMENT_ONLY / POST_HOC_DEVELOPMENT_POLICY_SELECTION。仅seed42，不是独立确认或完整RQ4。', '',
        '## 决策与冻结', '',
        '既有CONTROL在Gaussian sigma15上的平均绝对AP损失最大，故只选择NoiseAug中的Gaussian；概率固定0.25，sigma均匀5.5–30并沿用精确测试值排除。概率没有通过本轮AP搜索优化。',
        '另外75%仅不受新增定向增强影响，常规增强仍然存在。photons字段仅为兼容原schema保留，实际不会抽取ShotNoise。',
        '原FM_P允许Normal+Gaussian；FM_D允许单独NoiseAug。960训练沿用VM37探索扩展，并非原FM_D的Train640/Train1280叶节点。原UVL没有修改。',
        '策略SHA：`' + study.sha256(parent / 'policy.json') + '`；冻结清单：`' + str(parent / 'execution_freeze.json') + '`。', '',
        '## 公平性与限制', '',
        'YOLO26s、960、10轮、seed42、6471训练图、548开发图、同一健康初始化、AdamW、batch2/nbs64、FP32、workers0、学习率和普通增强均与既有两组相同。固定epoch10 EMA。',
        'close_mosaic=10且总训练10轮，因此epoch0关闭Mosaic。原1280/100轮方案不变。',
        '新增策略由已经观察的同一val退化损失选择，再在同一val评价，存在明确的开发选择偏差；不能称作独立泛化证据。三组顺序执行，温度、宿主负载和耗时也可能不同。',
        '## AP50:95与Tiny AP', '',
        '数值为百分数；置信区间和CSV使用AP原值。', '',
        '|分辨率|条件|CONTROL AP95 / Tiny|ALL_FIVE AP95 / Tiny|定向 AP95 / Tiny|',
        '|---|---|---:|---:|---:|']
    def cell(arm, res, condition):
        rows = [r for r in metrics if r['arm'] == arm and r['resolution'] == res and r['condition'] == condition]
        if not rows:
            return 'NOT_EVALUATED'
        point = {r['size']: r['map50_95'] for r in rows}
        return f"{100*point['all']:.3f} / {100*point['tiny']:.3f}"
    for resolution in (640, 1280):
        for condition in ('Normal', 'JPEG_Q25', 'JPEG_Q75', 'Gaussian_s15'):
            lines.append('|' + '|'.join([str(resolution), condition, cell('CONTROL', resolution, condition),
                                         cell('ALL_FIVE', resolution, condition), cell(ARM, resolution, condition)]) + '|')
    lines += ['', '## 配对差异与clean代价', '',
        '200次固定图像配对Bootstrap，seed20261112，与既有两组使用相同图像位置索引。每次重新累计COCO AP；不是平均单图AP。区间为逐项百分位95%区间，未作多重比较或训练seed推断。', '',
        '|分辨率|条件|相对组|AP95差 / 95%CI（百分点）|', '|---|---|---|---:|']
    for row in comparisons:
        if row['size'] == 'all' and row['metric'] == 'map50_95' and row['status'] == 'ESTIMATED':
            lo, hi = row['difference_CI95']
            lines.append(f"|{row['resolution']}|{row['context']}|{row['comparator']}|{100*row['strategy_minus_comparator']:+.3f} [{100*lo:+.3f}, {100*hi:+.3f}]|")
    lines += ['', 'Normal差值即clean准确率收益/代价。退化损失定义为同一策略Normal AP减退化AP；损失变小但绝对AP下降不能称作本任务成功。Tiny、AR及各尺寸对比见CSV/JSON。', '',
              '## 实际资源与验收', '', '```json', json.dumps(result, ensure_ascii=False, indent=2), '```', '',
              '本轮上限：训练18000秒、8项评价10800秒、停止余量1800秒。沿用原36小时总账和2026-11-20 21:20:36 UTC固定截止；没有新增或延长授权。',
              '90°C立即停止、87°C连续30秒停止；OOM/非有限值/损坏输入或保存失败均不重试。未记录的功耗、风扇转速与室温不作推测。', '',
              '## 复现与证据', '',
              '`' + str(SOURCE) + '`：仅使用本轮冻结RunID执行一次，不重复历史组。',
              '策略、源码SHA、参数、数据清单、初始化SHA、CPU连接测试及预算：`' + str(parent) + '`。',
              '逐配置原始预测、GT、COCO指标、200次Bootstrap数组：`' + str(parent / 'evaluation') + '`。',
              '训练参数、逐batch/step健康证据、10份FP32权重及保存重载证明：`' + str(parent / ('YOLO26s_seed42_' + ARM)) + '`。',
              '本次只有一个预定义策略，无论正负均保留，不按新AP调参或自动开展下一轮。test-dev/UAVDT边界保持不变。']
    if error:
        lines += ['', '停止原因：`' + error + '`；未完成指标保持NOT_EVALUATED。']
    report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    save(parent / 'final_report.json', source(report))
    meta = load(parent / 'run.json')
    meta.update(status=status, finished_UTC=datetime.now(timezone.utc).isoformat())
    study.save(parent / 'run.json', meta, parent, replace=True)
    (parent / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>VM38</h1><p>' + status +
        '</p><a href="configuration_metrics.csv">Metrics</a> | <a href="paired_comparisons.json">Paired comparisons</a> | <a href="execution_result.json">Execution</a>', encoding='utf-8')
    study.seal(parent)
    print('VM38_FINISHED', status, str(report), flush=True)


def execute(parent):
    study.runtime(parent, cuda=False)
    check(parent)
    if load(parent / 'cpu_checks.json')['status'] != 'PASS':
        raise ValueError('The new policy CPU connection failed.')
    if old.available(study.run_path(recovery.AUTH)) < sum(CAPS.values()):
        raise PermissionError('The complete training and evaluation no longer fit.')
    lock = study.RUNS / '.pilot10_gpu.lock'
    if lock.exists() or (study.RUNS / '.gpu.lock').exists():
        raise RuntimeError('An existing GPU worker lock is present.')
    old.resource_check()
    save(parent / 'execution_ticket.json', {'RunID': parent.name, 'PID': os.getpid(),
         'started_UTC': datetime.now(timezone.utc).isoformat(), 'attempt': 1, 'automatic_retry': False})
    save(lock, {'RunID': parent.name, 'PID': os.getpid()})
    status, error = 'PARTIAL_WITH_LIMITATIONS', None
    try:
        run_stage(parent, parent / ('YOLO26s_seed42_' + ARM), 'training', CAPS['training'], 'train')
        run_stage(parent, parent / 'evaluation', 'evaluation', CAPS['evaluation'], 'evaluate')
        status = 'VM38_TARGETED_PILOT_COMPLETE_DEVELOPMENT_ONLY'
    except BaseException as exc:
        error = str(exc)
        save(parent / 'failure.json', {'error': error, 'traceback': traceback.format_exc(), 'automatic_retry': False})
    finally:
        if lock.exists() and load(lock).get('RunID') == parent.name:
            lock.unlink()
        summarize(parent, status, error)
    return 0 if status == 'VM38_TARGETED_PILOT_COMPLETE_DEVELOPMENT_ONLY' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--execute')
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--mode', choices=('train', 'evaluate'), default='train')
    args = parser.parse_args()
    if args.worker:
        raise SystemExit(0 if worker(args.worker.resolve(strict=True), args.mode) in ('TRAINING_COMPLETED', 'COMPLETE') else 1)
    if args.prepare:
        prepare()
    elif args.execute:
        raise SystemExit(execute(run_path(args.execute)))
    else:
        parser.error('Choose --prepare or --execute RunID.')
