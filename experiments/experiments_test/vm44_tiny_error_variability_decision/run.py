"""VM44 CPU-only error analysis and measured-evidence detector selection."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

os.environ.update(CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1', YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true', OMP_NUM_THREADS='2')
from vm44_io import *
from selector import Controller, configuration_id, disjoint_groups, legal, state

AUTHOR = Path(r'C:\Users\w\.codex\attachments\a20ea97a-7982-4d97-9791-5459331e61bc\已粘贴的文本.txt')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def budget_snapshot():
    folder = HERE.parent/'vm43_external_pilot'
    sys.path.insert(0, str(folder))
    import vm43_common
    return vm43_common.budget()


def external_inputs(out):
    entries = seal_entries(V43/'SHA256SUMS.txt')
    entries.update(seal_entries(V43/'analysis_SHA256SUMS.txt'))
    for name in ('configuration_metrics.csv', 'per_class_metrics.csv', 'paired_comparisons.csv', 'bootstrap_intervals.csv', 'timing_by_image.csv',
                 'gap_status.csv', 'execution_protocol.json', 'scientific_acceptance.json', 'final_acceptance.json', 'source_provenance.json'):
        verify_source(V43/name, entries)
    if read(V43/'scientific_acceptance.json')['status'] != 'REAL_FOUR_CELLS_COMPLETED':
        raise ValueError('VM43 is not scientifically complete.')
    spec = read(V43/'execution_protocol.json')
    gt_ref = spec['GT']; payload = read(gt_ref['path'], gt_ref['SHA256'], 'EXTERNAL_DESCRIPTIVE_GT_ONLY')
    if len(payload['records']) != 1610 or len(payload['coco']['annotations']) != 74687:
        raise ValueError('External population differs.')
    for cell in spec['cells']:
        folder = V43/cell['cell_id']
        for name in ('predictions.jsonl.gz', 'image_manifest.jsonl', 'metrics.json', 'bootstrap.json', 'acceptance.json'):
            verify_source(folder/name, entries, 'EXTERNAL_DESCRIPTIVE_ONLY')
        receipt = read(folder/'acceptance.json')
        if receipt['checkpoint'] != cell['checkpoint'] or receipt['images'] != 1610 or receipt['GT']['SHA256'] != gt_ref['SHA256']:
            raise ValueError('External checkpoint/GT identity differs.')
    save(out/'external_input_verification.json', {'status': 'PASS', 'RunID': V43.name, 'images': 1610, 'valid_GT': 74687,
        'scope': 'LIMITED_EXTERNAL_EXPLORATORY', 'historical_independence': 'UNKNOWN', 'controller_use': 'FORBIDDEN',
        'image_pixels_opened': 0, 'Gaussian_regeneration': False})
    return spec, payload


def development_inputs(out):
    seals = {r['path']: r['sha256'] for r in read(V40/'output_checksums.json')['files']}
    pairs = csv_read(V40/'VM36_84_pair_metrics.csv', seals[str(V40/'VM36_84_pair_metrics.csv')])
    conditions = csv_read(V40/'VM36_21_measured_conditions.csv', seals[str(V40/'VM36_21_measured_conditions.csv')])
    settings = csv_read(V40/'VM36_4_detector_settings.csv', seals[str(V40/'VM36_4_detector_settings.csv')])
    index = read(V40/'VM36_source_index.json', seals[str(V40/'VM36_source_index.json')])
    source_index = {r['pair_id']: r for r in index}
    if len(pairs) != 84 or len(conditions) != 21 or len(settings) != 4:
        raise ValueError('Development grid differs.')
    parsed, candidates, support = [], {}, {}
    source_details = []
    for row in pairs:
        folder = Path(row['source_folder']); old = source_index[row['pair_id']]
        values = {name: read(folder/name, old['SHA256'][name], 'DEVELOPMENT_ONLY') for name in ('configuration.json', 'metrics.json', 'bootstrap.json', 'timing.json', 'verification.json')}
        spec, metrics, boot, timing, verification = [values[n] for n in ('configuration.json', 'metrics.json', 'bootstrap.json', 'timing.json', 'verification.json')]
        if metrics['images'] != 548 or verification['status'] != 'VALIDATED' or row['observed_execution_status'] != 'VALIDATED':
            raise ValueError('Unaccepted development measurement.')
        if timing['FP32'] is not True or timing['TF32'] is not False:
            raise ValueError('Development runtime precision differs.')
        for key, metric, area in [('AP50', 'map50', 'all'), ('AP50_95', 'map50_95', 'all'), ('Tiny_AP', 'map50_95', 'tiny')]:
            actual = next(r[metric] for r in metrics['per_area'] if r['size'] == area)
            if not math.isclose(float(row[key]), actual, abs_tol=1e-12):
                raise ValueError('Development AP table differs from source.')
        tiny = next(r['ci95'] for r in boot['intervals'] if r['size'] == 'tiny' and r['metric'] == 'map50_95')
        if not all(math.isclose(float(row[k]), v, abs_tol=1e-12) for k, v in zip(('Tiny_CI_low', 'Tiny_CI_high'), tiny)):
            raise ValueError('Development interval differs.')
        if not math.isclose(float(row['cached_pipeline_mean_ms']), timing['mean_pipeline_ms'], abs_tol=1e-12):
            raise ValueError('Latency boundary differs.')
        result = {**row, 'D': spec['d']['id'], 'leaves': spec['p']['leaves'], 'evidence_scope': 'DEVELOPMENT_ONLY'}
        for key in ('AP50', 'AP50_95', 'Tiny_AP', 'Tiny_CI_low', 'Tiny_CI_high', 'cached_pipeline_mean_ms', 'latency_CI_low_ms', 'latency_CI_high_ms'):
            result[key] = float(row[key])
        if state(result) != row['original_rule_state'] or configuration_id(spec['p']['leaves']) != row['condition_id']:
            raise ValueError('Stored task decision or P ID differs.')
        parsed.append(result)
        if 'RollingShutter' not in spec['p']['leaves']:
            support[row['condition_id']] = 'MEASURED_IDENTITY_COORDINATES_ORIGINAL_GT_TASK'
        if spec['d']['id'] not in candidates:
            candidates[spec['d']['id']] = {'declared_leaves': spec['declared_FM_D']['selected_features'], 'checkpoint': spec['d'],
                'declaration_source': str(folder/'configuration.json'), 'full_training_feature_provenance': False,
                'training_limit': 'Historical Albu activation and selected best epoch UNKNOWN; declared leaf assignment is not proof of all training features.'}
        if spec['d'] != candidates[spec['d']['id']]['checkpoint']:
            raise ValueError('Historical D identity differs between schema versions.')
        source_details.append({'pair_id': row['pair_id'], 'folder': str(folder), 'prediction_reference_SHA256_NOT_RECOMPUTED': old['SHA256']['predictions.jsonl.gz'],
                               'GT_reference_SHA256_NOT_RECOMPUTED': old['SHA256']['ground_truth.json'], 'metadata_SHA_verified': True})
    import torch
    from main.training_health import model_health
    from main.labels import VIS_NAMES
    torch.set_num_threads(2)
    checked = {}
    for key, info in candidates.items():
        ck = info['checkpoint']; file = Path(ck['path'])
        source(file, ck['sha256'], 'DEVELOPMENT_FIXED_CHECKPOINT_CPU_ONLY')
        if str(file) not in checked:
            checkpoint = torch.load(file, map_location='cpu', weights_only=False)
            model = checkpoint.get('ema') or checkpoint['model']
            model_health(model)
            names = list(model.names.values()) if isinstance(model.names, dict) else model.names
            if names != VIS_NAMES:
                raise ValueError('Development native10 class order differs.')
            args = read_yaml(file.parent/'args.yaml')
            source(file.parent/'results.csv')
            if args['imgsz'] != 1280 or args['epochs'] != 100 or args['seed'] != 42 or 'visdrone_10class.yaml' not in args['data']:
                raise ValueError('Training origin does not match catalog.')
            checked[str(file)] = {'health': 'FINITE_MODEL_AND_BN_NONNEGATIVE', 'classes': names, 'args': args,
                                  'selected_epoch': 'UNKNOWN', 'initialization_bytes_SHA': 'NOT_PRESENT_IN_THIS_CATALOG; model name only', 'CUDA_initialized': torch.cuda.is_initialized()}
            del model, checkpoint
        info['eligible_historical_checkpoint'] = ck['precision'] == 'FP32' and ck['resolution'] in (640, 1280)
        info['CPU_health'] = checked[str(file)]['health']
        info['source_data'] = 'Historical VisDrone native10 training; val used for best selection; not an independent model comparison.'
    save(out/'development_evidence_verification.json', {'pairs': 84, 'P': 21, 'D': 4, 'scope': 'DEVELOPMENT_ONLY', 'metadata_verified': source_details,
        'historical_checkpoints': checked, 'full_FM_D_training_provenance': 'NOT_FULLY_VERIFIED', 'external_data_used': False})
    return parsed, conditions, candidates, support


def read_yaml(path):
    import yaml
    source(path)
    return yaml.safe_load(Path(path).read_text(encoding='utf-8'))


def controller_analysis(out, cnf, rows, conditions, candidates, support):
    controller = Controller(cnf, rows, candidates, support)
    traces, selections, abstentions, baselines, holdout = [], [], [], [], []
    for name, c in candidates.items():
        if not legal(cnf['fm_d'], c['declared_leaves']):
            raise ValueError('Measured D declaration violates original UVL.')
    for condition in conditions:
        leaves = condition['selected_leaves'].split('|'); pid = condition['configuration_id']
        answer = controller.select(leaves)
        traces.append(answer)
        if answer['selected']:
            picked = answer['selected']
            selections.append({'P': pid, 'D': picked['D'], 'Tiny_AP': picked['Tiny_AP'], 'Tiny_CI_low': picked['Tiny_CI_low'], 'Tiny_CI_high': picked['Tiny_CI_high'],
                               'mean_latency_ms': picked['cached_pipeline_mean_ms'], 'latency_CI_high_ms': picked['latency_CI_high_ms'], 'state': 'PASS', 'scope': 'IN_SAMPLE_LOOKUP_RESULT'})
        else:
            abstentions.append({'P': pid, 'status': answer['status'], 'reason': answer['reason'], 'scope': 'IN_SAMPLE_LOOKUP_RESULT'})
        choices = [r for r in rows if r['condition_id'] == pid]
        for label, fixed in [('BASELINE_1_FIXED_V8_640', 'ck05_Infer640'), ('BASELINE_2_PRIMARY_26_1280', 'ck01_Infer1280')]:
            row = next(r for r in choices if r['D'] == fixed)
            baselines.append({'method': label, 'P': pid, 'selected_D': fixed, 'task_state': state(row), 'abstain': False, 'Tiny_AP': row['Tiny_AP'], 'latency_ms': row['cached_pipeline_mean_ms'], 'evaluation': 'IN_SAMPLE_LOOKUP_RESULT'})
        for label in ('BASELINE_3_MEASURED_LOOKUP_ORACLE', 'VARIABILITY_CONTROLLER'):
            r = answer['selected']
            baselines.append({'method': label, 'P': pid, 'selected_D': r['D'] if r else None, 'task_state': 'PASS' if r else 'NO_CONFIRMED_FEASIBLE_DETECTOR',
                              'abstain': not bool(r), 'Tiny_AP': r['Tiny_AP'] if r else None, 'latency_ms': r['cached_pipeline_mean_ms'] if r else None, 'evaluation': 'IN_SAMPLE_LOOKUP_RESULT'})
        train, test = [r for r in rows if r['condition_id'] != pid], choices
        disjoint_groups(train, test)
        held = Controller(cnf, train, candidates, support).select(leaves)
        holdout.append({'heldout_P': pid, 'train_P_groups': 20, 'test_rows': 4, 'status': held['status'], 'selected_D': None if held['selected'] is None else held['selected']['D'],
                        'group_disjoint': True, 'evaluation': 'GROUP_HOLDOUT_RESULT', 'model': 'EXACT_LOOKUP_ABSTENTION_NO_LEARNING'})
    for p in conditions:
        baselines.append({'method': 'BASELINE_4_FEATURE_RULE', 'P': p['configuration_id'], 'selected_D': None, 'task_state': 'INSUFFICIENT_DATA_FOR_GENERALIZABLE_RULE',
                          'abstain': True, 'Tiny_AP': None, 'latency_ms': None, 'evaluation': 'NOT_EVALUATED'})
    rules = {'id': 'VM44_MEASURED_INTERVAL_SELECTOR_V1', 'inputs': ['Original UVL leaves', 'task thresholds', 'eligible measured historical D', 'development evidence only'],
        'ranking': 'Minimum measured cached latency among interval PASS; highest Tiny lower CI then D ID as ties', 'task': {'Tiny_AP95_min': .10, 'cached_mean_latency_ms_max': 100},
        'models': cnf, 'candidates': candidates, 'semantic_support': support, 'development_rows': rows,
        'training_recipe_selection': 'NOT_IMPLEMENTED_NO_NEW_TRAINED_CHECKPOINT', 'full_training_semantics_mode': 'strict_training=True abstains on unknown Albu/best-epoch provenance',
        'lookup_oracle': 'Same measured evidence and interval criteria, hence intentionally identical in-sample selection; not a learned or unseen-condition upper-bound guarantee.'}
    save(out/'FM_P_to_FM_D_decision_rules.json', rules)
    save(out/'controller_decisions.json', traces)
    csv_write(out/'Controller_supported_configurations.csv', [{ 'D': k, 'checkpoint': v['checkpoint']['checkpoint'], 'path': v['checkpoint']['path'], 'SHA256': v['checkpoint']['sha256'],
        'resolution': v['checkpoint']['resolution'], 'FP32': True, 'CPU_health': v['CPU_health'], 'historical_empirical_eligible': v['eligible_historical_checkpoint'],
        'full_training_feature_provenance': v['full_training_feature_provenance'], 'limitation': v['training_limit']} for k, v in candidates.items()])
    csv_write(out/'Controller_selection_trace.csv', [{'P': r['P'], 'status': r['status'], 'selected_D': r['selected']['D'] if r['selected'] else None, 'reason': r['reason'],
        'trace': json.dumps(r['trace']), 'scope': 'DEVELOPMENT_ONLY'} for r in traces])
    csv_write(out/'Controller_abstention_cases.csv', abstentions, ['P', 'status', 'reason', 'scope'])
    csv_write(out/'Controller_baseline_comparison.csv', baselines)
    csv_write(out/'Controller_group_holdout.csv', holdout)
    positive = len(selections)
    report = {'status': 'WORKING_MEASURED_SELECTOR_WITH_LIMITATIONS', 'D_supported_historical': 4, 'D_full_training_provenance': 0, 'P_measured': 21, 'pairs': 84,
        'interval_feasible_pairs': sum(state(r) == 'PASS' for r in rows), 'P_selected': positive, 'P_abstained': 21-positive,
        'decision_coverage_on_measured_P': positive/21, 'input_information': 'Externally supplied P, not image-based condition recognition',
        'lookup_baseline': 'IN_SAMPLE_LOOKUP_RESULT; agrees by construction, not a new sampling algorithm',
        'group_holdout': {'P_folds': 21, 'disjoint': True, 'coverage': sum(r['selected_D'] is not None for r in holdout)/21,
                          'predictive_accuracy': None, 'status': 'EXACT_LOOKUP_ABSTAINS_ON_UNSEEN_P'},
        'learned_feature_rule': 'INSUFFICIENT_DATA_FOR_GENERALIZABLE_RULE', 'reason': 'Only4 feasible P groups among21 purposeful conditions; shared548 images and prior exposure.',
        'external_fitting_rows': 0, 'full_space_Gap_prevalence': 'NOT_ESTIMABLE'}
    save(out/'Controller_validation_report.json', report)
    return report, selections, baselines


def future_protocol(out):
    sys.path.insert(0, str(HERE.parent/'cv261109'))
    import generator
    import cv2
    import numpy as np
    old = read(ROOT/'datasets/local_runs/vm36_supervisor_reproduction/vm36_supervisor_reproduction_protocol_v2_20261109T153241_baec0138/IMPLEMENTATION_BINDINGS_PROPOSAL.json')
    values = old['proposed_values_from_historical_implementation']
    params = {'Normal': {'gain': 1., 'bias': 0}, 'Motion_k21': {'kernel': 21, **values['Motion_k21']}, 'Haze_dense': {'beta': 1.5, **values['Haze_dense']}}
    combinations = [['Normal'], ['Normal', 'Motion_k21'], ['Normal', 'Haze_dense'], ['Normal', 'Motion_k21', 'Haze_dense']]
    fixture = np.random.default_rng(20261117).integers(0, 256, (96, 128, 3), dtype=np.uint8)
    checks = []
    for leaves in combinations:
        attrs = {f: params[f] for f in leaves}
        a, trace = generator.generate(fixture, leaves, attrs, 'SYNTHETIC_VM44_FIXTURE', 20261117)
        b, second = generator.generate(fixture, leaves, attrs, 'SYNTHETIC_VM44_FIXTURE', 20261117)
        assert np.array_equal(a, b) and trace == second and a.shape == fixture.shape and a.dtype == np.uint8
        checks.append({'P': configuration_id(leaves), 'leaves': leaves, 'pixel_SHA256': __import__('hashlib').sha256(a.tobytes()).hexdigest(),
                       'shape': list(a.shape), 'deterministic': True, 'trace': trace, 'scope': 'SYNTHETIC_CPU_PIXEL_TEST_ONLY'})
    save(out/'future_pixel_fixture_tests.json', {'status': 'PASS', 'cases': checks, 'real_images_rendered': 0, 'bbox_transform': 'Identity; photometric/filter operations only'})
    proposal = {'id': 'VM44_PROPOSED_MOTION_HAZE_2X2_V1', 'status': 'PROPOSED_NOT_MEASURED_NOT_GPU_ADMITTED', 'scientific_readiness': 'NUISANCE_BINDINGS_REQUIRE_SCIENTIFIC_APPROVAL',
        'purpose': 'Measure missing Motion x Haze interaction and its detector dependence, not maximize AP.', 'data': 'Full548 original VisDrone val development members, fixed original GT',
        'D': ['ck01_Infer1280', 'ck05_Infer1280'], 'P': [{'id': configuration_id(x), 'leaves': x} for x in combinations], 'cells': 8, 'images_per_cell': 548, 'image_inferences': 4384,
        'UVL_defined_values': {'Motion_kernel': 21, 'Haze_beta': 1.5}, 'proposed_non_supervisor_bindings': {'motion_angle': 'Uniform[0,180) degrees; independent deterministic per image/feature RNG, not calibrated flight motion', 'airlight': 220.,
            'status': 'HISTORICAL_IMPLEMENTATION_PROPOSAL_ONLY_NOT_SUPERVISOR_APPROVED', 'reason': 'Reuse historical proxy to make implementation explicit; no evidence establishes physical realism or author approval.'},
        'processing_order': ['Atmosphere', 'Illumination', 'Motion blur'], 'haze_model': 'I=J*exp(-1.5)+A*(1-exp(-1.5)); spatially constant depth proxy; no real 3D reconstruction',
        'GT': 'Original coordinates/area and ignore masks unchanged; do not remove low-visibility targets; no RollingShutter',
        'RNG': 'effect_seed(20261117,image_stem,feature), NumPy default_rng; same feature draws and pixels reused across D', 'image_cache': 'One native-pixel PNG per image/P; source SHA and pixel SHA verified',
        'inference': {'resolution': 1280, 'FP32': True, 'batch': 1, 'head': 'dense one-to-many', 'conf': .001, 'NMS_IoU': .7, 'max_det': 3000, 'max_nms': 30000},
        'primary': 'Within-D AP95 DID = both-motion-haze - motion - haze + normal; same image pairing', 'secondary': ['Tiny/Small DID', 'checkpoint DID difference', 'same cached pipeline timing'],
        'statistics': {'image_bootstrap': 200, 'seed': 20261117, 'same_indices_all_cells': True, 'primary_two_detector_tests': 'Holm correction if inferential p-values are introduced; otherwise descriptive pointwise intervals, no familywise significance'},
        'scope': 'DEVELOPMENT_ONLY; historical best-selection exposure; not independent confirmation or architecture causality',
        'annotation_safety': 'CPU_IDENTITY_COORDINATES_VALIDATED; synthetic visibility change is not a geometry warp', 'independent_gate': 'NOT_RESOLVED_BY_THIS_DEVELOPMENT_EXPERIMENT',
        'GPU_authorization': 'NOT_PROVIDED; VM43 permission does not apply', 'thermal': 'Fresh admission required;90C immediate,87C30s,2s polling',
        'training_policy_design_only': {'status': 'FUTURE_PREREGISTRATION_NOT_TRAINED', 'rule': 'On separate development discovery groups, choose a family by predeclared loss and matched contrast support; freeze recipe before training; never use VM43 object errors.',
                                      'probabilities_and_ranges': 'NOT_SELECTED; no new training recipe or model has been created'}}
    save(out/'Next_minimal_experiment_protocol.json', proposal)
    actual = read(V43/'resource_usage.json')
    linear = actual['formal_charged_worker_wall_seconds']*4384/6440
    old_cost = read(V41/'Complete_pair_cost_plan.json')
    plan = {'status': 'ESTIMATE_ONLY_NOT_ADMITTED', 'VM43_actual_worker_seconds': actual['formal_charged_worker_wall_seconds'], 'VM43_measured_images': 6440,
        'new_image_inferences': 4384, 'linear_seconds_NOT_MEASURED': linear, 'uncertainty_factor': 2., 'shutdown_reserve_seconds': 1800,
        'planning_seconds_NOT_GUARANTEED': linear*2+1800, 'VM41_previous_plan': next(p for p in old_cost['plans'] if p['design'] == 'MOTION_HAZE_2x2'),
        'unknown': ['New checkpoint prediction density and CPU Bootstrap cost', 'Haze/motion PNG generation and writing on548 real images', 'New-time device temperature and competing workload'],
        'resource_readiness': 'NOT_ADMITTED; needs measured bounded preflight under a new authorization', 'GPU_authorization': 'NONE',
        'old_budget_is_not_permission': True, 'required_gates': ['Scientific nuisance-binding review', 'Dataset/checkpoint/pixel SHA', 'Complete8-cell budget and unchanged expiry if authorized allocation applies', 'Fresh physical/device thermal gate']}
    save(out/'Next_experiment_cost_estimate.json', plan)
    return proposal, plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--select', type=Path, help='An accepted VM44 rules JSON; selection only, no raw data.')
    parser.add_argument('--leaves', nargs='+')
    parser.add_argument('--strict-training', action='store_true')
    parser.add_argument('--continue-cpu', help='Reuse sealed, completed CPU source/controller stages from an earlier VM44 RunID.')
    args = parser.parse_args()
    if args.select:
        if not args.leaves:
            parser.error('--leaves is required for selection')
        rules = json.loads(args.select.read_text(encoding='utf-8'))
        answer = Controller(rules['models'], rules['development_rows'], rules['candidates'], rules['semantic_support']).select(args.leaves, strict_training=args.strict_training)
        print(json.dumps(answer, ensure_ascii=False, indent=2)); return
    started, process_cpu = time.perf_counter(), time.process_time()
    from main.pipeline import new_run
    out, logger = new_run({'output_root': str(RUNS), 'GPU': 0, 'diagnostic_score': .25, 'diagnostic_IoU': .50}, HERE.name, 'analysis')
    runtime = out/'runtime'; runtime.mkdir()
    os.environ.update(MPLCONFIGDIR=str(runtime/'matplotlib'), YOLO_CONFIG_DIR=str(runtime/'ultralytics'))
    import psutil
    process = psutil.Process(); stop = threading.Event(); memory = {'peak_RSS_bytes': 0}
    def watch():
        while not stop.wait(.25):
            memory['peak_RSS_bytes'] = max(memory['peak_RSS_bytes'], process.memory_info().rss)
    thread = threading.Thread(target=watch, daemon=True); thread.start()
    stages = []
    def timed(name, function, *values):
        wall, cpu, rss = time.perf_counter(), time.process_time(), process.memory_info().rss
        result = function(*values)
        stages.append({'stage': name, 'wall_seconds': time.perf_counter()-wall, 'CPU_process_seconds': time.process_time()-cpu,
                       'RSS_before_bytes': rss, 'RSS_after_bytes': process.memory_info().rss, 'GPU_seconds': 0})
        return result
    before = budget_snapshot()
    sys.path.insert(0, str(HERE))
    save(out/'protocol.json', {'mode': 'CPU_ONLY', 'external_branch': 'DESCRIPTIVE_ONLY_NO_CONTROLLER_FITTING', 'development_branch': 'VM36_ONLY',
        'diagnostic': {'score_min': .25, 'IoU_min': .50, 'matching': 'Class-aware descending score, original prediction index ties, best available IoU then GT ID', 'duplicates': 'No deduplication; unmatched copies are FP',
            'FN_hierarchy': ['Competing match', 'Low-confidence same-class overlap', 'Wrong-class high-score overlap', 'Same-class IoU[.1,.5) near miss', 'No high-score localized candidate'], 'not_COCO_decomposition': True},
        'sample_replay': {'seed': 20261117, 'images_per_cell': 12, 'scalar_independent_matching': True},
        'controller': {'fixed_detector': 'ck05_Infer640', 'primary_1280': 'ck01_Infer1280', 'Tiny_min': .1, 'latency_max_ms': 100, 'ranking': 'Minimum mean latency among interval PASS, then Tiny lower CI, then D ID',
            'heldout': 'Leave whole P out; exact lookup must abstain', 'learned_rule': 'Do not fit if fewer than5 feasible P groups, given21 purposeful shared-image contexts'},
        'scope': {'external': 'LIMITED_EXTERNAL_EXPLORATORY', 'historical_independence': 'UNKNOWN', 'development': 'DEVELOPMENT_ONLY'}, 'GPU_authorized': False})
    source(AUTHOR, role='USER_VM44_INSTRUCTION')
    for f in HERE.glob('*.py'): source(f, role='VM44_SOURCE')
    for f in ('labels.py', 'predictor_stable.py', 'statistics.py'): source(PUBLIC/'code/main'/f, role='UNCHANGED_REFERENCE_IMPLEMENTATION')
    try:
        external_spec, payload = timed('external_source_verification', external_inputs, out)
        logger.info('Accepted external inputs verified. No images decoded and no network forward.')
        for name, digest in [('fm_p.uvl', 'e6eadd633d1c14e0d6bc425f5e54421d592d766c65a6e5f20c61bc34dd76fb5c'), ('fm_d.uvl', '4022d674822b97dbbe3712646b9a0f970af95d175ae358e63582029bc107571a')]: source(UVL/name, digest, 'ORIGINAL_SUPERVISOR_UVL')
        if args.continue_cpu:
            if Path(args.continue_cpu).name != args.continue_cpu: raise ValueError('Expected a VM44 RunID.')
            prior = (RUNS/args.continue_cpu).resolve(strict=True)
            accepted = seal_entries(prior/'SHA256SUMS.txt')
            import shutil
            files = ['FM_P_to_FM_D_decision_rules.json', 'controller_decisions.json', 'Controller_supported_configurations.csv', 'Controller_selection_trace.csv',
                'Controller_abstention_cases.csv', 'Controller_baseline_comparison.csv', 'Controller_group_holdout.csv', 'Controller_validation_report.json',
                'development_evidence_verification.json', 'compiled_original_UVL.json', 'UVL_test_cases.json', 'flamapy.log']
            for name in files:
                verify_source(prior/name, accepted, 'REUSED_COMPLETED_CPU_STAGE'); shutil.copyfile(prior/name, out/name)
            for name in ('selector.py', 'uvl_bridge.py', 'tests_cpu.py', 'diagnostics.py', 'reporting.py'):
                if sha(HERE/name) != sha(prior/'failed_source'/name): raise ValueError('Reused algorithm source differs: '+name)
            rule = read(prior/'FM_P_to_FM_D_decision_rules.json')
            rows, candidates, support, cnf = rule['development_rows'], rule['candidates'], rule['semantic_support'], rule['models']
            accepted_v40 = {r['path']: r['sha256'] for r in read(V40/'output_checksums.json')['files']}
            conditions = csv_read(V40/'VM36_21_measured_conditions.csv', accepted_v40[str(V40/'VM36_21_measured_conditions.csv')])
            original_index = read(V40/'VM36_source_index.json', accepted_v40[str(V40/'VM36_source_index.json')])
            for item in original_index:
                for name in ('configuration.json', 'metrics.json', 'bootstrap.json', 'timing.json', 'verification.json'):
                    source(Path(item['source_folder'])/name, item['SHA256'][name], 'REUSED_STAGE_SOURCE_SHA_RECHECK_ONLY')
            controller_report = read(out/'Controller_validation_report.json')
            selections = [{'P': a['P'], 'D': a['selected']['D']} for a in read(prior/'controller_decisions.json') if a['selected']]
            baselines = csv_read(prior/'Controller_baseline_comparison.csv')
            for r in baselines: r['abstain'] = r['abstain'] == 'True'
            for info in candidates.values(): source(info['checkpoint']['path'], info['checkpoint']['sha256'], 'REUSED_HEALTH_EXACT_WEIGHT_BYTES')
            save(out/'CPU_stage_reuse.json', {'source_RunID': prior.name, 'seal': source(prior/'SHA256SUMS.txt'), 'reused_files': files,
                'source_algorithms_unchanged': True, 'correction': 'Explicit VM44 module loading; no scientific or controller algorithm change.',
                'new_model_health_or_controller_evaluation_repeated': False})
            logger.info('Reused sealed development health, original UVL and controller stages; no repeated experiment.')
        else:
            rows, conditions, candidates, support = timed('development_metadata_and_CPU_checkpoint_health', development_inputs, out)
            logger.info('Development84-pair metadata and4 D settings verified separately.')
            cases = {'fm_p': [c['selected_leaves'].split('|') for c in conditions]+[['Normal', 'Motion_k21'], ['Normal', 'Haze_dense'], ['Normal', 'Motion_k21', 'Haze_dense'], ['LowLight'], ['Normal', 'Shot_high'], ['Normal', 'RollingShutter'], ['Overexposed_strong', 'Haze_dense']],
                     'fm_d': [c['declared_leaves'] for c in candidates.values()]}
            cases['fm_d'].append([f for f in candidates['ck05_Infer1280']['declared_leaves'] if f != 'NMS']+['EndToEnd_NMSFree'])
            save(out/'UVL_test_cases.json', cases)
            request = [str(ROOT/'.venv_flamapy/Scripts/python.exe'), '-B', '-X', 'utf8', str(HERE/'uvl_bridge.py'), '--models', str(UVL), '--output', str(out/'compiled_original_UVL.json'), '--cases', str(out/'UVL_test_cases.json')]
            child = subprocess.run(request, capture_output=True, text=True, encoding='utf-8', timeout=120, env=os.environ.copy())
            (out/'flamapy.log').write_text(child.stdout+'\n'+child.stderr, encoding='utf-8')
            if child.returncode: raise RuntimeError('Original UVL compilation failed; see flamapy.log.')
            cnf = read(out/'compiled_original_UVL.json', role='VM44_COMPILED_ACTUAL_UVL')
            for p in conditions:
                if not legal(cnf['fm_p'], p['selected_leaves'].split('|')): raise ValueError('Measured P fails actual UVL.')
            controller_report, selections, baselines = timed('development_selector_and_group_holdout', controller_analysis, out, cnf, rows, conditions, candidates, support)
        diagnostic = module('vm44_diagnostics', HERE/'diagnostics.py')
        sys.modules['diagnostics'] = diagnostic
        test_module = module('vm44_cpu_tests', HERE/'tests_cpu.py')
        tests = timed('new_VM44_CPU_tests', test_module.execute_tests, out, cnf, rows, candidates, support)
        if tests['failed'] or tests['errors']: raise AssertionError('VM44 CPU tests failed.')
        fn, fp, sizes, classes = timed('streamed_external_diagnostics', diagnostic.analyze, out, external_spec, payload, logger)
        transitions_rows = timed('paired_GT_transitions', diagnostic.transitions, out, external_spec)
        proposal, costs = timed('future_protocol_synthetic_CPU_only', future_protocol, out)
        reporter = module('vm44_reporting', HERE/'reporting.py')
        reports, plots, interpretation = timed('plots_and_Chinese_reports', reporter.deliver, out, external_spec, fn, fp, sizes, classes, transitions_rows, rows, conditions, controller_report, selections, baselines, proposal, costs)
        after = budget_snapshot()
        if before['authorization'] != after['authorization'] or before['charges'] != after['charges']: raise ValueError('GPU ledger changed in CPU-only task.')
        for item in SOURCES.values():
            if sha(item['path']) != item['SHA256']: raise ValueError('Input changed during VM44: '+item['path'])
        stop.set(); thread.join()
        import torch
        if torch.cuda.is_initialized(): raise AssertionError('CPU-only VM44 initialized CUDA.')
        save(out/'source_provenance.json', {'RunID': out.name, 'files': list(SOURCES.values()), 'external_scope': 'LIMITED_EXTERNAL_EXPLORATORY', 'historical_independence': 'UNKNOWN',
            'controller_training_sources': 'VM36 development metadata only', 'testdev_for_fitting': False, 'raw_predictions_reused': True, 'images_regenerated': 0})
        save(out/'resource_usage.json', {'CPU_stage_wall_seconds': time.perf_counter()-started, 'CPU_process_seconds': time.process_time()-process_cpu,
            'FlamaPy_child_CPU_seconds': 0 if args.continue_cpu else cnf['CPU_process_seconds'], 'reused_FlamaPy_CPU_seconds_REFERENCE': cnf['CPU_process_seconds'] if args.continue_cpu else None,
            'prior_CPU_process_seconds': 'NOT_MEASURED_IN_FAILED_IMPORT_RUNS; excluded from current-run measurement' if args.continue_cpu else None,
            **memory, 'CUDA_initialized': False, 'GPU_seconds': 0, 'new_GPU_budget_charge_seconds': 0, 'budget_unchanged': True,
            'original_budget_reporting_only': after, 'independent_process': True, 'raw_image_decode_count': 0})
        csv_write(out/'CPU_stage_timing.csv', stages)
        measured = json.loads((out/'resource_usage.json').read_text(encoding='utf-8'))
        for path in reports:
            with path.open('a', encoding='utf-8') as f:
                f.write(f"\n\n本Run实测CPU分析墙钟{measured['CPU_stage_wall_seconds']:.3f}秒，主进程CPU时间{measured['CPU_process_seconds']:.3f}秒，新FlamaPy子进程CPU{measured['FlamaPy_child_CPU_seconds']:.3f}秒，RSS采样峰值{memory['peak_RSS_bytes']/1024**3:.3f}GiB；GPU=0。新VM44测试{tests['passed']}/{tests['tests']}通过，独立匹配对照48图通过。各分析阶段墙钟及RAM快照见`CPU_stage_timing.csv`；不把仅供报告的GPU余额当新授权。\n")
                if args.continue_cpu:
                    f.write(f"\n复用已封存CPU阶段：`{args.continue_cpu}`。此前两次CPU工程启动分别遇到旧字段版本与同名模块导入问题，均保留失败日志/源码/RunID；未执行GPU，也未生成外部诊断。已完成来源、健康、UVL和查表阶段本次不重做，仅复核SHA。旧失败进程CPU时间未捕获，不编入本Run实测时间。\n")
        save(out/'final_acceptance.json', {'status': 'VM44_COMPLETE_WITH_LIMITATIONS', 'RunID': out.name, 'tests': tests, 'external_cells_diagnosed': 4, 'external_images_per_cell': 1610,
            'controller': controller_report, 'interpretation': interpretation, 'GPU_executed': False, 'full_RQ2_RQ4_solved': False,
            'future_GPU': 'NOT_AUTHORIZED_NOT_STARTED', 'reports': [str(p) for p in reports], 'plots': [str(p) for p in plots]})
        record = json.loads((out/'run.json').read_text()); record.update(status='COMPLETE_WITH_LIMITATIONS', finished_UTC=datetime.now(timezone.utc).isoformat())
        (out/'run.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
        (out/'diary.md').write_text('VM44: descriptive external error diagnosis; separate development-only measured-evidence selection. No new inference, training or policy tuning. Frozen diagnostic score=.25/IoU=.50. Original UVL and source hashes retained. See final_acceptance.json for limitations and reports.\n', encoding='utf-8')
        logger.info('VM44 complete with limitations; CPU wall %.3fs; GPU0.', time.perf_counter()-started)
        from main.visualization import build_run_index
        build_run_index(out.parent, HERE.name)
        files = [p for p in out.rglob('*') if p.is_file() and 'runtime' not in p.relative_to(out).parts]+reports+list(HERE.glob('*.py'))
        with (out/'SHA256SUMS.txt').open('x', encoding='utf-8') as f:
            for p in sorted(files): f.write(sha(p)+'  '+str(p)+'\n')
        print('VM44_COMPLETED', out, flush=True)
    except BaseException:
        stop.set(); thread.join(timeout=2)
        (out/'failure_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        logger.exception('VM44 blocked; original evidence untouched. No GPU retry.'); raise


if __name__ == '__main__':
    main()
