"""Seal post-cost evidence and retain blocked training outcomes explicitly."""
import argparse
import csv
from pathlib import Path
import shutil
import sys
import time

POST = Path(__file__).resolve().parent
sys.path.insert(0, str(POST.parent))
from study import *
from safety import available_seconds

ADMISSION = 'vm37_yolo26_rq4_postcost_admission_cpu_20261112T130427_8e926bc0'
FROZEN = 'vm37_yolo26_rq4_postcost_protocol_cpu_20261112T132257_50b16e94'
GAPS = 'vm37_yolo26_rq4_historical_gap_evidence_cpu_20261112T132501_6d741736'
CONTROL = 'vm37_yolo26_rq4_full_epoch_yolo26s_control_20261111T232439_301fb25d'


def execute(loader_id, bundle_id, verification_id):
    out = create('postcost_final_cpu')
    runtime(out, cuda=False)
    started = time.perf_counter()
    loader = run_path(loader_id)
    bundle = run_path(bundle_id)
    verification = run_path(verification_id)
    for path, required in ((loader, ('loader_cost.json',)), (bundle, ('bundle_result.json',)),
                           (verification, ('verification.json', 'health_and_pixels.json')),
                           (run_path(GAPS), ('historical_deficits.csv', 'historical_quality_states.csv'))):
        verify_hashes(path, required)
    cost = load(loader / 'loader_cost.json')
    tested = load(verification / 'verification.json')
    if tested['status'] != 'CPU_VERIFIED_GPU_NOT_VALIDATED':
        raise ValueError('Portable CPU verification did not pass.')
    allocation_path = RUNS / 'compute_allocation.json'
    allocation = load(allocation_path)
    allocation_before = load(run_path(ADMISSION) / 'resource.json')['ledger_SHA256']
    if sha256(allocation_path) != allocation_before:
        raise ValueError('Allocation changed; recompute admission from actual state.')
    sources = []
    for row in load(run_path(FROZEN) / 'private_asset_sources.json'):
        actual = sha256(Path(row['path']))
        if actual != row['sha256']:
            raise ValueError('Original private asset changed.')
        sources.append({'path': row['path'], 'sha256': actual})
    accepted = load(PREP / 'FROZEN_DEVELOPMENT_EVALUATION.json')['conditions']
    for row in accepted:
        for name in ('metrics.json', 'bootstrap.json', 'configuration.json'):
            path = Path(row['source_folder']) / name
            if sha256(path) != row['source_SHA256'][name]:
                raise ValueError('Accepted scientific evidence changed.')
            sources.append({'path': str(path), 'sha256': row['source_SHA256'][name]})
    with (run_path(GAPS) / 'historical_deficits.csv').open(encoding='utf-8-sig') as stream:
        deficits = list(csv.DictReader(stream))
    with (run_path(GAPS) / 'historical_quality_states.csv').open(encoding='utf-8-sig') as stream:
        states = list(csv.DictReader(stream))
    selected = [row for row in deficits if row['size'] == 'all' and row['metric'] == 'map50_95' and
        set(row['leaves'].split('|')) in ({'Normal'}, {'Normal', 'JPEG_q25'}, {'Normal', 'JPEG_q75'}, {'Normal', 'Gauss_s15'})]
    with (out / 'historical_pilot_contexts.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=selected[0], lineterminator='\n')
        writer.writeheader()
        writer.writerows(selected)
    control = load(run_path(CONTROL) / 'training_acceptance.json')
    checkpoint = Path(control['checkpoint']['path'])
    checkpoint_bytes = checkpoint.stat().st_size
    if sha256(checkpoint) != control['checkpoint']['sha256']:
        raise ValueError('Accepted cost checkpoint changed.')
    now = utc()
    component = {
        'CONTROL_training_setup_validation_save_seconds_measured': control['training_setup_epoch_internal_val_save_seconds'],
        'CONTROL_loader_seconds_measured': control['loader_seconds'],
        'ALL_FIVE_CPU_loader': cost,
        'ALL_FIVE_GPU_full_epoch': 'NOT_MEASURED',
        'close_mosaic_epoch_for_pilot': 10,
        'closed_mosaic_ALL_FIVE_loader': 'NOT_MEASURED',
        'pilot_both_arms_complete_runtime': 'NOT_VERIFIED',
        'CONTROL20_linear_scenario_hours': control['training_setup_epoch_internal_val_save_seconds'] * 20 / 3600,
        'CONTROL100_linear_scenario_hours': control['training_setup_epoch_internal_val_save_seconds'] * 100 / 3600,
        'CONTROL100_linear_plus25percent_scenario_hours': control['training_setup_epoch_internal_val_save_seconds'] * 100 * 1.25 / 3600,
        'measured_checkpoint_bytes': checkpoint_bytes,
        'pilot40_snapshot_size_projection_bytes': checkpoint_bytes * 40,
        'formal1200_snapshot_size_projection_bytes': checkpoint_bytes * 1200,
        'interpretation': 'Projections are scenarios, not observed20/100-epoch times or reserved storage. Warmup/update cadence, close_mosaic, effects, model family and validation response affect cost.'}
    status = {
        'UTC': now.isoformat(), 'local_time': now.astimezone().isoformat(),
        'status': 'COMPLETE_CPU_WITH_TRAINING_BLOCKED',
        'resource_admission': 'NO_GO_UNVERIFIED_COMPLETE_MATCHED_COST_AND_THERMAL_READINESS',
        'formal100': 'RESOURCE_BLOCKED', 'pilot20': 'PILOT_RESOURCE_BLOCKED',
        'pilot20_reason': ['Complete matched training/evaluation/margin cost is not measured or admitted.',
                          'Physical cooling/adapter reliability and safe sustained GPU load remain unverified.',
                          'Historical sampled92C would violate the newly frozen90C hard-stop rule.'],
        'absolute_deadline_UTC': allocation['absolute_deadline_UTC'],
        'charged_worker_seconds': sum(row['charged_seconds'] for row in allocation['completed_workers']),
        'available_seconds': available_seconds(allocation, now), 'allocation_reset': False, 'allocation_unchanged': True,
        'disk_free_bytes': shutil.disk_usage(ROOT).free,
        'new_GPU_training': 0, 'new_GPU_predictions': 0, 'matched_pilot_pairs': 0, 'formal100_completed_arms': 0,
        'ALL_FIVE_minus_CONTROL_AP': 'NOT_EVALUATED', 'degradation_loss_change': 'NOT_EVALUATED',
        'augmentation_benefit': 'NOT_EVALUATED', 'scope': 'DEVELOPMENT_ONLY',
        'broad_policy': 'VM37_ALL_FIVE_V1 unchanged; independent probability0.5 gates; not proven Gap-guided.',
        'portable_status': tested['status'], 'portable_bundle': load(bundle / 'bundle_result.json'),
        'CPU_unit_tests': tested['unit_tests'], 'initialization_health_checks': len(load(verification / 'health_and_pixels.json')['initializations']),
        'evaluation_pixel_checks': len(load(verification / 'health_and_pixels.json')['pixel_checks']),
        'historical_deficit_rows': len(deficits), 'historical_quality_state_rows': len(states),
        'historical_reference_pairs': len(accepted), 'historical_bootstrap_samples': 200,
        'historical_intervals': 'Paired saved AP draws; pointwise image uncertainty only; no new COCO evaluation or training-seed inference.',
        'next_action': 'Confirm cooling and a safe bounded ALL_FIVE cost protocol, or admit the complete original study on a suitable GPU server with an explicit new resource allocation. Do not start CONTROL alone.'}
    save(out / 'final_status.json', status, out)
    save(out / 'cost_components.json', component, out)
    save(out / 'protected_inputs.json', {'status': 'PASS', 'files': sources,
         'raw_images': 'Loader post-hashes and two portable dry-runs checked full6471/548 memberships.',
         'large_archives': 'NOT_RECOPIED_NOT_REHASHED; no assertion of a full145GiB archive rehash.'}, out)
    save(out / 'inputs.json', {name: source(run_path(identity) / 'SHA256SUMS.txt') for name, identity in
          (('admission', ADMISSION), ('protocol', FROZEN), ('gaps', GAPS), ('loader', loader_id),
           ('bundle', bundle_id), ('verification', verification_id))}, out)
    write_atomic(out / 'diary.md', '# Post-cost decision\n\nCompleted safe CPU verification and local packaging. Retained the formal100-epoch matrix and the separate20-epoch paired pilot. Neither matched study is admitted. No new GPU task or budget reset; no augmentation benefit is inferred.\n', out)
    save(out / 'run.json', {'RunID': out.name, 'ExperimentID': EXPERIMENT, 'mode': 'POSTCOST_FINAL_CPU',
        'status': status['status'], 'GPU_seconds': 0, 'CPU_wall_seconds': time.perf_counter() - started}, out, replace=True)
    seal(out)
    print(out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loader', required=True)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--verification', required=True)
    args = parser.parse_args()
    execute(args.loader, args.bundle, args.verification)
