"""Run isolated local CPU acceptance of the versioned training package."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

POST = Path(__file__).resolve().parent
sys.path.insert(0, str(POST.parent))
from study import *
import portable


def execute(bundle, data_root):
    out = create('portable_v3_acceptance_cpu')
    runtime(out, cuda=False)
    began = time.perf_counter()
    entry = bundle / 'main/experiments/vm37_yolo26_rq4/post_cost'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='', OMP_NUM_THREADS='2')
    ledger_before = sha256(RUNS / 'compute_allocation.json')
    manifest_sha = sha256(bundle / 'bundle_manifest.json')
    commands = []
    def run(name, tail, limit=600, expected=0):
        command = [sys.executable, '-B', '-X', 'utf8', *map(str, tail)]
        available = max(1, 3600 - (time.perf_counter() - began))
        with (out / (name + '.log')).open('x', encoding='utf-8') as stream:
            process = subprocess.Popen(command, cwd=bundle, env=env, stdout=stream, stderr=subprocess.STDOUT)
            try:
                code = process.wait(timeout=min(limit, available))
            except subprocess.TimeoutExpired:
                from safety import stop_owned
                stop_owned(process)
                raise TimeoutError('CPU validation deadline reached.')
        value = {'stage': name, 'command': command, 'returncode': code, 'expected_returncode': expected,
                 'log_SHA256': sha256(out / (name + '.log'))}
        commands.append(value)
        if (expected == 0 and code != 0) or (expected != 0 and code == 0):
            raise RuntimeError('CPU acceptance stage failed: ' + name)
        print('CPU_STAGE', name, 'PASS', flush=True)
        return (out / (name + '.log')).read_text(encoding='utf-8').strip().splitlines()[-1]
    try:
        portable.verify_bundle(bundle)
        for kind in ('benchmark26', 'benchmark8'):
            protocol = load(bundle / (kind + '.json'))
            if (protocol['full_matrix_training_arms'] != len(portable.matrix(protocol))
                    or protocol['planned_counters_per_arm'] != expected_steps(6471, 100, stop_epochs=1)
                    or protocol['evaluation_configurations_per_matched_pair'] != 0):
                raise ValueError('Benchmark metadata must describe its two first-epoch cost arms.')
        run('unit', [entry / 'test_v3.py', '--output', out / 'unit'], 180)
        run('comparison', [entry / 'compare_v3.py', '--bundle', bundle, '--data-root', data_root,
                           '--output-root', out / 'comparisons'], 1800)
        receipts = list((out / 'comparisons').glob('*/CPU_receipt.json'))
        if len(receipts) != 1:
            raise ValueError('Expected exactly one completed CPU receipt.')
        comparison = receipts[0].parent
        receipt = load(receipts[0])
        if receipt['status'] != 'CPU_EXACT_PARITY_PASS' or receipt['golden_pixel_checks'] != 150 or receipt['healthy_initializations'] != 6:
            raise ValueError('The integrated comparison did not pass.')
        plans = []
        for kind, pipeline in [('formal100', 'optimized'), ('pilot20', 'optimized'), ('benchmark26', 'optimized'),
                               ('benchmark8', 'optimized'), ('benchmark26', 'baseline')]:
            path = Path(run('dry_' + kind + '_' + pipeline, [entry / 'portable_v3.py', '--bundle', bundle,
                '--data-root', data_root, '--protocol', kind, '--input-pipeline', pipeline,
                '--output-root', out / 'dry_runs']))
            dry = load(path / 'dry_run.json')
            if dry['GPU_seconds'] != 0 or list(path.glob('*/WORKER_STARTED.json')):
                raise ValueError('A CPU dry-run entered a worker.')
            summary = load(path / 'plans.json')
            rows = []
            for protocol_path in sorted(path.glob('*/protocol.json')):
                p = load(protocol_path)
                rows.append({'architecture': p['architecture'], 'seed': p['args']['seed'], 'arm': p['arm'],
                    'initialization_SHA256': p['initialization']['sha256'], 'epochs': p['args']['epochs'],
                    'execution_epochs': p['execution_epochs'], 'workers': p['args']['workers'],
                    'batch': p['args']['batch'], 'AMP': p['args']['amp'], 'input_pipeline': p['input_pipeline'],
                    'train_images': len(p['train']), 'development_images': len(p['development']),
                    'batches': p['expected']['batches'], 'updates': p['expected']['updates']})
            for row in rows:
                if row['workers'] != 0 or row['batch'] != 2 or row['AMP'] or row['train_images'] != 6471 or row['development_images'] != 548:
                    raise ValueError('Frozen training contract changed.')
                partner = next(other for other in rows if other['architecture'] == row['architecture'] and other['seed'] == row['seed'] and other['arm'] != row['arm'])
                if row['initialization_SHA256'] != partner['initialization_SHA256']:
                    raise ValueError('Matched initialization differs.')
            plans.append({'kind': kind, 'pipeline': pipeline, 'RunID': path.name, 'plans': rows, 'summary': summary})
        for kind in ('formal100', 'pilot20', 'benchmark26', 'benchmark8'):
            run('refused_' + kind, [entry / 'portable_v3.py', '--bundle', bundle, '--data-root', data_root,
                '--protocol', kind, '--authorization', bundle / ('authorization_v3_' + kind + '.example.json'),
                '--cpu-receipt', receipts[0], '--output-root', out / 'refused'], 180, expected=1)
            if 'New protocol-specific author approval is required' not in (out / ('refused_' + kind + '.log')).read_text():
                raise ValueError('Default refusal failed for an unexpected reason.')
        run('legacy_v2_dry', [entry / 'portable.py', '--bundle', bundle, '--data-root', data_root, '--protocol', 'pilot20'])
        portable.verify_bundle(bundle)
        portable.resolve_data(bundle, data_root)
        if sha256(RUNS / 'compute_allocation.json') != ledger_before:
            raise ValueError('The historical local GPU allocation was modified.')
        if sha256(bundle / 'bundle_manifest.json') != manifest_sha:
            raise ValueError('The package changed during CPU validation.')
        save(out / 'plan_verification.json', plans, out)
        save(out / 'verification.json', {'status': 'CPU_VERIFIED_GPU_ORCHESTRATOR_NOT_RUN',
             'bundle': str(bundle), 'bundle_manifest_SHA256': manifest_sha, 'unit_tests': load(out / 'unit/test_results.json'),
             'comparison_RunID': comparison.name, 'comparison_path': str(comparison), 'CPU_receipt': source(receipts[0]),
             'default_refusals': 4, 'v3_dry_runs': 5, 'legacy_v2_entrypoint': 'PASS',
             'historical_GPU_budget_unchanged': True, 'all_dataset_sources_rehashed_after': True,
             'GPU_seconds': 0, 'CPU_wall_seconds': time.perf_counter() - began, 'commands': commands,
             'Linux_tested': False, 'server_full_epoch_benchmark': 'NOT_RUN', 'formal100_training': 'NOT_RUN'}, out)
        write_atomic(out / 'diary.md', '# Optimized portable acceptance\n\nTested isolated package modules, complete '
            'matched plans, six native10 states,150 golden pixel checks and deterministic input parity. '
            'Default GPU authorization was refused. Rehashed all source data and the unchanged historical ledger. GPU=0.\n', out)
        save(out / 'run.json', {'RunID': out.name, 'ExperimentID': EXPERIMENT,
            'status': 'CPU_VERIFIED_GPU_ORCHESTRATOR_NOT_RUN', 'GPU_seconds': 0}, out, replace=True)
    except BaseException:
        save(out / 'failure.json', {'traceback': traceback.format_exc(), 'automatic_retry': False, 'commands': commands}, out)
        save(out / 'run.json', {'RunID': out.name, 'ExperimentID': EXPERIMENT,
            'status': 'CPU_VERIFICATION_FAILED', 'GPU_seconds': 0}, out, replace=True)
        seal(out)
        raise
    seal(out)
    print(out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    execute(args.bundle.resolve(strict=True), args.data_root.resolve(strict=True))
