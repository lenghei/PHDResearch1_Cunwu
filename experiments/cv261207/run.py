"""Audit and run frozen CV261207 pairs under a local wall-clock budget."""
from pathlib import Path
import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

from common import (RUNS, SCRIPT, FMP, FMD, VAL, TEST, BASELINE, CHECKPOINT, PUBLIC, ROOT,
                    load, source, save, seal, sha256, resolve_run, new_run, verify_hashes, file_id)
from audit import input_audit, split_audit, verify_uvls
from evaluation import Device, pair, runtime, verify, PROTOCOL
from gap_analysis import static_audit
from sampling import models, freeze, detector_design, check_generator, uniform, SEED

BASELINE_SHA = 'e564c819603bf36ac6d93e3acb0359e7d5a75503ce78458209e9a28113280667'


def preflight(args):
    out = new_run('audit')
    print('[CV261207] Audit evidence:', out, flush=True)
    try:
        save(out / 'input_audit.json', input_audit(args))
        fmp, fmd, prior = models()
        configs = prior.enumerate_configs(fmp)
        save(out / 'model_validation.json', verify_uvls(configs, fmp, fmd, SEED))
        save(out / 'generator_validation.json', check_generator(fmp, configs))
        print('[CV261207] UVL and deterministic generator checks passed.', flush=True)
        checkpoint = load(BASELINE / 'training/checkpoint.json')
        if checkpoint['best_checkpoint_SHA256'] != BASELINE_SHA or sha256(CHECKPOINT) != BASELINE_SHA:
            raise ValueError('The CV261102 weight differs from the accepted run.')
        if load(BASELINE / 'manifest.json')['status'] != 'PASS':
            raise ValueError('CV261102 has no accepted baseline.')
        freeze(out, configs, fmp, prior, detector_design(fmd, BASELINE_SHA))
        runtime(out)
        save(out / 'static_gap.json', static_audit(fmp, configs))
        split = split_audit(out, args.history)
        save(out / 'dataset_gate.json', split)
        save(out / 'blockers.json', {
            'formal_evaluation': split['reason'],
            'rolling_shutter': 'Geometry-preserving harness excludes this feature pending GT and ignore warp validation.',
            'detector_space': {'SAHI': 'Package not available in the inspected environment.',
                               'TTA': 'Not verified under the fixed native-ten NMS protocol.',
                               'INT8': 'No calibrated runtime artifact.',
                               'LEAF_YOLO': 'No locally bound and verified compatible checkpoint.',
                               'COCO_ZeroShot': 'COCO 80 and VisDrone native10 label spaces differ.',
                               'EndToEnd_NMSFree': 'Separate postprocessing protocol required.'}})
        save(out / 'run.json', {'RunID': out.name, 'ExperimentID': 'cv261207', 'status': 'PASS_CPU_AUDIT',
                                'formal_evaluation': 'DATASET_BLOCKED'})
        seal(out)
        print('[CV261207] Preflight complete | formal evaluation DATASET_BLOCKED', flush=True)
    except Exception:
        save(out / 'failure.json', {'status': 'PARTIAL', 'traceback': traceback.format_exc()})
        raise


def audit_plan(name):
    out = resolve_run(name)
    verify_hashes(out, ('run.json', 'selection_manifest.json', 'dataset_gate.json'))
    if load(out / 'run.json')['status'] != 'PASS_CPU_AUDIT':
        raise ValueError('Preflight did not pass.')
    manifest = load(out / 'selection_manifest.json')
    if (manifest['FM_P_sha256'] != sha256(FMP) or manifest['FM_D_sha256'] != sha256(FMD)
            or manifest['pairs'][:1] == []):
        raise ValueError('The audited selection no longer matches the installed models.')
    return out, manifest


def readiness(audit, approval, specs):
    gate = load(audit / 'dataset_gate.json')
    if gate['archive_status'] != 'PASS':
        raise ValueError('DATASET_BLOCKED: original archive admission failed.')
    if approval is None:
        raise ValueError('DATASET_BLOCKED: historical isolation review required.')
    decision = load(approval)
    if (decision.get('audit_RunID') != audit.name or decision.get('status') != 'AUTHOR_REVIEWED_ADMIT'
            or decision.get('test_dev_sha256') != gate['archives']['test-dev']['sha256']
            or decision.get('dataset_gate_sha256') != sha256(audit / 'dataset_gate.json')
            or decision.get('selection_sha256') != sha256(audit / 'selection_manifest.json')
            or decision.get('protocol') != PROTOCOL):
        raise ValueError('DATASET_BLOCKED: missing signed-off evidence or frozen identity mismatch.')
    from admission import verify_review
    verify_review(decision, {s['detector']['checkpoint_sha256'] for s in specs}, ROOT)
    if sha256(TEST) != decision['test_dev_sha256']:
        raise ValueError('Test-dev ZIP changed after preflight.')


def compare_repeats(left, right):
    import gzip
    first, second = resolve_run(left), resolve_run(right)
    for path in (first, second):
        verify(path)
    a, b = [load(path / 'configuration.json') for path in (first, second)]
    if first == second or a != b:
        raise ValueError('Repeat runs must have different RunIDs and identical configurations.')
    comparable = ['metrics.json', 'ground_truth.json', 'image_manifest.jsonl']
    if a['split'] == TEST.stem:
        comparable.append('bootstrap.json')
    for name in comparable:
        if sha256(first / name) != sha256(second / name):
            raise ValueError(f'Repeat evidence differs: {name}')
    with gzip.open(first / 'predictions.jsonl.gz', 'rb') as stream_a, gzip.open(second / 'predictions.jsonl.gz', 'rb') as stream_b:
        while True:
            block = stream_a.read(1024 * 1024)
            if block != stream_b.read(1024 * 1024):
                raise ValueError('Repeat raw prediction content differs.')
            if not block:
                break
    return {'status': 'PASS', 'runs': [left, right], 'pair_id': a['pair_id'],
            'split': a['split'], 'images': a['images'],
            'metrics_sha256': sha256(first / 'metrics.json'),
            'image_manifest_sha256': sha256(first / 'image_manifest.jsonl'),
            'raw_predictions_identical': True}


def batch_admission(args):
    if len(args.run) != 2:
        raise ValueError('Batch requires two verified full qualification runs via --run.')
    gate = compare_repeats(*args.run)
    if gate['split'] != TEST.stem or gate['images'] != 1610:
        raise ValueError('Batch requires a reproduced 1,610-image test-dev pair.')
    for name in args.run:
        path = resolve_run(name)
        if (load(path / 'run.json')['mode'] != 'FORMAL'
                or load(path / 'configuration.json')['dataset_sha256'] != sha256(TEST)):
            raise ValueError('Formal repeat input changed.')
        verify_hashes(path, ('bootstrap.json',))
    paths = [resolve_run(name) for name in args.run]
    measured_seconds = [load(p/'timing.json')['pair_wall_seconds'] for p in paths]
    sizes = [sum(f.stat().st_size for f in p.iterdir() if f.is_file()) for p in paths]
    if any(not 0 < x <= 28800 for x in measured_seconds):
        raise ValueError('Invalid qualification timings.')
    return {'qualification_runs':args.run, 'pair_wall_seconds':measured_seconds,
            'pair_evidence_bytes':sizes, 'seconds_reserved_per_pair':1.25*max(measured_seconds),
            'bytes_reserved_per_pair':2*max(sizes), 'scope':'Measured pair plus planning margin; other pairs may cost more.'}


def processed_pairs(selection):
    from common import fingerprint
    hashes = fingerprint()
    members = {p['pair_id']: p for p in selection['pairs']}
    archive_sha = sha256(TEST)
    completed = set()
    for path in RUNS.glob('cv261207_formal_*'):
        if not (path / 'SHA256SUMS.txt').is_file():
            continue
        cfg = load(path / 'configuration.json')
        if cfg['pair_id'] not in members or cfg['dataset_sha256'] != archive_sha:
            continue
        run = verify(path)
        spec = members[cfg['pair_id']]
        if (run['mode'] == 'FORMAL' and cfg['code_sha256'] == hashes and cfg['images'] == 1610
                and cfg['degradation'] == spec['degradation'] and cfg['detector'] == spec['detector']
                and cfg['render_seed'] == selection['render_seed']):
            verify_hashes(path, ('bootstrap.json',))
            completed.add(cfg['pair_id'])
    return completed


def choose(args, manifest, formal):
    specs = manifest['pairs']
    if formal:
        if args.mode == 'batch':
            return specs
        if args.pair_id is None:
            raise ValueError('Full mode requires a frozen --pair-id.')
        chosen = [p for p in specs if p['pair_id'] == args.pair_id]
        if len(chosen) != 1:
            raise ValueError('Pair not in the frozen manifest.')
        return chosen
    detectors = [p['detector'] for p in specs[:5]]
    leaves = sorted(('Normal', 'Motion_k7', 'Gauss_s5'))
    if args.mode == 'demo':
        fmp, _, prior = models()
        eligible = [c for c in prior.enumerate_configs(fmp) if 'RollingShutter' not in c]
        leaves = list(uniform(eligible, 1, args.seed)[0])
        detectors = detectors[:1]
    p = {'degradation_id': file_id(leaves, 'p_'), 'leaves': leaves}
    if args.mode != 'smoke-grid':
        detectors = detectors[:1]
    return [{'pair_id': p['degradation_id'] + '__' + d['detector_id'],
             'degradation': p, 'detector': d} for d in detectors]


def worker(args):
    started = time.perf_counter()
    cost = None
    audit, selection = audit_plan(args.audit)
    formal = args.mode in ('full', 'batch')
    specs = choose(args, selection, formal)
    if formal:
        readiness(audit, args.approval, specs)
    if args.mode == 'batch':
        cost = batch_admission(args)
        completed = processed_pairs(selection)
        specs = [spec for spec in specs if spec['pair_id'] not in completed]
    if not specs:
        return
    index = list(range(1610)) if formal else [0] if args.mode == 'demo' else [0, 50, 110, 165, 220, 295, 380, 500]
    archive = TEST if formal else VAL
    if sha256(archive) != load(audit / 'dataset_gate.json')['archives']['test-dev' if formal else 'val']['sha256']:
        raise ValueError('Input archive SHA changed after preflight.')
    out = None
    device = None
    for spec in specs:
        if cost and args.budget_seconds-(time.perf_counter()-started) < cost['seconds_reserved_per_pair']:
            print('[CV261207] Budget reserve reached; remaining pairs stay queued.', flush=True)
            return 2
        reserve = cost['bytes_reserved_per_pair'] if cost else 0
        if shutil.disk_usage(RUNS).free < 8 * 1024**3 + reserve:
            raise RuntimeError('Less than 8 GiB free; pair not started.')
        out = new_run('formal' if formal else 'smoke')
        try:
            if device is None:
                runtime(out)
                device = Device(BASELINE_SHA)
            save(out / 'source_audit.json', {'RunID': audit.name, 'selection_sha256': sha256(audit / 'selection_manifest.json'),
                                             'qualification': source(args.approval) if formal else None})
            pair(out, spec, archive, index, device, formal)
        except Exception:
            save(out / 'failure.json', {'status': 'FAILED', 'traceback': traceback.format_exc()})
            raise


def supervise(args):
    if args.budget_seconds is None or not 1 <= args.budget_seconds <= (28800 if args.mode in ('full', 'batch') else 600):
        raise ValueError('Supply an explicit bounded --budget-seconds.')
    audit, selection = audit_plan(args.audit)
    if args.mode in ('full', 'batch'):
        readiness(audit, args.approval, choose(args, selection, True))
    cost = batch_admission(args) if args.mode == 'batch' else None
    out = new_run('execution')
    lock = RUNS / '.gpu.lock'
    token = lock.open('x', encoding='utf-8')
    status = 'FAILED'
    start = time.perf_counter()
    try:
        with token:
            token.write(out.name + '\n')
        if cost:
            save(out / 'qualification_cost.json', cost)
        active = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                text=True, capture_output=True, check=True, timeout=10)
        if max(map(int, active.stdout.split())) > 20:
            raise RuntimeError('GPU utilization exceeds the admission limit.')
        with (out / 'execution.log').open('x', encoding='utf-8') as log:
            env = dict(os.environ, CV261207_WORKER=out.name, PYTHONDONTWRITEBYTECODE='1')
            remaining = args.budget_seconds - (time.perf_counter()-start)
            if remaining <= 0:
                raise TimeoutError('Budget elapsed during preflight.')
            result = subprocess.run([sys.executable, '-B', str(SCRIPT), *sys.argv[1:], '--worker'],
                                    timeout=remaining, stdout=log, stderr=subprocess.STDOUT, env=env)
        status = {0:'PASS', 2:'PARTIAL'}.get(result.returncode, 'FAILED')
        if result.returncode not in (0, 2):
            raise RuntimeError(f'GPU worker failed; see {out / "execution.log"}')
    except subprocess.TimeoutExpired:
        status = 'PARTIAL'
        raise
    finally:
        lock.unlink()
        save(out / 'run.json', {'RunID': out.name, 'ExperimentID': 'cv261207', 'status': status,
                                'mode': args.mode, 'budget_seconds': args.budget_seconds,
                                'worker_and_preflight_wall_seconds': time.perf_counter()-start})
        seal(out)
        print(f'[CV261207] {status}: {out}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', default='preflight',
                        choices=('preflight', 'admission', 'publish-admission', 'smoke', 'smoke-grid', 'demo', 'full', 'batch', 'verify',
                                 'regenerate', 'repeat-check', 'figures', 'publish', 'learn'))
    parser.add_argument('--source', type=Path, action='append', default=[])
    parser.add_argument('--history', type=Path, action='append', default=[])
    parser.add_argument('--overlay', type=Path)
    parser.add_argument('--audit')
    parser.add_argument('--run', action='append', default=[])
    parser.add_argument('--pair-id')
    parser.add_argument('--approval', type=Path)
    parser.add_argument('--budget-seconds', type=int)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.mode == 'preflight':
        preflight(args)
    elif args.mode == 'admission':
        from admission import run
        run(audit_plan(args.audit)[0])
    elif args.mode == 'publish-admission':
        from outputs import publish_admission
        if len(args.run) != 1:
            parser.error('publish-admission requires one --run.')
        publish_admission(resolve_run(args.run[0]))
    elif args.mode in ('verify', 'regenerate'):
        for name in args.run:
            print(json.dumps(verify(resolve_run(name), regenerate=args.mode == 'regenerate')))
        if not args.run:
            parser.error('--run is required.')
    elif args.mode == 'repeat-check':
        if len(args.run) != 2:
            parser.error('repeat-check requires two --run arguments.')
        print(json.dumps(compare_repeats(*args.run)))
    elif args.mode in ('figures', 'publish', 'learn'):
        from outputs import execute
        execute(args)
    elif args.worker:
        if (RUNS / '.gpu.lock').read_text(encoding='utf-8').strip() != os.environ.get('CV261207_WORKER'):
            raise ValueError('Use the bounded worker supervisor.')
        sys.exit(worker(args))
    else:
        supervise(args)


if __name__ == '__main__':
    main()
