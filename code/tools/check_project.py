"""CPU checks and saved-prediction metric regression."""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]
LOCAL_ROOT = REPOSITORY.parent
sys.path.insert(0, str(REPOSITORY / 'code'))
from main.config import load_config, sha256, save


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=REPOSITORY / 'experiments/vm1_flamapy_yolo26s_validation/config.local.yaml')
    parser.add_argument('--run-dir', type=Path)
    parser.add_argument('--flamapy-python', '--uvl-python', dest='flamapy_python', type=Path)
    parser.add_argument('--require-flamapy', '--require-uvl', dest='require_uvl', action='store_true')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--timeout', type=int, default=180)
    args = parser.parse_args(argv)
    if args.timeout < 1:
        parser.error('--timeout must be positive')
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')
    output = (args.output_dir or LOCAL_ROOT / 'datasets/local_runs/checks' / f'check_{stamp}_{uuid.uuid4().hex[:6]}').resolve()
    if output.is_relative_to(REPOSITORY):
        parser.error('Detailed check outputs must stay outside the publication root.')
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'GPU_inference': 'NOT_RUN', 'checks': []}
    environment = os.environ.copy()
    environment.update(PYTHONPATH=str(REPOSITORY / 'code'),
                       UAVGAP_TEST_CONFIG=str(args.config.resolve()),
                       PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8',
                       CUDA_VISIBLE_DEVICES='-1', YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true')
    for name in ['YOLO_CONFIG_DIR', 'MPLCONFIGDIR', 'TORCH_HOME', 'TMP', 'TEMP']:
        folder = output / 'runtime' / name
        folder.mkdir(parents=True)
        environment[name] = str(folder)
    (Path(environment['YOLO_CONFIG_DIR']) / 'Ultralytics').mkdir()

    def execute(name, command):
        print(f'Checking {name}...', flush=True)
        started = time.perf_counter()
        with (output / f'{name}.stdout.log').open('w', encoding='utf-8') as stdout, \
                (output / f'{name}.stderr.log').open('w', encoding='utf-8') as stderr:
            result = subprocess.run(command, cwd=REPOSITORY, env=environment,
                                    stdout=stdout, stderr=stderr, timeout=args.timeout)
        report['checks'].append({'name': name, 'exit_code': result.returncode,
                                 'seconds': time.perf_counter() - started, 'command': command})
        if result.returncode:
            raise RuntimeError(f'{name} failed; see {output}')

    try:
        cfg = load_config(args.config)
        report['checkpoint_SHA256'] = cfg['checkpoint_sha256']
        execute('cpu_tests', [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'code/tests', '-v'])
        flamapy_python = args.flamapy_python
        if flamapy_python is None:
            candidate = LOCAL_ROOT / ('.venv_flamapy/Scripts/python.exe' if os.name == 'nt' else '.venv_flamapy/bin/python')
            flamapy_python = candidate if candidate.is_file() else None
        if flamapy_python is not None:
            execute('uvl', [str(flamapy_python.resolve()), '-B', 'experiments/vm1_flamapy_yolo26s_validation/run.py', '--uvl-only',
                            '--output', str(output / 'uvl.json')])
            report['uvl'] = json.loads((output / 'uvl.json').read_text(encoding='utf-8'))
        elif args.require_uvl:
            raise FileNotFoundError('Specify an existing FlamaPy interpreter with --flamapy-python.')
        else:
            report['uvl'] = 'NOT_RUN: no FlamaPy interpreter supplied'
        if args.run_dir:
            source = args.run_dir.resolve()
            summary = json.loads((source / 'summary.json').read_text(encoding='utf-8'))
            if summary['Status'] != 'PASS':
                raise ValueError('Metric regression requires a completed run.')
            for filename, key in [('metrics.json', 'metrics_SHA256'), ('ground_truth.json', 'GT_SHA256'),
                                  ('predictions_raw.json', 'predictions_SHA256')]:
                if sha256(source / filename) != summary[key]:
                    raise ValueError('Saved artifact SHA mismatch: ' + filename)
            execute('regenerate', [sys.executable, '-B', 'experiments/experiments_test/vm03_gpu_pilot/run.py',
                                   '--regenerate', str(source), '--output', str(output / 'metrics_regenerated.json')])
            actual = json.loads((output / 'metrics_regenerated.json').read_text(encoding='utf-8'))
            expected = json.loads((source / 'metrics.json').read_text(encoding='utf-8'))
            if actual != expected:
                raise ValueError('Regenerated metrics differ from saved metrics.')
            report['metric_regression'] = {'status': 'EXACT_MATCH', 'source_run': summary['RunID']}
        else:
            report['metric_regression'] = 'NOT_RUN: pass --run-dir to check saved predictions'
        report['source_SHA256'] = {str(p.relative_to(REPOSITORY)): sha256(p)
                                   for p in sorted((REPOSITORY / 'code').rglob('*.py'))}
        report['status'] = 'PASS'
    except Exception as exc:
        report['status'] = 'FAILED'
        report['error'] = repr(exc)
    save(output / 'check.json', report)
    print(f"{report['status']}: {output}", flush=True)
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
