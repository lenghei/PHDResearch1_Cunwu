"""Assemble a local-only training bundle without copying original datasets."""
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import hashlib
import zipfile

POST = Path(__file__).resolve().parent
sys.path.insert(0, str(POST.parent))
from study import *

FROZEN = 'vm37_yolo26_rq4_postcost_protocol_cpu_20261112T132257_50b16e94'
ADMISSION = 'vm37_yolo26_rq4_postcost_admission_cpu_20261112T130427_8e926bc0'
COMMIT = '6e59a753705999d03eb4c280748c66778bb4b312'


def execute():
    out = create('portable_bundle_cpu')
    runtime(out, cuda=False)
    started = time.perf_counter()
    frozen = run_path(FROZEN)
    verify_hashes(frozen, ('private_asset_sources.json', 'portable_dataset_manifest.json', 'portable_initializations.json'))
    bundle = out / 'portable_bundle'
    bundle.mkdir()
    tracked = subprocess.run(['git', 'ls-tree', '-r', '--name-only', COMMIT, '--', 'code', 'experiments'],
                             cwd=PUBLIC, capture_output=True, text=True, check=True).stdout.splitlines()
    exported = []
    for relative in tracked:
        if Path(relative).suffix not in ('.py', '.uvl'):
            continue
        data = subprocess.run(['git', 'show', COMMIT + ':' + relative], cwd=PUBLIC, capture_output=True, check=True).stdout
        target = bundle / 'main' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        exported.append(relative)
    for path in POST.iterdir():
        if path.suffix in ('.py', '.md', '.json'):
            target = bundle / 'main/experiments/vm37_yolo26_rq4/post_cost' / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    for name in ('pilot20_protocol.json', 'formal100_protocol.json', 'portable_dataset_manifest.json',
                 'portable_initializations.json', 'evaluation_manifest.json', 'ordinary_training_args.json', 'environment_observed.json'):
        shutil.copyfile(frozen / name, bundle / name)
    shutil.copyfile(run_path(ADMISSION) / 'safety_policy.json', bundle / 'safety_policy.json')
    assets = load(frozen / 'private_asset_sources.json')
    source_environment = run_path(load(RUNS / 'source_environment.json')['RunID'])
    wheel = load(source_environment / 'environment_setup.json')['wheel']
    if sha256(Path(wheel['path'])) != wheel['sha256']:
        raise ValueError('Accepted official wheel changed.')
    assets.append({'path': wheel['path'], 'asset': 'assets/wheels/ultralytics-8.4.210-py3-none-any.whl', 'sha256': wheel['sha256']})
    with zipfile.ZipFile(wheel['path']) as archive:
        sources = {name.removeprefix('ultralytics/'): hashlib.sha256(archive.read(name)).hexdigest()
                   for name in archive.namelist() if name.startswith('ultralytics/') and name.endswith(('.py', '.yaml'))}
    save(bundle / 'ultralytics_sources.json', sources, out)
    evaluation = load(bundle / 'evaluation_manifest.json')
    old = load(PREP / 'FROZEN_DEVELOPMENT_EVALUATION.json')['conditions']
    by_leaves = {tuple(sorted(row['leaves'])): row for row in old
                 if row['checkpoint'] == 'ck01' and row['inference_resolution'] == 1280}
    if len(by_leaves) != 21:
        raise ValueError('Formal evaluation must preserve21 accepted contexts.')
    evaluation['contexts'] = []
    for leaves, row in sorted(by_leaves.items()):
        source_dir = Path(row['source_folder'])
        config = load(source_dir / 'configuration.json')
        if sha256(source_dir / 'configuration.json') != row['source_SHA256']['configuration.json']:
            raise ValueError('Accepted recipe changed.')
        identity = 'Normal' if leaves == ('Normal',) else row['pair_id'].split('__')[0]
        asset = 'assets/formal_evaluation_pixels/' + identity + '.jsonl'
        assets.append({'path': str(source_dir / 'image_manifest.jsonl'), 'asset': asset,
                       'sha256': row['source_SHA256']['image_manifest.jsonl']})
        evaluation['contexts'].append({'id': identity, 'p': config['p'], 'historical_source_pair': row['pair_id'],
                'pixel_manifest_asset': asset, 'pixel_manifest_SHA256': row['source_SHA256']['image_manifest.jsonl']})
    evaluation['not_full21_condition_gap'] = False
    evaluation['bootstrap_samples'] = 200
    save(bundle / 'formal_evaluation_manifest.json', evaluation, out)
    for row in assets:
        source_path = Path(row['path'])
        if sha256(source_path) != row['sha256']:
            raise ValueError('Private asset changed: ' + str(source_path))
        target = bundle / row['asset']
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError('Duplicate asset destination.')
        shutil.copyfile(source_path, target)
        if sha256(target) != row['sha256']:
            raise ValueError('Copied asset differs.')
    observed = load(bundle / 'environment_observed.json')
    write_atomic(bundle / 'requirements.lock.txt', '--extra-index-url https://download.pytorch.org/whl/cu124\n' +
        '\n'.join(name + '==' + version for name, version in sorted(observed['packages'].items())) + '\n', out)
    # Record installed distributions; do not install or change this environment.
    distributions = {dist.metadata['Name']: dist.version for dist in importlib.metadata.distributions() if dist.metadata['Name']}
    save(bundle / 'installed_distributions.json', distributions, out)
    write_atomic(bundle / 'constraints.lock.txt', '\n'.join(name + '==' + version for name, version in sorted(distributions.items(), key=lambda row: row[0].lower())) + '\n', out)
    original_author = load(RUNS / 'compute_allocation.json')
    save(bundle / 'original_allocation_READ_ONLY.json', original_author, out)
    save(bundle / 'authorization.example.json', {'approved': False, 'protocol_id': 'VM37_FORMAL100',
        'bundle_manifest_SHA256': 'SET_AFTER_REVIEW', 'arms': [[a, s, arm] for a in ('YOLO26s', 'YOLOv8s') for s in (42,43,44) for arm in ('CONTROL','ALL_FIVE')],
        'cooling_confirmed': False, 'power_confirmed': False, 'machine_admitted': False,
        'total_seconds': 0, 'absolute_deadline_UTC': None, 'completed_workers': [],
        'resource_admission': {'status': 'NOT_ADMITTED', 'ALL_FIVE_GPU_cost_measured': False,
            'required_complete_matrix_seconds': None, 'cost_evidence_relative': 'NOT_SET', 'cost_evidence_SHA256': 'NOT_SET'},
        'scope': 'Template only. A new server allocation requires explicit author approval; this is not an allocation.'}, out)
    write_atomic(bundle / 'README.md', '''# VM37 portable training

Local-only bundle. No data or weights have been uploaded.

Use Python3.11 and a separately prepared environment matching `requirements.lock.txt` and `constraints.lock.txt`. CUDA/driver compatibility and exact package-source checks require admission on the destination machine. No installer runs automatically.

Set `DATA_ROOT` to the directory containing `VisDrone2019-DET-train` and `VisDrone2019-DET-val`. Supply the original images and annotations with the recorded hashes; the bundle contains approved derived labels and the one original-byte validation override.

CPU dry-run:

```text
python -B main/experiments/vm37_yolo26_rq4/post_cost/portable.py --bundle . --data-root DATA_ROOT --protocol formal100
```

Use `--protocol pilot20` for the separate20-epoch exploratory protocol. Dry-run verifies source/data/assets and writes plans; it does not train or run model inference.

Execution requires a separately reviewed authorization JSON, complete-matrix cost admission, machine readiness and a fixed resource deadline:

```text
python -B main/experiments/vm37_yolo26_rq4/post_cost/portable.py --bundle . --data-root DATA_ROOT --protocol formal100 --authorization AUTHOR_APPROVED.json
```

The authorization template refuses execution. Do not reuse the expired local admission window. Resource admission must include both arms, all seeds/models, generation, validation, saving, research evaluation, bootstrap and a safety margin. A single immutable ticket prevents automatic retry. One GPU worker runs at a time.

Training uses the accepted trainer and FP32 health/saving hooks. Each arm starts from its native10 initialization. Fixed final-epoch EMA is evaluated; early cost checkpoints are never reused. Failed runs retain evidence. Exact training resume is not validated: RNG, sampler and pending gradients are not restored. Do not resume a partial arm as an exact matched run.

Pilot:20 epochs/arm, YOLO26s, seed42,4 contexts,640/1280. Formal:100 epochs/arm, YOLO26s/YOLOv8s,seeds42/43/44,21 contexts,640/1280. Both are development evaluations. ALL_FIVE is broad stochastic augmentation, not proven Gap-guided training. No test-dev admission is included.

The portable orchestration is CPU-tested only. Real multi-epoch GPU execution and new-machine stability remain unvalidated. Larger GPUs may not remove CPU augmentation bottlenecks with the frozen workers=0 setting. Budget and temperature stops preserve partial evidence and do not retry.
''', out)
    files = {p.relative_to(bundle).as_posix(): sha256(p) for p in sorted(bundle.rglob('*')) if p.is_file()}
    save(bundle / 'bundle_manifest.json', {'source_commit': COMMIT, 'files': files,
          'original_dataset_copies': 0, 'private_assets': len(assets), 'external_uploads': 0,
          'GPU_execution': 'NOT_RUN', 'source_paths': exported}, out)
    save(out / 'bundle_result.json', {'bundle': str(bundle), 'bundle_manifest_SHA256': sha256(bundle / 'bundle_manifest.json'),
        'files': len(files), 'bytes': sum(p.stat().st_size for p in bundle.rglob('*') if p.is_file()),
        'private_assets': len(assets), 'source_commit': COMMIT, 'wall_seconds': time.perf_counter() - started,
        'status': 'ASSEMBLED_NOT_GPU_VALIDATED', 'dataset_copies': 0, 'external_uploads': 0}, out)
    save(out / 'inputs.json', {'protocol_RunID': FROZEN, 'admission_RunID': ADMISSION,
          'original_ledger': source(RUNS / 'compute_allocation.json')}, out)
    write_atomic(out / 'diary.md', '# Portable bundle\n\nCopied accepted source, native10 initializations and small private provenance assets into a new local run. Original data and archives remain in place. No GPU task, upload or new resource allocation.\n', out)
    save(out / 'run.json', {'RunID': out.name, 'ExperimentID': EXPERIMENT, 'mode': 'PORTABLE_BUNDLE_CPU',
                          'status': 'ASSEMBLED_NOT_GPU_VALIDATED', 'GPU_seconds': 0}, out, replace=True)
    seal(out)
    print(out, flush=True)


if __name__ == '__main__':
    execute()
