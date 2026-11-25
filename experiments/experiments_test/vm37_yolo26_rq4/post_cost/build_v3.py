"""Assemble a separate optimized package from the verified v2 bundle."""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

POST = Path(__file__).resolve().parent
sys.path.insert(0, str(POST.parent))
from study import *
import portable

PRIOR = 'vm37_yolo26_rq4_portable_v2_cpu_20261112T215126_7275ee89'
ACCEPTED = 'vm37_yolo26_rq4_input_optimization_summary_cpu_20261113T101822_9f633b8c'
REGRESSION = 'vm37_yolo26_rq4_optimized_input_regression_20261113T101042_cb2c96be'
PREFIX = 'vm37_yolo26_rq4_all_five_short_cost_20261112T195422_06010323'


def execute():
    out = create('portable_v3_build_cpu')
    runtime(out, cuda=False)
    began = time.perf_counter()
    prior = run_path(PRIOR) / 'bundle'
    original = portable.verify_bundle(prior)
    accepted = load(run_path(REGRESSION) / 'numerical_comparison.json')
    if accepted['status'] != 'GPU_NUMERICAL_EXACT_PASS' or not all(accepted['checks'].values()):
        raise ValueError('Accepted numerical input regression is incomplete.')
    for name in ('fast_input.py', 'input_ops.py'):
        frozen = load(run_path(REGRESSION) / 'protocol.json')['sources']
        key = next(key for key in frozen if Path(key).name == name)
        if sha256(POST / name) != frozen[key]:
            raise ValueError('Verified numerical implementation changed.')
    bundle = out / 'bundle'
    bundle.mkdir()
    for relative, expected in original['files'].items():
        target = bundle / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(prior / relative, target)
        if sha256(target) != expected:
            raise ValueError('A v2 frozen member did not copy exactly.')
    names = ('portable_v3.py', 'evaluate_v3.py', 'compare_v3.py', 'test_v3.py', 'fast_input.py', 'input_ops.py',
             'verify_input.py', 'profile_input.py', 'test_input.py')
    for name in names:
        shutil.copyfile(POST / name, bundle / 'main/experiments/vm37_yolo26_rq4/post_cost' / name)
    policy_sha = sha256(bundle / 'assets/TARGETED_POLICY.json')
    formal = load(bundle / 'formal100_protocol.json')
    for kind, architecture in (('benchmark26', 'YOLO26s'), ('benchmark8', 'YOLOv8s')):
        protocol = {**formal, 'id': 'VM37_SERVER_EPOCH1_' + architecture.upper() + '_V3',
            'status': 'PREPARED_GPU_NOT_AUTHORIZED', 'architectures': [architecture], 'seeds': [42],
            'benchmark_only': True, 'execute_default': False, 'completed_epochs_required': 1,
            'schedule_horizon_epochs': 100, 'training_epochs_to_execute': 1,
            'candidate_per_arm_hard_seconds': 7200, 'candidate_total_hard_seconds': 14400,
            'full_matrix_training_arms': 2, 'planned_counters_per_arm': expected_steps(6471, 100, stop_epochs=1),
            'nominal_formal_counters_per_arm': formal['planned_counters_per_arm'],
            'selection': 'Epoch1 EMA for engineering cost only', 'contexts_per_checkpoint': 0,
            'inference_resolutions': [], 'evaluation_configurations_per_matched_pair': 0,
            'scope': 'Engineering cost only; full6471/548, not a100-epoch arm.',
            'metrics': 'Library development validation only; research COCO/bootstrap cost remains separate.'}
        save(bundle / (kind + '.json'), protocol, out)
    prefix_rows = [json.loads(line) for line in (run_path(PREFIX) / 'training_batches.jsonl').read_text().splitlines()]
    ids = [identity for row in prefix_rows[:8] for identity in row['image_ids']]
    save(bundle / 'cpu_comparison_manifest.json', {'image_ids': ids, 'source_RunID': PREFIX,
         'source_SHA256': sha256(run_path(PREFIX) / 'training_batches.jsonl'),
         'selection': 'First8 actual training batches; no response-based selection.', 'training_images_copied': 0}, out)
    save(bundle / 'input_regression_evidence.json', {
         'RunID': REGRESSION, 'source_acceptance_SHA256': sha256(run_path(REGRESSION) / 'numerical_comparison.json'),
         'reference_state_SHA256': accepted['reference']['sha256'], 'new_state_SHA256': accepted['new_state']['sha256'],
         'checks': accepted['checks'], 'batches': 128, 'optimizer_updates': 128,
         'scope': 'Historical Windows GPU input/state regression; not a GPU validation of this v3 orchestrator.'}, out)
    boundaries = {'evaluation_population': 'VisDrone2019-DET validation548; DEVELOPMENT_ONLY',
         'test_dev': 'DATASET_BLOCKED; independent admission unchanged; no inference or training authorized.',
         'UAVDT': 'BLOCKED_ANNOTATION_PROVENANCE; no compatible DET/ignore protocol admitted.',
         'new_dataset_substitution': False, 'ALL_FIVE': 'Five independent0.5 gates; broad stochastic augmentation.',
         'Gap_guided_strategy': 'Not implemented or established by this package.',
         'haze': 'Uncalibrated full-frame constant-depth proxy.', 'exact_training_resume': 'NOT_VALIDATED'}
    save(bundle / 'SCIENTIFIC_BOUNDARIES.json', boundaries, out)
    shutil.copyfile(HERE / 'YOLOV8_ABLATIONS.csv', bundle / 'YOLOV8_ABLATIONS_FROZEN.csv')
    for kind in ('formal100', 'pilot20', 'benchmark26', 'benchmark8'):
        spec = load(bundle / (kind + '_protocol.json' if kind in ('formal100', 'pilot20') else kind + '.json'))
        template = {'approved': False, 'authorization_scope': 'NEW_TARGET_SERVER_ONLY', 'protocol_id': spec['id'],
            'bundle_manifest_SHA256': 'SET_AFTER_REVIEW', 'input_pipeline': 'optimized',
            'arms': [list(row) for row in portable.matrix(spec)],
            'cooling_confirmed': False, 'power_confirmed': False, 'machine_admitted': False,
            'CPU_receipt_SHA256': 'SET_AFTER_TARGET_CPU_COMPARISON', 'total_seconds': 0,
            'absolute_deadline_UTC': None, 'completed_workers': [], 'purpose':
            'TARGET_SERVER_FULL_EPOCH_COST_ONLY' if kind.startswith('benchmark') else 'MATCHED_RESEARCH_STUDY',
            'per_arm_hard_seconds': 7200 if kind.startswith('benchmark') else None,
            'resource_admission': {'status': 'NOT_ADMITTED', 'ALL_FIVE_GPU_cost_measured': False,
                'required_complete_matrix_seconds': None, 'cost_evidence_relative': 'NOT_SET',
                'cost_evidence_SHA256': 'NOT_SET', 'measured_architectures': [], 'evaluation_and_bootstrap_included': False},
            'scope': 'Unapproved template, not a new compute allocation.'}
        save(bundle / ('authorization_v3_' + kind + '.example.json'), template, out)
    population = load(bundle / 'portable_dataset_manifest.json')
    dataset = ROOT / 'datasets/VisDrone'
    storage = {}
    for split, rows in population.items():
        image_bytes = sum((bundle / row['image_override'] if row['image_override'] else dataset / row['image_relative']).stat().st_size for row in rows)
        annotation_bytes = sum((dataset / row['annotation_relative']).stat().st_size for row in rows)
        storage[split] = {'images': len(rows), 'image_bytes': image_bytes, 'annotation_bytes': annotation_bytes}
    checkpoint = load(run_path('vm37_yolo26_rq4_all_five_full_epoch_cost_20261112T200430_2ab983f6') / 'training_acceptance.json')['checkpoint']
    environment = load(bundle / 'environment_observed.json')
    specification = {'reference': environment, 'local_environment_modified': False,
        'observed_training': {'model': 'YOLO26s', 'imgsz': 1280, 'batch': 2, 'AMP': False, 'workers': 0,
            'complete_epoch_maximum_NVML_MiB': 7935, 'short_optimized_regression_maximum_NVML_MiB': 5533,
            'CPU_input_sampled_peak_RSS_bytes': 784134144},
        'server_planning_not_certified_minima': {'exclusive_logical_CPU_threads': 8, 'RAM_GiB': 32,
            'GPU_memory_GiB': 16, 'training_worker': 1, 'native_effect_threads': 2,
            'Torch_threads': 2, 'OpenCV_threads': 2},
        'dataset_storage': storage, 'measured_YOLO26s_epoch_checkpoint_bytes': checkpoint['bytes'],
        'YOLO26s_six_arms_100_checkpoints_storage_scenario_bytes': checkpoint['bytes'] * 600,
        'YOLOv8s_checkpoint_storage': 'NOT_MEASURED', 'research_prediction_and_bootstrap_storage': 'NOT_MEASURED_FOR_THE_COMPLETE_MATRIX',
        'server_storage_admission': 'Must cover dataset,12x100 checkpoints, raw predictions, bootstrap evidence, environment and reserve.',
        'Linux_validation': 'NOT_PERFORMED; exact pins, official wheel/source checks and target-host CPU parity required.',
        'CUDA_driver': 'A driver compatible with the observed PyTorch cu124 runtime is required; minimum driver version not asserted.',
        'cross_host_bitwise_training': 'NOT_ESTABLISHED'}
    save(bundle / 'SERVER_ENVIRONMENT.json', specification, out)
    docs = '''# VM37 optimized portable package v3.1

This private package preserves v2, the original training code, scientific protocols, initializations and evaluation assets. `portable.py` remains the baseline executor. `portable_v3.py` uses the same trainer with opt-in versioned input handling. Both matched arms use one frozen pipeline choice. Workers remain zero.

## Environment and data

Use a new Python 3.11 environment matching `requirements.lock.txt`, `constraints.lock.txt` and `environment_observed.json`: Ultralytics8.4.210, PyTorch2.6.0+cu124, torchvision0.21.0+cu124. The official Ultralytics wheel is in `assets/wheels/`. No installer runs automatically. Do not modify the old local environment.

Linux environment preparation commands, for manual execution on the target server:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.lock.txt -c constraints.lock.txt
python -m pip install --no-deps --force-reinstall assets/wheels/ultralytics-8.4.210-py3-none-any.whl
```

Linux wheels, system OpenCV libraries and the target GPU/driver remain unverified. Preserve the exact OpenCV/JPEG implementation; do not silently replace packages to make a check pass. Target-host CPU comparison and measured GPU admission are required. Cross-machine bitwise training is not claimed.

Planning provision:8 exclusive logical CPU threads,32GiB RAM and16GiB GPU memory. These are headroom recommendations, not measured minimum requirements. The local8GiB laptop reached7935MiB in the original full-epoch worker. Workers=0, native-effect threads=2, Torch threads=2, OpenCV threads=2 remain frozen. See `SERVER_ENVIRONMENT.json` for measured dataset bytes and checkpoint storage scenarios. A complete100-epoch matrix retains1200 epoch checkpoints; measure YOLOv8s checkpoint/prediction sizes before storage admission.

`DATA_ROOT` must contain original `VisDrone2019-DET-train` and `VisDrone2019-DET-val`, with images and annotations. All6471/548 members and hashes are verified. Approved derived labels and one original-byte validation override are already in the package. Do not copy historical research archives. Never point outputs at the dataset, `main/` or `assets/`.

## CPU validation and dry-runs

Run from this package root; replace only the external data and output paths:

```sh
python -B main/experiments/vm37_yolo26_rq4/post_cost/test_v3.py --output /work/results/v3_unit
python -B main/experiments/vm37_yolo26_rq4/post_cost/compare_v3.py --bundle . --data-root /data/VisDrone --output-root /work/results/vm37
python -B main/experiments/vm37_yolo26_rq4/post_cost/portable_v3.py --bundle . --data-root /data/VisDrone --protocol benchmark26 --output-root /work/results/vm37
python -B main/experiments/vm37_yolo26_rq4/post_cost/portable_v3.py --bundle . --data-root /data/VisDrone --protocol formal100 --output-root /work/results/vm37
```

The comparator checks six healthy native10 initializations and150 saved evaluation-pixel checks first, then covers both architectures, seeds42/43/44, CONTROL/ALL_FIVE, epoch0/1 and the epoch90 closed-Mosaic state. Each comparison uses8 fixed batches and2 alternating repeats. It checks tensors, boxes, classes, native pixel hashes, parameters, activation, Mosaic IDs, occurrence counts, cache order and RNG. Timings describe CPU input replay only.

Use `--input-pipeline baseline` for a separate baseline dry-run. The selection is frozen for the entire authorized matrix; never mix implementations between arms.

## Bounded server cost benchmark: prepared, not authorized

`benchmark26` is a matched YOLO26s CONTROL/ALL_FIVE engineering measurement. Both start from the same seed42 native10 initialization, with6471 train and548 development images. Training declares the original100-epoch schedule but stops after epoch1. It is not a completed100-epoch arm. The candidate cap is7200s per arm and14400s total; the author may approve a smaller finite cap. It includes initialization, source verification, loading, forward/backward, health checks, internal validation and FP32 save/reload. Research COCO, full corruptions and bootstrap costs remain separate.

After separate explicit authorization, review a copy of `authorization_v3_benchmark26.example.json`, bind the target CPU receipt and bundle SHA, and set a real new finite budget and UTC expiry. Defaults refuse execution. Future command, not run during package preparation:

```sh
python -B main/experiments/vm37_yolo26_rq4/post_cost/portable_v3.py --bundle . --data-root /data/VisDrone --protocol benchmark26 --output-root /work/results/vm37 --cpu-receipt /work/results/vm37/CPU_RUN/CPU_receipt.json --authorization /work/author_approved_benchmark26.json
```

`benchmark8` prepares the same cost measurement for YOLOv8s. Do not extrapolate its speed from YOLO26s. Each execution has one immutable ticket and one GPU worker. Stop on numerical/device failure, OOM, sampled temperature>=90C, or>=87C for30s. No retry, automatic tuning, budget reset or deadline extension.

## Formal matched study: unchanged, not authorized

The formal protocol remains YOLO26s and YOLOv8s x seeds42/43/44 x CONTROL/ALL_FIVE x100 epochs. All arms use FP32,1280,batch2,nbs64,AdamW lr0=.01,warmup3,workers0 and fixed epoch100 EMA. Each pair shares the same native10 initialization SHA. Every epoch uses full548-image internal validation. Final research evaluation retains21 contexts,640/1280,native10,ignore-union COCO and200 paired image-bootstrap draws. The12 arms imply504 final evaluation configurations. The five original YOLOv8s augmentation ablations remain separately frozen in `YOLOV8_ABLATIONS_FROZEN.csv`; they are not silently added to the12-arm authorization.

Measured target-machine costs for both models, full evaluation/bootstrap, checkpoint storage and a safety reserve must fit before either arm starts. After separate author approval only:

```sh
python -B main/experiments/vm37_yolo26_rq4/post_cost/portable_v3.py --bundle . --data-root /data/VisDrone --protocol formal100 --output-root /work/results/vm37 --cpu-receipt /work/results/vm37/CPU_RUN/CPU_receipt.json --authorization /work/author_approved_formal100.json
```

Historical local48-hour authorization is read-only evidence, not a server allocation. Partial runs preserve valid checkpoints and logs. Exact training resume is not validated: do not reuse an interrupted or cost checkpoint as a new arm's initialization.

ALL_FIVE is broad stochastic augmentation with five independent0.5 gates. It is not an established Gap-guided strategy. Haze is an uncalibrated constant-depth proxy. Test-dev and UAVDT remain blocked under the existing admission boundaries. This package claims no AP, robustness or empirical-gap improvement.
'''
    (bundle / 'README.md').write_text(docs, encoding='utf-8')
    revision = {'version': 'VM37_PORTABLE_V3.1', 'prior_manifest_SHA256': sha256(prior / 'bundle_manifest.json'),
        'original_protocols_unchanged': all(sha256(bundle / name) == sha256(prior / name) for name in
            ('formal100_protocol.json', 'pilot20_protocol.json', 'portable_dataset_manifest.json', 'portable_initializations.json',
             'evaluation_manifest.json', 'formal_evaluation_manifest.json', 'assets/TARGETED_POLICY.json')),
        'base_executor_SHA256': sha256(bundle / 'main/experiments/vm37_yolo26_rq4/post_cost/portable.py'),
        'optimized_source_files': {name: sha256(POST / name) for name in names},
        'policy_SHA256': policy_sha, 'new_GPU_seconds': 0, 'GPU_v3_orchestrator': 'NOT_VALIDATED',
        'local_budget_modified': False, 'raw_images_copied': 0}
    if not revision['original_protocols_unchanged']:
        raise ValueError('A scientific protocol changed.')
    save(bundle / 'V3_REVISION.json', revision, out)
    files = {path.relative_to(bundle).as_posix(): sha256(path) for path in sorted(bundle.rglob('*')) if path.is_file()}
    save(bundle / 'bundle_manifest.json', {**original, 'files': files,
        'execution_wrapper_version': 'VM37_PORTABLE_V3.1', 'assembly_git_HEAD_before_final_commit': subprocess.run(
        ['git', 'rev-parse', 'HEAD'], cwd=PUBLIC, capture_output=True, text=True, check=True).stdout.strip()}, out)
    portable.verify_bundle(bundle)
    save(out / 'bundle_result.json', {'status': 'ASSEMBLED_V3_CPU_VALIDATION_PENDING', 'bundle': str(bundle),
        'bundle_manifest_SHA256': sha256(bundle / 'bundle_manifest.json'), 'files': len(files),
        'bytes': sum((bundle / name).stat().st_size for name in files), 'revision': revision,
        'CPU_seconds': time.perf_counter() - began, 'GPU_seconds': 0}, out)
    save(out / 'inputs.json', {'prior_v2_manifest': source(prior / 'bundle_manifest.json'),
        'accepted_input_regression': source(run_path(REGRESSION) / 'numerical_comparison.json'),
        'CPU_report_provenance': source(run_path(ACCEPTED) / 'report_provenance.json'),
        'original_budget_before': source(RUNS / 'compute_allocation.json')}, out)
    write_atomic(out / 'diary.md', '# Portable v3 assembly\n\nCreated an independent package from frozen v2 members. '
        'Integrated the verified native pipeline for both arms. Preserved protocols, initialization and boundaries. '
        'Prepared unapproved server benchmark/formal commands. GPU=0.\n', out)
    save(out / 'run.json', {'RunID': out.name, 'ExperimentID': EXPERIMENT,
        'status': 'ASSEMBLED_V3_CPU_VALIDATION_PENDING', 'GPU_seconds': 0}, out, replace=True)
    seal(out)
    print(out, flush=True)


if __name__ == '__main__':
    execute()
