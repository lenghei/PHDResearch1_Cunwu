"""CV261102: independent new local YOLOv8s fine-tuning -> 548-image evaluation.

No previous experiments, reference CSVs, or GitHub network calls are used.
This task depends on the existing uavgap/main/code/main evaluation implementation.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
import uuid
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
PUBLIC_ROOT = TASK_DIR.parents[1]  # uavgap/main
sys.path.insert(0, str(PUBLIC_ROOT / 'code'))
from prepare_data import NAMES, digest, prepare, list_members, scan_archive_annotations  # noqa: E402

TASK = 'cv261102'
DEFAULTS = {
    'epochs': 100, 'train_imgsz': 640, 'eval_imgsz': 1280,
    'batch': 4, 'eval_batch': 2, 'workers': 0, 'seed': 42,
}


def save_json(path: Path, item: object) -> None:
    """Atomic replacement so a broken session retains a valid manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.temporary')
    temporary.write_text(json.dumps(item, indent=2, ensure_ascii=True, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def read_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def export_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f'Refusing to export empty CSV: {path.name}')
    keys = list(rows[0])
    if any(set(row) != set(keys) for row in rows):
        raise ValueError(f'CSV schema differs by row: {path.name}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def record_git() -> dict:
    try:
        a = subprocess.run(['git', '-C', str(PUBLIC_ROOT), 'rev-parse', 'HEAD'],
                           capture_output=True, text=True, check=False)
        b = subprocess.run(['git', '-C', str(PUBLIC_ROOT), 'status', '--porcelain'],
                           capture_output=True, text=True, check=False)
        return {'commit': a.stdout.strip() if a.returncode == 0 else None,
                'working_tree_clean': b.returncode == 0 and not b.stdout.strip()}
    except OSError:
        return {'commit': None, 'working_tree_clean': None}


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', type=Path, default=TASK_DIR / 'config.local.yaml',
                        help='Local YAML; copy from config.example.yaml and edit paths')
    parser.add_argument('--preflight', action='store_true', help='Check data/GPU/dependencies without training')
    parser.add_argument('--epochs', type=int, default=None, help='Override YAML training epoch count')
    parser.add_argument('--batch', type=int, default=None, help='Override YAML training batch; try 2 if OOM')
    parser.add_argument('--train-imgsz', type=int, default=None, help='Override training input resolution')
    parser.add_argument('--eval-imgsz', type=int, default=None, help='Override evaluation input resolution')
    return parser.parse_args(argv)


def read_config(args) -> dict:
    import yaml

    cfg_path = args.config.expanduser().resolve()
    if not cfg_path.is_file():
        raise FileNotFoundError(f'Config missing: {cfg_path}\nCopy config.example.yaml to config.local.yaml and edit local paths.')
    cfg = yaml.safe_load(cfg_path.read_text(encoding='utf-8'))
    if not isinstance(cfg, dict):
        raise ValueError('Local config YAML must contain key/value settings.')
    extras = set(cfg) - {'train_zip', 'data_yaml', 'val_zip', 'weights', 'output_root', *DEFAULTS}
    if extras:
        raise ValueError('Unrecognized config keys: ' + ', '.join(sorted(extras)))
    result = dict(DEFAULTS)
    result.update(cfg)
    result['config_source_SHA256'] = digest(cfg_path)
    result['config_source'] = str(cfg_path)
    for key, attr in [('epochs', 'epochs'), ('batch', 'batch'),
                      ('train_imgsz', 'train_imgsz'), ('eval_imgsz', 'eval_imgsz')]:
        if getattr(args, attr) is not None:
            result[key] = getattr(args, attr)
    for key in ('epochs', 'batch', 'eval_batch', 'workers', 'seed', 'train_imgsz', 'eval_imgsz'):
        value = result[key]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f'{key} must be an integer')
    if result['epochs'] < 1 or result['batch'] < 1 or result['eval_batch'] < 1 or result['workers'] < 0:
        raise ValueError('epochs, batch and eval_batch must be positive; workers non-negative')
    if result['seed'] < 0:
        raise ValueError('seed must be non-negative')
    if min(result['train_imgsz'], result['eval_imgsz']) < 320:
        raise ValueError('Both input image sizes must be at least 320')
    if bool(result.get('train_zip')) == bool(result.get('data_yaml')):
        raise ValueError('Set exactly one of train_zip or data_yaml in the config.')
    for key in ('train_zip', 'data_yaml', 'val_zip', 'weights', 'output_root'):
        item = result.get(key)
        if key in ('val_zip', 'weights') and not item:
            raise ValueError(f'Missing required {key} in config.')
        if not item:
            result[key] = None
            continue
        expanded = os.path.expandvars(str(item))
        if expanded.startswith('CHANGEME') or '${' in expanded or '%' in expanded:
            raise ValueError('Set a real local path for ' + key)
        path = Path(expanded).expanduser()
        if not path.is_absolute():
            path = cfg_path.parent / path
        result[key] = path.resolve()
    if result['output_root'] is None:
        result['output_root'] = (PUBLIC_ROOT.parent / 'datasets/local_runs/cv261102').resolve()
    if result['output_root'].is_relative_to(PUBLIC_ROOT.resolve()):
        raise ValueError('Output root cannot be inside main (GitHub sync directory).')
    return result



def validate_existing_split(desc: dict, yaml_path: Path, source_zip: Path, split: str, expected: int) -> set[str]:
    """Reject preconverted splits whose source images differ from the original ZIP."""
    import hashlib
    import zipfile

    base = Path(str(desc.get('path') or yaml_path.parent)).expanduser()
    if not base.is_absolute():
        base = (yaml_path.parent / base).resolve()
    location = Path(str(desc[split]))
    if not location.is_absolute():
        location = base / location
    if location.is_dir():
        paths = sorted(p for p in location.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    elif location.is_file() and location.suffix.lower() == '.txt':
        paths = []
        for line in location.read_text(encoding='utf-8-sig').splitlines():
            line = line.strip()
            if not line:
                continue
            candidate = Path(line)
            if not candidate.is_absolute():
                candidate = (location.parent / candidate).resolve()
            paths.append(candidate)
    else:
        raise ValueError(f'Cannot locate {split} images in YAML: {location}')
    if len(paths) != expected or len({p.stem for p in paths}) != expected:
        raise ValueError(f'Expected {expected} unique {split} images, found {len(paths)}')
    with zipfile.ZipFile(source_zip) as archive:
        members, _ = list_members(archive)
        source = {Path(member).stem: member for member in members}
        if set(source) != {p.stem for p in paths}:
            raise ValueError(f'{split} stems do not match the annotated source ZIP')
        for item in paths:
            if not item.is_file():
                raise FileNotFoundError(item)
            if hashlib.sha256(item.read_bytes()).digest() != hashlib.sha256(archive.read(source[item.stem])).digest():
                raise ValueError(f'{split} image differs from original source: {item.name}')
    return set(source)


def preflight(cfg: dict) -> dict:
    import torch
    import yaml
    import zipfile
    from main.labels import VIS_NAMES
    from main.evaluation import evaluate, regenerate  # noqa: F401
    import ultralytics

    if VIS_NAMES != NAMES:
        raise ValueError('Training class order does not match the evaluation native10 protocol.')
    for key in ('val_zip', 'weights', 'train_zip', 'data_yaml'):
        path = cfg[key]
        if path is not None and not path.is_file():
            raise FileNotFoundError(f'{key}: {path}')
    if cfg['weights'].name.lower() != 'yolov8s.pt':
        raise ValueError('Use the original local yolov8s.pt pretrained weights, not ck05 or renamed fine-tuned weights.')
    with zipfile.ZipFile(cfg['val_zip']) as z:
        images, _ = list_members(z)
        if len(images) != 548:
            raise ValueError('VisDrone2019-DET-val must contain exactly 548 images.')
        val_annotation_scan = scan_archive_annotations(z)
    count_train = None
    train_annotation_scan = None
    if cfg['train_zip']:
        with zipfile.ZipFile(cfg['train_zip']) as z:
            images, _ = list_members(z)
            count_train = len(images)
            if count_train != 6471:
                raise ValueError(f'Expected 6471 full VisDrone2019-DET-train images, found {count_train}')
            train_annotation_scan = scan_archive_annotations(z)
    else:
        desc = yaml.safe_load(cfg['data_yaml'].read_text(encoding='utf-8'))
        if not isinstance(desc, dict):
            raise ValueError('data_yaml must contain a mapping.')
        names = desc.get('names')
        if isinstance(names, dict):
            names = [v for _, v in sorted(names.items(), key=lambda x: int(x[0]))]
        if names != NAMES or int(desc.get('nc', 10)) != 10 or not desc.get('train') or not desc.get('val'):
            raise ValueError('Existing data_yaml must declare VisDrone native10 train, val and class order.')
        base = Path(str(desc.get('path') or cfg['data_yaml'].parent)).expanduser()
        if not base.is_absolute():
            base = (cfg['data_yaml'].parent / base).resolve()
        train_location = Path(str(desc['train']))
        if not train_location.is_absolute():
            train_location = base / train_location
        if train_location.is_file() and train_location.suffix.lower() == '.txt':
            train_paths = [Path(line.strip()) for line in train_location.read_text(encoding='utf-8-sig').splitlines()
                           if line.strip()]
        elif train_location.is_dir():
            train_paths = [p for p in train_location.rglob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        else:
            raise ValueError(f'Cannot locate the full training split from data_yaml: {train_location}')
        count_train = len(train_paths)
        train_stems = {p.stem for p in train_paths}
        if count_train != 6471 or len(train_stems) != count_train:
            raise ValueError(f'Expected full 6471 unique-image train split; data_yaml resolves to {count_train}')
        # A preconverted YAML must use the original annotated validation images.
        val_stems = validate_existing_split(desc, cfg['data_yaml'], cfg['val_zip'], 'val', 548)
        if train_stems & val_stems:
            raise ValueError('Train and val share image IDs; split contamination')
    if not torch.cuda.is_available():
        raise RuntimeError('Real CUDA GPU required; CPU fallback is forbidden for this experiment.')
    return {'gpu': torch.cuda.get_device_name(0), 'gpu_total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
            'torch': torch.__version__, 'cuda': torch.version.cuda,
            'ultralytics': ultralytics.__version__, 'python': sys.version, 'os': platform.platform(),
            'train_images': count_train, 'validation_images': 548,
            'train_annotation_scan': train_annotation_scan,
            'validation_annotation_scan': val_annotation_scan,
            'initial_weights_SHA256': digest(cfg['weights']), 'val_zip_SHA256': digest(cfg['val_zip']),
            'train_zip_SHA256': digest(cfg['train_zip']) if cfg['train_zip'] else None,
            'data_yaml_SHA256': digest(cfg['data_yaml']) if cfg['data_yaml'] else None}


def clock_epochs(model, target: Path):
    """Save on_fit_epoch_end events immediately, including per-epoch training-only duration."""
    fields = ['epoch', 'train_seconds', 'train_plus_validation_seconds', 'clock', 'recorded_at_UTC']
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8-sig', newline='') as stream:
        csv.DictWriter(stream, fieldnames=fields).writeheader()
    opened: dict[int, dict] = {}
    recorded: set[int] = set()

    def sync_gpu(trainer):
        # A GPU synchronization is needed for wall-clock epoch timings.
        import torch
        if getattr(trainer.device, 'type', None) == 'cuda' and torch.cuda.is_available():
            torch.cuda.synchronize()

    def on_train_epoch_start(trainer):
        sync_gpu(trainer)
        opened[trainer.epoch + 1] = {'start': time.perf_counter()}

    def on_train_epoch_end(trainer):
        sync_gpu(trainer)
        record = opened.get(trainer.epoch + 1)
        if record is not None:
            record['train_end'] = time.perf_counter()

    def on_fit_epoch_end(trainer):
        epoch = trainer.epoch + 1
        record = opened.get(epoch)
        if epoch in recorded or record is None or 'train_end' not in record:
            return
        sync_gpu(trainer)
        now = time.perf_counter()
        train_seconds = record['train_end'] - record['start']
        total_seconds = now - record['start']
        if not (math.isfinite(train_seconds) and math.isfinite(total_seconds) and 0 < train_seconds <= total_seconds):
            raise RuntimeError('Invalid per-epoch wall-clock evidence')
        row = dict(epoch=epoch, train_seconds=repr(train_seconds),
                   train_plus_validation_seconds=repr(total_seconds),
                   clock='time.perf_counter; live trainer callbacks',
                   recorded_at_UTC=dt.datetime.now(dt.timezone.utc).isoformat())
        with target.open('a', encoding='utf-8-sig', newline='') as stream:
            w = csv.DictWriter(stream, fieldnames=fields)
            w.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        recorded.add(epoch)
        print(f'[CV261102] Epoch {epoch}: train {train_seconds:.2f}s | train+validation {total_seconds:.2f}s', flush=True)

    model.add_callback('on_train_epoch_start', on_train_epoch_start)
    model.add_callback('on_train_epoch_end', on_train_epoch_end)
    model.add_callback('on_fit_epoch_end', on_fit_epoch_end)
    return recorded


def verify_replay(validation: Path):
    """Fresh recomputation from saved COCO prediction and GT, without GPU."""
    from main.evaluation import regenerate

    measured = read_json(validation / 'metrics.json')
    recalculated = regenerate(validation)
    for group in ('per_area', 'per_class'):
        if len(measured[group]) != len(recalculated[group]):
            raise ValueError(f'COCO regeneration mismatch: {group} length')
        for a, b in zip(measured[group], recalculated[group]):
            if a.keys() != b.keys():
                raise ValueError(f'COCO regeneration mismatch: {group} field names')
            for name in a:
                x, y = a[name], b[name]
                if isinstance(x, (float, int)) and not isinstance(x, bool) and isinstance(y, (float, int)) and not isinstance(y, bool):
                    if not math.isclose(x, y, rel_tol=1e-10, abs_tol=1e-10):
                        raise ValueError(f'COCO regeneration mismatch: {group}.{name}')
                elif x != y:
                    raise ValueError(f'COCO regeneration mismatch: {group}.{name}')
    return {'status': 'PASS', 'metric_records': {k: len(measured[k]) for k in ('per_area', 'per_class')},
            'GT_SHA256': digest(validation / 'ground_truth.json'),
            'predictions_COCO_SHA256': digest(validation / 'predictions_coco.json'),
            'metrics_SHA256': digest(validation / 'metrics.json')}


def export_reports(run: Path, train_wall_seconds: float, eval_wall_seconds: float, expected_epochs: int):
    validation = run / 'validation'
    result = read_json(validation / 'summary.json')
    metrics = read_json(validation / 'metrics.json')
    if result['images'] != 548 or result['full_validation_status'] != 'FULL_VALIDATION_PASS':
        raise ValueError('Full validation status must be PASS on all 548 images')
    gt = read_json(validation / 'ground_truth.json')
    if len(gt['images']) != 548:
        raise ValueError('Evaluation GT must contain exactly 548 source images')
    overall = next(row for row in metrics['per_area'] if row['size'] == 'all')
    csv_dir = run / 'csv'
    export_csv(csv_dir / 'metrics_overall.csv', [dict(RunID=run.name, **overall)])
    export_csv(csv_dir / 'metrics_by_area.csv', metrics['per_area'])
    export_csv(csv_dir / 'metrics_by_class.csv', metrics['per_class'])
    export_csv(csv_dir / 'evaluation_timing.csv', [dict(images=548,
        network_seconds=result['network_seconds'], coco_seconds=result['evaluation_seconds'],
        evaluation_pipeline_seconds=result['total_seconds'], evaluation_wall_seconds=eval_wall_seconds,
        train_stage_wall_seconds=train_wall_seconds,
        meaning='measured training/evaluation wall clocks, not GPU-only kernel time')])
    with (run / 'training' / 'epoch_times.csv').open('r', encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_epochs or [int(r['epoch']) for r in rows] != list(range(1, expected_epochs+1)):
        raise ValueError(f'Epoch time evidence incomplete: {len(rows)} measured; expected {expected_epochs}')
    export_csv(csv_dir / 'training_epoch_times.csv', rows)
    paths = sorted(csv_dir.glob('*.csv'))
    return {'images': 548, 'trained_epochs': len(rows),
            'AP50': overall['map50'], 'AP50_95': overall['map50_95'], 'AR50_95': overall['ar50_95'],
            'csv_SHA256': {p.name: digest(p) for p in paths},
            'metrics_SHA256': digest(validation / 'metrics.json'),
            'predictions_raw_SHA256': digest(validation / 'predictions_raw.json'),
            'GT_SHA256': digest(validation / 'ground_truth.json')}


def code_hashes() -> dict:
    """Record all shared Python modules and CV261102 code used by the run."""
    shared = PUBLIC_ROOT / 'code' / 'main'
    if not (shared / 'evaluation.py').is_file():
        raise FileNotFoundError(f'Missing shared evaluation module: {shared / "evaluation.py"}')
    files = sorted(shared.rglob('*.py')) + sorted(TASK_DIR.glob('*.py'))
    files += sorted((TASK_DIR / 'tests').glob('test_*.py'))
    return {p.relative_to(PUBLIC_ROOT).as_posix(): digest(p) for p in files}


def main(argv=None):
    args = parse(argv)
    cfg = read_config(args)
    env = preflight(cfg)
    source_hashes = code_hashes()
    print('[CV261102] Preflight PASS: 548 val | YOLOv8s | GPU ' + env['gpu'], flush=True)
    for name in ('train_annotation_scan', 'validation_annotation_scan'):
        report = env[name]
        if report is not None:
            print(f'[CV261102] {name}: {report["rows"]} source rows; '
                  f'{report["zero_area"]} active zero-area records excluded from YOLO targets',
                  flush=True)
    if args.preflight:
        print('[CV261102] Preflight only; no training or output Run created.', flush=True)
        return

    from ultralytics import YOLO
    import yaml

    root = cfg['output_root']
    root.mkdir(parents=True, exist_ok=True)
    run_id = f'{TASK}_full_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}'
    run = root / run_id
    run.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    status = dict(RunID=run_id, ExperimentID=TASK, status='RUNNING',
                  scope='FRESH_TRAINING_AND_DEVELOPMENT_VALIDATION_NOT_INDEPENDENT_TEST',
                  created_at_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
                  configuration={k: str(v) if isinstance(v, Path) else v for k, v in cfg.items()},
                  system=env, code_SHA256=source_hashes, git=record_git(),
                  source='VisDrone2019-DET', historical_csv='NOT_USED')
    save_json(run / 'manifest.json', status)
    try:
        os.environ.update(YOLO_AUTOINSTALL='false', YOLO_OFFLINE='true', PYTHONDONTWRITEBYTECODE='1')
        prepared_seconds = 0.
        if cfg['train_zip']:
            t = time.perf_counter()
            data_yaml, conversion = prepare(cfg['train_zip'], cfg['val_zip'], run / 'prepared_dataset')
            prepared_seconds = time.perf_counter() - t
            save_json(run / 'data_preparation.json', conversion)
            print(f'[CV261102] Converted {conversion["train"]["images"]} train and 548 val images', flush=True)
            print('[CV261102] Excluded zero-area target records: '
                  f'train={conversion["train"]["annotation_accounting"]["zero_area"]}; '
                  f'val={conversion["val"]["annotation_accounting"]["zero_area"]}', flush=True)
        else:
            data_yaml = cfg['data_yaml']
        training = run / 'training'
        training.mkdir()
        model = YOLO(str(cfg['weights']))
        # A renamed historical native-ten ckXX checkpoint is not a new pretrained 80-class base.
        if len(model.names) != 80:
            raise ValueError('Initial checkpoint must be pretrained COCO YOLOv8s with 80 classes.')
        observed = clock_epochs(model, training / 'epoch_times.csv')
        options = dict(data=str(data_yaml), epochs=cfg['epochs'], imgsz=cfg['train_imgsz'],
                       batch=cfg['batch'], workers=cfg['workers'], seed=cfg['seed'],
                       device='0', amp=True, deterministic=True, val=True,
                       save=True, plots=False, cache=False, patience=cfg['epochs'],
                       project=str(training), name='yolov8s', exist_ok=False)
        save_json(training / 'requested_args.json', options)
        t = time.perf_counter()
        model.train(**options)
        train_seconds = time.perf_counter() - t
        best = training / 'yolov8s' / 'weights' / 'best.pt'
        if not best.is_file():
            raise FileNotFoundError('Training did not produce best.pt')
        if observed != set(range(1, cfg['epochs']+1)):
            raise ValueError(f'Only {len(observed)}/{cfg["epochs"]} real epochs measured; no fabricated time values')
        print('[CV261102] Training complete. Starting real 548-image evaluation.', flush=True)
        save_json(training / 'checkpoint.json', dict(best_checkpoint_SHA256=digest(best),
            actual_epochs=len(observed), training_wall_seconds=train_seconds,
            ultralytics_results_SHA256=digest(training / 'yolov8s/results.csv') if (training / 'yolov8s/results.csv').is_file() else None,
            ultralytics_args_SHA256=digest(training / 'yolov8s/args.yaml') if (training / 'yolov8s/args.yaml').is_file() else None))

        from main.evaluation import evaluate
        transient = run / '_evaluation_work'
        evcfg = dict(dataset_zip=str(cfg['val_zip']), expected_images=548, smoke_images=5,
                     checkpoint=str(best), checkpoint_sha256=digest(best), output_root=str(transient),
                     imgsz=cfg['eval_imgsz'], batch=cfg['eval_batch'], device='0', allow_cpu_fallback=False)
        (run / 'evaluation_config.yaml').write_text(yaml.safe_dump(evcfg, sort_keys=False), encoding='utf-8')
        t = time.perf_counter()
        calculated = evaluate(str(run / 'evaluation_config.yaml'), full=True)
        validation = run / 'validation'
        calculated.rename(validation)
        if transient.is_dir():
            transient.rmdir()
        summary = read_json(validation / 'summary.json')
        summary.update(RunID=run_id, ExperimentID=TASK, checkpoint_id='new_best_yolov8s',
                       scope='NEW_548_IMAGE_DEVELOPMENT_VALIDATION_NOT_TEST_DEV')
        save_json(validation / 'summary.json', summary)
        eval_seconds = time.perf_counter() - t
        replay = verify_replay(validation)
        save_json(run / 'coco_regeneration.json', replay)
        result = export_reports(run, train_seconds, eval_seconds, cfg['epochs'])
        status.update(status='PASS', stage='COMPLETE', result=result,
                      source_checkpoint_SHA256=digest(best),
                      training_wall_seconds=train_seconds, prepare_wall_seconds=prepared_seconds,
                      evaluation_wall_seconds=eval_seconds,
                      coco_regeneration='PASS', total_wall_seconds=time.perf_counter()-started,
                      completed_at_UTC=dt.datetime.now(dt.timezone.utc).isoformat())
        save_json(run / 'manifest.json', status)
        print(f'[CV261102] PASS {run}', flush=True)
        print(f'[CV261102] AP50={result["AP50"]:.4%} | AP50:95={result["AP50_95"]:.4%}', flush=True)
        print('[CV261102] CSV ' + str(run / 'csv'), flush=True)
    except BaseException as error:
        status.update(status='PARTIAL', error=repr(error),
                      stopped_at_UTC=dt.datetime.now(dt.timezone.utc).isoformat(),
                      elapsed_seconds=time.perf_counter()-started)
        save_json(run / 'manifest.json', status)
        (run / 'failure_traceback.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__':
    main()
