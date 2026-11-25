"""Verify CV261102 source evidence and regenerate COCO metrics without GPU inference.

The default audit reruns the COCO metrics from saved predictions and ground truth.
--integrity-only checks file hashes and cross-file consistency without replay;
that mode is explicitly NOT scientific metric verification.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from prepare_data import digest

EXPECTED_CSV = frozenset({
    'metrics_overall.csv', 'metrics_by_area.csv', 'metrics_by_class.csv',
    'evaluation_timing.csv', 'training_epoch_times.csv',
})


def load_json(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def same_csv_cell(actual: str, expected) -> bool:
    if expected is None:
        return actual == ''
    if isinstance(expected, bool):
        return actual == str(expected)
    if isinstance(expected, (float, int)):
        try:
            return math.isfinite(float(actual)) and math.isclose(
                float(actual), float(expected), rel_tol=1e-12, abs_tol=1e-12
            )
        except (ValueError, OverflowError):
            return False
    return actual == str(expected)


def verify_csv_rows(actual: list[dict], expected: list[dict], title: str) -> None:
    if len(actual) != len(expected):
        raise ValueError(f'{title}: row count differs from source evidence')
    for number, (observed, source) in enumerate(zip(actual, expected), 1):
        if set(observed) != set(source):
            raise ValueError(f'{title}: columns differ at row {number}')
        for name, value in source.items():
            if not same_csv_cell(observed[name], value):
                raise ValueError(f'{title}: mismatched {name} at row {number}')


def audit(run: Path, recompute: bool = True) -> dict:
    run = Path(run).expanduser().resolve()
    manifest = load_json(run / 'manifest.json')
    if (manifest.get('RunID') != run.name or manifest.get('ExperimentID') != 'cv261102'
            or manifest.get('status') != 'PASS' or manifest.get('stage') != 'COMPLETE'):
        raise ValueError('Run identity or completion status differs')
    result = manifest['result']
    if result.get('images') != 548 or result.get('trained_epochs', 0) < 1:
        raise ValueError('Expected complete 548-image validation and training')

    csv_dir = run / 'csv'
    if set(result['csv_SHA256']) != EXPECTED_CSV:
        raise ValueError('Expected five tracked CSV outputs')
    for filename, expected_sha in sorted(result['csv_SHA256'].items()):
        path = csv_dir / filename
        if not path.is_file() or digest(path) != expected_sha:
            raise ValueError('CSV SHA mismatch: ' + filename)

    validation = run / 'validation'
    required = (
        ('metrics.json', 'metrics_SHA256'),
        ('predictions_raw.json', 'predictions_raw_SHA256'),
        ('ground_truth.json', 'GT_SHA256'),
    )
    for filename, key in required:
        path = validation / filename
        if not path.is_file() or digest(path) != result[key]:
            raise ValueError('Evidence SHA mismatch: ' + filename)

    checkpoint = run / 'training/yolov8s/weights/best.pt'
    if not checkpoint.is_file() or digest(checkpoint) != manifest['source_checkpoint_SHA256']:
        raise ValueError('New trained checkpoint file/SHA mismatch')
    training_info = load_json(run / 'training/checkpoint.json')
    if training_info['best_checkpoint_SHA256'] != digest(checkpoint):
        raise ValueError('Checkpoint SHA differs from training metadata')
    if training_info['actual_epochs'] != result['trained_epochs']:
        raise ValueError('Training epoch count differs from checkpoint metadata')
    for key, relative in (
        ('ultralytics_results_SHA256', 'training/yolov8s/results.csv'),
        ('ultralytics_args_SHA256', 'training/yolov8s/args.yaml'),
    ):
        if training_info.get(key) is None:
            raise ValueError('Missing native Ultralytics training evidence: ' + relative)
        if digest(run / relative) != training_info[key]:
            raise ValueError('Training log hash differs: ' + relative)

    summary = load_json(validation / 'summary.json')
    gt = load_json(validation / 'ground_truth.json')
    if (summary.get('images') != 548 or summary.get('full_validation_status') != 'FULL_VALIDATION_PASS'
            or len(gt.get('images', [])) != 548):
        raise ValueError('Validation summary or GT does not describe 548 images')
    if summary.get('RunID') != run.name or summary.get('ExperimentID') != 'cv261102':
        raise ValueError('Evaluation summary/run linkage differs')
    metrics = load_json(validation / 'metrics.json')
    overall = [row for row in metrics['per_area'] if row['size'] == 'all']
    if len(overall) != 1 or not metrics.get('per_class'):
        raise ValueError('Expected one overall score and per-class scores')
    for field, key in [('AP50', 'map50'), ('AP50_95', 'map50_95'), ('AR50_95', 'ar50_95')]:
        if not math.isclose(float(result[field]), float(overall[0][key]), rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError('Metric differs from saved COCO report: ' + field)
    verify_csv_rows(read_csv(csv_dir / 'metrics_overall.csv'),
                    [dict(RunID=run.name, **overall[0])], 'metrics_overall.csv')
    verify_csv_rows(read_csv(csv_dir / 'metrics_by_area.csv'), metrics['per_area'], 'metrics_by_area.csv')
    verify_csv_rows(read_csv(csv_dir / 'metrics_by_class.csv'), metrics['per_class'], 'metrics_by_class.csv')

    original_epochs = read_csv(run / 'training/epoch_times.csv')
    copied_epochs = read_csv(csv_dir / 'training_epoch_times.csv')
    if original_epochs != copied_epochs:
        raise ValueError('Exported epoch timing differs from observed callback log')
    if (len(original_epochs) != result['trained_epochs'] or
            [int(row['epoch']) for row in original_epochs] != list(range(1, len(original_epochs) + 1))):
        raise ValueError('Expected contiguous live epoch observations')
    for row in original_epochs:
        train_seconds = float(row['train_seconds'])
        total_seconds = float(row['train_plus_validation_seconds'])
        if not all(map(math.isfinite, (train_seconds, total_seconds))) or not 0 < train_seconds <= total_seconds:
            raise ValueError('Invalid observed epoch duration')
        if row.get('clock') != 'time.perf_counter; live trainer callbacks' or not row.get('recorded_at_UTC'):
            raise ValueError('Missing epoch timing source annotation')

    timing = read_csv(csv_dir / 'evaluation_timing.csv')
    if len(timing) != 1 or timing[0].get('images') != '548':
        raise ValueError('Expected one 548-image evaluation timing record')
    for csv_field, manifest_field in (
        ('evaluation_wall_seconds', 'evaluation_wall_seconds'),
        ('train_stage_wall_seconds', 'training_wall_seconds'),
    ):
        if not same_csv_cell(timing[0][csv_field], manifest[manifest_field]):
            raise ValueError('Timing differs from run manifest: ' + csv_field)
    for csv_field, summary_field in (
        ('network_seconds', 'network_seconds'),
        ('coco_seconds', 'evaluation_seconds'),
        ('evaluation_pipeline_seconds', 'total_seconds'),
    ):
        if not same_csv_cell(timing[0][csv_field], summary[summary_field]):
            raise ValueError('Timing differs from evaluation: ' + csv_field)
    for row in timing:
        for name in ('evaluation_wall_seconds', 'train_stage_wall_seconds', 'network_seconds',
                     'coco_seconds', 'evaluation_pipeline_seconds'):
            if not math.isfinite(float(row[name])) or float(row[name]) < 0:
                raise ValueError('Non-finite or negative recorded time: ' + name)

    replay = load_json(run / 'coco_regeneration.json')
    if (replay.get('status') != 'PASS' or
            replay.get('GT_SHA256') != digest(validation / 'ground_truth.json') or
            replay.get('predictions_COCO_SHA256') != digest(validation / 'predictions_coco.json') or
            replay.get('metrics_SHA256') != digest(validation / 'metrics.json')):
        raise ValueError('Recorded COCO regeneration or evidence hashes mismatch')

    if recompute:
        from run import verify_replay
        new_replay = verify_replay(validation)
        if new_replay != replay:
            raise ValueError('Fresh COCO regeneration differs from stored replay')

    return {'status': 'PASS' if recompute else 'INTEGRITY_ONLY',
            'RunID': run.name, 'images': 548, 'epochs': len(original_epochs),
            'AP50': result['AP50'], 'AP50_95': result['AP50_95'],
            'coco_recomputed': recompute, 'file_hashes': 'PASS',
            'interpretation': 'Development validation, not independent test-dev'}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True, help='Full local CV261102 run directory')
    parser.add_argument('--integrity-only', action='store_true',
                        help='Skip COCO regeneration. Does not establish metrics correctness.')
    args = parser.parse_args(argv)
    print(json.dumps(audit(args.run, recompute=not args.integrity_only),
                     indent=2, ensure_ascii=True, allow_nan=False))


if __name__ == '__main__':
    main()
