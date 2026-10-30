"""Publish one verified experiment run as an independent cloud snapshot.

Usage (from Git root):
  python -B code/tools/export_dashboard.py --run <local RunID directory>
  python -B code/tools/export_dashboard.py --run <VM2 replay directory> --archive <historical archive root>
"""
import argparse
import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path

PUBLIC = Path(__file__).resolve().parents[2]
RUNS_ROOT = PUBLIC / 'reports' / 'results'
sys.path.insert(0, str(PUBLIC / 'code'))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def accepted(run):
    record = read(run / 'run.json')
    result = read(run / 'result.json')
    if (record.get('status') != 'PASS' or result.get('status') != 'PASS' or
            record.get('RunID') != run.name or
            record.get('ExperimentID') not in
            ('vm1_flamapy_yolo26s_validation', 'vm2_degradation_gap_analysis')):
        raise ValueError('Run is not a matching, completed VM1/VM2 execution.')
    date = re.search(r'(\d{8})T(\d{2})(\d{2})(\d{2})(?:_|$)', run.name)
    if not date:
        raise ValueError('RunID lacks UTC timestamp.')
    d = date.group(1)
    record['started_at'] = f'{d[:4]}-{d[4:6]}-{d[6:]}T{date[2]}:{date[3]}:{date[4]}Z'
    return record, result


def vm1_bundle(run, record, result):
    report_file = run / 'summary.json'
    if not report_file.is_file():
        raise ValueError('VM1 run needs root summary.json (create it using vm1_report.py first).')
    report = read(report_file)
    if (report.get('RunID') != run.name or report.get('status') != 'PASS' or
            report.get('model', {}).get('checkpoint_id') != 'ck01'):
        raise ValueError('VM1 report identity/checkpoint mismatch.')
    summaries = result.get('summaries', [])
    if len(summaries) != 2 or {s['configuration']['name'] for s in summaries} != {'CLEAN', 'DEGRADED'}:
        raise ValueError('Only completed full CLEAN/DEGRADED paired runs may be published as VM1 results.')
    if result.get('images_per_condition') != 548 or result.get('conditions') != 2:
        raise ValueError('VM1 public results must reflect full 548-image paired evaluation.')
    by_name = {c['name']: c for c in report.get('conditions', [])}
    if set(by_name) != {'CLEAN', 'DEGRADED'}:
        raise ValueError('VM1 summary must report both conditions.')
    if report.get('dataset', {}).get('images_per_condition') != 548:
        raise ValueError('VM1 report image count mismatch.')
    actual_sha = report.get('model', {}).get('SHA256')
    for item in summaries:
        name = item['configuration']['name']
        for file, key in [('ground_truth.json', 'GT_SHA256'),
                          ('predictions_raw.jsonl', 'predictions_SHA256'),
                          ('predictions_coco.json', 'predictions_coco_SHA256'),
                          ('metrics.json', 'metrics_SHA256')]:
            path = run / name / file
            if sha(path) != item[key]:
                raise ValueError('VM1 evidence mismatch: ' + str(path))
        values = item['aggregate']
        observed = by_name[name]
        for key, ref in [('AP50', 'map50'), ('AP50_95', 'map50_95'), ('AR50_95', 'ar50_95')]:
            if abs(observed[key] - values[ref]) > 1e-12:
                raise ValueError('VM1 metrics mismatch for ' + name)
        if item['checkpoint_SHA256'] != actual_sha:
            raise ValueError('VM1 checkpoint SHA mismatch.')
    # Public metadata is a strict whitelist. No local paths or raw predictions.
    fields = ('schema_version', 'RunID', 'ExperimentID', 'mode', 'status', 'scope',
              'dataset', 'model', 'inference', 'flamapy', 'conditions',
              'paired_comparison', 'timing', 'environment', 'data_note')
    report = {key: report[key] for key in fields if key in report}
    report['model'] = dict(report['model'])
    training = dict(report['model'].get('training_configured') or {})
    training.pop('data', None)  # May contain private Windows dataset paths.
    report['model']['training_configured'] = training
    result_public = {'status': 'PASS', 'scope': result.get('scope'),
                     'images_per_condition': result['images_per_condition'],
                     'conditions': 2, 'checkpoint_id': 'ck01',
                     'total_seconds': result.get('total_seconds'),
                     'GPU_inference': 'COMPLETED'}
    bundle = {**record, 'files': {'run.json': record, 'result.json': result_public,
                                  'summary.json': report,
                                  'provenance.json': {'source_RunID': run.name,
                                                      'source_summary_SHA256': sha(report_file),
                                                      'source_result_SHA256': sha(run / 'result.json'),
                                                      'checkpoint_SHA256': actual_sha,
                                                      'dataset_ZIP_SHA256': report['dataset']['ZIP_SHA256']}}}
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=['condition', 'images', 'AP50', 'AP50_95', 'AR50_95', 'inference_call_seconds'])
    writer.writeheader()
    for c in report['conditions']:
        writer.writerow({k: c.get(v) for k, v in [('condition', 'name'), ('images', 'images'),
                                                  ('AP50', 'AP50'), ('AP50_95', 'AP50_95'),
                                                  ('AR50_95', 'AR50_95'),
                                                  ('inference_call_seconds', 'inference_call_seconds')]})
    return bundle, {'metrics.csv': buf.getvalue().encode('utf-8')}


def vm2_bundle(run, record, result, archive):
    if archive is None:
        raise ValueError('VM2 requires --archive to verify historical metric sources.')
    from main.gap_analysis import verify_measurement
    rows = read(run / 'measured_grids.json')
    if not rows or len({(r['cohort'], r['checkpoint'], r['configuration_id']) for r in rows}) != len(rows):
        raise ValueError('Missing or duplicated VM2 measurement rows.')
    archive = archive.resolve()
    for row in rows:
        metric = (archive / row['metric_source']).resolve()
        if not metric.is_relative_to(archive) or sha(metric) != row['metric_source_SHA256']:
            raise ValueError('VM2 underlying metric source SHA mismatch.')
        verify_measurement(row, read(metric))
    public_result = dict(result)
    bundle = {**record, 'files': {
        'run.json': record,
        'result.json': public_result,
        'measured_grids.json': rows,
        'interaction_replay.json': read(run / 'interaction_replay.json'),
        'provenance.json': {'source_RunID': run.name,
                            'source_result_SHA256': sha(run / 'result.json'),
                            'source_grid_SHA256': sha(run / 'measured_grids.json'),
                            'protocol': 'historical development replay; cohorts remain separate'}}}
    return bundle, {}


def publish(run, archive=None, output=RUNS_ROOT):
    run = Path(run).resolve()
    record, result = accepted(run)
    experiment = record['ExperimentID']
    if experiment == 'vm1_flamapy_yolo26s_validation':
        bundle, attachments = vm1_bundle(run, record, result)
    else:
        bundle, attachments = vm2_bundle(run, record, result, archive)
    raw = (json.dumps(bundle, indent=2, allow_nan=False, ensure_ascii=False) + '\n').encode('utf-8')
    if len(raw) > 2_000_000 or re.search(rb'[A-Za-z]:[\\/]', raw):
        raise ValueError('Public snapshot too large or contains a Windows absolute path.')
    run_folder = Path(output) / experiment / run.name
    payloads = {'snapshot.json': raw, **attachments}
    manifest = {'schema': 2, 'RunID': run.name, 'ExperimentID': experiment,
                'mode': record['mode'], 'status': 'PASS', 'started_at': record['started_at'],
                'scope': result.get('scope'), 'bundle': 'snapshot.json',
                'SHA256': hashlib.sha256(raw).hexdigest(),
                'files': {key: hashlib.sha256(value).hexdigest() for key, value in payloads.items()}}
    payloads['manifest.json'] = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode('utf-8')
    if run_folder.exists():
        if any(not (run_folder / name).is_file() or (run_folder / name).read_bytes() != value
               for name, value in payloads.items()):
            raise FileExistsError('Published RunID already exists with different content: ' + str(run_folder))
    else:
        run_folder.mkdir(parents=True)
        for name, value in payloads.items():
            (run_folder / name).write_bytes(value)
    print('Published:', run_folder)
    return run_folder


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Publish a single verified VM1/VM2 run.')
    p.add_argument('--run', required=True, type=Path, help='Exact local RunID directory')
    p.add_argument('--archive', type=Path, help='Required for historical VM2 source verification')
    args = p.parse_args()
    publish(args.run, args.archive)
