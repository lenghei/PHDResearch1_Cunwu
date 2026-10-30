"""Export explicitly selected, accepted runs for the read-only cloud viewer."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

PUBLIC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PUBLIC / 'code'))
from main.gap_analysis import verify_measurement


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def accepted(run, experiment):
    record = read(run / 'run.json')
    if record['status'] != 'PASS' or record['ExperimentID'] != experiment or record['RunID'] != run.name:
        raise ValueError('Select a completed, identity-matched run: ' + run.name)
    if read(run / 'result.json')['status'] != 'PASS':
        raise ValueError('Run result is not PASS.')
    date = re.search(r'(\d{8})T(\d{2})(\d{2})(\d{2})_', run.name)
    if date is None:
        raise ValueError('Missing recorded Run UTC timestamp.')
    day = date[1]
    record['started_at'] = f'{day[:4]}-{day[4:6]}-{day[6:]}T{date[2]}:{date[3]}:{date[4]}Z'
    return record


def bundle(run, experiment):
    record = accepted(run, experiment)
    return {**record, 'files': {'run.json': record}}, record


def export(vm1, vm2, archive, output):
    first, first_record = bundle(vm1, 'vm1_flamapy_yolo26s_validation')
    gate = read(vm1 / 'checkpoint_gate.json')
    data = read(vm1 / 'dataset_gate.json')
    bridge = read(vm1 / 'bridge.json')
    if any(v['status'] != 'PASS' for v in [gate, data, bridge]):
        raise ValueError('VM1 gate is not PASS.')
    summary = read(vm1 / 'result.json')
    first['files']['result.json'] = summary
    first['files']['provenance.json'] = {
        'source_RunID': vm1.name, 'artifact_SHA256': {n: sha(vm1 / n) for n in
            ['run.json', 'result.json', 'checkpoint_gate.json', 'dataset_gate.json', 'bridge.json', 'source_SHA256.json']},
        'source_code_SHA256': read(vm1 / 'source_SHA256.json'),
        'dataset': {'name': 'VisDrone2019-DET-val', 'images': data['images'], 'ZIP_SHA256': data['ZIP_SHA256'],
                    'scope': data['scope']},
        'model': {k: gate[k] for k in ['checkpoint_id', 'SHA256', 'architecture', 'class_order']},
        'UVL_SHA256': bridge['UVL_SHA256'], 'protocol': 'native10; configured pr01; CPU checks only',
        'detector_metrics': 'NOT_EVALUATED; no YOLO26s AP exported from CPU acceptance'}
    second, second_record = bundle(vm2, 'vm2_degradation_gap_analysis')
    rows = read(vm2 / 'measured_grids.json')
    if len(rows) != 297 or len({(r['cohort'], r['checkpoint'], r['configuration_id']) for r in rows}) != 297:
        raise ValueError('Expected separate 243-point grid and 54-point JPEG cohorts.')
    if any(r['images'] != 300 for r in rows):
        raise ValueError('Image-count mismatch.')
    models = {}
    for row in rows:
        metric = (archive / row['metric_source']).resolve()
        if not metric.is_relative_to(archive.resolve()) or sha(metric) != row['metric_source_SHA256']:
            raise ValueError('Underlying metric SHA mismatch.')
        measured = read(metric)
        verify_measurement(row, measured)
        if row['measurement_kind'] == 'grid':
            model_sha = measured['checkpoint_SHA256']
            if row['checkpoint'] in models and models[row['checkpoint']]['SHA256'] != model_sha:
                raise ValueError('Mixed checkpoints within a model ID.')
            models[row['checkpoint']] = {'SHA256': model_sha, 'task': 'native10', 'protocol': 'pr01'}
    for name in ['result.json', 'measured_grids.json', 'interaction_replay.json']:
        second['files'][name] = read(vm2 / name)
    second['files']['provenance.json'] = {
        'source_RunID': vm2.name,
        'artifact_SHA256': {n: sha(vm2 / n) for n in ['run.json', 'result.json', 'measured_grids.json',
                                                   'interaction_replay.json', 'provenance.json', 'source_SHA256.json']},
        'source_code_SHA256': read(vm2 / 'source_SHA256.json'),
        'sources': read(vm2 / 'provenance.json'), 'models': models,
        'protocol': 'pr01; native10; FP32; one-to-many; ignore-union; COCO AP; maxDet=3000',
        'scope': 'DEVELOPMENT_VALIDATION; fixed 300-image ds04; dv02 grid and dv03 JPEG separate',
        'independence': 'VisDrone val used for historical checkpoint selection; not independent test data',
        'counts': {'grid': 243, 'jpeg_subgrid': 54, 'new_GPU_experiments': 0}}
    manifest = {'schema': 1, 'scope': 'Verified summary snapshots; local complete evidence is not published', 'runs': []}
    output.mkdir(parents=True, exist_ok=True)
    payloads = {}
    for name, value, record in [('vm1_summary.json', first, first_record), ('vm2_summary.json', second, second_record)]:
        raw = json.dumps(value, indent=2, allow_nan=False).encode('utf-8') + b'\n'
        if len(raw) >= 1_000_000 or re.search(rb'\b[A-Za-z]:[\\/]', raw):
            raise ValueError('Export contains private paths or exceeds size limit.')
        payloads[name] = raw
        manifest['runs'].append({**record, 'bundle': name, 'SHA256': hashlib.sha256(raw).hexdigest()})
    payloads['dashboard_manifest.json'] = json.dumps(manifest, indent=2).encode('utf-8') + b'\n'
    for name, raw in payloads.items():
        path = output / name
        if path.exists() and path.read_bytes() != raw:
            raise FileExistsError('Existing snapshot differs; review before replacing: ' + name)
    for name, raw in payloads.items():
        if not (output / name).exists():
            (output / name).write_bytes(raw)
        print(name, len(raw), hashlib.sha256(raw).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vm1-run', type=Path, required=True)
    parser.add_argument('--vm2-run', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args()
    export(args.vm1_run.resolve(), args.vm2_run.resolve(), args.archive.resolve(), PUBLIC / 'reports/results')
