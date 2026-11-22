"""Automatic discovery of published UAV-GAP studies and read-only result summaries.

Legacy VM1/VM2 pages are left to dashboard.py. This module renders newly
published run formats without changing the dashboard whenever a study is added.
"""
from __future__ import annotations

import csv
import hashlib
import io
import math
import re
from pathlib import Path

from dashboard_data import PUBLIC, DashboardStore, DataError, decode, read_published_snapshot

KNOWN_STUDIES = {
    'vm1_flamapy_yolo26s_validation': 'VM1 · FlamaPy–YOLO26s validation',
    'vm2_degradation_gap_analysis': 'VM2 · Degradation interactions',
    'vm03_gpu_pilot': 'VM03 · Historical YOLOv8s baseline',
}
LEGACY_STUDIES = frozenset(KNOWN_STUDIES)
RUN_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,220}$')
HASH = re.compile(r'^[0-9a-f]{64}$')
CSV_LIMIT = 1_000_000
MAX_CSV_FILES = 30
MAX_ROWS = 5_000
ASSET_LIMIT = 2_000_000
PUBLIC_FIGURES = {
    'gap_tradeoffs.png': 'Gap transitions and trade-offs',
    'Figure_C_class_degradation_heatmap.png': 'AP50:95 change by class and degradation',
}
CLASS_TABLE = 'per_class_all_conditions.csv'
PAIRED_METRICS = {'ap': 'AP50:95', 'tiny': 'Tiny AP50:95', 'ms': 'Cached latency'}
DISALLOWED_COLUMNS = re.compile(r'password|secret|token|api.?key|credential|authorization|private|(?:^|_)path|filepath|command|config|weights|user(name)?|email', re.I)
PRIVATE_VALUES = re.compile(
    r'(^[A-Za-z]:[\\/]|/home/|/Users/|\\Users\\|(?:^|\s)\\\\|gh[pousr]_|github_pat_)', re.I)
PUBLIC_COLUMN = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_. %():+\-]{0,119}')
METRICS = ('AP50', 'AP50_95', 'AR50_95')


def safe_child(parent: Path, name: str, root: Path) -> Path:
    """Restrict all reads to valid names and paths below the configured result root."""
    if not isinstance(name, str) or not RUN_NAME.fullmatch(name) or name in ('.', '..'):
        raise DataError('Invalid published directory name.')
    path = parent / name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise DataError('Published directory escapes result root.')
    return path


def read_manifest(path: Path) -> dict:
    value = decode(path, 65_536)
    if not isinstance(value, dict):
        raise DataError('Invalid run manifest.')
    return value


def _time(record: dict, folder: Path) -> str:
    value = record.get('started_at') or record.get('created_at_UTC')
    if isinstance(value, str) and re.match(r'^\d{4}-\d\d-\d\dT', value):
        return value
    match = re.search(r'(\d{8})T(\d{6})', folder.name)
    if match:
        date, clock = match.groups()
        return f'{date[:4]}-{date[4:6]}-{date[6:]}T{clock[:2]}:{clock[2:4]}:{clock[4:]}+00:00'
    return 'Unknown'


def _public_record(folder: Path, study: str) -> dict | None:
    manifest_path = folder / 'manifest.json'
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return None
    try:
        manifest = read_manifest(manifest_path)
    except DataError:
        return None
    if manifest.get('RunID') != folder.name or manifest.get('ExperimentID') != study:
        return None
    schema = manifest.get('schema')
    if schema == 2:
        if (manifest.get('bundle') != 'snapshot.json' or
                not isinstance(manifest.get('SHA256'), str) or
                not HASH.fullmatch(manifest['SHA256'])):
            return None
        kind = 'snapshot'
    elif 'result' in manifest and isinstance(manifest['result'], dict):
        kind = 'experiment'
    else:
        return None
    return {
        'RunID': folder.name,
        'ExperimentID': study,
        'mode': manifest.get('mode') or ('training-evaluation' if kind == 'experiment' else 'snapshot'),
        'status': str(manifest.get('status', 'UNKNOWN')),
        'started_at': _time(manifest, folder),
        'format': kind,
    }


def _published_store(store):
    """Published snapshots are available to both local and cloud viewers."""
    return store if store.mode == 'cloud' else DashboardStore(mode='cloud', public=PUBLIC)


def _local_record(folder: Path, study: str) -> dict | None:
    """Discover native experiment result files without requiring run.json."""
    meta = folder / 'result.json'
    if not meta.is_file() or meta.is_symlink():
        return None
    try:
        result = decode(meta, 1_000_000)
    except DataError:
        return None
    if not isinstance(result, dict) or result.get('RunID') != folder.name or result.get('ExperimentID') != study:
        return None
    selection_file = folder / 'selection.json'
    if selection_file.is_file() and not selection_file.is_symlink():
        try:
            selection = decode(selection_file, 256_000)
        except DataError:
            return None
        if (not isinstance(selection, dict) or selection.get('RunID') != folder.name or
                selection.get('ExperimentID') != study):
            return None
        kind = 'local_fmp'
        stamp = selection.get('started_at_UTC') or result.get('finished_at_UTC')
        mode = 'FM_P_GENERATION'
    else:
        kind = 'local_json'
        stamp = result.get('started_at_UTC') or result.get('finished_at_UTC')
        mode = result.get('mode') or 'local experiment'
    return {'RunID': folder.name, 'ExperimentID': study, 'mode': mode,
            'status': str(result.get('status', 'UNKNOWN')),
            'started_at': stamp if isinstance(stamp, str) and stamp.startswith('20') else _time(result, folder),
            'format': kind, 'source': 'local'}


def public_runs(store, study: str) -> list[dict]:
    """Merge approved public snapshots and local runs; local evidence wins."""
    published = _published_store(store)
    saved: dict[str, dict] = {}
    folder = safe_child(published.root, study, published.root)
    if folder.is_dir():
        for run in folder.iterdir():
            if not run.is_dir() or run.is_symlink() or not RUN_NAME.fullmatch(run.name):
                continue
            record = _public_record(run, study)
            if record:
                record['source'] = 'published'
                if store.mode == 'local':
                    record['format'] = 'published_' + record['format']
                saved[run.name] = record
    if store.mode == 'local':
        local_folder = safe_child(store.root, study, store.root)
        if local_folder.is_dir():
            for run in local_folder.iterdir():
                if not run.is_dir() or run.is_symlink() or not RUN_NAME.fullmatch(run.name):
                    continue
                record = _local_record(run, study)
                if record:
                    saved[run.name] = record
            # Historical VM and future run.json/summary.json formats.
            for record in store.history(study):
                if record['RunID'] not in saved or saved[record['RunID']].get('source') == 'published':
                    saved[record['RunID']] = {**record, 'format': 'local', 'source': 'local'}
    return sorted(saved.values(), key=lambda r: (r['started_at'], r['RunID']), reverse=True)


def discover_studies(store) -> dict[str, str]:
    """Discover experiment directories from the two allowed source roots."""
    roots = [_published_store(store).root]
    if store.mode == 'local':
        roots.insert(0, store.root)
    studies = set()
    for root in roots:
        if not root.is_dir():
            continue
        for path in root.iterdir():
            if path.is_dir() and not path.is_symlink() and RUN_NAME.fullmatch(path.name):
                studies.add(path.name)
    selected = []
    for study in studies:
        try:
            if public_runs(store, study):
                selected.append(study)
        except (DataError, OSError):
            continue
    selected.sort(key=lambda name: (0 if name in KNOWN_STUDIES else 1, name))
    labels = {}
    for name in selected:
        if name in KNOWN_STUDIES:
            label = KNOWN_STUDIES[name]
        else:
            head, sep, tail = name.partition('_')
            label = head.upper() + (' · ' + tail.replace('_', ' ') if sep else '')
        labels[label] = name
    return labels


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(256 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()



def _csv_sha_matches(path: Path, checksum: str) -> bool:
    """Accept exact bytes or Git's sole LF-to-CRLF text normalization.

    Original Windows-generated CSVs can be committed with LF-only newlines,
    while the saved experiment manifest hashes their original CRLF bytes.
    No metric values, encoding, delimiters, or other content may change.
    """
    if _sha(path) == checksum:
        return True
    if path.stat().st_size > CSV_LIMIT:
        return False
    content = path.read_bytes()
    if b'\r' in content or b'\n' not in content:
        return False
    windows = content.replace(b'\n', b'\r\n')
    return hashlib.sha256(windows).hexdigest() == checksum


def _csv_rows(path: Path) -> list[dict]:
    if path.stat().st_size > CSV_LIMIT:
        raise DataError('Published CSV exceeds size limit: ' + path.name)
    rows = []
    try:
        with path.open('r', encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if (not reader.fieldnames or len(reader.fieldnames) > 40 or
                    any(not name.strip() for name in reader.fieldnames) or
                    len(set(reader.fieldnames)) != len(reader.fieldnames)):
                raise DataError('Invalid published CSV header: ' + path.name)
            for row in reader:
                if len(rows) >= MAX_ROWS or None in row or None in row.values():
                    raise DataError('Invalid or oversized published CSV: ' + path.name)
                rows.append(row)
    except (csv.Error, UnicodeError) as exc:
        raise DataError('Cannot parse published CSV: ' + path.name) from exc
    return rows


def _filtered(row: dict) -> dict:
    """Display measurement columns, not machine-specific paths or private strings."""
    safe = {}
    for name, raw in row.items():
        if not name or not PUBLIC_COLUMN.fullmatch(name) or DISALLOWED_COLUMNS.search(name):
            continue
        if raw is None:
            continue
        value = str(raw).strip()
        if len(value) > 180 or PRIVATE_VALUES.search(value):
            continue
        if re.fullmatch(r'-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', value):
            number = float(value)
            if math.isfinite(number):
                safe[name] = int(number) if number.is_integer() and abs(number) < 1e15 else number
        elif re.fullmatch(r'[a-zA-Z0-9_][a-zA-Z0-9 _.\-+:]{0,119}', value):
            safe[name] = value
    return safe


def _float(value, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise DataError('Invalid metric: ' + label) from exc
    if not math.isfinite(result):
        raise DataError('Non-finite metric: ' + label)
    return result


def load_experiment(store, study: str, run_id: str) -> dict:
    """Load and verify a raw experiment manifest and all declared public CSV files."""
    group = safe_child(store.root, study, store.root)
    folder = safe_child(group, run_id, store.root)
    manifest = read_manifest(folder / 'manifest.json')
    if manifest.get('ExperimentID') != study or manifest.get('RunID') != run_id:
        raise DataError('Experiment identity mismatch.')
    result = manifest.get('result')
    if not isinstance(result, dict):
        raise DataError('Published experiment result is missing.')
    csv_dir = folder / 'csv'
    expected = result.get('csv_SHA256') or {}
    if not isinstance(expected, dict):
        raise DataError('Invalid published CSV checksums.')
    if len(expected) > MAX_CSV_FILES:
        raise DataError('Too many published CSV files.')
    tables = {}
    for filename, checksum in expected.items():
        if (not isinstance(filename, str) or not filename.endswith('.csv') or
                not RUN_NAME.fullmatch(filename) or
                not isinstance(checksum, str) or not HASH.fullmatch(checksum)):
            raise DataError('Invalid CSV manifest entry.')
        path = safe_child(csv_dir, filename, store.root)
        if not path.is_file() or not _csv_sha_matches(path, checksum):
            raise DataError('Missing or modified published CSV: ' + filename)
        tables[filename] = [_filtered(row) for row in _csv_rows(path)]
    if csv_dir.is_dir() and not csv_dir.is_symlink():
        # Future experiments can add CSV tables without altering dashboard code.
        for path in sorted(csv_dir.glob('*.csv')):
            if (path.name not in tables and len(tables) < MAX_CSV_FILES and
                    not path.is_symlink() and RUN_NAME.fullmatch(path.name)):
                tables[path.name] = [_filtered(row) for row in _csv_rows(path)]
    overall = next((r for r in tables.get('metrics_overall.csv', [])
                    if r.get('size') == 'all'), None)
    value = {}
    for metric, csv_field in [('AP50', 'map50'), ('AP50_95', 'map50_95'), ('AR50_95', 'ar50_95')]:
        recorded = result.get(metric)
        if recorded is not None:
            value[metric] = _float(recorded, metric)
        if overall is not None and csv_field in overall:
            measured = _float(overall[csv_field], csv_field)
            if metric in value and abs(measured - value[metric]) > 1e-9:
                raise DataError('Manifest and CSV metrics disagree: ' + metric)
            value[metric] = measured
    output = {
        'RunID': run_id,
        'ExperimentID': study,
        'status': str(manifest.get('status', 'UNKNOWN')),
        'scope': str(manifest.get('scope', '')),
        'started_at': _time(manifest, folder),
        'images': result.get('images'),
        'epochs': result.get('trained_epochs'),
        'metrics': value,
        'timing': {k: manifest.get(k) for k in
                   ('prepare_wall_seconds', 'training_wall_seconds', 'evaluation_wall_seconds',
                    'total_wall_seconds') if isinstance(manifest.get(k), (float, int))},
        'tables': tables,
    }
    system = manifest.get('system') or {}
    output['gpu'] = system.get('gpu', '') if isinstance(system, dict) else ''
    # Neither local configuration paths nor raw manifest content is exposed.
    return output


def _generated_summary(result: dict, summary: dict, provenance: dict) -> dict:
    """Use saved measurements only; generation is not detector evaluation."""
    config = summary.get('configuration', {}) if isinstance(summary, dict) else {}
    if not isinstance(config, dict):
        config = {}
    audit = provenance.get('audit', {}) if isinstance(provenance, dict) else {}
    timing = summary.get('timing', {}) if isinstance(summary, dict) else {}
    metrics = summary.get('metrics')
    metrics = metrics if isinstance(metrics, dict) else {}
    duration = result.get('elapsed_seconds')
    if not isinstance(duration, (int, float)) and isinstance(timing, dict):
        duration = timing.get('wall_seconds')
    return {
        'configuration_id': result.get('configuration_id') or config.get('id'),
        'valid_configurations': result.get('FlamaPy_valid_configurations'),
        'features': config.get('features') or [],
        'parameters': config.get('numeric_parameters') or {},
        'effect_order': config.get('effect_order') or [],
        'elapsed_seconds': duration if isinstance(duration, (int, float)) else None,
        'audit': {k: audit.get(k) for k in ('status', 'files_checked', 'images_regenerated') if k in audit},
        'inference': metrics.get('model_inference', result.get('model_inference', 'NOT_RECORDED')),
        'AP': metrics.get('AP', result.get('AP_metrics', 'NOT_EVALUATED')),
    }


def _snapshot_files(folder: Path, study: str, run_id: str) -> tuple[dict, dict]:
    manifest = read_manifest(safe_child(folder, 'manifest.json', folder))
    if (manifest.get('schema') != 2 or manifest.get('ExperimentID') != study or
            manifest.get('RunID') != run_id or manifest.get('bundle') != 'snapshot.json'):
        raise DataError('Published snapshot manifest identity mismatch.')
    expected = manifest.get('files', {})
    checksum = manifest.get('SHA256')
    if (not isinstance(expected, dict) or not isinstance(checksum, str) or not HASH.fullmatch(checksum)
            or expected.get('snapshot.json', checksum) != checksum):
        raise DataError('Invalid published snapshot checksums.')
    path = safe_child(folder, 'snapshot.json', folder)
    bundle = read_published_snapshot(path, checksum)
    if bundle.get('ExperimentID') != study or bundle.get('RunID') != run_id:
        raise DataError('Published snapshot identity mismatch.')
    files = bundle.get('files')
    if not isinstance(files, dict):
        raise DataError('Invalid published snapshot contents.')
    return manifest, files


def _class_rows(rows: list[dict]) -> list[dict]:
    allowed = {'condition', 'category_name', 'category_id', 'gt_count', 'tiny_gt_count',
               'images_with_GT', 'images_with_tiny_GT', 'contribution_to_macro_AP50_95_delta_pp'}
    measures = ('AP50', 'AP50_95', 'AR50_95', 'Tiny_AP50', 'Tiny_AP50_95', 'Tiny_AR50_95')
    for metric in measures:
        allowed.update((f'CONTROL_{metric}_pct', f'TARGETED_{metric}_pct', f'delta_{metric}_pp',
                        f'delta_{metric}_CI95_low_pp', f'delta_{metric}_CI95_high_pp'))
    required_numbers = ('CONTROL_AP50_95_pct', 'TARGETED_AP50_95_pct', 'delta_AP50_95_pp',
                        'CONTROL_Tiny_AP50_95_pct', 'TARGETED_Tiny_AP50_95_pct', 'delta_Tiny_AP50_95_pp')
    keys = set()
    cleaned = []
    for row in rows:
        condition, category, cid = (row.get(k) for k in ('condition', 'category_name', 'category_id'))
        if (not isinstance(condition, str) or not isinstance(category, str) or
                type(cid) is not int or cid < 1 or (condition, cid) in keys or
                any(k not in row for k in required_numbers)):
            raise DataError('Invalid per-class CSV fields or duplicate class rows.')
        keys.add((condition, cid))
        entry = {k: v for k, v in row.items() if k in allowed}
        for name, value in entry.items():
            if name in ('condition', 'category_name'):
                continue
            if value == 'NA' and name.endswith(('_pct', '_pp')):
                continue
            if (name.endswith('_pct') and (type(value) not in (int, float) or not 0 <= value <= 100) or
                    name.endswith('_pp') and (type(value) not in (int, float) or not -100 <= value <= 100) or
                    name.endswith(('_count', '_GT')) and (type(value) is not int or value < 0)):
                raise DataError('Invalid per-class CSV metric or count: ' + name)
        cleaned.append(entry)
    return cleaned


def _snapshot_assets(folder: Path, expected: dict) -> tuple[dict, dict]:
    csv_names = [name for name in expected if name.lower().endswith('.csv')]
    if len(csv_names) > MAX_CSV_FILES:
        raise DataError('Too many published CSV files.')
    tables, figures = {}, {}
    for name in csv_names + [name for name in PUBLIC_FIGURES if name in expected]:
        checksum = expected[name]
        if not isinstance(checksum, str) or not HASH.fullmatch(checksum):
            raise DataError('Invalid published asset checksum.')
        path = safe_child(folder, name, folder)
        if not path.is_file():
            raise DataError('Missing published asset: ' + name)
        if name in PUBLIC_FIGURES:
            if path.stat().st_size > ASSET_LIMIT:
                raise DataError('Published figure exceeds size limit: ' + name)
            data = path.read_bytes()
            if (hashlib.sha256(data).hexdigest() != checksum or
                    not data.startswith(b'\x89PNG\r\n\x1a\n')):
                raise DataError('Published figure checksum or format mismatch: ' + name)
            figures[name] = data
            continue
        if path.stat().st_size > CSV_LIMIT:
            raise DataError('Published CSV exceeds size limit: ' + name)
        if not _csv_sha_matches(path, checksum):
            raise DataError('Published CSV checksum mismatch: ' + name)
        rows = [_filtered(row) for row in _csv_rows(path)]
        tables[name] = _class_rows(rows) if name == CLASS_TABLE else [row for row in rows if row]
    return tables, figures


def _summary_tables(summary: dict) -> dict:
    def fields(value, prefix='', depth=0):
        for key, item in value.items():
            if not PUBLIC_COLUMN.fullmatch(key) or DISALLOWED_COLUMNS.search(key):
                continue
            name = f'{prefix}.{key}' if prefix else key
            if isinstance(item, dict) and depth < 3:
                yield from fields(item, name, depth + 1)
            elif isinstance(item, list) and len(item) <= 40:
                for index, member in enumerate(item, 1):
                    if type(member) in (str, int, float):
                        yield f'{name}.{index}', member
            elif type(item) in (str, int, float):
                yield name, item

    return {key: rows for key, value in summary.items()
            if key != 'paired_conditions' and PUBLIC_COLUMN.fullmatch(key)
            and isinstance(value, list) and 0 < len(value) <= MAX_ROWS
            and all(isinstance(row, dict) for row in value)
            and (rows := [clean for row in value if (clean := _filtered(dict(fields(row))))])}


def _paired_results(summary: dict) -> dict:
    source = summary.get('paired_conditions')
    if source is None:
        return {}
    if not isinstance(source, list) or not source or not all(isinstance(row, dict) for row in source):
        raise DataError('Invalid paired condition results.')
    arms = source[0].get('arms')
    if not isinstance(arms, dict) or len(arms) != 2 or 'CONTROL' not in arms:
        raise DataError('Invalid paired training arms.')
    control, targeted = 'CONTROL', next(arm for arm in arms if arm != 'CONTROL')
    if not RUN_NAME.fullmatch(targeted):
        raise DataError('Invalid paired training arm name.')
    conditions = []
    for row in source:
        label = _filtered({'label': row.get('label'), 'condition': row.get('condition')})
        saved_arms = row.get('arms')
        saved_delta = row.get('delta')
        if (len(label) != 2 or not isinstance(saved_arms, dict) or set(saved_arms) != {control, targeted}
                or not all(isinstance(arm, dict) for arm in saved_arms.values())
                or not isinstance(saved_delta, dict)
                or not all(isinstance(saved_delta.get(metric), dict) for metric in PAIRED_METRICS)):
            raise DataError('Invalid paired condition or arm.')
        arms = {arm: {key: _float(saved_arms[arm].get(key), key) for key in PAIRED_METRICS}
                for arm in (control, targeted)}
        for arm, values in arms.items():
            values['state'] = saved_arms[arm].get('state')
            if (values['state'] not in ('PASS', 'FAIL', 'UNCERTAIN') or values['ms'] < 0 or
                    not all(0 <= values[key] <= 100 for key in ('ap', 'tiny'))):
                raise DataError('Invalid paired metric or Gap state.')
        delta = {key: _float(saved_delta[key].get('value'), key) for key in PAIRED_METRICS}
        if any(abs(arms[targeted][key] - arms[control][key] - delta[key]) > 1e-9 for key in delta):
            raise DataError('Paired snapshot differences do not match arm metrics.')
        conditions.append({**label, 'arms': arms, 'delta': delta})
    if len({row['condition'] for row in conditions}) != len(conditions):
        raise DataError('Duplicate paired conditions.')
    closures = summary.get('confirmed_gap_closures')
    confirmed = sum(row['arms'][control]['state'] == 'FAIL' and row['arms'][targeted]['state'] == 'PASS'
                    for row in conditions)
    if type(closures) is not int or closures != confirmed:
        raise DataError('Invalid confirmed Gap closure count.')
    return {'arm_names': (control, targeted), 'conditions': conditions, 'closures': closures}


def load_snapshot(store, study: str, run_id: str) -> dict:
    """Read verified public artifacts, including when the dashboard is in local mode."""
    root = _published_store(store).root
    folder = safe_child(safe_child(root, study, root), run_id, root)
    manifest, files = _snapshot_files(folder, study, run_id)
    summary = files.get('summary.json', {})
    result = files.get('result.json', {})
    provenance = files.get('provenance.json', {})
    if not all(isinstance(value, dict) for value in (result, summary, provenance)):
        raise DataError('Invalid published metadata.')
    for value in (result, summary):
        if value.get('ExperimentID', study) != study or value.get('RunID', run_id) != run_id:
            raise DataError('Published summary identity mismatch.')
    tables, figures = _snapshot_assets(folder, manifest.get('files', {}))
    paired = _paired_results(summary)
    metrics = {key: value[key] for value in (result, summary) for key in METRICS
               if type(value.get(key)) in (float, int)}
    generated = _generated_summary(result, summary, provenance)
    dataset = summary.get('dataset') if isinstance(summary.get('dataset'), dict) else {}
    images_per_condition = summary.get('images_per_condition', result.get('images_per_condition',
                                                                         dataset.get('images_per_condition')))
    images = images_per_condition or result.get('generated_images') or result.get('images') or dataset.get('images')
    epochs_per_arm = summary.get('epochs_per_strategy')
    epochs = epochs_per_arm or result.get('trained_epochs') or summary.get('epochs')
    return {'RunID': run_id, 'ExperimentID': study, 'metrics': metrics,
            'status': result.get('status', summary.get('status', manifest.get('status', 'UNKNOWN'))),
            'tables': tables, 'summary_tables': _summary_tables(summary), 'figures': figures, 'paired': paired,
            'units': _filtered({key: summary.get(key) for key in ('ap_units', 'delta_ap_units', 'did_units')}),
            'scope': result.get('scope') or summary.get('scope') or manifest.get('scope', ''),
            'images': images, 'epochs': epochs, 'timing': {}, 'gpu': '', 'started_at': _time(manifest, folder),
            'images_per_condition': images_per_condition, 'epochs_per_strategy': epochs_per_arm,
            'training_images': summary.get('training_images'), 'cells': summary.get('cells'),
            'image_evaluations': summary.get('image_evaluations'), 'seed': summary.get('seed'),
            'generation': generated, 'preview_root': None}


def load_local_run(store, study: str, run_id: str) -> dict:
    """Read a native private run with strict RunID and experiment identity."""
    if store.mode != 'local':
        raise DataError('Private runs are only available in local mode.')
    group = safe_child(store.root, study, store.root)
    folder = safe_child(group, run_id, store.root)
    result = decode(folder / 'result.json', 1_000_000)
    if not isinstance(result, dict) or result.get('RunID') != run_id or result.get('ExperimentID') != study:
        raise DataError('Local run identity mismatch.')
    selection_file = folder / 'selection.json'
    if selection_file.is_file():
        selection = decode(selection_file, 256_000)
        if (not isinstance(selection, dict) or selection.get('RunID') != run_id or
                selection.get('ExperimentID') != study):
            raise DataError('Local selection identity mismatch.')
    else:
        selection = {}
    checksum = result.get('image_manifest_SHA256')
    if checksum is not None:
        manifest = folder / 'image_manifest.jsonl'
        if not isinstance(checksum, str) or not HASH.fullmatch(checksum) or not manifest.is_file():
            raise DataError('Local image manifest missing or invalid.')
        if manifest.is_symlink() or _sha(manifest) != checksum:
            raise DataError('Local image manifest checksum mismatch.')
    config = selection.get('selection', {})
    summary = {'configuration': {
        'id': config.get('configuration_id'),
        'features': config.get('selected_leaves', []),
        'numeric_parameters': selection.get('parameters_from_UVL', {}),
        'effect_order': selection.get('effect_order', [])},
        'metrics': {'model_inference': result.get('model_inference'), 'AP': result.get('AP_metrics')}}
    generation = _generated_summary(result, summary, {})
    return {'RunID': run_id, 'ExperimentID': study, 'status': result['status'],
            'scope': result.get('scope', ''), 'images': result.get('generated_images', result.get('images')),
            'epochs': result.get('trained_epochs'), 'metrics': {}, 'timing': {}, 'tables': {},
            'gpu': '', 'started_at': selection.get('started_at_UTC', ''),
            'generation': generation, 'preview_root': folder}


def local_preview_pairs(folder: Path, count: int = 3) -> list[tuple[bytes, bytes]]:
    """Read at most three safe local comparison pairs without serving raw images."""
    pairs = []
    root = folder.resolve()
    for index in range(1, min(count, 3) + 1):
        preview = root / 'preview'
        files = [preview / f'{prefix}_{index:04d}.jpg' for prefix in ('original', 'degraded')]
        if not all(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(root)
                   and 0 < p.stat().st_size <= 8_000_000 for p in files):
            continue
        pairs.append(tuple(p.read_bytes() for p in files))
    return pairs


def _render_paired_results(st, pd, paired: dict) -> None:
    import plotly.express as px

    control, targeted = paired['arm_names']
    rows = paired['conditions']
    st.subheader('Paired development results')
    st.metric('Confirmed Gap closures', paired['closures'])
    motion = next((row for row in rows if row['condition'] == 'Motion_k21'), None)
    if motion:
        a, b = motion['arms'][control], motion['arms'][targeted]
        st.caption(f"Motion: {a['ap']:.3f}% to {b['ap']:.3f}% ({motion['delta']['ap']:+.3f} pp) "
                   f"| Tiny: {a['tiny']:.3f}% to {b['tiny']:.3f}% "
                   f"| Gap: {a['state']} to {b['state']}")
    tabs = st.tabs(['AP50:95', 'Tiny AP50:95', 'Cached latency'])
    for tab, (key, title) in zip(tabs, PAIRED_METRICS.items()):
        unit = 'ms' if key == 'ms' else '%'
        frame = pd.DataFrame([{'Condition': row['label'], 'Training arm': arm, title: row['arms'][arm][key]}
                              for row in rows for arm in (control, targeted)])
        fig = px.bar(frame, x='Condition', y=title, color='Training arm', barmode='group',
                     labels={title: f'{title} ({unit})'}, color_discrete_sequence=['#4463c7', '#087f8c'])
        fig.update_layout(margin=dict(l=16, r=16, t=18, b=16), height=330, legend_title_text='')
        fig.update_traces(hovertemplate=f'%{{x}}<br>%{{y:.3f}} {unit}<extra>%{{fullData.name}}</extra>')
        tab.plotly_chart(fig, width='stretch', alt=f'{title} by condition and training arm')

    table = []
    for row in rows:
        a, b = row['arms'][control], row['arms'][targeted]
        values = {'Condition': row['condition'], 'Gap transition': f"{a['state']} to {b['state']}"}
        for key, title in PAIRED_METRICS.items():
            unit, diff_unit = ('ms', 'ms') if key == 'ms' else ('%', 'pp')
            values[f'{control} {title} ({unit})'] = a[key]
            values[f'{targeted} {title} ({unit})'] = b[key]
            values[f'{title} change ({diff_unit})'] = row['delta'][key]
        table.append(values)
    st.dataframe(table, hide_index=True, width='stretch', alt='Paired metrics and Gap transitions',
                 column_config={name: st.column_config.NumberColumn(format='%.3f')
                                for name in table[0] if name not in ('Condition', 'Gap transition')})


def _render_class_results(st, pd, rows: list[dict]) -> None:
    st.subheader('Per-class results')
    conditions = list(dict.fromkeys(row['condition'] for row in rows))
    classes = list(dict.fromkeys(row['category_name'] for row in sorted(rows, key=lambda r: r['category_id'])))
    motion = [row for row in rows if row['condition'] == 'Motion_k21']
    gains = sorted((row for row in motion if type(row['delta_AP50_95_pp']) in (int, float)),
                   key=lambda row: row['delta_AP50_95_pp'], reverse=True)
    net = sum(row['delta_AP50_95_pp'] for row in gains)
    if len(gains) >= 3 and net > 0:
        leaders = ', '.join(f"{row['category_name']} {row['delta_AP50_95_pp']:+.3f} pp" for row in gains[:3])
        share = 100 * sum(row['delta_AP50_95_pp'] for row in gains[:3]) / net
        st.caption(f'Motion: {leaders}. Together they explain {share:.1f}% of the net gain in '
                   'class-averaged AP50:95.')

    left, middle, right = st.columns(3)
    options = ['All conditions', *conditions]
    condition = left.selectbox('Degradation condition', options,
                               index=options.index('Motion_k21') if motion else 0)
    selected = middle.multiselect('Classes', classes, placeholder='All classes')
    sort_options = {'AP50:95 change (highest first)': 'delta_AP50_95_pp',
                    'Tiny AP50:95 change (highest first)': 'delta_Tiny_AP50_95_pp',
                    'Class ID': 'category_id'}
    sort = sort_options[right.selectbox('Sort by', list(sort_options))]
    frame = pd.DataFrame(rows)
    display = frame if condition == 'All conditions' else frame[frame['condition'] == condition]
    if selected:
        display = display[display['category_name'].isin(selected)]
    display = display.sort_values(sort, ascending=sort == 'category_id',
                                  key=lambda column: pd.to_numeric(column, errors='coerce'))
    columns = {'condition': 'Condition', 'category_name': 'Class',
               'CONTROL_AP50_95_pct': 'Control AP50:95 (%)', 'TARGETED_AP50_95_pct': 'Targeted AP50:95 (%)',
               'delta_AP50_95_pp': 'AP50:95 change (pp)', 'CONTROL_Tiny_AP50_95_pct': 'Control Tiny AP (%)',
               'TARGETED_Tiny_AP50_95_pct': 'Targeted Tiny AP (%)', 'delta_Tiny_AP50_95_pp': 'Tiny AP change (pp)',
               'gt_count': 'GT count', 'tiny_gt_count': 'Tiny GT count'}
    if display.empty:
        st.info('No class rows match the selected filters.')
    else:
        decimals = {key: st.column_config.NumberColumn(label, format='%.3f') for key, label in columns.items()
                    if key.endswith(('_pct', '_pp'))}
        st.dataframe(display, column_order=list(columns), column_config={**columns, **decimals},
                     hide_index=True, width='stretch', alt='Per-class AP and Tiny AP comparisons')
    st.download_button('Download all class results', frame.to_csv(index=False).encode('utf-8'),
                       file_name=CLASS_TABLE, mime='text/csv')


def render_discovered_study(store, study: str) -> None:
    """Shared local/cloud UI for all experiments discovered from result folders."""
    import pandas as pd
    import streamlit as st

    st.title(study.replace('_', ' '))
    st.caption('Saved evidence · read-only · no inference or training')
    runs = public_runs(store, study)
    if not runs:
        st.info('No saved runs found.')
        return
    labels = {row['RunID']: row for row in runs}
    selected = st.selectbox('Run · newest first', list(labels),
                            format_func=lambda name: f"{labels[name]['started_at'][:19]} · {name} · {labels[name]['status']} · {labels[name]['source']}")
    current = labels[selected]
    try:
        if current['format'] in ('snapshot', 'published_snapshot'):
            data = load_snapshot(store, study, selected)
        elif current['format'] in ('experiment', 'published_experiment'):
            data = load_experiment(_published_store(store), study, selected)
        elif current['format'] in ('local_fmp', 'local_json'):
            data = load_local_run(store, study, selected)
        else:
            view = store.view(study, selected)
            value = view.read('result.json', {}) or view.read('summary.json', {})
            value = value if isinstance(value, dict) else {}
            data = {'RunID': selected, 'status': current['status'], 'metrics': {},
                    'images': value.get('images'), 'epochs': None, 'scope': value.get('scope', ''),
                    'timing': {}, 'tables': {}, 'gpu': '', 'started_at': current['started_at']}
    except (DataError, OSError, UnicodeError, ValueError, KeyError) as exc:
        st.error('Result could not be verified: ' + str(exc))
        return

    gen = data.get('generation') or {}
    top = st.columns(4)
    status = str(data.get('status') or 'UNKNOWN')
    matched = status == 'MATCHED_PILOT_COMPLETED_WITH_LIMITATIONS'
    top[0].metric('Status', 'MATCHED PILOT' if matched else status)
    images = data.get('images')
    top[1].metric('Images', f'{images} per condition' if data.get('images_per_condition') else
                  str(images) if images is not None else '—')
    if gen.get('valid_configurations') is not None:
        top[2].metric('Valid FM_P configs', str(gen['valid_configurations']))
        top[3].metric('Generation time', f"{gen['elapsed_seconds']:.2f} s" if gen.get('elapsed_seconds') is not None else '—')
    else:
        epochs = data.get('epochs')
        top[2].metric('Epochs', f'{epochs} per arm' if data.get('epochs_per_strategy') else
                      str(epochs) if epochs is not None else '—')
        top[3].metric('Source', current['source'])
    st.caption(f"RunID: {selected} · source: {current['source']} · {data.get('scope', '')}")
    if matched:
        st.caption(status)
    paired = data.get('paired') or {}
    details = {'Total evaluations': data.get('image_evaluations'), 'Training images': data.get('training_images'),
               'Evaluation cells': data.get('cells'), 'Training arms': len(paired['arm_names']) if paired else None,
               'Seed': data.get('seed')}
    details = {name: value for name, value in details.items() if type(value) is int}
    if details:
        for col, (name, value) in zip(st.columns(len(details)), details.items()):
            col.metric(name, str(value))
    if paired:
        _render_paired_results(st, pd, paired)
    metrics = data.get('metrics') or {}
    if metrics:
        cols = st.columns(len(metrics))
        for col, (key, number) in zip(cols, metrics.items()):
            col.metric(key.replace('_', ':'), f'{100 * number:.2f}%')
    if data.get('gpu'):
        st.caption('GPU: ' + data['gpu'])

    if gen.get('configuration_id'):
        st.subheader('FM_P degradation configuration')
        st.code(str(gen['configuration_id']))
        st.write('Selected features: ' + ', '.join(str(f) for f in gen.get('features', [])))
        parameters = gen.get('parameters') or {}
        if parameters:
            st.dataframe(pd.DataFrame([{'Feature': feature, 'Parameters': str(values)}
                                       for feature, values in parameters.items()]),
                         hide_index=True, width='stretch')
        if gen.get('effect_order'):
            st.caption('Effect order: ' + ' → '.join(gen['effect_order']))
        st.info('Image generation only · detector inference: ' + str(gen.get('inference')) +
                ' · AP: ' + str(gen.get('AP')))
        audit = gen.get('audit') or {}
        if audit:
            st.subheader('Saved reproducibility audit')
            st.dataframe(pd.DataFrame([audit]), hide_index=True, width='stretch')
        if data.get('preview_root'):
            st.subheader('Local original / degraded previews')
            pairs = local_preview_pairs(data['preview_root'])
            if pairs:
                for index, (original, degraded) in enumerate(pairs, 1):
                    left, right = st.columns(2)
                    left.image(original, caption=f'Original image {index}', use_container_width=True)
                    right.image(degraded, caption=f'Degraded image {index}', use_container_width=True)
            else:
                st.caption('This local run has no saved preview images.')
        elif current['source'] == 'published':
            st.caption('The published snapshot contains validated metrics and parameters, not original image previews.')

    duration = data.get('timing') or {}
    if duration:
        st.subheader('Recorded timing')
        st.dataframe(pd.DataFrame([{'Stage': key.replace('_', ' '), 'Seconds': value}
                                   for key, value in duration.items()]), hide_index=True, width='stretch')
    tables = data.get('tables') or {}
    class_rows = tables.get(CLASS_TABLE)
    if class_rows:
        _render_class_results(st, pd, class_rows)

    summary_tables = data.get('summary_tables') or {}
    if summary_tables:
        label = st.selectbox('Saved result table', list(summary_tables),
                             format_func=lambda name: name.replace('_', ' ').capitalize())
        units = data['units']
        if units:
            st.caption('Saved units: ' + ', '.join(f"{k.replace('_', ' ')} = {v}" for k, v in units.items()))
        st.dataframe(pd.DataFrame(summary_tables[label]), hide_index=True, width='stretch',
                     alt='Measurements from the published snapshot')

    for name, image in (data.get('figures') or {}).items():
        st.image(image, caption=PUBLIC_FIGURES[name], alt=PUBLIC_FIGURES[name], width='stretch')

    tables = {name: rows for name, rows in tables.items() if name != CLASS_TABLE and rows}
    area = tables.get('metrics_by_area.csv') or []
    if area:
        st.subheader('Accuracy by object size')
        frame = pd.DataFrame(area)
        if {'size', 'map50', 'map50_95'}.issubset(frame.columns):
            frame = frame[frame['size'].isin(['tiny', 'small', 'medium', 'large'])]
            if not frame.empty:
                st.bar_chart(frame.set_index('size')[['map50', 'map50_95']].astype(float) * 100,
                             y_label='Average precision (%)')
    epochs = tables.get('training_epoch_times.csv') or []
    if epochs:
        st.subheader('Training time by epoch')
        frame = pd.DataFrame(epochs)
        if {'epoch', 'train_plus_validation_seconds'}.issubset(frame.columns):
            st.line_chart(frame.sort_values('epoch').set_index('epoch')[['train_plus_validation_seconds']],
                          y_label='Seconds')
    if tables:
        st.subheader('Published CSV tables')
        choice = st.selectbox('Table', sorted(tables))
        frame = pd.DataFrame(tables[choice])
        st.dataframe(frame, hide_index=True, width='stretch')
        st.download_button('Download displayed table', frame.to_csv(index=False).encode('utf-8'),
                           file_name=choice, mime='text/csv')
    elif not (gen.get('configuration_id') or class_rows or summary_tables or paired):
        st.info('This run has no CSV tables. Only saved summary metadata is displayed.')
