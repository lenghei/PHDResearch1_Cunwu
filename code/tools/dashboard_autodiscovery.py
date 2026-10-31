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

from dashboard_data import DataError, decode

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
DISALLOWED_COLUMNS = re.compile(r'password|secret|token|api.?key|private|(?:^|_)path|filepath|command|config|weights|user(name)?|email', re.I)
PRIVATE_VALUES = re.compile(r'(^[A-Za-z]:[\\/]|/home/|/Users/|\\Users\\|(?:^|\s)\\\\)', re.I)
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


def public_runs(store, study: str) -> list[dict]:
    study_dir = safe_child(store.root, study, store.root)
    if not study_dir.is_dir():
        return []
    saved = {}
    for folder in study_dir.iterdir():
        if not folder.is_dir() or folder.is_symlink():
            continue
        if not RUN_NAME.fullmatch(folder.name):
            continue
        item = _public_record(folder, study)
        if item is not None:
            saved[folder.name] = item
    if store.mode == 'local':
        # Historical local runs use run.json/summary.json instead of manifest.json.
        for item in store.history(study):
            saved.setdefault(item['RunID'], {**item, 'format': 'local'})
    return sorted(saved.values(), key=lambda r: (r['started_at'], r['RunID']), reverse=True)


def discover_studies(store) -> dict[str, str]:
    """Discover studies at each Streamlit rerun; no registry updates are needed."""
    if not store.root.is_dir():
        return {}
    folders = [p for p in store.root.iterdir() if p.is_dir() and not p.is_symlink()
               and RUN_NAME.fullmatch(p.name)]
    selected = []
    for path in folders:
        try:
            if public_runs(store, path.name):
                selected.append(path.name)
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
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) > 40:
            raise DataError('Invalid published CSV header: ' + path.name)
        for row in reader:
            if len(rows) >= MAX_ROWS or None in row:
                raise DataError('Invalid or oversized published CSV: ' + path.name)
            rows.append(row)
    return rows


def _filtered(row: dict) -> dict:
    """Display measurement columns, not machine-specific paths or private strings."""
    safe = {}
    for name, raw in row.items():
        if not name or DISALLOWED_COLUMNS.search(name):
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
        elif re.fullmatch(r'[a-zA-Z0-9 _.\-+:]{1,120}', value):
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


def load_snapshot(store, study: str, run_id: str) -> dict:
    view = store.view(study, run_id)
    summary = view.read('summary.json', {})
    result = view.read('result.json', {})
    value = {}
    for source in (result, summary):
        if not isinstance(source, dict):
            continue
        for metric in METRICS:
            if metric in source and isinstance(source[metric], (float, int)):
                value[metric] = source[metric]
    return {'RunID': run_id, 'ExperimentID': study, 'metrics': value,
            'status': result.get('status', 'UNKNOWN') if isinstance(result, dict) else 'UNKNOWN',
            'tables': {}, 'scope': result.get('scope', '') if isinstance(result, dict) else '',
            'images': result.get('images_per_condition') if isinstance(result, dict) else None,
            'epochs': None, 'timing': {}, 'gpu': '', 'started_at': ''}


def render_discovered_study(store, study: str) -> None:
    """Generic Streamlit display for newly published experiments."""
    import pandas as pd
    import streamlit as st

    st.title(study.replace('_', ' '))
    st.caption('Saved experimental measurements · read-only · no inference or training')
    runs = public_runs(store, study)
    if not runs:
        st.info('No published runs found.')
        return
    labels = {row['RunID']: row for row in runs}
    selected = st.selectbox('Run · newest first', list(labels),
                            format_func=lambda name: f"{labels[name]['started_at'][:19]} · {name} · {labels[name]['status']}")
    current = labels[selected]
    try:
        if current['format'] == 'snapshot':
            data = load_snapshot(store, study, selected)
        elif current['format'] == 'experiment':
            data = load_experiment(store, study, selected)
        else:
            view = store.view(study, selected)
            value = view.read('result.json', {}) or view.read('summary.json', {})
            value = value if isinstance(value, dict) else {}
            data = {'RunID': selected, 'status': current['status'], 'metrics': {},
                    'images': value.get('images'), 'epochs': None, 'scope': value.get('scope', ''),
                    'timing': {}, 'tables': {}, 'gpu': '', 'started_at': current['started_at']}
    except (DataError, OSError, UnicodeError, ValueError) as exc:
        st.error('Published result could not be verified: ' + str(exc))
        return

    metric_cols = st.columns(4)
    metric_cols[0].metric('Status', data.get('status') or 'UNKNOWN')
    metric_cols[1].metric('Images', str(data['images']) if data.get('images') is not None else '—')
    metric_cols[2].metric('Epochs', str(data['epochs']) if data.get('epochs') is not None else '—')
    metric_cols[3].metric('Run format', current['format'])
    metrics = data.get('metrics') or {}
    if metrics:
        cols = st.columns(len(metrics))
        for col, (key, number) in zip(cols, metrics.items()):
            col.metric(key.replace('_', ':'), f'{100 * number:.2f}%')
    st.caption(f"RunID: {selected} · {data.get('scope', '')}")
    if data.get('gpu'):
        st.caption('GPU: ' + data['gpu'])

    duration = data.get('timing') or {}
    if duration:
        st.subheader('Recorded timing')
        st.dataframe(pd.DataFrame([{'Stage': key.replace('_', ' '), 'Seconds': value}
                                   for key, value in duration.items()]), hide_index=True, width='stretch')

    tables = data.get('tables') or {}
    area = tables.get('metrics_by_area.csv') or []
    if area:
        st.subheader('Accuracy by object size')
        frame = pd.DataFrame(area)
        if {'size', 'map50', 'map50_95'}.issubset(frame.columns):
            frame = frame[frame['size'].isin(['tiny', 'small', 'medium', 'large'])]
            if not frame.empty:
                plotted = frame.set_index('size')[['map50', 'map50_95']].astype(float) * 100
                st.bar_chart(plotted, y_label='Average precision (%)')
    epochs = tables.get('training_epoch_times.csv') or []
    if epochs:
        st.subheader('Training time by epoch')
        frame = pd.DataFrame(epochs)
        if {'epoch', 'train_plus_validation_seconds'}.issubset(frame.columns):
            frame = frame.sort_values('epoch').set_index('epoch')
            st.line_chart(frame[['train_plus_validation_seconds']], y_label='Seconds')

    if tables:
        st.subheader('Published CSV tables')
        choice = st.selectbox('Table', sorted(tables))
        frame = pd.DataFrame(tables[choice])
        st.dataframe(frame, hide_index=True, width='stretch')
        st.download_button('Download displayed table', frame.to_csv(index=False).encode('utf-8'),
                           file_name=choice, mime='text/csv')
    else:
        st.info('This run has no published CSV tables. Only saved summary metadata is displayed.')
