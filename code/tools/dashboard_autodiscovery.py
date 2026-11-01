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

from dashboard_data import PUBLIC, DashboardStore, DataError, decode

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


def _generated_summary(result: dict, summary: dict, provenance: dict) -> dict:
    """Use saved measurements only; generation is not detector evaluation."""
    config = summary.get('configuration', {}) if isinstance(summary, dict) else {}
    if not isinstance(config, dict):
        config = {}
    audit = provenance.get('audit', {}) if isinstance(provenance, dict) else {}
    timing = summary.get('timing', {}) if isinstance(summary, dict) else {}
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
        'inference': (summary.get('metrics') or {}).get('model_inference', result.get('model_inference', 'NOT_RECORDED')),
        'AP': (summary.get('metrics') or {}).get('AP', result.get('AP_metrics', 'NOT_EVALUATED')),
    }


def load_snapshot(store, study: str, run_id: str) -> dict:
    """Read a SHA-verified published snapshot, also from the local dashboard."""
    view = _published_store(store).view(study, run_id)
    summary = view.read('summary.json', {}) or {}
    result = view.read('result.json', {}) or {}
    provenance = view.read('provenance.json', {}) or {}
    if not isinstance(result, dict) or not isinstance(summary, dict) or not isinstance(provenance, dict):
        raise DataError('Invalid published metadata.')
    metrics = {}
    for source in (result, summary):
        for key in METRICS:
            if key in source and isinstance(source[key], (float, int)):
                metrics[key] = source[key]
    generated = _generated_summary(result, summary, provenance)
    images = result.get('generated_images') or result.get('images_per_condition') or result.get('images')
    if images is None:
        images = (summary.get('dataset') or {}).get('images')
    return {'RunID': run_id, 'ExperimentID': study, 'metrics': metrics,
            'status': result.get('status', summary.get('status', 'UNKNOWN')),
            'tables': {}, 'scope': result.get('scope', ''), 'images': images,
            'epochs': result.get('trained_epochs'), 'timing': {}, 'gpu': '', 'started_at': '',
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
    top[0].metric('Status', str(data.get('status') or 'UNKNOWN'))
    top[1].metric('Images', str(data['images']) if data.get('images') is not None else '—')
    if gen.get('valid_configurations') is not None:
        top[2].metric('Valid FM_P configs', str(gen['valid_configurations']))
        top[3].metric('Generation time', f"{gen['elapsed_seconds']:.2f} s" if gen.get('elapsed_seconds') is not None else '—')
    else:
        top[2].metric('Epochs', str(data['epochs']) if data.get('epochs') is not None else '—')
        top[3].metric('Source', current['source'])
    st.caption(f"RunID: {selected} · source: {current['source']} · {data.get('scope', '')}")
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
    elif not gen.get('configuration_id'):
        st.info('This run has no CSV tables. Only saved summary metadata is displayed.')
