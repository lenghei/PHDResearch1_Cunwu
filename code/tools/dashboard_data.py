"""Read-only result store: one independent published snapshot per run."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import re
from pathlib import Path

PUBLIC = Path(__file__).resolve().parents[2]
METADATA = ('result.json', 'summary.json', 'timing.json', 'configuration.json',
            'provenance.json', 'run.json', 'paired_comparison.json')
DATA = ('measured_grids.json', 'interaction_replay.json')
CLOUD_METADATA = ('result.json', 'summary.json', 'timing.json', 'provenance.json',
                  'run.json', 'paired_comparison.json')
RUN_NAME = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,220}$')


class DataError(ValueError):
    pass


def decode(path, limit):
    try:
        if path.stat().st_size > limit:
            raise DataError('Result file exceeds size limit: ' + path.name)
        return json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError, UnicodeError) as exc:
        if isinstance(exc, DataError):
            raise
        raise DataError('Cannot read result: ' + path.name) from exc


class DashboardStore:
    def __init__(self, mode=None, public=None):
        self.mode = mode or os.environ.get('UAVGAP_DASHBOARD_MODE', 'cloud')
        if self.mode not in ('local', 'cloud'):
            raise DataError('UAVGAP_DASHBOARD_MODE must be local or cloud.')
        public = Path(public or PUBLIC).resolve()
        self.root = (public / 'reports/results') if self.mode == 'cloud' else Path(
            os.environ.get('UAVGAP_RUNS_ROOT', str(public.parent / 'datasets/local_runs'))).resolve()

    def child(self, parent, name):
        if not isinstance(name, str) or not RUN_NAME.fullmatch(name) or name in ('.', '..'):
            raise DataError('Invalid result name.')
        path = (parent / name).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise DataError('Result path escapes its data root.')
        return path

    def history(self, study):
        group = self.child(self.root, study)
        if not group.is_dir():
            return []
        rows = []
        for folder in sorted(group.iterdir()):
            if not folder.is_dir() or folder.is_symlink() or not folder.resolve().is_relative_to(group.resolve()):
                continue
            if not RUN_NAME.fullmatch(folder.name):
                continue
            if self.mode == 'cloud':
                meta = folder / 'manifest.json'
                if not meta.is_file():
                    continue
                try:
                    record = decode(meta, 32_768)
                    if (record.get('schema') != 2 or record.get('RunID') != folder.name or
                            record.get('ExperimentID') != study or
                            record.get('bundle') != 'snapshot.json' or
                            not re.fullmatch(r'[0-9a-f]{64}', record.get('SHA256', ''))):
                        raise DataError('Invalid published run manifest: ' + folder.name)
                    rows.append({key: record[key] for key in
                                 ('RunID', 'ExperimentID', 'mode', 'status', 'started_at')})
                except (KeyError, TypeError) as exc:
                    raise DataError('Incomplete published run manifest: ' + folder.name) from exc
                continue
            meta = next((folder / n for n in ('run.json', 'summary.json') if (folder / n).is_file()), None)
            if meta is None or not meta.resolve().is_relative_to(self.root):
                continue
            try:
                record = decode(meta, 1_000_000)
            except DataError:
                record = {'status': 'UNREADABLE'}
            match = re.search(r'(?:^|_)(\d{8}T\d{6})(?:_|$)', folder.name)
            try:
                date = datetime.datetime.strptime(match[1], '%Y%m%dT%H%M%S').replace(tzinfo=datetime.timezone.utc) if match else None
            except ValueError:
                date = None
            fallback = date is None
            date = date or datetime.datetime.fromtimestamp(meta.stat().st_mtime, datetime.timezone.utc)
            rows.append({'RunID': folder.name, 'ExperimentID': study,
                         'mode': record.get('mode') or ('full-validation' if str(record.get('scope', '')).startswith('FULL') else 'smoke'),
                         'status': record.get('status', record.get('Status', 'UNKNOWN')),
                         'started_at': date.isoformat(),
                         'time_source': 'metadata_mtime' if fallback else 'RunID_UTC'})
        return sorted(rows, key=lambda r: (r['started_at'], r['RunID']), reverse=True)

    def view(self, study, run_id):
        if not any(r['RunID'] == run_id for r in self.history(study)):
            raise DataError('Unknown saved run.')
        folder = self.child(self.child(self.root, study), run_id)
        if self.mode == 'cloud':
            record = decode(self.child(folder, 'manifest.json'), 32_768)
            path = self.child(folder, record['bundle'])
            if not path.is_file():
                raise DataError('Published snapshot is missing: ' + folder.name)
            if path.stat().st_size > 2_000_000:
                raise DataError('Published snapshot exceeds size limit: ' + folder.name)
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['SHA256']:
                raise DataError('Published snapshot checksum mismatch: ' + folder.name)
            bundle = decode(path, 2_000_000)
            if bundle.get('RunID') != run_id or bundle.get('ExperimentID') != study:
                raise DataError('Published snapshot identity mismatch: ' + folder.name)
            return RunView(self, None, bundle, f'reports/results/{study}/{run_id}')
        return RunView(self, folder, None, str(folder))


class RunView:
    def __init__(self, store, folder, bundle, display_path):
        self.store, self.folder, self.bundle, self.display_path = store, folder, bundle, display_path

    def read(self, name, default=None):
        if name not in METADATA + DATA:
            raise DataError('This file is not available in the viewer.')
        if self.bundle is not None:
            if name not in CLOUD_METADATA + DATA:
                raise DataError('Private configuration is unavailable in cloud mode.')
            return self.bundle.get('files', {}).get(name, default)
        path = self.store.child(self.folder, name)
        return decode(path, 10_000_000) if path.is_file() else default

    def exists(self, name):
        return self.read(name) is not None

    def metadata_names(self):
        return [n for n in (CLOUD_METADATA if self.bundle is not None else METADATA) if self.exists(n)]

    def bytes(self, name):
        return json.dumps(self.read(name), indent=2, allow_nan=False).encode('utf-8')

    def images(self):
        if self.bundle is not None:
            return []
        folder = self.store.child(self.folder, 'preview')
        return [p for p in sorted(folder.glob('image_*.jpg')) if p.resolve().is_relative_to(self.store.root)]

    def comparison_sets(self):
        """Audited four-panel previews; source run and cloud thumbnails remain separate."""
        names = ('original.jpg', 'degraded.jpg', 'clean_predictions.jpg', 'degraded_predictions.jpg')
        if self.bundle is not None:
            study = self.bundle['ExperimentID']
            run_id = self.bundle['RunID']
            root = self.store.child(self.store.child(self.store.root, study), run_id)
        else:
            root = self.folder
            run_id = root.name
        previews = self.store.child(root, 'image_comparison')
        index_file = self.store.child(previews, 'manifest.json')
        if not index_file.is_file():
            return []
        index = decode(index_file, 60_000)
        if index.get('schema') != 1 or index.get('RunID') != run_id:
            raise DataError('Invalid comparison manifest.')
        entries = index.get('images', [])
        if not isinstance(entries, list) or len(entries) > 5:
            raise DataError('Invalid comparison image count.')
        public_files = {}
        if self.bundle is not None:
            public_record = decode(self.store.child(root, 'manifest.json'), 32_768)
            public_files = public_record.get('files') or {}
            if public_files.get('image_comparison/manifest.json') != hashlib.sha256(index_file.read_bytes()).hexdigest():
                raise DataError('Published comparison manifest checksum mismatch.')
        result = []
        for entry in entries:
            image_id = entry.get('image_id')
            if type(image_id) is not int or not 1 <= image_id <= 548:
                raise DataError('Invalid comparison image ID.')
            folder = self.store.child(previews, f'{image_id:04d}')
            pictures = {}
            for name in names:
                file = self.store.child(folder, name)
                if not file.is_file() or file.stat().st_size > 2_000_000:
                    raise DataError('Comparison image missing or exceeds size limit: ' + name)
                data = file.read_bytes()
                checksum = hashlib.sha256(data).hexdigest()
                if entry.get('files', {}).get(name) != checksum:
                    raise DataError('Comparison image checksum mismatch: ' + name)
                if self.bundle is not None:
                    expected = f'image_comparison/{image_id:04d}/{name}'
                    if public_files.get(expected) != checksum:
                        raise DataError('Published image not listed in source manifest: ' + expected)
                pictures[name] = data
            result.append({'image_id': image_id, 'source_file': entry.get('source_file'),
                           'recipe': entry.get('recipe', {}), 'pictures': pictures,
                           'clean_boxes': entry.get('displayed_clean_boxes'),
                           'degraded_boxes': entry.get('displayed_degraded_boxes'),
                           'confidence': index.get('display_confidence', .25)})
        return result

    def log_text(self):
        if self.bundle is not None:
            return None
        path = self.store.child(self.folder, 'execution.log')
        if not path.is_file():
            return None
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 18000))
            return stream.read().decode('utf-8', errors='replace')
