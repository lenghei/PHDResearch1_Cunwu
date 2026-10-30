"""Bounded, read-only data access for the dashboard."""
import datetime
import hashlib
import json
import os
import re
from pathlib import Path

PUBLIC = Path(__file__).resolve().parents[2]
METADATA = ('result.json', 'summary.json', 'timing.json', 'configuration.json', 'provenance.json', 'run.json')
DATA = ('measured_grids.json', 'interaction_replay.json')
CLOUD_METADATA = ('result.json', 'summary.json', 'timing.json', 'provenance.json', 'run.json')


class DataError(ValueError):
    pass


def decode(path, limit):
    if path.stat().st_size > limit:
        raise DataError('Result file exceeds the display size limit: ' + path.name)
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (ValueError, OSError) as exc:
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
        if not isinstance(name, str) or not name or Path(name).name != name or '/' in name or '\\' in name:
            raise DataError('Invalid result name.')
        path = (parent / name).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise DataError('Result path escapes its data root.')
        return path

    def manifest(self):
        path = self.child(self.root, 'dashboard_manifest.json')
        if not path.is_file():
            return {'runs': []}
        value = decode(path, 100_000)
        if value.get('schema') != 1 or not isinstance(value.get('runs'), list):
            raise DataError('Unsupported published result manifest.')
        return value

    def history(self, study):
        if self.mode == 'cloud':
            return sorted([r for r in self.manifest()['runs'] if r['ExperimentID'] == study],
                          key=lambda r: r['started_at'], reverse=True)
        group = self.child(self.root, study)
        if not group.is_dir():
            return []
        rows = []
        for folder in group.iterdir():
            if not folder.is_dir() or not folder.resolve().is_relative_to(group):
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
                         'started_at': date.isoformat(), 'time_source': 'metadata_mtime' if fallback else 'RunID_UTC'})
        return sorted(rows, key=lambda r: (r['started_at'], r['RunID']), reverse=True)

    def view(self, study, run_id):
        records = self.history(study)
        record = next((r for r in records if r['RunID'] == run_id), None)
        if record is None:
            raise DataError('Unknown saved run.')
        if self.mode == 'cloud':
            path = self.child(self.root, record['bundle'])
            if not path.is_file():
                raise DataError('Published summary is missing: ' + path.name)
            if hashlib.sha256(path.read_bytes()).hexdigest() != record['SHA256']:
                raise DataError('Published summary checksum mismatch: ' + path.name)
            bundle = decode(path, 1_000_000)
            if bundle.get('RunID') != run_id or bundle.get('ExperimentID') != study:
                raise DataError('Published summary identity mismatch.')
            return RunView(self, None, bundle, path.name)
        folder = self.child(self.child(self.root, study), run_id)
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

    def log_text(self):
        if self.bundle is not None:
            return None
        path = self.store.child(self.folder, 'execution.log')
        if not path.is_file():
            return None
        with path.open('rb') as stream:
            stream.seek(max(0, path.stat().st_size - 18000))
            return stream.read().decode('utf-8', errors='replace')
