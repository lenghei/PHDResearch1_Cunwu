"""Local VM44 paths and immutable evidence files."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PUBLIC = ROOT/'main'
sys.path.insert(0, str(PUBLIC/'code'))
RUNS = ROOT/'datasets/local_runs'/HERE.name
V43 = ROOT/'datasets/local_runs/vm43_external_pilot/vm43_external_pilot_execution_20261116T222929_46c7e0ba'
V40 = ROOT/'datasets/local_runs/vm40_uvl_gap_independent_validation/vm40_uvl_gap_independent_validation_audit_20261116T011556_955084fb'
V41 = ROOT/'datasets/local_runs/vm41_degradation_coverage/vm41_degradation_coverage_coverage_20261116T020508_dfa7879d'
UVL = HERE.parent/'vm36_supervisor_reproduction/feature_models'
SOURCES = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def source(path, expected=None, role='INPUT'):
    path = Path(path).resolve(strict=True)
    if str(path) not in SOURCES:
        SOURCES[str(path)] = {'path': str(path), 'SHA256': sha(path), 'bytes': path.stat().st_size, 'role': role}
    item = SOURCES[str(path)]
    if expected is not None and item['SHA256'] != expected:
        raise ValueError('Source SHA differs: '+str(path))
    return item


def read(path, expected=None, role='INPUT'):
    source(path, expected, role)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def csv_read(path, expected=None):
    source(path, expected)
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(RUNS.resolve()):
        raise ValueError('VM44 JSON output must stay local.')
    tmp = path.with_name(path.name+'.partial')
    with tmp.open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n'); f.flush(); os.fsync(f.fileno())
    if path.exists():
        raise FileExistsError(path)
    tmp.rename(path)


def csv_write(path, rows, fields=None):
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    if not fields:
        raise ValueError('An empty table needs explicit columns.')
    with Path(path).open('x', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)


def seal_entries(path):
    result = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        digest, name = line.split('  ', 1)
        file = Path(name)
        if not file.is_absolute():
            file = Path(path).parent/file
        result[str(file.resolve())] = digest
    return result


def verify_source(path, entries, role='ACCEPTED_INPUT'):
    path = Path(path).resolve(strict=True)
    if str(path) not in entries:
        raise ValueError('Input absent from accepted seal: '+str(path))
    return source(path, entries[str(path)], role)
