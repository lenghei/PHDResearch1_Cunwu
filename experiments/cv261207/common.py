"""Paths, immutable run IDs and evidence utilities."""
from datetime import datetime, timezone
from hashlib import sha256 as digest
from importlib.util import spec_from_file_location, module_from_spec
from pathlib import Path
from uuid import uuid4
import json
import os
import sys

HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
PUBLIC = EXPERIMENTS.parent
ROOT = PUBLIC.parent
RUNS = ROOT / 'datasets/local_runs/cv261207'
sys.dont_write_bytecode = True
sys.path.insert(0, str(PUBLIC / 'code'))
from main.config import sha256, save

FMP = EXPERIMENTS / 'cv261109/fm_p.uvl'
FMD = EXPERIMENTS / 'cv261116/fm_d.uvl'
ARCHIVES = ROOT / 'datasets/VisDrone'
VAL = ARCHIVES / 'VisDrone2019-DET-val.zip'
TEST = ARCHIVES / 'VisDrone2019-DET-test-dev.zip'
BASELINE = ROOT / 'datasets/local_runs/cv261102/cv261102_full_20261031T000013_d0be8587'
STATIC = ROOT / 'datasets/local_runs/cv261116/cv261116_static_20261101T223010_016ba75c'
PROFILE = ROOT / 'datasets/local_runs/cv261116/cv261116_gpu_20261101T225631_5bc4e6ce'
CHECKPOINT = ROOT / 'models/checkpoints/yolov8s_cft640/best.pt'
SCRIPT = HERE / 'run.py'
FLAMA = ROOT / '.venv_flamapy' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def source(path):
    return {'name': path.name, 'path': str(path), 'sha256': sha256(path), 'bytes': path.stat().st_size}


def new_run(mode):
    name = f'cv261207_{mode}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid4().hex[:8]}'
    out = RUNS / name
    out.mkdir(parents=True)
    return out


def resolve_run(name):
    if not name.startswith('cv261207_') or Path(name).name != name:
        raise ValueError('Expected a CV261207 RunID.')
    path = (RUNS / name).resolve(strict=True)
    if path.parent != RUNS.resolve():
        raise ValueError('Run outside the local result store.')
    return path


def import_file(name, path):
    spec = spec_from_file_location(name, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprint():
    sources = [*HERE.glob('*.py'), FMP, FMD,
               EXPERIMENTS / 'cv261109/generator.py', EXPERIMENTS / 'cv261109/model.py',
               EXPERIMENTS / 'cv261123/sampling.py', EXPERIMENTS / 'cv261116/model.py']
    sources.extend((PUBLIC / 'code/main') / f'{n}.py' for n in
                   ('dataset', 'labels', 'predictor', 'statistics'))
    return {str(p.relative_to(PUBLIC)).replace(os.sep, '/'): sha256(p) for p in sorted(sources)}


def seal(out):
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name != 'SHA256SUMS.txt')
    with (out / 'SHA256SUMS.txt').open('x', encoding='utf-8') as stream:
        for path in files:
            stream.write(f'{sha256(path)}  {path.name}\n')


def verify_hashes(out, required):
    checks = {}
    for line in (out / 'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        value, name = line.split('  ', 1)
        if not name or Path(name).name != name or name in checks:
            raise ValueError('Invalid checksum manifest.')
        if sha256(out / name) != value:
            raise ValueError(f'Evidence hash mismatch: {name}')
        checks[name] = value
    if not set(required).issubset(checks):
        raise ValueError('Missing checksummed evidence.')
    return checks


def file_id(selected, prefix):
    data = json.dumps(sorted(selected), separators=(',', ':'), ensure_ascii=True)
    return prefix + digest(data.encode()).hexdigest()[:16]
