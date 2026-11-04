"""Read-only input, split and feature-model checks."""
from collections import Counter
from importlib import metadata
from pathlib import Path, PurePosixPath
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile

from common import (ROOT, PUBLIC, EXPERIMENTS, ARCHIVES, FMP, FMD, VAL, TEST,
                    BASELINE, STATIC, PROFILE, FLAMA, load, source, sha256, save, file_id)


def command(*args, timeout=240):
    return subprocess.run(args, check=True, capture_output=True, text=True,
                          encoding='utf-8', errors='replace', timeout=timeout).stdout.strip()


def compare_overlay(path):
    if path is None:
        return {'status': 'NOT_SUPPLIED'}
    with zipfile.ZipFile(path) as archive:
        rows = []
        names = set()
        for member in archive.infolist():
            name = PurePosixPath(member.filename)
            if member.is_dir():
                continue
            if (name.is_absolute() or '..' in name.parts or name.as_posix() in names
                    or '\\' in member.filename):
                raise ValueError('Unsafe or duplicate ZIP member.')
            names.add(name.as_posix())
            target = ROOT.joinpath(*name.parts).resolve()
            if not target.is_relative_to(ROOT):
                raise ValueError('ZIP member resolves outside the project.')
            expected = hashlib.sha256(archive.read(member)).hexdigest()
            actual = sha256(target) if target.is_file() else None
            rows.append({'member': name.as_posix(), 'zip_sha256': expected, 'installed_sha256': actual,
                         'status': 'MATCH' if expected == actual else 'DIFFERS' if actual else 'MISSING'})
    return {'source': source(path), 'status': 'MATCH' if all(r['status'] == 'MATCH' for r in rows)
            else 'DIFFERENCES_PRESERVED', 'members': rows}


def archive_ids(path):
    def ids(names, folder, suffix):
        return [PurePosixPath(n).stem for n in names
                if PurePosixPath(n).parent.name == folder and n.lower().endswith(suffix)
                and '__MACOSX' not in n and not PurePosixPath(n).name.startswith('._')]
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        images = ids(names, 'images', ('.jpg', '.jpeg', '.png'))
        labels = ids(names, 'annotations', ('.txt',))
    if (len(images) != len(set(images)) or len(labels) != len(set(labels))
            or set(images) != set(labels) or len(names) != len(set(names))):
        raise ValueError(f'Unmatched or duplicate ZIP members: {path.name}')
    return set(images)


def split_audit(out, history):
    from archives import OriginalZIP
    pools = {}
    inputs = {}
    for key, count in (('train', 6471), ('val', 548), ('test-dev', 1610)):
        path = ARCHIVES / f'VisDrone2019-DET-{key}.zip'
        pools[key] = archive_ids(path)
        if len(pools[key]) != count:
            raise ValueError(f'{key} member count mismatch.')
        inputs[key] = dict(source(path), images=count)
    overlap = {key: sorted(pools[key] & pools['test-dev']) for key in ('train', 'val')}
    archive = OriginalZIP(TEST, 1610)
    issues = []
    class_counts = Counter()
    with (out / 'test_dev_members.jsonl').open('x', encoding='utf-8') as stream:
        try:
            for i in range(1610):
                try:
                    _, rec, provenance = archive.record(i)
                    class_counts.update(o['class_id'] + 1 for o in rec['objects'])
                    stream.write(json.dumps(dict(provenance, image_stem=rec['stem'])) + '\n')
                except (ValueError, UnicodeError, zipfile.BadZipFile) as exc:
                    issues.append({'member': archive.images[i], 'error': str(exc)})
        finally:
            archive.close()
    prior = [{'source': source(path), 'recorded_status': load(path).get('status')}
             for path in history]
    return {'status': 'DATASET_BLOCKED',
            'archive_status': 'PASS' if not issues and not any(overlap.values()) else 'FAILED',
            'reason': 'Historical test-dev use remains unverified. Formal evaluation is locked.',
            'archives': inputs, 'test_dev_train_val_id_overlap': overlap,
            'test_dev_records_read': 1610-len(issues), 'native10_objects_before_ignore_union': class_counts,
            'test_dev_record_errors': issues, 'history': prior,
            'test_dev_member_manifest_sha256': sha256(out / 'test_dev_members.jsonl')}


def verify_uvls(configs, fmp, fmd, seed):
    p = json.loads(command(str(FLAMA), '-B', str(FMP.with_name('flamapy_select.py')),
                           '--model', str(FMP), '--seed', str(seed)).splitlines()[-1])
    d = json.loads(command(str(FLAMA), '-B', str(FMD.with_name('flamapy_count.py')),
                           '--model', str(FMD)).splitlines()[-1])
    attrs = fmp.read_uvl_attributes(FMP)
    if (p['valid_count'] != len(configs) or p['configuration_id'] != file_id(p['selected_leaves'], 'p_')
            or set().union(*map(set, configs)) != set(attrs)
            or len({file_id(row, 'p_') for row in configs}) != len(configs)
            or any(fmp.configuration_id(row) != file_id(row, 'p_') for row in configs)):
        raise ValueError('FM_P parser/enumerator/identity mismatch.')
    if d['valid_count'] != fmd.enumerate_count()['independent_configurations']:
        raise ValueError('FM_D factorization and enumeration disagree.')
    p.update(leaves=len(attrs), dead_leaves=0, satisfiable=True,
             independent_configuration_ids_checked=len(configs), attributes=attrs,
             processing_order=fmp.ORDER, UVL_SHA256=sha256(FMP))
    return {'FM_P': p, 'FM_D': d, 'FM_D_independent': fmd.enumerate_count()}


def input_audit(args):
    checkpoints = sorted((ROOT / 'models/checkpoints').rglob('*.pt'))
    existing = [FMP, FMD, BASELINE / 'manifest.json', BASELINE / 'coco_regeneration.json',
                BASELINE / 'training/checkpoint.json', BASELINE / 'training/yolov8s/args.yaml',
                STATIC / 'coverage_rules.json', STATIC / 'static_gap.json',
                STATIC / 'training_source_audit.json', PROFILE / 'result.json',
                PROFILE / 'measurements.csv', *checkpoints]
    for exp in ('cv261102', 'cv261109', 'cv261116', 'cv261123'):
        existing.append(EXPERIMENTS / exp / 'run.py')
    existing.append(EXPERIMENTS / 'cv261123/figures.py')
    packages = {}
    for name in ('torch', 'ultralytics', 'opencv-python', 'numpy', 'pycocotools',
                 'albumentations', 'matplotlib', 'sahi', 'scikit-learn'):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {'git': {'commit': command('git', '-C', str(PUBLIC), 'rev-parse', 'HEAD'),
                    'root': command('git', '-C', str(PUBLIC), 'rev-parse', '--show-toplevel'),
                    'status': command('git', '-C', str(PUBLIC), 'status', '--porcelain')},
            'sources': [source(path) for path in (*args.source, *existing)],
            'overlay': compare_overlay(args.overlay), 'disk': shutil.disk_usage(ROOT)._asdict(),
            'python': sys.version, 'packages': packages,
            'gpu': command('nvidia-smi', '--query-gpu=name,driver_version,memory.total,memory.free,utilization.gpu',
                           '--format=csv'), 'ai_assistance': 'Codex assisted code implementation, testing and audit tooling.'}
