"""Convert original VisDrone DET ZIPs into training-only YOLO native-10 labels.

Never modifies source ZIPs. Validation images are used during fine-tuning for
checkpoint selection; the subsequent 548-image evaluation is not independent.
"""
from __future__ import annotations

import hashlib
import math
import zipfile
from pathlib import Path, PurePosixPath

NAMES = [
    'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
    'tricycle', 'awning-tricycle', 'bus', 'motor',
]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def list_members(z: zipfile.ZipFile):
    members = z.namelist()
    if len(members) != len(set(members)):
        raise ValueError('Source ZIP has duplicated member paths.')
    images = sorted(n for n in members if '/images/' in n and
                    n.lower().endswith(('.jpg', '.jpeg', '.png')) and
                    not n.startswith('__MACOSX/') and not PurePosixPath(n).name.startswith('._'))
    annotations = [n for n in members if '/annotations/' in n and n.lower().endswith('.txt') and
                   not n.startswith('__MACOSX/') and not PurePosixPath(n).name.startswith('._')]
    labels = {PurePosixPath(n).stem: n for n in annotations}
    if len(labels) != len(annotations):
        raise ValueError('Multiple annotation files have the same image stem.')
    stems = [PurePosixPath(n).stem for n in images]
    if not images or len(set(stems)) != len(stems) or set(stems) != set(labels):
        raise ValueError('Images and original annotations are not one-to-one.')
    return images, labels


def parse_annotation(line: str, number: int, source_name: str = '<annotation>'):
    """Check the original eight-field record without treating zero area as corruption."""
    context = f'{source_name}:{number}'
    fields = line.split(',')
    # Some original VisDrone records end with a trailing comma.
    # Accept one empty final field, but never discard an extra value.
    if len(fields) == 9 and not fields[-1].strip():
        fields.pop()
    if len(fields) != 8:
        raise ValueError(f'{context}: expected 8 VisDrone fields, got {len(fields)}; row={line[:160]!r}')
    try:
        values = [float(field.strip()) for field in fields]
    except ValueError as error:
        raise ValueError(f'{context}: non-numeric VisDrone record; row={line[:160]!r}') from error
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f'{context}: non-finite VisDrone record; row={line[:160]!r}')
    x, y, w, h, score, category = values[:6]
    if int(category) != category or not 0 <= category <= 11:
        raise ValueError(f'{context}: invalid VisDrone category {category}; row={line[:160]!r}')
    if w < 0 or h < 0:
        raise ValueError(f'{context}: negative box dimensions; row={line[:160]!r}')
    return x, y, w, h, score, int(category)


def scan_archive_annotations(archive: zipfile.ZipFile) -> dict:
    """Preflight all annotation records before starting a costly conversion or GPU run."""
    _, labels = list_members(archive)
    counts = dict(annotation_files=len(labels), rows=0, ignored=0,
                  zero_area=0, candidate_objects=0)
    examples = []
    for source_name in sorted(labels.values()):
        raw = archive.read(source_name)
        for number, line in enumerate(raw.decode('utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            x, y, w, h, score, category = parse_annotation(line, number, source_name)
            counts['rows'] += 1
            if category in (0, 11) or score == 0:
                counts['ignored'] += 1
            elif w == 0 or h == 0:
                counts['zero_area'] += 1
                if len(examples) < 20:
                    examples.append(dict(annotation=source_name, line=number, box=[x, y, w, h],
                                         category=category))
            else:
                counts['candidate_objects'] += 1
    counts['zero_area_examples'] = examples
    return counts


def label_rows(raw: bytes, width: int, height: int, *,
               annotation_path: str = '<annotation>', statistics: dict | None = None):
    """Convert native classes 1..10; exclude pseudo-classes and zero-area boxes."""
    if width <= 0 or height <= 0:
        raise ValueError(f'{annotation_path}: invalid image dimensions {width}x{height}')
    rows = []
    for number, line in enumerate(raw.decode('utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        x, y, w, h, score, category = parse_annotation(line, number, annotation_path)
        if statistics is not None:
            statistics['rows'] += 1
        if category in (0, 11) or score == 0:
            if statistics is not None:
                statistics['ignored'] += 1
            continue
        if w == 0 or h == 0:
            # The official train split contains zero-area records. They cannot
            # become valid YOLO targets and must not abort the whole dataset.
            if statistics is not None:
                statistics['zero_area'] += 1
                if len(statistics['zero_area_examples']) < 20:
                    statistics['zero_area_examples'].append(dict(
                        annotation=annotation_path, line=number, box=[x, y, w, h],
                        category=category))
            continue
        left, top = max(0., x), max(0., y)
        right, bottom = min(float(width), x + w), min(float(height), y + h)
        if right <= left or bottom <= top:
            if statistics is not None:
                statistics['outside_image'] += 1
            continue
        xc = (left + right) / (2. * width)
        yc = (top + bottom) / (2. * height)
        wn = (right - left) / width
        hn = (bottom - top) / height
        if not all(0 <= v <= 1 for v in [xc, yc, wn, hn]):
            raise ValueError(f'{annotation_path}:{number}: invalid normalized box')
        rows.append(f'{category - 1} {xc:.8f} {yc:.8f} {wn:.8f} {hn:.8f}')
        if statistics is not None:
            statistics['kept'] += 1
    return rows


def convert_split(source: Path, destination: Path, split: str, expected: int | None):
    import cv2
    import numpy as np

    image_dir = destination / 'images' / split
    label_dir = destination / 'labels' / split
    image_dir.mkdir(parents=True, exist_ok=False)
    label_dir.mkdir(parents=True, exist_ok=False)
    stems, boxes = set(), 0
    stats = dict(rows=0, ignored=0, zero_area=0, outside_image=0, kept=0,
                 zero_area_examples=[])
    with zipfile.ZipFile(source) as z:
        images, labels = list_members(z)
        if expected is not None and len(images) != expected:
            raise ValueError(f'{split}: expected {expected} images, found {len(images)}.')
        for image_path in images:
            name = PurePosixPath(image_path).name
            stem = PurePosixPath(name).stem
            blob = z.read(image_path)
            decoded = cv2.imdecode(np.frombuffer(blob, dtype=np.uint8), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if decoded is None:
                raise ValueError(f'Image decode failed for {name}.')
            height, width = decoded.shape[:2]
            rows = label_rows(z.read(labels[stem]), width, height,
                              annotation_path=labels[stem], statistics=stats)
            (image_dir / name).write_bytes(blob)  # retain original image bytes
            (label_dir / (stem + '.txt')).write_text('\n'.join(rows) + ('\n' if rows else ''), encoding='utf-8')
            boxes += len(rows)
            stems.add(stem)
    if boxes != stats['kept'] or stats['rows'] != sum(stats[key] for key in (
            'kept', 'ignored', 'zero_area', 'outside_image')):
        raise AssertionError(f'{split}: conversion accounting mismatch')
    return {'split': split, 'images': len(stems), 'labels': boxes,
            'annotation_accounting': stats,
            'source_zip_SHA256': digest(source), 'stems': stems}


def prepare(train_zip: Path, val_zip: Path, dest: Path):
    import yaml

    dest.mkdir(parents=True, exist_ok=False)
    train = convert_split(train_zip, dest, 'train', 6471)
    val = convert_split(val_zip, dest, 'val', 548)
    if train['stems'].intersection(val['stems']):
        raise ValueError('Train and val share image stems; split contamination.')
    yaml_path = dest / 'visdrone10.yaml'
    yaml_path.write_text(yaml.safe_dump({
        'path': str(dest.resolve()), 'train': 'images/train', 'val': 'images/val',
        'nc': 10, 'names': NAMES,
    }, sort_keys=False, allow_unicode=True), encoding='utf-8')
    return yaml_path, {
        'train': {k: v for k, v in train.items() if k != 'stems'},
        'val': {k: v for k, v in val.items() if k != 'stems'},
        'yaml_SHA256': digest(yaml_path),
        'note': 'Ignore/other and zero-area regions excluded from training targets; evaluation uses original ignore-union protocol.',
    }
