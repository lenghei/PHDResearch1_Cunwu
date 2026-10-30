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
    labels = {PurePosixPath(n).stem: n for n in members if '/annotations/' in n and n.lower().endswith('.txt') and
              not n.startswith('__MACOSX/') and not PurePosixPath(n).name.startswith('._')}
    stems = [PurePosixPath(n).stem for n in images]
    if not images or len(set(stems)) != len(stems) or set(stems) != set(labels):
        raise ValueError('Images and original annotations are not one-to-one.')
    return images, labels


def label_rows(raw: bytes, width: int, height: int):
    """Category 1..10 -> YOLO 0..9; ignore region (0), others (11), score 0 excluded."""
    rows = []
    for number, line in enumerate(raw.decode('utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        cells = [float(x.strip()) for x in line.split(',')]
        if len(cells) < 8 or not all(math.isfinite(v) for v in cells):
            raise ValueError(f'Malformed VisDrone annotation line {number}.')
        x, y, w, h, score, category = cells[:6]
        if w <= 0 or h <= 0 or int(category) != category or not 0 <= category <= 11:
            raise ValueError(f'Invalid VisDrone box or category at line {number}.')
        if category in (0, 11) or score == 0:
            continue
        left, top = max(0., x), max(0., y)
        right, bottom = min(float(width), x + w), min(float(height), y + h)
        if right <= left or bottom <= top:
            continue
        xc = (left + right) / (2. * width)
        yc = (top + bottom) / (2. * height)
        wn = (right - left) / width
        hn = (bottom - top) / height
        if not all(0 <= v <= 1 for v in [xc, yc, wn, hn]):
            raise ValueError(f'Invalid normalized box at line {number}.')
        rows.append(f'{int(category) - 1} {xc:.8f} {yc:.8f} {wn:.8f} {hn:.8f}')
    return rows


def convert_split(source: Path, destination: Path, split: str, expected: int | None):
    import cv2
    import numpy as np

    image_dir = destination / 'images' / split
    label_dir = destination / 'labels' / split
    image_dir.mkdir(parents=True, exist_ok=False)
    label_dir.mkdir(parents=True, exist_ok=False)
    stems, boxes = set(), 0
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
            rows = label_rows(z.read(labels[stem]), width, height)
            (image_dir / name).write_bytes(blob)
            (label_dir / (stem + '.txt')).write_text('\n'.join(rows) + ('\n' if rows else ''), encoding='utf-8')
            boxes += len(rows)
            stems.add(stem)
    return {'split': split, 'images': len(stems), 'labels': boxes,
            'source_zip_SHA256': digest(source), 'stems': stems}


def prepare(train_zip: Path, val_zip: Path, dest: Path):
    import yaml

    dest.mkdir(parents=True, exist_ok=False)
    train = convert_split(train_zip, dest, 'train', None)
    if train['images'] < 100:
        raise ValueError('Expected the full VisDrone train split, not a tiny sample.')
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
        'note': 'Ignore/other regions omitted from training targets; evaluation uses original ignore-union protocol.',
    }
