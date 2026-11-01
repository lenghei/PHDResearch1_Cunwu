"""Supervisor FM_P semantics and numeric attributes read from the UVL itself."""
from __future__ import annotations
import hashlib
import math
import itertools
import json
import re
from pathlib import Path

GROUPS = {
    'illumination': ('Normal', 'Overexposed_mild', 'Overexposed_strong',
                     'Underexposed_mild', 'Underexposed_strong', 'LowLight'),
    'motion': ('Motion_k7', 'Motion_k15', 'Motion_k21'),
    'defocus': ('Defocus_r3', 'Defocus_r6'),
    'gaussian': ('Gauss_s5', 'Gauss_s15', 'Gauss_s30'),
    'shot': ('Shot_low', 'Shot_high'),
    'compression': ('JPEG_q75', 'JPEG_q50', 'JPEG_q25'),
    'atmosphere': ('Haze_light', 'Haze_dense', 'Rain'),
    'altitude': ('Downscale_x1_5', 'Downscale_x2'),
    'rolling': ('RollingShutter',),
}
ALL_LEAVES = frozenset(itertools.chain.from_iterable(GROUPS.values()))
MODEL_FEATURES = frozenset(('Degradation', 'Illumination', 'Blur', 'MotionBlur',
                            'DefocusBlur', 'Noise', 'GaussianNoise', 'ShotNoise',
                            'Compression', 'Atmosphere', 'Altitude'))
ORDER = ('atmosphere', 'illumination', 'blur', 'rolling_shutter',
         'noise', 'downscale', 'compression')
ATTRIBUTE_RE = re.compile(r'^\s*([A-Za-z][\w]*)\s*\{([^{}]*)\}\s*$')
NUMERIC_RE = re.compile(r'^\s*([A-Za-z][\w]*)\s+(-?(?:\d+(?:\.\d*)?|\.\d+))\s*$')


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_uvl_attributes(path: Path) -> dict:
    attributes = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        match = ATTRIBUTE_RE.match(line)
        if not match or match.group(1) not in ALL_LEAVES:
            continue
        feature, body = match.groups()
        if feature in attributes:
            raise ValueError('Duplicate UVL attribute block: ' + feature)
        values = {}
        for entry in body.split(','):
            part = NUMERIC_RE.fullmatch(entry)
            if not part:
                raise ValueError('Invalid numeric UVL attribute: ' + entry)
            key, number = part.groups()
            if key in values:
                raise ValueError('Duplicate UVL attribute: ' + key)
            values[key] = float(number) if '.' in number else int(number)
        attributes[feature] = values
    if set(attributes) != ALL_LEAVES:
        raise ValueError('UVL leaf attribute set differs from the documented FM_P.')
    required_by_group = {
        'illumination': {'gain', 'bias'}, 'motion': {'kernel','angle_min','angle_max'},
        'defocus': {'radius'}, 'gaussian': {'sigma'}, 'shot': {'photons'},
        'compression': {'quality'}, 'atmosphere': None, 'altitude': {'factor'},
        'rolling': {'max_shift_fraction', 'waves'},
    }
    for group, members in GROUPS.items():
        for feature in members:
            required = required_by_group[group]
            if group == 'atmosphere':
                required = {'beta', 'airlight'} if feature.startswith('Haze') else {'streaks','stroke_length','opacity'}
            if set(attributes[feature]) != required:
                raise ValueError('Unexpected numeric UVL attributes for ' + feature)
            if not all(math.isfinite(float(x)) for x in attributes[feature].values()):
                raise ValueError('Nonfinite UVL numeric attribute: ' + feature)
    for feature in GROUPS['compression']:
        if not 1 <= attributes[feature]['quality'] <= 100:
            raise ValueError('Invalid JPEG quality: ' + feature)
    for feature in GROUPS['motion']:
        size = attributes[feature]['kernel']
        if int(size) != size or size < 3 or size % 2 != 1:
            raise ValueError('Invalid motion blur kernel: ' + feature)
    for feature in GROUPS['shot']:
        if attributes[feature]['photons'] <= 0:
            raise ValueError('Invalid photon level: ' + feature)
    return attributes


def _one(features, group):
    return len(set(features).intersection(GROUPS[group]))


def valid_leaves(features) -> bool:
    """Independent model-structure oracle; FlamaPy is still required for runs."""
    selected = set(features)
    if not selected.issubset(ALL_LEAVES) or _one(selected, 'illumination') != 1:
        return False
    if any(_one(selected, g) > 1 for g in ('motion', 'defocus', 'gaussian', 'shot',
                                            'compression', 'atmosphere', 'altitude', 'rolling')):
        return False
    if 'LowLight' in selected and not _one(selected, 'shot'):
        return False
    if 'Rain' in selected and ('Overexposed_strong' in selected or 'Overexposed_mild' in selected):
        return False
    if 'Haze_dense' in selected and 'Overexposed_strong' in selected:
        return False
    if 'RollingShutter' in selected and not _one(selected, 'motion'):
        return False
    if 'Shot_high' in selected and not {'LowLight', 'Underexposed_strong'}.intersection(selected):
        return False
    return True


def combinations():
    """Enumerate structured leaf selections independently from the UVL parser."""
    optional = lambda names: [()] + [(n,) for n in names]
    blur = [()] + [(n,) for n in GROUPS['motion'] + GROUPS['defocus']] + [
        (m, d) for m in GROUPS['motion'] for d in GROUPS['defocus']]
    noise = [()] + [(n,) for n in GROUPS['gaussian'] + GROUPS['shot']] + [
        (g, s) for g in GROUPS['gaussian'] for s in GROUPS['shot']]
    for values in itertools.product(GROUPS['illumination'], blur, noise,
                                    optional(GROUPS['compression']),
                                    optional(GROUPS['atmosphere']),
                                    optional(GROUPS['altitude']), ((), ('RollingShutter',))):
        first, *parts = values
        selected = frozenset((first, *itertools.chain.from_iterable(parts)))
        if valid_leaves(selected):
            yield tuple(sorted(selected))


def configuration_id(leaves) -> str:
    value = json.dumps(sorted(leaves), separators=(',', ':'), ensure_ascii=True)
    return 'p_' + hashlib.sha256(value.encode('utf-8')).hexdigest()[:16]


def recipe(leaves, attributes: dict) -> dict:
    leaves = set(leaves)
    if not valid_leaves(leaves):
        raise ValueError('Invalid FM_P leaf configuration.')
    return {feature: attributes.get(feature, {}) for feature in sorted(leaves)}


def clean_selection(leaves) -> bool:
    return set(leaves) == {'Normal'}
