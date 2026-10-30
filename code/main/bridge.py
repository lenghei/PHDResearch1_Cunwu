"""Saved FlamaPy-to-inference configuration bridge."""
import random
from pathlib import Path

from .config import save, sha256
from .uvl import enumerate_uvl

LEVELS = [
    ['Normal', 'Overexposed', 'Underexposed'],
    ['BlurNone', 'BlurModerate', 'BlurSevere'],
    ['NoiseNone', 'NoiseModerate', 'NoiseSevere'],
]


def features(configuration):
    return sorted(getattr(k, 'name', str(k)) for k, value in configuration.elements.items() if value)


def numeric_parameters(indices):
    i, b, n = indices
    return {'alpha': [1., 1.3, .8][i], 'beta': [0., 80., -50.][i],
            'kernel': [1, 7, 21][b], 'noise_variance': [0., 15., 30.][n]}


def build_bridge(model_folder, output, seed=20261027, resolution=1280):
    import importlib.metadata
    folder = Path(model_folder)
    dataset, data_configs = enumerate_uvl(folder / 'dataset.uvl')
    yolo, model_configs = enumerate_uvl(folder / 'yolo.uvl')
    rows = []
    for config in data_configs:
        selected = features(config)
        indices = tuple(next(j for j, name in enumerate(names) if name in selected) for names in LEVELS)
        rows.append((indices, selected))
    rows.sort()
    clean = next(row for row in rows if row[0] == (0, 0, 0))
    degraded = random.Random(seed).choice([row for row in rows if row[0] != (0, 0, 0)])
    model = next(features(c) for c in model_configs
                 if 'YOLO26sFT1280' in features(c) and f'R{resolution}' in features(c))
    result = {
        'status': 'PASS', 'seed': seed,
        'dataset': {'valid_count': len(data_configs), 'constraint_count': len(dataset.ctcs)},
        'yolo': {'valid_count': len(model_configs), 'constraint_count': len(yolo.ctcs)},
        'versions': {name: importlib.metadata.version(name) for name in ['flamapy-fw', 'flamapy-fm', 'flamapy-sat', 'uvlparser']},
        'UVL_SHA256': {name: sha256(folder / name) for name in ['dataset.uvl', 'yolo.uvl']},
        'model_features': model, 'checkpoint_id': 'ck01', 'resolution': resolution,
        'order': ['illumination', 'motion_blur', 'Gaussian_noise'],
        'scope': 'Original baseline UVLs; not final supervisor FM_P/FM_D',
        'conditions': [{'name': 'CLEAN' if indices == (0, 0, 0) else 'DEGRADED',
                        'configuration_id': f'ck01_I{indices[0]}B{indices[1]}N{indices[2]}_R{resolution}',
                        'indices': list(indices), 'features': selected,
                        'parameters': numeric_parameters(indices)} for indices, selected in [clean, degraded]],
    }
    if (len(data_configs), len(model_configs)) != (27, 9):
        raise ValueError('Baseline UVL configuration counts changed.')
    save(output, result)
    return result
