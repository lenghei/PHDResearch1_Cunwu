"""Freeze a uniform executable cohort and keep the full-space design separate."""
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
import random

from common import EXPERIMENTS, FMP, FMD, sha256, import_file, file_id, save

SEED = 20261104
RENDER_SEED = 20261207


def models():
    prior = import_file('cv261123_sampling', EXPERIMENTS / 'cv261123/sampling.py')
    fmp = prior.load_fmp_module(FMP.with_name('model.py'))
    fmd = import_file('cv261116_model', FMD.with_name('model.py'))
    return fmp, fmd, prior


def detector_design(fmd, checkpoint_sha):
    rows = []
    for resolution, precision in ((640, 'FP32'), (960, 'FP32'), (1280, 'FP32'),
                                  (640, 'FP16'), (1280, 'FP16')):
        row = fmd.declared_profile_configuration('YOLOv8', 'Small', 'VisDrone_FT', 'Train640',
                                    resolution, precision, 'FullFrame', 'NMS')
        leaves = row['selected_features']
        if row['configuration_id'] != file_id(leaves, 'd_'):
            raise ValueError('Detector ID mismatch.')
        rows.append({'detector_id': row['configuration_id'], 'leaves': leaves,
                     'checkpoint': 'cv261102_yolov8s_cft640', 'checkpoint_sha256': checkpoint_sha,
                     'resolution': resolution, 'precision': precision,
                     'training_trace_status': 'ALBU_ACTIVATION_UNVERIFIED',
                     'identity_scope': 'declared UVL leaves plus checkpoint SHA'})
    return rows


def uniform(population, size, seed):
    return random.Random(seed).sample(population, size)


def freeze(out, configs, fmp, prior, detectors):
    design = prior.make_plan(fmp, configs, seed=SEED)
    eligible = [c for c in configs if 'RollingShutter' not in c]
    sample = uniform(eligible, 100, SEED)
    specs = [{'degradation_id': fmp.configuration_id(c), 'leaves': list(c)} for c in sample]
    pairs = [{'pair_id': p['degradation_id'] + '__' + d['detector_id'],
              'degradation': p, 'detector': d} for p in specs for d in detectors]
    if len(set(x['pair_id'] for x in pairs)) != 500:
        raise ValueError('Expected 500 unique pairs.')
    manifest = {'status': 'PLANNED', 'seed': SEED, 'render_seed': RENDER_SEED,
                'target_split': 'VisDrone2019-DET-test-dev', 'FM_P_sha256': sha256(FMP),
                'FM_D_sha256': sha256(FMD), 'sampling': 'Exact enumeration, uniform without replacement',
                'eligible_FM_P': len(eligible), 'full_FM_P': len(configs),
                'rolling_shutter_excluded': len(configs)-len(eligible),
                'per_configuration_inclusion_probability': len(sample)/len(eligible),
                'full_space_histogram': Counter(map(len, configs)),
                'pairwise': {'definition': design['pairwise_semantics'],
                             'feasible': design['feasible_pairwise_states'],
                             'achieved': design['feasible_pairwise_states'],
                             'impossible': 4*300-design['feasible_pairwise_states'],
                             'greedy_cover_size': design['greedy_cover_count'],
                             'cohort': design['pairwise']},
                'full_space_uniform_candidates': design['uniform'],
                'active': 'NOT_EVALUATED', 'pairs': pairs}
    save(out / 'selection_manifest.json', manifest)
    return manifest


def check_generator(fmp, configs):
    import numpy as np
    import sys
    sys.path.insert(0, str(FMP.parent))
    try:
        renderer = import_file('cv261109_generator', FMP.with_name('generator.py'))
    finally:
        sys.path.remove(str(FMP.parent))
    image = np.random.default_rng(SEED).integers(0, 256, (79, 103, 3), dtype=np.uint8)
    attrs = fmp.read_uvl_attributes(FMP)
    exercised = set()
    for leaf in sorted(fmp.ALL_LEAVES):
        selection = next(c for c in configs if leaf in c)
        first, trace = renderer.generate(image, selection, attrs, 'audit_fixture', RENDER_SEED)
        second, trace_repeat = renderer.generate(image, selection, attrs, 'audit_fixture', RENDER_SEED)
        if not np.array_equal(first, second) or trace != trace_repeat:
            raise ValueError('Nondeterministic renderer.')
        exercised.update(row['feature'] for row in trace)
    if exercised != set(attrs):
        raise ValueError('Some degradation leaves were not exercised.')
    return {'status': 'PASS', 'fixture': 'synthetic deterministic renderer test',
            'leaves_exercised': sorted(exercised), 'geometry_warning':
            'RollingShutter displaces pixels per row with reflected borders; original GT is invalid.'}
