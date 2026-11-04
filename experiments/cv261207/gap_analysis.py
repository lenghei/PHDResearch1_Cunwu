"""Static coverage under explicit definitions and conservative empirical labels."""
from collections import Counter
import inspect

from common import BASELINE, STATIC, load, source


def static_audit(fmp, configs):
    import yaml
    import albumentations as albu
    import ultralytics
    from ultralytics.data import augment

    args_path = BASELINE / 'training/yolov8s/args.yaml'
    args = yaml.safe_load(args_path.read_text(encoding='utf-8'))
    original = load(BASELINE / 'manifest.json')
    rules_path = STATIC / 'coverage_rules.json'
    rules = load(rules_path)
    source_path = __import__('pathlib').Path(inspect.getfile(augment))
    active = (ultralytics.__version__ == original['system']['ultralytics'] and
              'lut_val = np.clip(x * (r[2] + 1)' in inspect.getsource(augment.RandomHSV) and
              args['hsv_v'] == 0.4)
    variants = {name: set() for name in ('exact_identity', 'conservative', 'supervisor_proxy')}
    proxy = set().union(*map(set, rules['covered_levels'].values()))
    leaves = {}
    for family, members in fmp.GROUPS.items():
        for leaf in members:
            status, transform, params, probability = 'UNSUPPORTED', None, None, None
            rationale = 'No matching transform in the inspected baseline training pipeline.'
            if leaf == 'Normal':
                status, transform = 'EXACT', 'identity recipe'
                rationale = 'Identity of the degradation generator; does not assert a clean training-pixel frequency.'
            elif leaf in {'Overexposed_mild', 'Underexposed_mild'}:
                status = 'WEAK_PROXY' if active else 'UNKNOWN'
                transform, params = 'RandomHSV.value', [1-args['hsv_v'], 1+args['hsv_v']]
                rationale = 'Multiplicative brightness range matches the gain only; the additive offset is absent.'
            elif leaf in fmp.GROUPS['illumination']:
                transform, params = 'RandomHSV.value', [1-args['hsv_v'], 1+args['hsv_v']]
                rationale = 'The strong additive offsets or low-light gain are not reproduced.'
            elif leaf == 'Defocus_r3':
                status, transform = 'UNKNOWN', 'Albumentations.Blur/MedianBlur'
                params, probability = list(albu.Blur().blur_limit), .01
                rationale = 'Runtime source permits a weak kernel proxy; no training-run activation log verified.'
            elif family == 'compression':
                transform, params, probability = 'Albumentations.ImageCompression', [75, 100], 0
                rationale = 'Disabled in the inspected default source.'
            elif family == 'altitude':
                status, transform, params = 'WEAK_PROXY', 'RandomPerspective.scale / Mosaic', [1-args['scale'], 1+args['scale']]
                probability = {'mosaic_before_close': args['mosaic'], 'close_mosaic_epochs': args['close_mosaic']}
                rationale = 'Changes target geometry and labels during training; no fixed-grid down/up sampling equivalent.'
            leaves[leaf] = {'family': family, 'classification': status, 'transform': transform,
                            'parameter_range': params, 'probability': probability, 'rationale': rationale,
                            'source_version': ultralytics.__version__}
    variants['exact_identity'] = {'Normal'}
    variants['conservative'] = {'Normal'} | ({'Overexposed_mild', 'Underexposed_mild'} if active else set())
    variants['supervisor_proxy'] = proxy
    counts = {}
    for name, allowed in variants.items():
        accepted = [row for row in configs if set(row).issubset(allowed)]
        by_leaf = Counter(x for row in accepted for x in row)
        population = Counter(x for row in configs for x in row)
        counts[name] = {'covered': len(accepted), 'gap': len(configs)-len(accepted),
                        'population': len(configs), 'covered_leaves': sorted(allowed),
                        'covered_histogram': Counter(map(len, accepted)),
                        'population_histogram': Counter(map(len, configs)),
                        'by_leaf': {leaf: {'covered_when_present': by_leaf[leaf], 'present_total': population[leaf]}
                                    for leaf in sorted(leaves)},
                        'by_family': {family: {'covered_when_present': sum(bool(set(row).intersection(members)) for row in accepted),
                                               'present_total': sum(bool(set(row).intersection(members)) for row in configs)}
                                      for family, members in fmp.GROUPS.items()}}
    return {'scope': 'Rule-based static coverage sensitivity for CV261102; not a proof of pixel reachability.',
            'definitions': counts, 'leaves': leaves, 'args': source(args_path), 'proxy_rules': source(rules_path),
            'augmentation_source': source(source_path), 'albumentations_version': albu.__version__,
            'albumentations_transform_sources': {name: source(__import__('pathlib').Path(inspect.getfile(cls)))
                                                  for name, cls in [('Blur', albu.Blur), ('ImageCompression', albu.ImageCompression)]},
            'hsv_probability_note': 'RandomHSV is a pipeline step with no independent probability gate.',
            'historical_albumentations_activation': 'UNKNOWN',
            'training_args': {key: args.get(key) for key in ('hsv_v', 'scale', 'mosaic', 'close_mosaic', 'shear', 'perspective')}}


def pair_outcome(ci, latency_ms, tau, limit_ms):
    if latency_ms is not None and latency_ms > limit_ms:
        return 'FAIL'
    if ci is None or any(x is None for x in ci) or latency_ms is None:
        return 'NOT_EVALUATED'
    lower, upper = ci
    if upper < tau:
        return 'FAIL'
    if lower >= tau:
        return 'PASS'
    return 'UNCERTAIN'


def subset_outcome(outcomes):
    if 'PASS' in outcomes:
        return 'COVERED_TESTED_SUBSET'
    if outcomes and all(x == 'FAIL' for x in outcomes):
        return 'GAP_TESTED_SUBSET'
    return 'UNRESOLVED'
