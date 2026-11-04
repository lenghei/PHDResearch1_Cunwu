"""Independent FM_D enumeration and supervisor FM_P proxy coverage rules."""
from __future__ import annotations
import hashlib
import itertools
import json
from pathlib import Path

ARCH = ('YOLOv8', 'YOLO11', 'YOLO26', 'LEAF_YOLO')
SCALE = ('Nano', 'Small')
WEIGHTS = ('COCO_ZeroShot', 'VisDrone_FT')
TRAIN_RES = ('Train640', 'Train1280')
AUG = ('Albu_Default', 'NoiseAug', 'MotionBlurAug', 'ExposureAug', 'JPEGAug', 'HazeAug')
INFER_RES = ('Infer640', 'Infer960', 'Infer1280', 'Infer1536')
STRATEGY = ('FullFrame', 'SAHI_Slicing')
POST = ('NMS', 'EndToEnd_NMSFree')
PRECISION = ('FP32', 'FP16', 'INT8')
PRE = ('CLAHE_Pre', 'Gamma_Pre', 'Denoise_Pre', 'Deblur_Pre')
LEAVES = set(ARCH + SCALE + WEIGHTS + TRAIN_RES + AUG + INFER_RES + STRATEGY + POST + PRECISION + PRE + ('HSV_Default', 'Geometric_Default', 'TTA'))
SUPERVISOR_COUNT = 635904
SUPERVISOR_TRAIN_VARIANTS = 844


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def training_variants():
    """Enumerate training-only decisions independent of FlamaPy parsing."""
    for architecture, scale, weights, resolution, bits in itertools.product(
        ARCH, SCALE, WEIGHTS, TRAIN_RES, range(1 << len(AUG))
    ):
        additions = frozenset(name for index, name in enumerate(AUG) if (bits >> index) & 1)
        if architecture == 'LEAF_YOLO' and (scale != 'Nano' or weights != 'VisDrone_FT' or resolution != 'Train640'):
            continue
        if weights == 'COCO_ZeroShot' and (resolution != 'Train640' or additions.intersection(AUG[1:])):
            continue
        yield (architecture, scale, weights, resolution, additions)


def enumerate_count():
    """Analytically count all inference branches for each valid training variant."""
    total = 0
    variants = 0
    for architecture, scale, weights, train_res, additions in training_variants():
        variants += 1
        postprocesses = POST if architecture == 'YOLO26' else ('NMS',)
        # The LEAF-specific NMS constraint is already satisfied here.
        if architecture == 'LEAF_YOLO':
            postprocesses = ('NMS',)
        for postprocess in postprocesses:
            for strategy in STRATEGY:
                tta_options = (False,) if strategy == 'SAHI_Slicing' else (False, True)
                # Preprocessing is optional OR(4): zero selected, or any nonempty subset.
                total += len(INFER_RES) * len(PRECISION) * (1 << len(PRE)) * len(tta_options)
    return {'independent_configurations': total, 'independent_training_variants': variants,
            'expected_configurations': SUPERVISOR_COUNT,
            'expected_training_variants': SUPERVISOR_TRAIN_VARIANTS,
            'constraints': 4, 'leaf_features': len(LEAVES)}


def load_rules(path):
    rules = json.loads(Path(path).read_text(encoding='utf-8'))
    if rules['schema'] != 1 or set(rules['covered_levels']) != {
        'illumination','motion','defocus','gaussian','shot','compression','atmosphere','altitude','rolling'
    }:
        raise ValueError('Incomplete coverage rule families.')
    if len(rules['rules']) != 9 or {r['feature_family'] for r in rules['rules']} != set(rules['covered_levels']):
        raise ValueError('Coverage evidence table is incomplete.')
    if 'PROVISIONAL' not in rules['verification_status']:
        raise ValueError('Do not label supervisor proxy assumptions as exact coverage.')
    return rules


def gap_analysis(fmp_module, rules):
    cover = rules['covered_levels']
    families = fmp_module.GROUPS
    for name, allowed in cover.items():
        if not set(allowed).issubset(set(families[name])):
            raise ValueError('Coverage rule contains a nonexistent FM_P feature: ' + name)
    count = 0
    covered = []
    by_feature = {feature: {'present_total': 0, 'covered_when_present': 0} for feature in fmp_module.ALL_LEAVES}
    total = 0
    for selected in fmp_module.combinations():
        total += 1
        s = set(selected)
        accepted = all(not s.intersection(members) or s.intersection(members).issubset(cover[family])
                       for family, members in families.items())
        for feature in s:
            by_feature[feature]['present_total'] += 1
            if accepted:
                by_feature[feature]['covered_when_present'] += 1
        if accepted:
            count += 1
            covered.append(fmp_module.configuration_id(selected))
    if total != 46368:
        raise ValueError('FM_P model does not reproduce the checked 46,368 configurations.')
    return {'FM_P_configurations': total, 'proxy_covered': count,
            'proxy_gap': total - count,
            'proxy_gap_fraction': (total - count) / total,
            'covered_configuration_ids': covered,
            'per_leaf': by_feature,
            'coverage_status': rules['verification_status'],
            'interpretation': 'Heuristic training-augmentation proxy; NOT measured detection failure.'}


def strict_identity_gap(fmp_module):
    covered = sum(1 for x in fmp_module.combinations() if fmp_module.clean_selection(x))
    return {'policy': 'EXACT_IDENTITY_ONLY', 'covered': covered,
            'gap': 46368 - covered,
            'note': 'Strict lower-bound coverage from no alteration. Not a physical equivalence oracle.'}


def declared_profile_configuration(architecture, scale, weights, train_resolution,
                                   inference_resolution, precision, strategy, postprocessing,
                                   tta=False, preprocessing='None'):
    """Validate the declared measured branch against the four FM_D constraints.

    Training augmentation remains an unverified attribute until source audit.
    """
    values = (architecture, scale, weights, train_resolution,
              f'Infer{inference_resolution}', precision, strategy, postprocessing)
    domains = (ARCH, SCALE, WEIGHTS, TRAIN_RES, INFER_RES, PRECISION, STRATEGY, POST)
    for item, domain in zip(values, domains):
        if item not in domain:
            raise ValueError('Invalid FM_D feature: ' + str(item))
    if architecture == 'LEAF_YOLO' and (scale != 'Nano' or weights != 'VisDrone_FT'
                                        or train_resolution != 'Train640' or postprocessing != 'NMS'):
        raise ValueError('LEAF_YOLO violates supervisor FM_D restrictions.')
    if postprocessing == 'EndToEnd_NMSFree' and architecture != 'YOLO26':
        raise ValueError('End-to-end NMS-free requires YOLO26.')
    if weights == 'COCO_ZeroShot' and train_resolution != 'Train640':
        raise ValueError('COCO_ZeroShot requires Train640.')
    if strategy == 'SAHI_Slicing' and tta:
        raise ValueError('The supervisor draft excludes SAHI with TTA.')
    if preprocessing not in ('None',) + PRE:
        raise ValueError('Invalid preprocessing feature.')
    selected = set(values) | {'HSV_Default', 'Geometric_Default'}
    if tta:
        selected.add('TTA')
    if preprocessing != 'None':
        selected.add(preprocessing)
    return {'selected_features':sorted(selected),
            'configuration_id':'d_' + hashlib.sha256(
                json.dumps(sorted(selected), separators=(',', ':')).encode()).hexdigest()[:16],
            'training_augmentation_status':'UNVERIFIED_SOURCE_AUDIT_REQUIRED',
            'configuration_status':'VALID_DECLARED_FEATURES_NOT_FULL_TRAINING_PROVENANCE'}
