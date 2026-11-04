"""Bounded FM_D audit: FlamaPy parsing/SAT plus exact structural factorization.

PySATConfigurationsNumber can enumerate hundreds of thousands of solutions.
It is intentionally NOT used here. The exact count is computed algebraically
only for the audited, unchanged FM_D UVL v1. Any model edit fails closed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path


# A versioned, strict reference for the *structure and constraints* used by the
# analytic proof. Merely having the same 34 leaf names is insufficient.
EXPECTED_UVL = """namespace YOLODetector

features
    Detector {abstract}
        mandatory
            Architecture
                alternative
                    YOLOv8
                    YOLO11
                    YOLO26
                    LEAF_YOLO
            ModelScale
                alternative
                    Nano
                    Small
            Weights
                alternative
                    COCO_ZeroShot
                    VisDrone_FT
            TrainResolution
                alternative
                    Train640
                    Train1280
            TrainAugmentation
                mandatory
                    HSV_Default
                    Geometric_Default
                optional
                    Albu_Default
                    NoiseAug
                    MotionBlurAug
                    ExposureAug
                    JPEGAug
                    HazeAug
            InferResolution
                alternative
                    Infer640
                    Infer960
                    Infer1280
                    Infer1536
            InferenceStrategy
                alternative
                    FullFrame
                    SAHI_Slicing
            PostProcessing
                alternative
                    NMS
                    EndToEnd_NMSFree
            Precision
                alternative
                    FP32
                    FP16
                    INT8
        optional
            Preprocessing
                or
                    CLAHE_Pre
                    Gamma_Pre
                    Denoise_Pre
                    Deblur_Pre
            TTA

constraints
    LEAF_YOLO => Nano & VisDrone_FT & Train640 & NMS
    EndToEnd_NMSFree => YOLO26
    COCO_ZeroShot => Train640 & !NoiseAug & !MotionBlurAug & !ExposureAug & !JPEGAug & !HazeAug
    SAHI_Slicing => !TTA
"""


def canonical_uvl(value: str) -> tuple[str, ...]:
    """Ignore trailing whitespace/empty lines, preserve UVL indentation."""
    return tuple(line.rstrip() for line in value.splitlines() if line.strip())


def validate_uvl_contract(path: Path) -> str:
    """Guard the factorization against subtle tree or constraint edits."""
    content = path.read_text(encoding='utf-8-sig')
    if canonical_uvl(content) != canonical_uvl(EXPECTED_UVL):
        raise ValueError('FM_D UVL structure or constraints changed; exact proof is no longer applicable.')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def factorized_count() -> dict:
    """Exact count for the guarded UVL, using its disjoint architecture cases.

    COCO_ZeroShot: train640, 2 scales, Albu_Default optional -> 4 per
      normal architecture. LEAF_YOLO cannot use COCO_ZeroShot.
    VisDrone_FT: 2 scales * 2 training resolutions * 2**6 augment subsets
      -> 256 per normal architecture; LEAF_YOLO restricts scale and
      training resolution -> 64.
    YOLO26 permits 2 postprocessors; each other architecture permits 1.
    Per postprocessor: 4 inference resolutions * 3 precision levels *
      2**4 optional preprocessing combinations * (2 FullFrame TTA states
      + 1 SAHI state, which excludes TTA) -> 576.
    """
    augmentation_subsets = 1 << 6
    coco_per_normal = 2 * 2
    ft_per_normal = 2 * 2 * augmentation_subsets
    normal_training = coco_per_normal + ft_per_normal
    leaf_training = augmentation_subsets
    training_variants = 3 * normal_training + leaf_training
    post_weighted = (1 + 1 + 2) * normal_training + leaf_training
    per_post = 4 * 3 * (1 << 4) * (2 + 1)
    total = post_weighted * per_post
    if (normal_training, training_variants, post_weighted, per_post, total) != (260, 844, 1104, 576, 635904):
        raise AssertionError('FM_D factorization proof changed unexpectedly.')
    return {
        'valid_count': total,
        'training_variants': training_variants,
        'proof_terms': {
            'training_per_regular_architecture': normal_training,
            'training_leaf_yolo': leaf_training,
            'weighted_postprocessing_cases': post_weighted,
            'inference_cases_per_postprocessor': per_post,
        },
        'count_method': 'EXACT_FACTORIZED_COUNT_OF_STRICT_UVL_V1',
    }


def analyze(path: Path) -> dict:
    from flamapy.metamodels.fm_metamodel.transformations.uvl_reader import UVLReader
    from flamapy.core.discover import DiscoverMetamodels

    path = path.resolve(strict=True)
    source_sha = validate_uvl_contract(path)
    fm = UVLReader(str(path)).transform()
    if len(fm.ctcs) != 4:
        raise ValueError('FlamaPy parsed a different number of cross-tree constraints.')
    features = fm.get_features()
    leaf_names = {x.name for x in features if not getattr(x, 'relations', [])}
    from model import LEAVES
    if leaf_names != LEAVES or len(leaf_names) != 34:
        raise ValueError('FlamaPy parsed a different leaf set.')

    # These are SAT *decision* queries, not #SAT enumeration.
    dm = DiscoverMetamodels()
    is_sat = dm.use_operation_from_file('PySATSatisfiable', str(path))
    if is_sat is not True:
        raise ValueError('FlamaPy PySATSatisfiable did not confirm a satisfiable UVL.')
    dead = dm.use_operation_from_file('PySATDeadFeatures', str(path))
    if dead is None or not isinstance(dead, (list, tuple, set)):
        raise ValueError('FlamaPy dead-feature operation did not return a feature list.')
    if dead:
        raise ValueError('Dead FM_D features: ' + ', '.join(str(x) for x in dead))

    counted = factorized_count()
    versions = {}
    for name in ('flamapy-fw', 'flamapy-fm', 'flamapy-sat', 'uvlparser'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        'status': 'PASS',
        'backend': 'FlamaPy UVL parser + PySATSatisfiable + PySATDeadFeatures',
        'count_backend': counted['count_method'],
        'valid_count': counted['valid_count'],
        'training_variants': counted['training_variants'],
        'proof_terms': counted['proof_terms'],
        'constraint_count': len(fm.ctcs),
        'leaf_count': len(leaf_names),
        'dead_feature_count': len(dead),
        'satisfiable': True,
        'UVL_SHA256': source_sha,
        'count_limitation': 'Not an independent FlamaPy BDD/#SAT count; count applies only to the exact audited UVL semantics.',
        'packages': versions,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.model), sort_keys=True, allow_nan=False))
