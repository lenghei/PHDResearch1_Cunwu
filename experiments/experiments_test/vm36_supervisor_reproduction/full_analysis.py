"""Paired contrasts and incomplete-subset accounting."""
import math
import numpy as np


def binary_sample_size(population, half_width, z=1.96):
    if population < 1 or not 0 < half_width < 1:
        raise ValueError('Invalid population or precision.')
    return math.ceil(population*z*z*.25/(half_width*half_width*(population-1)+z*z*.25))


def paired_effect(terms, size, metric='map50_95'):
    """Combine point estimates and identical image-bootstrap draws."""
    reference = None
    estimate = 0.0
    values = None
    for coefficient, measured, bootstrap in terms:
        draws = bootstrap['draws']
        identity = (tuple(draws['image_ids']), draws['samples'], draws['seed'], draws['sampling'])
        if reference is not None and identity != reference:
            raise ValueError('Bootstrap pairing differs.')
        reference = identity
        point = next(row for row in measured['per_area'] if row['size'] == size)[metric]
        sample = np.asarray(draws['values'][size][metric], dtype=float)
        if point is None or len(sample) != draws['samples'] or not np.isfinite(sample).all():
            raise ValueError('Missing or nonfinite contrast evidence.')
        estimate += coefficient*point
        values = coefficient*sample if values is None else values+coefficient*sample
    if reference is None:
        raise ValueError('A contrast requires measured terms.')
    return {'estimate': estimate, 'CI95_low': float(np.quantile(values, .025)),
            'CI95_high': float(np.quantile(values, .975)), 'images': len(reference[0]),
            'draws': reference[1]}


def matched_contrasts(accepted):
    rows = []
    for checkpoint in ('ck05', 'ck01'):
        for size in ('all', 'tiny', 'small', 'medium', 'large'):
            for quality in (25, 75):
                keys = [(checkpoint, 1280, quality), (checkpoint, 640, quality)]
                if not all(key in accepted for key in keys):
                    continue
                terms = [(sign, accepted[key][1], accepted[key][2])
                         for sign, key in zip((1, -1), keys)]
                rows.append({'checkpoint': checkpoint, 'size': size,
                    'contrast': '1280-640 at Q'+str(quality), **paired_effect(terms, size),
                    'scope': 'POINTWISE_PAIRED_IMAGE_DEVELOPMENT_ONLY'})
            keys = [(checkpoint, 1280, 25), (checkpoint, 640, 25),
                    (checkpoint, 1280, 75), (checkpoint, 640, 75)]
            if all(key in accepted for key in keys):
                terms = [(sign, accepted[key][1], accepted[key][2])
                         for sign, key in zip((1, -1, -1, 1), keys)]
                rows.append({'checkpoint': checkpoint, 'size': size,
                    'contrast': '(1280-640 at Q25)-(1280-640 at Q75)',
                    **paired_effect(terms, size), 'scope': 'POINTWISE_PAIRED_IMAGE_DEVELOPMENT_ONLY'})
    return rows


def tested_subset_state(outcomes, expected):
    if len(outcomes) > expected:
        raise ValueError('More outcomes than declared detector settings.')
    if 'PASS' in outcomes:
        return 'COVERED_TESTED_SUBSET'
    if len(outcomes) == expected and all(value == 'FAIL' for value in outcomes):
        return 'GAP_TESTED_SUBSET'
    return 'UNRESOLVED'


def checkpoint_contrasts(accepted):
    rows=[]
    for size in ('all','tiny','small','medium','large'):
        for resolution in (640,1280):
            keys=[('ck01',resolution,25),('ck01',resolution,75),
                  ('ck05',resolution,25),('ck05',resolution,75)]
            if not all(key in accepted for key in keys):continue
            terms=[(sign,accepted[key][1],accepted[key][2])
                   for sign,key in zip((1,-1,-1,1),keys)]
            rows.append({'size':size,'resolution':resolution,
                'contrast':'(ck01 Q25-Q75)-(ck05 Q25-Q75)',**paired_effect(terms,size),
                'scope':'POST_HOC_DESCRIPTIVE_POINTWISE_FIXED_CHECKPOINTS'})
    return rows
