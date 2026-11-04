"""Exact FM_P enumeration and auditable sampling; no detector evaluations."""
from __future__ import annotations
from collections import Counter
import hashlib
import importlib.util
import json
import random
from pathlib import Path


def load_fmp_module(source: Path):
    """Use the existing CV261109 implementation rather than copying model semantics."""
    source = Path(source)
    spec = importlib.util.spec_from_file_location('uavgap_cv261109_fmp_model', source)
    if spec is None or spec.loader is None:
        raise ValueError('Cannot load the existing FM_P model module.')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def enumerate_configs(module) -> list[tuple[str, ...]]:
    configs = list(module.combinations())
    if len(configs) != 46368 or len(set(configs)) != len(configs):
        raise ValueError('FM_P cardinality or uniqueness differs from the checked model.')
    if len(module.ALL_LEAVES) != 25 or any(not module.valid_leaves(c) for c in configs):
        raise ValueError('FM_P leaf semantics differ from the checked model.')
    return configs


def compute_binary_pair_masks(configs, leaves):
    """All feasible t=2 presence/absence states, including unselected leaves."""
    pairs = [(i, j) for i in range(len(leaves)) for j in range(i + 1, len(leaves))]
    masks, feasible = [], 0
    for selected in configs:
        active = set(selected)
        flags = [int(x in active) for x in leaves]
        mask = 0
        for k, (i, j) in enumerate(pairs):
            mask |= 1 << (4 * k + 2 * flags[i] + flags[j])
        masks.append(mask)
        feasible |= mask
    return masks, feasible


def make_plan(module, configs, seed=20261104, pairwise_budget=80, uniform_budget=100,
              first_pairs=500, detector_slots=5) -> dict:
    if not isinstance(seed, int) or pairwise_budget < 1 or uniform_budget < 1:
        raise ValueError('Invalid sampling budget.')
    if detector_slots < 1 or first_pairs % detector_slots != 0:
        raise ValueError('First-pair budget must be divisible by detector slots.')
    rng = random.Random(seed)
    leaves = sorted(module.ALL_LEAVES)
    masks, feasible = compute_binary_pair_masks(configs, leaves)
    uncovered = feasible
    greedy = []
    while uncovered:
        choices = [(m & uncovered).bit_count() for m in masks]
        best = max(choices)
        if best == 0:
            raise AssertionError('Feasible pair states were not covered.')
        idx = choices.index(best)
        greedy.append(idx)
        uncovered &= ~masks[idx]
        if len(greedy) > pairwise_budget:
            raise ValueError('Binary pairwise cover exceeds configured budget.')
    reserved = set(greedy)
    remainder = [i for i in range(len(configs)) if i not in reserved]
    padding = rng.sample(remainder, pairwise_budget - len(greedy))
    pairwise_indices = greedy + padding
    # Uniform means random sample from the complete valid model, not a sample
    # stratified by selected-feature count or hand-picked severity.
    uniform_indices = rng.sample(range(len(configs)), uniform_budget)
    chosen_for_first = pairwise_indices[:]
    for index in uniform_indices:
        if index not in chosen_for_first and len(chosen_for_first) < first_pairs // detector_slots:
            chosen_for_first.append(index)
    if len(chosen_for_first) < first_pairs // detector_slots:
        raise ValueError('Not enough unique degradation configurations for first pairs.')
    included = 0
    for i in pairwise_indices:
        included |= masks[i]
    if included != feasible:
        raise AssertionError('Final pairwise plan does not cover every feasible binary state.')
    def row(index, cohort):
        leaves_selected = configs[index]
        return {'configuration_id': module.configuration_id(leaves_selected),
                'selected_leaves': list(leaves_selected), 'leaf_count': len(leaves_selected),
                'sampling_cohort': cohort, 'valid_space_index': index}
    first_ids = set(chosen_for_first)
    return {
        'schema': 1,
        'seed': seed,
        'status': 'SAMPLING_DESIGN_ONLY_NO_YOLO_INFERENCE',
        'FM_P_valid_count': len(configs), 'FM_P_leaf_count': len(leaves),
        'pairwise_semantics': 'feasible_binary_presence_absence_pairs_over_25_FM_P_leaves',
        'feasible_pairwise_states': feasible.bit_count(),
        'greedy_cover_count': len(greedy),
        'pairwise_budget': pairwise_budget,
        'uniform_budget': uniform_budget,
        'active_sampling': 'NOT_SELECTED_REQUIRES_LEARNED_MODEL',
        'first_phase': {'planned_pairs': first_pairs,
                        'degradation_configs': first_pairs // detector_slots,
                        'detector_slots': detector_slots,
                        'checkpoint_binding': 'REQUIRED_BEFORE_EVALUATION',
                        'inference': 'NOT_RUN',
                        'reporting_split': 'VisDrone2019-DET-test-dev',
                        'test_dev_images_per_pair': 1610,
                        'planned_total_image_inferences': first_pairs * 1610},
        'pairwise': [row(i, 'pairwise') for i in pairwise_indices],
        'uniform': [row(i, 'uniform') for i in uniform_indices],
        'first_phase_fmp': [row(i, 'first500') for i in chosen_for_first],
        'selected_feature_histogram': dict(sorted(Counter(len(x) for x in configs).items())),
    }


def write_plan_files(plan, target: Path):
    import csv
    target.mkdir(parents=True, exist_ok=True)
    def write_csv(filename, rows, names):
        with (target / filename).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, names)
            writer.writeheader()
            writer.writerows(rows)
    all_rows = []
    for cohort in ('pairwise', 'uniform'):
        for row in plan[cohort]:
            all_rows.append({'configuration_id': row['configuration_id'],
                             'cohort': cohort, 'selected_leaves': '|'.join(row['selected_leaves']),
                             'selected_leaf_count': row['leaf_count'],
                             'status': 'SELECTED_NOT_MEASURED'})
    write_csv('sampled_fmp_configurations.csv', all_rows, list(all_rows[0]))
    pairs = []
    for row in plan['first_phase_fmp']:
        for slot in range(1, plan['first_phase']['detector_slots'] + 1):
            pairs.append({'pair_id': f"pair_{len(pairs)+1:04d}",
                          'FM_P_configuration_id': row['configuration_id'],
                          'FM_D_slot': f'D_SLOT_{slot}', 'checkpoint_sha256': 'NOT_BOUND',
                          'status': 'PLANNED_NOT_EXECUTED',
                          'planned_split': 'VisDrone2019-DET-test-dev', 'AP50_95': '',
                          'latency_ms': ''})
    if len(pairs) != plan['first_phase']['planned_pairs']:
        raise AssertionError('Unexpected initial-pair plan size.')
    write_csv('first_500_pairs_PLAN_ONLY.csv', pairs, list(pairs[0]))
    brief = {key: value for key, value in plan.items() if key not in ('pairwise', 'uniform', 'first_phase_fmp')}
    (target / 'sampling_summary.json').write_text(json.dumps(brief, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return len(pairs)
