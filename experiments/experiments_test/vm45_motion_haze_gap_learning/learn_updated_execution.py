"""Whole-P holdout on the updated VM45 development index."""
import argparse, ast, csv, json, sys, time
from pathlib import Path
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from vm45_common import read, ref, save, csv_write, V44


def summary(y, p):
    n = len(y)
    positive = int((y == 0).sum())
    negative = int((y == 1).sum())
    tp = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 1) & (p == 0)).sum())
    tn = int(((y == 1) & (p == 1)).sum())
    fn = int(((y == 0) & (p == 1)).sum())
    return {'scored_rows': n, 'reference_PASS': positive, 'reference_FAIL': negative,
            'actual_FAIL_predicted_PASS': fp, 'false_PASS_rate': fp / negative if negative else None,
            'PASS_prediction_precision': tp / (tp + fp) if tp + fp else None,
            'PASS_predictions': tp + fp,
            'accuracy': (tp + tn) / n if n else None,
            'balanced_accuracy': (tp / positive + tn / negative) / 2 if positive and negative else None,
            'confusion_PASS_FAIL': [[tp, fn], [fp, tn]],
            'prediction_coverage': float(np.mean(p >= 0)) if n else None,
            'abstention_rate': float(np.mean(p < 0)) if n else None}


def learn(out):
    started = time.perf_counter()
    with (out / 'development_canonical_index.csv').open(encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 90 and len({(r['condition_id'], r['D']) for r in rows}) == 90
    rules = read(V44 / 'FM_P_to_FM_D_decision_rules.json')
    leaves = rules['models']['fm_p']['leaves']
    ds = sorted(rules['candidates'])
    names = leaves + ['D_' + d for d in ds] + ['TIMING_VM41']
    x = np.array([[int(v in ast.literal_eval(r['leaves'])) for v in leaves]
                  + [int(r['D'] == d) for d in ds]
                  + [int(r['timing_protocol'] == 'VM41_MATCHED_CACHED_GAP_V1')]
                  for r in rows])
    y = np.array([{'PASS': 0, 'FAIL': 1, 'UNCERTAIN': -1}[r['original_rule_state']] for r in rows])
    groups = np.array([r['condition_id'] for r in rows])
    assert len(set(groups)) == 24
    definitions = {
        'UPDATED_SHALLOW_TREE': lambda: DecisionTreeClassifier(max_depth=2, min_samples_leaf=4,
                                  class_weight='balanced', random_state=20261117),
        'UPDATED_RANDOM_FOREST_REFERENCE': lambda: RandomForestClassifier(n_estimators=100,
                                  max_depth=3, min_samples_leaf=4, class_weight='balanced',
                                  random_state=20261117, n_jobs=1)}
    predictions, folds, results = [], [], []
    valid = y >= 0
    for name, factory in definitions.items():
        p = np.full(len(rows), -1)
        for group in sorted(set(groups)):
            train = (groups != group) & valid
            test = groups == group
            assert not set(groups[train]) & set(groups[test])
            p[test] = factory().fit(x[train], y[train]).predict(x[test])
            folds.append({'method': name, 'heldout_P': group,
                          'train_P_ids': json.dumps(sorted(set(groups[train]))),
                          'validation_P_ids': json.dumps([group]),
                          'training_rows': int(train.sum()), 'test_rows': int(test.sum()),
                          'same_P_leakage': False})
        score = summary(y[valid], p[valid])
        rng = np.random.default_rng(20261117)
        boot = []
        for _ in range(200):
            draw = rng.choice(sorted(set(groups)), len(set(groups)), replace=True)
            indices = np.concatenate([np.where((groups == g) & valid)[0] for g in draw])
            boot.append(summary(y[indices], p[indices]))
        cis = {}
        for key in ('false_PASS_rate', 'PASS_prediction_precision', 'balanced_accuracy'):
            values = [s[key] for s in boot if s[key] is not None]
            cis[key] = {'ci95': np.percentile(values, [2.5, 97.5]).tolist() if values else None,
                        'valid_draws': len(values)}
        results.append({'method': name, **score, 'group_bootstrap': cis,
                        'uncertain_reference_excluded': int((~valid).sum()), 'admitted': False})
        for i, row in enumerate(rows):
            predictions.append({'method': name, 'P': row['condition_id'], 'D': row['D'],
                                'actual': row['original_rule_state'],
                                'predicted': 'PASS' if p[i] == 0 else 'FAIL',
                                'scored': bool(valid[i]), 'same_P_in_train': False,
                                'timing_protocol': row['timing_protocol']})
    results.append({'method': 'ALWAYS_FAIL_REFERENCE', **summary(y[valid], np.ones(valid.sum(), dtype=int)), 'admitted': False})
    csv_write(out / 'updated_rule_heldout_results.csv', predictions)
    csv_write(out / 'updated_rule_group_folds.csv', folds)
    save(out / 'updated_rule_validation.json', {
        'status': 'DIAGNOSTIC_ONLY_ABSTAIN_FOR_NEW_ENVIRONMENTS', 'source': ref(out / 'development_canonical_index.csv'),
        'updated_training_population': len(rows), 'P_groups': len(set(groups)),
        'features': names, 'models': results, 'sklearn_version': sklearn.__version__,
        'interpreter': sys.executable, 'CPU_wall_seconds': time.perf_counter() - started,
        'method': 'Unchanged fixed tree/RF hyperparameters; whole-P holdout; no tuning',
        'adaptation': 'New 90-row diagnostic with explicit timing-cohort indicator; not a repeat of old84-only experiment',
        'interval_scope': '200 whole-P resamples of fixed OOF predictions; no refit or independent-image claim',
        'limitations': ['Shared548 development images', 'Sparse feasible P groups',
                       'Historical/new timing cohorts differ', 'No independent safe-release validation'],
        'old84_only_refit': False, 'external_testdev_rows': 0, 'admitted_rules': 0,
        'source_code': ref(__file__)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    learn(parser.parse_args().out.resolve(strict=True))
