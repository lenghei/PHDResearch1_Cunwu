"""Fit grouped baseline, tree, rule list and forest models on measured pairs."""
from collections import Counter

from gap_analysis import pair_outcome

DEFAULT_TAU = .10
DEFAULT_LIMIT = 50.0


def fold_membership(train, test):
    if {r['group'] for r in train} & {r['group'] for r in test}:
        return 'DEGRADATION_LEAKAGE'
    for row in train+test:
        images=row.get('image_sha256')
        if (not isinstance(images,list) or not images or len(images)!=len(set(images))
                or any(not isinstance(s,str) or len(s)!=64 or any(c not in '0123456789abcdef' for c in s) for s in images)):
            return 'IMAGE_MEMBERSHIP_UNVERIFIED'
    if set().union(*(set(r['image_sha256']) for r in train)) & set().union(*(set(r['image_sha256']) for r in test)):
        return 'IMAGE_LEAKAGE'
    return 'PASS'


def training_rows(records, tau=DEFAULT_TAU, limit=DEFAULT_LIMIT):
    rows = []
    for record in records:
        if record['run']['mode'] != 'FORMAL':
            continue
        c = record['configuration']
        detector = c['detector']
        leaves = c['degradation']['leaves']
        features = {f'P:{leaf}': 1 for leaf in leaves}
        features.update({f'D:{leaf}': 1 for leaf in detector['leaves']})
        features['D:resolution'] = detector['resolution']
        from sampling import models
        from common import FMP
        fmp = models()[0]
        params = fmp.read_uvl_attributes(FMP)
        for leaf in leaves:
            features.update({f'P:{leaf}:{key}': value for key, value in params[leaf].items()})
        ci = record['bootstrap']['intervals']['tiny']['ci95']
        status = pair_outcome(ci, record['timing']['mean_pipeline_ms'], tau, limit)
        if status in ('PASS', 'FAIL', 'UNCERTAIN'):
            rows.append({'group': c['degradation']['degradation_id'], 'label': status,
                         'features': features, 'run': record['run']['RunID'],
                         'image_sha256': c.get('evaluation_image_sha256')})
    return rows


def evaluate(records):
    rows = training_rows(records)
    groups = {row['group'] for row in rows}
    counts = Counter(row['label'] for row in rows)
    group_counts = {label: len({r['group'] for r in rows if r['label'] == label}) for label in counts}
    if len(groups) < 20 or len(counts) < 2 or min(group_counts.values()) < 5:
        return {'status': 'ML_INSUFFICIENT_EVIDENCE', 'labeled_pairs': len(rows),
                'degradation_groups': len(groups), 'class_counts': counts,
                'reason': 'Requires at least 20 groups, with each observed label present in five groups.'}
    if any(not row['image_sha256'] for row in rows):
        return {'status':'ML_INSUFFICIENT_EVIDENCE', 'labeled_pairs':len(rows),
                'reason':'Image membership is required to verify image-disjoint grouped folds.'}
    try:
        from sklearn.dummy import DummyClassifier
        from sklearn.feature_extraction import DictVectorizer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
        from sklearn.model_selection import StratifiedGroupKFold
        from sklearn.tree import DecisionTreeClassifier, export_text
    except ImportError:
        return {'status': 'BLOCKED', 'reason': 'scikit-learn is unavailable.'}

    observations = [r['features'] for r in rows]
    labels = [r['label'] for r in rows]
    group_ids = [r['group'] for r in rows]
    scores = []
    for seed in (42, 73, 101):
        splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        for fold, (train, test) in enumerate(splitter.split(observations, labels, group_ids)):
            train_groups = {group_ids[i] for i in train}
            test_groups = {group_ids[i] for i in test}
            if train_groups & test_groups:
                raise ValueError('Degradation leakage between grouped folds.')
            boundary=fold_membership([rows[i] for i in train],[rows[i] for i in test])
            if boundary!='PASS':
                return {'status':'ML_INSUFFICIENT_EVIDENCE', 'labeled_pairs':len(rows),
                        'reason':boundary}
            vector = DictVectorizer(sparse=False)
            x_train = vector.fit_transform([observations[i] for i in train])
            x_test = vector.transform([observations[i] for i in test])
            y_train = [labels[i] for i in train]
            y_test = [labels[i] for i in test]
            if set(y_train) != set(counts) or set(y_test) != set(counts):
                return {'status': 'ML_INSUFFICIENT_EVIDENCE', 'reason': 'A grouped fold lacks a label.',
                        'labeled_pairs': len(rows), 'class_counts': counts}
            candidates = {'majority': DummyClassifier(strategy='most_frequent'),
                          'rule_stump': DecisionTreeClassifier(max_depth=1, min_samples_leaf=4, random_state=seed),
                          'tree': DecisionTreeClassifier(max_depth=3, min_samples_leaf=4, random_state=seed),
                          'forest': RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=4,
                                                           random_state=seed, n_jobs=1)}
            for name, model in candidates.items():
                model.fit(x_train, y_train)
                pred = model.predict(x_test)
                rule = export_text(model, feature_names=list(vector.get_feature_names_out())) if name in ('tree','rule_stump') else None
                scores.append({'seed': seed, 'fold': fold, 'model': name, 'rule': rule,
                               'test_groups': sorted(test_groups),
                               'balanced_accuracy': balanced_accuracy_score(y_test, pred),
                               'macro_f1': f1_score(y_test, pred, average='macro', zero_division=0),
                               'fail_precision': precision_score(y_test, pred, labels=['FAIL'], average='macro', zero_division=0),
                               'fail_recall': recall_score(y_test, pred, labels=['FAIL'], average='macro', zero_division=0),
                               'confusion_labels': sorted(counts),
                               'confusion': confusion_matrix(y_test, pred, labels=sorted(counts)).tolist(),
                               'held_out': [{'RunID': rows[i]['run'], 'label': labels[i], 'predicted': str(p)}
                                            for i, p in zip(test, pred)]})
    return {'status': 'VALIDATED_GROUPED_EVALUATION', 'labeled_pairs': len(rows),
            'class_counts': counts, 'model_complexity': 'Fixed before scoring', 'scores': scores}
