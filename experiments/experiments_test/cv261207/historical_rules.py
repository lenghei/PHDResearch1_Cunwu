"""Describe measured resolution gains; do not score shared-image holdouts."""
import argparse
import subprocess
import sys
from pathlib import Path

from common import FLAMA, load, save, source


def fit_rules(out):
    with (out / 'rules_cpu.log').open('x', encoding='utf-8') as log:
        subprocess.run([str(FLAMA), '-B', str(Path(__file__)), '--run', str(out)],
                       check=True, timeout=120, stdout=log, stderr=subprocess.STDOUT)


def fit(out):
    import csv
    import numpy as np
    import sklearn
    from sklearn.tree import DecisionTreeRegressor, export_text
    from historical_review import table, csv_file

    data = [r for r in table(out / 'resolution_differences.csv')
            if r['metric'] == 'AP50_95' and r['high'] == '1280' and r['low'] == '640']
    features = ('Severe blur', 'JPEG Q85', 'JPEG Q50', 'YOLOv8s', 'YOLO11s')
    x = np.array([[int(r['blur']) == 2, r['jpeg'] == 'Q85', r['jpeg'] == 'Q50',
                   r['checkpoint'] == 'ck05', r['checkpoint'] == 'ck06'] for r in data], dtype=float)
    y = np.array([float(r['estimate']) for r in data])
    model = DecisionTreeRegressor(max_depth=2, min_samples_leaf=3, random_state=42)
    model.fit(x, y)
    predicted = model.predict(x)
    csv_file(out / 'rule_predictions.csv', [{**r, 'fitted_gain':float(p),
        'absolute_error':float(abs(p-y[i])), 'evaluation':'IN_SAMPLE_DESCRIPTIVE'}
        for i,(r,p) in enumerate(zip(data,predicted))])
    rules = export_text(model, feature_names=list(features), decimals=6)
    (out / 'learned_rules.txt').write_text(rules, encoding='utf-8')
    save(out / 'rule_validation.json', {
        'scope':'DEVELOPMENT_ONLY_POST_HOC', 'model':'DecisionTreeRegressor',
        'sklearn_version':sklearn.__version__, 'max_depth':2, 'min_samples_leaf':3,
        'random_state':42, 'rows':len(data), 'shared_images':300,
        'target':'AP50:95(1280) - AP50:95(640)', 'features':features,
        'categorical_encoding':'Indicator variables; no ordinal illumination assumption.',
        'in_sample_MAE':float(abs(predicted-y).mean()),
        'held_out_status':'NOT_EVALUATED',
        'reason':'All aggregate responses share the same development images. Configuration or checkpoint splits alone do not establish image-disjoint validation.',
        'causal_attribution':'NOT_SUPPORTED',
        'source':source(out / 'resolution_differences.csv'),
        'future_validation':'Freeze rules and thresholds, split unseen scenes before evaluation, then recompute AP separately on image-disjoint and configuration-disjoint partitions.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    fit(Path(args.run))
