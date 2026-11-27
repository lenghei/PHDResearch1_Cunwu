"""Count FM_D without materializing hundreds of thousands of configurations."""
from __future__ import annotations
import argparse
import importlib.metadata
import json
from pathlib import Path


def analyze(path: Path):
    from flamapy.metamodels.fm_metamodel.transformations.uvl_reader import UVLReader
    from flamapy.core.discover import DiscoverMetamodels
    path = path.resolve()
    fm = UVLReader(str(path)).transform()
    if len(fm.ctcs) != 4:
        raise ValueError('Expected exactly four FM_D cross-tree constraints.')
    features = fm.get_features()
    leaf_names = sorted(x.name for x in features if not getattr(x, 'relations', []))
    from model import LEAVES
    if set(leaf_names) != LEAVES or len(leaf_names) != 34:
        raise ValueError('Parsed FM_D leaf set differs from supervisor draft.')
    dm = DiscoverMetamodels()
    count = dm.use_operation_from_file('PySATConfigurationsNumber', str(path))
    # DiscoverMetamodels may return a direct integer or a typed result wrapper.
    if isinstance(count, dict):
        count = count.get('result', count.get('number_of_configurations'))
    if isinstance(count, str):
        count = int(count.strip())
    if type(count) is not int:
        raise ValueError('Unexpected FlamaPy count return type: ' + type(count).__name__)
    dead = dm.use_operation_from_file('PySATDeadFeatures', str(path))
    if dead is None:
        raise ValueError('FlamaPy dead-feature check returned no result.')
    if not isinstance(dead, (list, tuple, set)):
        raise ValueError('FlamaPy dead-feature check returned unexpected type.')
    if dead:
        raise ValueError('Dead FM_D features: ' + ', '.join(str(x) for x in dead))
    versions = {}
    for name in ('flamapy-fw', 'flamapy-fm', 'flamapy-sat', 'uvlparser'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {'status': 'PASS', 'backend': 'PySATConfigurationsNumber',
            'valid_count': count, 'constraint_count': len(fm.ctcs),
            'leaf_count': len(leaf_names), 'dead_feature_count': len(dead), 'packages': versions}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.model), sort_keys=True, allow_nan=False))
