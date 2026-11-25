"""Executed by the dedicated FlamaPy interpreter; outputs one JSON selection."""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import random
from pathlib import Path


def select(model: Path, seed: int, requested: str | None = None):
    from flamapy.metamodels.fm_metamodel.transformations.uvl_reader import UVLReader
    from flamapy.metamodels.pysat_metamodel.transformations.fm_to_pysat import FmToPysat
    from flamapy.metamodels.pysat_metamodel.operations.pysat_configurations import PySATConfigurations
    from model import MODEL_FEATURES, ALL_LEAVES, configuration_id, valid_leaves

    fm = UVLReader(str(model)).transform()
    configurations = PySATConfigurations().execute(FmToPysat(fm).transform()).get_result()
    rows = set()
    for config in configurations:
        selected = {getattr(node, 'name', str(node)) for node, state in config.elements.items() if state}
        unknown = selected - MODEL_FEATURES - ALL_LEAVES
        if unknown:
            raise ValueError('Unexpected selected UVL features: ' + repr(sorted(unknown)))
        leaves = tuple(sorted(selected.intersection(ALL_LEAVES)))
        if not valid_leaves(leaves):
            raise ValueError('FlamaPy selected an invalid FM_P configuration.')
        rows.add(leaves)
    if len(configurations) != len(rows):
        raise ValueError('Duplicate FlamaPy feature configurations detected.')
    if len(fm.ctcs) != 5:
        raise ValueError('Expected five supervisor cross-tree constraints.')
    if len(rows) != 46368:
        raise ValueError(f'FM_P count differs from the supervisor model: {len(rows)}')
    ordered = sorted(rows)
    if requested:
        chosen = next((s for s in ordered if configuration_id(s) == requested), None)
        if chosen is None:
            raise ValueError('Requested configuration_id is not valid in FM_P.')
    else:
        chosen = random.Random(seed).choice(ordered)
    versions = {}
    for package in ('flamapy-fw', 'flamapy-fm', 'flamapy-sat', 'uvlparser'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return dict(status='PASS', valid_count=len(rows), constraint_count=len(fm.ctcs),
                FlamaPy_packages=versions,
                selected_leaves=list(chosen), configuration_id=configuration_id(chosen),
                selection_seed=seed, selector='FlamaPy.PySATConfigurations',
                UVL_SHA256=hashlib.sha256(model.read_bytes()).hexdigest())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--configuration-id')
    args = parser.parse_args()
    print(json.dumps(select(args.model.resolve(), args.seed, args.configuration_id),
                     sort_keys=True, allow_nan=False))
