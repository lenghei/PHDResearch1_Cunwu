"""Enumerate FM_P with FlamaPy BDD and freeze a uniform sample."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
from random import Random

from flamapy.metamodels.bdd_metamodel.operations.bdd_configurations import BDDConfigurations
from flamapy.metamodels.bdd_metamodel.operations.bdd_configurations_number import BDDConfigurationsNumber
from flamapy.metamodels.bdd_metamodel.transformations.fm_to_bdd import FmToBDD
from flamapy.metamodels.fm_metamodel.transformations.uvl_reader import UVLReader


def freeze(path, output, seed, size):
    model = UVLReader(str(path)).transform()
    leaves = {f.name for f in model.get_features() if not f.relations}
    constraints = len(model.ctcs)
    bdd = FmToBDD(model).transform()
    count = BDDConfigurationsNumber().execute(bdd).get_result()
    configs = BDDConfigurations().execute(bdd).get_result()
    rows = sorted(tuple(sorted(name for name, chosen in config.elements.items()
                               if chosen and name in leaves)) for config in configs)
    if (len(leaves), constraints, count, len(rows), len(set(rows))) != (25, 5, 46368, 46368, 46368):
        raise ValueError("Unexpected FM_P leaf, constraint, or configuration count.")
    payload = "".join("|".join(row) + "\n" for row in rows).encode("ascii")
    indices = Random(seed).sample(range(count), size)
    data = {"leaf_count": len(leaves), "leaf_features": sorted(leaves), "constraint_count": constraints,
            "population": count, "uvl_sha256": sha256(path.read_bytes()).hexdigest(),
            "leaf_catalog_sha256": sha256(payload).hexdigest(),
            "method": "FlamaPy BDD full enumeration; canonical leaf order; uniform without replacement",
            "seed": seed, "inclusion_probability": size/count, "uniform_indices": indices,
            "packages": {name: version(name) for name in ("flamapy-bdd", "flamapy-fm", "dd")},
            "bdd_backend": type(bdd.bdd).__module__, "catalog": rows}
    with output.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, separators=(",", ":"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--size", type=int, required=True)
    args = parser.parse_args()
    freeze(args.model, args.output, args.seed, args.size)
