"""Sample complete FM_D configurations and check them with FlamaPy SAT."""
import argparse
import json
from pathlib import Path
from random import Random

from flamapy.metamodels.fm_metamodel.transformations.uvl_reader import UVLReader
from flamapy.metamodels.pysat_metamodel.transformations.fm_to_pysat import FmToPysat
from pysat.solvers import Solver

from export_configuration_validation import (
    ARCH, SCALE, WEIGHTS, TRAIN_RES, INFER_RES, STRATEGY, POST, PRECISION,
    AUG_OPTIONAL, MANDATORY_AUG, PREPROCESS, read_and_check_uvl, sha_id, valid_config,
)


def sample(path, seed):
    sha = read_and_check_uvl(path)
    model = UVLReader(str(path)).transform()
    sat = FmToPysat(model).transform()
    leaf_vars = {f.name: sat.variables[f.name] for f in model.get_features() if not f.relations}
    random = Random(seed)
    groups = (ARCH, SCALE, WEIGHTS, TRAIN_RES, INFER_RES, STRATEGY, POST, PRECISION)
    optional = (*AUG_OPTIONAL, *PREPROCESS, "TTA")
    rows, seen = [], set()
    with Solver(bootstrap_with=sat.get_all_clauses().clauses) as solver:
        while len(rows) < 8:
            leaves = {random.choice(group) for group in groups}
            leaves.update(MANDATORY_AUG)
            leaves.update(name for name in optional if random.getrandbits(1))
            key = sha_id("d_", leaves)
            if key in seen or not valid_config(leaves):
                continue
            if not solver.solve(assumptions=[var if name in leaves else -var for name, var in leaf_vars.items()]):
                raise ValueError(f"FlamaPy rejected {key}.")
            seen.add(key)
            rows.append({"configuration_id": key, "leaves": sorted(leaves), "status": "SAT_VALID"})
    return {"seed": seed, "uvl_sha256": sha, "samples": rows, "scope": "LOGIC_SAMPLE"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = sample(Path(__file__).with_name("fm_d.uvl"), args.seed)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(payload)
    else:
        print(payload, end="")
