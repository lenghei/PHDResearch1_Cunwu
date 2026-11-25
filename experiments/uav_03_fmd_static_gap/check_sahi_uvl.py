"""Count the leaf settings added by the scoped SAHI/TTA rule."""
from hashlib import sha256
import json
from pathlib import Path

from flamapy.metamodels.fm_metamodel.transformations.uvl_reader import UVLReader
from flamapy.metamodels.pysat_metamodel.transformations.fm_to_pysat import FmToPysat
from pysat.solvers import Solver


def joint_settings(path):
    fm = UVLReader(str(path)).transform()
    model = FmToPysat(fm).transform()
    leaves = {f.name: model.variables[f.name] for f in fm.get_features() if not f.relations}
    joint = [leaves["SAHI_Slicing"], leaves["TTA"]]
    results = []
    with Solver(bootstrap_with=model.get_all_clauses().clauses) as sat:
        while sat.solve(assumptions=joint):
            solution = set(sat.get_model())
            results.append(sorted(name for name, var in leaves.items() if var in solution))
            if len(results) > 1:
                raise ValueError("More than one SAHI/TTA setting was opened.")
            sat.add_clause([-var if var in solution else var for var in leaves.values()])
    return results


if __name__ == "__main__":
    original = Path(__file__).with_name("fm_d.uvl")
    revised = original.with_name("fm_d_sahi_tta.uvl")
    if sha256(original.read_bytes()).hexdigest() != "4022d674822b97dbbe3712646b9a0f970af95d175ae358e63582029bc107571a":
        raise ValueError("The original model differs from the counted baseline.")
    before, after = [p.read_text(encoding="utf-8").splitlines() for p in (original, revised)]
    if before[:-1] != after[:-1] or before[-1].strip() != "SAHI_Slicing => !TTA":
        raise ValueError("The scoped variant changed the model outside the SAHI/TTA rule.")
    if joint_settings(original):
        raise ValueError("The original SAHI/TTA exclusion differs.")
    new = joint_settings(revised)
    expected = {"YOLOv8","Small","VisDrone_FT","Train640","Infer640","FP32","NMS",
                "HSV_Default","Geometric_Default","SAHI_Slicing","TTA"}
    if len(new) != 1 or set(new[0]) != expected:
        raise ValueError("The new setting differs from the CPU compatibility fixture.")
    print(json.dumps({"original_configurations":635904,"added_settings":new,
          "revised_configurations":635905,"revised_sha256":sha256(revised.read_bytes()).hexdigest()}))
