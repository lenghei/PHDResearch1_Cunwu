"""Check archived Ultralytics defaults and count the FM_P coverage policies with SAT."""
import argparse
import ast
from hashlib import sha256
import json
from pathlib import Path

from flamapy.metamodels.fm_metamodel.transformations.uvl_reader import UVLReader
from flamapy.metamodels.pysat_metamodel.transformations.fm_to_pysat import FmToPysat
from pysat.solvers import Solver

HERE = Path(__file__).resolve().parent
SOURCE = (HERE.parents[2] / "datasets/local_runs/cv261207/"
          "cv261207_supervisor_compliance_20261109T121122_70d953f1/sources/ultralytics_8.4.33_augment.py")
SOURCE_SHA = "37a32d0d0555db44bd84f149673d9a85d7be1529e793e4e204a5001b62201dae"
FMP = HERE.parent / "uav_02_fmp/fm_p.uvl"
EXPECTED = {"Blur": .01, "MedianBlur": .01, "ToGray": .01, "CLAHE": .01,
            "RandomBrightnessContrast": 0., "RandomGamma": 0., "ImageCompression": 0.}


def source_defaults(path):
    payload = path.read_bytes()
    if sha256(payload).hexdigest() != SOURCE_SHA:
        raise ValueError("Ultralytics 8.4.33 source SHA differs.")
    module = ast.parse(payload.decode("utf-8-sig"))
    classes = {stmt.name: stmt for stmt in module.body if isinstance(stmt, ast.ClassDef)}
    albu = next(stmt for stmt in classes["Albumentations"].body
                if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__")
    transforms = next(stmt.value.body for stmt in ast.walk(albu) if isinstance(stmt, ast.Assign)
                      and isinstance(stmt.value, ast.IfExp) and isinstance(stmt.value.body, ast.List)
                      and any(isinstance(target,ast.Name) and target.id=="T" for target in stmt.targets))
    observed = {call.func.attr: next(ast.literal_eval(kw.value) for kw in call.keywords if kw.arg == "p")
                for call in transforms.elts}
    hsv = ast.unparse(classes["RandomHSV"])
    if observed != EXPECTED or "np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain]" not in hsv:
        raise ValueError("Default transform list or HSV draw differs.")
    if "lut_val = np.clip(x * (r[2] + 1), 0, 255).astype(dtype)" not in hsv:
        raise ValueError("HSV value transform differs.")
    return observed


def sat_count(sat, variables, allowed):
    count = 0
    with Solver(bootstrap_with=sat.get_all_clauses().clauses) as solver:
        excluded = [-var for name,var in variables.items() if name not in allowed]
        while solver.solve(assumptions=excluded):
            positive = set(solver.get_model())
            solver.add_clause([-var if var in positive else var for var in variables.values()])
            count += 1
    return count


def verify(path):
    transforms = source_defaults(path)
    rules = json.loads((HERE/"coverage_rules.json").read_bytes())
    fm = UVLReader(str(FMP)).transform()
    sat = FmToPysat(fm).transform()
    leaves = {f.name:sat.variables[f.name] for f in fm.get_features() if not f.relations}
    policies = {"STRICT_IDENTITY":{"Normal"},
                "BRIGHTNESS_ONLY":{"Normal","Overexposed_mild","Underexposed_mild"},
                "SUPERVISOR_PROXY":{name for levels in rules["covered_levels"].values() for name in levels}}
    counts = {name:sat_count(sat,leaves,allowed) for name,allowed in policies.items()}
    if len(leaves)!=25 or counts != {"STRICT_IDENTITY":1,"BRIGHTNESS_ONLY":3,"SUPERVISOR_PROXY":18}:
        raise ValueError("The static coverage counts differ.")
    return {"source_version":"8.4.33","source_sha256":SOURCE_SHA,"hsv_v":.4,"hsv_gain":[.6,1.4],
            "hsv_offset":0,"albu_default_p":transforms,"motion_blur_in_defaults":False,
            "noise_in_defaults":False,"historical_augmentation_activation":"UNKNOWN",
            "FM_P_SHA256":sha256(FMP.read_bytes()).hexdigest(),"FM_P_population":46368,
            "coverage_counts":counts,"proxy_uncovered_fraction":(46368-18)/46368}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=SOURCE)
    parser.add_argument("--output",type=Path)
    args = parser.parse_args()
    payload=json.dumps(verify(args.source),indent=2)+"\n"
    if args.output:
        with args.output.open("x",encoding="utf-8") as file:
            file.write(payload)
    print(payload,end="")
