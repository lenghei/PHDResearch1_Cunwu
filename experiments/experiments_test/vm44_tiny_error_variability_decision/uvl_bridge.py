"""Export the original UVL CNF using the existing FlamaPy environment."""
import argparse
import json
from pathlib import Path
import time


def compile_uvl(path, cases):
    from flamapy.metamodels.fm_metamodel.transformations.uvl_reader import UVLReader
    from flamapy.metamodels.pysat_metamodel.transformations.fm_to_pysat import FmToPysat
    fm = UVLReader(str(path)).transform()
    sat = FmToPysat(fm).transform()
    leaves = sorted(f.name for f in fm.get_features() if not f.relations)
    from pysat.solvers import Solver
    checks = []
    with Solver(bootstrap_with=sat.get_all_clauses().clauses) as solver:
        for selected in cases:
            valid = len(selected) == len(set(selected)) and set(selected).issubset(leaves) and solver.solve(assumptions=[sat.variables[f] if f in selected else -sat.variables[f] for f in leaves])
            checks.append({'leaves': selected, 'PySAT_reference': bool(valid)})
    return {'variables': sat.variables, 'clauses': sat.get_all_clauses().clauses, 'leaves': leaves, 'reference_cases': checks}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--models', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cases', type=Path, required=True)
    args = p.parse_args()
    start = time.perf_counter(); cpu = time.process_time()
    cases = json.loads(args.cases.read_text(encoding='utf-8'))
    result = {name: compile_uvl(args.models/(name+'.uvl'), cases[name]) for name in ('fm_p', 'fm_d')}
    result['wall_seconds'] = time.perf_counter()-start
    result['CPU_process_seconds'] = time.process_time()-cpu
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
