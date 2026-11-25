"""Check complete leaf assignments against the frozen UVL CNF."""
import argparse
import json
from pathlib import Path


def check(model_path, selections):
    from flamapy.metamodels.fm_metamodel.transformations.uvl_reader import UVLReader
    from flamapy.metamodels.pysat_metamodel.transformations.fm_to_pysat import FmToPysat
    from pysat.solvers import Solver
    fm=UVLReader(str(model_path)).transform()
    leaves={f.name for f in fm.get_features() if not f.relations}
    sat=FmToPysat(fm).transform();results=[]
    with Solver(bootstrap_with=sat.get_all_clauses().clauses) as solver:
        for item in selections:
            chosen=set(item['leaves'])
            valid=chosen.issubset(leaves) and solver.solve(assumptions=[sat.variables[f] if f in chosen else -sat.variables[f] for f in sorted(leaves)])
            results.append({'id':item['id'],'SAT_valid':bool(valid),'leaves':sorted(chosen)})
    return results


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=check(args.model,json.loads(args.input.read_text(encoding='utf-8')))
    with args.output.open('x',encoding='utf-8') as stream:json.dump(result,stream,indent=2)
