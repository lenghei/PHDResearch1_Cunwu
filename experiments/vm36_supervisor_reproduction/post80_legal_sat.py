"""Check candidate membership in the original UVL using FlamaPy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def validate(model,design):
    from flamapy.metamodels.fm_metamodel.transformations.uvl_reader import UVLReader
    from flamapy.metamodels.pysat_metamodel.transformations.fm_to_pysat import FmToPysat
    from flamapy.metamodels.pysat_metamodel.operations.pysat_configurations import PySATConfigurations
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cv261109'))
    from model import ALL_LEAVES,valid_leaves,configuration_id
    fm=UVLReader(str(model)).transform()
    configs=PySATConfigurations().execute(FmToPysat(fm).transform()).get_result()
    population={tuple(sorted(getattr(f,'name',str(f)) for f,v in c.elements.items() if v and getattr(f,'name',str(f)) in ALL_LEAVES)) for c in configs}
    if len(population)!=46368 or len(fm.ctcs)!=5:raise ValueError('Original logical population changed.')
    rows=[]
    for case in json.loads(design.read_text(encoding='utf-8'))['candidate_checks']:
        selected=tuple(sorted(case['leaves']));legal=selected in population
        if legal!=valid_leaves(selected) or legal!=case['expected_legal']:raise ValueError('SAT/reference validity differs.')
        rows.append({'name':case['name'],'configuration_id':configuration_id(selected),'legal':legal,'leaves':list(selected)})
    return {'status':'VALIDATED','logical_configurations':len(population),'constraints':len(fm.ctcs),
        'UVL_SHA256':hashlib.sha256(model.read_bytes()).hexdigest(),'cases':rows,
        'engine':'FlamaPy UVLReader, FmToPysat, PySATConfigurations; exact membership, no response input.'}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--design',type=Path,required=True)
    a=p.parse_args();print(json.dumps(validate(a.model,a.design),sort_keys=True))
