"""Validate PDF-aligned metadata without weakening the legacy v1 guard."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def boolean_text(text):
    stripped=re.sub(r'\s+\{(?=[^{}]*\d)[^{}]*\}','',text)
    return tuple(line.rstrip() for line in stripped.splitlines() if line.strip())


def validate(path,domain,seed=20261109):
    from flamapy.metamodels.fm_metamodel.transformations.uvl_reader import UVLReader
    from flamapy.core.discover import DiscoverMetamodels
    experiments=Path(__file__).resolve().parent.parent
    legacy=experiments/('cv261109' if domain=='FM_P' else 'cv261116')
    original=legacy/('fm_p.uvl' if domain=='FM_P' else 'fm_d.uvl')
    if boolean_text(path.read_text(encoding='utf-8'))!=boolean_text(original.read_text(encoding='utf-8')):
        raise ValueError('Boolean tree or constraints changed; previous factorization cannot be reused.')
    fm=UVLReader(str(path)).transform();dm=DiscoverMetamodels()
    sat=dm.use_operation_from_file('PySATSatisfiable',str(path))
    dead=dm.use_operation_from_file('PySATDeadFeatures',str(path))
    if sat is not True or not isinstance(dead,(list,tuple,set)) or dead:raise ValueError('SAT/dead-feature check failed.')
    leaves={f.name for f in fm.get_features() if not f.relations}
    if domain=='FM_P':
        sys.path.insert(0,str(legacy))
        old=module('vm36_original_fmp_selector',legacy/'flamapy_select.py')
        result=old.select(path,seed)
    else:
        old=module('vm36_guarded_fmd_count',legacy/'flamapy_count.py')
        old.validate_uvl_contract(original)
        # The original guard remains intact; only the numeric-free Boolean projection is transferred.
        if old.canonical_uvl(original.read_text(encoding='utf-8'))!=boolean_text(path.read_text(encoding='utf-8')):
            raise ValueError('Aligned model is not an attributes-only extension of the guarded proof.')
        result=old.factorized_count()
        result['count_method']='EXACT_FACTORIZATION_TRANSFER_AFTER_STRICT_BOOLEAN_PROJECTION_EQUALITY'
    expected=(25,5,46368) if domain=='FM_P' else (34,4,635904)
    if (len(leaves),len(fm.ctcs),result['valid_count'])!=expected:raise ValueError('Model counts differ.')
    result.update(status='VALIDATED',satisfiable=sat,dead_features=[],leaves=len(leaves),constraints=len(fm.ctcs),
        UVL_SHA256=hashlib.sha256(path.read_bytes()).hexdigest(),
        method_limitation='SAT checks genuinely executed; no new independent BDD/#SAT count is claimed.',
        legacy_guard_modified=False,Boolean_projection_identical=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--domain',choices=('FM_P','FM_D'),required=True)
    p.add_argument('--seed',type=int,default=20261109);a=p.parse_args()
    print(json.dumps(validate(a.model.resolve(strict=True),a.domain,a.seed),sort_keys=True))
