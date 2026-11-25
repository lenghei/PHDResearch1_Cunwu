"""Parse the frozen models with the installed validated FlamaPy workflow."""
import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys


def signature(model):
    features=[]
    for f in model.get_features():
        features.append((f.name,f.parent.name if f.parent else None,
            sorted((r.card_min,r.card_max,tuple(c.name for c in r.children)) for r in f.relations),
            sorted((a.name,a.default_value) for a in f.attributes)))
    return sorted(features), sorted(str(c.ast) for c in model.ctcs)


def audit(out, experiment_root):
    from flamapy.metamodels.fm_metamodel.transformations.uvl_reader import UVLReader
    legacy=experiment_root/'vm36_supervisor_reproduction'
    sys.path.insert(0,str(legacy))
    spec=importlib.util.spec_from_file_location('vm40_validated_workflow',legacy/'validate_models.py')
    validation=importlib.util.module_from_spec(spec);spec.loader.exec_module(validation)
    result={}
    for domain,filename in [('FM_P','fm_p.uvl'),('FM_D','fm_d.uvl')]:
        path=legacy/'feature_models'/filename
        model=UVLReader(str(path)).transform()
        parsed_pdf=UVLReader(str(out/('pdf_'+filename))).transform()
        proof=validation.validate(path,domain)
        proof['PDF_tree_attributes_constraints_equal']=signature(model)==signature(parsed_pdf)
        proof['PDF_model_features']=len(parsed_pdf.get_features())
        if not proof['PDF_tree_attributes_constraints_equal']:
            raise ValueError('PDF and repository model signatures differ: '+domain)
        tree=[]
        def visit(f,depth):
            relation=None
            if f.parent:
                relation=next(r for r in f.parent.relations if f in r.children)
            kind='ROOT' if relation is None else 'MANDATORY' if relation.is_mandatory() else 'OPTIONAL' if relation.is_optional() else 'ALTERNATIVE' if relation.is_alternative() else 'OR' if relation.is_or() else 'CARDINALITY'
            tree.append({'domain':domain,'feature':f.name,'parent':f.parent.name if f.parent else '',
                'depth':depth,'relation':kind,'group_min':relation.card_min if relation else 1,
                'group_max':relation.card_max if relation else 1,'leaf':not f.relations,
                'attributes':json.dumps({a.name:a.default_value for a in f.attributes},sort_keys=True),
                'source':str(path),'origin':'ORIGINAL_PDF_DEFINITION'})
            for r in f.relations:
                for child in r.children:visit(child,depth+1)
        visit(model.root,0)
        with (out/(domain+'_feature_tree.csv')).open('x',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(tree[0]));writer.writeheader();writer.writerows(tree)
        proof['tree']=tree
        proof['constraint_count']=len(model.ctcs)
        proof['constraints']=[str(c.ast) for c in model.ctcs]
        result[domain]=proof
    (out/'UVL_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:{x:v[x] for x in ('valid_count','leaves','constraints','satisfiable','dead_features','PDF_tree_attributes_constraints_equal')} for k,v in result.items()}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--experiments',type=Path,required=True);args=parser.parse_args()
    audit(args.out.resolve(strict=True),args.experiments.resolve(strict=True))
