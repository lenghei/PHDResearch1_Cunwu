"""Count declared static assumptions separately from measured empirical gaps."""
import argparse,json
from pathlib import Path
def analyze(out,model_dir,rules_path):
 from flamapy.metamodels.fm_metamodel.transformations.uvl_reader import UVLReader
 from flamapy.metamodels.pysat_metamodel.transformations.fm_to_pysat import FmToPysat
 from pysat.solvers import Solver
 from vm45_common import save,csv_write
 models={}
 for name in ('fm_p','fm_d'):
  fm=UVLReader(str(model_dir/(name+'.uvl'))).transform();sat=FmToPysat(fm).transform();models[name]=(fm,sat)
 fm,sat=models['fm_p'];leaves=sorted(f.name for f in fm.get_features() if not f.relations)
 assumptions={'STRICT_IDENTITY_ONLY':{'Normal'},'GAIN_DOWNSCALE_APPROX':{'Normal','Overexposed_mild','Underexposed_mild','Downscale_x1_5','Downscale_x2'},
 'SUPERVISOR_WEAK_ALBU_APPROX':{'Normal','Overexposed_mild','Underexposed_mild','Downscale_x1_5','Downscale_x2','Defocus_r3'}}
 counts={k:0 for k in assumptions};total=0
 with Solver(bootstrap_with=sat.get_all_clauses().clauses) as solver:
  while solver.solve():
   assignment=set(solver.get_model());chosen={f for f in leaves if sat.variables[f] in assignment};total+=1
   for k,allowed in assumptions.items():counts[k]+=chosen.issubset(allowed)
   solver.add_clause([-sat.variables[f] if f in chosen else sat.variables[f] for f in leaves])
 assert total==46368
 dead={}
 for name,(fm,sat) in models.items():
  with Solver(bootstrap_with=sat.get_all_clauses().clauses) as solver:dead[name]=[f for f in sat.variables if not solver.solve(assumptions=[sat.variables[f]])]
 original=(model_dir/'fm_p.uvl').read_text(encoding='utf-8')
 for name in ('fm_p_empirical_candidate.uvl','derived_uvl_constraints.uvl'):
  (out/name).write_text(original,encoding='utf-8')
  parsed=FmToPysat(UVLReader(str(out/name)).transform()).transform()
  assert parsed.get_all_clauses().clauses==models['fm_p'][1].get_all_clauses().clauses
 rows=json.loads(rules_path.read_text())['development_rows'];groups=sorted({r['condition_id'] for r in rows});summary=[]
 for key,allowed in assumptions.items():
  for p in groups:
   rs=[r for r in rows if r['condition_id']==p];states=[r['original_rule_state'] for r in rs]
   empirical='COVERED_TESTED_D_SUBSET' if 'PASS' in states else 'GAP_TESTED_D_SUBSET' if set(states)=={'FAIL'} else 'UNRESOLVED'
   summary.append({'assumption':key,'P':p,'static_covered':set(rs[0]['leaves']).issubset(allowed),'empirical_state':empirical,'D_measured':4,'scope':'PURPOSEFUL_DEVELOPMENT_SUBSET_ONLY'})
 csv_write(out/'static_empirical_gap_analysis.csv',summary)
 save(out/'UVL_analysis.json',{'original_P_valid':total,'original_D_valid_REUSED_ACCEPTED':635904,'derived_P_valid':total,'dead_features':dead,'derived_CNF_identical':True,'new_empirical_constraints':0,
 'static_sensitivity':[{'assumption':k,'rule_kind':'TRAINING_COVERAGE_RULE','allowed_leaves':sorted(assumptions[k]),'covered':v,'outside_coverage':total-v,'status':'EXPLICIT_MAPPING_ASSUMPTION_NOT_MEASURED_FAILURE_RATE'} for k,v in counts.items()],
 'unknown_P_outside_measured_grid':total-len(groups),'empirical_full_space_fraction':'NOT_ESTIMABLE','constraint_types':['PHYSICAL_VALIDITY_CONSTRAINT','TRAINING_COVERAGE_RULE','EMPIRICAL_FAILURE_RULE','DEPLOYMENT_REQUIREMENT'],
 'reason_no_constraints_promoted':'Joint P-D classifier paths have insufficient grouped evidence; not physical impossibilities or validated P-only constraints.'})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--models',type=Path,required=True);p.add_argument('--rules',type=Path,required=True);a=p.parse_args();analyze(a.out,a.models,a.rules)
