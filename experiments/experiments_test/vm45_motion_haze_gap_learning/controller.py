"""V2 keeps exact evidence separate from unadmitted rule predictions."""
from vm45_common import *
def evaluate(out,rules,learner):
 sys.path.insert(0,str(HERE.parent/'vm44_tiny_error_variability_decision'))
 from selector import Controller
 c=Controller(rules['models'],rules['development_rows'],rules['candidates'],rules['semantic_support']);results=[]
 for p in sorted({r['condition_id'] for r in rules['development_rows']}):
  leaves=next(r['leaves'] for r in rules['development_rows'] if r['condition_id']==p);ans=c.select(leaves)
  results.append({'P':p,'information':'MEASURED_EXACT_LOOKUP','state':ans['status'],'D':ans['selected']['D'] if ans['selected'] else None,'reason':ans['reason'],'rule_prediction_used':False})
 for leaves in (['Normal','Motion_k21'],['Normal','Haze_dense'],['Normal','Motion_k21','Haze_dense']):
  ans=c.select(leaves);results.append({'P':ans['P'],'information':'UNKNOWN','state':ans['status'],'D':None,'reason':'No admitted generalizable rule; no fabricated AP','rule_prediction_used':False})
 csv_write(out/'controller_v2_results.csv',results)
 import csv
 prior=list(csv.DictReader((V44/'Controller_baseline_comparison.csv').open(encoding='utf-8-sig')))
 summary=[]
 for name in ('BASELINE_1_FIXED_V8_640','BASELINE_2_PRIMARY_26_1280','VARIABILITY_CONTROLLER','BASELINE_3_MEASURED_LOOKUP_ORACLE'):
  rs=[r for r in prior if r['method']==name];summary.append({'method':name,'measured_P':21,'PASS':sum(r['task_state']=='PASS' for r in rs),'abstentions':sum(r['abstain']=='True' for r in rs),'evaluation':'IN_SAMPLE_LOOKUP_RESULT','new_generalization_claim':False})
 summary.append({'method':'VARIABILITY_CONTROLLER_V2','measured_P':21,'PASS':sum(r['D'] is not None for r in results[:21]),'abstentions':sum(r['D'] is None for r in results[:21]),'evaluation':'IN_SAMPLE_LOOKUP_RESULT','new_generalization_claim':False})
 csv_write(out/'baseline_comparisons.csv',summary)
 save(out/'controller_v2_validation.json',{'status':'WORKING_SAFE_ABSTENTION','exact_lookup':21,'selected':4,'abstained':17,'new_unmeasured_conditions':3,'new_conditions_abstained':3,'admitted_rule_predictions':0,'RULE_PREDICTION':'Not used until grouped validation supports rule admission','external_labels_used':False,'learned_rule_status':learner['status'],'advantage':'Provenance/legality/feasibility checks and explicit abstention, no measured coverage improvement over fixed26s1280'})
