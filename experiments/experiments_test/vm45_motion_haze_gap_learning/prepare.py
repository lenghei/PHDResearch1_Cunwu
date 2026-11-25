"""Review bindings and prepare metadata without admitting a GPU task."""
import json
from vm45_common import *
def prepare(out,rules,budget):
 from pypdf import PdfReader
 pdf=ROOT/'datasets/research_archive/supervisor/Research_plan_Cunwu_UAV_gap.pdf';ref(pdf)
 pages=[p.extract_text() or '' for p in PdfReader(pdf).pages]
 save(out/'supervisor_plan_read.json',{'source':ref(pdf),'pages':pages,'conflicts':['PDF p15 disallows AI-authored analysis; these outputs are machine-assisted technical evidence for human review, not manuscript prose. User explicitly requests this workflow.','PDF test-dev reporting preference does not establish historical independence; VM43 remains limited exploratory.','PDF daily push conflicts with current user no-push instruction; no Git changes.']})
 for n in ('fm_p.uvl','fm_d.uvl'):ref(UVL/n)
 old=read(V44/'Next_minimal_experiment_protocol.json');cost=read(V44/'Next_experiment_cost_estimate.json')
 generator=HERE.parent/'cv261109/generator.py';ref(generator)
 review={'status':'BLOCKED_SCIENTIFIC_BINDING_ADMISSION','parameter_source':ref(generator),'original_UVL':ref(UVL/'fm_p.uvl'),
 'parameters':[{'name':'kernel','value':21,'source':'Supervisor UVL','unit':'pixels'}, {'name':'beta','value':1.5,'source':'Supervisor UVL','unit':'dimensionless optical-depth proxy; no metric depth calibration'},
 {'name':'motion_angle','proposed_value':'Uniform[0,180) degrees','source':'Historical implementation proposal','approved':False,'assumption':'Orientation-averaged linear PSF, not measured UAV motion'},
 {'name':'airlight','proposed_value':220,'source':'Historical implementation proposal','approved':False,'assumption':'Same scalar BGR8 airlight; constant transmission exp(-beta), no depth map'}],
 'validation_evidence':ref(V44/'future_pixel_fixture_tests.json'),'physical_model_basis':'Saved generator I=J exp(-beta)+A(1-exp(-beta)); no saved calibration proves real-weather validity.',
 'remaining_limitations':['No independent scientific admission for nuisance values','No supervisor approval invented','No actual depth distribution','Identity coordinates preserve boxes/ignore, not visibility','Not the VM38 Gaussian training range'],
 'full_cache_status':'BLOCKED_SCIENTIFIC_BINDING_ADMISSION; no548-image cache generated with unadmitted bindings','RollingShutter':'BLOCKED_UNVERIFIED_BOX_MAPPING'}
 save(out/'parameter_binding_review.json',review)
 manifest=[]
 for p in old['P']:
  for d in old['D']:manifest.append({'P_id':p['id'],'leaves':'|'.join(p['leaves']),'D':d,'images':548,'status':'PROPOSED_NOT_MEASURED','FP32':True,'resolution':1280,'batch':1})
 csv_write(out/'configuration_manifest.csv',manifest)
 normal=next(r for r in rules['development_rows'] if r['leaves']==['Normal'] and r['D']=='ck01_Infer1280')
 path=Path(normal['source_folder']); records=[json.loads(x) for x in (path/'image_manifest.jsonl').read_text().splitlines()];ref(path/'image_manifest.jsonl');ref(path/'ground_truth.json')
 assert len(records)==548 and len({r['source']['ZIP_member'] for r in records})==548
 csv_write(out/'frozen_development_members.csv',[{'image_id':r['image_id'],'ZIP_member':r['source']['ZIP_member'],'image_SHA256':r['source']['image_SHA256'],'annotation_SHA256':r['source']['annotation_SHA256'],'shape':r['shape'],'GT_status':'REUSED_ACCEPTED_ORIGINAL_COORDINATES'} for r in records])
 gate={'status':'BLOCKED','scientific_binding_admission':False,'VM45_specific_GPU_authorization':False,'authorization_search_scope':'Current VM45 user instruction explicitly requires separate applicable permission; no such record supplied. VM43/old ledger not reused as permission.',
 'budget_reporting_only':budget,'complete_cost_ESTIMATE':cost,'thermal':'NOT_PERFORMED_NO_GPU_AUTHORIZATION','GPU_probe_executed':False,'Normal_reuse':'NOT_ADMITTED_FOR_NEW_MATCHED_TIMING; old raw AP remains historical only','cache':'NOT_GENERATED_UNADMITTED_BINDINGS','GPU_seconds':0}
 save(out/'gpu_admission.json',gate)
 save(out/'experiment_protocol.json',{**old,'VM45_scope':'CPU_STUDY_ONLY_GPU_BLOCKED','new_GPU_authorization':False,'scientific_bindings':'PROPOSED_NOT_ADMITTED','bootstrap':'200 shared image draws for future real8-cell evaluation; NOT_PERFORMED','no_external_labels_for_learning':True})
 return gate
