"""Verify VM45 cells and derive paired contrasts without new inference."""
import argparse,ast,csv,gzip,hashlib,html,json,math,os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
import sys,time,subprocess
import numpy as np
from vm45_common import ROOT,HERE,V44,read,ref,save,csv_write,sha
sys.path.insert(0,str(HERE.parent/'vm43_external_pilot'))

RUNS=ROOT/'datasets/local_runs/vm45_motion_haze_gap_learning'

def paired(point,draws):
 if point is None:return None,[None,None],0
 valid=np.array([x for x in draws if x is not None],dtype=float)
 return point,np.percentile(valid,[2.5,97.5]).tolist() if len(valid) else [None,None],len(valid)

def finish(identity):
 start=time.perf_counter();cpu=time.process_time()
 if Path(identity).name!=identity or not identity.startswith(HERE.name+'_execution_'):raise ValueError('Expected a VM45 execution RunID.')
 out=(RUNS/identity).resolve(strict=True)
 if (out/'final_acceptance.json').exists():raise FileExistsError('Final receipt already exists; do not overwrite.')
 save(out/'postprocessing_command.json',{'actual_command':sys.orig_argv,'GPU':False,'source':ref(__file__)})
 save(out/'bootstrap_CPU_regression_receipt.json',{'status':'PASS_PREVIOUSLY_OBSERVED_THIS_EXECUTION','test_source':ref(HERE/'tests/test_execution_backend.py'),
  'command':[str(ROOT/'.venv_rq4/Scripts/python.exe'),'-B','-X','utf8','-m','unittest','discover','-s',str(HERE/'tests'),'-p','test_execution_backend.py','-v'],
  'observation':'Ran1 test in0.963s; OK. Three synthetic CPU images,200 area draws exactly match accepted reference; class1 equivalence and absent-class null checked.',
  'evidence_origin':'Initial execution tool transcript; carried forward without rerunning successful test','separate_test_execution_UTC':'NOT_RECORDED','GPU':False})
 spec=read(out/'experiment_protocol.json');cells={};receipts=[];metrics_rows=[];class_rows=[];gap_rows=[];timings=[];intervals=[]
 complete=(out/'execution_end.json').exists() and read(out/'execution_end.json')['status']=='REAL_EIGHT_CELLS_COMPLETED'
 preservation=[]
 if (out/'execution_source_provenance.json').exists():
  for entry in read(out/'execution_source_provenance.json')['files']:
   actual=sha(entry['path']);assert actual==entry['SHA256'],entry['path']
   preservation.append({'path':entry['path'],'SHA256':actual,'status':'UNCHANGED'})
 save(out/'input_preservation_verification.json',{'files':preservation,'scope':'Frozen registered input files; no claim about unregistered filesystem contents'})
 indices=np.random.default_rng(spec['bootstrap_seed']).integers(0,548,(200,548))
 gt=read(spec['GT']['path']);assert sha(spec['GT']['path'])==spec['GT']['SHA256']
 cache=read(out/'pixel_cache_manifest.json') if (out/'pixel_cache_manifest.json').exists() else {}
 for entries in cache.values():
  for entry in entries:assert sha(entry['path'])==entry['file_SHA256']
 for condition in spec['conditions']:
  for detector in spec['models']:
   cid=condition['condition']+'__'+detector['D'];folder=out/cid
   if not (folder/'acceptance.json').exists():continue
   a=read(folder/'acceptance.json');m=read(folder/'metrics.json');b=read(folder/'bootstrap.json');t=read(folder/'timing.json')
   assert a['images']==m['images']==548 and a['GT']==spec['GT']
   for field in ('raw','GT','metrics','bootstrap','checkpoint'):
    p=a[field];assert sha(p['path'])==p['SHA256']
   assert a['checkpoint']['SHA256']==detector['checkpoint']['sha256']
   manifest=[json.loads(x) for x in (folder/'image_manifest.jsonl').read_text().splitlines()]
   assert [r['image_id'] for r in manifest]==[r['id'] for r in gt['records']]
   for row,pixel in zip(manifest,cache[condition['condition']]):
    assert {k:v for k,v in row.items() if k!='tensor'}==pixel
    assert row['tensor']=={'NCHW':[1,3,1280,1280],'dtype':'torch.float32'}
   assert b['draws']['image_ids']==[r['id'] for r in gt['records']] and b['draws']['samples']==200 and b['draws']['seed']==spec['bootstrap_seed']
   cells[detector['D'],condition['condition']]=(m,b,t)
   receipts.append({'cell_id':cid,'images':548,'raw':a['raw'],'checkpoint':a['checkpoint'],'GT':a['GT'],'status':'SHA_TENSOR_PIXEL_POPULATION_VERIFIED'})
   for r in m['per_area']:metrics_rows.append({'cell_id':cid,'D':detector['D'],'condition':condition['condition'],'P_id':condition['P_id'],'scope':spec['scientific_scope'],**r})
   for r in m['per_class']:class_rows.append({'cell_id':cid,'D':detector['D'],'condition':condition['condition'],**r})
   for r in b['intervals']:intervals.append({'cell_id':cid,**r})
   lat=np.array([r['cached_pipeline_ms'] for r in t['per_image']]);lat_ci=np.percentile(lat[indices].mean(axis=1),[2.5,97.5]).tolist()
   tiny=next(r for r in b['intervals'] if r['kind']=='area' and r['size']=='tiny' and r['metric']=='map50_95')
   q=tiny['ci95'];state='FAIL' if lat_ci[1]>100 or q[1]<.1 else 'PASS' if q[0]>=.1 else 'UNCERTAIN'
   gap_rows.append({'cell_id':cid,'P_id':condition['P_id'],'D':detector['D'],'condition':condition['condition'],'Tiny_AP':tiny['estimate'],'Tiny_CI_low':q[0],'Tiny_CI_high':q[1],
    'cached_pipeline_mean_ms':float(lat.mean()),'latency_CI_low_ms':lat_ci[0],'latency_CI_high_ms':lat_ci[1],'state':state})
   for r in t['per_image']:timings.append({'cell_id':cid,**r})
 assert not complete or len(cells)==8
 effects=[];cross=[]
 def endpoint(d,condition,area,metric,category=None):
  m,b,t=cells[d,condition]
  base=m['per_area'] if category is None else m['per_class'];r=next(r for r in base if r['size']==area and (category is None or r['category_id']==category))
  values=b['draws']['values'][area][metric] if category is None else b['draws']['class_values'][area+':'+str(category)][metric]
  return r[metric],values
 if complete:
  for d in [r['D'] for r in spec['models']]:
   for area in ('all','tiny','small','medium','large','coco_small'):
    for category in (None,*range(1,11)):
     for metric in ('map50','map50_95','ar50','ar50_95'):
      corners=[endpoint(d,c,area,metric,category) for c in ('Normal','Motion_k21','Haze_dense','Motion_Haze')]
      for name,weights in [('Motion_effect',[-1,1,0,0]),('Haze_effect',[-1,0,1,0]),('Interaction_DID',[1,-1,-1,1])]:
       point=sum(w*x[0] for w,x in zip(weights,corners)) if all(x[0] is not None for x in corners) else None
       draws=[sum(w*x[1][i] for w,x in zip(weights,corners)) if all(x[1][i] is not None for x in corners) else None for i in range(200)]
       value,ci,n=paired(point,draws)
       effects.append({'D':d,'size':area,'category_id':category,'metric':metric,'effect':name,'effect_raw':value,'effect_pp':value*100 if value is not None else None,
        'interval_lower':ci[0],'interval_upper':ci[1],'paired_valid_draws':n,'number_of_images':548,'scope':spec['scientific_scope'],'status':'ESTIMATED_POINTWISE_ONE_HISTORICAL_CHECKPOINT' if n else 'NOT_ESTIMABLE'})
      left=[endpoint('ck05_Infer1280',c,area,metric,category) for c in ('Normal','Motion_k21','Haze_dense','Motion_Haze')]
      right=[endpoint('ck01_Infer1280',c,area,metric,category) for c in ('Normal','Motion_k21','Haze_dense','Motion_Haze')]
      if d=='ck01_Infer1280':
       weights=[1,-1,-1,1]
       point=sum(w*(g[0]-c[0]) for w,c,g in zip(weights,left,right)) if all(x[0] is not None for x in left+right) else None
       draw=[sum(w*(g[1][i]-c[1][i]) for w,c,g in zip(weights,left,right)) if all(x[1][i] is not None for x in left+right) else None for i in range(200)]
       value,ci,n=paired(point,draw)
       cross.append({'contrast':'DID_ck01_minus_DID_ck05','size':area,'category_id':category,'metric':metric,'effect_raw':value,'effect_pp':value*100 if value is not None else None,'interval_lower':ci[0],'interval_upper':ci[1],
        'paired_valid_draws':n,'number_of_images':548,'scope':spec['scientific_scope'],'interpretation':'Checkpoint contrast; no architecture-causal or real-weather claim'})
  csv_write(out/'matched_evaluation_metrics.csv',metrics_rows);csv_write(out/'per_class_metrics.csv',class_rows);csv_write(out/'paired_bootstrap_intervals.csv',intervals)
  csv_write(out/'factorial_interactions.csv',effects);csv_write(out/'cross_detector_interactions.csv',cross);csv_write(out/'gap_status.csv',gap_rows);csv_write(out/'timing_by_image.csv',timings)
  latency_effects=[];latency_draws={};weights_by_effect={'Motion_effect':[-1,1,0,0],'Haze_effect':[-1,0,1,0],'Interaction_DID':[1,-1,-1,1]}
  for d in [r['D'] for r in spec['models']]:
   arrays=[np.array([r['cached_pipeline_ms'] for r in cells[d,c][2]['per_image']]) for c in ('Normal','Motion_k21','Haze_dense','Motion_Haze')]
   for name,weights in weights_by_effect.items():
    values=sum(w*a for w,a in zip(weights,arrays));draws=values[indices].mean(axis=1);latency_draws[d,name]=draws
    ci=np.percentile(draws,[2.5,97.5]);latency_effects.append({'D':d,'effect':name,'mean_difference_ms':float(values.mean()),'CI_low_ms':float(ci[0]),'CI_high_ms':float(ci[1]),'samples':200,
     'scope':'Paired-image conditional timing contrast; fixed execution order, no repeated-machine-state uncertainty'})
  draws=latency_draws['ck01_Infer1280','Interaction_DID']-latency_draws['ck05_Infer1280','Interaction_DID'];ci=np.percentile(draws,[2.5,97.5])
  latency_effects.append({'D':'ck01_minus_ck05','effect':'Cross_detector_DID','mean_difference_ms':next(r['mean_difference_ms'] for r in latency_effects if r['D']=='ck01_Infer1280' and r['effect']=='Interaction_DID')-next(r['mean_difference_ms'] for r in latency_effects if r['D']=='ck05_Infer1280' and r['effect']=='Interaction_DID'),
   'CI_low_ms':float(ci[0]),'CI_high_ms':float(ci[1]),'samples':200,'scope':'Conditional image interval; not hardware causal effect'})
  csv_write(out/'latency_interactions.csv',latency_effects)
  local=[]
  for condition in spec['conditions']:
   selected=[r for r in gap_rows if r['condition']==condition['condition']]
   local.append({'P_id':condition['P_id'],'condition':condition['condition'],'measured_D':2,'candidate_scope':'ck01/ck05 Infer1280 only',
    'local_status':'COVERED' if any(r['state']=='PASS' for r in selected) else 'GAP' if all(r['state']=='FAIL' for r in selected) else 'UNCERTAIN',
    'status_over_original_four_D':'NOT_FULLY_MEASURED_NEW_P' if condition['condition']!='Normal' else 'MIXED_TIMING_HISTORICAL_AND_NEW',
    'full_space_prevalence':'NOT_ESTIMABLE'})
  csv_write(out/'local_P_gap_status.csv',local)
  export_updated_evidence(out,spec,cells,gap_rows)
  learner=HERE/'learn_updated_execution.py'
  command=[str(ROOT/'.venv_flamapy/Scripts/python.exe'),'-B','-X','utf8',str(learner),'--out',str(out)]
  save(out/'updated_learning_command.json',{'command':command,'source':ref(learner),'budget_seconds':600,'old84_only_refit':False})
  with (out/'updated_learning.log').open('x',encoding='utf-8') as log:
   try:
    child=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600)
    if child.returncode:save(out/'updated_learning_failure.json',{'returncode':child.returncode,'retry':False,'real_evaluation_preserved':True})
   except subprocess.TimeoutExpired:
    save(out/'updated_learning_failure.json',{'stop_reason':'CPU_DIAGNOSTIC_600S_CAP','retry':False,'real_evaluation_preserved':True})
 from vm43_common import budget
 resource={'budget_after':budget(),'GPU_phase_charges':[],'CPU_cache':read(out/'dataset_gate.json') if (out/'dataset_gate.json').exists() else None,'kernel_busy_seconds':'NOT_MEASURED',
  'unit':'Supervised GPU-worker wall includes CPU COCO/Bootstrap; CPU-only cache and postprocessing recorded separately','new_training':False,'retries':0}
 for phase in ('preflight','formal'):
  f=out/phase/'safety_supervisor.json'
  if f.exists():resource[phase]=read(f)
  charge=ROOT/'datasets/local_runs/vm37_yolo26_rq4/vm37_yolo26_rq4_pilot10_authorization_cpu_20261113T212036_50286ea7/charges'/(out.name+'_'+phase+'.json')
  if charge.exists():resource['GPU_phase_charges'].append(read(charge))
 resource['total_charged_seconds']=sum(r['charged_seconds'] for r in resource['GPU_phase_charges'])
 telemetry=[]
 for phase in ('preflight','formal'):
  path=out/phase/'gpu_telemetry.jsonl'
  if not path.exists():continue
  values=[json.loads(line) for line in path.read_text().splitlines()]
  telemetry.append({'phase':phase,'records':len(values),'max_temperature_C':max(v['temperature_C'] for v in values),
   'max_used_MiB':max(v['used_MiB'] for v in values),
   'max_accepted_record_gap_seconds':max((b['monotonic_seconds']-a['monotonic_seconds'] for a,b in zip(values,values[1:])),default=None),
   'raw':ref(path),'note':'nvidia-smi sampling wall includes query time; safety stop receipt is authoritative'})
 resource['telemetry_summary']=telemetry
 resource['cell_costs']=[{'D':d,'condition':c,'cached_pipeline_seconds':sum(r['cached_pipeline_ms'] for r in t['per_image'])/1000,
  'CPU_evaluation_cost':t['evaluation_cost'],'peak_torch_allocated_bytes':t['peak_allocated_bytes'],'peak_torch_reserved_bytes':t['peak_reserved_bytes']}
  for (d,c),(m,b,t) in cells.items()]
 resource['formal_accepted_image_records']=len(cells)*548
 resource['engineering_probe_image_records']=len(read(out/'preflight_result.json')['checks']) if (out/'preflight_result.json').exists() else 0
 resource['completed_programmed_warmups']=30 if complete else 'See completed worker stages; no inferred complete count'
 reports=report(out,spec,cells,effects,cross,gap_rows,complete,resource)
 save(out/'execution_protocol.json',{'frozen_protocol':ref(out/'experiment_protocol.json'),'author':ref(out/'author_authorization.json'),'source_RunID':out.name,'classification':'DEVELOPMENT_ONLY_SYNTHETIC_EXPLORATORY'})
 save(out/'source_provenance.json',{'verified_cells':receipts,'original_inputs':read(out/'execution_source_provenance.json') if (out/'execution_source_provenance.json').exists() else None,
  'postprocessor':ref(__file__),'external_testdev_used_for_learning':False,'original_files_modified':False})
 resource['CPU_postprocessing_wall_seconds_before_seal']=time.perf_counter()-start;resource['CPU_postprocessing_process_seconds']=time.process_time()-cpu
 save(out/'actual_resource_usage.json',resource)
 save(out/'final_acceptance.json',{'status':'REAL_EIGHT_CELLS_COMPLETED_WITH_LIMITATIONS' if complete else 'INCOMPLETE','verified_cells':len(cells),'formal_images':len(cells)*548,
  'bootstrap':{'samples':200,'shared_image_indices':True,'class_and_area_intervals':True},'new_rule_promotions':0,'scope':spec['scientific_scope'],
  'reports':[str(p) for p in reports],'resource':ref(out/'actual_resource_usage.json'),'postprocessing_GPU':False,'retry':False})
 r=read(out/'run.json');r.update(status='COMPLETE_WITH_LIMITATIONS' if complete else 'INCOMPLETE');(out/'run.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
 (out/'execution_diary.md').write_text('VM45 author-approved synthetic exploration. Source-bound cache and8 cells, fixed order and original checkpoints. No training. Actual gate/stop records retained. Reported DID is paired pooled-AP contrast, not causal weather or architecture evidence. Unsafe rules remain rejected.\n',encoding='utf-8')
 links=''.join(f'<li><a href="{p.as_uri()}">{p.name}</a></li>' for p in reports)
 links+=''.join(f'<li><a href="{name}">{name}</a></li>' for name in ('matched_evaluation_metrics.csv','factorial_interactions.csv','cross_detector_interactions.csv','gap_status.csv','local_P_gap_status.csv','controller_v2_measured_results.csv','updated_rule_validation.json','actual_resource_usage.json','source_provenance.json','final_acceptance.json') if (out/name).exists())
 (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>VM45 real execution</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto}img{max-width:100%}</style><h1>VM45 real execution</h1><p>'+out.name+'</p><p>Verified cells:'+str(len(cells))+'/8. DEVELOPMENT_ONLY synthetic exploration.</p><ul>'+links+'</ul><p>Raw predictions, metrics, class intervals, timing and source hashes are retained in this Run.</p><img src="plots/real_AP.png"><img src="plots/real_DID.png">',encoding='utf-8')
 from main.visualization import build_run_index
 build_run_index(out.parent,HERE.name)
 files=[p for p in out.rglob('*') if p.is_file() and 'runtime' not in p.relative_to(out).parts]+reports+[Path(__file__).resolve(),HERE/'learn_updated_execution.py']
 with (out/'SHA256SUMS.txt').open('x',encoding='utf-8') as f:
  for file in sorted(files):f.write(sha(file)+'  '+str(file)+'\n')
 print(json.dumps({'RunID':out.name,'cells':len(cells),'status':'COMPLETE_WITH_LIMITATIONS' if complete else 'INCOMPLETE','reports':[str(p) for p in reports],'budget_remaining':resource['budget_after']['remaining_seconds']},ensure_ascii=False),flush=True)

def export_updated_evidence(out,spec,cells,gaps):
 old=read(V44/'FM_P_to_FM_D_decision_rules.json');observations=[]
 for r in old['development_rows']:observations.append({**r,'timing_protocol':'VM36_HISTORICAL_CACHED','measurement_role':'HISTORICAL'})
 new=[]
 for row in gaps:
  m=cells[row['D'],row['condition']][0];allm=next(r for r in m['per_area'] if r['size']=='all')
  r={**row,'condition_id':row['P_id'],'leaves':next(r['leaves'] for r in spec['conditions'] if r['condition']==row['condition']),
   'original_rule_state':row['state'],'evidence_scope':'DEVELOPMENT_ONLY','source_folder':str(out/(row['condition']+'__'+row['D'])),'AP50':allm['map50'],'AP50_95':allm['map50_95'],
   'timing_protocol':'VM41_MATCHED_CACHED_GAP_V1','measurement_role':'NEW_REAL_VM45'}
  new.append(r)
 csv_write(out/'development_observation_index.csv',observations+new)
 newkeys={(r['condition_id'],r['D']) for r in new};canonical=[r for r in observations if (r['condition_id'],r['D']) not in newkeys]+new
 csv_write(out/'development_canonical_index.csv',canonical)
 tree_source=RUNS/'vm45_motion_haze_gap_learning_cpu_study_20261117T110831_36b2fc57/in_sample_tree.txt'
 assert tree_source.read_text().splitlines()==['|--- Normal <= 0.50','|   |--- class: 1','|--- Normal >  0.50','|   |--- D_ck05_Infer640 <= 0.50','|   |   |--- class: 0','|   |--- D_ck05_Infer640 >  0.50','|   |   |--- class: 1']
 # Evaluate the already frozen shallow-tree decision, without refitting old84 rows.
 predictions=[]
 for r in new:
  predicted='FAIL' if 'Normal' not in r['leaves'] or r['D']=='ck05_Infer640' else 'PASS'
  predictions.append({'condition':r['condition'],'P':r['condition_id'],'D':r['D'],'predicted_state':predicted,'actual_state':r['state'],
   'unseen_P':r['condition']!='Normal','scope':'FROZEN_TREE_NEW_CONDITION_DIAGNOSTIC_SAME_DEVELOPMENT_IMAGES','admitted_for_selection':False})
 csv_write(out/'frozen_rule_new_condition_results.csv',predictions)
 unseen=[r for r in predictions if r['unseen_P'] and r['actual_state']!='UNCERTAIN']
 fp=sum(r['predicted_state']=='PASS' and r['actual_state']=='FAIL' for r in unseen)
 actual_fail=sum(r['actual_state']=='FAIL' for r in unseen);predicted_pass=sum(r['predicted_state']=='PASS' for r in unseen)
 controller=[]
 for c in spec['conditions']:
  available=[r for r in gaps if r['condition']==c['condition']]
  feasible=sorted([r for r in available if r['state']=='PASS'],key=lambda r:(r['cached_pipeline_mean_ms'],-r['Tiny_CI_low'],r['D']))
  for name in ('MEASURED_EXACT_LOOKUP','RULE_BASED_SAFE_RELEASE','FIXED_ck05_Infer1280','FIXED_ck01_Infer1280'):
   choice=feasible[0] if name=='MEASURED_EXACT_LOOKUP' and feasible else next((r for r in available if name=='FIXED_'+r['D']),None)
   controller.append({'condition':c['condition'],'controller':name,'D':choice['D'] if choice else 'ABSTAIN',
    'measured_state':choice['state'] if choice else 'ABSTAIN','interval_certified_coverage':bool(choice and choice['state']=='PASS'),
    'cached_pipeline_mean_ms':choice['cached_pipeline_mean_ms'] if choice else None,
    'scope':'Retrospective4-P measured subset; lookup is not heldout generalization'})
 csv_write(out/'controller_v2_measured_results.csv',controller)
 save(out/'rule_and_controller_update.json',{'raw_observations':92,'unique_P_D':len(canonical),'P_conditions':len({r['condition_id'] for r in canonical}),
  'duplicate_policy':'New Normal1280 measurements supersede2 old Normal1280 rows in canonical index only; original observations remain readonly',
  'timing_cohorts':'Kept explicit; legacy84 and VM41 timing boundaries not silently treated as identical',
  'old84_classifier_refit':False,'new_rules_admitted':0,'frozen_tree_source':ref(tree_source),'RF_new_condition_prediction':'NOT_EVALUATED; old estimator not serialized, no prohibited old84 refit',
  'frozen_tree_unseen_conditions':{'scored_rows':len(unseen),'actual_FAIL_predicted_PASS':fp,'actual_FAIL_denominator':actual_fail,
   'false_PASS_rate':fp/actual_fail if actual_fail else None,'PASS_prediction_denominator':predicted_pass,
   'PASS_prediction_precision':(predicted_pass-fp)/predicted_pass if predicted_pass else None},
  'independent_safety_admission':'NOT_ESTABLISHED; no independent heldout image/scene set or preregistered supported false-PASS bound',
  'controller':'MEASURED_EXACT_LOOKUP can use new cells that actually pass fixed Tiny/latency intervals; rule-based release remains ABSTAIN',
  'new_measured_feasible_cells':sum(r['state']=='PASS' for r in gaps),'new_frozen_rule_actual_FAIL_predicted_PASS':sum(r['predicted_state']=='PASS' and r['actual_state']=='FAIL' and r['unseen_P'] for r in predictions),
  'external_testdev_for_training':False})

def report(out,spec,cells,effects,cross,gaps,complete,resource):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 (out/'plots').mkdir()
 table=lambda hs,rs:'\n'.join(['| '+' | '.join(hs)+' |','| '+' | '.join(['---']*len(hs))+' |',*['| '+' | '.join(str(v) for v in r)+' |' for r in rs]])
 learning='未完成新增开发证据分组诊断；失败日志保留，不能推广规则。'
 if (out/'updated_rule_validation.json').exists():
  models=read(out/'updated_rule_validation.json')['models']
  fmt=lambda value:'NA' if value is None else f'{value:.4f}'
  learning=table(['新增90行诊断','False-PASS/实际FAIL','预测PASS精确率','平衡准确率','覆盖率','弃权率'],[[m['method'],str(m['actual_FAIL_predicted_PASS'])+'/'+str(m['reference_FAIL']),fmt(m['PASS_prediction_precision']),fmt(m['balanced_accuracy']),fmt(m['prediction_coverage']),fmt(m['abstention_rate'])] for m in models])
 if (out/'rule_and_controller_update.json').exists():
  risk=read(out/'rule_and_controller_update.json')['frozen_tree_unseen_conditions']
  learning+='\n\n冻结旧树在3个新增P×2D上的迁移诊断：'+json.dumps(risk,ensure_ascii=False)+'。仅同548图像、无新环境安全准入。'
 summary=[]
 if complete:
  for c in spec['conditions']:
   for d in spec['models']:
    m=cells[d['D'],c['condition']][0];a={r['size']:r for r in m['per_area']};g=next(r for r in gaps if r['D']==d['D'] and r['condition']==c['condition'])
    summary.append([c['condition'],d['D'],f"{a['all']['map50']*100:.3f}",f"{a['all']['map50_95']*100:.3f}",f"{a['tiny']['map50_95']*100:.3f}",f"{a['all']['ar50_95']*100:.3f}",f"{g['cached_pipeline_mean_ms']:.3f}",g['state']])
  fig,ax=plt.subplots(figsize=(9,5))
  for j,d in enumerate(spec['models']):
   vals=[next(r['map50_95'] for r in cells[d['D'],c['condition']][0]['per_area'] if r['size']=='all')*100 for c in spec['conditions']]
   ax.bar(np.arange(4)+(j-.5)*.35,vals,.33,label=d['D'])
  ax.set(xticks=range(4),xticklabels=[c['condition'] for c in spec['conditions']],ylabel='AP50:95 (%)',title='Real Motion x Haze synthetic development experiment');ax.legend();fig.tight_layout();fig.savefig(out/'plots/real_AP.png',dpi=180);fig.savefig(out/'plots/real_AP.pdf');plt.close(fig)
  selected=[r for r in effects if r['size']=='all' and r['category_id'] is None and r['metric']=='map50_95']
  fig,ax=plt.subplots(figsize=(9,5))
  for i,r in enumerate(selected):
   ax.hlines(i,r['interval_lower']*100,r['interval_upper']*100,color='#27867d');ax.plot(r['effect_pp'],i,'o',color='#27867d')
  ax.axvline(0,color='gray');ax.set(yticks=range(len(selected)),yticklabels=[r['D']+' / '+r['effect'] for r in selected],xlabel='Effect (pp), paired image95% interval',title='Fixed checkpoint contrasts; not architecture causality');fig.tight_layout();fig.savefig(out/'plots/real_DID.png',dpi=180);fig.savefig(out/'plots/real_DID.pdf');plt.close(fig)
  effect_table=table(['D','效应','AP95差pp','配对区间pp'],[[r['D'],r['effect'],f"{r['effect_pp']:+.3f}",f"[{r['interval_lower']*100:.3f},{r['interval_upper']*100:.3f}] (n={r['paired_valid_draws']})"] for r in selected])
  cr=next(r for r in cross if r['size']=='all' and r['category_id'] is None and r['metric']=='map50_95')
  effect_table+=f"\n\n跨checkpoint DID（ck01−ck05）：{cr['effect_pp']:+.3f}pp，区间[{cr['interval_lower']*100:.3f},{cr['interval_upper']*100:.3f}]pp。\n"
 else:
  effect_table='NOT_EVALUATED：八格未全部完成，不发布完整主比较。'
  for name in ('real_AP','real_DID'):
   fig,ax=plt.subplots();ax.axis('off');ax.text(.1,.5,'INCOMPLETE / NOT_EVALUATED');fig.savefig(out/'plots'/(name+'.png'));plt.close(fig)
 gate_text='成功完成所有八格' if complete else '不完整；停止原因见execution_failure.json及阶段监督日志'
 body=f'''# VM45 Motion×Haze真实执行科学报告

RunID：`{out.name}`。{gate_text}。原548张开发图、两历史checkpoint、原图native10 GT/ignore、FP32 batch1 Infer1280，DEVELOPMENT_ONLY_SYNTHETIC_EXPLORATORY。未重新训练、未选最优epoch、未根据中途AP改参数。

## 参数与科学范围

kernel21/beta1.5来自原导师UVL；角度Uniform[0,180)及airlight220由本次作者明确允许，标记AUTHOR_ASSUMPTION，不是导师签字或真实天气/相机/飞行标定。Haze为常深度光学厚度代理，BGR编码强度并非线性辐亮度。相同P的共享PNG每模型逐图检查相同SHA；GT坐标、面积和ignore不变。RollingShutter排除。

实际顺序为Haze→Normal恒等曝光→Motion，不是任意交换。Haze公式为clip_uint8(J·exp(−1.5)+(1−exp(−1.5))·220)，beta是常深度归一化代理而非m⁻¹实测消光系数；220为BGR8数字强度。Motion为原图像素21×21归一化栅格直线核、cv2.filter2D、BORDER_REFLECT_101；角度单位度。每图每特征seed取SHA256(master_seed|image_stem|feature)前8字节大端整数，master_seed=20261117。Motion单项与复合条件使用同图同角度，光度/卷积不移动GT坐标，不代表遮挡或物理成像标定。源码及具体逐图角度见parameter_binding_review.json和pixel_cache_manifest.json。

## 真实指标

下表百分数，JSON/CSV保持0–1原值。完整类别/尺度AP50、AP95、AR及200次区间保存。原始预测、GT来源及逐图NCHW FP32保留。

有效GT38593框，其中Tiny11842框。原图bbox面积口径：Tiny<256px²，Small为[256,1024)，Medium为[1024,9216)，Large≥9216；额外coco_small沿用既有评价器。21px模糊核相对Tiny目标本身已很大。低AP是有效配置的经验性能结果，不是SAT非法、数值异常或YOLO软件错误。

{table(['条件','D','AP50','AP95','Tiny AP95','AR95','缓存管线ms','固定判据'],summary) if summary else '无完整结果'}

## 配对效应

{effect_table}

200次同图位置Bootstrap，每次用相同GT/预测集合重累计COCO AP；复用已验收匹配/累计算法，不平均单图AP。四角点DID=复合−Motion−Haze+Normal；跨D为ck01−ck05。类别缺GT保持null，paired_valid_draws记录有效配对数；非全部200有效的逐类区间条件于有GT的重采样，不伪称完整支持。

点态区间没有家族多重校正；这是固定两历史checkpoint、同548开发图、既有选模暴露的描述性受控合成交互，不是训练seed显著性、独立图像确认、YOLO架构因果或真实天气物理因果。

DID只描述AP尺度的非加性。正DID也可能来自AP下界和损失饱和，不表示复合退化改善检测；应同时查看四格绝对AP与单项损失。没有可靠航次ID，图像Bootstrap没有控制真实航次聚类相关。历史best的准确选中epoch及实际Albu激活仍UNKNOWN，跨checkpoint差值不能单独归因为架构。

## Tiny、时延与局部Gap

固定Tiny AP95≥.10、缓存管线均值≤100ms；按冻结规则Tiny上界<.10或时延上界>100判FAIL，Tiny下界≥.10且时延上界≤100判PASS，其他UNCERTAIN。AP变化不等于Gap关闭。只讨论4个P×2个D局部覆盖，不将21或24个有目的P推为46368空间失败比例。时延包括缓存SHA/读取/解码/像素SHA、预测、框提取、逐图预测及manifest写入flush；加载、预热、缓存生成、COCO/Bootstrap计入任务成本，不混入逐图管线。固定执行顺序、热状态和缓存效应仍限制时延因果解释。

local_P_gap_status.csv只给本次两个Infer1280候选的局部COVERED/GAP/UNCERTAIN；新增3个P尚未测原四候选的两个Infer640，不声称全候选不存在可行解。latency_interactions.csv给同图时延单因素/DID及跨D差，区间不包含重复硬件状态的不确定性。原静态映射敏感性1、9、18/46368仅为不同覆盖假设，不是经验失败率；沿用已有VM45结果，不重新计算或修改UVL。

## 开发证据与规则安全

新观察加入只读联合索引。原84行不重新拟合；旧Normal两行与新两行都保留在92条观察中，canonical索引只采用新Normal，90个唯一P-D、24个P。历史VM36和新VM41计时边界显式标记，没有静默拼成完全同口径。旧浅树在新条件上的固定标签输出单独保存；旧RF没有序列化，不为重建它而重复拟合旧84行。

现有预测PASS精确率37.50%，本次预先规定没有独立安全验收就不推广规则。0条新规则获准。新实测区间PASS可供精确查表，规则预测仍ABSTAIN；同P不能跨训练/验证，两种计时cohort和共享图像局限保留。VM43 test-dev未进入规则训练。

{learning}

新增诊断只在完整8格后采用90个唯一P-D、24个P全组留出。树/RF参数沿用冻结定义，显式加入时延cohort指标；不重复旧84行单独拟合、不调参。条件于固定OOF预测的200次整P重采样区间不等于独立图像或训练seed证据。固定YOLO、实测查表与安全弃权控制器的逐P差异见controller_v2_measured_results.csv。无法支持新环境安全释放，规则控制器弃权率100%。

## 资源、保护与未完成研究

实际阶段监督与计费见actual_resource_usage.json；合计计费{resource['total_charged_seconds']:.3f}秒，为含CPU统计的GPU-worker墙钟，不是CUDA kernel忙时。缓存CPU时间单独记录。账本余额{resource['budget_after']['remaining_seconds']:.3f}秒，原截止{resource['budget_after']['deadline_UTC']}不变。90°C即停、87°C30秒停，实际监测目标1秒；无重试。仍在有效期内的用户物理检查记录被复用，并非本次新现场确认，当前AC、设备和真实预检另有记录。

{table(['阶段','采样数','最高°C','最大设备占用MiB','最大相邻记录间隔s'],[[r['phase'],r['records'],r['max_temperature_C'],r['max_used_MiB'],f"{r['max_accepted_record_gap_seconds']:.3f}"] for r in resource['telemetry_summary']])}

没有完整连续功耗、风扇或CUDA kernel忙时记录，不估算能耗。推理低温不证明长训练散热已经解决。

正常图也在本轮重新评价，未混用历史计时。保存全部原预测、共享像素、配置/模型/GT SHA。原VM36–VM45历史、权重、UVL和账本旧条目不修改；只追加实际阶段计费。无Git操作。正式1280/100epoch多seed与五增强消融仍未完成，这八格不是八次训练，也不代表完整RQ4完成。

证据：{out}。执行入口execute_vm45.py仅当前一次授权；finish_execution.py只做CPU验收/图表。已验收统计不重跑。原监督停止原因、失败trace与partial文件均保留，不因不完整而删去。
'''
 brief=f'''# VM45 导师会议短汇报

RunID：`{out.name}`。{gate_text}，已核验{len(cells)}/8格。

{effect_table}

这是两历史模型、548开发图、作者允许的合成工程假设下的配对结果。角度/airlight没有真实天气标定；不是架构因果。Tiny和完整时延必须同时通过才可称该局部P被当前D覆盖，AP增益不等于Gap关闭。

规则方面：旧37.5%预测PASS精确率不安全，不允许自动放行。新实测可用于查表；没有独立安全留出验收则继续ABSTAIN。正式100轮RQ4不因八格评价而完成。详细数值见科学报告和原值CSV，而非重跑旧实验。
'''
 matrix=[{'RQ':'RQ1','state':'PARTIAL','evidence':'原UVL来源/合法性和实际548图像素GT验证；非全空间可执行证明'},
 {'RQ':'RQ2','state':'MEASURED_LOCAL_INTERACTIONS' if complete else 'NOT_EVALUATED_NEW','evidence':'真实Motion/Haze四角点及共享索引区间，仅两固定checkpoint；非架构因果'},
 {'RQ':'RQ3','state':'PARTIAL','evidence':'固定Tiny/缓存时延的4P×2D局部Gap；未知全空间，原静态映射假设未改变'},
 {'RQ':'RQ4','state':'NOT_EVALUATED_FORMAL','evidence':'未训练；1280/100轮12臂、多seed、五消融仍缺；VM43 Gaussian改善未关闭其Tiny判据'}]
 csv_write(out/'rq_evidence_matrix.csv',matrix)
 paper='# VM45 RQ1–RQ4手稿证据矩阵\n\nRunID：`'+out.name+'`。\n\n'+table(['RQ','状态','可用证据与边界'],[[r['RQ'],r['state'],r['evidence']] for r in matrix])+'\n\n本文件是可追溯证据清单，不是代写论文正文。外部历史独立性未升级；所有新结果属于开发合成探索。原始预测、来源SHA和分母优先于图表摘要。\n'
 paths=[]
 for title,text in [('VM45_真实MotionHaze_科学报告_CN',body),('VM45_真实MotionHaze_导师短汇报_CN',brief),('VM45_真实MotionHaze_RQ证据矩阵_CN',paper)]:
  path=ROOT/'datasets/reportcn'/(title+'_'+out.name.split('_')[-1]+'.md')
  with path.open('x',encoding='utf-8') as f:f.write(text)
  paths.append(path)
 return paths

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);finish(p.parse_args().run)
