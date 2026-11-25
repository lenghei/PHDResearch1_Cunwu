"""One author-approved VM45 eight-cell attempt with an owned GPU supervisor."""
import argparse
from datetime import datetime, timedelta, timezone
import csv
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'main/code'))
sys.path.insert(0,str(HERE.parent/'vm43_external_pilot'))
import vm43_common as old

RUNS=ROOT/'datasets/local_runs/vm45_motion_haze_gap_learning'
AUTHORIZATION_ID='VM45_AUTHOR_FINAL_EXECUTION_DECISION_20261117_ONCE_V1'
V44=ROOT/'datasets/local_runs/vm44_tiny_error_variability_decision/vm44_tiny_error_variability_decision_analysis_20261117T031628_9ebff202'
V43=ROOT/'datasets/local_runs/vm43_external_pilot/vm43_external_pilot_execution_20261116T222929_46c7e0ba'
READ=old.load
SHA=old.sha
SOURCES={}

def load_module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

backend=load_module('vm45_accepted_vm43_backend',HERE.parent/'vm43_external_pilot/backend.py')

def source(path):
 path=Path(path).resolve(strict=True)
 result={'path':str(path),'SHA256':SHA(path),'bytes':path.stat().st_size};SOURCES[str(path)]=result;return result

def save(path,value):
 path=Path(path)
 if not path.resolve().is_relative_to(RUNS.resolve()):raise ValueError('VM45 writes must stay local.')
 tmp=path.with_name(path.name+'.partial')
 with tmp.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
 if path.exists():raise FileExistsError(path)
 tmp.rename(path)

def csv_write(path,rows):
 with Path(path).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in rows for k in row)));w.writeheader();w.writerows(rows)

def original_seal(path):
 result={}
 for line in Path(path).read_text(encoding='utf-8').splitlines():
  digest,name=line.split('  ',1);p=Path(name)
  if not p.is_absolute():p=Path(path).parent/p
  result[str(p.resolve())]=digest
 return result

def approved_sources(out):
 rules=READ(V44/'FM_P_to_FM_D_decision_rules.json')
 seal=original_seal(V44/'SHA256SUMS.txt')
 for name in ('FM_P_to_FM_D_decision_rules.json','future_pixel_fixture_tests.json'):
  assert source(V44/name)['SHA256']==seal[str((V44/name).resolve())]
 normal=next(r for r in rules['development_rows'] if r['leaves']==['Normal'] and r['D']=='ck01_Infer1280')
 folder=Path(normal['source_folder']);original=original_seal(folder/'SHA256SUMS.txt')
 for name in ('ground_truth.json','image_manifest.jsonl','configuration.json'):
  assert source(folder/name)['SHA256']==original[str((folder/name).resolve())]
 payload=READ(folder/'ground_truth.json');records=payload['records'];assert len(records)==548
 manifest=[json.loads(line) for line in (folder/'image_manifest.jsonl').read_text().splitlines()]
 assert [r['image_id'] for r in manifest]==[r['id'] for r in records]
 members=[]
 for m,r in zip(manifest,records):
  s=m['source'];assert s['shape']==[r['height'],r['width'],3]
  members.append({'image_id':r['stem'],'width':r['width'],'height':r['height'],'source_image_SHA256':s['image_SHA256'],'annotation_SHA256':s['annotation_SHA256']})
 pop={'images':members,'source_ZIP_accepted_certificate':payload['source_archive']}
 attrs={'Normal':{'gain':1.,'bias':0},'Motion_k21':{'kernel':21,'angle_min':0.,'angle_max':180.},'Haze_dense':{'beta':1.5,'airlight':220.}}
 plans=[('Normal',['Normal']),('Motion_k21',['Normal','Motion_k21']),('Haze_dense',['Normal','Haze_dense']),('Motion_Haze',['Normal','Motion_k21','Haze_dense'])]
 sys.path.insert(0,str(HERE.parent/'vm44_tiny_error_variability_decision'))
 from selector import legal,configuration_id
 conditions=[]
 for name,leaves in plans:
  assert legal(rules['models']['fm_p'],leaves)
  conditions.append({'condition':name,'P_id':configuration_id(leaves),'leaves':leaves,'attributes':{f:attrs[f] for f in leaves}})
 models=[]
 for identity in ('ck05_Infer1280','ck01_Infer1280'):
  c=rules['candidates'][identity];ck=c['checkpoint'];assert source(ck['path'])['SHA256']==ck['sha256']
  assert ck['resolution']==1280 and ck['precision']=='FP32' and legal(rules['models']['fm_d'],c['declared_leaves'])
  models.append({'D':identity,'checkpoint':ck,'historical_training_limit':c['training_limit']})
 assert source(old.HERE.parent/'vm36_supervisor_reproduction/feature_models/fm_p.uvl')['SHA256']=='e6eadd633d1c14e0d6bc425f5e54421d592d766c65a6e5f20c61bc34dd76fb5c'
 assert source(old.HERE.parent/'vm36_supervisor_reproduction/feature_models/fm_d.uvl')['SHA256']=='4022d674822b97dbbe3712646b9a0f970af95d175ae358e63582029bc107571a'
 spec={'id':'VM45_AUTHOR_ASSUMPTION_MOTION_HAZE_8CELL_V1','models':models,'conditions':conditions,'images':548,'cells':8,'inferences':4384,'GT':source(folder/'ground_truth.json'),
  'population':pop,'native10':True,'resolution':1280,'FP32':True,'TF32':False,'batch':1,'head':'dense one-to-many','conf':.001,'NMS_IoU':.7,'max_det':3000,'max_nms':30000,
  'bootstrap_samples':200,'bootstrap_seed':20261117,'image_seed':20261117,'warmups_per_cell':3,'order':'Normal,Motion,Haze,MotionHaze; each ck05 then ck01; fixed before AP',
  'scientific_scope':'DEVELOPMENT_ONLY_SYNTHETIC_EXPLORATORY','parameter_label':'AUTHOR_ASSUMPTION','supervisor_signature':'NOT_PROVIDED','physical_calibration':False,
  'timing_protocol':source(old.V41/'Future_Gap_protocol.json'),'reuse_old_Normal':False,'RollingShutter':'EXCLUDED','no_new_training':True,'no_automatic_retry':True,
  'rule_safety_admission':{'predeclared_condition':'No rule promotion or predicted-PASS release in this exploratory small-P study; absent independent safe holdout evidence means ABSTAIN','false_PASS_safety_bound':'NOT_ESTABLISHED','no_posthoc_threshold_tuning':True}}
 save(out/'experiment_protocol.json',spec)
 save(out/'parameter_binding_review.json',{'status':'AUTHOR_EXPLICITLY_ACCEPTED_CONTROLLED_SYNTHETIC_ASSUMPTIONS','label':'AUTHOR_ASSUMPTION','source':source(HERE.parent/'cv261109/generator.py'),
 'kernel_and_beta':'Original UVL21 and1.5','angle':'Uniform[0,180) degrees, normalized rasterized linear21px PSF, reflect101 borders','airlight':'220 scalar BGR8 encoded intensity, not linear radiance',
 'haze_equation':'J*exp(-beta)+(1-exp(-beta))*220, normalized constant depth1, clip/truncateuint8','calibrated_real_weather_or_flight':False,'supervisor_approved_values':False,
 'GT':'Identity coordinates and ignore; all original objects retained according to existing ignore rule','randomness':'Accepted per-image/per-feature hash seed20261117; same motion draw in motion-only and joint cases','no_training_augmentation_change':True})
 return spec,payload

def cache(out,spec,payload):
 import cv2,numpy as np
 from main.labels import ground_truth
 cv2.setNumThreads(2)
 start=time.perf_counter();root=out/'pixel_cache';root.mkdir()
 archive=backend.OriginalZIP(spec['population'],spec['population']['source_ZIP_accepted_certificate']);ops=backend.renderer();cases={}
 try:
  decoded=[]
  for i,r in enumerate(payload['records']):
   image,parsed=archive.record(i,decode=True)
   assert all(parsed[k]==r[k] for k in ('id','stem','width','height','objects','ignore_regions','other_regions'))
   decoded.append(image)
  rebuilt,_=ground_truth(payload['records'],'native10');assert rebuilt==payload['coco']
  for cond in spec['conditions']:
   dest=root/cond['condition'];dest.mkdir();entries=[]
   for i,(image,r) in enumerate(zip(decoded,payload['records'])):
    pix,trace=ops.generate(image,cond['leaves'],cond['attributes'],r['stem'],spec['image_seed'])
    again,trace2=ops.generate(image,cond['leaves'],cond['attributes'],r['stem'],spec['image_seed'])
    assert np.array_equal(pix,again) and trace==trace2 and pix.shape==image.shape
    ok,data=cv2.imencode('.png',pix);assert ok
    file=dest/(r['stem']+'.png');file.write_bytes(data.tobytes())
    check=cv2.imdecode(np.frombuffer(file.read_bytes(),np.uint8),cv2.IMREAD_COLOR);assert np.array_equal(check,pix)
    entries.append({'image_id':r['id'],'stem':r['stem'],'path':str(file),'file_SHA256':SHA(file),'pixel_SHA256':hashlib.sha256(pix.tobytes()).hexdigest(),
      'source_image_SHA256':spec['population']['images'][i]['source_image_SHA256'],'annotation_SHA256':spec['population']['images'][i]['annotation_SHA256'],'P_id':cond['P_id'],'effects':trace,'shape':list(pix.shape)})
   cases[cond['condition']]=entries;print('CACHE_COMPLETE',cond['condition'],len(entries),flush=True)
 finally:archive.close()
 save(out/'pixel_cache_manifest.json',cases)
 save(out/'dataset_gate.json',{'status':'PASS','images':548,'conditions':4,'PNG_files':2192,'image_decode_once':True,'all_source_image_and_label_SHA':True,'GT_exact_original':True,
  'valid_GT_instances':len(payload['coco']['annotations']),'pixel_determinism_and_PNG_roundtrip':True,'shared_pixels_across_detectors':True,'CPU_cache_wall_seconds':time.perf_counter()-start})
 return cases

def query():
 r=subprocess.run(['nvidia-smi','--query-gpu=temperature.gpu,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=1.5,check=True)
 lines=r.stdout.strip().splitlines()
 if len(lines)!=1:raise RuntimeError('One GPU required.')
 t,u,f,v=[int(x.strip()) for x in lines[0].split(',')]
 return {'UTC':datetime.now(timezone.utc).isoformat(),'temperature_C':t,'used_MiB':u,'free_MiB':f,'utilization_percent':v}

def supervised(command,phase,cap,env,policy):
 from safety import decide,stop_owned
 import psutil
 first=query()
 if first['temperature_C']>policy['preflight_maximum_temperature_C'] or first['free_MiB']<policy['preflight_minimum_free_MiB']:raise RuntimeError('Temperature/VRAM preflight failed.')
 phase.mkdir();start=time.perf_counter();samples=[];reason=None
 with (phase/'worker.log').open('x',encoding='utf-8') as log,(phase/'gpu_telemetry.jsonl').open('x',encoding='utf-8') as trace:
  process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
  try:
   while process.poll() is None:
    tick=time.perf_counter();elapsed=tick-start
    if elapsed>=cap-15:reason='WALL_CLOCK_HARD_STOP'
    else:
     try:
      row=query();row['monotonic_seconds']=time.perf_counter()-start;samples.append(row);trace.write(json.dumps(row)+'\n');trace.flush();reason=decide(policy,samples)
     except BaseException:reason='TELEMETRY_FAILURE'
    if reason:stop_owned(process);break
    time.sleep(max(0,1.-(time.perf_counter()-tick)))
   code=process.wait(timeout=15)
  except BaseException:stop_owned(process);raise
 result={'returncode':code,'stop_reason':reason,'wall_seconds':time.perf_counter()-start,'worker_PID':process.pid,'poll_target_seconds':1,'maximum_observed_temperature_C':max((r['temperature_C'] for r in samples),default=None),'retry':False}
 save(phase/'safety_supervisor.json',result);return result

def evaluation(raw,records,seed):
 import numpy as np
 from main.labels import ground_truth,mapped_predictions
 from main.statistics import evaluate_coco,resample_evaluator,_metric_rows,METRIC_NAMES
 with gzip.open(raw,'rt',encoding='utf-8') as f:rows=[json.loads(line) for line in f]
 backend.validate_predictions(rows,{r['id'] for r in records})
 gt,regions=ground_truth(records,'native10');pred=[p for r in rows for p in r['predictions']];mapped=mapped_predictions(pred,records,regions,'native10','finetuned',3000)
 start=time.perf_counter();areas,classes,evaluator=evaluate_coco(gt,mapped,3000);coco=time.perf_counter()-start
 backend.install_accumulator();start=time.perf_counter();rng=np.random.default_rng(seed)
 av={r['size']:{m:[] for m in METRIC_NAMES} for r in areas};cv={r['size']+':'+str(r['category_id']):{m:[] for m in METRIC_NAMES} for r in classes}
 for i in range(200):
  boot=resample_evaluator(evaluator,rng.integers(0,len(records),len(records)));aa,cc=_metric_rows(boot,include_per_class=True,include_gt_counts=False)
  for r in aa:
   for m in METRIC_NAMES:av[r['size']][m].append(r[m])
  for r in cc:
   for m in METRIC_NAMES:cv[r['size']+':'+str(r['category_id'])][m].append(r[m])
  if (i+1)%50==0:print('BOOTSTRAP',raw.parent.name,i+1,flush=True)
 intervals=[]
 for kind,base,values in [('area',areas,av),('class',classes,cv)]:
  for r in base:
   key=r['size'] if kind=='area' else r['size']+':'+str(r['category_id'])
   for metric in METRIC_NAMES:
    valid=[x for x in values[key][metric] if x is not None]
    intervals.append({'kind':kind,'size':r['size'],'category_id':r.get('category_id'),'metric':metric,'estimate':r[metric],
      'ci95':np.percentile(valid,[2.5,97.5]).tolist() if valid else [None,None],'n_valid':len(valid),'samples':200,'seed':seed})
 draws={'image_ids':[r['id'] for r in records],'samples':200,'seed':seed,'sampling':'numpy.default_rng.integers, sorted source image positions','values':av,'class_values':cv}
 return {'per_area':areas,'per_class':classes,'raw_count':len(pred),'mapped_count':len(mapped),'images':len(records)}, {'intervals':intervals,'draws':draws}, {'COCO_seconds':coco,'bootstrap_seconds':time.perf_counter()-start}

def worker(out,mode):
 ticket=READ(out/(mode+'_lease.json'));token=os.environ.get('VM45_WORKER_TOKEN','')
 if not token or hashlib.sha256(token.encode()).hexdigest()!=ticket['worker_token_SHA256']:raise PermissionError('Owned supervisor token required.')
 if datetime.now(timezone.utc)>=datetime.fromisoformat(ticket['deadline_UTC']):raise PermissionError('Worker lease expired.')
 spec=READ(out/'experiment_protocol.json');assert SHA(out/'experiment_protocol.json')==ticket['protocol_SHA256']
 for path,digest in READ(out/'frozen_worker_sources.json'):
  if SHA(path)!=digest:raise ValueError('Frozen worker implementation changed.')
 old.study.runtime(out/'runtime',cuda=True)
 import cv2,numpy as np,torch
 torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 cache=READ(out/'pixel_cache_manifest.json');records=READ(spec['GT']['path'])['records'];assert SHA(spec['GT']['path'])==spec['GT']['SHA256']
 adapters={};checks=[];start=time.perf_counter()
 chosen=sorted(records,key=lambda r:(-r['width']*r['height'],r['stem']))[:6]
 for detector in spec['models']:
  identity=detector['D'];ck=detector['checkpoint'];assert SHA(ck['path'])==ck['sha256']
  tick=time.perf_counter();adapter=backend.RealDetector(ck['path']);adapters[identity]=adapter
  print('MODEL_LOADED',identity,time.perf_counter()-tick,flush=True)
 if mode=='preflight':
  for identity,adapter in adapters.items():
   image=cv2.imread(cache['Normal'][chosen[0]['id']-1]['path'])
   for _ in range(3):adapter.predict(image)
   for cond in spec['conditions']:
    for r in chosen:
     entry=cache[cond['condition']][r['id']-1];assert SHA(entry['path'])==entry['file_SHA256'];image=cv2.imread(entry['path'])
     adapter.synchronize();t=time.perf_counter();res=adapter.predict(image);adapter.synchronize();raw,tensor=adapter.extract(res,r['id']);backend.validate_predictions([{'image_id':r['id'],'predictions':raw}],{r['id']})
     checks.append({'D':identity,'condition':cond['condition'],'image_id':r['id'],'seconds':time.perf_counter()-t,'predictions':len(raw),'tensor':tensor})
  save(out/'preflight_result.json',{'status':'REAL_ENGINEERING_PROBE_PASS','checks':checks,'AP':'NOT_EVALUATED','wall_seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()});return
 cells=[]
 for cond in spec['conditions']:
  for detector in spec['models']:
   identity=detector['D'];adapter=adapters[identity];cell_id=cond['condition']+'__'+identity;folder=out/cell_id;folder.mkdir();print('CELL_START',cell_id,flush=True)
   image=cv2.imread(cache[cond['condition']][0]['path'])
   for _ in range(3):adapter.predict(image)
   adapter.synchronize();torch.cuda.reset_peak_memory_stats();times=[];partial=folder/'predictions.jsonl.gz.partial';raw=folder/'predictions.jsonl.gz'
   with gzip.open(partial,'xt',encoding='utf-8') as f,(folder/'image_manifest.jsonl').open('x',encoding='utf-8') as mf:
    for entry in cache[cond['condition']]:times.append(backend.timed_prediction(entry,adapter,f,mf))
   with gzip.open(partial,'rt',encoding='utf-8') as f:backend.validate_predictions([json.loads(s) for s in f],{r['id'] for r in records})
   partial.rename(raw)
   metrics,boot,cost=evaluation(raw,records,spec['bootstrap_seed'])
   save(folder/'metrics.json',metrics);save(folder/'bootstrap.json',boot)
   save(folder/'timing.json',{'per_image':times,'evaluation_cost':cost,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
   save(folder/'acceptance.json',{'status':'REAL_CELL_COMPLETED','images':548,'P_id':cond['P_id'],'condition':cond['condition'],'D':identity,'checkpoint':source(detector['checkpoint']['path']),
     'raw':source(raw),'GT':spec['GT'],'metrics':source(folder/'metrics.json'),'bootstrap':source(folder/'bootstrap.json'),'scope':spec['scientific_scope']})
   cells.append(cell_id);print('CELL_COMPLETE',cell_id,flush=True)
 save(out/'worker_result.json',{'status':'REAL_EIGHT_CELLS_COMPLETED','cells':cells,'images_per_cell':548,'formal_predictions':4384,'wall_seconds':time.perf_counter()-start,'scope':spec['scientific_scope']})

def charge(out,phase,seconds,result):
 path=old.AUTH/'charges'/(out.name+'_'+phase+'.json')
 with path.open('x',encoding='utf-8') as f:json.dump({'RunID':out.name,'stage':'VM45_'+phase,'charged_seconds':seconds,'supervisor':result,'budget_reset':False},f,indent=2)
 return {'path':str(path),'SHA256':SHA(path)}

def run_phase(out,mode,cap,token,policy):
 now=datetime.now(timezone.utc);save(out/(mode+'_lease.json'),{'mode':mode,'worker_token_SHA256':hashlib.sha256(token.encode()).hexdigest(),
  'protocol_SHA256':SHA(out/'experiment_protocol.json'),'deadline_UTC':(now+timedelta(seconds=cap)).isoformat(),'cap_seconds':cap,'attempt':1})
 command=[sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),'--worker',out.name,'--mode',mode]
 start=time.perf_counter();result=None
 try:result=supervised(command,out/mode,cap,dict(os.environ,VM45_WORKER_TOKEN=token),policy)
 finally:charge(out,mode,time.perf_counter()-start,result)
 return result

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--worker');p.add_argument('--mode',choices=['preflight','formal']);a=p.parse_args()
 if a.worker:
  if Path(a.worker).name!=a.worker or not a.worker.startswith(HERE.name+'_execution_'):raise ValueError('Invalid VM45 RunID.')
  out=(RUNS/a.worker).resolve(strict=True)
  try:worker(out,a.mode)
  except BaseException:
   save(out/(a.mode+'_failure.json'),{'traceback':traceback.format_exc(),'retry':False});raise
  return
 if not a.execute:p.error('Explicit --execute is required; this entry is bound to the current author decision.')
 for previous in RUNS.glob('*_execution_*/author_authorization.json'):
  if READ(previous).get('authorization_id')==AUTHORIZATION_ID:raise PermissionError('This author authorization already has an attempt; no retry.')
 from main.pipeline import new_run
 started=time.perf_counter();out,logger=new_run({'output_root':str(RUNS),'scope':'VM45_AUTHOR_APPROVED_8CELL_ONLY'},HERE.name,'execution')
 (out/'runtime').mkdir();old.budget()
 policy=READ(old.BUNDLE/'safety_policy.json');assert (policy['immediate_stop_temperature_C'],policy['sustained_stop_temperature_C'],policy['sustained_seconds'],policy['poll_seconds'])==(90,87,30,2)
 lock=old.study.RUNS/'.pilot10_gpu.lock';acquired=False;stopped=None
 try:
  authority={'authorization_id':AUTHORIZATION_ID,'source':'CURRENT_USER_EXPLICIT_VM45_FINAL_EXECUTION_DECISION','created_UTC':datetime.now(timezone.utc).isoformat(),
   'author_scientific_quote':'允许将以下历史工程实现参数用于受控合成退化的探索性实验：kernel=21, angle~Uniform[0,180), beta=1.5, airlight=220。',
   'author_GPU_quote':'我授权Codex在原有36小时GPU预算内，有条件执行VM45的完整8格实验。',
   'scope':'8cells x548 validation, ck05/ck01 Infer1280 FP32 batch1 native10','parameter_label':'AUTHOR_ASSUMPTION',
   'no_supervisor_signature_or_physical_calibration':True,'no_budget_reset_or_extension':True,'no_retry':True,'no_training':True,'original_authorization':old.budget()['authorization']}
  save(out/'author_authorization.json',authority)
  for f in (Path(__file__).resolve(),HERE.parent/'vm43_external_pilot/backend.py',HERE.parent/'cv261109/generator.py',ROOT/'main/code/main/statistics.py',ROOT/'main/code/main/labels.py',ROOT/'main/code/main/predictor_stable.py'):source(f)
  spec,payload=approved_sources(out)
  import shutil,psutil
  if shutil.disk_usage(out).free<12*1024**3 or psutil.virtual_memory().available<8*1024**3:raise RuntimeError('Disk/RAM admission failed.')
  before=old.budget();assert before['remaining_seconds']>=15000 and datetime.now(timezone.utc)+timedelta(seconds=15000)<datetime.fromisoformat(before['deadline_UTC'])
  cache(out,spec,payload)
  if lock.exists() or (old.study.RUNS/'.gpu.lock').exists():raise RuntimeError('Existing research GPU owner lock.')
  with lock.open('x',encoding='utf-8') as f:json.dump({'RunID':out.name,'PID':os.getpid()},f)
  acquired=True
  hardware=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version,temperature.gpu,memory.total,memory.free,utilization.gpu,power.draw','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
  (out/'device_identity.txt').write_text(hardware.stdout,encoding='utf-8');assert 'GPU-1dcc4d8e-3346-93c1-53db-f899d1f9efaf' in hardware.stdout
  processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10)
  (out/'GPU_processes.txt').write_text(processes.stdout+'\n'+processes.stderr,encoding='utf-8')
  for line in processes.stdout.splitlines():
   if 'python' in line.lower():raise RuntimeError('Competing Python GPU workload detected.')
  import ctypes
  class Power(ctypes.Structure):_fields_=[('ACLineStatus',ctypes.c_ubyte),('BatteryFlag',ctypes.c_ubyte),('BatteryLifePercent',ctypes.c_ubyte),('SystemStatusFlag',ctypes.c_ubyte),('BatteryLifeTime',ctypes.c_ulong),('BatteryFullLifeTime',ctypes.c_ulong)]
  power=Power();assert ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power)) and power.ACLineStatus==1
  physical=old.study.RUNS/'vm37_yolo26_rq4_pilot960_authorization_record_cpu_20261114T110935_8e5d6543/physical_and_execution_authorization.json'
  physical_data=READ(physical);assert datetime.now(timezone.utc)<datetime.fromisoformat(physical_data['expiry_UTC'])
  assert all(physical_data[k] is True for k in ('hard_ventilated_surface','unobstructed_intake_exhaust','fans_normal','reliable_AC_adapter'))
  save(out/'physical_power_admission.json',{'AC_online_actual':True,'valid_user_physical_record_reused':source(physical),'new_physical_reconfirmation':False,'unknown':['room temperature','fan RPM','OEM mode'],'current_device_reading':source(out/'device_identity.txt')})
  sources=[(str(f),SHA(f)) for f in (Path(__file__).resolve(),HERE.parent/'vm43_external_pilot/backend.py',HERE.parent/'cv261109/generator.py')]
  save(out/'frozen_worker_sources.json',sources)
  token=secrets.token_hex(24);preflight=run_phase(out,'preflight',600,token,policy)
  if preflight['returncode']!=0 or preflight['stop_reason'] is not None:raise RuntimeError('Engineering preflight failed: '+str(preflight))
  assert READ(out/'preflight_result.json')['status']=='REAL_ENGINEERING_PROBE_PASS'
  measured=[]
  for f in V43.glob('s42_*/timing.json'):
   v=READ(f);seconds=sum(r['cached_pipeline_ms'] for r in v['per_image'])/1000+sum(v['evaluation_cost'].values());measured.append({'source':source(f),'total_measured_seconds':seconds,'images':1610})
  maximum=max(r['total_measured_seconds']/r['images'] for r in measured);cache_seconds=READ(out/'dataset_gate.json')['CPU_cache_wall_seconds']
  cap=math.ceil((maximum*4384*2+cache_seconds+60+600)/60)*60;reserve=1800;b=old.budget()
  assert cap+reserve<=b['remaining_seconds'] and datetime.now(timezone.utc)+timedelta(seconds=cap+reserve)<datetime.fromisoformat(b['deadline_UTC'])
  save(out/'full_cost_admission.json',{'status':'ADMITTED_CONSERVATIVE_EMPIRICAL_PLANNING_BOUND','measured_worst_cell_sources':measured,'response_margin_factor':2,
   'cache_CPU_seconds_MEASURED':cache_seconds,'startup_allowance_NOT_MEASURED':60,'final_verification_allowance_NOT_MEASURED':600,'worker_cap_seconds':cap,'shutdown_reserve_seconds':reserve,
   'remaining_seconds_at_admission':b['remaining_seconds'],'limitations':'No deterministic runtime guarantee; dense historical models and classwise Bootstrap may vary; hard cap stops incomplete result with no retry.'})
  save(out/'thermal_admission.json',{'status':'ACTUAL_SUPERVISED_PREFLIGHT_PASS','evidence':source(out/'preflight_result.json'),'supervisor':preflight,'physical_record':source(physical),'current_AC_online':True,
   'thresholds_unchanged':True,'inference_probe_is_not_long_training_stability_confirmation':True})
  save(out/'formal_execution_ticket.json',{'status':'ALL_OBJECTIVE_GATES_PASSED','attempt':1,'authorization':source(out/'author_authorization.json'),'cost':source(out/'full_cost_admission.json'),
   'thermal':source(out/'thermal_admission.json'),'dataset':source(out/'dataset_gate.json'),'protocol':source(out/'experiment_protocol.json'),'no_retry':True})
  logger.info('All VM45 gates passed. Starting ONE eight-cell worker; cap=%ss reserve=%ss.',cap,reserve)
  formal=run_phase(out,'formal',cap,token,policy)
  success=formal['returncode']==0 and formal['stop_reason'] is None and (out/'worker_result.json').exists()
  save(out/'execution_end.json',{'status':'REAL_EIGHT_CELLS_COMPLETED' if success else 'INCOMPLETE','formal_supervisor':formal,'budget_after':old.budget(),'CPU_and_supervised_wall_seconds':time.perf_counter()-started,'retry':False})
  if not success:raise RuntimeError('Formal worker incomplete; no retry.')
  print('VM45_EIGHT_CELLS_FINISHED',out,flush=True)
 except BaseException:
  stopped=traceback.format_exc();save(out/'execution_failure.json',{'status':'BLOCKED_OR_INCOMPLETE','traceback':stopped,'retry':False,'budget_after':old.budget()});print(stopped,flush=True)
 finally:
  if acquired and lock.exists() and READ(lock).get('RunID')==out.name:lock.unlink()
  save(out/'execution_source_provenance.json',{'files':list(SOURCES.values()),'old_evaluator_and_generator_unchanged':True})
 if stopped:sys.exit(1)

if __name__=='__main__':main()
