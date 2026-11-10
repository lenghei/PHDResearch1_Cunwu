"""Execute four frozen Normal-only references under one new7200-second cap."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from protocol import ROOT,HERE,EXPERIMENT,FMP,FMD,load,save,source,sha256,write_atomic
from common import seal,verify_hashes,import_file
from historical_review import table,csv_file
from main.pipeline import new_run
from post80_replay import ANALYSIS,run_path
from full_development import PREP
from joint_batch import cached_pixels,measure_pair,resource_guard

SCIENCE='vm36_supervisor_reproduction_post80_science_20261110T190438_45783294'
VERSION='VM36_NORMAL_REFERENCE_FULL548_FP32_V1'
BUDGET=7200


def contract(plan):
    from alignment import parameters
    from sampling import models
    pm,_,_=models();pairs=plan['new_pairs'];expected={(c,r) for c in ('ck01','ck05') for r in (640,1280)}
    if len(pairs)!=4 or len({p['id'] for p in pairs})!=4:raise ValueError('Expected four unique frozen pairs.')
    if {(p['d']['checkpoint'],p['d']['resolution']) for p in pairs}!=expected:raise ValueError('Detector grid changed.')
    for pair in pairs:
        p,d=pair['p'],pair['d']
        if p['leaves']!=['Normal'] or p['attributes']!=parameters(FMP,['Normal']) or p['attributes']!={'Normal':{'gain':1.,'bias':0.}}:
            raise ValueError('Only the original Normal identity is authorized.')
        if p['id']!=pm.configuration_id(['Normal']) or d['id']!=d['checkpoint']+'_Infer'+str(d['resolution']) or pair['id']!=p['id']+'__'+d['id']:
            raise ValueError('Frozen pair identity differs.')
        if pair['images']!=548 or d['precision']!='FP32':raise ValueError('Image population or precision changed.')
    if plan['seed']!=20261108 or plan['bootstrap_draws']!=200 or plan['precision']!='FP32':raise ValueError('Statistical protocol changed.')
    return pairs


def estimate_pair(reference,completed):
    return max([reference]+[r['complete_seconds'] for r in completed])*1.25+90


def authorization(request):
    frozen=run_path(SCIENCE)/'FROZEN_MINIMUM_GPU_PLAN.json'
    verify_hashes(frozen.parent,('FROZEN_MINIMUM_GPU_PLAN.json','LEGAL_CONTRAST_SAT.json','FINAL_EVIDENCE_STATUS.json'))
    contract(load(frozen))
    out,_=new_run({'output_root':str(frozen.parent.parent)},EXPERIMENT,'normal4_authorization')
    shutil.copyfile(request,out/'author_task.txt')
    save(out/'authorization.json',{'scope':'VM36_FOUR_NORMAL_REFERENCE_PAIRS_ONLY','source_task':source(request),
        'preserved_task':source(out/'author_task.txt'),'frozen_plan':source(frozen),'new_pairs':4,'images_per_pair':548,
        'worker_hard_seconds':BUDGET,'recorded_at_UTC':datetime.now(timezone.utc).isoformat(),
        'old_budgets_reused':False,'additional_batches_allowed':False,'training_allowed':False},out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'NORMAL4_AUTHORIZATION','status':'RECORDED'},out,replace=True)
    seal(out);print('AUTHORIZATION',out,flush=True)
    return out/'authorization.json'


def prepare(auth):
    from alignment import parameters
    from sampling import models
    from archives import OriginalZIP
    from development_v1 import generator
    started=time.perf_counter();a=load(auth)
    if a['scope']!='VM36_FOUR_NORMAL_REFERENCE_PAIRS_ONLY' or a['worker_hard_seconds']!=7200 or a['new_pairs']!=4:
        raise ValueError('Wrong execution allocation.')
    frozen=Path(a['frozen_plan']['path'])
    if sha256(frozen)!=a['frozen_plan']['sha256']:raise ValueError('Preregistered plan changed.')
    minimum=load(frozen);contract(minimum)
    old=load(Path(minimum['source_protocol']['path']))
    out,_=new_run({'output_root':str(frozen.parent.parent)},EXPERIMENT,'normal4_full548')
    shutil.copyfile(Path(__file__),out/'executed_source.py');save(out/'inputs.json',{'authorization':source(auth),'worker_source':source(Path(__file__))},out)
    try:
        from evaluation import runtime
        runtime(out)
        import torch
        from ultralytics import YOLO
        from ultralytics.data.augment import LetterBox
        from main.labels import VIS_NAMES
        for key,record in minimum['source_data'].items():
            if sha256(Path(record['path']))!=record['sha256']:raise ValueError('Protected input changed: '+key)
        if sha256(Path(minimum['source_protocol']['path']))!=minimum['source_protocol']['sha256']:raise ValueError('Original protocol changed.')
        dependencies=load(Path(minimum['source_protocol']['path']).parent/'source_SHA256.json');proof={}
        for relative in ('code/main/statistics.py','code/main/labels.py','code/main/predictor.py',
            'experiments/cv261207/evaluation.py','experiments/cv261207/development_v1.py','experiments/cv261207/development_fast_coco.py',
            'experiments/vm36_supervisor_reproduction/joint_batch.py'):
            actual=sha256(ROOT/'main'/relative)
            if actual!=dependencies[relative]:raise ValueError('Scientific implementation changed: '+relative)
            proof[relative]=actual
        if any(out.parent.glob('*.lock')):raise RuntimeError('An existing VM36 lock prevents another GPU worker.')
        resources=resource_guard(initial=True)
        running=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=15)
        if any('python' in line.lower() and line.split(',')[0].strip()!=str(os.getpid()) for line in running.stdout.splitlines()):
            raise RuntimeError('Another Python CUDA process is active.')
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable.')
        pm,dm,_=models();health={};pairs=[]
        for case in minimum['new_pairs']:
            spec={k:case[k] for k in ('id','p','d','images')};d=spec['d'];ck=d['checkpoint']
            if sha256(Path(d['path']))!=d['sha256']:raise ValueError('Checkpoint SHA changed.')
            if ck not in health:
                net=YOLO(d['path']).model.cpu().float().eval()
                if list(net.names.values())!=VIS_NAMES:raise ValueError('Native10 class order differs.')
                invalid=[k for k,v in net.state_dict().items() if v.is_floating_point() and
                    (not bool(torch.isfinite(v).all()) or k.endswith('running_var') and bool((v<0).any()))]
                if invalid:raise ValueError('Unhealthy checkpoint: '+str(invalid))
                health[ck]={'source':source(Path(d['path'])),'class_order':VIS_NAMES,'parameters':sum(p.numel() for p in net.parameters()),'CPU_tensor_health':'VALIDATED'}
                del net
            spec['declared_FM_D']=dm.declared_profile_configuration('YOLO26' if ck=='ck01' else 'YOLOv8','Small','VisDrone_FT','Train1280',d['resolution'],'FP32','FullFrame','NMS')
            pairs.append(spec)
        validation=[]
        for domain,path in (('FM_P',FMP),('FM_D',FMD)):
            command=[str(ROOT/'.venv_flamapy/Scripts/python.exe'),'-B',str(HERE/'validate_models.py'),'--model',str(path),'--domain',domain]
            result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
            write_atomic(out/(domain+'_FlamaPy.log'),result.stdout+result.stderr,out)
            if result.returncode:raise ValueError('Current FlamaPy validation failed: '+domain)
            validation.append(json.loads(result.stdout))
        payload=load(Path(old['GT']['path']));expected=load(Path(old['image_members']['path']));archive=OriginalZIP(Path(old['ZIP']['path']),548)
        renderer=generator();pixels=[];cpu_tensors=[]
        if list(renderer.ORDER)!=minimum['stage_order']:raise ValueError('Rendering order differs.')
        try:
            for i in range(548):
                image,record,member=archive.record(i)
                if record!=payload['records'][i] or member!=expected[i]:raise ValueError('Original image/annotation identity differs.')
                normal,trace=renderer.generate(image,['Normal'],parameters(FMP,['Normal']),record['stem'],minimum['seed'])
                if not np.array_equal(image,normal) or normal.dtype!=np.uint8:raise ValueError('Normal is not an exact pixel identity.')
                pixels.append({'image_id':record['id'],'pixel_sha256':hashlib.sha256(image.tobytes()).hexdigest(),'shape':list(image.shape),'source':member})
                if i<6:
                    again,trace2=renderer.generate(image,['Normal'],parameters(FMP,['Normal']),record['stem'],minimum['seed'])
                    if trace!=trace2 or not np.array_equal(normal,again):raise ValueError('Rendering is not deterministic.')
                    for resolution in (640,1280):
                        square=LetterBox((resolution,resolution),auto=False,stride=32)(image=normal)
                        tensor=torch.from_numpy(np.ascontiguousarray(square[None,...,::-1].transpose(0,3,1,2))).float()/255
                        if list(tensor.shape)!=[1,3,resolution,resolution] or tensor.dtype!=torch.float32:
                            raise ValueError('CPU LetterBox tensor contract differs.')
                        cpu_tensors.append({'image_id':record['id'],'NCHW':list(tensor.shape),'dtype':str(tensor.dtype),'device':'CPU_GEOMETRY_CHECK'})
        finally:archive.close()
        refs=load(run_path(ANALYSIS)/'EVIDENCE_SOURCES.json')['conditions']
        if len(refs)!=80 or any(p['id'] in {r['pair_id'] for r in refs} for p in pairs):raise ValueError('A new pair already exists in the80-point input.')
        reference=max(load(Path(r['folder'])/'timing.json')['wall_seconds'] for r in refs)
        cache_allowance=load(frozen.parent/'GPU_RESOURCE_GATE.json')['reference_cache_max_seconds']
        screening=4*estimate_pair(reference,[])+cache_allowance*1.25+180
        if screening>BUDGET:raise RuntimeError('Conservative four-pair screening exceeds the new budget.')
        plan={'version':VERSION,'pairs':pairs,**{k:old[k] for k in ('GT','image_members','ZIP','FMP','FMD','generator','seed','stage_order')},
            'worker_budget_seconds':BUDGET,'bootstrap_draws':200,'authorization':source(auth),'minimum_plan':source(frozen),
            'scope':'DEVELOPMENT_ONLY','images':548,'reference_pair_seconds':reference,'reference_cache_seconds':cache_allowance,
            'scientific_dependencies':proof,'thresholds':[.05,.10,.15],'latency_limits_ms':[50,100],
            'measurement_function':source(HERE/'joint_batch.py'),'no_training':True,'original80_repeated':False}
        save(out/'EXECUTION_PROTOCOL.json',plan,out);save(out/'NORMAL_PIXEL_PROOFS.json',pixels,out)
        save(out/'PREFLIGHT.json',{'status':'PASS','images':548,'source_annotations_verified':548,'Normal_identity_all548':True,
            'determinism_checks':6,'checkpoints':health,'UVL_validation':validation,'resources':resources,
            'CPU_LetterBox_checks':cpu_tensors,
            'planned_actual_tensor_contract':'[1,3,R,R] float32, R640/1280; predictor asserts every warmup and recorded image.',
            'actual_GPU_tensor_status':'Verified inside the authorized worker; no preflight inference performed.',
            'conservative_four_pair_screen_seconds':screening,'CPU_preflight_seconds':time.perf_counter()-started},out)
        print('PREFLIGHT_PASS',out,screening,flush=True)
        return out
    except BaseException:
        save(out/'preflight_failure.json',{'traceback':traceback.format_exc(),'GPU_started':False,'retry':False},out)
        save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'NORMAL4_FULL548','status':'BLOCKED_PREFLIGHT'},out,replace=True)
        seal(out);raise


def worker(out):
    from evaluation import runtime
    from development_v1 import Model
    runtime(out);began=time.perf_counter();plan=load(out/'EXECUTION_PROTOCOL.json');completed=[];active=None;status='VALIDATED';devices={}
    fast=import_file('vm36_normal_fast',HERE.parent/'cv261207/development_fast_coco.py');sys.modules['new_fast_coco_accumulate']=fast
    payload=load(Path(plan['GT']['path']));proofs=load(out/'NORMAL_PIXEL_PROOFS.json');resource_events=[]
    try:
        for relative,digest in plan['scientific_dependencies'].items():
            if sha256(ROOT/'main'/relative)!=digest:raise ValueError('Scientific source changed after preflight.')
        for number,spec in enumerate(plan['pairs'],1):
            if BUDGET-(time.perf_counter()-began)<estimate_pair(plan['reference_pair_seconds'],completed):status='PARTIAL_BUDGET';break
            active=spec['id'];before=resource_guard();start=time.perf_counter();(out/'pairs'/active).mkdir(parents=True)
            t=time.perf_counter();rows=cached_pixels(out,plan,spec);cache_seconds=time.perf_counter()-t
            if [(r['image_id'],r['pixel_sha256'],r['shape']) for r in rows]!=[(r['image_id'],r['pixel_sha256'],r['shape']) for r in proofs]:
                raise ValueError('Normal cache differs from original decoded pixels.')
            ck=spec['d']['checkpoint']
            if ck not in devices:devices[ck]=Model(spec['d'])
            timing=measure_pair(out,plan,spec,rows,devices[ck],start,payload)
            folder=out/'pairs'/active;manifest=[json.loads(s) for s in (folder/'image_manifest.jsonl').read_text().splitlines()]
            if [r['image_id'] for r in manifest]!=[r['id'] for r in payload['records']] or any(r['tensor']!={'NCHW':[1,3,spec['d']['resolution'],spec['d']['resolution']],'dtype':'torch.float32'} for r in manifest):
                raise ValueError('Actual tensor/image contract changed.')
            after=resource_guard();resource_events.append({'pair_id':active,'before':before,'after':after})
            save(folder/'CACHE_COST.json',{'seconds':cache_seconds,'first_generation':number==1,'same_pixels_as_original':True,
                'manifest':source(out/'cache_manifests'/(spec['p']['id']+'.json'))},out)
            write_atomic(folder/'index.html','<!doctype html><meta charset="utf-8"><h1>'+active+'</h1><a href="metrics.json">COCO metrics</a> · <a href="bootstrap.json">Paired draws</a>',out)
            seal(folder);completed.append({'pair_id':active,'complete_seconds':time.perf_counter()-start,'cache_seconds':cache_seconds,'folder':str(folder)})
            save(out/'progress.json',{'completed':completed,'elapsed_seconds':time.perf_counter()-began,'remaining_seconds':BUDGET-(time.perf_counter()-began)},out,replace=(out/'progress.json').exists())
            print('NORMAL_PAIR_ACCEPTED',number,'/4',active,completed[-1]['complete_seconds'],flush=True)
    except BaseException:
        status='PARTIAL_ERROR';save(out/'failure.json',{'active_pair':active,'traceback':traceback.format_exc(),'retry':False},out)
    save(out/'RESOURCE_EVENTS.json',resource_events,out)
    save(out/'worker_result.json',{'status':status if len(completed)==4 else status if status!='VALIDATED' else 'PARTIAL',
        'completed':len(completed),'planned':4,'saved_image_records':548*len(completed),'warmup_calls':3*len(completed),
        'worker_wall_seconds':time.perf_counter()-began,'new_training_runs':0,'original80_repeated':False},out)


def execute(auth):
    from development_guard import bounded_run
    out=prepare(auth);lock=out.parent/'.normal_reference_gpu.lock';ticket=out.parent/('.normal4_'+sha256(auth)[:16]+'.ticket')
    token=os.open(ticket,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(token,str(out).encode());os.close(token)
    token=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(token,str(os.getpid()).encode());os.close(token)
    began=time.perf_counter()
    try:
        with (out/'worker.log').open('x',encoding='utf-8') as stream:
            code=bounded_run([sys.executable,'-B',str(Path(__file__)),'--worker',out.name],stream,BUDGET,
                dict(os.environ,PYTHONDONTWRITEBYTECODE='1',YOLO_AUTOINSTALL='false',OMP_NUM_THREADS='2',PYTHONIOENCODING='utf-8'))
        if code:raise RuntimeError('Normal worker failed; no retry.')
    except BaseException:save(out/'supervisor_failure.json',{'traceback':traceback.format_exc(),'retry':False},out)
    finally:
        if lock.exists() and lock.read_text()==str(os.getpid()):lock.unlink()
        result=load(out/'worker_result.json') if (out/'worker_result.json').exists() else {'status':'PARTIAL_INTERRUPTED','completed':0}
        save(out/'supervisor.json',{'hard_budget_seconds':BUDGET,'actual_supervision_seconds':time.perf_counter()-began,'retry':False,'ticket':str(ticket)},out)
        save(out/'EVIDENCE_TREE.json',{str(p.relative_to(out)):sha256(p) for p in (out/'pairs').rglob('*') if p.is_file()},out)
        write_atomic(out/'diary.md','# Four Normal references\n\nOriginal decoded pixels, gain1 and bias0. The four frozen settings use the unchanged full548 FP32 predictor, raw COCO replay and200 paired draws. The new7200-second worker budget is independent of all historical runs.\n',out)
        write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>Normal reference measurements</h1><p>'+result['status']+'</p><a href="worker_result.json">Actual completion</a>',out)
        save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'NORMAL4_FULL548',**result},out,replace=True)
        seal(out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--record-authorization',type=Path);g.add_argument('--execute',type=Path);g.add_argument('--worker');a=p.parse_args()
    if a.record_authorization:print(authorization(a.record_authorization),flush=True)
    elif a.execute:execute(a.execute)
    else:worker(run_path(a.worker))
