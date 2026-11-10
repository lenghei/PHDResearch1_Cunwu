"""Measure the frozen joint-coverage additions under one four-hour cap."""
import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import numpy as np
from protocol import ROOT,HERE,EXPERIMENT,FMP,FMD,load,save,source,sha256,write_atomic
from common import import_file,seal,verify_hashes
from historical_review import table,csv_file
from main.pipeline import new_run
from full_development import PREP

DESIGN_ID='vm36_supervisor_reproduction_joint_design_20261109T211826_4ada1a21'
ANALYSIS_ID='vm36_supervisor_reproduction_factorial_analysis_20261109T210914_61857e58'
RENDER_ID='vm36_supervisor_reproduction_renderer_preflight_20261109T212219_3aac07df'
VERSION='VM36_JOINT18_FULL548_FP32_V1'
BUDGET=14400


def gpu_snapshot():
    command=['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.free,temperature.gpu,utilization.gpu',
        '--format=csv,noheader,nounits']
    result=subprocess.run(command,capture_output=True,text=True,check=True,timeout=15)
    row=next(csv.reader(result.stdout.splitlines(),skipinitialspace=True))
    return {'name':row[0],'driver':row[1],'total_MiB':int(row[2]),'free_MiB':int(row[3]),
        'temperature_C':int(row[4]),'utilization_percent':int(row[5])}


def resource_guard(initial=False):
    gpu=gpu_snapshot();free=shutil.disk_usage(ROOT).free
    if gpu['temperature_C']>=85:raise RuntimeError('GPU temperature reached the frozen 85 C policy stop.')
    if gpu['free_MiB']<(3072 if initial else 512):raise RuntimeError('Insufficient free GPU memory.')
    if free<8*1024**3:raise RuntimeError('Less than 8 GiB disk reserve.')
    return {'GPU':gpu,'free_disk_bytes':free}


def coverage_contract(rows):
    from sampling import models
    fmp,_,prior=models();blocked=set(load(PREP/'IMPLEMENTATION_BINDINGS_PROPOSAL.json')['proposed_values_from_historical_implementation'])
    population=[p for p in prior.enumerate_configs(fmp) if not blocked.intersection(p)]
    names=sorted(fmp.ALL_LEAVES)+['D_Checkpoint_ck01','D_Infer1280']
    def features(p,ck,res):return list(p)+(['D_Checkpoint_ck01'] if ck=='ck01' else [])+(['D_Infer1280'] if int(res)==1280 else [])
    full=[features(p,ck,r) for p in population for ck in ('ck01','ck05') for r in (640,1280)]
    _,feasible=prior.compute_binary_pair_masks(full,names)
    masks,_=prior.compute_binary_pair_masks([features(r['leaves'].split('|'),r['checkpoint'],r['resolution']) for r in rows],names)
    reused=0;all_states=0
    for row,mask in zip(rows,masks):
        all_states|=mask
        if row['role']=='REUSE_ACCEPTED':reused|=mask
    expected=(len(population),len(full),feasible.bit_count(),reused.bit_count(),all_states==feasible)
    if expected!=(1872,7488,1033,459,True):raise ValueError('Joint projection coverage differs.')
    if len(rows)!=26 or len({r['pair_id'] for r in rows})!=26:raise ValueError('Plan identity/count differs.')
    if sum(r['role']=='PROPOSED_NOT_RUN' for r in rows)!=18:raise ValueError('Expected eighteen new points.')
    return {'admitted_FMP':1872,'detector_settings':4,'projected_pairs':7488,'feasible_states':1033,
        'existing_states':459,'planned_total_states':all_states.bit_count(),'scope':'Restricted projection only; not full FM_D.'}


def prepare(authorization):
    import torch
    from ultralytics import YOLO
    from main.labels import VIS_NAMES
    from alignment import parameters,require_runtime_bindings
    start=time.perf_counter();root=ROOT/'datasets/local_runs'/EXPERIMENT
    design=root/DESIGN_ID;analysis=root/ANALYSIS_ID;render=root/RENDER_ID
    for path,required in ((design,('JOINT_COVERAGE_PLAN.csv','DESIGN_ACCEPTANCE.json')),
        (analysis,('EVIDENCE_SOURCES.json','EVIDENCE_ACCEPTANCE.json')),(render,('result.json','PIXEL_PROVENANCE.json'))):
        verify_hashes(path,required)
    if load(render/'result.json')['status']!='CPU_PREFLIGHT_PASS':raise ValueError('Renderer engineering preflight failed.')
    rows=table(design/'JOINT_COVERAGE_PLAN.csv');contract=coverage_contract(rows)
    refs=load(analysis/'EVIDENCE_SOURCES.json');original=load(Path(refs['original_protocol']['path']))
    accepted={r['pair_id'] for r in refs['conditions']}
    if accepted!={r['pair_id'] for r in rows if r['role']=='REUSE_ACCEPTED'}:raise ValueError('Accepted eight-pair identity differs.')
    for r in refs['conditions']:verify_hashes(Path(r['pair_folder']),('metrics.json','predictions.jsonl.gz','ground_truth.json','bootstrap.json'))
    for key in ('FMP','FMD','GT','image_members','ZIP','generator'):
        record=original[key]
        if sha256(Path(record['path']))!=record['sha256']:raise ValueError('Source changed: '+key)
    dependencies=load(Path(refs['original_protocol']['path']).parent/'source_SHA256.json')
    for relative in ('code/main/statistics.py','code/main/labels.py','code/main/predictor.py',
        'experiments/cv261207/evaluation.py','experiments/cv261207/development_v1.py','experiments/cv261207/development_fast_coco.py'):
        if sha256(ROOT/'main'/relative)!=dependencies[relative]:raise ValueError('Scientific implementation changed: '+relative)
    if any((root/name).exists() for name in ('.full548_gpu.lock','.missing_corners_gpu.lock','.vm36_joint_gpu.lock')):
        raise RuntimeError('An existing VM36 GPU worker lock prevents execution.')
    resources=resource_guard(initial=True)
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable.')
    detectors={s['d']['id']:s['d'] for s in original['pairs']};health={}
    for ck in ('ck01','ck05'):
        d=next(d for d in detectors.values() if d['checkpoint']==ck)
        if sha256(Path(d['path']))!=d['sha256']:raise ValueError('Checkpoint SHA differs.')
        model=YOLO(d['path']).model.cpu().float().eval()
        if list(model.names.values())!=VIS_NAMES:raise ValueError('Native10 order differs.')
        invalid=[k for k,v in model.state_dict().items() if v.is_floating_point() and
            (not bool(torch.isfinite(v).all()) or k.endswith('running_var') and bool((v<0).any()))]
        if invalid:raise ValueError('Unhealthy tensors: '+str(invalid))
        health[ck]={'source':source(Path(d['path'])),'class_order':VIS_NAMES,'CPU_tensor_health':'VALIDATED'}
        del model
    from sampling import models
    pm,dm,_=models();proposal=load(PREP/'IMPLEMENTATION_BINDINGS_PROPOSAL.json');pairs=[]
    for row in rows:
        if row['role']!='PROPOSED_NOT_RUN':continue
        leaves=row['leaves'].split('|');attributes=parameters(FMP,leaves)
        require_runtime_bindings(attributes,proposal)
        if 'RollingShutter' in leaves or pm.configuration_id(leaves)!=row['p_id']:raise ValueError('Unadmitted geometry or identity.')
        d=detectors[row['d_id']]
        declared=dm.declared_profile_configuration('YOLO26' if d['checkpoint']=='ck01' else 'YOLOv8','Small','VisDrone_FT','Train1280',d['resolution'],'FP32','FullFrame','NMS')
        pairs.append({'id':row['pair_id'],'p':{'id':row['p_id'],'leaves':leaves,'attributes':attributes},
            'd':d,'declared_FM_D':declared,'images':548,'selection':'Frozen feature-only greedy coverage; no AP selection.'})
    slowest=max(load(Path(r['pair_folder'])/'timing.json')['wall_seconds'] for r in refs['conditions'])
    out,_=new_run({'output_root':str(root)},EXPERIMENT,'joint18_4h')
    plan={**original,'version':VERSION,'pairs':pairs,'worker_budget_seconds':BUDGET,
        'reference_pair_seconds':slowest,'authorization':source(authorization),'design':source(design/'JOINT_COVERAGE_PLAN.csv'),
        'source_analysis':source(analysis/'SHA256SUMS.txt'),'source_analysis_RunID':ANALYSIS_ID,
        'planned_new_pairs':18,'first_pair_cost_gate':True,'temperature_policy_stop_C':85,
        'stage_order':['atmosphere','illumination','blur','rolling_shutter','noise','downscale','compression'],
        'thresholds':[.05,.10,.15],'latency_limits_ms':[50,100],'threshold_scope':'Sensitivity only; deployment requirement remains under review.'}
    save(out/'EXECUTION_PROTOCOL.json',plan,out);write_atomic(out/'PROTOCOL_SHA256.txt',sha256(out/'EXECUTION_PROTOCOL.json')+'  EXECUTION_PROTOCOL.json\n',out)
    save(out/'PREFLIGHT.json',{'status':'PASS','resources':resources,'checkpoints':health,'coverage_contract':contract,
        'reference_complete_pair_seconds':slowest,'conservative_JPEG_screening_seconds':18*slowest*1.25,
        'screening_limit':'New recipe generation and matching density differ; first full548 new pair decides continuation.',
        'preflight_seconds':time.perf_counter()-start},out)
    save(out/'EXPERIMENT_DESIGN_JUSTIFICATION.json',{'RQ':'RQ1/RQ2 joint coverage; RQ3 conditional measured outcomes',
        'full_image_population':548,'new_pairs':18,'accepted_reused':8,'planned_total_pairs':26,
        'necessary_count':'Eighteen frozen greedy additions to cover the restricted 1033-state objective; no global optimality claim.',
        'controls':'Original UVLs, seven-stage renderer, native10 original GT/ignore, fixed checkpoints, FP32, pr01, 200 paired image draws.',
        'acceptance':'Only full548 sealed/replayed pairs count. Missing or blocked points are not failures.',
        'scientific_limit':'Configuration coverage is not fault coverage, architecture causality or an independent test.'},out)
    save(out/'inputs.json',{'design':source(design/'SHA256SUMS.txt'),'renderer_preflight':source(render/'SHA256SUMS.txt'),
        'accepted_eight':source(analysis/'SHA256SUMS.txt'),'author_instruction':source(authorization),'worker':source(Path(__file__))},out)
    csv_file(out/'TRAINING_METADATA.csv',[r for r in table(PREP/'CHECKPOINT_ADMISSION.csv') if r['checkpoint'] in health])
    return out


def cached_pixels(out,plan,spec):
    import cv2
    from archives import OriginalZIP
    from development_v1 import generator
    cache=ROOT/'datasets/cache'/EXPERIMENT/out.name/spec['p']['id'];manifest=out/'cache_manifests'/(spec['p']['id']+'.json')
    if manifest.exists():
        rows=load(manifest)
        for r in rows:
            if sha256(Path(r['path']))!=r['file_sha256']:raise ValueError('Verified cache changed.')
        return rows
    cache.mkdir(parents=True);manifest.parent.mkdir(exist_ok=True)
    payload=load(Path(plan['GT']['path']));sources=load(Path(plan['image_members']['path']));archive=OriginalZIP(Path(plan['ZIP']['path']),548)
    renderer=generator();rows=[]
    if list(renderer.ORDER)!=plan['stage_order']:raise ValueError('Seven-stage effect order changed.')
    try:
        for i in range(548):
            image,record,proof=archive.record(i)
            if record!=payload['records'][i] or proof!=sources[i]:raise ValueError('Full-split input identity differs.')
            pixels,trace=renderer.generate(image,spec['p']['leaves'],spec['p']['attributes'],record['stem'],plan['seed'])
            if pixels.shape!=image.shape or pixels.dtype!=np.uint8:raise ValueError('Image geometry/dtype changed.')
            if i==0:
                repeated,again=renderer.generate(image,spec['p']['leaves'],spec['p']['attributes'],record['stem'],plan['seed'])
                if not np.array_equal(pixels,repeated) or trace!=again:raise ValueError('Nondeterministic pixels.')
            path=cache/(record['stem']+'.png');ok,encoded=cv2.imencode('.png',pixels)
            if not ok:raise ValueError('PNG encoding failed.')
            path.write_bytes(encoded.tobytes())
            rows.append({'image_id':record['id'],'path':str(path),'file_sha256':sha256(path),
                'pixel_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),'source':proof,'effects':trace,'shape':list(pixels.shape)})
    finally:archive.close()
    save(manifest,rows,out);return rows


def worker(out):
    from evaluation import runtime
    from development_v1 import Model
    runtime(out);plan=load(out/'EXECUTION_PROTOCOL.json');started=time.perf_counter()
    from main.statistics import evaluate_coco
    fast=import_file('vm36_joint_fast',HERE.parent/'cv261207/development_fast_coco.py');sys.modules['new_fast_coco_accumulate']=fast
    for key in ('FMP','FMD','GT','image_members','ZIP','generator'):
        r=plan[key]
        if sha256(Path(r['path']))!=r['sha256']:raise ValueError('Protected source changed: '+key)
    payload=load(Path(plan['GT']['path']));completed=[];devices={};stopped='ALL_ATTEMPTED';active=None
    try:
        for number,spec in enumerate(plan['pairs'],1):
            remaining=BUDGET-(time.perf_counter()-started)
            estimate=max([plan['reference_pair_seconds']]+[r['complete_wall_seconds'] for r in completed])*1.25+90
            if completed and remaining<estimate:stopped='BUDGET_ADMISSION_STOP';break
            active=spec['id'];resource_guard();pair_start=time.perf_counter();folder=out/'pairs'/active;folder.mkdir(parents=True)
            t=time.perf_counter();rows=cached_pixels(out,plan,spec);cache_seconds=time.perf_counter()-t
            key=spec['d']['checkpoint']
            if key not in devices:devices[key]=Model(spec['d'])
            timing=measure_pair(out,plan,spec,rows,devices[key],pair_start,payload)
            records=[json.loads(line) for line in (folder/'image_manifest.jsonl').read_text().splitlines()]
            expected=[r['id'] for r in payload['records']]
            if [r['image_id'] for r in records]!=expected or any(r['tensor']!={'NCHW':[1,3,spec['d']['resolution'],spec['d']['resolution']],'dtype':'torch.float32'} for r in records):
                raise ValueError('Actual tensor/image identity differs.')
            resource_guard();save(folder/'CACHE_COST.json',{'seconds':cache_seconds,'images':548,'cache_manifest':source(out/'cache_manifests'/(spec['p']['id']+'.json'))},out)
            from joint_reporting import report_pair
            report_pair(out,folder,spec,number,timing)
            seal(folder);cost=time.perf_counter()-pair_start
            completed.append({'pair_id':active,'pair_number':number,'complete_wall_seconds':cost,'folder':str(folder)})
            save(out/'progress.json',{'completed':completed,'remaining_seconds':BUDGET-(time.perf_counter()-started),
                'planned_new_pairs':18,'reused_pairs':8},out,replace=(out/'progress.json').exists())
            if len(completed)==1:save(out/'FIRST_COMPLETE_PAIR_COST.json',{'measured_seconds':cost,'cache_seconds':cache_seconds,
                'remaining_pairs':17,'conservative_remaining_seconds':17*cost*1.25,'hard_budget_seconds':BUDGET},out)
            print('PAIR_ACCEPTED',number,active,cost,flush=True)
    except BaseException:
        stopped='SAFETY_OR_EXECUTION_STOP';save(out/'failure.json',{'active_pair':active,'traceback':traceback.format_exc(),'retry':False},out)
    save(out/'worker_result.json',{'status':'VALIDATED' if len(completed)==18 else 'PARTIAL','completed':len(completed),
        'planned':18,'worker_wall_seconds':time.perf_counter()-started,'stop_reason':stopped},out)


def supervise(authorization):
    from development_guard import bounded_run
    out=prepare(authorization);lock=out.parent/'.vm36_joint_gpu.lock'
    ticket=out.parent/('.joint18_'+sha256(authorization)[:16]+'.ticket')
    token=os.open(ticket,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(token,str(out).encode());os.close(token)
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    began=time.perf_counter()
    try:
        with (out/'worker.log').open('x',encoding='utf-8') as stream:
            code=bounded_run([sys.executable,'-B',str(Path(__file__)),'--worker',str(out)],stream,BUDGET,
                dict(os.environ,PYTHONDONTWRITEBYTECODE='1',YOLO_AUTOINSTALL='false',OMP_NUM_THREADS='2',PYTHONIOENCODING='utf-8'))
        if code:raise RuntimeError('Worker exit was nonzero; no retry.')
    except BaseException:
        save(out/'supervisor_failure.json',{'traceback':traceback.format_exc(),'retry':False},out)
    finally:
        lock.unlink();result=load(out/'worker_result.json') if (out/'worker_result.json').exists() else {'status':'PARTIAL'}
        save(out/'supervisor.json',{'hard_budget_seconds':BUDGET,'actual_child_supervision_seconds':time.perf_counter()-began,
            'authorization_ticket':str(ticket),'automatic_restart':False},out)
        save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'JOINT18_FULL548',**result},out,replace=True)
        print(out,flush=True)


def measure_pair(out,plan,spec,rows,device,begin,payload):
    import cv2
    from evaluation import evaluate_saved
    from main.labels import mapped_predictions
    from main.statistics import evaluate_coco,bootstrap_coco
    folder=out/'pairs'/spec['id'];cases={spec['p']['id']:rows}
    records=payload['records'];gt=payload['coco']
    regions={r['id']:r['ignore_regions']+r['other_regions'] for r in records}
    predict=device.detector(spec['d']);torch=device.torch;rows=cases[spec['p']['id']]
    sample=cv2.imread(rows[0]['path'])
    for _ in range(3):predict(sample)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();raw=[];times=[]
    with gzip.open(folder/'predictions.jsonl.gz.partial','xt',encoding='utf-8') as stream,(folder/'image_manifest.jsonl').open('x',encoding='utf-8') as manifest:
        for index,row in enumerate(rows):
            t=time.perf_counter();path=Path(row['path'])
            if sha256(path)!=row['file_sha256']:raise ValueError('Cache changed.')
            image=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
            if hashlib.sha256(image.tobytes()).hexdigest()!=row['pixel_sha256']:raise ValueError('Pixel SHA differs.')
            torch.cuda.synchronize();p0=time.perf_counter();result,tensor=predict(image);torch.cuda.synchronize();p1=time.perf_counter()
            boxes=result.boxes.data.detach().cpu().numpy()
            if not np.isfinite(boxes).all():raise ValueError('Nonfinite prediction.')
            predictions=[]
            for x1,y1,x2,y2,score,cat in boxes:
                if int(cat)!=cat or not 0<=cat<10 or not 0<=score<=1:raise ValueError('Invalid prediction.')
                if x2>x1 and y2>y1:predictions.append({'image_id':row['image_id'],'class_id':int(cat),'bbox':[float(x1),float(y1),float(x2-x1),float(y2-y1)],'score':float(score)})
            stream.write(json.dumps({'image_id':row['image_id'],'predictions':predictions})+'\n');stream.flush();raw.extend(predictions)
            manifest.write(json.dumps({**row,'tensor':tensor})+'\n');manifest.flush()
            times.append({'image_id':row['image_id'],'predict_ms':(p1-p0)*1000,'forward_ms':result.speed['inference'],'pipeline_ms':(time.perf_counter()-t)*1000})
            if (index+1)%100==0:
                print(spec['id'],'images',index+1,'/548',flush=True)
                resource_guard()
    (folder/'predictions.jsonl.gz.partial').rename(folder/'predictions.jsonl.gz')
    shutil.copyfile(Path(plan['GT']['path']),folder/'ground_truth.json')
    mapped=mapped_predictions(raw,records,regions,'native10','finetuned',3000)
    t=time.perf_counter();areas,classes,ev=evaluate_coco(gt,mapped,3000);eval_time=time.perf_counter()-t
    metrics={'per_area':areas,'per_class':classes,'raw_count':len(raw),'mapped_count':len(mapped),'images':548}
    save(folder/'metrics.json',metrics,out)
    t=time.perf_counter()
    if evaluate_saved(folder)[0]!=metrics:raise ValueError('COCO replay differs.')
    replay=time.perf_counter()-t;t=time.perf_counter()
    ci,draws=bootstrap_coco(ev,samples=200,seed=plan['seed'],progress_callback=lambda a,b:print('bootstrap',a,'/',b,flush=True))
    save(folder/'bootstrap.json',{'intervals':ci,'draws':draws},out)
    timing={'samples':times,'wall_seconds':time.perf_counter()-begin,'evaluation_seconds':eval_time,'replay_seconds':replay,
        'bootstrap_seconds':time.perf_counter()-t,'mean_pipeline_ms':float(np.mean([r['pipeline_ms'] for r in times])),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
        'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'CUDA':torch.version.cuda,'FP32':True,'TF32':False}
    save(folder/'timing.json',timing,out);save(folder/'configuration.json',spec,out)
    save(folder/'verification.json',{'status':'VALIDATED','COCO_recomputed_from_raw':True,'images':548,'protocol_sha256':sha256(out/'EXECUTION_PROTOCOL.json')},out)
    return timing

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--worker',type=Path);parser.add_argument('--authorization',type=Path);args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.execute and args.authorization:supervise(args.authorization)
    else:parser.error('Explicit --execute and the current authorization file are required.')
