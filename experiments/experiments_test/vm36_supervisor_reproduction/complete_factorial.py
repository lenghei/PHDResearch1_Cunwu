"""Complete only the two missing ck01 corners under a new bounded budget."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from protocol import ROOT,HERE,EXPERIMENT,load,save,source,sha256,write_atomic
from common import import_file,seal,verify_hashes
from main.pipeline import new_run

SOURCE_ID='vm36_supervisor_reproduction_full548_20261109T162524_f629a969'
VERSION='VM36_MISSING_CK01_CORNERS_V1'


def missing_specs(plan,completed):
    pending=[s for s in plan['pairs'] if s['id'] not in completed]
    if len(pending)!=2 or any(s['d']['checkpoint']!='ck01' or s['d']['resolution']!=1280 for s in pending):
        raise ValueError('Expected only two missing ck01/1280 configurations.')
    if {tuple(s['p']['leaves']) for s in pending}!={('Normal','JPEG_q25'),('Normal','JPEG_q75')}:
        raise ValueError('Unexpected degradation selection.')
    return pending


def verified_cache(out,plan):
    reference=plan['cache_reference']
    if sha256(Path(reference['path']))!=reference['sha256']:raise ValueError('Cache manifest changed.')
    all_rows=load(Path(reference['path']));cases={}
    ids=sorted(r['id'] for r in load(Path(plan['GT']['path']))['records'])
    for spec in plan['pairs']:
        rows=all_rows[spec['p']['id']]
        if len(rows)!=548 or sorted(r['image_id'] for r in rows)!=ids:raise ValueError('Cache membership differs.')
        for row in rows:
            if sha256(Path(row['path']))!=row['file_sha256']:raise ValueError('Cache bytes changed.')
        cases[spec['p']['id']]=rows
    save(out/'cache_reuse.json',{'source':reference,'images':sum(map(len,cases.values())),
        'new_pixel_generation':0,'pixel_validation':'Verified again after decode for every actual inference.'},out)
    return cases


def prepare(authorization):
    import torch
    from ultralytics import YOLO
    from main.labels import VIS_NAMES
    old=ROOT/'datasets/local_runs'/EXPERIMENT/SOURCE_ID
    verify_hashes(old,('EXECUTION_PROTOCOL.json','RESULT_SUMMARY.json','EVIDENCE_TREE.json','cache_manifest.json'))
    original=load(old/'EXECUTION_PROTOCOL.json');completed=set()
    dependencies=load(old/'source_SHA256.json')
    for relative in ('experiments/experiments_test/vm36_supervisor_reproduction/full_development.py',
        'experiments/experiments_test/cv261207/evaluation.py','experiments/experiments_test/cv261207/development_v1.py',
        'experiments/experiments_test/cv261207/development_fast_coco.py','code/main/statistics.py','code/main/labels.py','code/main/predictor.py'):
        if sha256(ROOT/'main'/relative)!=dependencies[relative]:raise ValueError('Scientific dependency changed: '+relative)
    for spec in original['pairs']:
        folder=old/'pairs'/spec['id']
        if (folder/'SHA256SUMS.txt').exists():
            verify_hashes(folder,('metrics.json','bootstrap.json','predictions.jsonl.gz','ground_truth.json'))
            completed.add(spec['id'])
    pairs=missing_specs(original,completed)
    if len(completed)!=6:raise ValueError('The accepted six-pair source differs.')
    if (old.parent/'.full548_gpu.lock').exists():raise ValueError('A full548 worker lock exists.')
    free=shutil.disk_usage(ROOT).free
    if free<8*1024**3:raise RuntimeError('Insufficient disk space.')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable.')
    gpu_free,gpu_total=torch.cuda.mem_get_info(0)
    if gpu_free<3*1024**3:raise RuntimeError('Insufficient free GPU memory.')
    spec=pairs[0]['d'];weight=Path(spec['path'])
    if sha256(weight)!=spec['sha256']:raise ValueError('Checkpoint identity differs.')
    model=YOLO(weight).model.cpu().float().eval()
    if list(model.names.values())!=VIS_NAMES:raise ValueError('Native10 order differs.')
    invalid=[key for key,value in model.state_dict().items() if value.is_floating_point() and
        (not bool(torch.isfinite(value).all()) or key.endswith('running_var') and bool((value<0).any()))]
    if invalid:raise ValueError('Unhealthy checkpoint tensors: '+str(invalid))
    slowest=max(load(old/'pairs'/identity/'timing.json')['wall_seconds'] for identity in completed)
    estimate=2*slowest*1.25+120
    if estimate>1800:raise RuntimeError('Both conservative pair estimates exceed the new budget.')
    out,_=new_run({'output_root':str(old.parent)},EXPERIMENT,'missing_corners')
    plan={**original,'version':VERSION,'pairs':pairs,'worker_budget_seconds':1800,
        'reference_pair_seconds':slowest,'original_protocol':source(old/'EXECUTION_PROTOCOL.json'),
        'cache_reference':source(old/'cache_manifest.json'),'source_RunID':SOURCE_ID,
        'authorization':source(authorization),'new_image_inferences':1096,
        'scientific_changes':[],'operational_changes':['New immutable RunID','Two missing pairs only','Reuse verified pixels','New 1800-second cap']}
    save(out/'EXECUTION_PROTOCOL.json',plan,out)
    write_atomic(out/'PROTOCOL_SHA256.txt',sha256(out/'EXECUTION_PROTOCOL.json')+'  EXECUTION_PROTOCOL.json\n',out)
    save(out/'EXPERIMENT_DESIGN_JUSTIFICATION.json',{'RQ':'RQ2','required_total_corners':8,'accepted_reused':6,
        'new_corners':2,'images_per_corner':548,'rationale':'Complete both fixed-checkpoint 2x2 resolution x JPEG contrasts without repeating accepted pairs.'},out)
    save(out/'RESOURCE_PREFLIGHT.json',{'status':'PASS','free_disk_bytes':free,'GPU_free_bytes':gpu_free,
        'GPU_total_bytes':gpu_total,'device':torch.cuda.get_device_name(0),'checkpoint':source(weight),
        'tensor_health':'VALIDATED_CPU','class_order':VIS_NAMES,'reference_slowest_pair_seconds':slowest,
        'conservative_two_pair_seconds':estimate,'new_worker_cap_seconds':1800,'old_budget_modified':False},out)
    save(out/'inputs.json',{'accepted_source':source(old/'SHA256SUMS.txt'),'source_protocol':source(old/'EXECUTION_PROTOCOL.json'),
        'author_instruction':source(authorization),'worker':source(Path(__file__))},out)
    return out


def supervise(authorization):
    from development_guard import bounded_run
    out=prepare(authorization);lock=out.parent/'.missing_corners_gpu.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        with (out/'worker.log').open('x',encoding='utf-8') as stream:
            code=bounded_run([sys.executable,'-B',str(Path(__file__)),'--worker',str(out)],stream,1800,
                dict(os.environ,PYTHONDONTWRITEBYTECODE='1',YOLO_AUTOINSTALL='false',OMP_NUM_THREADS='2'))
        if code:raise RuntimeError('Worker failed; no automatic retry.')
    except BaseException:
        save(out/'failure.json',{'traceback':traceback.format_exc(),'automatic_retry':False},out)
    finally:
        lock.unlink()
        result=load(out/'worker_result.json') if (out/'worker_result.json').exists() else {'status':'PARTIAL','completed':0}
        save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'MISSING_CORNERS',**result},out,replace=True)
        print(out,flush=True)


def worker(out):
    import cv2,numpy as np
    from development_v1 import Model
    from evaluation import runtime,evaluate_saved
    from main.labels import mapped_predictions
    from main.statistics import evaluate_coco,bootstrap_coco
    runtime(out);plan=load(out/'EXECUTION_PROTOCOL.json');start=time.perf_counter()
    for key in ('FMP','FMD','GT','image_members','ZIP','generator'):
        r=plan[key]
        if sha256(Path(r['path']))!=r['sha256']:raise ValueError('Input changed: '+key)
    cases=verified_cache(out,plan);payload=load(Path(plan['GT']['path']));records=payload['records'];gt=payload['coco']
    regions={r['id']:r['ignore_regions']+r['other_regions'] for r in records}
    fast=import_file('vm36_full_fast',HERE.parent/'cv261207/development_fast_coco.py');sys.modules['new_fast_coco_accumulate']=fast
    completed=[];device=None;checkpoint=None
    for spec in plan['pairs']:
        remaining=plan['worker_budget_seconds']-(time.perf_counter()-start)
        if completed and remaining<max(plan['reference_pair_seconds'],completed[-1]['wall_seconds'])*1.25+60:break
        folder=out/'pairs'/spec['id'];folder.mkdir(parents=True);begin=time.perf_counter()
        if checkpoint!=spec['d']['checkpoint']:
            device=Model(spec['d']);checkpoint=spec['d']['checkpoint']
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
                if (index+1)%100==0:print(spec['id'],'images',index+1,'/548',flush=True)
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
        seal(folder);completed.append({'pair':spec['id'],'checkpoint':checkpoint,'resolution':spec['d']['resolution'],'JPEG':spec['p']['leaves'][-1] if False else next(x for x in spec['p']['leaves'] if x.startswith('JPEG')),'wall_seconds':timing['wall_seconds'],'folder':str(folder)})
        save(out/'progress.json',{'completed':completed,'remaining_seconds':plan['worker_budget_seconds']-(time.perf_counter()-start)},out,replace=(out/'progress.json').exists())
        if len(completed)==1:save(out/'FIRST_FULL_PAIR_COST.json',{'measured_seconds':timing['wall_seconds'],'remaining_pairs':len(plan['pairs'])-1,'screening_remaining_seconds':(len(plan['pairs'])-1)*timing['wall_seconds'],'hard_budget_seconds':plan['worker_budget_seconds']},out)
        print('PAIR_VALIDATED',spec['id'],timing['wall_seconds'],flush=True)
    save(out/'worker_result.json',{'status':'VALIDATED' if len(completed)==len(plan['pairs']) else 'PARTIAL','completed':len(completed),'planned':len(plan['pairs']),'worker_wall_seconds':time.perf_counter()-start},out)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--worker',type=Path);parser.add_argument('--authorization',type=Path);args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.execute and args.authorization:supervise(args.authorization)
    else:parser.error('Explicit --execute is required; no GPU job was started.')
