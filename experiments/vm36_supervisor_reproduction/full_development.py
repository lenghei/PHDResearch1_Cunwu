"""Bounded complete-validation Resolution x JPEG measurements."""
import argparse
import csv
import gzip
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
from datetime import datetime,timezone
from protocol import ROOT,PUBLIC,HERE,EXPERIMENT,FMP,FMD,STUDY,load,save,source,sha256,write_atomic
from alignment import parameters
from main.pipeline import new_run
from common import import_file,seal,verify_hashes
from historical_review import csv_file,table

VERSION='VM36_FULL548_JPEG_ENDPOINTS_V1'
PREP=ROOT/'datasets/local_runs'/EXPERIMENT/'vm36_supervisor_reproduction_protocol_v2_20261109T153241_baec0138'


def freeze():
    verify_hashes(PREP,('DATASET_ADMISSION.json','MODEL_ALIGNMENT.json','FEATURE_MODEL_VALIDATION.json'))
    if load(PREP/'DATASET_ADMISSION.json')['development_validation']['status']!='VALIDATED':raise ValueError('Dataset not admitted.')
    if shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('Less than 8 GiB free disk space.')
    out,logger=new_run({'output_root':str(ROOT/'datasets/local_runs'/EXPERIMENT)},EXPERIMENT,'full548')
    from sampling import models
    pm,dm,_=models();specs={x['checkpoint']:x for x in load(STUDY/'protocol.json')['detectors']}
    from supervisor_compliance import numeric_attributes
    attrs=numeric_attributes(FMD.read_text(encoding='utf-8'));pairs=[]
    for ck,arch in [('ck05','YOLOv8'),('ck01','YOLO26')]:
        for feature in ('Infer640','Infer1280'):
            resolution=int(attrs[feature,'imgsz'])
            d={**specs[ck],'id':ck+'_'+feature,'resolution':resolution}
            declared=dm.declared_profile_configuration(arch,'Small','VisDrone_FT','Train1280',resolution,'FP32','FullFrame','NMS')
            for jpeg in ('JPEG_q25','JPEG_q75'):
                leaves=['Normal',jpeg];pid=pm.configuration_id(leaves)
                pairs.append({'id':pid+'__'+d['id'],'p':{'id':pid,'leaves':leaves,'attributes':parameters(FMP,leaves)},
                     'd':d,'declared_FM_D':declared,'images':548,'status':'PLANNED'})
    dataset=load(PREP/'full548_data_references.json')
    gt=Path(dataset['full548_ground_truth.json']['path']);images=Path(dataset['full548_image_manifest.json']['path'])
    seed=load(STUDY/'protocol.json')['bootstrap_seed']
    plan={'version':VERSION,'scope':'DEVELOPMENT_ONLY','pairs':pairs,'images':548,'seed':seed,'draws':200,
      'worker_budget_seconds':3600,'first_pair_gate':True,'FMP':source(FMP),'FMD':source(FMD),
      'GT':source(gt),'image_members':source(images),'ZIP':source(ROOT/'datasets/VisDrone/VisDrone2019-DET-val.zip'),
      'generator':source(HERE.parent/'cv261109/generator.py'),
      'scientific_scope':'Two original primary checkpoints, matched resolution x compression contrasts. No motion binding, classifier or population gap claim.',
      'authorization':'Current author instruction permits scientifically valid finite execution; 500/900 are examples, not targets.',
      'timing':'Cached PNG hash/decode + synchronized predictor/NMS + raw write; forward profiler separate; not standardized deployment latency.',
      'precision':'FP32; dense one-to-many; square Letterbox; native10; existing pr01 unchanged.',
      'statistical_scope':'200 paired image resamples, inherited seed; pointwise percentile intervals only. No training-seed significance or flight-cluster claim.'}
    save(out/'EXECUTION_PROTOCOL.json',plan,out)
    write_atomic(out/'PROTOCOL_SHA256.txt',sha256(out/'EXECUTION_PROTOCOL.json')+'  EXECUTION_PROTOCOL.json\n',out)
    design={'status':'PLANNED','RQ':'RQ2','hypothesis':'Does the 1280-minus-640 AP contrast change between JPEG Q25 and Q75 within each primary checkpoint?',
        'minimum_conditions_per_checkpoint':2**2,'primary_checkpoints':2,'total_pairs':len(pairs),'image_inferences':len(pairs)*548,
        'alternatives':{'one_checkpoint_four_pairs':'Cheaper, but cannot compare the two original primary detector families.',
            'three_checkpoints_twelve_pairs':'Supporting YOLO11 is unnecessary for the first primary comparison.',
            'three_JPEG_levels':'Needed for curvature, not the predefined endpoint difference-in-differences.'},
        'coverage':'Complete 2x2 factorial for this restricted contrast; not cross-model full-space pairwise coverage.',
        'held_constant':'All548 IDs, GT, checkpoint within contrasts, Normal illumination, no blur/noise/atmosphere/downscale/rolling, FP32, NMS and evaluator.',
        'acceptance':'All four corner pairs per checkpoint complete, raw replay exact, same bootstrap indices. Report either effect direction.',
        'population_gap_precision':'Not specified; no uniform population estimate or arbitrary epsilon chosen.',
        'excluded_branches':{'motion':'Original direction is unspecified; binding review remains unresolved.','formal_testdev':'DATASET_BLOCKED','RQ4':'RESOURCE_BLOCKED pending matched 3-seed training cost/admission.'}}
    save(out/'EXPERIMENT_DESIGN_JUSTIFICATION.json',design,out)
    write_atomic(out/'EXPERIMENT_DESIGN_JUSTIFICATION.md','# Minimal complete-validation design\n\n'+json.dumps(design,indent=2)+'\n',out)
    z=1.96;rows=[]
    for N in (46368,26496):
        for epsilon in (.02,.05,.075,.10,.15):
            n0=math.ceil(z*z*.25/(epsilon*epsilon));n=math.ceil(N*z*z*.25/(epsilon*epsilon*(N-1)+z*z*.25))
            rows.append({'population':N,'illustrative_half_width':epsilon,'confidence':.95,'unadjusted_n':n0,'finite_population_n':n,
                 'inclusion_probability':n/N,'selection':'SENSITIVITY_ONLY_NOT_SELECTED'})
    csv_file(out/'SAMPLING_SIZE_CALCULATION.csv',rows)
    save(out/'EXECUTION_BUDGET.json',{'worker_hard_seconds':3600,'estimated_pairs':8,'first_pair_measured_seconds':None,
        'old_248_screening_reference':source(PREP/'RESOURCE_BUDGET.json'),'no_budget_extension':True,'new_training':False},out)
    return out


def generate_cache(out,plan):
    import cv2,numpy as np
    from archives import OriginalZIP
    from development_v1 import generator
    gen=generator();archive=OriginalZIP(Path(plan['ZIP']['path']),548)
    cache=ROOT/'datasets/cache'/EXPERIMENT/out.name;cache.mkdir(parents=True)
    records=load(Path(plan['GT']['path']))['records'];sources=load(Path(plan['image_members']['path']))
    cases={};start=time.perf_counter()
    try:
        for p in {s['p']['id']:s['p'] for s in plan['pairs']}.values():
            folder=cache/p['id'];folder.mkdir();rows=[]
            for i in range(548):
                image,rec,proof=archive.record(i)
                if proof!=sources[i] or rec!=records[i]:raise ValueError('Full548 source identity changed.')
                pixels,trace=gen.generate(image,p['leaves'],p['attributes'],rec['stem'],plan['seed'])
                if i==0:
                    repeated,again=gen.generate(image,p['leaves'],p['attributes'],rec['stem'],plan['seed'])
                    if not np.array_equal(pixels,repeated) or trace!=again:raise ValueError('Nondeterministic degradation.')
                if pixels.shape!=image.shape:raise ValueError('Geometry changed.')
                path=folder/(rec['stem']+'.png');ok,encoded=cv2.imencode('.png',pixels)
                if not ok:raise ValueError('PNG save failed.')
                path.write_bytes(encoded.tobytes())
                rows.append({'image_id':rec['id'],'path':str(path),'file_sha256':sha256(path),
                  'pixel_sha256':hashlib.sha256(pixels.tobytes()).hexdigest(),'source':proof,'effects':trace,'shape':list(pixels.shape)})
            cases[p['id']]=rows
    finally:archive.close()
    save(out/'cache_manifest.json',cases,out)
    save(out/'cache_cost.json',{'wall_seconds':time.perf_counter()-start,'images':1096,'bytes':sum(Path(r['path']).stat().st_size for rs in cases.values() for r in rs)},out)
    return cases


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
    cases=generate_cache(out,plan);payload=load(Path(plan['GT']['path']));records=payload['records'];gt=payload['coco']
    regions={r['id']:r['ignore_regions']+r['other_regions'] for r in records}
    fast=import_file('vm36_full_fast',HERE.parent/'cv261207/development_fast_coco.py');sys.modules['new_fast_coco_accumulate']=fast
    completed=[];device=None;checkpoint=None
    for spec in plan['pairs']:
        remaining=plan['worker_budget_seconds']-(time.perf_counter()-start)
        if completed and remaining<completed[-1]['wall_seconds']*1.25+60:break
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
        if len(completed)==1:save(out/'FIRST_FULL_PAIR_COST.json',{'measured_seconds':timing['wall_seconds'],'remaining_pairs':7,'screening_remaining_seconds':7*timing['wall_seconds'],'hard_budget_seconds':3600},out)
        print('PAIR_VALIDATED',spec['id'],timing['wall_seconds'],flush=True)
    save(out/'worker_result.json',{'status':'VALIDATED' if len(completed)==8 else 'PARTIAL','completed':len(completed),'planned':8,'worker_wall_seconds':time.perf_counter()-start},out)


def supervise():
    from development_guard import bounded_run
    out=freeze();lock=out.parent/'.full548_gpu.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    try:
        with (out/'worker.log').open('x',encoding='utf-8') as stream:
            code=bounded_run([sys.executable,'-B',str(Path(__file__)),'--worker',str(out)],stream,3600,
                dict(os.environ,PYTHONDONTWRITEBYTECODE='1',YOLO_AUTOINSTALL='false',OMP_NUM_THREADS='2'))
        if code:raise RuntimeError('Worker failed; no automatic retry.')
    except BaseException:
        save(out/'failure.json',{'traceback':traceback.format_exc(),'no_retry':True},out)
    finally:
        lock.unlink();result=load(out/'worker_result.json') if (out/'worker_result.json').exists() else {'status':'PARTIAL'}
        save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'FULL548_R_JPEG','scope':'DEVELOPMENT_ONLY',**result},out,replace=True)
        print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',type=Path);args=p.parse_args()
    if args.worker:worker(args.worker)
    else:supervise()
