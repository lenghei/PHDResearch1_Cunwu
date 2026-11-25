"""Bounded new-image development measurements; independent testing stays blocked."""
import argparse
import copy
import csv
from datetime import datetime, timezone
import gzip
from hashlib import sha256 as digest
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from contextlib import redirect_stdout

import numpy as np
from common import ROOT,HERE,RUNS,FMP,FMD,VAL,TEST,FLAMA,load,save,source,sha256,new_run,seal,resolve_run,verify_hashes,import_file
from archives import OriginalZIP
from historical_review import table,csv_file
from sampling import models
from evaluation import Device,runtime,evaluate_saved

BASE=RUNS/'cv261207_rq_acceptance_20261107T220632_8271fd92'
ADMIT=RUNS/'cv261207_admission_20261104T211249_32d83def'
OLD=RUNS/'cv261207_vm26review_20261106T212756_fdf80ed8'
SEED=20261108
PROTOCOL='CV261207_DEV248_NATIVE10_DENSE_FP32_V1'


def checked(path,expected):
    if sha256(path)!=expected:raise ValueError('Input SHA mismatch: '+path.name)
    return path


def prepare():
    out=new_run('development')
    runtime(out)
    for folder in (BASE,ADMIT,OLD):verify_hashes(folder,('run.json',))
    gate=load(ADMIT/'admission.json')
    prior=load(RUNS/'cv261207_audit_20261104T193649_d1b63492/dataset_gate.json')
    checked(VAL,prior['archives']['val']['sha256']);checked(TEST,gate['dataset_sha256'])
    save(out/'admission.json',{'status':'DATASET_BLOCKED','recorded_by':'programmatic evidence reconciliation; not author sign-off',
        'source':source(ADMIT/'admission.json'),'verified_file_disjointness':gate['archive_overlap'],
        'missing':['Historical checkpoint-creation membership for ck01/ck05/ck06','Complete manual/external test-dev selection and tuning history'],
        'options':['Recover exact checkpoint-specific history','Admit genuinely new native10 scenes with permission and frozen hypotheses',
                   'Approve explicitly development-only analysis; this does not admit independent test-dev'],
        'absence_of_search_hits_is_not_proof':True,'new_testdev_forward':0})
    pmodel,dmodel,_=models()
    leaves=[['Downscale_x1_5','Normal'],['JPEG_q25','Normal']]
    configs=[]
    for selected in leaves:
        if not pmodel.valid_leaves(selected):raise ValueError('Invalid new degradation.')
        configs.append({'id':pmodel.configuration_id(selected),'leaves':selected})
    old_stems={r['stem'] for r in load(ROOT/'datasets/research_archive/variability/experiments/vm03_gpu_pilot/runs/vm03_r003/records.json')}
    old_sha=set(load(OLD/'provenance.json')['image_sha256'])
    archive=OriginalZIP(VAL,548);records=[];members=[];indices=[]
    try:
        for i,name in enumerate(archive.images):
            if Path(name).stem in old_stems:continue
            image,rec,proof=archive.record(i)
            if proof['image_SHA256'] in old_sha:raise ValueError('Image byte overlap with rule-fitting population.')
            indices.append(i);records.append(rec);members.append(proof)
    finally:archive.close()
    if len(records)!=248:raise ValueError('Expected complete 248-image complement.')
    save(out/'records.json',records);save(out/'image_members.json',members)
    save(out/'dataset_gate.json',{'status':'DEVELOPMENT_ONLY','images':248,'source_split':'VisDrone2019-DET-val',
        'image_ID_overlap_old300':0,'image_SHA_overlap_old300':0,'checkpoint_selection_exposure':'VAL_USED_HISTORICALLY',
        'independent_test':False,'source_zip':source(VAL),'indices':indices,
        'members_sha256':sha256(out/'image_members.json'),'GT_classes':'native10','ignore_rule':'pr01 ignore union'})
    candidates={r['id']:r for r in gate['candidates']}
    detectors=[]
    for cid in ('ck01','ck05','ck06'):
        c=candidates[cid]
        checked(Path(c['checkpoint']['path']),c['checkpoint_sha256'])
        if not c['native10'] or c['numerical_issues']:raise ValueError('Candidate not healthy/native10.')
        for resolution in (1280,640):
            detectors.append({'id':cid+'_R'+str(resolution),'checkpoint':cid,
                'path':c['checkpoint']['path'],'sha256':c['checkpoint_sha256'],'resolution':resolution,'precision':'FP32',
                'training_resolution':1280,'training_epochs':100,'training_seed':42,
                'historical_Albu_activation':'UNKNOWN'})
    pairs=[{'id':p['id']+'__'+d['id'],'p':p,'d':d} for d in detectors for p in configs]
    plan={'version':PROTOCOL,'scope':'DEVELOPMENT_ONLY','status':'FROZEN_BEFORE_NEW_RESPONSE',
          'images':248,'conditions':configs,'detectors':detectors,'pairs':pairs,'seed':SEED,
          'render_seed':SEED,'bootstrap_seed':SEED,'bootstrap_draws':200,
          'global_budget_seconds':7200,'pair_hard_seconds':1800,'canary_hard_seconds':600,
          'tail_reserve_seconds':900,'first_pair':pairs[0]['id'],'identical_repeat_required':True,
          'thresholds':[.05,.10,.15],'latency_limits_ms':[25,50,100],
          'latency_policy':'95% percentile upper bound for the mean paired-image cached-PNG decode + inference/NMS + raw write wall time; generation reported separately.',
          'selection':'All 248 validation images outside the historical rule-fitting 300; two purposively frozen new factor combinations, not a uniform FM_P sample.',
          'heldout_configuration_rationale':'Neither isolated Downscale_x1_5 nor JPEG_q25 appears in the VM09/VM26 fitting grid.',
          'v1_manifest':source(RUNS/'cv261207_audit_20261104T193649_d1b63492/selection_manifest.json'),
          'v2_manifest':source(BASE/'cross_architecture_v2.json'),
          'code':[source(HERE/n) for n in ('development_v1.py','development_fast_coco.py')],
          'FMP':source(FMP),'FMD':source(FMD),'generator':source(FMP.with_name('generator.py'))}
    save(out/'protocol.json',plan)
    save(out/'phase0.json',{'status':'PASS_DEVELOPMENT_ADMISSION','formal':'DATASET_BLOCKED',
            'new_images':248,'new_configurations':2,'planned_pairs':12,'no_old_model_or_registry_write':True,
            'budget_seconds':7200,'canary_scope':'One clean-control 64/16 engineering epoch only, conditional on remaining budget and CPU admission.'})
    (out/'diary.md').write_text('# Development continuation\n\nFrozen 248-image complement and two new configurations before new inference. Formal test-dev remains blocked.\n',encoding='utf-8')
    save(out/'run.json',{'RunID':out.name,'ExperimentID':'cv261207','status':'PREPARED','mode':'NEW_DEVELOPMENT_STUDY','scope':'DEVELOPMENT_ONLY'})
    print(out,flush=True)


def generator():
    sys.path.insert(0,str(FMP.parent))
    try:return import_file('cv261109_generator',FMP.with_name('generator.py'))
    finally:sys.path.remove(str(FMP.parent))


def cache(study):
    import cv2
    plan=load(study/'protocol.json');gen=generator();pmodel,_,_=models();attrs=pmodel.read_uvl_attributes(FMP)
    archive=OriginalZIP(VAL,548);rows=[];start=time.perf_counter()
    try:
        for p in plan['conditions']:
            folder=study/'cache'/p['id'];folder.mkdir(parents=True)
            for i in load(study/'dataset_gate.json')['indices']:
                image,rec,proof=archive.record(i)
                pixels,effects=gen.generate(image,p['leaves'],attrs,rec['stem'],plan['render_seed'])
                if pixels.shape!=image.shape or pixels.dtype!=np.uint8:raise ValueError('Geometry changed.')
                path=folder/(rec['stem']+'.png')
                ok,data=cv2.imencode('.png',pixels)
                if not ok:raise ValueError('Cache encoder failed.')
                path.write_bytes(data.tobytes())
                rows.append({'condition':p['id'],'image_id':rec['id'],'stem':rec['stem'],'file':str(path.relative_to(study)),
                    'source_image_sha256':proof['image_SHA256'],'annotation_sha256':proof['annotation_SHA256'],
                    'file_sha256':sha256(path),'pixel_sha256':digest(pixels.tobytes()).hexdigest(),'shape':list(pixels.shape),
                    'effects':effects})
    finally:archive.close()
    save(study/'cache_manifest.json',rows)
    save(study/'cache_cost.json',{'seconds':time.perf_counter()-start,'images':len(rows),'bytes':sum((study/r['file']).stat().st_size for r in rows)})


class Model(Device):
    def __init__(self,spec):
        import torch
        from ultralytics import YOLO
        from main.labels import VIS_NAMES
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable.')
        self.torch=torch;self.free,self.total=torch.cuda.mem_get_info(0)
        if self.free<3*1024**3:raise RuntimeError('Insufficient free GPU memory.')
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        checked(Path(spec['path']),spec['sha256']);self.model=YOLO(spec['path'])
        if list(self.model.names.values())!=VIS_NAMES:raise ValueError('Class-order mismatch.')
        for key,value in self.model.model.state_dict().items():
            if value.is_floating_point() and (not torch.isfinite(value).all() or key.endswith('running_var') and (value<0).any()):
                raise ValueError('Unhealthy model tensor: '+key)
        self.params=sum(p.numel() for p in self.model.model.parameters())
        self.checkpoint=source(Path(spec['path']));self.pristine=copy.deepcopy(self.model.model)
        self.predictor_key=None;self.flops={}


def bootstrap_exact(ev,study,out):
    from main.statistics import bootstrap_coco,resample_evaluator,_metric_rows
    fast=import_file('development_fast_coco',HERE/'development_fast_coco.py')
    sys.modules['new_fast_coco_accumulate']=fast
    n=len(ev.params.imgIds);rng=np.random.default_rng(SEED)
    errors=[]
    for _ in range(2):
        indices=rng.integers(0,n,n);sample=resample_evaluator(ev,indices)
        fast_values={k:sample.eval[k].copy() for k in ('precision','recall','scores')}
        with redirect_stdout(StringIO()):sample.accumulate()
        errors.append(max(float(abs(sample.eval[k]-fast_values[k]).max()) for k in fast_values))
    if max(errors)>1e-12:raise ValueError('Fast accumulation differs from reference.')
    start=time.perf_counter()
    ci,draws=bootstrap_coco(ev,samples=200,seed=SEED,
        progress_callback=lambda done,total: print('Bootstrap',done,'/',total,flush=True))
    save(out/'bootstrap.json',{'intervals':ci,'draws':draws,'reference_max_abs_error':max(errors),
                             'seconds':time.perf_counter()-start,'fast_source':source(HERE/'development_fast_coco.py')})


def worker(study,pair_id):
    import cv2
    from main.labels import ground_truth,mapped_predictions
    from main.statistics import evaluate_coco
    from visualization import write_preview
    plan=load(study/'protocol.json');spec=next(r for r in plan['pairs'] if r['id']==pair_id)
    for item in [*plan['code'],plan['FMP'],plan['FMD'],plan['generator']]:
        checked(Path(item['path']),item['sha256'])
    out=new_run('dev_pair');runtime(out);start=time.perf_counter()
    save(out/'binding.json',{'study':study.name,'pair_id':pair_id,'protocol_sha256':sha256(study/'protocol.json')})
    print('PAIR_RUN',out.name,flush=True)
    records=load(study/'records.json');member={r['image_id']:r for r in load(study/'image_members.json')}
    cases=[r for r in load(study/'cache_manifest.json') if r['condition']==spec['p']['id']]
    if [r['image_id'] for r in cases]!=[r['id'] for r in records]:raise ValueError('Cache membership mismatch.')
    device=Model(spec['d']);predict=device.detector(spec['d']);torch=device.torch
    save(out/'configuration.json',{'protocol':PROTOCOL,'study':study.name,'images':248,'split':VAL.stem,
           'pair':spec,'dataset_sha256':sha256(VAL),'cache_manifest_sha256':sha256(study/'cache_manifest.json'),
           'scope':'DEVELOPMENT_ONLY','checkpoint_sha256':spec['d']['sha256'],
           'image_SHA256':[member[r['id']]['image_SHA256'] for r in records]})
    save(out/'environment.json',{'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,
           'python':sys.version,'parameters':device.params,'input_precision':'FP32','TF32':False})
    raw=[];timing=[]
    warm=cv2.imdecode(np.frombuffer((study/cases[0]['file']).read_bytes(),np.uint8),cv2.IMREAD_COLOR)
    for _ in range(3):predict(warm)
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    with gzip.open(out/'predictions.jsonl.gz.partial','xt',encoding='utf-8') as stream,(out/'image_manifest.jsonl.partial').open('x',encoding='utf-8') as manifest:
        for index,case in enumerate(cases):
            begin=time.perf_counter();path=checked(study/case['file'],case['file_sha256'])
            image=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
            if list(image.shape)!=case['shape'] or digest(image.tobytes()).hexdigest()!=case['pixel_sha256']:
                raise ValueError('Degraded pixel identity differs.')
            torch.cuda.synchronize();t=time.perf_counter();result,tensor=predict(image);torch.cuda.synchronize()
            predicted=time.perf_counter();boxes=result.boxes.data.detach().cpu().numpy()
            if not np.isfinite(boxes).all():raise ValueError('Nonfinite prediction.')
            observed=[]
            for x1,y1,x2,y2,score,cat in boxes:
                if int(cat)!=cat or not 0<=cat<10 or not 0<=score<=1:raise ValueError('Invalid class/score.')
                if x2>x1 and y2>y1:observed.append({'image_id':case['image_id'],'class_id':int(cat),
                    'bbox':[float(x1),float(y1),float(x2-x1),float(y2-y1)],'score':float(score)})
            stream.write(json.dumps({'image_id':case['image_id'],'predictions':observed})+'\n');stream.flush()
            manifest.write(json.dumps({**case,'tensor':tensor})+'\n');manifest.flush();raw.extend(observed)
            timing.append({'image_id':case['image_id'],'predict_ms':(predicted-t)*1000,'forward_ms':result.speed['inference'],
                           'pipeline_ms':(time.perf_counter()-begin)*1000})
            if index==0:write_preview(out,image,records[0],observed)
            if (index+1)%50==0:print('Images',index+1,'/ 248',flush=True)
    (out/'predictions.jsonl.gz.partial').rename(out/'predictions.jsonl.gz')
    (out/'image_manifest.jsonl.partial').rename(out/'image_manifest.jsonl')
    gt,regions=ground_truth(records,'native10');save(out/'ground_truth.json',{'coco':gt,'records':records})
    pred=mapped_predictions(raw,records,regions,'native10','finetuned',3000)
    t=time.perf_counter();areas,classes,ev=evaluate_coco(gt,pred,3000);evaluation=time.perf_counter()-t
    metric={'per_area':areas,'per_class':classes,'raw_count':len(raw),'mapped_count':len(pred),'images':248}
    save(out/'metrics.json',metric)
    t=time.perf_counter()
    if evaluate_saved(out)[0]!=metric:raise ValueError('Raw prediction replay differs.')
    replay_seconds=time.perf_counter()-t
    bootstrap_exact(ev,study,out)
    latency=np.array([r['pipeline_ms'] for r in timing]);rng=np.random.default_rng(SEED)
    mean_draws=latency[rng.integers(0,len(latency),(200,len(latency)))].mean(axis=1)
    save(out/'timing.json',{'samples':timing,'mean_pipeline_ms':float(latency.mean()),
         'mean_pipeline_ci95':np.quantile(mean_draws,[.025,.975]).tolist(),
         'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),
         'cpu_evaluation_seconds':evaluation,'cpu_replay_seconds':replay_seconds,
         'pair_wall_seconds':time.perf_counter()-start,'clock':'cached PNG read/hash/decode + prediction/NMS + compressed raw writes',
         'generation_cost_included':False,'cold_initialization_in_per_image_latency':False,'warmups':3})
    save(out/'verification.json',{'status':'PASS','COCO_recomputed_from_raw':True,'metrics_sha256':sha256(out/'metrics.json')})
    save(out/'run.json',{'RunID':out.name,'ExperimentID':'cv261207','status':'PASS','mode':'DEVELOPMENT_PAIR',
                          'scope':'DEVELOPMENT_ONLY','images':248,'pair_id':pair_id,'study':study.name})
    seal(out);print('PAIR_PASS',out.name,flush=True)


def verify_pair(name,study):
    out=resolve_run(name);verify_hashes(out,('run.json','configuration.json','metrics.json','bootstrap.json','timing.json','predictions.jsonl.gz','ground_truth.json','image_manifest.jsonl'))
    r=load(out/'run.json')
    if r['status']!='PASS' or r['study']!=study.name or r['images']!=248:raise ValueError('Invalid completed pair.')
    return out


def repeat_check(left,right,study):
    a,b=verify_pair(left,study),verify_pair(right,study)
    for name in ('configuration.json','metrics.json','ground_truth.json','image_manifest.jsonl'):
        if sha256(a/name)!=sha256(b/name):raise ValueError('Repeat differs: '+name)
    with gzip.open(a/'predictions.jsonl.gz','rb') as x,gzip.open(b/'predictions.jsonl.gz','rb') as y:
        if x.read()!=y.read():raise ValueError('Repeat predictions differ.')
    first,second=load(a/'bootstrap.json'),load(b/'bootstrap.json')
    if first['intervals']!=second['intervals'] or first['draws']!=second['draws']:
        raise ValueError('Repeat bootstrap differs.')
    return {'status':'PASS','runs':[left,right],'metrics_sha256':sha256(a/'metrics.json'),
            'raw_prediction_content_equal':True,'paired_bootstrap_equal':True}


def execute(study,resume=False):
    ticket=study/'execution_ticket.json'
    if resume:
        original=load(ticket)
        elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(original['started_utc'])).total_seconds()
        if elapsed<0 or elapsed>=7200 or (study/'failure.json').exists():raise ValueError('Original deadline or failure forbids resume.')
        completed=load(study/'progress.json')['completed_runs']
        for name in completed:verify_pair(name,study)
    else:
        with ticket.open('x',encoding='utf-8') as stream:json.dump({'started_utc':datetime.now(timezone.utc).isoformat(),'budget_seconds':7200},stream)
        elapsed=0.;completed=[]
    started=time.perf_counter()-elapsed;plan=load(study/'protocol.json')
    lock=RUNS/'.gpu.lock'
    with lock.open('x',encoding='utf-8') as stream:stream.write(study.name)
    try:
        if (study/'cache_manifest.json').exists():
            for item in load(study/'cache_manifest.json'):checked(study/item['file'],item['file_sha256'])
        else:cache(study)
        schedule=[plan['pairs'][0],plan['pairs'][0],*plan['pairs'][1:]]
        if any(load(resolve_run(name)/'run.json')['pair_id']!=schedule[i]['id'] for i,name in enumerate(completed)):
            raise ValueError('Resume order differs from frozen schedule.')
        for index,spec in enumerate(schedule):
            if index<len(completed):continue
            elapsed=time.perf_counter()-started;remaining=7200-elapsed
            if index>=2:
                times=[load(resolve_run(n)/'timing.json')['pair_wall_seconds'] for n in completed[:2]]
                reserve=1.25*max(times)
                save(study/'resource_gate.json',{'qualification_runs':completed[:2],'seconds_reserved_per_pair':reserve,
                    'remaining_seconds':remaining,'tail_reserve_seconds':900,'status':'MEASURED_DEVELOPMENT_BUDGET_ONLY'})
                if remaining<reserve+900:break
            budget=min(1800,remaining-900)
            if budget<1:break
            before={p.name for p in RUNS.glob('cv261207_dev_pair_*')}
            log=study/f'pair_{index:02}.log'
            with log.open('x',encoding='utf-8') as stream:
                result=subprocess.run([sys.executable,'-B',str(Path(__file__)),'worker','--run',study.name,'--pair',spec['id']],
                        stdout=stream,stderr=subprocess.STDOUT,timeout=budget,
                        env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONIOENCODING':'utf-8'})
            fresh=[p.name for p in RUNS.glob('cv261207_dev_pair_*') if p.name not in before]
            if result.returncode or len(fresh)!=1:raise RuntimeError('Worker failed; no retry. Inspect '+log.name)
            verify_pair(fresh[0],study);completed.append(fresh[0])
            save(study/'progress.json',{'completed_runs':completed,'unique_pairs':len({load(resolve_run(n)/'run.json')['pair_id'] for n in completed}),
                          'elapsed_seconds':time.perf_counter()-started,'remaining_seconds':7200-(time.perf_counter()-started)})
            if index==1:save(study/'reproducibility.json',repeat_check(*completed[:2],study))
            print('Completed',index+1,'/ 13 runs |',fresh[0],flush=True)
        save(study/'execution_result.json',{'status':'COMPLETE' if len(completed)==13 else 'PARTIAL',
             'completed_runs':completed,'wall_seconds':time.perf_counter()-started,'remaining_seconds':7200-(time.perf_counter()-started),
             'formal_pairs':0,'new_training_runs':0})
    except BaseException:
        save(study/'failure.json',{'status':'PARTIAL','traceback':traceback.format_exc(),'completed_runs':completed,
                                 'elapsed_seconds':time.perf_counter()-started,'no_automatic_retry':True})
        raise
    finally:lock.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','execute','worker'))
    parser.add_argument('--run');parser.add_argument('--pair');parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    if args.mode=='prepare':prepare()
    elif args.mode=='execute':execute(resolve_run(args.run),args.resume)
    else:
        study=resolve_run(args.run)
        if (RUNS/'.gpu.lock').read_text(encoding='utf-8')!=study.name:raise ValueError('Use the supervised execution entry.')
        worker(study,args.pair)
