"""Prepare or explicitly authorize three matched full-split measurements."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
from protocol import ROOT,HERE,EXPERIMENT,load,save,source,sha256,write_atomic
from common import seal,verify_hashes
from matched_followup import ANALYSIS_ID,SEVERE_ID
from joint_batch import measure_pair,resource_guard
from main.pipeline import new_run

BUDGET=3600
VERSION='VM36_MATCHED_CONTEXT3_FP32_FULL548_V2'


def run_path(identity):
    if Path(identity).name!=identity:raise ValueError('Expected a plain RunID.')
    return ROOT/'datasets/local_runs'/EXPERIMENT/identity


def prepare(design):
    os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['YOLO_AUTOINSTALL']='false'
    verify_hashes(design,('B_ONE_CONTEXT_MANIFEST.json','NEXT_MATCHED_PROTOCOL.json','COST_AND_DISK.json'))
    proposal=load(design/'NEXT_MATCHED_PROTOCOL.json');pairs=load(design/'B_ONE_CONTEXT_MANIFEST.json')['pairs']
    if len(pairs)!=3 or len({s['id'] for s in pairs})!=3 or any(s['p']['id']!=SEVERE_ID or s['images']!=548 for s in pairs):raise ValueError('Unexpected matched design.')
    current=design.parent/ANALYSIS_ID;refs=load(current/'EVIDENCE_SOURCES.json')
    old=Path(refs['batch_protocol']['path']).parent;base=load(old/'EXECUTION_PROTOCOL.json')
    cache_file=old/'cache_manifests'/(SEVERE_ID+'.json');rows=load(cache_file)
    if len(rows)!=548 or len({r['image_id'] for r in rows})!=548:raise ValueError('Incomplete original pixel cache.')
    measured=next(Path(r['folder']) for r in refs['accepted_pairs'] if load(Path(r['folder'])/'configuration.json')['p']['id']==SEVERE_ID)
    verify_hashes(measured,('image_manifest.jsonl','configuration.json'))
    original_rows=[{k:v for k,v in json.loads(line).items() if k!='tensor'} for line in (measured/'image_manifest.jsonl').read_text().splitlines()]
    if rows!=original_rows or [r['image_id'] for r in rows]!=[r['id'] for r in load(Path(base['GT']['path']))['records']]:
        raise ValueError('Pixels do not match the accepted severe-context measurement.')
    import cv2
    import torch
    from ultralytics import YOLO
    from main.labels import VIS_NAMES
    for row in rows:
        p=Path(row['path'])
        if sha256(p)!=row['file_sha256']:raise ValueError('Source cache file changed.')
        pixels=cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        if list(pixels.shape)!=row['shape'] or hashlib.sha256(pixels.tobytes()).hexdigest()!=row['pixel_sha256']:
            raise ValueError('Cached pixels changed.')
    existing={r['pair_id'] for r in refs['accepted_pairs']}
    if any(s['id'] in existing for s in pairs):raise ValueError('Accepted pair would be repeated.')
    health={}
    for spec in pairs:
        d=spec['d'];key=d['checkpoint']
        if key in health:continue
        if sha256(Path(d['path']))!=d['sha256']:raise ValueError('Checkpoint SHA differs.')
        model=YOLO(d['path']).model.cpu().float().eval()
        if list(model.names.values())!=VIS_NAMES:raise ValueError('Native10 order differs.')
        invalid=[k for k,v in model.state_dict().items() if v.is_floating_point() and
            (not bool(torch.isfinite(v).all()) or k.endswith('running_var') and bool((v<0).any()))]
        if invalid:raise ValueError('Unhealthy checkpoint.')
        health[key]={'source':source(Path(d['path'])),'native10':VIS_NAMES,'tensor_health':'CPU_VERIFIED'}
    out,_=new_run({'output_root':str(design.parent)},EXPERIMENT,'matched3_preparation')
    plan={**base,'version':VERSION,'pairs':pairs,'worker_budget_seconds':BUDGET,
        'parent_design':source(design/'B_ONE_CONTEXT_MANIFEST.json'),'cache_manifest':source(cache_file),
        'cache_reused_from':old.name,'prior_accepted':source(current/'SHA256SUMS.txt'),
        'scientific_question':proposal['primary'],'secondary_question':proposal['secondary'],
        'context_warning':proposal['context_warning'],'selection':proposal['selection'],
        'cost_screening':load(design/'COST_AND_DISK.json')['screening_per_pair_seconds'],
        'measure_pair_source':source(HERE/'joint_batch.py')}
    save(out/'EXECUTION_PROTOCOL.json',plan,out);save(out/'CHECKPOINT_HEALTH.json',health,out)
    save(out/'GPU_READINESS.json',{'status':'CPU_PREPARED_AWAITING_NEW_GPU_AUTHORIZATION','actual_GPU_preflight':'NOT_RUN',
        'new_pairs':3,'images_per_pair':548,'proposed_worker_budget_seconds':BUDGET,'authorized_worker_seconds':None,
        'cache_images_verified':548,'new_model_forward_backward':0,
        'safe_resume_scope':'Completed sealed pairs only; no exact mid-pair or cross-reboot trajectory claim.'},out)
    shutil.copyfile(Path(__file__),out/'executed_source.py')
    save(out/'inputs.json',{'design':source(design/'SHA256SUMS.txt'),'source':source(Path(__file__))},out)
    write_atomic(out/'diary.md','# Matched3 preparation\n\nVerified three missing identities, original548-pixel cache and CPU checkpoint health. No GPU initialization, inference or training. A separate exact-budget author record is required.\n',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'MATCHED3_PREPARATION','status':'AWAITING_EXPLICIT_GPU_BUDGET'},out,replace=True)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>Matched3 CPU preparation</h1><p>GPU not run.</p><a href="GPU_READINESS.json">Readiness</a> · <a href="EXECUTION_PROTOCOL.json">Frozen proposal</a>',out)
    seal(out);print(out,flush=True)


def validate_authorization(preparation,authorization):
    a=load(authorization)
    expected={'scope':'VM36_MATCHED3_FULL548','preparation_RunID':preparation.name,
        'protocol_SHA256':sha256(preparation/'EXECUTION_PROTOCOL.json'),'worker_budget_seconds':BUDGET,'attempts':1}
    if any(a.get(k)!=v for k,v in expected.items()) or not a.get('author_instruction'):
        raise ValueError('A new exact-scope author record is required; prior budgets cannot be reused.')
    return a


def completed_from(previous,plan):
    if load(previous/'EXECUTION_PROTOCOL.json')!=plan:raise ValueError('Resume protocol changed.')
    found={}
    for spec in plan['pairs']:
        folder=previous/'pairs'/spec['id']
        if not (folder/'SHA256SUMS.txt').exists():continue
        verify_hashes(folder,('configuration.json','metrics.json','bootstrap.json','predictions.jsonl.gz','ground_truth.json','verification.json'))
        if load(folder/'configuration.json')!=spec or load(folder/'metrics.json')['images']!=548:
            raise ValueError('Completed output identity differs.')
        verified=load(folder/'verification.json')
        if not verified['COCO_recomputed_from_raw'] or verified['protocol_sha256']!=sha256(previous/'EXECUTION_PROTOCOL.json'):
            raise ValueError('Missing matching raw replay.')
        if sha256(folder/'ground_truth.json')!=plan['GT']['sha256']:raise ValueError('Resume GT changed.')
        found[spec['id']]=str(folder)
    return found


def worker(out):
    from evaluation import runtime
    from development_v1 import Model
    runtime(out);start=time.perf_counter();plan=load(out/'EXECUTION_PROTOCOL.json')
    rows=load(Path(plan['cache_manifest']['path']));payload=load(Path(plan['GT']['path']))
    result=load(out/'REUSED_COMPLETED.json');completed=[];devices={};active=None
    try:
        for key in ('FMP','FMD','GT','image_members','ZIP','generator','cache_manifest','measure_pair_source'):
            record=plan[key]
            if sha256(Path(record['path']))!=record['sha256']:raise ValueError('Frozen input changed: '+key)
        import importlib.util
        fast_path=HERE.parent/'cv261207/development_fast_coco.py'
        spec=importlib.util.spec_from_file_location('new_fast_coco_accumulate',fast_path)
        fast=importlib.util.module_from_spec(spec);spec.loader.exec_module(fast);sys.modules['new_fast_coco_accumulate']=fast
        for condition in plan['pairs']:
            if condition['id'] in result:continue
            if completed and BUDGET-(time.perf_counter()-start)<max(plan['cost_screening'],max(r['seconds'] for r in completed)*1.25+90):break
            active=condition['id'];resource_guard(initial=not devices);pair_start=time.perf_counter()
            folder=out/'pairs'/active;folder.mkdir(parents=True)
            key=condition['d']['checkpoint']
            if key not in devices:devices[key]=Model(condition['d'])
            timing=measure_pair(out,plan,condition,rows,devices[key],pair_start,payload)
            manifest=[json.loads(line) for line in (folder/'image_manifest.jsonl').read_text().splitlines()]
            if len(manifest)!=548 or any(r['tensor']!={'NCHW':[1,3,condition['d']['resolution'],condition['d']['resolution']],'dtype':'torch.float32'} for r in manifest):
                raise ValueError('Actual full-split tensor check failed.')
            metrics=load(folder/'metrics.json');all_area=next(r for r in metrics['per_area'] if r['size']=='all')
            save(folder/'CACHE_COST.json',{'seconds':0,'reuse':plan['cache_manifest'],'images':548},out)
            write_atomic(folder/'report.md','# Matched context measurement\n\nPROBLEM: Complete one missing checkpoint/resolution corner.\nWHY: The frozen severe-context contrast needs this point.\nSOLUTION: The original548 pixels, GT and native10 FP32 protocol; existing cache reused.\nEVIDENCE: '+json.dumps(all_area)+'\n\nFixed-checkpoint development evidence only. Metrics, bootstrap and timing remain separate.\n',out)
            seal(folder);completed.append({'pair_id':active,'seconds':time.perf_counter()-pair_start,'folder':str(folder)})
            save(out/'progress.json',{'completed':completed,'reused':result,'remaining_seconds':BUDGET-(time.perf_counter()-start)},out,replace=(out/'progress.json').exists())
            print('MATCHED_PAIR_ACCEPTED',active,flush=True)
    except BaseException:save(out/'failure.json',{'pair_id':active,'traceback':traceback.format_exc(),'automatic_retry':False},out)
    save(out/'worker_result.json',{'status':'VALIDATED' if len(completed)+len(result)==3 else 'PARTIAL',
        'new_completed':len(completed),'reused_completed':len(result),'planned':3,'worker_seconds':time.perf_counter()-start},out)


def execute(preparation,authorization):
    verify_hashes(preparation,('EXECUTION_PROTOCOL.json','GPU_READINESS.json'))
    a=validate_authorization(preparation,authorization);plan=load(preparation/'EXECUTION_PROTOCOL.json')
    lock=preparation.parent/'.matched_context_gpu.lock'
    if any((preparation.parent/name).exists() for name in ('.vm36_joint_gpu.lock','.full548_gpu.lock','.missing_corners_gpu.lock',lock.name)):
        raise RuntimeError('Another VM36 worker lock is present.')
    resumed=completed_from(run_path(a['resume_from']),plan) if a.get('resume_from') else {}
    ticket=preparation.parent/('.matched3_'+sha256(authorization)[:16]+'.ticket')
    fd=os.open(ticket,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,preparation.name.encode());os.close(fd)
    out,_=new_run({'output_root':str(preparation.parent)},EXPERIMENT,'matched3_full548')
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    from development_guard import bounded_run
    begin=time.perf_counter()
    try:
        shutil.copyfile(preparation/'EXECUTION_PROTOCOL.json',out/'EXECUTION_PROTOCOL.json')
        save(out/'REUSED_COMPLETED.json',resumed,out);save(out/'authorization.json',a,out)
        with (out/'worker.log').open('x',encoding='utf-8') as stream:
            code=bounded_run([sys.executable,'-B',str(Path(__file__)),'--worker',str(out)],stream,BUDGET,
                dict(os.environ,YOLO_AUTOINSTALL='false',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',PYTHONIOENCODING='utf-8'))
        if code:raise RuntimeError('Worker failed; no automatic retry.')
    except BaseException:save(out/'supervisor_failure.json',{'traceback':traceback.format_exc(),'automatic_retry':False},out)
    finally:
        lock.unlink();result=load(out/'worker_result.json') if (out/'worker_result.json').exists() else {'status':'PARTIAL'}
        save(out/'supervisor.json',{'hard_budget_seconds':BUDGET,'actual_supervision_seconds':time.perf_counter()-begin,'automatic_restart':False},out)
        save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'MATCHED3_FULL548',**result},out,replace=True)
        print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',metavar='DESIGN_RUN');g.add_argument('--execute',metavar='PREPARATION_RUN');g.add_argument('--worker',type=Path)
    p.add_argument('--authorization',type=Path);args=p.parse_args()
    if args.prepare:prepare(run_path(args.prepare))
    elif args.worker:worker(args.worker)
    elif args.authorization:execute(run_path(args.execute),args.authorization)
    else:p.error('A new explicit author budget record is required for execution.')
