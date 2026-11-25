"""Complete the frozen20-by-four grid under a new total-day authorization."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
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
from common import seal,verify_hashes,import_file
from historical_review import table,csv_file
from full_development import PREP
from full_analysis import paired_effect,tested_subset_state
from matched_followup import ANALYSIS_ID,SEVERE_ID
from joint_batch import measure_pair,resource_guard
from main.pipeline import new_run

VERSION='VM36_FULL20_FOUR_SETTINGS_COMPLETION_V1'


def path_for(identity):
    if Path(identity).name!=identity:raise ValueError('Expected a plain RunID.')
    return ROOT/'datasets/local_runs'/EXPERIMENT/identity


def remaining_seconds(authorization):
    deadline=datetime.fromisoformat(authorization['global_deadline_UTC'])
    return (deadline-datetime.now(timezone.utc)).total_seconds()


def validate_day_budget(a):
    start=datetime.fromisoformat(a['global_start_UTC']);deadline=datetime.fromisoformat(a['global_deadline_UTC'])
    if (deadline-start).total_seconds()!=86400:raise ValueError('Total-day budget was reset or extended.')
    parent=a['parent_authorization']
    if sha256(Path(parent['path']))!=parent['sha256'] or load(Path(parent['path']))['recorded_at_UTC']!=a['global_start_UTC']:
        raise ValueError('Original new-study start differs.')


def read_pair(folder):
    checks=verify_hashes(folder,('configuration.json','metrics.json','bootstrap.json','timing.json',
        'predictions.jsonl.gz','ground_truth.json','verification.json','image_manifest.jsonl'))
    spec=load(folder/'configuration.json');m=load(folder/'metrics.json');b=load(folder/'bootstrap.json');t=load(folder/'timing.json')
    if m['images']!=548 or b['draws']['samples']!=200 or len(b['draws']['image_ids'])!=548:
        raise ValueError('Not a full548,200-draw measurement.')
    v=load(folder/'verification.json')
    if not v['COCO_recomputed_from_raw'] or v['protocol_sha256']!=sha256(folder.parent.parent/'EXECUTION_PROTOCOL.json'):
        raise ValueError('Raw replay protocol differs.')
    return {'spec':spec,'metrics':m,'bootstrap':b,'timing':t,'folder':str(folder),'checks':checks}


def initial_sources(first):
    base=first.parent/ANALYSIS_ID;refs=load(base/'EVIDENCE_SOURCES.json')
    verify_hashes(base,('summary.json','EVIDENCE_SOURCES.json'))
    if load(first/'run.json')['status']!='VALIDATED' or load(first/'worker_result.json')['new_completed']!=3:
        raise ValueError('The initial three matched points are incomplete.')
    found={}
    for row in refs['accepted_pairs']:
        item=read_pair(Path(row['folder']));key=item['spec']['id']
        if key in found:raise ValueError('Repeated source point.')
        found[key]=item
    for spec in load(first/'EXECUTION_PROTOCOL.json')['pairs']:
        item=read_pair(first/'pairs'/spec['id'])
        if spec['id'] in found or item['spec']!=spec:raise ValueError('New matched identity differs.')
        found[spec['id']]=item
    if len(found)!=29:raise ValueError('Expected26 original plus3 newly measured points.')
    if not (first/'SHA256SUMS.txt').exists():
        save(first/'EVIDENCE_TREE.json',{str(p.relative_to(first)):sha256(p) for p in (first/'pairs').rglob('*') if p.is_file()},first)
        write_atomic(first/'diary.md','# Three matched corners\n\nAll three full548-image points completed under their3600-second cap. Raw COCO replay and200 paired draws are saved. Original26 were not repeated.\n',first)
        seal(first)
    return found


def prepare(design,first,authorization,resume=None):
    a=load(authorization)
    if a['scope']!='VM36_FULL54_WITHIN_ONE_DAY' or a['new_pairs_total']!=54 or a['global_wall_budget_seconds']!=86400:
        raise ValueError('Exact new full54 authorization required.')
    validate_day_budget(a)
    if remaining_seconds(a)<=180:raise ValueError('Global day budget exhausted.')
    verify_hashes(design,('B_COMPLETE20_MANIFEST.json','COST_AND_DISK.json'))
    if sha256(design/'B_COMPLETE20_MANIFEST.json')!=a['design_manifest_SHA256']:raise ValueError('Frozen full54 manifest changed.')
    found=initial_sources(first)
    all_new=load(design/'B_COMPLETE20_MANIFEST.json')['pairs']
    if resume:
        if a.get('resume_from_RunID')!=resume.name or load(resume/'run.json')['status'] not in ('VALIDATED','PARTIAL'):
            raise ValueError('Resume needs a new explicit record and a terminal source run.')
        previous=load(resume/'EXECUTION_PROTOCOL.json')
        if previous['global_deadline_UTC']!=a['global_deadline_UTC'] or previous['design']['sha256']!=a['design_manifest_SHA256']:
            raise ValueError('Resume changed the frozen design or reset its budget.')
        for spec in all_new:
            folder=resume/'pairs'/spec['id']
            if not (folder/'SHA256SUMS.txt').exists():continue
            item=read_pair(folder)
            if item['spec']!=spec or spec['id'] in found:raise ValueError('Repeated or changed resume point.')
            found[spec['id']]=item
    pairs=[s for s in all_new if s['id'] not in found]
    if len(all_new)!=54 or len(pairs)!=80-len(found) or len({s['id'] for s in pairs})!=len(pairs):raise ValueError('Remaining-grid identity differs.')
    if not resume and len(pairs)!=51:raise ValueError('Expected exactly51 remaining points for the first attempt.')
    pids={r['spec']['p']['id'] for r in found.values()};dids={r['spec']['d']['id'] for r in found.values()}
    expected={p+'__'+d for p in pids for d in dids}
    if len(pids)!=20 or len(dids)!=4 or set(found)|{s['id'] for s in pairs}!=expected:
        raise ValueError('Not the original20-by-four population.')
    for s in pairs:
        if s['p']['id'] not in pids or s['d']['id'] not in dids or s['images']!=548:
            raise ValueError('A new degradation or setting was added.')
    for d in {r['spec']['d']['path']:r['spec']['d']['sha256'] for r in found.values()}.items():
        if sha256(Path(d[0]))!=d[1]:raise ValueError('Checkpoint changed.')
    out,_=new_run({'output_root':str(first.parent)},EXPERIMENT,'full20_preparation')
    shutil.copyfile(Path(__file__),out/'executed_source.py')
    pixel_refs={};pixel_dir=out/'pixel_manifests';pixel_dir.mkdir()
    expected_ids=load(first/'EXECUTION_PROTOCOL.json')['GT']
    gt_ids=[r['id'] for r in load(Path(expected_ids['path']))['records']]
    for p in sorted(pids):
        item=next(v for v in found.values() if v['spec']['p']['id']==p)
        rows=[{k:v for k,v in json.loads(line).items() if k!='tensor'} for line in (Path(item['folder'])/'image_manifest.jsonl').read_text().splitlines()]
        if [r['image_id'] for r in rows]!=gt_ids:raise ValueError('Pixel/source image positions differ.')
        save(pixel_dir/(p+'.json'),rows,out);pixel_refs[p]=source(pixel_dir/(p+'.json'))
    base=load(first/'EXECUTION_PROTOCOL.json');cost=load(design/'COST_AND_DISK.json')
    pairs.sort(key=lambda s:(s['d']['checkpoint'],s['d']['resolution'],s['p']['id']))
    plan={**base,'version':VERSION,'pairs':pairs,'pixel_manifests':pixel_refs,'global_authorization':source(authorization),
        'authorization':source(authorization),'worker_budget_seconds':None,'worker_budget_seconds_max':86400,
        'planned_new_pairs':len(pairs),'cache_manifest':None,
        'design':source(design/'B_COMPLETE20_MANIFEST.json'),'source_analysis_RunID':ANALYSIS_ID,
        'source_analysis':source(first.parent/ANALYSIS_ID/'SHA256SUMS.txt'),
        'first3_RunID':first.name,'initial_pairs':len(found),'target_pairs':80,'remaining_pairs':len(pairs),
        'per_pair_screening_seconds':cost['screening_per_pair_seconds'],
        'scope':'Complete the same20 purposive P configurations x four fixed detector settings; no population estimate.',
        'order':'Checkpoint, resolution, then P ID; fixed before new responses. No PASS early stopping in this complete-grid study.',
        'pixel_policy':'Reuse the exact accepted548-pixel manifests; verify file/pixel SHA immediately before each prediction.',
        'global_deadline_UTC':a['global_deadline_UTC'],'global_wall_budget_seconds':86400}
    save(out/'EXECUTION_PROTOCOL.json',plan,out)
    save(out/'ACCEPTED_SOURCE_INDEX.json',[{'pair_id':k,'folder':v['folder'],'manifest':source(Path(v['folder'])/'SHA256SUMS.txt')} for k,v in found.items()],out)
    save(out/'inputs.json',{'first3':source(first/'SHA256SUMS.txt'),'design':source(design/'SHA256SUMS.txt'),
        'global_authorization':source(authorization),'source':source(Path(__file__))},out)
    if shutil.disk_usage(ROOT).free<len(pairs)*cost['screening_saved_pair_bytes']+8*1024**3:
        raise RuntimeError('Predicted evidence storage plus8GiB reserve does not fit.')
    save(out/'PREPARATION.json',{'status':'CPU_PREPARED','initial_pairs':len(found),'remaining_new_pairs':len(pairs),'target_pairs':80,
        'remaining_screening_seconds':len(pairs)*cost['screening_per_pair_seconds'],'remaining_global_seconds_at_prepare':remaining_seconds(a),
        'disk_free_bytes':shutil.disk_usage(ROOT).free,'new_degradation_population':False,'new_checkpoints':False,
        'initial3_actual_worker_seconds':load(first/'worker_result.json')['worker_seconds'],
        'budget_warning':'Initial severe-context timing is not a forecast for other conditions. Stop if projected cost no longer fits the total-day deadline.'},out)
    save(out/'PIXEL_MANIFEST_SHA256.json',{p.name:sha256(p) for p in pixel_dir.iterdir()},out)
    write_atomic(out/'diary.md','# Full20 preparation\n\nVerified26 original and3 new measurements. Froze51 remaining points from the existing54-point complement. Identical original pixels and metrics are retained; all budgets share the original new-study start.\n',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'FULL20_PREPARATION','status':'CPU_PREPARED'},out,replace=True)
    seal(out);print(out,flush=True)


def worker(out):
    from evaluation import runtime
    from development_v1 import Model
    runtime(out);plan=load(out/'EXECUTION_PROTOCOL.json');a=load(Path(plan['global_authorization']['path']))
    start=time.perf_counter();budget=load(out/'LAUNCH_BUDGET.json')['worker_hard_seconds']
    fast=import_file('vm36_full20_fast',HERE.parent/'cv261207/development_fast_coco.py');sys.modules['new_fast_coco_accumulate']=fast
    payload=load(Path(plan['GT']['path']));completed=[];devices={};active=None;stop='ALL_COMPLETED'
    try:
        for key in ('FMP','FMD','GT','image_members','ZIP','generator','measure_pair_source','global_authorization'):
            record=plan[key]
            if sha256(Path(record['path']))!=record['sha256']:raise ValueError('Protected input changed: '+key)
        for number,spec in enumerate(plan['pairs'],1):
            left=min(budget-(time.perf_counter()-start),remaining_seconds(a)-120)
            ref=max([plan['per_pair_screening_seconds']]+[r['complete_seconds']*1.25+90 for r in completed])
            if left<ref:stop='BUDGET_ADMISSION_STOP';break
            if (len(plan['pairs'])-len(completed))*ref>left:stop='PROJECTED_TOTAL_EXCEEDS_DAY';break
            resource_guard(initial=not devices);active=spec['id'];begin=time.perf_counter()
            p_ref=plan['pixel_manifests'][spec['p']['id']]
            if sha256(Path(p_ref['path']))!=p_ref['sha256']:raise ValueError('Pixel manifest changed.')
            rows=load(Path(p_ref['path']));folder=out/'pairs'/active;folder.mkdir(parents=True)
            key=spec['d']['checkpoint']
            if key not in devices:devices[key]=Model(spec['d'])
            t=measure_pair(out,plan,spec,rows,devices[key],begin,payload)
            manifest=[json.loads(line) for line in (folder/'image_manifest.jsonl').read_text().splitlines()]
            if [r['image_id'] for r in manifest]!=[r['id'] for r in payload['records']] or any(r['tensor']!={'NCHW':[1,3,spec['d']['resolution'],spec['d']['resolution']],'dtype':'torch.float32'} for r in manifest):
                raise ValueError('Actual tensor or image population differs.')
            m=load(folder/'metrics.json');all_area=next(r for r in m['per_area'] if r['size']=='all')
            save(folder/'CACHE_COST.json',{'seconds':0,'cache_manifest':p_ref,'source':'Reused accepted pixels; no regeneration.'},out)
            write_atomic(folder/'report.md','# Full20 missing-point measurement\n\nPROBLEM: Complete a missing fixed-P/fixed-D response.\nWHY: Coverage alone does not identify all four detector outcomes.\nSOLUTION: All548 original images, native10 FP32 and the unchanged COCO/ignore protocol.\nEVIDENCE: '+json.dumps(all_area)+'\n\nSee configuration.json, bootstrap.json and timing.json. Development evidence only; no architecture causality.\n',out)
            seal(folder);elapsed=time.perf_counter()-begin
            completed.append({'pair_id':active,'number':number,'complete_seconds':elapsed,'folder':str(folder)})
            save(out/'progress.json',{'completed':completed,'remaining_points':len(plan['pairs'])-len(completed),'remaining_global_seconds':remaining_seconds(a),
                'projected_remaining_screening_seconds':(len(plan['pairs'])-len(completed))*ref},out,replace=(out/'progress.json').exists())
            print('FULL20_PAIR_ACCEPTED',len(completed),'/',len(plan['pairs']),active,elapsed,flush=True)
    except BaseException:
        stop='SAFETY_OR_EXECUTION_STOP';save(out/'failure.json',{'pair_id':active,'traceback':traceback.format_exc(),'automatic_retry':False},out)
    save(out/'worker_result.json',{'status':'VALIDATED' if len(completed)==len(plan['pairs']) else 'PARTIAL','completed_new_pairs':len(completed),
        'planned_new_pairs':len(plan['pairs']),'worker_seconds':time.perf_counter()-start,'stop_reason':stop},out)


def execute(preparation,authorization):
    verify_hashes(preparation,('EXECUTION_PROTOCOL.json','PREPARATION.json','ACCEPTED_SOURCE_INDEX.json'))
    plan=load(preparation/'EXECUTION_PROTOCOL.json');a=load(authorization)
    validate_day_budget(a)
    if source(authorization)!=plan['global_authorization'] or a['scope']!='VM36_FULL54_WITHIN_ONE_DAY':raise ValueError('Authorization differs.')
    seconds=int(remaining_seconds(a)-180)
    if seconds<len(plan['pairs'])*plan['per_pair_screening_seconds']:raise RuntimeError('Conservative remaining-grid cost does not fit the global day budget.')
    root=preparation.parent;lock=root/'.full20_completion_gpu.lock'
    if any((root/name).exists() for name in ('.matched_context_gpu.lock','.vm36_joint_gpu.lock',lock.name)):
        raise RuntimeError('Another VM36 GPU worker is active.')
    ticket=root/('.full20_'+sha256(authorization)[:16]+'.ticket')
    fd=os.open(ticket,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,preparation.name.encode());os.close(fd)
    out,_=new_run({'output_root':str(root)},EXPERIMENT,'full20_remaining51')
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    from development_guard import bounded_run
    begin=time.perf_counter()
    try:
        shutil.copyfile(preparation/'EXECUTION_PROTOCOL.json',out/'EXECUTION_PROTOCOL.json')
        shutil.copyfile(Path(__file__),out/'executed_source.py')
        save(out/'LAUNCH_BUDGET.json',{'worker_hard_seconds':seconds,'global_deadline_UTC':a['global_deadline_UTC'],
            'same_original_start_UTC':a['global_start_UTC'],'reserved_postprocessing_seconds':180,'no_budget_reset':True},out)
        save(out/'inputs.json',{'preparation':source(preparation/'SHA256SUMS.txt'),'authorization':source(authorization),'source':source(Path(__file__))},out)
        with (out/'worker.log').open('x',encoding='utf-8') as stream:
            code=bounded_run([sys.executable,'-B',str(Path(__file__)),'--worker',str(out)],stream,seconds,
                dict(os.environ,YOLO_AUTOINSTALL='false',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',PYTHONIOENCODING='utf-8'))
        if code:raise RuntimeError('Worker failed; no automatic retry.')
    except BaseException:save(out/'supervisor_failure.json',{'traceback':traceback.format_exc(),'automatic_retry':False},out)
    finally:
        lock.unlink();result=load(out/'worker_result.json') if (out/'worker_result.json').exists() else {'status':'PARTIAL'}
        save(out/'supervisor.json',{'actual_supervision_seconds':time.perf_counter()-begin,'worker_hard_seconds':seconds,
            'global_deadline_UTC':a['global_deadline_UTC'],'automatic_restart':False},out)
        save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'FULL20_REMAINING51',**result},out,replace=True)
        print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',metavar='DESIGN_RUN');g.add_argument('--execute',metavar='PREPARATION_RUN');g.add_argument('--worker',type=Path)
    p.add_argument('--first3');p.add_argument('--resume-from');p.add_argument('--authorization',type=Path);args=p.parse_args()
    if args.prepare and args.first3 and args.authorization:prepare(path_for(args.prepare),path_for(args.first3),args.authorization,path_for(args.resume_from) if args.resume_from else None)
    elif args.worker:worker(args.worker)
    elif args.execute and args.authorization:execute(path_for(args.execute),args.authorization)
    else:p.error('Preparation needs --first3 and the new authorization; execution needs the exact authorization.')
