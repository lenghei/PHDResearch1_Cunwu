"""Recover saved evidence and continue only unfinished development pairs."""
import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

from common import HERE, RUNS, load, save, source, sha256, new_run, resolve_run, seal
from development_v1 import bootstrap_exact, verify_pair, checked
from evaluation import evaluate_saved

STUDY='cv261207_development_20261107T235344_5fcd2d79'
FIRST='cv261207_dev_pair_20261107T235417_9a537fcd'
REPEAT='cv261207_dev_pair_20261107T235723_5c48222f'


def durable(out):
    for p in out.iterdir():
        if p.is_file():
            with p.open('r+b') as stream: os.fsync(stream.fileno())


def process_gate():
    command="Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -match 'development_v1.py.*worker|development_canary_v1.py.*worker' } | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
    value=subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True,check=True)
    if value.stdout.strip(): raise RuntimeError('An experiment worker is still running: '+value.stdout)


def recover():
    process_gate();study=resolve_run(STUDY);first=verify_pair(FIRST,study);old=resolve_run(REPEAT)
    out=new_run('shutdown_recovery');start=time.perf_counter()
    evidence=[]
    for p in sorted(old.iterdir()):
        if p.is_file(): evidence.append(source(p))
    save(out/'interrupted_evidence.json',{'user_report':'Manual forced power-off on 2026-11-08; cause not inferred.',
        'source_run':REPEAT,'source_files':evidence,'original_ticket':source(study/'execution_ticket.json'),
        'old_run_unchanged':True,'GPU_reexecution':False})
    for name in ('configuration.json','metrics.json','ground_truth.json','image_manifest.jsonl'):
        if sha256(first/name)!=sha256(old/name): raise ValueError('Repeat source differs: '+name)
        shutil.copyfile(old/name,out/name)
    with gzip.open(first/'predictions.jsonl.gz','rb') as a,gzip.open(old/'predictions.jsonl.gz','rb') as b:
        left,right=a.read(),b.read()
        if left!=right or len(right.splitlines())!=248:raise ValueError('Saved repeat predictions differ or are incomplete.')
    shutil.copyfile(old/'predictions.jsonl.gz',out/'predictions.jsonl.gz')
    metrics,ev=evaluate_saved(out)
    if metrics!=load(first/'metrics.json'):raise ValueError('CPU replay differs.')
    bootstrap_exact(ev,study,out)
    original=load(first/'bootstrap.json');fresh=load(out/'bootstrap.json')
    if original['intervals']!=fresh['intervals'] or original['draws']!=fresh['draws']:
        raise ValueError('Recovered repeat bootstrap differs.')
    save(out/'timing.json',{'status':'NOT_RECOVERABLE','mean_pipeline_ms':None,'mean_pipeline_ci95':None,
         'pair_wall_seconds':None,'reason':'Original timing file contains NUL bytes after forced power-off.',
         'CPU_recovery_wall_seconds':time.perf_counter()-start})
    save(out/'verification.json',{'status':'PASS','COCO_recomputed_from_raw':True,
        'repeat_predictions_equal':True,'repeat_metrics_equal':True,'repeat_bootstrap_equal':True,
        'timing_repeat':'NOT_VERIFIED','metrics_sha256':sha256(out/'metrics.json')})
    save(out/'run.json',{'RunID':out.name,'ExperimentID':'cv261207','status':'PASS','mode':'CPU_REPEAT_RECOVERY',
         'scope':'DEVELOPMENT_ONLY','images':248,'pair_id':load(first/'run.json')['pair_id'],'study':STUDY,
         'source_interrupted_run':REPEAT})
    lock=RUNS/'.gpu.lock'
    if lock.exists():
        if lock.read_text(encoding='utf-8')!=STUDY:raise ValueError('Unrecognized lock owner.')
        target=out/'stale_gpu_lock.txt'
        if lock.resolve().parent!=RUNS.resolve() or target.resolve().parent!=out.resolve():raise ValueError('Move boundary error.')
        lock.rename(target)
    seal(out);durable(out);print('CPU_RECOVERY_PASS',out.name,flush=True)


def execute(recovery):
    process_gate();study=resolve_run(STUDY);verify_pair(FIRST,study);verify_pair(recovery.name,study)
    if not load(recovery/'verification.json')['repeat_bootstrap_equal']:raise ValueError('Repeat gate not met.')
    plan=load(study/'protocol.json');completed=[FIRST];out=new_run('development_continuation')
    budget=5400.;started=time.perf_counter();lock=RUNS/'.gpu.lock'
    save(out/'author_execution_ticket.json',{'authorization':'User continuation after manual shutdown, 2026-11-08',
        'started_utc':datetime.now(timezone.utc).isoformat(),'budget_seconds':budget,
        'original_study':STUDY,'old_ticket_unchanged':source(study/'execution_ticket.json'),
        'repeat_recovery':recovery.name,'remaining_pairs':11,'training_authorized_here':False,
        'code':source(Path(__file__))})
    for item in load(study/'cache_manifest.json'):checked(study/item['file'],item['file_sha256'])
    with lock.open('x',encoding='utf-8') as stream:stream.write(STUDY)
    try:
        measured=load(resolve_run(FIRST)/'timing.json')['pair_wall_seconds']
        save(out/'resource_gate.json',{'measured_complete_pair_seconds':measured,'repeat_AP_and_prediction_verified':True,
            'repeat_timing':'LOST','forecast_per_pair_seconds':1.5*measured,'tail_reserve_seconds':900,
            'planned_remaining_pairs':11,'forecast_remaining_pair_seconds':11*1.5*measured,
            'new_continuation_budget_seconds':budget,'old_ticket_not_resumed':True})
        for index,spec in enumerate(plan['pairs'][1:],1):
            remaining=budget-(time.perf_counter()-started)
            if remaining<max(900,1.5*measured)+900:break
            before={p.name for p in RUNS.glob('cv261207_dev_pair_*')}
            with (out/f'pair_{index:02}.log').open('x',encoding='utf-8') as stream:
                result=subprocess.run([sys.executable,'-B',str(HERE/'development_v1.py'),'worker','--run',STUDY,'--pair',spec['id']],
                    stdout=stream,stderr=subprocess.STDOUT,timeout=min(1800,remaining-900),
                    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONIOENCODING':'utf-8'})
            fresh=[p for p in RUNS.glob('cv261207_dev_pair_*') if p.name not in before]
            if result.returncode or len(fresh)!=1:raise RuntimeError('Worker failed; no retry. Inspect pair log.')
            verify_pair(fresh[0].name,study);durable(fresh[0]);completed.append(fresh[0].name)
            save(out/'progress.json',{'completed_runs':completed,'unique_pairs':len(completed),
                 'elapsed_seconds':time.perf_counter()-started,'remaining_seconds':budget-(time.perf_counter()-started)})
            durable(out);print('Completed unique pairs',len(completed),'/ 12 |',fresh[0].name,flush=True)
        save(out/'result.json',{'status':'COMPLETE' if len(completed)==12 else 'PARTIAL','completed_runs':completed,
            'repeat_recovery':recovery.name,'unique_pairs':len(completed),'formal_pairs':0,
            'scope':'DEVELOPMENT_ONLY','wall_seconds':time.perf_counter()-started,'budget_seconds':budget})
    except BaseException:
        save(out/'failure.json',{'status':'PARTIAL','completed_runs':completed,'traceback':traceback.format_exc(),
             'wall_seconds':time.perf_counter()-started,'no_retry':True})
        raise
    finally:
        if lock.exists() and lock.read_text(encoding='utf-8')==STUDY:lock.unlink()
        seal(out);durable(out)
    print('CONTINUATION_FINISHED',out.name,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('recover','execute'));parser.add_argument('--recovery')
    args=parser.parse_args()
    if args.mode=='recover':recover()
    else:execute(resolve_run(args.recovery))
