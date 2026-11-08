"""One clean-control engineering canary within the continuation deadline."""
import argparse
from datetime import datetime,timezone
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from common import HERE,RUNS,load,save,new_run,resolve_run,source,seal,sha256,verify_hashes
from development_recovery import process_gate,durable
from development_guard import bounded_run


def execute(preparation,continuation):
    process_gate();ticket=load(continuation/'author_execution_ticket.json');result=load(continuation/'result.json')
    for existing in RUNS.glob('cv261207_clean_canary_*'):
        prior=existing/'author_execution_ticket.json'
        if prior.exists() and load(prior).get('continuation')==continuation.name:
            raise RuntimeError('This continuation already has a canary attempt; no automatic retry.')
    elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(ticket['started_utc'])).total_seconds()
    remaining=ticket['budget_seconds']-elapsed
    if result['status']!='COMPLETE' or remaining<900:raise RuntimeError('No admitted complete study or sufficient budget.')
    verify_hashes(preparation,('protocol.json','selection.json','dataset.yaml','full_label_audit.csv'))
    out=new_run('clean_canary');lock=RUNS/'.gpu.lock';start=time.perf_counter()
    save(out/'author_execution_ticket.json',{'source_authorization':'2026-11-07 instructions Stage 5 plus 2026-11-08 continuation',
          'preparation':preparation.name,'continuation':continuation.name,'budget_seconds':600,
          'continuation_remaining_seconds_at_start':remaining,'group':'clean_control','attempts':1,
          'scope':'CANARY_ENGINEERING_ONLY / DEVELOPMENT_ONLY','long_training_authorized':False,
          'worker_source':source(HERE/'development_canary_v1.py'),'supervisor_source':source(Path(__file__))})
    with lock.open('x',encoding='utf-8') as stream:stream.write(out.name)
    try:
        with (out/'execution.log').open('x',encoding='utf-8') as stream:
            status=bounded_run([sys.executable,'-B',str(HERE/'development_canary_v1.py'),'worker',
                     '--run',preparation.name,'--output',out.name],stream,timeout=590,
                     env={**os.environ,'PYTHONIOENCODING':'utf-8','PYTHONDONTWRITEBYTECODE':'1'})
        if status or not (out/'result.json').exists():raise RuntimeError('Canary failed; inspect execution.log. No retry.')
        checked=load(out/'result.json')
        if checked['status']!='CANARY_PASS' or checked['optimizer_steps']!=1:raise ValueError('Canary acceptance failed.')
        save(out/'supervisor.json',{'status':'CANARY_PASS','total_worker_wall_seconds':time.perf_counter()-start,
             'worker_limit_seconds':600,'FP32_reload':load(out/'checkpoint_health.json'),
             'long_training':'NOT_STARTED','targeted_augmentation_training':'NOT_EVALUATED','gap_closure':'NOT_EVALUATED'})
    except BaseException:
        save(out/'failure.json',{'status':'BLOCKED','traceback':traceback.format_exc(),'wall_seconds':time.perf_counter()-start,
             'automatic_retry':False,'long_training':'NOT_STARTED'})
        if not (out/'run.json').exists():save(out/'run.json',{'RunID':out.name,'ExperimentID':'cv261207','status':'BLOCKED','mode':'RQ4_CLEAN_CANARY'})
    finally:
        if lock.exists() and lock.read_text(encoding='utf-8')==out.name:lock.unlink()
        if not (out/'SHA256SUMS.txt').exists():seal(out)
        # The worker seals before its supervisor result exists; keep that seal immutable.
        save(out/'supervisor_evidence.json',{'supervisor':source(out/'supervisor.json') if (out/'supervisor.json').exists() else None,
             'failure':source(out/'failure.json') if (out/'failure.json').exists() else None,
             'execution_log':source(out/'execution.log'),'worker_seal':source(out/'SHA256SUMS.txt')})
        durable(out)
    print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--preparation',required=True);parser.add_argument('--continuation',required=True)
    args=parser.parse_args();execute(resolve_run(args.preparation),resolve_run(args.continuation))
