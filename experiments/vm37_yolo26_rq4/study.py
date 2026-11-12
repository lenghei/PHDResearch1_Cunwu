"""Local evidence paths and the single new RQ4 compute allocation."""
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
PUBLIC=HERE.parents[1]
ROOT=PUBLIC.parent
EXPERIMENT=HERE.name
RUNS=ROOT/'datasets/local_runs'/EXPERIMENT
VM36=HERE.parent/'vm36_supervisor_reproduction'
sys.path.insert(0,str(PUBLIC/'code'))
sys.path.append(str(VM36));sys.path.append(str(HERE.parent/'cv261207'))
from protocol import load, save, sha256, source, write_atomic
from main.pipeline import new_run
from common import seal, verify_hashes
from rq4_real_canary import PREPARED, SOURCE
from post80_replay import run_path as legacy_run

PREP=legacy_run(PREPARED)


def create(mode):
    return new_run({'output_root':str(RUNS)},EXPERIMENT,mode)[0]


def run_path(identity):
    if Path(identity).name!=identity or not identity.startswith(EXPERIMENT+'_'):raise ValueError('Invalid RunID.')
    return (RUNS/identity).resolve(strict=True)


def utc():
    return datetime.now(timezone.utc)


def runtime(out,cuda=True):
    os.environ.update(CUDA_VISIBLE_DEVICES='0' if cuda else '',YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',PYTHONDONTWRITEBYTECODE='1')
    for key in ('YOLO_CONFIG_DIR','MPLCONFIGDIR','TORCH_HOME','TORCH_EXTENSIONS_DIR','CUDA_CACHE_PATH','TEMP','TMP'):
        path=out/'runtime'/key;path.mkdir(parents=True,exist_ok=True);os.environ[key]=str(path)
    import tempfile
    tempfile.tempdir=str(out/'runtime/TEMP')


def budget_start(out,cap,phase):
    ledger=RUNS/'compute_allocation.json';lock=RUNS/'.gpu.lock'
    allocation=load(ledger)
    if allocation['total_seconds']!=172800 or allocation['admission_seconds']!=7200:raise ValueError('Allocation changed.')
    if lock.exists():raise RuntimeError('A GPU worker lease already exists; inspect it before continuing.')
    now=utc()
    if allocation['first_worker_UTC'] is None:
        allocation['first_worker_UTC']=now.isoformat()
        allocation['absolute_deadline_UTC']=(now+timedelta(seconds=172800)).isoformat()
        allocation['admission_deadline_UTC']=(now+timedelta(seconds=7200)).isoformat()
    remaining=allocation['total_seconds']-sum(r['charged_seconds'] for r in allocation['completed_workers'])
    remaining=min(remaining,(datetime.fromisoformat(allocation['absolute_deadline_UTC'])-now).total_seconds())
    if phase=='ADMISSION':
        remaining=min(remaining,(datetime.fromisoformat(allocation['admission_deadline_UTC'])-now).total_seconds())
    if remaining<60:raise RuntimeError('RESOURCE_BLOCKED: no safe worker time remains.')
    cap=min(cap,remaining)
    handle=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
    os.write(handle,json.dumps({'RunID':out.name,'parent_PID':os.getpid(),'started_UTC':now.isoformat()}).encode());os.close(handle)
    save(ledger,allocation,RUNS,replace=True)
    lease={'RunID':out.name,'phase':phase,'worker_seconds':cap,'started_UTC':now.isoformat(),
        'deadline_UTC':(now+timedelta(seconds=cap)).isoformat(),'allocation_author_SHA256':allocation['author_SHA256'],
        'budget_reset':False,'allocation_before':allocation}
    save(out/'WORKER_LEASE.json',lease,out)
    return cap


def budget_finish(out,seconds,status):
    ledger=RUNS/'compute_allocation.json';allocation=load(ledger)
    if any(r['RunID']==out.name for r in allocation['completed_workers']):raise RuntimeError('Worker already charged.')
    allocation['completed_workers'].append({'RunID':out.name,'charged_seconds':seconds,'status':status,'finished_UTC':utc().isoformat()})
    save(ledger,allocation,RUNS,replace=True)
    lock=RUNS/'.gpu.lock'
    if lock.exists() and load(lock)['RunID']==out.name:lock.unlink()


def expected_steps(images,epochs,batch=2,nbs=64,warmup_epochs=3,stop_epochs=None):
    import numpy as np
    batches=(images+batch-1)//batch
    accumulation=min(max(round(nbs/batch),1),batches*epochs)
    warmup=round(min(warmup_epochs,max(epochs-1,0))*batches)
    last=-1;count=0
    for position in range(batches*(stop_epochs or epochs)):
        active=max(1,int(np.interp(position,[0,warmup],[1,accumulation]).round())) if warmup and position<=warmup else accumulation
        if position-last>=active:count+=1;last=position
    return {'batches':batches*(stop_epochs or epochs),'updates':count,'warmup_batches':warmup,
        'last_update_position':last,'pending_microbatches':batches*(stop_epochs or epochs)-1-last}
