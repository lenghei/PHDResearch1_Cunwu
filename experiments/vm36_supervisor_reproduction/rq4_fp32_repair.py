"""Explicit FP32 canary repair within the original GPU deadline."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import rq4_real_canary as canary
from protocol import ROOT, EXPERIMENT, load, save, source, sha256, write_atomic
from main.pipeline import new_run
from common import seal, verify_hashes
from post80_replay import run_path
from development_guard import bounded_run


def remaining_allocation_seconds(allocation, now=None):
    start=datetime.fromisoformat(allocation['start_UTC'])
    deadline=datetime.fromisoformat(allocation['deadline_UTC'])
    if allocation['worker_seconds']!=600 or (deadline-start).total_seconds()!=600 or allocation['budget_reset_allowed'] is not False:
        raise ValueError('The original600-second allocation changed.')
    if allocation['formal_training_seconds']!=0:raise ValueError('Canary scope changed.')
    return (deadline-(now or datetime.now(timezone.utc))).total_seconds()


def repair(failed_id):
    failed=run_path(failed_id)
    verify_hashes(failed,('GPU_ALLOCATION.json','CANARY_PROTOCOL.json','worker_result.json'))
    allocation=load(failed/'GPU_ALLOCATION.json')
    deadline=datetime.fromisoformat(allocation['deadline_UTC'])
    remaining=remaining_allocation_seconds(allocation)
    if remaining<60 or load(failed/'worker_result.json')['status']!='CANARY_FAILED':
        raise RuntimeError('No admitted repair time or failed source.')
    root=failed.parent
    ticket=root/('.rq4_fp32_repair_'+sha256(failed/'GPU_ALLOCATION.json')[:16]+'_'+sha256(Path(canary.__file__))[:8]+'.ticket')
    handle=os.open(ticket,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(handle,failed_id.encode());os.close(handle)
    out,_=new_run({'output_root':str(root)},EXPERIMENT,'rq4_fp32_repair')
    plan=load(failed/'CANARY_PROTOCOL.json')
    plan['args']['amp']=False
    plan['precision_diagnostic']={'mode':'EXPLICIT_FP32_REPAIR','failed_run':source(failed/'worker_result.json'),
        'reason':'Finite detection loss but nonfinite scaled gradients at AMP scale65536. Retain strict health checks.',
        'causal_diagnosis':'SUSPECTED_AMP_OVERFLOW_NOT_YET_PROVEN','scientific_comparison':False}
    for key in ('data','project'):
        plan['args'][key]=str(out/'dataset.yaml') if key=='data' else str(out)
    plan['operational_overrides'].append('Explicit FP32 training repair; same optimizer,LR,batch,resolution,data,seed; original deadline retained.')
    canary.canary_contract(plan)
    save(out/'CANARY_PROTOCOL.json',plan,out)
    save(out/'GPU_ALLOCATION.json',allocation,out)
    save(out/'REPAIR_AUTHORIZATION.json',{'original_allocation':source(failed/'GPU_ALLOCATION.json'),
        'remaining_seconds_at_admission':remaining,'budget_reset':False,'precision_change':'AMPTrue_to_explicit_FP32',
        'basis':'Author requests diagnosis and implementation correction after failure.'},out)
    shutil.copyfile(Path(__file__),out/'executed_repair_source.py')
    shutil.copyfile(Path(canary.__file__),out/'executed_canary_source.py')
    lock=root/'.rq4_real_canary_gpu.lock'
    handle=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(handle,str(os.getpid()).encode());os.close(handle)
    began=time.perf_counter()
    try:
        remaining=max(0,(deadline-datetime.now(timezone.utc)).total_seconds())
        with (out/'worker.log').open('x',encoding='utf-8') as log:
            code=bounded_run([sys.executable,'-B','-X','utf8',str(Path(canary.__file__)),'--worker',out.name],log,remaining,
                dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',OMP_NUM_THREADS='2',PYTHONIOENCODING='utf-8'))
        if code:raise RuntimeError('Repair worker failed.')
    except BaseException:
        save(out/'supervisor_failure.json',{'traceback':traceback.format_exc(),'retry':False},out)
    finally:
        if lock.exists() and lock.read_text()==str(os.getpid()):lock.unlink()
    result=load(out/'worker_result.json') if (out/'worker_result.json').exists() else {'status':'CANARY_INTERRUPTED'}
    save(out/'supervisor.json',{'actual_seconds':time.perf_counter()-began,'original_deadline_UTC':deadline.isoformat(),
        'deadline_preserved':True,'budget_reset':False},out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'RQ4_EXPLICIT_FP32_REPAIR',**result},out,replace=True)
    write_atomic(out/'diary.md','# FP32 engineering repair\n\nSame600-second allocation and immutable failed AMP evidence. Explicit precision change; no optimizer change, no full training or scientific matched pair.\n',out)
    seal(out)
    print(out,flush=True)


if __name__=='__main__':
    repair(sys.argv[1])
