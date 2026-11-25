"""VM48 default-deny training and target full-epoch benchmark launcher."""
import argparse
from datetime import datetime,timezone,timedelta
import gc,json,os,secrets,subprocess,sys,time,traceback
from pathlib import Path
from unittest.mock import patch
from common48 import *


def query_gpu():
    r=subprocess.run(['nvidia-smi','--query-gpu=uuid,temperature.gpu,memory.free,memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=1.4)
    rows=r.stdout.strip().splitlines()
    if len(rows)!=1:raise RuntimeError('Isolate exactly one NVIDIA GPU.')
    uid,t,free,used,util=[v.strip() for v in rows[0].split(',')]
    return {'UTC':datetime.now(timezone.utc).isoformat(),'uuid':uid,'temperature_C':int(t),'free_MiB':int(free),'used_MiB':int(used),'utilization_percent':int(util)}


def supervised(command,out,cap,env,policy,expected_uuid):
    from safety import decide,stop_owned
    start=time.perf_counter();initial=query_gpu()
    if initial['uuid']!=expected_uuid or initial['temperature_C']>policy['preflight_maximum_temperature_C'] or initial['free_MiB']<policy['preflight_minimum_free_MiB']:
        raise RuntimeError('Current GPU identity, temperature or memory admission failed.')
    samples=[];reason=None
    with (out/'worker.log').open('x',encoding='utf-8') as log,(out/'gpu_telemetry.jsonl').open('x',encoding='utf-8') as trace:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        try:
            while process.poll() is None:
                tick=time.perf_counter()
                if tick-start>=cap-15:reason='WALL_CLOCK_HARD_STOP'
                else:
                    try:
                        row=query_gpu();row['monotonic_seconds']=time.perf_counter()-start
                        if row['uuid']!=expected_uuid:reason='GPU_IDENTITY_CHANGED'
                        if samples and row['monotonic_seconds']-samples[-1]['monotonic_seconds']>2:reason='TELEMETRY_INTERVAL_EXCEEDED'
                        samples.append(row);trace.write(json.dumps(row)+'\n');trace.flush()
                        reason=reason or decide(policy,samples)
                    except BaseException:reason='TELEMETRY_FAILURE'
                if reason:stop_owned(process);break
                time.sleep(max(0.,1-(time.perf_counter()-tick)))
            code=process.wait(timeout=15)
        except BaseException:stop_owned(process);raise
    result={'returncode':code,'stop_reason':reason,'worker_wall_seconds':time.perf_counter()-start,'max_temperature_C':max((r['temperature_C'] for r in samples),default=None),'retry':False}
    save(out/'supervisor.json',result)
    return result


def ledger_path(package,auth):
    identity=auth.get('allocation_id','')
    if not identity or Path(identity).name!=identity or any(x in identity.upper() for x in ('VM37','REPLACE','EXAMPLE','SYNTHETIC')):raise PermissionError('A new author-issued allocation ID is required.')
    return package/'datasets/local_runs/vm48_allocations'/identity/'ledger.json'


def read_ledger(package,auth):
    path=ledger_path(package,auth)
    if not path.exists():return {'total_seconds':auth['total_seconds'],'expires_UTC':auth['expires_UTC'],'charges':[]}
    value=load(path)
    if (value['total_seconds'],value['expires_UTC'])!=(auth['total_seconds'],auth['expires_UTC']):raise PermissionError('Allocation total or expiry cannot be reset.')
    return value


def charge(package,auth,run_id,stage,seconds,result):
    path=ledger_path(package,auth);value=read_ledger(package,auth)
    value['charges'].append({'RunID':run_id,'stage':stage,'charged_seconds':seconds,'result':result,'UTC':datetime.now(timezone.utc).isoformat()})
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.partial')
    with temporary.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    temporary.replace(path)


def own_ticket(out,package):
    if not out.is_relative_to(package/'datasets/local_runs'):raise PermissionError('Worker output is outside the package-local protected scope.')
    parent=out.parent;ticket=load(parent/'execution_ticket.json');token=os.environ.get('VM48_WORKER_TOKEN','')
    if not token or hashlib.sha256(token.encode()).hexdigest()!=ticket['worker_token_SHA256']:raise PermissionError('Owned VM48 supervisor token required.')
    if ticket['package_manifest_SHA256']!=sha(package/'package_manifest.json'):raise PermissionError('Package changed.')
    if ticket['authorization_SHA256']!=sha(parent/'authorization.json'):raise PermissionError('Authorization changed.')
    if ticket['plan_SHA256'][out.name]!=sha(out/'protocol.json'):raise PermissionError('Arm plan changed.')
    lease=load(out/'WORKER_LEASE.json')
    if datetime.now(timezone.utc)>=datetime.fromisoformat(lease['deadline_UTC']):raise PermissionError('Worker lease expired.')
    auth=load(parent/'authorization.json');receipt=load(parent/'CPU_receipt.json')
    validate_authorization(package,ticket['mode'],ticket['rows'],auth,receipt)
    if remaining(auth,read_ledger(package,auth)['charges'])<=0:raise PermissionError('Allocation expired or exhausted.')
    return load(out/'protocol.json'),package


def training_worker(out,package):
    verify_package(package);plan,_=own_ticket(out,package)
    from trainer48 import trainer_factory
    # Keep the accepted training implementation and replace only its admission adapter.
    with patch.object(v3,'check_worker',lambda candidate:own_ticket(candidate,package)),patch.object(v3,'trainer_factory',trainer_factory):
        status=v3.training_worker(out)
    if status!='TRAINING_COMPLETED':return status
    receipt=load(out/'training_acceptance.json');receipt.update(arm_id=plan['arm_id'],strategy=plan['arm'],executor_version=VERSION,
        initialization_SHA256=plan['initialization']['sha256'],expected_pending_microbatches=plan['expected']['pending_microbatches'],
        snapshot_selection='FIXED_EPOCH100_EMA' if plan['execution_epochs']==100 else 'EPOCH1_ENGINEERING_ONLY')
    temporary=out/'training_acceptance.extended.partial';temporary.write_text(json.dumps(receipt,indent=2),encoding='utf-8');temporary.replace(out/'training_acceptance.json')
    return status


def make_plans(package,data_root,mode,architecture,out,verify_data=True):
    population=base.resolve_data(package,data_root,verify=verify_data);rows=matrix(package,mode,architecture)
    for row in rows:
        folder=out/row['arm_id'];folder.mkdir()
        plan=plan_arm(package,population,row,folder,mode)
        plan.update(package_manifest_SHA256=sha(package/'package_manifest.json'),source_commit=load(package/'package_manifest.json')['source_commit'])
        save(folder/'protocol.json',plan)
    save(out/'plans.json',{'version':VERSION,'mode':mode,'rows':rows,'data_root':str(data_root),'package':str(package),'GPU':False,'scientific_results':'NOT_EVALUATED'})
    return rows


def execute(package,data_root,mode,architecture,output,authorization,receipt,do_execute):
    verify_package(package);base.environment_check(package)
    rows=matrix(package,mode,architecture)
    auth=load(authorization) if authorization else None
    cpu=load(receipt) if receipt else None
    if do_execute:
        if auth is None or cpu is None:raise PermissionError('Explicit arm-specific authorization and target CPU receipt required.')
        validate_authorization(package,mode,rows,auth,cpu)
        if remaining(auth,read_ledger(package,auth)['charges'])<auth['resource_admission']['required_complete_group_seconds']:
            raise PermissionError('Remaining allocation cannot cover the complete requested group.')
    output=output.resolve()
    if not output.is_relative_to(package/'datasets/local_runs'):raise PermissionError('Use package-local datasets/local_runs; arbitrary external outputs are not admitted.')
    output.mkdir(parents=True,exist_ok=False)
    base.runtime(output,gpu=False)
    rows=make_plans(package,data_root,mode,architecture,output)
    if not do_execute:
        save(output/'dry_run.json',{'status':'CPU_DRY_RUN_ONLY','planned_arms':len(rows),'training_arms_run':0,'GPU_seconds':0})
        print('Prepared',len(rows),'arms. No GPU execution.',output,flush=True);return
    with (output/'authorization.json').open('xb') as stream:stream.write(Path(authorization).read_bytes())
    save(output/'CPU_receipt.json',cpu)
    if auth['resource_admission']['required_complete_group_seconds']>remaining(auth,read_ledger(package,auth)['charges']):
        raise PermissionError('Preparation consumed the reserve for the complete requested group.')
    tokens=package/'datasets/local_runs/vm48_authorization_tokens';tokens.mkdir(parents=True,exist_ok=True)
    token_file=tokens/(auth['authorization_id']+'.json')
    save(token_file,{'RunID':output.name,'output':str(output),'scope':'Single attempt; no automatic retry or resume'})
    lock=package/'datasets/local_runs/.vm48_gpu.lock'
    with lock.open('x',encoding='utf-8') as f:json.dump({'RunID':output.name,'PID':os.getpid()},f)
    token=secrets.token_hex(24)
    save(output/'execution_ticket.json',{'RunID':output.name,'mode':mode,'rows':rows,'package_manifest_SHA256':sha(package/'package_manifest.json'),
       'authorization_SHA256':sha(output/'authorization.json'),'worker_token_SHA256':hashlib.sha256(token.encode()).hexdigest(),
       'plan_SHA256':{r['arm_id']:sha(output/r['arm_id']/'protocol.json') for r in rows}})
    policy=load(package/'safety_policy.json');status='INCOMPLETE'
    try:
        if (policy['immediate_stop_temperature_C'],policy['sustained_stop_temperature_C'],policy['sustained_seconds'])!=(90,87,30):raise ValueError('Safety thresholds changed.')
        for row in rows:
            folder=output/row['arm_id'];cap=auth['arm_authorizations'][row['arm_id']]['hard_seconds']
            if cap+auth['resource_admission']['safety_reserve_seconds']>remaining(auth,read_ledger(package,auth)['charges']):raise PermissionError('Arm no longer fits remaining budget.')
            now=datetime.now(timezone.utc);deadline=min(now+timedelta(seconds=cap),datetime.fromisoformat(auth['expires_UTC']))
            save(folder/'WORKER_LEASE.json',{'deadline_UTC':deadline.isoformat(),'cap_seconds':cap,'attempt':1})
            command=[sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),'--package',str(package),'--worker',str(folder)]
            started=time.perf_counter();result=None
            try:result=supervised(command,folder,cap,dict(os.environ,VM48_WORKER_TOKEN=token,CUDA_VISIBLE_DEVICES='0'),policy,auth['GPU_uuid'])
            finally:charge(package,auth,output.name,row['arm_id'],time.perf_counter()-started,result)
            if result['stop_reason']:raise RuntimeError('Supervisor stopped arm: '+result['stop_reason'])
            if result['returncode'] or not (folder/'training_acceptance.json').exists():raise RuntimeError('Arm failed; see worker log. No retry.')
        status='BENCHMARK_GROUP_COMPLETED_ENGINEERING_ONLY' if mode=='benchmark' else 'TRAINING_GROUP_COMPLETED_RESEARCH_EVALUATION_NOT_RUN'
    except BaseException:save(output/'failure.json',{'traceback':traceback.format_exc(),'retry':False})
    finally:
        if lock.exists() and load(lock).get('RunID')==output.name:lock.unlink()
    save(output/'execution_result.json',{'status':status,'training_group':mode,'exact_resume_validated':False,'formal_RQ4_completed':False})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);p.add_argument('--data-root',type=Path)
    p.add_argument('--mode',choices=['formal','ablations','benchmark'],default='formal');p.add_argument('--architecture',choices=['YOLO26s','YOLOv8s'])
    p.add_argument('--output',type=Path);p.add_argument('--authorization',type=Path);p.add_argument('--cpu-receipt',type=Path)
    p.add_argument('--execute',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--worker',type=Path)
    a=p.parse_args();package=a.package.resolve(strict=True)
    if a.worker:
        training_worker(a.worker.resolve(strict=True),package);return
    if not a.data_root or not a.output:p.error('--data-root and --output are required.')
    if a.execute and a.dry_run:p.error('Choose dry-run or execute.')
    execute(package,a.data_root.resolve(strict=True),a.mode,a.architecture,a.output,a.authorization,a.cpu_receipt,a.execute)


if __name__=='__main__':main()
