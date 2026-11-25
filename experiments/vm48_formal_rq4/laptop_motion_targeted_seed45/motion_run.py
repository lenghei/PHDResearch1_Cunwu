"""One supervised seed45 attempt with a separate author allocation."""
import argparse,ctypes,csv,json,os,secrets,shutil,subprocess,sys,time,traceback
from datetime import timedelta
from motion_common import *

TICKET=RUNS/'MOTION_SEED45_EXECUTION_TICKET.json'
GPU_UUID='GPU-1dcc4d8e-3346-93c1-53db-f899d1f9efaf'

def ac_online():
    class Power(ctypes.Structure):
        _fields_=[('ACLineStatus',ctypes.c_ubyte),('BatteryFlag',ctypes.c_ubyte),('BatteryLifePercent',ctypes.c_ubyte),('SystemStatusFlag',ctypes.c_ubyte),('BatteryLifeTime',ctypes.c_ulong),('BatteryFullLifeTime',ctypes.c_ulong)]
    p=Power()
    return bool(ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(p))) and p.ACLineStatus==1

def query():
    text=subprocess.run(['nvidia-smi','--query-gpu=uuid,temperature.gpu,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=1.4).stdout.strip().splitlines()
    if len(text)!=1:raise RuntimeError('Unexpected GPU inventory.')
    uuid,t,u,f,v=[x.strip() for x in text[0].split(',')]
    if uuid!=GPU_UUID:raise RuntimeError('Wrong GPU.')
    return {'UTC':datetime.now(timezone.utc).isoformat(),'temperature_C':int(t),'used_MiB':int(u),'free_MiB':int(f),'utilization_percent':int(v)}

def admission(out):
    import psutil
    check(out);q=query()
    if q['temperature_C']>65 or q['free_MiB']<6000:raise RuntimeError('Initial temperature/VRAM admission failed.')
    if shutil.disk_usage(out).free<35*1024**3 or psutil.virtual_memory().available<8*1024**3:raise RuntimeError('Disk/RAM admission failed.')
    if not ac_online():raise RuntimeError('AC power is required.')
    physical=ROOT/'datasets/local_runs/vm37_yolo26_rq4/vm37_yolo26_rq4_pilot960_authorization_record_cpu_20261114T110935_8e5d6543/physical_and_execution_authorization.json'
    r=load(physical)
    assert datetime.now(timezone.utc)<datetime.fromisoformat(r['expiry_UTC'])
    assert all(r[k] is True for k in ['hard_ventilated_surface','unobstructed_intake_exhaust','fans_normal','reliable_AC_adapter'])
    apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=2).stdout
    if any('python' in x.lower() for x in apps.splitlines()):raise RuntimeError('Competing GPU Python process.')
    identity=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,driver_version,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=2).stdout
    return {'actual_GPU':identity,'sample':q,'AC_online':True,'free_disk_bytes':shutil.disk_usage(out).free,'available_RAM_bytes':psutil.virtual_memory().available,
        'physical_declaration':source(physical),'prior_declaration_still_valid':True,'new_physical_confirmation':False,'unknown':['room temperature','fan RPM','OEM mode'],
        'safety':{'immediate_C':90,'sustained_C':87,'sustained_seconds':30,'maximum_sample_interval_seconds':2,'disk_reserve_GiB':25},'GPU_forward_calls':0}

def supervise(out,stage,cap,token):
    import psutil
    safety=module('seed45_accepted_process_safety',ROOT/'main/experiments/experiments_test/vm37_yolo26_rq4/post_cost/safety.py')
    folder=out/(stage+'_supervision');folder.mkdir();started=time.perf_counter();reason=None;process=None;samples=[];hot_since=None;code=None
    deadline=min(datetime.now(timezone.utc)+timedelta(seconds=cap),datetime.fromisoformat(load(out/'budget_ledger.json')['expiry_UTC']))
    protocol=out/('training_protocol.json' if stage=='training' else 'evaluation_protocol.json')
    save(out/(stage+'_lease.json'),{'deadline_UTC':deadline.isoformat(),'cap_seconds':cap,'token_SHA256':sha_text(token),'protocol_SHA256':sha(protocol),'attempt':1})
    command=[sys.executable,'-B','-X','utf8',str(HERE/'motion_run.py'),'--worker',stage,'--run',str(out),'--cap',str(cap)]
    (folder/'command.txt').write_text(subprocess.list2cmdline(command)+'\n',encoding='utf-8')
    env=dict(os.environ,VM48_SEED45_TOKEN=token,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    try:
        with (folder/'worker.log').open('x',encoding='utf-8') as log,(folder/'gpu_telemetry.jsonl').open('x',encoding='utf-8') as trace:
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,cwd=str(ROOT))
            save(folder/'process.json',{'PID':process.pid,'supervisor_PID':os.getpid(),'started_UTC':datetime.now(timezone.utc).isoformat(),'cap_seconds':cap})
            while process.poll() is None:
                tick=time.perf_counter();elapsed=tick-started
                if elapsed>=cap-20 or datetime.now(timezone.utc)>=deadline-timedelta(seconds=20):reason='HARD_BUDGET_OR_EXPIRY_STOP'
                else:
                    try:
                        q=query();q['monotonic_seconds']=time.perf_counter()-started
                        if samples and q['monotonic_seconds']-samples[-1]['monotonic_seconds']>2:reason='TELEMETRY_INTERVAL_EXCEEDED'
                        elif not samples and q['monotonic_seconds']>2:reason='INITIAL_TELEMETRY_DELAY'
                        if q['temperature_C']>=90:reason='GPU_TEMPERATURE_AT_LEAST_90C'
                        if q['temperature_C']>=87:
                            if hot_since is None:hot_since=q['monotonic_seconds']
                            if q['monotonic_seconds']-hot_since>=30:reason='GPU_TEMPERATURE_87C_FOR_30S'
                        else:hot_since=None
                        q['disk_free_bytes']=shutil.disk_usage(out).free;q['RAM_available_bytes']=psutil.virtual_memory().available;q['AC_online']=ac_online()
                        if q['disk_free_bytes']<25*1024**3:reason='DISK_RESERVE_BELOW_25GiB'
                        if not q['AC_online']:reason='AC_POWER_LOST'
                        samples.append(q);trace.write(json.dumps(q)+'\n');trace.flush()
                    except BaseException as e:reason='TELEMETRY_FAILURE: '+str(e)
                if reason:safety.stop_owned(process);break
                time.sleep(max(0,.75-(time.perf_counter()-tick)))
            code=process.wait(timeout=15)
    except BaseException:
        reason=reason or 'SUPERVISOR_EXCEPTION';
        if process is not None and process.poll() is None:safety.stop_owned(process)
        raise
    finally:
        elapsed=time.perf_counter()-started
        result={'returncode':code,'stop_reason':reason,'wall_seconds':elapsed,'worker_PID':process.pid if process else None,'attempt':1,'retry':False,
            'peak_temperature_C':max((r['temperature_C'] for r in samples),default=None),'peak_used_MiB':max((r['used_MiB'] for r in samples),default=None),
            'maximum_observed_sample_interval_seconds':max((b['monotonic_seconds']-a['monotonic_seconds'] for a,b in zip(samples,samples[1:])),default=None)}
        save(folder/'supervisor.json',result);charge(out,stage,elapsed,result)
    return result

def sha_text(t):
    import hashlib
    return hashlib.sha256(t.encode()).hexdigest()

def execute(out):
    check(out);assert (out/'source_provenance.json').exists()
    if (out/'execution_started.json').exists():raise PermissionError('This attempt already started. No automatic retry.')
    locks=ROOT/'datasets/local_runs/vm37_yolo26_rq4';lock=locks/'.pilot10_gpu.lock';owned=False;failure=None
    try:
        if remaining(out)<36000-5:raise PermissionError('The new complete allocation is not available.')
        save(out/'initial_device_admission.json',admission(out))
        if lock.exists() or (locks/'.gpu.lock').exists():raise RuntimeError('A project GPU owner lock exists.')
        with lock.open('x',encoding='utf-8') as f:json.dump({'RunID':out.name,'PID':os.getpid(),'new_budget':True},f)
        owned=True
        save(out/'execution_started.json',{'UTC':datetime.now(timezone.utc).isoformat(),'attempt':1,'budget_seconds':36000,'old_ledger_unchanged':True})
        result=supervise(out,'training',28000,secrets.token_hex(24))
        if result['stop_reason']:raise RuntimeError('Training supervisor stop: '+result['stop_reason'])
        if result['returncode']!=0:raise RuntimeError('Training worker failed; inspect training_failure.json and worker.log.')
        status=load(out/'training_result.json')['status']
        if status!='MATCHED_10EPOCH_TRAINING_COMPLETED':raise RuntimeError(status)
        if remaining(out)<8000:raise RuntimeError('Complete eight-cell evaluation plus reserve does not fit.')
        result=supervise(out,'evaluation',6000,secrets.token_hex(24))
        if result['stop_reason']:raise RuntimeError('Evaluation supervisor stop: '+result['stop_reason'])
        if result['returncode']!=0:raise RuntimeError('Evaluation worker failed; inspect evaluation_failure.json and worker.log.')
        assert load(out/'analysis_acceptance.json')['cells']==8
    except BaseException:
        failure=traceback.format_exc();save(out/'execution_failure.json',{'traceback':failure,'retry':False});print(failure,flush=True)
    finally:
        if owned and lock.exists() and load(lock).get('RunID')==out.name:lock.unlink()
        from motion_reports import finish
        finish(out,failure)
    print('FINAL_RESULT',str(out),flush=True)
    return 1 if failure else 0

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepare',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--worker',choices=['training','evaluation']);p.add_argument('--run');p.add_argument('--cap',type=float,default=28000);a=p.parse_args()
    ticket=load(TICKET);out=Path(a.run) if a.run else Path(ticket['path']);assert out.resolve()==Path(ticket['path']).resolve()
    if a.worker:
        lease=load(out/(a.worker+'_lease.json'));token=os.environ.get('VM48_SEED45_TOKEN','');assert token and sha_text(token)==lease['token_SHA256']
        assert datetime.now(timezone.utc)<datetime.fromisoformat(lease['deadline_UTC']) and a.cap==lease['cap_seconds'];check(out)
        protocol=out/('training_protocol.json' if a.worker=='training' else 'evaluation_protocol.json');assert sha(protocol)==lease['protocol_SHA256']
        try:
            if a.worker=='training':
                from motion_training import worker
                worker(out,a.cap)
            else:
                from motion_evaluation import worker
                worker(out)
        except BaseException:
            save(out/(a.worker+'_failure.json'),{'traceback':traceback.format_exc(),'retry':False});raise
    elif a.prepare:
        from motion_prepare import prepare
        prepare(out);print('CPU_PREPARATION_COMPLETE',out,flush=True)
    elif a.execute:sys.exit(execute(out))
    else:p.error('Use --prepare or --execute. GPU execution is never implicit.')

if __name__=='__main__':main()
