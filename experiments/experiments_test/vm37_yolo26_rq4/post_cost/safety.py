"""Finite-budget admission and fail-closed monitoring of an owned worker."""
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time


def available_seconds(ledger,now=None):
    now=now or datetime.now(timezone.utc)
    consumed=sum(row['charged_seconds'] for row in ledger['completed_workers'])
    return max(0.,min(ledger['total_seconds']-consumed,(datetime.fromisoformat(ledger['absolute_deadline_UTC'])-now).total_seconds()))


def decide(policy,samples,telemetry_ok=True):
    if not telemetry_ok:return 'TELEMETRY_FAILURE'
    if not samples:return None
    latest=samples[-1]
    if latest['temperature_C']>=policy['immediate_stop_temperature_C']:return 'TEMPERATURE_HARD_STOP'
    threshold=policy['sustained_stop_temperature_C'];start=len(samples)-1
    while start>=0 and samples[start]['temperature_C']>=threshold:start-=1
    if start<len(samples)-2 and latest['monotonic_seconds']-samples[start+1]['monotonic_seconds']>=policy['sustained_seconds']:
        return 'SUSTAINED_TEMPERATURE_STOP'
    return None


def query():
    command=['nvidia-smi','--query-gpu=temperature.gpu,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']
    result=subprocess.run(command,capture_output=True,text=True,timeout=10,check=True)
    rows=result.stdout.strip().splitlines()
    if len(rows)!=1:raise RuntimeError('This protocol requires one visible NVIDIA device.')
    temperature,used,free,utilization=[int(v.strip()) for v in rows[0].split(',')]
    return {'UTC':datetime.now(timezone.utc).isoformat(),'temperature_C':temperature,'used_MiB':used,'free_MiB':free,'utilization_percent':utilization}


def stop_owned(process):
    import psutil
    try:
        parent=psutil.Process(process.pid);children=parent.children(recursive=True)
        for child in reversed(children):
            try:child.kill()
            except psutil.NoSuchProcess:pass
        try:parent.kill()
        except psutil.NoSuchProcess:pass
    except psutil.NoSuchProcess:pass
    process.wait(timeout=15)


def run_owned(command,out,cap,policy,env):
    initial=query()
    if initial['temperature_C']>policy['preflight_maximum_temperature_C'] or initial['free_MiB']<policy['preflight_minimum_free_MiB']:
        raise RuntimeError('GPU temperature or memory preflight failed.')
    start=time.perf_counter();samples=[];stop_reason=None
    with (out/'worker.log').open('x',encoding='utf-8') as log,(out/'gpu_telemetry.jsonl').open('x',encoding='utf-8') as telemetry:
        process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        try:
            while process.poll() is None:
                elapsed=time.perf_counter()-start
                if elapsed>=max(0,cap-15):stop_reason='WALL_CLOCK_HARD_STOP'
                else:
                    try:
                        sample=query();sample['monotonic_seconds']=elapsed;samples.append(sample)
                        telemetry.write(json.dumps(sample)+'\n');telemetry.flush();stop_reason=decide(policy,samples)
                    except Exception:stop_reason='TELEMETRY_FAILURE'
                if stop_reason:stop_owned(process);break
                time.sleep(min(policy['poll_seconds'],max(.1,cap-15-elapsed)))
            code=process.wait(timeout=15)
        except BaseException:
            stop_owned(process);raise
    result={'returncode':code,'stop_reason':stop_reason,'wall_seconds':time.perf_counter()-start,
            'worker_PID':process.pid,'automatic_retry':False}
    (out/'safety_supervisor.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result
