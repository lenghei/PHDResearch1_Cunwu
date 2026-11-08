"""Enforce the active continuation budget on its Windows worker tree."""
import argparse
from datetime import datetime,timezone
import json
import subprocess
import time
from common import load,save,new_run,resolve_run,source,seal


def stop_tree(pid):
    return subprocess.run(['taskkill','/PID',str(pid),'/T','/F'],capture_output=True,text=True,errors='replace').returncode


def bounded_run(command,stream,timeout,env):
    child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,env=env)
    try:
        code=child.wait(timeout=timeout)
        return code
    except BaseException:
        stop_tree(child.pid)
        child.wait(timeout=15)
        raise


def guard(continuation):
    ticket=load(continuation/'author_execution_ticket.json');study=ticket['original_study']
    started=datetime.fromisoformat(ticket['started_utc']);out=new_run('budget_guard')
    save(out/'inputs.json',{'continuation':continuation.name,'ticket':source(continuation/'author_execution_ticket.json'),
          'source':source(__import__('pathlib').Path(__file__)),'purpose':'Guard venv launcher and base-Python descendants; no new GPU work.'})
    command="Get-CimInstance Win32_Process | Where-Object { $_.Name -match '^python' -and $_.CommandLine -like '*development_v1.py*worker*--run*"+study+"*' } | Select-Object ProcessId,@{Name='created_utc';Expression={$_.CreationDate.ToUniversalTime().ToString('o')}} | ConvertTo-Json -Compress"
    polls=0;status='PASS_NO_TIMEOUT';stops=[]
    while True:
        now=datetime.now(timezone.utc);elapsed=(now-started).total_seconds()
        query=subprocess.run(['powershell','-NoProfile','-Command',command],capture_output=True,text=True,check=True,timeout=10)
        found=json.loads(query.stdout) if query.stdout.strip() else []
        if isinstance(found,dict):found=[found]
        polls+=1
        failure=(continuation/'failure.json').exists()
        for worker in found:
            age=(now-datetime.fromisoformat(worker['created_utc'])).total_seconds()
            if failure or elapsed>=ticket['budget_seconds']-10 or age>=1790:
                stops.append({'pid':worker['ProcessId'],'worker_age_seconds':age,'continuation_elapsed_seconds':elapsed,
                      'exit_code':stop_tree(worker['ProcessId']),'reason':'PARENT_FAILED' if failure else 'BUDGET_GUARD'})
                status='STOPPED_OWNED_WORKER'
        if stops or not found and ((continuation/'result.json').exists() or failure):break
        time.sleep(min(5,max(.1,ticket['budget_seconds']-10-elapsed)))
    save(out/'result.json',{'status':status,'polls':polls,'stops':stops,'continuation':continuation.name,
             'GPU_experiments_started':0,'scope':'ENGINEERING_BUDGET_GUARD'})
    save(out/'run.json',{'RunID':out.name,'ExperimentID':'cv261207','mode':'BUDGET_GUARD','status':status});seal(out)
    print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--continuation',required=True);args=parser.parse_args()
    guard(resolve_run(args.continuation))
