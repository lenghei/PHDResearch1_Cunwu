"""Local paths, immutable inputs and the new seed45 allocation."""
import hashlib,importlib.util,json,os,sys,time
from datetime import datetime,timezone
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
RUNS=ROOT/'datasets/local_runs/vm48_formal_rq4'
sys.path[:0]=[str(ROOT/'main/code'),str(ROOT/'main/experiments/experiments_test/vm37_yolo26_rq4'),str(HERE.parent)]

def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

recovery=module('vm48_seed45_accepted_recovery',ROOT/'main/experiments/experiments_test/vm37_yolo26_rq4/local_recovery_v1.py')
v3=recovery.v3;base=v3.base
VM45=ROOT/'datasets/local_runs/vm45_motion_haze_gap_learning/vm45_motion_haze_gap_learning_execution_20261117T172830_0d9cb1ee'
TRANSFER=ROOT/'datasets/local_runs/vm48_formal_rq4/laptop_transfer_pilot_execution_20261118T100506_d3fa1688'
BUNDLE=recovery.baseline.bundle()
REQUEST=Path(r'C:\Users\w\.codex\attachments\3ea4f040-72b5-4684-a3e1-83e1e1671fee\已粘贴的文本.txt')
load=base.load;sha=base.digest

def save(path,value,replace=False):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    if not p.resolve().is_relative_to(RUNS.resolve()):raise PermissionError('Private VM48 outputs only.')
    if p.exists() and not replace:raise FileExistsError(p)
    t=p.with_name(p.name+'.partial')
    with t.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    t.replace(p)

def remaining(out):
    a=load(out/'author_authorization.json');b=load(out/'budget_ledger.json')
    if a['new_independent_budget_seconds']!=36000 or b['total_seconds']!=36000 or a['expiry_UTC']!=b['expiry_UTC']:raise PermissionError('Allocation changed.')
    return max(0,min(36000-sum(r['charged_seconds'] for r in b['charges']),(datetime.fromisoformat(b['expiry_UTC'])-datetime.now(timezone.utc)).total_seconds()))

def charge(out,stage,seconds,result):
    b=load(out/'budget_ledger.json')
    if any(r['stage']==stage for r in b['charges']):raise PermissionError('No stage retry.')
    b['charges'].append({'stage':stage,'charged_seconds':seconds,'result':result,'finished_UTC':datetime.now(timezone.utc).isoformat()})
    save(out/'budget_ledger.json',b,replace=True)

def check(out):
    if not out.resolve().is_relative_to(RUNS.resolve()) or not out.name.startswith('motion_targeted_seed45_execution_'):raise PermissionError('Wrong Run.')
    a=load(out/'author_authorization.json')
    if a['source_SHA256']!=sha(REQUEST) or sha(out/'author_request.txt')!=sha(REQUEST):raise PermissionError('Author source changed.')
    if a['scope']!='VM48_MOTION_TARGETED_SEED45' or a['arms']!=['CONTROL','MOTION_TARGETED'] or a['retry']:raise PermissionError('Wrong scope.')
    if not load(out/'weekly_reports.json')['completed_before_GPU']:raise PermissionError('Weekly reports must precede GPU.')
    if remaining(out)<=0:raise PermissionError('Allocation exhausted or expired.')
    if (out/'frozen_sources.json').exists():
        for r in load(out/'frozen_sources.json'):
            if sha(r['path'])!=r['SHA256']:raise ValueError('Frozen source changed: '+r['path'])

def source(p):
    p=Path(p).resolve(strict=True);return {'path':str(p),'SHA256':sha(p),'bytes':p.stat().st_size}
