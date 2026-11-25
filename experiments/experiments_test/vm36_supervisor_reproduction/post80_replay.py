"""Recompute the accepted80 outputs on CPU without repeating inference."""
import argparse
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from protocol import ROOT,HERE,EXPERIMENT,load,save,source,sha256,write_atomic
from common import seal,verify_hashes
from historical_review import csv_file
from main.pipeline import new_run

ANALYSIS='vm36_supervisor_reproduction_full20_results_20261110T161658_46e4aed4'
ACCEPTANCE='vm36_supervisor_reproduction_full20_acceptance_20261110T161705_6e2cc033'
REQUIRED=('configuration.json','metrics.json','predictions.jsonl.gz','ground_truth.json',
          'bootstrap.json','image_manifest.jsonl','verification.json','timing.json')


def run_path(identity):
    if Path(identity).name!=identity:raise ValueError('Use a plain RunID.')
    return ROOT/'datasets/local_runs'/EXPERIMENT/identity


def prepare(request):
    original=run_path(ANALYSIS);admit=run_path(ACCEPTANCE)
    verify_hashes(original,('EVIDENCE_SOURCES.json','MEASURED_DESIGN.json','summary.json'))
    verify_hashes(admit,('ACCEPTANCE.json',))
    if not load(admit/'ACCEPTANCE.json')['all80_measured']:raise ValueError('The80-cell study is not accepted.')
    records=[]
    for row in load(original/'EVIDENCE_SOURCES.json')['conditions']:
        folder=Path(row['folder']);timing=load(folder/'timing.json')
        records.append({'pair_id':row['pair_id'],'folder':str(folder),'manifest':source(folder/'SHA256SUMS.txt'),
            'expected_metrics':source(folder/'metrics.json'),'historical_replay_seconds':timing['replay_seconds']})
    if len(records)!=80 or len({r['pair_id'] for r in records})!=80:raise ValueError('Expected80 unique inputs.')
    slow=max(records,key=lambda r:r['historical_replay_seconds'])
    fast=min(records,key=lambda r:r['historical_replay_seconds'])
    ordered=[slow,fast]+[r for r in records if r not in (slow,fast)]
    reference=sum(r['historical_replay_seconds'] for r in records)
    out,_=new_run({'output_root':str(original.parent)},EXPERIMENT,'post80_cpu_replay')
    shutil.copyfile(Path(__file__),out/'executed_source.py')
    plan={'version':'VM36_POST80_CPU_REPLAY_V1','source_analysis':source(original/'SHA256SUMS.txt'),
        'source_acceptance':source(admit/'SHA256SUMS.txt'),'request':source(request),'pairs':ordered,
        'historical_sum_replay_seconds':reference,'CPU_worker_wall_limit_seconds':math.ceil(reference*1.5+300),
        'order':'Highest and lowest historical CPU cost as preflight, then the original source order; no AP selection.',
        'GPU_allowed':False,'max_parallel_evaluators':1,'evaluation':'Unchanged evaluation.evaluate_saved: native10, ignore union, original coordinates, maxDet3000.',
        'source_files':{str(p.relative_to(ROOT)):source(p) for p in (HERE.parent/'cv261207/evaluation.py',
            ROOT/'main/code/main/labels.py',ROOT/'main/code/main/statistics.py')}}
    save(out/'REPLAY_PROTOCOL.json',plan,out)
    save(out/'inputs.json',{'request':source(request),'source':source(Path(__file__)),
        'accepted_results':source(original/'SHA256SUMS.txt')},out)
    return out


def worker(out):
    from evaluation import evaluate_saved
    began=time.perf_counter();plan=load(out/'REPLAY_PROTOCOL.json');completed=[];active=None;status='COMPLETE'
    for record in plan['source_files'].values():
        if sha256(Path(record['path']))!=record['sha256']:raise ValueError('Evaluator source changed.')
    try:
        for i,row in enumerate(plan['pairs']):
            remaining=plan['CPU_worker_wall_limit_seconds']-(time.perf_counter()-began)
            if completed and remaining<max(r['seconds'] for r in completed)*1.25+30:
                status='PARTIAL_BUDGET';break
            active=row['pair_id'];folder=Path(row['folder']);t=time.perf_counter()
            if sha256(folder/'SHA256SUMS.txt')!=row['manifest']['sha256']:raise ValueError('Input manifest changed.')
            checks=verify_hashes(folder,REQUIRED)
            measured,evaluator=evaluate_saved(folder);expected=load(folder/'metrics.json')
            if measured!=expected:
                save(out/('MISMATCH_'+active+'.json'),{'expected':expected,'recomputed':measured},out)
                raise ValueError('Exact metric replay differs: '+active)
            if measured['images']!=548:raise ValueError('Unexpected image count.')
            result={'pair_id':active,'seconds':time.perf_counter()-t,'exact_equality':True,'images':548,
                'raw_count':measured['raw_count'],'mapped_count':measured['mapped_count'],
                'prediction_SHA256':checks['predictions.jsonl.gz'],'GT_SHA256':checks['ground_truth.json'],
                'original_metrics_SHA256':checks['metrics.json']}
            save(out/'results'/(active+'.json'),{'source_folder':str(folder),'verification':result,'metrics':measured},out)
            completed.append(result)
            save(out/'progress.json',{'completed':completed,'elapsed_seconds':time.perf_counter()-began},out,replace=(out/'progress.json').exists())
            print('CPU_REPLAY',len(completed),'/80',active,round(result['seconds'],3),flush=True)
            del evaluator;gc.collect()
    except BaseException:
        status='PARTIAL_ERROR';save(out/'failure.json',{'active_pair':active,'traceback':traceback.format_exc(),'retry':False},out)
    csv_file(out/'REPLAY_VERIFICATION.csv',completed)
    save(out/'worker_result.json',{'status':status,'replayed_pairs':len(completed),'expected_pairs':80,
        'CPU_worker_seconds':time.perf_counter()-began,'GPU_experiments':0,'automatic_retry':False},out)


def execute(request):
    from development_guard import bounded_run
    out=prepare(request);plan=load(out/'REPLAY_PROTOCOL.json');began=time.perf_counter()
    try:
        with (out/'worker.log').open('x',encoding='utf-8') as stream:
            code=bounded_run([sys.executable,'-B',str(Path(__file__)),'--worker',out.name],stream,
                plan['CPU_worker_wall_limit_seconds'],dict(os.environ,CUDA_VISIBLE_DEVICES='',
                    PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',YOLO_AUTOINSTALL='false'))
        if code:raise RuntimeError('CPU worker failed; no retry.')
    except BaseException:
        save(out/'supervisor_failure.json',{'traceback':traceback.format_exc(),'retry':False},out)
    result=load(out/'worker_result.json') if (out/'worker_result.json').exists() else {'status':'PARTIAL_INTERRUPTED','replayed_pairs':0}
    save(out/'supervisor.json',{'hard_budget_seconds':plan['CPU_worker_wall_limit_seconds'],
        'actual_seconds':time.perf_counter()-began,'GPU_allowed':False},out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'POST80_CPU_REPLAY',**result},out,replace=True)
    write_atomic(out/'diary.md','# Post80 CPU replay\n\nRecomputed saved548-image predictions and GT with the unchanged evaluator. Exact equality covers area metrics, per-class AP/AR and prediction counts. No inference or training.\n',out)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>Post80 CPU replay</h1><p>'+result['status']+'</p><a href="REPLAY_VERIFICATION.csv">Exact metric replay</a>',out)
    seal(out);print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--request',type=Path);g.add_argument('--worker');args=parser.parse_args()
    worker(run_path(args.worker)) if args.worker else execute(args.request)
