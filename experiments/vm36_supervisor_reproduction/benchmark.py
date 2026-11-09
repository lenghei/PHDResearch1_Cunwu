"""Bounded CPU equivalence benchmark on existing prediction evidence."""
import argparse
import importlib.util
import inspect
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace
from protocol import ROOT,HERE,EXPERIMENT,load,save,source,sha256,write_atomic
from common import verify_hashes,seal
from main.pipeline import new_run

PAIR='cv261207_dev_pair_20261107T235417_9a537fcd'


def work(out):
    import numpy as np
    from evaluation import evaluate_saved
    from main.statistics import bootstrap_coco
    from pycocotools.cocoeval import COCOeval
    pair=ROOT/'datasets/local_runs/cv261207'/PAIR
    verify_hashes(pair,('metrics.json','ground_truth.json','predictions.jsonl.gz','bootstrap.json'))
    started=time.perf_counter();cpu=time.process_time()
    metrics,ev=evaluate_saved(pair)
    if metrics!=load(pair/'metrics.json'):raise ValueError('Replayed COCO metrics differ.')
    evaluation=time.perf_counter()-started;saved=load(pair/'bootstrap.json')
    count=saved['draws']['samples'];seed=saved['draws']['seed']
    if count!=200:raise ValueError('Expected the original 200-draw procedure.')
    base=time.perf_counter();cpu0=time.process_time()
    sys.modules['new_fast_coco_accumulate']=SimpleNamespace(accumulate_fast=lambda e:e.accumulate())
    reference_ci,reference_draws=bootstrap_coco(ev,samples=count,seed=seed)
    ref_wall=time.perf_counter()-base;ref_cpu=time.process_time()-cpu0
    fast_path=HERE.parent/'cv261207/development_fast_coco.py'
    spec=importlib.util.spec_from_file_location('vm36_cached_accumulator',fast_path)
    fast=importlib.util.module_from_spec(spec);spec.loader.exec_module(fast)
    sys.modules['new_fast_coco_accumulate']=fast
    base=time.perf_counter();cpu0=time.process_time()
    fast_ci,fast_draws=bootstrap_coco(ev,samples=count,seed=seed)
    fast_wall=time.perf_counter()-base;fast_cpu=time.process_time()-cpu0
    same=reference_ci==fast_ci==saved['intervals'] and reference_draws==fast_draws==saved['draws']
    if not same:raise ValueError('Accumulator uncertainty differs; the optimization is not accepted.')
    report={'status':'VALIDATED','scope':'ENGINEERING_ONLY_REPLAY_BENCHMARK','source_pair':PAIR,
        'images':metrics['images'],'new_inference_pairs':0,'GPU_seconds':0,'seed':seed,'draws':count,
        'original_COCO_replay_matches_saved':True,'full_reference_draws_equal_fast':True,
        'full_CIs_equal_saved':True,'reference_bootstrap_wall_seconds':ref_wall,'reference_bootstrap_CPU_seconds':ref_cpu,
        'fast_bootstrap_wall_seconds':fast_wall,'fast_bootstrap_CPU_seconds':fast_cpu,
        'accumulation_speedup':ref_wall/fast_wall,'COCO_replay_wall_seconds':evaluation,
        'worker_wall_seconds':time.perf_counter()-started,'worker_CPU_seconds':time.process_time()-cpu,
        'timing_scope':'Both bootstrap variants share already computed matches; COCO replay is separate. This is not a complete new 548/1610-image inference pair.',
        'fast_implementation':source(fast_path),'reference_implementation':source(Path(inspect.getfile(COCOeval))),
        'new_algorithm_claim':False,'safety':'The existing validated vectorized accumulator is reused; no installed package or historical source was changed.'}
    save(out/'efficiency_result.json',report,out)
    save(out/'reference_bootstrap.json',{'intervals':reference_ci,'draws':reference_draws},out)
    save(out/'fast_bootstrap.json',{'intervals':fast_ci,'draws':fast_draws},out)


def main(args):
    if args.worker:
        path=Path(args.worker).resolve()
        if not path.is_relative_to((ROOT/'datasets/local_runs'/EXPERIMENT).resolve()):raise ValueError('Worker output escapes the study.')
        work(path);return
    from development_guard import bounded_run
    parent=ROOT/'datasets/local_runs'/EXPERIMENT/args.protocol_run
    proposal=parent/'EXPERIMENT_PROTOCOL.json'
    if sha256(proposal)!=(parent/'PROTOCOL_SHA256.txt').read_text().split()[0]:raise ValueError('Protocol changed.')
    out,logger=new_run({'output_root':str(ROOT/'datasets/local_runs'/EXPERIMENT)},EXPERIMENT,'cpu_benchmark')
    save(out/'execution_ticket.json',{'status':'PLANNED','CPU_worker_limit_seconds':600,'GPU_seconds':0,
        'parent_protocol':source(proposal),'source_pair':PAIR,'draws':200,
        'selection':'Fixed first accepted pair; selected as an engineering regression input, without using its AP value.'},out)
    before={n:sha256(ROOT/'datasets/local_runs/cv261207'/PAIR/n) for n in ('metrics.json','ground_truth.json','predictions.jsonl.gz','bootstrap.json')}
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    command=[sys.executable,'-B',str(Path(__file__)),'--worker',str(out)]
    status='FAILED';start=time.perf_counter()
    try:
        with (out/'worker.log').open('x',encoding='utf-8') as stream:
            code=bounded_run(command,stream,600,env)
        if code:raise RuntimeError('CPU benchmark worker failed.')
        for n,h in before.items():
            if sha256(ROOT/'datasets/local_runs/cv261207'/PAIR/n)!=h:raise ValueError('Source evidence changed.')
        if load(out/'efficiency_result.json')['status']!='VALIDATED':raise ValueError('Benchmark incomplete.')
        status='VALIDATED'
    except BaseException:
        save(out/'failure.json',{'traceback':traceback.format_exc(),'no_automatic_retry':True},out)
        raise
    finally:
        save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'CPU_EVALUATION_BENCHMARK',
            'status':status,'wall_seconds':time.perf_counter()-start,'GPU_seconds':0,'new_inference_pairs':0},out,replace=True)
        seal(out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol-run');p.add_argument('--worker');a=p.parse_args()
    if not a.worker and (not a.protocol_run or Path(a.protocol_run).name!=a.protocol_run):p.error('Supply a plain --protocol-run RunID.')
    main(a)
