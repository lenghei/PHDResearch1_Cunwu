"""CPU acceptance of sealed full-validation evidence; never starts inference."""
import argparse
from datetime import datetime,timezone
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from protocol import ROOT,PUBLIC,HERE,EXPERIMENT,load,save,source,sha256,write_atomic
from common import verify_hashes,seal
from main.pipeline import new_run


def verify(study):
    started=time.perf_counter()
    out,_=new_run({'output_root':str(study.parent)},EXPERIMENT,'acceptance')
    required=('RESULT_SUMMARY.json','summary.json','result.json','CONFIGURATION_METRICS.csv',
        'INTERACTION_EFFECTS.csv','EVIDENCE_TREE.json','PRESERVED_INPUTS.json')
    checks=verify_hashes(study,required)
    for name,digest in load(study/'EVIDENCE_TREE.json').items():
        path=(study/name).resolve()
        if not path.is_relative_to(study.resolve()) or sha256(path)!=digest:
            raise ValueError('Nested evidence changed: '+name)
    report=load(study/'RESULT_SUMMARY.json');plan=load(study/'EXECUTION_PROTOCOL.json')
    summary=load(study/'summary.json');metadata=load(study/'result.json')
    import hashlib
    digest=hashlib.sha256(json.dumps(summary,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if digest!=metadata['summary_canonical_sha256']:raise ValueError('Dashboard summary identity differs.')
    accepted=[]
    for spec in plan['pairs']:
        folder=study/'pairs'/spec['id']
        if not (folder/'SHA256SUMS.txt').exists():continue
        verify_hashes(folder,('metrics.json','predictions.jsonl.gz','ground_truth.json','bootstrap.json','verification.json'))
        metric=load(folder/'metrics.json');bootstrap=load(folder/'bootstrap.json')
        gt=load(folder/'ground_truth.json')['coco'];expected=sorted(r['id'] for r in gt['images'])
        ids=[];count=0
        with gzip.open(folder/'predictions.jsonl.gz','rt',encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line);ids.append(row['image_id']);count+=len(row['predictions'])
                for p in row['predictions']:
                    if p['image_id']!=row['image_id'] or not 0<=p['class_id']<10:
                        raise ValueError('Prediction class or image differs.')
                    if not np.isfinite(p['bbox']+[p['score']]).all():raise ValueError('Nonfinite raw prediction.')
        if sorted(ids)!=expected or len(ids)!=548 or count!=metric['raw_count']:
            raise ValueError('Raw prediction membership/count differs.')
        if bootstrap['draws']['image_ids']!=expected:raise ValueError('Bootstrap population differs.')
        for interval in bootstrap['intervals']:
            values=[v for v in bootstrap['draws']['values'][interval['size']][interval['metric']] if v is not None]
            if len(values)!=interval['n_valid'] or interval['n_samples']!=200:raise ValueError('Bootstrap count differs.')
            if values and not np.allclose(np.quantile(values,[.025,.975]),interval['ci95'],rtol=0,atol=1e-14):
                raise ValueError('Saved percentile interval differs from draws.')
        if not load(folder/'verification.json')['COCO_recomputed_from_raw']:raise ValueError('Missing raw replay acceptance.')
        accepted.append({'pair':spec['id'],'images':len(ids),'raw_predictions':count,
            'source':source(folder/'SHA256SUMS.txt'),'raw_replay':'EXACT_REPLAY_DURING_WORKER',
            'bootstrap_draws':200,'CI_from_saved_draws':'VERIFIED'})
    if len(accepted)!=report['completed_pairs'] or report['image_inferences']!=548*len(accepted):
        raise ValueError('Completed count differs.')
    protected=load(study/'PRESERVED_INPUTS.json')['inputs']
    for name,item in protected.items():
        if sha256(Path(item['path']))!=item['sha256']:raise ValueError('Protected input changed: '+name)
    save(out/'INPUT_VERIFICATION.json',{'source_RunID':study.name,'root_manifest':source(study/'SHA256SUMS.txt'),
        'pairs':accepted,'protected_inputs':protected,'new_GPU_inferences':0},out)
    tests=[]
    for label,python,pattern in [('CPU',ROOT/'.venv/Scripts/python.exe','test_*.py'),
        ('DASHBOARD',ROOT/'.venv_dashboard/Scripts/python.exe','test_dashboard.py')]:
        command=[str(python),'-B','-m','unittest','discover','-s',str(HERE/'tests'),'-p',pattern,'-v']
        result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',
            timeout=120,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        write_atomic(out/(label+'_TESTS.log'),result.stdout+'\n'+result.stderr,out)
        tests.append({'name':label,'returncode':result.returncode,'command':command})
        if result.returncode:raise ValueError('Acceptance tests failed: '+label)
    decision={'RunID':out.name,'ExperimentID':EXPERIMENT,'source_RunID':study.name,
        'status':'PARTIAL','engineering_acceptance':'VALIDATED','verified_full548_pairs':len(accepted),
        'formal_testdev_pairs':0,'pending_pairs':report['pending_pairs'],'tests':tests,
        'RQ1':'LOGIC_COUNTS_AND_PAIRWISE_CORE_VERIFIED; physical justification remains limited',
        'RQ2':'YOLOv8s resolution x JPEG descriptive DID supported; ck01/ck05 JPEG contrast at 640 only',
        'RQ2_learning':'ML_INSUFFICIENT_EVIDENCE; no grouped held-out score reported',
        'RQ3':'Threshold sensitivities over two JPEG selections/four detector settings; no uniform population estimate',
        'RQ4':'RESOURCE_BLOCKED; matched 2 families x 3 seeds x 2 arms = 12 training runs before ablations',
        'independent_data':'DATASET_BLOCKED: unknown historical test-dev selection/tuning use',
        'full_audio_transcript':'BLOCKED: no locally available recognizer/model; no external upload',
        'GPU_resume':'NOT_VALIDATED; verified complete pairs retained; no automatic restart',
        'new_GPU_inferences':0,'wall_seconds':time.perf_counter()-started}
    save(out/'acceptance.json',decision,out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'EVIDENCE_ACCEPTANCE',
        'status':'PARTIAL','finished_at_utc':datetime.now(timezone.utc).isoformat()},out,replace=True)
    write_atomic(out/'diary.md','# Evidence acceptance\n\nVerified the sealed development run, raw membership, draw intervals, protected input hashes and read-only dashboard. No detector forward or extra COCO evaluation. Scientific status remains PARTIAL.\n',out)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>VM36 evidence acceptance</h1><p>PARTIAL: engineering checks passed; research gates remain.</p><a href="acceptance.json">Acceptance</a>',out)
    seal(out);print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args()
    if Path(args.run).name!=args.run:raise ValueError('Expected a plain RunID.')
    verify(ROOT/'datasets/local_runs'/EXPERIMENT/args.run)
