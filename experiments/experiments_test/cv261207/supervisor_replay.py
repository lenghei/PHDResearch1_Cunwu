"""Verify saved development pairs and optionally regenerate COCO metrics on CPU."""
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from common import load,source,new_run,resolve_run,seal,verify_hashes
from supervisor_compliance import save


def replay(names,recalculate=False):
    out=new_run('supervisor_replay');rows=[]
    for name in names:
        pair=resolve_run(name)
        checks=verify_hashes(pair,('configuration.json','metrics.json','verification.json',
             'predictions.jsonl.gz','ground_truth.json','bootstrap.json'))
        if load(pair/'configuration.json')['protocol']!='CV261207_DEV248_NATIVE10_DENSE_FP32_V1':
            raise ValueError('This replay entry accepts only the frozen development protocol.')
        if not load(pair/'verification.json')['COCO_recomputed_from_raw']:raise ValueError('Missing original raw replay certificate.')
        if recalculate:
            from evaluation import evaluate_saved
            metrics,_=evaluate_saved(pair)
            if metrics!=load(pair/'metrics.json'):raise ValueError('COCO replay differs: '+name)
        rows.append({'RunID':name,'input_hashes':checks,'hash_verification':'PASS',
              'COCO_regeneration':'PASS' if recalculate else 'NOT_RERUN',
              'image_bootstrap':'Original paired draws verified by SHA; not recomputed.'})
    save(out/'replay.json',{'status':'PASS','GPU_seconds':0,'new_inference_pairs':0,'pairs':rows},out)
    save(out/'run.json',{'RunID':out.name,'status':'PASS','mode':'CPU_EVIDENCE_REPLAY','GPU_seconds':0},out)
    seal(out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pair',action='append',required=True)
    p.add_argument('--metrics',action='store_true',help='Recompute saved COCO AP; never run inference.')
    args=p.parse_args();replay(args.pair,args.metrics)
