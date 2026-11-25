"""Verify VM26 evidence and reproduce development interaction estimates on CPU."""
import argparse
from collections import Counter
import copy
import csv
from hashlib import sha256 as hash_bytes
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time
import traceback

import numpy as np

from common import ROOT, RUNS, HERE, load, new_run, source, save, sha256, seal, resolve_run
VM = ROOT / 'datasets/research_archive/variability'
V26 = VM / 'experiments/vm26_jpeg_ablation/runs/vm26_r001'
BASE = VM / 'experiments/vm03_gpu_pilot/runs/vm03_r003'
AREAS = ('all','tiny','small')
METRICS = ('map50','map50_95','ar50_95')
CHECKPOINTS = ('ck01','ck05','ck06')


def table(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def csv_file(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def relocated(old):
    normal = str(old).replace('\\','/')
    token = '/Variability/'
    if token not in normal:
        raise ValueError('Unexpected historical source root.')
    relative = normal.split(token, 1)[1]
    path = (VM / relative).resolve(strict=True)
    if not path.is_relative_to(VM.resolve()):
        raise ValueError('Historical path escapes the archive.')
    return path


def checked(path, expected):
    if sha256(path) != expected:
        raise ValueError(f'Historical SHA mismatch: {path.name}')
    return path


def effects(values):
    gain = lambda b,j: values[b,j,1280]-values[b,j,640]
    first = gain(0,'Q50')-gain(0,'None')
    second = gain(2,'Q50')-gain(2,'None')
    return {'JPEG_DID_B0':first, 'JPEG_DID_B2':second, 'R_B_J':second-first}


def prepare(out):
    ledger = {r['path']:r['SHA256'] for r in table(V26/'artifact_SHA256.csv')}
    required = ('configuration_metrics.csv','interaction_effects.csv','headline_effects_and_intervals.csv',
                'bootstrap_acceptance.json','bootstrap_pointwise.csv','bootstrap_joint36.csv',
                'bootstrap/source_manifest.json','bootstrap/indices.npy','bootstrap/plan.json',
                'recipe_freeze.json','model_health.json')
    for name in required:
        checked(V26/'results'/name, ledger['results/'+name])
    lock = load(BASE/'execution_lock.json')
    for name in ('records.json','ground_truth.json','ignore_regions.json','sampling_manifest.csv','dataset_gate.json'):
        checked(BASE/name, lock['files'][name])
    records = load(BASE/'records.json')
    gt = load(BASE/'ground_truth.json')
    if ([r['id'] for r in records] != list(range(1,301)) or len(gt['images']) != 300
            or len({r['image_sha256'] for r in records}) != 300
            or len(gt['annotations']) != 21756):
        raise ValueError('Unexpected shared image or GT population.')
    configs = table(V26/'results/configuration_metrics.csv')
    expected = {(c,b,j,r) for c in CHECKPOINTS for b in (0,2)
                for j in ('None','Q85','Q50') for r in (640,960,1280)}
    found = {(r['CheckpointID'],int(r['blur']),r['jpeg'],int(r['resolution'])) for r in configs}
    if len(configs)!=54 or found != expected:
        raise ValueError('VM26 configuration grid is incomplete.')
    models = table(VM/'models/checkpoints.csv')
    weights = {r['CheckpointID']:r['SHA256'] for r in models if r['CheckpointID'] in CHECKPOINTS}
    proofs = []
    for row in configs:
        metric = checked(relocated(row['source_metric']),row['source_metric_SHA256'])
        raw = checked(relocated(row['source_raw']),row['raw_SHA256'])
        per_area = load(metric)['per_area']
        area = {r['size']:r for r in per_area}
        values = {'AP50':area['all']['map50'], 'AP50_95':area['all']['map50_95'],
                  'AR':area['all']['ar50_95'],
                  **{size.title()+'_AP':area[size]['map50_95'] for size in ('tiny','small','medium','large')}}
        if any(abs(values[k]-float(row[k]))>1e-14 for k in values):
            raise ValueError('Aggregate and cell metrics disagree.')
        if (row['checkpoint_SHA256']!=weights[row['CheckpointID']] or int(row['image_count'])!=300
                or row['label_space']!='NATIVE10' or row['ProtocolID']!='pr01'
                or float(row['illumination'])!=0 or float(row['noise'])!=0):
            raise ValueError('Mixed model, dataset or evaluation identities.')
        ids, image_sha = [], []
        raw_count=0
        with raw.open(encoding='utf-8') as stream:
            for line in stream:
                item=json.loads(line)
                if item.get('checkpoint_SHA256',row['checkpoint_SHA256'])!=row['checkpoint_SHA256']:
                    raise ValueError('Per-image checkpoint mismatch.')
                ids.append(item['image_id'])
                original=item.get('original_id') or item.get('stem')
                if original is not None and original != records[len(ids)-1]['stem']:
                    raise ValueError('Per-image original identity mismatch.')
                raw_count += len(item['predictions'])
        if ids != list(range(1,301)):
            raise ValueError('Incomplete, duplicated or shuffled predictions.')
        proofs.append({'config_id':row['config_id'], 'CheckpointID':row['CheckpointID'],
                       'checkpoint_sha256':row['checkpoint_SHA256'],
                       'metric_file':str(metric), 'metric_sha256':row['source_metric_SHA256'],
                       'raw_file':str(raw), 'raw_sha256':row['raw_SHA256'], 'image_records':len(ids),
                       'prediction_count':raw_count, 'GT_sha256':sha256(BASE/'ground_truth.json')})
    indices=np.load(V26/'results/bootstrap/indices.npy',allow_pickle=False)
    if not np.array_equal(indices,np.random.default_rng(2026102926).integers(0,300,(200,300))):
        raise ValueError('Shared bootstrap indices differ from the frozen seed.')
    sources=load(V26/'results/bootstrap/source_manifest.json')
    arrays={}
    for row in sources:
        folder=V26/'results/bootstrap'/row['config_id']
        complete=load(checked(folder/'complete.json',ledger[f'results/bootstrap/{row["config_id"]}/complete.json']))
        array=checked(folder/'draws.npy',complete['SHA256'])
        if (complete['status']!='PASS' or complete['draws']!=200
                or complete['paired_indices_SHA256']!=sha256(V26/'results/bootstrap/indices.npy')
                or complete['source_manifest_SHA256']!=sha256(V26/'results/bootstrap/source_manifest.json')
                or complete['cache_SHA256']!=row['cache_SHA256']):
            raise ValueError('Bootstrap receipt identity mismatch.')
        checked(relocated(row['point_file']),row['point_SHA256'])
        checked(relocated(row['cache_manifest']),row['cache_manifest_SHA256'])
        values=np.load(array,allow_pickle=False)
        if values.shape!=(200,3,3) or not np.isfinite(values).all():
            raise ValueError('Invalid bootstrap draws.')
        arrays[row['checkpoint'],row['blur'],row['jpeg'],row['resolution']]=values
    if len(arrays)!=24:
        raise ValueError('Expected 24 paired bootstrap endpoints.')
    save(out/'provenance.json',{'scope':'DEVELOPMENT_ONLY','RunID':'vm26_r001',
          'dataset':'ds04','images':300,'GT_instances':21756,'protocol':'pr01',
          'image_sha256':[r['image_sha256'] for r in records],
          'inputs':[source(V26/'artifact_SHA256.csv'),*[source(V26/'results'/name) for name in required],
                    source(BASE/'execution_lock.json'),source(BASE/'records.json'),source(BASE/'ground_truth.json')],
          'configurations':proofs,'bootstrap_endpoints':24,'bootstrap_draws':200})
    csv_file(out/'configuration_metrics.csv',configs)
    return configs, arrays


def point_effects(configs):
    sizes={'AP50_95':('all','map50_95'),'AP50':('all','map50'),
           'Tiny_AP':('tiny','map50_95'),'Small_AP':('small','map50_95'),
           'Medium_AP':('medium','map50_95'),'Large_AP':('large','map50_95'),'AR':('all','ar50_95')}
    contrasts, differences=[],[]
    for c in CHECKPOINTS:
        subset=[r for r in configs if r['CheckpointID']==c]
        for metric in sizes:
            v={(int(r['blur']),r['jpeg'],int(r['resolution'])):float(r[metric]) for r in subset}
            for b in (0,2):
                for j in ('None','Q85','Q50'):
                    for high,low in ((960,640),(1280,640),(1280,960)):
                        differences.append({'checkpoint':c,'blur':b,'jpeg':j,'metric':metric,
                                            'high':high,'low':low,'estimate':v[b,j,high]-v[b,j,low]})
            for name,value in effects(v).items():
                contrasts.append({'checkpoint':c,'metric':metric,'effect':name,'estimate':value})
            for j in ('None','Q85','Q50'):
                value=(v[2,j,1280]-v[2,j,640])-(v[0,j,1280]-v[0,j,640])
                contrasts.append({'checkpoint':c,'metric':metric,'effect':'BLUR_DID_'+j,'estimate':value})
    old={(r['checkpoint'],r['metric'],r['effect']):float(r['estimate'])
         for r in table(V26/'results/interaction_effects.csv')}
    if any(abs(old[r['checkpoint'],r['metric'],r['effect']]-r['estimate'])>1e-13 for r in contrasts):
        raise ValueError('Recomputed contrasts differ from VM26.')
    return contrasts,differences


def intervals(configs, arrays):
    points={(r['CheckpointID'],int(r['blur']),r['jpeg'],int(r['resolution'])):
            {a: {k:next(x for x in load(relocated(r['source_metric']))['per_area'] if x['size']==a)[k]
                 for k in METRICS} for a in AREAS} for r in configs}
    rows, family, columns, estimates, gains=[],[],[],[],[]
    for c in CHECKPOINTS:
        for ai,a in enumerate(AREAS):
            for mi,m in enumerate(METRICS):
                values={(b,j,r):arrays[c,b,j,r][:,ai,mi] for b in (0,2) for j in ('None','Q50') for r in (640,1280)}
                pv={(b,j,r):points[c,b,j,r][a][m] for b,j,r in values}
                for name,d in effects(values).items():
                    estimate=float(effects(pv)[name]);lo,hi=np.quantile(d,[.025,.975])
                    meta={'checkpoint':c,'size':a,'metric':m,'effect':name}
                    rows.append({**meta,'estimate':estimate,'pointwise_low':float(lo),'pointwise_high':float(hi),
                                 'samples':200,'seed':2026102926})
                    if mi==1 or mi==2 and ai==0:
                        family.append(meta);columns.append(d);estimates.append(estimate)
                for b in (0,2):
                    for j in ('None','Q50'):
                        d=values[b,j,1280]-values[b,j,640]
                        lo,hi=np.quantile(d,[.025,.975])
                        gains.append({'checkpoint':c,'size':a,'metric':m,'blur':b,'jpeg':j,
                                      'estimate':pv[b,j,1280]-pv[b,j,640],
                                      'pointwise_low':float(lo),'pointwise_high':float(hi),
                                      'samples':200,'seed':2026102926})
    draw=np.column_stack(columns); point=np.asarray(estimates)
    sd=np.std(draw,axis=0,ddof=1)
    z=np.divide(abs(draw-point),sd,out=np.zeros_like(draw),where=sd>1e-15)
    critical=float(np.quantile(z.max(axis=1),.95))
    joint=[{**meta,'estimate':float(point[i]),'joint36_low':float(point[i]-critical*sd[i]),
            'joint36_high':float(point[i]+critical*sd[i]),'critical':critical} for i,meta in enumerate(family)]
    old={(r['checkpoint'],r['size'],r['metric'],r['effect']):r for r in table(V26/'results/bootstrap_joint36.csv')}
    for r in joint:
        reference=old[tuple(r[k] for k in ('checkpoint','size','metric','effect'))]
        if any(abs(r[k]-float(reference[k]))>1e-12 for k in ('estimate','joint36_low','joint36_high','critical')):
            raise ValueError('Joint bootstrap interval replay differs.')
    return rows,joint,gains


def replay(out, cid):
    from main.labels import mapped_predictions
    from main.statistics import evaluate_coco, _metric_rows
    from evaluation import runtime
    from contextlib import redirect_stdout
    from io import StringIO
    from pycocotools.cocoeval import COCOeval

    runtime(out)
    started=time.perf_counter()
    proof=next(r for r in load(out/'provenance.json')['configurations'] if r['config_id']==cid)
    raw=checked(Path(proof['raw_file']),proof['raw_sha256'])
    records=load(BASE/'records.json')
    gt=load(checked(BASE/'ground_truth.json',proof['GT_sha256']))
    region=load(BASE/'ignore_regions.json')
    region={int(k):v for k,v in region.items()}
    predictions=[]
    with raw.open(encoding='utf-8') as stream:
        for line in stream:
            predictions.extend(json.loads(line)['predictions'])
    pred=mapped_predictions(predictions,records,region,'native10','finetuned',3000)
    areas,_,ev=evaluate_coco(gt,pred,3000)
    expected={r['size']:r for r in load(Path(proof['metric_file']))['per_area']}
    point_error=max(abs(row[k]-expected[row['size']][k]) for row in areas for k in METRICS)
    if point_error>1e-12:
        raise ValueError('Raw prediction COCO replay differs.')
    # A fresh match calculation is reused across paired aggregate AP accumulations.
    n=300;na=len(ev.params.areaRng);nk=len(ev.params.catIds)
    ev.evalImgs=[ev.evalImgs[(k*na+a)*n+i] for k in range(nk) for a in range(3) for i in range(n)]
    ev.params.areaRng=ev.params.areaRng[:3];ev.params.areaRngLbl=list(AREAS)
    ev._paramsEval=copy.deepcopy(ev.params)
    indices=np.load(V26/'results/bootstrap/indices.npy',allow_pickle=False)
    saved=np.load(V26/f'results/bootstrap/{cid}/draws.npy',allow_pickle=False)
    replayed=[]
    for index in (0,99,199):
        sample=COCOeval(iouType='bbox')
        sample.params=copy.deepcopy(ev.params)
        sample._paramsEval=copy.deepcopy(sample.params)
        sample.evalImgs=[ev.evalImgs[g*n+int(i)] for g in range(nk*3) for i in indices[index]]
        with redirect_stdout(StringIO()):
            sample.accumulate()
        metrics=_metric_rows(sample,include_per_class=False,include_gt_counts=False)[0]
        value=np.array([[row[k] for k in METRICS] for row in metrics])
        error=float(abs(value-saved[index]).max())
        if error>1e-12:
            raise ValueError('Fresh paired COCO bootstrap draw differs.')
        replayed.append({'draw_index':index,'max_abs_error':error})
    save(out/f'replay_{cid}.json',{'status':'PASS','config_id':cid,'source_raw_sha256':proof['raw_sha256'],
         'GT_sha256':proof['GT_sha256'],'point_max_abs_error':point_error,
         'fresh_COCO_match_and_accumulate':True,'fresh_bootstrap_draws':replayed,
         'seconds':time.perf_counter()-started})
    print('Fresh COCO replay PASS:',cid,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('review','replay'))
    parser.add_argument('--run');parser.add_argument('--config')
    args=parser.parse_args()
    if args.mode=='replay':
        replay(resolve_run(args.run),args.config)
        return
    out=new_run('vm26review')
    save(out/'protocol.json',{'scope':'DEVELOPMENT_ONLY_POST_HOC','source_RunID':'vm26_r001',
                             'fresh_replay_configs':['ck01_I0B0N0_JNone_R640','ck01_I0B0N0_JQ50_R1280'],
                             'bootstrap_replay_indices':[0,99,199],
                             'rule_depth':2,'practical_effect_threshold':.01,
                             'held_out_image_and_configuration_validation':'NOT_EVALUATED',
                             'reason':'All configuration aggregates and their prior bootstrap draws share the same 300 images.'})
    print('Review:',out,flush=True)
    try:
        configs,arrays=prepare(out)
        contrasts,diffs=point_effects(configs)
        point,joint,gains=intervals(configs,arrays)
        for name,rows in [('interaction_effects',contrasts),('resolution_differences',diffs),
                          ('bootstrap_pointwise',point),('bootstrap_joint36',joint),('resolution_gain_intervals',gains)]:
            csv_file(out/(name+'.csv'),rows)
        print('54 sources and 200 paired draws per endpoint verified; intervals reproduced.',flush=True)
        for cid in load(out/'protocol.json')['fresh_replay_configs']:
            with (out/f'cpu_{cid}.log').open('x',encoding='utf-8') as log:
                result=subprocess.run([sys.executable,'-B',str(Path(__file__)),'replay','--run',out.name,'--config',cid],
                                      timeout=600,stdout=log,stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError('Raw replay failed; see local CPU log.')
        from historical_rules import fit_rules
        fit_rules(out)
        from historical_figures import render
        render(out)
        save(out/'run.json',{'RunID':out.name,'ExperimentID':'cv261207','status':'COMPLETE_WITH_LIMITATIONS',
                            'mode':'historical_vm26_review','scope':'DEVELOPMENT_ONLY',
                            'formal_pairs_measured':0})
        seal(out)
    except BaseException:
        save(out/'failure.json',{'status':'PARTIAL','traceback':traceback.format_exc()})
        print(out,flush=True)
        raise
    print('COMPLETE_WITH_LIMITATIONS:',out,flush=True)


if __name__=='__main__':
    main()
