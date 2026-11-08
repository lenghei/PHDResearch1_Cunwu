"""Summarize sealed development pairs and preserve missing outcomes."""
import argparse
from collections import Counter
from pathlib import Path
import json
import subprocess
import sys
import numpy as np

from common import HERE,ROOT,RUNS,FLAMA,load,save,source,sha256,new_run,resolve_run,seal,verify_hashes
from historical_review import csv_file,table
from development_v1 import BASE,verify_pair


def outcome(ap_ci,latency_ci,tau,limit):
    if ap_ci is not None and ap_ci[1]<tau:return 'FAIL'
    if latency_ci is not None and latency_ci[0]>limit:return 'FAIL'
    if ap_ci is None or latency_ci is None:return 'NOT_EVALUATED'
    if ap_ci[0]>=tau and latency_ci[1]<=limit:return 'PASS'
    return 'UNCERTAIN'


def analyze(continuation):
    result=load(continuation/'result.json');ticket=load(continuation/'author_execution_ticket.json')
    study=resolve_run(ticket['original_study']);plan=load(study/'protocol.json');out=new_run('development_analysis')
    names=result['completed_runs'];rows=[];ci_rows=[];classes=[];cost=[];draws={};evidence=[]
    if len(set(names))!=len(names):raise ValueError('Duplicate measurement RunID.')
    for name in names:
        p=verify_pair(name,study);cfg=load(p/'configuration.json');spec=cfg['pair']
        if spec not in plan['pairs']:raise ValueError('Measurement outside frozen plan.')
        metrics=load(p/'metrics.json');boot=load(p/'bootstrap.json');timing=load(p/'timing.json')
        if boot['draws']['seed']!=plan['bootstrap_seed'] or boot['draws']['samples']!=200:raise ValueError('Bootstrap plan differs.')
        condition='Downscale x1.5' if 'Downscale_x1_5' in spec['p']['leaves'] else 'Extra JPEG Q25'
        common={'RunID':name,'pair_id':spec['id'],'condition_id':spec['p']['id'],'condition':condition,
                'checkpoint':spec['d']['checkpoint'],'checkpoint_sha256':cfg['checkpoint_sha256'],
                'resolution':spec['d']['resolution'],'images':248,'scope':'DEVELOPMENT_ONLY'}
        for area in metrics['per_area']:
            rows.append({**common,**area,'metrics_sha256':sha256(p/'metrics.json'),
                'prediction_sha256':sha256(p/'predictions.jsonl.gz'),'GT_sha256':sha256(p/'ground_truth.json'),
                'mean_pipeline_ms':timing['mean_pipeline_ms'],'latency_ci95_low':timing['mean_pipeline_ci95'][0],
                'latency_ci95_high':timing['mean_pipeline_ci95'][1]})
        ci_rows.extend({**common,**{k:v for k,v in r.items() if k!='ci95'},'ci95_low':r['ci95'][0],
                       'ci95_high':r['ci95'][1],'bootstrap_sha256':sha256(p/'bootstrap.json')} for r in boot['intervals'])
        classes.extend({**common,**r,'metrics_sha256':sha256(p/'metrics.json')} for r in metrics['per_class'])
        cost.append({**common,**{k:v for k,v in timing.items() if k not in ('samples','mean_pipeline_ci95')}})
        draws[spec['id']]=boot['draws']
        evidence.extend(source(p/n) for n in ('configuration.json','metrics.json','bootstrap.json','timing.json',
                                               'predictions.jsonl.gz','ground_truth.json','image_manifest.jsonl','SHA256SUMS.txt'))
    if len({r['pair_id'] for r in rows})!=len(names):raise ValueError('Repeated condition counted as new pair.')
    csv_file(out/'configuration_metrics.csv',rows);csv_file(out/'confidence_intervals.csv',ci_rows)
    csv_file(out/'per_class_metrics.csv',classes);csv_file(out/'timing.csv',cost)
    gains=[]
    for c in ('ck01','ck05','ck06'):
        for condition in plan['conditions']:
            for size in ('all','tiny','small','medium','large'):
                matches=[r for r in rows if r['checkpoint']==c and r['condition_id']==condition['id'] and r['size']==size]
                if {r['resolution'] for r in matches}!={640,1280}:continue
                low=next(r for r in matches if r['resolution']==640);high=next(r for r in matches if r['resolution']==1280)
                a,b=draws[high['pair_id']],draws[low['pair_id']]
                if a['image_ids']!=b['image_ids'] or a['sampling']!=b['sampling']:raise ValueError('Bootstrap image pairing differs.')
                for metric in ('map50','map50_95','ar50_95'):
                    delta=np.array(a['values'][size][metric])-np.array(b['values'][size][metric])
                    bounds=np.quantile(delta,[.025,.975])
                    gains.append({'checkpoint':c,'condition':high['condition'],'condition_id':condition['id'],
                          'size':size,'metric':metric,'estimate':high[metric]-low[metric],
                          'ci95_low':float(bounds[0]),'ci95_high':float(bounds[1]),'resamples':200,
                          'method':'pointwise paired image percentile bootstrap','seed':plan['bootstrap_seed'],
                          'RunID_1280':high['RunID'],'RunID_640':low['RunID'],'scope':'DEVELOPMENT_ONLY'})
    csv_file(out/'resolution_effects.csv',gains)
    classified=[];static=table(BASE/'static_coverage_by_checkpoint.csv')
    for r in rows:
        if r['size'] not in ('all','tiny','small'):continue
        ci=next(x for x in ci_rows if x['pair_id']==r['pair_id'] and x['size']==r['size'] and x['metric']=='map50_95')
        leaves=set(next(p['p']['leaves'] for p in plan['pairs'] if p['id']==r['pair_id']))
        for tau in plan['thresholds']:
            for limit in plan['latency_limits_ms']:
                state=outcome([ci['ci95_low'],ci['ci95_high']],[r['latency_ci95_low'],r['latency_ci95_high']],tau,limit)
                for rule in static:
                    if rule['checkpoint']!=r['checkpoint']:continue
                    classified.append({k:r[k] for k in ('pair_id','checkpoint','condition','condition_id','resolution','size')} |
                        {'tau':tau,'latency_limit_ms':limit,'state':state,'policy':rule['policy'],
                         'static_gap':not leaves.issubset(set(rule['covered_leaves'].split('|'))),
                         'AP_ci_low':ci['ci95_low'],'AP_ci_high':ci['ci95_high'],
                         'latency_ci_low':r['latency_ci95_low'],'latency_ci_high':r['latency_ci95_high']})
    csv_file(out/'pair_states.csv',classified)
    confusion=[];gaps=[]
    for size in ('all','tiny','small'):
        for tau in plan['thresholds']:
            for limit in plan['latency_limits_ms']:
                for policy in ('exact_identity','conservative','supervisor_proxy'):
                    subset=[r for r in classified if (r['size'],r['tau'],r['latency_limit_ms'],r['policy'])==(size,tau,limit,policy)]
                    complete=[r for r in subset if r['state'] in ('PASS','FAIL')]
                    counts=Counter((r['static_gap'],r['state']=='FAIL') for r in complete)
                    tp,fp,fn,tn=(counts[True,True],counts[True,False],counts[False,True],counts[False,False])
                    both={r['state'] for r in complete}=={'PASS','FAIL'}
                    confusion.append({'size':size,'tau':tau,'latency_limit_ms':limit,'policy':policy,
                         'TP':tp,'FP':fp,'FN':fn,'TN':tn,'confirmed':len(complete),'total_pairs':len(subset),
                         'uncertain_or_missing':len(subset)-len(complete),'both_classes':both,
                         'sensitivity':tp/(tp+fn) if both else None,'specificity':tn/(tn+fp) if both else None,
                         'balanced_accuracy':((tp/(tp+fn)+tn/(tn+fp))/2) if both else None,
                         'agreement':(tp+tn)/len(complete) if both else None,
                         'status':'FINITE_SUBSET_DIAGNOSTIC' if both else 'UNDEFINED_MISSING_OUTCOME_CLASS'})
                for p in plan['conditions']:
                    subset=[r for r in classified if (r['size'],r['tau'],r['latency_limit_ms'],r['policy'],r['condition_id'])==
                            (size,tau,limit,'exact_identity',p['id'])]
                    count=Counter(r['state'] for r in subset)
                    state='PASS_EXISTS' if count['PASS'] else 'ALL_SIX_FAIL' if len(subset)==6 and count['FAIL']==6 else 'UNRESOLVED'
                    gaps.append({'condition_id':p['id'],'size':size,'tau':tau,'latency_limit_ms':limit,
                         'tested_detectors':len(subset),'PASS':count['PASS'],'FAIL':count['FAIL'],'UNCERTAIN':count['UNCERTAIN'],
                         'NOT_EVALUATED':count['NOT_EVALUATED'],'subset_state':state,'population_gap':'NOT_EVALUATED'})
    csv_file(out/'static_empirical_diagnostics.csv',confusion);csv_file(out/'subset_gaps.csv',gaps)
    members=load(study/'image_members.json');sha=Counter(r['image_SHA256'] for r in members)
    save(out/'validation_boundaries.json',{'scope':'DEVELOPMENT_ONLY','formal_pairs':0,'test_dev_gate':'DATASET_BLOCKED',
       'image_ID_overlap_old300':0,'image_SHA_overlap_old300':0,'selected_image_IDs':248,'unique_image_byte_SHA':len(sha),
       'duplicate_byte_groups':[s for s,n in sha.items() if n>1],
       'bootstrap_limitations':['One byte-identical image pair remains in the frozen 248-ID population.',
          'Image resampling does not control scene/flight clustering or checkpoint training-seed variability.',
          'Intervals are pointwise exploratory intervals, without simultaneous familywise correction.'],
       'configurations':2,'configuration_selection':'Purposive, not uniform FM_P sampling',
       'latency':'Cached PNG read/hash/decode + prediction/NMS + gzip raw writes; generation and initialization separate.',
       'interactions':'Same-checkpoint resolution effects in two new conditions; no isolated Blur/JPEG DID is identified by this grid.',
       'checkpoint_selection_exposure':'All validation images were historically available for model selection.',
       'repeat':load(resolve_run(result['repeat_recovery'])/'verification.json')})
    save(out/'sources.json',{'continuation':source(continuation/'result.json'),'frozen_protocol':source(study/'protocol.json'),
         'code':source(Path(__file__)),'measurement_sources':evidence,'source_runs':names})
    csv_file(out/'measured_learning_features.csv',table(BASE/'measured_learning_features.csv'))
    with (out/'learning.log').open('x',encoding='utf-8') as stream:
        subprocess.run([str(FLAMA),'-B',str(Path(__file__)),'fit','--run',out.name],stdout=stream,stderr=subprocess.STDOUT,
                       timeout=120,check=True)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':'cv261207','status':'PARTIAL','mode':'DEVELOPMENT_ANALYSIS',
         'unique_new_pairs':len(names),'scope':'DEVELOPMENT_ONLY','source_continuation':continuation.name})
    seal(out);print(out,flush=True)


def fit(out):
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.tree import DecisionTreeRegressor,export_text
    from sklearn.ensemble import RandomForestRegressor
    train=table(out/'measured_learning_features.csv');test=[r for r in table(out/'configuration_metrics.csv') if r['size']=='all']
    features=[json.loads(r['features']) for r in train];target=np.array([float(r['AP50_95']) for r in train]);new=[]
    for r in test:
        template=json.loads(next(t['features'] for t in train if t['checkpoint']==r['checkpoint'] and
                     t['configuration_group']=='I0B0N0_None' and int(json.loads(t['features'])['D:inference_resolution'])==int(r['resolution'])))
        if r['condition']=='Extra JPEG Q25':
            template.update({'P:extra_JPEG':1,'P:JPEG_quality_missing':0,'P:JPEG_quality':25,'P:JPEG_q25':1})
        else:template['P:Downscale_x1_5']=1
        new.append(template)
    vector=DictVectorizer(sparse=False);x=vector.fit_transform(features);xt=vector.transform(new);y=np.array([float(r['map50_95']) for r in test])
    scores=[];predictions=[];rules=[]
    for seed in (42,73,101):
        methods={'rule_stump':DecisionTreeRegressor(max_depth=1,min_samples_leaf=3,random_state=seed),
                 'tree':DecisionTreeRegressor(max_depth=3,min_samples_leaf=3,random_state=seed),
                 'forest':RandomForestRegressor(n_estimators=100,max_depth=5,min_samples_leaf=3,random_state=seed,n_jobs=1)}
        for name,model in methods.items():
            model.fit(x,target);pred=model.predict(xt)
            scores.append({'method':name,'seed':seed,'fit_rows':len(train),'heldout_rows':len(test),'heldout_groups':2,
               'MAE':float(np.mean(abs(pred-y))),'RMSE':float(np.sqrt(np.mean((pred-y)**2))),
               'validation':'IMAGE_AND_CONFIGURATION_DISJOINT_DEVELOPMENT_REGRESSION',
               'scope':'TWO_PURPOSIVE_OUT_OF_DOMAIN_CONDITIONS; checkpoint selection exposure remains'})
            predictions.extend({'pair_id':r['pair_id'],'condition':r['condition'],'checkpoint':r['checkpoint'],
              'method':name,'seed':seed,'measured_AP':float(y[i]),'predicted_AP':float(pred[i])} for i,r in enumerate(test))
            if name!='forest':rules.append(name+f' seed={seed}\n'+export_text(model,feature_names=list(vector.get_feature_names_out()),decimals=5))
    csv_file(out/'heldout_regression.csv',scores);csv_file(out/'heldout_predictions.csv',predictions)
    (out/'rules.txt').write_text('\n\n'.join(rules),encoding='utf-8')
    save(out/'learning_acceptance.json',{'status':'PARTIAL_DEVELOPMENT_REGRESSION','classifier':'ML_INSUFFICIENT_EVIDENCE',
        'fit_rows':279,'fit_image_IDs':300,'heldout_image_IDs':248,'heldout_configurations':len(test),'heldout_degradation_groups':2,
        'balanced_accuracy':None,'macro_F1':None,'failure_precision':None,'failure_recall':None,
        'reason':'Historical fitting data lack complete joint AP/latency class labels; only two new held-out degradation groups.',
        'out_of_domain':['Downscale leaf absent from fit; DictVectorizer ignores this unseen feature.',
                         'Q25 is below observed Q50/Q85; shallow trees cannot extrapolate JPEG quality.',
                         'Historical factor renderer and FM_P renderer have different pixel semantics.'],
        'parameters':'Reused fixed rq_learning.py definitions; no test-response tuning.',
        'training_seed_generalization':'NOT_EVALUATED','independent_test_generalization':'NOT_EVALUATED'})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('analyze','fit'));parser.add_argument('--run',required=True)
    args=parser.parse_args();run=resolve_run(args.run)
    if args.mode=='fit':fit(run)
    else:analyze(run)
