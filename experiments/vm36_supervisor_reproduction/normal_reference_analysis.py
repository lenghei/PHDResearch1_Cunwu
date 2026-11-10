"""Replay new Normal predictions and complete matched contrasts on CPU."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
from protocol import ROOT,HERE,EXPERIMENT,load,save,source,sha256,write_atomic
from common import seal,verify_hashes
from historical_review import table,csv_file
from main.pipeline import new_run
from full_development import PREP
from full_analysis import paired_effect,tested_subset_state
from full20_completion import read_pair
from full20_results import latency_interval_state
from gap_analysis import pair_outcome
from post80_replay import ANALYSIS,run_path
from post80_science import classification_scores
from normal_reference import SCIENCE,contract

CONDITIONS={'p_59915f60d81c6f17':'JPEG Q25','p_5a06deca286f53b2':'JPEG Q75','p_9031c68f63a468d1':'Gaussian sigma15'}
NORMAL='p_cdbfeb94b010c58d'
SIZES=('all','tiny','small','medium','large')
METRICS=('map50_95','map50','ar50_95')


def metric(item,size,name):return next(r[name] for r in item['metrics']['per_area'] if r['size']==size)


def timing_difference(normal,degraded,key,seed):
    left=normal['timing']['samples'];right=degraded['timing']['samples']
    if [r['image_id'] for r in left]!=[r['image_id'] for r in right] or len(left)!=548:raise ValueError('Timing image pairs differ.')
    original=np.asarray([r[key] for r in left]);changed=np.asarray([r[key] for r in right]);values=changed-original
    indices=np.random.default_rng(seed).integers(0,548,(200,548));low,high=np.quantile(values[indices].mean(axis=1),[.025,.975])
    return {'normal_mean_ms':float(original.mean()),'degraded_mean_ms':float(changed.mean()),
        'difference_ms':float(values.mean()),'CI95_low_ms':float(low),'CI95_high_ms':float(high)}


def contrasts(found,seed):
    differences=[];latencies=[];interactions=[];classes=[]
    for p,name in CONDITIONS.items():
        for ck in ('ck01','ck05'):
            for resolution in (640,1280):
                nkey=NORMAL+'__'+ck+'_Infer'+str(resolution);dkey=p+'__'+ck+'_Infer'+str(resolution)
                complete=nkey in found and dkey in found
                for size in SIZES:
                    for m in METRICS:
                        values=paired_effect([(1,found[dkey]['metrics'],found[dkey]['bootstrap']),(-1,found[nkey]['metrics'],found[nkey]['bootstrap'])],size,m) if complete else {'estimate':None,'CI95_low':None,'CI95_high':None,'images':548,'draws':200}
                        differences.append({'condition':name,'P':p,'checkpoint':ck,'resolution':resolution,'size':size,'metric':m,
                            'normal_value':metric(found[nkey],size,m) if complete else None,'degraded_value':metric(found[dkey],size,m) if complete else None,
                            **values,'status':'CONTROLLED_COMPARISON' if complete else 'NOT_ESTIMABLE',
                            'direction':'Degraded minus Normal, fixed checkpoint and resolution','normal_pair':nkey,'degraded_pair':dkey,
                            'normal_source':found[nkey]['folder'] if complete else None,'degraded_source':found[dkey]['folder'] if complete else None})
                if complete:
                    for key in ('predict_ms','forward_ms','pipeline_ms'):
                        latencies.append({'condition':name,'checkpoint':ck,'resolution':resolution,'component':key,
                            **timing_difference(found[nkey],found[dkey],key,seed),'images':548,'draws':200,
                            'evidence_kind':'DESCRIPTIVE_CROSS_RUN_TIMING','normal_pair':nkey,'degraded_pair':dkey,
                            'limitation':'Runs were not interleaved or clock-calibrated. Image-bootstrap timing intervals do not cover hardware/run variability.'})
                    reference={(r['size'],r['category_id']):r for r in found[nkey]['metrics']['per_class']}
                    for r in found[dkey]['metrics']['per_class']:
                        original=reference[r['size'],r['category_id']]
                        for m in METRICS:
                            classes.append({'condition':name,'checkpoint':ck,'resolution':resolution,'size':r['size'],
                                'category_id':r['category_id'],'category_name':r['category_name'],'metric':m,
                                'normal_value':original[m],'degraded_value':r[m],
                                'difference':r[m]-original[m] if r[m] is not None and original[m] is not None else None,
                                'CI_status':'NOT_EVALUATED_PER_CLASS_BOOTSTRAP','normal_pair':nkey,'degraded_pair':dkey})
            keys=[p+'__'+ck+'_Infer1280',p+'__'+ck+'_Infer640',NORMAL+'__'+ck+'_Infer1280',NORMAL+'__'+ck+'_Infer640']
            complete=all(k in found for k in keys)
            for size in SIZES:
                for m in METRICS:
                    values=paired_effect([(sign,found[k]['metrics'],found[k]['bootstrap']) for sign,k in zip((1,-1,-1,1),keys)],size,m) if complete else {'estimate':None,'CI95_low':None,'CI95_high':None,'images':548,'draws':200}
                    interactions.append({'condition':name,'P':p,'checkpoint':ck,'size':size,'metric':m,**values,
                        'normal_resolution_change':metric(found[keys[2]],size,m)-metric(found[keys[3]],size,m) if complete else None,
                        'degraded_resolution_change':metric(found[keys[0]],size,m)-metric(found[keys[1]],size,m) if complete else None,
                        'status':'CONTROLLED_COMPARISON' if complete else 'NOT_ESTIMABLE',
                        'definition':'[degraded1280-degraded640]-[Normal1280-Normal640]','paired_sources':'|'.join(keys)})
    return differences,latencies,interactions,classes


def subset_states(found,seed):
    pids=sorted({r['spec']['p']['id'] for r in found.values()}|{NORMAL});settings=[c+'_Infer'+str(r) for c in ('ck01','ck05') for r in (640,1280)]
    indices=np.random.default_rng(seed).integers(0,548,(200,548));rows=[];gaps=[]
    for p in pids:
        for d in settings:
            item=found.get(p+'__'+d)
            if item:
                ci=next(r['ci95'] for r in item['bootstrap']['intervals'] if r['size']=='tiny' and r['metric']=='map50_95')
                t=np.asarray([r['pipeline_ms'] for r in item['timing']['samples']]);latency=list(map(float,np.quantile(t[indices].mean(axis=1),[.025,.975])))
            for tau in (.05,.10,.15):
                for limit in (50,100):
                    rows.append({'P':p,'D':d,'pair_id':p+'__'+d,'tau':tau,'L_cached_ms':limit,
                        'state':pair_outcome(ci,latency[1],tau,limit) if item else 'NOT_EVALUATED',
                        'latency_interval_state':latency_interval_state(ci,latency,tau,limit) if item else 'NOT_EVALUATED',
                        'Tiny_AP_low':ci[0] if item else None,'Tiny_AP_high':ci[1] if item else None,
                        'cached_latency_low_ms':latency[0] if item else None,'cached_latency_high_ms':latency[1] if item else None})
        for tau in (.05,.10,.15):
            for limit in (50,100):
                selected=[r for r in rows if r['P']==p and r['tau']==tau and r['L_cached_ms']==limit]
                gaps.append({'P':p,'tau':tau,'L_cached_ms':limit,'subset_state':tested_subset_state([r['state'] for r in selected],4),
                    'latency_interval_subset_state':tested_subset_state([r['latency_interval_state'] for r in selected],4),
                    'measured_settings':sum(r['state']!='NOT_EVALUATED' for r in selected),'declared_settings':4})
    return rows,gaps


def gap_and_learning(out,found,seed):
    pairs,gaps=subset_states(found,seed);csv_file(out/'PAIR_STATES.csv',pairs);csv_file(out/'FINITE21_GAPS.csv',gaps)
    old_gap=table(run_path(ANALYSIS)/'FINITE20_GAPS.csv')
    for row in old_gap:
        current=next(r for r in gaps if r['P']==row['p_id'] and r['tau']==float(row['tau']) and r['L_cached_ms']==int(row['L_cached_ms']))
        if current['subset_state']!=row['subset_state']:raise ValueError('An original20-P outcome changed.')
    recipes={r['spec']['p']['id']:r['spec']['p']['leaves'] for r in found.values()};policies=table(PREP/'STATIC_COVERAGE_REFERENCE.csv')
    comparisons=[];cohorts=[]
    for name,ids in (('ORIGINAL20',set(recipes)-{NORMAL}),('AUGMENTED21',set(recipes)|{NORMAL})):
        for tau in (.05,.1,.15):
            for limit in (50,100):
                states=[r for r in gaps if r['P'] in ids and r['tau']==tau and r['L_cached_ms']==limit]
                counts=Counter(r['subset_state'] for r in states)
                cohorts.append({'cohort':name,'selected_P':len(ids),'tau':tau,'L_cached_ms':limit,
                    **{k:counts[k] for k in ('GAP_TESTED_SUBSET','COVERED_TESTED_SUBSET','UNRESOLVED')}})
                for policy in ('exact_identity','conservative','supervisor_proxy'):
                    confusion=Counter()
                    for row in states:
                        if row['subset_state']=='UNRESOLVED':continue
                        predicted=not any(set(recipes.get(row['P'],['Normal'])).issubset(set(p['covered_leaves'].split('|')))
                            for p in policies if p['policy']==policy and p['checkpoint'] in ('ck01','ck05'))
                        confusion[(row['subset_state']=='GAP_TESTED_SUBSET',predicted)]+=1
                    tp,tn,fp,fn=(confusion[True,True],confusion[False,False],confusion[False,True],confusion[True,False]);n=tp+tn+fp+fn
                    comparisons.append({'cohort':name,'policy':policy,'tau':tau,'L_cached_ms':limit,'selected_P':len(ids),
                        'decisive_P':n,'unresolved_P':counts['UNRESOLVED'],'TP_gap':tp,'TN_gap':tn,'FP_gap':fp,'FN_gap':fn,
                        'accuracy':(tp+tn)/n if n else None,'constant_gap_accuracy':(tp+fn)/n if n else None,
                        'scope':'Descriptive selected-cohort agreement, not independent or population performance.'})
    csv_file(out/'COHORT_GAP_COUNTS.csv',cohorts);csv_file(out/'STATIC_AGREEMENT.csv',comparisons)
    primary=[r for r in pairs if r['tau']==.1 and r['L_cached_ms']==100];predictions=[];folds=[]
    pids=sorted({r['P'] for r in primary})
    for pid in pids:
        training=[r for r in primary if r['P']!=pid and r['state'] in ('PASS','FAIL')]
        testing=[r for r in primary if r['P']==pid and r['state'] in ('PASS','FAIL')];majority=Counter(r['state'] for r in training).most_common(1)[0][0]
        actual_training_groups=sorted({r['P'] for r in training})
        folds.append({'heldout_P':pid,'train_P_IDs':'|'.join(actual_training_groups),'train_groups':len(actual_training_groups),
            'train_PASS':sum(r['state']=='PASS' for r in training),'train_FAIL':sum(r['state']=='FAIL' for r in training),
            'test_PASS':sum(r['state']=='PASS' for r in testing),'test_FAIL':sum(r['state']=='FAIL' for r in testing)})
        predictions.extend({'pair_id':r['pair_id'],'heldout_P':pid,'truth':r['state'],'prediction':majority} for r in testing)
    csv_file(out/'LEARNING_GROUP_FOLDS.csv',folds);csv_file(out/'MAJORITY_PREDICTIONS.csv',predictions)
    save(out/'LEARNING_FEASIBILITY.json',{'status':'ML_INSUFFICIENT_EVIDENCE','selected_P_groups':len(pids),
        'class_balance':dict(Counter(r['state'] for r in primary)),'PASS_P_groups':len({r['P'] for r in primary if r['state']=='PASS'}),
        'independent_scenes':'UNKNOWN','baseline':classification_scores([r['truth'] for r in predictions],[r['prediction'] for r in predictions]),
        'interpretable_model_fitted':False,'independent_image_validation':False,'new_images_added':0,
        'scope':'Configuration-grouped development diagnostic; a new Normal P does not repair historical image-selection exposure or weak minority-group support.'},out)
    return cohorts


def load_sources(worker):
    minimum=load(run_path(SCIENCE)/'FROZEN_MINIMUM_GPU_PLAN.json');contract(minimum)
    original=run_path(ANALYSIS);verify_hashes(original,('EVIDENCE_SOURCES.json','summary.json'))
    found={r['pair_id']:read_pair(Path(r['folder'])) for r in load(original/'EVIDENCE_SOURCES.json')['conditions']}
    if len(found)!=80:raise ValueError('Accepted80 source identity differs.')
    new=[]
    for spec in minimum['new_pairs']:
        folder=worker/'pairs'/spec['id']
        if not (folder/'SHA256SUMS.txt').exists():continue
        item=read_pair(folder)
        if any(item['spec'][k]!=spec[k] for k in ('id','p','d','images')) or spec['id'] in found:
            raise ValueError('A new scientific configuration differs from its frozen reference.')
        found[spec['id']]=item;new.append(spec['id'])
    seed=minimum['seed'];gt_sha=next(iter(found.values()))['checks']['ground_truth.json']
    for item in found.values():
        if item['checks']['ground_truth.json']!=gt_sha or item['bootstrap']['draws']['seed']!=seed or item['bootstrap']['draws']['samples']!=200:
            raise ValueError('GT or paired-bootstrap protocol differs.')
    leaves={'p_59915f60d81c6f17':'JPEG_q25','p_5a06deca286f53b2':'JPEG_q75','p_9031c68f63a468d1':'Gauss_s15'}
    from alignment import parameters
    from protocol import FMP
    for p,leaf in leaves.items():
        for ck in ('ck01','ck05'):
            for resolution in (640,1280):
                item=found[p+'__'+ck+'_Infer'+str(resolution)]
                if set(item['spec']['p']['leaves'])!={'Normal',leaf} or item['spec']['p']['attributes']!=parameters(FMP,['Normal',leaf]):
                    raise ValueError('A reference is not the specified one-factor degradation.')
    return found,new,minimum


def accept_new_evidence(out,worker,found,new):
    plan=load(worker/'EXECUTION_PROTOCOL.json');supervisor=load(worker/'supervisor.json')
    if supervisor['hard_budget_seconds']!=7200 or supervisor['actual_supervision_seconds']>7200:
        raise ValueError('The authorized worker budget was not satisfied.')
    recorded=load(worker/'inputs.json')['worker_source']
    if sha256(Path(recorded['path']))!=recorded['sha256'] or sha256(worker/'executed_source.py')!=recorded['sha256']:
        raise ValueError('The measured worker source changed.')
    for key in ('GT','image_members','ZIP','FMP','FMD','generator'):
        if sha256(Path(plan[key]['path']))!=plan[key]['sha256']:raise ValueError('Protected source changed after inference: '+key)
    for relative,digest in plan['scientific_dependencies'].items():
        if sha256(ROOT/'main'/relative)!=digest:raise ValueError('Evaluation/prediction source changed.')
    proofs=load(worker/'NORMAL_PIXEL_PROOFS.json');expected=[(r['image_id'],r['pixel_sha256'],r['shape']) for r in proofs];checks=[]
    for identity in new:
        item=found[identity];spec=item['spec'];folder=Path(item['folder'])
        if sha256(Path(spec['d']['path']))!=spec['d']['sha256']:raise ValueError('Source checkpoint changed.')
        manifest=[json.loads(s) for s in (folder/'image_manifest.jsonl').read_text().splitlines()]
        if [(r['image_id'],r['pixel_sha256'],r['shape']) for r in manifest]!=expected:
            raise ValueError('Normal input pixels are not the original decoded population.')
        required={'NCHW':[1,3,spec['d']['resolution'],spec['d']['resolution']],'dtype':'torch.float32'}
        if any(r['tensor']!=required for r in manifest) or not item['timing']['FP32'] or item['timing']['TF32']:
            raise ValueError('Actual input precision or shape differs.')
        checks.append({'pair_id':identity,'images':len(manifest),'actual_tensor':required,'original_pixels_equal':True,
            'GT_SHA256':item['checks']['ground_truth.json'],'checkpoint_SHA256':spec['d']['sha256']})
    save(out/'NORMAL_ACCEPTANCE.json',{'accepted_pairs':len(checks),'full_four_pairs':len(checks)==4,
        'checks':checks,'source_and_budget_unchanged':True,'scientific_scope':'DEVELOPMENT_ONLY'},out)


def continuation_sources(out):
    science=run_path(SCIENCE)
    names=('INDEPENDENT_ADMISSION_UPDATE.json','UAVDT_PROTOCOL_UPDATE.json','RQ4_READINESS_UPDATE.json')
    verify_hashes(science,names)
    admission,uavdt,training=(load(science/name) for name in names)
    refs=[row['source'] for row in admission['direct_sources_rechecked']]
    for row in admission['checkpoint_history']:
        refs.extend({'path':row[k+'_source'],'sha256':row[k+'_SHA256']} for k in ('args','curve'))
    refs.extend((uavdt['source_archive'],training['CPU_loader_adapter']))
    verified=[]
    for ref in refs:
        path=Path(ref['path'])
        if sha256(path)!=ref['sha256']:raise ValueError('Continuation evidence changed: '+str(path))
        verified.append(source(path))
    save(out/'CONTINUATION_SOURCE_RECHECK.json',{'verified_sources':verified,
        'scope':'Bounded recheck of named source evidence; no inference, training or broad history search.',
        'new_independence_evidence':False,'negative_search_hits_used_as_proof':False,
        'admission':admission['status'],'UAVDT':uavdt['status'],'training':training['status']},out)


def analyze(out,worker):
    from evaluation import evaluate_saved
    began=time.perf_counter();found,new,minimum=load_sources(worker);accept_new_evidence(out,worker,found,new)
    verification=[];normal=[];per_class=[]
    for identity in new:
        item=found[identity];folder=Path(item['folder']);t=time.perf_counter();replayed,ev=evaluate_saved(folder)
        if replayed!=item['metrics']:raise ValueError('New Normal raw replay differs.')
        verification.append({'pair_id':identity,'exact_equality':True,'seconds':time.perf_counter()-t,
            'prediction_SHA256':item['checks']['predictions.jsonl.gz'],'GT_SHA256':item['checks']['ground_truth.json']})
        del ev
        spec=item['spec']
        for area in item['metrics']['per_area']:
            normal.append({'pair_id':identity,'checkpoint':spec['d']['checkpoint'],'resolution':spec['d']['resolution'],**area,
                'AP95_CI':next(r['ci95'] for r in item['bootstrap']['intervals'] if r['size']==area['size'] and r['metric']=='map50_95'),
                'source_folder':item['folder'],'checkpoint_SHA256':spec['d']['sha256']})
        per_class.extend({'pair_id':identity,**r} for r in item['metrics']['per_class'])
    csv_file(out/'NEW_RAW_REPLAY.csv',verification);csv_file(out/'NORMAL_METRICS.csv',normal);csv_file(out/'NEW_PER_CLASS_METRICS.csv',per_class)
    changes,latency,did,classes=contrasts(found,minimum['seed'])
    for frozen in minimum['contrasts']:
        expected=[r['pair_id'] for r in frozen['terms']]
        if [r['coefficient'] for r in frozen['terms']]!=[1,-1,-1,1]:raise ValueError('Frozen contrast signs differ.')
        row=next(r for r in did if r['size']=='all' and r['metric']=='map50_95' and r['paired_sources'].split('|')==expected)
        if not row:raise ValueError('A preregistered primary contrast is absent.')
    csv_file(out/'WITHIN_RESOLUTION_CONTRASTS.csv',changes);csv_file(out/'LATENCY_DIFFERENCES.csv',latency)
    csv_file(out/'RESOLUTION_DID.csv',did);csv_file(out/'PER_CLASS_POINT_DIFFERENCES.csv',classes)
    cohorts=gap_and_learning(out,found,minimum['seed'])
    save(out/'METRIC_PROVENANCE.json',{'pairs':[{'pair_id':k,'source_folder':v['folder'],'SHA256':v['checks']} for k,v in found.items()],
        'old_pairs_repeated':False,'new_normal_pairs':new,'source_original80':source(run_path(ANALYSIS)/'SHA256SUMS.txt'),
        'new_worker':source(worker/'SHA256SUMS.txt'),'minimum_plan':source(run_path(SCIENCE)/'FROZEN_MINIMUM_GPU_PLAN.json')},out)
    continuation_sources(out)
    save(out/'CONTINUATION_GATES.json',{'RQ1':'PARTIAL','RQ2':'PARTIAL','RQ3':'PARTIAL','RQ4':'NOT_EVALUATED',
        'testdev':'DATASET_BLOCKED','UAVDT':'NOT_ADMITTED_FOR_INDEPENDENT_EVALUATION','new_testdev_inference':0,'new_UAVDT_inference':0,
        'RQ4_training':'TRAINING_NOT_READY; no initialization/range/budget or production-trainer gate is repaired by Normal inference.',
        'verified_full_pool_CPU_adapter':source(run_path(SCIENCE)/'RQ4_READINESS_UPDATE.json'),
        'checkpoint_selection_history':source(run_path(SCIENCE)/'INDEPENDENT_ADMISSION_UPDATE.json'),
        'UAVDT_source_semantics':source(run_path(SCIENCE)/'UAVDT_PROTOCOL_UPDATE.json'),'new_training_runs':0,
        'independent_validation_results':0,'new_historical_exposure_evidence':False},out)
    figures(out,changes,did)
    report(out,worker,found,new,changes,did,cohorts,latency)
    save(out/'analysis_result.json',{'status':'NORMAL4_COMPLETED_DEVELOPMENT_ONLY' if len(new)==4 else 'PARTIAL_WITH_LIMITATIONS',
        'new_pairs':len(new),'combined_pairs':len(found),'saved_new_image_records':548*len(new),'original80_rerun':False,
        'CPU_analysis_seconds':time.perf_counter()-began,'new_training_runs':0,'independent_validation_results':0},out)


def figures(out,changes,did):
    os.environ['MPLCONFIGDIR']=str(out/'runtime/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if not any(r['status']=='CONTROLLED_COMPARISON' for r in changes):
        save(out/'FIGURE_PROVENANCE.json',{'status':'NOT_ESTIMABLE_NO_ACCEPTED_NORMAL','outputs':[]},out)
        write_atomic(out/'FIGURE_CAPTIONS_EN.md','No complete Normal-reference comparison was available. No estimated figure was generated.\n',out)
        return
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12,5.2),sharex=True,layout='constrained')
    for ax,ck,name in zip(axes,('ck01','ck05'),('YOLO26s','YOLOv8s')):
        rows=[r for r in changes if r['checkpoint']==ck and r['size']=='all' and r['metric']=='map50_95' and r['status']=='CONTROLLED_COMPARISON']
        for i,r in enumerate(rows):
            ax.errorbar(r['estimate'],i,xerr=[[r['estimate']-r['CI95_low']],[r['CI95_high']-r['estimate']]],fmt='o',capsize=3,color='#28658c')
        ax.set_yticks(range(len(rows)),[r['condition']+' / '+str(r['resolution']) for r in rows],fontsize=8)
        ax.invert_yaxis();ax.axvline(0,color='.6',linewidth=.8);ax.set_title(name);ax.set_xlabel('AP50:95: degraded - Normal (0-1)');ax.grid(axis='x',alpha=.15)
        if not rows:ax.text(.5,.5,'No complete matched comparison',transform=ax.transAxes,ha='center')
    fig.suptitle('One additional degradation, same checkpoint and inference resolution')
    fig.savefig(out/'controlled_degradation_effects.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');names=list(CONDITIONS.values())
    for i,(ck,name,color) in enumerate((('ck01','YOLO26s','#28658c'),('ck05','YOLOv8s','#a35b20'))):
        rows=[r for r in did if r['checkpoint']==ck and r['size']=='all' and r['metric']=='map50_95' and r['status']=='CONTROLLED_COMPARISON']
        for r in rows:
            x=names.index(r['condition'])+(i-.5)*.12
            ax.errorbar(x,r['estimate'],yerr=[[r['estimate']-r['CI95_low']],[r['CI95_high']-r['estimate']]],fmt='o',capsize=3,color=color,label=name if r==rows[0] else None)
    ax.set_xticks(range(3),names);ax.axhline(0,color='.6',linewidth=.8);ax.set_ylabel('Resolution DID in AP50:95 (0-1)')
    ax.set_title('(Degraded1280 - degraded640) - (Normal1280 - Normal640)');ax.legend(frameon=False);ax.grid(axis='y',alpha=.15)
    fig.savefig(out/'resolution_degradation_interactions.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained');colors=('#28658c','#a35b20','#287655')
    for row_index,(ck,name) in enumerate((('ck01','YOLO26s'),('ck05','YOLOv8s'))):
        for col_index,res in enumerate((640,1280)):
            ax=axes[row_index,col_index]
            for j,condition in enumerate(names):
                rows=[r for r in changes if r['checkpoint']==ck and r['resolution']==res and r['condition']==condition and r['metric']=='map50_95' and r['status']=='CONTROLLED_COMPARISON']
                for r in rows:
                    x=SIZES.index(r['size'])+(j-1)*.14
                    ax.errorbar(x,r['estimate'],yerr=[[r['estimate']-r['CI95_low']],[r['CI95_high']-r['estimate']]],fmt='o',capsize=2,color=colors[j],label=condition if r['size']=='all' else None)
            ax.set_xticks(range(5),SIZES);ax.axhline(0,color='.6',linewidth=.8);ax.set_title(name+' / '+str(res));ax.set_ylabel('Degraded - Normal AP50:95')
            ax.grid(axis='y',alpha=.15)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.savefig(out/'size_dependent_degradation_effects.png',dpi=160);plt.close(fig)
    save(out/'FIGURE_PROVENANCE.json',{'inputs':[source(out/n) for n in ('WITHIN_RESOLUTION_CONTRASTS.csv','RESOLUTION_DID.csv')],
        'outputs':[source(out/n) for n in ('controlled_degradation_effects.png','resolution_degradation_interactions.png','size_dependent_degradation_effects.png')]},out)
    write_atomic(out/'FIGURE_CAPTIONS_EN.md','# Figure captions\n\nControlled additional-degradation effects. Points show degraded-minus-Normal COCO AP50:95 under a fixed checkpoint and inference resolution. Each corner uses the same 548 validation image identities, original-coordinate GT and native 10-class evaluation. Bars are pointwise 95% paired-image percentile intervals from 200 shared resamples. JPEG Q75 is additional recompression, not the Normal baseline. Source: WITHIN_RESOLUTION_CONTRASTS.csv.\n\nResolution interaction. DID equals the 1280-minus-640 AP change under the specified degradation minus the corresponding change under Normal. The six primary definitions were frozen before Normal inference. These are exploratory development comparisons with no familywise or training-seed significance claim. Source: RESOLUTION_DID.csv.\n\nSize dependence. The overall result and four original-coordinate area bins are shown for both checkpoints and inference resolutions. Absolute AP, AP50, AR and source identities are provided in the CSV files. Sources: WITHIN_RESOLUTION_CONTRASTS.csv and METRIC_PROVENANCE.json.\n',out)


def report(out,worker,found,new,changes,did,cohorts,latency):
    plan=load(worker/'EXECUTION_PROTOCOL.json');result=load(worker/'worker_result.json');supervisor=load(worker/'supervisor.json')
    learning=load(out/'LEARNING_FEASIBILITY.json');static=table(out/'STATIC_AGREEMENT.csv');metrics=[]
    text=['# VM36：Normal参考与合法匹配对照','',
        '## PROBLEM / WHY','',
        '原80组缺少无额外退化的Normal参考，无法直接量化额外JPEG或Gaussian σ=15相对源图的变化。本轮补齐这一参考，完成同权重、同分辨率的一因素计算对照。比较仍属探索性开发实验。','',
        '## SOLUTION','',
        f'本轮实际新增{len(new)}/4组，每组548张VisDrone2019-DET-val；新增保存{548*len(new)}条逐图记录。原80组不重跑，累计{len(found)}组。',
        'Normal仅gain=1.0、bias=0，其余额外退化全部关闭。它保留原JPEG解码像素，不代表无损相机采集。数值来自PDF对齐的FM_P，FM_D及原COCO、ignore-union、native10、FP32/NMS路径均保持。',
        '模型历史训练分辨率1280、seed42；100epochs为历史配置记录，best所选epoch仍未知。推理分辨率640/1280，不重新训练。',
        '推理batch=1，square Letterbox，FP32、TF32关闭；one-to-many路径，conf=0.001、NMS IoU=0.7、multi-label、class-aware、max_det=3000、max_nms=30000。',
        '尺寸按原图GT面积分组：tiny<256，small为[256,1024)，medium为[1024,9216)，large≥9216像素²；这里的small不等于COCO惯用的全部area≤1024分组。']
    for ck,name in (('ck01','YOLO26s'),('ck05','YOLOv8s')):
        spec=next(v['spec']['d'] for v in found.values() if v['spec']['d']['checkpoint']==ck)
        text.append(f'- {name} / {ck} SHA256：`{spec["sha256"]}`。')
    text+=['','## RESULT：Normal绝对指标','',
        '| 模型 | 输入 | AP50 | AP50:95与95%区间 | AR50:95 | Tiny AP | Small AP | Medium AP | Large AP |',
        '|---|---:|---:|---|---:|---:|---:|---:|---:|']
    for identity in new:
        item=found[identity];spec=item['spec'];area=next(r for r in item['metrics']['per_area'] if r['size']=='all')
        ci=next(r['ci95'] for r in item['bootstrap']['intervals'] if r['size']=='all' and r['metric']=='map50_95')
        sizes=[metric(item,s,'map50_95') for s in SIZES[1:]]
        text.append(f'| {spec["d"]["checkpoint"]} | {spec["d"]["resolution"]} | {area["map50"]:.6f} | {area["map50_95"]:.6f} [{ci[0]:.6f}, {ci[1]:.6f}] | {area["ar50_95"]:.6f} | '+' | '.join(f'{v:.6f}' for v in sizes)+' |')
    text+=['','以上AP和AR均为0到1原值。AR使用原评估器maxDet=3000。完整尺寸、类别指标见NORMAL_METRICS.csv和NEW_PER_CLASS_METRICS.csv。','',
        '### 同分辨率的额外退化差值','',
        '| 额外退化 | 模型 | 输入 | Normal AP50:95 | 退化AP50:95 | 差值与95%区间 |','|---|---|---:|---:|---:|---|']
    for r in changes:
        if r['size']!='all' or r['metric']!='map50_95':continue
        if r['status']=='NOT_ESTIMABLE':text.append(f'| {r["condition"]} | {r["checkpoint"]} | {r["resolution"]} | — | — | NOT_ESTIMABLE |');continue
        text.append(f'| {r["condition"]} | {r["checkpoint"]} | {r["resolution"]} | {r["normal_value"]:.6f} | {r["degraded_value"]:.6f} | {r["estimate"]:+.6f} [{r["CI95_low"]:+.6f}, {r["CI95_high"]:+.6f}] |')
    text+=['','差值为“退化−Normal”，保留所有正、负和跨0区间。AP50、AR及四个尺寸的差值和配对区间均在WITHIN_RESOLUTION_CONTRASTS.csv；逐类差值仅报告点估计，未计算的逐类Bootstrap不填区间。','',
        '### 分辨率交互','',
        '| 额外退化 | 模型 | Normal的1280−640 | 退化的1280−640 | DID与95%区间 |','|---|---|---:|---:|---|']
    for r in did:
        if r['size']!='all' or r['metric']!='map50_95':continue
        if r['status']=='NOT_ESTIMABLE':text.append(f'| {r["condition"]} | {r["checkpoint"]} | — | — | NOT_ESTIMABLE |');continue
        text.append(f'| {r["condition"]} | {r["checkpoint"]} | {r["normal_resolution_change"]:+.6f} | {r["degraded_resolution_change"]:+.6f} | {r["estimate"]:+.6f} [{r["CI95_low"]:+.6f}, {r["CI95_high"]:+.6f}] |')
    text+=['','DID=[退化1280−退化640]−[Normal1280−Normal640]。负DID表示分辨率收益相对Normal减弱；需要查看该条件自身的1280−640差值，才能判断是否反转。','',
        '### 实际时间与资源','',f'新GPU worker实际墙钟{result.get("worker_wall_seconds",0):.3f}秒，父监督墙钟{supervisor["actual_supervision_seconds"]:.3f}秒，硬上限7200秒。初始化、缓存、推理、评价、原始预测复算及200次Bootstrap包含在worker预算内；CPU预检另计。',
        '每组3次标准热身单独记录，不计入548张保存记录。新Normal逐图实际输入均为[1,3,640,640]或[1,3,1280,1280]、float32。','',
        '| 设置 | 预测流程均值ms | 缓存输入流程均值ms | Torch allocated峰值MiB | Torch reserved峰值MiB |','|---|---:|---:|---:|---:|']
    for identity in new:
        t=found[identity]['timing']
        text.append(f'| {identity.split("__")[1]} | {np.mean([r["predict_ms"] for r in t["samples"]]):.3f} | {t["mean_pipeline_ms"]:.3f} | {t["peak_allocated_bytes"]/1024**2:.1f} | {t["peak_reserved_bytes"]/1024**2:.1f} |')
    text+=['','LATENCY_DIFFERENCES.csv分别报告forward、predict/NMS及缓存输入流程的均值差和图像重采样区间。旧退化与新Normal不是交错运行，时钟、缓存压缩率和系统负载可能影响差值；它不是标准化边缘部署时延结论。','',
        '## CONCLUSION：测量子集与静态预测','',
        '| 固定条件队列 | GAP | COVERED | UNRESOLVED |','|---|---:|---:|---:|']
    for r in cohorts:
        if r['tau']==.1 and r['L_cached_ms']==100:text.append(f'| {r["cohort"]} | {r["GAP_TESTED_SUBSET"]} | {r["COVERED_TESTED_SUBSET"]} | {r["UNRESOLVED"]} |')
    text+=['','主场景沿用Tiny AP50:95≥0.10与缓存输入时延≤100ms。只涉及四个固定设置，不是完整FM_D的经验gap，也不是总体失败比例。原20条件结果保持不变；新增Normal作为目的性参考单独记入21条件队列。','',
        '| 队列 | 静态策略 | 与实测一致率 | 恒预测gap对照 |','|---|---|---:|---:|']
    for r in static:
        if float(r['tau'])==.1 and int(r['L_cached_ms'])==100:
            text.append(f'| {r["cohort"]} | {r["policy"]} | {float(r["accuracy"]):.6f} | {float(r["constant_gap_accuracy"]):.6f} |')
    text+=['','原20条件上的三个静态误报仍是Normal+JPEG Q25、Normal+JPEG Q75、Normal+Gaussian15。新Normal参考有助于量化退化幅度；它不会自动证明模型为何能处理某类退化。静态覆盖定义和实际AP达标定义应分别报告。',
        f'RQ2：主要场景标签为{json.dumps(learning["class_balance"],ensure_ascii=False)}，PASS涉及{learning["PASS_P_groups"]}个P组。配置留组多数类对照见LEARNING_FEASIBILITY.json及折成员表；依然ML_INSUFFICIENT_EVIDENCE，没有拟合或宣称有效失败规则。','',
        '## LIMITATIONS / 后续准入','',
        'CONTROLLED_COMPARISON：本轮Normal与三种额外退化的同checkpoint、同分辨率对照。EXPLORATORY_ASSOCIATION：原复合退化的多因素上下文关系。INDEPENDENT_VALIDATION：本轮为0。',
        '548张val仍有历史选模暴露，不能变成独立测试；原重复图像字节与未知航次相关保留。200次区间为固定权重、固定退化实现的点态图像Bootstrap，不覆盖训练seed或多重比较。Gaussian只检验指定σ=15，不解释为独立物理噪声机制。',
        'test-dev仍DATASET_BLOCKED；UAVDT原图来源、类别/忽略/指标口径及历史暴露仍未准入，未进行新外部推理。RQ4完整训练池CPU适配器证据保留，但初始化、五类增强范围和生产训练预算仍未冻结；本轮训练0次。',
        '本轮再次核对既有索引指向的训练配置、训练曲线、历史评价配置、UAVDT工具包和CPU数据适配器SHA，明细见CONTINUATION_SOURCE_RECHECK.json。已有来源记录保持，但这不是新增独立性证明；未以搜索无命中代替历史使用证据。',
        'RQ1/RQ2/RQ3仍PARTIAL，RQ4仍NOT_EVALUATED。没有把未测设置算成失败，没有使用旧预算，也未扩张到28组复合对照。','',
        '## 原始证据与复现','',f'- 新4组Run：{worker}',f'- 本轮CPU分析：{out}',f'- 原80组来源：{run_path(ANALYSIS)}',
        '- METRIC_PROVENANCE.json连接每组预测、GT、配置、checkpoint及Bootstrap SHA。NEW_RAW_REPLAY.csv保存本轮4组CPU再复算的一致性。',
        '- 所有图的数据与输出SHA见FIGURE_PROVENANCE.json，英文图注见FIGURE_CAPTIONS_EN.md。',
        '- CPU重算入口：'+str(HERE/'normal_reference_analysis.py')+'；在项目根目录使用.venv运行，参数为`--worker-run '+worker.name+'`。创建新分析Run，不重复GPU。',
        '- GPU旧授权票据不可重复使用；未来重放需要新的明确授权、独立Run和有限预算。']
    note=ROOT/'datasets/reportcn'/('vm36_supervisor_reproduction_Normal参考对照_'+out.name.split('_')[-1]+'.md')
    write_atomic(note,'\n'.join(text)+'\n',out);save(out/'REPORT_REFERENCE.json',source(note),out)


def execute(worker):
    from development_guard import bounded_run
    if load(worker/'run.json')['status']=='RUNNING':raise ValueError('GPU worker has not stopped.')
    verify_hashes(worker,('worker_result.json','supervisor.json','EXECUTION_PROTOCOL.json','NORMAL_PIXEL_PROOFS.json',
        'PREFLIGHT.json','inputs.json','executed_source.py','EVIDENCE_TREE.json'))
    out,_=new_run({'output_root':str(worker.parent)},EXPERIMENT,'normal4_analysis');shutil.copyfile(Path(__file__),out/'executed_source.py')
    save(out/'inputs.json',{'worker':source(worker/'SHA256SUMS.txt'),'source':source(Path(__file__))},out);started=time.perf_counter()
    try:
        with (out/'worker.log').open('x',encoding='utf-8') as stream:
            code=bounded_run([sys.executable,'-B',str(Path(__file__)),'--analysis-worker',out.name,'--input-run',worker.name],stream,1800,
                dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
        if code:raise RuntimeError('CPU analysis failed; see preserved log.')
    except BaseException:save(out/'failure.json',{'traceback':traceback.format_exc(),'automatic_retry':False},out)
    result=load(out/'analysis_result.json') if (out/'analysis_result.json').exists() else {'status':'PARTIAL_ANALYSIS_ERROR'}
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'NORMAL_REFERENCE_RESULTS',**result,
        'CPU_supervision_seconds':time.perf_counter()-started},out,replace=True)
    write_atomic(out/'diary.md','# Normal-reference contrasts\n\nReused the accepted80 responses, replayed only new Normal predictions, and completed specified one-factor contrasts with shared image-bootstrap draws. Preserved incomplete cells and kept cross-run latency descriptive. No new training or independent test.\n',out)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>Normal-reference matched contrasts</h1><p>'+result['status']+'</p>'+''.join('<p><a href="'+n+'">'+n+'</a></p>' for n in ('REPORT_REFERENCE.json','NORMAL_METRICS.csv','WITHIN_RESOLUTION_CONTRASTS.csv','RESOLUTION_DID.csv','LATENCY_DIFFERENCES.csv','COHORT_GAP_COUNTS.csv','STATIC_AGREEMENT.csv','LEARNING_FEASIBILITY.json','METRIC_PROVENANCE.json'))+''.join('<img style="max-width:100%" src="'+n+'">' for n in ('controlled_degradation_effects.png','resolution_degradation_interactions.png','size_dependent_degradation_effects.png') if (out/n).exists()),out)
    seal(out)
    from main.visualization import build_run_index
    build_run_index(out.parent,EXPERIMENT)
    latest=load(out.parent/'latest.json');latest.update(latest_normal_reference_results_RunID=out.name,normal_reference_status=result['status'],
        normal_reference_new_pairs=result.get('new_pairs',0),normal_reference_combined_pairs=result.get('combined_pairs',80),
        status=result['status'],combined_measured_pairs=result.get('combined_pairs',80),formal_measured_pairs=0)
    save(out.parent/'latest.json',latest,out,replace=True)
    landing=out.parent/'index.html';html=landing.read_text(encoding='utf-8')
    link='<p><a href="'+out.name+'/index.html">Latest Normal-reference scientific results</a> · '+str(result.get('combined_pairs',80))+' accepted development pairs · 0 independent pairs</p>'
    write_atomic(landing,html.replace('<div class="cards">',link+'<div class="cards">',1),out,replace=True)
    print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--worker-run');g.add_argument('--analysis-worker');p.add_argument('--input-run');a=p.parse_args()
    if a.worker_run:execute(run_path(a.worker_run))
    elif a.input_run:analyze(run_path(a.analysis_worker),run_path(a.input_run))
    else:p.error('The analysis worker requires its source RunID.')

