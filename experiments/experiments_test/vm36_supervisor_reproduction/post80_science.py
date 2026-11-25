"""Analyze accepted responses and freeze missing legal contrasts; no GPU entry."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from protocol import ROOT,HERE,EXPERIMENT,FMP,FMD,PRIOR,load,save,source,sha256,write_atomic
from common import seal,verify_hashes
from historical_review import csv_file,table
from main.pipeline import new_run
from full_development import PREP
from full20_completion import read_pair
from full20_results import resolution_effects,finite20_marginals,context_effects,outcomes
from post80_replay import ANALYSIS,ACCEPTANCE,run_path
from alignment import parameters,require_runtime_bindings

SEVERE='p_8d98aefd645e8b60'
DATA='vm36_supervisor_reproduction_training_data_cpu_v2_20261110T110058_057fb01c'
READINESS='vm36_supervisor_reproduction_post26_admission_readiness_20261110T111123_2a3e725a'


def check_numeric_table(path,expected,keys,fields):
    actual=table(path);lookup={tuple(str(r[k]) for k in keys):r for r in actual}
    if len(lookup)!=len(actual) or len(actual)!=len(expected):raise ValueError('Table identity/count differs: '+path.name)
    for row in expected:
        saved=lookup[tuple(str(row[k]) for k in keys)]
        for field in fields:
            value=row[field];other=saved[field]
            if value is None:
                if other!='':raise ValueError('Missing value was filled.')
            elif not math.isclose(float(value),float(other),rel_tol=0,abs_tol=1e-14):
                raise ValueError('Numeric extraction differs: '+path.name+' '+field)


def existing_evidence(out):
    a=run_path(ANALYSIS);verify_hashes(a,('CONDITION_METRICS.csv','RESOLUTION_EFFECTS.csv','FINITE20_MARGINALS.csv',
        'STATIC_AGREEMENT.csv','MEASURED_DESIGN.json','EVIDENCE_SOURCES.json','CONFIDENCE_INTERVALS.csv'))
    found={};gt_sha=None;pixel_sets={};draw_identity=None
    for ref in load(a/'EVIDENCE_SOURCES.json')['conditions']:
        item=read_pair(Path(ref['folder']));spec=item['spec'];b=item['bootstrap'];identity=spec['id']
        if identity in found:raise ValueError('Repeated accepted measurement.')
        if gt_sha and item['checks']['ground_truth.json']!=gt_sha:raise ValueError('GT differs.')
        gt_sha=item['checks']['ground_truth.json']
        d=(b['draws']['seed'],b['draws']['samples'],tuple(b['draws']['image_ids']),b['draws']['sampling'])
        if draw_identity and d!=draw_identity:raise ValueError('Bootstrap pairing differs.')
        draw_identity=d
        for interval in b['intervals']:
            values=[v for v in b['draws']['values'][interval['size']][interval['metric']] if v is not None]
            if len(values)!=interval['n_valid'] or values and not np.allclose(np.quantile(values,[.025,.975]),interval['ci95'],rtol=0,atol=1e-14):
                raise ValueError('Saved CI differs from saved COCO resamples.')
        manifest=[json.loads(s) for s in (Path(item['folder'])/'image_manifest.jsonl').read_text().splitlines()]
        if [r['image_id'] for r in manifest]!=list(d[2]):raise ValueError('Image identities differ.')
        signature=hashlib.sha256(json.dumps([(r['image_id'],r['pixel_sha256']) for r in manifest]).encode()).hexdigest()
        pid=spec['p']['id']
        if pid in pixel_sets and pixel_sets[pid]!=signature:raise ValueError('Different pixels for one P.')
        pixel_sets[pid]=signature;found[identity]=item
    if len(found)!=80 or len(pixel_sets)!=20:raise ValueError('Incomplete accepted grid.')
    checks=[]
    for ck in ('ck01','ck05'):
        spec=next(v['spec']['d'] for v in found.values() if v['spec']['d']['checkpoint']==ck)
        if sha256(Path(spec['path']))!=spec['sha256']:raise ValueError('Checkpoint changed.')
        checks.append({'checkpoint':ck,'identity':source(Path(spec['path'])),
            'original_CPU_health':next(r for r in table(PREP/'CHECKPOINT_ADMISSION.csv') if r['checkpoint']==ck)})
    comparisons=[('RESOLUTION_EFFECTS.csv',resolution_effects(found),('p_id','checkpoint','size','metric')),
        ('FINITE20_MARGINALS.csv',finite20_marginals(found),('contrast','size','metric'))]
    effects,_=context_effects(found);comparisons.append(('CONTEXT_INTERACTIONS.csv',effects,('p_id','checkpoint','size','metric')))
    for name,rows,keys in comparisons:
        check_numeric_table(a/name,rows,keys,('estimate','CI95_low','CI95_high'))
    gt=load(Path(next(iter(found.values()))['folder'])/'ground_truth.json')
    first_manifest=manifest
    byte_shas=[r['source']['image_sha256'] if 'image_sha256' in r['source'] else r['source'].get('image_SHA256') for r in first_manifest]
    save(out/'EVIDENCE_RECONSTRUCTION.json',{'status':'RECONSTRUCTED_FROM_SAVED_METRICS_AND_COCO_DRAWS',
        'accepted_pairs':80,'images_per_pair':548,'P_configurations':20,'detector_settings':4,'GT_SHA256':gt_sha,
        'bootstrap_seed':d[0],'bootstrap_draws':d[1],'image_ids':list(d[2]),'pixel_signatures':pixel_sets,
        'checkpoint_checks':checks,'COCO_metric_replay':'SEPARATE_CPU_REPLAY_RUN_REQUIRED',
        'source_analysis':source(a/'SHA256SUMS.txt'),'image_byte_identity_source':source(Path(next(iter(found.values()))['folder'])/'image_manifest.jsonl')},out)
    csv_file(out/'RESOLUTION_EFFECTS.csv',resolution_effects(found));csv_file(out/'FINITE20_MARGINALS.csv',finite20_marginals(found))
    csv_file(out/'CONTEXT_INTERACTIONS.csv',effects)
    return found,gt


def legal_design(found):
    from sampling import models
    pm,_,_=models();recipes={v['spec']['p']['id']:v['spec']['p'] for v in found.values()}
    severe=set(recipes[SEVERE]['leaves']);candidates=[('NORMAL_REFERENCE',{'Normal'},True,'Shared missing reference for Gaussian and additional JPEG comparisons')]
    for feature in ('Defocus_r6','Downscale_x1_5','Gauss_s30'):
        candidates.append(('REMOVE_'+feature,severe-{feature},True,'One degradation-family intervention within the severe anchor'))
    candidates += [('SHOT_HIGH_TO_LOW',severe-{'Shot_high'}|{'Shot_low'},True,'Photon level20 to100; LowLight still requires ShotNoise'),
        ('JPEG25_TO75',severe-{'JPEG_q25'}|{'JPEG_q75'},True,'Additional JPEG quality25 to75'),
        ('LOWLIGHT_TO_UNDEREXPOSED',severe-{'LowLight'}|{'Underexposed_strong'},True,'One illumination-level change with Shot_high retained'),
        ('NORMAL_AND_SHOT_LOW',severe-{'LowLight','Shot_high'}|{'Normal','Shot_low'},True,'Minimal legal two-family alternative; not an illumination-only effect'),
        ('ILLEGAL_REMOVE_SHOT',severe-{'Shot_high'},False,'LowLight => ShotNoise'),
        ('ILLEGAL_LOWLIGHT_TO_NORMAL',severe-{'LowLight'}|{'Normal'},False,'Shot_high => LowLight | Underexposed_strong')]
    checks=[];new={};contrasts=[]
    for name,leaves,legal,reason in candidates:
        selected=sorted(leaves)
        if pm.valid_leaves(selected)!=legal:raise ValueError('Contrast legality changed.')
        checks.append({'name':name,'leaves':selected,'expected_legal':legal,'reason':reason})
        if not legal:continue
        pid=pm.configuration_id(selected);spec={'id':pid,'leaves':selected,'attributes':parameters(FMP,selected)}
        require_runtime_bindings(spec['attributes'],load(PREP/'IMPLEMENTATION_BINDINGS_PROPOSAL.json'))
        if pid not in recipes:new[pid]=spec
        if name=='NORMAL_REFERENCE':
            for reference in ('p_9031c68f63a468d1','p_59915f60d81c6f17','p_5a06deca286f53b2'):
                contrasts.append({'name':name+'__'+reference,'existing_reference':reference,'new_P':pid,
                    'changed_families':'gaussian' if 'Gauss_s15' in recipes[reference]['leaves'] else 'compression',
                    'estimand':'Within-checkpoint Resolution DID: deltaR(existing degraded P) - deltaR(Normal-only).',
                    'claim':'CONTROLLED_COMPARISON after measurement; currently NOT_EVALUATED'})
        else:contrasts.append({'name':name,'existing_reference':SEVERE,'new_P':pid,
            'changed_families':'illumination+shot' if name=='NORMAL_AND_SHOT_LOW' else name,
            'estimand':'Within-checkpoint Resolution DID: deltaR(modified P) - deltaR(severe reference).',
            'claim':'Joint-context contrast, not a single-factor effect' if name=='NORMAL_AND_SHOT_LOW' else 'One-family computational intervention; conditional on the fixed operator order'})
    return {'version':'VM36_POST80_LEGAL_CONTRASTS_V1','candidate_checks':checks,'new_P_configurations':list(new.values()),
        'contrasts':contrasts,'selection':'Feature structure and missing matched references only; no AP-valued selection function.',
        'study_scope':'EXPLORATORY: design developed after the80-cell responses were seen.',
        'primary_minimal_plan':{'name':'NORMAL_REFERENCE','new_P':1,'new_pairs':4,'images_per_pair':548,
            'purpose':'One shared baseline identifies Gaussian15 presence and additional JPEG25/75 contrasts on a Normal background for both checkpoints.'},
        'compound_anchor_alternative':{'new_P':7,'new_pairs':28,'purpose':'Six one-family changes and one legal joint change around the previously studied severe context.'},
        'GPU_status':'RESOURCE_BLOCKED','previous_allocation':'The54-pair scope was completed. Unused clock time does not authorize new configurations.',
        'no_execution_entry_in_this_stage':True,'same_source_population':548,'precision':'FP32','bootstrap_draws':200,
        'processing_order':['atmosphere','illumination','blur','rolling_shutter','noise','downscale','compression']}


def render_checks(out,design,found):
    from development_v1 import generator
    from archives import OriginalZIP
    plan=load(run_path('vm36_supervisor_reproduction_full20_remaining51_20261110T120947_b5f3997d')/'EXECUTION_PROTOCOL.json')
    renderer=generator();rows=[]
    if list(renderer.ORDER)!=plan['stage_order']:raise ValueError('Renderer order changed.')
    fixture=np.arange(79*103*3,dtype=np.uint8).reshape(79,103,3)
    for spec in design['new_P_configurations']:
        first,trace=renderer.generate(fixture,spec['leaves'],spec['attributes'],'POST80_SYNTHETIC',plan['seed'])
        again,trace2=renderer.generate(fixture,spec['leaves'],spec['attributes'],'POST80_SYNTHETIC',plan['seed'])
        if not np.array_equal(first,again) or trace!=trace2 or first.shape!=fixture.shape:raise ValueError('Renderer fixture differs.')
        rows.append({'P':spec['id'],'input':'SYNTHETIC_ONLY','shape':list(first.shape),'pixel_SHA256':hashlib.sha256(first.tobytes()).hexdigest(),'deterministic':True})
    archive=OriginalZIP(Path(plan['ZIP']['path']),548)
    normal=next(s for s in design['new_P_configurations'] if s['leaves']==['Normal'])
    try:
        for i in range(6):
            image,record,proof=archive.record(i)
            first,trace=renderer.generate(image,normal['leaves'],normal['attributes'],record['stem'],plan['seed'])
            if not np.array_equal(image,first):raise ValueError('Normal reference changes decoded pixels.')
            rows.append({'P':normal['id'],'input':record['stem'],'shape':list(first.shape),
                'pixel_SHA256':hashlib.sha256(first.tobytes()).hexdigest(),'source':proof,'unchanged_original_pixels':True})
    finally:archive.close()
    save(out/'RENDERER_CHECKS.json',{'status':'CPU_RENDER_CHECKS_PASSED','records':rows,
        'original_GT_unchanged':True,'new_full_dataset_caches':0,'new_GPU_inferences':0,
        'geometry':'All admitted effects retain original pixel coordinates; downscale returns to the source dimensions.',
        'JPEG_None':'No additional recompression of decoded JPEG, not lossless original acquisition.'},out)


def classification_scores(true,pred):
    cells=Counter(zip(true,pred));tp=cells['FAIL','FAIL'];fn=cells['FAIL','PASS'];fp=cells['PASS','FAIL'];tn=cells['PASS','PASS']
    precision=tp/(tp+fp) if tp+fp else None;recall=tp/(tp+fn) if tp+fn else None
    specificity=tn/(tn+fp) if tn+fp else None
    f1fail=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
    f1pass=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.
    return {'TP_failure':tp,'FN_failure':fn,'FP_failure':fp,'TN_failure':tn,'failure_precision':precision,
        'failure_recall':recall,'balanced_accuracy':(recall+specificity)/2 if recall is not None and specificity is not None else None,
        'macro_F1':(f1fail+f1pass)/2,'accuracy':(tp+tn)/len(true),'decisive_samples':len(true)}


def learning_and_counterexamples(out,found):
    from sampling import models
    pm,_,_=models();seed=next(iter(found.values()))['bootstrap']['draws']['seed']
    pair_states,_,gaps=outcomes(found,seed)
    primary=[r for r in pair_states if r['tau']==.1 and r['L_cached_ms']==100]
    pids=sorted({r['p_id'] for r in primary});features=[];folds=[];predictions=[]
    specs={v['spec']['id']:v['spec'] for v in found.values()}
    for r in primary:
        spec=specs[r['pair_id']]
        features.append({'pair_id':r['pair_id'],'P_group':r['p_id'],'label':r['state'],
            'checkpoint':spec['d']['checkpoint'],'Infer1280':int(spec['d']['resolution']==1280),
            **{leaf:int(leaf in spec['p']['leaves']) for leaf in sorted(pm.ALL_LEAVES)}})
    for pid in pids:
        train=[r for r in primary if r['p_id']!=pid and r['state'] in ('PASS','FAIL')]
        test=[r for r in primary if r['p_id']==pid and r['state'] in ('PASS','FAIL')]
        majority=Counter(r['state'] for r in train).most_common(1)[0][0]
        folds.append({'heldout_P':pid,'train_groups':19,'heldout_groups':1,'train_PASS':sum(r['state']=='PASS' for r in train),
            'train_FAIL':sum(r['state']=='FAIL' for r in train),'test_PASS':sum(r['state']=='PASS' for r in test),
            'test_FAIL':sum(r['state']=='FAIL' for r in test),'majority_prediction':majority,
            'train_P_IDs':'|'.join(p for p in pids if p!=pid),'validation_P_IDs':pid})
        predictions.extend({'pair_id':r['pair_id'],'heldout_P':pid,'truth':r['state'],'prediction':majority} for r in test)
    scores=classification_scores([r['truth'] for r in predictions],[r['prediction'] for r in predictions])
    counts=Counter(r['state'] for r in primary);pass_groups=len({r['p_id'] for r in primary if r['state']=='PASS'})
    csv_file(out/'FEATURE_MATRIX.csv',features);csv_file(out/'GROUP_FOLDS.csv',folds);csv_file(out/'MAJORITY_HELDOUT_PREDICTIONS.csv',predictions)
    save(out/'LEARNING_DIAGNOSTIC.json',{'status':'ML_INSUFFICIENT_EVIDENCE','checkpoint_settings':4,'unique_P_groups':20,
        'shared_validation_images':548,'independent_scene_count':'UNKNOWN','pair_class_balance':dict(counts),
        'PASS_P_groups':pass_groups,'majority_grouped_development_baseline':scores,'interpretable_classifier_fitted':False,
        'model_uncertainty':'NOT_ESTIMATED; only three distinct P groups contain passing assignments.',
        'sample_support':'A missed passing P changes positive-group recall by1/3. This is insufficient for a stable failure-rule claim.',
        'selection':'No response-trained model or hyperparameter search; folds use P identity and exclude uncertain labels.',
        'protocol_A':'Configuration-held-out development diagnostics only; all four settings of each P share its fold.',
        'protocol_B':'Independent image/scene validation not established; original predictions cannot retrospectively undo selection exposure.',
        'protocol_C':'External evaluation not admitted; test-dev history and UAVDT semantics remain unresolved.'},out)
    policies=table(PREP/'STATIC_COVERAGE_REFERENCE.csv');rows=[]
    covered={r['p_id'] for r in gaps if r['tau']==.1 and r['L_cached_ms']==100 and r['subset_state']=='COVERED_TESTED_SUBSET'}
    for r in primary:
        if r['p_id'] not in covered:continue
        item=found[r['pair_id']];spec=item['spec'];tiny=next(m for m in item['metrics']['per_area'] if m['size']=='tiny')
        for policy in ('exact_identity','conservative','supervisor_proxy'):
            proxy=next(p for p in policies if p['policy']==policy and p['checkpoint']==spec['d']['checkpoint'])
            static_covered=set(spec['p']['leaves']).issubset(set(proxy['covered_leaves'].split('|')))
            rows.append({**r,'policy':policy,'static_predicted_gap':not static_covered,
                'features':'|'.join(spec['p']['leaves']),'numeric_attributes':json.dumps(spec['p']['attributes'],sort_keys=True),
                'Tiny_AP50_95':tiny['map50_95'],'mean_cached_pipeline_ms':item['timing']['mean_pipeline_ms'],
                'raw_source_folder':item['folder'],'prediction_SHA256':item['checks']['predictions.jsonl.gz'],
                'GT_SHA256':item['checks']['ground_truth.json'],'metrics_SHA256':item['checks']['metrics.json'],
                'reason_for_static_mismatch':'Static producibility is not a necessary condition for measured detection success; causal mechanism not isolated.',
                'historical_Albu_activation':'UNKNOWN'})
    csv_file(out/'STATIC_FALSE_POSITIVE_SETTINGS.csv',rows)
    save(out/'STATIC_INTERPRETATION.json',{'conditions':len(covered),'passing_assignments':sum(r['state']=='PASS' for r in primary if r['p_id'] in covered),
        'all_three_policies_predict_gap_for_all20':True,'static_agreement':source(run_path(ANALYSIS)/'STATIC_AGREEMENT.csv'),
        'training_arguments':{ck:source(ROOT/f'models/checkpoints/{folder}/args.yaml') for ck,folder in
            (('ck01','ck01_yolo26s_ft1280'),('ck05','ck05_yolov8s_ft1280'))},
        'source_interpretation':['An optional augmentation being absent is not proof that a detector cannot recognize affected inputs.',
            'JPEG denotes additional encoding of already JPEG-encoded source imagery; do not call its absence lossless acquisition.',
            'Saved augmentation arguments and later installed source do not prove historical runtime activation or all pretraining exposures.',
            'No invariance, denoising or physical mechanism is inferred from these counterexamples.'],
        'universal_rule_static_gap_implies_failure':'REFUTED_ON_THESE_THREE_DEVELOPMENT_CONDITIONS'},out)
    return primary,gaps


def uniform_planning(out,found):
    from sampling import models
    pm,_,prior=models();blocked=set(load(PREP/'IMPLEMENTATION_BINDINGS_PROPOSAL.json')['proposed_values_from_historical_implementation'])
    population=[p for p in prior.enumerate_configs(pm) if not blocked.intersection(p)]
    if len(population)!=1872:raise ValueError('Executable population changed.')
    costs=[v['timing']['wall_seconds'] for v in found.values()];maximum=max(costs);rows=[]
    for n_pop,scope,admitted in ((1872,'Parameter-bound FM_P x four fixed settings',True),(46368,'Full logical FM_P; render bindings incomplete',False)):
        for error in (.15,.10,.05):
            approximate=math.ceil(n_pop*1.959963984540054**2*.25/(error**2*(n_pop-1)+1.959963984540054**2*.25))
            sufficient=min(n_pop,math.ceil(n_pop/(1+4*.05*(n_pop-1)*error**2)))
            rows.append({'population':n_pop,'scope':scope,'sampling':'SRS without replacement of P; measure all four fixed D',
                'desired_absolute_error':error,'confidence':.95,'normal_FPC_planning_n':approximate,
                'Chebyshev_sufficient_n':sufficient,'Chebyshev_basis':'Var(mean)<=0.25*(N-n)/(n*(N-1)); P(abs(error)>=e)<=Var/e^2.',
                'normal_planned_pairs':4*approximate,'normal_pair_cost_screen_seconds':4*approximate*maximum*1.25 if admitted else None,
                'execution':'NOT_SELECTED_NOT_RUN','admitted_projection':admitted})
    csv_file(out/'UNIFORM_SAMPLE_SIZE_SCENARIOS.csv',rows)
    save(out/'POPULATION_ESTIMATION_PLAN.json',{'status':'NOT_EXECUTED','estimand':'Fixed confidence-rule gap prevalence on the1872-P/four-setting projection; not full FM_D or field failure prevalence.',
        'estimator':'SRS finite-population sample mean, with unresolved P reported as separate lower/upper classification bounds.',
        'normal_approximation':'Planning only, not an exact coverage guarantee. Conservative sufficient counts are derived from finite-population variance and Chebyshev.',
        'oracle_uncertainty':'Sampling uncertainty and pointwise image-bootstrap uncertainty are separate; no joint95% guarantee is implied.',
        'existing20_reuse':'A selected historical P can be reused only if it independently falls in the prospectively frozen uniform draw; never force all20 into it.',
        'no_probability_sample_drawn':True,'full_FM_D_gap':'NOT_IDENTIFIED by four fixed settings.'},out)


def prepare(request):
    started=time.perf_counter();out,_=new_run({'output_root':str(run_path(ANALYSIS).parent)},EXPERIMENT,'post80_science')
    shutil.copyfile(Path(__file__),out/'executed_source.py')
    found,gt=existing_evidence(out);design=legal_design(found);save(out/'MATCHED_CONTRAST_PREREGISTRATION.json',design,out)
    command=[str(ROOT/'.venv_flamapy/Scripts/python.exe'),'-B',str(HERE/'post80_legal_sat.py'),
        '--model',str(FMP),'--design',str(out/'MATCHED_CONTRAST_PREREGISTRATION.json')]
    result=subprocess.run(command,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
    write_atomic(out/'FlamaPy.log',result.stdout+result.stderr,out)
    if result.returncode:raise RuntimeError('FlamaPy legality check failed.')
    sat=json.loads(result.stdout);save(out/'LEGAL_CONTRAST_SAT.json',sat,out);render_checks(out,design,found)
    primary,gaps=learning_and_counterexamples(out,found);uniform_planning(out,found)
    pair_costs=[v['timing']['wall_seconds'] for v in found.values()];cache_costs=[]
    for v in found.values():
        p=Path(v['folder'])/'CACHE_COST.json'
        if p.exists():cache_costs.append(load(p)['seconds'])
    cache_allowance=max(cache_costs) if cache_costs else None
    save(out/'GPU_RESOURCE_GATE.json',{'status':'RESOURCE_BLOCKED','new_inference_pairs':0,'new_training_runs':0,
        'scope_reason':'Previous author record covered only the54 now-completed missing points. A different P requires a new finite allocation.',
        'minimum_new_P':design['primary_minimal_plan']['new_P'],'minimum_new_pairs':design['primary_minimal_plan']['new_pairs'],
        'reference_complete_pair_max_seconds':max(pair_costs),'reference_cache_max_seconds':cache_allowance,
        'conservative_minimum_batch_screen_seconds':4*max(pair_costs)*1.25+(cache_allowance or 0)*1.25+300,
        'estimate_limit':'Screening from measured historical complete points plus a cache allowance; actual first new complete pair must gate continuation.',
        'no_existing_budget_reused':True},out)
    questions=[
        ('A1','Within-checkpoint inference-resolution contrasts on measured P','CONTROLLED_COMPARISON','RESOLUTION_EFFECTS.csv','Fixed pixels, GT and checkpoint; no training-seed or physical causal claim.'),
        ('A2','Isolated JPEG25-versus75 interaction','CONTROLLED_COMPARISON','CONTEXT_INTERACTIONS.csv','The original eight matched corners support the JPEG endpoint contrast.'),
        ('A3','Attribute compound differences to individual components','UNSUPPORTED_CLAIM','MATCHED_CONTRAST_PREREGISTRATION.json','Missing single-family P counterfactuals; legal design frozen but unmeasured.'),
        ('B1','Configuration-held-out majority baseline','DESCRIPTIVE_ASSOCIATION','LEARNING_DIAGNOSTIC.json','Not an independently validated rule; minority support is three P groups.'),
        ('B2','Independent failure prediction','UNSUPPORTED_CLAIM','LEARNING_DIAGNOSTIC.json','Image/scene independence not established; no classifier fitted.'),
        ('C1','Static-gap false positives','EXPLORATORY_FINDING','STATIC_FALSE_POSITIVE_SETTINGS.csv','Measured counterexamples, not identified physical mechanisms.'),
        ('C2','Full-space failure prevalence from17/20','UNSUPPORTED_CLAIM','POPULATION_ESTIMATION_PLAN.json','Purposeful configurations do not provide uniform inclusion probabilities.'),
        ('D1','Targeted training shrinks the gap','UNSUPPORTED_CLAIM','RQ4_READINESS_UPDATE.json','No targeted-augmentation training or matched-seed efficacy experiment.')]
    csv_file(out/'QUESTION_TO_EVIDENCE.csv',[{'question':q,'estimand':e,'category':c,'evidence_file':f,'limitation':l} for q,e,c,f,l in questions])
    save(out/'inputs.json',{'request':source(request),'source':source(Path(__file__)),'source_analysis':source(run_path(ANALYSIS)/'SHA256SUMS.txt'),
        'original_pdf_extraction':source(PRIOR/'pdf_pages.json'),'FMP':source(FMP),'FMD':source(FMD)},out)
    save(out/'PREPARATION_RESULT.json',{'status':'CPU_DESIGN_COMPLETE_GPU_RESOURCE_BLOCKED','seconds':time.perf_counter()-started},out)
    print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',type=Path,required=True);a=p.parse_args();prepare(a.prepare)
