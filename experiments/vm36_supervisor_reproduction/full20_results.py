"""Regenerate matched-context contrasts and finite20 subset outcomes on CPU."""
import argparse
from collections import Counter
from itertools import combinations
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
from protocol import ROOT,HERE,EXPERIMENT,load,save,source,sha256,write_atomic
from common import seal,verify_hashes
from historical_review import table,csv_file
from full_development import PREP
from full_analysis import paired_effect,tested_subset_state
from full20_completion import path_for,initial_sources,read_pair
from matched_followup import ANALYSIS_ID,SEVERE_ID
from main.pipeline import new_run


def latency_interval_state(ap_ci,latency_ci,tau,limit):
    if ap_ci is None or latency_ci is None or any(v is None for v in (*ap_ci,*latency_ci)):
        return 'NOT_EVALUATED'
    if ap_ci[0]>ap_ci[1] or latency_ci[0]>latency_ci[1]:raise ValueError('Invalid interval order.')
    if ap_ci[1]<tau or latency_ci[0]>limit:return 'FAIL'
    if ap_ci[0]>=tau and latency_ci[1]<=limit:return 'PASS'
    return 'UNCERTAIN'


def outcomes(found,seed):
    from gap_analysis import pair_outcome
    indices=np.random.default_rng(seed).integers(0,548,(200,548));pairs=[];gaps=[]
    pids=sorted({v['spec']['p']['id'] for v in found.values()});dids=sorted({v['spec']['d']['id'] for v in found.values()})
    if len(pids)!=20 or len(dids)!=4:raise ValueError('Subset definition changed.')
    for item in found.values():
        b=item['bootstrap'];t=item['timing'];s=item['spec']
        ci=next(r['ci95'] for r in b['intervals'] if r['size']=='tiny' and r['metric']=='map50_95')
        latency_ci=list(map(float,np.quantile(np.asarray([r['pipeline_ms'] for r in t['samples']])[indices].mean(axis=1),[.025,.975])))
        latency=latency_ci[1]
        for tau in (.05,.10,.15):
            for limit in (50,100):
                pairs.append({'pair_id':s['id'],'p_id':s['p']['id'],'d_id':s['d']['id'],'tau':tau,'L_cached_ms':limit,
                    'state':pair_outcome(ci,latency,tau,limit),'Tiny_AP_CI_low':ci[0],'Tiny_AP_CI_high':ci[1],
                    'mean_cached_latency_lower95_ms':latency_ci[0],'mean_cached_latency_upper95_ms':latency,
                    'latency_interval_state':latency_interval_state(ci,latency_ci,tau,limit),
                    'quality_only_state':'FAIL' if ci[1]<tau else 'PASS' if ci[0]>=tau else 'UNCERTAIN',
                    'source_metrics_SHA256':item['checks']['metrics.json']})
    complete=[]
    for p in pids:
        for tau in (.05,.10,.15):
            for limit in (50,100):
                states=[];interval_states=[];quality_states=[]
                for d in dids:
                    row=next((r for r in pairs if r['p_id']==p and r['d_id']==d and r['tau']==tau and r['L_cached_ms']==limit),None)
                    value=row['state'] if row else 'NOT_EVALUATED';states.append(value)
                    alternate=row['latency_interval_state'] if row else 'NOT_EVALUATED';interval_states.append(alternate)
                    quality=row['quality_only_state'] if row else 'NOT_EVALUATED';quality_states.append(quality)
                    complete.append({'p_id':p,'d_id':d,'tau':tau,'L_cached_ms':limit,'state':value,'latency_interval_state':alternate,'quality_only_state':quality})
                gaps.append({'p_id':p,'tau':tau,'L_cached_ms':limit,'subset_state':tested_subset_state(states,4),
                    'latency_interval_subset_state':tested_subset_state(interval_states,4),
                    'quality_only_subset_state':tested_subset_state(quality_states,4),
                    'measured_settings':sum(s!='NOT_EVALUATED' for s in states),'declared_settings':4,
                    'gap_scope':'These20 purposive P states and four fixed settings only.'})
    return pairs,complete,gaps


def context_effects(found):
    indexed={(v['spec']['p']['id'],v['spec']['d']['id']):v for v in found.values()}
    recipes={v['spec']['p']['id']:v['spec']['p'] for v in found.values()}
    reference=next(p for p,r in recipes.items() if set(r['leaves'])=={'Normal','JPEG_q25'})
    rows=[];missing=[]
    for p in sorted(recipes):
        if p==reference:continue
        definitions={}
        for ck in ('ck01','ck05'):
            keys=[(p,ck+'_Infer1280'),(p,ck+'_Infer640'),(reference,ck+'_Infer1280'),(reference,ck+'_Infer640')]
            if not all(k in indexed for k in keys):
                missing.append({'p_id':p,'checkpoint':ck,'contrast':'Resolution x context bundle','status':'NOT_ESTIMABLE',
                    'missing_keys':'|'.join(a+'__'+b for a,b in keys if (a,b) not in indexed)});continue
            definitions[ck]=[(sign,indexed[k]['metrics'],indexed[k]['bootstrap']) for sign,k in zip((1,-1,-1,1),keys)]
        if len(definitions)==2:definitions['ck01-minus-ck05']=definitions['ck01']+[(-s,m,b) for s,m,b in definitions['ck05']]
        for ck,terms in definitions.items():
            for size in ('all','tiny','small','medium','large'):
                for metric in ('map50_95','map50','ar50_95'):
                    rows.append({'p_id':p,'features':'|'.join(recipes[p]['leaves']),'reference_p_id':reference,
                        'checkpoint':ck,'size':size,'metric':metric,**paired_effect(terms,size,metric),
                        'context_frozen_before_missing_corner_measurement':p==SEVERE_ID,
                        'scope':'Fixed-checkpoint, pointwise paired-image development contrast; other features form a context bundle, not isolated causal components.'})
    return rows,missing


def resolution_effects(found):
    indexed={(v['spec']['p']['id'],v['spec']['d']['id']):v for v in found.values()}
    recipes={v['spec']['p']['id']:v['spec']['p'] for v in found.values()};rows=[]
    for p,recipe in sorted(recipes.items()):
        for ck in ('ck01','ck05'):
            keys=[(p,ck+'_Infer1280'),(p,ck+'_Infer640')]
            complete=all(k in indexed for k in keys)
            for size in ('all','tiny','small','medium','large'):
                for metric in ('map50_95','map50','ar50_95'):
                    effect=paired_effect([(sign,indexed[k]['metrics'],indexed[k]['bootstrap']) for sign,k in zip((1,-1),keys)],size,metric) if complete else {'estimate':None,'CI95_low':None,'CI95_high':None,'images':548,'draws':200}
                    rows.append({'p_id':p,'features':'|'.join(recipe['leaves']),'checkpoint':ck,'size':size,'metric':metric,
                        'contrast':'1280-minus-640 within the same context and checkpoint',**effect,
                        'status':'ESTIMABLE' if complete else 'NOT_ESTIMABLE',
                        'scope':'Pointwise paired-image development contrast; distinct from a difference-in-differences against another context.'})
    return rows


def static_comparison(found,gaps):
    policies=table(PREP/'STATIC_COVERAGE_REFERENCE.csv');names=sorted({r['policy'] for r in policies})
    checkpoints={v['spec']['d']['checkpoint'] for v in found.values()}
    recipes={v['spec']['p']['id']:v['spec']['p'] for v in found.values()};rows=[];summary=[]
    for policy in names:
        for row in gaps:
            covered=any(set(recipes[row['p_id']]['leaves']).issubset(set(p['covered_leaves'].split('|'))) for p in policies if p['policy']==policy and p['checkpoint'] in checkpoints)
            rows.append({**row,'policy':policy,'static_covered':covered,'static_predicted_gap':not covered,
                'comparison_scope':'Finite selected20, four-setting subset; not full-space predictive validation.'})
        for tau in (.05,.10,.15):
            for limit in (50,100):
                selected=[r for r in rows if r['policy']==policy and r['tau']==tau and r['L_cached_ms']==limit]
                decisive=[r for r in selected if r['subset_state']!='UNRESOLVED']
                tp=sum(r['static_predicted_gap'] and r['subset_state']=='GAP_TESTED_SUBSET' for r in decisive)
                fp=sum(r['static_predicted_gap'] and r['subset_state']=='COVERED_TESTED_SUBSET' for r in decisive)
                fn=sum(not r['static_predicted_gap'] and r['subset_state']=='GAP_TESTED_SUBSET' for r in decisive)
                tn=sum(not r['static_predicted_gap'] and r['subset_state']=='COVERED_TESTED_SUBSET' for r in decisive)
                summary.append({'policy':policy,'tau':tau,'L_cached_ms':limit,'selected_P':20,'decisive_P':len(decisive),
                    'unresolved_P':20-len(decisive),'TP_gap':tp,'FP_gap':fp,'FN_gap':fn,'TN_gap':tn,
                    'accuracy_on_decisive_selected_P':(tp+tn)/len(decisive) if decisive else None,
                    'gap_precision':tp/(tp+fp) if tp+fp else None,'gap_recall':tp/(tp+fn) if tp+fn else None,
                    'scope':'Conditional descriptive agreement; unresolved excluded, no unbiased population accuracy.'})
    return rows,summary


def finite20_marginals(found):
    indexed={(v['spec']['p']['id'],v['spec']['d']['id']):v for v in found.values()}
    pids=sorted({p for p,d in indexed});definitions={};rows=[]
    for ck in ('ck01','ck05'):
        for resolution in (640,1280):
            definitions[f'{ck}_R{resolution}_mean']=[(1/20,(p,f'{ck}_Infer{resolution}')) for p in pids]
        definitions[ck+'_1280_minus640_mean']=[(sign/20,(p,ck+'_Infer'+str(res))) for p in pids for sign,res in ((1,1280),(-1,640))]
    for resolution in (640,1280):
        definitions[f'ck01_minus_ck05_R{resolution}_mean']=[(sign/20,(p,ck+'_Infer'+str(resolution))) for p in pids for sign,ck in ((1,'ck01'),(-1,'ck05'))]
    for name,terms in definitions.items():
        complete=len(pids)==20 and all(k in indexed for sign,k in terms)
        for size in ('all','tiny','small','medium','large'):
            for metric in ('map50_95','map50','ar50_95'):
                value=paired_effect([(sign,indexed[k]['metrics'],indexed[k]['bootstrap']) for sign,k in terms],size,metric) if complete else {'estimate':None,'CI95_low':None,'CI95_high':None,'images':548,'draws':200}
                rows.append({'contrast':name,'size':size,'metric':metric,**value,'selected_contexts':20,
                    'status':'ESTIMABLE' if complete else 'NOT_ESTIMABLE',
                    'scope':'Equal mean of20 genuine context-level AP/AR outputs; paired shared-image bootstrap. Not pooled-image COCO AP or a uniform population estimate.'})
    return rows


def single_factor_identifiability(found):
    from sampling import models
    pm,_,_=models();recipes={v['spec']['p']['id']:set(v['spec']['p']['leaves']) for v in found.values()}
    observed={(v['spec']['p']['id'],v['spec']['d']['id']) for v in found.values()};rows=[]
    for family,members in pm.GROUPS.items():
        eligible=[]
        for left,right in combinations(sorted(recipes),2):
            changed=[name for name,features in pm.GROUPS.items() if recipes[left].intersection(features)!=recipes[right].intersection(features)]
            if changed==[family]:eligible.append((left,right))
        if not eligible:
            rows.append({'degradation_family':family,'left_P':None,'right_P':None,'checkpoint':None,
                'matched_P_pairs':0,'required_points':4,'observed_points':0,'status':'NOT_ESTIMABLE_NO_SINGLE_FACTOR_P_PAIR'})
        for left,right in eligible:
            for ck in ('ck01','ck05'):
                keys=[(p,ck+'_Infer'+str(r)) for p in (left,right) for r in (640,1280)]
                rows.append({'degradation_family':family,'left_P':left,'right_P':right,'checkpoint':ck,
                    'matched_P_pairs':len(eligible),'required_points':4,'observed_points':sum(k in observed for k in keys),
                    'status':'DIRECTLY_ESTIMABLE' if all(k in observed for k in keys) else 'NOT_ESTIMABLE_MISSING_D_CORNERS'})
    return rows


def refresh_results(out):
    from joint_analysis import refresh
    refresh(out)
    summary=load(out/'summary.json');latest=load(out.parent/'latest.json')
    latest.update({'reused_conditions':summary['combined_measured_pairs']-summary['new_measured_pairs'],
        'latest_complete20_analysis_RunID':out.name})
    save(out.parent/'latest.json',latest,out,replace=True)


def analyze(worker):
    if load(worker/'run.json')['status']=='RUNNING':raise ValueError('Worker is still running.')
    started=time.perf_counter();plan=load(worker/'EXECUTION_PROTOCOL.json');first=worker.parent/plan['first3_RunID']
    found=initial_sources(first);new_count=0
    for spec in plan['pairs']:
        folder=worker/'pairs'/spec['id']
        if not (folder/'SHA256SUMS.txt').exists():continue
        item=read_pair(folder)
        if spec['id'] in found or item['spec']!=spec:raise ValueError('Duplicate or changed point.')
        found[spec['id']]=item;new_count+=1
    out,_=new_run({'output_root':str(worker.parent)},EXPERIMENT,'full20_results')
    shutil.copyfile(Path(__file__),out/'executed_source.py')
    rows=[];classes=[];ci_rows=[];sources=[]
    expected_gt=plan['GT']['sha256'];expected_ids=sorted(r['id'] for r in load(Path(plan['GT']['path']))['records'])
    for identity,item in found.items():
        s=item['spec'];b=item['bootstrap'];t=item['timing']
        if item['checks']['ground_truth.json']!=expected_gt or b['draws']['seed']!=plan['seed'] or b['draws']['image_ids']!=expected_ids:
            raise ValueError('GT or bootstrap populations differ.')
        for interval in b['intervals']:
            values=[v for v in b['draws']['values'][interval['size']][interval['metric']] if v is not None]
            if len(values)!=interval['n_valid'] or values and not np.allclose(np.quantile(values,[.025,.975]),interval['ci95'],atol=1e-14,rtol=0):
                raise ValueError('Saved interval does not match saved resamples.')
            ci_rows.append({'pair_id':identity,**interval})
        for area in item['metrics']['per_area']:
            ci=next(r['ci95'] for r in b['intervals'] if r['size']==area['size'] and r['metric']=='map50_95')
            rows.append({'pair_id':identity,'p_id':s['p']['id'],'checkpoint':s['d']['checkpoint'],
                'checkpoint_SHA256':s['d']['sha256'],'resolution':s['d']['resolution'],
                'features':'|'.join(s['p']['leaves']),'images':548,**area,'AP_CI_low':ci[0],'AP_CI_high':ci[1],
                'mean_cached_pipeline_ms':t['mean_pipeline_ms'],'source_RunID':Path(item['folder']).parent.parent.name,
                'metrics_SHA256':item['checks']['metrics.json']})
        classes.extend({'pair_id':identity,**r,'metrics_SHA256':item['checks']['metrics.json']} for r in item['metrics']['per_class'])
        sources.append({'pair_id':identity,'folder':item['folder'],'manifest':source(Path(item['folder'])/'SHA256SUMS.txt')})
    save(out/'MEASURED_DESIGN.json',{'degradation_configurations':{v['spec']['p']['id']:v['spec']['p'] for v in found.values()},
        'detector_settings':{v['spec']['d']['id']:v['spec']['d'] for v in found.values()},
        'measured_pairs':sorted(found),'images_per_pair':548,'bootstrap_seed':plan['seed'],
        'selection':'Complete four-setting comparison on the20 previously selected contexts; no new probability sample.',
        'training_epochs_field':'Historical configured epochs, not the verified selected-best checkpoint epoch.',
        'execution_status':'Only sealed measurements enter this index. Frozen source specs retain their original planning fields.'},out)
    pairs,all_states,gaps=outcomes(found,plan['seed']);effects,missing=context_effects(found);static,agreement=static_comparison(found,gaps)
    csv_file(out/'FINITE20_MARGINALS.csv',finite20_marginals(found))
    csv_file(out/'RESOLUTION_EFFECTS.csv',resolution_effects(found))
    csv_file(out/'SINGLE_FACTOR_IDENTIFIABILITY.csv',single_factor_identifiability(found))
    learning=[]
    for tau in (.05,.10,.15):
        for limit in (50,100):
            selected=[r for r in pairs if r['tau']==tau and r['L_cached_ms']==limit]
            counts=Counter(r['state'] for r in selected)
            learning.append({'tau':tau,'L_cached_ms':limit,'measured_pairs':len(selected),
                **{state:counts[state] for state in ('PASS','FAIL','UNCERTAIN')},
                'PASS_degradation_groups':len({r['p_id'] for r in selected if r['state']=='PASS'}),
                'FAIL_degradation_groups':len({r['p_id'] for r in selected if r['state']=='FAIL'}),
                'independent_image_gate':'NOT_PASSED','held_out_classifier_score':'NOT_REPORTED'})
    csv_file(out/'LEARNING_FEASIBILITY.csv',learning)
    save(out/'LEARNING_ADMISSION.json',{'status':'ML_INSUFFICIENT_EVIDENCE','classifier_fitted':False,
        'current_boundary':'All labels use the same historical checkpoint/method-selection validation population. A P-group split is retrospective configuration diagnostics, not independent-image validation.',
        'required_new_evidence':['Checkpoint-specific verified untouched native10 images or a resolved test-dev admission',
            'Frozen image-byte and reliable scene separation, with duplicate bytes grouped together',
            'Prospectively held-out P configurations and no held-out hyperparameter selection',
            'Decisive class support and independent group counts justified for the intended precision',
            'Matched training seeds if making detector-family training-variance claims'],
        'additional_points_alone':'Completing the80-cell table identifies those responses; it cannot restore historical image independence.',
        'no_new_data_selected':True},out)
    save(out/'FUNCTION_IDENTIFIABILITY.json',{'finite_cell_means':80,'observed_cell_means':len(found),
        'unobserved_cell_means':80-len(found),'saturated_cell_model_rank':len(found),'saturated_residual_df':0,
        'model_fitted':False,'meaning':'Cell-response identification and independent predictive validation are different. No saturated fit is used as generalization evidence.'},out)
    for name,data in (('CONDITION_METRICS.csv',rows),('PER_CLASS_METRICS.csv',classes),('CONFIDENCE_INTERVALS.csv',ci_rows),
        ('PAIR_STATES.csv',pairs),('ALL_SETTING_OUTCOMES.csv',all_states),('FINITE20_GAPS.csv',gaps),
        ('CONTEXT_INTERACTIONS.csv',effects),('STATIC_EMPIRICAL_FINITE20.csv',static),('STATIC_AGREEMENT.csv',agreement)):
        csv_file(out/name,data)
    if missing:csv_file(out/'NOT_ESTIMABLE_CONTRASTS.csv',missing)
    primary=[r for r in gaps if r['tau']==.10 and r['L_cached_ms']==100];counts=Counter(r['subset_state'] for r in primary)
    alternate_counts=Counter(r['latency_interval_subset_state'] for r in primary)
    save(out/'ORACLE_UNCERTAINTY_AUDIT.json',{'original_policy_preserved':True,
        'legacy':'FAIL includes AP upper<tau OR latency upper>L; the latter is conservative admission rejection, not necessarily definite latency exceedance.',
        'latency_interval_v1':'FAIL only if AP upper<tau OR latency lower>L; PASS requires AP lower>=tau AND latency upper<=L; otherwise UNCERTAIN.',
        'same_thresholds_size_and_timing_boundary':True,'new_raw_measurements':0,
        'primary_legacy_subset_counts':dict(counts),'primary_latency_interval_subset_counts':dict(alternate_counts),
        'changed_pair_scenario_labels':sum(r['state']!=r['latency_interval_state'] for r in pairs),
        'confidence_scope':'Both audits use marginal pointwise intervals; neither claims simultaneous AP/latency95% confidence.'},out)
    status='VALIDATED' if len(found)==80 else 'PARTIAL'
    save(out/'EVIDENCE_SOURCES.json',{'conditions':sources,'original26':source(worker.parent/ANALYSIS_ID/'SHA256SUMS.txt'),
        'first3':source(first/'SHA256SUMS.txt'),'remainder_protocol':source(worker/'EXECUTION_PROTOCOL.json')},out)
    a=load(Path(plan['global_authorization']['path']));elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(a['global_start_UTC'])).total_seconds()
    if not (worker/'SHA256SUMS.txt').exists():
        save(worker/'EVIDENCE_TREE.json',{str(p.relative_to(worker)):sha256(p) for p in (worker/'pairs').rglob('*') if p.is_file()},worker)
        write_atomic(worker/'diary.md','# Complete-grid remainder\n\n'+str(new_count)+' sealed full548 points, without repeating the original26 or initial3. The global day deadline was not reset.\n',worker)
        seal(worker)
    resource={'first3':load(first/'supervisor.json'),'remainder':load(worker/'supervisor.json'),
        'global_elapsed_seconds_at_analysis':elapsed,'global_deadline_UTC':a['global_deadline_UTC'],'authorized_global_seconds':86400,
        'total_new_saved_image_records':(new_count+3)*548,'total_new_warmup_calls':(new_count+3)*3,
        'summed_new_forward_profiler_seconds':sum(sum(s['forward_ms'] for s in item['timing']['samples'])/1000 for item in found.values() if Path(item['folder']).parent.parent in (first,worker)),
        'peak_new_torch_allocated_bytes':max(item['timing']['peak_allocated_bytes'] for item in found.values() if Path(item['folder']).parent.parent in (first,worker)),
        'peak_new_torch_reserved_bytes':max(item['timing']['peak_reserved_bytes'] for item in found.values() if Path(item['folder']).parent.parent in (first,worker)),
        'new_evidence_bytes':sum(p.stat().st_size for root in (first,worker) for p in root.rglob('*') if p.is_file()),
        'new_pixel_generation_seconds':0,'source_pixel_policy':'Reused accepted caches; old cache generation is not counted as new work.',
        'new_phase_seconds':{key:sum(item['timing'][key] for item in found.values() if Path(item['folder']).parent.parent in (first,worker)) for key in ('evaluation_seconds','replay_seconds','bootstrap_seconds')},
        'timing_scope':'Worker supervision and global elapsed include different boundaries. Forward profiler is nested, not GPU occupancy or energy. No historical-budget reset.'}
    save(out/'RESOURCE_RESULT.json',resource,out)
    summary={'schema':'VM36_READ_ONLY_SUMMARY_V1','RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'COMPLETE20_SUBSET_RESULTS',
        'status':status,'joint_study':True,'complete20_study':True,'scope':'DEVELOPMENT_ONLY','validated_validation_images':548,
        'new_measured_pairs':new_count+3,'combined_measured_pairs':len(found),'formal_measured_pairs':0,
        'coverage':{**load(worker.parent/ANALYSIS_ID/'COVERAGE_ACCEPTANCE.json'),
            'original_core_pairs':26,'accepted_pairs':len(found),'planned_pairs':80,
            'scope':'Original26-pair core already witnesses all1033 binary states; this study completes responses on20P x fourD, not the full7488 projection.'},
        'worker_wall_seconds':resource['first3']['actual_supervision_seconds']+resource['remainder']['actual_supervision_seconds'],
        'metrics':[r for r in rows if r['size']=='all'],'gap_sensitivity':gaps,
        'finite20_primary_gap_counts':dict(counts),'source_hashes':{'sources':source(out/'EVIDENCE_SOURCES.json')},
        'primary_latency_interval_gap_counts':dict(alternate_counts),
        'learning':{'status':'ML_INSUFFICIENT_EVIDENCE','reason':'Same historical-selection validation images; a complete configuration grid is not independent-image validation.'},
        'gates':{'RQ1':'Restricted projection only; full UVL unchanged.','RQ2':'Matched context-bundle contrasts; no isolated factor or independent held-out classifier claim.',
            'RQ3':'Finite20/four-setting oracle classification; no uniform population estimate.','RQ4':'NOT_EVALUATED; training admission remains blocked.',
            'testdev':'DATASET_BLOCKED','UAVDT':'Source audit advanced; external evaluation not admitted.'}}
    save(out/'summary.json',summary,out)
    save(out/'result.json',{'RunID':out.name,'status':status,'mode':summary['mode'],
        'summary_canonical_sha256':hashlib.sha256(json.dumps(summary,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()},out)
    cpu={}
    for name,identity in (
        ('design','vm36_supervisor_reproduction_matched_followup_design_20261110T105657_56b11e1d'),
        ('training_data','vm36_supervisor_reproduction_training_data_cpu_v2_20261110T110058_057fb01c'),
        ('seed_diagnostic','vm36_supervisor_reproduction_loader_seed_diagnostic_20261110T110444_70a8cd35'),
        ('image_integrity','vm36_supervisor_reproduction_training_image_integrity_20261110T123852_0df355e2'),
        ('independence_and_RQ4','vm36_supervisor_reproduction_post26_admission_readiness_20261110T111123_2a3e725a')):
        cpu[name]=source(worker.parent/identity/'SHA256SUMS.txt')
    save(out/'CPU_BRANCH_REFERENCES.json',cpu,out)
    save(out/'STATISTICAL_SCOPE.json',{'AP':'COCO AP50:95, distinct from AP50; AR50:95 uses the saved evaluator maxDet3000.',
        'bootstrap':'200 paired image-index resamples, original548 image coordinates and inherited seed.',
        'CI_scope':'Marginal, pointwise percentile95% intervals. No familywise, joint AP/latency or training-seed confidence guarantee.',
        'configuration_sampling':'20 purposively selected P configurations; no probability population estimator.',
        'zero_AP':'A degenerate empirical bootstrap at[0,0] does not establish population AP is exactly zero.',
        'corruption_seed':'One deterministic realization per image/recipe; no repeated-corruption-seed inference.',
        'matched_context_selection':'The severe context was selected using earlier development evidence, then its missing corners were frozen. No historical blind-discovery claim.',
        'new_classifier_or_UVL_pruning_rule':'NOT_FITTED_NOT_VALIDATED'},out)
    write_report(out,worker,summary,effects,resource,agreement)
    save(out/'inputs.json',{'worker':source(worker/'SHA256SUMS.txt'),'source':source(Path(__file__))},out)
    write_atomic(out/'diary.md','# Finite20 results\n\nOnly sealed548-image points counted. Recomputed matched contrasts from paired draw arrays, retained uncertainty and static-policy counterexamples, and made no independent-image or population claim.\n',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':summary['mode'],'status':status,'CPU_seconds':time.perf_counter()-started},out,replace=True)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>VM36 finite20 results</h1><p>'+status+' · '+str(len(found))+'/80</p><a href="CONDITION_METRICS.csv">Metrics</a> · <a href="FINITE20_GAPS.csv">Subset gaps</a> · <a href="CONTEXT_INTERACTIONS.csv">Paired context contrasts</a> · <a href="STATIC_AGREEMENT.csv">Static agreement</a>',out)
    seal(out)
    refresh_results(out);print(out,flush=True)


def write_report(out,worker,s,effects,resource,agreement):
    complete=s['combined_measured_pairs']==80
    text=['# VM36 '+('完整20×4检测器子集实验' if complete else '20×4子集实验：部分完成'),'',
        'PROBLEM：同一退化配置下，四个固定检测器设置是否均失败；分辨率收益是否随复合条件改变。',
        'WHY：原26组仅实现特征状态覆盖，不能把17个未决P当作gap。',
        'SOLUTION：复用原26组和已完成3组，新增冻结补集中的剩余点；每组548张val，native10 FP32、相同原像素、GT、ignore、COCO和200次配对图像重采样。',
        'WHY IS IT A GOOD SOLUTION：完整设置逐项记录，未测仍未知；交互只在真实角点齐全时计算。','',
        f'本轮新增实测{s["new_measured_pairs"]}/54组；累计{s["combined_measured_pairs"]}/80组；独立test-dev为0组。',
        '模型：ck01 YOLO26s与ck05 YOLOv8s，历史FT1280、seed42；本轮推理640/1280。原SHA、参数与逐图记录见EVIDENCE_SOURCES.json及各configuration.json。',
        'MEASURED_DESIGN.json汇总20个退化的准确数值参数、四个设置和checkpoint完整SHA。历史100 epochs为配置上限，所选best的实际epoch未核实。',
        '样本：20个既定目的性退化配置，非均匀随机总体样本。没有增加新的退化或模型。',
        '主要场景：Tiny AP50:95阈值0.10，缓存输入处理流程上限100ms；其他冻结场景为敏感性分析。',
        '主要状态计数：'+json.dumps(s['finite20_primary_gap_counts'],ensure_ascii=False),
        '旧时延上界规则保留，其FAIL含保守验收拒绝；不全等于时延确证超标。上下界分开的latency_interval_v1敏感性计数：'+json.dumps(s['primary_latency_interval_gap_counts'],ensure_ascii=False),
        'DID(P) = [AP1280(P) − AP640(P)] − [AP1280(Normal+JPEG Q25) − AP640(Normal+JPEG Q25)]。JPEG Q75行与此前Q25减Q75的端点对比方向相反，原结果未改。',
        '上述是固定20P/四D子集的分类，不是全FM_D或总体gap比例；CI跨阈值的状态保留UNRESOLVED。','',
        '| 模型 | 原预定重度复合条件相对Q25参考的Resolution DID | 95% CI |','|---|---:|---|']
    for row in effects:
        if row['p_id']==SEVERE_ID and row['size']=='all' and row['metric']=='map50_95':
            text.append(f'|{row["checkpoint"]}|{row["estimate"]:.6f}|[{row["CI95_low"]:.6f}, {row["CI95_high"]:.6f}]|')
    text+=['','复合条件同时包含LowLight、Defocus、Gaussian、Shot与Downscale；不能把此DID归因于单个成分。其他P对比为探索性、点态区间，不声称多重比较控制或训练seed显著性。',
        '原预定重度条件的Tiny AP观测为0时，Bootstrap区间可退化为[0,0]；这只是本图像集的经验结果，不代表真实场景总体AP必为0。该条件按既有失败选择后冻结，属于目标化开发诊断。',
        'SINGLE_FACTOR_IDENTIFIABILITY.csv按原UVL语义分组检查实际P配对；补齐D设置不能代替缺少的单因素P反事实。',
        'STATIC_AGREEMENT.csv只在已判定的选定P上比较原三静态策略；未决分母单列，不把静态未覆盖等同实际失败。',
        '所有AP50、AP50:95、AR、尺寸、原生十类及区间均见原值CSV，无插值。',
        'FINITE20_MARGINALS.csv为20个既定条件的等权AP/AR均值和配对差值，不是逐图AP平均，也不是均匀总体估计或混合图片后的COCO AP。',
        'RESOLUTION_EFFECTS.csv逐条件记录同一checkpoint的1280减640差值及配对区间。相对参考的DID为负，只说明分辨率收益减弱，不等于该条件下1280一定低于640。',
        f'新worker监督时间合计{s["worker_wall_seconds"]:.3f}秒；从本轮首次授权起累计墙钟{resource["global_elapsed_seconds_at_analysis"]:.3f}秒。24小时截止未重置。',
        f'本轮Torch allocated峰值{resource["peak_new_torch_allocated_bytes"]/1024**3:.3f} GiB，reserved峰值{resource["peak_new_torch_reserved_bytes"]/1024**3:.3f} GiB。阶段计时见RESOURCE_RESULT.json，嵌套forward时间不再与worker墙钟相加。',
        'ML_INSUFFICIENT_EVIDENCE：所有配置共享历史选模val，完整网格不等于独立保留图像。没有拟合饱和模型或编造分类准确率。',
        'RQ4仍NOT_EVALUATED。6471图标签/种子CPU记录另存；无GPU训练、无正式训练成本预测。test-dev仍DATASET_BLOCKED；UAVDT仅源和协议审计。',
        'CPU_BRANCH_REFERENCES.json连接本轮全6471图解码、标签及JPEG完整性记录；3条零面积行和4个重复框在独立候选标签视图中处理。原数据未改，生产训练loader未接入。',
        '16图CPU增强重复检查中，标准启动同seed未完全复现，构建dataset后重新固定seed可复现；这不代表完整训练轨迹或GPU恢复已验收。',
        '保留548 ID中的重复字节、图像/场景相关、未知部分创建期训练来源等限制。',
        '证据：'+str(out),'复现：full20_results.py --worker-run '+worker.name+'（CPU，不重跑推理）']
    note=ROOT/'datasets/reportcn'/('vm36_supervisor_reproduction_'+('完整20x4结果_' if complete else '20x4部分结果_')+out.name.split('_')[-1]+'.md')
    write_atomic(note,'\n'.join(text)+'\n',out);save(out/'REPORT_REFERENCE.json',source(note),out)
    write_atomic(out/'FIGURE_CAPTIONS_EN.md','# Source-driven captions\n\nMatched context contrast. Each difference-in-differences compares the 1280-minus-640 AP change between the fixed compound context and its Normal/JPEG Q25 reference. Each corner uses 548 shared validation images and a fixed checkpoint. Intervals use 200 paired image draws. These are context-bundle contrasts, not component ablations. Source: CONTEXT_INTERACTIONS.csv.\n\nFinite detector-subset gap. Outcomes use Tiny AP confidence bounds and cached-input pipeline latency for the 20 selected degradation configurations and four fixed settings. Unknown or interval-ambiguous states remain unresolved. Source: FINITE20_GAPS.csv. No new decorative figures were generated.\n',out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker-run',required=True);args=p.parse_args();analyze(path_for(args.worker_run))
