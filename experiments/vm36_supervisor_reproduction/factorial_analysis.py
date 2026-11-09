"""Join immutable measurements; calculate paired factorial contrasts on CPU."""
import argparse
from collections import Counter
import hashlib
import gzip
import json
from pathlib import Path
import sys
import time
import numpy as np
from protocol import ROOT,HERE,EXPERIMENT,FMP,FMD,load,save,source,sha256,write_atomic
from common import seal,verify_hashes
from historical_review import csv_file,table
from full_analysis import paired_effect,tested_subset_state
from full_development import PREP
from complete_factorial import SOURCE_ID
from main.pipeline import new_run

METRICS=('map50_95','map50','ar50_95','ar50')
SIZES=('all','tiny','small','medium','large')


def contrast_terms(ck,quality=None):
    if quality is not None:return [(1,(ck,1280,quality)),(-1,(ck,640,quality))]
    return [(1,(ck,1280,25)),(-1,(ck,640,25)),(-1,(ck,1280,75)),(1,(ck,640,75))]


def analyze(new):
    started=time.perf_counter();old=new.parent/SOURCE_ID
    verify_hashes(old,('EXECUTION_PROTOCOL.json','RESULT_SUMMARY.json','EVIDENCE_TREE.json'))
    if load(new/'run.json')['status']!='VALIDATED' or load(new/'worker_result.json')['completed']!=2:
        raise ValueError('Both missing corners must pass before combined analysis.')
    oldplan=load(old/'EXECUTION_PROTOCOL.json');newplan=load(new/'EXECUTION_PROTOCOL.json')
    for key in ('FMP','FMD','GT','image_members','ZIP','generator','images','seed','draws','precision','timing'):
        if newplan[key]!=oldplan[key]:raise ValueError('Scientific protocol differs: '+key)
    for spec in newplan['pairs']:
        if spec not in oldplan['pairs']:raise ValueError('New condition is outside the original design.')
    for key in ('FMP','FMD','GT','image_members','ZIP','generator'):
        record=oldplan[key]
        if sha256(Path(record['path']))!=record['sha256']:raise ValueError('Protected source changed: '+key)
    if not (new/'SHA256SUMS.txt').exists():
        for spec in newplan['pairs']:verify_hashes(new/'pairs'/spec['id'],('metrics.json','bootstrap.json','verification.json'))
        save(new/'MEASUREMENT_ACCEPTANCE.json',{'status':'VALIDATED','new_pairs':2,'new_image_inferences':1096,
            'original_six_repeated':False,'worker_cap_seconds':1800,'GPU_resume_validation':'NOT_PERFORMED',
            'source_six_manifest':source(old/'SHA256SUMS.txt')},new)
        write_atomic(new/'diary.md','# Missing corner measurements\n\nTwo ck01/1280 JPEG endpoints on all548 validation images; existing cache reused and checked. New bounded worker, not a reset or a demonstrated crash-recovery run. Predictions replayed through the same COCO evaluator; 200 paired image draws.\n',new)
        write_atomic(new/'index.html','<!doctype html><meta charset="utf-8"><h1>VM36 · two missing corners</h1><p>2/2 measured; 1,096 new image-inference operations; development only.</p><a href="worker_result.json">Worker result</a> · <a href="MEASUREMENT_ACCEPTANCE.json">Acceptance</a>',new)
        save(new/'EVIDENCE_TREE.json',{str(p.relative_to(new)).replace('\\','/'):sha256(p)
            for p in sorted((new/'pairs').rglob('*')) if p.is_file()},new)
        seal(new)
    verify_hashes(new,('MEASUREMENT_ACCEPTANCE.json','EVIDENCE_TREE.json'))
    out,_=new_run({'output_root':str(old.parent)},EXPERIMENT,'factorial_analysis')
    accepted={};timings={};references=[];rows=[];classes=[];terms_by_name={};pair_checks=[]
    expected_ids=sorted(r['id'] for r in load(Path(oldplan['GT']['path']))['records'])
    for spec in oldplan['pairs']:
        locations=[run/'pairs'/spec['id'] for run in (old,new) if (run/'pairs'/spec['id']/'SHA256SUMS.txt').exists()]
        if len(locations)!=1:raise ValueError('Missing or repeated configuration: '+spec['id'])
        folder=locations[0];checks=verify_hashes(folder,('metrics.json','bootstrap.json','timing.json','configuration.json',
            'predictions.jsonl.gz','ground_truth.json','verification.json','image_manifest.jsonl'))
        parent=folder.parent.parent
        if load(folder/'configuration.json')!=spec:raise ValueError('Pair configuration differs.')
        if load(folder/'verification.json')['protocol_sha256']!=sha256(parent/'EXECUTION_PROTOCOL.json'):raise ValueError('Pair protocol identity differs.')
        if checks['ground_truth.json']!=oldplan['GT']['sha256']:raise ValueError('GT bytes differ.')
        d=spec['d'];quality=int(next(x for x in spec['p']['leaves'] if x.startswith('JPEG')).split('q')[1])
        key=(d['checkpoint'],d['resolution'],quality);metric=load(folder/'metrics.json');boot=load(folder/'bootstrap.json');timing=load(folder/'timing.json')
        if metric['images']!=548:raise ValueError('Incomplete image population.')
        if sha256(Path(d['path']))!=d['sha256']:raise ValueError('Checkpoint changed.')
        if boot['draws']['image_ids']!=expected_ids or boot['draws']['seed']!=oldplan['seed'] or boot['draws']['samples']!=200:
            raise ValueError('Bootstrap identity differs from the source population.')
        for interval in boot['intervals']:
            values=[v for v in boot['draws']['values'][interval['size']][interval['metric']] if v is not None]
            if len(values)!=interval['n_valid'] or interval['n_samples']!=200:raise ValueError('Bootstrap draw count differs.')
            if values and not np.allclose(np.quantile(values,[.025,.975]),interval['ci95'],rtol=0,atol=1e-14):
                raise ValueError('Saved interval differs from saved draws.')
        if parent==new:
            ids=[];raw_count=0
            with gzip.open(folder/'predictions.jsonl.gz','rt',encoding='utf-8') as stream:
                for line in stream:
                    r=json.loads(line);ids.append(r['image_id']);raw_count+=len(r['predictions'])
                    if any(p['image_id']!=r['image_id'] or not 0<=p['class_id']<10 or not np.isfinite(p['bbox']+[p['score']]).all() for p in r['predictions']):
                        raise ValueError('Invalid raw prediction.')
            if ids!=expected_ids or raw_count!=metric['raw_count']:raise ValueError('New raw stream count differs.')
        if not load(folder/'verification.json')['COCO_recomputed_from_raw']:raise ValueError('Missing raw COCO replay.')
        pair_checks.append({'pair_id':spec['id'],'source_RunID':parent.name,'COCO_replay':'VERIFIED_AT_MEASUREMENT',
            'CI_from_saved_draws':'VERIFIED','actual_NCHW':[1,3,d['resolution'],d['resolution']],
            'raw_stream_audit':'VERIFIED_NOW' if parent==new else 'REUSED_HASH_VERIFIED_ACCEPTANCE'})
        manifest=[json.loads(line) for line in (folder/'image_manifest.jsonl').read_text().splitlines()]
        if len(manifest)!=548 or sorted(r['image_id'] for r in manifest)!=boot['draws']['image_ids']:raise ValueError('Image identities differ.')
        for r in manifest:
            if r['tensor']!={'NCHW':[1,3,d['resolution'],d['resolution']],'dtype':'torch.float32'}:raise ValueError('Actual tensor differs.')
        accepted[key]=(spec,metric,boot);timings[key]=timing
        references.append({'pair_id':spec['id'],'source_RunID':parent.name,'pair_folder':str(folder),
            'checkpoint_SHA256':d['sha256'],'pair_manifest':source(folder/'SHA256SUMS.txt'),
            'prediction':source(folder/'predictions.jsonl.gz'),'GT':source(folder/'ground_truth.json'),
            'metrics':source(folder/'metrics.json'),'bootstrap':source(folder/'bootstrap.json')})
        for area in metric['per_area']:
            ci=next(r['ci95'] for r in boot['intervals'] if r['size']==area['size'] and r['metric']=='map50_95')
            rows.append({'pair_id':spec['id'],'checkpoint':key[0],'resolution':key[1],'JPEG':quality,'images':548,
                **area,'AP_CI_low':ci[0],'AP_CI_high':ci[1],'mean_cached_pipeline_ms':timing['mean_pipeline_ms'],
                'peak_allocated_bytes':timing['peak_allocated_bytes'],'source_RunID':parent.name,'metrics_SHA256':checks['metrics.json']})
        for category in metric['per_class']:classes.append({'checkpoint':key[0],'resolution':key[1],'JPEG':quality,
            **category,'source_RunID':parent.name,'metrics_SHA256':checks['metrics.json']})
    if len(accepted)!=8:raise ValueError('Eight unique measured conditions are required.')
    for ck in ('ck05','ck01'):
        for q in (25,75):terms_by_name[(ck,'1280-640 at Q'+str(q))]=contrast_terms(ck,q)
        terms_by_name[(ck,'(1280-640 at Q25)-(1280-640 at Q75)')]=contrast_terms(ck)
    terms_by_name[('ck01-minus-ck05','DID_ck01-DID_ck05')]=contrast_terms('ck01')+[(-sign,key) for sign,key in contrast_terms('ck05')]
    effects=[];per_class_effects=[]
    for (ck,label),definition in terms_by_name.items():
        terms=[(sign,accepted[key][1],accepted[key][2]) for sign,key in definition]
        for size in SIZES:
            for m in METRICS:effects.append({'checkpoint':ck,'contrast':label,'size':size,'metric':m,
                **paired_effect(terms,size,m),'scope':'POINTWISE_PAIRED_IMAGE_DEVELOPMENT_ONLY'})
        for category in range(1,11):
            for m in METRICS:
                values=[(sign,next(r for r in accepted[key][1]['per_class'] if r['category_id']==category)) for sign,key in definition]
                value=None if any(r[m] is None for _,r in values) else sum(sign*r[m] for sign,r in values)
                per_class_effects.append({'checkpoint':ck,'contrast':label,'category_id':category,
                    'category_name':values[0][1]['category_name'],'metric':m,'estimate':value,
                    'confidence_interval':'NOT_EVALUATED_PER_CLASS','scope':'DESCRIPTIVE_FIXED_CHECKPOINTS'})
    csv_file(out/'CONFIGURATION_METRICS.csv',rows);csv_file(out/'PER_CLASS_METRICS.csv',classes)
    csv_file(out/'INTERACTION_EFFECTS.csv',effects);csv_file(out/'PER_CLASS_EFFECTS.csv',per_class_effects)
    save(out/'EVIDENCE_SOURCES.json',{'conditions':references,'original_protocol':source(old/'EXECUTION_PROTOCOL.json'),
        'completion_protocol':source(new/'EXECUTION_PROTOCOL.json'),'new_conditions':2,'reused_conditions':6},out)
    save(out/'EVIDENCE_ACCEPTANCE.json',{'status':'VALIDATED','conditions':pair_checks,'GPU_resume':'NOT_PERFORMED',
        'all_original_inputs_unchanged':'Referenced SHA values verified; no whole-archive claim.'},out)
    save(out/'protocol.json',{'scope':'DEVELOPMENT_ONLY','source_protocol':source(old/'EXECUTION_PROTOCOL.json'),
        'hypothesis':'Fixed-checkpoint resolution x original JPEG endpoint interaction and the difference between checkpoints.',
        'images':548,'checkpoints':['ck05','ck01'],'resolutions':[640,1280],'JPEG':[25,75],
        'bootstrap':{'unit':'image position','seed':oldplan['seed'],'samples':200,'interval':'pointwise percentile'},
        'per_class_interval':'NOT_EVALUATED; per-class draws were not saved in the source bootstrap'},out)
    states,gaps=gap_sensitivity(accepted,timings,oldplan['seed'])
    csv_file(out/'PAIR_STATES.csv',states);csv_file(out/'SUBSET_GAPS.csv',gaps)
    costs={'old_worker_seconds':load(old/'worker_result.json')['worker_wall_seconds'],
        'new_worker_seconds':load(new/'worker_result.json')['worker_wall_seconds'],
        'new_cap_seconds':1800,'total_measured_conditions':8,'new_image_inferences':1096,
        'forward_profiler_seconds':sum(x['forward_ms'] for t in timings.values() for x in t['samples'])/1000,
        'peak_allocated_bytes':max(t['peak_allocated_bytes'] for t in timings.values()),
        'peak_reserved_bytes':max(t['peak_reserved_bytes'] for t in timings.values()),
        'memory_scope':'Torch worker allocated/reserved peaks; total-device peak was not sampled.',
        'latency_definition':oldplan['timing'],'total_GPU_occupancy':'NOT_MEASURED'}
    save(out/'COSTS.json',costs,out)
    plot(out,rows,effects)
    followup(out,accepted,states,timings)
    status='VALIDATED'
    summary={'schema':'VM36_READ_ONLY_SUMMARY_V1','RunID':out.name,'ExperimentID':EXPERIMENT,'status':status,
        'mode':'FULL548_FACTORIAL','measured_study':True,'scope':'DEVELOPMENT_ONLY','validated_validation_images':548,
        'new_measured_pairs':2,'combined_measured_pairs':8,'formal_measured_pairs':0,
        'worker_wall_seconds':costs['old_worker_seconds']+costs['new_worker_seconds'],
        'forward_profiler_seconds':costs['forward_profiler_seconds'],'metrics':[r for r in rows if r['size']=='all'],
        'effects':[r for r in effects if r['metric']=='map50_95'],'source_completion_RunID':new.name,
        'source_hashes':{'original':source(old/'SHA256SUMS.txt'),'completion':source(new/'SHA256SUMS.txt')},
        'gates':[{'branch':k,'status':v} for k,v in {'Eight conditions':'VALIDATED','Independent test-dev':'DATASET_BLOCKED',
            'Grouped failure learner':'ML_INSUFFICIENT_EVIDENCE','Population gap':'NOT_ESTIMATED',
            'Matched augmentation':'RESOURCE_BLOCKED','GPU interruption recovery':'NOT_VALIDATED'}.items()]}
    save(out/'summary.json',summary,out)
    save(out/'result.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':summary['mode'],'status':status,
        'summary_canonical_sha256':hashlib.sha256(json.dumps(summary,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()},out)
    write_report(out,summary,costs,effects,rows)
    save(out/'inputs.json',{'old':source(old/'SHA256SUMS.txt'),'new_protocol':source(new/'EXECUTION_PROTOCOL.json'),
        'analysis_source':source(Path(__file__))},out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':summary['mode'],'status':status,
        'scientific_scope':'DEVELOPMENT_ONLY; RQ1-RQ4 not fully accepted','analysis_seconds':time.perf_counter()-started},out,replace=True)
    write_atomic(out/'diary.md','# Full factorial analysis\n\nJoined six accepted and two new immutable conditions by exact identity. Paired image intervals, size effects, per-class point effects and subset-gap sensitivities are separate from population estimates. No independent test, new training or GPU recovery claim.\n',out)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>VM36 · eight-condition development study</h1><p>8/8 measured; 6 reused + 2 new; formal independent pairs: 0.</p><a href="CONFIGURATION_METRICS.csv">Metrics</a> · <a href="INTERACTION_EFFECTS.csv">Paired effects</a><p><img width="900" src="resolution_jpeg.png"></p><p><img width="900" src="interaction_intervals.png"></p>',out)
    seal(out);refresh_history(out);print(out,flush=True)


def refresh_history(out):
    import re,html
    history=[]
    for folder in out.parent.iterdir():
        if not folder.is_dir() or not (folder/'run.json').is_file():continue
        record=load(folder/'run.json');match=re.search(r'\d{8}T\d{6}',folder.name)
        history.append({'RunID':folder.name,'mode':record.get('mode'),'status':record.get('status'),
            'UTC':match.group() if match else '', 'page':folder.name+('/index.html' if (folder/'index.html').is_file() else '/run.json')})
    history.sort(key=lambda r:r['UTC'],reverse=True)
    body='<!doctype html><meta charset="utf-8"><title>VM36 evidence</title><style>body{max-width:1100px;margin:40px auto;font:17px Arial}td,th{padding:10px;text-align:left}code{font-size:12px}</style><h1>VM36 evidence history</h1><p><a href="'+out.name+'/index.html">Latest eight-condition results and figures</a></p><p>8/8 development conditions · 6 reused + 2 new · 0 formal independent pairs</p><table><tr><th>UTC</th><th>Mode</th><th>Status</th><th>Evidence</th></tr>'
    for row in history:body+='<tr><td>'+row['UTC']+'</td><td>'+html.escape(str(row['mode']))+'</td><td>'+html.escape(str(row['status']))+'</td><td><a href="'+row['page']+'">Open</a><details><summary>RunID</summary><code>'+row['RunID']+'</code></details></td></tr>'
    write_atomic(out.parent/'index.html',body+'</table>',out,replace=True)
    save(out.parent/'latest.json',{'ExperimentID':EXPERIMENT,'latest_combined_RunID':out.name,
        'status':'VALIDATED_DEVELOPMENT_FACTORIAL','measured_conditions':8,'new_conditions':2,
        'formal_measured_pairs':0,'history':history},out,replace=True)


def gap_sensitivity(accepted,timings,seed):
    from gap_analysis import pair_outcome
    states=[];gaps=[];indices=np.random.default_rng(seed).integers(0,548,(200,548))
    static=table(PREP/'STATIC_COVERAGE_REFERENCE.csv')
    for key,(spec,metric,boot) in accepted.items():
        latency=np.quantile(np.asarray([s['pipeline_ms'] for s in timings[key]['samples']])[indices].mean(axis=1),[.025,.975])
        ci=next(r['ci95'] for r in boot['intervals'] if r['size']=='tiny' and r['metric']=='map50_95')
        for tau in (.05,.10,.15):
            for limit in (50,100):states.append({'checkpoint':key[0],'resolution':key[1],'JPEG':key[2],'tau':tau,'L_cached_ms':limit,
                'state':pair_outcome(ci,float(latency[1]),tau,limit),'AP_CI':ci,'mean_latency_CI':latency.tolist(),
                'scope':'PREVIOUSLY_DEFINED_THRESHOLD_SENSITIVITY_ONLY'})
    for q in (25,75):
        for tau in (.05,.10,.15):
            for limit in (50,100):
                subset=[r for r in states if r['JPEG']==q and r['tau']==tau and r['L_cached_ms']==limit]
                row={'JPEG':q,'tau':tau,'L_cached_ms':limit,'detector_settings':4,
                    'empirical_state':tested_subset_state([r['state'] for r in subset],4)}
                for policy in ('exact_identity','conservative','supervisor_proxy'):
                    row[policy+'_static_covered']=any({'Normal','JPEG_q'+str(q)}.issubset(set(r['covered_leaves'].split('|')))
                        for r in static if r['policy']==policy and r['checkpoint'] in ('ck05','ck01'))
                gaps.append(row)
    return states,gaps


def plot(out,rows,effects):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,5))
    for ck,name in (('ck05','YOLOv8s'),('ck01','YOLO26s')):
        for q,style in ((25,'--'),(75,'-')):
            selected=sorted([r for r in rows if r['size']=='all' and r['checkpoint']==ck and r['JPEG']==q],key=lambda r:r['resolution'])
            line,=ax.plot([r['resolution'] for r in selected],[100*r['map50_95'] for r in selected],style,marker='o',label=name+' Q'+str(q))
            ax.vlines([r['resolution'] for r in selected],[100*r['AP_CI_low'] for r in selected],[100*r['AP_CI_high'] for r in selected],color=line.get_color(),alpha=.6)
    ax.set(xlabel='Inference resolution',ylabel='COCO AP50:95 (%)',title='Fixed checkpoints · resolution x JPEG endpoints')
    ax.legend();fig.text(.1,.01,'548 validation images; 200 paired image draws; pointwise 95% intervals. Development only.',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'resolution_jpeg.png',dpi=180);plt.close(fig)
    selected=[r for r in effects if r['metric']=='map50_95' and r['contrast'].startswith(('(', 'DID'))]
    fig,ax=plt.subplots(figsize=(10,7))
    for i,r in enumerate(selected):
        ax.plot(100*r['estimate'],i,'o');ax.hlines(i,100*r['CI95_low'],100*r['CI95_high'])
    ax.set_yticks(range(len(selected)),[r['checkpoint']+' / '+r['size'] for r in selected]);ax.axvline(0,color='grey',lw=1)
    ax.set(xlabel='DID or checkpoint DID difference (AP percentage points)',title='(1280−640 at Q25) − (1280−640 at Q75)')
    fig.text(.1,.01,'Same images and bootstrap positions; fixed checkpoints; no architecture-causality or training-seed claim.',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'interaction_intervals.png',dpi=180);plt.close(fig)
    save(out/'FIGURE_PROVENANCE.json',{'metrics':source(out/'CONFIGURATION_METRICS.csv'),
        'effects':source(out/'INTERACTION_EFFECTS.csv'),'script':source(Path(__file__))},out)


def followup(out,accepted,states,timings):
    from sampling import models
    from full_analysis import binary_sample_size
    fmp,_,prior=models();population=prior.enumerate_configs(fmp)
    blocked=set(load(PREP/'IMPLEMENTATION_BINDINGS_PROPOSAL.json')['proposed_values_from_historical_implementation'])
    eligible=[c for c in population if not blocked.intersection(c)]
    masks,feasible=prior.compute_binary_pair_masks(eligible,sorted(fmp.ALL_LEAVES));uncovered=feasible;selected=[]
    while uncovered:
        gains=[(m&uncovered).bit_count() for m in masks];i=gains.index(max(gains));selected.append(i);uncovered&=~masks[i]
    csv_file(out/'PARAMETER_ADMITTED_PAIRWISE_CORE.csv',[{'configuration_id':fmp.configuration_id(eligible[i]),
        'leaves':'|'.join(eligible[i]),'status':'SELECTED_NOT_MEASURED','population':len(eligible)} for i in selected])
    from development_v1 import generator
    from alignment import parameters,require_runtime_bindings
    renderer=generator();proposal=load(PREP/'IMPLEMENTATION_BINDINGS_PROPOSAL.json')
    fixture=np.random.default_rng(20261109).integers(0,256,(96,160,3),dtype=np.uint8)
    smoke=[];covered=set()
    for leaf in sorted(set().union(*map(set,eligible))):
        chosen=next(c for c in eligible if leaf in c)
        attributes=require_runtime_bindings(parameters(FMP,chosen),proposal)
        pixels,trace=renderer.generate(fixture,chosen,attributes,'cpu_fixture',20261109)
        repeated,again=renderer.generate(fixture,chosen,attributes,'cpu_fixture',20261109)
        if pixels.shape!=fixture.shape or pixels.dtype!=np.uint8 or not np.array_equal(pixels,repeated) or trace!=again:
            raise ValueError('Parameter-admitted renderer smoke test failed.')
        covered.update(chosen)
        smoke.append({'target_leaf':leaf,'configuration_id':fmp.configuration_id(chosen),'attributes':attributes,
            'pixel_SHA256':hashlib.sha256(pixels.tobytes()).hexdigest(),'shape':list(pixels.shape),
            'status':'SYNTHETIC_CPU_RENDERER_ONLY_NOT_AP'})
    save(out/'RENDERER_SMOKE.json',{'source':source(HERE.parent/'cv261109/generator.py'),'leaf_coverage':sorted(covered),'cases':smoke},out)
    sampling=[];mean_pair=float(np.mean([t['wall_seconds'] for t in timings.values()]))
    for half in (.02,.05,.075,.10,.15):
        n=binary_sample_size(len(eligible),half)
        sampling.append({'population':len(eligible),'candidate_half_width':half,'confidence':.95,'n':n,
            'detector_settings':4,'full548_pairs':n*4,'screening_worker_hours':n*4*mean_pair/3600,
            'status':'PLANNING_ONLY; mission threshold/precision and additional budget not admitted'})
    csv_file(out/'UNIFORM_RESOURCE_REQUIREMENTS.csv',sampling)
    learning=[]
    for tau in (.05,.10,.15):
        for limit in (50,100):
            counts=Counter(r['state'] for r in states if r['tau']==tau and r['L_cached_ms']==limit)
            learning.append({'tau':tau,'L_cached_ms':limit,'configurations':8,'JPEG_groups':2,
                **{k:counts[k] for k in ('PASS','FAIL','UNCERTAIN','NOT_EVALUATED')},
                'status':'ML_INSUFFICIENT_EVIDENCE','reason':'Two JPEG endpoint groups share the same images; mission labels not fixed; no untouched configuration/image population.'})
    csv_file(out/'LEARNING_ADMISSION.csv',learning)
    design=[]
    for ck,res,q in accepted:
        c=float(ck=='ck01');r=float(res==1280);j=float(q==75)
        design.append([1,c,r,j,c*r,c*j,r*j,c*r*j])
    save(out/'LEARNING_IDENTIFIABILITY.json',{'configurations':len(design),'full_three_factor_columns':8,
        'matrix_rank':int(np.linalg.matrix_rank(np.asarray(design))),
        'full_factorial_residual_configuration_df':len(design)-int(np.linalg.matrix_rank(np.asarray(design))),
        'JPEG_level_groups':2,'shared_original_image_population':548,
        'conclusion':'A saturated model can fit this grid; this is not a grouped held-out validation or an independently validated failure rule.',
        'heldout_metrics':'NOT_REPORTED'},out)
    save(out/'RQ1_RQ4_NEXT_GATES.json',{'RQ1':{'full_legal_FMP':len(population),'parameter_admitted_stratum':len(eligible),
        'excluded_leaves_pending_bindings':sorted(blocked),'stratum_pairwise_core':len(selected),
        'stratum_feasible_binary_states':feasible.bit_count(),'scope':'Admission filter, not additional logical constraints; still requires renderer smoke/geometry checks.'},
        'RQ2':'8-condition descriptive interactions accepted; grouped generalization and failure-rule learning not accepted.',
        'RQ3':'Complete four-setting subset sensitivities available; no full-FM_D or uniform population estimate.',
        'RQ4':{'minimum_matched_primary_runs':12,'fully_seed_matched_five_ablation_runs':15,'total_with_ablations':27,
            'status':'RESOURCE_BLOCKED','other_blockers':['Three invalid original training annotation rows require a separately frozen derived loader/label policy.',
                'Augmentation ranges must contain, but not equal, test values; no approved ranges or full matched cost budget.',
                'Historical stripped checkpoints do not establish three matched training seeds.']},
        'independent_testdev':'DATASET_BLOCKED; file disjointness is not proof of no historical selection/tuning.',
        'new_GPU_work_after_completion':0,'next_executable_branch':'CPU audit of renderer geometry and candidate training-label policy; GPU work needs a new recorded resource allocation.'},out)


def write_report(out,summary,costs,effects,rows):
    identities={r['pair_id'].split('__')[1].split('_')[0]:r['checkpoint_SHA256'] for r in load(out/'EVIDENCE_SOURCES.json')['conditions']}
    text=['# VM36：八条件完整开发实验','',
        '1. Research problem: RQ2，分辨率与原始JPEG端点交互；比较两个固定checkpoint。',
        '2. Why necessary: 补齐原2×2设计的两个缺失角点，不重跑已验收六组。',
        '3. Dataset: VisDrone2019-DET validation，DEVELOPMENT_ONLY；曾参与历史选模。',
        '4. Full image count: 每组完整548张，八组共4384张次，本轮新增1096张次。',
        '5. Architecture/SHA: ck05 YOLOv8s '+identities['ck05']+'；ck01 YOLO26s '+identities['ck01']+'。',
        '6. Resolution: 历史训练1280；本次推理640/1280。历史单seed42，不代表训练seed稳定性。',
        '7. Degradation: Normal；JPEG质量25/75，原图解码后额外编码一次；其他退化关闭。Q75不是CLEAN。',
        '8. Sampling: 每checkpoint两个因素的完整四角点，最少四组；非均匀总体抽样。',
        '9. Controlled: 相同图像、GT、像素缓存、FP32、dense one-to-many、原NMS/ignore/COCO口径。',
        '10. Actual conditions: 八组已测，旧Run六组只读引用，新Run两组。',
        '11. Metrics: 真正AP50/AP50:95、AR、尺寸与逐类指标，CSV均保留0—1原值。','',
        '| 模型 | 输入 | JPEG | AP50 | AP50:95 | AR50:95 |','|---|---:|---:|---:|---:|---:|']
    for r in rows:
        if r['size']=='all':text.append(f'|{r["checkpoint"]}|{r["resolution"]}|{r["JPEG"]}|{r["map50"]:.6f}|{r["map50_95"]:.6f}|{r["ar50_95"]:.6f}|')
    text+=['','12. Intervals: 200次相同原图索引配对重采样，点态百分位95%区间；固定权重，不控制训练seed、航次聚类或多重比较。逐类仅有点估计，未伪造逐类区间。','',
        '| 对比 | 尺寸 | 差值 | 95% CI |','|---|---|---:|---|']
    for r in effects:
        if r['metric']=='map50_95' and r['contrast'].startswith(('(', 'DID')):
            text.append(f'|{r["checkpoint"]}: {r["contrast"]}|{r["size"]}|{r["estimate"]:.6f}|[{r["CI95_low"]:.6f},{r["CI95_high"]:.6f}]|')
    text += ['',f'13. Actual runtime: 新worker {costs["new_worker_seconds"]:.3f}秒，上限1800秒；旧worker {costs["old_worker_seconds"]:.3f}秒，旧预算未改。计时边界和显存见COSTS.json。',
        '14. Supported: 仅支持本548图、两固定checkpoint、两JPEG端点的条件效应和交互比较。结果方向、零附近区间均如实保留。',
        '15. Unsupported/blockers: 不推断架构因果、Q50曲率、独立测试、训练seed显著性或全空间gap。Grouped ML、独立test-dev和匹配增强仍有准入缺口。',
        '16. Evidence/reproduction: EVIDENCE_SOURCES.json逐项指向原Run与SHA；CPU复算命令为factorial_analysis.py --completion-run '+summary['source_completion_RunID'] if 'source_completion_RunID' in summary else '16. Evidence: EVIDENCE_SOURCES.json逐项指向原Run与SHA；运行factorial_analysis.py --completion-run <missing-corners-RunID>可创建新的CPU汇总Run。',
        '', '## 导师汇报四栏','',
        'PROBLEM：检测器设置与图像退化存在条件交互，不能由平均AP代替。',
        'WHY IS IT A PROBLEM?：观察同权重640→1280收益随JPEG质量的实际变化，见INTERACTION_EFFECTS.csv。',
        'SOLUTION：导师对齐双UVL、最小四角点、全548图、同图配对区间和原预测复算。',
        'WHY IS IT A GOOD SOLUTION?：八个角点均可追溯，比较不混图像或评价器；不能由此宣称完整自动学习/增强闭环已完成。']
    path=ROOT/'datasets/reportcn'/('vm36_supervisor_reproduction_八条件结果_'+out.name.split('_')[-1]+'.md')
    write_atomic(path,'\n'.join(text)+'\n',out);save(out/'REPORT_REFERENCE.json',source(path),out)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--completion-run',required=True);args=parser.parse_args()
    if Path(args.completion_run).name!=args.completion_run:raise ValueError('Expected a plain RunID.')
    analyze(ROOT/'datasets/local_runs'/EXPERIMENT/args.completion_run)
