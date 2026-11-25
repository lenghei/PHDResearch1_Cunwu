"""Audit accepted measurements and compare explicit follow-up estimands on CPU."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import time
import numpy as np
from protocol import ROOT,HERE,EXPERIMENT,FMP,FMD,load,save,source,sha256,write_atomic
from common import seal,verify_hashes
from full_development import PREP
from historical_review import table,csv_file
from main.pipeline import new_run

ANALYSIS_ID='vm36_supervisor_reproduction_joint_results_20261110T091428_8b6672a2'
ALIAS_ID='vm36_supervisor_reproduction_joint_results_20261110T091110_94cc7886'
ACCEPTANCE_ID='vm36_supervisor_reproduction_joint_acceptance_20261110T091448_7d26f93a'
SEVERE_ID='p_8d98aefd645e8b60'
THRESHOLD=.10
LATENCY=100


def missing_product(pids,settings,observed):
    return [(p,d) for p in sorted(pids) for d in sorted(settings) if (p,d) not in observed]


def gap_bounds(states,expected=4):
    if len(states)!=expected:raise ValueError('Expected all declared settings, including unknowns.')
    if 'PASS' in states:return 'COVERED_TESTED_SUBSET'
    if all(s=='FAIL' for s in states):return 'GAP_TESTED_SUBSET'
    return 'UNRESOLVED'


def resumable_pairs(plan,completed):
    allowed={r['id']:r for r in plan}
    if len(allowed)!=len(plan) or len({r['pair_id'] for r in completed})!=len(completed):
        raise ValueError('Duplicate pair identity.')
    for row in completed:
        if row['pair_id'] not in allowed or row.get('status')!='SEALED_FULL548' or row.get('images')!=548:
            raise ValueError('Incomplete or unplanned output cannot be resumed as completed.')
        if row['configuration_sha256']!=hashlib.sha256(json.dumps(allowed[row['pair_id']],sort_keys=True).encode()).hexdigest():
            raise ValueError('Changed configuration.')
    done={r['pair_id'] for r in completed}
    return [r for r in plan if r['id'] not in done]


def audit_inputs(out,root):
    current=root/ANALYSIS_ID;alias=root/ALIAS_ID;acceptance=root/ACCEPTANCE_ID
    verify_hashes(current,('summary.json','CONDITION_METRICS.csv','EVIDENCE_SOURCES.json','PAIR_STATES.csv'))
    verify_hashes(alias,('CONDITION_METRICS.csv','PER_CLASS_METRICS.csv','CONFIDENCE_INTERVALS.csv'))
    verify_hashes(acceptance,('ACCEPTANCE.json','PAIR_CHECKS.json'))
    for filename in ('CONDITION_METRICS.csv','PER_CLASS_METRICS.csv','CONFIDENCE_INTERVALS.csv'):
        if sha256(current/filename)!=sha256(alias/filename):raise ValueError('Summary aliases differ.')
    refs=load(current/'EVIDENCE_SOURCES.json');accepted={};recipes={};settings={};identity_keys=set();rows=[]
    for record in refs['accepted_pairs']:
        folder=Path(record['folder'])
        checks=verify_hashes(folder,('configuration.json','metrics.json','bootstrap.json','predictions.jsonl.gz','ground_truth.json','verification.json','timing.json'))
        spec=load(folder/'configuration.json');metrics=load(folder/'metrics.json');boot=load(folder/'bootstrap.json')
        if metrics['images']!=548 or not load(folder/'verification.json')['COCO_recomputed_from_raw']:
            raise ValueError('Missing full-split raw replay.')
        p=spec['p']['id'];d=spec['d']['id'];key=(p,d)
        if key in accepted:raise ValueError('Repeated measurement.')
        recipes[p]=spec['p'];settings[d]=spec['d'];accepted[key]={'spec':spec,'folder':folder,'metrics':metrics,'bootstrap':boot,'timing':load(folder/'timing.json')}
        identity=(p,d,spec['d']['sha256'],checks['ground_truth.json'],checks['predictions.jsonl.gz'])
        if identity in identity_keys:raise ValueError('Repeated raw measurement identity.')
        identity_keys.add(identity)
        rows.append({'p_id':p,'d_id':d,'source_RunID':folder.parent.parent.name,'prediction_SHA256':checks['predictions.jsonl.gz'],
            'GT_SHA256':checks['ground_truth.json'],'metric_SHA256':checks['metrics.json'],'pair_folder':str(folder)})
    if (len(accepted),len(recipes),len(settings))!=(26,20,4):raise ValueError('Unexpected accepted population.')
    csv_file(out/'MEASUREMENT_IDENTITIES.csv',rows)
    save(out/'REPLICATION_AUDIT.json',{'measured_pairs':26,'image_records':26*548,'distinct_degradations':20,'detector_settings':4,
        'summary_versions':2,'independent_experimental_replications':False,'raw_measurement_identities':len(identity_keys),
        'reason':'The two summaries reference the same predictions and their metric CSV bytes are identical.',
        'current':source(current/'SHA256SUMS.txt'),'alias':source(alias/'SHA256SUMS.txt'),'acceptance':source(acceptance/'SHA256SUMS.txt')},out)
    return current,accepted,recipes,settings


def matched_relations(out,accepted,recipes,settings):
    from full_analysis import paired_effect
    rows=[];effects=[]
    groups=defaultdict(list)
    for pid,p in recipes.items():groups[tuple(x for x in sorted(p['leaves']) if not x.startswith('JPEG_q'))].append(pid)
    for context,pids in groups.items():
        q25=next((p for p in pids if 'JPEG_q25' in recipes[p]['leaves']),None)
        q75=next((p for p in pids if 'JPEG_q75' in recipes[p]['leaves']),None)
        if not q25 or not q75:continue
        for ck in ('ck01','ck05'):
            keys=[(q25,ck+'_Infer1280'),(q25,ck+'_Infer640'),(q75,ck+'_Infer1280'),(q75,ck+'_Infer640')]
            complete=all(k in accepted for k in keys)
            rows.append({'relation':'Resolution x JPEG endpoints','checkpoint':ck,'context':'|'.join(context),
                'required_points':4,'observed_points':sum(k in accepted for k in keys),'directly_estimable':complete,
                'classification':'MATCHED_CONTROLLED_CONTRAST' if complete else 'NOT_ESTIMABLE'})
            if complete:
                term=[(weight,accepted[k]['metrics'],accepted[k]['bootstrap']) for weight,k in zip((1,-1,-1,1),keys)]
                for size in ('all','tiny','small','medium','large'):
                    for metric in ('map50_95','map50','ar50_95'):
                        value=paired_effect(term,size,metric)
                        effects.append({'checkpoint':ck,'context':'|'.join(context),'size':size,'metric':metric,**value})
    csv_file(out/'IDENTIFIABILITY_AUDIT.csv',rows)
    csv_file(out/'VERIFIED_MATCHED_EFFECTS.csv',effects)
    save(out/'CLAIM_BOUNDARIES.json',{'matched':'Original JPEG endpoint contrasts only; selected compound bundle contrasts require additional corners.',
        'descriptive':'Observed compound configurations and selected-pair failure counts; no isolated factor attribution.',
        'predictive':'ML_INSUFFICIENT_EVIDENCE; no independent-image held-out score.',
        'population':'No probability sample; purposive pairwise points cannot estimate full-space gap prevalence.',
        'causal':'No architecture/component causal effect, training-seed effect or calibrated sensor mechanism.'},out)
    return effects


def followup_designs(out,root,current,accepted,recipes,settings):
    from sampling import models
    from alignment import parameters
    pm,_,_=models();pids=set(recipes);dids=set(settings)
    all_missing=missing_product(pids,dids,accepted)
    if len(all_missing)!=54:raise ValueError('The20x4 complement is not54.')
    scenario=[r for r in table(current/'ALL_SETTING_OUTCOMES.csv') if float(r['tau'])==THRESHOLD and float(r['L_cached_ms'])==LATENCY]
    by_p=defaultdict(dict)
    for row in scenario:by_p[row['p_id']][row['d_id']]=row['state']
    unresolved={p for p,s in by_p.items() if gap_bounds(list(s.values()))=='UNRESOLVED'}
    decisive_missing=missing_product(unresolved,dids,accepted)
    rows=[]
    for p in sorted(pids):
        states=by_p[p]
        rows.append({'p_id':p,'features':'|'.join(recipes[p]['leaves']),'tau_tiny_AP':THRESHOLD,'L_cached_ms':LATENCY,
            'subset_state':gap_bounds(list(states.values())),**states,'missing_pairs':sum((p,d) not in accepted for d in dids)})
    csv_file(out/'GAP_SETTING_AUDIT.csv',rows)
    covered=sum(r['subset_state']=='COVERED_TESTED_SUBSET' for r in rows);gaps=sum(r['subset_state']=='GAP_TESTED_SUBSET' for r in rows)
    save(out/'FINITE20_GAP_BOUNDS.json',{'degradations':20,'covered':covered,'confirmed_gaps':gaps,'unresolved':len(unresolved),
        'gap_count_lower':gaps,'gap_count_upper':gaps+len(unresolved),'population_gap_estimate':None,
        'scenario':{'metric':'tiny_AP50_95','tau':THRESHOLD,'L_cached_ms':LATENCY,'scope':'Sensitivity; deployment requirement remains under review.'},
        'uncertainty':'Even all80 measured points may retain UNCERTAIN intervals. Measuring missing points does not guarantee a binary label.'},out)
    pure={'leaves':['Normal']};pure['id']=pm.configuration_id(pure['leaves']);pure['attributes']=parameters(FMP,pure['leaves'])
    recipes={**recipes,pure['id']:pure}
    gaussian=next(p for p in pids if set(recipes[p]['leaves'])=={'Normal','Gauss_s15'})
    if SEVERE_ID not in pids:raise ValueError('Expected severe-compound witness is absent.')
    reference=next(p for p in pids if set(recipes[p]['leaves'])=={'Normal','JPEG_q25'})
    reference75=next(p for p in pids if set(recipes[p]['leaves'])=={'Normal','JPEG_q75'})
    witness_missing=missing_product({SEVERE_ID},dids,accepted)
    gaussian_one=missing_product({pure['id'],gaussian},{'ck01_Infer640','ck01_Infer1280'},accepted)
    gaussian_both=missing_product({pure['id'],gaussian},dids,accepted)
    definitions=[
        ('A_EXISTING_JPEG',[],{reference,reference75},'Recheck the already matched JPEG endpoint interaction.','No new scientific measurements.','Already identifiable; development only.'),
        ('A_GAUSSIAN_ONE',gaussian_one,{pure['id'],gaussian},'Does Gaussian sigma15 change the resolution effect within ck01?','Two levels of Gaussian presence x two resolutions, other factors fixed.','One fixed checkpoint; no model-generation comparison.'),
        ('A_GAUSSIAN_BOTH',gaussian_both,{pure['id'],gaussian},'Does the Gaussian-by-resolution contrast differ between the two checkpoints?','Two checkpoints x two resolutions x Gaussian absent/present.','Checkpoint bundles; no architecture causality or independent test.'),
        ('B_ONE_CONTEXT',witness_missing,{SEVERE_ID,reference},'Complete four settings for one severe compound; compare its resolution effect with the existing Q25 reference.','All four D settings at the chosen compound, plus reused four-setting reference.','Compound-vs-reference bundle; cannot isolate five simultaneous degradation factors or estimate prevalence.'),
        ('B_DECISIVE_ONLY',decisive_missing,unresolved,'Resolve current20-state gap classification where no PASS is already known.','Missing settings only for currently unresolved P; stop after any PASS if gap-only.','Threshold-specific adaptive diagnostic; remaining CI uncertainty can prevent classification.'),
        ('B_COMPLETE20',all_missing,pids,'Describe all20 chosen degradation configurations across all four settings.','Complete20x4 repeated-measures grid.','20 purposive P states, not a uniform population sample or full FM_D.')]
    resource=load(current/'RESOURCE_RESULT.json');costs=[r['complete_wall_seconds'] for r in resource['new_pair_costs']]
    cached_costs=[r['timing']['wall_seconds'] for r in accepted.values()]
    cost_reference=max(costs+cached_costs);screening_per_pair=cost_reference*1.25+90
    pair_max=max(sum(p.stat().st_size for p in r['folder'].iterdir() if p.is_file()) for r in accepted.values())
    pair_upper=max(pair_max*2,64*1024**2)
    gt=load(Path(load(Path(load(current/'EVIDENCE_SOURCES.json')['batch_protocol']['path']))['GT']['path']))
    raw_pixel_bytes=sum(r['width']*r['height']*3 for r in gt['records'])
    widths=[r['AP_CI_high']-r['AP_CI_low'] for r in load(current/'summary.json')['metrics']]
    plans=[];manifest=[]
    for identity,missing,chosen,question,rationale,limitations in definitions:
        pairs=[]
        for p,d in missing:
            if not pm.valid_leaves(recipes[p]['leaves']):raise ValueError('Illegal proposed degradation.')
            spec={'id':p+'__'+d,'p':recipes[p],'d':settings[d],'images':548,'status':'PROPOSED_NOT_RUN'}
            pairs.append(spec);manifest.append({'design_id':identity,'pair_id':spec['id'],'p_id':p,'d_id':d,
                'features':'|'.join(spec['p']['leaves']),'attributes':json.dumps(spec['p']['attributes'],sort_keys=True),
                'checkpoint_SHA256':spec['d']['sha256'],'resolution':spec['d']['resolution'],'images':548,'status':'NOT_RUN'})
        save(out/(identity+'_MANIFEST.json'),{'design_id':identity,'pairs':pairs,'question':question,'new_measurements':0},out)
        plans.append({'design_id':identity,'question':question,'new_pairs':len(missing),'new_saved_image_records':548*len(missing),
            'degradation_configurations':len(chosen),'detector_settings':2 if identity=='A_GAUSSIAN_ONE' else 4,
            'necessary_setting_IDs':'ck01_Infer640|ck01_Infer1280' if identity=='A_GAUSSIAN_ONE' else '|'.join(sorted(dids)),
            'necessary_degradation_IDs':'|'.join(sorted(chosen)),
            'sampling_rationale':rationale,'scientific_limitations':limitations,
            'screening_seconds':len(missing)*screening_per_pair,'screening_disk_bytes':len(missing)*pair_upper,
            'precision':'No prospective CI width/power claim; full548 paired draws. Existing CI widths are reference only.',
            'new_GPU_authorization':'NOT_PROVIDED'})
    csv_file(out/'CANDIDATE_DESIGNS.csv',plans);csv_file(out/'MISSING_PAIRS.csv',manifest)
    if (len(witness_missing),len(gaussian_one),len(gaussian_both))!=(3,3,7):raise ValueError('Minimal corner counts differ.')
    proposal={'status':'DESIGN_FROZEN_RESOURCE_BLOCKED','recommended':'B_ONE_CONTEXT','new_pairs':3,'images_per_pair':548,
        'target_P':recipes[SEVERE_ID],'reference_P':recipes[reference],
        'selection':'Retrospective targeted development witness: existing severe-compound failure and existing matched Q25 reference; not response-blind sampling.',
        'primary':'Finite four-setting outcome at Tiny AP tau0.10/L100ms, with frozen other sensitivity scenarios retained.',
        'secondary':'Per-checkpoint Resolution x compound-context DID and checkpoint difference, using all eight required corners.',
        'context_warning':'A multi-feature bundle contrast; no individual noise/illumination/defocus attribution.',
        'metrics':['AP50','AP50_95','AR','tiny_AP','small_AP','medium_AP','large_AP','per_class_AP'],
        'statistics':'Same548 image positions;200 paired image draws, inherited seed; pointwise95% CIs, no seed/flight inference or independent confirmation.',
        'budget_proposal_seconds':3600,'recorded_authorized_GPU_budget_seconds':None,
        'conservative_three_pair_screening_seconds':3*screening_per_pair,
        'execution_gate':'No GPU or new training until an explicit new finite budget is recorded. Do not call joint_batch.py.',
        'safe_resume':'Reuse only sealed full548 pairs with unchanged P/D, GT, source pixels, model/protocol and raw hashes. Incomplete outputs stay incomplete; no claimed exact mid-forward/cross-reboot recovery.',
        'worker_backend':'NEW_VERSION_REQUIRED; this CPU task does not start a worker.'}
    save(out/'NEXT_MATCHED_PROTOCOL.json',proposal,out)
    baseline=[]
    for (p,d),item in accepted.items():
        if p not in (SEVERE_ID,reference):continue
        for area in item['metrics']['per_area']:
            ci=next(v['ci95'] for v in item['bootstrap']['intervals'] if v['size']==area['size'] and v['metric']=='map50_95')
            baseline.append({'p_id':p,'d_id':d,**area,'AP95_CI_low':ci[0],'AP95_CI_high':ci[1],
                'metrics_SHA256':sha256(item['folder']/'metrics.json')})
    csv_file(out/'FOLLOWUP_BASELINES.csv',baseline)
    save(out/'COST_AND_DISK.json',{'complete_new_pairs':len(costs),'measured_complete_pair_seconds':costs,
        'complete_pair_min_median_max':list(map(float,(min(costs),np.median(costs),max(costs)))),
        'conservative_reference_max_seconds':cost_reference,'screening_per_pair_seconds':screening_per_pair,
        'screening_method':'max observed full/new or cached/old pair *1.25 +90s; a screening scenario, not a prediction or guarantee.',
        'old_new_timing_boundary':'Old8 excludes pixel generation; new18 includes it. No speedup is inferred.',
        'max_saved_pair_bytes':pair_max,'screening_saved_pair_bytes':pair_upper,'raw_full548_pixel_bytes':raw_pixel_bytes,
        'disk_policy':'Reuse verified20 recipe caches. New clean control can stream fixed-size blocks; do not require another full image archive.',
        'free_disk_bytes':shutil.disk_usage(ROOT).free,'reserve_bytes':8*1024**3,
        'measured_AP_CI_width_min_median_max':list(map(float,(min(widths),np.median(widths),max(widths)))),
        'prospective_CI_precision':'NOT_KNOWN; do not transfer these widths to unmeasured corners.'},out)
    return proposal,plans


def uniform_planning(out):
    rows=[]
    for population,scope in ((1872,'Parameter-admitted FM_P only'),(46368,'Full logical FM_P; missing renderer bindings')):
        for half_width in (.05,.075,.10):
            z=1.959963984540054;v=.25
            n=math.ceil(population*z*z*v/(half_width**2*(population-1)+z*z*v))
            rows.append({'population':population,'scope':scope,'illustrative_half_width':half_width,'confidence':.95,
                'SRSWOR_degradations':n,'maximum_four_setting_pairs':4*n,'inclusion_probability':n/population,
                'selected':False,'assumptions':'Binary decisive subset gap; worst-case normal-approximation planning with finite correction, not guaranteed CI precision.',
                'uncertain_policy':'Report lower/upper gap bounds; uncertain labels are not failures.'})
    csv_file(out/'UNIFORM_PRECISION_SENSITIVITY.csv',rows)


def report(out,current,proposal,plans,effects):
    lines=['# VM36：26组后的可辨识性与下一步设计','',
        'PROBLEM：从覆盖转向可验证的交互和有限检测器子集gap；本轮只做CPU核验。',
        'WHY：20个退化配置只测26个P-D点，二阶覆盖完整不等于20×4响应完整。',
        'SOLUTION：核对原预测/GT哈希、两版汇总同源性、真实角点、逐设置状态，再比较最少新增点数。',
        'WHY IS IT A GOOD SOLUTION：缺失点明确列出，不用回归补角点，不把未知当失败。','',
        '原始26组不重跑、不修改；8b6672a2与94cc7886为同一批测量的汇总版本，不是两次重复实验。',
        '完整548图、native10、FP32、历史FT1280/seed42；准确模型SHA和参数在MEASUREMENT_IDENTITIES.csv及各候选manifest。','',
        '| 设计 | 新增组数 | 主要用途 | 估算worker小时 |','|---|---:|---|---:|']
    for p in plans:lines.append(f'|{p["design_id"]}|{p["new_pairs"]}|{p["question"]}|{p["screening_seconds"]/3600:.2f}|')
    lines+=['','表中时间为实测成本加余量的筛查，不是执行保证；没有新GPU结果。',
        '建议B_ONE_CONTEXT：补同一重度复合退化的3个缺失设置；完整四设置与已有Q25参考可支持复合条件整体×分辨率对比。',
        '该点按已见失败选择，属于目标化开发诊断。不能拆成各单因素因果，也不能据此估总体gap比例。',
        '全20×4需额外54组；若只回答已存在PASS之外的阈值特定gap问题，可跳过已被PASS证明覆盖的P。具体缺失数见候选表。',
        '即使补完所有设置，跨阈值CI仍可能是UNCERTAIN，不能保证20个P均得到二元标签。',
        '独立均匀总体估计是另一设计；精度敏感性表未选定样本量，未把这些目的混在一起。','',
        'ML_INSUFFICIENT_EVIDENCE：共享548张历史选模val；没有独立图像/模型选择边界和足够的类别支持。没有训练或独立保留集分数。',
        '现有JPEG配对结果只支持两个端点、固定权重和本图像集；已重新从保存的配对Bootstrap值复算。',
        'test-dev/UAVDT与训练数据工程检查使用本轮独立CPU记录；本报告不宣称它们已经准入。',
        '新GPU预算尚未提供。建议三组完整548图，worker总上限3600秒，包含预热、缓存、推理、评价和200次Bootstrap；当前状态RESOURCE_BLOCKED。',
        '证据：'+str(out),'原26组：'+str(current),
        '复现：.venv/Scripts/python.exe -B main/experiments/experiments_test/vm36_supervisor_reproduction/matched_followup.py','']
    note=ROOT/'datasets/reportcn'/('vm36_supervisor_reproduction_26组后设计审计_'+out.name.split('_')[-1]+'.md')
    write_atomic(note,'\n'.join(lines),out);save(out/'REPORT_REFERENCE.json',source(note),out)
    captions=['# Source-driven captions','',
        'Resolution and JPEG endpoint contrasts. COCO AP50:95 uses548 validation images and fixed checkpoints. Intervals use200 paired image resamples. JPEG Q75 is not CLEAN. The checkpoint-difference interval includes zero.',
        '', 'Measured configuration coverage. The26 accepted P-D measurements witness1033 feasible binary pair states within the1872-by-four projection. This does not measure all7488 configurations or fault-detection coverage.',
        '', 'Conditional subset gap. Outcomes use Tiny AP confidence bounds and cached-input pipeline latency. Unmeasured settings remain unknown; counts describe the20 chosen degradation configurations only.',
        '', 'No additional figures or measured values were generated for the proposed follow-up. Sources: VERIFIED_MATCHED_EFFECTS.csv, GAP_SETTING_AUDIT.csv and the accepted summary.']
    write_atomic(out/'FIGURE_CAPTIONS_EN.md','\n'.join(captions)+'\n',out)


def execute():
    started=time.perf_counter();root=ROOT/'datasets/local_runs'/EXPERIMENT
    out,_=new_run({'output_root':str(root)},EXPERIMENT,'matched_followup_design')
    shutil.copyfile(Path(__file__),out/'executed_source.py')
    current,accepted,recipes,settings=audit_inputs(out,root)
    effects=matched_relations(out,accepted,recipes,settings)
    proposal,plans=followup_designs(out,root,current,accepted,recipes,settings)
    uniform_planning(out);report(out,current,proposal,plans,effects)
    save(out/'inputs.json',{'accepted_analysis':source(current/'SHA256SUMS.txt'),
        'plan':source(ROOT/'datasets/research_archive/supervisor/Research_plan_Cunwu_UAV_gap.pdf'),
        'FMP':source(FMP),'FMD':source(FMD),'source':source(Path(__file__))},out)
    write_atomic(out/'diary.md','# Post26 design audit\n\nVerified raw-evidence identities and summary aliases. Recomputed available paired contrasts from saved bootstrap draws. Compared controlled contrasts, conditional subset completion and probability sampling separately. No new detector output or training.\n',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'POST26_DESIGN_AUDIT','status':'COMPLETE_WITH_LIMITATIONS',
        'wall_seconds':time.perf_counter()-started,'new_GPU_measurements':0,'new_training_runs':0},out,replace=True)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>VM36 post26 design audit</h1><p>CPU only; no new measurements.</p><a href="CANDIDATE_DESIGNS.csv">Alternatives</a> · <a href="NEXT_MATCHED_PROTOCOL.json">Proposed next study</a> · <a href="GAP_SETTING_AUDIT.csv">Observed outcomes</a>',out)
    seal(out)
    from joint_analysis import refresh
    refresh(current);print(out,flush=True)


if __name__=='__main__':execute()
