"""Summarize accepted joint measurements without treating missing pairs as failures."""
import argparse
from collections import Counter
from itertools import combinations
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
from full_analysis import tested_subset_state
from joint_batch import DESIGN_ID,ANALYSIS_ID
from main.pipeline import new_run


def projection_features(spec):
    d=spec['d']
    return spec['p']['leaves']+(['D_Checkpoint_ck01'] if d['checkpoint']=='ck01' else [])+(['D_Infer1280'] if d['resolution']==1280 else [])


def finish(batch,prior_summary=None):
    started=time.perf_counter();root=batch.parent;analysis=root/ANALYSIS_ID;design=root/DESIGN_ID
    if load(batch/'run.json')['status']=='RUNNING':raise ValueError('The bounded worker is still running.')
    verify_hashes(analysis,('EVIDENCE_SOURCES.json','EVIDENCE_ACCEPTANCE.json'));verify_hashes(design,('JOINT_COVERAGE_PLAN.csv','DESIGN_ACCEPTANCE.json'))
    plan=load(batch/'EXECUTION_PROTOCOL.json');refs=load(analysis/'EVIDENCE_SOURCES.json')
    original=load(Path(refs['original_protocol']['path']));specs={s['id']:s for s in original['pairs']}
    specs.update({s['id']:s for s in plan['pairs']})
    expected_images=sorted(r['id'] for r in load(Path(plan['GT']['path']))['records'])
    accepted={};records=[];metrics_rows=[];class_rows=[];ci_rows=[];new_count=0
    locations={r['pair_id']:Path(r['pair_folder']) for r in refs['conditions']}
    for spec in plan['pairs']:
        folder=batch/'pairs'/spec['id']
        if (folder/'SHA256SUMS.txt').exists():locations[spec['id']]=folder;new_count+=1
    for identity,folder in locations.items():
        check=verify_hashes(folder,('metrics.json','bootstrap.json','timing.json','configuration.json','predictions.jsonl.gz','ground_truth.json','verification.json','image_manifest.jsonl'))
        spec=load(folder/'configuration.json');m=load(folder/'metrics.json');b=load(folder/'bootstrap.json');t=load(folder/'timing.json')
        if spec!=specs[identity] or m['images']!=548 or check['ground_truth.json']!=plan['GT']['sha256']:
            raise ValueError('Pair identity or full-split GT differs.')
        verification=load(folder/'verification.json')
        if not verification['COCO_recomputed_from_raw'] or verification['protocol_sha256']!=sha256(folder.parent.parent/'EXECUTION_PROTOCOL.json'):
            raise ValueError('Missing matching raw replay evidence.')
        if b['draws']['seed']!=plan['seed'] or b['draws']['samples']!=200 or b['draws']['image_ids']!=expected_images:
            raise ValueError('Bootstrap identity differs.')
        manifests=[json.loads(line) for line in (folder/'image_manifest.jsonl').read_text().splitlines()]
        if sorted(r['image_id'] for r in manifests)!=expected_images or any(r['tensor']!={'NCHW':[1,3,spec['d']['resolution'],spec['d']['resolution']],'dtype':'torch.float32'} for r in manifests):
            raise ValueError('Image population or actual input tensor differs.')
        for interval in b['intervals']:
            values=[v for v in b['draws']['values'][interval['size']][interval['metric']] if v is not None]
            if len(values)!=interval['n_valid'] or interval['n_samples']!=200:
                raise ValueError('Saved bootstrap sample count differs.')
            if values and not np.allclose(np.quantile(values,[.025,.975]),interval['ci95'],atol=1e-14,rtol=0):
                raise ValueError('Interval differs from saved draws.')
            ci_rows.append({'pair_id':identity,**interval})
        accepted[identity]=(spec,m,b,t)
        records.append({'pair_id':identity,'source_RunID':folder.parent.parent.name,'folder':str(folder),
            'manifest':source(folder/'SHA256SUMS.txt'),'metrics':source(folder/'metrics.json'),
            'prediction':source(folder/'predictions.jsonl.gz'),'GT':source(folder/'ground_truth.json')})
        for row in m['per_area']:
            ci=next(r['ci95'] for r in b['intervals'] if r['size']==row['size'] and r['metric']=='map50_95')
            metrics_rows.append({'pair_id':identity,'checkpoint':spec['d']['checkpoint'],'resolution':spec['d']['resolution'],
                'features':'|'.join(spec['p']['leaves']),'images':548,**row,'AP_CI_low':ci[0],'AP_CI_high':ci[1],
                'mean_cached_pipeline_ms':t['mean_pipeline_ms'],'peak_allocated_bytes':t['peak_allocated_bytes'],
                'source_RunID':folder.parent.parent.name,'metrics_SHA256':check['metrics.json']})
        for row in m['per_class']:class_rows.append({'pair_id':identity,**row,'metrics_SHA256':check['metrics.json']})
    out,_=new_run({'output_root':str(root)},EXPERIMENT,'joint_results')
    csv_file(out/'CONDITION_METRICS.csv',metrics_rows);csv_file(out/'PER_CLASS_METRICS.csv',class_rows);csv_file(out/'CONFIDENCE_INTERVALS.csv',ci_rows)
    if prior_summary:
        files=('CONDITION_METRICS.csv','PER_CLASS_METRICS.csv','CONFIDENCE_INTERVALS.csv')
        verify_hashes(prior_summary,files)
        if any(sha256(out/name)!=sha256(prior_summary/name) for name in files):
            raise ValueError('A report-only summary revision changed metrics.')
        save(out/'SUMMARY_REVISION.json',{'prior_summary':source(prior_summary/'SHA256SUMS.txt'),
            'reason':'Use the final source version and corrected Markdown table cells.',
            'metric_CSV_bytes_unchanged':True,'new_GPU_inferences':0},out)
    statuses=[]
    for spec in plan['pairs']:
        identity=spec['id'];folder=batch/'pairs'/identity
        status='ACCEPTED' if identity in accepted else 'INCOMPLETE' if folder.exists() else 'NOT_RUN'
        statuses.append({'pair_id':identity,'status':status,'images_accepted':548 if identity in accepted else 0})
    csv_file(out/'NEW_PAIR_STATUS.csv',statuses)
    coverage=coverage_table(out,design,accepted,specs)
    states,gaps,static=empirical_tables(out,plan,accepted)
    failure_associations(out,accepted,states)
    learning=learning_feasibility(out,plan,accepted,states)
    save(out/'EVIDENCE_SOURCES.json',{'accepted_pairs':records,'batch_protocol':source(batch/'EXECUTION_PROTOCOL.json'),
        'design':source(design/'JOINT_COVERAGE_PLAN.csv'),'eight_pair_acceptance':source(analysis/'SHA256SUMS.txt')},out)
    worker=load(batch/'worker_result.json') if (batch/'worker_result.json').exists() else {'status':'PARTIAL'}
    supervisor=load(batch/'supervisor.json')
    status='VALIDATED' if new_count==18 else 'PARTIAL'
    summary={'schema':'VM36_READ_ONLY_SUMMARY_V1','RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'JOINT_COVERAGE_RESULTS',
        'status':status,'joint_study':True,'scope':'DEVELOPMENT_ONLY','validated_validation_images':548,
        'new_measured_pairs':new_count,'combined_measured_pairs':len(accepted),'formal_measured_pairs':0,
        'new_saved_image_predictions':new_count*548,'warmup_predictor_calls_per_accepted_pair':3,
        'coverage':coverage,'worker_wall_seconds':supervisor['actual_child_supervision_seconds'],
        'metrics':[r for r in metrics_rows if r['size']=='all'],'gap_sensitivity':gaps,
        'learning':learning,'source_hashes':{'protocol':source(batch/'EXECUTION_PROTOCOL.json'),'design':source(design/'SHA256SUMS.txt')},
        'gates':{'RQ1':'Restricted coverage verified; original logical models unchanged.',
            'RQ2':learning['status'],'RQ3':'Conditional tested-subset sensitivity only; no population gap estimate.',
            'RQ4':'RESOURCE_BLOCKED; no matched three-seed augmentation training.',
            'independent_testdev':'DATASET_BLOCKED','UAVDT':'Original GT/ignore recovered; mixed SOT separated; conversion edges documented; external evaluation not admitted.'}}
    save(out/'summary.json',summary,out)
    save(out/'result.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':summary['mode'],'status':status,
        'summary_canonical_sha256':hashlib.sha256(json.dumps(summary,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()},out)
    components=[]
    for spec,m,b,t in accepted.values():
        folder=batch/'pairs'/spec['id']
        if not folder.exists():continue
        components.append({'pair_id':spec['id'],'forward_profiler_seconds':sum(row['forward_ms'] for row in t['samples'])/1000,
            'synchronized_predictor_wall_seconds':sum(row['predict_ms'] for row in t['samples'])/1000,
            'cached_pipeline_wall_seconds':sum(row['pipeline_ms'] for row in t['samples'])/1000,
            'cache_generation_seconds':load(folder/'CACHE_COST.json')['seconds'],
            **{key:t[key] for key in ('evaluation_seconds','replay_seconds','bootstrap_seconds','peak_allocated_bytes','peak_reserved_bytes')}})
    save(out/'RESOURCE_RESULT.json',{'supervisor':supervisor,'worker':worker,
        'new_pair_costs':load(batch/'progress.json')['completed'] if (batch/'progress.json').exists() else [],
        'accepted_pair_saved_prediction_images':new_count*548,'accepted_pair_warmup_calls':new_count*3,
        'warmup_scope':'Three first-image predictor calls per accepted pair; separately excluded from the saved548-image population. Any incomplete pair requires its own log review.',
        'disk_bytes_batch':sum(p.stat().st_size for p in batch.rglob('*') if p.is_file()),
        'disk_bytes_cache':sum(p.stat().st_size for p in (ROOT/'datasets/cache'/EXPERIMENT/batch.name).rglob('*') if p.is_file()),
        'filesystem_free_bytes_at_analysis':shutil.disk_usage(ROOT).free,
        'new_pair_components':components,'GPU_temperature_peak':'NOT_CONTINUOUSLY_RECORDED',
        'limitations':'Forward profiler is nested within predictor wall time and cached pipeline time; do not add nested components. These are not GPU occupancy or energy measurements. Worker includes generation, evaluation and bootstrap. Old pair wall times exclude cache generation and are not a speedup baseline.'},out)
    save(out/'inputs.json',{'batch':source(batch/'EXECUTION_PROTOCOL.json'),'analysis_source':source(Path(__file__))},out)
    save(out/'MATCHED_CONTRAST_REFERENCES.json',{'source':source(analysis/'INTERACTION_EFFECTS.csv'),
        'scope':'The original eight JPEG endpoint corners support matched resolution-by-JPEG contrasts. New coverage points with different P recipes do not create matched detector contrasts.',
        'new_contrasts':'NOT_ESTIMATED_WITHOUT_MATCHED_CORNERS'},out)
    auxiliary={}
    for mode,identity in (
        ('independence','vm36_supervisor_reproduction_continuation_audit_20261110T002932_d2a7e70c'),
        ('training','vm36_supervisor_reproduction_training_readiness_20261110T004858_72ea4f79'),
        ('UAVDT_source','vm36_supervisor_reproduction_uavdt_toolkit_source_20261110T004050_44d7a453'),
        ('UAVDT_reconciliation','vm36_supervisor_reproduction_uavdt_reconciliation_20261110T004456_299551e8'),
        ('UAVDT_edges','vm36_supervisor_reproduction_uavdt_boundary_review_20261110T004947_9b01a0a5')):
        auxiliary[mode]=source(root/identity/'SHA256SUMS.txt')
    save(out/'AUXILIARY_AUDITS.json',auxiliary,out)
    report_cn(out,batch,summary,metrics_rows,coverage)
    write_atomic(out/'diary.md','# Joint measurement acceptance\n\nCounted only sealed full548 pairs. Kept planned coverage, measured states, missing detector outcomes and empirical failures separate. No saturated training-accuracy or independent-image claim.\n',out)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>VM36 joint coverage results</h1><p>'+status+' · '+str(new_count)+'/18 new pairs · '+str(coverage['measured_states'])+'/1033 states in the restricted projection.</p><p><a href="CONDITION_METRICS.csv">Metrics</a> · <a href="PAIR_STATES.csv">Measured outcomes</a> · <a href="MEASURED_SUBSET_GAPS.csv">Subset gaps</a> · <a href="COVERAGE_MATRIX.csv">Coverage</a></p>',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':summary['mode'],'status':status,'CPU_seconds':time.perf_counter()-started},out,replace=True)
    if not (batch/'SHA256SUMS.txt').exists():
        save(batch/'EVIDENCE_TREE.json',{str(p.relative_to(batch)).replace('\\','/'):sha256(p) for p in sorted((batch/'pairs').rglob('*')) if p.is_file()},batch)
        write_atomic(batch/'diary.md','# Bounded joint batch\n\n'+str(new_count)+'/18 sealed complete pairs; original eight not repeated. Budget fixed at14400 seconds. See supervisor, worker result and per-pair reports.\n',batch)
        seal(batch)
    save(out/'BATCH_ACCEPTANCE_REFERENCE.json',source(batch/'SHA256SUMS.txt'),out);seal(out)
    refresh(out);print(out,flush=True)


def coverage_table(out,design,accepted,specs):
    from sampling import models
    _,_,prior=models();contract=load(design/'DESIGN_ACCEPTANCE.json');names=contract['feature_names']
    rows=table(design/'JOINT_COVERAGE_PLAN.csv')
    planned=[projection_features(specs[r['pair_id']]) for r in rows]
    masks,feasible=prior.compute_binary_pair_masks(planned,names)
    measured=0
    for row,mask in zip(rows,masks):
        if row['pair_id'] in accepted:measured|=mask
    if feasible.bit_count()!=1033:raise ValueError('Frozen projected target differs.')
    detail=[]
    for k,(left,right) in enumerate(combinations(names,2)):
        for a,b in ((0,0),(0,1),(1,0),(1,1)):
            bit=1<<(4*k+2*a+b)
            detail.append({'left':left,'right':right,'left_state':a,'right_state':b,
                'projection_feasible':bool(feasible&bit),'measured':bool(measured&bit),
                'accepted_witnesses':'|'.join(r['pair_id'] for r,m in zip(rows,masks) if m&bit and r['pair_id'] in accepted)})
    csv_file(out/'COVERAGE_MATRIX.csv',detail)
    result={'planned_pairs':26,'accepted_pairs':len(accepted),'projected_legal_pairs':7488,
        'feasible_states':1033,'measured_states':measured.bit_count(),'unknown_states':(feasible&~measured).bit_count(),
        'scope':'1872 parameter-admitted FM_P configurations x four actual settings; not full FM_D; not fault-detection coverage.'}
    save(out/'COVERAGE_ACCEPTANCE.json',result,out);return result


def empirical_tables(out,plan,accepted):
    from gap_analysis import pair_outcome
    indices=np.random.default_rng(plan['seed']).integers(0,548,(200,548));states=[];static=[]
    policies=table(PREP/'STATIC_COVERAGE_REFERENCE.csv')
    for identity,(spec,m,b,t) in accepted.items():
        ci=next(r['ci95'] for r in b['intervals'] if r['size']=='tiny' and r['metric']=='map50_95')
        latency=float(np.quantile(np.asarray([r['pipeline_ms'] for r in t['samples']])[indices].mean(axis=1),.975))
        for tau in plan['thresholds']:
            for limit in plan['latency_limits_ms']:
                state=pair_outcome(ci,latency,tau,limit)
                states.append({'pair_id':identity,'p_id':spec['p']['id'],'d_id':spec['d']['id'],'tau':tau,'L_cached_ms':limit,
                    'state':state,'Tiny_AP_CI':ci,'mean_latency_upper95_ms':latency})
                for policy in (p for p in policies if p['checkpoint']==spec['d']['checkpoint']):
                    covered=set(spec['p']['leaves']).issubset(set(policy['covered_leaves'].split('|')))
                    static.append({'pair_id':identity,'tau':tau,'L_cached_ms':limit,'policy':policy['policy'],
                        'static_covered':covered,'measured_pair_state':state,
                        'comparison_scope':'Source-policy coverage versus one detector outcome; not full-space gap truth.'})
    pids={s['p']['id'] for s in plan['pairs']}|{v[0]['p']['id'] for v in accepted.values()}
    dids={v[0]['d']['id'] for v in accepted.values()};full=[];gaps=[]
    if len(dids)!=4:raise ValueError('Tested setting definition differs.')
    for p in sorted(pids):
        for tau in plan['thresholds']:
            for limit in plan['latency_limits_ms']:
                outcomes=[]
                for d in sorted(dids):
                    matching=next((r for r in states if r['p_id']==p and r['d_id']==d and r['tau']==tau and r['L_cached_ms']==limit),None)
                    value=matching['state'] if matching else 'NOT_EVALUATED';outcomes.append(value)
                    full.append({'p_id':p,'d_id':d,'tau':tau,'L_cached_ms':limit,'state':value})
                gaps.append({'p_id':p,'tau':tau,'L_cached_ms':limit,'measured_detector_settings':sum(v!='NOT_EVALUATED' for v in outcomes),
                    'declared_detector_settings':4,'state':tested_subset_state(outcomes,4),
                    'scope':'Purposive tested P selections; not a uniform gap prevalence estimate.'})
    csv_file(out/'PAIR_STATES.csv',states);csv_file(out/'ALL_SETTING_OUTCOMES.csv',full)
    csv_file(out/'MEASURED_SUBSET_GAPS.csv',gaps);csv_file(out/'STATIC_EMPIRICAL_COMPARISON.csv',static)
    return states,gaps,static


def learning_feasibility(out,plan,accepted,states):
    from learning import fold_membership
    images=load(Path(plan['image_members']['path']));sha=sorted({r['image_SHA256'] for r in images})
    unique_groups=sorted({s['p']['id'] for s,_,_,_ in accepted.values()})
    boundary=fold_membership([{'group':unique_groups[0],'image_sha256':sha}],
        [{'group':unique_groups[1],'image_sha256':sha}])
    rows=[]
    for tau in plan['thresholds']:
        for limit in plan['latency_limits_ms']:
            selected=[r for r in states if r['tau']==tau and r['L_cached_ms']==limit]
            counts=Counter(r['state'] for r in selected)
            rows.append({'tau':tau,'L_cached_ms':limit,'measured_pairs':len(selected),'degradation_groups':len(unique_groups),
                **{k:counts[k] for k in ('PASS','FAIL','UNCERTAIN')},
                'PASS_groups':len({r['p_id'] for r in selected if r['state']=='PASS'}),
                'FAIL_groups':len({r['p_id'] for r in selected if r['state']=='FAIL'}),
                'image_disjoint_gate':boundary,'heldout_accuracy':'NOT_REPORTED'})
    csv_file(out/'LEARNING_FEASIBILITY.csv',rows)
    feature_names=sorted({name for spec,_,_,_ in accepted.values() for name in projection_features(spec)})
    features=[projection_features(spec) for spec,_,_,_ in accepted.values()]
    design=np.asarray([[1.]+[float(name in selected) for name in feature_names] for selected in features])
    pids=[spec['p']['id'] for spec,_,_,_ in accepted.values()]
    heldout=[]
    for group in unique_groups:
        train=design[[p!=group for p in pids]];test=design[[p==group for p in pids]]
        rank=int(np.linalg.matrix_rank(train))
        heldout.append({'heldout_p_id':group,'training_rows':len(train),'heldout_rows':len(test),
            'training_rank':rank,'heldout_functions_in_training_row_space':sum(int(np.linalg.matrix_rank(np.vstack([train,row])))==rank for row in test)})
    save(out/'LEARNING_DESIGN_RANK.json',{'rows':len(design),'columns':design.shape[1],
        'feature_names':['intercept']+feature_names,'rank':int(np.linalg.matrix_rank(design)),
        'leave_recipe_out':heldout,'response_fitted':False,
        'scope':'Linear feature support diagnostic only; no AP prediction score or proof of image-independent generalization.'},out)
    result={'status':'ML_INSUFFICIENT_EVIDENCE','strict_image_disjoint_gate':boundary,'evaluation_image_IDs':548,
        'unique_image_byte_hashes':len(sha),'degradation_groups':len(unique_groups),
        'reason':'All configuration AP labels use the same image population. Grouping degradation IDs alone does not satisfy the existing image-disjoint gate.',
        'not_run':'No saturated fit, training accuracy or fabricated held-out metrics.',
        'next':'Freeze image-SHA-disjoint and degradation-disjoint label populations, benchmark genuine AP reaccumulation, then admit a bounded learning study.'}
    save(out/'LEARNING_ADMISSION.json',result,out);return result


def failure_associations(out,accepted,states):
    names=sorted({leaf for spec,_,_,_ in accepted.values() for leaf in spec['p']['leaves']})
    selected={(row['tau'],row['L_cached_ms']) for row in states};rows=[]
    for tau,limit in sorted(selected):
        scenario=[row for row in states if row['tau']==tau and row['L_cached_ms']==limit]
        for name in names:
            for present in (False,True):
                members=[row for row in scenario if (name in accepted[row['pair_id']][0]['p']['leaves'])==present]
                counts=Counter(row['state'] for row in members)
                rows.append({'feature':name,'present':present,'tau_tiny_AP':tau,'L_cached_ms':limit,
                    'measured_pairs':len(members),**{key:counts[key] for key in ('PASS','FAIL','UNCERTAIN')},
                    'pair_ids':'|'.join(row['pair_id'] for row in members),
                    'scope':'Descriptive selected-pair counts; correlated features, checkpoint and resolution are not controlled. No population or causal estimate.'})
    csv_file(out/'DESCRIPTIVE_FAILURE_ASSOCIATIONS.csv',rows)


def report_cn(out,batch,summary,rows,coverage,evidence_root=None):
    evidence_root=evidence_root or out
    audits=load(evidence_root/'AUXILIARY_AUDITS.json')
    uavdt=load(Path(audits['UAVDT_reconciliation']['path']).parent/'RECONCILIATION_RESULT.json')
    edges=load(Path(audits['UAVDT_edges']['path']).parent/'ANNOTATION_SOURCE_DECISION.json')
    training=load(Path(audits['training']['path']).parent/'TRAINING_READINESS.json')
    lines=['# VM36联合覆盖实测总结','',
        'PROBLEM：RQ1/RQ2联合配置覆盖，以及RQ3在明确四设置子集上的实测状态。',
        'WHY IS IT A PROBLEM：FM_P单域覆盖不等于两域覆盖；静态未覆盖也不等于实际检测失败。',
        'SOLUTION：冻结18个新增点，全548图验证，引用原八组；逐组保存预测、GT、原COCO复算、200次图像区间、成本。',
        'WHY IS IT A GOOD SOLUTION：下面只列已封存实测值；覆盖、实际失败与未测状态分开，不按期望填数。','',
        f'Images: 每组548张VisDrone2019-DET val，原生十类；新增验收{summary["new_measured_pairs"]}/18组，累计{summary["combined_measured_pairs"]}/26组。',
        'Models: ck01 YOLO26s、ck05 YOLOv8s；FT1280、历史seed42；SHA和训练来源见批次TRAINING_METADATA.csv及逐组configuration.json。',
        'Parameters: 直接读取原FM_P数值；七阶段顺序不变；FP32、原dense NMS、ignore-union及COCO口径。',
        'Selection: 冻结贪心联合覆盖清单，不读取AP选点、不声称全局最优。',
        f'Actual child supervision: {summary["worker_wall_seconds"]:.3f}秒，上限14400秒；不是纯GPU forward时间。',
        f'新增保存的逐图预测：{summary["new_saved_image_predictions"]}张次；每组另有3次预热调用，不计入548图评价人口。',
        f'Coverage: {coverage["measured_states"]}/1033个状态，仅限1872×4=7488点投影；不等于故障覆盖率。','',
        '| checkpoint | 输入 | 退化选择 | AP50 | AP50:95 | AR50:95 |','|---|---:|---|---:|---:|---:|']
    for row in rows:
        if row['size']=='all':lines.append(f'|{row["checkpoint"]}|{row["resolution"]}|{row["features"].replace("|", ", ")}|{float(row["map50"]):.6f}|{float(row["map50_95"]):.6f}|{float(row["ar50_95"]):.6f}|')
    lines+=['','区间和尺寸/逐类指标见CONFIDENCE_INTERVALS.csv、CONDITION_METRICS.csv、PER_CLASS_METRICS.csv。',
        'DESCRIPTIVE_FAILURE_ASSOCIATIONS.csv按特征列出已测PASS/FAIL/UNCERTAIN计数；覆盖选点非随机，因素彼此相关，不能解释为总体故障率或单因素因果。',
        '原八组JPEG配对效应由MATCHED_CONTRAST_REFERENCES.json引用；新增不同退化点不拼成缺失角点。',
        'RQ2：共享548图不满足已有图像隔离学习关卡，未报告虚假的保留集分数。',
        'RQ3：阈值仍为历史敏感性场景；未测设置为NOT_EVALUATED，不能当FAIL，不能外推全FM_D或总体gap比例。',
        'RQ4：三seed匹配训练及五项消融未运行，增强参数绑定、派生标签/loader和额外有限训练预算仍未准入。',
        'Independent test-dev：仍DATASET_BLOCKED；本轮已取得UAVDT原始GT/ignore并核对M序列，但图片来源、类别/评价版本和模型暴露尚未准入外部评价。',
        'Limitations：历史选模val、固定checkpoint、图像/场景相关、未知创建期部分来源；无架构因果或训练seed推断。',
        f'UAVDT进展：已恢复{uavdt["M_sequences"]}个M序列原始GT和ignore，共核对{uavdt["M_frames_compared"]}帧；混入目录的{sum(r["SOT_files_excluded"] for r in uavdt["SOT_exclusion"])}张SOT图单独排除。{edges["reviewed_frames"]}帧边界差异与W−1/H−1裁剪一致，但历史转换原因未证实，原始文件未改。',
        f'RQ4成本筛查：若采用完整三seed匹配消融情景，共{training["total_runs_if_full_seed_matched"]}次训练，历史库内累计时间倍数为{training["historical_total_multiple_hours"]:.2f}小时；这不是当前设备运行预测，也不是已授权的训练预算。',
        'AUXILIARY_AUDITS.json链接独立性、训练成本、原始标注来源和边界核查的封存Run与SHA。RQ4成本只是历史累计时间倍数的筛查情景，不是当前训练耗时预测。',
        '548个图像ID仅547个不同图片字节SHA；保留全部合法图像。区间为固定权重、逐图点态百分位区间，未控制航次聚类、训练seed和多重比较。',
        'Evidence: '+str(evidence_root),'Batch RunID: '+batch.name,
        'Reproduction: joint_analysis.py --batch-run '+batch.name+'（CPU，不重跑GPU）；逐组原始预测复算见evaluation.evaluate_saved。']
    path=ROOT/'datasets/reportcn'/('vm36_supervisor_reproduction_联合覆盖实测_'+out.name.split('_')[-1]+'.md')
    write_atomic(path,'\n'.join(lines)+'\n',out);save(out/'REPORT_REFERENCE.json',source(path),out)


def revise_report(analysis):
    verify_hashes(analysis,('summary.json','EVIDENCE_SOURCES.json','CONDITION_METRICS.csv','AUXILIARY_AUDITS.json'))
    summary=load(analysis/'summary.json');references=load(analysis/'EVIDENCE_SOURCES.json')
    batch=Path(references['batch_protocol']['path']).parent
    out,_=new_run({'output_root':str(analysis.parent)},EXPERIMENT,'joint_report_revision')
    report_cn(out,batch,summary,table(analysis/'CONDITION_METRICS.csv'),summary['coverage'],analysis)
    save(out/'inputs.json',{'analysis':source(analysis/'SHA256SUMS.txt'),'source':source(Path(__file__))},out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'REPORT_FORMAT_REVISION','status':'VALIDATED',
        'new_measurements':0,'reason':'Render feature separators within one table cell; source metrics and old reports unchanged.'},out,replace=True)
    write_atomic(out/'diary.md','# Report rendering revision\n\nCorrected Markdown table cell separators using the sealed summary CSV. No measurements or statistics recalculated. Original report retained.\n',out)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>VM36 report rendering</h1><p>No new measurements.</p><a href="REPORT_REFERENCE.json">Chinese report reference</a>',out)
    seal(out);refresh(analysis);print(out,flush=True)


def refresh(out):
    from factorial_analysis import refresh_history
    refresh_history(out)
    text='<!doctype html><meta charset="utf-8"><h1>VM36 latest measurements</h1><p><a href="'+out.name+'/index.html">Joint-coverage results and raw evidence</a></p><p>Development validation only; no independent test-dev result.</p>'
    write_atomic(out.parent/'index.html',text+''.join('<p><a href="'+r['page']+'">'+r['RunID']+'</a> · '+str(r['status'])+'</p>' for r in load(out.parent/'latest.json')['history']),out,replace=True)
    latest=load(out.parent/'latest.json');summary=load(out/'summary.json')
    latest.update({'latest_joint_analysis_RunID':out.name,'status':summary['status'],
        'measured_conditions':summary['combined_measured_pairs'],'new_conditions':summary['new_measured_pairs'],
        'reused_conditions':8,'formal_measured_pairs':0,'measured_states':summary['coverage']['measured_states'],
        'coverage_scope':summary['coverage']['scope']})
    save(out.parent/'latest.json',latest,out,replace=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--batch-run');group.add_argument('--report-only');parser.add_argument('--prior-analysis');args=parser.parse_args()
    identity=args.batch_run or args.report_only
    if Path(identity).name!=identity:raise ValueError('Expected a plain RunID.')
    path=ROOT/'datasets/local_runs'/EXPERIMENT/identity
    prior=None
    if args.prior_analysis:
        if not args.batch_run or Path(args.prior_analysis).name!=args.prior_analysis:parser.error('Use a plain prior RunID with --batch-run.')
        prior=path.parent/args.prior_analysis
    finish(path,prior) if args.batch_run else revise_report(path)
