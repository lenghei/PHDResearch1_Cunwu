"""Source-linked factual delivery, with no new scientific interpretation."""
from collections import Counter
from datetime import datetime,timezone
import csv
import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys
from protocol import ROOT,PUBLIC,HERE,EXPERIMENT,PRIOR,BASE,DEV,STUDY,ADMIT,FMP,FMD,LEGACY_FMP,LEGACY_FMD,load,save,source,sha256,table,csv_file,write_atomic
from common import verify_hashes,seal


def checkpoints(out):
    import torch
    from ultralytics import YOLO
    from main.labels import VIS_NAMES
    torch.set_num_threads(2)
    candidates={c['id']:c for c in load(ADMIT/'admission.json')['candidates']}
    catalog=table(ROOT/'datasets/research_archive/migration_20261104/model_catalog.csv')
    study={r['checkpoint']:r for r in load(STUDY/'protocol.json')['detectors']};rows=[]
    for cid in ('ck01','ck05','ck06'):
        entry=study[cid];path=Path(entry['path']);before=sha256(path)
        if before!=entry['sha256']:raise ValueError('Checkpoint SHA differs.')
        net=YOLO(path).model.cpu().float().eval()
        if list(net.names.values())!=VIS_NAMES:raise ValueError('Class order differs.')
        bad=[k for k,v in net.state_dict().items() if v.is_floating_point() and (not bool(torch.isfinite(v).all()) or (k.endswith('running_var') and bool((v<0).any())))]
        if bad:raise ValueError('Unhealthy tensors: '+','.join(bad))
        c=candidates[cid];config=c['training_protocol'];record=next(r for r in catalog if r['sha256']==before)
        curve=ROOT/record['train_curve'];history=table(curve)
        last={k.strip():v for k,v in history[-1].items()}
        rows.append({'checkpoint':cid,'model':c['model'],'architecture_class':type(net).__name__,
            'checkpoint_sha256':before,'native10_class_order':json.dumps(VIS_NAMES),
            'CPU_tensor_health':'VALIDATED','parameters':sum(p.numel() for p in net.parameters()),
            'training_imgsz':config['imgsz'],'configured_epochs':config['epochs'],'seed':config['seed'],
            'optimizer':config['optimizer'],'lr0':config['lr0'],'batch':config['batch'],'nbs':config.get('nbs'),
            'AMP_training':config['amp'],'training_library_version':c['checkpoint_version'],
            'COCO_initialization_name':record['initialization_or_model'],'initialization_creation_SHA':'UNKNOWN',
            'selected_epoch':'UNKNOWN_STRIPPED_CHECKPOINT_EPOCH_MINUS_ONE','checkpoint_selection':c['checkpoint_selection'],
            'training_data_byte_provenance':c['training_membership'],'historical_testdev_use':'UNKNOWN_PRIOR_USE',
            'development_status':'CPU_HEALTH_VALIDATED; SCIENTIFIC_REVIEW_PENDING','formal_status':'DATASET_BLOCKED',
            'training_args_sha256':sha256(ROOT/record['training_config']),'training_curve_sha256':sha256(curve),
            'reported_last_epoch':last.get('epoch'),'reported_training_curve_seconds':last.get('time'),
            'training_time_definition':'Library cumulative time in results.csv; not independently instrumented total wall time.'})
        if sha256(path)!=before:raise ValueError('Source weight changed.')
        del net
    csv_file(out/'CHECKPOINT_ADMISSION.csv',rows)
    return rows


def historical(out):
    required=['configuration_metrics.csv','resolution_effects.csv','heldout_regression.csv','pair_states.csv','static_empirical_diagnostics.csv']
    verify_hashes(DEV,required)
    metrics=table(DEV/'configuration_metrics.csv');prediction_refs=[]
    for row in metrics:
        if row['size']!='all':continue
        pair=ROOT/'datasets/local_runs/cv261207'/row['RunID']
        checks=verify_hashes(pair,('metrics.json','predictions.jsonl.gz','ground_truth.json','bootstrap.json'))
        prediction_refs.append({'source_RunID':row['RunID'],'scope':'HISTORICAL_DEVELOPMENT_248_REFERENCE',
            'images':248,'checkpoint_sha256':row['checkpoint_sha256'],
            'raw_predictions':source(pair/'predictions.jsonl.gz'),'matching_GT':source(pair/'ground_truth.json'),
            'metrics':source(pair/'metrics.json'),'bootstrap':source(pair/'bootstrap.json')})
    csv_file(out/'HISTORICAL_REFERENCE_METRICS.csv',metrics)
    csv_file(out/'HISTORICAL_RESOLUTION_EFFECTS.csv',table(DEV/'resolution_effects.csv'))
    csv_file(out/'HISTORICAL_STATIC_EMPIRICAL.csv',table(DEV/'static_empirical_diagnostics.csv'))
    csv_file(out/'HISTORICAL_RULE_DIAGNOSTICS.csv',table(DEV/'heldout_regression.csv'))
    csv_file(out/'STATIC_COVERAGE_REFERENCE.csv',table(PRIOR/'static_coverage_policies.csv'))
    csv_file(out/'LEAF_CHECKPOINT_COVERAGE_REFERENCE.csv',table(PRIOR/'leaf_checkpoint_coverage.csv'))
    csv_file(out/'INFERENCE_ATTRIBUTES_REFERENCE.csv',table(PRIOR/'tested_detector_attributes.csv'))
    save(out/'PREDICTION_EVIDENCE_MANIFEST.json',{'status':'NOT_EVALUATED_FOR_NEW_STUDY',
        'new_predictions':[],'new_measured_pairs':0,'historical_references':prediction_refs,
        'restriction':'Historical 248-image metrics are not combined with 300-image metrics or relabeled full548 measurements.'},out)
    with (out/'CONFIGURATION_METRICS.csv').open('x',encoding='utf-8',newline='') as stream:
        csv.writer(stream).writerow(['RunID','pair_id','images','AP50','AP50_95','AR50_95','Tiny_AP','Small_AP','Medium_AP','Large_AP','metric_source_SHA256'])
    save(out/'STATISTICAL_METHODS.json',{'status':'PLANNED','new_confidence_intervals':[],
        'source_reference':source(DEV/'confidence_intervals.csv'),
        'candidate_draws':200,'candidate_seed':20261207,'unit':'image position, paired within contrast',
        'original_area_bins':'tiny<256;small[256,1024);medium[1024,9216);large>=9216',
        'multiplicity':'NEEDS_SCIENTIFIC_REVIEW','flight_clusters':'UNKNOWN','training_seed_stability':'NOT_EVALUATED',
        'restriction':'Existing paired intervals are conditional on fixed checkpoints and do not control training-seed or scene variability.'},out)
    save(out/'INDEPENDENT_EVALUATION_ADMISSION.json',{'status':'BLOCKED','scientific_gate':'FORMAL_DATASET_BLOCKED',
        'source':source(ADMIT/'admission.json'),'prior_use_state':'UNKNOWN_PRIOR_USE',
        'file_disjointness':'VERIFIED_FILE_IDENTITIES_ONLY','verified_no_prior_use':False,
        'missing':['Creation-time train bytes and initialization hashes','Selected checkpoint epoch evidence','Historical model/method decisions influenced by test-dev'],
        'proposals_for_review':['Recover direct historical records and an accountable usage inventory; preserve unknowns.',
            'If independence cannot be established, use a separately preregistered prospective holdout with compatible labels, licences and independent scene identities.'],
        'substitution':'No automatic dataset switch; test-dev and UAVDT are not used for new tuning.'},out)
    return metrics


def resource(out,benchmark):
    timing=table(DEV/'timing.csv');fields=['pair_wall_seconds','GPU_forward_profiler_seconds','cpu_evaluation_seconds','cpu_replay_seconds','bootstrap_seconds']
    total={k:sum(float(r[k]) for r in timing) for k in fields}
    original_mean=total['pair_wall_seconds']/len(timing)
    report={'status':'BLOCKED_FOR_NEW_GPU_BATCH','historical_248_pair_count':len(timing),'historical_totals':total,
        'source':source(DEV/'timing.csv'),'not_additive':'Nested phases are not independent wall-time totals.',
        'screening_only_linear_548_seconds_per_pair':original_mean*548/248,
        'screening_only_linear_1610_seconds_per_pair':original_mean*1610/248,
        'screening_estimate_status':'NOT_ADMITTED; object density, initialization, I/O and CPU matching do not scale uniformly.',
        'actual_548_complete_pair_wall_seconds':None,'actual_1610_complete_pair_wall_seconds':None,
        'next_cost_gate':'After scientific review, authorize one bounded complete-split pair, then budget the batch from measurement.',
        'GPU_batch_authorization':None,'training_authorization':None,'new_GPU_seconds':0}
    if benchmark:report['CPU_benchmark']=source(benchmark/'efficiency_result.json')
    save(out/'RESOURCE_BUDGET.json',report,out)
    return report


def definitions(out):
    content='''# Definitions and corrections

This record separates verified quantities from historical notes and proposals.

| Term | Meaning |
|---|---|
| FM_P configuration | A legal selection of degradation features under the original constraints |
| Checkpoint | A trained or initialized model identified by file SHA256 |
| Inference setting | Resolution, precision, head and post-processing applied to a fixed checkpoint |
| Evaluation pair | One exact degradation selection and detector setting on a specified full image population |
| Source image | An original image ID plus its byte SHA; distinct IDs can share identical bytes |
| Image-inference operation | One image processed by one detector setting; not a configuration count |
| Raw prediction | A saved scored box before the declared mapping/ignore evaluation step |
| IoU | Intersection area divided by union area of the two boxes |
| AP50 | Precision-recall AP at IoU 0.50 |
| AP50:95 | Mean precision-recall AP at IoU .50, .55, ..., .95; not ordinary classification accuracy |
| AR | COCO-style maximum recall under the declared per-image/category detection cap |

- VisDrone2019-DET split counts are 6,471 train, 548 val, 1,610 test-dev and 1,580 test-challenge. The challenge archive has no public labels here.
- The theoretical product is 29,485,596,672 legal model pairs. It is not an experiment count.
- Pairwise state coverage does not mathematically guarantee discovery of 90% or any fixed fraction of empirical failures.
- Manually selected historical factors are not retroactively called uniform or pairwise samples.
- V1's 500 and V2's 900 pairs remain planned. VM36 currently has no new detector measurements.
- Historical JPEG Q85 is not an original Q25/Q50/Q75 leaf. Gaussian variance is not sigma.
- Source support or current defaults do not prove that a historical training transform was activated.
- No verified camera speed, pitch angle or real sensor calibration is supplied by the sketches.
- Coding and analysis assistance must not be described as formatting-only assistance. Student reasoning and approval are separate from generated code.
- The PDF-aligned VM36 replicas preserve the original Boolean models and numerical definitions. Twelve historical implementation attributes remain a separate unapproved proposal.
- Draft Nano/Small params_m attributes are not actual parameter counts. Stripped epoch=-1 is not evidence of a selected epoch.
- Cached-PNG pipeline timing is not device-only forward timing or a standardized UAV deployment benchmark.
- Image bootstrap is not training-seed validation. A subset study is not complete-split independent evidence.

Original source files and historical results are preserved. Targeted legacy-note excerpts and line numbers remain in the private evidence Run.
'''
    write_atomic(out/'DEFINITIONS_AND_CORRECTIONS.md',content,out)


def finish(out,benchmark=None):
    if (out/'SHA256SUMS.txt').exists():raise ValueError('Run is sealed.')
    if sha256(out/'EXPERIMENT_PROTOCOL.json')!=(out/'PROTOCOL_SHA256.txt').read_text().split()[0]:raise ValueError('Protocol hash changed.')
    if benchmark:verify_hashes(benchmark,('efficiency_result.json','reference_bootstrap.json','fast_bootstrap.json'))
    from evaluation import runtime
    runtime(out)
    models=checkpoints(out);metrics=historical(out);budget=resource(out,benchmark);definitions(out)
    data=load(out/'DATASET_ADMISSION.json');alignment=load(out/'MODEL_ALIGNMENT.json');logic=load(out/'FEATURE_MODEL_VALIDATION.json')
    sampling=load(out/'sampling_verification.json');decisions=table(out/'DECISION_REGISTER.csv')
    if data['development_validation']['images']!=548 or data['development_validation']['status']!='VALIDATED':raise ValueError('Full validation gate failed.')
    if any(r['numeric_equality']!='EXACT' for r in alignment['models'].values()):raise ValueError('Model alignment failed.')
    evidence={'status':'PARTIAL','RQ1':{'status':'PARTIAL','logical_models':'VALIDATED','numeric_alignment':'VALIDATED',
        'physical_constraints':'NEEDS_SCIENTIFIC_REVIEW','runtime_extensions':'NOT_APPROVED'},
        'RQ2':{'status':'PARTIAL','new_measured_pairs':0,'historical_responses':279,'historical_248_pairs':12,
             'classifier':'ML_INSUFFICIENT_EVIDENCE','source':source(DEV/'learning_acceptance.json')},
        'RQ3':{'status':'PARTIAL','static_reference_counts':[1,3,18],'static_denominator':46368,
             'new_full548_empirical_gap':'NOT_EVALUATED','full_FM_D_gap':'NOT_EVALUATED','historical_subset_source':source(DEV/'subset_gaps.csv')},
        'RQ4':{'status':'NOT_EVALUATED','engineering_canary':'One update, not augmentation efficacy.',
             'source':source(BASE/'rq4_matched_protocol.json')},'formal_pairs':0,'new_GPU_seconds':0}
    save(out/'RQ1_RQ4_EVIDENCE.json',evidence,out)
    gates=[{'gate':'PDF numeric alignment','status':'VALIDATED'},
       {'gate':'Full548 validation data','status':'VALIDATED'}, {'gate':'Scientific protocol approval','status':'NEEDS_SCIENTIFIC_REVIEW'},
       {'gate':'Motion-orientation binding','status':'NEEDS_SCIENTIFIC_REVIEW'},
       {'gate':'Historical train labels','status':data['training_gate']},
       {'gate':'Independent test-dev','status':'DATASET_BLOCKED'}, {'gate':'Full recording transcript','status':'FULL_TRANSCRIPT_BLOCKED'},
       {'gate':'New GPU budget','status':'NOT_APPROVED'}, {'gate':'RQ4 efficacy','status':'NOT_EVALUATED'}]
    bench=load(benchmark/'efficiency_result.json') if benchmark else None
    summary={'schema':'VM36_READ_ONLY_SUMMARY_V1','RunID':out.name,'ExperimentID':EXPERIMENT,'status':'PARTIAL',
        'scope':'SOURCE_ALIGNMENT_AND_CPU_ENGINEERING; no new detector response',
        'validated_validation_images':548,'new_measured_pairs':0,'formal_measured_pairs':0,'GPU_seconds':0,
        'model_alignment':[{'model':d,'leaves':25 if d=='FM_P' else 34,'constraints':5 if d=='FM_P' else 4,
            'legal_configurations':46368 if d=='FM_P' else 635904,'PDF_numeric_attributes':r['PDF_defined_attributes'],
            'numeric_match':r['numeric_equality']} for d,r in alignment['models'].items()],
        'gates':gates,'decisions':[{'ID':r['DecisionID'],'Question':r['ExactQuestion'],'Student review':r['StudentReviewStatus']} for r in decisions],
        'source_hashes':{p.name:sha256(p) for p in (FMP,FMD,out/'EXPERIMENT_PROTOCOL.json',out/'DATASET_ADMISSION.json',out/'FEATURE_MODEL_VALIDATION.json')},
        'CPU_benchmark':{k:bench[k] for k in ('images','draws','reference_bootstrap_wall_seconds','fast_bootstrap_wall_seconds','accumulation_speedup','full_CIs_equal_saved')} if bench else None}
    save(out/'summary.json',summary,out)
    save(out/'result.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'SUPERVISOR_ALIGNED_PREPARATION','status':'PARTIAL',
        'scope':summary['scope'],'new_measured_pairs':0,'formal_measured_pairs':0,'summary_sha256':sha256(out/'summary.json'),
        'summary_canonical_sha256':hashlib.sha256(json.dumps(summary,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()},out)
    save(out/'RESULT_VERIFICATION.json',{'status':'PARTIAL','model_numeric_equality':'VALIDATED','dataset_population':'VALIDATED',
        'new_metrics':'NOT_EVALUATED','scientific_approval':'NOT_GRANTED','student_alignment_directive_applied':True,
        'historical_models_unchanged':{'FM_P':sha256(LEGACY_FMP),'FM_D':sha256(LEGACY_FMD)},
        'CPU_benchmark':bench['status'] if bench else 'NOT_EVALUATED'},out)
    report_cn(out,summary,models,budget,bench)
    figures=[];previous=load(PRIOR/'figure_evidence.json')
    for i in range(1,13):
        key=f'fig{i:02}'
        figures.append({'figure':key,'research_question':('RQ1' if i<=7 else 'RQ2' if i in (8,9,11) else 'RQ3' if i==10 else 'RQ4'),
           'population':'See unchanged source Run; historical 300 and new248 cohorts remain separate.',
           'variables_and_units':'Defined by the source CSV and plot axes; no replacement values entered.',
           'selection_rationale':'Source sampling/condition design; student must explain and approve its relevance.',
           'source_RunID':PRIOR.name,'source_CSV_sha256':sha256(PRIOR/f'figures/{key}.csv'),
           'figure_sha256':sha256(PRIOR/f'figures/{key}.pdf'),
           'numerical_result':'NOT_EVALUATED' if i==12 else 'UNCHANGED_SAVED_SOURCE_CSV',
           'supported_scope':'Existing logical/development/engineering evidence only.',
           'not_supported':'New full548 results, formal independence or matched gap closure.',
           'student_interpretation':'NEEDS_SCIENTIFIC_REVIEW'})
    csv_file(out/'FIGURE_EXPLANATION_REGISTER.csv',figures)
    write_atomic(out/'diary.md','# VM36 factual record\n\nThe first proposal used historical model definitions and was sealed PARTIAL. After the student required exact supervisor alignment, V2 restores only the PDF-defined model attributes. Historical files remain unchanged. Full548 data evidence is reused by SHA. A new validator preserves the old strict guard and checks the Boolean projection before count transfer. No detector was run. Scientific interpretation and runtime bindings remain pending.\n',out)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><title>VM36 evidence</title><style>body{font:17px Arial;max-width:1100px;margin:40px auto}li{margin:14px 0}</style><h1>VM36 · Supervisor-aligned preparation</h1><p>PARTIAL · New detector pairs: 0 · GPU: 0 s · Formal test-dev: blocked</p><ul>'+''.join(f'<li><a href="{n}">{n}</a></li>' for n in ('EXPERIMENT_PROTOCOL.md','MODEL_ALIGNMENT.csv','DECISION_REGISTER.csv','DATASET_ADMISSION.json','SAMPLING_RATIONALE.md','CHECKPOINT_ADMISSION.csv','HISTORICAL_REFERENCE_METRICS.csv','RESOURCE_BUDGET.json','summary.json'))+'</ul>',out)
    print(out,flush=True)


def report_cn(out,summary,models,budget,benchmark):
    val=load(out/'DATASET_ADMISSION.json')['development_validation'];alignment=load(out/'MODEL_ALIGNMENT.json')
    lines=['# VM36：导师计划对齐与实验准备','',
      '1. **研究问题**：保留导师原来的双特征模型、交互失效、静态/实测gap和定向增强四个问题。',
      '2. **为什么需要处理**：旧模型的数值定义与导师稿不完全相同；小图像子集也不能替代完整数据划分。',
      '3. **本轮处理方法及依据**：从导师PDF读取数值定义，在VM36建立独立对齐副本；复用已验证实现，核对完整548张val，不改历史数据。',
      '4. **现有证据**：两模型数值一致、Boolean结构未变，SAT和无死特征检查实际执行；548张val检查通过。尚无VM36新检测结果，不据此认定方法有效。',
      '',f'RunID：`{out.name}`。总状态PARTIAL。','',
      '## 两棵特征模型与导师稿是否一致','',
      '- 新FM_P的30个数值属性与PDF逐项一致；旧实现额外的12个属性没有放进导师对齐副本。',
      '- 新FM_D恢复PDF中的10个数值属性；Nano/Small的2.5M/10M是草稿元数据，不当作真实checkpoint参数量。',
      '- 特征树和约束没有改变；FM_P仍为25叶、5约束、46368合法配置。FM_D仍为34叶、4约束、635904合法配置、844训练配置。',
      '- 真实运行了FlamaPy解析、SAT与无死特征检查。FM_D使用经Boolean结构相等性保护的原计数推导，没有冒称新BDD/#SAT计数。',
      '- 旧校验器拒绝新增数值属性的错误日志保留。新校验器只支持“数值元数据改变、Boolean结构完全相同”的情形；原校验器没有被修改。',
      '',f'本轮模型：`{FMP}`；`{FMD}`。',
      '旧cv261109/fm_p.uvl及cv261116/fm_d.uvl保留为历史文件，不再冒充VM36的导师对齐定义。',
      '', '## 尚需确认的实现细节','',
      '- 导师稿规定Motion_k21的核长21，但没有规定方向。旧实现采用固定种子随机0—180°；这是额外实现选择，并非导师已给出的参数或真实飞行标定。',
      '- 雾的airlight、雨线长度/透明度、rolling shutter位移/频率也属于旧实现扩展。均在IMPLEMENTATION_BINDINGS_PROPOSAL.json单列，默认不启用。',
      '- 没有你的具体审核，不会把这些扩展参数自动填回新模型，也不会启动包含运动模糊的新科学实验。',
      '', '## 全量数据核验','',
      '- train 6471图/6471标注；val 548图/548标注；test-dev 1610图/1610标注；test-challenge 1580图、无公开GT。',
      '- 548张val全部按原ZIP字节解码，全部保留，没有按AP挑图或删图。原生十类、原像素尺寸及ignore-union协议保留。',
      f'- 固定ignore规则后共有{val["GT_after_ignore"]}个GT：tiny {val["size_counts_after_ignore"]["tiny"]}、small {val["size_counts_after_ignore"]["small"]}、medium {val["size_counts_after_ignore"]["medium"]}、large {val["size_counts_after_ignore"]["large"]}。',
      '- val有一组字节重复图像，保留不同原始ID及其标注；后续划分不得拆开相同图像字节。图像bootstrap也不能当成独立航次证据。',
      '- train仍有3条零高度框，其中包含ignore框和一条有效类目标框；没有修改原标签。未来训练须另行审核派生标签策略。',
      '- test-dev文件不交叉不等于科学独立，历史使用状态UNKNOWN_PRIOR_USE，正式评价仍为DATASET_BLOCKED。',
      '', '## 采样和实验数量','',
      '- 原500/900配对计划不变；约4000对是导师更大目标，不是已完成实验。理论积为29485596672，不是执行数。',
      '- 原pairwise集80配置：21个贪心核心+59个补充点；覆盖1163可行二阶状态，37不可行。没有承诺90%失效发现率。',
      '- 两个uniform总体分别是46368与26496；各抽100个，纳入概率不能混用。active尚未运行。',
      '- 36条件候选方案尚未获批准；每条件必须使用全部548张，计划19728张次。VM36目前新测量0对，正式0/500，训练0次，GPU用时0。',
      '', '## 旧结果如何使用','',
      '- 历史279条去重响应和12对248图开发结果只作参考，不重新标成548图实验；AP不能通过平均或拼接表格代替重新评价。',
      '- 真正AP50/AP50:95、AR和尺寸分层数值仍可从HISTORICAL_REFERENCE_METRICS.csv及原预测/GT追溯；CONFIGURATION_METRICS.csv只有表头，表示新实验尚未运行。',
      '- 原配对200次图像bootstrap区间与训练seed变异分开；至少三个匹配训练seed尚未完成。',
      '- RQ2失效分类器仍是ML_INSUFFICIENT_EVIDENCE；旧AP回归或描述性交互不算已验证的失效规则。',
      '- RQ3仅有明确小子集的静态/实测比较。Tiny阈值0.10、100 ms缓存管线的4 PASS/6 FAIL/2 UNCERTAIN不能换称50 ms部署结果。',
      '- RQ4仍NOT_EVALUATED；一更新canary不是增强效果实验。UAVDT原生标注/ignore边界未恢复，仍阻塞。',
      '', '## 资源与评价加速','']
    if benchmark:
        lines.append(f'- 相同旧248图预测上，原累积器200次bootstrap耗时{benchmark["reference_bootstrap_wall_seconds"]:.3f}秒；复用快速实现耗时{benchmark["fast_bootstrap_wall_seconds"]:.3f}秒。全部抽样值、区间及COCO指标一致。此为CPU工程验收，不是新研究结果。')
    lines += ['- 旧12对墙钟2640.344秒，forward 45.049秒，CPU评价518.647秒、重算557.432秒、bootstrap1233.869秒；阶段可能嵌套，不相加冒充GPU占用。',
      f'- 简单按图数线性筛查：548图每对约{budget["screening_only_linear_548_seconds_per_pair"]:.1f}秒，1610图约{budget["screening_only_linear_1610_seconds_per_pair"]:.1f}秒。这不是已测耗时，也不是准入预算；目标密度和CPU匹配成本可能改变。',
      '- 下一步须先审核科学协议，再授权一个有界完整划分计时对；测出实际成本后再批量运行。',
      '', '## 录音状态','',
      '- 原录音Windows元数据时长2545.931675秒，约42分25.93秒。未发现当前环境可用本地ASR与模型，没有上传外部服务。',
      '- 旧文档只声称到14:30，连前段也未逐字听核。完整转写和中文全文翻译均BLOCKED；没有补写后28分钟，没有据告别语认定实际会议结束。',
      '- 三份meeting_20261109_*文件明确标出来源与缺口，供本人核对，不能直接作为导师原话引用。',
      '', '## 四段式汇报准备表','',
      '| 部分 | 由本人回答的问题 | 可核对材料 |','|---|---|---|',
      '| Problem | 我究竟研究什么部署条件与配置之间的问题？ | 导师PDF第1、4页、原两棵树 |',
      '| Why | 哪些实测或明确假设说明值得研究？ | 原预测、GT、条件和范围；不要只展示图 |',
      '| Solution | 我为什么选择这些配置、样本和评价边界？ | DECISION_REGISTER、采样规则与准确配置ID |',
      '| Why good | 什么证据能支持或反驳方案？目前缺什么？ | 独立验证、失效分类、gap前后对照；尚缺的如实说明 |',
      '', '这里仅列证据和待回答问题，不替本人撰写科研结论。每张旧图另有FIGURE_EXPLANATION_REGISTER.csv，提交导师前需本人核对解释。',
      '', '## 运行与文件关系','',
      '- run.py prepare：新建提案Run并冻结协议，不运行YOLO。',
      '- alignment.py：由PDF数值生成独立UVL副本；缺少审核时拒绝注入实现参数。',
      '- validate_models.py：在FlamaPy环境检查原计划对应的模型。',
      '- evidence.py：原始ZIP、548图、采样和会议来源审计；可通过SHA复用已完成检查。',
      '- benchmark.py：仅对已存预测做CPU重算和加速一致性验证。',
      '- reporting.py：真实权重CPU健康核验、事实表和中文说明。',
      '- Streamlit的VM36页只读已存summary，不训练、不推理、不重算AP。未发布审核摘要，云端不会读取本地文件。',
      '',f'证据目录：`{out}`',f'固定入口：`{out.parent / "index.html"}`',
      '结论：基础模型数值对齐和全量val工程检查已完成；科学协议、额外实现选择、独立性和GPU预算仍待审核。']
    path=ROOT/'datasets/reportcn/vm36_supervisor_reproduction_实验事实与使用说明.md'
    write_atomic(path,'\n'.join(lines)+'\n',out)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--benchmark-run')
    args=p.parse_args()
    for value in (args.run,args.benchmark_run):
        if value and Path(value).name!=value:raise ValueError('Expected plain RunIDs.')
    root=ROOT/'datasets/local_runs'/EXPERIMENT
    finish(root/args.run,root/args.benchmark_run if args.benchmark_run else None)
