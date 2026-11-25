"""Reuse accepted VM45 predictions; quantify unsafe PASS decisions."""
import csv
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import sys
import time

os.environ.update(CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1')
from vm45_common import ROOT, HERE, UVL, V44, ref, read, save, sha, csv_write

PRIOR = ROOT/'datasets/local_runs/vm45_motion_haze_gap_learning/vm45_motion_haze_gap_learning_cpu_study_20261117T110831_36b2fc57'


def csv_rows(path):
    ref(path)
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def ratio(n, d):
    return n/d if d else None


def risk(rows, method):
    scored = [r for r in rows if r['true_state'] in ('PASS', 'FAIL')]
    true_pass = sum(r['true_state']=='PASS' and r['prediction']=='PASS' for r in scored)
    false_pass = sum(r['true_state']=='FAIL' and r['prediction']=='PASS' for r in scored)
    missed_pass = sum(r['true_state']=='PASS' and r['prediction']=='FAIL' for r in scored)
    correct_fail = sum(r['true_state']=='FAIL' and r['prediction']=='FAIL' for r in scored)
    actual_fail = false_pass+correct_fail
    actual_pass = true_pass+missed_pass
    return {'method': method, 'scored_rows': len(scored), 'actual_FAIL': actual_fail, 'actual_PASS': actual_pass,
            'actual_FAIL_predicted_PASS': false_pass, 'actual_PASS_predicted_PASS': true_pass,
            'predicted_PASS_count': true_pass+false_pass, 'false_PASS_rate_among_actual_FAIL': ratio(false_pass, actual_fail),
            'PASS_prediction_precision': ratio(true_pass, true_pass+false_pass),
            'unsafe_fraction_among_predicted_PASS': ratio(false_pass, true_pass+false_pass),
            'false_PASS_fraction_all_scored': ratio(false_pass, len(scored)),
            'balanced_accuracy': .5*(true_pass/actual_pass+correct_fail/actual_fail),
            'accuracy': (true_pass+correct_fail)/len(scored), 'confusion_PASS_FAIL': [[true_pass, missed_pass], [false_pass, correct_fail]],
            'UNCERTAIN_predicted_PASS_NOT_scored': sum(r['true_state']=='UNCERTAIN' and r['prediction']=='PASS' for r in rows),
            'rule_admission': 'REJECTED_UNSAFE_OR_UNVALIDATED', 'evaluation_scope': 'GROUP_HELDOUT_DEVELOPMENT_ONLY_FIXED_OOF_PREDICTIONS'}


def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(str(v).replace('|','/') for v in row)+' |' for row in rows]])


def main():
    started, cpu = time.perf_counter(), time.process_time()
    from main.pipeline import new_run
    out, logger = new_run({'output_root': str(PRIOR.parent), 'source_RunID': PRIOR.name,
                          'scope': 'CPU_RISK_AND_REPORT_CONTINUATION', 'GPU_authorized': False}, HERE.name, 'continuation')
    (out/'runtime').mkdir(); (out/'plots').mkdir()
    os.environ['MPLCONFIGDIR'] = str(out/'runtime/matplotlib')
    ref(__file__)
    verified = 0
    for seal in ('SHA256SUMS.txt', 'supplementary_SHA256SUMS.txt'):
        ref(PRIOR/seal)
        for line in (PRIOR/seal).read_text(encoding='utf-8').splitlines():
            expected, file = line.split('  ',1)
            if ref(file)['SHA256'] != expected:
                raise ValueError('Accepted source changed: '+file)
            verified += 1
    for name, expected in [('fm_p.uvl','e6eadd633d1c14e0d6bc425f5e54421d592d766c65a6e5f20c61bc34dd76fb5c'),
                           ('fm_d.uvl','4022d674822b97dbbe3712646b9a0f970af95d175ae358e63582029bc107571a')]:
        if ref(UVL/name)['SHA256'] != expected: raise ValueError('Original UVL changed.')
    sys.path.insert(0,str(HERE.parent/'vm43_external_pilot'))
    import vm43_common
    before = vm43_common.budget()
    save(out/'continuation_scope.json', {'source': 'USER_EXPLICIT_CURRENT_MESSAGE', 'created_UTC': datetime.now(timezone.utc).isoformat(),
        'instruction': 'Autonomous CPU execution. GPU execution only with genuinely applicable VM45 authorization and all admission gates.',
        'VM45_GPU_permission_granted_by_this_message': False, 'prior_results_reused': True,
        'no_repeat': ['UVL counting','VM43 inference','VM44 diagnostics','VM45 classifier fitting'],
        'scientific_reporting_change': 'Expose false-PASS denominators; replace duplicate reports with distinct documents under new filenames.'})
    learning = read(PRIOR/'rule_learning_validation.json')
    predictions = csv_rows(PRIOR/'held_out_rule_results.csv')
    folds = csv_rows(PRIOR/'rule_group_folds.csv')
    for f in folds:
        if set(json.loads(f['train_P_ids'])) & set(json.loads(f['validation_P_ids'])):
            raise ValueError('P leakage in accepted folds.')
    risks = []
    for method in ('SHALLOW_TREE','RANDOM_FOREST_REFERENCE'):
        rows = [r for r in predictions if r['method']==method]
        result = risk(rows, method)
        saved = next(r for r in learning['models'] if r['method']==method)
        assert result['confusion_PASS_FAIL']==saved['confusion_PASS_FAIL']
        assert abs(result['balanced_accuracy']-saved['balanced_accuracy'])<1e-12
        risks.append(result)
    reference = [{**r,'prediction':'FAIL'} for r in predictions if r['method']=='SHALLOW_TREE']
    risks.append(risk(reference,'ALWAYS_FAIL_REFERENCE'))
    csv_write(out/'false_PASS_risk.csv', risks)
    grouped = []
    for method in ('SHALLOW_TREE','RANDOM_FOREST_REFERENCE'):
        rs = [r for r in predictions if r['method']==method]
        for group in sorted({r['P'] for r in rs}):
            chosen = [r for r in rs if r['P']==group and r['prediction']=='PASS']
            grouped.append({'method':method,'P':group,'predicted_PASS_D':len(chosen),
                'actual_FAIL_in_predicted_PASS':sum(r['true_state']=='FAIL' for r in chosen),
                'actual_UNCERTAIN_in_predicted_PASS':sum(r['true_state']=='UNCERTAIN' for r in chosen),
                'selection_admission':'REJECTED_RULE_PREDICTIONS','true_labels_used_for_selection':False})
    csv_write(out/'controller_selection_risk.csv', grouped)
    proposal = read(PRIOR/'parameter_binding_review.json')
    generator = HERE.parent/'cv261109/generator.py'
    source = ref(generator)
    # This freezes a review candidate, not scientific approval or execution permission.
    bindings = {'status':'AUTHOR_ASSUMPTION_PROPOSED_FROZEN_NOT_EXECUTION_ADMITTED',
        'asserted_by':'CODEX_PROPOSAL_FOR_AUTHOR_REVIEW_NOT_AUTHOR_SIGNATURE',
        'source':source,'supervisor_defined':{'kernel':21,'beta':1.5},
        'bindings':{'motion_angle_distribution':{'kind':'Uniform','low':0,'high_exclusive':180,'unit':'degrees modulo180'},
                    'airlight':{'value':220,'unit':'same scalar in BGR uint8 encoded intensities, range0..255'},
                    'depth':{'kind':'constant_normalized_depth','value':1,'beta_unit':'dimensionless optical-depth proxy'}},
        'haze_equation':'I=J*exp(-beta)+(1-exp(-beta))*A; implemented in float32 then clip0..255 and uint8 truncation',
        'motion_equation':'Normalized rasterized linear21x21 PSF; cv2.filter2D/BORDER_REFLECT_101',
        'order':['Atmosphere','Illumination','Blur','RollingShutter','Noise','Downscale','JPEG'],
        'RNG':'Per-image/per-feature hash seed; shared generated pixels across detectors',
        'GT':'Identity image coordinates and unchanged ignore-union; no visibility-based GT removal',
        'label':'AUTHOR_ASSUMPTION','physically_calibrated':False,'supervisor_approved':False,
        'actual_scientific_admission':'MISSING; historical source explicitly NEEDS_SCIENTIFIC_REVIEW/approval null',
        'limitations':['No measured flight motion or depth distribution','Encoded BGR intensity is not linear scene radiance','Beta is not measured fog density','Synthetic haze is not real weather','Kernel length alone has no flight-time or speed calibration'],
        'CPU_validation_reused':ref(V44/'future_pixel_fixture_tests.json'),
        'training_augmentation_not_changed':'VM38 Gaussian p.25/sigma5.5-30 unrelated to these inference-test bindings',
        'RollingShutter':'EXCLUDED_UNVERIFIED_BBOX_IGNORE_MAPPING'}
    save(out/'parameter_binding_freeze.json', bindings)
    # Search only the VM45 surfaces; a negative search is not universal proof of absence.
    search = [str(p) for root in (HERE,PRIOR.parent) for p in root.rglob('*.json')
              if any(w in p.name.lower() for w in ('authoriz','approval','admission'))]
    gate = {'status':'BLOCKED','GPU_executed':False,'VM45_specific_execution_authorization':False,
        'reason':'No genuinely applicable VM45 execution grant supplied or validated. Current message is conditional; VM43-only permission excluded.',
        'authorization_search_scope':search,'absence_of_search_hits_is_not_global_proof':True,
        'scientific_binding_admission':False,'complete_cost_admission':False,'current_device_health':'NOT_PERFORMED_NO_GPU_PERMISSION',
        'current_thermal_admission':'NOT_PERFORMED_NO_GPU_PERMISSION','planned_seconds_NOT_GUARANTEE':12227,
        'budget_reporting_only':before,'original_deadline_unchanged':True,
        'safety':{'immediate_C':90,'sustained_C':87,'sustained_seconds':30,'poll_seconds':2,'retries':0}}
    save(out/'gpu_admission.json',gate)
    future = read(PRIOR/'experiment_protocol.json')
    save(out/'frozen_experiment_reference.json',{'source':ref(PRIOR/'experiment_protocol.json'),'design':future,
        'cells':8,'images_per_cell':548,'image_inferences':4384,'precision':'FP32','batch':1,'native10':True,
        'Normal_reuse':'NOT_ADMITTED; future complete timing boundary compatibility unproven',
        'bootstrap':'200 shared image indices; pooled COCO AP accumulation; not run',
        'actual_predictions':0,'status':'BLOCKED_NOT_EXECUTED'})
    for name in ('matched_evaluation_metrics.csv','factorial_interactions.csv','cross_detector_interactions.csv','paired_bootstrap_intervals.csv'):
        csv_write(out/name,[{'status':'NOT_EVALUATED','reason':'VM45 GPU and scientific admission missing','measured_cells':0}],['status','reason','measured_cells'])
    gap = read(PRIOR/'UVL_analysis.json')
    controller = read(PRIOR/'controller_v2_validation.json')
    save(out/'Gap_and_controller_status.json',{'empirical_scope':'Existing21P x4D development only','COVERED':4,'GAP':17,'UNCERTAIN_P':0,
        'pair_states':{'PASS':6,'FAIL':76,'UNCERTAIN':2},'new_Motion_Haze_Gap_change':'NOT_EVALUATED',
        'static_sensitivity':gap['static_sensitivity'],'full_space_empirical_gap':'NOT_ESTIMABLE',
        'controller':controller,'new_rule_predictions_admitted':0,'selection_policy':'Only measured interval-feasible evidence; otherwise ABSTAIN/UNKNOWN',
        'new_training_or_rule_fit':False,'VM43_testdev_used_for_fitting':False})
    reports = write_reports(out, risks, gate, gap)
    make_plot(out, risks)
    after = vm43_common.budget()
    assert before['authorization']==after['authorization'] and before['charges']==after['charges']
    import psutil
    save(out/'actual_resource_usage.json',{'CPU_wall_seconds_before_seal':time.perf_counter()-started,
        'CPU_process_seconds':time.process_time()-cpu,'current_RSS_bytes':psutil.Process().memory_info().rss,
        'peak_RSS':'NOT_SAMPLED','GPU_seconds':0,'new_GPU_charged_seconds':0,'original_budget_reporting_only':after})
    from vm45_common import SOURCES
    for item in list(SOURCES.values()):
        if sha(item['path'])!=item['SHA256']:raise ValueError('Input changed during continuation.')
    save(out/'source_provenance.json',{'files':list(SOURCES.values()),'source_RunID':PRIOR.name,'source_sealed_files_verified':verified,
        'new_classifier_fit':False,'new_COCO_or_Bootstrap_run':False,'GPU':False})
    save(out/'final_acceptance.json',{'status':'PARTIAL_WITH_LIMITATIONS','CPU_continuation':'COMPLETED','GPU':'BLOCKED',
        'new_GPU_cells':0,'Motion_Haze_AP_DID':'NOT_EVALUATED','false_PASS_count_per_classifier':10,
        'false_PASS_rate':10/76,'PASS_prediction_precision':6/16,'unsafe_rules_rejected':True,
        'distinct_report_bodies':True,'reports':[str(p) for p in reports],'original_results_modified':False})
    record=json.loads((out/'run.json').read_text());record.update(status='COMPLETE_WITH_LIMITATIONS',finished_UTC=datetime.now(timezone.utc).isoformat())
    (out/'run.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    (out/'execution_diary.md').write_text('Reused sealed VM45 OOF predictions. Recomputed risk denominators without fitting, inference, UVL counting or diagnostics. Bound engineering assumptions are proposed only. GPU remains blocked; unsafe rules rejected. Three distinct technical reports generated.\n',encoding='utf-8')
    links=''.join(f'<li><a href="{p.as_uri()}">{p.name}</a></li>' for p in reports)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>VM45 continuation</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto}</style><h1>VM45 continuation: false-PASS safety</h1><p>'+out.name+'</p><p>GPU BLOCKED. No Motion/Haze AP or DID measured. CPU risk reporting completed.</p><ul>'+links+'</ul><img src="plots/false_PASS_risk.png" style="max-width:100%"><p>See false_PASS_risk.csv, gpu_admission.json, source_provenance.json and SHA256SUMS.txt.</p>',encoding='utf-8')
    from main.visualization import build_run_index
    build_run_index(out.parent,HERE.name)
    logger.info('VM45 CPU continuation completed; GPU remains blocked.')
    files=[p for p in out.rglob('*') if p.is_file() and 'runtime' not in p.relative_to(out).parts]+reports+[Path(__file__).resolve()]
    with (out/'SHA256SUMS.txt').open('x',encoding='utf-8') as f:
        for p in sorted(files):f.write(sha(p)+'  '+str(p)+'\n')
    print('VM45_CONTINUATION_COMPLETED',out,flush=True)


def write_reports(out, risks, gate, gap):
    numeric=table(['方法','实际FAIL→PASS','False-PASS/实际FAIL','PASS预测精确率','准确率','平衡准确率'],
        [[r['method'],r['actual_FAIL_predicted_PASS'],f"{r['actual_FAIL_predicted_PASS']}/{r['actual_FAIL']}={r['false_PASS_rate_among_actual_FAIL']:.2%}",
          f"{r['PASS_prediction_precision']:.2%}" if r['PASS_prediction_precision'] is not None else 'NA（没有PASS预测）',
          f"{r['accuracy']:.2%}",f"{r['balanced_accuracy']:.2%}"] for r in risks])
    report=f'''# VM45 连续研究：科学报告与false-PASS安全分析

RunID：`{out.name}`。来源Run：`{PRIOR.name}`。本轮真实新增工作是既有留出预测的风险复算、参数假设登记、准入核查及报告修正，不是重新拟合、推理或交互测量。

## 1. 实际GPU与交互状态

GPU BLOCKED：当前消息要求适用专项授权，没有给出VM45八格的实际执行许可；VM43专项许可不能沿用。角度与airlight的独立科学准入也不存在。没有执行GPU预检、生成全量候选缓存、COCO重算或200次Bootstrap。Motion/Haze绝对AP、各单因素效应、Motion×Haze DID及跨模型差均NOT_EVALUATED，没有用历史Normal补缺角点。

## 2. false-PASS风险：明确分母

{numeric}

两分类器的混淆矩阵均为[[6,0],[10,66]]，行是真值PASS/FAIL，列是预测PASS/FAIL。76个实际FAIL中10个被判PASS，false-PASS率13.16%；16个预测PASS中只有6个正确，PASS精确率37.50%，错误放行比例62.50%。全82个有确定标签的评价行中，错误放行10/82=12.20%。两项UNCERTAIN只从标签评分排除，不当FAIL；其预测仍保留，见风险CSV。

平衡准确率93.42%不能代表安全选择可靠。相比之下，Always-FAIL准确率92.68%更高，却完全找不到PASS，平衡准确率50%；它不产生false-PASS，但不能成为有用的适配策略。两分类器的总体准确率87.80%，并非全面胜过基线。

这些是21折完整P留出、固定OOF预测的开发诊断，42份折定义已复核无同P交叉。548张图像由全部条件共享，PASS仅涉及4个P组；不是独立图像泛化或训练seed显著性。未根据本次风险结果调整树深、概率阈值或特征。

## 3. 控制器选择风险

把预测PASS直接当部署许可，会将10个实际FAIL对放行；另外对真值UNCERTAIN的预测PASS不能计成已证实可行。模型级评分不等于某条规则的独立验收，因此仍0条学得规则被准入。V2只接受同条件实测区间PASS，21个已测P选择4、弃权17；3个未测Motion/Haze条件全部UNKNOWN。与固定YOLO26s/1280相比未增加可行覆盖，只提供合法性、资格、区间检查及明确弃权。Oracle有同条件真实响应，只是数据内参考。

## 4. Motion/Haze参数候选

原UVL给出kernel21、beta1.5；Uniform[0,180)角度与airlight220来自旧实现提案。现在以AUTHOR_ASSUMPTION登记冻结为待作者审阅的候选，**不是作者签名、导师批准或GPU科学准入**。来源`cv261109/generator.py`及已验收CPU夹具都有SHA。

Motion是图像像素长度21的栅格化线形PSF，随机角度单位度，归一化后filter2D，边界反射。Haze用I=J exp(-beta)+(1-exp(-beta))A，A=220是BGR uint8编码强度，不是测得线性辐亮度；深度假设归一化常数1，beta为无量纲光学厚度代理。它不是实测雾浓度或真实三维深度，也没有飞行速度、曝光时长标定。复用确定性per-image/per-feature RNG和身份坐标GT/ignore验证，不重新生成Gaussian。RollingShutter仍排除。

冻结实验仍为4P×ck05/ck01 Infer1280、FP32/batch1/native10、每格548图，共4384张次；同P共享PNG，VM41完整缓存管线计时，200次同图索引重累计COCO AP。Normal只有所有边界完全相符才可复用，当前未作此准入。

## 5. RQ3已测Gap与静态假设

沿用实测21P×4D：4 COVERED、17 GAP、0个P级UNCERTAIN；对级标签仍6 PASS/76 FAIL/2 UNCERTAIN。没有新Motion/Haze测量，所以新交互对Gap的影响NOT_EVALUATED。静态覆盖假设分别1、9、18/46368，反映映射与联合覆盖假设，不是失败率。46347个未测P不能靠有目的样本得到精确经验Gap比例，原UVL不修改。`!Normal`树路径里的Normal是光照叶，不是“所有非干净图像都失败”。

## 6. 成本和下一步

原权威账本本次读取余额{gate['budget_reporting_only']['remaining_seconds']:.3f}秒，原截止{gate['budget_reporting_only']['deadline_UTC']}。12227秒是先前完整八格的规划外推，不是实测保证；本轮没有完整成本或当前设备/热安全准入。全部GPU分支必须另有真实专项授权，再按90°C即停、87°C持续30秒停和2秒监测执行，不重试、不改精度或缩图。

下一项仍是取得真实科学绑定决定、适用VM45许可及实测完整资源/热准入后的八格Motion×Haze，不是继续重做CPU审计。正式1280/100轮、12臂和五项YOLOv8s消融仍未完成；960/10轮Gaussian改进没有关闭Tiny任务Gap。VM43外部范围继续LIMITED_EXTERNAL_EXPLORATORY/独立性UNKNOWN，未进入规则训练。

## 7. 代码、证据与限制

实际命令见command.txt；新入口`{HERE/'continue_vm45.py'}`只复用旧预测复算风险，不调用原run.py整套实验。false_PASS_risk.csv记录全部分母，controller_selection_risk.csv列每组潜在放行风险，parameter_binding_freeze.json与gpu_admission.json分开候选假设和执行许可。原指标、学习结果、UVL、预测和权重不覆盖，无Git操作。

本技术分析由自动工具生成，遵循原PDF限制应由研究者核验和重写后用于论文，不作为已获导师认可的论文文字。没有新Motion/Haze结果或未测提升承诺。
'''
    brief=f'''# VM45 导师汇报：三分钟结论

RunID：`{out.name}`。

1. **新GPU结果0**：VM45专项GPU许可和科学绑定准入缺失。Motion/Haze AP及DID未测，不用计划替代结果。
2. **关键安全发现**：虽然平衡准确率93.42%，两模型各把10/76实际FAIL判PASS；预测PASS只有6/16正确，精确率37.50%。不能把这种PASS用于自动放行。
3. **系统保持安全弃权**：原有21P中选择4、弃权17；未测Motion/Haze返回UNKNOWN。0条候选规则被推广进原UVL，没有新覆盖优势。
4. **Gap仍有限范围**：4/21有实测可行D，17/21在四个已测D中无可行项；不能推广成整个46368空间失败率。
5. **参数定位**：kernel21/beta1.5来自导师UVL；角度均匀0–180和airlight220仅为AUTHOR_ASSUMPTION候选，不是自然天气标定或批准。
6. **下一科学动作**：在真实VM45准入齐备后完成八格4384张次；不要再重复旧推理或学习。正式100轮RQ4未完成。

建议当面展示false_PASS_risk.csv、原混淆矩阵和未知P的弃权返回。老师若问“93%为什么不用”，回答：平衡准确率与放行安全的分母不同，预测PASS中62.5%其实FAIL。若问新交互，明确回答NOT_EVALUATED。

完整技术依据在本Run网页及科学报告；这份简报只保留汇报结论，不复制完整报告。
'''
    matrix=[{'RQ':'RQ1','existing':'原UVL来源/SAT/有限生成验证','this_continuation':'只核验SHA，未重数','claim_allowed':'模型来源可追溯','missing':'全空间可执行性、物理约束标定','status':'PARTIAL'},
            {'RQ':'RQ2','existing':'VM36开发84对、VM45整P留出','this_continuation':'false-PASS10/76；PASS精确率6/16','claim_allowed':'开发条件分组诊断及失败候选；不安全规则拒绝','missing':'真实Motion/Haze八格DID、可信新条件泛化','status':'PARTIAL'},
            {'RQ':'RQ3','existing':'4 COVERED/17 GAP已测子集；静态1/9/18敏感性','this_continuation':'维持原UVL及未知区间','claim_allowed':'局部Gap与假设依赖','missing':'概率采样或充分验证的全空间估计','status':'PARTIAL'},
            {'RQ':'RQ4','existing':'960/10轮Gaussian探索；Tiny仍FAIL','this_continuation':'没有训练或新Gap收缩测量','claim_allowed':'探索性收益与正常精度代价','missing':'1280/100轮12臂、多seed、五单增强消融及同P重评价','status':'NOT_EVALUATED_FORMAL'}]
    csv_write(out/'rq_manuscript_evidence_matrix.csv',matrix)
    manuscript=f'''# VM45 RQ1–RQ4手稿证据矩阵

RunID：`{out.name}`。这是一份证据清单，不是论文正文；每条可写主张需研究者核验原预测/日志。

{table(['RQ','已有证据','本轮新增','可支持主张','仍缺','状态'],[[r[k] for k in ('RQ','existing','this_continuation','claim_allowed','missing','status')] for r in matrix])}

证据链：VM36→VM44冻结规则输入→`{PRIOR.name}`整组留出预测→本Run风险复算。外部VM43只保留原有限探索结论，不作为规则训练/阈值选择数据。所有新Motion/Haze AP和DID必须待真实八格完成后再进入结果章节。

明确禁止的当前表述：独立图像泛化已证明；全部YOLO架构的因果规律；完整FM空间Gap率17/21；候选失败路径是物理非法条件；正式RQ4已闭合；平衡准确率93.42%意味着可靠自动选模。

未修改原正式协议：YOLOv8s/YOLO26s、seed42/43/44、CONTROL/ALL_FIVE、1280训练100轮，以及五项v8单增强消融。其完整预算、目标设备、训练/评价仍需独立准入，不以约5小时历史余额推定足够。
'''
    directory=ROOT/'datasets/reportcn';stamp=out.name.split('_')[-1]
    docs=[('VM45_科学报告_falsePASS修订_CN_'+stamp+'.md',report),('VM45_导师短汇报_CN_'+stamp+'.md',brief),('VM45_RQ手稿证据矩阵_CN_'+stamp+'.md',manuscript)]
    paths=[]
    for name,content in docs:
        path=directory/name
        with path.open('x',encoding='utf-8') as f:f.write(content)
        paths.append(path)
    assert len({p.read_text(encoding='utf-8') for p in paths})==3
    return paths


def make_plot(out,risks):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].bar(['Tree','RF','Always FAIL'],[r['false_PASS_rate_among_actual_FAIL']*100 for r in risks]);axes[0].set(ylabel='Actual FAIL predicted PASS (%)',title='Denominator:76 actual FAIL')
    axes[1].bar(['True PASS','Actual FAIL'],[6,10],color=['#288c79','#ab5868']);axes[1].set(ylabel='Count within16 predicted PASS',title='Tree/RF PASS precision:37.5%')
    fig.text(.01,.01,'Fixed whole-P held-out development predictions only; no independent-image confirmation',fontsize=8);fig.tight_layout(rect=(0,.06,1,1))
    fig.savefig(out/'plots/false_PASS_risk.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
