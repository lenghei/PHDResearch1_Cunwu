"""Build source-driven figures and the post80 scientific evidence report."""
import argparse
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from protocol import ROOT,HERE,EXPERIMENT,load,save,source,sha256,write_atomic
from common import seal,verify_hashes
from historical_review import table,csv_file
from full20_results import static_comparison,outcomes,refresh_results
from full20_completion import read_pair
from post80_replay import ANALYSIS,run_path
from post80_science import check_numeric_table


def save_verified(path,data,out):
    if path.exists():
        if load(path)!=data:raise ValueError('Previously frozen output differs: '+path.name)
    else:save(path,data,out)


def figures(out):
    import numpy as np
    os.environ['MPLCONFIGDIR']=str(out/'runtime/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    provenance=out/'FIGURE_PROVENANCE.json'
    if provenance.exists() and load(provenance).get('renderer_version')=='POST80_PLOT_V2':
        for r in load(provenance)['outputs']:
            if sha256(Path(r['path']))!=r['sha256']:raise ValueError('A completed figure changed.')
        return
    effects=table(out/'RESOLUTION_EFFECTS.csv');pids=sorted({r['p_id'] for r in effects})
    mapping=[{'plot_id':f'{i+1:02d}','P':p,'features':next(r['features'] for r in effects if r['p_id']==p)} for i,p in enumerate(pids)]
    mapping_file=out/'PLOT_CONTEXT_MAPPING.csv'
    if mapping_file.exists():
        if table(mapping_file)!=mapping:raise ValueError('Plot context mapping changed.')
    else:csv_file(mapping_file,mapping)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(12.5,9),sharey=True,layout='constrained')
    for ax,ck,name in zip(axes,('ck01','ck05'),('YOLO26s','YOLOv8s')):
        selected={r['p_id']:r for r in effects if r['checkpoint']==ck and r['size']=='all' and r['metric']=='map50_95'}
        values=np.array([float(selected[p]['estimate']) for p in pids]);low=np.array([float(selected[p]['CI95_low']) for p in pids]);high=np.array([float(selected[p]['CI95_high']) for p in pids])
        ax.errorbar(values,np.arange(20),xerr=[values-low,high-values],fmt='o',capsize=2,color='#28658c',markersize=4)
        ax.axvline(0,color='.6',linewidth=.8);ax.set_title(name);ax.set_xlabel('AP50:95 difference: 1280 - 640 (0-1)');ax.grid(axis='x',alpha=.15)
    aliases={'Overexposed_mild':'Over-mild','Overexposed_strong':'Over-strong','Underexposed_mild':'Under-mild',
        'Underexposed_strong':'Under-strong','LowLight':'Low light','Normal':'Normal','Shot_low':'S100','Shot_high':'S20',
        'Defocus_r3':'D3','Defocus_r6':'D6','Gauss_s5':'G5.5','Gauss_s15':'G15','Gauss_s30':'G30',
        'Downscale_x1_5':'DS1.5','Downscale_x2':'DS2','JPEG_q25':'J25','JPEG_q50':'J50','JPEG_q75':'J75'}
    axes[0].set_yticks(range(20),[r['plot_id']+' '+','.join(aliases.get(f,f) for f in r['features'].split('|')) for r in mapping],fontsize=7.5);axes[0].invert_yaxis()
    fig.suptitle('Within-context resolution effects: 548 shared validation images')
    fig.savefig(out/'resolution_effects.png',dpi=160);plt.close(fig)
    means=table(out/'FINITE20_MARGINALS.csv');sizes=('all','tiny','small','medium','large')
    fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
    for i,(ck,name,color) in enumerate((('ck01','YOLO26s','#28658c'),('ck05','YOLOv8s','#a35b20'))):
        selected={r['size']:r for r in means if r['contrast']==ck+'_1280_minus640_mean' and r['metric']=='map50_95'}
        values=np.array([float(selected[s]['estimate']) for s in sizes]);low=np.array([float(selected[s]['CI95_low']) for s in sizes]);high=np.array([float(selected[s]['CI95_high']) for s in sizes])
        ax.errorbar(np.arange(5)+(i-.5)*.13,values,yerr=[values-low,high-values],fmt='o',capsize=3,label=name,color=color)
    ax.set_xticks(range(5),sizes);ax.axhline(0,color='.6',linewidth=.8);ax.set_ylabel('Mean AP50:95 difference (0-1)')
    ax.set_title('1280 - 640: equal mean across 20 selected contexts');ax.legend(frameon=False);ax.grid(axis='y',alpha=.15)
    fig.savefig(out/'size_dependent_resolution.png',dpi=160);plt.close(fig)
    unique={r['pair_id']:r for r in table(out/'STATIC_FALSE_POSITIVE_SETTINGS.csv')};rows=[unique[p] for p in sorted(unique)]
    fig,ax=plt.subplots(figsize=(10,6),layout='constrained')
    for i,r in enumerate(rows):
        value=float(r['Tiny_AP50_95']);lo=float(r['Tiny_AP_CI_low']);hi=float(r['Tiny_AP_CI_high'])
        ax.errorbar(value,i,xerr=[[value-lo],[hi-value]],fmt='o',capsize=3,color='#287655' if r['state']=='PASS' else '#777777')
    labels=[r['features'].replace('Normal|','').replace('|Normal','')+' / '+r['d_id'] for r in rows]
    ax.set_yticks(range(len(rows)),labels,fontsize=8);ax.invert_yaxis();ax.axvline(.1,color='#a35b20',linestyle='--',label='Tiny AP threshold 0.10')
    ax.set_xlabel('Tiny AP50:95 (0-1)');ax.set_title('All four settings for each static-gap counterexample');ax.legend(frameon=False)
    fig.savefig(out/'static_gap_counterexamples.png',dpi=160);plt.close(fig)
    save(provenance,{'renderer_version':'POST80_PLOT_V2','inputs':[source(out/n) for n in ('RESOLUTION_EFFECTS.csv','FINITE20_MARGINALS.csv','STATIC_FALSE_POSITIVE_SETTINGS.csv')],
        'outputs':[source(out/n) for n in ('resolution_effects.png','size_dependent_resolution.png','static_gap_counterexamples.png')],
        'scope':'Fixed-checkpoint development evidence; pointwise paired-image intervals, no training-seed or multiplicity claim.'},out,replace=provenance.exists())
    write_atomic(out/'FIGURE_CAPTIONS_EN.md','# Figure captions\n\nWithin-context resolution effects. Each point is the 1280-minus-640 AP50:95 difference under the same P configuration, checkpoint, original pixels and 548-image validation population. Bars are pointwise 95% intervals from 200 shared image resamples. D is defocus radius; G is Gaussian sigma; S is photon level; DS is downscale factor; J is additional JPEG quality. Context IDs map to PLOT_CONTEXT_MAPPING.csv. Source: RESOLUTION_EFFECTS.csv.\n\nSize-dependent mean effects. The figure averages the 20 preselected context-level AP values equally; it is neither pooled-image COCO AP nor a probability-population estimate. It separates overall, tiny, small, medium and large original-coordinate bins. Source: FINITE20_MARGINALS.csv.\n\nStatic-gap counterexamples. All four measured settings are shown for each of the three P configurations where at least one setting passes. The line is the frozen 0.10 Tiny AP threshold; the separate 100 ms cached-input latency test is recorded in the source table. Static noncoverage does not imply detection failure. Source: STATIC_FALSE_POSITIVE_SETTINGS.csv.\n',out,replace=(out/'FIGURE_CAPTIONS_EN.md').exists())


def validate_tables(out,found):
    original=run_path(ANALYSIS);expected=[];intervals=[]
    for identity,item in found.items():
        expected.extend({'pair_id':identity,**r} for r in item['metrics']['per_area'])
        intervals.extend({'pair_id':identity,**r} for r in item['bootstrap']['intervals'])
    check_numeric_table(original/'CONDITION_METRICS.csv',expected,('pair_id','size'),('map50','map50_95','ar50','ar50_95','gt_count','max_det','image_count'))
    saved={tuple(r[k] for k in ('pair_id','size','metric')):r for r in table(original/'CONFIDENCE_INTERVALS.csv')}
    if len(saved)!=len(intervals):raise ValueError('CI row count differs.')
    for r in intervals:
        actual=saved[tuple(r[k] for k in ('pair_id','size','metric'))]
        if ast.literal_eval(actual['ci95'])!=r['ci95'] or int(actual['n_valid'])!=r['n_valid']:
            raise ValueError('CI extraction differs.')
    seed=next(iter(found.values()))['bootstrap']['draws']['seed'];_,_,gaps=outcomes(found,seed)
    _,agreement=static_comparison(found,gaps)
    check_numeric_table(original/'STATIC_AGREEMENT.csv',agreement,('policy','tau','L_cached_ms'),
        ('decisive_P','unresolved_P','TP_gap','FP_gap','FN_gap','TN_gap','accuracy_on_decisive_selected_P','gap_precision','gap_recall'))
    save_verified(out/'SUMMARY_TABLE_RECONSTRUCTION.json',{'condition_rows':len(expected),'CI_rows':len(intervals),
        'all_numeric_extractions_match':True,'static_policy_rows':len(agreement),'pointwise_CI_draws_reused':200,
        'new_COCO_resamples':0},out)


def freeze_minimum_plan(out,found):
    source_plan=run_path('vm36_supervisor_reproduction_full20_remaining51_20261110T120947_b5f3997d')/'EXECUTION_PROTOCOL.json'
    old=load(source_plan);design=load(out/'MATCHED_CONTRAST_PREREGISTRATION.json')
    for key in ('GT','image_members','ZIP','FMP','FMD','generator'):
        if sha256(Path(old[key]['path']))!=old[key]['sha256']:raise ValueError('Protected protocol input changed: '+key)
    normal=next(p for p in design['new_P_configurations'] if p['leaves']==['Normal'])
    detectors={v['spec']['d']['id']:v['spec']['d'] for v in found.values()}
    for spec in {r['checkpoint']:r for r in detectors.values()}.values():
        if sha256(Path(spec['path']))!=spec['sha256']:raise ValueError('Checkpoint changed after preparation.')
    pairs=[{'id':normal['id']+'__'+key,'p':normal,'d':d,'images':548,'status':'NOT_RUN'} for key,d in sorted(detectors.items())]
    if any(p['id'] in found for p in pairs) or len(pairs)!=4:raise ValueError('The minimum plan is not four missing settings.')
    contrasts=[]
    for c in design['contrasts']:
        if not c['name'].startswith('NORMAL_REFERENCE__'):continue
        for ck in ('ck01','ck05'):
            terms=[(1,c['existing_reference'],1280),(-1,c['existing_reference'],640),(-1,normal['id'],1280),(1,normal['id'],640)]
            contrasts.append({'name':c['name']+'__'+ck,'metric':'AP50:95','size':'all',
                'terms':[{'coefficient':s,'pair_id':p+'__'+ck+'_Infer'+str(r)} for s,p,r in terms]})
    save_verified(out/'FROZEN_MINIMUM_GPU_PLAN.json',{'status':'DESIGN_FROZEN_RESOURCE_BLOCKED','version':'VM36_NORMAL_REFERENCE_V1',
        'new_pairs':pairs,'contrasts':contrasts,'seed':old['seed'],'bootstrap_draws':200,
        'source_protocol':source(source_plan),'source_data':{k:old[k] for k in ('GT','image_members','ZIP','FMP','FMD','generator')},
        'stage_order':old['stage_order'],'precision':'FP32','actual_tensor_NCHW':'[1,3,resolution,resolution]',
        'inference':'Unchanged one-to-many, conf0.001, class-aware multi-label NMS IoU0.7, max_det3000, max_nms30000, TF32 off.',
        'evaluation':'Original native10, original bbox areas and ignore-union COCO evaluator; no head or protocol replacement.',
        'confidence_policy':'Exploratory pointwise paired-image95% intervals; no familywise significance declaration.',
        'GPU_authorization':None,'worker_budget_seconds':None,'training_allowed':False,
        'required_before_execution':'New explicit finite allocation, source/health/resource preflight and a new immutable execution RunID.'},out)
    matrix=table(out/'FEATURE_MATRIX.csv')
    save_verified(out/'FEATURE_SCHEMA.json',{'predictors':[k for k in matrix[0] if k not in ('pair_id','P_group','label')],
        'metadata_not_predictors':['pair_id','P_group','label'],'illumination':'Mutually exclusive categorical indicators; not a continuous intensity scale.',
        'response_features_forbidden':['AP','GT counts','bootstrap outcomes','held-out labels'],
        'classifier_fitted':False},out)


def report(out,replay,loader):
    bench=load(replay/'worker_result.json');gpu=load(out/'GPU_RESOURCE_GATE.json');learn=load(out/'LEARNING_DIAGNOSTIC.json')
    design=load(out/'MATCHED_CONTRAST_PREREGISTRATION.json');false=table(out/'STATIC_FALSE_POSITIVE_SETTINGS.csv')
    passes={r['pair_id']:r for r in false if r['state']=='PASS'};means=table(out/'FINITE20_MARGINALS.csv')
    precision=next(r for r in table(out/'UNIFORM_SAMPLE_SIZE_SCENARIOS.csv') if r['population']=='1872' and r['desired_absolute_error']=='0.1')
    lines=['# VM36：80组之后的科学分析','','## PROBLEM / WHY','',
        '检验分辨率变化与哪些退化相关、静态gap为何出现误判，以及现有证据能否支持失败预测和定向训练。原20×4结果已完整，但覆盖率本身不能解释交互或证明泛化。','',
        '## SOLUTION','',
        f'读取80组保存的预测、原生十类GT、COCO指标及200次配对重采样。每组548张VisDrone val，不新增图像或检测器，不重复GPU推理。此次CPU从原始预测重新评价{bench["replayed_pairs"]}/80组，逐项比较面积/类别AP、AR及预测数量，实际worker墙钟{bench["CPU_worker_seconds"]:.3f}秒。',
        '模型为ck01 YOLO26s及ck05 YOLOv8s，历史FT1280、seed42，推理640/1280；完整SHA、参数和原始目录仍见原结果MEASURED_DESIGN.json及EVIDENCE_SOURCES.json。best实际选中epoch及创建期数据暴露仍未完全核实。','',
        f'源表：[20条件平均及配对区间]({(out/"FINITE20_MARGINALS.csv").as_posix()})；[原始预测、GT及Run索引]({(run_path(ANALYSIS)/"EVIDENCE_SOURCES.json").as_posix()})。','',
        '## RESULT：已有证据支持什么','',
        '| 模型 | 尺寸 | 1280−640的20条件平均AP50:95差值 | 点态95%区间 |','|---|---|---:|---|']
    for ck,name in (('ck01','YOLO26s'),('ck05','YOLOv8s')):
        for size in ('all','tiny','small','medium','large'):
            r=next(r for r in means if r['contrast']==ck+'_1280_minus640_mean' and r['size']==size and r['metric']=='map50_95')
            lines.append(f'| {name} | {size} | {float(r["estimate"]):+.6f} | [{float(r["CI95_low"]):+.6f}, {float(r["CI95_high"]):+.6f}] |')
    lines+=['','AP差值按0到1原值显示。Tiny/Small的平均收益为正，Medium/Large的平均差值为负；整体平均差值区间含0。整体AP不是各尺寸AP的线性加权平均，不能把这些值直接相加解释。分组均值仅针对20个目的性条件，不是均匀总体结论。',
        '同一checkpoint内的分辨率对照可作为CONTROLLED_COMPARISON；原JPEG八角点支持JPEG端点交互。复合上下文差异不能单独归因于Gaussian、Defocus或光照，缺少的反事实仍是UNSUPPORTED_CLAIM。','',
        '### 静态误判的三个条件','',
        '主场景保持Tiny AP阈值0.10、缓存输入流程时延上限100ms。三种静态策略都把20个条件判为gap，实测却有3个条件被某设置覆盖。以下是全部通过设置；其余设置也保留在STATIC_FALSE_POSITIVE_SETTINGS.csv。','',
        '| P条件 | 通过设置 | Tiny AP50:95 | 95%区间 | 时延均值上界(ms) |','|---|---|---:|---|---:|']
    for r in passes.values():
        folder=Path(r['raw_source_folder'])
        lines.append(f'| [{r["features"].replace("|"," + ")}]({(folder/"configuration.json").as_posix()}) | [{r["d_id"]}]({(folder/"metrics.json").as_posix()}) | {float(r["Tiny_AP50_95"]):.6f} | [{float(r["Tiny_AP_CI_low"]):.6f}, {float(r["Tiny_AP_CI_high"]):.6f}] | {float(r["mean_cached_latency_upper95_ms"]):.3f} |')
    lines+=['','两种定义不同：静态策略判断增强配方覆盖，实测判断AP与时延是否达标。把静态未覆盖直接用作失败预测，会在这三个条件误报。这里未识别出具体的物理或网络机制。',
        '原训练6471张图片均为JPEG编码；JPEG_q25/q75表示额外重压缩，不是从无损采集图首次压缩。是否启用某增强、预训练暴露和实际不变性仍不能由args.yaml单独推断。',
        '静态85%一致率与恒预测gap基线相同，没有额外的分类区分能力。','',
        '### 失败预测能做什么','',
        f'80组的主要标签为PASS={learn["pair_class_balance"].get("PASS",0)}、FAIL={learn["pair_class_balance"].get("FAIL",0)}、UNCERTAIN={learn["pair_class_balance"].get("UNCERTAIN",0)}；PASS只涉及{learn["PASS_P_groups"]}个P组。按P留组，所有四个设置一起保留，排除未决标签。',
        '仅计算训练折多数类的开发对照，没有拟合决策树或搜索超参数。独立图像/场景边界未通过；少数类P组太少，ML_INSUFFICIENT_EVIDENCE。GROUP_FOLDS.csv保存实际组成员，LEARNING_DIAGNOSTIC.json保存分母、混淆矩阵及指标。',
        f'排除2组未决标签后，多数类对照在78组上的准确率为{learn["majority_grouped_development_baseline"]["accuracy"]:.6f}，balanced accuracy={learn["majority_grouped_development_baseline"]["balanced_accuracy"]:.6f}，macro F1={learn["majority_grouped_development_baseline"]["macro_F1"]:.6f}。高普通准确率主要反映类别不均衡。',
        '配置保留、图像保留、外部验证是三个不同目标，本轮没有INDEPENDENT_VALIDATION。','',
        '## 下一项最小合法对照','',
        f'最小方案新增{design["primary_minimal_plan"]["new_P"]}个P（Normal、无额外退化）×四个设置，共4组、2192张次。它同时为已有Normal+Gauss_s15、Normal+JPEG_q25、Normal+JPEG_q75提供相同背景参考。选择依据是缺失反事实结构，没有用新AP选条件。',
        'FlamaPy在原46368个合法配置中核验候选成员。原UVL参数和五条约束不变；Normal只使用gain=1、bias=0。6张固定原图的CPU像素检查确认该配方不改变已解码像素。',
        '重度锚点的备选设计需7个新P、28组：分别改变Defocus、Downscale、Gaussian、Shot级别、JPEG级别、光照级别，另含一个合法光照+Shot联合变化。不能删除LowLight要求的ShotNoise，也不能在保留Shot_high时换为Normal。联合改法不解释为单因素作用。',
        '4组方案仅回答Normal背景上的Gaussian/JPEG对照；重度复合条件的成分分解属于28组备选方案，两者不能互相替代。',
        f'4组的保守成本筛查为{gpu["conservative_minimum_batch_screen_seconds"]:.1f}秒，依据已实测完整组及缓存开销，不是新实验实测时间。GPU状态RESOURCE_BLOCKED：旧授权的54组范围已经完成，剩余时钟不等于新配置授权。所有新比较标记EXPLORATORY，尚无新AP。','',
        '## RQ4与独立数据','',
        '全6471图候选标签视图已接入现有Ultralytics数据集校验；343200个目标全部有效。原图未复制，原始图片与标注SHA保持不变；3条零面积行和4个重复框只在派生标签视图处理。',
        '在1280、batch2、workers0下，两次各16锚图的CPU常规增强/组批输出及标签SHA完全复现。Mosaic可能读取其他训练图，实际访问序列另存，不能把锚图数当成全部曝光量。没有forward/backward、optimizer或EMA训练。',
        '初始化SHA、五类定向增强范围与缺失参数、生产训练器、GPU显存及完整训练成本仍未冻结。准备了两模型×控制/定向两臂的4-run探索计划，不替代三seed与逐增强消融；本轮不训练。',
        'test-dev仍DATASET_BLOCKED。直接重读checkpoint元数据、曲线和配置，不能用缺少日志或检索无命中证明没有历史调参。UAVDT官方DET/MOT格式与忽略规则保留，但原图来源、类别语义、条件标签和独立性仍不足；没有跨数据集AP。','',
        '## CONCLUSION / LIMITATION','',
        'RQ1 PARTIAL；RQ2 PARTIAL；RQ3 PARTIAL；RQ4 NOT_EVALUATED。当前推进的是合法反事实设计、静态误判解释和真实数据链路准备，不是新的泛化或训练有效性结论。',
        '20个P非概率样本，17/20不是完整空间失败率。均匀采样方案分别列出1872个可执行P投影与46368个逻辑P的精度/成本，后者仍有渲染绑定缺口；没有抽样执行。',
        f'例如1872个P投影、误差目标±0.10时，正态/FPC规划为{precision["normal_FPC_planning_n"]}个P（{precision["normal_planned_pairs"]}组）；保守Chebyshev充分样本量为{precision["Chebyshev_sufficient_n"]}个P。前者不是精确95%保证，成本表仅作筛查，不代表已测GPU用时。',
        '图像Bootstrap只针对固定权重和固定退化实现，区间为点态；真实航次聚类、训练seed和多重比较结论未覆盖。','',
        '## 复现与证据','',f'- 本轮目录：{out}',f'- 原始80组：{run_path(ANALYSIS)}',f'- CPU复算：{replay}',f'- 全训练池CPU适配器：{loader}',
        '- QUESTION_TO_EVIDENCE.csv区分可控对照、描述性关联、探索发现及不支持的主张。',
        '- MATCHED_CONTRAST_PREREGISTRATION.json保存数值参数、合法替代及未执行状态。',
        '- FROZEN_MINIMUM_GPU_PLAN.json保存4个准确pair ID、权重/数据SHA、随机种子、六个对比的符号定义及尚无GPU授权的状态。',
        '- FIGURE_PROVENANCE.json和FIGURE_CAPTIONS_EN.md连接三张图的真实源表。',
        '- 复算80组：post80_replay.py --request <本次指令文本路径>；只读预测，CPU执行。',
        '- 重建分析：post80_science.py --prepare <本次指令文本路径>；生成新Run，不会运行GPU。',
        '- 完整数据链路：post80_loader.py；CPU执行，生成新适配器Run。',
        '- 数据来源与训练准备：post80_provenance.py --science-run <本轮RunID> --loader-run <适配器RunID>。',
        '- 完成汇总：post80_finalize.py --science-run <本轮RunID> --replay-run <复算RunID> --loader-run <适配器RunID>。']
    note=ROOT/'datasets/reportcn'/('vm36_supervisor_reproduction_下一阶段科学证据_'+out.name.split('_')[-1]+'.md')
    write_atomic(note,'\n'.join(lines)+'\n',out);save(out/'REPORT_REFERENCE.json',source(note),out)


def finish(out,replay,loader):
    if (out/'SHA256SUMS.txt').exists():raise ValueError('This science run is already sealed.')
    for child,file in ((replay,'worker_result.json'),(loader,'LOADER_ACCEPTANCE.json')):verify_hashes(child,(file,))
    result=load(replay/'worker_result.json')
    if result['replayed_pairs']!=80 or result['status']!='COMPLETE':raise ValueError('Full80 CPU replay is not complete.')
    found={r['pair_id']:read_pair(Path(r['folder'])) for r in load(run_path(ANALYSIS)/'EVIDENCE_SOURCES.json')['conditions']}
    validate_tables(out,found);freeze_minimum_plan(out,found)
    checks=table(replay/'REPLAY_VERIFICATION.csv')
    if {r['pair_id'] for r in checks}!=set(found) or len(checks)!=80:raise ValueError('CPU replay membership differs.')
    for r in checks:
        if r['prediction_SHA256']!=found[r['pair_id']]['checks']['predictions.jsonl.gz']:raise ValueError('CPU replay source differs.')
    figures(out);report(out,replay,loader)
    shutil.copyfile(Path(__file__),out/'executed_finalize.py')
    shutil.copyfile(HERE/'post80_provenance.py',out/'executed_provenance.py')
    save(out/'FINAL_EVIDENCE_STATUS.json',{'status':'CPU_COMPLETE_WITH_SCIENTIFIC_AND_RESOURCE_BLOCKERS',
        'original_pairs_preserved':80,'raw_CPU_replayed_pairs':80,'new_GPU_pairs':0,'new_training_runs':0,
        'RQ1':'PARTIAL','RQ2':'PARTIAL','RQ3':'PARTIAL','RQ4':'NOT_EVALUATED',
        'GPU':'RESOURCE_BLOCKED','ML':'ML_INSUFFICIENT_EVIDENCE','testdev':'DATASET_BLOCKED',
        'independent_validation_results':0,'CPU_replay_seconds':result['CPU_worker_seconds'],
        'source_inputs':[source(replay/'SHA256SUMS.txt'),source(loader/'SHA256SUMS.txt')],
        'source_components':[source(Path(__file__)),source(HERE/'post80_provenance.py'),source(HERE/'post80_legal_sat.py')]},out)
    write_atomic(out/'diary.md','# Scientific stage after80 measurements\n\nReconstructed saved contrasts, verified complete raw CPU replay, isolated static-gap counterexamples and documented insufficient classifier evidence. Froze legal missing contrasts without response-based selection. A full-pool CPU label adapter passed; no GPU or training was started.\n',out)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>VM36 post80 scientific evidence</h1><p>CPU complete; new GPU, independent testing and training remain blocked.</p>'+''.join('<p><a href="'+p+'">'+p+'</a></p>' for p in ('QUESTION_TO_EVIDENCE.csv','STATIC_FALSE_POSITIVE_SETTINGS.csv','LEARNING_DIAGNOSTIC.json','FINITE20_MARGINALS.csv','PLOT_CONTEXT_MAPPING.csv','FROZEN_MINIMUM_GPU_PLAN.json','GPU_RESOURCE_GATE.json','RQ4_READINESS_UPDATE.json','FIGURE_CAPTIONS_EN.md','REPORT_REFERENCE.json'))+''.join('<img style="max-width:100%" src="'+p+'">' for p in ('resolution_effects.png','size_dependent_resolution.png','static_gap_counterexamples.png')),out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'POST80_SCIENTIFIC_EVIDENCE','status':'COMPLETE_WITH_BLOCKERS'},out,replace=True)
    seal(out);refresh_results(run_path(ANALYSIS))
    latest=load(out.parent/'latest.json');latest.update(latest_post80_science_RunID=out.name,latest_post80_science_status='COMPLETE_WITH_BLOCKERS');save(out.parent/'latest.json',latest,out,replace=True)
    landing=out.parent/'index.html';html=landing.read_text(encoding='utf-8')
    link='<p><a href="'+out.name+'/index.html">Latest scientific analysis and source-driven figures</a></p>'
    write_atomic(landing,html.replace('<h1>VM36 latest measurements</h1>','<h1>VM36 latest measurements</h1>'+link),out,replace=True)
    print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--science-run',required=True);p.add_argument('--replay-run');p.add_argument('--loader-run');p.add_argument('--plots-only',action='store_true');a=p.parse_args()
    if a.plots_only:figures(run_path(a.science_run))
    elif a.replay_run and a.loader_run:finish(run_path(a.science_run),run_path(a.replay_run),run_path(a.loader_run))
    else:p.error('Full replay and loader RunIDs are required for finalization.')
