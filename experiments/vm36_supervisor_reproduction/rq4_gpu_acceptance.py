"""Verify saved canaries and freeze the remaining matched-training decisions."""
import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from protocol import ROOT, HERE, EXPERIMENT, load, save, source, sha256, write_atomic
from common import seal, verify_hashes
from main.pipeline import new_run
from post80_replay import run_path
from rq4_real_canary import PREPARED, SOURCE

REVIEW='vm36_supervisor_reproduction_rq4_training_review_cpu_20261111T095859_6043d9ba'


def csv_out(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def verify_canary(out):
    verify_hashes(out,('CANARY_PROTOCOL.json','worker_result.json','REAL_TRAINING_ACCEPTANCE.json',
        'RELOAD_INFERENCE_ACCEPTANCE.json','training_batches.jsonl','metrics.json','predictions.jsonl.gz','ground_truth.json'))
    result=load(out/'worker_result.json');training=load(out/'REAL_TRAINING_ACCEPTANCE.json')
    plan=load(out/'CANARY_PROTOCOL.json');rows=read_lines(out/'training_batches.jsonl')
    reload=load(out/'RELOAD_INFERENCE_ACCEPTANCE.json')
    if result['status']!='CANARY_PASS' or not result['source_inputs_unchanged']:raise ValueError('Canary is not accepted.')
    expected={'epochs_completed':1,'training_exposures':64,'forward_batches':32,'backward_batches':32,'optimizer_updates':7,'EMA_updates':7}
    if any(training[k]!=value for k,value in expected.items()):raise ValueError('Actual training counts differ.')
    anchors=[identity for row in rows for identity in row['image_ids']]
    if len(rows)!=32 or len(anchors)!=64 or len(set(anchors))!=64 or set(anchors)!={r['image_id'] for r in plan['train']}:
        raise ValueError('Training membership differs.')
    if any(not r['gradients_finite'] or r['tensor']['NCHW']!=[2,3,1280,1280] for r in rows):raise ValueError('Batch health/shape differs.')
    if not reload['raw_COCO_replay_equal'] or reload['images']!=16:raise ValueError('Reload inference/replay failed.')
    if sha256(Path(training['checkpoint']['path']))!=training['checkpoint']['sha256']:raise ValueError('Checkpoint SHA differs.')
    return plan,training,rows,reload


def finalize(failed_ids, passed_id):
    began=time.perf_counter();passed=run_path(passed_id)
    plan,training,batches,reload=verify_canary(passed)
    failures=[run_path(value) for value in failed_ids]
    allocations=[];stages=[]
    for run in failures+[passed]:
        verify_hashes(run,('CANARY_PROTOCOL.json','worker_result.json','GPU_ALLOCATION.json','supervisor.json'))
        allocation=load(run/'GPU_ALLOCATION.json');allocations.append(allocation)
        result=load(run/'worker_result.json');protocol=load(run/'CANARY_PROTOCOL.json')
        end=datetime.fromtimestamp((run/'supervisor.json').stat().st_mtime,timezone.utc)
        if end>datetime.fromisoformat(allocation['deadline_UTC']):raise ValueError('Stage ended after the original deadline.')
        stages.append({'RunID':run.name,'AMP':protocol['args']['amp'],'status':result['status'],
            'worker_seconds':result['worker_seconds'],'source_inputs_unchanged':result['source_inputs_unchanged'],
            'supervisor_file_mtime_UTC':end.isoformat(),
            'protocol':source(run/'CANARY_PROTOCOL.json'),'result':source(run/'worker_result.json')})
    if any(row!=allocations[0] for row in allocations):raise ValueError('GPU allocation was reset.')
    if sum(row['worker_seconds'] for row in stages)>600:raise ValueError('Worker allocation exceeded.')
    for run in failures+[passed]:
        supervisor=load(run/'supervisor.json')
        if supervisor.get('budget_reset',False):raise ValueError('GPU budget was reset.')
    out,_=new_run({'output_root':str(passed.parent)},EXPERIMENT,'rq4_real_gpu_acceptance_cpu')
    shutil.copyfile(Path(__file__),out/'executed_source.py')
    prepared=run_path(PREPARED);review=run_path(REVIEW);admitted=run_path(SOURCE)
    verify_hashes(prepared,('SUPERVISOR_REQUIREMENTS.json','MATCHED_RQ4_MATRIX_PROPOSAL.csv','OPTIMIZER_EXPOSURE_PLAN.json'))
    verify_hashes(review,('SCIENTIFIC_PARAMETER_PACKET.json',))
    requirements=load(prepared/'SUPERVISOR_REQUIREMENTS.json');packet=load(review/'SCIENTIFIC_PARAMETER_PACKET.json')
    for key in ('FMP','FMD','generator','GT','image_members'):
        ref=requirements['frozen_source_hashes'][key]
        if sha256(Path(ref['path']))!=ref['sha256']:raise ValueError('Protected scientific source changed: '+key)
    refs={'accepted_preparation':source(prepared/'SHA256SUMS.txt'),'scientific_packet':source(review/'SCIENTIFIC_PARAMETER_PACKET.json'),
        'source_admission':source(admitted/'INITIALIZATION_SOURCE_GATE.json'),'canary_stages':stages,
        'original_supervisor_requirements':source(prepared/'SUPERVISOR_REQUIREMENTS.json'),
        'source':source(Path(__file__))}
    save(out/'inputs.json',refs,out)
    save(out/'GPU_BUDGET_ACCEPTANCE.json',{'status':'PASS','original_allocation':allocations[0],
        'stages':stages,'worker_seconds_sum':sum(r['worker_seconds'] for r in stages),
        'budget_reset':False,'further_formal_training_seconds_authorized':0,
        'note':'The initial600-second canary deadline is shared by all corrective stages. Unused canary time is not a full-training allocation.'},out)
    initializations=load(prepared/'CANDIDATE_INITIALIZATION_MANIFEST.json');matrix=[]
    for initialization in initializations:
        for arm in ('DEFAULT_CONTROL','ALL_FIVE'):
            matrix.append({'architecture':initialization['architecture'],'seed':initialization['seed'],'arm':arm,
                'initialization_SHA256':initialization['native10_candidate']['sha256'],'epochs':100,'imgsz':1280,
                'batch':2,'nbs':64,'optimizer':'AdamW','lr0':.01,'cos_lr':True,'warmup_epochs':3,
                'train_images':6471,'development_images':548,'checkpoint_selection':'FIXED_EPOCH100_EMA_DESIGN_CHOICE',
                'precision':'PENDING_COMMON_PROTOCOL: AMP_FAILED_FP32_ENGINEERING_PASSED','workers':'PENDING:0_ENGINEERING_ONLY',
                'recipe_status':'CONVENTIONAL_CONTROL_DEFINED' if arm=='DEFAULT_CONTROL' else 'BLOCKED_DESIGN',
                'execution_status':'RESOURCE_BLOCKED_NOT_RUN'})
    if len(matrix)!=12:raise ValueError('Primary matched design is not12 runs.')
    csv_out(out/'MATCHED_TRAINING_DESIGN.csv',matrix)
    design={'status':'PARTIALLY_FROZEN_NOT_EXECUTABLE','primary_runs':12,'initialization_decision':'FRESH_MATCHED_COCO_DERIVED_NATIVE10',
        'reason':'Historical selected epochs and creation-time initialization bytes are unknown. A historicalFT is not a matched fresh control.',
        'same_initialization_within_seed_pair':True,'training_members':source(prepared/'TRAINING_DATA_VERIFICATION.csv'),
        'development_members':source(prepared/'VALIDATION_LOADER_ACCEPTANCE.json'),
        'training_parameters':{k:plan['args'][k] for k in ('imgsz','batch','nbs','optimizer','lr0','lrf','cos_lr','warmup_epochs','warmup_bias_lr','momentum','weight_decay')},
        'epochs':100,'seeds':[42,43,44],'checkpoint_selection':'Fixed epoch100 EMA; a declared design choice, not a quoted supervisor requirement.',
        'full_source_update_plan':source(prepared/'OPTIMIZER_EXPOSURE_PLAN.json'),
        'augmentation_parameters':packet,'scientific_recipe_status':'BLOCKED_DESIGN',
        'activation_probabilities':None,'noise_mixture_probability':None,'approved_ranges':None,
        'motion_orientation':'UNAPPROVED_HISTORICAL_PROPOSAL_ONLY','haze_airlight':'UNAPPROVED_HISTORICAL_PROPOSAL_ONLY',
        'ablation':'Five per-family ablations onYOLOv8s required; leave-one-out versus add-one-in remains a documented unresolved choice.',
        'ablation_runs':'5 single-seed exploratory or15 three-seed runs; NOT_RUN',
        'precision_workers':'FP32/workers0 exercised; a shared full protocol must explicitly adopt and freeze them or validate another policy.',
        'budget':'RESOURCE_BLOCKED: no documented unconsumed full matched-training allocation',
        'full_epoch_cost':'NOT_MEASURED; mini-epoch throughput cannot establish100-epoch feasibility',
        'independent_testdev':'DATASET_BLOCKED','historical_evaluation_pairs_preserved':84,
        'new_matched_scientific_pairs':0,'new_robustness_evaluations':'NOT_EVALUATED',
        'evaluation_template':source(prepared/'FROZEN_DEVELOPMENT_EVALUATION.json')}
    save(out/'SCIENTIFIC_TRAINING_FREEZE.json',design,out)
    metrics=load(passed/'metrics.json')
    csv_out(out/'canary_area_metrics.csv',metrics['per_area']);csv_out(out/'canary_per_class_metrics.csv',metrics['per_class'])
    csv_out(out/'training_batches.csv',[{k:v for k,v in row.items() if k not in ('tensor','image_ids','loss_items')} for row in batches])
    events=read_lines(passed/'health_events.jsonl')
    updates=[row for row in events if row['event']=='optimizer_and_EMA_healthy']
    csv_out(out/'optimizer_updates.csv',[{'update':r['optimizer_updates'],'EMA':r['EMA_updates'],
        'gradient_norm':r['gradient_norm'],'learning_rates':json.dumps(r['learning_rates']),
        'check_and_step_seconds':r['check_and_step_seconds']} for r in updates])
    os_env=__import__('os').environ
    os_env['CUDA_VISIBLE_DEVICES']='';os_env['MPLCONFIGDIR']=str(out/'runtime/matplotlib')
    os_env['YOLO_CONFIG_DIR']=str(out/'runtime/ultralytics')
    os_env['YOLO_AUTOINSTALL']='false';os_env['YOLO_OFFLINE']='true';os_env['PYTHONDONTWRITEBYTECODE']='1'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(9,3.2),layout='constrained')
    axes[0].plot(range(1,33),[r['loss'] for r in batches]);axes[0].set(xlabel='Training batch',ylabel='Detection loss',title='Engineering canary:32 batches')
    axes[1].plot(range(1,33),[r['batch_wall_seconds'] for r in batches]);axes[1].set(xlabel='Training batch',ylabel='Seconds',title='Measured batch time')
    figure.savefig(out/'canary_training.png',dpi=170);plt.close(figure)
    env={name:importlib.metadata.version(name) for name in ('torch','ultralytics','numpy','opencv-python','albumentations')}
    save(out/'package_versions.json',env,out)
    command=[sys.executable,'-B','-X','utf8','-m','unittest','discover','-s',str(HERE/'tests'),'-p','test_rq4_real_canary.py','-v']
    tested=subprocess.run(command,cwd=ROOT/'main',capture_output=True,text=True,encoding='utf-8',timeout=120)
    write_atomic(out/'changed_entry_tests.log',tested.stdout+tested.stderr,out)
    if tested.returncode:raise RuntimeError('Changed-entry CPU regression failed.')
    test_count=int(re.search(r'Ran (\d+) tests',tested.stdout+tested.stderr).group(1))
    all_ap=next(r for r in metrics['per_area'] if r['size']=='all')
    training_wall=training['training_setup_epoch_val_save_seconds'];peak=training['training_validation_save_GPU_peak']
    summary={'status':'PARTIAL_WITH_LIMITATIONS','real_canary_status':'CANARY_PASS','precision':'EXPLICIT_FP32_REPAIR',
        'actual_training':training,'engineering_metrics':all_ap,'canary_worker_seconds':load(passed/'worker_result.json')['worker_seconds'],
        'original_worker_seconds_sum':sum(r['worker_seconds'] for r in stages),'changed_entry_CPU_tests_passed':test_count,
        'scientific_training_runs':0,'complete_matched_training_pairs':0,'old_evaluation_pairs':84,
        'targeted_recipe':'BLOCKED_DESIGN','full_matched_training':'RESOURCE_BLOCKED','independent_testdev':'DATASET_BLOCKED',
        'full_epoch_cost':'NOT_MEASURED','seed_variability':'NOT_EVALUATED','empirical_gap_change':'NOT_EVALUATED',
        'bootstrap':'NOT_EVALUATED_ENGINEERING_ONLY','exact_resume':'NOT_VALIDATED',
        'protected_sources_checked':list(requirements['frozen_source_hashes'].keys())[:5],
        'original_archives_recopied':False,'automatic_full_training':False}
    save(out/'FINAL_RQ4_STATUS.json',summary,out)
    note=ROOT/'datasets/reportcn'/('vm36_supervisor_reproduction_RQ4真实训练验收与配对方案_'+out.name.split('_')[-1]+'.md')
    text=[
        '# RQ4真实GPU训练验收与配对设计','',
        '## 1. PROBLEM','',
        '导师RQ4要求比较同条件控制训练与针对退化的增强训练。本轮先检查真实YOLO训练链路；工程Canary不是增强有效性实验。',
        f'最终验收RunID：`{out.name}`。真实训练RunID：`{passed.name}`。','',
        '## 2. WHY IS IT A PROBLEM?','',
        '- 首次AMP训练loss有限，但scale65536下首层梯度非有限；Health Gate在第一次更新前终止。optimizer/EMA均0次。',
        '- 改用明确标记的FP32后，32个batch和7次更新均健康；首次保存被局部计数回调的pickle错误阻断。该失败Run及partial文件保留。',
        '- 移除保存前的计数回调后完成新工程验证。没有替换Inf、跳过检查、降低batch/分辨率或重置预算。',
        '- AMP溢出是当前解释；没有在同一已保存增强tensor上做AMP/FP32配对诊断，不能认定唯一数值根因，也没有GPU硬件故障证据。','',
        '## 3. SOLUTION','',
        '### 实际工程训练','',
        f'- YOLOv8s，官方COCO源派生的native10初始化SHA：`{plan["initialization"]["sha256"]}`。',
        '- 固定64张训练/16张开发图；seed42、1280、batch2、workers0、nbs64、AdamW lr0=0.01、cosine、warmup3。',
        '- FP32为本轮显式修复条件；优化器、学习率、数据及常规增强保持原候选设置。没有新增目标退化。',
        '- 常规HSV/Mosaic等仍开启；现有Albumentations还包含Blur/MedianBlur/ToGray/CLAHE各p=.01，不能把控制臂称为完全无增强。',
        '- 实际完成1epoch、64锚图曝光、32forward、32backward、7optimizer和7EMA更新。Mosaic辅助图访问与64个锚图曝光不是同一计数。',
        '- warmup的库内最小100batch影响短Canary，所以不能按64/64推算更新次数。最后仍有7个microbatch的未提交梯度；未私自追加flush。它不是精确resume检查点。',
        f'- 成功worker {summary["canary_worker_seconds"]:.3f}秒；训练初始化/epoch/内部验证/保存段{training_wall:.3f}秒。',
        f'- 训练+验证GPU峰值allocated={peak["allocated"]}bytes，reserved={peak["reserved"]}bytes。',
        f'- FP32权重：`{training["checkpoint"]["path"]}`；SHA `{training["checkpoint"]["sha256"]}`。',
        '- 在线模型和EMA均FP32；保存前健康检查、临时保存、参数及buffer重载逐张量一致后提交；重新加载后完成16图研究协议预测。','',
        '| 开发工程评价口径 | 图像 | AP50 | AP50:95 |','|---|---:|---:|---:|',
        '| Ultralytics内部rect/maxDet300 |16|0|0|',
        f'| 研究square1280/ignore-union/maxDet3000 |16|{all_ap["map50"]:.12f}|{all_ap["map50_95"]:.12f}|','',
        'AP为0–1原值。内部GT1548与研究有效GT1509的过滤口径不同，不能混用。极小训练的低AP只描述工程输出，不是方法失败或成功的证据；没有训练前同口径比较和统计区间。',
        '研究预测和GT已保存，COCO从原始预测回放与首次结果完全一致；尺寸/逐类AP和AR见本Run CSV。','',
        '### 配对科学方案','',
        '- 主矩阵：YOLOv8s/YOLO26s × seed42/43/44 × CONTROL/ALL_FIVE＝12次，均NOT_RUN。每个seed的两臂使用同一份初始化SHA。',
        '- 采用新建COCO派生native10匹配控制/增强臂，避免将历史best与新训练末轮混为一组；这是fresh retraining设计，不是从历史FT继续训练。',
        '- 6471训练图、548内部开发图、1280、100epoch、batch2、nbs64及源配置优化参数固定。固定epoch100 EMA作为本方案选择规则，属于明确设计选择，并非导师原文唯一规定。',
        '- 全程共同精度/worker协议仍需冻结：FP32/workers0已工程验证，历史AMP/workers2不能直接沿用准入。YOLO26s真实训练尚未验收。',
        '- 五类增强的训练范围、触发概率、Noise混合、motion方向、haze airlight仍缺少充分支持，保留BLOCKED_DESIGN，不把CPU fixture或历史提案直接当批准参数。',
        '- YOLOv8s每类增强消融保留5项要求；单seed探索共17次或完整三seed共27次。leave-one-out与add-one-in尚未唯一确定。',
        '- 源码100epoch计划：647100锚图曝光、323600batch、10917optimizer/EMA更新，含最后1图batch和跨epoch梯度累积；这些是源码计划，不是实测。',
        '- 没有新的完整配对训练预算，RESOURCE_BLOCKED。64图mini-epoch不能可靠外推6471图×100epoch，未启动完整epoch成本测试。',
        '- 后续评价仍引用冻结Normal/JPEG25/JPEG75/Gaussian15及复合条件，同一548GT、原图面积及ignore-union口径。新训练权重另编ID。','',
        '## 4. WHY IS IT A GOOD SOLUTION?','',
        '训练链路已经用真实YOLO损失、反向传播、计数、健康检查及FP32重载验证。两次失败没有掩盖，最终结果与原始文件一一对应。',
        '目前只能得出FP32工程链路可执行。完整训练对照0组、增强收益NOT_EVALUATED、gap变化NOT_EVALUATED、训练seed变异NOT_EVALUATED。test-dev仍DATASET_BLOCKED；未作新的独立测试。','',
        '### 预算与保护','',
        f'三阶段worker累计{summary["original_worker_seconds_sum"]:.3f}秒，共用最初600秒授权和同一UTC截止时间；修复没有新建600秒预算。',
        '两次失败Run、84个旧评价点、冻结UVL/生成器/548GT与成员文件均保留。已复核列出的SHA；没有对整个145GiB档案重新哈希，因此不宣称完成全盘逐文件审计。',
        f'新增入口{test_count}项CPU测试通过，包括显式精度变更、禁止配置漂移、固定数据计数、warmup更新计划、计数回调移除后序列化及修复共用原截止时间。','',
        '### 证据与复现','',
        f'- 本轮汇总：`{out}`',f'- 成功Canary：`{passed}`',
        *['- 保留失败：`'+str(run)+'`' for run in failures],
        '- 汇总脚本只读已有证据，不启动GPU：',
        '`'+str(ROOT/'.venv/Scripts/python.exe')+' -B '+str(HERE/'rq4_gpu_acceptance.py')+' --failed '+' '.join(failed_ids)+' --passed '+passed_id+'`',
        '- 不重新执行已消耗的GPU票据。后续完整训练需要明确有限预算、确定增强配方及共同精度/worker协议。']
    write_atomic(note,'\n'.join(text)+'\n',out);save(out/'REPORT_REFERENCE.json',source(note),out)
    write_atomic(out/'diary.md','# RQ4 real GPU acceptance\n\nVerified failed AMP, failed serialization and accepted FP32 engineering evidence. Froze source-supported matched design; unsupported recipes and full training remain blocked. No GPU work in this stage.\n',out)
    complete_bookkeeping(out,summary,passed,time.perf_counter()-began)
    print(out,flush=True);print(note,flush=True)


def complete_bookkeeping(out,summary,passed,cpu_seconds):
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>RQ4 real GPU acceptance</h1><p>CANARY_PASS: explicit FP32 engineering repair. Full matched training:RESOURCE_BLOCKED.</p><img style="max-width:100%" src="canary_training.png"><p><a href="FINAL_RQ4_STATUS.json">Actual status</a> | <a href="canary_area_metrics.csv">Engineering metrics</a> | <a href="MATCHED_TRAINING_DESIGN.csv">Matched design</a> | <a href="SCIENTIFIC_TRAINING_FREEZE.json">Open decisions</a></p>',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'RQ4_REAL_GPU_ACCEPTANCE_CPU',
        'status':summary['status'],'real_canary_status':'CANARY_PASS','complete_matched_training_pairs':0,
        'CPU_seconds':cpu_seconds},out,replace=True)
    seal(out)
    from main.visualization import build_run_index
    build_run_index(out.parent,EXPERIMENT)
    latest=load(out.parent/'latest.json');latest.update(latest_RQ4_results_RunID=out.name,
        latest_RQ4_backend_RunID=passed.name,RQ4_training_status='CANARY_PASS_FULL_TRAINING_RESOURCE_BLOCKED',
        combined_measured_pairs=84,formal_measured_pairs=0,matched_training_pairs=0)
    save(out.parent/'latest.json',latest,out,replace=True)


def recover_bookkeeping(identity):
    out=run_path(identity)
    if not identity.startswith(EXPERIMENT+'_rq4_real_gpu_acceptance_cpu_') or (out/'SHA256SUMS.txt').exists():
        raise ValueError('Only an unsealed current acceptance run can finish bookkeeping.')
    summary=load(out/'FINAL_RQ4_STATUS.json')
    inputs=load(out/'inputs.json');passed=Path(inputs['canary_stages'][-1]['result']['path']).parent
    verify_canary(passed)
    ref=load(out/'REPORT_REFERENCE.json')
    if sha256(Path(ref['path']))!=ref['sha256']:raise ValueError('Report changed before sealing.')
    save(out/'FINALIZATION_RECOVERY.json',{'reason':'RunStore already creates command.txt; retain its original command.',
        'source':source(Path(__file__)),'GPU_tasks':0,'COCO_recalculations':0,'CPU_tests_repeated':0},out)
    complete_bookkeeping(out,summary,passed,None)
    print(out,flush=True);print(ref['path'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--failed',nargs='+');parser.add_argument('--passed')
    parser.add_argument('--complete-bookkeeping')
    args=parser.parse_args()
    if args.complete_bookkeeping:recover_bookkeeping(args.complete_bookkeeping)
    elif args.failed and args.passed:finalize(args.failed,args.passed)
    else:parser.error('Provide failed/passed RunIDs or one unsealed acceptance RunID.')
