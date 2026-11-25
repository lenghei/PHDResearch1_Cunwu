"""Compare existing initialization bytes with saved official release metadata."""
import argparse
from pathlib import Path
import shutil
import time
from protocol import ROOT,HERE,EXPERIMENT,load,save,source,sha256,write_atomic
from common import seal,verify_hashes
from main.pipeline import new_run
from post80_replay import run_path

RELEASE='https://github.com/ultralytics/assets/releases/tag/v8.4.0'


def admit(metadata,prepared,review):
    verify_hashes(prepared,('TRAINING_ADMISSION.json','CANDIDATE_INITIALIZATION_MANIFEST.json'))
    verify_hashes(review,('INITIALIZATION_REPRODUCIBILITY.json','PREPARATION_OUTPUT_SHA256.json','SCIENTIFIC_PARAMETER_PACKET.json'))
    if load(prepared/'TRAINING_ADMISSION.json')['failed_CPU_branches']:raise ValueError('CPU preparation is incomplete.')
    for relative,digest in load(review/'PREPARATION_OUTPUT_SHA256.json').items():
        if sha256(prepared/relative)!=digest:raise ValueError('Nested CPU evidence changed: '+relative)
    payload=load(metadata)
    if payload.get('html_url')!=RELEASE or payload.get('tag_name')!='v8.4.0':raise ValueError('Unexpected release metadata.')
    entries=[]
    for name in ('yolov8s.pt','yolo26s.pt'):
        assets=[r for r in payload['assets'] if r['name']==name]
        if len(assets)!=1:raise ValueError('Missing or duplicate release asset.')
        asset=assets[0];local=ROOT/'models/checkpoints'/name;digest=sha256(local)
        expected='https://github.com/ultralytics/assets/releases/download/v8.4.0/'+name
        if asset['browser_download_url']!=expected or asset.get('digest')!='sha256:'+digest or asset['size']!=local.stat().st_size:
            raise ValueError('The current initialization does not match the official release digest.')
        entries.append({'name':name,'local':source(local),'official_asset_id':asset['id'],'official_digest':asset['digest'],
            'official_size':asset['size'],'official_download_URL':expected,'official_updated_at':asset['updated_at'],'status':'OFFICIAL_RELEASE_DIGEST_MATCH'})
    digests={row['local']['sha256'] for row in entries}
    candidates=load(prepared/'CANDIDATE_INITIALIZATION_MANIFEST.json')
    if len(candidates)!=6 or any(row['source']['sha256'] not in digests for row in candidates):raise ValueError('A candidate initialization uses a different source.')
    out,_=new_run({'output_root':str(prepared.parent)},EXPERIMENT,'rq4_source_admission_cpu');began=time.perf_counter()
    shutil.copyfile(metadata,out/'official_release_metadata.json');shutil.copyfile(Path(__file__),out/'executed_source.py')
    save(out/'inputs.json',{'preparation':source(prepared/'SHA256SUMS.txt'),'review':source(review/'SHA256SUMS.txt'),
        'release_metadata':source(metadata),'source':source(Path(__file__))},out)
    save(out/'INITIALIZATION_SOURCE_GATE.json',{'status':'PRESENT_SOURCE_BYTES_VERIFIED','release_URL':RELEASE,
        'API_URL':'https://api.github.com/repos/ultralytics/assets/releases/tags/v8.4.0','metadata':source(out/'official_release_metadata.json'),
        'checkpoints':entries,'candidate_native10_files':6,'weight_downloads':0,'source_weight_changes':0,
        'historical_initialization_used_these_bytes':'UNKNOWN; digest identity today is not a creation-time training record.'},out)
    gate=load(prepared/'TRAINING_ADMISSION.json');plan=load(prepared/'OPTIMIZER_EXPOSURE_PLAN.json')
    final={'status':'RQ4_CPU_PREPARED_GPU_NOT_ADMITTED','train_images':6471,'validation_images':548,
        'positive_training_targets':343200,'CPU_dataset_checks':9,'repeats_per_check':2,
        'native10_candidate_initializations':6,'official_source_digests_verified':2,'training_runs':0,'GPU_tasks':0,
        'scientific_training_results':'NOT_EVALUATED','real_GPU_training_cost':'NOT_MEASURED',
        'historical_best_epoch':'UNKNOWN','historical_initialization_link':'UNKNOWN','independent_testdev':'DATASET_BLOCKED',
        'remaining':gate['blockers'],'actual_full_CPU_preparation_seconds':gate['CPU_seconds'],
        'minimal_training_matrix_runs':17,'full_three_seed_ablation_matrix_runs':27,
        'matrix_status':'PROPOSED_NOT_RUN','no_old84_repeated':True}
    save(out/'FINAL_RQ4_STATUS.json',final,out)
    text=['# VM36：RQ4训练准备最终汇总','',
        '## PROBLEM / WHY','',
        '目标是为导师RQ4建立公平的控制/增强训练起点，而不是把已有84组推理结果当作增强训练成功。','',
        '## METHOD','',
        '复用原始FM_P/FM_D及84组开发证据，全程CPU，不改原始图像、标注、权重和历史Run。现有本地COCO源文件只读核对官方资产SHA，不下载或替换模型。',
        '训练候选：YOLOv8s和YOLO26s，1280、100epochs、batch2、nbs64、AdamW lr0=0.01、cos_lr=true、warmup3；seed42/43/44。这些沿用历史配置的数值是候选，不宣称最优或已获正式训练准入。','',
        '## RESULT','',
        '- 全部6471训练图和标注SHA/解码通过，343200个有效native10框；3条零面积行和4个重复框只在派生标签中排除。',
        '- 完整548图内部验证loader通过；1张原ZIP字节私有视图替代本地异常字节输入，原文件未改。内部rect验证与研究square/ignore-union COCO评价分开。',
        '- 九种CPU连接场景各16锚图、两次重复通过；Mosaic开启时每次实际访问64张次，关闭时16张次。控制组与原库像素和目标输出一致，目标增强组与控制组几何目标一致。',
        '- 两模型×三seed共6份native10 FP32候选初始化已保存并重载一致；同模型三seed权重状态不同，seed42重建完全相同。',
        '- 两份现有COCO源文件与官方发布资产SHA和字节数完全一致：','',
        '| 文件 | SHA256 |','|---|---|']
    text.extend('| '+row['name']+' | `'+row['local']['sha256']+'` |' for row in entries)
    text+=['',f'来源：[Ultralytics assets v8.4.0]({RELEASE})。原始API元数据和摘要随本Run保存。当前文件来源已核对，不代表已证明历史ck01/ck05训练使用过同一字节。',
        f'完整数据/初始化/loader CPU准备实测{gate["CPU_seconds"]:.3f}秒。候选100epoch的源码计划为647100张次、323600个batch、{plan["optimizer_and_EMA_updates"]}次更新；这些计数不是训练实测。',
        '真实GPU训练、optimizer/EMA更新、训练AP/AR均为0次/NOT_EVALUATED；GPU训练速度、显存、验证和训练保存成本NOT_MEASURED。','',
        '## CONCLUSION','',
        '已完成可客观验证的CPU准备，原84组没有重跑。完整方案12次三seed主对照＋15次消融，共27次；最小方案保留全部12次主对照，仅先做5次单seed消融，共17次。全部NOT_RUN。',
        '建议优先考虑从已核实的官方COCO源新建匹配控制/增强臂：历史初始化与best轮次未知，且库版本和派生标签与历史存在差别。它是待确认的实验选择，当前没有据此启动训练。',
        '冻结评价模板保留原21种退化上下文、两档推理分辨率和548张val。新权重需要新checkpoint ID；历史84组只作上下文，不能替代新控制臂。','',
        '## LIMITATIONS / 真实训练前尚需完成','',
        '1. 确认五类增强范围、概率及Noise混合概率；范围候选与精确测试值排除清单已有记录，没有默设p=0.5等科学参数。',
        '2. 确认motion方向与haze airlight。旧实现提案0..180°与airlight220也未获批准；CPU fixture的11..169°与210仅用于连接测试。',
        '3. 确认fresh retrain/继续FT、best-on-val/固定末轮、leave-one-out/单增强式消融。原导师文本没有唯一确定所有实现选择。',
        '4. 完成实际trainer的epoch/worker预取、Health Gate及FP32保存接入。当前CPU只测试workers0，历史workers2未准入；现用8.4.61默认half保存，不能冒充FP32训练保存已通过。',
        '5. 记录新的有限GPU训练预算，再单独准备64训练图/16开发图、最多600秒的工程canary。原推理预算不能用于训练；当前prototype.train()拒绝执行。',
        '历史最佳轮次与当时初始化SHA仍UNKNOWN，test-dev仍BLOCKED。当前无新独立测试，不得声称RQ4已经证明gap缩小。','',
        '## 完整证据与复现','',
        '- 数据、loader、参数矩阵：'+str(prepared),'- 初始化重复性及参数决策：'+str(review),'- 官方来源与最终状态：'+str(out),
        '- 中文CPU报告：'+load(prepared/'REPORT_REFERENCE.json')['path'],
        '- 中文参数报告：'+load(review/'REPORT_REFERENCE.json')['path'],
        '- 本步复现（只读本地官方元数据，CPU）：'+str(HERE/'rq4_source_admission.py')+' --metadata <official_release_metadata.json> --prepared-run '+prepared.name+' --review-run '+review.name]
    note=ROOT/'datasets/reportcn'/('vm36_supervisor_reproduction_RQ4最终汇总_'+out.name.split('_')[-1]+'.md')
    write_atomic(note,'\n'.join(text)+'\n',out);save(out/'REPORT_REFERENCE.json',source(note),out)
    write_atomic(out/'diary.md','# RQ4 source admission\n\nCompared local source bytes with saved official release digests and verified candidate initialization provenance. Historical training use remains unknown. No training or new inference.\n',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'RQ4_SOURCE_ADMISSION',**final,
        'bookkeeping_seconds':time.perf_counter()-began},out,replace=True)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>RQ4 preparation evidence</h1><p>'+final['status']+'</p><p><a href="../'+prepared.name+'/index.html">Data, loader and training matrix</a></p><p><a href="../'+review.name+'/index.html">Initialization and scientific decisions</a></p><p><a href="INITIALIZATION_SOURCE_GATE.json">Official source digest verification</a></p><p><a href="FINAL_RQ4_STATUS.json">Final status</a></p><img style="max-width:100%" src="../'+prepared.name+'/training_class_counts.png">',out)
    seal(out)
    from main.visualization import build_run_index
    build_run_index(out.parent,EXPERIMENT)
    latest=load(out.parent/'latest.json');latest.update(latest_RQ4_results_RunID=out.name,RQ4_training_status=final['status'],
        combined_measured_pairs=84,formal_measured_pairs=0)
    save(out.parent/'latest.json',latest,out,replace=True);print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--metadata',required=True,type=Path)
    parser.add_argument('--prepared-run',required=True);parser.add_argument('--review-run',required=True);args=parser.parse_args()
    admit(args.metadata,run_path(args.prepared_run),run_path(args.review_run))
