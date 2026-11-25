"""Finalize the supervisor evidence record after CPU tests and visual review."""
import argparse
from datetime import datetime,timezone
import html
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from common import ROOT,HERE,PUBLIC,FMP,FMD,load,source,sha256,resolve_run,verify_hashes,seal
from supervisor_compliance import BASE,DEV,VERSION,matrix,save,write_atomic
from supervisor_finish import remaining
from historical_review import table,csv_file


def validate_visual_review(out):
    evidence=out/'visual_review_input.json'
    if not evidence.is_file():raise ValueError('Record the actual PDF inspection in visual_review_input.json first.')
    review=load(evidence);pdf=out/'figures/supervisor_figures.pdf'
    if review.get('status')!='PASS' or not review.get('reviewer') or review.get('PDF_sha256')!=sha256(pdf):
        raise ValueError('Missing or stale visual-review evidence.')
    expected={f'page-{i:02}.png' for i in range(1,13)}
    if set(review.get('rendered_pages',{}))!=expected:raise ValueError('All twelve rendered pages are required.')
    for name,value in review['rendered_pages'].items():
        if sha256(out/'pdf_qa_final'/name)!=value:raise ValueError('Rendered-page hash differs.')
    return {'status':'PASS','pages':12,'recorded_visual_review':review,'review_evidence':source(evidence),
            'PDF':source(pdf),'supervisor_signoff':'UNRECORDED'}


def final_audit(out):
    if (out/'SHA256SUMS.txt').exists():raise ValueError('Acceptance is already sealed.')
    if remaining(out)<=0:raise TimeoutError('Original audit wall budget exceeded.')
    visual=validate_visual_review(out)
    beginning=time.perf_counter()
    original_sources=load(out/'evidence_sources.json');checks=[]
    for key,ref in original_sources.items():
        actual=sha256(Path(ref['path']))
        if actual!=ref['sha256']:raise ValueError('Historical source changed: '+key)
        checks.append({'role':key,'artifact':ref['name'],'sha256':actual,'status':'UNCHANGED'})
    csv_file(out/'preserved_inputs.csv',checks)
    request=load(out/'profile_request.json')
    for c in request['checkpoints']:
        if sha256(Path(c['path']))!=c['sha256']:raise ValueError('Checkpoint changed.')
    # Separate the greedy core from the larger frozen cohort.
    plan=load(ROOT/'datasets/local_runs/cv261207/cv261207_audit_20261104T193649_d1b63492/selection_manifest.json')
    from sampling import models
    fmp,_,prior=models();cohort=plan['pairwise']['cohort'];core=plan['pairwise']['greedy_cover_size']
    selections=[tuple(r['selected_leaves']) for r in cohort]
    if len(set(selections))!=len(cohort) or any(not fmp.valid_leaves(s) for s in selections):raise ValueError('Pairwise membership error.')
    masks,_=prior.compute_binary_pair_masks(selections,sorted(fmp.ALL_LEAVES))
    combined=0;greedy=0
    for i,mask in enumerate(masks):
        combined|=mask
        if i<core:greedy|=mask
    if combined.bit_count()!=plan['pairwise']['feasible'] or combined!=greedy:raise ValueError('Pairwise coverage differs.')
    rec=load(out/'sampling_reconciliation.json')
    rec.update(pairwise_greedy_core_configurations=core,pairwise_padding_configurations=len(cohort)-core,
        pairwise_actual_covered_states=combined.bit_count(),pairwise_greedy_covered_states=greedy.bit_count(),
        approximate_60_100='The existing frozen cohort has 80 configurations; its greedy core has 21. Neither is changed.',
        correction='Stage 1/2 prose confused the 21-point core with the full 80-point cohort. The source manifest and plotted 80-point count are unchanged.')
    save(out/'sampling_reconciliation_v2.json',rec,out)
    profiles=load(out/'operation_profiles.json')['profiles'];lookup={(r['checkpoint'],r['resolution']):r for r in profiles}
    timing=table(DEV/'timing.csv');attributes=[]
    for row in timing:
        pair=resolve_run(row['RunID']);verify_hashes(pair,('timing.json','environment.json','metrics.json','bootstrap.json'))
        environment=load(pair/'environment.json');p=lookup[row['checkpoint'],int(row['resolution'])]
        attributes.append({**row,'checkpoint_parameters':p['checkpoint_parameters'],'GFLOPs_2xMAC':p['GFLOPs_2xMAC'],
            'FLOPs_method':p['method'],'FLOPs_limitations':p['limitations'],'hardware':environment['gpu'],
            'Torch':environment['torch'],'Ultralytics':p['ultralytics'],'precision':environment['input_precision'],
            'operation_profile_sha256':sha256(out/'operation_profiles.json'),'timing_sha256':sha256(pair/'timing.json')})
    csv_file(out/'tested_detector_attributes.csv',attributes)
    counts={field:sum(float(r[field]) for r in timing) for field in ('pair_wall_seconds','cpu_evaluation_seconds',
         'cpu_replay_seconds','bootstrap_seconds','GPU_forward_profiler_seconds','synchronized_prediction_seconds')}
    save(out/'measured_resource_totals.json',{'source':source(DEV/'timing.csv'),'totals':counts,
         'scope':'12 unique previously accepted development pairs; excludes repeat timing lost in the interruption.',
         'not_additive':'Nested forward, prediction, evaluation and bootstrap phases must not be added to worker wall time.',
         'new_GPU_seconds':0,'new_inference_pairs':0},out)
    modules=['test_cv261207','test_development_canary','test_development_recovery','test_development_v1',
             'test_historical_review','test_rq_evidence','test_supervisor_compliance']
    command=[sys.executable,'-B','-m','unittest',*[f'tests.{m}' for m in modules],'-v']
    # The tests directory is a namespace, so load it explicitly without site-package name collisions.
    script='import sys,unittest;sys.path.insert(0,sys.argv[1]);s=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in sys.argv[2:]);r=unittest.TextTestRunner(verbosity=2).run(s);raise SystemExit(not r.wasSuccessful())'
    command=[sys.executable,'-B','-c',script,str(HERE/'tests'),*modules]
    with (out/'tests.log').open('x',encoding='utf-8') as stream:
        result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,timeout=min(120,remaining(out)-10))
    if result.returncode:raise ValueError('CPU regression failed; see tests.log.')
    text=(out/'tests.log').read_text(encoding='utf-8');import re
    test_count=int(re.search(r'Ran (\d+) tests?',text)[1])
    save(out/'test_results.json',{'status':'PASS','tests':test_count,'command':command,'GPU_seconds':0,
        'scope':'Existing six CV261207 protocol suites and the new supervisor-audit suite; dashboard source was not changed.'},out)
    commands=[]
    for name in ('supervisor_compliance.py','supervisor_finish.py','supervisor_figures.py','supervisor_replay.py'):
        c=[sys.executable,'-B',str(HERE/name),'--help']
        r=subprocess.run(c,capture_output=True,text=True,timeout=20)
        if r.returncode:raise ValueError('CLI import failed: '+name)
        commands.append({'command':c,'status':'PASS'})
    save(out/'CLI_checks.json',commands,out)
    sources=load(out/'context.json')['source_evidence']
    for key,file in [('parameters','operation_profiles.json'),('generator','demo_manifest.json'),
        ('sampling','sampling_reconciliation_v2.json'),('compatibility','detector_compatibility.csv'),
        ('static','leaf_checkpoint_coverage.csv'),('catalog','checkpoint_readiness.csv'),('release','replay_commands.csv')]:
        sources[key]={'RunID':out.name,'artifact':file,'sha256':sha256(out/file)}
    c=load(out/'continuation_result.json')
    report=matrix(out,load(out/'pdf_pages.json')['pages'],sources,4,c['operation_profiles'],c['CPU_demo'])
    save(out/'visual_QA.json',visual,out)
    save(out/'environment.json',{'python':sys.version,'executable':sys.executable,
       'packages':{p:importlib.metadata.version(p) for p in ('numpy','opencv-python','torch','ultralytics','matplotlib','ultralytics-thop')},
       'GPU_used':False},out)
    commands=[]
    for r in table(DEV/'configuration_metrics.csv'):
        if r['size']!='all':continue
        commands.append({'result':r['pair_id'],'RunID':r['RunID'],'metric_source_sha256':r['metrics_sha256'],
             'command':f"python -B main/experiments/experiments_test/cv261207/supervisor_replay.py --pair {r['RunID']} --metrics",
             'scope':'CPU COCO regeneration from saved predictions and GT; original paired bootstrap draws stay unchanged.'})
    csv_file(out/'pair_replay_commands.csv',commands)
    save(out/'source_versions.json',{p.name:source(p) for p in [HERE/'supervisor_compliance.py',HERE/'supervisor_requirements.py',
         HERE/'supervisor_finish.py',HERE/'supervisor_profile.py',HERE/'supervisor_figures.py',HERE/'supervisor_replay.py',Path(__file__)]},out)
    report_cn(out,report,profiles,counts,test_count)
    command_text='\n'.join(' '.join(r['command']) for r in commands if isinstance(r.get('command'),list))
    write_atomic(out/'command.txt','Final acceptance: python -B main/experiments/experiments_test/cv261207/supervisor_accept.py --run '+out.name+'\nSee execution_ticket.json, continuation_source.json, CLI_checks.json, tests.log and pair_replay_commands.csv.\n',out)
    save(out/'acceptance.json',{'status':'PARTIAL','engineering_audit':'PASS','tests':test_count,'new_GPU_seconds':0,
        'new_training_runs':0,'formal_pairs':0,'development_pairs_reused':12,'source_preservation':'CHECKED_REFERENCED_INPUTS_ONLY',
        'whole_145_GiB_archive_rehashed':False,'publication_policy':'REVIEW_REQUIRED','supervisor_signoff':'UNRECORDED',
        'CPU_acceptance_wall_seconds':time.perf_counter()-beginning,'remaining_budget_seconds':remaining(out)},out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':'cv261207','mode':'SUPERVISOR_COMPLIANCE','status':'PARTIAL',
         'engineering_audit':'PASS','scope':'CPU_ONLY','formal_pairs':0,'development_pairs_reused':12,'GPU_seconds':0},out,replace=True)
    links=f'<h1>Supervisor plan: evidence audit</h1><p>PARTIAL · Formal test-dev pairs: 0 · New GPU work: 0</p><ul><li><a href="compliance_stage_04.md">Requirements matrix</a></li><li><a href="figures/index.html">Twelve figures and source tables</a></li><li><a href="figures/supervisor_figures.pdf">PDF figure book</a></li><li><a href="tested_detector_attributes.csv">Measured attributes</a></li><li><a href="pair_replay_commands.csv">Raw metric replay commands</a></li></ul>'
    d=load(out/'demo_manifest.json')
    for row in d['images']:
        target=Path(row['preview']['path']);relative=os.path.relpath(target,out).replace(os.sep,'/')
        links+=f'<h2>{html.escape(row["source"]["ZIP_member"])}</h2><p>CPU generation only; original left, generated right. No AP evaluation.</p><img src="{html.escape(relative)}" width="100%">'
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><title>Supervisor compliance</title><style>body{font:17px Arial;max-width:1200px;margin:32px auto}</style>'+links,out)
    fixed=out.parent/'rq_index.html';old=fixed.read_text(encoding='utf-8')
    marker='<!-- supervisor-compliance-'+out.name+' -->'
    if marker not in old:write_atomic(fixed,old+marker+f'<h2>2026-11-09 supervisor plan audit</h2><p><a href="{out.name}/index.html">Latest requirements, CPU readiness, source-linked figures and blockers</a></p>',out,replace=True)
    save(out/'nested_evidence_manifest.json',{str(p.relative_to(out)).replace(os.sep,'/'):sha256(p)
         for p in sorted(out.rglob('*')) if p.is_file() and not p.is_relative_to(out/'staging')},out)
    seal(out);print(out,flush=True)


def report_cn(out,report,profiles,cost,tests):
    path=ROOT/'datasets/reportcn/cv261207_导师计划逐项验收_20261109.md'
    lines=['# 导师计划逐项验收（2026-11-09）','',f'RunID：`{out.name}`。总状态：PARTIAL；本轮CPU工程审计通过。',
      '',f'已阅读原计划17页；按原页码和原文建立{len(report["requirements"])}项对照。原始日期未调整，导师现场验收状态均为UNRECORDED。',
      '', '| 研究问题 | 当前状态 | 现有证据与缺口 |','|---|---|---|',
      '| RQ1 双特征模型 | PARTIAL | 两个UVL的SAT、无死特征与计数证据可追溯；物理约束依据及数值属性来源仍需审核。|',
      '| RQ2 退化与检测配置关系 | PARTIAL | 历史279条去重响应；新增开发集12对，3模型×2条件×2分辨率，每对248个图像ID。正式test-dev仍为0/500。|',
      '| RQ3 静态与实测gap | PARTIAL | 三种静态策略和有限子集阈值分析已保存；没有全FM_D经验gap或已验证的失效分类器。|',
      '| RQ4 定向增强缩小gap | NOT_EVALUATED | 一次64张训练曝光、1次optimizer/EMA更新仅为工程canary；没有匹配的定向增强前后实验。|',
      '', '## 本轮实际完成','',
      f'- 原PDF、UVL、11份模型目录权重及已有证据SHA核验；新回归测试与旧协议测试合计{tests}项通过。没有重拷145 GiB资料库。',
      '- 三张原图按一个固定随机合法FM_P配置重复生成，像素和trace完全一致。缓存放在datasets/cache/cv261207下。不运行YOLO、不计算新AP。',
      '- 六个CPU实际输入尺寸运算量profile完成，未从小输入外推。THOP可能遗漏不支持算子；数值不代表GPU吞吐、NMS成本或部署延迟。',
      '- 12图资料包保留旧版本；更新采样说明、50/100 ms下静态对照、RQ4未评价面板；逐页渲染检查。',
      '', '## 查到的差异','',
      '- FM_P：计划中的30个数值全部吻合；实现另有12个扩展属性，未冒称来自原计划。',
      '- FM_D：原文件缺少计划中的10个数值属性。未擅自补写；Nano/Small草稿参数量不能当作真实模型参数。',
      '- 采样：冻结pairwise集是80个配置，包括21个贪心覆盖核心和59个补充点；两者均覆盖1163个可行二阶状态。前两阶段审计文字混淆了核心与全队列，最终表已修正，旧manifest没有改。',
      '- SUP_C1—SUP_C4与RQ1—RQ4分开；导师第四项贡献是复现包，不是定向增强训练。',
      '', '## 实测模型属性','', '| checkpoint | 参数量 | 输入 | GFLOPs（2×MAC）|','|---|---:|---:|---:|']
    for p in profiles:lines.append(f'|{p["checkpoint"]}|{p["checkpoint_parameters"]}|{p["resolution"]}|{p["GFLOPs_2xMAC"]:.6f}|')
    lines+=['','## 已有结果与边界','',
      '- 历史VM26三模型的Resolution×Blur×JPEG效应和原联合区间继续保留；新248图的六个1280−640整体AP差为正，不能等同于任意尺度都受益。详见原resolution_effects.csv。',
      '- 新开发集有248个ID、247种图像字节，其中重复图像的标注不同。保留原始记录；图像bootstrap不能解释为独立航次或训练seed区间。',
      '- 已有回归使用历史279响应拟合、12个新响应评分，仅两个新退化组；不是独立测试或合格PASS/FAIL分类泛化。',
      '- Tiny阈值0.10、100 ms时为4 PASS、6 FAIL、2 UNCERTAIN；50 ms时12对均FAIL。计时是缓存PNG读取、预测/NMS和写出，不是已准入部署延迟。',
      '- 原静态1/3/18覆盖数是不同策略的结果，不是实测失败概率。',
      '', '## 已保存的执行成本','',
      f'- 12对旧开发实验worker墙钟合计{cost["pair_wall_seconds"]:.3f}秒；其中GPU forward计时{cost["GPU_forward_profiler_seconds"]:.3f}秒、CPU评价{cost["cpu_evaluation_seconds"]:.3f}秒、重算{cost["cpu_replay_seconds"]:.3f}秒、bootstrap {cost["bootstrap_seconds"]:.3f}秒。',
      '- 这些是嵌套阶段，不能相加当作总GPU占用。本轮新增GPU时间为0，训练次数为0。重复实验丢失的计时未估造。',
      '', '## 代码与文件关系','',
      '- supervisor_requirements.py：原文、页码和验收条件。',
      '- supervisor_compliance.py：核对原计划、已有输入和参数来源，保存第一、二阶段矩阵。',
      '- supervisor_finish.py：有界CPU示范和profile；沿用原1800秒上限。',
      '- supervisor_profile.py：隔离CPU运算量测量；不启用CUDA。',
      '- supervisor_figures.py：从已保存数值导出图表。',
      '- supervisor_replay.py：默认只核验旧pair哈希；加--metrics才从保存的预测和GT重算COCO。',
      '- supervisor_accept.py：CPU回归、最终矩阵、源证据保护和归档；拒绝覆盖已封存Run。',
      '',f'完整结果：`{out}`',f'固定入口：`{out.parent / "rq_index.html"}`',
      f'图表：`{out / "figures/supervisor_figures.pdf"}`',
      '逐项表：本地 Run 中的 compliance_stage_*.md 与 JSON。',
      '逐对复算命令：本Run的pair_replay_commands.csv；每项均指向真实RunID和指标SHA。',
      '', '## 尚不能推进的分支','',
      '- test-dev与train/val文件不相交可验证，但历史选模、调参与使用记录仍不完整；不把无搜索命中当作未使用证明。',
      '- UAVDT仍缺原生DET/ignore来源和兼容评价协议，未进行真实场景验证。',
      '- 至少三个同协议训练seed、定向增强消融、完整gap估计均未完成；Paper2的15轮seed43/44续训不算主基线重复。',
      '- 需导师审核五个场景约束的依据、FM_D缺失属性及FM_P扩展属性；不改冻结原文件。',
      '- 原计划要求raw结果进Git，当前约定raw仅本地；已列出私有/LFS仓库或不可变外部证据档案的选择，尚未获发布决定。没有上传数据、权重或私密配置。',
      '- 70—100 GPU小时是原计划估计，不是本轮授权。500、900和约4000对是不同计划，均未因本轮核验变成完成。',
      '', '后续先处理上述数据与协议准入；本轮不继续训练，也不撰写论文结论。']
    write_atomic(path,'\n'.join(lines)+'\n',out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    final_audit(resolve_run(args.run))
