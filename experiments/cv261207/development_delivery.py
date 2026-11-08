"""Publish small verified summaries; retain full predictions locally."""
import argparse
from datetime import datetime,timezone
import html
from pathlib import Path
import shutil
from common import ROOT,PUBLIC,HERE,RUNS,load,save,new_run,resolve_run,source,sha256,seal,verify_hashes
from historical_review import table,csv_file
from outputs import publish_snapshot


def deliver(analysis,figures,canary=None):
    checks=verify_hashes(analysis,('configuration_metrics.csv','resolution_effects.csv','validation_boundaries.json',
           'heldout_regression.csv','learning_acceptance.json','sources.json'))
    verify_hashes(figures,('figure_evidence.json','rq1_rq4_figures.pdf','run.json'))
    out=new_run('development_delivery');meta=load(analysis/'run.json');boundary=load(analysis/'validation_boundaries.json')
    rows=table(analysis/'configuration_metrics.csv');gains=table(analysis/'resolution_effects.csv')
    training=load(canary/'result.json') if canary and (canary/'result.json').exists() else {'status':'BLOCKED' if canary else 'NOT_EVALUATED'}
    total=meta['unique_new_pairs'];reg=table(analysis/'heldout_regression.csv')
    rq=[{'RQ':'RQ1','status':'PARTIAL','evidence':'Original UVL logic/counts unchanged; two additional renderable conditions executed.',
         'remaining':'Physical calibration and full deployment executability.'},
        {'RQ':'RQ2','status':'PARTIAL_DEVELOPMENT','evidence':f'{total} new measured pairs on the 248-ID complement; fixed descriptive models evaluated on image/configuration-disjoint development responses.',
         'remaining':'Two purposive degradation groups; OOD features; valid PASS/FAIL classifier and independent confirmation absent.'},
        {'RQ':'RQ3','status':'PARTIAL_SUBSET','evidence':'Measured Tiny AP and cached-input mean-latency intervals classify the new finite subset.',
         'remaining':'No population gap estimator or independently validated learned static constraint.'},
        {'RQ':'RQ4','status':'NOT_EVALUATED','evidence':'Clean-control canary: '+training['status']+'; five augmentation hooks unit-tested.',
         'remaining':'Targeted-arm training and matched gap closure NOT_EVALUATED; long training not started.'}]
    brief={'RunID':out.name,'analysis_RunID':analysis.name,'figures_RunID':figures.name,'scope':'DEVELOPMENT_ONLY',
        'unique_new_pairs':total,'formal_pairs':0,'historical_unique_responses':279,'image_IDs':248,'unique_image_hashes':247,
        'independent_test_gate':'DATASET_BLOCKED','research_questions':rq,
        'pairs':[{k:r[k] for k in ('RunID','condition','checkpoint','resolution','map50','map50_95','mean_pipeline_ms',
                  'latency_ci95_low','latency_ci95_high','metrics_sha256','prediction_sha256','GT_sha256')} for r in rows if r['size']=='all'],
        'resolution_effects':[r for r in gains if r['metric']=='map50_95' and r['size'] in ('all','tiny','small')],
        'heldout_regression':reg,'learning':load(analysis/'learning_acceptance.json'),
        'heldout_relations':table(analysis/'heldout_relation_summary.csv') if (analysis/'heldout_relation_summary.csv').exists() else [],
        'cross_checkpoint_effects':[r for r in table(analysis/'cross_checkpoint_effects.csv') if r['metric']=='map50_95' and
              r['size'] in ('all','tiny','small')] if (analysis/'cross_checkpoint_effects.csv').exists() else [],
        'states':[r for r in table(analysis/'pair_states.csv') if r['size']=='tiny' and float(r['tau'])==.1 and
                    float(r['latency_limit_ms'])==100 and r['policy']=='exact_identity'],
        'diagnostics':[r for r in table(analysis/'static_empirical_diagnostics.csv') if r['size']=='tiny' and
                    float(r['tau'])==.1 and float(r['latency_limit_ms'])==100],
        'canary':{k:training.get(k) for k in ('status','group','network_image_exposures','batches','optimizer_steps','EMA_updates',
                   'worker_seconds','peak_allocated_bytes','peak_reserved_bytes')},
        'gap_closure':'NOT_EVALUATED','repeat':boundary['repeat'],'limitations':boundary['bootstrap_limitations'],
        'timing_scope':boundary['latency'],'rules':(analysis/'rules.txt').read_text(encoding='utf-8'),
        'source_RunIDs':load(analysis/'sources.json')['source_runs'],'source_hashes':checks}
    timing=table(analysis/'timing.csv')
    brief['execution_cost']={'unique_pair_worker_seconds':sum(float(r['pair_wall_seconds']) for r in timing),
          'unique_pair_GPU_forward_profiler_seconds':sum(float(r['GPU_forward_profiler_seconds']) for r in timing),
          'unique_pair_CPU_evaluation_seconds':sum(float(r['cpu_evaluation_seconds']) for r in timing),
          'unique_pair_CPU_replay_seconds':sum(float(r['cpu_replay_seconds']) for r in timing),
          'unique_pair_bootstrap_seconds':sum(float(r['bootstrap_seconds']) for r in timing),
          'peak_allocated_bytes':max(int(r['peak_allocated_bytes']) for r in timing),
          'peak_reserved_bytes':max(int(r['peak_reserved_bytes']) for r in timing),
          'repeated_run_worker_seconds':None,'GPU_occupancy_hours':None,
          'timing_warning':'Worker wall includes CPU work. Repeated-run timing lost. GPU occupancy and energy were not measured.'}
    if canary and training['status']=='CANARY_PASS':
        receipt=load(canary/'author_execution_ticket.json');prep=load(resolve_run(receipt['preparation'])/'protocol.json')
        val=table(canary/'training/results.csv')[-1]
        brief['canary'].update({'RunID':canary.name,'initial_checkpoint_sha256':prep['initial_checkpoint']['sha256'],
              'saved_checkpoint_sha256':training['checkpoint']['sha256'],
              'supervised_worker_wall_seconds':load(canary/'supervisor.json')['total_worker_wall_seconds'],
              'validation_AP50':float(val['metrics/mAP50(B)']),'validation_AP50_95':float(val['metrics/mAP50-95(B)']),
              'validation_protocol':'Ultralytics 8.4.61 built-in engineering validation; not research pr01 ignore-union COCO.',
              'validation_images':16,'matched_before_after':'NOT_EVALUATED'})
    save(out/'summary.json',brief);csv_file(out/'rq_acceptance.csv',rq)
    text=['# Development evidence update','',f'Analysis `{analysis.name}`. New measured pairs: **{total}/12**.',
          'Formal independent pairs remain **0/500 — DATASET_BLOCKED**.','',
          'The 248 validation IDs are outside the historical 300 fitting IDs and byte hashes. They include 247 unique image byte hashes.',
          'Validation was historically available for checkpoint selection. These are development results, not independent test results.','',
          'Together, the historical 300 and new 248 IDs cover all 548 validation images. None is an untouched future test image; aggregate AP is not pooled across the different study protocols.',
          'AP denotes AP50:95 over IoU 0.50:0.05:0.95 and 101 recall points. AP50 is reported separately. AR averages recall across the same IoUs at the custom 3,000 detection cap.',
          'Original-coordinate area bins are tiny <256; small 256–<1024; medium 1024–<9216; large >=9216 square pixels. The custom bins are disjoint. Missing class metrics remain null in source JSON and blank in CSV.',
          'The native10 ignore-union protocol and dense detection cap differ from official VisDrone leaderboard evaluation.','',
          '## Measured AP and cost','', '| Checkpoint | Condition | Input | AP50 | AP50:95 | Mean pipeline ms |',
          '|---|---|---:|---:|---:|---:|']
    for r in brief['pairs']:text.append(f"| {r['checkpoint']} | {r['condition']} | {r['resolution']} | {100*float(r['map50']):.3f}% | {100*float(r['map50_95']):.3f}% | {float(r['mean_pipeline_ms']):.3f} |")
    text+=['','The timed pipeline reads, hashes and decodes a cached PNG, predicts/NMS, and writes compressed raw results. Generation, cold initialization, evaluation and bootstrap are separately recorded.',
          'The repeated run reproduced predictions and AP exactly. Its damaged timing was not reconstructed.','',
          'Measured unique-pair worker wall: '+f"{brief['execution_cost']['unique_pair_worker_seconds']:.3f} s; "+
          'GPU forward profiler: '+f"{brief['execution_cost']['unique_pair_GPU_forward_profiler_seconds']:.3f} s. "+
          'Worker wall is not GPU occupancy. Repeated-run timing and energy are unavailable.','',
          '## Same-checkpoint resolution contrasts','', '| Checkpoint | Condition | Size | AP50:95 difference, pp | Pointwise 95% interval, pp |',
          '|---|---|---|---:|---|']
    for r in brief['resolution_effects']:text.append(f"| {r['checkpoint']} | {r['condition']} | {r['size']} | {100*float(r['estimate']):+.3f} | [{100*float(r['ci95_low']):+.3f}, {100*float(r['ci95_high']):+.3f}] |")
    text+=['','Two conditions do not identify an isolated Blur×JPEG interaction. The historical VM26 interaction estimates remain separate.',
           'The 200 paired image draws do not cover flight clustering, repeated scenes or detector training seeds. Intervals are pointwise.','',
           '## Learned responses','', 'The fixed tree, rule-stump and forest definitions were fitted on the historical 279 responses only.',
           'New configurations and image IDs were held out. The two test conditions are out of distribution; this is limited development regression.',
           'Historical joint AP/latency labels are incomplete, so balanced accuracy, macro F1 and failure precision/recall remain unreported.',
           '', '## RQ4','', 'Clean-control canary: **'+training['status']+'**. Targeted augmentation efficacy and gap closure: **NOT_EVALUATED**.',
           'A short canary cannot establish a reliable full-training budget or method efficacy.','',
           '## Remaining supervisor decisions','',
           '- Resolve checkpoint-specific historical test-dev use, or admit a genuinely new native10 image cohort.',
           '- Decide whether the two-group OOD diagnostic supports expanding a development-only study.',
           '- Approve a measured training budget and matched efficacy protocol separately from engineering readiness.','',
           'Source tables, raw prediction/GT checksums and bootstrap draws are indexed in the analysis run.']
    at=text.index('## RQ4')
    learned=['| Method | Fitting seed | AP MAE | AP RMSE | Resolution-effect MAE | Direction correct / total |',
             '|---|---:|---:|---:|---:|---:|']
    for r in reg:
        relation=next((x for x in brief['heldout_relations'] if x['method']==r['method'] and x['seed']==r['seed']),None)
        learned.append(f"| {r['method']} | {r['seed']} | {float(r['MAE']):.6f} | {float(r['RMSE']):.6f} | "+
             (f"{float(relation['effect_MAE']):.6f} | {relation['direction_correct']}/{relation['direction_denominator']} |" if relation else '— | — |'))
    learned+=['','Errors use AP raw units (0–1). Seeds are fitting seeds, not independent detector training seeds.',
              'All six measured resolution effects are positive. Direction accuracy cannot establish false-relation control without negative cases.','']
    text[at:at]=learned
    text+=['','## Data and uncertainty limits','',*['- '+item for item in boundary['bootstrap_limitations']]]
    if training['status']=='CANARY_PASS':
        text+=['',f"Canary: {training['network_image_exposures']} network exposures, {training['batches']} batches, "+
               f"{training['optimizer_steps']} optimizer and {training['EMA_updates']} EMA update; FP32 save/reload passed.",
               f"Supervised worker wall: {brief['canary']['supervised_worker_wall_seconds']:.3f} s. "+
               f"Torch peak allocated/reserved: {training['peak_allocated_bytes']}/{training['peak_reserved_bytes']} bytes.",
               f"The 16-image built-in engineering validator reported AP50={brief['canary']['validation_AP50']:.5f}, "+
               f"AP50:95={brief['canary']['validation_AP50_95']:.5f}. These are not pr01 research metrics or before/after efficacy estimates."]
    (out/'report.md').write_text('\n'.join(text),encoding='utf-8')
    save(out/'inputs.json',{'analysis':source(analysis/'SHA256SUMS.txt'),'figures':source(figures/'SHA256SUMS.txt'),
                            'canary':source(canary/'supervisor_evidence.json') if canary else None})
    save(out/'run.json',{'RunID':out.name,'ExperimentID':'cv261207','status':'PARTIAL','mode':'DEVELOPMENT_DELIVERY',
           'source_analysis':analysis.name,'scope':'DEVELOPMENT_ONLY','unique_new_pairs':total})
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>CV261207 latest evidence</title><style>body{font:18px Arial;max-width:1050px;margin:40px auto}li{margin:16px}</style><h1>CV261207 development update</h1><p>'+str(total)+' new measured pairs; independent test-dev remains blocked.</p><ul><li><a href="report.md">Measured results and limitations</a></li><li><a href="../'+figures.name+'/index.html">Twelve figures</a></li><li><a href="../'+analysis.name+'/configuration_metrics.csv">All metrics and source hashes</a></li><li><a href="../'+analysis.name+'/heldout_regression.csv">Held-out development regression</a></li><li><a href="../'+analysis.name+'/static_empirical_diagnostics.csv">Static versus measured subset</a></li></ul>',encoding='utf-8')
    seal(out)
    public=publish_snapshot(out.name,'PARTIAL','DEVELOPMENT_ONLY',{'development_study':brief})
    private_tokens=('E:\\','C:\\','local_runs\\','checkpoint_path')
    if any(token in (public/'snapshot.json').read_text(encoding='utf-8') for token in private_tokens):raise ValueError('Private path in public summary.')
    finalize_links(out,analysis,figures,public)


def finalize_links(out,analysis,figures,public):
    verify_hashes(out,('summary.json','report.md','run.json'))
    brief=load(out/'summary.json');total=brief['unique_new_pairs']
    if brief['analysis_RunID']!=analysis.name or brief['figures_RunID']!=figures.name:raise ValueError('Delivery source differs.')
    if sha256(public/'snapshot.json')!=load(public/'manifest.json')['SHA256']:raise ValueError('Public snapshot changed.')
    for name in ('RQ_EVIDENCE.md','RQ_ACCEPTANCE_DETAILS.md'):
        path=HERE/name;old=path.read_text(encoding='utf-8')
        marker='<!-- development-update-20261108 -->'
        if marker not in old:
            notice=marker+f'\nLatest update: [new development measurements](DEVELOPMENT_EVIDENCE.md), `{out.name}`. The earlier acceptance below is retained as a dated historical state.\n\n'
            path.write_text(notice+old,encoding='utf-8')
    shutil.copyfile(out/'report.md',HERE/'DEVELOPMENT_EVIDENCE.md')
    local=RUNS/'rq_index.html';old_index=local.read_text(encoding='utf-8')
    marker='<!-- latest-development-update -->'
    if marker not in old_index:
        local.write_text(old_index+marker+f'<h2>2026-11-08 development continuation</h2><p><a href="{out.name}/index.html">New measurements, uncertainty and figures</a></p>',encoding='utf-8')
    pointer=load(RUNS/'latest_rq_evidence.json');pointer['development_delivery_RunID']=out.name;pointer['development_analysis_RunID']=analysis.name
    pointer['development_figures_RunID']=figures.name;save(RUNS/'latest_rq_evidence.json',pointer)
    cn=ROOT/'datasets/reportcn/cv261207_关机恢复与新增实验结果_20261108.md'
    cn.write_text('\n'.join(['# CV261207 关机恢复与新增结果','',f'交付运行：`{out.name}`。分析：`{analysis.name}`。',
       '',f'新增实测 **{total}/12 个配对**，每个248图；另有一次完全相同的重复推理，CPU已恢复并复核预测、AP和200次bootstrap。',
       '原重复运行的耗时文件、完成标记和校验清单被强制关机损坏，保留原样；没有伪造耗时，也没有重复GPU推理。',
       '248个图像ID对应247份不同图片字节，存在一对完全相同图片。固定样本没有事后删除。',
       '这对图片的有效框为41/40个、忽略区域为2/0个；保留原始GT，不擅自认定哪份标注正确。区间不能解释为独立场景抽样证据。',
       '历史300图加本次248图，已覆盖全部548个val图像ID。以后不能把它们改称未看过的最终测试集，也不能按图片数加权拼接两批AP。',
       '', '## 结果怎么看','',
       '- AP50：IoU=0.50的平均精度；AP50:95：IoU从0.50到0.95平均，本文AP默认指后者。',
       '- 同权重分辨率差：同一个模型、同一批图片和退化像素，1280减640。尺寸分组依据原图框面积。',
       '- 95%区间：200次配对图片重采样后重新累计COCO AP；不是单图AP平均。未控制真实航次、重复场景和训练seed。',
       '- 实测PASS要求AP区间下界达标且平均流水耗时区间上界达标；不确定值保留UNCERTAIN。',
       '- 两种新增退化是Downscale x1.5和额外JPEG Q25；不能用它们声称识别了单独的Blur×JPEG因果交互。',
       '', '## 代码和结果位置','',
       '- `development_v1.py`：冻结图片、退化缓存、真实预测、原生十类COCO评价和bootstrap。',
       '- `development_recovery.py`：原始现场只读，恢复重复运行的CPU统计，跳过已验收配对。',
       '- `development_analysis.py`：同权重分辨率差、尺寸分层、AP/耗时判定、固定学习模型的开发集保留评价。',
       '- `development_canary_v1.py`：小样本干净继续训练工程检查；五种退化钩子已单测，未做正式方法训练。',
       '- `development_figures.py`、`development_delivery.py`：图表、证据说明、只读仪表盘摘要。',
       '',f'完整图表：`{figures}`。完整数值：`{analysis}`。',
       f'本地固定入口：`{local}`。公开摘要：`{public}`。',
       '', '## 仍未完成','',
       '- 独立test-dev：DATASET_BLOCKED，正式0/500；不是把剩余248张val当独立测试。',
       '- 规则学习：有图片及配置分离的开发回归诊断，但仅两个新退化组；PASS/FAIL分类训练证据仍不足。',
       '- 完整空间经验gap、学习到的UVL约束、五组增强方法效果和gap闭合：未验证。',
       '- 长训练、自动推送GitHub：未执行。',
       '', '详细AP和区间见公开英文说明 `main/experiments/cv261207/DEVELOPMENT_EVIDENCE.md`，每个数值能追溯到本地原始证据。']),encoding='utf-8')
    print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--analysis',required=True);parser.add_argument('--figures',required=True);parser.add_argument('--canary');parser.add_argument('--link-existing')
    args=parser.parse_args()
    if args.link_existing:
        out=resolve_run(args.link_existing);finalize_links(out,resolve_run(args.analysis),resolve_run(args.figures),PUBLIC/'reports/results/cv261207'/out.name)
    else:deliver(resolve_run(args.analysis),resolve_run(args.figures),resolve_run(args.canary) if args.canary else None)
