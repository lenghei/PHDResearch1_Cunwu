"""Cross-seed summaries from measured, accepted configuration results."""
import csv
from datetime import datetime, timezone
import json
from pathlib import Path


def summarize(parent, exp):
    import numpy as np
    load,save,study=exp.load,exp.save,exp.study
    result=load(parent/'execution_result.json')
    values={};provenance=[]
    sources=[study.run_path(exp.guided.REFERENCE_ID)/'evaluation',exp.previous_guided()/'evaluation']
    sources += [exp.path(i)/'evaluation' for i in result['completed_seed_RunIDs']]
    for folder in sources:
        index,observed=exp.guided.read_values(folder)
        if any(key in values for key in observed):raise ValueError('Duplicate source configuration.')
        values.update(observed)
        provenance += [{'source_RunID':folder.parent.name,'source_folder':str(folder/r['folder']),**r} for r in index]
    metrics=[]
    for (architecture,seed,arm,res,context),value in values.items():
        metrics += [{'architecture':architecture,'seed':seed,'arm':arm,'resolution':res,'condition':context,**r}
                    for r in value['metrics']['per_area']]
    exp.guided.rows_csv(parent/'configuration_metrics.csv',metrics)
    save(parent/'joint_result_index.json',provenance)
    matched={(k[0],k[1],'ALL_FIVE' if k[2]==exp.ARMS[1] else 'CONTROL',k[3],k[4]):v
             for k,v in values.items() if k[2] in exp.ARMS}
    comparisons=exp.old.research.paired_contrasts(matched)
    for row in comparisons:
        if 'ALL_FIVE_minus_CONTROL' in row:row['GUIDED_minus_CONTROL']=row.pop('ALL_FIVE_minus_CONTROL')
    save(parent/'per_seed_comparisons.json',comparisons)
    exp.guided.rows_csv(parent/'per_seed_comparisons.csv',comparisons)
    seeds=sorted({key[1] for key in values if key[2]==exp.ARMS[1]})
    contexts=('Normal','JPEG_Q25','JPEG_Q75','Gaussian_s15')
    stats=[]
    for res in (640,1280):
        for context in contexts:
            for size in ('all','tiny','small','medium','large'):
                selected=[r for r in comparisons if r['resolution']==res and r['context']==context
                          and r['size']==size and r['metric']=='map50_95' and r['status']=='ESTIMATED']
                points=[r['GUIDED_minus_CONTROL'] for r in selected]
                stats.append({'resolution':res,'context':context,'size':size,'n_training_seeds':len(points),
                    'seeds':','.join(str(r['seed']) for r in selected),
                    'mean_paired_difference':float(np.mean(points)) if points else None,
                    'sample_SD':float(np.std(points,ddof=1)) if len(points)>1 else None,
                    'minimum':min(points) if points else None,'maximum':max(points) if points else None,
                    'positive_seed_count':sum(v>0 for v in points),
                    'training_seed_significance':'NOT_TESTED'})
    exp.guided.rows_csv(parent/'cross_seed_variability.csv',stats)
    losses=[]
    for key,value in values.items():
        arch,seed,arm,res,context=key
        normal=values[(arch,seed,arm,res,'Normal')]
        for size in ('all','tiny'):
            point=lambda v:next(r['map50_95'] for r in v['metrics']['per_area'] if r['size']==size)
            draws=np.asarray(normal['draws']['values'][size]['map50_95'])-np.asarray(value['draws']['values'][size]['map50_95'])
            ci=np.percentile(draws,[2.5,97.5])
            losses.append({'seed':seed,'arm':arm,'resolution':res,'context':context,'size':size,
                'degradation_loss':point(normal)-point(value),'CI95_low':float(ci[0]),'CI95_high':float(ci[1])})
    exp.guided.rows_csv(parent/'degradation_losses.csv',losses)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots=parent/'plots';plots.mkdir()
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    colors=('#52667c','#288b8a','#d58d38')
    for column,res in enumerate((640,1280)):
        for row,size in enumerate(('all','tiny')):
            ax=axes[row,column]
            for j,seed in enumerate(seeds):
                selected=[next(x for x in comparisons if x['seed']==seed and x['resolution']==res and
                    x['context']==c and x['size']==size and x['metric']=='map50_95') for c in contexts]
                points=100*np.asarray([x['GUIDED_minus_CONTROL'] for x in selected])
                intervals=100*np.asarray([x['difference_CI95'] for x in selected])
                positions=np.arange(4)+(j-(len(seeds)-1)/2)*.12
                ax.plot(positions,points,marker='o',linewidth=1,color=colors[j],label='seed'+str(seed))
                ax.vlines(positions,intervals[:,0],intervals[:,1],color=colors[j],linewidth=1)
                ax.hlines(intervals[:,0],positions-.025,positions+.025,color=colors[j],linewidth=1)
                ax.hlines(intervals[:,1],positions-.025,positions+.025,color=colors[j],linewidth=1)
            ax.axhline(0,color='#777777',linewidth=.8)
            ax.set_xticks(range(4),('Normal','JPEG25','JPEG75','Gauss15'))
            ax.set_ylabel(('AP95' if size=='all' else 'Tiny AP')+' difference (pp)')
            ax.set_title(f'Inference {res}; training 960')
            ax.spines[['right','top']].set_visible(False);ax.grid(axis='y',alpha=.2)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncols=3,frameon=False)
    fig.suptitle('GUIDED minus CONTROL; pointwise image intervals, not seed intervals')
    figure=plots/'matched_seed_differences.png';fig.savefig(figure,dpi=180);plt.close(fig)
    def comparison(seed,res,context,size='all'):
        return next((r for r in comparisons if r['seed']==seed and r['resolution']==res and
                    r['context']==context and r['size']==size and r['metric']=='map50_95'),None)
    def cell(seed,res,context,size='all'):
        value=comparison(seed,res,context,size)
        if value is None:return 'NOT_RUN'
        if value['status']!='ESTIMATED':return 'NOT_ESTIMABLE'
        lo,hi=value['difference_CI95']
        return f"{100*value['GUIDED_minus_CONTROL']:+.3f} [{100*lo:+.3f},{100*hi:+.3f}]"
    report=study.ROOT/'datasets/reportcn'/('vm39_multi_seed_confirmation_科学比较_'+parent.name[-8:]+'.md')
    lines=['# VM39：冻结Gaussian策略的跨训练seed确认','', '**'+result['status']+'**','',
        f'总RunID：`{parent.name}`。完整配对训练seed：'+','.join(map(str,seeds))+'。',
        '**DEVELOPMENT_ONLY / POST_HOC_POLICY_SELECTION**。seed42复用旧结果；seed43/44未参与策略调参，但仍使用同一548图开发集，不是新图像独立验证。','',
        '## 方法与匹配条件','',
        'YOLO26s、960、6471训练图、548开发图、10轮、FP32、batch2/nbs64、workers0、AdamW lr0=0.01、warmup3及原cosine日程/普通增强。固定epoch10 EMA。close_mosaic=10，因此从epoch0关闭Mosaic。',
        'GUIDED完整复用VM38政策字典：Gaussian概率0.25，sigma均匀5.5–30；精确值排除及逐图/epoch/occurrence/family SHA派生RNG语义不变。没有根据新结果调整参数。',
        '每个seed的两组使用同一份健康native10初始化；不同seed使用已验收的对应seed初始化，因此本研究的seed变化包含初始十类head、数据顺序和增强随机性。',
        '旧resolution adapter的recovery_protocol只作为未改动的960形状检查契约；真实seed绑定在VM39协议、初始化SHA和args.seed中，且在worker训练开始时核对实际初始状态。',
        '训练条件唯一差别是新增Gaussian策略。原1280/100轮正式RQ4和原YOLOv8s五项消融不变。','',
        '## 绝对AP50:95（%）','',
        '|Seed|推理尺寸|策略|Normal|JPEG25|JPEG75|Gaussian15|','|---|---|---|---:|---:|---:|---:|']
    for seed in seeds:
        for res in (640,1280):
            for arm in exp.ARMS:
                points=[next(r['map50_95'] for r in metrics if r['seed']==seed and r['arm']==arm and r['resolution']==res
                             and r['condition']==c and r['size']=='all') for c in contexts]
                lines.append(f'|{seed}|{res}|{arm}|'+'|'.join(f'{100*v:.3f}' for v in points)+'|')
    lines += ['', '## 每seed配对差异及图像区间','',
        '数值为GUIDED−CONTROL（百分点），括号为200次配对图像Bootstrap百分位95%区间。每次重新累计COCO AP，同一548图位置索引，seed20261112。不是平均单图AP。','',
        '|尺寸|条件|seed42|seed43|seed44|','|---|---|---:|---:|---:|']
    for res in (640,1280):
        for c in contexts:lines.append(f'|{res}|{c}|'+'|'.join(cell(s,res,c) for s in (42,43,44))+'|')
    lines += ['', '## Tiny AP配对差异','',
        'Tiny按原图GT框面积<256像素²分组；不随网络分辨率重分组。','',
        '|尺寸|条件|seed42|seed43|seed44|','|---|---|---:|---:|---:|']
    for res in (640,1280):
        for c in contexts:lines.append(f'|{res}|{c}|'+'|'.join(cell(s,res,c,'tiny') for s in (42,43,44))+'|')
    lines += ['', f'![Matched seed differences]({figure.as_posix()})','',
        '## 实际跨seed变动','',
        '下表直接使用各seed的配对点差异。SD为训练seed之间的样本标准差，范围为实测最小/最大值；不是置信区间，也不将200次图像Bootstrap当作200个训练seed。','',
        '|尺寸|条件|n|平均差±seed SD（百分点）|实测范围|正向seed数|','|---|---|---:|---:|---:|---:|']
    for row in stats:
        if row['size']!='all':continue
        sd='NA' if row['sample_SD'] is None else f"{100*row['sample_SD']:.3f}"
        lines.append(f"|{row['resolution']}|{row['context']}|{row['n_training_seeds']}|{100*row['mean_paired_difference']:+.3f} ± {sd}|[{100*row['minimum']:+.3f},{100*row['maximum']:+.3f}]|{row['positive_seed_count']}|")
    lines += ['', 'Normal差值即clean收益/代价。每组绝对Tiny AP、退化损失（Normal−对应退化）及损失变化区间，见configuration_metrics.csv、degradation_losses.csv及per_seed_comparisons.csv。JPEG结果和所有负面点均保留。',
        'ALL_FIVE只有已验收seed42，作为旧参考保留在联合索引/指标表，不与缺失的seed43/44 ALL_FIVE混合比较。','',
        '## 执行、来源与预算','',
        f"新增worker墙钟：{result['new_worker_seconds']:.3f}秒（{result['new_worker_seconds']/3600:.3f}小时），包含worker内CPU评价/Bootstrap，并非纯CUDA时间。",
        f"原36小时总账剩余：{result['budget']['remaining_seconds']/3600:.3f}小时。固定截止仍为2026-11-20 21:20:36.593408 UTC，未新增或延长额度。",
        '90°C立即停止、87°C连续30秒停止及全部loss/梯度/optimizer/model/EMA/BN检查与FP32原子保存/重载保护均沿用。无自动重试或降配置。','',
        '|Seed|RunID|初始化SHA256|评价项数|','|---|---|---|---:|']
    evidence=[]
    for identity in result['attempted_seed_RunIDs']:
        out=exp.path(identity);seed=load(out/'seed_run.json')['seed']
        completed=(out/'seed_acceptance.json').exists()
        initial=load(out/('YOLO26s_seed'+str(seed)+'_CONTROL')/'protocol.json')['initialization']
        lines.append(f"|{seed}|{identity}|`{initial['sha256']}`|{16 if completed else 'PARTIAL/NOT_EVALUATED'}|")
        for name in ('seed_acceptance.json','execution_freeze.json','seed_run.json'):
            if (out/name).exists():evidence.append(exp.guided.source(out/name))
        for arm in exp.ARMS:
            folder=out/('YOLO26s_seed'+str(seed)+'_'+arm)
            epochs=len((folder/'epoch_progress.jsonl').read_text().splitlines()) if (folder/'epoch_progress.jsonl').exists() else 0
            lines += ['', f"seed{seed} {arm}：已完成{epochs}轮；日志与有效checkpoint位于`{folder}`。"]
            if (folder/'training_acceptance.json').exists():
                a=load(folder/'training_acceptance.json');evidence.append(exp.guided.source(folder/'training_acceptance.json'))
                lines.append('固定epoch10 EMA SHA256：`'+a['selected_checkpoint']['sha256']+'`。')
        lines += ['', '逐配置原始预测、GT、指标、Bootstrap、trace：`'+str(out/'evaluation')+'`。']
    if result['error']:lines += ['', '阻塞/停止原因：`'+result['error']+'`。未完成组不当作有效科学配对。']
    positives=[r for r in stats if r['size']=='all' and r['context']=='Gaussian_s15']
    consistent=len(seeds)>1 and all(r['positive_seed_count']==len(seeds) for r in positives)
    lines += ['', '## 结论与限制','',
        ('所完成的'+str(len(seeds))+'个seed在两档推理分辨率的Gaussian整体AP差异方向均为正，支持该冻结策略在当前开发设置下的方向重复性。'
         if consistent else '尚未得到所有已完成seed在两档分辨率均正向的Gaussian改善证据；以实际差异、范围及缺失状态为准，不强行确认重复性。'),
        'clean代价及JPEG迁移需逐表判断，不因Gaussian改善而宣称所有条件全面优越。新seed没有参与政策优化，但政策最初使用seed42同一开发图反馈选择；独立图像准入限制不变。',
        '即使完成3个训练seed，样本仍很少。这里只报告观察到的seed变动，不声称训练seed显著性、独立泛化或完整RQ4完成。图像区间未控制可靠航次聚类和多重比较。',
        '本轮结束后停止，不自动增加seed、策略、训练轮数或推送GitHub。','',
        '## 复现和文件','',
        f'- 冻结Gaussian政策：`{exp.previous_guided()/"policy.json"}`，SHA256 `{study.sha256(exp.previous_guided()/"policy.json")}`。',
        f'- 执行源码：`{exp.SOURCE}`，SHA256 `{study.sha256(exp.SOURCE)}`。',
        f'- 报告源码：`{Path(__file__).resolve()}`，SHA256 `{study.sha256(Path(__file__).resolve())}`。',
        f'- 参数/预算/输入来源：`{parent/"protocol.json"}`、`{parent/"resource_admission.json"}`、`{parent/"execution_freeze.json"}`。',
        f'- 联合引用索引：`{parent/"joint_result_index.json"}`；所有旧结果只读，没有复制旧预测或数据。',
        f'- 完整指标：`{parent/"configuration_metrics.csv"}`；配对区间：`{parent/"per_seed_comparisons.csv"}`。',
        f'- seed变动：`{parent/"cross_seed_variability.csv"}`；退化损失：`{parent/"degradation_losses.csv"}`。',
        '- 实际启动命令（已执行，不应重复）：','', '```powershell',
        f"& '{study.ROOT/'.venv_rq4/Scripts/python.exe'}' -B -X utf8 '{exp.SOURCE}' --execute '{parent.name}'",'```']
    report.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(parent/'final_report.json',exp.guided.source(report))
    save(parent/'report_evidence.json',{'sources':evidence,'raw_results_modified':False,
        'new_evaluation_configurations':16*len(result['completed_seed_RunIDs']),
        'reused_evaluation_configurations':24,'n_complete_training_seeds':len(seeds),
        'report_SHA256':study.sha256(report),'figure_SHA256':study.sha256(figure),
        'scope':exp.SCOPE,'training_seed_significance':'NOT_TESTED'})
    meta=load(parent/'run.json');meta.update(status=result['status'],finished_UTC=datetime.now(timezone.utc).isoformat())
    study.save(parent/'run.json',meta,parent,replace=True)
    (parent/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>VM39 matched seeds</h1><p>'+result['status']+
        '</p><img src="plots/matched_seed_differences.png" style="width:100%;max-width:1200px">'
        '<p><a href="configuration_metrics.csv">AP</a> | <a href="per_seed_comparisons.csv">Paired intervals</a> | '
        '<a href="cross_seed_variability.csv">Seed variability</a> | <a href="execution_result.json">Execution</a></p>',encoding='utf-8')
    study.seal(parent)
    print('VM39_REPORT',report,flush=True)
