"""Paired scientific summaries from completed seed groups only."""
import argparse,csv,json,sys,hashlib
from pathlib import Path
import numpy as np
import run as pilot

READ=pilot.READ;SAVE=pilot.SAVE;CSV=pilot.csv_write

def interval(a):
    x=np.asarray(a,dtype=float);x=x[np.isfinite(x)]
    return np.percentile(x,[2.5,97.5]).tolist() if len(x) else [None,None]

def metric_row(m,size):return next(r for r in m['per_area'] if r['size']==size)

def state(tiny,latency):return pilot.backend.gap_state(tiny,latency)

def finish(out):
    spec=READ(out/'protocol.json');cells={};metrics=[];classes=[];comparison=[];transitions=[];indices=None;accepted_seeds=[]
    for seed in (42,43,44):
        result=out/f'group_{seed}_result.json';supervisor=out/f'group_{seed}/supervisor.json'
        if not result.exists() or not supervisor.exists():continue
        safety=READ(supervisor)
        if safety['returncode'] or safety['stop_reason']:continue
        completed=READ(result);assert completed['status']=='COMPLETE_MATCHED_SEED_GROUP' and len(completed['cells'])==8
        accepted_seeds.append(seed)
        for name in completed['cells']:
            folder=out/f'group_{seed}'/name;a=READ(folder/'acceptance.json')
            for n,k in [('predictions.jsonl.gz','raw_SHA256'),('metrics.json','metrics_SHA256'),('bootstrap.json','bootstrap_SHA256'),('timing.json','timing_SHA256')]:assert pilot.SHA(folder/n)==a[k]
            m=READ(folder/'metrics.json');b=READ(folder/'bootstrap.json');t=READ(folder/'timing.json');d=b['draws'];assert d['samples']==200 and d['seed']==spec['bootstrap_seed']
            if indices is None:
                image_ids=d['image_ids'];assert len(image_ids)==548
                indices=np.random.default_rng(d['seed']).integers(0,548,(200,548))
            assert d['image_ids']==image_ids
            lat=np.array([r['cached_pipeline_ms'] for r in t['per_image']]);lat_draws=lat[indices].mean(axis=1)
            tiny=next(r['ci95'] for r in b['intervals'] if r['kind']=='area' and r['size']=='tiny' and r['metric']=='map50_95')
            status=state(tiny,interval(lat_draws));key=(seed,a['strategy'],a['condition'])
            cells[key]={'metrics':m,'draws':d,'latency':lat,'latency_draws':lat_draws,'state':status,'tiny_CI':tiny,'acceptance':a,'folder':str(folder),
                'FP_draws':np.asarray(m['FP_IoU50_counts_by_image'])[indices].mean(axis=1)}
            r={'seed':seed,'strategy':a['strategy'],'condition':a['condition'],'P_id':a['P_id'],'images':548,'checkpoint_SHA256':a['checkpoint_SHA256'],'GT_SHA256':a['GT_SHA256'],'pixel_manifest_SHA256':a['pixel_manifest_SHA256'],'state':status,'AP50':metric_row(m,'all')['map50'],'AP50_95':metric_row(m,'all')['map50_95'],
                'Tiny_AP50_95':metric_row(m,'tiny')['map50_95'],'Small_AP50_95':metric_row(m,'small')['map50_95'],'Medium_AP50_95':metric_row(m,'medium')['map50_95'],'Large_AP50_95':metric_row(m,'large')['map50_95'],'AR50_95':metric_row(m,'all')['ar50_95'],
                'FP_IoU50_per_image':m['FP_IoU50_per_image'],'cached_pipeline_ms':float(lat.mean()),'cached_latency_CI_low':interval(lat_draws)[0],'cached_latency_CI_high':interval(lat_draws)[1],'Tiny_CI_low':tiny[0],'Tiny_CI_high':tiny[1],'raw_predictions':m['raw_count'],'mapped_predictions':m['mapped_count'],'source_metrics':str(folder/'metrics.json'),'metrics_SHA256':a['metrics_SHA256'],'scope':'DEVELOPMENT_ONLY'}
            metrics.append(r)
            classes.extend({'seed':seed,'strategy':a['strategy'],'condition':a['condition'],**r} for r in m['per_class'])
    if indices is not None:SAVE(out/'paired_bootstrap_indices.json',{'image_ids':image_ids,'seed':spec['bootstrap_seed'],'indices':indices.tolist(),'samples':200,'same_across_cells_and_seeds':True})
    for seed in accepted_seeds:
        for condition in [c['condition'] for c in spec['conditions']]:
            c=cells[seed,'CONTROL',condition];g=cells[seed,'VARIABILITY_GUIDED',condition]
            for size in c['draws']['values']:
                for metric in ('map50','map50_95','ar50_95'):
                    delta=metric_row(g['metrics'],size)[metric]-metric_row(c['metrics'],size)[metric]
                    draws=np.asarray(g['draws']['values'][size][metric],float)-np.asarray(c['draws']['values'][size][metric],float);ci=interval(draws)
                    comparison.append({'seed':seed,'condition':condition,'size':size,'metric':metric,'CONTROL':metric_row(c['metrics'],size)[metric],'GUIDED':metric_row(g['metrics'],size)[metric],'GUIDED_minus_CONTROL':delta,'CI95_low':ci[0],'CI95_high':ci[1],'n_valid':int(np.isfinite(draws).sum()),'unit':'AP or AR raw0to1'})
            for name,ca,ga,point in [('cached_pipeline_ms',c['latency_draws'],g['latency_draws'],float(g['latency'].mean()-c['latency'].mean())),('FP_IoU50_per_image',c['FP_draws'],g['FP_draws'],g['metrics']['FP_IoU50_per_image']-c['metrics']['FP_IoU50_per_image'])]:
                ci=interval(ga-ca);comparison.append({'seed':seed,'condition':condition,'size':'all','metric':name,'CONTROL':float(np.mean(ca)),'GUIDED':float(np.mean(ga)),'GUIDED_minus_CONTROL':point,'CI95_low':ci[0],'CI95_high':ci[1],'n_valid':200,'unit':'ms or false positives/image'})
            transitions.append({'seed':seed,'condition':condition,'P_id':c['acceptance']['P_id'],'CONTROL_state':c['state'],'GUIDED_state':g['state'],'transition':c['state']+' -> '+g['state'],'confirmed_Gap_closed':c['state']=='FAIL' and g['state']=='PASS','AP_improvement_alone_is_not_closure':True,'CONTROL_Tiny_CI':json.dumps(c['tiny_CI']),'GUIDED_Tiny_CI':json.dumps(g['tiny_CI'])})
    CSV(out/'configuration_metrics.csv',metrics);CSV(out/'per_class_metrics.csv',classes);CSV(out/'paired_strategy_comparisons.csv',comparison);CSV(out/'tiny_gap_transitions.csv',transitions)
    # Class contrasts use the same pooled-AP bootstrap draws.
    perclass=[]
    for seed in accepted_seeds:
        for condition in [c['condition'] for c in spec['conditions']]:
            c=cells[seed,'CONTROL',condition];g=cells[seed,'VARIABILITY_GUIDED',condition]
            for row in c['metrics']['per_class']:
                k=row['size']+':'+str(row['category_id']);other=next(r for r in g['metrics']['per_class'] if r['size']==row['size'] and r['category_id']==row['category_id'])
                for metric in ('map50','map50_95','ar50_95'):
                    draws=np.asarray(g['draws']['class_values'][k][metric],float)-np.asarray(c['draws']['class_values'][k][metric],float);ci=interval(draws)
                    perclass.append({'seed':seed,'condition':condition,'size':row['size'],'category_id':row['category_id'],'metric':metric,'CONTROL':row[metric],'GUIDED':other[metric],'difference':other[metric]-row[metric] if other[metric] is not None and row[metric] is not None else None,'CI95_low':ci[0],'CI95_high':ci[1],'n_valid':int(np.isfinite(draws).sum())})
    CSV(out/'paired_per_class_comparisons.csv',perclass)
    consistency=[]
    for condition in [c['condition'] for c in spec['conditions']]:
        for size in ('all','tiny','small'):
            for metric in ('map50','map50_95','ar50_95'):
                rows=[r for r in comparison if r['condition']==condition and r['size']==size and r['metric']==metric]
                if rows:
                    x=[r['GUIDED_minus_CONTROL'] for r in rows];consistency.append({'condition':condition,'size':size,'metric':metric,'complete_seeds':len(rows),'seeds':json.dumps(accepted_seeds),'mean_matched_difference':float(np.mean(x)),'minimum':min(x),'maximum':max(x),'sample_SD':float(np.std(x,ddof=1)) if len(x)>1 else None,'positive_seeds':sum(v>0 for v in x),'image_CI_is_not_seed_significance':True})
    CSV(out/'seed_consistency.csv',consistency)
    did=[]
    for seed in accepted_seeds:
        for size in ('all','tiny','small'):
            by={}
            for strategy in ('CONTROL','VARIABILITY_GUIDED'):
                value=0.;draw=0.
                for condition,sign in [('Motion_Haze',1),('Motion_k21',-1),('Haze_dense',-1),('Normal',1)]:
                    v=cells[seed,strategy,condition];value+=sign*metric_row(v['metrics'],size)['map50_95'];draw=draw+sign*np.asarray(v['draws']['values'][size]['map50_95'],float)
                by[strategy]=(value,draw);ci=interval(draw);did.append({'seed':seed,'size':size,'strategy':strategy,'DID':value,'CI95_low':ci[0],'CI95_high':ci[1]})
            value=by['VARIABILITY_GUIDED'][0]-by['CONTROL'][0];ci=interval(by['VARIABILITY_GUIDED'][1]-by['CONTROL'][1]);did.append({'seed':seed,'size':size,'strategy':'GUIDED_DID_minus_CONTROL_DID','DID':value,'CI95_low':ci[0],'CI95_high':ci[1]})
    CSV(out/'Motion_Haze_DID.csv',did)
    gap=[];controllers=[]
    for seed in accepted_seeds:
        for strategy in ('CONTROL','VARIABILITY_GUIDED'):
            labels=[cells[seed,strategy,c['condition']]['state'] for c in spec['conditions']]
            gap.append({'seed':seed,'strategy':strategy,'P_count':4,'COVERED':labels.count('PASS'),'GAP':labels.count('FAIL'),'UNCERTAIN':labels.count('UNCERTAIN'),'gap_count_lower':labels.count('FAIL'),'gap_count_upper':labels.count('FAIL')+labels.count('UNCERTAIN'),'D_scope':'One savedYOLO26s epoch10 checkpoint atInfer1280'})
        for condition in [c['condition'] for c in spec['conditions']]:
            for method in ('FIXED_CONTROL','FIXED_GUIDED','DEVELOPMENT_EXACT_LOOKUP','SAFE_UNKNOWN_P'):
                choice=None;label='ABSTAIN';scope='No unknown-P rule admitted'
                if method.startswith('FIXED'):
                    choice='CONTROL' if method=='FIXED_CONTROL' else 'VARIABILITY_GUIDED';label=cells[seed,choice,condition]['state'];scope='Fixed candidate baseline; no predicted-PASS claim'
                elif method=='DEVELOPMENT_EXACT_LOOKUP':
                    ok=[s for s in ('CONTROL','VARIABILITY_GUIDED') if cells[seed,s,condition]['state']=='PASS']
                    if ok:choice=min(ok,key=lambda s:float(cells[seed,s,condition]['latency'].mean()));label='PASS'
                    scope='RETROSPECTIVE_IN_SAMPLE_LOOKUP; current verified labels used to build table, not deployment input or heldout validation'
                controllers.append({'seed':seed,'condition':condition,'method':method,'selected_strategy':choice,'verified_state':label,'abstain':choice is None,'scope':scope,'independent_false_PASS_validation':'NOT_EVALUATED','unknown_P_deployment_authorized':False})
    CSV(out/'Gap_counts.csv',gap);CSV(out/'controller_comparison.csv',controllers)
    if cells:plots(out,accepted_seeds,cells,comparison,transitions)
    preliminary={'completed_seeds':accepted_seeds,'complete_cells':len(cells),'measured_image_records':len(cells)*548,'programmed_warmup_image_records':len(accepted_seeds)*6,'scientific_scope':spec['limitations'],'partial_group_not_in_primary_statistics':True,'formal_RQ4':'NOT_EVALUATED','training_arms_run':0,'safe_controller':'ABSTAIN_ON_UNKNOWN_P','new_training_authorized':False}
    SAVE(out/'analysis_acceptance.json',preliminary)
    reports(out,accepted_seeds,cells,comparison,gap,consistency)

def plots(out,seeds,cells,comparison,transitions):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'plots';folder.mkdir();conditions=['Normal','Motion_k21','Haze_dense','Motion_Haze'];colors=['#31688e','#d98b34']
    def save(fig,name):
        fig.tight_layout();fig.savefig(folder/(name+'.png'),dpi=180);fig.savefig(folder/(name+'.pdf'));plt.close(fig)
    fig,axes=plt.subplots(1,len(seeds),figsize=(5*len(seeds),4),squeeze=False)
    for ax,seed in zip(axes.flat,seeds):
        x=np.arange(4)
        for j,strategy in enumerate(['CONTROL','VARIABILITY_GUIDED']):ax.bar(x+(j-.5)*.36,[metric_row(cells[seed,strategy,c]['metrics'],'all')['map50_95']*100 for c in conditions],.36,label=strategy,color=colors[j])
        ax.set(xticks=x,xticklabels=['Normal','Motion','Haze','Composite'],ylabel='AP50:95 (%)',title=f'Seed {seed}');ax.legend(fontsize=7)
    save(fig,'condition_AP_comparison')
    fig,ax=plt.subplots(figsize=(6,4))
    for seed in seeds:
        for c in conditions:ax.scatter(metric_row(cells[seed,'CONTROL',c]['metrics'],'all')['map50_95']*100,metric_row(cells[seed,'VARIABILITY_GUIDED',c]['metrics'],'all')['map50_95']*100,label=f'{seed} {c}')
    ax.plot([0,40],[0,40],'--',color='grey');ax.set(xlabel='CONTROL AP (%)',ylabel='GUIDED AP (%)',title='Matched seed/condition comparisons');ax.legend(fontsize=6,ncol=2);save(fig,'seed_strategy_comparison')
    fig,axes=plt.subplots(1,len(seeds),figsize=(5*len(seeds),4),squeeze=False)
    for ax,seed in zip(axes.flat,seeds):
        x=np.arange(4)
        for j,strategy in enumerate(['CONTROL','VARIABILITY_GUIDED']):
            vals=[metric_row(cells[seed,strategy,c]['metrics'],'tiny')['map50_95']*100 for c in conditions];ci=np.array([cells[seed,strategy,c]['tiny_CI'] for c in conditions])*100
            ax.bar(x+(j-.5)*.36,vals,.36,color=colors[j],label=strategy);ax.errorbar(x+(j-.5)*.36,vals,yerr=np.maximum(0,np.array([vals-ci[:,0],ci[:,1]-vals])),fmt='none',ecolor='black',capsize=2)
        ax.axhline(10,color='red',linestyle='--',label='10% threshold');ax.set(xticks=x,xticklabels=['Normal','Motion','Haze','Composite'],ylabel='Tiny AP (%)',title=f'Seed {seed}');ax.legend(fontsize=6)
    save(fig,'Tiny_AP_threshold')
    labels=['PASS','FAIL','UNCERTAIN'];a=np.zeros((3,3),int)
    for r in transitions:a[labels.index(r['CONTROL_state']),labels.index(r['GUIDED_state'])]+=1
    fig,ax=plt.subplots(figsize=(5,4));im=ax.imshow(a,cmap='Blues')
    for i in range(3):
        for j in range(3):ax.text(j,i,str(a[i,j]),ha='center',va='center')
    ax.set(xticks=range(3),yticks=range(3),xticklabels=labels,yticklabels=labels,xlabel='GUIDED state',ylabel='CONTROL state',title='Measured P state transitions');fig.colorbar(im,ax=ax);save(fig,'Gap_transition_matrix')
    fig,ax=plt.subplots(figsize=(5,4));rows=[r for r in comparison if r['condition']=='Normal' and r['size']=='all' and r['metric']=='map50_95'];v=[-r['GUIDED_minus_CONTROL']*100 for r in rows]
    ax.bar([str(r['seed']) for r in rows],v,color=colors[1]);ax.axhline(0,color='grey');ax.set(ylabel='CONTROL - GUIDED AP (percentage points)',xlabel='Training seed',title='Normal accuracy cost');save(fig,'Normal_accuracy_cost')
    fig,axes=plt.subplots(1,len(seeds),figsize=(5*len(seeds),4),squeeze=False)
    for ax,seed in zip(axes.flat,seeds):
        x=np.arange(4)
        for j,strategy in enumerate(['CONTROL','VARIABILITY_GUIDED']):ax.bar(x+(j-.5)*.36,[cells[seed,strategy,c]['latency'].mean() for c in conditions],.36,label=strategy,color=colors[j])
        ax.axhline(100,color='red',linestyle='--');ax.set(xticks=x,xticklabels=['Normal','Motion','Haze','Composite'],ylabel='Full cached pipeline (ms/image)',title=f'Seed {seed}');ax.legend(fontsize=7)
    save(fig,'cached_latency_comparison')

def reports(out,seeds,cells,comparison,gap,consistency):
    def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(str(v) for v in r)+' |' for r in rows]])
    rows=[]
    for r in comparison:
        if r['size'] in ('all','tiny') and r['metric']=='map50_95':rows.append([r['seed'],r['condition'],r['size'],f"{r['CONTROL']*100:.3f}",f"{r['GUIDED']*100:.3f}",f"{r['GUIDED_minus_CONTROL']*100:+.3f}",f"[{r['CI95_low']*100:+.3f},{r['CI95_high']*100:+.3f}]"])
    numeric=table(['seed','P','面积','CONTROL AP%','GUIDED AP%','差值百分点','配对95%区间'],rows)
    tiny=[r for r in comparison if r['size']=='tiny' and r['metric']=='map50_95'];closed=sum(cells[s,'CONTROL',c]['state']=='FAIL' and cells[s,'VARIABILITY_GUIDED',c]['state']=='PASS' for s in seeds for c in ['Normal','Motion_k21','Haze_dense','Motion_Haze'])
    budget=pilot.accepted.budget();scope='DEVELOPMENT_ONLY / POST_HOC_POLICY_SELECTION；同开发图历史暴露；不是独立外部验证。'
    main=f'''# VM48 笔记本跨退化定向增强实验

RunID：`{out.name}`。已完成seed：{seeds}，完整格{len(cells)}，正式评价张次{len(cells)*548}；不完整组不进入配对统计。只读复用原960/10轮fixed epoch10 EMA，没有新训练，没有改正式100轮包。

{numeric}

数值来自548图池化COCO AP，200次同图索引重采样；不平均单图AP。不同seed分别配对；只有最多3个训练seed，图像区间不代表训练seed显著性。所有置信界沿用VM45，AP提升本身不构成Gap关闭。Tiny阈值.10与缓存管线100ms不变。已有FAIL→PASS共{closed}项，详见tiny_gap_transitions.csv；UNCERTAIN单列。

{scope}

像素全部来自VM45的2192个封存PNG，完整文件/解码像素SHA及GT身份核对后复用，未复制大缓存。Motion21/beta1.5来自原UVL；角度Uniform[0,180)与airlight220是作者允许本轮复用的工程近似，不是导师签字值或天气/飞行标定。FP/image是在原推理conf.001、IoU.50、ignore-union之后的非忽略未匹配框；不是部署conf阈值下性能。

所有P关联原FM_P ID。D共享YOLO26/Small/VisDrone_FT/Infer1280/FullFrame/NMS/FP32及GUIDED的NoiseAug身份；Train960不在原FM_D Train640/1280域内，明确标为探索性扩展，不伪装成合法Train1280完整配置。Gaussian概率.25、sigma5.5–30及RNG/排除语义均沿用VM38，不根据新结果改策略。

Normal代价定义为CONTROL−GUIDED AP；所有正/负/跨0结果保留。Motion_Haze_DID.csv另存四角点DID与两策略DID差。配置、逐类AP、FP、时延及完整配对结果都在CSV；原始预测/GT引用、Bootstrap及每图时延保存在每格目录。

控制器只比较固定CONTROL、固定GUIDED、当前开发证据精确查表和未知P安全弃权。查表是当前结果建立的in-sample证据表，不是未知P预测成功；没有用当前GT当部署输入。未知P继续ABSTAIN，不重新拟合旧84行，也不使用VM43 test-dev。

下一项最小匹配训练提案：先完成所有未测现有seed评价；若仍需新训练，采用独立seed45、YOLO26s、960/10轮、CONTROL对不变Gaussian策略、全6471/548及本轮8格评价。先生成并核验同一健康初始化，再单独冻结和授权。基于旧seed43/44两臂实测约5小时及本轮评价成本，需新预算与余量，不能沿用本次推理余额；8–10小时仅为待核实规划区间，不是保证。训练散热须重新准入，不能用本轮推理温度证明训练稳定。正式1280/100轮RQ4仍NOT_EVALUATED。

分析开始时账本余额{budget['remaining_seconds']:.3f}秒；最终扣费后的权威余额见resource_usage.json及本报告末尾。温度、监督器、磁盘和停止原因均保留；无重试、无预算重置。来源及SHA见source_provenance.json、checkpoint_manifest.csv、pixel_manifest.csv、SHA256SUMS.txt。
'''
    summary=f'''# VM48 导师实验结果简报

完成seed {seeds}；{len(cells)}格×548图＝{len(cells)*548}张次。研究的是已有Gaussian增强能否迁移到Motion/Haze，不是新训练或正式100轮RQ4。

{numeric}

已证实FAIL→PASS {closed}项。其余状态及Normal代价必须与AP一并报告。图像区间只描述固定训练seed；跨seed正负变化见seed_consistency.csv。当前结果仍开发集/事后策略选择，不是独立泛化。未知P控制器ABSTAIN。

图：plots/condition_AP_comparison.png、Tiny_AP_threshold.png、Gap_transition_matrix.png、Normal_accuracy_cost.png及cached_latency_comparison.png。原始数字与来源在本Run；没有根据AP选择后续seed，组边界只使用实际成本。
'''
    gaptext=f'''# VM48 RQ4探索性Gap评价

RunID：`{out.name}`。D范围为各seed的单一CONTROL或GUIDED checkpoint、Infer1280；P仅Normal/Motion/Haze/Composite四个目的条件。

{table(['seed','策略','COVERED','GAP','UNCERTAIN','Gap数上下界'],[[r['seed'],r['strategy'],r['COVERED'],r['GAP'],r['UNCERTAIN'],str([r['gap_count_lower'],r['gap_count_upper']])] for r in gap])}

已证实FAIL→PASS {closed}项。只有固定Tiny置信界和完整缓存时延准则达标才能写Gap关闭；FAIL→UNCERTAIN不是关闭，AP提升也不是关闭。两策略Gap数差及三态上下界按相同P计算，不能解释为全FM_P空间Gap比例。

固定策略与开发证据查表比较见controller_comparison.csv。查表使用当前验证标签建表，因此只支持回顾性证据查询；无未知P留出安全证明，不得报告新的低false-PASS泛化能力。未知P保持ABSTAIN。{scope}原UVL、VM43边界、正式1280/100轮研究不变。
'''
    paths=[]
    for name,text in [('VM48_笔记本跨退化定向增强实验_CN.md',main),('VM48_导师实验结果简报_CN.md',summary),('VM48_RQ4探索性Gap评价_CN.md',gaptext)]:
        p=pilot.ROOT/'datasets/reportcn'/name
        if p.exists():p=p.with_name(p.stem+'_'+out.name.split('_')[-1]+p.suffix)
        with p.open('x',encoding='utf-8') as f:f.write(text)
        paths.append(str(p))
    SAVE(out/'report_paths.json',paths)
    links=''.join('<li><a href="'+Path(p).as_uri()+'">'+Path(p).name+'</a></li>' for p in paths)
    images=''.join('<h2>'+p.stem+'</h2><img style="max-width:1100px;width:100%" src="plots/'+p.name+'">' for p in sorted((out/'plots').glob('*.png'))) if (out/'plots').exists() else ''
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>VM48 laptop transfer pilot</title><h1>VM48 laptop transfer pilot</h1><p>Complete seeds: '+str(seeds)+'. DEVELOPMENT_ONLY. No new training.</p><ul>'+links+'</ul>'+images,encoding='utf-8')

def seal_after_charge(out):
    analysis=READ(out/'analysis_acceptance.json');b=pilot.accepted.budget();charges=[p for p in b['charges'] if Path(p['path']).name.startswith(out.name+'_')]
    used=sum(READ(r['path'])['charged_seconds'] for r in charges)
    groups=[]
    for seed in (42,43,44):
        p=out/f'group_{seed}/supervisor.json'
        if p.exists():groups.append({'seed':seed,**READ(p)})
    SAVE(out/'resource_usage.json',{'total_charged_worker_and_CPU_seconds':used,'charges':charges,'budget_after':b,'group_supervisors':groups,'GPU_training_seconds':0,'GPU_kernel_busy_seconds':'NOT_RECORDED; worker wall is charged','untouched_reserve_minimum_seconds':1200})
    status='COMPLETE_3_SEEDS_DEVELOPMENT_ONLY' if len(analysis['completed_seeds'])==3 else ('PARTIAL_WITH_LIMITATIONS' if analysis['completed_seeds'] else 'BLOCKED_NO_COMPLETED_SEED')
    SAVE(out/'final_acceptance.json',{'status':status,**analysis,'budget_after_final_CPU_analysis':b,'untouched1200_second_reserve_preserved':b['remaining_seconds']>=1200})
    run=READ(out/'run.json');run['status']=status
    (out/'run.json').write_text(json.dumps(run,indent=2),encoding='utf-8')
    footer=f"\n最终计费（监督worker＋CPU准备/分析）：{used:.3f}秒；原账本余额{b['remaining_seconds']:.3f}秒，原截止{b['deadline_UTC']}。GPU忙时未独立记录，不把worker墙钟冒称kernel时间。最终状态：{status}。\n"
    reports=READ(out/'report_paths.json')
    for p in reports:
        with Path(p).open('a',encoding='utf-8') as f:f.write(footer)
    with (out/'SHA256SUMS.txt').open('x',encoding='utf-8') as f:
        for p in sorted([p for p in out.rglob('*') if p.is_file() and 'runtime' not in '/'.join(p.relative_to(out).parts)]+[Path(p) for p in reports]):
            if p.name!='SHA256SUMS.txt':f.write(pilot.SHA(p)+'  '+str(p)+'\n')
    return status

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();finish(a.run.resolve(strict=True))
