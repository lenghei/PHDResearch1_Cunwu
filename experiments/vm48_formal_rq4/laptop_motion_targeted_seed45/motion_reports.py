"""Factual local summaries for completed or interrupted seed45 work."""
import csv,html,json,time
import numpy as np
from motion_common import *

def read_csv(path):
    if not Path(path).exists():return []
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def table(rows,keys):
    if not rows:return 'NOT_EVALUATED'
    return '| '+' | '.join(keys)+' |\n| '+' | '.join(['---']*len(keys))+' |\n'+'\n'.join('| '+' | '.join(str(r.get(k,'')) for k in keys)+' |' for r in rows)

def finish(out,failure):
    from motion_evaluation import csv_write
    started=time.perf_counter();ledger=load(out/'budget_ledger.json');charges=ledger['charges'];epochs={a:len(list((out/a).glob('epoch*_complete.json'))) for a in ['CONTROL','MOTION_TARGETED']}
    rows=read_csv(out/'configuration_metrics.csv');comparisons=read_csv(out/'paired_AP_comparisons.csv');transitions=read_csv(out/'tiny_gap_transitions.csv')
    safe=len(charges)==2 and all(c['result']['returncode']==0 and c['result']['stop_reason'] is None for c in charges)
    complete=safe and epochs=={'CONTROL':10,'MOTION_TARGETED':10} and len(rows)==8 and (out/'analysis_acceptance.json').exists()
    old=load(out/'old_budget_READ_ONLY_snapshot.json');old_same=all(sha(r['path'])==r['SHA256'] for r in old['files'])
    status='MATCHED_PILOT_COMPLETED_WITH_LIMITATIONS' if complete and old_same else 'PARTIAL_WITH_LIMITATIONS' if any(epochs.values()) else 'BLOCKED'
    telemetry=[]
    for c in charges:
        p=out/(c['stage']+'_supervision')/'gpu_telemetry.jsonl'
        if p.exists():
            with p.open(encoding='utf-8') as f:telemetry.extend({'stage':c['stage'],**json.loads(s)} for s in f)
    csv_write(out/'GPU_telemetry.csv',telemetry)
    resources={'GPU_worker_supervision_charged_seconds':sum(c['charged_seconds'] for c in charges),'charges':charges,'new_total_seconds':36000,'new_remaining_seconds':remaining(out),
        'expiry_UTC':ledger['expiry_UTC'],'old_ledger_files_unchanged':old_same,'old_ledger_remaining_reference_seconds':old.get('remaining_seconds',4343.0300212),
        'peak_temperature_C':max((r['temperature_C'] for r in telemetry),default=None),'peak_GPU_used_MiB':max((r['used_MiB'] for r in telemetry),default=None),
        'minimum_disk_free_bytes':min((r['disk_free_bytes'] for r in telemetry),default=None),'formal_evaluation_images':len(rows)*548,'formal_result_not_inferred_from_internal_validation':True}
    save(out/'resource_usage.json',resources)
    augment=read_csv(out/'augmentation_event_log.csv');stats=[]
    for arm in ['CONTROL','MOTION_TARGETED']:
        for ep in sorted({int(r['epoch']) for r in augment if r['strategy']==arm}):
            group=[r for r in augment if r['strategy']==arm and int(r['epoch'])==ep];active=[r for r in group if r['active']=='True'];angles=[float(r['angle_deg']) for r in active]
            stats.append({'strategy':arm,'epoch':ep,'logged_image_loads':len(group),'active':len(active),'realized_probability':len(active)/len(group),
                **{'kernel_'+str(k):sum(r['kernel']==str(k) for r in active) for k in [5,9,13,17,25]},'evaluation_kernel21_uses':sum(r['kernel']=='21' for r in active),
                'angle_min':min(angles) if angles else None,'angle_max':max(angles) if angles else None,'active_pixels_changed':sum(r['pixel_changed']=='True' for r in active)})
    csv_write(out/'augmentation_summary.csv',stats)
    weights=sorted(out.glob('*/weights/epoch*_fp32.pt'))
    (out/'checkpoint_SHA256SUMS.txt').write_text(''.join(sha(p)+'  '+str(p)+'\n' for p in weights),encoding='utf-8')
    acceptance={'status':status,'RunID':out.name,'actual_epochs':epochs,'complete_evaluation_cells':len(rows),'formal_evaluation_images':len(rows)*548,'both_arms_completed':complete,
        'scope':['DEVELOPMENT_ONLY','SINGLE_SEED45','POST_HOC_POLICY_SELECTION','Controlled synthetic Motion/Haze','Not formal1280/100epoch RQ4'],
        'per_cell_warmup_verified':complete,'old_ledger_unchanged':old_same,'failure':failure,'new_remaining_seconds':remaining(out),'retry':False,
        'image_bootstrap_not_training_seed_significance':True,'unknown_P_controller':'ABSTAIN','no_test_dev_used':True,'no_AP_benefit_assumed':True}
    save(out/'final_acceptance.json',acceptance)
    figures=plots(out,rows,comparisons) if complete else []
    evidence='\n'.join('- `'+str(out/n)+'`' for n in ['training_protocol.json','training_arm_manifest.csv','initialization_SHA.json','epoch_metrics.csv','augmentation_event_log.csv','augmentation_summary.csv','checkpoint_SHA256SUMS.txt','resource_usage.json','final_acceptance.json','SHA256SUMS.txt'])
    metrics=table(rows,['strategy','condition','AP50','AP50_95','Tiny_AP50_95','cached_pipeline_ms','state'])
    effects=table([r for r in comparisons if r['size'] in ['all','tiny'] and r['metric']=='map50_95'],['condition','size','CONTROL','MOTION_TARGETED','difference','CI95_low','CI95_high'])
    gap=table(transitions,['condition','transition','confirmed_Gap_closed'])
    costs=table([{'stage':c['stage'],'seconds':c['charged_seconds'],'stop_reason':c['result']['stop_reason']} for c in charges],['stage','seconds','stop_reason'])
    common=f'RunID: `{out.name}`\n\n状态：{status}。实际轮数：{epochs}。完整科研评价 {len(rows)}/8 格，{len(rows)*548} 张次。\n'
    full=common+f'\n本次干预为 Motion p=0.25、训练核 {{5,9,13,17,25}}、角度 Uniform[0,180)。评价 Motion 核为21。两臂相同 seed45、原生10类、初始化、6471/548图、960/10轮、FP32、batch2、workers0、AdamW与普通增强；选固定第10轮 EMA。close_mosaic=10使此10轮探索从首轮关闭 Mosaic；正式1280/100轮协议未变。\n\n'+metrics+'\n\n'+effects+'\n\n'+gap+'\n\n'+costs+f'\n\n新账本剩余 {resources["new_remaining_seconds"]:.3f} 秒，固定到期 {ledger["expiry_UTC"]}；旧账本校验保持不变：{old_same}。GPU峰温 {resources["peak_temperature_C"]}°C、显存占用峰值 {resources["peak_GPU_used_MiB"]} MiB。监督总墙钟含训练/验证/保存/推理/统计，不等于纯CUDA事件时间。\n\n首轮成本关卡不使用AP。CONTROL首轮之后保留同一进程内的模型、EMA、优化器、调度器、scaler、待累积梯度与数据对象；TARGETED完成后核验状态并恢复全局RNG，再继续CONTROL。该机制不是磁盘精确续训证明。若关卡未通过，两组首轮仅为部分训练证据。\n\n200次同图配对Bootstrap重新累计汇总COCO AP，区间为逐项95%图像区间；未控制真实航次聚类，不代表训练seed显著性。策略由开发数据选择，不能声称独立泛化或与历史Gaussian三seed合并。未知P控制器继续ABSTAIN。AP提升只有同时通过Tiny及完整缓存时延的固定置信界，才可称已测范围Gap关闭。\n\n'
    full+=('实际阻塞：\n```text\n'+failure+'\n```\n' if failure else '下一步：先由本次单seed结果判断是否值得另行授权Motion策略多seed重复；不自动增加训练。\n')
    full+='\n复现入口：`'+str(HERE/'motion_run.py')+'`。本次授权已使用，禁止直接重跑；同一Run的证据用于只读复算。命令与监督日志见各 `*_supervision/command.txt`、`worker.log`。\n\n'+evidence
    short=common+'\n问题：开发证据中的Motion Tiny失败区，能否通过单一可解释Motion训练干预改善？\n\n'+effects+'\n\n'+gap+'\n\n'+f'代价：{resources["GPU_worker_supervision_charged_seconds"]:.1f}秒监督worker；余额{remaining(out):.1f}秒。仅一个训练seed、开发集、探索性策略。没有完整8格时，不提供训练收益结论。\n'
    rq=common+'\n|研究项|当前证据|本次边界|\n|---|---|---|\n|RQ1 配置/采样|沿用原UVL和已验收枚举；本次没有重计数|Train960是明确探索扩展|\n|RQ2 交互|复用VM45/47真实交互；新8格完成后提供干预后的Motion/Haze响应|不是物理因果|\n|RQ3 Gap|Tiny>=.10及缓存管线<=100ms固定置信界|仅4P和这两个训练策略；未知P弃权|\n|RQ4 增强适配|Motion定向CONTROL匹配干预|单seed960/10轮，不代替正式1280/100轮|\n\n'+gap+'\n\n'+evidence
    names=['VM48_Motion定向训练_科学报告_CN.md','VM48_Motion定向训练_导师简报_CN.md','VM48_Motion定向训练_RQ4证据_CN.md'];paths=[]
    for name,body in zip(names,[full,short,rq]):
        p=ROOT/'datasets/reportcn'/name
        if p.exists():raise FileExistsError('Preserve an existing report: '+str(p))
        p.write_text(body,encoding='utf-8');paths.append(str(p))
    for p in load(out/'weekly_reports.json')['paths']:
        p=Path(p);backup=out/('before_update_'+p.name);backup.write_bytes(p.read_bytes())
        if p.name.endswith('_EN.md'):note=f'\n\n## Seed45 update\nRun `{out.name}`: {status}; epochs {epochs}; {len(rows)}/8 evaluation cells. No incomplete result is treated as an AP improvement. See `{paths[0]}`.\n'
        else:note=f'\n\n## Seed45后续实测更新\nRun `{out.name}`：{status}；实际轮数{epochs}；完整科研评价{len(rows)}/8格。\n\n'+effects+'\n\n'+gap+'\n\n完整证据：`'+paths[0]+'`。\n'
        with p.open('a',encoding='utf-8') as f:f.write(note)
    save(out/'report_paths.json',{'reports':paths,'figures':figures,'weekly_updated':load(out/'weekly_reports.json')['paths']})
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Seed45 results</title><h1>Motion-targeted seed45</h1><pre>'+html.escape(common+metrics+'\n\n'+effects+'\n\n'+gap)+'</pre>'+''.join('<p><img width="850" src="'+str(Path(p).relative_to(out)).replace('\\','/')+'"></p>' for p in figures if p.endswith('.png')),encoding='utf-8')
    save(out/'report_generation.json',{'CPU_wall_seconds':time.perf_counter()-started,'outside_GPU_worker':True,'no_new_COCO_or_GPU_calls':True})
    snapshots=out/'execution_source';snapshots.mkdir()
    for p in HERE.glob('*.py'):(snapshots/p.name).write_bytes(p.read_bytes())
    (out/'SHA256SUMS.txt').write_text(''.join(sha(p)+'  '+str(p.relative_to(out))+'\n' for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.txt'),encoding='utf-8')

def plots(out,rows,comparisons):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=out/'plots';dest.mkdir();result=[];conditions=['Normal','Motion_k21','Haze_dense','Motion_Haze'];arms=['CONTROL','MOTION_TARGETED'];colors=['#3675a8','#dd8638']
    def emit(fig,name):
        fig.tight_layout()
        for ext in ['png','pdf']:
            p=dest/(name+'.'+ext);fig.savefig(p,dpi=180);result.append(str(p))
        plt.close(fig)
    def values(arm,key):return [float(next(r[key] for r in rows if r['strategy']==arm and r['condition']==c)) for c in conditions]
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(4)
    for j,a in enumerate(arms):ax.bar(x+(j-.5)*.36,np.array(values(a,'AP50_95'))*100,.36,label=a,color=colors[j])
    ax.set(xticks=x,xticklabels=['Normal','Motion','Haze','Composite'],ylabel='AP50:95 (%)',title='Seed45 matched intervention');ax.legend(fontsize=8);emit(fig,'strategy_AP')
    fig,ax=plt.subplots(figsize=(7,4))
    for j,a in enumerate(arms):
        rr=[next(r for r in rows if r['strategy']==a and r['condition']==c) for c in conditions];v=np.array([float(r['Tiny_AP50_95']) for r in rr])*100
        lo=np.array([float(r['Tiny_CI_low']) for r in rr])*100;hi=np.array([float(r['Tiny_CI_high']) for r in rr])*100
        ax.errorbar(x+(j-.5)*.12,v,yerr=[np.maximum(0,v-lo),np.maximum(0,hi-v)],fmt='o',label=a,color=colors[j])
    ax.axhline(10,color='red',ls='--');ax.set(xticks=x,xticklabels=['Normal','Motion','Haze','Composite'],ylabel='Tiny AP50:95 (%)',title='Paired-image pointwise intervals');ax.legend(fontsize=8);emit(fig,'Tiny_threshold')
    for condition,name in [('Normal','Normal_accuracy_cost'),('Motion_Haze','Composite_effect')]:
        rr=[r for r in comparisons if r['condition']==condition and r['metric']=='map50_95'];v=np.array([float(r['difference']) for r in rr])*100
        fig,ax=plt.subplots(figsize=(7,4));ax.bar([r['size'] for r in rr],v,color=colors[1]);ax.axhline(0,color='grey');ax.set(ylabel='TARGETED - CONTROL (pp)',title=condition+' AP change');emit(fig,name)
    fig,ax=plt.subplots(figsize=(8,5));rr=[r for r in comparisons if r['metric']=='map50_95' and r['size'] in ['all','tiny']]
    v=np.array([float(r['difference']) for r in rr])*100;lo=np.array([float(r['CI95_low']) for r in rr])*100;hi=np.array([float(r['CI95_high']) for r in rr])*100
    ax.errorbar(v,np.arange(len(v)),xerr=[np.maximum(0,v-lo),np.maximum(0,hi-v)],fmt='o');ax.axvline(0,color='grey');ax.set(yticks=np.arange(len(v)),yticklabels=[r['condition']+' '+r['size'] for r in rr],xlabel='AP change (pp)',title='Single-seed paired image intervals');emit(fig,'paired_differences')
    epochs=read_csv(out/'epoch_metrics.csv');fig,axes=plt.subplots(1,2,figsize=(10,4))
    for j,a in enumerate(arms):
        rr=[r for r in epochs if r['strategy']==a];axes[0].plot([int(r['epoch']) for r in rr],[float(r['active_epoch_seconds']) for r in rr],label=a,color=colors[j])
    axes[0].set(xlabel='Epoch',ylabel='Active epoch seconds');axes[0].legend(fontsize=7)
    for stage in ['training','evaluation']:
        path=out/(stage+'_supervision/gpu_telemetry.jsonl')
        if path.exists():
            with path.open(encoding='utf-8') as f:data=[json.loads(s) for s in f]
            axes[1].plot([r['monotonic_seconds']/3600 for r in data],[r['temperature_C'] for r in data],label=stage)
    axes[1].axhline(90,color='red',ls='--');axes[1].set(xlabel='Stage hours',ylabel='GPU temperature (C)');axes[1].legend(fontsize=7);emit(fig,'training_resources')
    return result
