"""Withdraw unadmitted timing claims without changing measured AP evidence."""
import argparse,csv,json,os,time,shutil,importlib.util
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
_spec=importlib.util.spec_from_file_location('vm48_transfer_review_source',Path(__file__).with_name('run.py'))
pilot=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(pilot)

def main(out):
    started=time.perf_counter();budget=pilot.accepted.budget()
    if budget['remaining_seconds']<1260:raise RuntimeError('Review must preserve1200-second reserve.')
    assert (out/'timing_protocol_deviation.json').exists()
    analysis=pilot.READ(out/'analysis_acceptance.json');seeds=analysis['completed_seeds']
    archive=out/'before_timing_review';archive.mkdir()
    mapping=[]
    def keep(path):
        path=Path(path);target=archive/path.name
        if target.exists():raise FileExistsError(target)
        digest=pilot.SHA(path);shutil.move(str(path),str(target));mapping.append({'original':str(path),'preserved_copy':str(target),'SHA256':digest})
    def csvread(name):
        with (out/name).open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
    cfg=csvread('configuration_metrics.csv') if (out/'configuration_metrics.csv').exists() else []
    original_AP_fields=[{k:r[k] for k in ['seed','strategy','condition','AP50','AP50_95','Tiny_AP50_95','Small_AP50_95','Medium_AP50_95','Large_AP50_95','AR50_95','FP_IoU50_per_image']} for r in cfg]
    lookup={}
    for r in cfg:
        r['state']='FAIL' if float(r['Tiny_CI_high'])<.10 else 'UNCERTAIN'
        r['state_basis']='Tiny upperCI provesFAIL; otherwise latency unadmitted so noPASS'
        r['timing_fully_VM41_compliant']=False;r['cached_latency_diagnostic_only']=True
        lookup[int(r['seed']),r['strategy'],r['condition']]=r['state']
    for name in ['configuration_metrics.csv','tiny_gap_transitions.csv','Gap_counts.csv','controller_comparison.csv','final_acceptance.json','resource_usage.json','SHA256SUMS.txt']:
        if (out/name).exists():keep(out/name)
    if cfg:pilot.csv_write(out/'configuration_metrics.csv',cfg)
    transitions=[];counts=[];controllers=[]
    conditions=['Normal','Motion_k21','Haze_dense','Motion_Haze']
    for seed in seeds:
        for condition in conditions:
            c=lookup[seed,'CONTROL',condition];g=lookup[seed,'VARIABILITY_GUIDED',condition]
            transitions.append({'seed':seed,'condition':condition,'CONTROL_state':c,'GUIDED_state':g,'transition':c+' -> '+g,'confirmed_Gap_closed':False,'basis':'NoPASSrelease: required per-cell warmups not met; AP intervals unchanged'})
            for method in ['FIXED_CONTROL','FIXED_GUIDED','DEVELOPMENT_EXACT_LOOKUP','SAFE_UNKNOWN_P']:
                choice='CONTROL' if method=='FIXED_CONTROL' else ('VARIABILITY_GUIDED' if method=='FIXED_GUIDED' else None)
                controllers.append({'seed':seed,'condition':condition,'method':method,'selected_strategy':choice,'verified_state':lookup[seed,choice,condition] if choice else 'ABSTAIN','abstain':choice is None,'independent_false_PASS_validation':'NOT_EVALUATED','unknown_P_deployment_authorized':False,'scope':'No lookupPASS without admitted latency; fixed methods select candidates without predictedPASS claims'})
        for strategy in ['CONTROL','VARIABILITY_GUIDED']:
            labels=[lookup[seed,strategy,c] for c in conditions];n=labels.count('FAIL')
            counts.append({'seed':seed,'strategy':strategy,'P_count':4,'COVERED':0,'GAP_proven_by_Tiny_upperCI':n,'UNCERTAIN_due_to_unadmitted_latency':4-n,'gap_count_lower':n,'gap_count_upper':4,'Gap_closure':'NOT_CONFIRMED_PROTOCOL_DEVIATION'})
    pilot.csv_write(out/'tiny_gap_transitions.csv',transitions);pilot.csv_write(out/'Gap_counts.csv',counts);pilot.csv_write(out/'controller_comparison.csv',controllers)
    diffs=[]
    for seed in seeds:
        c=next(r for r in counts if r['seed']==seed and r['strategy']=='CONTROL');g=next(r for r in counts if r['seed']==seed and r['strategy']=='VARIABILITY_GUIDED')
        diffs.append({'seed':seed,'reduction_lower_bound':c['gap_count_lower']-g['gap_count_upper'],'reduction_upper_bound':c['gap_count_upper']-g['gap_count_lower'],'point_Gap_reduction':'NOT_ESTIMABLE_UNADMITTED_LATENCY','Gap_bootstrap_CI':'NOT_EVALUATED_UNADMITTED_LATENCY'})
    pilot.csv_write(out/'Gap_interval_bounds.csv',diffs)
    comparisons=csvread('paired_strategy_comparisons.csv') if (out/'paired_strategy_comparisons.csv').exists() else []
    if comparisons:
        keep(out/'paired_strategy_comparisons.csv')
        for r in comparisons:
            if r['metric'] in ('cached_pipeline_ms','FP_IoU50_per_image'):
                c=next(v for v in cfg if v['seed']==r['seed'] and v['condition']==r['condition'] and v['strategy']=='CONTROL')
                g=next(v for v in cfg if v['seed']==r['seed'] and v['condition']==r['condition'] and v['strategy']=='VARIABILITY_GUIDED')
                r['CONTROL']=c[r['metric']];r['GUIDED']=g[r['metric']]
                assert abs(float(r['GUIDED_minus_CONTROL'])-(float(g[r['metric']])-float(c[r['metric']])))<1e-10
        pilot.csv_write(out/'paired_strategy_comparisons.csv',comparisons)
    transfer=[]
    for seed in seeds:
        for size in ['all','tiny']:
            for condition in conditions[1:]:
                values={}
                for strategy in ['CONTROL','VARIABILITY_GUIDED']:
                    for name in ['Normal',condition]:
                        r=next(v for v in cfg if int(v['seed'])==seed and v['strategy']==strategy and v['condition']==name)
                        folder=Path(r['source_metrics']).parent
                        m=pilot.READ(folder/'metrics.json');b=pilot.READ(folder/'bootstrap.json')['draws']
                        values[strategy,name]=(next(v['map50_95'] for v in m['per_area'] if v['size']==size),np.asarray(b['values'][size]['map50_95'],float))
                c=values['CONTROL',condition];g=values['VARIABILITY_GUIDED',condition];cn=values['CONTROL','Normal'];gn=values['VARIABILITY_GUIDED','Normal']
                corrected=g[0]-c[0]-(gn[0]-cn[0]);draw=g[1]-c[1]-(gn[1]-cn[1]);ci=np.percentile(draw,[2.5,97.5])
                transfer.append({'seed':seed,'condition':condition,'size':size,'CONTROL_degradation_loss':cn[0]-c[0],'GUIDED_degradation_loss':gn[0]-g[0],
                    'absolute_degraded_AP_gain':g[0]-c[0],'Normal_AP_gain':gn[0]-cn[0],'Normal_adjusted_AP_gain':corrected,'CI95_low':float(ci[0]),'CI95_high':float(ci[1]),
                    'interpretation':'Reduced loss can reflect lower Normal AP; do not substitute for absolute degraded AP or Gap closure'})
    pilot.csv_write(out/'transfer_specificity.csv',transfer)
    def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(str(v) for v in r)+' |' for r in rows]])
    rows=[]
    for r in comparisons:
        if r['metric']=='map50_95' and r['size'] in ['all','tiny']:
            rows.append([r['seed'],r['condition'],r['size'],f"{float(r['CONTROL'])*100:.3f}",f"{float(r['GUIDED'])*100:.3f}",f"{float(r['GUIDED_minus_CONTROL'])*100:+.3f}",f"[{float(r['CI95_low'])*100:+.3f},{float(r['CI95_high'])*100:+.3f}]"])
    numbers=table(['seed','P','面积','CONTROL%','GUIDED%','差值pp','配对95%区间pp'],rows)
    warning='执行入口预热频率实现错误：每模型3次，而VM41要求每格3次。原始预测、COCO AP及同图Bootstrap仍有效；完整缓存计时的组件未变，但不能把这些时延写成完整VM41准入。未重跑任何GPU。没有按新AP修改参数。'
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if seeds:
        labels=['PASS','FAIL','UNCERTAIN'];grid=np.zeros((3,3),int)
        for r in transitions:grid[labels.index(r['CONTROL_state']),labels.index(r['GUIDED_state'])]+=1
        fig,ax=plt.subplots(figsize=(6,4));im=ax.imshow(grid,cmap='Blues')
        for i in range(3):
            for j in range(3):ax.text(j,i,str(grid[i,j]),ha='center',va='center')
        ax.set(xticks=range(3),yticks=range(3),xticklabels=labels,yticklabels=labels,xlabel='GUIDED conservative state',ylabel='CONTROL conservative state',title='No PASS release: latency protocol deviation')
        fig.colorbar(im,ax=ax);fig.tight_layout()
        for ext in ['png','pdf']:
            p=out/'plots'/('Gap_transition_matrix.'+ext)
            if p.exists():keep(p)
            fig.savefig(p,dpi=180)
        plt.close(fig)
        fig,axes=plt.subplots(1,len(seeds),figsize=(5*len(seeds),4),squeeze=False)
        for ax,seed in zip(axes.flat,seeds):
            x=np.arange(4)
            for j,strategy in enumerate(['CONTROL','VARIABILITY_GUIDED']):
                vals=[float(next(r['cached_pipeline_ms'] for r in cfg if int(r['seed'])==seed and r['strategy']==strategy and r['condition']==c)) for c in conditions]
                ax.bar(x+(j-.5)*.36,vals,.36,label=strategy)
            ax.axhline(100,color='red',linestyle='--');ax.set(xticks=x,xticklabels=['Normal','Motion','Haze','Composite'],ylabel='Diagnostic cached pipeline (ms)',title=f'Seed {seed}: warmup deviation');ax.legend(fontsize=7)
        fig.tight_layout()
        for ext in ['png','pdf']:
            p=out/'plots'/('cached_latency_comparison.'+ext)
            if p.exists():keep(p)
            fig.savefig(p,dpi=180)
        plt.close(fig)
    gap_table=table(['seed','策略','Tiny置信界证明FAIL','未决','Gap数上下界'],[[r['seed'],r['strategy'],r['GAP_proven_by_Tiny_upperCI'],r['UNCERTAIN_due_to_unadmitted_latency'],str([r['gap_count_lower'],4])] for r in counts])
    full=f'''# VM48 笔记本跨退化定向增强实验

RunID：`{out.name}`。最终状态PARTIAL_WITH_LIMITATIONS。完整AP评价seed：{seeds}，{len(cfg)}格、{len(cfg)*548}张次；另有每组6张预热，不算正式评价。没有新训练，使用原960/10轮fixed epoch10 EMA。

{warning}

{numbers}

AP50、各尺寸/十类AP、AR及FP/image见configuration_metrics.csv、per_class_metrics.csv、paired_strategy_comparisons.csv及paired_per_class_comparisons.csv。所有AP是548图池化COCO结果，200次相同图像索引重采样，不平均每图AP；区间是逐项95%百分位，并非多重比较同时保证。Normal代价为CONTROL−GUIDED。不同seed分别比较，跨seed范围/SD在seed_consistency.csv；最多3个seed不能冒称大量训练seed显著性。

Motion_Haze_DID.csv保存每策略四角点DID及策略DID差。像素来自VM45封存2192个PNG，完整核验文件/像素SHA、GT与成员身份，未复制大缓存。kernel21/beta1.5来自原UVL；角度Uniform[0,180)及airlight220为作者认可本轮复用的工程近似，未物理标定。FP/image定义为原conf.001、COCO IoU.50、ignore-union后的非忽略未匹配框数/548，不是另选部署阈值。

当前没有任何可靠的FAIL→PASS或Gap关闭结论。Tiny upperCI<.10足以证明FAIL，其余因时延未准入保留UNCERTAIN；这不是把缺失结果填0。Gap可识别上下界见Gap_interval_bounds.csv；不能用AP变化冒充Gap缩小。

全部来自同一开发图像与历史方法选择暴露：DEVELOPMENT_ONLY / POST_HOC_POLICY_SELECTION。P关联原FM_P ID；D共享YOLO26、Small、VisDrone_FT、Infer1280、FullFrame、NMS、FP32及GUIDED的NoiseAug。Train960不在原FM_D训练分辨率域内，明确是探索扩展，不改成Train1280，不改原UVL。没有读取VM43 test-dev调参。

固定CONTROL、固定GUIDED、开发证据查表与安全弃权见controller_comparison.csv。时延未完整准入，查表不释放PASS；未知P始终ABSTAIN。不能报告未知P规则安全泛化成功。

Normal校正后的收益与退化损失另存transfer_specificity.csv；退化损失减少可能来自Normal AP变差，不能替代绝对AP。实际结果仅Motion整体AP三seed均正向；Haze和复合条件方向不一致，Tiny收益弱。先单独授权修正逐格预热的时延验证，本轮不补跑。若另做最小匹配训练，提议独立seed45、YOLO26s、960/10轮、CONTROL对单Motion定向策略，全6471/548及相同8格评价；具体提案见next_matched_training_proposal.json。候选新总预算10小时、不是现有余额扩容；先有完整首轮成本与新训练散热准入才能开始匹配两臂。Motion训练成本尚未实测，不能保证10小时完成。正式1280/100轮RQ4仍NOT_EVALUATED。

原始脚本/冻结协议及审查前记录保存在before_timing_review，防止掩盖实现错误。没有GPU重试、参数优化或预算重置。逐格原始预测、COCO/Bootstrap与时延文件未修改。
'''
    brief=f'''# VM48 导师实验结果简报

完成seed {seeds}的真实匹配AP评价：{len(cfg)}格×548＝{len(cfg)*548}张次。Gaussian指导训练是否迁移Motion/Haze，按下表及配对区间判断，不按结果补选seed。

{numbers}

限制：{warning}因此只能陈述AP/Tiny/AR迁移，不能陈述Gap关闭或完整时延合规。开发集/事后策略选择；最多3训练seed。未知P控制器ABSTAIN；正式100轮研究未完成。
'''
    gaptext=f'''# VM48 RQ4探索性Gap评价

RunID：`{out.name}`。{warning}

{gap_table}

只在Tiny upperCI<.10时证明FAIL，其他组合保留UNCERTAIN，无PASS释放。不能确认Motion/Haze复合Gap缩小。Gap_interval_bounds.csv保存部分识别界；完整Gap变化和其Bootstrap区间为NOT_EVALUATED_UNADMITTED_LATENCY，不编造0收益。

同4个有目的P，各seed单一CONTROL或GUIDED detector，不外推全FM_P空间。固定策略可作AP基线；开发查表及未知P安全控制器均不能获准PASS。原UVL、VM43隔离边界和正式100轮协议不变。
'''
    reports=pilot.READ(out/'report_paths.json')
    for path,text in zip(reports,[full,brief,gaptext]):
        keep(path)
        with Path(path).open('x',encoding='utf-8') as f:f.write(text)
    pilot.SAVE(out/'next_matched_training_proposal.json',{'status':'PROPOSED_NOT_AUTHORIZED_NOT_TRAINED','reason':'Small positive Motion overall AP transfer, weak Tiny effect, inconsistent Haze/composite; test direct Motion targeting rather than asserting broad Gaussian transfer',
        'model':'YOLO26s','seed':45,'training_resolution':960,'epochs':10,'train_images':6471,'development_images':548,'FP32':True,'batch':2,'nbs':64,'workers':0,'optimizer':'AdamW','lr0':.01,'warmup_epochs':3,'close_mosaic':10,
        'arms':['CONTROL','MOTION_GUIDED_CANDIDATE'],'initialization':'Same healthy native10 initialization within pair, generated from the accepted COCO source for seed45 before freezing; currently not created',
        'policy_candidate':{'MotionBlurAug_probability':.25,'odd_kernel_values':[9,11,13,17,19],'angle_degrees':'Uniform[0,180)','excluded_test_kernels':[7,15,21],'other_added_families':False,'status':'CANDIDATE_REQUIRES_SEPARATE_FREEZE_AND_AUTHORIZATION'},
        'new_allocation_candidate_seconds':36000,'candidate_caps_seconds':{'two_full_epoch_cost_benchmarks':3600,'CONTROL_10epoch':10800,'MOTION_GUIDED_10epoch':14400,'frozen8cell_evaluation':5400,'shutdown_reserve':1800},
        'measured_reference':'Existing seed43 CONTROL8486.58s / GaussianGUIDED9461.10s; new Motion cost UNKNOWN; current8cell evaluation costs in group supervisors',
        'training_cost_reliability':'ESTIMATE_NOT_RELIABLE_FOR_NEW_MOTION_ARM','admission':'New explicit allocation/deadline, fresh thermal/cost/storage gate, complete pair fit; no current training permission','historical_budget_reset':False})
    assert original_AP_fields==[{k:r[k] for k in original_AP_fields[0]} for r in cfg] if cfg else True
    preserved=pilot.READ(out/'source_provenance.json')['files']
    assert all(pilot.SHA(r['path'])==r['SHA256'] for r in preserved)
    pilot.SAVE(out/'timing_review_receipt.json',{'review_source':{'path':str(Path(__file__).resolve()),'SHA256':pilot.SHA(__file__)},'AP_points_unchanged':True,'raw_COCO_and_bootstrap_files_unchanged':True,
        'frozen_worker_sources_and_weights_unchanged':True,'latency_and_FP_point_columns_corrected_to_actual_sample_means':True,'timing_admitted':False,'new_GPU_calls':0,'new_COCO_evaluations':0,'source_of_seal_failure':'Parent module name collision; scientific worker and CPU metric generation completed'})
    pilot.SAVE(out/'review_preservation_map.json',mapping)
    pilot.charge(out,'CPU_scientific_timing_review',time.perf_counter()-started)
    b=pilot.accepted.budget();own=[r for r in b['charges'] if Path(r['path']).name.startswith(out.name+'_')];seconds=sum(pilot.READ(r['path'])['charged_seconds'] for r in own)
    pilot.SAVE(out/'resource_usage.json',{'total_charged_worker_and_CPU_seconds':seconds,'charges':own,'budget_after':b,'GPU_training_seconds':0,'kernel_busy_seconds':'NOT_RECORDED','untouched1200s_reserve_preserved':b['remaining_seconds']>=1200})
    final={**analysis,'status':'PARTIAL_WITH_LIMITATIONS','AP_evaluation_completed_seeds':seeds,'timing_fully_VM41_compliant':False,'Gap_closure':'NOT_CONFIRMED_PROTOCOL_DEVIATION','false_PASS_release':False,'budget_after':b,'untouched1200s_reserve_preserved':b['remaining_seconds']>=1200,'review':'review_preservation_map.json'}
    pilot.SAVE(out/'final_acceptance.json',final)
    r=pilot.READ(out/'run.json');r['status']='PARTIAL_WITH_LIMITATIONS';(out/'run.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    footer=f"\n实际计费{seconds:.3f}秒，账本余额{b['remaining_seconds']:.3f}秒，原截止{b['deadline_UTC']}；kernel忙时未独立记录。GPU训练0。\n"
    for path in reports:
        with Path(path).open('a',encoding='utf-8') as f:f.write(footer)
    html=(out/'index.html').read_text(encoding='utf-8');html=html.replace('<h1>VM48 laptop transfer pilot</h1>','<h1>VM48 laptop transfer pilot — PARTIAL_WITH_LIMITATIONS</h1><p>AP valid; per-cell warmup deviation prevents VM41 latency admission and Gap closure claims.</p>');(out/'index.html').write_text(html,encoding='utf-8')
    with (out/'SHA256SUMS.txt').open('x',encoding='utf-8') as f:
        for p in sorted([q for q in out.rglob('*') if q.is_file() and not any(x.startswith('runtime_') for x in q.relative_to(out).parts)]+[Path(q) for q in reports]):
            if p!=out/'SHA256SUMS.txt':f.write(pilot.SHA(p)+'  '+str(p)+'\n')
    print(json.dumps({'complete_seeds':seeds,'cells':len(cfg),'image_records':len(cfg)*548,'status':final['status'],'remaining_seconds':b['remaining_seconds']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();main(a.run.resolve(strict=True))
