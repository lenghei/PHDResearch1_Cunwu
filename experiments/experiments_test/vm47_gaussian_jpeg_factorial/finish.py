"""Verify VM47 raw evidence and compute paired factorial contrasts on CPU."""
import argparse,csv,hashlib,json,math,os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from pathlib import Path
import sys,time
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'main/code'));sys.path.insert(0,str(HERE.parent/'vm43_external_pilot'))
RUNS=ROOT/'datasets/local_runs'/HERE.name
CONDITIONS=('Normal','Gauss_s15','JPEG_q25','Gauss_s15_JPEG_q25')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def ref(path):
    p=Path(path).resolve(strict=True);return {'path':str(p),'SHA256':sha(p),'bytes':p.stat().st_size}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def table(path,rows,fields=None):
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def interval(point,draws):
    valid=[x for x in draws if x is not None]
    return {'estimate':point,'CI_low':float(np.percentile(valid,2.5)) if valid else None,
            'CI_high':float(np.percentile(valid,97.5)) if valid else None,'valid_draws':len(valid)}


def binary_score(rows):
    scored=[r for r in rows if r['actual'] in ('PASS','FAIL')]
    fail=sum(r['actual']=='FAIL' for r in scored);pos=sum(r['actual']=='PASS' for r in scored)
    fp=sum(r['actual']=='FAIL' and r['predicted']=='PASS' for r in scored)
    tp=sum(r['actual']=='PASS' and r['predicted']=='PASS' for r in scored)
    fn=pos-tp;tn=fail-fp;pp=fp+tp
    return {'scored_pairs':len(scored),'actual_PASS':pos,'actual_FAIL':fail,'UNCERTAIN':sum(r['actual']=='UNCERTAIN' for r in rows),
            'actual_FAIL_predicted_PASS':fp,'false_PASS_rate':fp/fail if fail else None,'false_PASS_denominator':fail,
            'PASS_precision':tp/pp if pp else None,'PASS_prediction_denominator':pp,'failure_recall':tn/fail if fail else None,
            'balanced_accuracy':(tp/pos+tn/fail)/2 if pos and fail else None,'confusion_PASS_FAIL':[[tp,fn],[fp,tn]],
            'raw_prediction_coverage':1. if rows else None,'raw_prediction_abstention':0. if rows else None,
            'safe_controller_abstention':1.,'deployment_promotions':0,'independent_image_generalization':False}


def finish(identity):
    started=time.perf_counter()
    if Path(identity).name!=identity or not identity.startswith(HERE.name+'_execution_'):raise ValueError('Expected VM47 execution RunID.')
    out=(RUNS/identity).resolve(strict=True)
    if (out/'final_acceptance.json').exists():raise FileExistsError('Immutable final receipt exists.')
    spec=read(out/'experiment_protocol.json');complete=(out/'execution_end.json').exists() and read(out/'execution_end.json')['status']=='REAL_EIGHT_CELLS_COMPLETED'
    prior=read(out/'execution_source_provenance.json');preserved=[]
    for r in prior['files']:
        assert sha(r['path'])==r['SHA256'],r['path'];preserved.append(r)
    save(out/'preservation_verification.json',{'status':'REGISTERED_INPUTS_UNCHANGED','files':preserved,'no_claim_about_unregistered_filesystem':True})
    payload=read(spec['GT']['path']);assert sha(spec['GT']['path'])==spec['GT']['SHA256']
    ids=[r['id'] for r in payload['records']];cache=read(out/'pixel_cache_manifest.json') if (out/'pixel_cache_manifest.json').exists() else {}
    for entries in cache.values():
        for r in entries:assert sha(r['path'])==r['file_SHA256']
    indices=np.load(out/'paired_bootstrap_indices.npy',allow_pickle=False)
    assert np.array_equal(indices,np.random.default_rng(spec['bootstrap_seed']).integers(0,548,(200,548)))
    cells={};metrics=[];classes=[];intervals=[];gaps=[];per_image=[];receipts=[]
    for c in spec['conditions']:
        for d in spec['models']:
            folder=out/(c['condition']+'__'+d['D'])
            if not (folder/'acceptance.json').exists():continue
            a=read(folder/'acceptance.json');m=read(folder/'metrics.json');b=read(folder/'bootstrap.json');t=read(folder/'timing.json')
            assert a['images']==m['images']==548 and a['GT']==spec['GT']
            for field in ('raw','GT','metrics','bootstrap','checkpoint'):assert sha(a[field]['path'])==a[field]['SHA256']
            assert a['checkpoint']['SHA256']==d['checkpoint']['sha256']
            manifest=[json.loads(x) for x in (folder/'image_manifest.jsonl').read_text().splitlines()]
            assert [r['image_id'] for r in manifest]==ids
            for row,pixel in zip(manifest,cache[c['condition']]):
                assert {k:v for k,v in row.items() if k!='tensor'}==pixel
                assert row['tensor']=={'NCHW':[1,3,1280,1280],'dtype':'torch.float32'}
            assert b['draws']['image_ids']==ids and b['draws']['samples']==200 and b['draws']['seed']==spec['bootstrap_seed']
            key=(d['D'],c['condition']);cells[key]=(m,b,t)
            receipts.append({'cell':folder.name,'raw':a['raw'],'GT':a['GT'],'checkpoint':a['checkpoint'],'metrics':a['metrics'],'bootstrap':a['bootstrap'],'images':548})
            for r in m['per_area']:metrics.append({'cell_id':folder.name,'D':d['D'],'condition':c['condition'],**r})
            for r in m['per_class']:classes.append({'cell_id':folder.name,'D':d['D'],'condition':c['condition'],**r})
            for r in b['intervals']:intervals.append({'cell_id':folder.name,**r})
            tiny=next(r for r in b['intervals'] if r['kind']=='area' and r['size']=='tiny' and r['metric']=='map50_95')
            latency=np.array([r['cached_pipeline_ms'] for r in t['per_image']]);ci=np.percentile(latency[indices].mean(axis=1),[2.5,97.5])
            state='FAIL' if ci[1]>100 or tiny['ci95'][1]<.1 else 'PASS' if tiny['ci95'][0]>=.1 else 'UNCERTAIN'
            gaps.append({'condition':c['condition'],'P_id':c['P_id'],'D':d['D'],'Tiny_AP':tiny['estimate'],'Tiny_CI_low':tiny['ci95'][0],
               'Tiny_CI_high':tiny['ci95'][1],'cached_pipeline_mean_ms':float(latency.mean()),'latency_CI_low_ms':float(ci[0]),'latency_CI_high_ms':float(ci[1]),'state':state})
            for r in t['per_image']:per_image.append({'cell_id':folder.name,**r})
    assert not complete or len(cells)==8
    table(out/'actual_metrics.csv',metrics,['cell_id','D','condition','size','map50','map50_95','ar50','ar50_95','gt_count','evaluated_categories','image_count','max_det','area_reference'])
    table(out/'per_class_metrics.csv',classes,list(classes[0]) if classes else ['cell_id','status'])
    table(out/'paired_bootstrap_intervals.csv',intervals,list(intervals[0]) if intervals else ['cell_id','status'])
    table(out/'gap_by_detector.csv',gaps,list(gaps[0]) if gaps else ['condition','D','state'])
    if per_image:table(out/'timing_by_image.csv',per_image)
    effects=[];cross=[]
    def endpoint(d,c,area,metric,category=None):
        m,b,t=cells[d,c];base=m['per_area'] if category is None else m['per_class']
        r=next(r for r in base if r['size']==area and (category is None or r['category_id']==category))
        draws=b['draws']['values'][area][metric] if category is None else b['draws']['class_values'][area+':'+str(category)][metric]
        return r[metric],draws
    def contrast(corners,weights):
        point=sum(w*r[0] for w,r in zip(weights,corners)) if all(r[0] is not None for r in corners) else None
        draws=[sum(w*r[1][i] for w,r in zip(weights,corners)) if all(r[1][i] is not None for r in corners) else None for i in range(200)]
        return point,draws
    if complete:
        for area in ('all','tiny','small','medium','large','coco_small'):
            for category in (None,*range(1,11)):
                for metric in ('map50','map50_95','ar50','ar50_95'):
                    did={}
                    for d in [r['D'] for r in spec['models']]:
                        corners=[endpoint(d,c,area,metric,category) for c in CONDITIONS]
                        for name,weights in [('Gaussian_effect',[-1,1,0,0]),('JPEG_effect',[-1,0,1,0]),('Interaction_DID',[1,-1,-1,1])]:
                            point,draws=contrast(corners,weights)
                            effects.append({'D':d,'size':area,'category_id':category,'metric':metric,'effect':name,**interval(point,draws),
                               'effect_pp':point*100 if point is not None else None,'scope':'DEVELOPMENT_ONLY_SYNTHETIC_EXPLORATORY'})
                            if name=='Interaction_DID':did[d]=(point,draws)
                    left,right=did['ck05_Infer1280'],did['ck01_Infer1280']
                    point=right[0]-left[0] if left[0] is not None and right[0] is not None else None
                    draws=[g-c if c is not None and g is not None else None for c,g in zip(left[1],right[1])]
                    cross.append({'contrast':'DID_ck01_minus_ck05','size':area,'category_id':category,'metric':metric,**interval(point,draws),'effect_pp':point*100 if point is not None else None})
        table(out/'factorial_effects.csv',effects);table(out/'cross_detector_DID.csv',cross)
        table(out/'Gaussian_JPEG_DID.csv',[r for r in effects if r['effect']=='Interaction_DID'])
    local=[];controllers=[];heldout=[]
    if complete:
        for c in spec['conditions']:
            measured=[r for r in gaps if r['condition']==c['condition']]
            feasible=sorted([r for r in measured if r['state']=='PASS'],key=lambda r:(r['cached_pipeline_mean_ms'],-r['Tiny_CI_low'],r['D']))
            local.append({'condition':c['condition'],'P_id':c['P_id'],'measured_D':2,'scope':'ck05/ck01 Infer1280 only',
                 'local_status':'COVERED' if feasible else 'GAP' if all(r['state']=='FAIL' for r in measured) else 'UNCERTAIN','full_FM_P_prevalence':'NOT_ESTIMABLE'})
            for method in ('SAFE_RULE_CONTROLLER','MEASURED_EXACT_LOOKUP','FIXED_ck05_Infer1280','FIXED_ck01_Infer1280'):
                selected=feasible[0] if method=='MEASURED_EXACT_LOOKUP' and feasible else next((r for r in measured if method=='FIXED_'+r['D']),None)
                controllers.append({'condition':c['condition'],'method':method,'D':selected['D'] if selected else 'ABSTAIN','measured_state':selected['state'] if selected else 'ABSTAIN',
                    'certified_coverage':bool(selected and selected['state']=='PASS'),'latency_ms':selected['cached_pipeline_mean_ms'] if selected else None,
                    'scope':'Retrospective measured lookup, not heldout predictive generalization or deployment approval'})
        with (out/'frozen_rule_predictions.csv').open(encoding='utf-8-sig') as f:predictions=list(csv.DictReader(f))
        for p in predictions:
            if p['unseen_full_P']!='True':continue
            actual=next(r for r in gaps if r['condition']==p['condition'] and r['D']==p['D'])['state']
            heldout.append({'condition':p['condition'],'P_id':p['P_id'],'D':p['D'],'predicted':p['predicted_state'],'actual':actual,
                'safe_action':'ABSTAIN','rule_refitted':False,'unseen_complete_P':True,'independent_images':False})
        assert len(heldout)==2
        table(out/'heldout_rule_validation.csv',heldout);table(out/'Controller_comparison.csv',controllers);table(out/'Gap_status.csv',local)
        score=binary_score(heldout)
    else:score={'status':'NOT_EVALUATED_INCOMPLETE_EIGHT_CELLS','deployment_promotions':0}
    save(out/'False_PASS_validation.json',score)
    import vm43_common
    budget=vm43_common.budget();charges=[];telemetry=[]
    for phase in ('preflight','formal'):
        charge=vm43_common.AUTH/'charges'/(out.name+'_'+phase+'.json')
        if charge.exists():charges.append(read(charge))
        path=out/phase/'gpu_telemetry.jsonl'
        if path.exists():
            samples=[json.loads(x) for x in path.read_text().splitlines()]
            telemetry.append({'phase':phase,'samples':len(samples),'max_C':max(r['temperature_C'] for r in samples),'max_used_MiB':max(r['used_MiB'] for r in samples),
                'max_record_interval_seconds':max((b['monotonic_seconds']-a['monotonic_seconds'] for a,b in zip(samples,samples[1:])),default=0),'raw':ref(path)})
    resource={'GPU_worker_wall_charged_seconds':sum(r['charged_seconds'] for r in charges),'charges':charges,'budget_after':budget,'telemetry':telemetry,
              'CPU_cache':read(out/'dataset_gate.json') if (out/'dataset_gate.json').exists() else None,
              'cost_definition':'Supervised GPU-worker wall includes CPU COCO and Bootstrap; not CUDA busy time or energy.',
              'kernel_busy_seconds':'NOT_MEASURED','power_and_fan_history':'NOT_RECORDED','training':False,'retries':0}
    resource['cell_costs']=[{'D':d,'condition':c,'cached_pipeline_seconds':sum(r['cached_pipeline_ms'] for r in t['per_image'])/1000,
        'CPU_evaluation_cost':t['evaluation_cost'],'peak_torch_allocated_bytes':t['peak_allocated_bytes'],'peak_torch_reserved_bytes':t['peak_reserved_bytes']}
        for (d,c),(m,b,t) in cells.items()]
    if complete:
        plots(out,spec,cells,effects,cross,gaps,local,score,resource)
    docs=reports(out,spec,cells,effects,cross,gaps,local,heldout,score,resource,complete)
    resource['CPU_postprocess_wall_seconds_before_seal']=time.perf_counter()-started
    save(out/'actual_resource_usage.json',resource)
    save(out/'source_provenance.json',{'execution':ref(out/'execution_source_provenance.json'),'verified_cells':receipts,'postprocessor':ref(__file__),
         'protocol':ref(out/'experiment_protocol.json'),'paired_bootstrap_plan':ref(out/'paired_bootstrap_plan.json'),'paired_indices':ref(out/'paired_bootstrap_indices.npy'),
         'frozen_rule':ref(out/'frozen_rule_protocol.json'),'frozen_predictions':ref(out/'frozen_rule_predictions.csv'),'no_VM43_testdev_for_learning':True,
         'new_rule_fitting':False,'scientific_scope':spec['scientific_scope']})
    save(out/'postprocessing_command.json',{'actual_command':sys.orig_argv,'GPU':False})
    save(out/'final_acceptance.json',{'status':'REAL_EIGHT_CELLS_COMPLETED_WITH_LIMITATIONS' if complete else 'PARTIAL_OR_BLOCKED','verified_cells':len(cells),
        'formal_image_records':len(cells)*548,'shared_bootstrap_draws':200,'rule_heldout_complete_P':1 if complete else 0,'deployment_promotions':0,
        'reports':[str(p) for p in docs],'source':ref(out/'source_provenance.json'),'resource':ref(out/'actual_resource_usage.json')})
    r=read(out/'run.json');r['status']='COMPLETE_WITH_LIMITATIONS' if complete else 'PARTIAL';(out/'run.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    (out/'diary.md').write_text('VM47 separate focused8-cell synthetic development study. Gaussian15/JPEG25 and original seed frozen before AP. Original VM46 preserved. One owned attempt, original ledger/expiry, no training/retry. Shared pixels and paired pooled-COCO Bootstrap. Frozen rule diagnostic is one P/two detectors only; safe controller ABSTAIN.\n',encoding='utf-8')
    links=''.join(f'<li><a href="{p.as_uri()}">{p.name}</a></li>' for p in docs)
    links+=''.join(f'<li><a href="{name}">{name}</a></li>' for name in ('actual_metrics.csv','factorial_effects.csv','Gaussian_JPEG_DID.csv','cross_detector_DID.csv','gap_by_detector.csv','Gap_status.csv','heldout_rule_validation.csv','False_PASS_validation.json','Controller_comparison.csv','actual_resource_usage.json','source_provenance.json') if (out/name).exists())
    charts=''.join(f'<img style="width:100%;max-width:1050px" src="plots/{name}.png">' for name in ('four_cell_AP','DID','Tiny_AP','Gap','controller_safety','actual_worker_cost')) if complete else ''
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>VM47 Gaussian × JPEG</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto}</style><h1>VM47 Gaussian × JPEG</h1><p>'+out.name+'</p><p>'+str(len(cells))+'/8 verified cells; DEVELOPMENT_ONLY synthetic exploration.</p><ul>'+links+'</ul>'+charts,encoding='utf-8')
    from main.visualization import build_run_index
    build_run_index(RUNS,HERE.name)
    files=[p for p in out.rglob('*') if p.is_file() and 'runtime' not in p.relative_to(out).parts]+docs+[Path(__file__).resolve()]
    with (out/'SHA256SUMS.txt').open('x',encoding='utf-8') as f:
        for p in sorted(files):f.write(sha(p)+'  '+str(p)+'\n')
    print(json.dumps({'RunID':out.name,'cells':len(cells),'complete':complete,'heldout_rule':score,'charged_seconds':resource['GPU_worker_wall_charged_seconds'],
                     'remaining_seconds':budget['remaining_seconds'],'reports':[str(p) for p in docs]},ensure_ascii=False),flush=True)


def plots(out,spec,cells,effects,cross,gaps,local,score,resource):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (out/'plots').mkdir()
    def export(fig,name):
        fig.tight_layout();fig.savefig(out/'plots'/(name+'.png'),dpi=170);fig.savefig(out/'plots'/(name+'.pdf'));plt.close(fig)
    for size,name in [('all','four_cell_AP'),('tiny','Tiny_AP')]:
        fig,ax=plt.subplots(figsize=(10,5))
        for j,d in enumerate(spec['models']):
            rows=[next(r for r in gaps if r['D']==d['D'] and r['condition']==c) for c in CONDITIONS]
            values=[next(r['map50_95'] for r in cells[d['D'],c][0]['per_area'] if r['size']==size)*100 for c in CONDITIONS]
            ax.bar(np.arange(4)+(j-.5)*.34,values,.32,label=d['D'])
        if size=='tiny':ax.axhline(10,ls='--',color='gray',label='Fixed Tiny threshold10%')
        ax.set(xticks=range(4),xticklabels=['Normal','Gaussian15','JPEG25','Gaussian15+JPEG25'],ylabel='AP50:95 (%)',title='VM47 '+size+' original-area AP; synthetic development');ax.legend();export(fig,name)
    fig,ax=plt.subplots(figsize=(10,5));selected=[r for r in effects if r['size'] in ('all','tiny') and r['category_id'] is None and r['metric']=='map50_95' and r['effect']=='Interaction_DID']
    for i,r in enumerate(selected):ax.hlines(i,r['CI_low']*100,r['CI_high']*100,color='#287e8e');ax.plot(r['effect_pp'],i,'o',color='#287e8e')
    ax.axvline(0,color='gray');ax.set(yticks=range(len(selected)),yticklabels=[r['D']+' / '+r['size'] for r in selected],xlabel='DID (pp), paired-image95% interval',title='Gaussian × JPEG; positive DID is not a detection improvement');export(fig,'DID')
    from matplotlib.colors import ListedColormap
    states={'FAIL':0,'UNCERTAIN':1,'PASS':2};matrix=np.array([[states[next(r for r in gaps if r['D']==d['D'] and r['condition']==c)['state']] for d in spec['models']] for c in CONDITIONS])
    fig,ax=plt.subplots(figsize=(7,5));ax.imshow(matrix,cmap=ListedColormap(['#e69b93','#f3d58c','#96cfba']),vmin=0,vmax=2)
    for i in range(4):
        for j in range(2):ax.text(j,i,['FAIL','UNCERTAIN','PASS'][matrix[i,j]],ha='center',va='center')
    ax.set(xticks=range(2),xticklabels=[d['D'] for d in spec['models']],yticks=range(4),yticklabels=CONDITIONS,title='Fixed Tiny + cached-latency interval rule; measured two-D scope');export(fig,'Gap')
    fig,ax=plt.subplots(figsize=(8,4));ax.bar(['Scored outcomes','False PASS','UNCERTAIN','Safe ABSTAIN'],[score['scored_pairs'],score['actual_FAIL_predicted_PASS'],score['UNCERTAIN'],2],color=['#4488aa','#bb5555','#d7af50','#888888'])
    ax.set(ylabel='P-D count',title='Frozen-rule transfer: ONE new P, TWO correlated checkpoints');export(fig,'controller_safety')
    fig,ax=plt.subplots(figsize=(8,4));ax.bar([r['stage'].replace('VM47_','') for r in resource['charges']],[r['charged_seconds']/60 for r in resource['charges']],color='#4488aa')
    ax.set(ylabel='Charged worker wall (minutes)',title='Includes CPU COCO/Bootstrap; CUDA busy time not measured');export(fig,'actual_worker_cost')


def reports(out,spec,cells,effects,cross,gaps,local,heldout,score,resource,complete):
    def md(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(str(x) for x in r)+' |' for r in rows]])
    fmt=lambda v:'NA' if v is None else f'{v:.3f}'
    main=[];endpoint=[];did_text='NOT_EVALUATED：完整四角点未完成。'
    if complete:
        for c in CONDITIONS:
            for d in spec['models']:
                m=cells[d['D'],c][0];a=next(r for r in m['per_area'] if r['size']=='all');g=next(r for r in gaps if r['D']==d['D'] and r['condition']==c)
                main.append([c,d['D'],fmt(a['map50']*100),fmt(a['map50_95']*100),fmt(g['Tiny_AP']*100),fmt(a['ar50_95']*100),fmt(g['cached_pipeline_mean_ms']),g['state']])
        selected=[r for r in effects if r['size'] in ('all','tiny') and r['category_id'] is None and r['metric']=='map50_95' and r['effect']=='Interaction_DID']
        for r in selected:endpoint.append([r['D'],r['size'],fmt(r['effect_pp']),f"[{r['CI_low']*100:.3f},{r['CI_high']*100:.3f}]"])
        did_text=md(['D','尺度','DID百分点','配对95%区间百分点'],endpoint)
        for size in ('all','tiny'):
            r=next(r for r in cross if r['size']==size and r['category_id'] is None and r['metric']=='map50_95')
            did_text+=f"\n\n跨checkpoint DID差（ck01−ck05），{size}：{r['effect_pp']:+.3f}pp，[{r['CI_low']*100:.3f},{r['CI_high']*100:.3f}]pp。"
    local_text=md(['P','已测候选范围','状态'],[[r['condition'],r['scope'],r['local_status']] for r in local]) if local else 'NOT_EVALUATED'
    rule_text=md(['D','冻结预测','实测状态','安全动作'],[[r['D'],r['predicted'],r['actual'],r['safe_action']] for r in heldout]) if heldout else 'NOT_EVALUATED'
    state='八格全部完成' if complete else '不完整，停止原因见监督器与失败日志；没有补齐或重试'
    thermal=md(['阶段','最高°C','最大设备占用MiB','最大采样间隔s'],[[r['phase'],r['max_C'],r['max_used_MiB'],fmt(r['max_record_interval_seconds'])] for r in resource['telemetry']])
    body=f'''# VM47 Gaussian×JPEG科学实验报告

RunID：`{out.name}`。{state}，已核验{len(cells)}/8格、{len(cells)*548}条逐图记录。本次是独立登记的较小实验，不是完成VM46的16格；其冻结文件保持不变。

## 冻结与真实执行

548张VisDrone validation原图，历史ck05 YOLOv8s与ck01 YOLO26s，1280/FP32/batch1/native10。原UVL的Gauss_s15 sigma15、JPEG_q25 quality25与Normal gain1/bias0；既有生成器按恒等光照→Gaussian→JPEG顺序，Gaussian单项及复合共享每图每特征噪声种子，主seed20261108来自历史执行协议。原图已为JPEG，Normal不是无损采集源；新增编码只做一次。不是标定相机噪声或真实JPEG链路等效实验。

原图GT、ignore-union和面积不变，Tiny<256px²。两个模型逐图使用同一PNG及像素SHA；原图、标签、PNG、checkpoint、预测、评价器SHA保留。NCHW为[1,3,1280,1280]，dense one-to-many、conf.001、NMS IoU.7、max_det3000/max_nms30000、TF32=False。Normal本轮重测，没有把历史缓存计时拼入四角点。

## 实测绝对指标

百分数仅用于展示，原值CSV/JSON为0–1。AR为AR50:95，时延为完整缓存管线ms。

{md(['条件','D','AP50','AP50:95','Tiny AP95','AR95','时延ms','固定判据'],main) if main else 'NOT_EVALUATED'}

完整类别/尺度AP和AR、逐格200次区间见per_class_metrics.csv及paired_bootstrap_intervals.csv；类别无GT保留null，valid_draws不足200的区间条件于有GT的重采样。低AP不等于数值异常或非法配置。

## Gaussian×JPEG交互

{did_text}

DID=联合−Gaussian−JPEG+Normal。200次同图位置重采样，每次重新累计汇总COCO AP，不平均单图AP。VM47沿用已验收默认整数rng.integers并保存实际共享索引，seed20261117；VM46的int32 CPU规划索引文件没有作为本次统计输入或产生旧AP，原件保留。原评价/Bootstrap算法没有修改。

正DID只说明AP尺度上的非加性，可能包含AP下界、损失饱和或压缩对噪声的作用，不能自动称为性能改善或物理因果。必须结合四格绝对AP。区间为点态95%，未做家族多重校正；548开发图有选模暴露且无可靠航次聚类，图像Bootstrap不是训练seed显著性。单个历史checkpoint的选中epoch及实际Albu激活仍UNKNOWN，跨D差不能归因为架构。

## 局部经验Gap

{local_text}

固定Tiny AP95≥.10、缓存均值≤100ms。沿用原区间规则：Tiny上界<.10或时延上界>100为FAIL；Tiny下界≥.10且时延上界≤100为PASS，其余UNCERTAIN。缺测为NOT_EVALUATED，不能冒充UNCERTAIN。仅讨论两个已测Infer1280候选，不估计全FM_P失败比例，不把AP变化等同Gap关闭。

缓存计时含完整性SHA、文件读取/解码、预处理、forward/NMS、框提取、预测和manifest写入flush；模型加载、预热、共享缓存生成、COCO/Bootstrap另记成本。固定先后次序与热/缓存状态限制纯时延因果解释。

## 冻结规则迁移与安全

原VM46冻结规则原样回放，没有重拟合或读取未来结果调规则。Normal、Gaussian、JPEG均为已测完整P，只将联合P的两格作为未见完整P诊断。

{rule_text}

{json.dumps(score,ensure_ascii=False)}

false-PASS分母为实际FAIL；PASS精确率分母为有确定真值的预测PASS；UNCERTAIN单列，零分母为NA。两检测器共享同一P与图像，不是两个独立环境样本。原始分类覆盖率与安全部署覆盖率分开：安全控制器仍ABSTAIN，0条规则获得部署许可。

实测查表可在本次固定P上比较已证实PASS的D，但使用完成后标签的查表不是留出预测泛化。固定两模型、精确查表及安全弃权逐P结果见Controller_comparison.csv。未来若要验证规则，须AP盲预指定更多未见完整P并涵盖足够PASS/FAIL，再单独满足预算与独立图像准入；不能因两格表现良好放行，也不自动运行VM46其余条件。

## 资源、保护与研究边界

实际GPU-worker计费{resource['GPU_worker_wall_charged_seconds']:.3f}秒，包含CPU COCO和200次Bootstrap，不等于CUDA kernel忙时。缓存CPU和后处理单列；功耗/风扇连续记录缺失，不估算能耗。余额{resource['budget_after']['remaining_seconds']:.3f}秒，原截止{resource['budget_after']['deadline_UTC']}未变。完整成本包络、真实设备/显存预检、所属CPU进程树终止测试和热监测记录保留。

{thermal}

90°C即停、87°C持续30秒停，监控目标1秒；无自动重试。沿用仍在有效期内的物理检查声明，当前供电与设备另查，并非新的人工作证。有限推理温度不能证明长训练稳定。原VM36–VM46、模型和UVL内容不改，只追加本Run实际GPU扣费；无训练或Git操作。

这是DEVELOPMENT_ONLY_SYNTHETIC_EXPLORATORY，不是独立测试集、自然天气、跨环境适应或训练收益证据。正式1280/100轮12臂和五项v8消融未由本次评价完成。

全部证据：{out}。真实执行命令见command.txt及preflight/formal_command.json，复核入口finish.py只读原预测并做CPU对比，不重新推理或重新计算已有COCO AP。
'''
    brief=f'''# VM47 导师短汇报

RunID：`{out.name}`。{state}，{len(cells)}/8格。

{did_text}

{local_text}

联合P为唯一未见完整配置，冻结规则的实际结果如下：

{rule_text}

只有1个P、2个相关检测器，无法支持可靠跨环境泛化；安全控制器保持ABSTAIN。正DID不是检测改善，须同时看绝对AP。本次为开发图合成探索，VM46未改、无训练。

实际计费{resource['GPU_worker_wall_charged_seconds']:.3f}秒，余额{resource['budget_after']['remaining_seconds']:.3f}秒，原期限不变。
'''
    rows=[{'RQ':'RQ1','state':'MEASURED_VALID_SUBSET' if complete else 'PARTIAL','evidence':'原UVL合法四条件、共享PNG/GT与checkpoint来源；非全空间可执行性证明'},
          {'RQ':'RQ2','state':'LOCAL_GAUSSIAN_JPEG_MEASURED' if complete else 'NOT_EVALUATED_COMPLETE','evidence':'四角点AP/Tiny DID和共享图像区间；唯一未见P仅两相关D，不是可靠规则泛化'},
          {'RQ':'RQ3','state':'LOCAL_TWO_D_GAP' if complete else 'PARTIAL','evidence':'固定Tiny/缓存时延三态与查表；无全空间Gap率或预测放行许可'},
          {'RQ':'RQ4','state':'FORMAL_STILL_INCOMPLETE','evidence':'未训练；1280/100epoch多seed、五消融与独立确认限制保持'}]
    table(out/'RQ_evidence_matrix.csv',rows)
    matrix='# VM47 RQ证据矩阵\n\nRunID：`'+out.name+'`。\n\n'+md(['RQ','状态','证据与限制'],[[r['RQ'],r['state'],r['evidence']] for r in rows])+'\n\n原始预测/GT、Bootstrap、时延及SHA优先于摘要图；本轮不声称独立图像验证、架构因果、Gap全域关闭或完整RQ4。\n'
    docs=[]
    for name,text in [('VM47_科学实验报告_CN.md',body),('VM47_导师短汇报_CN.md',brief),('VM47_RQ证据矩阵_CN.md',matrix)]:
        path=ROOT/'datasets/reportcn'/name
        if path.exists():path=path.with_name(path.stem+'_'+out.name.split('_')[-1]+path.suffix)
        with path.open('x',encoding='utf-8') as f:f.write(text)
        docs.append(path)
    return docs


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);finish(p.parse_args().run)
