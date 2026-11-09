"""Summarize only sealed full548 pairs and matched bootstrap contrasts."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from datetime import datetime,timezone
import numpy as np
from protocol import ROOT,HERE,EXPERIMENT,load,save,source,sha256,write_atomic,FMP,FMD
from common import seal,verify_hashes
from historical_review import csv_file
from full_analysis import matched_contrasts,tested_subset_state,checkpoint_contrasts


def finish(out):
    began=time.perf_counter()
    if (out/'SHA256SUMS.txt').exists():
        verify_hashes(out,('RESULT_SUMMARY.json','summary.json','result.json'))
        print('Existing sealed summary verified:',out,flush=True)
        return
    terminal=load(out/'run.json')
    if terminal.get('status')=='RUNNING':raise ValueError('Worker has not stopped.')
    plan=load(out/'EXECUTION_PROTOCOL.json');rows=[];accepted={};latencies={};classrows=[];tensor_rows=[]
    if sha256(out/'EXECUTION_PROTOCOL.json')!=(out/'PROTOCOL_SHA256.txt').read_text().split()[0]:raise ValueError('Protocol changed.')
    for spec in plan['pairs']:
        folder=out/'pairs'/spec['id']
        if not (folder/'SHA256SUMS.txt').is_file():continue
        verify_hashes(folder,('metrics.json','bootstrap.json','verification.json','configuration.json','timing.json','predictions.jsonl.gz','ground_truth.json'))
        metric=load(folder/'metrics.json');assert metric['images']==548
        timing=load(folder/'timing.json');boot=load(folder/'bootstrap.json')
        expected_ids=sorted(im['id'] for im in load(folder/'ground_truth.json')['coco']['images'])
        if boot['draws']['image_ids']!=expected_ids or boot['draws']['samples']!=plan['draws'] or boot['draws']['seed']!=plan['seed']:
            raise ValueError('Unexpected bootstrap identity.')
        if load(folder/'verification.json')['protocol_sha256']!=sha256(out/'EXECUTION_PROTOCOL.json'):
            raise ValueError('Pair protocol identity differs.')
        quality=int(next(x for x in spec['p']['leaves'] if x.startswith('JPEG')).split('q')[1])
        key=spec['d']['checkpoint'],spec['d']['resolution'],quality
        accepted[key]=(spec,metric,boot);latencies[key]=timing
        for row in metric['per_class']:classrows.append({'pair_id':spec['id'],'checkpoint':key[0],
            'resolution':key[1],'JPEG':quality,**row,'metrics_SHA256':sha256(folder/'metrics.json')})
        manifest=[json.loads(line) for line in (folder/'image_manifest.jsonl').read_text(encoding='utf-8').splitlines()]
        if len(manifest)!=548 or sorted(r['image_id'] for r in manifest)!=expected_ids:raise ValueError('Incomplete inference membership.')
        if any(r['tensor']!={'NCHW':[1,3,key[1],key[1]],'dtype':'torch.float32'} for r in manifest):raise ValueError('Unexpected network input.')
        tensor_rows.append({'pair_id':spec['id'],'images':548,'NCHW':[1,3,key[1],key[1]],'dtype':'torch.float32',
             'manifest_SHA256':sha256(folder/'image_manifest.jsonl')})
        for area in metric['per_area']:
            ci=next(c for c in boot['intervals'] if c['size']==area['size'] and c['metric']=='map50_95')
            rows.append({'pair_id':spec['id'],'checkpoint':key[0],'resolution':key[1],'JPEG':quality,'images':548,
                 **area,'AP_CI_low':ci['ci95'][0],'AP_CI_high':ci['ci95'][1],
                 'mean_cached_pipeline_ms':timing['mean_pipeline_ms'],'peak_allocated_bytes':timing['peak_allocated_bytes'],
                 'metrics_SHA256':sha256(folder/'metrics.json'),'raw_SHA256':sha256(folder/'predictions.jsonl.gz')})
    csv_file(out/'CONFIGURATION_METRICS.csv',rows) if rows else write_atomic(out/'CONFIGURATION_METRICS.csv','pair_id,checkpoint,images,AP50,AP50_95\n',out)
    if classrows:csv_file(out/'PER_CLASS_METRICS.csv',classrows)
    csv_file(out/'TENSOR_VERIFICATION.csv',tensor_rows)
    effects=matched_contrasts(accepted)
    if effects:csv_file(out/'INTERACTION_EFFECTS.csv',effects)
    cross_effects=checkpoint_contrasts(accepted)
    if cross_effects:csv_file(out/'CHECKPOINT_INTERACTIONS.csv',cross_effects)
    states=[]
    from gap_analysis import pair_outcome
    for key,(spec,metric,boot) in accepted.items():
        timing=latencies[key];values=np.array([x['pipeline_ms'] for x in timing['samples']]);ix=np.random.default_rng(plan['seed']).integers(0,548,(200,548))
        latency=np.quantile(values[ix].mean(axis=1),[.025,.975]).tolist()
        for tau in (.05,.10,.15):
            for L in (50,100):
                ci=next(c['ci95'] for c in boot['intervals'] if c['size']=='tiny' and c['metric']=='map50_95')
                states.append({'pair_id':spec['id'],'JPEG':key[2],'tau':tau,'L_cached_ms':L,
                    'state':pair_outcome(ci,latency[1],tau,L),'AP_CI':ci,'latency_mean_CI':latency,
                    'scope':'SENSITIVITY_ONLY_NOT_DEPLOYMENT_ADMISSION'})
    if states:csv_file(out/'THRESHOLD_SENSITIVITY.csv',states)
    from full_development import PREP
    static=__import__('historical_review').table(PREP/'STATIC_COVERAGE_REFERENCE.csv')
    gaps=[]
    for quality in (25,75):
        for tau in (.05,.10,.15):
            for L in (50,100):
                subset=[r for r in states if r['JPEG']==quality and r['tau']==tau and r['L_cached_ms']==L]
                covered={}
                for policy in ('exact_identity','conservative','supervisor_proxy'):
                    rules=[r for r in static if r['policy']==policy and r['checkpoint'] in ('ck05','ck01')]
                    covered[policy]=any({'Normal','JPEG_q'+str(quality)}.issubset(set(r['covered_leaves'].split('|'))) for r in rules)
                gaps.append({'JPEG':quality,'tau':tau,'L_cached_ms':L,'measured_detector_settings':len(subset),
                    'declared_detector_settings':4,'empirical_state':tested_subset_state([r['state'] for r in subset],4),
                    **{p+'_static_covered':v for p,v in covered.items()},
                    'scope':'SENSITIVITY_ONLY; not a uniform population estimate or deployment admission'})
    csv_file(out/'SUBSET_GAP_SENSITIVITY.csv',gaps)
    from coverage_evidence import export
    save(out/'PAIRWISE_CORE_ACCEPTANCE.json',export(out,PREP),out)
    protected={}
    for key in ('FMP','FMD','GT','image_members','ZIP','generator'):
        entry=plan[key]
        if sha256(Path(entry['path']))!=entry['sha256']:raise ValueError('Original input changed: '+key)
        protected[key]=entry
    for spec in plan['pairs']:
        d=spec['d']
        if sha256(Path(d['path']))!=d['sha256']:raise ValueError('Original checkpoint changed.')
        protected[d['checkpoint']]={'path':d['path'],'sha256':d['sha256']}
    expected_sources=load(out/'source_SHA256.json')
    worker_path=Path(__file__).with_name('full_development.py')
    if sha256(worker_path)!=expected_sources[str(worker_path.relative_to(ROOT/'main')).replace('\\','/')]:raise ValueError('Executed worker source changed.')
    save(out/'PRESERVED_INPUTS.json',{'status':'VALIDATED','inputs':protected,
        'scope':'Exact referenced files and checkpoint identities; not a whole-archive scan.'},out)
    pending=[spec['id'] for spec in plan['pairs'] if not (out/'pairs'/spec['id']/'SHA256SUMS.txt').exists()]
    save(out/'RECOVERY_STATE.json',{'completed_verified_pairs':[spec['id'] for spec,_,_ in accepted.values()],
        'pending_pairs':pending,'resume_unit':'Verified complete configuration; no incomplete raw stream is silently reused.',
        'CPU_metric_regeneration':'AVAILABLE','cross_reboot_GPU_resume':'NOT_VALIDATED',
        'original_budget_reset_allowed':False,'automatic_restart':False,
        'next_requirement':'A separately recorded resource decision is required for pending GPU work.'},out)
    status='VALIDATED' if len(accepted)==8 else 'PARTIAL'
    report={'RunID':out.name,'ExperimentID':EXPERIMENT,'status':status,'mode':'FULL548_R_JPEG',
       'scope':'DEVELOPMENT_ONLY','completed_pairs':len(accepted),'planned_pairs':8,'image_inferences':len(accepted)*548,
       'formal_pairs':0,'CLEAN':'NOT_EVALUATED','motion':'NOT_EVALUATED_UNSPECIFIED_ORIENTATION',
       'RQ1':'PARTIAL_PHYSICAL_JUSTIFICATION','RQ2':'DESCRIPTIVE_MATCHED_INTERACTIONS; ML_INSUFFICIENT_EVIDENCE',
       'RQ3':'TESTED_SUBSET_THRESHOLD_SENSITIVITY; NO_POPULATION_ESTIMATE','RQ4':'RESOURCE_BLOCKED; NOT_EVALUATED',
       'intervals':'200 paired image percentile draws; pointwise only; duplicate bytes and scenes are not independent flights.',
       'new_training_runs':0,'fullspace_estimate':'NOT_EVALUATED','legacy_500_900':'Historical examples, not required targets.',
       'pending_pairs':pending,'cross_checkpoint_resolution_JPEG':'NOT_ESTIMABLE' if pending else 'AVAILABLE',
       'checkpoint_JPEG_640':'POST_HOC_DESCRIPTIVE' if cross_effects else 'NOT_ESTIMABLE'}
    save(out/'RESULT_SUMMARY.json',report,out)
    worker=load(out/'worker_result.json') if (out/'worker_result.json').is_file() else {'status':'INTERRUPTED','worker_wall_seconds':None}
    timing_summary={'worker_wall_seconds':worker.get('worker_wall_seconds'),
        'cache_wall_seconds':load(out/'cache_cost.json')['wall_seconds'],
        'pair_wall_seconds':sum(t['wall_seconds'] for t in latencies.values()),
        'evaluation_seconds':sum(t['evaluation_seconds'] for t in latencies.values()),
        'replay_seconds':sum(t['replay_seconds'] for t in latencies.values()),
        'bootstrap_seconds':sum(t['bootstrap_seconds'] for t in latencies.values()),
        'forward_profiler_seconds':sum(r['forward_ms'] for t in latencies.values() for r in t['samples'])/1000,
        'peak_allocated_bytes':max((t['peak_allocated_bytes'] for t in latencies.values()),default=None),
        'peak_reserved_bytes':max((t['peak_reserved_bytes'] for t in latencies.values()),default=None),
        'definition':'Nested components; do not add them to worker wall time. Cached-input timing is not deployment latency.'}
    save(out/'TIMING_SUMMARY.json',timing_summary,out)
    summary={'schema':'VM36_READ_ONLY_SUMMARY_V1','RunID':out.name,'ExperimentID':EXPERIMENT,
        'status':status,'mode':'FULL548_R_JPEG','scope':'DEVELOPMENT_ONLY','measured_study':True,
        'validated_validation_images':548,'new_measured_pairs':len(accepted),'formal_measured_pairs':0,
        'worker_wall_seconds':worker.get('worker_wall_seconds'),
        'forward_profiler_seconds':sum(r['forward_ms'] for t in latencies.values() for r in t['samples'])/1000,
        'peak_allocated_bytes':max((t['peak_allocated_bytes'] for t in latencies.values()),default=None),
        'metrics':[r for r in rows if r['size']=='all'],'effects':effects,'checkpoint_effects':cross_effects,'threshold_sensitivity':gaps,
        'source_hashes':{key:plan[key] for key in ('FMP','FMD','GT','image_members','ZIP','generator')},
        'gates':[{'branch':k,'status':v} for k,v in {'RQ2':'MATCHED_EFFECTS_ONLY; ML_INSUFFICIENT_EVIDENCE',
            'RQ3':'SUBSET_SENSITIVITY_ONLY','RQ4':'RESOURCE_BLOCKED','Independent test-dev':'DATASET_BLOCKED',
            'Motion':'UNSPECIFIED_ORIENTATION; NOT_EVALUATED','CLEAN':'NOT_EVALUATED'}.items()]}
    save(out/'summary.json',summary,out)
    save(out/'result.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':summary['mode'],'status':status,
        'summary_canonical_sha256':hashlib.sha256(json.dumps(summary,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()},out)
    allrows=[r for r in rows if r['size']=='all']
    if allrows:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(9,5))
        labels={'ck05':'YOLOv8s','ck01':'YOLO26s'}
        for ck in ('ck05','ck01'):
            for q,style in ((25,'--'),(75,'-')):
                selected=sorted([r for r in allrows if r['checkpoint']==ck and r['JPEG']==q],key=lambda r:r['resolution'])
                if selected:
                    line,=ax.plot([r['resolution'] for r in selected],[100*r['map50_95'] for r in selected],style,marker='o',label=labels[ck]+' JPEG Q'+str(q))
                    ax.vlines([r['resolution'] for r in selected],[100*r['AP_CI_low'] for r in selected],
                        [100*r['AP_CI_high'] for r in selected],color=line.get_color(),alpha=.6)
        ax.set(xlabel='Inference resolution',ylabel='COCO AP50:95 (%)',title='Full548 validation · fixed-checkpoint resolution x JPEG')
        ax.legend();fig.text(.1,.01,'548 images per point; 200 paired image draws; pointwise 95% intervals. Development only.',fontsize=8)
        fig.tight_layout(rect=(0,.03,1,1));fig.savefig(out/'resolution_jpeg.png',dpi=180);fig.savefig(out/'resolution_jpeg.svg');plt.close(fig)
    if effects:
        primary=[r for r in effects if r['contrast'].startswith('(')]
        if primary:
            fig,ax=plt.subplots(figsize=(8,4))
            for i,r in enumerate(primary):
                ax.plot(100*r['estimate'],i,'o',color='tab:blue')
                ax.hlines(i,100*r['CI95_low'],100*r['CI95_high'],color='tab:blue')
            ax.axvline(0,color='grey',lw=1);ax.set_yticks(range(len(primary)),[r['checkpoint']+' / '+r['size'] for r in primary])
            ax.set(xlabel='Resolution x JPEG DID (AP percentage points)',title='(1280−640 at Q25) − (1280−640 at Q75)')
            fig.text(.1,.01,'Same 548 image positions; fixed weights; pointwise 95% image-bootstrap intervals. Development only.',fontsize=8)
            fig.tight_layout(rect=(0,.04,1,1));fig.savefig(out/'interaction_intervals.png',dpi=180);fig.savefig(out/'interaction_intervals.svg');plt.close(fig)
    text=['# VM36 全量开发实验','', 'EXPERIMENT ID: '+EXPERIMENT,'RESEARCH QUESTION: RQ2：分辨率效应是否随JPEG强度变化。',
       'DATASET: VisDrone2019-DET','DATASET SPLIT: validation（开发用途，曾参与选模）','NUMBER OF IMAGES: 每对548张，未删图。',
       f'NUMBER OF CONFIGURATIONS: 已验收{len(accepted)}/8对。','NUMBER OF DETECTOR CHECKPOINTS: YOLOv8s、YOLO26s两个原主要家族。',
       'MODEL ARCHITECTURE: ck05 YOLOv8s；ck01 YOLO26s。','TRAINING RESOLUTION: 历史FT1280。',
       'INFERENCE RESOLUTION: FM_D的Infer640/Infer1280属性。','TRAINING SEEDS: 历史seed42；未做训练seed重复。',
       'DEGRADATION PARAMETERS: 直接读取导师对齐FM_P：Normal；JPEG_q25/q75；其他退化关闭。',
       'SAMPLING METHOD: 两因素2×2完整角点，每checkpoint4对。','WHY THESE CONFIGURATIONS: 一个差中之差需要四角点；两个主要家族合计8对。Q50不属于端点交互的必要点，不能据此分析曲率。',
       'CONTROLLED VARIABLES: 同一548图、GT、同权重、FP32、原NMS与ignore-union；同一退化像素跨分辨率复用。',
       'EVALUATION METRICS: 真正COCO AP50/AP50:95、AR、尺寸与逐类AP；原预测重算核对。',
       'STATISTICAL METHOD: 相同seed和索引的200次配对图像bootstrap；点态区间，不推断训练seed或独立航次。',
       f'ACTUAL EXECUTION TIME: worker实测{worker.get("worker_wall_seconds")}秒，上限3600秒；CPU评价、复算和区间均在预算内，见TIMING_SUMMARY.json。嵌套阶段不能相加冒充GPU占用。',
       'ACTUAL RESULTS:','', '| 模型 | 输入 | JPEG | AP50 | AP50:95 | 95% CI |','|---|---:|---:|---:|---:|---|']
    for r in allrows:text.append(f'|{r["checkpoint"]}|{r["resolution"]}|{r["JPEG"]}|{r["map50"]:.6f}|{r["map50_95"]:.6f}|[{r["AP_CI_low"]:.6f},{r["AP_CI_high"]:.6f}]|')
    text+=['','配对差值（AP原值；乘100为百分点）：','',
        '| 模型 | 尺寸 | 对比 | 差值 | 95% CI |','|---|---|---|---:|---|']
    for r in effects:text.append(f'|{r["checkpoint"]}|{r["size"]}|{r["contrast"]}|{r["estimate"]:.6f}|[{r["CI95_low"]:.6f},{r["CI95_high"]:.6f}]|')
    text+=['','CONFIDENCE INTERVALS: INTERACTION_EFFECTS.csv保存配对差值和差中之差区间。',
       'SUPPORTED CONCLUSION: 仅支持本548图开发总体上、固定两个checkpoint与两个JPEG端点的条件效应。正负结果均保留。',
       'LIMITATIONS: 非独立test-dev；仅单训练seed；未测CLEAN、motion或全空间gap；样本量精度表仅列备选epsilon，未任意选定总体估计精度。',
       f'预算内未执行：{len(pending)}组；未计算缺角点交互。跨模型640下的JPEG效应对比见CHECKPOINT_INTERACTIONS.csv，属于事后描述分析，不归因于模型架构。',
       '完整接续限制：已有封存组可复核并引用；跨重启GPU接续未实测。不能重置本轮预算。',
       'RAW EVIDENCE PATH: '+str(out),'REPRODUCTION COMMAND: full_results.py --run '+out.name,
       'ACCEPTANCE STATUS: '+status]
    report_path=ROOT/'datasets/reportcn'/('vm36_supervisor_reproduction_全量JPEG实验_'+out.name.split('_')[-1]+'.md')
    write_atomic(report_path,'\n'.join(text)+'\n',out)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>VM36 full548 development</h1><p>'+status+' · '+str(len(accepted))+'/8 pairs · no formal independent evaluation</p><a href="CONFIGURATION_METRICS.csv">Metrics</a> · <a href="INTERACTION_EFFECTS.csv">Matched effects</a><p><img width="900" src="resolution_jpeg.png"></p>',out)
    save(out/'FIGURE_PROVENANCE.json',{'source':source(out/'CONFIGURATION_METRICS.csv'),'script':source(Path(__file__)),'scope':'DEVELOPMENT_ONLY'},out)
    save(out/'ANALYSIS_SOURCE_SHA256.json',{str(p.relative_to(ROOT/'main')).replace('\\','/'):sha256(p)
        for p in sorted(HERE.rglob('*.py'))},out)
    write_atomic(out/'diary.md','# Full548 development\n\n'+f'{len(accepted)}/8 pairs verified; {len(accepted)*548} image-inference operations. Worker: {worker.get("worker_wall_seconds")} s.\n\n'
        +'Original JPEG Q25/Q75 and Infer640/Infer1280 attributes; 200 paired image draws. No formal test, new training or missing-corner imputation.\n\n'
        +'The budget gate preserved completed pairs and did not launch pending configurations. RQ2 learning, population gap estimation and matched augmentation remain unaccepted.\n',out)
    save(out/'REPORT_REFERENCE.json',source(report_path),out)
    save(out/'ANALYSIS_TIMING.json',{'wall_seconds':time.perf_counter()-began,'GPU_experiments':0,
        'scope':'CPU evidence verification, aggregation and figures; separate from the stopped inference worker.'},out)
    save(out/'EVIDENCE_TREE.json',{str(p.relative_to(out)).replace('\\','/'):sha256(p)
        for p in sorted((out/'pairs').rglob('*')) if p.is_file()},out)
    seal(out)
    history=[]
    for folder in sorted(out.parent.iterdir(),reverse=True):
        if not folder.is_dir() or not (folder/'run.json').is_file():continue
        record=load(folder/'run.json');target='index.html' if (folder/'index.html').exists() else 'run.json'
        history.append((folder.name,record.get('status','UNKNOWN'),target))
    import re
    history.sort(key=lambda r:re.search(r'\d{8}T\d{6}',r[0]).group(0) if re.search(r'\d{8}T\d{6}',r[0]) else '',reverse=True)
    write_atomic(out.parent/'index.html','<!doctype html><meta charset="utf-8"><h1>VM36 evidence history</h1><p>Latest: '+str(len(accepted))+'/8 full548 development pairs; formal evaluation remains blocked.</p><ul>'
        +''.join(f'<li><a href="{name}/{target}">{name}</a> · {state}</li>' for name,state,target in history)+'</ul>',out,replace=True)
    save(out.parent/'latest.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'FULL548_R_JPEG',
        'status':status,'new_measured_pairs':len(accepted),'formal_measured_pairs':0},out,replace=True)
    print(report_path,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    if Path(args.run).name!=args.run:raise ValueError('Expected plain RunID.')
    finish(ROOT/'datasets/local_runs'/EXPERIMENT/args.run)
