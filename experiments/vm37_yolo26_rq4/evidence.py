"""Export measured engineering results without altering source runs."""
import csv
import json
import time
from study import *


def table(path,rows):
    if not rows:return
    with path.open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def collect():
    candidates=sorted(p for p in RUNS.glob(EXPERIMENT+'_*') if p.is_dir() and (p/'SHA256SUMS.txt').exists())
    recoveries={}
    for folder in candidates:
        if (folder/'worker_result.json').exists():
            value=load(folder/'worker_result.json')
            if value.get('training_RunID') and value['status']=='PASS':recoveries[value['training_RunID']]=folder
    measurements=[];metrics=[];classes=[];provenance=[]
    for folder in candidates:
        if not (folder/'training_acceptance.json').exists():continue
        verify_hashes(folder,('training_acceptance.json','protocol.json','health_events.jsonl'))
        result=load(folder/'training_acceptance.json');plan=load(folder/'protocol.json')
        head_folder=folder if (folder/'research_head_acceptance.json').exists() else recoveries.get(folder.name)
        health=[json.loads(line) for line in (folder/'health_events.jsonl').read_text().splitlines()]
        batches=[json.loads(line) for line in (folder/'training_batches.jsonl').read_text().splitlines()]
        saves=[r for r in health if r['event']=='FP32_snapshot_reloaded_equal']
        telemetry=[json.loads(line) for line in (folder/'gpu_telemetry.jsonl').read_text().splitlines()] if (folder/'gpu_telemetry.jsonl').exists() else []
        supervisor=load(folder/'supervisor.json')
        record={'RunID':folder.name,'architecture':plan['architecture'],'arm':plan.get('arm','CONTROL'),
            'scope':plan['scope'],'train_images':result['training_images'],'development_images':result['validation_images'],
            'completed_epochs':result['completed_epochs'],'forward_batches':result['forward_batches'],
            'backward_batches':result['backward_batches'],'optimizer_updates':result['optimizer_updates'],'EMA_updates':result['EMA_updates'],
            'training_setup_epoch_validation_save_seconds':result['training_setup_epoch_internal_val_save_seconds'],
            'worker_wall_seconds':supervisor.get('wall_seconds',supervisor.get('seconds')),
            'loader_seconds':result['loader_seconds'],'batch_compute_and_health_seconds':result['forward_backward_health_seconds'],
            'forward_loss_wall_seconds':result.get('forward_loss_wall_seconds'),'backward_wall_seconds':result.get('backward_wall_seconds'),
            'internal_validation_seconds':sum(r['seconds'] for r in result['internal_validation']),
            'save_reload_seconds':sum(r['save_reload_seconds'] for r in saves),
            'peak_allocated_bytes':result['GPU_peak_bytes']['allocated'],'peak_reserved_bytes':result['GPU_peak_bytes']['reserved'],
            'NVIDIA_max_sampled_used_MiB':max((r['used_MiB'] for r in telemetry),default=None),
            'maximum_temperature_C':max((r['temperature_C'] for r in telemetry),default=None),
            'healthy_update_events':sum(r['event']=='optimizer_and_EMA_healthy' for r in health),
            'FP32_reload_events':len(saves),'all_recorded_batches_healthy':all(r['healthy'] for r in batches),
            'initialization_SHA256':plan['initialization']['sha256'],'checkpoint_SHA256':result['checkpoint']['sha256'],
            'training_acceptance_SHA256':sha256(folder/'training_acceptance.json'),
            'training_Run_status':load(folder/'run.json')['status'],'research_RunID':head_folder.name if head_folder else None,
            'research_images':None,'research_wall_seconds':None,'research_COCO_seconds':None,'raw_replay_seconds':None,
            'mean_predict_call_ms_excludes_decode_COCO':None}
        measurements.append(record)
        if head_folder is None:continue
        verify_hashes(head_folder,('metrics.json','research_head_acceptance.json','predictions.jsonl.gz','ground_truth.json'))
        head=load(head_folder/'research_head_acceptance.json');value=load(head_folder/'metrics.json')
        if head['status']!='PASS' or not head['COCO_replay_equal']:raise ValueError('Unaccepted research metrics.')
        record.update(research_images=head['images'],research_wall_seconds=head['wall_seconds'],
                      research_COCO_seconds=head['COCO_seconds'],raw_replay_seconds=head['replay_seconds'],
                      mean_predict_call_ms_excludes_decode_COCO=sum(r['inference_ms'] for r in head['actual_inputs'])/head['images'])
        metric_sha=sha256(head_folder/'metrics.json');gt_sha=sha256(head_folder/'ground_truth.json')
        prediction_sha=sha256(head_folder/'predictions.jsonl.gz')
        for row in value['per_area']:
            metrics.append({'training_RunID':folder.name,'evaluation_RunID':head_folder.name,'scope':plan['scope'],
                'size':row['size'],'AP50':row['map50'],'AP50_95':row['map50_95'],'AR50_95':row['ar50_95'],
                'GT_count':row['gt_count'],'images':row['image_count'],'raw_predictions':value['raw_count'],
                'ignore_filtered_predictions':value['mapped_count'],'metrics_SHA256':metric_sha})
            for name in ('map50','map50_95','ar50_95'):
                provenance.append({'training_RunID':folder.name,'evaluation_RunID':head_folder.name,'metric':name,
                    'size':row['size'],'value':row[name],'source_file':head_folder.name+'/metrics.json',
                    'source_SHA256':metric_sha,'GT_SHA256':gt_sha,
                    'predictions_SHA256':prediction_sha,'checkpoint_SHA256':result['checkpoint']['sha256']})
        for row in value['per_class']:
            classes.append({'training_RunID':folder.name,'evaluation_RunID':head_folder.name,'size':row['size'],
                'class':row['category_name'],'AP50':row['map50'],'AP50_95':row['map50_95'],'AR50_95':row['ar50_95'],
                'GT_count':row['gt_count'],'metrics_SHA256':metric_sha})
    return measurements,metrics,classes,provenance


def figures(out,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    full=[r for r in rows if r['train_images']==6471]
    if not full:return
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    colors=['#456990','#E09F3E','#6B8F71','#9B7E9B','#9D9D9D']
    labels=[]
    for i,row in enumerate(full):
        label=('Original' if i==0 else 'Grouped checks')+' / '+row['arm'];labels.append(label)
        components=[row['loader_seconds'],row['batch_compute_and_health_seconds'],row['internal_validation_seconds'],row['save_reload_seconds']]
        components.append(row['training_setup_epoch_validation_save_seconds']-sum(components))
        if components[-1]<0:raise ValueError('Timing components exceed their containing interval.')
        bottom=0.
        for j,value in enumerate(components):
            axes[0].bar(i,value/60,bottom=bottom,color=colors[j],label=['Loader','Forward/backward + health','Internal validation','FP32 save/reload','Setup/other (residual)'][j] if i==0 else None)
            bottom+=value/60
        folder=run_path(row['RunID']);batches=[json.loads(line) for line in (folder/'training_batches.jsonl').read_text().splitlines()]
        exposures=[];loss=[];seen=0
        for batch in batches:seen+=len(batch['image_ids']);exposures.append(seen);loss.append(batch['loss'])
        window=50
        smooth=[sum(loss[max(0,j-window+1):j+1])/len(loss[max(0,j-window+1):j+1]) for j in range(len(loss))]
        axes[1].plot(exposures,smooth,label=label,linewidth=2 if i==0 else 1,linestyle='-' if i==0 else '--')
    axes[0].set_xticks(range(len(labels)),labels,rotation=15,ha='right');axes[0].set_ylabel('Measured minutes');axes[0].legend(fontsize=7)
    axes[1].set_xlabel('Training anchor-image exposures');axes[1].set_ylabel('Loss (trailing 50 batches)');axes[1].legend(fontsize=7)
    fig.suptitle('Complete first-epoch cost studies; not full matched training')
    for ext in ('png','pdf'):fig.savefig(out/('epoch_cost_and_loss.'+ext),dpi=180)
    plt.close(fig)


def execute():
    started=time.perf_counter();out=create('evidence_summary_cpu');runtime(out,cuda=False)
    rows,metrics,classes,provenance=collect()
    table(out/'training_measurements.csv',rows);table(out/'research_metrics.csv',metrics)
    table(out/'per_class_metrics.csv',classes);table(out/'results_provenance.csv',provenance)
    figures(out,rows)
    save(out/'scientific_status.json',{'full100_arms_completed':0,'matched_scientific_pairs':0,
        'augmentation_effectiveness':'NOT_EVALUATED','TARGETED_minus_CONTROL':'NOT_EVALUATED',
        'paired_uncertainty':'NOT_EVALUATED_NO_COMPLETED_FORMAL_PAIR','empirical_gap_change':'NOT_EVALUATED',
        'test_dev':'DATASET_BLOCKED','UAVDT':'BLOCKED_ANNOTATION_PROVENANCE','scope':'DEVELOPMENT_ENGINEERING_AND_COST_ONLY'},out)
    write_atomic(out/'diary.md','# Measured evidence summary\n\nRead sealed VM37 results and exported traceable training costs, research AP/AR and object-size/class metrics. No training, inference or COCO recalculation.\n',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'status':'COMPLETE_WITH_LIMITATIONS',
        'CPU_seconds':time.perf_counter()-started,'GPU_seconds':0},out,replace=True)
    seal(out);print(out,flush=True)
    return out


if __name__=='__main__':execute()
