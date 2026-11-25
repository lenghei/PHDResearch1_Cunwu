"""Source-linked analysis of the accepted CONTROL and ALL_FIVE cost runs."""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import sys

POST=Path(__file__).resolve().parent
sys.path.insert(0,str(POST.parent))
from study import *
from safety import available_seconds

CONTROL='vm37_yolo26_rq4_full_epoch_yolo26s_control_20261111T232439_301fb25d'
ALL_FIVE='vm37_yolo26_rq4_all_five_full_epoch_cost_20261112T200430_2ab983f6'
PREFIX='vm37_yolo26_rq4_all_five_short_cost_20261112T195422_06010323'


def lines(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            yield json.loads(line)


def execute():
    out=create('measured_cost_analysis_cpu');runtime(out,cuda=False)
    control=run_path(CONTROL);full=run_path(ALL_FIVE);prefix=run_path(PREFIX)
    for folder in (control,full):
        verify_hashes(folder,('training_acceptance.json','metrics.json','research_head_acceptance.json','protocol.json'))
        accepted=load(folder/'training_acceptance.json')
        assert accepted['completed_epochs']==1 and accepted['forward_batches']==accepted['backward_batches']==3236
        assert accepted['optimizer_updates']==accepted['EMA_updates']==784
        assert sha256(Path(accepted['checkpoint']['path']))==accepted['checkpoint']['sha256']
        assert load(folder/'research_head_acceptance.json')['COCO_replay_equal'] is True
        assert load(folder/'protected_inputs.json')['unchanged'] is True
    a=load(control/'training_acceptance.json');b=load(full/'training_acceptance.json')
    pa=load(control/'protocol.json');pb=load(full/'protocol.json')
    train_key=lambda r:(r['image_id'],r['image_SHA256'],r['derived_label_SHA256'])
    val_key=lambda r:(r['stem'],r['source_image_SHA256'],r['derived_label_SHA256'])
    matching={'initialization_SHA_equal':pa['initialization']['sha256']==pb['initialization']['sha256'],
        'training_membership_and_SHA_equal':sorted(map(train_key,pa['train']))==sorted(map(train_key,pb['train'])),
        'validation_membership_and_SHA_equal':sorted(map(val_key,pa['development']))==sorted(map(val_key,pb['development'])),
        'nominal_horizons':{'CONTROL':pa['args']['epochs'],'ALL_FIVE':pb['args']['epochs']},
        'both_stopped_after_epoch':1,'interpretation':'Cost references only; different runs and nominal horizons. No completed20-epoch matched pair.'}
    assert all(matching[key] for key in ('initialization_SHA_equal','training_membership_and_SHA_equal','validation_membership_and_SHA_equal'))
    rows=list(lines(full/'training_batches.jsonl'))
    assert len(rows)==3236 and all(row['healthy'] and row['epoch']==1 and not row['AMP'] for row in rows)
    anchors=[i for row in rows for i in row['image_ids']]
    assert len(anchors)==len(set(anchors))==6471
    assert all(row['NCHW'][1:]==[3,1280,1280] for row in rows)
    short=list(lines(prefix/'training_batches.jsonl'))
    fields=('image_ids','NCHW','loss','loss_items','optimizer_updates','EMA_updates','accumulation','AMP')
    prefix_trace_equal=all(all(x[k]==y[k] for k in fields) for x,y in zip(short,rows[:128]))
    events=list(lines(full/'health_events.jsonl'))
    updates=[r for r in events if r['event']=='optimizer_and_EMA_healthy']
    snapshots=[r for r in events if r['event']=='FP32_snapshot_reloaded_equal']
    assert len(updates)==784 and len(snapshots)==1 and snapshots[0]['SHA256']==b['checkpoint']['sha256']
    counts=Counter();native_calls=0
    for row in lines(full/'augmentation_events.jsonl'):
        native_calls+=1
        assert row['epoch']==0
        for effect in row['effects']:
            assert effect['probability']==.5
            counts[(effect['family'],'total')]+=1
            counts[(effect['family'],'active')]+=effect['active']
    activation={family:{'active':counts[(family,'active')],'total':counts[(family,'total')],
        'observed_fraction':counts[(family,'active')]/counts[(family,'total')]} for family in sorted({key[0] for key in counts})}
    raw_ids=[];raw_predictions=0
    with gzip.open(full/'predictions.jsonl.gz','rt',encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);raw_ids.append(row['image_id']);raw_predictions+=len(row['predictions'])
    metrics=load(full/'metrics.json')
    assert len(raw_ids)==len(set(raw_ids))==548 and raw_predictions==metrics['raw_count']
    ledger=load(RUNS/'compute_allocation.json')
    assert ledger['total_seconds']==172800 and ledger['absolute_deadline_UTC']=='2026-11-13T22:03:21.577924+00:00'
    pair20=20*(a['training_setup_epoch_internal_val_save_seconds']+b['training_setup_epoch_internal_val_save_seconds'])
    facts={'status':'MEASURED_COST_COMPLETE_PILOT_BLOCKED','UTC':utc().isoformat(),'CONTROL_RunID':CONTROL,
        'ALL_FIVE_RunID':ALL_FIVE,'PREFIX_RunID':PREFIX,'matching':matching,
        'CONTROL_training_scope_seconds':a['training_setup_epoch_internal_val_save_seconds'],
        'ALL_FIVE_training_scope_seconds':b['training_setup_epoch_internal_val_save_seconds'],
        'training_scope_ratio_observed':b['training_setup_epoch_internal_val_save_seconds']/a['training_setup_epoch_internal_val_save_seconds'],
        'CONTROL_loader_seconds':a['loader_seconds'],'ALL_FIVE_loader_seconds':b['loader_seconds'],
        'ALL_FIVE_forward_backward_health_seconds':b['forward_backward_health_seconds'],
        'ALL_FIVE_worker':load(full/'cost_result.json'),'short_gate':load(prefix/'final_result.json'),
        'prefix_128_trace_fields_equal':prefix_trace_equal,'not_a_weight_state_or_training_seed_replication':True,
        'actual_native_pixel_calls':native_calls,'family_activation':activation,
        'raw_predictions':raw_predictions,'raw_prediction_images':len(raw_ids),'GT_source':source(full/'ground_truth.json'),
        'checkpoint':b['checkpoint'],'initialization_SHA256':pb['initialization']['sha256'],
        'paired20_training_linear_scenario_seconds':pair20,'paired20_plus25percent_scenario_seconds':pair20*1.25,
        'current_available_seconds':available_seconds(ledger),'absolute_deadline_UTC':ledger['absolute_deadline_UTC'],
        'historical_plus_new_charged_seconds':sum(r['charged_seconds'] for r in ledger['completed_workers']),
        'pilot20':'PILOT_RESOURCE_BLOCKED','formal100':'RESOURCE_BLOCKED','completed20_pairs':0,
        'scenario_limit':'Constant first-epoch cost only. Warmup, close_mosaic and later epochs can change speed. Full16-config evaluation/bootstrap is additional and unmeasured.',
        'conclusion':'The complete pair cannot be admitted conservatively within the current window; do not start CONTROL alone.',
        'formal_augmentation_benefit':'NOT_EVALUATED','degraded_AP':'NOT_EVALUATED','paired_CI':'NOT_EVALUATED_COST_ONLY',
        'scope':'ENGINEERING_ONLY / DEVELOPMENT_ONLY','system_settings_or_dependencies_changed':False}
    save(out/'cost_analysis.json',facts,out)
    metric_rows=[]
    for arm,folder in (('CONTROL',control),('ALL_FIVE',full)):
        for row in load(folder/'metrics.json')['per_area']:
            metric_rows.append({'arm':arm,'RunID':folder.name,'size':row['size'],'AP50':row['map50'],
                'AP50_95':row['map50_95'],'AR50_95':row['ar50_95'],'GT_count':row['gt_count'],
                'source_metrics_SHA256':sha256(folder/'metrics.json'),'source_metrics':str(folder/'metrics.json'),
                'scope':'One-epoch cost checkpoint, not completed pilot results.'})
    with (out/'engineering_clean_metrics.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=metric_rows[0],lineterminator='\n');writer.writeheader();writer.writerows(metric_rows)
    save(out/'inputs.json',{name:source(folder/'SHA256SUMS.txt') for name,folder in
        (('CONTROL',control),('ALL_FIVE',full),('short_gate',prefix))},out)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,(axis,temp_axis)=plt.subplots(2,1,figsize=(10,7))
    for i,(name,m) in enumerate((('CONTROL',a),('ALL_FIVE',b))):
        loader=m['loader_seconds']/60;compute=m['forward_backward_health_seconds']/60
        other=m['training_setup_epoch_internal_val_save_seconds']/60-loader-compute
        axis.barh(i,loader,label='CPU loading/transforms' if i==0 else None,color='#4c78a8')
        axis.barh(i,compute,left=loader,label='Forward/backward/health' if i==0 else None,color='#f58518')
        axis.barh(i,other,left=loader+compute,label='Setup/validation/save residual' if i==0 else None,color='#b9bec5')
    axis.set_yticks([0,1],['CONTROL','ALL_FIVE']);axis.set_xlabel('Measured training-scope wall time (min)')
    axis.legend(fontsize=8);axis.grid(axis='x',alpha=.2)
    telemetry=list(lines(full/'gpu_telemetry.jsonl'))
    temp_axis.plot([r['monotonic_seconds']/60 for r in telemetry],[r['temperature_C'] for r in telemetry],lw=.8)
    temp_axis.axhline(90,color='red',ls='--',label='Immediate stop90C');temp_axis.axhline(87,color='orange',ls=':',label='Sustained stop87C/30s')
    temp_axis.set(xlabel='ALL_FIVE worker elapsed time (min)',ylabel='Sampled GPU temperature (C)')
    temp_axis.legend(fontsize=8);temp_axis.grid(alpha=.2)
    fig.suptitle('One-epoch engineering measurements; not full training or augmentation efficacy')
    fig.tight_layout();fig.savefig(out/'cost_and_temperature.png',dpi=150);plt.close(fig)
    write_atomic(out/'diary.md','# Measured ALL_FIVE cost\n\nVerified actual epoch counts, healthy snapshot, raw prediction membership and independent COCO replay. Compared measured cost scopes only. Kept lower engineering CLEAN values and blocked the20-epoch pair; no augmentation-benefit or independent-generalization claim.\n',out)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><title>VM37 measured cost</title>'
        '<style>body{font:16px system-ui;max-width:1050px;margin:40px auto}img{max-width:100%}</style>'
        '<h1>VM37 measured ALL_FIVE cost</h1><p>One epoch accepted. Complete20/100-epoch paired training remains blocked.</p>'
        '<img src="cost_and_temperature.png"><p><a href="cost_analysis.json">Cost and resource evidence</a></p>'
        '<p><a href="engineering_clean_metrics.csv">Engineering CLEAN metrics and sources</a></p>',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'MEASURED_COST_ANALYSIS_CPU',
        'status':facts['status'],'GPU_seconds':0},out,replace=True)
    seal(out)
    from main.visualization import build_run_index
    build_run_index(RUNS,EXPERIMENT)
    print(out,flush=True)


if __name__=='__main__':execute()
