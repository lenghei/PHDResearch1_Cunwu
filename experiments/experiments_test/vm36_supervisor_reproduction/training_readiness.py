"""Source-backed matched-training cost screening; no training is started."""
from pathlib import Path
import time
from protocol import ROOT,EXPERIMENT,load,save,source,write_atomic
from full_development import PREP
from historical_review import table,csv_file
from common import seal,verify_hashes
from main.pipeline import new_run


def execute():
    start=time.perf_counter();out,_=new_run({'output_root':str(ROOT/'datasets/local_runs'/EXPERIMENT)},EXPERIMENT,'training_readiness')
    verify_hashes(PREP,('CHECKPOINT_ADMISSION.csv',))
    catalog=table(ROOT/'datasets/research_archive/migration_20261104/model_catalog.csv');rows=[]
    for ck in ('ck01','ck05'):
        model=next(r for r in catalog if r['model_id'].startswith(ck+'_'))
        curve=ROOT/model['train_curve'];history=table(curve);last={k.strip():v for k,v in history[-1].items()}
        seconds=float(last['time']);epochs=int(float(last['epoch']))
        rows.append({'checkpoint':ck,'architecture':'YOLO26s' if ck=='ck01' else 'YOLOv8s',
            'reported_final_epoch':epochs,'historical_curve_seconds':seconds,'source_curve':source(curve),
            'source_args':source(ROOT/model['training_config']),
            'matched_primary_runs':6,'historical_time_multiple_primary_seconds':6*seconds,
            'seed_matched_ablation_runs':15 if ck=='ck05' else 0,
            'historical_time_multiple_ablation_seconds':15*seconds if ck=='ck05' else 0})
    csv_file(out/'HISTORICAL_TRAINING_COSTS.csv',rows)
    baseline=sum(r['historical_time_multiple_primary_seconds'] for r in rows)
    ablations=sum(r['historical_time_multiple_ablation_seconds'] for r in rows)
    proposal=ROOT/'datasets/local_runs'/EXPERIMENT/'vm36_supervisor_reproduction_research_acceptance_20261109T211213_255c0fa1/TRAINING_LABEL_POLICY_DRAFT.json'
    result={'status':'RESOURCE_AND_PROTOCOL_BLOCKED','primary_matched_runs':12,'five_seed_matched_ablation_runs':15,
        'total_runs_if_full_seed_matched':27,'historical_primary_time_multiple_hours':baseline/3600,
        'historical_ablation_time_multiple_hours':ablations/3600,'historical_total_multiple_hours':(baseline+ablations)/3600,
        'estimate_quality':'SCREENING_ONLY_NOT_A_CURRENT_RUNTIME_PREDICTION; source library cumulative times, not independently instrumented GPU hours.',
        'training_resolution':1280,'configured_historical_epoch_budget':100,'seed_count':3,'seed_values':'NOT_FROZEN',
        'augmentations':['NoiseAug','MotionBlurAug','ExposureAug','JPEGAug','HazeAug'],
        'source_plan':'Original supervisor plan, Closing the gap; no augmentation family removed.',
        'controls':['Same verified initialization for matched arms','Same6471 image IDs and one approved derived label view',
            'Same training library, optimizer, LR schedule, epochs and checkpoint selection','Matched seeds and fixed evaluation conditions'],
        'blockers':['No separately authorized matched-training budget','Unfrozen augmentation ranges/bindings; do not reuse exact test values',
            'Initialization creation SHA and full original checkpoint selection provenance unresolved',
            'Derived zero-area label policy is a proposal, not an active training view','Full training decoding/loader admission not complete',
            'Formal test-dev independence unresolved'],
        'training_label_proposal':source(proposal),'new_training_runs':0,'wall_seconds':time.perf_counter()-start}
    save(out/'TRAINING_READINESS.json',result,out)
    write_atomic(out/'diary.md','# Matched training readiness\n\nRead original curve/argument evidence and counted the full three-seed controls plus five ablations. Historical cumulative times are screening references only. No budget, seed or augmentation-binding approval is inferred. No detector training.\n',out)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>RQ4 matched-training readiness</h1><p>Blocked for real training.</p><a href="TRAINING_READINESS.json">Protocol gaps and costs</a>',out)
    save(out/'inputs.json',{'catalog':source(ROOT/'datasets/research_archive/migration_20261104/model_catalog.csv'),'source':source(Path(__file__))},out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'TRAINING_READINESS','status':'RESOURCE_AND_PROTOCOL_BLOCKED'},out,replace=True)
    seal(out);print(out,flush=True)


if __name__=='__main__':execute()
