"""Project matched-pair costs from complete-epoch measurements."""
import argparse
import shutil
from datetime import datetime
from study import *


def measured(run_id,arm):
    folder=run_path(run_id);verify_hashes(folder,('training_acceptance.json','protocol.json'))
    result=load(folder/'training_acceptance.json');plan=load(folder/'protocol.json')
    actual=(result['completed_epochs'],result['training_images'],result['validation_images'],
            result['forward_batches'],result['backward_batches'],result['optimizer_updates'],result['EMA_updates'])
    if actual!=(1,6471,548,3236,3236,784,784):raise ValueError('Incomplete cost epoch or update plan.')
    if plan.get('arm','CONTROL')!=arm or plan['architecture']!='YOLO26s':raise ValueError('Cost arm identity differs.')
    if sha256(Path(result['checkpoint']['path']))!=result['checkpoint']['sha256']:raise ValueError('Cost checkpoint changed.')
    events=[__import__('json').loads(s) for s in (folder/'health_events.jsonl').read_text().splitlines()]
    if len([r for r in events if r['event']=='optimizer_and_EMA_healthy'])!=784:raise ValueError('Incomplete health/update evidence.')
    update_seconds=sum(r['check_and_step_seconds'] for r in events if r['event']=='optimizer_and_EMA_healthy')
    save_seconds=sum(r['save_reload_seconds'] for r in events if r['event']=='FP32_snapshot_reloaded_equal')
    validation_seconds=sum(r['seconds'] for r in result['internal_validation'])
    total=result['training_setup_epoch_internal_val_save_seconds']
    components=result['loader_seconds']+result['forward_backward_health_seconds']+validation_seconds+save_seconds
    one_time=max(0.,total-components)
    adjusted=one_time+100*(result['loader_seconds']+result['forward_backward_health_seconds']-update_seconds+
                            validation_seconds+save_seconds)+expected_steps(6471,100)['updates']*update_seconds/784
    return {'RunID':run_id,'arm':arm,'measurement':result,'sources':[source(folder/'training_acceptance.json'),source(folder/'protocol.json')],
            'first_epoch_linear_seconds':100*total,'component_update_adjusted_seconds':adjusted,
            'optimizer_health_step_seconds':update_seconds,'validation_seconds':validation_seconds,'save_reload_seconds':save_seconds,
            'one_time_residual_seconds':one_time}


def execute(control_id,target_id=None):
    out=create('matched_budget_cpu');runtime(out,cuda=False)
    policy=load(HERE/'COST_ADMISSION_POLICY.json');arms=[measured(control_id,'CONTROL')]
    if target_id:arms.append(measured(target_id,'ALL_FIVE'))
    historical=[]
    for row in load(PREP/'FROZEN_DEVELOPMENT_EVALUATION.json')['conditions']:
        if row['checkpoint']=='ck01' and row['inference_resolution']==1280:
            path=Path(row['source_folder'])/'timing.json'
            if sha256(path)!=row['source_SHA256']['timing.json']:raise ValueError('Historical timing changed.')
            historical.append({'pair_id':row['pair_id'],'source':source(path),'wall_seconds':load(path)['wall_seconds']})
    if len(historical)!=21:raise ValueError('The evaluation reserve needs all21 frozen contexts.')
    reserve=42*max(row['wall_seconds'] for row in historical)
    allocation=load(RUNS/'compute_allocation.json')
    spent=sum(r['charged_seconds'] for r in allocation['completed_workers'])
    remaining=min(allocation['total_seconds']-spent,(datetime.fromisoformat(allocation['absolute_deadline_UTC'])-utc()).total_seconds())
    conservative=[r['first_epoch_linear_seconds']*policy['uncertainty_multiplier'] for r in arms]
    free=shutil.disk_usage(ROOT).free
    snapshot_bytes=max(r['measurement']['checkpoint']['bytes'] for r in arms)
    disk_required=200*snapshot_bytes+policy['disk_reserve_bytes']
    if conservative[0]+reserve>remaining:decision='RESOURCE_BLOCKED'
    elif len(arms)!=2:decision='NEEDS_TARGETED_COST'
    elif sum(conservative)+reserve>remaining or disk_required>free:decision='RESOURCE_BLOCKED'
    else:decision='RESOURCE_FIT_REQUIRES_FROZEN_FORMAL_ENTRY'
    result={'status':decision,'arms':arms,'remaining_seconds':remaining,'charged_worker_seconds':spent,
            'conservative_arm_seconds':conservative,'evaluation_reserve_seconds':reserve,
            'pair_total_seconds':sum(conservative)+reserve if len(arms)==2 else None,
            'targeted_cost_measured':len(arms)==2,'disk_free_bytes':free,'minimum_pair_disk_bytes':disk_required,
            'disk_limitations':'Evaluation artifacts add space; the checkpoint estimate is a minimum, not a complete storage forecast.',
            'historical_timing_inputs':historical,'cost_policy':source(HERE/'COST_ADMISSION_POLICY.json'),
            'forecast_limitations':'One first epoch is not a measured100-epoch runtime. Warmup, target density, CPU loading and memory pressure can change later costs. The25% margin is a planning choice.',
            'full100_arms_completed':0,'matched_scientific_pairs':0,'budget_reset':False}
    save(out/'budget_admission.json',result,out)
    write_atomic(out/'diary.md','# Matched training resource gate\n\nUsed complete6471/548 first-epoch measurements and verified historical evaluation timings. No shortened arm, new inference or GPU training was executed by this analysis.\n',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'status':'COMPLETE_WITH_LIMITATIONS','decision':decision,'GPU_seconds':0},out,replace=True)
    seal(out);print(out,flush=True)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--control',required=True);parser.add_argument('--target')
    args=parser.parse_args();execute(args.control,args.target)
