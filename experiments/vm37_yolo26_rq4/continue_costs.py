"""Continue accepted engineering and cost gates within the existing allocation."""
import argparse
import time
import traceback
from study import *


def execute(recovery_id):
    recovered=run_path(recovery_id)
    verify_hashes(recovered,('worker_result.json','research_head_acceptance.json','metrics.json'))
    head=load(recovered/'research_head_acceptance.json')
    if load(recovered/'worker_result.json')['status']!='PASS' or head['images']!=548 or not head['COCO_replay_equal']:
        raise ValueError('Complete-epoch research inference has not passed.')
    out=create('cost_gate_continuation');runtime(out,cuda=False)
    save(out/'inputs.json',{'recovery':source(recovered/'SHA256SUMS.txt'),'authorization':load(RUNS/'compute_allocation.json')['authorization'],
                          'cost_policy':source(HERE/'COST_ADMISSION_POLICY.json')},out)
    started=time.perf_counter();stages=[];status='FAILED'
    try:
        from cpu_tests import execute as cpu_test
        from training import execute as train
        from budget_analysis import execute as assess
        # Each call creates a separate immutable Run and consumes the same ledger.
        regression=cpu_test();stages.append({'stage':'FOCUSED_CPU_REGRESSION','RunID':regression.name})
        canary=train('canary','YOLO26s');stages.append({'stage':'OPTIMIZED_CANARY','RunID':canary.name})
        if load(canary/'run.json')['status']!='PASS':raise RuntimeError('Optimized GPU canary failed; no automatic retry.')
        control=train('full_epoch','YOLO26s','CONTROL');stages.append({'stage':'CONTROL_COST','RunID':control.name})
        if load(control/'run.json')['status']!='PASS':raise RuntimeError('Optimized complete control epoch failed; no automatic retry.')
        decision=assess(control.name);stages.append({'stage':'RESOURCE_GATE','RunID':decision.name})
        gate=load(decision/'budget_admission.json')['status']
        if gate=='NEEDS_TARGETED_COST':
            targeted=train('full_epoch','YOLO26s','ALL_FIVE');stages.append({'stage':'TARGETED_COST','RunID':targeted.name})
            if load(targeted/'run.json')['status']!='PASS':raise RuntimeError('Targeted cost epoch failed; no automatic retry.')
            decision=assess(control.name,targeted.name);stages.append({'stage':'MATCHED_RESOURCE_GATE','RunID':decision.name})
            gate=load(decision/'budget_admission.json')['status']
        status=gate
    except BaseException:
        save(out/'failure.json',{'traceback':traceback.format_exc(),'retry':False},out)
    save(out/'continuation_result.json',{'status':status,'stages':stages,'orchestrator_wall_seconds':time.perf_counter()-started,
         'formal100_training_started':False,'budget_reset':False,'new_training_authorization_created':False},out)
    write_atomic(out/'diary.md','# Cost-gate continuation\n\nExecuted only accepted engineering and complete first-epoch cost stages. The existing48-hour allocation was retained. A failed stage is not retried. Any formal100-epoch entry still needs implementation and validation after a positive resource gate.\n',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'status':status},out,replace=True)
    seal(out);print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--recovery',required=True)
    execute(parser.parse_args().recovery)
