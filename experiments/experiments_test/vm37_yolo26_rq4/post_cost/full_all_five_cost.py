"""One guarded ALL_FIVE full-epoch cost measurement after the short gate."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

POST = Path(__file__).resolve().parent
sys.path.insert(0, str(POST.parent))
from study import *
from safety import available_seconds, query, run_owned
from short_cost import ac_connected

PREFIX = 'vm37_yolo26_rq4_all_five_short_cost_20261112T195422_06010323'
CAP = 6600
TICKET = RUNS / 'all_five_full_cost_ticket.json'


def prepare():
    prefix = run_path(PREFIX)
    verify_hashes(prefix, ('final_result.json', 'short_measurement.json', 'physical_confirmation.json', 'CPU_initialization_health.json'))
    result = load(prefix / 'final_result.json')
    if result['status'] != 'SHORT_PREFIX_COMPLETED_ENGINEERING_ONLY' or result['supervisor']['stop_reason'] is not None:
        raise RuntimeError('The bounded thermal/health gate did not pass.')
    if TICKET.exists() or available_seconds(load(RUNS/'compute_allocation.json')) < CAP + 60:
        raise RuntimeError('The one-attempt full-cost budget gate failed.')
    current = query()
    policy = load(prefix / 'safety_policy.json')
    if current['temperature_C'] > policy['preflight_maximum_temperature_C'] or current['free_MiB'] < policy['preflight_minimum_free_MiB'] or current['utilization_percent'] > 10:
        raise RuntimeError('GPU is not ready for the bounded cost measurement.')
    ac_connected()
    out = create('all_five_full_epoch_cost')
    runtime(out, cuda=False)
    plan = load(prefix / 'protocol.json')
    plan['args'].update(data=str(out/'dataset.yaml'),project=str(out),name='trainer')
    plan.update(mode='FULL_EPOCH_COST', scope='FULL_EPOCH_COST_ONLY_NOT_PILOT_RESULT',
        expected=expected_steps(6471,20,stop_epochs=1),stop_after_completed_epochs=1,completed_epochs=0,
        worker_hard_seconds=CAP,validation='Full548 internal validation plus frozen research replay if completed.',
        AP='ENGINEERING_CLEAN_ONLY_IF_MEASURED_NOT_MATCHED_BENEFIT',
        snapshot_role='One-epoch engineering/cost checkpoint; forbidden as pilot or formal initialization.',
        stop_rule='Complete epoch1 of the declared20-epoch scheduling horizon, then stop. No20-epoch arm.',
        full_epoch_cost='TO_BE_MEASURED')
    plan.pop('stop_after_batches',None)
    plan['COCO_source'] = next(row['source'] for row in load(PREP/'CANDIDATE_INITIALIZATION_MANIFEST.json')
                               if row['architecture']=='YOLO26s' and row['seed']==42)
    plan['sources'][str(Path(__file__).relative_to(PUBLIC))]=sha256(Path(__file__))
    save(out/'protocol.json',plan,out)
    save(out/'safety_policy.json',policy,out)
    shutil.copyfile(prefix/'physical_confirmation.json',out/'physical_confirmation.json')
    save(out/'inputs.json',{'short_gate':source(prefix/'SHA256SUMS.txt'),
        'original_allocation_before':source(RUNS/'compute_allocation.json'),
        'policy':source(HERE/'TARGETED_POLICY.json'),'GPU_before':current,
        'CONTROL_reference_RunID':'vm37_yolo26_rq4_full_epoch_yolo26s_control_20261111T232439_301fb25d'},out)
    snapshots=out/'source_snapshots';snapshots.mkdir()
    for relative in plan['sources']:
        shutil.copyfile(PUBLIC/relative,snapshots/relative.replace('/','__').replace('\\','__'))
    write_atomic(out/'diary.md','# ALL_FIVE full-epoch cost\n\nA distinct full-data cost measurement after the successful128-batch gate. Starts from the same healthy native10 initialization, not the prefix state. Epoch1 uses the separate20-epoch horizon; stop after one complete epoch. No matched pilot or augmentation-efficacy claim. Frozen90C/87C stops and original allocation remain active.\n',out)
    print(out,flush=True)


def worker(out):
    if load(TICKET)['RunID']!=out.name or load(TICKET)['protocol_SHA256']!=sha256(out/'protocol.json'):
        raise PermissionError('No matching one-attempt cost ticket.')
    save(out/'WORKER_STARTED.json',{'PID':os.getpid(),'UTC':utc().isoformat()},out)
    # Reuse the accepted training, health, FP32 saving and research replay code.
    from training import worker as accepted_worker
    accepted_worker(out)


def execute(out):
    if TICKET.exists():
        raise RuntimeError('The full-cost attempt is already consumed; no automatic retry.')
    plan=load(out/'protocol.json')
    if plan['args']['epochs']!=20 or plan['stop_after_completed_epochs']!=1 or plan['arm']!='ALL_FIVE':
        raise ValueError('The prepared cost scope changed.')
    ac_connected()
    save(TICKET,{'RunID':out.name,'protocol_SHA256':sha256(out/'protocol.json'),
        'created_UTC':utc().isoformat(),'retry_allowed':False,'short_gate_RunID':PREFIX},RUNS)
    cap=budget_start(out,CAP,'FULL_DATA_COST')
    began=time.perf_counter();supervisor=None;error=None
    try:
        supervisor=run_owned([sys.executable,'-B','-X','utf8',str(Path(__file__)),'--worker',out.name],out,cap,
            load(out/'safety_policy.json'),dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',
                YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',OMP_NUM_THREADS='2',PYTHONIOENCODING='utf-8'))
    except BaseException:
        error=traceback.format_exc()
    result=load(out/'worker_result.json') if (out/'worker_result.json').exists() else {'status':'INTERRUPTED'}
    status=result['status']
    if error or supervisor is None or supervisor['returncode'] or supervisor['stop_reason']:
        status='PARTIAL_WITH_LIMITATIONS'
    charged=time.perf_counter()-began
    budget_finish(out,charged,status)
    traces=[load_line for line in (out/'gpu_telemetry.jsonl').read_text().splitlines() if (load_line:=json.loads(line))] if (out/'gpu_telemetry.jsonl').exists() else []
    save(out/'cost_result.json',{'status':status,'worker':result,'supervisor':supervisor,'error':error,
        'charged_seconds':charged,'remaining_seconds':available_seconds(load(RUNS/'compute_allocation.json')),
        'absolute_deadline_UTC':load(RUNS/'compute_allocation.json')['absolute_deadline_UTC'],'budget_reset':False,
        'maximum_sampled_temperature_C':max((r['temperature_C'] for r in traces),default=None),
        'maximum_sampled_memory_used_MiB':max((r['used_MiB'] for r in traces),default=None),
        'pilot20':'NOT_ADMITTED','formal100':'RESOURCE_BLOCKED','augmentation_benefit':'NOT_EVALUATED',
        'scope':'Single full epoch for cost only; no complete20/100-epoch matched arm.'},out)
    save(out/'nested_SHA256.json',{str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file() and
        p.parent!=out and 'runtime' not in p.parts},out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'ALL_FIVE_FULL_EPOCH_COST',
        'status':status,'matched_pairs':0},out,replace=True)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>ALL_FIVE full-epoch cost</h1>'
        '<p>'+status+'</p><p>Engineering/cost evidence only; no completed matched training pair.</p>'
        '<a href="cost_result.json">Budget and temperature</a><br><a href="training_acceptance.json">Actual training</a>'
        '<br><a href="research_head_acceptance.json">Research metric replay</a>',out)
    seal(out)
    from main.visualization import build_run_index
    build_run_index(RUNS,EXPERIMENT)
    print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--execute')
    parser.add_argument('--worker')
    args=parser.parse_args()
    if args.prepare:
        prepare()
    elif args.execute:
        execute(run_path(args.execute))
    elif args.worker:
        worker(run_path(args.worker))
    else:
        parser.error('Prepare, or execute one prepared cost RunID.')
