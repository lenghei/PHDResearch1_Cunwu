"""Prepare/test VM43; execute only with separate documented admissions."""
import argparse
from datetime import datetime,timedelta,timezone
import json
import os
from pathlib import Path
import secrets
import sys
import time
import traceback

from vm43_common import HERE,ROOT,PUBLIC,RUNS,EXPERIMENT,V42,V41,AUTH,BUNDLE,study,load,save,sha,source,module,frozen,budget,gates
sys.path.insert(0,str(HERE))
import backend


def verify_sealed(out):
    for line in (out/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        expected,file=line.split('  ',1)
        if sha(Path(file))!=expected:raise ValueError('Sealed artifact changed: '+file)


def prepare():
    from main.pipeline import new_run
    out,logger=new_run({'output_root':str(RUNS)},EXPERIMENT,'implementation');study.runtime(out,cuda=False);started=time.perf_counter()
    try:
        verify_sealed(V42)
        spec=frozen();input_files=[V42/'Pilot_4cell_frozen_protocol.json',V42/'Independent_admission_decision.json',
            V42/'Historical_split_usage_matrix.csv',V42/'Scientific_blockers_and_actions.csv',V41/'Future_Gap_protocol.json',
            V41/'Timing_protocol_comparison.csv',HERE.parent/'vm41_degradation_coverage/timing_protocol.py']
        input_files.extend(PUBLIC/'code/main'/f for f in ('dataset.py','labels.py','statistics.py','predictor.py','predictor_stable.py','training_health.py'))
        input_files.extend((HERE.parent/'cv261109/generator.py',HERE.parent/'cv261109/model.py',HERE.parent/'cv261207/development_fast_coco.py'))
        checkpoint_records=[]
        for strategy in ('CONTROL','VARIABILITY_GUIDED'):
            r=next(c for c in spec['cells'] if c['strategy']==strategy);checkpoint=r['checkpoint'];current=source(checkpoint['path'])
            if current['SHA256']!=checkpoint['SHA256']:raise ValueError('Frozen epoch10 checkpoint SHA mismatch.')
            checkpoint_records.append({'strategy':strategy,'checkpoint':current,'selection':'FIXED_SEED42_EPOCH10_EMA','new_health_test':'NOT_REPEATED; accepted prior health/reload evidence retained'})
        save(out/'checkpoint_provenance.json',checkpoint_records)
        save(out/'pilot_protocol_verified.json',{'status':'EXACT_VM42_OPTION_A_VERIFIED','source':source(V42/'Pilot_4cell_frozen_protocol.json'),
            'cells':spec['cells'],'cells_count':4,'image_count':1610,'seed':42,'no_policy_change':True,
            'execution_order':'VM42 stored order, fixed before any AP; two models each loaded once',
            'clean_accuracy_tolerance':'Not numerically specified in VM42; report measured clean cost, do not invent an acceptable-loss cut-off.'})
        payload=backend.prepare_gt(out,spec);logger.info('Full1610 annotation/GT input prepared; no full image decode or AP.')
        save(out/'dataset_membership.json',{'status':'SOURCE_SHA_AND_FROZEN_GT_VALIDATED','images':len(payload['records']),
            'class_order':spec['native10_class_order'],'population':source(spec['population']['path']),
            'ZIP_certificate':load(spec['population']['path'])['source_ZIP_accepted_certificate'],
            'GT':source(out/'ground_truth_input.json'),'GT_valid_instances_after_ignore':len(payload['coco']['annotations']),
            'source_image_full_decode':'VM40 accepted evidence reused; this integration tests only one image',
            'scientific_independence':'UNKNOWN_NOT_UPGRADED','annotation_mapping':'Unchanged native10/class0-or-score0 ignore/class11 other + original raster union'})
        tests=module('vm43_cpu_tests',HERE/'tests_cpu.py').run(out,payload['records'])
        save(out/'real_evaluator_test_results.json',tests);logger.info('CPU acceptance: %d passed',tests['passed'])
        admission=gates(out)
        prior=load(V42/'Independent_admission_decision.json')
        save(out/'scientific_admission.json',{'status':'BLOCKED','new_primary_evidence_received':False,
            'prior_independent_status':prior['status'],'independent_admitted':False,'limited_external_decision_provided':False,
            'current_user_scope':'Conditional execution only; does not itself document the separate scientific override',
            'decision_labels_supported':['INDEPENDENT_ADMITTED only with primary evidence','LIMITED_EXTERNAL_EXPLORATORY only with explicit separate documented decision'],
            'exact_blocker':'No new historical-use/selection/scene provenance evidence or separate limited-exploratory decision.'})
        save(out/'gpu_resource_admission.json',admission)
        save(out/'execution_inputs.json',{'protocol':source(V42/'Pilot_4cell_frozen_protocol.json'),'GT':source(out/'ground_truth_input.json'),
            'files':[source(f) for f in input_files]+[source(f) for f in HERE.glob('*.py')],
            'budget_before':budget(),'new_GPU':0})
        finish_preparation(out,started)
        print('CPU_IMPLEMENTED_AND_TESTED_GPU_BLOCKED',out,flush=True)
    except BaseException as exc:
        save(out/'preparation_failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'GPU':0})
        record=load(out/'run.json');record['status']='BLOCKED_CPU_IMPLEMENTATION'
        (out/'run.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
        logger.exception('VM43 CPU acceptance blocked.');raise


def finish_preparation(out,started):
    reporter=module('vm43_reporter',HERE/'reporting.py');report_paths=reporter.prepare_report(out)
    reporter.page(out,report_paths)
    for item in load(out/'execution_inputs.json')['files']:
        if sha(item['path'])!=item['SHA256']:raise ValueError('Input/source changed during CPU acceptance.')
    for relative,digest in load(out/'source_SHA256.json').items():
        if relative.startswith('experiments/'+EXPERIMENT+'/'):continue
        if sha(PUBLIC/relative)!=digest:raise ValueError('Historical public code changed: '+relative)
    before=load(out/'execution_inputs.json')['budget_before'];after=budget()
    if before['authorization']!=after['authorization'] or before['charges']!=after['charges']:raise ValueError('Budget changed during CPU preparation.')
    save(out/'resource_usage.json',{'CPU_wall_seconds':time.perf_counter()-started,'GPU_seconds':0,
        'GPU_charged_seconds':0,'new_training':False,'scientific_predictions':0,'full_source_image_decode_repeated':False})
    save(out/'final_acceptance.json',{'status':'CPU_IMPLEMENTED_AND_UNIT_TESTED_GPU_BLOCKED','real_four_cell_evaluation':'NOT_RUN',
        'CPU_acceptance':load(out/'real_evaluator_test_results.json')['status'],'scientific_admission':'BLOCKED',
        'GPU_executed':False,'actual_AP':'NOT_EVALUATED','mocked_scientific_outputs':False,'source_and_old_budget_unchanged':True,
        'required_future_decisions':['Documented independent admission or explicit limited external scientific decision','Explicit valid four-cell GPU authorization',
            'Complete four-cell cost admitted with reserve before unchanged deadline','Renewed thermal admission'],
        'reports':[str(p) for p in report_paths]})
    record=load(out/'run.json');record.update(status='COMPLETE_WITH_LIMITATIONS',GPU_seconds=0,
        finished_UTC=datetime.now(timezone.utc).isoformat())
    (out/'run.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    seal(out,report_paths)


def seal(out,report_paths=()):
    files=[p for p in out.rglob('*') if p.is_file() and 'runtime' not in p.relative_to(out).parts and p.name!='SHA256SUMS.txt']+list(report_paths)+list(HERE.glob('*.py'))
    with (out/'SHA256SUMS.txt').open('x',encoding='utf-8') as f:
        for p in sorted(files):f.write(sha(p)+'  '+str(p)+'\n')
    from main.visualization import build_run_index
    build_run_index(out.parent,EXPERIMENT)


def worker_entry(out):
    ticket=load(out/'execution_ticket.json');token=os.environ.get('VM43_WORKER_TOKEN','')
    if not token or __import__('hashlib').sha256(token.encode()).hexdigest()!=ticket['worker_token_SHA256']:raise PermissionError('Worker requires owned-supervisor token.')
    if datetime.now(timezone.utc)>=datetime.fromisoformat(ticket['worker_deadline_UTC']):raise PermissionError('Worker lease expired.')
    prepared=Path(ticket['prepared_run']);verify_sealed(prepared)
    admission=gates(prepared,Path(ticket['decision']),Path(ticket['authorization']),Path(ticket['thermal']))
    if not admission['all_gates_pass']:raise PermissionError('Worker gates no longer pass.')
    for item in load(prepared/'execution_inputs.json')['files']:
        if sha(item['path'])!=item['SHA256']:raise ValueError('Frozen source changed.')
    gt=load(prepared/'execution_inputs.json')['GT']
    if sha(gt['path'])!=gt['SHA256']:raise ValueError('Frozen GT input changed.')
    study.runtime(out,cuda=True)
    try:backend.worker(out,frozen(),admission['evaluation_scope'],Path(gt['path']))
    except BaseException as exc:
        save(out/'worker_failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'matched_comparison':'INCOMPLETE','retry':False})
        raise


def execute(prepared,decision,authorization,thermal):
    verify_sealed(prepared)
    admission=gates(prepared,decision,authorization,thermal)
    if not admission['all_gates_pass']:raise PermissionError(json.dumps(admission['failures']))
    approval=load(authorization);identifier=sha(authorization)
    for p in RUNS.glob(EXPERIMENT+'_*/execution_ticket.json'):
        if load(p).get('authorization_SHA256')==identifier:raise PermissionError('This authorization already has an attempted execution; no retry.')
    from main.pipeline import new_run
    from safety import run_owned
    out,_=new_run({'output_root':str(RUNS)},EXPERIMENT,'execution');study.runtime(out,cuda=False)
    lock=study.RUNS/'.pilot10_gpu.lock'
    if lock.exists() or (study.RUNS/'.gpu.lock').exists():raise RuntimeError('Existing GPU owner lock.')
    token=secrets.token_hex(24);now=datetime.now(timezone.utc);cap=approval['max_worker_seconds']
    deadline=now+timedelta(seconds=cap)
    if cap+approval['shutdown_reserve_seconds']>budget()['remaining_seconds']:raise PermissionError('Complete unit no longer fits.')
    save(out/'execution_ticket.json',{'prepared_run':str(prepared),'decision':str(decision),'authorization':str(authorization),'thermal':str(thermal),
        'authorization_SHA256':identifier,'worker_token_SHA256':__import__('hashlib').sha256(token.encode()).hexdigest(),
        'worker_deadline_UTC':deadline.isoformat(),'worker_cap_seconds':cap,'attempt':1})
    with lock.open('x',encoding='utf-8') as f:json.dump({'RunID':out.name,'PID':os.getpid()},f)
    start=time.perf_counter();result=None
    try:
        command=[sys.executable,'-B','-X','utf8',str(HERE/'run.py'),'--worker',out.name]
        result=run_owned(command,out,cap,load(BUNDLE/'safety_policy.json'),dict(os.environ,VM43_WORKER_TOKEN=token))
    finally:
        elapsed=time.perf_counter()-start
        charge=AUTH/'charges'/(out.name+'_evaluation.json')
        with charge.open('x',encoding='utf-8') as f:json.dump({'RunID':out.name,'stage':'VM43_OPTION_A_EVALUATION','charged_seconds':elapsed,
            'status':'ATTEMPT_ENDED','authorization_SHA256':identifier,'no_budget_reset':True},f,indent=2)
        if lock.exists() and load(lock).get('RunID')==out.name:lock.unlink()
        actual=load(out/'worker_result.json') if (out/'worker_result.json').exists() else None
        success=bool(result and result['returncode']==0 and result['stop_reason'] is None and actual and actual['status']=='REAL_FOUR_CELLS_COMPLETED')
        save(out/'final_acceptance.json',{'status':'REAL_FOUR_CELLS_COMPLETED' if success else 'INCOMPLETE','GPU_executed':True,
            'supervisor_stop_reason':result['stop_reason'] if result else 'SUPERVISOR_FAILURE','worker_result':actual,
            'charged_seconds':elapsed,'retry':False})
        seal(out)
    print('EXECUTION_ENDED',out,flush=True)


def resolve(identity):
    if Path(identity).name!=identity or not identity.startswith(EXPERIMENT+'_'):raise ValueError('Invalid VM43 RunID.')
    return (RUNS/identity).resolve(strict=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='VM43 defaults to CPU implementation acceptance.')
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--prepare',action='store_true');mode.add_argument('--verify')
    mode.add_argument('--execute');mode.add_argument('--worker')
    parser.add_argument('--scientific-decision',type=Path);parser.add_argument('--gpu-authorization',type=Path);parser.add_argument('--thermal-admission',type=Path)
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.verify:verify_sealed(resolve(args.verify));print('VERIFIED',args.verify)
    elif args.worker:worker_entry(resolve(args.worker))
    else:
        if not all((args.scientific_decision,args.gpu_authorization,args.thermal_admission)):parser.error('Three separate admission files are required; VM43 does not generate them.')
        execute(resolve(args.execute),args.scientific_decision.resolve(strict=True),args.gpu_authorization.resolve(strict=True),args.thermal_admission.resolve(strict=True))
