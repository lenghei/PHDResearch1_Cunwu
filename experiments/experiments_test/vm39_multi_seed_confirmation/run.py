"""Matched seed confirmation using the accepted VM37/VM38 workers."""
import argparse
import copy
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('vm38_reuse', HERE.parent / 'vm38_variability_guided_augmentation/run.py')
guided = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guided)
old, recovery, study = guided.old, guided.recovery, guided.study
load, save = old.load, old.save
SOURCE = Path(__file__).resolve()
EXPERIMENT = HERE.name
RUNS = study.ROOT / 'datasets/local_runs' / EXPERIMENT
GUIDED42 = 'vm38_variability_guided_augmentation_targeted960_20261114T215310_e8bba12b'
ARMS = ('CONTROL', 'VARIABILITY_GUIDED')
CAPS = {'CONTROL': 12000, 'VARIABILITY_GUIDED': 15000, 'evaluation': 14400, 'reserve': 1800}
SCOPE = 'DEVELOPMENT_ONLY / POST_HOC_POLICY_SELECTION'


def create(mode):
    return study.new_run({'output_root': str(RUNS)}, EXPERIMENT, mode)[0]


def path(identity):
    if Path(identity).name != identity or not identity.startswith(EXPERIMENT + '_'):
        raise ValueError('Invalid VM39 RunID.')
    return (RUNS / identity).resolve(strict=True)


def previous_guided():
    return (guided.RUNS / GUIDED42).resolve(strict=True)


def budget():
    value = recovery.budget_snapshot()
    value.pop('GPU_seconds_this_run', None)
    return value


def check_parent(parent):
    for name, digest in load(parent / 'execution_freeze.json')['files'].items():
        if study.sha256(Path(name)) != digest:
            raise ValueError('Frozen source or evidence changed: ' + name)
    old.verify_freeze(study.run_path(recovery.AUTH))
    if old.available(study.run_path(recovery.AUTH)) <= 0:
        raise PermissionError('The original budget expired or was exhausted.')


def check(seed_run):
    parent = path(load(seed_run / 'seed_run.json')['parent_RunID'])
    check_parent(parent)
    for name, digest in load(seed_run / 'execution_freeze.json')['files'].items():
        if study.sha256(Path(name)) != digest:
            raise ValueError('Seed plan changed: ' + name)
    return parent


def make_seed(parent, seed, caps):
    out = create('seed' + str(seed))
    study.runtime(out, cuda=False)
    save(out / 'seed_run.json', {'parent_RunID': parent.name, 'seed': seed, 'caps': caps, 'scope': SCOPE})
    protocol = copy.deepcopy(recovery.candidate_protocol('B'))
    geometry_contract = copy.deepcopy(protocol)
    protocol.update(id='VM39_YOLO26S_960_10E_SEED' + str(seed), seeds=[seed], arms=list(ARMS),
                    scope=SCOPE, GPU_execution_authorized=True)
    save(out / 'scientific_protocol.json', protocol)
    population = old.v3.base.resolve_data(old.bundle(), study.ROOT / 'datasets/VisDrone', verify=False)
    reference = load(study.run_path(guided.REFERENCE_ID) / 'YOLO26s_seed42_CONTROL/protocol.json')
    plans = []
    policy = load(previous_guided() / 'policy.json')
    if policy != guided.recipe():
        raise ValueError('The accepted Gaussian policy differs.')
    for arm in ARMS:
        folder = out / ('YOLO26s_seed' + str(seed) + '_' + arm)
        folder.mkdir()
        plan = old.v3.make_plan(old.bundle(), population, protocol, 'YOLO26s', seed, arm,
                                folder, 'vm39', 'optimized')
        plan.update(mode=protocol['id'], scope=SCOPE, selection='FIXED_EPOCH10_EMA',
            protocol_name='vm39',
            targeted_recipe=copy.deepcopy(policy) if arm == ARMS[1] else None,
            recovery_protocol=copy.deepcopy(geometry_contract),
            recovery_protocol_usage='Read-only legacy size assertion only; args.seed and this VM39 protocol define the actual seed.',
            executor_version=recovery.VERSION, GPU_execution_authorized=True,
            augmentation_label='VM38 frozen Gaussian p0.25' if arm == ARMS[1] else 'CONTROL')
        if plan['args']['seed'] != seed or plan['args']['imgsz'] != 960:
            raise ValueError('Seed or resolution binding failed.')
        shared = ('architecture', 'COCO_source_SHA256', 'train', 'development', 'train_labels',
                  'val_labels', 'expected', 'execution_epochs', 'selection')
        if any(plan[k] != reference[k] for k in shared):
            raise ValueError('Accepted population or schedule changed.')
        args = [{k: v for k, v in p['args'].items() if k not in ('data', 'project', 'seed', 'model')}
                for p in (plan, reference)]
        if args[0] != args[1]:
            raise ValueError('Ordinary training parameters changed.')
        initial = next(r for r in load(old.bundle() / 'portable_initializations.json')
                       if r['architecture']=='YOLO26s' and r['seed']==seed)
        if plan['initialization']['sha256'] != initial['sha256']:
            raise ValueError('Wrong seed-specific initialization.')
        save(folder / 'protocol.json', plan)
        plans.append(plan)
    if plans[0]['initialization'] != plans[1]['initialization']:
        raise ValueError('The two arms do not share the same initialization.')
    evaluation = out / 'evaluation'
    evaluation.mkdir()
    save(evaluation / 'protocol.json', {'scope': SCOPE, 'seed': seed, 'completed_arms_required': 2})
    save(evaluation / 'evaluation_stage.json', {'contexts': load(old.bundle() / 'evaluation_manifest.json')['contexts']})
    save(out / 'plan_acceptance.json', {'status':'PASS', 'seed':seed, 'initialization':plans[0]['initialization'],
        'parameters_equal_except_seed_and_policy':True, 'accepted_tests_reused':True,
        'new_CPU_test_suite':False, 'scope':SCOPE})
    files = [out / 'seed_run.json', out / 'scientific_protocol.json', out / 'plan_acceptance.json',
             evaluation / 'protocol.json', evaluation / 'evaluation_stage.json']
    files += [out / ('YOLO26s_seed' + str(seed) + '_' + a) / 'protocol.json' for a in ARMS]
    save(out / 'execution_freeze.json', {'files': {str(p):study.sha256(p) for p in files},
         'frozen_UTC':datetime.now(timezone.utc).isoformat()})
    return out


def context(out):
    ticket = load(out / 'worker_ticket.json')
    seed_run = path(ticket['seed_RunID'])
    check(seed_run)
    if out.parent != seed_run or study.sha256(out / 'protocol.json') != ticket['protocol_SHA256']:
        raise PermissionError('Worker boundary or protocol changed.')
    if ticket.get('evaluation_stage_SHA256') and study.sha256(out / 'evaluation_stage.json') != ticket['evaluation_stage_SHA256']:
        raise ValueError('Evaluation stage changed.')
    if datetime.now(timezone.utc) >= datetime.fromisoformat(load(out / 'WORKER_LEASE.json')['deadline_UTC']):
        raise PermissionError('Worker lease expired.')
    return load(out / 'protocol.json'), seed_run


def worker(out, mode):
    study.runtime(out, cuda=True)
    plan, seed_run = context(out)
    if mode == 'train':
        Factory = recovery.trainer_factory(plan, out)
        with patch.object(old, 'worker_context', context), patch.object(old.v3, 'trainer_factory', lambda *_:Factory):
            return old.train_worker(out)
    research = old.research
    original_load = research.load
    protocol = load(seed_run / 'scientific_protocol.json')
    seed = protocol['seeds'][0]
    evaluation_spec = load(old.bundle() / 'evaluation_manifest.json')
    evaluation_spec['contexts'] = load(out / 'evaluation_stage.json')['contexts']
    def mapped_load(p):
        if Path(p) == old.bundle() / 'vm39_protocol.json':
            return protocol
        if Path(p) == old.bundle() / 'formal_evaluation_manifest.json':
            return evaluation_spec
        return original_load(p)
    def admitted(*_):
        context(out)
        for arm in ARMS:
            accepted = load(seed_run / ('YOLO26s_seed'+str(seed)+'_'+arm) / 'training_acceptance.json')
            if any(accepted[k] != v for k,v in {'completed_epochs':10,'anchor_exposures':64710,
                'forward_batches':32360,'backward_batches':32360,'optimizer_updates':1816,'EMA_updates':1816}.items()):
                raise ValueError('The complete matched training pair is required.')
    with patch.object(research,'load',mapped_load), patch.object(research,'check_worker_authorization',admitted), \
            patch.object(research,'paired_contrasts',lambda _:[]):
        research.evaluate(old.bundle(),study.ROOT/'datasets/VisDrone',seed_run,'vm39')
    return 'COMPLETE'


def stage(seed_run, out, name, cap, mode):
    check(seed_run)
    authorization = study.run_path(recovery.AUTH)
    if old.available(authorization) < cap + CAPS['reserve']:
        raise PermissionError('The worker and safety reserve do not fit.')
    old.cooldown(out)
    save(out / 'resource_before.json',old.resource_check())
    deadline = min(datetime.now(timezone.utc)+timedelta(seconds=cap),
        datetime.fromisoformat(load(authorization/'authorization.json')['absolute_expiry_UTC']))
    save(out/'WORKER_LEASE.json',{'deadline_UTC':deadline.isoformat(),'cap_seconds':cap})
    save(out/'worker_ticket.json',{'seed_RunID':seed_run.name,'protocol_SHA256':study.sha256(out/'protocol.json'),
        'evaluation_stage_SHA256':study.sha256(out/'evaluation_stage.json') if mode=='evaluate' else None})
    command=[sys.executable,'-B','-X','utf8',str(SOURCE),'--worker',str(out),'--mode',mode]
    save(out/'command.json',command)
    started,result=time.perf_counter(),None
    try:
        result=old.run_owned(command,out,cap,load(old.bundle()/'safety_policy.json'),
            dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2'))
        diagnosis=recovery.stage_diagnosis(out,result,mode)
        save(out/'stage_diagnosis.json',diagnosis)
        if diagnosis['status']!='STAGE_COMPLETED':
            raise RuntimeError(diagnosis['primary_reason'])
    finally:
        save(authorization/'charges'/(seed_run.name+'_'+name+'.json'),{'RunID':seed_run.name,
            'stage':name,'stage_path':str(out),'charged_seconds':time.perf_counter()-started,
            'hard_cap':cap,'supervisor':result,'finished_UTC':datetime.now(timezone.utc).isoformat()})


def accept_seed(out):
    import itertools
    seed=load(out/'seed_run.json')['seed']
    signature=hashlib.sha256()
    folders=[out/('YOLO26s_seed'+str(seed)+'_'+a) for a in ARMS]
    keys=('position','epoch','image_ids','NCHW','optimizer_updates','EMA_updates','accumulation')
    batches=exposures=0
    with (folders[0]/'training_batches.jsonl').open() as c,(folders[1]/'training_batches.jsonl').open() as g:
        for lines in itertools.zip_longest(c,g):
            if not all(lines):raise ValueError('Unequal batch count.')
            a,b=map(json.loads,lines)
            if any(a[k]!=b[k] for k in keys):raise ValueError('Within-seed sampling or step mismatch.')
            signature.update(json.dumps(a['image_ids']).encode());batches+=1;exposures+=len(a['image_ids'])
    if (batches,exposures)!=(32360,64710):raise ValueError('Exposure count differs.')
    initial=[]; rates=[]; reloads=[]
    for folder in folders:
        value=load(folder/'training_acceptance.json');initial.append(value['initialization'])
        events=[json.loads(line) for line in (folder/'health_events.jsonl').read_text().splitlines()]
        steps=[e for e in events if e['event']=='optimizer_and_EMA_healthy']
        if len(steps)!=1816:raise ValueError('Incomplete optimizer health evidence.')
        rates.append([e['learning_rates'] for e in steps])
        reloads.append(sum(e['event']=='FP32_snapshot_reloaded_equal' for e in events))
        if not load(folder/'protected_inputs.json')['unchanged']:raise ValueError('Input preservation failed.')
    if initial[0]!=initial[1] or rates[0]!=rates[1] or reloads!=[10,10]:
        raise ValueError('Initialization, actual LR or FP32 reload mismatch.')
    index,values=guided.read_values(out/'evaluation',verify=True)
    if len(index)!=16 or {r['seed'] for r in index}!={seed} or {r['arm'] for r in index}!=set(ARMS):
        raise ValueError('The sixteen evaluation configurations are incomplete.')
    for value in values.values():
        if value['metrics']['images']!=548 or value['draws']['samples']!=200:
            raise ValueError('Evaluation population or bootstrap count differs.')
    alias={(k[0],k[1],'ALL_FIVE' if k[2]==ARMS[1] else 'CONTROL',k[3],k[4]):v for k,v in values.items()}
    paired=old.research.paired_contrasts(alias)
    for row in paired:
        if 'ALL_FIVE_minus_CONTROL' in row:row['GUIDED_minus_CONTROL']=row.pop('ALL_FIVE_minus_CONTROL')
    save(out/'paired_comparisons.json',paired)
    save(out/'seed_acceptance.json',{'status':'MATCHED_SEED_COMPLETE','seed':seed,'initialization':initial[0],
        'anchor_exposures_per_arm':exposures,'batches_per_arm':batches,'updates_per_arm':1816,
        'actual_LR_equal':True,'FP32_reload_events':reloads,'sampling_order_SHA256':signature.hexdigest(),
        'evaluation_configurations':16,'saved_image_records':8768,'scope':SCOPE})
    meta=load(out/'run.json');meta['status']='MATCHED_SEED_COMPLETE'
    study.save(out/'run.json',meta,out,replace=True)
    (out/'diary.md').write_text(f'# Seed {seed}\n\nCompleted CONTROL and frozen VM38 policy, ten epochs each. Sixteen measured evaluations and paired image bootstrap. No policy tuning.\n',encoding='utf-8')
    study.seal(out)
    print('VM39_SEED_COMPLETED',seed,out,flush=True)


def prepare():
    out=create('confirmation')
    study.runtime(out,cuda=False)
    old.verify_freeze(study.run_path(recovery.AUTH))
    guided.check(previous_guided())
    if load(previous_guided()/'final_acceptance.json')['status']!='PASS':
        raise ValueError('Accepted VM38 evidence is required.')
    old.v3.base.verify_bundle(old.bundle())
    old.v3.base.environment_check(old.bundle())
    available=budget()
    if sum(CAPS.values())>available['remaining_seconds']:
        raise PermissionError('The complete seed43 pair and evaluation do not fit.')
    save(out/'author_authorization.json',{'source':'Current VM39 user instruction: seed43, then conditionally seed44.',
        'original_authorization':recovery.AUTH,'no_new_allocation':True,'no_deadline_extension':True,
        'no_retry':True,'policy_source':guided.source(previous_guided()/'policy.json'),'scope':SCOPE})
    save(out/'resource_admission.json',{'status':'SEED43_COMPLETE_PAIR_FITS','caps_seconds':CAPS,
        'complete_seed_required_seconds':sum(CAPS.values()),'budget':available,
        'measured_reference_worker_seconds':{'CONTROL_10_epochs':8131.553,'GUIDED_10_epochs':10261.991279900074,
            'two_arm_16_evaluations':10808.362,'GUIDED_8_evaluations':4709.062168800039},
        'conservative_basis':'CONTROL cap +47.6%, GUIDED cap +46.2%, evaluation cap +33.2% over corresponding full measured workers; plus 1800s reserve.',
        'seed44_rule':'After accepted seed43, each cap=max(base, ceil(seed43 observed*1.3 to60s)); launch only if their sum and reserve fit.',
        'new_CPU_test_suite':False,'new_portable_package':False})
    save(out/'protocol.json',{'id':'VM39_MATCHED_SEEDS_V1','model':'YOLO26s','seeds_in_order':[43,44],
        'reuse_seed42':True,'arms':list(ARMS),'imgsz':960,'epochs':10,'training_images':6471,'development_images':548,
        'policy_SHA256':study.sha256(previous_guided()/'policy.json'),'policy_tuning':False,
        'scope':SCOPE,'formal1280_100_epochs':'UNCHANGED_NOT_EXECUTED',
        'cross_seed_statistics':'Matched differences, sample SD and range; image bootstrap is not training-seed significance.'})
    files=[SOURCE,HERE/'report.py',Path(guided.__file__),Path(recovery.__file__),*old.source_paths(),
        out/'author_authorization.json',out/'resource_admission.json',out/'protocol.json',
        previous_guided()/'policy.json',previous_guided()/'final_acceptance.json',previous_guided()/'report_review.json',
        previous_guided()/'evaluation/result_index.json',
        study.run_path(guided.REFERENCE_ID)/'evaluation/result_index.json']
    save(out/'execution_freeze.json',{'files':{str(p):study.sha256(p) for p in files},
        'frozen_UTC':datetime.now(timezone.utc).isoformat()})
    (out/'diary.md').write_text('# VM39\n\nConfirm frozen VM38 policy across seeds. Reuse seed42. Seed44 requires complete-pair resource admission after seed43. Development only; no policy tuning.\n',encoding='utf-8')
    print('VM39_PREPARED',out,flush=True)
    return out


def execute(parent):
    import math
    study.runtime(parent,cuda=False)
    check_parent(parent)
    if old.available(study.run_path(recovery.AUTH))<sum(CAPS.values()):
        raise PermissionError('The full seed43 pair no longer fits.')
    lock=study.RUNS/'.pilot10_gpu.lock'
    if lock.exists() or (study.RUNS/'.gpu.lock').exists():raise RuntimeError('Existing GPU lock.')
    old.resource_check()
    save(parent/'execution_ticket.json',{'RunID':parent.name,'PID':os.getpid(),'attempt':1,
        'started_UTC':datetime.now(timezone.utc).isoformat(),'automatic_retry':False})
    save(lock,{'RunID':parent.name,'PID':os.getpid()})
    completed=[];attempted=[];status='PARTIAL_WITH_LIMITATIONS';error=None;caps=CAPS.copy()
    try:
        for seed in (43,44):
            remaining=old.available(study.run_path(recovery.AUTH))
            save(parent/('seed'+str(seed)+'_admission.json'),{'seed':seed,'caps':caps,'required_seconds':sum(caps.values()),
                'remaining_seconds':remaining,'admitted':remaining>=sum(caps.values()),'policy_or_AP_used_to_set_caps':False})
            if remaining<sum(caps.values()):
                error='SEED44_RESOURCE_NOT_ADMITTED_COMPLETE_PAIR_DOES_NOT_FIT'
                break
            out=make_seed(parent,seed,caps);attempted.append(out.name)
            save(parent/('seed'+str(seed)+'_ticket.json'),{'RunID':out.name,'caps':caps})
            for arm in ARMS:
                stage(out,out/('YOLO26s_seed'+str(seed)+'_'+arm),arm,caps[arm],'train')
            stage(out,out/'evaluation','evaluation',caps['evaluation'],'evaluate')
            accept_seed(out);completed.append(out.name)
            caps={**CAPS,**{name:max(CAPS[name],60*math.ceil(load(out/('evaluation' if name=='evaluation' else 'YOLO26s_seed'+str(seed)+'_'+name)/'safety_supervisor.json')['wall_seconds']*1.3/60))
                           for name in (*ARMS,'evaluation')}}
        if len(completed)==2:status='VM39_THREE_SEED_COMPLETE_DEVELOPMENT_ONLY'
    except BaseException as exc:
        error=str(exc)
        save(parent/'failure.json',{'error':error,'traceback':traceback.format_exc(),'automatic_retry':False})
    finally:
        if lock.exists() and load(lock).get('RunID')==parent.name:lock.unlink()
        charges=[load(p) for p in old.charge_files(study.run_path(recovery.AUTH)) if load(p)['RunID'] in attempted]
        save(parent/'execution_result.json',{'status':status,'error':error,'completed_seed_RunIDs':completed,
            'attempted_seed_RunIDs':attempted,'new_worker_seconds':sum(x['charged_seconds'] for x in charges),
            'budget':budget(),'scope':SCOPE})
        report_spec=importlib.util.spec_from_file_location('vm39_report',HERE/'report.py')
        reporter=importlib.util.module_from_spec(report_spec)
        report_spec.loader.exec_module(reporter)
        reporter.summarize(parent,sys.modules[__name__])
    print('VM39_FINISHED',status,parent,flush=True)
    return 0 if len(completed)==2 else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--execute')
    parser.add_argument('--worker',type=Path)
    parser.add_argument('--mode',choices=('train','evaluate'),default='train')
    args=parser.parse_args()
    if args.worker:raise SystemExit(0 if worker(args.worker.resolve(strict=True),args.mode) in ('TRAINING_COMPLETED','COMPLETE') else 1)
    if args.prepare:prepare()
    elif args.execute:raise SystemExit(execute(path(args.execute)))
    else:parser.error('Choose --prepare or --execute RunID.')
