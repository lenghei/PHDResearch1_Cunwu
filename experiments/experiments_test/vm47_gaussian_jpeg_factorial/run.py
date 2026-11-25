"""One VM47 eight-cell attempt using the accepted VM45 execution primitives."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
RUNS=ROOT/'datasets/local_runs'/HERE.name
sys.path.insert(0,str(ROOT/'main/code'))
sys.path.insert(0,str(HERE.parent/'vm43_external_pilot'))
import vm43_common as accepted


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


ENGINE_PATH=HERE.parent/'vm45_motion_haze_gap_learning/execute_vm45.py'
engine=module('vm47_accepted_execution_engine',ENGINE_PATH)
# Scope the adapter's writes to VM47; the historical source file stays unchanged.
engine.RUNS=RUNS
READ=accepted.load
SHA=accepted.sha
SAVE=engine.save
SOURCES={}
VM46=ROOT/'datasets/local_runs/vm46_heldout_variability/vm46_heldout_variability_admission_cpu_20261117T184741_6d462e31'
VM45=ROOT/'datasets/local_runs/vm45_motion_haze_gap_learning/vm45_motion_haze_gap_learning_execution_20261117T172830_0d9cb1ee'
REQUEST=Path(r'C:\Users\w\.codex\attachments\ae215acb-445d-4424-968b-4cc7a9ba73eb\已粘贴的文本.txt')
AUTH_ID='VM47_EXACT8_AUTHOR_ATTACHMENT_'+SHA(REQUEST)


def source(path):
    p=Path(path).resolve(strict=True)
    result={'path':str(p),'SHA256':SHA(p),'bytes':p.stat().st_size}
    SOURCES[str(p)]=result
    return result


def sealed(path,seal):
    records={}
    for line in Path(seal).read_text(encoding='utf-8').splitlines():
        h,name=line.split('  ',1);p=Path(name)
        records[str(p.resolve() if p.is_absolute() else (Path(seal).parent/p).resolve())]=h
    assert SHA(path)==records[str(Path(path).resolve())],str(path)
    return source(path)


def freeze(out):
    for name in ('protocol_freeze.json','frozen_rule_protocol.json','Heldout_rule_predictions.csv','Final_acceptance.json'):
        sealed(VM46/name,VM46/'SHA256SUMS.txt')
    assert READ(VM46/'Final_acceptance.json')['real_cells']==0
    parent=READ(VM46/'protocol_freeze.json')
    conditions=[]
    for c,name in zip(parent['conditions'][:4],('Normal','Gauss_s15','JPEG_q25','Gauss_s15_JPEG_q25')):
        conditions.append({**c,'condition':name})
    assert [set(c['leaves']) for c in conditions]==[{'Normal'},{'Normal','Gauss_s15'},{'Normal','JPEG_q25'},{'Normal','Gauss_s15','JPEG_q25'}]
    spec={**parent,'id':'VM47_FOCUSED_GAUSSIAN_JPEG_8CELL_V1','conditions':conditions,'images':548,'cells':8,'inferences':4384,'formal_inferences':4384,
          'scientific_scope':'DEVELOPMENT_ONLY_SYNTHETIC_EXPLORATORY','reuse_old_Normal':False,
          'order':'Normal,Gauss_s15,JPEG_q25,Gauss_s15_JPEG_q25; each ck05 then ck01',
          'smaller_distinct_experiment':'Prospectively specified VM47; not a completed or edited VM46 evaluation',
          'frozen_rule_source':source(VM46/'frozen_rule_protocol.json'),'parameter_label':'ORIGINAL_UVL_VALUES_EXISTING_SYNTHETIC_IMPLEMENTATION',
          'physical_calibration':False,'rule_safety_admission':'No deployment promotion; diagnostic on one unseen complete P only; safe controller ABSTAIN'}
    rules=READ(engine.V44/'FM_P_to_FM_D_decision_rules.json')
    sys.path.insert(0,str(HERE.parent/'vm44_tiny_error_variability_decision'))
    from selector import legal
    for c in conditions:
        assert legal(rules['models']['fm_p'],c['leaves'])
    for d in spec['models']:
        assert SHA(d['checkpoint']['path'])==d['checkpoint']['sha256'];source(d['checkpoint']['path'])
    for name,h in [('fm_p.uvl','e6eadd633d1c14e0d6bc425f5e54421d592d766c65a6e5f20c61bc34dd76fb5c'),('fm_d.uvl','4022d674822b97dbbe3712646b9a0f970af95d175ae358e63582029bc107571a')]:
        path=HERE.parent/'vm36_supervisor_reproduction/feature_models'/name;assert SHA(path)==h;source(path)
    source(spec['GT']['path']);assert SHA(spec['GT']['path'])==spec['GT']['SHA256']
    source(spec['population']['source_ZIP_accepted_certificate']['path'])
    SAVE(out/'experiment_protocol.json',spec)
    import csv
    with (VM46/'Heldout_rule_predictions.csv').open(encoding='utf-8-sig') as f:old=list(csv.DictReader(f))
    predictions=[]
    for c in conditions:
        for d in spec['models']:
            previous=next(r for r in old if r['P_id']==c['P_id'] and r['D']==d['D'])
            predictions.append({**previous,'condition':c['condition'],'unseen_full_P':c['condition']=='Gauss_s15_JPEG_q25',
                                'VM47_verified_state':'NOT_EVALUATED','frozen_in_VM46':True})
    engine.csv_write(out/'frozen_rule_predictions.csv',predictions)
    rule=READ(VM46/'frozen_rule_protocol.json')
    assert SHA(rule['source']['path'])==rule['source']['SHA256'];source(rule['source']['path'])
    SAVE(out/'frozen_rule_protocol.json',rule)
    SAVE(out/'freeze_receipt.json',{'UTC':datetime.now(timezone.utc).isoformat(),'protocol_SHA256':SHA(out/'experiment_protocol.json'),
          'predictions_SHA256':SHA(out/'frozen_rule_predictions.csv'),'source_VM46_protocol':source(VM46/'protocol_freeze.json'),
          'future_AP_accessed':0,'no_refit':True,'unseen_complete_P':1,'unseen_P_D_pairs':2})
    source(ENGINE_PATH);source(HERE.parent/'vm43_external_pilot/backend.py');source(HERE.parent/'cv261109/generator.py')
    SAVE(out/'parameter_provenance.json',{'source_VM46':source(VM46/'protocol_freeze.json'),'original_UVL_numeric_values':True,
         'Gauss_s15':'Independent BGR sample noise Normal(0,15), float32 addition then clip/truncate uint8; not calibrated sensor noise',
         'JPEG_q25':'One extra OpenCV JPEG encode at quality25 after Gaussian, then fixed BGR decode; original image is already JPEG',
         'order':'Normal identity illumination, Gaussian noise, JPEG; unselected stages skipped',
         'image_seed':spec['image_seed'],'shared_noise':'Same per-image/per-feature hash seed in Gaussian-only and joint conditions',
         'GT':'Unchanged original coordinates, area memberships and ignore-union','no_physical_camera_equivalence':True})
    return spec,READ(spec['GT']['path'])


def cost_admission(out,phase):
    old=READ(VM45/'actual_resource_usage.json');source(VM45/'actual_resource_usage.json')
    previous=READ(VM45/'full_cost_admission.json');source(VM45/'full_cost_admission.json')
    worst=max(r['cached_pipeline_seconds']+sum(r['CPU_evaluation_cost'].values()) for r in old['cell_costs'])
    factor=previous['response_margin_factor'];assert factor==2
    cache_allowance=2*old['CPU_cache']['CPU_cache_wall_seconds']
    cap=math.ceil((worst*8*factor+cache_allowance+60+600)/60)*60
    b=accepted.budget();now=datetime.now(timezone.utc)
    total=cap+1800+(600 if phase=='before_preflight' else 0)
    assert total<=b['remaining_seconds'],'Complete eight-cell conservative budget does not fit.'
    assert now+timedelta(seconds=total)<datetime.fromisoformat(b['deadline_UTC']),'Original expiry cannot cover complete run.'
    assert b['authorization']['SHA256']=='20259f1f92c45eff11721363d4a02b0f01090e6c31c52b93c3e3db0bf3785986'
    for r in b['charges']:source(r['path'])
    source(b['authorization']['path'])
    result={'status':'ADMITTED_EMPIRICAL_CONSERVATIVE_PLANNING_ENVELOPE','phase':phase,'UTC':now.isoformat(),'worst_measured_VM45_cell_seconds':worst,
            'uncertainty_factor':factor,'formal_worker_cap_seconds':cap,'preflight_allowance_seconds':600 if phase=='before_preflight' else 0,
            'CPU_cache_allowance_included_seconds_NOT_GPU_MEASURED':cache_allowance,'startup_allowance_seconds':60,'verification_allowance_seconds':600,
            'shutdown_reserve_seconds':1800,'complete_remaining_requirement_seconds':total,'actual_budget':b,
            'not_a_guarantee':'Unknown Gaussian/JPEG response cost can vary; fixed cap stops partial work without retry. CPU-only cache and postprocessing recorded separately.'}
    SAVE(out/('cost_'+phase+'.json'),result)
    return cap


def stop_capability(out):
    import psutil
    from safety import stop_owned
    command=[sys.executable,'-B','-c','import subprocess,sys,time;subprocess.Popen([sys.executable,"-B","-c","import time;time.sleep(60)"]);time.sleep(60)']
    p=subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    descendants=[]
    try:
        for _ in range(40):
            descendants=psutil.Process(p.pid).children(recursive=True)
            if descendants:break
            time.sleep(.05)
        assert descendants,'Owned CPU termination fixture did not start its child.'
        stop_owned(p);p.wait(timeout=10)
        assert all(not child.is_running() for child in descendants)
        SAVE(out/'owned_process_stop_test.json',{'status':'PASS','GPU':False,'parent_PID':p.pid,'child_PIDs':[child.pid for child in descendants],
             'scope':'Only processes created by this test; no unrelated process enumeration or termination'})
    finally:
        if p.poll() is None:stop_owned(p)


def phase(out,mode,cap,token,policy):
    now=datetime.now(timezone.utc)
    SAVE(out/(mode+'_lease.json'),{'RunID':out.name,'mode':mode,'scope':'VM47_ONLY','worker_token_SHA256':hashlib.sha256(token.encode()).hexdigest(),
         'protocol_SHA256':SHA(out/'experiment_protocol.json'),'deadline_UTC':(now+timedelta(seconds=cap)).isoformat(),'cap_seconds':cap,'attempt':1})
    command=[sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),'--worker',out.name,'--mode',mode]
    SAVE(out/(mode+'_command.json'),{'command':command})
    start=time.perf_counter();result=None
    try:result=engine.supervised(command,out/mode,cap,dict(os.environ,VM47_WORKER_TOKEN=token),policy)
    finally:
        charge=accepted.AUTH/'charges'/(out.name+'_'+mode+'.json')
        with charge.open('x',encoding='utf-8') as f:
            json.dump({'RunID':out.name,'stage':'VM47_'+mode,'charged_seconds':time.perf_counter()-start,'supervisor':result,'budget_reset':False},f,indent=2)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--worker');p.add_argument('--mode',choices=['preflight','formal']);args=p.parse_args()
    if args.worker:
        if Path(args.worker).name!=args.worker or not args.worker.startswith(HERE.name+'_execution_'):raise ValueError('Invalid VM47 RunID.')
        out=(RUNS/args.worker).resolve(strict=True);token=os.environ.get('VM47_WORKER_TOKEN','')
        ticket=READ(out/(args.mode+'_lease.json'))
        if ticket['scope']!='VM47_ONLY' or not token or hashlib.sha256(token.encode()).hexdigest()!=ticket['worker_token_SHA256']:raise PermissionError('VM47 owned worker lease required.')
        # The accepted primitive uses this internal variable; no VM45 permission is reused.
        os.environ['VM45_WORKER_TOKEN']=token
        try:engine.worker(out,args.mode)
        except BaseException:
            SAVE(out/(args.mode+'_failure.json'),{'traceback':traceback.format_exc(),'retry':False});raise
        return
    if not args.execute:p.error('Use --execute only for the current explicit VM47 eight-cell author decision.')
    for previous in RUNS.glob('*/author_authorization.json'):
        if READ(previous).get('authorization_id')==AUTH_ID:raise PermissionError('VM47 authorization already has an attempt; no retry.')
    from main.pipeline import new_run
    out,logger=new_run({'output_root':str(RUNS),'scope':'VM47_EXACT_EIGHT_CELLS_ONLY'},HERE.name,'execution')
    (out/'runtime').mkdir();started=time.perf_counter();lock=accepted.study.RUNS/'.pilot10_gpu.lock';owned=False;failed=None
    try:
        request=REQUEST.read_bytes();(out/'author_request.txt').write_bytes(request)
        SAVE(out/'author_authorization.json',{'authorization_id':AUTH_ID,'scope':'VM47_ONLY_4P_2D_FULL548_FP32_BATCH1_NATIVE10','author_source':source(out/'author_request.txt'),
             'source_attachment':source(REQUEST),'author_quote':'I authorize VM47’s exact frozen eight-cell GPU evaluation conditionally, within the remaining original36-hour GPU ledger and fixed deadline.',
             'scientific_scope':'DEVELOPMENT_ONLY_SYNTHETIC_EXPLORATORY','training':False,'retry':False,'budget_reset':False,'expiry_extended':False})
        source(__file__);spec,payload=freeze(out)
        cost_admission(out,'before_preflight')
        import psutil,shutil
        assert psutil.virtual_memory().available>=8*1024**3 and shutil.disk_usage(out).free>=12*1024**3
        engine.cache(out,spec,payload)
        stop_capability(out)
        if lock.exists() or (accepted.study.RUNS/'.gpu.lock').exists():raise RuntimeError('Existing project GPU owner lock.')
        with lock.open('x',encoding='utf-8') as f:json.dump({'RunID':out.name,'PID':os.getpid()},f)
        owned=True
        device=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version,temperature.gpu,memory.total,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
        (out/'device_identity.txt').write_text(device.stdout,encoding='utf-8')
        assert len(device.stdout.strip().splitlines())==1 and 'GPU-1dcc4d8e-3346-93c1-53db-f899d1f9efaf' in device.stdout
        processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
        (out/'GPU_processes.txt').write_text(processes.stdout,encoding='utf-8')
        assert not any('python' in line.lower() for line in processes.stdout.splitlines()),'Competing Python GPU workload.'
        import ctypes
        class Power(ctypes.Structure):_fields_=[('ACLineStatus',ctypes.c_ubyte),('BatteryFlag',ctypes.c_ubyte),('BatteryLifePercent',ctypes.c_ubyte),('SystemStatusFlag',ctypes.c_ubyte),('BatteryLifeTime',ctypes.c_ulong),('BatteryFullLifeTime',ctypes.c_ulong)]
        power=Power();assert ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power)) and power.ACLineStatus==1
        physical=accepted.study.RUNS/'vm37_yolo26_rq4_pilot960_authorization_record_cpu_20261114T110935_8e5d6543/physical_and_execution_authorization.json'
        record=READ(physical);assert datetime.now(timezone.utc)<datetime.fromisoformat(record['expiry_UTC'])
        assert all(record[k] is True for k in ('hard_ventilated_surface','unobstructed_intake_exhaust','fans_normal','reliable_AC_adapter'))
        SAVE(out/'physical_and_device_admission.json',{'AC_online_actual':True,'existing_valid_user_physical_declaration':source(physical),
             'new_physical_reconfirmation':False,'current_device':source(out/'device_identity.txt'),'unknown':['room temperature','fan RPM','OEM mode']})
        policy=READ(accepted.BUNDLE/'safety_policy.json');assert (policy['immediate_stop_temperature_C'],policy['sustained_stop_temperature_C'],policy['sustained_seconds'],policy['poll_seconds'])==(90,87,30,2)
        sources=[Path(__file__).resolve(),ENGINE_PATH,HERE.parent/'vm43_external_pilot/backend.py',HERE.parent/'cv261109/generator.py',HERE.parent/'cv261109/model.py',ROOT/'main/code/main/statistics.py',ROOT/'main/code/main/labels.py',ROOT/'main/code/main/predictor_stable.py']
        for f in sources:source(f)
        SAVE(out/'frozen_worker_sources.json',[(str(f),SHA(f)) for f in sources])
        token=secrets.token_hex(24);probe=phase(out,'preflight',600,token,policy)
        assert probe['returncode']==0 and probe['stop_reason'] is None,'VM47 preflight failed; no retry.'
        assert READ(out/'preflight_result.json')['status']=='REAL_ENGINEERING_PROBE_PASS'
        cap=cost_admission(out,'after_preflight')
        SAVE(out/'formal_execution_ticket.json',{'status':'ALL_GATES_PASSED','scope':'VM47_ONLY','attempt':1,'source':source(out/'author_authorization.json'),
             'protocol':source(out/'experiment_protocol.json'),'dataset':source(out/'dataset_gate.json'),'cost':source(out/'cost_after_preflight.json'),
             'preflight':source(out/'preflight_result.json'),'supervisor':probe,'owned_process_stop':source(out/'owned_process_stop_test.json'),'no_retry':True})
        logger.info('All VM47 gates passed. Starting ONE eight-cell worker; cap=%ss.',cap)
        formal=phase(out,'formal',cap,token,policy)
        success=formal['returncode']==0 and formal['stop_reason'] is None and (out/'worker_result.json').exists()
        SAVE(out/'execution_end.json',{'status':'REAL_EIGHT_CELLS_COMPLETED' if success else 'INCOMPLETE','formal_supervisor':formal,
             'budget_after':accepted.budget(),'CPU_and_supervised_wall_seconds':time.perf_counter()-started,'retry':False})
        if not success:raise RuntimeError('VM47 eight-cell worker incomplete; no retry.')
        print('VM47_EIGHT_CELLS_FINISHED',out,flush=True)
    except BaseException:
        failed=traceback.format_exc()
        SAVE(out/'execution_failure.json',{'status':'BLOCKED_OR_INCOMPLETE','traceback':failed,'budget_after':accepted.budget(),'retry':False})
        print(failed,flush=True)
    finally:
        if owned and lock.exists() and READ(lock).get('RunID')==out.name:lock.unlink()
        for item in engine.SOURCES.values():SOURCES[item['path']]=item
        SAVE(out/'execution_source_provenance.json',{'files':list(SOURCES.values()),'historical_sources_unchanged':True,'adapter':'VM47_SCOPED_ACCEPTED_ENGINE_V1'})
    if failed:sys.exit(1)


if __name__=='__main__':main()
