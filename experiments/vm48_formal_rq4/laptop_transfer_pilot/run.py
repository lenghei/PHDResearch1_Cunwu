"""Matched evaluation of saved 960/10-epoch models on VM45 pixels."""
import argparse,csv,gc,gzip,hashlib,importlib.util,json,math,os,secrets,shutil,subprocess,sys,time,traceback
from datetime import datetime,timedelta,timezone
from pathlib import Path
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
EXPERIMENTS=ROOT/'main/experiments/experiments_test'
RUNS=ROOT/'datasets/local_runs/vm48_formal_rq4'
sys.path[:0]=[str(ROOT/'main/code'),str(EXPERIMENTS/'vm43_external_pilot')]
import vm43_common as accepted

def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

engine=module('vm48_transfer_accepted_engine',EXPERIMENTS/'vm45_motion_haze_gap_learning/execute_vm45.py')
engine.RUNS=RUNS
backend=engine.backend
READ=accepted.load;SHA=accepted.sha;SAVE=engine.save
VM39=ROOT/'datasets/local_runs/vm39_multi_seed_confirmation/vm39_multi_seed_confirmation_confirmation_20261115T095834_9623ace0'
VM45=ROOT/'datasets/local_runs/vm45_motion_haze_gap_learning/vm45_motion_haze_gap_learning_execution_20261117T172830_0d9cb1ee'
AUTH_ID='VM48_LAPTOP_TRANSFER_24CELL_AUTHOR_DECISION_20261118_V1'
SOURCES={}

def source(path):
    p=Path(path).resolve(strict=True);r={'path':str(p),'SHA256':SHA(p),'bytes':p.stat().st_size};SOURCES[str(p)]=r;return r

def csv_write(path,rows):
    if not rows:return
    with Path(path).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def charge(out,stage,seconds,result=None):
    p=accepted.AUTH/'charges'/(out.name+'_'+stage+'.json')
    with p.open('x',encoding='utf-8') as f:json.dump({'RunID':out.name,'stage':'VM48_LAPTOP_'+stage,'charged_seconds':seconds,'supervisor':result,'budget_reset':False},f,indent=2)
    return source(p)

def freeze(out):
    import torch,numpy as np,cv2
    from main.training_health import model_health
    from main.labels import VIS_NAMES,ground_truth
    torch.set_num_threads(2);cv2.setNumThreads(2)
    index=READ(VM39/'joint_result_index.json');source(VM39/'joint_result_index.json');source(VM39/'final_acceptance.json')
    checkpoints=[];plans={};structure=None
    for seed in (42,43,44):
        for arm in ('CONTROL','VARIABILITY_GUIDED'):
            row=next(r for r in index if r['seed']==seed and r['arm']==arm and r['context']=='Normal' and r['resolution']==1280)
            parent=Path(row['source_folder']).parent.parent
            folder=parent/f'YOLO26s_seed{seed}_{arm}'
            receipt=READ(folder/'training_acceptance.json');plan=READ(folder/'protocol.json')
            source(folder/'training_acceptance.json');source(folder/'protocol.json')
            ck=receipt['selected_checkpoint'];path=Path(ck['path'])
            assert path.name=='epoch0010_fp32.pt' and SHA(path)==ck['sha256']==row['checkpoint_SHA256']
            assert all(receipt[k]==v for k,v in {'completed_epochs':10,'anchor_exposures':64710,'forward_batches':32360,'backward_batches':32360,'optimizer_updates':1816,'EMA_updates':1816}.items())
            assert len(receipt['epoch_checkpoints'])==10
            assert plan['args']['seed']==seed and plan['args']['epochs']==10 and plan['args']['imgsz']==960
            assert plan['args']['amp'] is False and plan['args']['batch']==2 and plan['args']['workers']==0 and not plan['args']['resume']
            payload=torch.load(path,map_location='cpu',weights_only=False);ema=payload['ema'];assert payload['epoch']==9 and ema is not None
            model_health(ema);assert list(ema.names.values())==VIS_NAMES
            assert payload['rq4_provenance']['initialization_SHA256']==plan['initialization']['sha256']
            assert all(t.dtype==torch.float32 for t in ema.state_dict().values() if t.is_floating_point())
            signature=[(k,list(t.shape)) for k,t in ema.state_dict().items()]
            if structure is None:structure=signature
            assert signature==structure
            if arm=='CONTROL':assert plan['targeted_recipe'] is None
            else:
                noise=plan['targeted_recipe']['families']['NoiseAug']
                assert set(plan['targeted_recipe']['families'])=={'NoiseAug'} and noise['probability']==.25 and noise['gaussian_probability']==1
                assert noise['sigma']['low']==5.5 and noise['sigma']['high']==30
            plans[seed,arm]=plan
            checkpoints.append({'seed':seed,'strategy':arm,'architecture':'YOLO26s','checkpoint':str(path),'SHA256':ck['sha256'],'initialization_SHA256':plan['initialization']['sha256'],
                'source_RunID':parent.name,'training_protocol':str(folder/'protocol.json'),'training_protocol_SHA256':SHA(folder/'protocol.json'),'receipt_SHA256':SHA(folder/'training_acceptance.json'),
                'epoch':10,'training_resolution':960,'FP32':True,'native10':True,'health':'PASS','FM_D_shared_features':'YOLO26;Small;VisDrone_FT;Infer1280;FullFrame;NMS;FP32',
                'FM_D_training_resolution':'OUTSIDE_ORIGINAL_DOMAIN:Train960 is an exploratory extension; never relabel as Train1280','NoiseAug':arm!='CONTROL'})
            source(path);del ema,payload;gc.collect()
        c=plans[seed,'CONTROL'];g=plans[seed,'VARIABILITY_GUIDED']
        assert c['initialization']['sha256']==g['initialization']['sha256']
        assert c['train']==g['train'] and c['development']==g['development'] and c['train_labels']==g['train_labels'] and c['val_labels']==g['val_labels']
        exclude={'data','project','name'}
        assert {k:v for k,v in c['args'].items() if k not in exclude}=={k:v for k,v in g['args'].items() if k not in exclude}
    original=READ(VM45/'experiment_protocol.json');source(VM45/'experiment_protocol.json');source(VM45/'parameter_binding_review.json')
    assert SHA(original['GT']['path'])==original['GT']['SHA256'];source(original['GT']['path'])
    records=READ(original['GT']['path'])['records'];assert len(records)==548
    assert ground_truth(records,'native10')[0]==READ(original['GT']['path'])['coco']
    cache=READ(VM45/'pixel_cache_manifest.json');source(VM45/'pixel_cache_manifest.json');pixel_rows=[]
    for condition in original['conditions']:
        entries=cache[condition['condition']];assert len(entries)==548
        assert [r['image_id'] for r in entries]==[r['id'] for r in records]
        for i,(entry,r) in enumerate(zip(entries,records)):
            p=Path(entry['path']);assert SHA(p)==entry['file_SHA256']
            image=cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
            assert image.shape==tuple(entry['shape'])==(r['height'],r['width'],3)
            assert hashlib.sha256(image.tobytes()).hexdigest()==entry['pixel_SHA256']
            assert entry['source_image_SHA256']==original['population']['images'][i]['source_image_SHA256']
            assert entry['annotation_SHA256']==original['population']['images'][i]['annotation_SHA256']
            pixel_rows.append({'condition':condition['condition'],'P_id':condition['P_id'],'image_id':r['id'],'stem':r['stem'],'path':entry['path'],'file_SHA256':entry['file_SHA256'],'pixel_SHA256':entry['pixel_SHA256'],'source_image_SHA256':entry['source_image_SHA256'],'annotation_SHA256':entry['annotation_SHA256']})
    csv_write(out/'checkpoint_manifest.csv',checkpoints);csv_write(out/'pixel_manifest.csv',pixel_rows)
    for n,h in [('fm_p.uvl','e6eadd633d1c14e0d6bc425f5e54421d592d766c65a6e5f20c61bc34dd76fb5c'),('fm_d.uvl','4022d674822b97dbbe3712646b9a0f970af95d175ae358e63582029bc107571a')]:
        p=EXPERIMENTS/'vm36_supervisor_reproduction/feature_models'/n;assert SHA(p)==h;source(p)
    spec={**original,'id':'VM48_LAPTOP_GAUSSIAN_TRANSFER_V1','checkpoints':checkpoints,'models':'Six saved matched epoch10 EMAs; no historical ck01/ck05 substitution',
        'seed_order':[42,43,44],'strategy_order':['CONTROL','VARIABILITY_GUIDED'],'cells_per_group':8,'maximum_cells':24,'maximum_image_records':13152,
        'training':'NO_NEW_TRAINING; saved960/10epoch policy only','cache_source':source(VM45/'pixel_cache_manifest.json'),'cache_reused_without_copy':True,
        'order':'Complete seed42 then43 then44; condition Normal,Motion,Haze,Composite; CONTROL thenGUIDED',
        'FP_image_definition':'COCO IoU0.50, all-area, maxDet3000, score>=0.001 after frozen ignore-union; unmatched nonignored detections /548',
        'GPU_warmup_policy':'3 Normal warmups per model immediately before its first timed cell; counted separately and charged',
        'Gap_definition':'Separate singleton CONTROL and GUIDED detector sets atInfer1280; Tiny interval + VM41 upper latency CI rule; no full-space prevalence',
        'budget_policy':{'untouched_shutdown_reserve_seconds':1200,'final_report_allowance_seconds':600,'next_group_margin_factor':1.35,'initial_VM45_worst_cell_factor':1.5,
            'selection_uses_AP':False,'group_boundary_only':True,'no_retry':True},
        'controller_policy':'Fixed baselines; retrospective exact-P verified evidence lookup (in-sample); unknown-P safe controller always ABSTAIN; no refit or deployment authorization',
        'limitations':['DEVELOPMENT_ONLY','POST_HOC_POLICY_SELECTION','Train960 outside original FM_D TrainResolution domain','Image Bootstrap is not training-seed significance','Not formal1280/100epoch RQ4']}
    SAVE(out/'protocol.json',spec);SAVE(out/'cache_reuse_acceptance.json',{'images':548,'conditions':4,'PNG_files_verified':2192,'file_and_pixel_SHA':True,'GT_exact':True,'new_images_generated':0})
    sources=[Path(__file__),HERE/'finish.py',EXPERIMENTS/'vm45_motion_haze_gap_learning/execute_vm45.py',EXPERIMENTS/'vm43_external_pilot/backend.py',EXPERIMENTS/'vm43_external_pilot/vm43_common.py',ROOT/'main/code/main/statistics.py',ROOT/'main/code/main/labels.py',ROOT/'main/code/main/predictor_stable.py']
    for p in sources:source(p)
    SAVE(out/'source_provenance.json',{'files':list(SOURCES.values()),'accepted_engine_reused_without_edits':True,'formal_VM48_package_not_modified':True})
    return spec

def evaluation(raw,records,seed):
    import numpy as np
    import main.statistics as stats
    original=stats.evaluate_coco;fp={}
    def observe(*args,**kwargs):
        result=original(*args,**kwargs);e=result[2];all_area=e.params.areaRng[e.params.areaRngLbl.index('all')]
        counts={r['id']:0 for r in records}
        for r in e.evalImgs:
            if r is not None and list(r['aRng'])==list(all_area) and r['maxDet']==3000:
                unmatched=(np.asarray(r['dtMatches'])[0]==0)&~np.asarray(r['dtIgnore'])[0].astype(bool)
                counts[r['image_id']]+=int(unmatched.sum())
        fp['per_image']=[counts[r['id']] for r in records];return result
    with patch.object(stats,'evaluate_coco',observe):metrics,boot,cost=engine.evaluation(raw,records,seed)
    metrics['FP_IoU50_per_image']=float(np.mean(fp['per_image']));metrics['FP_IoU50_counts_by_image']=fp['per_image']
    return metrics,boot,cost

def worker(out,seed):
    token=os.environ.get('VM48_TRANSFER_TOKEN','');ticket=READ(out/f'group_{seed}_lease.json')
    assert token and hashlib.sha256(token.encode()).hexdigest()==ticket['token_SHA256']
    assert SHA(out/'protocol.json')==ticket['protocol_SHA256'] and datetime.now(timezone.utc)<datetime.fromisoformat(ticket['deadline_UTC'])
    spec=READ(out/'protocol.json')
    for r in READ(out/'source_provenance.json')['files']:
        if r['path'].endswith('.py'):assert SHA(r['path'])==r['SHA256']
    accepted.study.runtime(out/f'runtime_{seed}',cuda=True)
    import cv2,numpy as np,torch
    torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    cache=READ(spec['cache_source']['path']);records=READ(spec['GT']['path'])['records'];assert SHA(spec['GT']['path'])==spec['GT']['SHA256']
    models=[];start=time.perf_counter();group=out/f'group_{seed}';cells=[];warmups=0
    for row in spec['checkpoints']:
        if row['seed']!=seed:continue
        assert SHA(row['checkpoint'])==row['SHA256'];adapter=backend.RealDetector(row['checkpoint'])
        image=cv2.imread(cache['Normal'][0]['path'])
        for _ in range(3):adapter.predict(image);warmups+=1
        adapter.synchronize();models.append((row,adapter));print('MODEL_READY',seed,row['strategy'],flush=True)
    for cond in spec['conditions']:
        for row,adapter in models:
            if shutil.disk_usage(out).free<6*1024**3:raise RuntimeError('Less than6GiB disk reserve; stop without retry.')
            cell_id=f"s{seed}_{row['strategy']}_{cond['condition']}";folder=group/cell_id;folder.mkdir();print('CELL_START',cell_id,flush=True)
            torch.cuda.reset_peak_memory_stats();times=[];partial=folder/'predictions.jsonl.gz.partial';raw=folder/'predictions.jsonl.gz'
            with gzip.open(partial,'xt',encoding='utf-8') as f,(folder/'image_manifest.jsonl').open('x',encoding='utf-8') as mf:
                for entry in cache[cond['condition']]:times.append(backend.timed_prediction(entry,adapter,f,mf))
            with gzip.open(partial,'rt',encoding='utf-8') as f:backend.validate_predictions([json.loads(s) for s in f],{r['id'] for r in records})
            partial.rename(raw);metrics,boot,cost=evaluation(raw,records,spec['bootstrap_seed'])
            SAVE(folder/'metrics.json',metrics);SAVE(folder/'bootstrap.json',boot)
            SAVE(folder/'timing.json',{'per_image':times,'evaluation_cost':cost,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'timing_protocol':'VM41_MATCHED_CACHED_GAP_V1'})
            acceptance={'status':'REAL_CELL_COMPLETED','seed':seed,'strategy':row['strategy'],'condition':cond['condition'],'P_id':cond['P_id'],'checkpoint_SHA256':row['SHA256'],'images':548,
                'raw_SHA256':SHA(raw),'metrics_SHA256':SHA(folder/'metrics.json'),'bootstrap_SHA256':SHA(folder/'bootstrap.json'),'timing_SHA256':SHA(folder/'timing.json'),'GT_SHA256':spec['GT']['SHA256'],'pixel_manifest_SHA256':spec['cache_source']['SHA256'],'scope':'DEVELOPMENT_ONLY'}
            SAVE(folder/'acceptance.json',acceptance);cells.append(cell_id);print('CELL_COMPLETE',cell_id,flush=True)
    assert len(cells)==8
    SAVE(out/f'group_{seed}_result.json',{'status':'COMPLETE_MATCHED_SEED_GROUP','seed':seed,'cells':cells,'image_records':4384,'warmup_image_records':warmups,'worker_wall_seconds':time.perf_counter()-start})

def supervised(command,folder,cap,env,policy):
    from safety import decide,stop_owned
    first=engine.query();assert first['temperature_C']<=policy['preflight_maximum_temperature_C'] and first['free_MiB']>=policy['preflight_minimum_free_MiB']
    folder.mkdir();started=time.perf_counter();samples=[];reason=None
    with (folder/'worker.log').open('x',encoding='utf-8') as log,(folder/'gpu_telemetry.jsonl').open('x',encoding='utf-8') as trace:
        p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        try:
            while p.poll() is None:
                tick=time.perf_counter()
                if tick-started>=cap-15:reason='HARD_BUDGET_STOP'
                else:
                    try:
                        r=engine.query();r['monotonic_seconds']=time.perf_counter()-started
                        if samples and r['monotonic_seconds']-samples[-1]['monotonic_seconds']>2:reason='TELEMETRY_INTERVAL_EXCEEDED'
                        samples.append(r);trace.write(json.dumps(r)+'\n');trace.flush();reason=reason or decide(policy,samples)
                    except BaseException:reason='TELEMETRY_FAILURE'
                if reason:stop_owned(p);break
                time.sleep(max(0,1-(time.perf_counter()-tick)))
            code=p.wait(timeout=15)
        except BaseException:stop_owned(p);raise
    r={'returncode':code,'stop_reason':reason,'worker_wall_seconds':time.perf_counter()-started,'peak_temperature_C':max((x['temperature_C'] for x in samples),default=None),'attempt':1,'retry':False}
    SAVE(folder/'supervisor.json',r);return r

def physical(out):
    import ctypes,psutil
    if shutil.disk_usage(out).free<6*1024**3 or psutil.virtual_memory().available<6*1024**3:raise RuntimeError('Disk or RAM admission failed.')
    device=subprocess.run(['nvidia-smi','--query-gpu=uuid,name,memory.free,temperature.gpu,driver_version','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=2)
    assert len(device.stdout.strip().splitlines())==1 and 'GPU-1dcc4d8e-3346-93c1-53db-f899d1f9efaf' in device.stdout
    processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=2)
    assert not any('python' in x.lower() for x in processes.stdout.splitlines())
    class Power(ctypes.Structure):_fields_=[('ACLineStatus',ctypes.c_ubyte),('BatteryFlag',ctypes.c_ubyte),('BatteryLifePercent',ctypes.c_ubyte),('SystemStatusFlag',ctypes.c_ubyte),('BatteryLifeTime',ctypes.c_ulong),('BatteryFullLifeTime',ctypes.c_ulong)]
    power=Power();assert ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power)) and power.ACLineStatus==1
    p=accepted.study.RUNS/'vm37_yolo26_rq4_pilot960_authorization_record_cpu_20261114T110935_8e5d6543/physical_and_execution_authorization.json';r=READ(p)
    assert datetime.now(timezone.utc)<datetime.fromisoformat(r['expiry_UTC']) and all(r[k] for k in ['hard_ventilated_surface','unobstructed_intake_exhaust','fans_normal','reliable_AC_adapter'])
    return {'device':device.stdout,'GPU_processes':processes.stdout,'AC_online':True,'free_disk_bytes':shutil.disk_usage(out).free,'available_RAM_bytes':psutil.virtual_memory().available,'valid_prior_physical_record':source(p),'new_physical_confirmation':False,'unknown':['current room temperature','fan RPM','OEM mode']}

def execute():
    for p in RUNS.glob('laptop_transfer_pilot_*/author_authorization.json'):
        if READ(p).get('authorization_id')==AUTH_ID:raise PermissionError('This authorization already has an attempt; no retry.')
    from main.pipeline import new_run
    out,log=new_run({'output_root':str(RUNS),'scope':'VM48_LAPTOP_TRANSFER_PILOT_ONLY'},'laptop_transfer_pilot','execution')
    start=time.perf_counter();prep_charged=False;lock=accepted.study.RUNS/'.pilot10_gpu.lock';owned=False;failure=None
    try:
        SAVE(out/'author_authorization.json',{'authorization_id':AUTH_ID,'source':'CURRENT_USER_EXPLICIT_VM48_LAPTOP_FIRST_RQ4_PILOT_MESSAGE_20261118','scope':'Saved960/10epoch26s CONTROL/GUIDED seeds42,43,44;4VM45P;Infer1280 FP32 batch1;max24cells',
            'quote':'本次授权在原剩余GPU预算和截止期限内，有条件执行下述冻结的小规模真实评价。','original_allocation':accepted.budget()['authorization'],'new_budget':False,'expiry_extended':False,'training':False,'retry':False,'shutdown_reserve_seconds':1200})
        before=accepted.budget();SAVE(out/'budget_before.json',before);assert before['remaining_seconds']>1800
        spec=freeze(out);SAVE(out/'initial_device_admission.json',physical(out))
        if lock.exists() or (accepted.study.RUNS/'.gpu.lock').exists():raise RuntimeError('Another project GPU owner lock exists.')
        with lock.open('x',encoding='utf-8') as f:json.dump({'RunID':out.name,'PID':os.getpid()},f)
        owned=True
        policy=READ(accepted.BUNDLE/'safety_policy.json');assert (policy['immediate_stop_temperature_C'],policy['sustained_stop_temperature_C'],policy['sustained_seconds'])==(90,87,30)
        charge(out,'CPU_preparation',time.perf_counter()-start);prep_charged=True
        measured=[]
        for seed in (42,43,44):
            remaining=accepted.budget();cap_base=None
            if not measured:
                historical=READ(VM45/'actual_resource_usage.json');worst=max(r['cached_pipeline_seconds']+sum(r['CPU_evaluation_cost'].values()) for r in historical['cell_costs']);cap_base=8*worst*1.5+120
            else:cap_base=max(measured)*1.35+120
            cap=math.ceil(cap_base/60)*60
            fits=cap+1800<=remaining['remaining_seconds'] and datetime.now(timezone.utc)+timedelta(seconds=cap+1800)<datetime.fromisoformat(remaining['deadline_UTC'])
            SAVE(out/f'group_{seed}_admission.json',{'seed':seed,'group_cap_seconds':cap,'reserve_untouched_seconds':1200,'final_CPU_allowance_seconds':600,'fits_complete_group':fits,'budget':remaining,'basis':'VM45 worst cell x8 x1.5 +120 beforeA; max completed group wall x1.35 +120 thereafter','AP_used':False,'current_device':physical(out) if fits else None})
            if not fits:
                SAVE(out/'group_boundary_stop.json',{'seed_not_started':seed,'reason':'Complete next seed group plus reserve does not fit','AP_based':False});break
            token=secrets.token_hex(24);now=datetime.now(timezone.utc)
            SAVE(out/f'group_{seed}_lease.json',{'seed':seed,'token_SHA256':hashlib.sha256(token.encode()).hexdigest(),'protocol_SHA256':SHA(out/'protocol.json'),'deadline_UTC':(now+timedelta(seconds=cap)).isoformat(),'attempt':1,'cap_seconds':cap})
            cmd=[sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),'--worker',out.name,'--seed',str(seed)]
            SAVE(out/f'group_{seed}_command.json',{'command':cmd});log.info('Starting complete seed%s eight-cell group; cap=%ss.',seed,cap)
            tick=time.perf_counter();result=None
            try:result=supervised(cmd,out/f'group_{seed}',cap,dict(os.environ,VM48_TRANSFER_TOKEN=token),policy)
            finally:charge(out,f'group_{seed}',time.perf_counter()-tick,result)
            if result['stop_reason'] or result['returncode'] or not (out/f'group_{seed}_result.json').exists():raise RuntimeError('Group stopped; no retry. '+str(result))
            assert READ(out/f'group_{seed}_result.json')['status']=='COMPLETE_MATCHED_SEED_GROUP';measured.append(result['worker_wall_seconds'])
            log.info('Seed%s group completed; charged wall=%.1fs; remaining=%.1fs.',seed,measured[-1],accepted.budget()['remaining_seconds'])
        SAVE(out/'execution_end.json',{'completed_seeds':[s for s in (42,43,44) if (out/f'group_{s}_result.json').exists()],'budget':accepted.budget(),'GPU_training':0})
    except BaseException:
        failure=traceback.format_exc();SAVE(out/'execution_failure.json',{'traceback':failure,'retry':False});print(failure,flush=True)
    finally:
        if not prep_charged:charge(out,'CPU_preparation',time.perf_counter()-start)
        if owned and lock.exists() and READ(lock).get('RunID')==out.name:lock.unlink()
    charged=sum(READ(r['path'])['charged_seconds'] for r in accepted.budget()['charges'] if Path(r['path']).name.startswith(out.name+'_'))
    charge(out,'CPU_orchestration',max(0,time.perf_counter()-start-charged))
    remaining=accepted.budget()['remaining_seconds'];cap=min(600,remaining-1200)
    if cap>15:
        tick=time.perf_counter();result=None
        try:
            cmd=[sys.executable,'-B','-X','utf8',str(HERE/'finish.py'),'--run',str(out)]
            with (out/'final_analysis.log').open('x',encoding='utf-8') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=cap)
        finally:charge(out,'CPU_final_analysis',time.perf_counter()-tick,{'returncode':result.returncode if result else None})
        if result and result.returncode:raise RuntimeError('Final analysis failed; retain evidence and inspect log.')
        final=module('vm48_transfer_report_seal',HERE/'finish.py')
        final.seal_after_charge(out)
    print('VM48_LAPTOP_PILOT_FINISHED',out,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--worker');p.add_argument('--seed',type=int,choices=[42,43,44]);a=p.parse_args()
    if a.worker:
        if Path(a.worker).name!=a.worker or not a.worker.startswith('laptop_transfer_pilot_execution_'):raise PermissionError('Invalid owned RunID.')
        out=(RUNS/a.worker).resolve(strict=True)
        try:worker(out,a.seed)
        except BaseException:SAVE(out/f'group_{a.seed}_failure.json',{'traceback':traceback.format_exc(),'retry':False});raise
    elif a.execute:execute()
    else:p.error('Explicit current author authorization requires --execute; no automatic execution.')

if __name__=='__main__':main()
