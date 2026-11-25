"""CPU software validation; fixtures never constitute formal training evidence."""
import argparse,copy,csv,gc,gzip,hashlib,json,os,random,subprocess,sys,time,traceback
os.environ.update(CUDA_VISIBLE_DEVICES='-1',YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',NO_ALBUMENTATIONS_UPDATE='1')
from pathlib import Path
from datetime import datetime,timezone,timedelta
from common48 import *
import numpy as np


def run_checks(package,data_root,out,regression=None):
    out.mkdir(parents=True,exist_ok=False);base.runtime(out,gpu=False)
    checks=[];details={};started=time.perf_counter()
    def check(name,fn):
        tick=time.perf_counter()
        try:result=fn();checks.append({'test':name,'status':'PASS','wall_seconds':time.perf_counter()-tick});details[name]=result
        except BaseException:checks.append({'test':name,'status':'FAIL','wall_seconds':time.perf_counter()-tick});details[name]={'traceback':traceback.format_exc()}
        print(checks[-1]['status'],name,flush=True)
    check('package_integrity',lambda: {'files':len(verify_package(package)['files'])})
    check('installed_environment',lambda:(base.environment_check(package) or host_identity()))
    population={}
    def data():
        population.update(base.resolve_data(package,data_root))
        classes=set();objects=0
        for split,n in [('train',6471),('val',548)]:
            assert len(population[split])==n
            for r in population[split]:
                for line in Path(r['label_path']).read_text().splitlines():
                    if not line.strip():continue
                    values=list(map(float,line.split()));assert len(values)==5 and all(np.isfinite(values))
                    cls,x,y,w,h=values;assert int(cls)==cls and 0<=cls<=9 and 0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1
                    classes.add(int(cls));objects+=1
        assert not ({r['image_id'] for r in population['train']}&{r['image_id'] for r in population['val']})
        return {'train':6471,'development':548,'SHA_triplets_verified':7019,'valid_label_boxes':objects,'class_ids':sorted(classes),'scene_independence':'NOT_ESTABLISHED'}
    check('complete_data_and_labels',data)
    def plans():
        rows=load(package/'vm48_protocol.json')['training_matrix'];assert len(rows)==17
        primary=[r for r in rows if r['role']=='PRIMARY'];ablations=[r for r in rows if r['role']=='ABLATION']
        assert len(primary)==12 and len(ablations)==5 and {r['seed'] for r in ablations}=={42}
        manifest=[]
        for row in rows:
            plan=plan_arm(package,population,row,out/row['arm_id'],'formal' if row['role']=='PRIMARY' else 'ablations')
            assert plan['expected']=={'batches':323600,'updates':10917,'warmup_batches':9708,'last_update_position':323574,'pending_microbatches':25}
            a=plan['args'];assert a['close_mosaic']==10 and a['mosaic']==1.0 and a['epochs']==100 and a['batch']==2 and a['nbs']==64
            assert a['workers']==0 and not a['amp'] and a['optimizer']=='AdamW' and a['lr0']==.01 and a['cos_lr'] and a['warmup_epochs']==3
            manifest.append({'arm_id':row['arm_id'],'initialization_SHA256':plan['initialization']['sha256'],'seed':a['seed'],'expected':plan['expected']})
        for arch in ('YOLOv8s','YOLO26s'):
            for seed in (42,43,44):
                pair=[r for r in manifest if r['arm_id'].startswith(arch+'_seed'+str(seed)+'_') and 'SINGLE_' not in r['arm_id']]
                assert len(pair)==2 and pair[0]['initialization_SHA256']==pair[1]['initialization_SHA256']
        return {'arms':manifest,'primary':12,'ablations':5,'benchmark_horizon100_execution_epochs1':True,'partial_last_batch_images':1}
    check('17_arm_configuration_and_accumulation',plans)
    def initializations():
        import torch
        from main.training_health import model_health,cpu_state,same_state
        from main.labels import VIS_NAMES
        summaries=[]
        for row in load(package/'portable_initializations.json'):
            path=package/row['asset'];assert sha(path)==row['sha256'] and row['role']=='UNTRAINED_NATIVE10_INITIALIZATION'
            payload=torch.load(path,map_location='cpu',weights_only=False);model=payload['model'];model_health(model)
            assert list(model.names.values())==VIS_NAMES
            assert all(not x.is_floating_point() or x.dtype==torch.float32 for x in model.state_dict().values())
            summaries.append({'architecture':row['architecture'],'seed':row['seed'],'SHA256':row['sha256'],'role':row['role'],'healthy':True})
            del model,payload
        assert len(summaries)==6
        return {'native10':summaries,'no_finetuned_best_or_pilot_weights':True}
    check('six_real_initialization_identity_health',initializations)
    def effects():
        from main.training_effects import NativeEffects,TEST_VALUES,FAMILIES
        from input_ops import ArithmeticEffects
        base.renderer();summary={};comparisons=0;image=np.random.default_rng(48).integers(0,256,(129,97,3),dtype=np.uint8)
        definitions=load(package/'augmentation_protocol.json')['strategies']
        for name,entry in definitions.items():
            if entry['recipe_asset'] is None:continue
            recipe=load(package/entry['recipe_asset']);counts={f:0 for f in recipe['families']};variants=set()
            for seed in (42,43,44):
                native=NativeEffects(recipe,seed);optimized=ArithmeticEffects(recipe,seed)
                for epoch in (0,90):
                    for occurrence in range(16):
                        before=(random.getstate(),np.random.get_state(),__import__('torch').get_rng_state().clone())
                        a,ta=native.apply(image,'CPU_SYNTHETIC_IMAGE',epoch,occurrence);b,tb=optimized.apply(image,'CPU_SYNTHETIC_IMAGE',epoch,occurrence)
                        assert np.array_equal(a,b) and ta==tb and a.shape==image.shape and a.dtype==np.uint8
                        assert before[0]==random.getstate() and np.array_equal(before[1][1],np.random.get_state()[1]) and __import__('torch').equal(before[2],__import__('torch').get_rng_state())
                        for e in ta:
                            if not e['active']:continue
                            counts[e['family']]+=1
                            if 'variant' in e:variants.add(e['variant'])
                            for key,value in e['parameters'].items():assert all(abs(value-t)>1e-9 for t in TEST_VALUES.get(key,()))
                        comparisons+=1
            assert all(v>0 for v in counts.values())
            if 'NoiseAug' in recipe['families']:assert variants=={'gaussian','shot'}
            summary[name]={'active_events':counts,'noise_variants':sorted(variants),'pixel_trace_rng_equal':True}
        full=load(package/definitions['ALL_FIVE_TARGETED']['recipe_asset'])
        for family in FAMILIES:
            subset=load(package/definitions['SINGLE_'+family]['recipe_asset'])
            for occurrence in range(20):
                _,all_trace=NativeEffects(full,42).apply(image,'SYNTHETIC',0,occurrence)
                _,one_trace=NativeEffects(subset,42).apply(image,'SYNTHETIC',0,occurrence)
                assert one_trace==[r for r in all_trace if r['family']==family]
        return {'comparisons':comparisons,'activation':summary,'heldout_exact_values_rejected':True,'scope':'Synthetic CPU transform tests, no scientific AP'}
    check('all_targeted_and_five_ablation_activation',effects)
    def albu():
        import torch
        from ultralytics.data.augment import Albumentations
        from trainer48 import RecordingCompose
        image=np.random.default_rng(48).integers(0,256,(64,96,3),dtype=np.uint8)
        reference=Albumentations();logged=Albumentations();assert reference.transform is not None and logged.transform is not None
        events=[];logged.transform=RecordingCompose(logged.transform,events)
        reference.transform.set_random_seed(42);logged.transform.set_random_seed(42)
        torch.manual_seed(42);names=set()
        for i in range(1024):
            random.seed(i);np.random.seed(i);a=reference({'img':image.copy()})['img']
            random.seed(i);np.random.seed(i);b=logged({'img':image.copy()})['img']
            assert np.array_equal(a,b)
        for event in events:
            for name,params in event['applied_transforms']:names.add(name)
        assert {'Blur','MedianBlur','ToGray','CLAHE'}.issubset(names)
        assert not {'RandomBrightnessContrast','RandomGamma','ImageCompression'}&names
        return {'actual_active_transform_types':sorted(names),'logged_calls':len(events),'logging_pixel_parity':True,'Albu_Default':'Observed2.0.8, not assumed from8.4.33 documentation'}
    check('actual_ordinary_Albumentations_and_logging_parity',albu)
    def health_restart():
        import torch
        from main.training_health import RQ4HealthHooks,model_health,cpu_state,same_state
        from ultralytics.utils.torch_utils import ModelEMA
        class Fixture(RQ4HealthHooks):pass
        fixture=Fixture();fixture.model=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.BatchNorm1d(4))
        fixture.ema=ModelEMA(fixture.model);fixture.optimizer=torch.optim.AdamW(fixture.model.parameters(),lr=.01)
        fixture.scheduler=torch.optim.lr_scheduler.LambdaLR(fixture.optimizer,lambda epoch:1.)
        fixture.scaler=torch.amp.GradScaler('cuda',enabled=False);fixture.rq4_optimizer_updates=0
        fixture.rq4_context={'output':str(out/'synthetic_health'),'selection':'FIXED_LAST','scope':'SYNTHETIC_SOFTWARE_FIXTURE_NOT_TRAINING','initialization_SHA256':'SYNTHETIC'}
        Path(fixture.rq4_context['output']).mkdir()
        fixture.loss=fixture.model(torch.randn(8,4)).square().mean();fixture.loss.backward();fixture.optimizer_step()
        assert fixture.ema.updates==fixture.rq4_optimizer_updates==1
        fixture.args=argparse.Namespace(seed=42);fixture.epoch=0;fixture.metrics={};fixture.fitness=0.;fixture.best_fitness=0.;fixture.read_results_csv=lambda:{}
        fixture.save_model();checkpoint=Path(fixture.rq4_context['output'])/'weights/epoch0001_fp32.pt'
        loaded=__import__('torch').load(checkpoint,map_location='cpu',weights_only=False)
        assert same_state(loaded['ema'],cpu_state(fixture.ema.ema)) and loaded['exact_resume_validated'] is False
        try:fixture.save_model();raise AssertionError('Overwriting must fail.')
        except FileExistsError:pass
        with torch.no_grad():fixture.model[1].running_var[0]=-1
        try:model_health(fixture.model);raise AssertionError('Negative BN variance must fail.')
        except FloatingPointError:pass
        command=[sys.executable,'-B','-X','utf8','-c',"import torch,sys; p=torch.load(sys.argv[1],map_location='cpu',weights_only=False); assert p['epoch']==0 and p['exact_resume_validated'] is False; assert all(not t.is_floating_point() or t.dtype==torch.float32 for t in p['ema'].state_dict().values()); print('CPU_RELOAD_PASS')",str(checkpoint)]
        result=subprocess.run(command,capture_output=True,text=True,check=True,timeout=60)
        return {'optimizer_EMA_checks':'PASS','atomic_FP32_save_reload':'PASS','subprocess_restart':result.stdout.strip(),'negative_BN_rejected':True,
                'exact_GPU_resume':'NOT_VALIDATED_AND_FORBIDDEN','fixture_scope':'SYNTHETIC_ONLY_NOT_AN_EPOCH100_RESULT'}
    check('health_FP32_model_restart_and_no_overwrite',health_restart)
    def final_epoch():
        for bad in (0,10,99):
            try:epoch100_contract({'completed_epochs':bad},{},'x');raise AssertionError('Incomplete epochs must fail.')
            except ValueError:pass
        counters={'completed_epochs':100,'anchor_exposures':647100,'forward_batches':323600,'backward_batches':323600,'optimizer_updates':10917,'EMA_updates':10917}
        try:epoch100_contract(counters,{'epoch':98,'ema':object()},'x');raise AssertionError('Best or epoch99 must fail.')
        except ValueError:pass
        return {'incomplete_and_wrong_epoch_rejected':True,'fixed_epoch100_not_best':True,'no_pilot_substitution':True}
    check('epoch100_contract_negative_cases',final_epoch)
    def inference_contract():
        import torch
        from types import SimpleNamespace
        from main.predictor_stable import StableNMSPredictor
        from evaluation48 import sized_predictor
        from ultralytics.utils.ops import scale_boxes
        image=np.random.default_rng(48).integers(0,256,(192,320,3),dtype=np.uint8)
        original=torch.tensor([[20.,30.,70.,90.]])
        for resolution in (640,1280):
            predictors=[StableNMSPredictor(overrides={'imgsz':resolution,'rect':False}),sized_predictor(StableNMSPredictor,resolution)(overrides={'imgsz':resolution,'rect':False})]
            tensors=[]
            for predictor in predictors:
                predictor.imgsz=[resolution,resolution];predictor.device=torch.device('cpu');predictor.model=SimpleNamespace(fp16=False,stride=32,pt=True)
                tensors.append(predictor.preprocess([image]))
            assert torch.equal(*tensors) and list(tensors[1].shape)==[1,3,resolution,resolution]
            gain=resolution/320;box=original*gain;box[:,[1,3]]+=(resolution-192*gain)/2
            assert torch.allclose(scale_boxes((resolution,resolution),box,(192,320)),original,atol=1e-5)
        return {'FP32_NCHW':[640,1280],'accepted_preprocessing_tensor_equal':True,'original_coordinate_roundtrip':True,'GPU':False}
    check('resolution_and_coordinate_CPU_contract',inference_contract)
    def gap():
        from gap48 import state,p_state,paired_gap,bootstrap_gap_draws
        assert state([.11,.12],[40,50])=='PASS' and state([.09,.11],[40,50])=='UNCERTAIN' and state([.08,.09],[40,50])=='FAIL'
        assert p_state(['FAIL','UNCERTAIN'])=='UNCERTAIN' and p_state(['FAIL','FAIL'])=='GAP'
        r=paired_gap(['GAP','UNCERTAIN','COVERED'],['COVERED','UNCERTAIN','COVERED']);assert r['known_Gap_count_reduction']==1 and r['three_state_reduction_bounds']==[0,2]
        cells={}
        for seed in (42,43,44):
            for arch in ('YOLOv8s','YOLO26s'):
                for strategy,tiny in [('CONTROL',.08),('ALL_FIVE_TARGETED',.12)]:
                    for resolution in (640,1280):
                        for p in ('SYNTHETIC_P0','SYNTHETIC_P1'):cells[(arch,seed,strategy,resolution,p)]={'tiny_draws':[tiny]*200,'latency_draws':[50.]*200}
        result=bootstrap_gap_draws(cells,['SYNTHETIC_P0','SYNTHETIC_P1'],[42,43,44]);assert all(r['CI95_count']==[2.,2.] for r in result)
        return {'three_state_counting':'PASS','paired_indices_and_complete_D_sets':'PASS','population_estimation':'NOT_ALLOWED','scope':'Synthetic software tests; not measured Gap reduction'}
    check('paired_evaluation_and_Gap_counting',gap)
    def safety():
        from safety import decide
        policy=load(package/'safety_policy.json')
        assert decide(policy,[{'temperature_C':90,'monotonic_seconds':0}])=='TEMPERATURE_HARD_STOP'
        assert decide(policy,[{'temperature_C':87,'monotonic_seconds':float(i)} for i in range(31)])=='SUSTAINED_TEMPERATURE_STOP'
        assert decide(policy,[],telemetry_ok=False)=='TELEMETRY_FAILURE'
        assert remaining({'total_seconds':100,'expires_UTC':(datetime.now(timezone.utc)-timedelta(seconds=1)).isoformat()})==0
        try:validate_authorization(package,'formal',matrix(package,'formal'),{'approved':False},{});raise AssertionError('No authorization must fail.')
        except PermissionError:pass
        rows=matrix(package,'benchmark','YOLO26s');receipt={'status':'CPU_SOFTWARE_PASS_NOT_GPU_VALIDATED','package_manifest_SHA256':sha(package/'package_manifest.json'),'host':host_identity()}
        canonical=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        auth={'approved':True,'authorization_id':'unit-metadata-in-memory-only','protocol_id':VERSION,'allocation_scope':'NEW_ARM_SPECIFIC_ALLOCATION',
            'historical_VM37_ledger_reuse':False,'package_manifest_SHA256':sha(package/'package_manifest.json'),'mode':'benchmark','arm_ids':[r['arm_id'] for r in rows],
            'arm_authorizations':{r['arm_id']:{'approved':True,'hard_seconds':100} for r in rows},'cooling_confirmed':True,'power_confirmed':True,'machine_admitted':True,
            'CPU_receipt_SHA256':canonical,'total_seconds':400,'expires_UTC':(datetime.now(timezone.utc)+timedelta(seconds=400)).isoformat(),
            'resource_admission':{'status':'AUTHOR_ADMITTED_COMPLETE_GROUP','safety_reserve_seconds':60,'evaluation_hard_seconds':0,'required_complete_group_seconds':260}}
        assert validate_authorization(package,'benchmark',rows,auth,receipt)>260
        for key,value in [('approved',False),('historical_VM37_ledger_reuse',True),('arm_ids',auth['arm_ids'][:1]),('total_seconds',200)]:
            bad=copy.deepcopy(auth);bad[key]=value
            try:validate_authorization(package,'benchmark',rows,bad,receipt);raise AssertionError('Admission mutation must fail: '+key)
            except PermissionError:pass
        return {'90C_immediate':True,'87C_30seconds':True,'telemetry_fail_closed':True,'expired_budget_rejected':True,'default_GPU_denied':True,
                'complete_pair_budget_and_arm_scope_mutations':'PASS','test_authorization':'IN_MEMORY_SOFTWARE_FIXTURE_ONLY_NOT_SAVED_OR_EXECUTED'}
    check('budget_and_stop_safety',safety)
    if regression:
        def real_metrics():
            from evaluation48 import saved_metrics
            from main.labels import ground_truth,mapped_predictions
            from main.statistics import evaluate_coco
            folder=Path(regression);accept=load(folder/'acceptance.json')
            for field in ('raw','GT','metrics'):
                assert sha(accept[field]['path'])==accept[field]['SHA256']
            records=load(accept['GT']['path'])['records'];expected=load(accept['metrics']['path'])
            with gzip.open(accept['raw']['path'],'rt',encoding='utf-8') as f:rows=[json.loads(s) for s in f]
            from cached_backend import validate_predictions
            validate_predictions(rows,{r['id'] for r in records})
            gt,regions=ground_truth(records,'native10');pred=[p for r in rows for p in r['predictions']]
            mapped=mapped_predictions(pred,records,regions,'native10','finetuned',3000)
            areas,classes,evaluator=evaluate_coco(gt,mapped,3000)
            for got,reference in [(areas,expected['per_area']),(classes,expected['per_class'])]:
                assert len(got)==len(reference)
                for a,b in zip(got,reference):
                    for k in ('map50','map50_95','ar50','ar50_95'):
                        assert a[k] is None and b[k] is None or a[k] is not None and b[k] is not None and abs(a[k]-b[k])<=1e-12
            result={'source_cell':str(folder),'raw_SHA256':accept['raw']['SHA256'],'GT_SHA256':accept['GT']['SHA256'],'reference_metrics_SHA256':accept['metrics']['SHA256'],
                    'images':548,'areas':len(areas),'per_class_rows':len(classes),'numeric_tolerance':1e-12,'GPU':False,'new_scientific_results':False}
            del evaluator,rows,pred,mapped;gc.collect();return result
        check('saved_real548_prediction_metric_regression',real_metrics)
    failed=[r['test'] for r in checks if r['status']=='FAIL']
    receipt={'status':'CPU_SOFTWARE_PASS_NOT_GPU_VALIDATED' if not failed else 'CPU_TEST_FAILED','package_manifest_SHA256':sha(package/'package_manifest.json'),
       'host':host_identity(),'checks':checks,'details':details,'failed':failed,'GPU_seconds':0,'formal_training_arms_run':0,
       'wall_seconds':time.perf_counter()-started,'exact_GPU_resume_validated':False,'server_GPU_tested':False}
    save(out/'cpu_test_results.json',receipt)
    print(json.dumps({'status':receipt['status'],'checks':len(checks),'failed':failed,'receipt':str(out/'cpu_test_results.json')}),flush=True)
    return 1 if failed else 0


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--real-regression',type=Path)
    a=p.parse_args();sys.exit(run_checks(a.package.resolve(strict=True),a.data_root.resolve(strict=True),a.output,a.real_regression))
