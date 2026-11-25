"""Freeze seed45 plans and check only the new sampler/pause integration."""
import copy,csv,gc,json,os,random,time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from motion_common import *

def prepare(out):
    check(out);base.runtime(out,gpu=False)
    import torch,cv2
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import init_seeds
    from main.training_health import model_health,same_state,cpu_state
    from main.labels import VIS_NAMES
    from motion_training import draw_explicit,KERNELS
    torch.set_num_threads(2);cv2.setNumThreads(2);base.environment_check(BUNDLE)
    source_weight=ROOT/'models/checkpoints/yolo26s.pt'
    initial_record=next(r for r in load(BUNDLE/'portable_initializations.json') if r['architecture']=='YOLO26s' and r['seed']==42)
    assert sha(source_weight)==initial_record['COCO_source_SHA256']
    payload=torch.load(source_weight,map_location='cpu',weights_only=False);pretrained=payload.get('ema') or payload['model'];model_health(pretrained)
    assert len(pretrained.names)==80 and pretrained.yaml.get('scale')=='s'
    init_seeds(45,deterministic=True);model=DetectionModel(cfg=copy.deepcopy(pretrained.yaml),ch=3,nc=10,verbose=False);model.load(pretrained,verbose=False)
    model=model.cpu().float();model.names=dict(enumerate(VIS_NAMES));model_health(model)
    initial=out/'YOLO26s_seed45_native10_fp32.pt'
    torch.save({'model':model,'epoch':-1,'seed':45,'origin_SHA256':sha(source_weight),'purpose':'UNTRAINED_NATIVE10_INITIALIZATION','not_historical_training_reproduction':True},initial)
    restored=torch.load(initial,map_location='cpu',weights_only=False)['model'];assert same_state(restored,cpu_state(model))
    save(out/'initialization_SHA.json',{'initialization':source(initial),'COCO_source':source(source_weight),'seed':45,'names':VIS_NAMES,'FP32':True,'save_reload_equal':True,'same_file_for_both_arms':True})
    del model,restored,pretrained,payload;gc.collect()
    population=base.resolve_data(BUNDLE,ROOT/'datasets/VisDrone',verify=True)
    reference=load(ROOT/'datasets/local_runs/vm37_yolo26_rq4/vm37_yolo26_rq4_pilot960_execution_20261114T110947_fe2f01b3/YOLO26s_seed42_CONTROL/protocol.json')
    recipe={'version':'VM48_MOTION_TARGETED_SEED45_P025_K5_9_13_17_25_V1','status':'APPROVED_TRAINING_RECIPE','scientific_approval':{'basis':'Author submitted this exact exploratory strategy in the current one-time instruction','source_SHA256':sha(REQUEST),'teacher_numeric_approval_claim':False},
        'order':['HazeAug','ExposureAug','MotionBlurAug','NoiseAug','JPEGAug'],'families':{'MotionBlurAug':{'probability':.25,'kernel':{'distribution':'integer_uniform','low':5,'high':25,'values':list(KERNELS),'actual_sampling':'Uniform categorical over values, not all integers in the bounding range'},'angle_min':0.,'angle_max':180.}},
        'selection':'Frozen before new outcomes; no use of evaluation kernel21','RNG':'Accepted per-family SHA(seed|epoch|image|occurrence); local probability gate, kernel choice and generator angle draw','geometry':'Native BGR photometric blur; original boxes/classes unchanged before shared ordinary augmentation'}
    plans={}
    for arm in ['CONTROL','MOTION_TARGETED']:
        folder=out/arm;folder.mkdir();p=copy.deepcopy(reference)
        p.update(mode='VM48_MOTION_TARGETED_SEED45',scope='DEVELOPMENT_ONLY / POST_HOC_POLICY_SELECTION / SINGLE_SEED45',arm=arm,protocol_name='VM48_MOTION_TARGETED_SEED45',
            initialization={'path':str(initial),'sha256':sha(initial)},targeted_recipe=recipe if arm=='MOTION_TARGETED' else None,selection='FIXED_EPOCH10_EMA',GPU_execution_authorized=True,
            augmentation_label='CONTROL' if arm=='CONTROL' else 'Author-frozen Motion p0.25, kernels5/9/13/17/25')
        p['args'].update(seed=45,model=str(initial),data=str(folder/'dataset.yaml'),project=str(folder),epochs=10,imgsz=960)
        p['train']=[{'image_id':r['image_id'],'image_path':r['image_path'],'image_SHA256':r['image_SHA256'],'derived_label_SHA256':r['label_SHA256']} for r in population['train']]
        p['development']=[{'stem':r['image_id'],'loader_path':r['image_path'],'source_image_SHA256':r['image_SHA256'],'derived_label_SHA256':r['label_SHA256']} for r in population['val']]
        p['train_labels']={r['image_id']:r['label_path'] for r in population['train']};p['val_labels']={r['image_id']:r['label_path'] for r in population['val']}
        assert p['expected']['batches']==32360 and p['expected']['updates']==1816
        save(folder/'protocol.json',p);plans[arm]=p
    exclude={'project','data'}
    assert {k:v for k,v in plans['CONTROL']['args'].items() if k not in exclude}=={k:v for k,v in plans['MOTION_TARGETED']['args'].items() if k not in exclude}
    with (out/'training_arm_manifest.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['strategy','seed','epochs','imgsz','initialization_SHA256','protocol_SHA256']);w.writeheader()
        for arm in plans:w.writerow({'strategy':arm,'seed':45,'epochs':10,'imgsz':960,'initialization_SHA256':sha(initial),'protocol_SHA256':sha(out/arm/'protocol.json')})
    spec={'id':'VM48_MOTION_TARGETED_SEED45','seed':45,'model':'YOLO26s','train_images':6471,'validation_images':548,'training_args':plans['CONTROL']['args'],'recipe':recipe,
        'initialization_SHA256':sha(initial),'epoch100_formal_protocol_changed':False,'fixed_epoch10_EMA':True,'budget_seconds':36000,'expiry_UTC':'2026-11-22T23:00:00+00:00',
        'stage_caps_seconds':{'training':28000,'evaluation':6000,'shutdown_reserve':2000},
        'order':'CONTROL epoch1; TARGETED epoch1; cost gate; TARGETED remaining9; CONTROL remaining9, all in one process, no disk resume',
        'pause_semantics':'CONTROL live model/EMA/optimizer/scheduler/scaler/pending gradients stay intact; Python/NumPy/Torch CPU/CUDA RNG restored after nested TARGETED; inactive elapsed time excluded fromCONTROL training diagnostics',
        'cost_gate':'No early AP use. Conservative remaining9 epochs each =9*(measured first epoch including setup for both)*1.10+120. Must fit training cap; evaluation6000 and reserve2000 held aside.',
        'first_epoch_not_a_training_completion':True,'new_strategy_not_pooled_with_Gaussian_seeds42_44':True,'scope':'DEVELOPMENT_ONLY / SINGLE_SEED45 / AUTHOR_PROPOSED_EXPLORATORY',
        'primary_endpoint':'Motion_k21 Tiny AP50:95 MOTION_TARGETED minus CONTROL','FP32':True,'evaluation_per_cell_warmups':3,'bootstrap_samples':200}
    save(out/'training_protocol.json',spec)
    # Exercise only the new explicit sampler; accepted image operations are reused.
    import input_ops
    from main.training_effects import NativeEffects
    from input_ops import ArithmeticEffects
    image=np.random.default_rng(1).integers(0,256,(80,120,3),dtype=np.uint8);trace=[];kernels=set();angles=[]
    with patch.object(input_ops,'draw',draw_explicit):
        op=ArithmeticEffects(recipe,45)
        state=random.getstate();numpy=np.random.get_state();torch_rng=torch.get_rng_state().clone()
        for i in range(400):
            pix,event=op.apply(image,str(i),0,0);again,e2=op.apply(image,str(i),0,0);assert np.array_equal(pix,again) and event==e2 and pix.shape==image.shape
            e=event[0]
            if e['active']:kernels.add(e['parameters']['kernel']);angles.append(e['angle_deg']);assert not np.array_equal(pix,image)
        assert random.getstate()==state and np.array_equal(np.random.get_state()[1],numpy[1]) and torch.equal(torch.get_rng_state(),torch_rng)
        assert kernels==set(KERNELS) and all(0<=a<180 for a in angles)
    pause=cpu_pause_check()
    save(out/'new_integration_CPU_checks.json',{'explicit_kernels':sorted(kernels),'evaluation21_excluded':21 not in kernels,'active_trials':len(angles),'trials':400,'pixel_geometry_and_repeatability':True,'global_RNG_not_consumed_by_motion':True,'new_pause_fixture':pause,'not_real_YOLO_resume_proof':True,'GPU_forward_calls':0,'accepted_tests_repeated':False})
    original=load(VM45/'experiment_protocol.json');cache=load(VM45/'pixel_cache_manifest.json');assert sha(original['GT']['path'])==original['GT']['SHA256']
    for name,entries in cache.items():
        assert len(entries)==548
        for e in entries:assert sha(e['path'])==e['file_SHA256']
    save(out/'evaluation_protocol.json',{**original,'id':'VM48_SEED45_FROZEN_EVALUATION_V1','training_seed':45,'strategies':['CONTROL','MOTION_TARGETED'],'primary_endpoint':'Motion_k21 Tiny AP95 paired delta','warmups_per_cell':3,'old_invalid_latency_reused':False,'pixel_cache_source':source(VM45/'pixel_cache_manifest.json'),'conditions_frozen_before_training':True})
    files=[p for p in HERE.glob('*.py')]+[out/a/'protocol.json' for a in plans]+[out/'initialization_SHA.json',initial,out/'evaluation_protocol.json',out/'training_protocol.json',Path(v3.__file__),Path(recovery.__file__),ROOT/'main/experiments/vm48_formal_rq4/trainer48.py',ROOT/'main/experiments/experiments_test/vm37_yolo26_rq4/training.py',ROOT/'main/code/main/training_health.py',ROOT/'main/code/main/training_effects.py',ROOT/'main/code/main/training_data.py',ROOT/'main/experiments/experiments_test/vm37_yolo26_rq4/post_cost/fast_input.py',ROOT/'main/experiments/experiments_test/vm37_yolo26_rq4/post_cost/input_ops.py',ROOT/'main/experiments/experiments_test/vm43_external_pilot/backend.py',ROOT/'main/experiments/experiments_test/vm45_motion_haze_gap_learning/execute_vm45.py',ROOT/'main/code/main/statistics.py',ROOT/'main/code/main/labels.py',ROOT/'main/code/main/predictor_stable.py']
    save(out/'frozen_sources.json',[source(p) for p in files])
    save(out/'source_provenance.json',{'frozen_sources':source(out/'frozen_sources.json'),'accepted_dataset_manifest':source(BUNDLE/'portable_dataset_manifest.json'),'initialization':source(initial),'COCO_source':source(source_weight),'renderer':source(ROOT/'main/experiments/experiments_test/cv261109/generator.py'),'weekly_completed_before_GPU':True,'original_formal17arm_package_changed':False})

def cpu_pause_check():
    import torch
    def one(pause):
        random.seed(45);np.random.seed(45);torch.manual_seed(45)
        a=torch.nn.Linear(2,1);opt=torch.optim.AdamW(a.parameters(),lr=.01);sched=torch.optim.lr_scheduler.StepLR(opt,step_size=1,gamma=.9);losses=[]
        for epoch in range(3):
            for step in range(3):
                x=torch.randn(2,2)*(random.random()+np.random.random());loss=a(x).square().mean();loss.backward();losses.append(float(loss.detach()))
                if (epoch*3+step)%2:opt.step();opt.zero_grad()
            sched.step()
            if pause and epoch==0:
                states=(random.getstate(),np.random.get_state(),torch.get_rng_state());before=[p.grad.clone() if p.grad is not None else None for p in a.parameters()]
                random.seed(81);np.random.seed(81);torch.manual_seed(81);b=torch.nn.Linear(2,1);other=torch.optim.AdamW(b.parameters(),lr=.01)
                for _ in range(4):other.zero_grad();b(torch.randn(2,2)).square().mean().backward();other.step();random.random();np.random.random()
                random.setstate(states[0]);np.random.set_state(states[1]);torch.set_rng_state(states[2])
                assert all(x is None and p.grad is None or x is not None and torch.equal(x,p.grad) for x,p in zip(before,a.parameters()))
        return a.state_dict(),opt.state_dict(),sched.state_dict(),losses,torch.get_rng_state(),random.getstate(),np.random.get_state()
    from main.training_health import same_tree
    x=one(False);y=one(True)
    assert same_tree(x[:6],y[:6]) and np.array_equal(x[6][1],y[6][1])
    return 'SYNTHETIC_CPU_EXACT: model/optimizer/scheduler/pending-gradient/RNG/loss parity; not GPU recovery'
