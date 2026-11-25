"""Accepted960 trainer with a bounded in-memory paired cost gate."""
import copy,csv,gc,hashlib,json,math,random,time
from pathlib import Path
from collections import Counter
from unittest.mock import patch
import numpy as np
from motion_common import *

KERNELS=(5,9,13,17,25)

class CostGateStop(RuntimeError):pass

def draw_explicit(spec,name,rng):
    from main.training_effects import draw
    if name=='kernel' and 'values' in spec[name]:return int(rng.choice(spec[name]['values']))
    return draw(spec,name,rng)

def rng_state():
    import torch
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}

def restore_rng(s):
    import torch
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch']);torch.cuda.set_rng_state_all(s['cuda'])

def fingerprint(x):
    import torch
    h=hashlib.sha256()
    def visit(v):
        if isinstance(v,torch.Tensor):h.update(str((v.dtype,list(v.shape))).encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str):h.update(str(k).encode());visit(v[k])
        elif isinstance(v,(list,tuple)):
            for a in v:visit(a)
        elif isinstance(v,np.ndarray):h.update(v.tobytes())
        else:h.update(repr(v).encode())
    visit(x);return h.hexdigest()

def live_state(trainer):
    return {'model':trainer.model.state_dict(),'EMA':trainer.ema.ema.state_dict(),'optimizer':trainer.optimizer.state_dict(),'scheduler':trainer.scheduler.state_dict(),'scaler':trainer.scaler.state_dict(),
        'pending_gradients':[p.grad for p in trainer.model.parameters()],'updates':trainer.rq4_optimizer_updates,'EMA_updates':trainer.ema.updates}

def trainer_factory(plan,folder,manager):
    Parent=recovery.trainer_factory(plan,folder)
    helper=module('vm48_seed45_albu_observer',ROOT/'main/experiments/vm48_formal_rq4/trainer48.py')
    class Trainer(Parent):
        def __init__(self,*a,**k):
            super().__init__(*a,**k);self.add_callback('on_fit_epoch_end',self.completed_epoch)
        def build_dataset(self,*a,**k):
            data=super().build_dataset(*a,**k);data.motion_albu_events=[]
            cfg=helper.install_albu(data.transforms,data.motion_albu_events)
            save(folder/('ordinary_augmentation_'+('train' if data.augment else 'val')+'.json'),{'configuration':cfg,'plan':plan['arm']})
            return data
        def _close_dataloader_mosaic(self):
            super()._close_dataloader_mosaic();data=self.train_loader.dataset
            helper.install_albu(data.transforms,data.motion_albu_events)
        def batch_end(self,trainer):
            events=list(self.train_loader.dataset.effect_events)
            super().batch_end(trainer)
            target=manager.out/'augmentation_event_log.csv';new=not target.exists()
            with target.open('a',encoding='utf-8',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['strategy','epoch','image_id','occurrence','active','kernel','angle_deg','family_seed','native_SHA256','effect_SHA256','pixel_changed'])
                if new:w.writeheader()
                for e in events:
                    effect=next((x for x in e['effects'] if x['family']=='MotionBlurAug'),{})
                    active=effect.get('active',False);params=effect.get('parameters',{})
                    if active:
                        assert params['kernel'] in KERNELS and params['kernel']!=21 and 0<=effect['angle_deg']<180
                    w.writerow({'strategy':plan['arm'],'epoch':self.epoch+1,'image_id':e['id'],'occurrence':e['occurrence'],'active':active,'kernel':params.get('kernel'),'angle_deg':effect.get('angle_deg'),'family_seed':effect.get('seed'),
                        'native_SHA256':e['native_pixel_SHA256'],'effect_SHA256':e['effect_pixel_SHA256'],'pixel_changed':e['native_pixel_SHA256']!=e['effect_pixel_SHA256']})
            albu=self.train_loader.dataset.motion_albu_events
            with (folder/'ordinary_augmentation_events.jsonl').open('a',encoding='utf-8') as f:
                for e in albu:f.write(json.dumps({'epoch':self.epoch+1,**e})+'\n')
            albu.clear()
        def completed_epoch(self,trainer):manager.completed(self,plan,folder)
    return Trainer

class PairManager:
    def __init__(self,out,plans,cap):
        self.out=out;self.plans=plans;self.cap=cap;self.started=time.perf_counter();self.active=None;self.trainers={};self.started_arm={};self.first={};self.epoch_rows=[]
    def launch(self,arm):
        plan=self.plans[arm];folder=self.out/arm;self.started_arm[arm]=time.perf_counter()
        t=trainer_factory(plan,folder,self)(overrides=plan['args']);self.trainers[arm]=t;self.active=t;t.train();self.accept(t,plan,folder)
    def completed(self,t,plan,folder):
        epoch=t.epoch+1
        row={'strategy':plan['arm'],'epoch':epoch,'active_epoch_seconds':t.epoch_time,'forward_batches':t.forward_count,'backward_batches':t.backward_count,'optimizer_updates':t.rq4_optimizer_updates,'EMA_updates':t.ema.updates,
            'internal_validation_seconds':t.validations[-1]['seconds'],'internal_validation_scope':'Library diagnostic, not research ignore-union COCO','loss':float(t.loss.detach()),'learning_rates':str([g['lr'] for g in t.optimizer.param_groups]),'checkpoint':str(t.rq4_selected)}
        self.epoch_rows.append(row)
        with (self.out/'epoch_metrics.csv').open('a',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(row))
            if f.tell()==0:w.writeheader()
            w.writerow(row)
        save(folder/f'epoch{epoch:02d}_complete.json',row)
        print('EPOCH_COMPLETE',plan['arm'],epoch,'active_seconds',t.epoch_time,flush=True)
        if epoch!=1:return
        self.first[plan['arm']]=time.perf_counter()-self.started_arm[plan['arm']]
        save(folder/'first_epoch_cost.json',{'seconds_including_setup':self.first[plan['arm']],'validation_and_save_included':True,'nominal_schedule_epochs':10,'completed_epochs':1,'data':[6471,548]})
        if plan['arm']=='MOTION_TARGETED':
            elapsed=time.perf_counter()-self.started;remaining_training=9*sum(self.first.values())*1.10+120
            fits=remaining_training<=self.cap-elapsed
            save(self.out/'complete_pair_cost_gate.json',{'admitted':fits,'first_epoch_seconds':self.first,'remaining9_each_conservative_seconds':remaining_training,'factor':1.10,'remaining_training_cap_seconds':self.cap-elapsed,
                'evaluation_cap_seconds':6000,'shutdown_reserve_seconds':2000,'early_AP_used':False,'in_memory_continuation':True,'cross_process_resume_validated':False})
            if not fits:raise CostGateStop('Both first epochs completed; conservative remaining pair does not fit.')
            return
        # The outer training loop remains alive, including its pending gradients.
        assert not getattr(t.train_loader.dataset,'pending',{})
        state=rng_state();before=fingerprint(live_state(t));rng_before=fingerprint(state);paused=time.perf_counter()
        torch=__import__('torch');torch.save(state,self.out/'CONTROL_live_pause_RNG.pt')
        self.launch('MOTION_TARGETED')
        assert fingerprint(live_state(t))==before,'Paused CONTROL state changed.'
        restore_rng(state);assert fingerprint(rng_state())==rng_before
        pause=time.perf_counter()-paused;t.epoch_time_start+=pause;t.train_time_start+=pause;self.active=t
        save(self.out/'CONTROL_live_pause_preservation.json',{'model_EMA_optimizer_scheduler_scaler_pending_gradients_equal':True,'global_Python_NumPy_Torch_CPU_CUDA_RNG_restored':True,'pause_seconds':pause,
            'cross_process_resume_claim':False,'CONTINUE_order':'TARGETED completes9 remaining epochs, then CONTROL completes9 remaining epochs','before_after_digest':before})
    def accept(self,t,plan,folder):
        from main.training_health import model_health
        expected=plan['expected'];assert t.epoch==9 and len(t.rows)==t.forward_count==t.backward_count==32360
        assert t.rq4_optimizer_updates==t.ema.updates==expected['updates']==1816
        for epoch in range(1,11):assert Counter(i for r in t.rows if r['epoch']==epoch for i in r['image_ids'])==Counter(r['image_id'] for r in plan['train'])
        model_health(t.model,gradients=True);model_health(t.ema.ema)
        weights=list((folder/'weights').glob('epoch*_fp32.pt'));assert len(weights)==10 and Path(t.rq4_selected).name=='epoch0010_fp32.pt'
        save(folder/'training_acceptance.json',{'completed_epochs':10,'anchor_exposures':64710,'forward_batches':32360,'backward_batches':32360,'optimizer_updates':1816,'EMA_updates':1816,'strategy':plan['arm'],
            'selected_checkpoint':source(t.rq4_selected),'epoch_checkpoints':[source(p) for p in sorted(weights)],'initialization_SHA256':plan['initialization']['sha256'],'internal_validation':t.validations,'FP32':True,'exact_resume_validated':False})

def worker(out,cap):
    check(out);base.runtime(out,gpu=True)
    import torch,cv2,yaml,input_ops
    from ultralytics.utils import callbacks
    from post80_loader import read_only_sources
    from main.labels import VIS_NAMES
    torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.is_available()
    plans={a:load(out/a/'protocol.json') for a in ['CONTROL','MOTION_TARGETED']}
    deadline=load(out/'training_lease.json')['deadline_UTC']
    for arm,plan in plans.items():
        folder=out/arm;save(folder/'WORKER_LEASE.json',{'deadline_UTC':deadline})
        (folder/'train_images.txt').write_text('\n'.join(r['image_path'] for r in plan['train'])+'\n',encoding='utf-8')
        (folder/'val_images.txt').write_text('\n'.join(r['loader_path'] for r in plan['development'])+'\n',encoding='utf-8')
        (folder/'dataset.yaml').write_text(yaml.safe_dump({'train':str(folder/'train_images.txt'),'val':str(folder/'val_images.txt'),'names':dict(enumerate(VIS_NAMES))}),encoding='utf-8')
    manager=PairManager(out,plans,cap);backward=torch.autograd.backward
    def timed_backward(*a,**k):
        t=manager.active;begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True);begin.record();tick=time.perf_counter();r=backward(*a,**k);end.record();torch.cuda.synchronize()
        t.backward_count+=1;t.backward_seconds=time.perf_counter()-tick;t.backward_CUDA_ms=begin.elapsed_time(end);return r
    status='FAILED';error=None
    try:
        base.renderer()
        with patch.object(callbacks,'add_integration_callbacks',lambda instance:None),patch.object(torch.autograd,'backward',timed_backward),patch.object(input_ops,'draw',draw_explicit),v3.output_scope(out),read_only_sources(out):manager.launch('CONTROL')
        assert all((out/a/'training_acceptance.json').exists() for a in plans);status='MATCHED_10EPOCH_TRAINING_COMPLETED'
    except CostGateStop as e:status='PARTIAL_COST_GATE_STOP';error=str(e)
    except BaseException:
        import traceback;error=traceback.format_exc();raise
    finally:
        save(out/'training_result.json',{'status':status,'completed_epochs':{a:len(list((out/a).glob('epoch*_complete.json'))) for a in plans},'error':error,'retry':False,'wall_seconds':time.perf_counter()-manager.started})
        for t in manager.trainers.values():
            data=getattr(getattr(t,'train_loader',None),'dataset',None)
            if hasattr(data,'close'):data.close()
