"""YOLO26-first FP32 engineering admission and full-data cost measurement."""
import argparse
import copy
import csv
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import threading
import traceback
from study import *


def prepare(mode,architecture,arm='CONTROL'):
    import yaml
    verify_hashes(PREP,('CANARY_IMAGE_CANDIDATES.json','TRAINING_DATA_VERIFICATION.csv','VALIDATION_LOADER_ACCEPTANCE.json','CANDIDATE_INITIALIZATION_MANIFEST.json'))
    if mode=='canary' and arm!='CONTROL':raise ValueError('The engineering canary uses the shared control pipeline.')
    if arm=='ALL_FIVE':
        receipt=load(RUNS/'policy_admission.json');policy_run=run_path(receipt['RunID'])
        verify_hashes(policy_run,('policy.json','CPU_policy_acceptance.json'))
        if receipt['status']!='PASS' or sha256(HERE/'TARGETED_POLICY.json')!=receipt['policy_SHA256']:
            raise ValueError('Targeted policy has no unchanged CPU admission.')
        if sha256(PUBLIC/'code/main/training_effects.py')!=receipt['effects_implementation_SHA256'] or sha256(HERE.parent/'cv261109/generator.py')!=receipt['generator_SHA256']:
            raise ValueError('Targeted sampling implementation changed after admission.')
    out=create(mode+'_'+architecture.lower()+'_'+arm.lower());runtime(out,cuda=False)
    candidates=load(PREP/'CANDIDATE_INITIALIZATION_MANIFEST.json')
    initialization=next(r for r in candidates if r['architecture']==architecture and r['seed']==42)
    for ref in (initialization['source'],initialization['native10_candidate']):
        if sha256(Path(ref['path']))!=ref['sha256']:raise ValueError('Initialization SHA changed.')
    ids=load(PREP/'CANARY_IMAGE_CANDIDATES.json');labels=load(PREP/'PRIVATE_LABEL_LOOKUP.json')
    if mode=='canary':train=ids['train'];development=ids['development'];epochs=1
    else:
        with (PREP/'TRAINING_DATA_VERIFICATION.csv').open(encoding='utf-8-sig') as stream:train=list(csv.DictReader(stream))
        development=load(PREP/'VALIDATION_LOADER_ACCEPTANCE.json')['population'];epochs=100
    original_args=ROOT/'models/checkpoints'/('ck01_yolo26s_ft1280' if architecture=='YOLO26s' else 'ck05_yolov8s_ft1280')/'args.yaml'
    args=yaml.safe_load(original_args.read_text());args.pop('save_dir',None)
    args.update(model=initialization['native10_candidate']['path'],data=str(out/'dataset.yaml'),epochs=epochs,imgsz=1280,
        batch=2,nbs=64,optimizer='AdamW',lr0=.01,cos_lr=True,warmup_epochs=3.,amp=False,workers=0,seed=42,device='0',
        project=str(out),name='trainer',plots=False,exist_ok=False,cache=False,save=True,save_period=-1,resume=False,
        end2end=False,cls_remap=False)
    target=load(HERE/'TARGETED_POLICY.json') if arm=='ALL_FIVE' else None
    plan={'mode':mode,'architecture':architecture,'arm':arm,'targeted_recipe':target,
        'scope':'CANARY_ENGINEERING_ONLY' if mode=='canary' else 'FULL_EPOCH_COST_ONLY',
        'args':args,'initialization':initialization['native10_candidate'],'COCO_source':initialization['source'],
        'train':train,'development':development,'train_labels':{r['image_id']:labels[r['image_id']] for r in train},
        'val_labels':{r['stem']:str(PREP/'validation_labels'/(r['stem']+'.txt')) for r in development},
        'new_target_augmentations':[],'expected':expected_steps(len(train),epochs,stop_epochs=1),
        'stop_after_completed_epochs':1,'scientific_training_arm':False,'library':'8.4.210',
        'initialization_policy':'Reuse the admitted official-COCO-derived native10 state; disable new class-name remapping to preserve its exact bytes.',
        'ordinary_augmentations':'Shared inherited HSV/geometry/mosaic and installed Albumentations.',
        'engineering_implementation':'Grouped full-tensor health checks and read-only native cache; original checks and pixel/label calculations retained. CPU parity tested.',
        'new_version_differences':['Warmup is capped at epochs-1; one-epoch canary has0 warmup iterations.',
            'One-epoch canary closes mosaic at startup; full-data cost uses the intended100-epoch schedule, stopped after epoch1.',
            'Loss items are dictionaries; strict health/FP32 saving replaces the library default sanitizer.',
            'Source dataset cache hash includes scan settings; preserve the private-label view with the new hash.'],
        'source_args':source(original_args),'sources':{str(p.relative_to(PUBLIC)):sha256(p) for p in
            [HERE/'training.py',HERE/'study.py',HERE/'research_inference.py',
             HERE.parent/'cv261109/generator.py',HERE.parent/'cv261109/model.py',
             VM36/'feature_models/fm_p.uvl',VM36/'feature_models/fm_d.uvl']+
            [PUBLIC/'code/main'/n for n in ('training_health.py','training_data.py','training_effects.py','predictor.py','predictor_stable.py','labels.py','statistics.py')]}}
    save(out/'protocol.json',plan,out)
    snapshots=out/'source_snapshots';snapshots.mkdir()
    for relative in plan['sources']:shutil.copyfile(PUBLIC/relative,snapshots/relative.replace('\\','__').replace('/','__'))
    return out


class TimedLoader:
    def __init__(self,loader,trainer):self.loader=loader;self.trainer=trainer
    def __len__(self):return len(self.loader)
    def __getattr__(self,name):return getattr(self.loader,name)
    def __iter__(self):
        iterator=iter(self.loader)
        for _ in range(len(self.loader)):
            tick=time.perf_counter();batch=next(iterator)
            self.trainer.loader_seconds=time.perf_counter()-tick
            self.trainer.anchor_ids=[Path(p).stem for p in batch['im_file']]
            yield batch


def verify_input_preservation(out,protected):
    evidence={'unchanged':all(sha256(Path(row['path']))==row['sha256'] for row in protected),'inputs':protected}
    save(out/'protected_inputs.json',evidence,out)
    if not evidence['unchanged']:raise ValueError('A frozen training or validation input changed.')
    return evidence


def trainer_class(plan,out):
    import torch
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from main.training_health import RQ4HealthHooks,model_health,finite_tree,same_state,cpu_state,epoch_start
    from main.training_data import trainer_class as dataset_trainer
    from main.training_effects import engineering_fixture

    class Trainer(RQ4HealthHooks,DetectionTrainer):
        def __init__(self,*args,**kwargs):
            self.rq4_context={'output':str(out),'selection':'FIXED_LAST','scope':plan['scope'],
                'initialization_SHA256':plan['initialization']['sha256'],'protocol_SHA256':sha256(out/'protocol.json')}
            self.rq4_optimizer_updates=0;self.rows=[];self.builds=0;self.forward_count=0;self.last_forward_started=None
            self.validations=[];self.loader_seconds=0.;self.anchor_ids=[];self.backward_count=0
            super().__init__(*args,**kwargs)
            self.add_callback('on_pretrain_routine_end',self.accept_start)
            self.add_callback('on_train_epoch_start',epoch_start)
            self.add_callback('on_train_batch_start',self.batch_start)
            self.add_callback('on_train_batch_end',self.batch_end)
            self.add_callback('on_train_epoch_end',self.epoch_end)
            self.add_callback('on_fit_epoch_end',self.stop_cost_epoch)

        def get_dataset(self):
            from main.labels import VIS_NAMES
            return {'train':str(out/'train_images.txt'),'val':str(out/'val_images.txt'),'names':dict(enumerate(VIS_NAMES)),
                'nc':10,'channels':3,'yaml_file':str(out/'dataset.yaml')}

        def build_dataset(self,img_path,mode='train',batch=None):
            target=plan['targeted_recipe'] if mode=='train' else None
            context={'labels':plan['train_labels'] if mode=='train' else plan['val_labels'],
                'evidence':str(out/'dataset_cache'/mode),'recipe':target or engineering_fixture(()),
                'seed':plan['args']['seed'],'engineering':target is None}
            return dataset_trainer(context).build_dataset(self,img_path,mode,batch)

        def _build_train_pipeline(self):
            self.builds+=1
            if self.builds!=1:raise RuntimeError('Automatic OOM configuration reduction is forbidden.')
            return super()._build_train_pipeline()

        def _handle_nan_recovery(self,epoch):
            self.check_batch();model_health(self.ema.ema)
            if not finite_tree(self.metrics):raise FloatingPointError('Nonfinite validation metrics.')
            return False

        def accept_start(self,trainer):
            initial=torch.load(plan['initialization']['path'],map_location='cpu',weights_only=False)['model']
            if not same_state(self.model,cpu_state(initial)):raise ValueError('Native10 initialization changed in the new trainer.')
            model_health(self.model);model_health(self.ema.ema)
            if self.amp or type(self.optimizer).__name__!='AdamW' or self.batch_size!=2:raise ValueError('Training contract differs.')
            self.model.end2end=False;self.ema.ema.end2end=False
            self.train_loader=TimedLoader(self.train_loader,self)
            self.install_counter()
            torch.cuda.reset_peak_memory_stats()
            self._oom_retries=3
            save(out/'trainer_admission.json',{'initial_state_equal':True,'amp':self.amp,'optimizer':type(self.optimizer).__name__,
                'head_class':type(self.model.model[-1]).__name__,'one2many_present':self.model.model[-1].cv2 is not None,
                'dual_training_head':getattr(self.model.model[-1],'one2one_cv2',None) is not None,
                'train_images':len(self.train_loader.dataset),'validation_images':len(self.test_loader.dataset),
                'expected':plan['expected'],'actual_warmup':self._get_warmup_iterations(len(self.train_loader)),
                'initial_accumulation':self.accumulate,'FP32':all(not t.is_floating_point() or t.dtype==torch.float32 for t in self.model.state_dict().values())},out)

        def install_counter(self):
            def started(module,inputs):
                if inputs and isinstance(inputs[0],dict):
                    self.forward_begin=time.perf_counter();self.forward_event=torch.cuda.Event(enable_timing=True);self.forward_event.record()
            def completed(module,inputs,output):
                if inputs and isinstance(inputs[0],dict):
                    event=torch.cuda.Event(enable_timing=True);event.record();torch.cuda.synchronize()
                    self.forward_count+=1;self.forward_seconds=time.perf_counter()-self.forward_begin
                    self.forward_CUDA_ms=self.forward_event.elapsed_time(event)
            self.counter_pre=self.model.register_forward_pre_hook(started)
            self.counter_hook=self.model.register_forward_hook(completed)

        def batch_start(self,trainer):
            torch.cuda.synchronize();self.batch_begin=time.perf_counter()
            if self.batch_size!=2 or self.args.batch!=2:raise RuntimeError('Batch reduction is forbidden.')
            self._oom_retries=3
            if utc()>=__import__('datetime').datetime.fromisoformat(load(out/'WORKER_LEASE.json')['deadline_UTC']):raise TimeoutError('Worker deadline reached.')
            self.event({'event':'batch_started','position':len(self.rows),'anchors':self.anchor_ids})

        def preprocess_batch(self,batch):
            batch=super().preprocess_batch(batch)
            if batch['img'].shape[1:]!=(3,1280,1280) or batch['img'].dtype!=torch.float32:raise ValueError('Training tensor differs.')
            self.tensor_shape=list(batch['img'].shape)
            return batch

        def batch_end(self,trainer):
            self.check_batch();torch.cuda.synchronize()
            row={'position':len(self.rows),'epoch':self.epoch+1,'image_ids':self.anchor_ids,'NCHW':self.tensor_shape,
                'loader_seconds':self.loader_seconds,'forward_backward_health_seconds':time.perf_counter()-self.batch_begin,
                'loss':float(self.loss.detach()),'loss_items':{k:float(v.detach()) for k,v in self.loss_items.items()},
                'optimizer_updates':self.rq4_optimizer_updates,'EMA_updates':self.ema.updates,'accumulation':self.accumulate,
                'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved(),
                'forward_loss_wall_seconds':self.forward_seconds,'forward_loss_CUDA_interval_ms':self.forward_CUDA_ms,
                'backward_wall_seconds':self.backward_seconds,'backward_CUDA_interval_ms':self.backward_CUDA_ms,
                'healthy':True,'AMP':False}
            self.rows.append(row)
            with (out/'training_batches.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')
            events=self.train_loader.dataset.effect_events
            with (out/'augmentation_events.jsonl').open('a',encoding='utf-8') as stream:
                for event in events:stream.write(json.dumps(event)+'\n')
            events.clear()
            if len(self.rows)%100==0:print('MEASURED_BATCH',len(self.rows),'/',plan['expected']['batches'],flush=True)

        def epoch_end(self,trainer):
            self.peak={'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}
            self.event({'event':'epoch_forward_backward_complete','epoch':self.epoch+1,'batches':len(self.rows),
                'optimizer_updates':self.rq4_optimizer_updates,'EMA_updates':self.ema.updates,'GPU_peak':self.peak})

        def validate(self):
            tick=time.perf_counter();value=super().validate();torch.cuda.synchronize()
            model_health(self.ema.ema)
            self.validations.append({'seconds':time.perf_counter()-tick,'metrics':value[0],
                'scope':'ULTRALYTICS_INTERNAL_DEVELOPMENT_ONLY_NOT_RESEARCH_IGNORE_UNION'})
            save(out/'internal_validation.json',self.validations,out,replace=(out/'internal_validation.json').exists())
            return value

        def save_model(self):
            self.counter_pre.remove();self.counter_hook.remove()
            result=RQ4HealthHooks.save_model(self)
            self.install_counter()
            return result

        def stop_cost_epoch(self,trainer):
            if self.epoch+1==plan['stop_after_completed_epochs']:self.stop=True

    return Trainer


def worker(out):
    tick=time.perf_counter();runtime(out);import torch
    from unittest.mock import patch
    from ultralytics.utils import callbacks
    from post80_loader import read_only_sources
    from main.training_health import model_health
    import yaml
    from main.labels import VIS_NAMES
    plan=load(out/'protocol.json');trainer=None;status='FAILED';failure=None
    try:
        if importlib.metadata.version('ultralytics')!=plan['library']:raise ValueError('Wrong training version.')
        for relative,digest in plan['sources'].items():
            if sha256(PUBLIC/relative)!=digest:raise ValueError('Frozen code changed.')
        torch.set_num_threads(2);__import__('cv2').setNumThreads(2)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable.')
        for ref in (plan['initialization'],plan['COCO_source']):
            if sha256(Path(ref['path']))!=ref['sha256']:raise ValueError('Source weights changed.')
        protected=[]
        for row in plan['train']:
            for path,digest in ((row['image_path'],row['image_SHA256']),(plan['train_labels'][row['image_id']],row['derived_label_SHA256'])):
                if sha256(Path(path))!=digest:raise ValueError('Training input changed.')
                protected.append({'path':path,'sha256':digest})
        for row in plan['development']:
            for path,digest in ((row['loader_path'],row['source_image_SHA256']),(plan['val_labels'][row['stem']],row['derived_label_SHA256'])):
                if sha256(Path(path))!=digest:raise ValueError('Validation input changed.')
                protected.append({'path':path,'sha256':digest})
        write_atomic(out/'train_images.txt','\n'.join(r['image_path'] for r in plan['train'])+'\n',out)
        write_atomic(out/'val_images.txt','\n'.join(r['loader_path'] for r in plan['development'])+'\n',out)
        write_atomic(out/'dataset.yaml',yaml.safe_dump({'train':str(out/'train_images.txt'),'val':str(out/'val_images.txt'),'names':dict(enumerate(VIS_NAMES))}),out)
        torch.cuda.reset_peak_memory_stats();setup_tick=time.perf_counter()
        original_backward=torch.autograd.backward
        def timed_backward(*args,**kwargs):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();began=time.perf_counter();result=original_backward(*args,**kwargs);end.record();torch.cuda.synchronize()
            trainer.backward_count+=1;trainer.backward_seconds=time.perf_counter()-began;trainer.backward_CUDA_ms=start.elapsed_time(end)
            return result
        with patch.object(callbacks,'add_integration_callbacks',lambda instance:None),patch.object(torch.autograd,'backward',timed_backward),read_only_sources(out):
            trainer=trainer_class(plan,out)(overrides=plan['args']);trainer.train()
        training_seconds=time.perf_counter()-setup_tick
        ids=[i for row in trainer.rows for i in row['image_ids']]
        if len(ids)!=len(plan['train']) or len(set(ids))!=len(ids) or set(ids)!={r['image_id'] for r in plan['train']}:
            raise ValueError('Incomplete epoch membership.')
        if len(trainer.rows)!=plan['expected']['batches'] or trainer.rq4_optimizer_updates!=plan['expected']['updates'] or trainer.ema.updates!=trainer.rq4_optimizer_updates:
            raise ValueError('Actual batch/update counts differ.')
        model_health(trainer.model,gradients=True);model_health(trainer.ema.ema)
        checkpoint=trainer.rq4_selected
        measured={'completed_epochs':1,'training_images':len(ids),'validation_images':len(plan['development']),
            'forward_batches':trainer.forward_count,'backward_batches':trainer.backward_count,'optimizer_updates':trainer.rq4_optimizer_updates,
            'EMA_updates':trainer.ema.updates,'pending_microbatches':plan['expected']['pending_microbatches'],
            'training_setup_epoch_internal_val_save_seconds':training_seconds,'GPU_peak_bytes':trainer.peak,
            'loader_seconds':sum(r['loader_seconds'] for r in trainer.rows),
            'forward_backward_health_seconds':sum(r['forward_backward_health_seconds'] for r in trainer.rows),
            'forward_loss_wall_seconds':sum(r['forward_loss_wall_seconds'] for r in trainer.rows),
            'backward_wall_seconds':sum(r['backward_wall_seconds'] for r in trainer.rows),
            'forward_loss_CUDA_interval_ms':sum(r['forward_loss_CUDA_interval_ms'] for r in trainer.rows),
            'backward_CUDA_interval_ms':sum(r['backward_CUDA_interval_ms'] for r in trainer.rows),
            'internal_validation':trainer.validations,'checkpoint':source(checkpoint),'precision':'FP32',
            'exact_resume':'NOT_VALIDATED','full100_training_arm_completed':False}
        save(out/'training_acceptance.json',measured,out)
        del trainer;trainer=None;gc.collect();torch.cuda.empty_cache()
        from research_inference import infer
        with patch.object(callbacks,'add_integration_callbacks',lambda instance:None),read_only_sources(out):infer(out,plan,checkpoint)
        verify_input_preservation(out,protected)
        status='PASS'
    except BaseException:
        failure=traceback.format_exc()
        measured=load(out/'training_acceptance.json') if (out/'training_acceptance.json').exists() else {}
        save(out/'failure.json',{'traceback':failure,'optimizer_updates':getattr(trainer,'rq4_optimizer_updates',measured.get('optimizer_updates')),
            'completed_batches':len(trainer.rows) if trainer is not None else measured.get('backward_batches'),'automatic_retry':False},out)
    save(out/'worker_result.json',{'status':status,'worker_seconds':time.perf_counter()-tick,
        'mode':plan['mode'],'architecture':plan['architecture'],'matched_scientific_pairs':0,'has_failure':failure is not None},out)


def execute(mode,architecture,arm='CONTROL'):
    from development_guard import bounded_run
    if mode=='full_epoch':
        accepted=load(RUNS/'admission_result.json')
        if accepted['status']!='PASS' or accepted['architecture']!='YOLO26s':raise ValueError('YOLO26 admission must pass first.')
    out=prepare(mode,architecture,arm)
    snap=subprocess.run(['nvidia-smi','--query-gpu=name,memory.free,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15,check=True).stdout.strip()
    parts=[r.strip() for r in snap.split(',')]
    if int(parts[1])<5*1024 or int(parts[2])>=85 or int(parts[3])>10:raise RuntimeError('GPU readiness failed.')
    save(out/'gpu_before.json',{'snapshot':snap},out)
    cap=budget_start(out,600 if mode=='canary' else 6600,'ADMISSION' if mode=='canary' else 'FULL_DATA_COST');began=time.perf_counter();error=None
    from telemetry import watch
    watcher=threading.Thread(target=watch,args=(out.name,),daemon=True);watcher.start()
    try:
        with (out/'worker.log').open('x',encoding='utf-8') as log:
            code=bounded_run([sys.executable,'-B','-X','utf8',str(Path(__file__)),'--worker',out.name],log,cap,
                dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',OMP_NUM_THREADS='2',PYTHONIOENCODING='utf-8'))
        if code:error='Worker exited with code'+str(code)
    except BaseException:error=traceback.format_exc()
    result=load(out/'worker_result.json') if (out/'worker_result.json').exists() else {'status':'INTERRUPTED'}
    seconds=time.perf_counter()-began;budget_finish(out,seconds,result['status'])
    watcher.join(timeout=12)
    save(out/'supervisor.json',{'wall_seconds':seconds,'error':error,'hard_cap':cap},out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,**result},out,replace=True)
    if mode=='canary' and architecture=='YOLO26s':
        save(RUNS/'admission_result.json',{'RunID':out.name,'architecture':architecture,'status':result['status']},RUNS,replace=(RUNS/'admission_result.json').exists())
    write_atomic(out/'diary.md','# '+mode+'\n\nFP32 training, strict health checks and atomic snapshots. Engineering/cost evidence only; no completed matched scientific pair.\n',out)
    save(out/'nested_SHA256.json',{str(p.relative_to(out)):sha256(p) for folder in ('weights','dataset_cache') for p in (out/folder).rglob('*') if p.is_file()},out)
    seal(out);print(out,flush=True)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=('canary','full_epoch'));parser.add_argument('--architecture',choices=('YOLO26s','YOLOv8s'),default='YOLO26s');parser.add_argument('--worker')
    parser.add_argument('--arm',choices=('CONTROL','ALL_FIVE'),default='CONTROL')
    args=parser.parse_args()
    if args.worker:worker(run_path(args.worker))
    elif args.mode:execute(args.mode,args.architecture,args.arm)
    else:parser.error('Choose one bounded mode.')
