"""One authorized YOLOv8s GPU engineering canary under a 600-second cap."""
import argparse
from collections import Counter
from datetime import datetime,timezone,timedelta
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from protocol import ROOT,HERE,EXPERIMENT,load,save,source,sha256,write_atomic
from common import seal,verify_hashes
from main.pipeline import new_run
from post80_replay import run_path

PREPARED='vm36_supervisor_reproduction_rq4_training_admission_cpu_20261111T095136_2986cfa7'
SOURCE='vm36_supervisor_reproduction_rq4_source_admission_cpu_20261111T100553_926d0f31'
BACKEND='vm36_supervisor_reproduction_rq4_backend_cpu_20261111T101533_5b6f7aa4'
BUDGET=600


def gpu_runtime(out):
    os.environ.update(CUDA_VISIBLE_DEVICES='0',YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',PYTHONDONTWRITEBYTECODE='1')
    for name in ('YOLO_CONFIG_DIR','MPLCONFIGDIR','TORCH_HOME','TORCH_EXTENSIONS_DIR','TORCHINDUCTOR_CACHE_DIR','CUDA_CACHE_PATH','TEMP','TMP'):
        path=out/'runtime'/name;path.mkdir(parents=True,exist_ok=True);os.environ[name]=str(path)
    import tempfile
    tempfile.tempdir=str(out/'runtime/TEMP')
    import torch,cv2
    torch.set_num_threads(2);cv2.setNumThreads(2)


def canary_contract(plan):
    args=plan['args']
    expected={'epochs':1,'imgsz':1280,'batch':2,'workers':0,'nbs':64,'optimizer':'AdamW','lr0':.01,
        'cos_lr':True,'warmup_epochs':3.,'seed':42,'amp':True,'resume':False}
    if plan.get('precision_diagnostic',{}).get('mode')=='EXPLICIT_FP32_REPAIR':
        expected['amp']=False
        if not plan['precision_diagnostic'].get('failed_run'):raise ValueError('Missing failed-run evidence.')
    if any(args.get(k)!=v for k,v in expected.items()):raise ValueError('The authorized canary configuration changed.')
    if len(plan['train'])!=64 or len(plan['development'])!=16:raise ValueError('Canary membership changed.')
    for rows,key in ((plan['train'],'image_id'),(plan['development'],'stem')):
        if len({r[key] for r in rows})!=len(rows):raise ValueError('Duplicate canary image identity.')
    if plan['worker_seconds']!=600 or plan['new_target_augmentations']!=[]:raise ValueError('Wrong scope or budget.')


def save_without_counter_hook(trainer):
    from rq4_training_backend import RQ4HealthHooks
    trainer.rq4_forward_handle.remove()
    return RQ4HealthHooks.save_model(trainer)


def prepare(request):
    import yaml
    from rq4_preparation import optimizer_exposure_plan
    for identity,names in ((PREPARED,('CANARY_IMAGE_CANDIDATES.json','CANDIDATE_INITIALIZATION_MANIFEST.json')),
            (SOURCE,('INITIALIZATION_SOURCE_GATE.json',)),(BACKEND,('BACKEND_CPU_ACCEPTANCE.json',))):
        verify_hashes(run_path(identity),names)
    selected=load(run_path(PREPARED)/'CANARY_IMAGE_CANDIDATES.json')
    initial=next(r for r in load(run_path(PREPARED)/'CANDIDATE_INITIALIZATION_MANIFEST.json') if r['architecture']=='YOLOv8s' and r['seed']==42)
    source_weight=initial['source'];native=initial['native10_candidate']
    for ref in (source_weight,native):
        if sha256(Path(ref['path']))!=ref['sha256']:raise ValueError('Initialization SHA changed.')
    labels=load(run_path(PREPARED)/'PRIVATE_LABEL_LOOKUP.json');val_labels={r['stem']:str(run_path(PREPARED)/'validation_labels'/(r['stem']+'.txt')) for r in selected['development']}
    protected=[]
    for row in selected['train']:
        for path,digest in ((Path(row['image_path']),row['image_SHA256']),(Path(labels[row['image_id']]),row['derived_label_SHA256'])):
            if sha256(path)!=digest:raise ValueError('Training input changed.')
            protected.append({'path':str(path),'sha256':digest})
    for row in selected['development']:
        for path,digest in ((Path(row['loader_path']),row['source_image_SHA256']),(Path(val_labels[row['stem']]),row['derived_label_SHA256'])):
            if sha256(path)!=digest:raise ValueError('Development input changed.')
            protected.append({'path':str(path),'sha256':digest})
    root=ROOT/'datasets/local_runs'/EXPERIMENT
    if any(root.glob('.*gpu.lock')):raise RuntimeError('An existing GPU experiment lock blocks the canary.')
    snapshot=subprocess.run(['nvidia-smi','--query-gpu=name,memory.free,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=15).stdout.strip()
    fields=[s.strip() for s in snapshot.split(',')]
    if int(fields[1])<5*1024 or int(fields[2])>=85 or int(fields[3])>10:raise RuntimeError('GPU readiness gate failed.')
    if shutil.disk_usage(ROOT).free<8*1024**3:raise RuntimeError('Insufficient disk reserve.')
    out,_=new_run({'output_root':str(root)},EXPERIMENT,'rq4_real_gpu_canary')
    shutil.copyfile(request,out/'author_task.txt');shutil.copyfile(Path(__file__),out/'executed_source.py')
    args=yaml.safe_load((ROOT/'models/checkpoints/ck05_yolov8s_ft1280/args.yaml').read_text())
    args.pop('save_dir',None)
    args.update(model=native['path'],data=str(out/'dataset.yaml'),epochs=1,workers=0,device='0',project=str(out),
        name='trainer',exist_ok=False,plots=False,cache=False,save=True,save_period=-1,resume=False)
    plan={'scope':'RQ4_REAL_YOLOV8S_ENGINEERING_CANARY','worker_seconds':BUDGET,'args':args,
        'initialization':native,'COCO_source':source_weight,'train':selected['train'],'development':selected['development'],
        'train_labels':{r['image_id']:labels[r['image_id']] for r in selected['train']},'val_labels':val_labels,
        'new_target_augmentations':[],'expected_update_plan':optimizer_exposure_plan(64,1,2,64,3.),
        'source_version':'8.4.61','protected_inputs':protected,'GPU_before':snapshot,
        'dependencies':{str(path.relative_to(ROOT/'main')):sha256(path) for path in [HERE/name for name in
            ('rq4_dataset.py','rq4_augmentation.py','rq4_training_backend.py')]+[ROOT/'main/code/main'/name for name in
            ('predictor.py','labels.py','statistics.py')]+[HERE.parent/'cv261207'/name for name in ('evaluation.py','development_v1.py')]},
        'operational_overrides':['One engineering epoch,64/16 subset,workers0,local outputs,no plots/no integrations',
            'FP32 internal validation to retain FP32 EMA; training AMP remains inherited true',
            'AMP check uses the same official YOLOv8s source instead of an implicit YOLO26n download',
            'No automatic OOM batch reduction or NaN recovery; no optimizer stripping'],
        'selection':'FIXED_LAST_ENGINEERING_ONLY','statistical_uncertainty':'NOT_EVALUATED_ENGINEERING_ONLY',
        'scientific_training_result':False,'author_request':source(out/'author_task.txt')}
    canary_contract(plan);save(out/'CANARY_PROTOCOL.json',plan,out)
    save(out/'AUTHORIZATION.json',{'scope':plan['scope'],'new_worker_seconds':600,'source_task':source(out/'author_task.txt'),
        'recorded_UTC':datetime.now(timezone.utc).isoformat(),'formal_training_seconds_authorized':0,
        'previous_budgets_reused':False,'automatic_retry':False},out)
    return out


class TimedLoader:
    def __init__(self,loader,trainer):self.loader=loader;self.trainer=trainer
    def __len__(self):return len(self.loader)
    def __getattr__(self,name):return getattr(self.loader,name)
    def __iter__(self):
        iterator=iter(self.loader)
        for _ in range(len(self.loader)):
            started=time.perf_counter();batch=next(iterator)
            self.trainer.rq4_loader_seconds=time.perf_counter()-started
            self.trainer.rq4_anchor_ids=[Path(p).stem for p in batch['im_file']]
            yield batch


def train_class(plan,out):
    import torch
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from rq4_dataset import trainer_class
    from rq4_augmentation import engineering_fixture
    from rq4_training_backend import RQ4HealthHooks,model_health,finite_tree,epoch_start

    class CanaryTrainer(RQ4HealthHooks,DetectionTrainer):
        def __init__(self,*args,**kwargs):
            canary_contract(plan)
            allocation=load(out/'GPU_ALLOCATION.json')
            if allocation['worker_seconds']!=600 or datetime.now(timezone.utc)>=datetime.fromisoformat(allocation['deadline_UTC']):
                raise RuntimeError('The canary allocation is absent or expired.')
            self.rq4_context={'output':str(out),'selection':'FIXED_LAST','scope':'CANARY_ENGINEERING_ONLY_DEVELOPMENT_ONLY',
                'initialization_SHA256':plan['initialization']['sha256'],'protocol_SHA256':sha256(out/'CANARY_PROTOCOL.json')}
            self.rq4_optimizer_updates=0;self.rq4_forwards=0;self.rq4_backwards=0;self.rq4_batches=[]
            self.rq4_seen_backward=-1;self.rq4_builds=0;self.rq4_validations=[];self.rq4_loader_seconds=0.
            super().__init__(*args,**kwargs)
            self.add_callback('on_pretrain_routine_end',self.accept_start)
            self.add_callback('on_train_epoch_start',epoch_start)
            self.add_callback('on_train_batch_start',self.batch_start)
            self.add_callback('on_train_batch_end',self.batch_end)
            self.add_callback('on_train_epoch_end',self.epoch_end)

        def get_dataset(self):
            from main.labels import VIS_NAMES
            return {'train':str(out/'train_images.txt'),'val':str(out/'val_images.txt'),
                'names':dict(enumerate(VIS_NAMES)),'nc':10,'channels':3,'yaml_file':str(out/'dataset.yaml')}

        def build_dataset(self,img_path,mode='train',batch=None):
            context={'labels':plan['train_labels'] if mode=='train' else plan['val_labels'],
                'evidence':str(out/'dataset_cache'/mode),'recipe':engineering_fixture(()),'seed':42,'engineering':True}
            return trainer_class(context).build_dataset(self,img_path,mode,batch)

        def _build_train_pipeline(self):
            self.rq4_builds+=1
            if self.rq4_builds!=1:raise RuntimeError('Automatic pipeline rebuild/OOM retry is forbidden.')
            return super()._build_train_pipeline()

        def _handle_nan_recovery(self,epoch):
            self.check_batch();model_health(self.ema.ema)
            if not finite_tree(self.metrics):raise FloatingPointError('Nonfinite validation metrics.')
            return False

        def accept_start(self,trainer):
            from rq4_training_backend import same_state,cpu_state
            initial=torch.load(plan['initialization']['path'],map_location='cpu',weights_only=False)['model']
            if not same_state(self.model,cpu_state(initial)):raise ValueError('Trainer did not load the frozen starting state.')
            model_health(self.model);model_health(self.ema.ema)
            if bool(self.amp)!=plan['args']['amp']:raise RuntimeError('Actual precision differs from the explicit protocol.')
            if type(self.optimizer).__name__!='AdamW' or self.batch_size!=2:raise ValueError('Optimizer or batch differs.')
            self.train_loader=TimedLoader(self.train_loader,self)
            def forward_finished(module,inputs,output):
                if inputs and isinstance(inputs[0],dict):self.rq4_forwards+=1
            self.rq4_forward_handle=self.model.register_forward_hook(forward_finished)
            self.rq4_initial_peak={'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}
            torch.cuda.reset_peak_memory_stats()
            save(out/'TRAINER_START_ACCEPTANCE.json',{'initial_state_equal':True,'AMP':self.amp,'batch':self.batch_size,
                'optimizer':type(self.optimizer).__name__,'train_images':len(self.train_loader.dataset),
                'development_images':len(self.test_loader.dataset),'validation_batch':self.test_loader.batch_size,
                'initial_accumulation':self.accumulate,'model_state_dtypes':sorted({str(v.dtype) for v in self.model.state_dict().values() if v.is_floating_point()}),
                'EMA_state_dtypes':sorted({str(v.dtype) for v in self.ema.ema.state_dict().values() if v.is_floating_point()})},out)

        def batch_start(self,trainer):
            if self.batch_size!=2 or self.args.batch!=2:raise RuntimeError('Batch configuration changed.')
            torch.cuda.synchronize();self.rq4_batch_begin=time.perf_counter()
            self.event({'event':'training_batch_started','batch_index':len(self.rq4_batches),
                'anchors':self.rq4_anchor_ids,'loader_seconds':self.rq4_loader_seconds})

        def preprocess_batch(self,batch):
            batch=super().preprocess_batch(batch)
            if list(batch['img'].shape)!=[2,3,1280,1280] or batch['img'].dtype!=torch.float32:raise ValueError('Actual training tensor changed.')
            self.rq4_tensor={'NCHW':list(batch['img'].shape),'dtype':str(batch['img'].dtype),'device':str(batch['img'].device)}
            return batch

        def mark_backward(self):
            index=len(self.rq4_batches)
            if self.rq4_seen_backward!=index:self.rq4_backwards+=1;self.rq4_seen_backward=index

        def optimizer_step(self):
            self.mark_backward()
            self.event({'event':'before_optimizer_health','batch_index':len(self.rq4_batches),
                'loss':float(self.loss.detach()),'AMP_scale':float(self.scaler.get_scale()),
                'forward_batches':self.rq4_forwards,'backward_batches':self.rq4_backwards})
            return RQ4HealthHooks.optimizer_step(self)

        def batch_end(self,trainer):
            self.mark_backward();self.check_batch();torch.cuda.synchronize()
            row={'batch_index':len(self.rq4_batches),'image_ids':self.rq4_anchor_ids,'images':len(self.rq4_anchor_ids),
                'tensor':self.rq4_tensor,'batch_wall_seconds':time.perf_counter()-self.rq4_batch_begin,
                'loader_seconds':self.rq4_loader_seconds,'loss':float(self.loss.detach()),'loss_items':self.loss_items.detach().cpu().tolist(),
                'accumulation':self.accumulate,'optimizer_updates':self.rq4_optimizer_updates,'EMA_updates':self.ema.updates,
                'AMP_scale':float(self.scaler.get_scale()),'gradients_finite':True,
                'allocated_bytes':torch.cuda.memory_allocated(),'reserved_bytes':torch.cuda.memory_reserved()}
            self.rq4_batches.append(row)
            with (out/'training_batches.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(row)+'\n')

        def epoch_end(self,trainer):
            self.rq4_epoch_peak={'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}
            self.event({'event':'training_epoch_forward_backward_complete','batches':len(self.rq4_batches),
                'exposures':sum(r['images'] for r in self.rq4_batches),'optimizer_updates':self.rq4_optimizer_updates,
                'EMA_updates':self.ema.updates,'GPU_peak_bytes':self.rq4_epoch_peak})

        def save_model(self):
            return save_without_counter_hook(self)

        def validate(self):
            started=time.perf_counter();amp=self.amp;self.amp=False
            try:result=super().validate()
            finally:self.amp=amp
            torch.cuda.synchronize();model_health(self.ema.ema)
            if not finite_tree(result):raise FloatingPointError('Validation metrics are nonfinite.')
            self.rq4_validations.append({'seconds':time.perf_counter()-started,'actual_metrics':result[0],
                'precision':'FP32','scope':'ULTRALYTICS_INTERNAL_CANARY_ONLY_NOT_RESEARCH_IGNORE_UNION'})
            save(out/'INTERNAL_VALIDATION.json',self.rq4_validations,out,replace=(out/'INTERNAL_VALIDATION.json').exists())
            return result

    return CanaryTrainer


def amp_checker(plan,out):
    from unittest.mock import patch
    import torch,ultralytics
    from ultralytics.utils.checks import check_amp
    original=ultralytics.YOLO
    def checked(model):
        calls=[]
        def same_architecture(*args,**kwargs):
            calls.append({'requested':str(args[0]),'actual':plan['COCO_source']})
            return original(plan['COCO_source']['path'])
        with patch.object(ultralytics,'YOLO',side_effect=same_architecture):passed=check_amp(model)
        save(out/'AMP_CHECK.json',{'passed':bool(passed),'calls':calls,
            'method':'Installed check_amp with the same official YOLOv8s source; original image,256 sizing,8-image FP32/AMP check and0.5 tolerance unchanged.',
            'no_implicit_YOLO26n_download':True},out)
        if not passed:raise RuntimeError('Source-backed AMP check failed; no silent AMP disable.')
        torch.cuda.empty_cache();return True
    return checked


def compatibility(out,plan,checkpoint):
    started=time.perf_counter()
    import cv2,numpy as np,gzip
    from development_v1 import Model
    from main.labels import ground_truth,mapped_predictions
    from main.statistics import evaluate_coco
    from evaluation import evaluate_saved
    reference=load(run_path(PREPARED)/'inputs.json')['accepted84']
    normal=Path(reference['path']).parent;worker=Path(load(normal/'METRIC_PROVENANCE.json')['new_worker']['path']).parent
    payload=load(Path(load(worker/'EXECUTION_PROTOCOL.json')['GT']['path']));by_stem={r['stem']:r for r in payload['records']}
    records=[by_stem[r['stem']] for r in plan['development']];raw=[];observed=[]
    device=Model({'path':str(checkpoint),'sha256':sha256(checkpoint)});predict=device.detector({'resolution':1280,'precision':'FP32'})
    with gzip.open(out/'predictions.jsonl.gz','xt',encoding='utf-8') as stream:
        for ref,record in zip(plan['development'],records):
            if sha256(Path(ref['loader_path']))!=ref['source_image_SHA256']:raise ValueError('Compatibility image changed.')
            pixels=cv2.imdecode(np.frombuffer(Path(ref['loader_path']).read_bytes(),np.uint8),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            device.torch.cuda.synchronize();tick=time.perf_counter();result,tensor=predict(pixels);device.torch.cuda.synchronize()
            predict_ms=(time.perf_counter()-tick)*1000;boxes=result.boxes.data.detach().cpu().numpy()
            if not np.isfinite(boxes).all():raise FloatingPointError('Reloaded checkpoint prediction is nonfinite.')
            preds=[{'image_id':record['id'],'class_id':int(c),'bbox':[float(x1),float(y1),float(x2-x1),float(y2-y1)],'score':float(score)}
                for x1,y1,x2,y2,score,c in boxes if x2>x1 and y2>y1]
            stream.write(json.dumps({'image_id':record['id'],'predictions':preds})+'\n');raw.extend(preds)
            observed.append({'image_id':record['id'],'source_SHA256':ref['source_image_SHA256'],'tensor':tensor,
                'prediction_count':len(preds),'predict_ms':predict_ms})
    gt,regions=ground_truth(records,'native10');save(out/'ground_truth.json',{'coco':gt,'records':records},out)
    mapped=mapped_predictions(raw,records,regions,'native10','finetuned',3000)
    tick=time.perf_counter();areas,classes,evaluator=evaluate_coco(gt,mapped,3000);evaluation_seconds=time.perf_counter()-tick
    metrics={'per_area':areas,'per_class':classes,'raw_count':len(raw),'mapped_count':len(mapped),'images':16}
    save(out/'metrics.json',metrics,out)
    tick=time.perf_counter()
    if evaluate_saved(out)[0]!=metrics:raise ValueError('Canary raw-prediction replay differs.')
    replay_seconds=time.perf_counter()-tick
    save(out/'RELOAD_INFERENCE_ACCEPTANCE.json',{'status':'PASS','checkpoint':source(checkpoint),'images':16,
        'scope':'CANARY_ENGINEERING_ONLY_DEVELOPMENT_ONLY','actual_inputs':observed,'raw_COCO_replay_equal':True,
        'research_metrics_not_a_matched_training_comparison':True,'bootstrap':'NOT_EVALUATED_ENGINEERING_ONLY',
        'compatibility_wall_seconds':time.perf_counter()-started,'COCO_evaluation_seconds':evaluation_seconds,
        'raw_replay_seconds':replay_seconds},out)


def worker(out):
    started=time.perf_counter();gpu_runtime(out);import torch
    from unittest.mock import patch
    import ultralytics.engine.trainer as engine
    from ultralytics.utils import callbacks
    from rq4_training_backend import model_health
    from post80_loader import read_only_sources
    plan=load(out/'CANARY_PROTOCOL.json');canary_contract(plan);trainer=None
    if plan.get('precision_diagnostic'):
        failed=plan['precision_diagnostic']['failed_run']
        if sha256(Path(failed['path']))!=failed['sha256'] or load(Path(failed['path']))['status']!='CANARY_FAILED':
            raise ValueError('FP32 repair failure evidence changed.')
    for relative,digest in plan['dependencies'].items():
        if sha256(ROOT/'main'/relative)!=digest:raise ValueError('An accepted dependency changed.')
    for ref in plan['protected_inputs']+[plan['initialization'],plan['COCO_source']]:
        if sha256(Path(ref['path']))!=ref['sha256']:raise ValueError('Pre-worker input SHA differs.')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable.')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    write_atomic(out/'train_images.txt','\n'.join(r['image_path'] for r in plan['train'])+'\n',out)
    write_atomic(out/'val_images.txt','\n'.join(r['loader_path'] for r in plan['development'])+'\n',out)
    import yaml
    from main.labels import VIS_NAMES
    write_atomic(out/'dataset.yaml',yaml.safe_dump({'train':str(out/'train_images.txt'),'val':str(out/'val_images.txt'),
        'names':dict(enumerate(VIS_NAMES))},sort_keys=False),out)
    status='CANARY_FAILED';error=None
    try:
        with patch.object(callbacks,'add_integration_callbacks',lambda instance:None),patch.object(engine,'check_amp',amp_checker(plan,out)),read_only_sources(out):
            trainer=train_class(plan,out)(overrides=plan['args'])
            training_started=time.perf_counter();trainer.train();training_wall=time.perf_counter()-training_started
        expected=plan['expected_update_plan']['optimizer_and_EMA_updates']
        ids=[image for row in trainer.rq4_batches for image in row['image_ids']]
        if len(trainer.rq4_batches)!=32 or len(ids)!=64 or set(ids)!={r['image_id'] for r in plan['train']} or len(set(ids))!=64:
            raise ValueError('Actual training exposures differ.')
        if trainer.rq4_optimizer_updates!=expected or trainer.ema.updates!=expected or trainer.rq4_forwards!=32 or trainer.rq4_backwards!=32:
            raise ValueError('Actual forward/backward/optimizer/EMA counts differ.')
        model_health(trainer.model,gradients=True);model_health(trainer.ema.ema)
        checkpoint=trainer.rq4_selected
        training={'epochs_completed':1,'training_exposures':64,'forward_batches':trainer.rq4_forwards,'backward_batches':trainer.rq4_backwards,
            'optimizer_updates':trainer.rq4_optimizer_updates,'EMA_updates':trainer.ema.updates,
            'pending_gradient_tensors':sum(p.grad is not None for p in trainer.model.parameters()),
            'pending_microbatches_after_last_optimizer':31-max(r['batch_index'] for r in trainer.rq4_batches if r['batch_index']==0 or r['optimizer_updates']>trainer.rq4_batches[r['batch_index']-1]['optimizer_updates']),
            'training_setup_epoch_val_save_seconds':training_wall,'internal_validation':trainer.rq4_validations,
            'checkpoint':source(checkpoint),'model_state_dtypes':sorted({str(v.dtype) for v in trainer.model.state_dict().values() if v.is_floating_point()}),
            'EMA_state_dtypes':sorted({str(v.dtype) for v in trainer.ema.ema.state_dict().values() if v.is_floating_point()}),
            'training_epoch_GPU_peak':trainer.rq4_epoch_peak,'initialization_GPU_peak':trainer.rq4_initial_peak,
            'training_validation_save_GPU_peak':{'allocated':torch.cuda.max_memory_allocated(),'reserved':torch.cuda.max_memory_reserved()}}
        save(out/'REAL_TRAINING_ACCEPTANCE.json',training,out)
        trainer.rq4_forward_handle.remove();del trainer;trainer=None;gc.collect();torch.cuda.empty_cache()
        with patch.object(callbacks,'add_integration_callbacks',lambda instance:None),read_only_sources(out):compatibility(out,plan,checkpoint)
        status='CANARY_PASS'
    except BaseException:
        error=traceback.format_exc();save(out/'failure.json',{'traceback':error,'automatic_retry':False,
            'actual_optimizer_updates':getattr(trainer,'rq4_optimizer_updates',0),'actual_forward_batches':getattr(trainer,'rq4_forwards',0),
            'actual_backward_batches':getattr(trainer,'rq4_backwards',0),'health_accepted_batches':len(getattr(trainer,'rq4_batches',[])),
            'AMP_scale':float(trainer.scaler.get_scale()) if trainer is not None and hasattr(trainer,'scaler') else None},out)
    save(out/'worker_result.json',{'status':status,'worker_seconds':time.perf_counter()-started,'hard_limit_seconds':600,
        'scope':'CANARY_ENGINEERING_ONLY_DEVELOPMENT_ONLY','scientific_training_runs':0,'full_training_epochs':0,
        'complete_matched_training_pairs':0,'source_inputs_unchanged':all(sha256(Path(r['path']))==r['sha256'] for r in plan['protected_inputs']+[plan['initialization'],plan['COCO_source']]),
        'has_failure':error is not None},out)


def execute(request):
    from development_guard import bounded_run
    out=prepare(request);root=out.parent;ticket=root/('.rq4_canary_'+sha256(out/'author_task.txt')[:16]+'.ticket')
    handle=os.open(ticket,os.O_WRONLY|os.O_CREAT|os.O_EXCL);os.write(handle,str(out).encode());os.close(handle)
    lock=root/'.rq4_real_canary_gpu.lock';handle=os.open(lock,os.O_WRONLY|os.O_CREAT|os.O_EXCL);os.write(handle,str(os.getpid()).encode());os.close(handle)
    began=time.perf_counter();start_UTC=datetime.now(timezone.utc)
    save(out/'GPU_ALLOCATION.json',{'worker_seconds':600,'start_UTC':start_UTC.isoformat(),
        'deadline_UTC':(start_UTC+timedelta(seconds=600)).isoformat(),'author_task_SHA256':sha256(out/'author_task.txt'),
        'budget_reset_allowed':False,'formal_training_seconds':0},out)
    try:
        with (out/'worker.log').open('x',encoding='utf-8') as log:
            code=bounded_run([sys.executable,'-B',str(Path(__file__)),'--worker',out.name],log,600,
                dict(os.environ,CUDA_VISIBLE_DEVICES='0',PYTHONDONTWRITEBYTECODE='1',YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',OMP_NUM_THREADS='2',PYTHONIOENCODING='utf-8'))
        if code:raise RuntimeError('GPU canary worker failed; evidence preserved.')
    except BaseException:save(out/'supervisor_failure.json',{'traceback':traceback.format_exc(),'automatic_retry':False},out)
    finally:
        if lock.exists() and lock.read_text()==str(os.getpid()):lock.unlink()
    result=load(out/'worker_result.json') if (out/'worker_result.json').exists() else {'status':'CANARY_INTERRUPTED'}
    save(out/'supervisor.json',{'hard_seconds':600,'actual_seconds':time.perf_counter()-began,'old_budgets_reused':False},out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'RQ4_REAL_GPU_CANARY',**result},out,replace=True)
    save(out/'NESTED_SHA256.json',{str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file()},out)
    write_atomic(out/'diary.md','# RQ4 real GPU canary\n\nNew600-second allocation, fixed64/16 engineering sample and inherited optimizer settings. No full matched arm, no old evaluation rerun and no scientific efficacy claim.\n',out)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>RQ4 real YOLO GPU canary</h1><p>'+result['status']+'</p><p><a href="worker_result.json">Actual status</a></p><p><a href="training_batches.jsonl">Training batches</a></p><p><a href="health_events.jsonl">Health checks</a></p><p><a href="REAL_TRAINING_ACCEPTANCE.json">Training acceptance</a></p>',out)
    seal(out)
    from main.visualization import build_run_index
    build_run_index(root,EXPERIMENT);print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--execute',type=Path);group.add_argument('--worker');args=parser.parse_args()
    if args.execute:execute(args.execute)
    else:worker(run_path(args.worker))
