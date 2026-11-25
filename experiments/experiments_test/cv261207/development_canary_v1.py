"""Pixel-augmentation hooks and one bounded clean-control training canary."""
import argparse
import copy
from datetime import datetime,timezone
from hashlib import sha256 as digest
import json
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from zipfile import ZipFile

import numpy as np
from common import ROOT,HERE,RUNS,ARCHIVES,load,save,source,sha256,new_run,resolve_run,seal,verify_hashes
from historical_review import csv_file
from evaluation import runtime


def augment_pixels(image,group,seed):
    import cv2
    rng=np.random.default_rng(seed);x=image.astype(np.float32);parameters={}
    if group=='clean_control':return image.copy(),parameters
    if group=='NoiseAug':
        sigma=float(rng.uniform(3,35))
        if sigma in (5.5,15,30):sigma=float(np.nextafter(sigma,35.))
        parameters={'sigma':sigma}
        x+=rng.normal(0,sigma,image.shape).astype(np.float32)
    elif group=='MotionBlurAug':
        k=int(rng.choice([9,11,13,17,19,23]));angle=float(rng.uniform(0,180));parameters={'kernel':k,'angle':angle}
        kernel=np.zeros((k,k),np.float32);kernel[k//2,:]=1
        kernel=cv2.warpAffine(kernel,cv2.getRotationMatrix2D((k/2-.5,k/2-.5),angle,1),(k,k));kernel/=kernel.sum()
        x=cv2.filter2D(image,-1,kernel).astype(np.float32)
    elif group=='ExposureAug':
        gain=float(rng.uniform(.35,1.4));bias=float(rng.uniform(-70,90));parameters={'gain':gain,'bias':bias};x=x*gain+bias
    elif group=='JPEGAug':
        quality=int(rng.choice([q for q in range(35,96) if q not in (25,50,75,85)]));parameters={'quality':quality}
        ok,data=cv2.imencode('.jpg',image,[cv2.IMWRITE_JPEG_QUALITY,quality])
        if not ok:raise ValueError('Augmentation encode failed.')
        x=cv2.imdecode(data,cv2.IMREAD_COLOR).astype(np.float32)
    elif group=='HazeAug':
        beta=float(rng.uniform(.15,1.8));light=float(rng.uniform(190,240))
        if beta in (.5,1.5):beta=float(np.nextafter(beta,1.8))
        parameters={'beta':beta,'airlight':light}
        t=float(np.exp(-beta));x=x*t+light*(1-t)
    else:raise ValueError('Unknown augmentation group.')
    return np.clip(x,0,255).astype(np.uint8),parameters


def prepare():
    import cv2
    cv2.setNumThreads(1)
    out=new_run('training_canary_prepare');runtime(out)
    audit=load(RUNS/'cv261207_audit_20261104T193649_d1b63492/dataset_gate.json')
    admission=load(RUNS/'cv261207_admission_20261104T211249_32d83def/admission.json')
    ck=next(c for c in admission['candidates'] if c['id']=='ck05')
    if sha256(Path(ck['checkpoint']['path']))!=ck['checkpoint_sha256']:raise ValueError('Starting weight changed.')
    old_ids={r['stem'] for r in load(ROOT/'datasets/research_archive/variability/experiments/vm03_gpu_pilot/runs/vm03_r003/records.json')}
    audits=[];exceptions=[];selection={};source_rows={}
    for split,count in (('train',6471),('val',548)):
        path=ARCHIVES/f'VisDrone2019-DET-{split}.zip'
        if sha256(path)!=audit['archives'][split]['sha256']:raise ValueError('Source ZIP changed.')
        with ZipFile(path) as z:
            images=sorted(n for n in z.namelist() if Path(n).parent.name=='images' and n.lower().endswith('.jpg') and '__MACOSX' not in n)
            labels={Path(n).stem:n for n in z.namelist() if Path(n).parent.name=='annotations' and n.endswith('.txt') and '__MACOSX' not in n}
            if len(images)!=count or len(labels)!=count:raise ValueError('Pool count differs.')
            eligible=[]
            for name in images:
                stem=Path(name).stem;image_bytes=z.read(name);label_bytes=z.read(labels[stem]);im=cv2.imdecode(np.frombuffer(image_bytes,np.uint8),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
                if im is None:raise ValueError('Image decode failed.')
                h,w=im.shape[:2];valid=[];excluded=0
                for line_num,line in enumerate(label_bytes.decode('utf-8-sig').splitlines(),1):
                    if not line.strip():continue
                    v=[float(x) for x in line.split(',') if x.strip()]
                    if len(v)<8 or not np.isfinite(v).all() or int(v[5])!=v[5] or not 0<=v[5]<=11:
                        raise ValueError('Unexplained malformed annotation.')
                    x,y,bw,bh,score,c=v[:6]
                    if bw<=0 or bh<=0:
                        excluded+=1;exceptions.append({'split':split,'image':stem,'line':line_num,'original':line,
                                  'reason':'NONPOSITIVE_AREA_EXCLUDED_IN_DERIVED_LABEL_VIEW','annotation_sha256':digest(label_bytes).hexdigest()});continue
                    if c in (0,11) or score==0:continue
                    x1,y1=max(0,x),max(0,y);x2,y2=min(w,x+bw),min(h,y+bh)
                    if x2<=x1 or y2<=y1:raise ValueError('Empty clipped native training box.')
                    valid.append(f'{int(c)-1} {(x1+x2)/2/w:.10f} {(y1+y2)/2/h:.10f} {(x2-x1)/w:.10f} {(y2-y1)/h:.10f}')
                row={'split':split,'stem':stem,'image_sha256':digest(image_bytes).hexdigest(),'annotation_sha256':digest(label_bytes).hexdigest(),
                     'width':w,'height':h,'native_boxes':len(valid),'excluded_zero_area':excluded,'duplicate_native_rows':len(valid)-len(set(valid))}
                audits.append(row);source_rows[split,stem]=(name,valid)
                if split=='train' or stem in old_ids:eligible.append(stem)
            chosen=sorted(random.Random(42).sample(sorted(eligible),64 if split=='train' else 16));selection[split]=chosen
            for stem in chosen:
                name,valid=source_rows[split,stem]
                image_dir=out/'dataset/images'/split;label_dir=out/'dataset/labels'/split
                image_dir.mkdir(parents=True,exist_ok=True);label_dir.mkdir(parents=True,exist_ok=True)
                (image_dir/(stem+'.jpg')).write_bytes(z.read(name))
                (label_dir/(stem+'.txt')).write_text('\n'.join(valid)+'\n',encoding='utf-8')
    csv_file(out/'full_label_audit.csv',audits)
    csv_file(out/'annotation_exceptions.csv',exceptions)
    save(out/'selection.json',selection)
    from main.labels import VIS_NAMES
    import yaml
    data={'path':str(out/'dataset'),'train':'images/train','val':'images/val','names':dict(enumerate(VIS_NAMES))}
    (out/'dataset.yaml').write_text(yaml.safe_dump(data,sort_keys=False),encoding='utf-8')
    proto={'version':'CV261207_MATCHED_CANARY_V2','scope':'CANARY_ENGINEERING_ONLY / DEVELOPMENT_ONLY',
        'initial_checkpoint':ck['checkpoint'],'source_original_protocol':source(BASE_PROTOCOL),
        'group':'clean_control','available_hooks':['clean_control','NoiseAug','MotionBlurAug','ExposureAug','JPEGAug','HazeAug'],
        'train':64,'development':16,'epochs':1,'imgsz':1280,'batch':2,'nbs':64,
        'optimizer':'AdamW','lr0':.0001,'warmup_epochs':0.,'warmup_bias_lr':.0001,'amp':False,'workers':0,'seed':42,
        'expected_network_image_exposures':64,'expected_batches':32,'expected_optimizer_updates':1,
        'budget_seconds':600,'training_source_version':'8.4.61',
        'training_label_policy':'Derived native10 view; exclude recorded nonpositive boxes, class0/11 and score0. Original ZIP unchanged.',
        'data_yaml':source(out/'dataset.yaml'),'selection':source(out/'selection.json'),
        'routine_augmentation':{k:ck['training_protocol'].get(k) for k in
            ('hsv_h','hsv_s','hsv_v','degrees','translate','scale','shear','perspective','flipud','fliplr','mosaic','mixup','copy_paste','close_mosaic')},
        'augmentation_order':'Optional pixel corruption after standard geometric/format pipeline; same geometry and isolated per-index/epoch RNG. CPU tested, target-arm GPU canaries not yet executed.',
        'data_files':[source(p) for p in sorted((out/'dataset').rglob('*')) if p.is_file()],
        'code':source(Path(__file__))}
    save(out/'protocol.json',proto)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':'cv261207','status':'PREPARED','mode':'MATCHED_CANARY_PREPARATION','full_train_images_audited':6471})
    seal(out);print(out,flush=True)


BASE_PROTOCOL=RUNS/'cv261207_rq_acceptance_20261107T220632_8271fd92/rq4_matched_protocol.json'


def healthy(module):
    import torch
    for name,value in module.state_dict().items():
        if value.is_floating_point() and (not torch.isfinite(value).all() or name.endswith('running_var') and (value<0).any()):
            raise ValueError('Unhealthy tensor: '+name)


def finite_state(value):
    import torch
    if isinstance(value,torch.Tensor) and value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError('Nonfinite optimizer state.')
    if isinstance(value,dict):
        for item in value.values():finite_state(item)
    elif isinstance(value,(list,tuple)):
        for item in value:finite_state(item)


def epoch_start(trainer):
    trainer._oom_retries=3
    trainer.train_loader.dataset.epoch=trainer.epoch


def worker(preparation,out):
    import torch
    from ultralytics import YOLO,__version__
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.utils.torch_utils import unwrap_model
    from ultralytics.data import build as data_build
    runtime(out);p=load(preparation/'protocol.json')
    if __version__!='8.4.61':raise ValueError('Trainer version changed.')
    for item in [p['initial_checkpoint'],p['data_yaml'],p['selection'],p['code'],*p['data_files']]:
        if sha256(Path(item['path']))!=item['sha256']:raise ValueError('Canary source changed.')
    import yaml
    shutil.copytree(preparation/'dataset',out/'dataset')
    data=yaml.safe_load((preparation/'dataset.yaml').read_text(encoding='utf-8'))
    data['path']=str(out/'dataset')
    data_file=out/'dataset.yaml';data_file.write_text(yaml.safe_dump(data,sort_keys=False),encoding='utf-8')
    if not torch.cuda.is_available() or torch.cuda.mem_get_info(0)[0]<3*1024**3:raise RuntimeError('CUDA admission failed.')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    start=time.perf_counter();events=[];counts={'batches':0,'images':0,'tensor_shapes':[]}
    class Wrapped:
        def __init__(self,base):self.base=base;self.epoch=0
        def __len__(self):return len(self.base)
        def __getattr__(self,name):return getattr(self.base,name)
        def __getitem__(self,index):
            item=self.base[index]
            if p['group']=='clean_control':return item
            im=item['img'].numpy().transpose(1,2,0)[:,:,::-1].copy()
            seed=int.from_bytes(digest(f"{p['seed']}|{self.epoch}|{index}".encode()).digest()[:8],'big')
            pixels,params=augment_pixels(im,p['group'],seed)
            with (out/'augmentation_samples.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps({'index':index,'epoch':self.epoch,'group':p['group'],'seed':seed,'parameters':params})+'\n')
            item['img']=torch.from_numpy(pixels[:,:,::-1].copy().transpose(2,0,1))
            return item
    class Trainer(DetectionTrainer):
        def preprocess_batch(self,batch):
            value=super().preprocess_batch(batch)
            shape=list(value['img'].shape)
            if shape[1:]!=[3,1280,1280] or not torch.isfinite(value['img']).all():raise ValueError('Training tensor differs.')
            counts['batches']+=1;counts['images']+=shape[0];counts['tensor_shapes'].append(shape)
            with (out/'training_batches.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps({'batch':counts['batches'],'NCHW':shape,'image_files':list(map(str,value['im_file']))})+'\n')
            return value
        def build_dataset(self,img_path,mode='train',batch=None):
            ds=super().build_dataset(img_path,mode,batch)
            return Wrapped(ds) if mode=='train' else ds
        def optimizer_step(self):
            grads=[v.grad for v in self.model.parameters() if v.grad is not None]
            if not grads or any(not torch.isfinite(v).all() for v in grads):raise ValueError('Nonfinite or absent gradients.')
            norm=float(torch.linalg.vector_norm(torch.stack([v.float().norm() for v in grads])))
            if not np.isfinite(norm):raise ValueError('Nonfinite gradient norm.')
            super().optimizer_step();healthy(self.model);healthy(self.ema.ema)
            finite_state(self.optimizer.state_dict())
            events.append({'update':len(events)+1,'EMA_updates':self.ema.updates,'gradient_norm':norm,
                           'lr':[g['lr'] for g in self.optimizer.param_groups]})
            save(out/'optimizer_events.json',events)
        def save_model(self):
            healthy(self.model);healthy(self.ema.ema)
            online=copy.deepcopy(unwrap_model(self.model)).float().cpu();ema=copy.deepcopy(self.ema.ema).float().cpu()
            temporary=Path(self.last).with_suffix('.pt.partial')
            if out.resolve() not in temporary.resolve().parents:raise ValueError('Checkpoint outside worker output.')
            if temporary.exists() or Path(self.last).exists():raise ValueError('Do not overwrite a checkpoint.')
            packet={'epoch':self.epoch,'model':online,'ema':ema,'updates':self.ema.updates,
                    'optimizer':copy.deepcopy(self.optimizer.state_dict()),'scaler':self.scaler.state_dict(),
                    'train_args':vars(self.args),'version':__version__,'best_fitness':self.best_fitness,
                    'train_metrics':{**self.metrics,'fitness':self.fitness},'train_results':self.read_results_csv(),
                    'date':datetime.now(timezone.utc).isoformat(),'source_SHA256':p['initial_checkpoint']['sha256']}
            torch.save(packet,temporary);reloaded=torch.load(temporary,map_location='cpu',weights_only=False)
            for key,model in (('model',online),('ema',ema)):
                healthy(reloaded[key])
                if any(not torch.equal(v,reloaded[key].state_dict()[k]) for k,v in model.state_dict().items()):
                    raise ValueError('FP32 save/reload differs.')
            temporary.rename(self.last)
            save(out/'checkpoint_health.json',{'status':'PASS','FP32':True,'all_model_and_EMA_tensors_finite':True,
                   'save_reload_exact':True,'path':str(self.last),'SHA256':sha256(self.last)})
            return True
        def final_eval(self):
            save(out/'finalization.json',{'default_half_strip':'DISABLED','metrics_scope':'ULTRALYTICS_BUILTIN_CANARY_ONLY; not pr01 ignore-union COCO'})
        def _handle_nan_recovery(self,epoch):
            if not torch.isfinite(self.loss).all() or self.fitness is not None and not np.isfinite(self.fitness):
                raise ValueError('Nonfinite loss or fitness; automatic recovery prohibited.')
            return False
    model=YOLO(p['initial_checkpoint']['path']);healthy(model.model)
    kwargs={k:p[k] for k in ('epochs','imgsz','batch','nbs','optimizer','lr0','warmup_epochs','warmup_bias_lr','amp','workers','seed')}
    kwargs.update({k:v for k,v in p['routine_augmentation'].items() if v is not None})
    kwargs.update(data=str(data_file),device='0',project=str(out),name='training',
                  plots=False,save=True,save_period=-1,cache=False,patience=100,resume=False,verbose=False)
    model.add_callback('on_train_epoch_start',epoch_start)
    model.add_callback('on_train_batch_end',lambda t: (_ for _ in ()).throw(ValueError('Nonfinite loss')) if not torch.isfinite(t.loss).all() else None)
    torch.cuda.reset_peak_memory_stats();model.train(trainer=Trainer,**kwargs)
    if len(events)!=1 or events[-1]['EMA_updates']!=1 or counts['images']!=64 or counts['batches']!=32:
        raise ValueError('Optimizer/EMA/exposure count differs.')
    checked_model=YOLO(str(out/'training/weights/last.pt'));healthy(checked_model.model)
    for item in [p['data_yaml'],p['selection'],*p['data_files']]:
        if sha256(Path(item['path']))!=item['sha256']:raise ValueError('Preparation evidence changed during training.')
    save(out/'result.json',{'status':'CANARY_PASS','scope':p['scope'],'group':p['group'],
        'network_image_exposures':counts['images'],'batches':counts['batches'],'optimizer_steps':len(events),'EMA_updates':events[-1]['EMA_updates'],
        'worker_seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'checkpoint':source(out/'training/weights/last.pt'),
        'accuracy_scope':'Engineering only, no gap closure or independent validation.',
        'full_training_estimate':'ESTIMATE_NOT_RELIABLE_FROM_ONE_64_IMAGE_EPOCH'})
    save(out/'run.json',{'RunID':out.name,'ExperimentID':'cv261207','status':'CANARY_PASS','mode':'RQ4_CLEAN_CANARY','preparation':preparation.name})
    seal(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','worker'));parser.add_argument('--run');parser.add_argument('--output')
    args=parser.parse_args()
    if args.mode=='prepare':prepare()
    else:
        prep=resolve_run(args.run);out=resolve_run(args.output)
        if not (out/'author_execution_ticket.json').exists() or (RUNS/'.gpu.lock').read_text(encoding='utf-8')!=out.name:
            raise ValueError('Use a single bounded authorized supervisor.')
        worker(prep,out)
