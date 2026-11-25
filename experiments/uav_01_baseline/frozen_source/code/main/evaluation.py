"""Fixed native-ten FP32 inference and genuine COCO evaluation."""
import os,time,traceback,datetime,uuid,re
from pathlib import Path
from .config import load_config,sha256,save

def evaluate(config_path,full=False):
 begin=time.perf_counter();cfg=load_config(config_path)
 print('[1/4] Loading dataset and checkpoint...',flush=True)
 prefix=cfg.get('run_prefix', 'git03_vm03')
 if not isinstance(prefix,str) or not re.fullmatch(r'[a-z][a-z0-9_]{0,63}',prefix):raise ValueError('Invalid configured run_prefix')
 out=Path(cfg['output_root'])/(prefix+'_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8]);out.mkdir(parents=True,exist_ok=False)
 for key in ['YOLO_CONFIG_DIR','MPLCONFIGDIR','TORCH_HOME','TMP','TEMP']:
  folder=out/'runtime'/key;folder.mkdir(parents=True);os.environ[key]=str(folder)
 (Path(os.environ['YOLO_CONFIG_DIR'])/'Ultralytics').mkdir()
 os.environ.update(YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',PYTHONDONTWRITEBYTECODE='1')
 save(out/'configuration.json',cfg)
 try:
  import numpy as np,torch
  from ultralytics import YOLO
  from .predictor import IEEEDetectionPredictor
  from .dataset import VisDroneZIP
  from .labels import VIS_NAMES,ground_truth,mapped_predictions
  from .statistics import evaluate_coco
  class TracedPredictor(IEEEDetectionPredictor):
   def preprocess(self,images):
    tensor=super().preprocess(images);self.input_trace=dict(NCHW=list(tensor.shape),dtype=str(tensor.dtype),device=str(tensor.device));return tensor
  device=cfg.get('device','0') if torch.cuda.is_available() else 'cpu'
  if device=='cpu' and not cfg.get('allow_cpu_fallback',True):raise RuntimeError('CUDA unavailable')
  archive=VisDroneZIP(cfg['dataset_zip'],cfg['expected_images']);n=len(archive.images) if full else cfg['smoke_images']
  model=YOLO(cfg['checkpoint'])
  if list(model.names.values())!=VIS_NAMES:raise ValueError('Native-ten class order mismatch')
  for name,tensor in model.model.state_dict().items():
   if tensor.is_floating_point() and (not torch.isfinite(tensor).all() or 'running_var' in name and (tensor<0).any()):raise ValueError('Unhealthy checkpoint:'+name)
  records=[];raw=[];trace=[];batch=cfg['batch'];network_seconds=0
  log_every=max(1,((n+batch-1)//batch)//10)
  print(f"[2/4] Inference: {n} images | input {cfg['imgsz']} | FP32 | device {device}",flush=True)
  for start in range(0,n,batch):
   loaded=[archive.record(i) for i in range(start,min(start+batch,n))];images=[x[0] for x in loaded]
   if device!='cpu':torch.cuda.synchronize()
   t=time.perf_counter()
   with torch.inference_mode():results=model.predict(source=images,predictor=TracedPredictor,imgsz=cfg['imgsz'],batch=batch,device=device,half=False,conf=.001,iou=.7,max_det=3000,end2end=False,rect=False,agnostic_nms=False,classes=None,augment=False,save=False,verbose=False)
   if device!='cpu':torch.cuda.synchronize()
   duration=time.perf_counter()-t;network_seconds+=duration
   actual=model.predictor.input_trace
   if actual['NCHW']!=[len(images),3,cfg['imgsz'],cfg['imgsz']] or actual['dtype']!='torch.float32':raise ValueError('Tensor protocol mismatch')
   if len(results)!=len(loaded):raise ValueError('Prediction/image mismatch')
   for (_,rec,prov),result in zip(loaded,results):
    records.append(rec);data=result.boxes.data.detach().cpu().numpy()
    if not np.isfinite(data).all():raise ValueError('Nonfinite prediction')
    for x1,y1,x2,y2,score,cls in data:
     if x2<=x1 or y2<=y1:continue
     if cls!=int(cls) or not 0<=cls<10 or not 0<=score<=1:raise ValueError('Prediction class/score invalid')
     raw.append(dict(image_id=rec['id'],class_id=int(cls),bbox=[float(x1),float(y1),float(x2-x1),float(y2-y1)],score=float(score)))
    trace.append(dict(**prov,**actual,batch_wall_seconds=duration))
   save(out/f'batch_{start:04d}.json',dict(records=trace[-len(loaded):],predictions=[p for p in raw if p['image_id'] in [l[1]['id'] for l in loaded]]))
   completed=min(start+batch,n)
   if start==0 or (start//batch+1)%log_every==0 or completed==n:
    print(f'  {completed}/{n} images ({completed/n:.0%}) | {len(raw)} predictions | {time.perf_counter()-begin:.1f}s elapsed',flush=True)
  print('[3/4] Computing COCO AP/AR on CPU...',flush=True)
  gt,regions=ground_truth(records,'native10');pred=mapped_predictions(raw,records,regions,'native10','finetuned',3000)
  save(out/'predictions_raw.json',raw);save(out/'predictions_coco.json',pred);save(out/'ground_truth.json',gt);save(out/'trace.json',trace)
  t=time.perf_counter();areas,classes,_=evaluate_coco(gt,pred,max_det=3000);evaluation_seconds=time.perf_counter()-t
  save(out/'metrics.json',dict(per_area=areas,per_class=classes))
  summary=dict(RunID=out.name,ExperimentID='vm03_gpu_pilot',Status='PASS',scope='FULL_DEVELOPMENT_VALIDATION' if full else 'SMOKE_ENGINEERING_ONLY',dataset='VisDrone2019-DET-val',images=n,checkpoint_id='ck05',checkpoint_SHA256=cfg['checkpoint_sha256'],protocol='pr01_native10_one2many_FP32_ignore_union_0.5_cap3000',device=device,network_seconds=network_seconds,evaluation_seconds=evaluation_seconds,total_seconds=time.perf_counter()-begin,raw_predictions=len(raw),predictions_SHA256=sha256(out/'predictions_raw.json'),GT_SHA256=sha256(out/'ground_truth.json'),metrics_SHA256=sha256(out/'metrics.json'),dataset_ZIP_SHA256=sha256(cfg['dataset_zip']),versions=dict(torch=torch.__version__,ultralytics=__import__('ultralytics').__version__,numpy=np.__version__),aggregate=next(a for a in areas if a['size']=='all'),full_validation_status='FULL_VALIDATION_PASS' if full and n==548 else 'FULL_VALIDATION_NOT_RUN')
  save(out/'summary.json',summary);archive.close();print('[4/4] Evaluation saved.',flush=True);print(str(out),flush=True);return out
 except BaseException as exc:
  save(out/'failure.json',dict(status='PARTIAL',error=repr(exc),traceback=traceback.format_exc()));raise

def regenerate(run_path):
 import json
 from .statistics import evaluate_coco
 path=Path(run_path);gt=json.loads((path/'ground_truth.json').read_text());pred=json.loads((path/'predictions_coco.json').read_text());areas,classes,_=evaluate_coco(gt,pred,3000)
 return dict(per_area=areas,per_class=classes)
