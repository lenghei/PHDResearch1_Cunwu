"""Saved-checkpoint inference with the frozen native10 research evaluator."""
import argparse
import gc
import gzip
import json
from pathlib import Path
import sys
import time
import traceback
from study import *


def infer(out,plan,checkpoint):
    import cv2,numpy as np,torch
    from ultralytics import YOLO
    from main.labels import VIS_NAMES,ground_truth,mapped_predictions
    from main.predictor_stable import StableNMSPredictor
    from main.statistics import evaluate_coco
    from main.training_health import model_health
    from evaluation import evaluate_saved
    refs=load(PREP/'SUPERVISOR_REQUIREMENTS.json')['frozen_source_hashes']
    if sha256(Path(refs['GT']['path']))!=refs['GT']['sha256']:raise ValueError('Frozen GT changed.')
    by_stem={r['stem']:r for r in load(Path(refs['GT']['path']))['records']}
    records=[by_stem[r['stem']] for r in plan['development']]
    model=YOLO(str(checkpoint));model_health(model.model)
    if list(model.names.values())!=VIS_NAMES:raise ValueError('Native10 class order differs.')
    traces=[];raw=[];observed=[];started=time.perf_counter()

    class Traced(StableNMSPredictor):
        def preprocess(self,images):
            value=super().preprocess(images)
            if list(value.shape)!=[1,3,1280,1280] or value.dtype!=torch.float32:raise ValueError('Research tensor changed.')
            self.trace={'NCHW':list(value.shape),'dtype':str(value.dtype)}
            return value

        def postprocess(self,preds,img,orig_imgs,**kwargs):
            value=preds[0] if isinstance(preds,(tuple,list)) else preds
            if self.model.end2end or value.ndim!=3 or value.shape[1]!=14:raise ValueError('Expected dense one-to-many native10 output.')
            if not torch.isfinite(value).all():raise FloatingPointError('Nonfinite dense predictions.')
            traces.append({'raw_shape':list(value.shape),'end2end':False,'dtype':str(value.dtype),
                'conf':self.args.conf,'iou':self.args.iou,'max_det':self.args.max_det,'rect':self.args.rect})
            return super().postprocess(preds,img,orig_imgs,**kwargs)

    with gzip.open(out/'predictions.jsonl.gz','xt',encoding='utf-8') as stream:
        for reference,record in zip(plan['development'],records):
            path=Path(reference['loader_path'])
            if sha256(path)!=reference['source_image_SHA256']:raise ValueError('Inference source SHA changed.')
            pixels=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            torch.cuda.synchronize();tick=time.perf_counter()
            with torch.inference_mode():
                result=model.predict(source=pixels,predictor=Traced,imgsz=1280,batch=1,device='0',quantize=None,nms=None,
                    conf=.001,iou=.7,max_det=3000,rect=False,agnostic_nms=False,classes=None,augment=False,save=False,verbose=False)[0]
            torch.cuda.synchronize();milliseconds=(time.perf_counter()-tick)*1000
            boxes=result.boxes.data.detach().cpu().numpy()
            if not np.isfinite(boxes).all() or result.orig_shape!=pixels.shape[:2]:raise ValueError('Prediction health/coordinates differ.')
            predictions=[{'image_id':record['id'],'class_id':int(c),'bbox':[float(x1),float(y1),float(x2-x1),float(y2-y1)],'score':float(score)}
                for x1,y1,x2,y2,score,c in boxes if x2>x1 and y2>y1]
            stream.write(json.dumps({'image_id':record['id'],'predictions':predictions})+'\n');raw.extend(predictions)
            observed.append({'image_id':record['id'],'tensor':model.predictor.trace,'predictions':len(predictions),'inference_ms':milliseconds})
    gt,regions=ground_truth(records,'native10');save(out/'ground_truth.json',{'coco':gt,'records':records},out)
    mapped=mapped_predictions(raw,records,regions,'native10','finetuned',3000)
    tick=time.perf_counter();areas,classes,_=evaluate_coco(gt,mapped,3000);evaluation_seconds=time.perf_counter()-tick
    metrics={'per_area':areas,'per_class':classes,'raw_count':len(raw),'mapped_count':len(mapped),'images':len(records)}
    save(out/'metrics.json',metrics,out)
    # Release the first COCO match graph before independent raw-file replay.
    del _,mapped,raw,gt,regions,model
    gc.collect();torch.cuda.empty_cache()
    tick=time.perf_counter()
    if evaluate_saved(out)[0]!=metrics:raise ValueError('Raw-prediction COCO replay differs.')
    save(out/'research_head_acceptance.json',{'status':'PASS','images':len(records),'checkpoint':source(checkpoint),
        'actual_outputs':traces,'actual_inputs':observed,'COCO_replay_equal':True,'COCO_seconds':evaluation_seconds,
        'replay_seconds':time.perf_counter()-tick,'wall_seconds':time.perf_counter()-started,
        'scope':'DEVELOPMENT_ENGINEERING_OR_COST_ONLY','bootstrap':'NOT_EVALUATED_ENGINEERING_ONLY'},out)


def worker(out,training_run):
    started=time.perf_counter();runtime(out);import torch
    from unittest.mock import patch
    from ultralytics.utils import callbacks
    from post80_loader import read_only_sources
    torch.set_num_threads(2);__import__('cv2').setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    status='FAILED'
    try:
        plan=load(training_run/'protocol.json');accepted=load(training_run/'training_acceptance.json');checkpoint=Path(accepted['checkpoint']['path'])
        if sha256(checkpoint)!=accepted['checkpoint']['sha256']:raise ValueError('Training snapshot changed.')
        with patch.object(callbacks,'add_integration_callbacks',lambda instance:None),read_only_sources(out):infer(out,plan,checkpoint)
        status='PASS'
    except BaseException:save(out/'failure.json',{'traceback':traceback.format_exc(),'automatic_retry':False},out)
    save(out/'worker_result.json',{'status':status,'worker_seconds':time.perf_counter()-started,'training_RunID':training_run.name,
        'new_training':False,'model':load(training_run/'protocol.json')['architecture']},out)


def recover(training_id):
    from development_guard import bounded_run
    training_run=run_path(training_id)
    verify_hashes(training_run,('training_acceptance.json','protocol.json'))
    plan=load(training_run/'protocol.json');is_canary=plan['mode']=='canary'
    out=create('canary_inference_recovery' if is_canary else 'full_epoch_inference_recovery')
    save(out/'inputs.json',{'training':source(training_run/'SHA256SUMS.txt'),'source':source(Path(__file__)),
        'predictor_adapter':source(PUBLIC/'code/main/predictor_stable.py')},out)
    cap=budget_start(out,600 if is_canary else 1800,'ADMISSION' if is_canary else 'FULL_DATA_COST');tick=time.perf_counter();error=None
    try:
        with (out/'worker.log').open('x',encoding='utf-8') as stream:
            bounded_run([sys.executable,'-B','-X','utf8',str(Path(__file__)),'--worker',out.name,'--training-run',training_id],stream,cap,
                dict(os.environ,CUDA_VISIBLE_DEVICES='0',YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',OMP_NUM_THREADS='2'))
    except BaseException:error=traceback.format_exc()
    result=load(out/'worker_result.json') if (out/'worker_result.json').exists() else {'status':'INTERRUPTED'}
    duration=time.perf_counter()-tick;budget_finish(out,duration,result['status'])
    save(out/'supervisor.json',{'seconds':duration,'error':error,'hard_cap':cap},out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'INFERENCE_RECOVERY',**result},out,replace=True)
    if result['status']=='PASS' and is_canary:
        save(RUNS/'admission_result.json',{'RunID':out.name,'training_RunID':training_id,'architecture':plan['architecture'],'status':'PASS'},RUNS,replace=True)
    write_atomic(out/'diary.md','# Canary inference recovery\n\nReused the healthy engineering checkpoint; no repeated training. The new API adapter retains the original NMS and COCO calculation.\n',out)
    seal(out);print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--recover');parser.add_argument('--worker');parser.add_argument('--training-run')
    args=parser.parse_args()
    if args.recover:recover(args.recover)
    elif args.worker:worker(run_path(args.worker),run_path(args.training_run))
    else:parser.error('Choose saved-checkpoint recovery or an admitted worker.')
