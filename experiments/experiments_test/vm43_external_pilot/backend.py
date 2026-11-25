"""Original-ZIP, cached FP32 prediction and unchanged native10 COCO evaluation."""
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path,PurePosixPath
import shutil
import time
from zipfile import ZipFile

from vm43_common import ROOT,HERE,PUBLIC,load,save,sha,source,module,output


def parse_annotation(data,record):
    objects=[];ignores=[];others=[];width,height=record['width'],record['height']
    for line in data.decode('utf-8-sig').splitlines():
        if not line.strip():continue
        v=[float(x) for x in line.split(',') if x.strip()]
        if len(v)<8 or not all(math.isfinite(x) for x in v):raise ValueError('Malformed annotation.')
        x,y,w,h,score,cat=v[:6]
        if w<=0 or h<=0 or cat!=int(cat) or not 0<=cat<=11:raise ValueError('Invalid annotation.')
        x1,y1=max(0,x),max(0,y);x2,y2=min(width,x+w),min(height,y+h)
        if x2<=x1 or y2<=y1:raise ValueError('Empty clipped annotation.')
        box=[x1,y1,x2-x1,y2-y1]
        if cat==0 or score==0:ignores.append(box)
        elif cat==11:others.append(box)
        else:objects.append({'class_id':int(cat)-1,'bbox':box})
    return {**record,'objects':objects,'ignore_regions':ignores,'other_regions':others}


class OriginalZIP:
    def __init__(self,population,certificate,verify_archive=True):
        self.members=population['images'];self.path=Path(certificate['path'])
        if verify_archive and sha(self.path)!=certificate['sha256']:raise ValueError('ZIP SHA differs.')
        self.archive=ZipFile(self.path);names=self.archive.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP entry.')
        self.images={PurePosixPath(n).stem:n for n in names if '/images/' in '/'+n.replace('\\','/') and n.lower().endswith('.jpg')}
        self.labels={PurePosixPath(n).stem:n for n in names if '/annotations/' in '/'+n.replace('\\','/') and n.endswith('.txt')}
        stems={r['image_id'] for r in self.members}
        if len(stems)!=len(self.members) or stems!=set(self.images) or stems!=set(self.labels):raise ValueError('Image/GT/frozen membership differs.')
    def record(self,i,decode=False):
        expected=self.members[i];stem=expected['image_id'];label=self.archive.read(self.labels[stem])
        if hashlib.sha256(label).hexdigest()!=expected['annotation_SHA256']:raise ValueError('Annotation SHA differs.')
        rec=parse_annotation(label,{'id':i+1,'stem':stem,'file_name':PurePosixPath(self.images[stem]).name,
            'width':expected['width'],'height':expected['height']})
        image=None
        if decode:
            import cv2,numpy as np
            raw=self.archive.read(self.images[stem])
            if hashlib.sha256(raw).hexdigest()!=expected['source_image_SHA256']:raise ValueError('Source image SHA differs.')
            image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            if image is None or image.shape!=(rec['height'],rec['width'],3) or image.dtype!=np.uint8:raise ValueError('Image decode shape/dtype differs.')
        return image,rec
    def close(self):self.archive.close()


def renderer():
    import sys
    sys.path.insert(0,str(HERE.parent/'cv261109'))
    return module('vm43_frozen_generator',HERE.parent/'cv261109/generator.py')


def prepare_gt(out,spec):
    from main.labels import ground_truth,VIS_NAMES
    from main.statistics import _validated_records
    population=load(spec['population']['path'])
    if population['native10_class_order']!=VIS_NAMES:raise ValueError('Native10 class order differs.')
    archive=OriginalZIP(population,population['source_ZIP_accepted_certificate'])
    try:records=[archive.record(i)[1] for i in range(len(population['images']))]
    finally:archive.close()
    gt,regions=ground_truth(records,'native10');_validated_records(gt,[])
    payload={'records':records,'coco':gt,'scope':'GT_INPUT_ONLY_NOT_AP_RESULTS','population_SHA256':sha(spec['population']['path'])}
    save(out/'ground_truth_input.json',payload)
    return payload


def validate_predictions(rows,image_ids):
    seen=[]
    for row in rows:
        iid=row['image_id'];seen.append(iid)
        if iid not in image_ids:raise ValueError('Unknown image ID.')
        for p in row['predictions']:
            if p['image_id']!=iid or isinstance(p['class_id'],bool) or not isinstance(p['class_id'],int) or not 0<=p['class_id']<10:raise ValueError('Invalid class/image mapping.')
            if len(p['bbox'])!=4 or not all(math.isfinite(float(v)) for v in [*p['bbox'],p['score']]):raise ValueError('Nonfinite prediction.')
            if p['bbox'][2]<=0 or p['bbox'][3]<=0 or not 0<=p['score']<=1:raise ValueError('Invalid prediction.')
    if len(seen)!=len(image_ids) or set(seen)!=set(image_ids):raise ValueError('Truncated/duplicate predictions.')


def install_accumulator():
    import sys
    sys.modules['new_fast_coco_accumulate']=module('new_fast_coco_accumulate',HERE.parent/'cv261207/development_fast_coco.py')


def evaluate_saved(path,records,samples,seed):
    from main.labels import ground_truth,mapped_predictions
    from main.statistics import evaluate_coco,bootstrap_coco
    with gzip.open(path,'rt',encoding='utf-8') as f:rows=[json.loads(line) for line in f]
    validate_predictions(rows,{r['id'] for r in records})
    gt,regions=ground_truth(records,'native10');raw=[p for r in rows for p in r['predictions']]
    mapped=mapped_predictions(raw,records,regions,'native10','finetuned',3000)
    start=time.perf_counter();areas,classes,evaluator=evaluate_coco(gt,mapped,3000);coco_seconds=time.perf_counter()-start
    install_accumulator();start=time.perf_counter();intervals,draws=bootstrap_coco(evaluator,samples=samples,seed=seed)
    metrics={'per_area':areas,'per_class':classes,'raw_count':len(raw),'mapped_count':len(mapped),'images':len(records)}
    return metrics,intervals,draws,{'COCO_seconds':coco_seconds,'bootstrap_seconds':time.perf_counter()-start}


def cache_pixels(out,spec):
    import cv2
    population=load(spec['population']['path']);archive=OriginalZIP(population,population['source_ZIP_accepted_certificate']);ops=renderer()
    cache=out/'pixel_cache';cache.mkdir();cases={}
    required=sum(r['width']*r['height']*3*2 for r in population['images'])+8*1024**3
    if shutil.disk_usage(out).free<required:raise RuntimeError('Insufficient space for full two-condition PNG cache and outputs.')
    conditions={r['condition']:r['P_attributes'] for r in spec['cells']}
    try:
        for name,attrs in conditions.items():
            dest=cache/name;dest.mkdir();entries=[]
            for i,expected in enumerate(population['images']):
                image,record=archive.record(i,decode=True)
                pixels,trace=ops.generate(image,list(attrs),attrs,record['stem'],spec['condition_rng']['master_seed'])
                repeated,again=ops.generate(image,list(attrs),attrs,record['stem'],spec['condition_rng']['master_seed'])
                if pixels.tobytes()!=repeated.tobytes() or trace!=again:raise ValueError('Generator is not deterministic.')
                ok,data=cv2.imencode('.png',pixels)
                if not ok:raise RuntimeError('PNG encoding failed.')
                file=dest/(record['stem']+'.png');file.write_bytes(data.tobytes())
                entries.append({'image_id':record['id'],'stem':record['stem'],'path':str(file),'file_SHA256':sha(file),
                    'pixel_SHA256':hashlib.sha256(pixels.tobytes()).hexdigest(),'source_image_SHA256':expected['source_image_SHA256'],'effects':trace})
            cases[name]=entries
    finally:archive.close()
    save(out/'pixel_cache_manifest.json',cases);return cases


def timed_prediction(entry,adapter,predictions,manifest):
    import cv2,numpy as np
    start=time.perf_counter();phase={};path=Path(entry['path'])
    if sha(path)!=entry['file_SHA256']:raise ValueError('Cached file integrity failed.')
    phase['cached_file_verification_ms']=(time.perf_counter()-start)*1000
    t=time.perf_counter();data=path.read_bytes();phase['image_read_ms']=(time.perf_counter()-t)*1000
    t=time.perf_counter();image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR);phase['decode_ms']=(time.perf_counter()-t)*1000
    if image is None:raise ValueError('Cache decode failed.')
    t=time.perf_counter()
    if hashlib.sha256(image.tobytes()).hexdigest()!=entry['pixel_SHA256']:raise ValueError('Cached pixel integrity failed.')
    phase['pixel_verification_ms']=(time.perf_counter()-t)*1000
    t=time.perf_counter();adapter.synchronize();phase['pre_predict_sync_ms']=(time.perf_counter()-t)*1000
    t=time.perf_counter();result=adapter.predict(image);s=time.perf_counter();adapter.synchronize();end=time.perf_counter()
    phase['inference_only_ms']=(end-t)*1000;phase['post_predict_sync_ms']=(end-s)*1000
    phase.update(adapter.profiles(result))
    t=time.perf_counter();raw,tensor=adapter.extract(result,entry['image_id']);phase['prediction_extraction_ms']=(time.perf_counter()-t)*1000
    validate_predictions([{'image_id':entry['image_id'],'predictions':raw}],{entry['image_id']})
    t=time.perf_counter();predictions.write(json.dumps({'image_id':entry['image_id'],'predictions':raw})+'\n');predictions.flush()
    manifest.write(json.dumps({**entry,'tensor':tensor})+'\n');manifest.flush()
    phase['output_and_manifest_write_flush_ms']=(time.perf_counter()-t)*1000
    phase['cached_pipeline_ms']=(time.perf_counter()-start)*1000
    return {'image_id':entry['image_id'],**phase,'tensor':tensor,'prediction_count':len(raw)}


class RealDetector:
    def __init__(self,checkpoint):
        import torch
        from ultralytics import YOLO,__version__
        from ultralytics.utils import callbacks
        from unittest.mock import patch
        from main.predictor_stable import StableNMSPredictor
        from main.training_health import model_health
        from main.labels import VIS_NAMES
        if __version__!='8.4.210' or not torch.cuda.is_available():raise RuntimeError('Frozen CUDA/Ultralytics runtime unavailable.')
        self.torch=torch
        with patch.object(callbacks,'add_integration_callbacks',lambda instance:None):self.model=YOLO(checkpoint)
        model_health(self.model.model)
        if list(self.model.names.values())!=VIS_NAMES:raise ValueError('Model native10 order differs.')
        class Traced(StableNMSPredictor):
            def preprocess(self,images):
                tensor=super().preprocess(images)
                if list(tensor.shape)!=[1,3,1280,1280] or tensor.dtype!=torch.float32:raise ValueError('Actual network NCHW/dtype differs.')
                self.input_tensor={'NCHW':list(tensor.shape),'dtype':str(tensor.dtype)};return tensor
            def postprocess(self,preds,img,orig_imgs,**kwargs):
                raw=preds[0] if isinstance(preds,(list,tuple)) else preds
                if self.model.end2end or raw.ndim!=3 or raw.shape[1]!=14 or not torch.isfinite(raw).all():raise ValueError('Unhealthy or incompatible dense output.')
                return super().postprocess(preds,img,orig_imgs,**kwargs)
        self.predictor=Traced
    def synchronize(self):self.torch.cuda.synchronize()
    def predict(self,image):
        self.last_shape=image.shape[:2]
        with self.torch.inference_mode():
            return self.model.predict(image,predictor=self.predictor,imgsz=1280,batch=1,device='0',
                quantize=None,nms=None,conf=.001,iou=.7,max_det=3000,rect=False,agnostic_nms=False,
                classes=None,augment=False,save=False,verbose=False)[0]
    def profiles(self,result):
        names={'preprocess':'preprocessing_profile_ms','inference':'forward_profile_ms','postprocess':'NMS_profile_ms'}
        values={v:float(result.speed[k]) for k,v in names.items()}
        if not all(math.isfinite(v) and v>=0 for v in values.values()):raise ValueError('Invalid stage profiler value.')
        return values
    def extract(self,result,image_id):
        if result.orig_shape!=self.last_shape:raise ValueError('Original-coordinate shape differs.')
        boxes=result.boxes.data.detach().cpu().numpy()
        if not self.torch.isfinite(self.torch.as_tensor(boxes)).all():raise ValueError('Nonfinite output boxes.')
        rows=[]
        for x1,y1,x2,y2,score,cat in boxes:
            if cat!=int(cat):raise ValueError('Nonintegral model class.')
            if x2>x1 and y2>y1:rows.append({'image_id':image_id,'class_id':int(cat),'bbox':[float(x1),float(y1),float(x2-x1),float(y2-y1)],'score':float(score)})
        return rows,self.model.predictor.input_tensor


def gap_state(quality,latency):
    if quality[0] is None or quality[1] is None:return 'NOT_ESTIMABLE'
    if latency[1]>100 or quality[1]<.10:return 'FAIL'
    return 'PASS' if quality[0]>=.10 else 'UNCERTAIN'


def compare_cells(cells,records,seed,draws):
    import numpy as np
    required={(a,c) for a in ('CONTROL','VARIABILITY_GUIDED') for c in ('Normal','Gaussian_s15')}
    if set(cells)!=required:raise ValueError('Four-cell comparison incomplete.')
    reference=next(iter(cells.values()))['draws']
    for value in cells.values():
        if any(value['draws'][k]!=reference[k] for k in ('image_ids','samples','seed','sampling')):raise ValueError('Bootstrap pairing differs.')
    rows=[]
    for condition in ('Normal','Gaussian_s15'):
        c,g=cells['CONTROL',condition],cells['VARIABILITY_GUIDED',condition]
        for area in ('all','tiny','small','medium','large'):
            for metric in ('map50','map50_95','ar50_95'):
                point=lambda v:next(x[metric] for x in v['metrics']['per_area'] if x['size']==area)
                left=c['draws']['values'][area][metric];right=g['draws']['values'][area][metric]
                valid=point(c) is not None and point(g) is not None and all(x is not None for x in left+right)
                delta=np.asarray(right)-np.asarray(left) if valid else None
                rows.append({'condition':condition,'size':area,'metric':metric,'difference':point(g)-point(c) if valid else None,
                    'ci95':np.percentile(delta,[2.5,97.5]).tolist() if valid else [None,None],
                    'status':'ESTIMATED_ONE_SEED_POINTWISE' if valid else 'NOT_ESTIMABLE'})
    nc,c,ng,g=[cells[k] for k in [('CONTROL','Normal'),('CONTROL','Gaussian_s15'),('VARIABILITY_GUIDED','Normal'),('VARIABILITY_GUIDED','Gaussian_s15')]]
    point=lambda v:next(x['map50_95'] for x in v['metrics']['per_area'] if x['size']=='all')
    x=(np.asarray(ng['draws']['values']['all']['map50_95'])-np.asarray(g['draws']['values']['all']['map50_95']))-(np.asarray(nc['draws']['values']['all']['map50_95'])-np.asarray(c['draws']['values']['all']['map50_95']))
    rows.append({'condition':'Gaussian_degradation_loss','size':'all','metric':'map50_95','difference':(point(ng)-point(g))-(point(nc)-point(c)),
        'ci95':np.percentile(x,[2.5,97.5]).tolist(),'status':'ESTIMATED_ONE_SEED_POINTWISE'})
    return rows


def csv_file(path,rows):
    if not rows:raise ValueError('No measured rows.')
    with output(path).open('x',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));writer.writeheader();writer.writerows(rows)


def worker(out,spec,execution_scope,gt_path):
    import cv2,numpy as np,torch
    torch.set_num_threads(2);cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.perf_counter();payload=load(gt_path);records=payload['records']
    cache=cache_pixels(out,spec);adapters={};results={};timing_rows=[];metrics_rows=[];gap_rows=[];interval_rows=[]
    rng=np.random.default_rng(20261116);indices=rng.integers(0,len(records),(200,len(records)))
    for cell in spec['cells']:
        if sha(cell['checkpoint']['path'])!=cell['checkpoint']['SHA256']:raise ValueError('Checkpoint changed.')
        key=cell['strategy']
        if key not in adapters:adapters[key]=RealDetector(cell['checkpoint']['path'])
        adapter=adapters[key];entries=cache[cell['condition']]
        sample=cv2.imread(entries[0]['path'])
        for _ in range(3):adapter.predict(sample)
        adapter.synchronize();torch.cuda.reset_peak_memory_stats()
        folder=out/cell['cell_id'];folder.mkdir();times=[]
        raw=folder/'predictions.jsonl.gz';partial=folder/'predictions.jsonl.gz.partial'
        with gzip.open(partial,'xt',encoding='utf-8') as stream,(folder/'image_manifest.jsonl').open('x',encoding='utf-8') as manifest:
            for entry in entries:times.append(timed_prediction(entry,adapter,stream,manifest))
        with gzip.open(partial,'rt',encoding='utf-8') as stream:validate_predictions([json.loads(line) for line in stream],{r['id'] for r in records})
        partial.rename(raw)
        metrics,intervals,bootstrap,cost=evaluate_saved(raw,records,200,20261116)
        save(folder/'metrics.json',metrics);save(folder/'bootstrap.json',{'intervals':intervals,'draws':bootstrap})
        save(folder/'timing.json',{'per_image':times,'evaluation_cost':cost,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'scope':execution_scope,'profiling':'Ultralytics library stage profiles are diagnostics, not additive reconstructed pipeline latency'})
        save(folder/'acceptance.json',{'status':'REAL_CELL_COMPLETED','images':1610,'scope':execution_scope,
            'checkpoint':cell['checkpoint'],'raw_predictions':source(raw),'GT':source(gt_path),
            'metrics':source(folder/'metrics.json'),'bootstrap':source(folder/'bootstrap.json')})
        results[key,cell['condition']]={'metrics':metrics,'draws':bootstrap}
        latency=np.asarray([r['cached_pipeline_ms'] for r in times]);lat_ci=np.percentile(latency[indices].mean(axis=1),[2.5,97.5]).tolist()
        tiny=next(r['ci95'] for r in intervals if r['size']=='tiny' and r['metric']=='map50_95')
        gap_rows.append({'cell_id':cell['cell_id'],'Tiny_CI':tiny,'latency_mean_ms':float(latency.mean()),'latency_CI_ms':lat_ci,
            'state':gap_state(tiny,lat_ci),'scope':execution_scope})
        for r in metrics['per_area']:metrics_rows.append({'cell_id':cell['cell_id'],'strategy':key,'condition':cell['condition'],'scope':execution_scope,**r})
        for r in times:timing_rows.append({'cell_id':cell['cell_id'],**r})
        for r in intervals:interval_rows.append({'cell_id':cell['cell_id'],**r})
    comparisons=compare_cells(results,records,20261116,200)
    for condition in ('Normal','Gaussian_s15'):
        left=[r for r in timing_rows if r['cell_id']==next(c['cell_id'] for c in spec['cells'] if c['strategy']=='CONTROL' and c['condition']==condition)]
        right=[r for r in timing_rows if r['cell_id']==next(c['cell_id'] for c in spec['cells'] if c['strategy']=='VARIABILITY_GUIDED' and c['condition']==condition)]
        delta=np.array([r['cached_pipeline_ms'] for r in right])-np.array([r['cached_pipeline_ms'] for r in left])
        comparisons.append({'condition':condition,'metric':'cached_pipeline_ms','difference':float(delta.mean()),
            'ci95':np.percentile(delta[indices].mean(axis=1),[2.5,97.5]).tolist(),'status':'ESTIMATED_ONE_SEED_POINTWISE'})
    csv_file(out/'configuration_metrics.csv',metrics_rows);csv_file(out/'paired_comparisons.csv',comparisons)
    csv_file(out/'bootstrap_intervals.csv',interval_rows);csv_file(out/'timing_by_image.csv',timing_rows);csv_file(out/'gap_status.csv',gap_rows)
    save(out/'worker_result.json',{'status':'REAL_FOUR_CELLS_COMPLETED','cells':4,'images_per_cell':1610,'seed':42,
        'scope':execution_scope,'wall_seconds':time.perf_counter()-started,'policy_updated':False})
