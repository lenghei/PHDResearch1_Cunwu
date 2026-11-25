"""Check a6471-image candidate label adapter without copying images or training."""
import argparse
from contextlib import contextmanager
import builtins
import gc
import hashlib
import inspect
import io
import itertools
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from unittest.mock import patch
from protocol import ROOT,HERE,EXPERIMENT,load,save,source,sha256,write_atomic
from common import seal,verify_hashes
from historical_review import table,csv_file
from main.pipeline import new_run
from post80_science import DATA
from post80_replay import run_path


def writable(path,root):
    if isinstance(path,int):return
    if not Path(path).resolve().is_relative_to(root.resolve()):raise PermissionError('Write outside the candidate run: '+str(path))


@contextmanager
def read_only_sources(root):
    original=builtins.open;original_io=io.open;original_os=os.open;original_unlink=Path.unlink
    def opened(path,mode='r',*args,**kwargs):
        if any(c in str(mode) for c in 'wax+'):writable(path,root)
        return original(path,mode,*args,**kwargs)
    def io_opened(path,mode='r',*args,**kwargs):
        if any(c in str(mode) for c in 'wax+'):writable(path,root)
        return original_io(path,mode,*args,**kwargs)
    def os_opened(path,flags,*args,**kwargs):
        if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):writable(path,root)
        return original_os(path,flags,*args,**kwargs)
    def unlinked(path,*args,**kwargs):
        writable(path,root);return original_unlink(path,*args,**kwargs)
    with patch('builtins.open',opened),patch('io.open',io_opened),patch('os.open',os_opened),patch.object(Path,'unlink',unlinked):
        yield


def cpu_check(out):
    os.environ.update(CUDA_VISIBLE_DEVICES='',YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',PYTHONDONTWRITEBYTECODE='1')
    for key in ('YOLO_CONFIG_DIR','MPLCONFIGDIR','TORCH_HOME'):
        p=out/'runtime'/key;p.mkdir(parents=True);os.environ[key]=str(p)
    import cv2
    import torch
    import yaml
    import ultralytics.data.dataset as ud
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.utils.torch_utils import init_seeds
    from main.labels import VIS_NAMES
    cv2.setNumThreads(2);torch.set_num_threads(2)
    before=run_path(DATA);verify_hashes(before,('TRAINING_IMAGE_LABEL_AUDIT.csv','DERIVED_LABEL_MANIFEST.json','ANNOTATION_EXCLUSIONS.csv'))
    records=table(before/'TRAINING_IMAGE_LABEL_AUDIT.csv');manifest=load(before/'DERIVED_LABEL_MANIFEST.json')
    if len(records)!=6471 or len(manifest)!=6471:raise ValueError('Training membership differs.')
    labels=out/'label_view';labels.mkdir();image_files=[];protected=[];jpeg=0
    for r in records:
        stem=r['image_id'];image=ROOT/'datasets/VisDrone/VisDrone2019-DET-train/images'/(stem+'.jpg')
        annotation=image.parent.parent/'annotations'/(stem+'.txt')
        for path,expected in ((image,r['image_SHA256']),(annotation,r['annotation_SHA256'])):
            if sha256(path)!=expected:raise ValueError('Original source differs: '+str(path))
            protected.append((path,expected))
        with image.open('rb') as stream:jpeg+=int(stream.read(3)==b'\xff\xd8\xff')
        label=before/'derived_labels_native10_v1'/(stem+'.txt')
        if sha256(label)!=manifest[label.name]:raise ValueError('Derived label changed.')
        shutil.copyfile(label,labels/label.name);image_files.append(str(image))
    listing=out/'image_list.txt';listing.write_text('\n'.join(image_files)+'\n',encoding='utf-8')
    args=yaml.safe_load((ROOT/'models/checkpoints/ck05_yolov8s_ft1280/args.yaml').read_text())
    fields=('imgsz','hsv_h','hsv_s','hsv_v','degrees','translate','scale','shear','perspective','flipud','fliplr',
        'bgr','mosaic','mixup','cutmix','copy_paste','copy_paste_mode')
    hyp=get_cfg(overrides={k:args[k] for k in fields if k in args});hyp.workers=0;hyp.cache=False;hyp.rect=False
    def one_pass():
        init_seeds(42,deterministic=True)
        with patch.object(ud,'img2label_paths',lambda files:[str(labels/(Path(f).stem+'.txt')) for f in files]),patch.object(ud,'NUM_THREADS',2):
            data=YOLODataset(img_path=str(listing),imgsz=1280,batch_size=2,augment=True,hyp=hyp,
                rect=False,cache=False,single_cls=False,stride=32,pad=0.,prefix='Post80 CPU: ',
                data={'names':dict(enumerate(VIS_NAMES)),'nc':10,'channels':3},task='detect')
        if len(data)!=6471:raise ValueError('The library dropped images.')
        targets=sum(len(r['bboxes']) for r in data.labels)
        if targets!=343200:raise ValueError('Library target count differs.')
        for r in data.labels:
            b=r['bboxes'];c=r['cls']
            if len(b) and (not __import__('numpy').isfinite(b).all() or (b[:,2:]<=0).any() or (c<0).any() or (c>=10).any()):
                raise ValueError('Invalid full-pool target.')
        data.npy_files=[out/'unused_npy'/(Path(p).stem+'.npy') for p in data.im_files]
        access=[];original=data.load_image
        def traced(index,*args,**kwargs):
            access.append(Path(data.im_files[index]).stem);return original(index,*args,**kwargs)
        data.load_image=traced
        init_seeds(42,deterministic=True)
        loader=torch.utils.data.DataLoader(data,batch_size=2,shuffle=False,num_workers=0,collate_fn=data.collate_fn,
            generator=torch.Generator().manual_seed(42))
        outputs=[]
        for batch in itertools.islice(loader,8):
            image=batch['img'];boxes=batch['bboxes'];classes=batch['cls']
            if tuple(image.shape)!=(2,3,1280,1280) or image.dtype!=torch.uint8:raise ValueError('Unexpected CPU batch.')
            if not torch.isfinite(boxes).all() or not torch.isfinite(classes).all():raise ValueError('Nonfinite transformed targets.')
            if len(boxes) and (bool((boxes<0).any()) or bool((boxes>1+1e-6).any()) or bool((boxes[:,2:]<=0).any())):
                raise ValueError('Transformed boxes are outside normalized geometry.')
            outputs.append({'anchor_ids':[Path(p).stem for p in batch['im_file']],'NCHW':list(image.shape),
                'targets':len(boxes),'pixels_SHA256':hashlib.sha256(image.numpy().tobytes()).hexdigest(),
                'targets_SHA256':hashlib.sha256(boxes.numpy().tobytes()+classes.numpy().tobytes()+batch['batch_idx'].numpy().tobytes()).hexdigest()})
        del loader,data;gc.collect();return {'outputs':outputs,'image_access_sequence':access,'targets':targets}
    with read_only_sources(out):
        first=one_pass();second=one_pass()
    if first!=second:raise ValueError('Candidate CPU loader repeat differs.')
    if not all(sha256(p)==h for p,h in protected):raise ValueError('A protected original changed.')
    result={'status':'CPU_FULL_LABEL_ADAPTER_VERIFIED_NOT_TRAINING_ADMITTED','full_library_images':6471,
        'native10_targets':343200,'source_JPEG_encoded_images':jpeg,'original_image_copies':0,'private_label_copies':6471,
        'zero_area_rows_excluded':3,'duplicate_positive_boxes_excluded':4,'anchors_per_repeat':16,'repeats':2,
        'seed':42,'workers':0,'batch':2,'imgsz':1280,'repeat_exact':True,'originals_unchanged':True,
        'new_model_forward_backward':0,'optimizer_steps':0,'GPU_experiments':0,
        'integration':'Versioned candidate label-path adapter; installed image verifier, transforms and collation unchanged.',
        'RNG':'Reseed after dataset construction; independent generator seed42, sequential anchors, fresh dataset each repeat.',
        'production_trainer':'NOT_INTEGRATED_NOT_TESTED; no efficacy or GPU determinism claim.',
        'current_augmentation':'Conventional archived args only; five target augmentation families not added.',
        'remaining':['Initialization source SHA not frozen','Training-only ranges and missing renderer choices not approved',
            'Production trainer, workers and matched optimizer/EMA/schedule unverified','No new finite training allocation or measured full-run cost'],
        'first_repeat':first,'source':source(Path(inspect.getsourcefile(YOLODataset)))}
    save(out/'LOADER_ACCEPTANCE.json',result,out)
    save(out/'SOURCE_PROTECTION.json',{'files':len(protected),'before_after_SHA_equal':True,
        'data_manifest':source(before/'TRAINING_IMAGE_LABEL_AUDIT.csv'),'derived_labels':source(before/'DERIVED_LABEL_MANIFEST.json')},out)


def execute():
    from development_guard import bounded_run
    out,_=new_run({'output_root':str(run_path(DATA).parent)},EXPERIMENT,'post80_full_loader_cpu')
    shutil.copyfile(Path(__file__),out/'executed_source.py');start=time.perf_counter()
    try:
        with (out/'worker.log').open('x',encoding='utf-8') as stream:
            code=bounded_run([sys.executable,'-B',str(Path(__file__)),'--worker',out.name],stream,600,
                dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',YOLO_AUTOINSTALL='false'))
        if code:raise RuntimeError('CPU loader worker failed; no retry.')
    except BaseException:
        save(out/'supervisor_failure.json',{'traceback':traceback.format_exc(),'automatic_retry':False},out)
    status=load(out/'LOADER_ACCEPTANCE.json')['status'] if (out/'LOADER_ACCEPTANCE.json').exists() else 'BLOCKED_CPU_LOADER'
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'POST80_FULL_LOADER_CPU','status':status,
        'CPU_wall_seconds':time.perf_counter()-start,'hard_limit_seconds':600,'GPU_tasks':0},out,replace=True)
    save(out/'inputs.json',{'source':source(Path(__file__)),'previous_data':source(run_path(DATA)/'SHA256SUMS.txt')},out)
    write_atomic(out/'diary.md','# Full-pool candidate label adapter\n\nNo source image copies or training. Verified original image/annotation bytes, copied the approved candidate labels into this run, and exercised full library label validation plus repeated CPU batches. Production training remains separate.\n',out)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>Full-pool CPU loader</h1><p>'+status+'</p><a href="LOADER_ACCEPTANCE.json">Actual acceptance</a>',out)
    seal(out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker');args=p.parse_args()
    if args.worker:
        try:cpu_check(run_path(args.worker))
        except BaseException:
            save(run_path(args.worker)/'failure.json',{'traceback':traceback.format_exc(),'automatic_retry':False},run_path(args.worker));raise
    else:execute()
