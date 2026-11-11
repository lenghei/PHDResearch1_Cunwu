"""Audit RQ4 inputs and exercise the native-effect dataset on CPU."""
import argparse
import copy
from collections import Counter
from contextlib import redirect_stdout
import gc
import hashlib
import inspect
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from zipfile import ZipFile
from protocol import ROOT,HERE,EXPERIMENT,FMP,FMD,PRIOR,load,save,source,sha256,write_atomic
from common import seal,verify_hashes
from historical_review import table,csv_file
from main.pipeline import new_run
from post80_replay import run_path
from post80_science import DATA

NORMAL='vm36_supervisor_reproduction_normal4_analysis_20261110T234523_1ce40f03'
HARD_CPU_SECONDS=1800


def optimizer_exposure_plan(images,epochs,batch,nbs,warmup_epochs):
    import numpy as np
    if min(images,epochs,batch,nbs)<1 or warmup_epochs<0:raise ValueError('Invalid training-count proposal.')
    batches=math.ceil(images/batch);warmup=max(round(warmup_epochs*batches),100) if warmup_epochs>0 else -1
    target=max(round(nbs/batch),1);last=-1;per_epoch=[];updates=0
    for epoch in range(epochs):
        count=0
        for i in range(batches):
            ni=i+batches*epoch
            accumulate=max(1,int(np.interp(ni,[0,warmup],[1,nbs/batch]).round())) if ni<=warmup else target
            if ni-last>=accumulate:last=ni;count+=1
        updates+=count
        per_epoch.append({'epoch':epoch+1,'image_exposures':images,'batches':batches,'last_batch_images':images-(batches-1)*batch,
            'planned_optimizer_and_EMA_updates':count})
    return {'scope':'SOURCE_DERIVED_PLAN_NOT_EXECUTED','images':images,'epochs':epochs,'batch':batch,'nbs':nbs,
        'image_exposures':images*epochs,'forward_backward_batches':batches*epochs,'optimizer_and_EMA_updates':updates,
        'warmup_batches':warmup,'steady_accumulation':target,'epoch_end_gradient_flush':False,'epochs_detail':per_epoch}


def runtime(out):
    os.environ.update(CUDA_VISIBLE_DEVICES='',YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',PYTHONDONTWRITEBYTECODE='1')
    for name in ('YOLO_CONFIG_DIR','MPLCONFIGDIR','TORCH_HOME'):
        path=out/'runtime'/name;path.mkdir(parents=True,exist_ok=True);os.environ[name]=str(path)
    import torch,cv2
    torch.set_num_threads(2);cv2.setNumThreads(2)


def requirements_and_sources(out):
    import importlib.metadata
    verify_hashes(run_path(NORMAL),('analysis_result.json','METRIC_PROVENANCE.json','CONTINUATION_GATES.json'))
    if load(run_path(NORMAL)/'analysis_result.json')['combined_pairs']!=84:raise ValueError('The accepted84 source differs.')
    proof=load(PRIOR/'pdf_provenance.json');pages=load(PRIOR/'pdf_pages.json')['pages']
    for name in ('original','canonical'):
        if sha256(Path(proof[name]['path']))!=proof[name]['sha256']:raise ValueError('Supervisor PDF source changed.')
    reference=load(run_path(NORMAL)/'METRIC_PROVENANCE.json')['new_worker']
    if sha256(Path(reference['path']))!=reference['sha256']:raise ValueError('The accepted Normal worker manifest changed.')
    parent=Path(reference['path']).parent
    verify_hashes(parent,('EXECUTION_PROTOCOL.json',))
    plan=load(parent/'EXECUTION_PROTOCOL.json')
    for name in ('FMP','FMD','generator','GT','image_members','ZIP'):
        if sha256(Path(plan[name]['path']))!=plan[name]['sha256']:raise ValueError('Accepted evaluation input changed: '+name)
    save(out/'SUPERVISOR_REQUIREMENTS.json',{'PDF_provenance':proof,'extracted_pages_source':source(PRIOR/'pdf_pages.json'),
        'pages':{'11':pages[10],'12':pages[11]},'primary_models':['YOLOv8s','YOLO26s'],'train_resolution':1280,
        'minimum_primary_seeds':3,'required_augmentations':['NoiseAug','MotionBlurAug','ExposureAug','JPEGAug','HazeAug'],
        'frozen_source_hashes':{name:plan[name] for name in ('FMP','FMD','generator','GT','image_members','ZIP')},
        'test_level_reuse':'Training ranges contain test levels, while exact held-out numeric settings are excluded.',
        'independent_testdev':'BLOCKED; val remains development/checkpoint-selection data.'},out)
    import ultralytics.engine.trainer as tr
    import ultralytics.data.base as base
    import ultralytics.data.dataset as dataset
    from ultralytics.data.augment import v8_transforms,Albumentations
    parts={'training_loop':tr.BaseTrainer._do_train,'setup':tr.BaseTrainer._setup_train,
        'optimizer_step':tr.BaseTrainer.optimizer_step,'save_model':tr.BaseTrainer.save_model,
        'final_eval':tr.BaseTrainer.final_eval,'load_image':base.BaseDataset.load_image,
        'get_image_and_label':base.BaseDataset.get_image_and_label,'get_labels':dataset.YOLODataset.get_labels,
        'transforms':v8_transforms,'optional_Albumentations':Albumentations}
    text={name:inspect.getsource(value) for name,value in parts.items()}
    for name,body in text.items():write_atomic(out/('source_'+name+'.txt'),body,out)
    save(out/'TRAINER_SOURCE_AUDIT.json',{'installed_versions':{name:importlib.metadata.version(name) for name in ('torch','ultralytics','numpy','opencv-python')},
        'sources':{name:source(Path(inspect.getsourcefile(value))) for name,value in parts.items()},
        'historical_checkpoint_library':'8.4.33','installed_library':importlib.metadata.version('ultralytics'),
        'version_identity_to_historical':'DIFFERENT; fresh matched arms require an explicitly new protocol.',
        'initial_accumulation':'max(round(nbs/batch),1)=32 at batch2/nbs64; warmup changes it per batch.',
        'warmup':'max(round(warmup_epochs*train_batches),100) when warmup_epochs>0; original args specify3.',
        'epoch_boundary':'last_opt_step initialized once before the epoch loop; no unconditional epoch-end gradient flush.',
        'save_model':'Installed default converts EMA and optimizer states to FP16. It skips a nonfinite EMA save.',
        'final_eval':'Installed final_eval calls strip_optimizer; historical best train_results may describe the final curve.',
        'automatic_recovery_tokens':{name:name in text['training_loop'] for name in ('OutOfMemoryError','_handle_nan_recovery')},
        'required_before_GPU':['New finite training authorization','Approved source/recipe/selection/worker protocol',
            'Numerical health and explicit save/reload checks','No claim of exact interrupted-training resume'],
        'old_84_inference_repeated':False},out)


def data_audit(out):
    import cv2,numpy as np
    from training_data_readiness import native_rows
    old=run_path(DATA);verify_hashes(old,('TRAINING_IMAGE_LABEL_AUDIT.csv','DERIVED_LABEL_MANIFEST.json','ANNOTATION_EXCLUSIONS.csv'))
    records=table(old/'TRAINING_IMAGE_LABEL_AUDIT.csv');labels=load(old/'DERIVED_LABEL_MANIFEST.json')
    if len(records)!=6471 or len(labels)!=6471:raise ValueError('Training membership differs.')
    checked=[];exceptions=[];duplicate_rows=[];class_counts=np.zeros(10,np.int64);lookup={};originals=[]
    image_root=ROOT/'datasets/VisDrone/VisDrone2019-DET-train/images'
    for i,record in enumerate(records):
        stem=record['image_id'];image=image_root/(stem+'.jpg');ann=image_root.parent/'annotations'/(stem+'.txt')
        for path,digest in ((image,record['image_SHA256']),(ann,record['annotation_SHA256'])):
            if sha256(path)!=digest:raise ValueError('Protected input differs: '+str(path))
            originals.append((path,digest))
        pixels=cv2.imdecode(np.frombuffer(image.read_bytes(),np.uint8),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        if pixels is None or pixels.dtype!=np.uint8:raise ValueError('Training decode failed.')
        h,w=pixels.shape[:2];values,invalid,crossing,duplicates=native_rows(ann.read_text(encoding='utf-8-sig'),w,h)
        label=old/'derived_labels_native10_v1'/(stem+'.txt')
        if sha256(label)!=labels[label.name]:raise ValueError('Existing derived label changed.')
        originals.append((label,labels[label.name]))
        actual=np.loadtxt(label,dtype=np.float32,ndmin=2) if len(values) else np.zeros((0,5),np.float32)
        if not np.array_equal(values,actual):raise ValueError('Derived coordinates differ from the original native10 conversion.')
        if crossing:raise ValueError('An unreviewed boundary-crossing box appeared.')
        if len(values):class_counts+=np.bincount(values[:,0].astype(int),minlength=10)
        lookup[stem]=str(label)
        for row in invalid:exceptions.append({'id':stem,**row})
        if duplicates:duplicate_rows.append({'id':stem,'duplicate_boxes':duplicates,'annotation_SHA256':record['annotation_SHA256']})
        checked.append({'image_id':stem,'image_path':str(image),'image_SHA256':record['image_SHA256'],
            'annotation_SHA256':record['annotation_SHA256'],'derived_label_SHA256':labels[label.name],
            'width':w,'height':h,'native10_targets':len(values)})
        if (i+1)%1000==0:print('TRAIN_DATA_VERIFIED',i+1,'/6471',flush=True)
    if len(exceptions)!=3 or sum(r['duplicate_boxes'] for r in duplicate_rows)!=4 or int(class_counts.sum())!=343200:
        raise ValueError('Known exceptions or target population changed.')
    csv_file(out/'TRAINING_DATA_VERIFICATION.csv',checked);csv_file(out/'ANNOTATION_EXCEPTIONS.csv',exceptions)
    csv_file(out/'DUPLICATE_BOX_CASES.csv',duplicate_rows)
    save(out/'PRIVATE_LABEL_LOOKUP.json',lookup,out)
    write_atomic(out/'training_images.txt','\n'.join(r['image_path'] for r in checked)+'\n',out)
    save(out/'TRAINING_DATA_ACCEPTANCE.json',{'status':'FULL6471_SOURCE_AND_DERIVED_LABELS_VERIFIED','images':6471,
        'positive_targets':int(class_counts.sum()),'class_counts_zero_based':class_counts.tolist(),
        'zero_area_rows':exceptions,'duplicates_removed':duplicate_rows,'image_copies':0,'source_labels_modified':False,
        'derived_view':source(old/'DERIVED_LABEL_MANIFEST.json'),'ignore_policy':
        'Only positive native10 targets enter standard detection loss. Ignore/other pixels are not loss-masked; this matches the conventional loader semantics and must be shared by all arms. Research evaluation retains original ignore-union filtering.',
        'coordinate_rule':'VisDrone IDs1..10 become0..9; xywh pixels become normalized center xywh in original decoded coordinates.',
        'training_data_change_vs_legacy':'One positive zero-height row is excluded; four duplicates follow library deduplication. Fresh matched controls are required.'},out)
    return checked,lookup,originals


def state_health(model):
    import torch
    bad=[name for name,value in model.state_dict().items() if value.is_floating_point() and
        (not bool(torch.isfinite(value).all()) or name.endswith('running_var') and bool((value<0).any()))]
    if bad:raise ValueError('Unhealthy tensors: '+str(bad))
    return {'parameters':sum(p.numel() for p in model.parameters()),'all_tensors_finite':True,'BN_variances_nonnegative':True,
        'floating_dtypes':sorted({str(v.dtype) for v in model.state_dict().values() if v.is_floating_point()})}


def validation_loader(out):
    import torch,numpy as np
    from types import SimpleNamespace
    from ultralytics.cfg import get_cfg
    from main.labels import VIS_NAMES
    from archives import OriginalZIP
    from training_data_readiness import native_rows
    from rq4_dataset import trainer_class
    from rq4_augmentation import engineering_fixture
    from post80_loader import read_only_sources
    worker=Path(load(run_path(NORMAL)/'METRIC_PROVENANCE.json')['new_worker']['path']).parent
    plan=load(worker/'EXECUTION_PROTOCOL.json');members=load(Path(plan['image_members']['path']))
    gt=load(Path(plan['GT']['path']));archive=OriginalZIP(Path(plan['ZIP']['path']),548)
    labels=out/'validation_labels';labels.mkdir();images=[];lookup={};proof=[];protected=[];exceptions=[];copies=0
    try:
        for i in range(548):
            pixels,record,member=archive.record(i)
            if member!=members[i] or record!=gt['records'][i]:raise ValueError('Validation source differs from the accepted84 population.')
            stem=record['stem'];local=ROOT/'datasets/VisDrone/VisDrone2019-DET-val/images'/(stem+'.jpg')
            if local.exists():protected.append((local,sha256(local)))
            if not local.exists() or sha256(local)!=member['image_SHA256']:
                local=out/'validation_original_byte_view'/(stem+'.jpg');local.parent.mkdir(exist_ok=True)
                local.write_bytes(archive.archive.read(archive.images[i]));copies+=1
                if sha256(local)!=member['image_SHA256']:raise ValueError('Private original-byte view differs.')
            annotation=archive.archive.read(archive.labels[stem]).decode('utf-8-sig')
            values,invalid,crossing,duplicates=native_rows(annotation,pixels.shape[1],pixels.shape[0])
            if crossing:raise ValueError('Validation box crosses native geometry.')
            label=labels/(stem+'.txt');label.write_text(''.join(' '.join(format(float(v),'.9g') for v in row)+'\n' for row in values),encoding='utf-8')
            images.append(str(local));lookup[stem]=str(label)
            proof.append({'image_id':record['id'],'stem':stem,'loader_path':str(local),'source_image_SHA256':member['image_SHA256'],
                'source_annotation_SHA256':member['annotation_SHA256'],'derived_label_SHA256':sha256(label),'native_shape':list(pixels.shape),
                'valid_internal_targets':len(values),'duplicates_removed':duplicates,'invalid_boxes_removed':len(invalid)})
            exceptions.extend({'stem':stem,**row} for row in invalid)
    finally:archive.close()
    listing=out/'validation_images.txt';write_atomic(listing,'\n'.join(images)+'\n',out)
    ctx={'labels':lookup,'evidence':str(out/'validation_dataset_check'),'recipe':engineering_fixture(()),'seed':42,'engineering':True}
    cls=trainer_class(ctx);trainer=object.__new__(cls);trainer.args=get_cfg(overrides={'imgsz':1280,'workers':0,'cache':False})
    trainer.data={'names':dict(enumerate(VIS_NAMES)),'nc':10,'channels':3};trainer.model=SimpleNamespace(stride=torch.tensor([32]))
    with read_only_sources(out):dataset=trainer.build_dataset(str(listing),mode='val',batch=2)
    if len(dataset)!=548:raise ValueError('The validation loader dropped images.')
    observed=[]
    for index in range(0,16,2):
        with read_only_sources(out):batch=dataset.collate_fn([dataset[index],dataset[index+1]])
        observed.append({'ids':[Path(p).stem for p in batch['im_file']],'NCHW':list(batch['img'].shape),
            'targets':len(batch['bboxes']),'labels_finite':bool(torch.isfinite(batch['bboxes']).all())})
    save(out/'VALIDATION_LOADER_ACCEPTANCE.json',{'status':'CPU_FULL548_VALIDATION_LOADER_VERIFIED','images':548,
        'private_original_byte_image_copies':copies,'population':proof,'exceptions':exceptions,'engineering_batches':observed,
        'added_target_augmentation':False,'actual_optimizer_steps':0,'actual_GPU_AP_evaluations':0,
        'metric_boundary':'Ultralytics internal validation is rect-mode and lacks research ignore-union filtering. It must not be labelled the frozen square-input research COCO evaluation.',
        'checkpoint_selection':'Still requires a common predeclared rule across arms.'},out)
    del dataset,trainer;gc.collect();return protected


def initialization_audit(out):
    import torch,yaml
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import init_seeds
    from main.labels import VIS_NAMES
    metadata=[];candidate=[];protected=[];initial=out/'candidate_initializations';initial.mkdir()
    for ck,folder,name in (('ck05','ck05_yolov8s_ft1280','yolov8s.pt'),('ck01','ck01_yolo26s_ft1280','yolo26s.pt')):
        where=ROOT/'models/checkpoints'/folder;args=yaml.safe_load((where/'args.yaml').read_text());history=[]
        for which in ('best','last'):
            path=where/(which+'.pt');protected.append((path,sha256(path)))
            data=torch.load(path,map_location='cpu',weights_only=False);model=data.get('ema') or data['model']
            if list(model.names.values())!=VIS_NAMES:raise ValueError('Historical native10 class order differs.')
            if which=='best':
                expected=next(r for r in load(run_path(NORMAL)/'METRIC_PROVENANCE.json')['pairs'] if '__'+ck+'_' in r['pair_id'])
                cfg=load(Path(expected['source_folder'])/'configuration.json')
                if sha256(path)!=cfg['d']['sha256']:raise ValueError('Accepted historical checkpoint changed.')
            history.append({'kind':which,'source':source(path),'direct_epoch':data.get('epoch'),'date':data.get('date'),
                'version':data.get('version'),'best_fitness':data.get('best_fitness'),'health':state_health(model),
                'state':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()}})
            del model,data;gc.collect()
        same=all(torch.equal(history[0]['state'][key],value) for key,value in history[1]['state'].items())
        metadata.append({'checkpoint':ck,'saved_args':source(where/'args.yaml'),'configured_epochs':args['epochs'],
            'records':[{k:v for k,v in record.items() if k!='state'} for record in history],
            'best_and_last_tensor_state_equal':same,'selected_historical_best_epoch':'UNKNOWN; stripped epoch=-1 is not reconstructed from curve length.',
            'historical_initialization_SHA256':'UNKNOWN; a present same-name pretrained file is not creation-time evidence.'})
        del history;gc.collect()
        path=ROOT/'models/checkpoints'/name;digest=sha256(path);protected.append((path,digest))
        data=torch.load(path,map_location='cpu',weights_only=False);pretrained=data.get('ema') or data['model']
        health=state_health(pretrained)
        if len(pretrained.names)!=80 or int(pretrained.model[-1].nc)!=80:raise ValueError('Expected a COCO80 source.')
        if pretrained.yaml.get('scale')!='s' and (pretrained.yaml.get('depth_multiple'),pretrained.yaml.get('width_multiple'))!=(.33,.5):
            raise ValueError('The initialization is not a small model.')
        if ck=='ck05' and not any(type(module).__name__=='C2f' for module in pretrained.modules()):raise ValueError('Expected YOLOv8 backbone.')
        if ck=='ck01' and not getattr(pretrained.model[-1],'end2end',False):raise ValueError('Expected YOLO26 dual-head source.')
        for seed in (42,43,44):
            init_seeds(seed,deterministic=True)
            with redirect_stdout(io.StringIO()):
                model=DetectionModel(cfg=copy.deepcopy(pretrained.yaml),ch=3,nc=10,verbose=False)
                model.load(pretrained,verbose=False)
            model=model.cpu().float();model.names=dict(enumerate(VIS_NAMES));state_health(model)
            if model.model[-1].nc!=10:raise ValueError('Native10 head mismatch.')
            target=initial/(ck+'_seed'+str(seed)+'_native10_fp32.pt');temporary=target.with_suffix('.partial')
            torch.save({'model':model,'epoch':-1,'origin_SHA256':digest,'seed':seed,'purpose':'CANDIDATE_INITIALIZATION_ONLY',
                'names':VIS_NAMES,'not_historical_training_reproduction':True},temporary)
            restored=torch.load(temporary,map_location='cpu',weights_only=False)['model'];state_health(restored)
            if state_health(restored)['floating_dtypes']!=['torch.float32']:raise ValueError('The candidate save is not FP32.')
            if not all(torch.equal(v,restored.state_dict()[k]) for k,v in model.state_dict().items()):raise ValueError('FP32 reload differs.')
            if target.exists():raise FileExistsError('Refusing to overwrite a candidate initialization.')
            temporary.rename(target)
            candidate.append({'architecture':'YOLOv8s' if ck=='ck05' else 'YOLO26s','seed':seed,'source':source(path),
                'source_COCO_health':health,'native10_candidate':source(target),'class_order':VIS_NAMES,
                'native10_health':state_health(restored),'FP32_reload_state_equal':True,'training_executed':False})
            del model,restored;gc.collect()
        del pretrained,data;gc.collect()
    save(out/'HISTORICAL_CHECKPOINT_AUDIT.json',metadata,out)
    save(out/'CANDIDATE_INITIALIZATION_MANIFEST.json',candidate,out)
    return protected


def loader_validation(out,checked,lookup):
    import torch,numpy as np,yaml
    from types import SimpleNamespace
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.torch_utils import init_seeds
    from main.labels import VIS_NAMES
    from rq4_dataset import trainer_class
    from rq4_augmentation import FAMILIES,engineering_fixture
    from post80_loader import read_only_sources
    args=yaml.safe_load((ROOT/'models/checkpoints/ck05_yolov8s_ft1280/args.yaml').read_text())
    fields=('imgsz','hsv_h','hsv_s','hsv_v','degrees','translate','scale','shear','perspective','flipud','fliplr',
        'bgr','mosaic','mixup','cutmix','copy_paste','copy_paste_mode','close_mosaic')
    result=[];full_counts=None;control_targets={}
    cases=[('CONTROL',()),('ALL_FIVE',FAMILIES)]+[(family,(family,)) for family in FAMILIES]+[('CONTROL_NO_MOSAIC',()),('ALL_FIVE_NO_MOSAIC',FAMILIES)]
    for label,active in cases:
        repeats=[]
        for repeat in range(2):
            folder=out/'loader_checks'/(label+'_'+str(repeat));folder.mkdir(parents=True)
            context={'labels':lookup,'evidence':str(folder),'recipe':engineering_fixture(active),'seed':42,'engineering':True,
                'label_cache':str(out/'verified_full_pool.cache')}
            cls=trainer_class(context);trainer=object.__new__(cls)
            trainer.args=get_cfg(overrides={k:args[k] for k in fields if k in args});trainer.args.workers=0;trainer.args.cache=False
            if label.endswith('NO_MOSAIC'):trainer.args.mosaic=0.
            trainer.model=SimpleNamespace(stride=torch.tensor([32]));trainer.data={'names':dict(enumerate(VIS_NAMES)),'nc':10,'channels':3}
            init_seeds(42,deterministic=True)
            with read_only_sources(out):dataset=trainer.build_dataset(str(out/'training_images.txt'),mode='train',batch=2)
            counts=(len(dataset),sum(len(r['cls']) for r in dataset.labels))
            if counts!=(6471,343200):raise ValueError('Production dataset-hook population differs.')
            full_counts=counts;init_seeds(42,deterministic=True)
            batches=[];started=time.perf_counter()
            for index in range(0,16,2):
                with read_only_sources(out):batch=dataset.collate_fn([dataset[index],dataset[index+1]])
                image=batch['img'];boxes=batch['bboxes'];classes=batch['cls']
                if list(image.shape)!=[2,3,1280,1280] or image.dtype!=torch.uint8:raise ValueError('CPU training batch differs.')
                if not torch.isfinite(boxes).all() or not torch.isfinite(classes).all() or len(boxes) and ((boxes[:,2:]<=0).any() or (boxes<0).any() or (boxes>1+1e-6).any() or (classes<0).any() or (classes>=10).any()):
                    raise ValueError('Geometric transforms produced invalid targets.')
                batches.append({'anchors':[Path(p).stem for p in batch['im_file']],'NCHW':list(image.shape),'targets':len(boxes),
                    'pixel_SHA256':hashlib.sha256(image.numpy().tobytes()).hexdigest(),
                    'target_SHA256':hashlib.sha256(boxes.numpy().tobytes()+classes.numpy().tobytes()+batch['batch_idx'].numpy().tobytes()).hexdigest()})
            events=dataset.effect_events
            if active and not all(len(e['effects'])==len(active) for e in events):raise ValueError('An effect was disconnected.')
            for event in events:
                expected=next(r for r in checked if r['image_id']==event['id'])
                if event['native_shape'][:2]!=[expected['height'],expected['width']]:raise ValueError('Effects were applied after an early resize.')
            save(folder/'BATCHES.json',batches,out);save(folder/'EFFECT_EVENTS.json',events,out)
            save(folder/'RECIPE.json',context['recipe'],out)
            repeats.append({'batches':batches,'events':events,'seconds':time.perf_counter()-started})
            if label.startswith('CONTROL') and repeat==0:
                from unittest.mock import patch
                from ultralytics.data.dataset import YOLODataset
                with read_only_sources(out),patch.object(YOLODataset,'get_labels',return_value=copy.deepcopy(dataset.labels)):
                    stock=YOLODataset(img_path=str(out/'training_images.txt'),imgsz=1280,batch_size=2,augment=True,
                        hyp=copy.deepcopy(trainer.args),rect=False,cache=False,single_cls=False,stride=32,pad=0.,data=trainer.data,task='detect')
                stock.npy_files=[folder/'unused_stock_npy'/(Path(p).stem+'.npy') for p in stock.im_files]
                init_seeds(42,deterministic=True);stock_batches=[]
                for index in range(0,16,2):
                    with read_only_sources(out):batch=stock.collate_fn([stock[index],stock[index+1]])
                    stock_batches.append({'anchors':[Path(p).stem for p in batch['im_file']],
                        'NCHW':list(batch['img'].shape),'targets':len(batch['bboxes']),
                        'pixel_SHA256':hashlib.sha256(batch['img'].numpy().tobytes()).hexdigest(),
                        'target_SHA256':hashlib.sha256(batch['bboxes'].numpy().tobytes()+batch['cls'].numpy().tobytes()+batch['batch_idx'].numpy().tobytes()).hexdigest()})
                if stock_batches!=batches:raise ValueError('The no-op native adapter differs from installed conventional transforms.')
                save(out/(label+'_STOCK_PARITY.json'),{'exact_batch_and_target_equality':True,'anchors':16,
                    'test_boundary':'Frozen prevalidated labels injected only for the stock comparison; installed image loader and transforms unchanged.',
                    'batches':stock_batches},out);del stock;gc.collect()
            del dataset,trainer;gc.collect()
        equal=all(repeats[0][key]==repeats[1][key] for key in ('batches','events'))
        if not equal:raise ValueError('CPU repeat differs for '+label)
        targets=[(row['anchors'],row['target_SHA256']) for row in repeats[0]['batches']]
        phase='NO_MOSAIC' if label.endswith('NO_MOSAIC') else 'MOSAIC'
        if label.startswith('CONTROL'):control_targets[phase]=targets
        elif targets!=control_targets[phase]:raise ValueError('Matched arms changed geometric targets or anchor order.')
        result.append({'arm':label,'scope':'ENGINEERING_FIXTURE_ONLY','full_dataset_images':full_counts[0],
            'full_dataset_targets':full_counts[1],'anchors_per_repeat':16,'repeats':2,'repeat_exact':equal,
            'native_image_accesses_per_repeat':len(repeats[0]['events']),'timed_transform_seconds':[r['seconds'] for r in repeats],
            'matched_control_geometric_targets_equal':True,
            'actual_optimizer_steps':0,'actual_EMA_updates':0})
        print('TRAINER_DATASET_CPU_PASS',label,flush=True)
    save(out/'PRODUCTION_DATASET_CPU_ACCEPTANCE.json',{'status':'CPU_DATASET_HOOK_VERIFIED_NOT_GPU_TRAINING',
        'checks':result,'source_images_copied':0,'workers':0,'formal_archived_workers':2,
        'worker_limitation':'Target-effect epoch/prefetch handling with workers2 is not admitted; changing formal workers requires an explicit versioned protocol.',
        'activation_status':'All technical fixtures use p=1 to test their connection. No scientific training probabilities have been selected.',
        'ignore_loss_masking':False,'conventional_geometry':'Installed mosaic, RandomPerspective, HSV, flip and Format follow the native effects.'},out)


def plans(out):
    from rq4_augmentation import FAMILIES,TEST_VALUES
    import yaml
    inits=load(out/'CANDIDATE_INITIALIZATION_MANIFEST.json') if (out/'CANDIDATE_INITIALIZATION_MANIFEST.json').exists() else []
    matrix=[]
    for architecture in ('YOLOv8s','YOLO26s'):
        for seed in (42,43,44):
            arms=['DEFAULT_CONTROL','ALL_FIVE']+(['WITHOUT_'+f for f in FAMILIES] if architecture=='YOLOv8s' else [])
            initial=next((r['native10_candidate'] for r in inits if r['architecture']==architecture and r['seed']==seed),None)
            for arm in arms:
                active=[] if arm=='DEFAULT_CONTROL' else [f for f in FAMILIES if arm!='WITHOUT_'+f]
                matrix.append({'architecture':architecture,'seed_candidate':seed,'arm':arm,'added_families':'|'.join(active),
                    'priority':'PRIMARY_THREE_SEED' if arm in ('DEFAULT_CONTROL','ALL_FIVE') else 'MINIMUM_SINGLE_SEED_ABLATION' if seed==42 else 'EXTENDED_ABLATION_REPEAT',
                    'initialization_SHA256':initial['sha256'] if initial else 'NOT_VERIFIED','epochs_candidate':100,'imgsz':1280,
                    'batch':2,'nbs':64,'optimizer':'AdamW','lr0':.01,'lrf':.01,'cos_lr':True,'warmup_epochs':3.,
                    'AMP_historical':True,'workers_historical':2,'train_images':6471,'validation_images':548,
                    'checkpoint_selection':'PENDING: historical best-on-val or fixed epoch100 EMA; use the same rule for every matched arm.',
                    'status':'NOT_RUN','scientific_approval':'REQUIRED','resource_budget':'NOT_AUTHORIZED'})
    csv_file(out/'MATCHED_RQ4_MATRIX_PROPOSAL.csv',matrix)
    save(out/'OPTIMIZER_EXPOSURE_PLAN.json',optimizer_exposure_plan(6471,100,2,64,3.),out)
    fields=('epochs','imgsz','batch','nbs','optimizer','lr0','lrf','momentum','weight_decay','cos_lr','warmup_epochs',
        'warmup_momentum','warmup_bias_lr','patience','amp','workers','deterministic','hsv_h','hsv_s','hsv_v','degrees',
        'translate','scale','shear','perspective','flipud','fliplr','bgr','mosaic','mixup','cutmix','copy_paste','close_mosaic')
    settings={}
    for name in ('ck05_yolov8s_ft1280','ck01_yolo26s_ft1280'):
        path=ROOT/'models/checkpoints'/name/'args.yaml';args=yaml.safe_load(path.read_text())
        settings[name]={'source':source(path),'recorded_values':{k:args[k] for k in fields}}
    save(out/'TRAINING_SETTINGS_PROPOSAL.json',{'status':'PROPOSED_NOT_AUTHORIZED','architectures':settings,
        'matching':'Use an identical admitted setting and initialization per architecture/seed across control, targeted and ablation arms.',
        'historical_settings_equal_except_model':settings['ck05_yolov8s_ft1280']['recorded_values']==settings['ck01_yolo26s_ft1280']['recorded_values'],
        'current_canary_cost':'NOT_MEASURED','checkpoint_save_policy':'FP32 source/reload health must be integrated before GPU; installed default half-save is not admitted.'},out)
    original=load(run_path(NORMAL)/'METRIC_PROVENANCE.json')['pairs'];evaluation=[]
    for row in original:
        cfg=load(Path(row['source_folder'])/'configuration.json')
        evaluation.append({'pair_id':row['pair_id'],'P_id':cfg['p']['id'],'leaves':cfg['p']['leaves'],
            'attributes':cfg['p']['attributes'],'checkpoint':cfg['d']['checkpoint'],'inference_resolution':cfg['d']['resolution'],
            'source_folder':row['source_folder'],'source_SHA256':row['SHA256']})
    save(out/'FROZEN_DEVELOPMENT_EVALUATION.json',{'status':'EXISTING84_REUSE_NO_NEW_INFERENCE','images_per_condition':548,
        'conditions':evaluation,'sampling':'Existing20 purposeful compound contexts plus Normal; no selection on new training AP.',
        'metrics':['AP50','AP50:95','AR','tiny','small','medium','large','native10 per-class'],
        'pair_CI':'Existing200 paired image resamples; training-seed variability reported separately, not pooled as independent images.',
        'gap':'Recount the same tested detector subset, same thresholds and timing boundary; not full FM_D.',
        'future_checkpoint_identity':'Register new checkpoint IDs and SHA per architecture/seed/arm. Never relabel an old ck01/ck05 response as a new training outcome.',
        'matched_baseline':'Newly trained control and targeted arms must share the same initialization, derived labels and current library. Historical84 responses are context, not a substitute for new matched controls.',
        'independent_testdev':'DATASET_BLOCKED'},out)
    alternatives=[
        {'decision':'Initialization','options':['Fresh matched COCO-derived control/target arms from the verified per-seed candidates','Continue matched arms from the fixed historical FT checkpoints'],
         'difference':'The first tests full retraining; the second tests additional adaptation and cannot be called the original100-epoch training comparison.'},
        {'decision':'Ranges','options':['Use the closed hull of each PDF test family, excluding its exact discrete test values','Choose broader documented ranges before training'],
         'test_values':TEST_VALUES,'difference':'The PDF defines test levels, not approved training bounds or activation probabilities. No production recipe is generated.'},
        {'decision':'Motion/Haze bindings','options':['Explicitly approve the existing renderer orientation interval and constant-depth airlight proxy','Provide a separately justified calibrated binding'],
         'difference':'The existing helper needs angle_min/angle_max and airlight; the PDF does not specify these. Technical fixture values are not a recommendation.'},
        {'decision':'Ablation','options':['Leave one family out from ALL_FIVE','Add one family at a time to DEFAULT_CONTROL'],
         'difference':'The current27-run proposal uses the former; the original text does not uniquely resolve the choice.'},
        {'decision':'Checkpoint selection','options':['Best checkpoint selected on validation using the same metric and tie rule','Fixed epoch100 EMA for every arm'],
         'difference':'The former follows historical args; the latter was an earlier proposal, not verified as the supervisor-required selection rule.'},
        {'decision':'Workers','options':['Validate workers2 epoch/prefetch and RNG handling before using the archived value','Admit workers0 for every arm as a new protocol'],
         'difference':'Only workers0 has been exercised in this engineering run; no silent substitution is allowed.'}]
    save(out/'SCIENTIFIC_DECISIONS_REQUIRED.json',{'status':'UNRESOLVED','decisions':alternatives,
        'augmentation_order':list(FAMILIES),'scientific_probabilities':'NOT_SELECTED','scientific_ranges':'NOT_SELECTED'},out)
    if (out/'TRAINING_DATA_VERIFICATION.csv').exists() and (out/'VALIDATION_LOADER_ACCEPTANCE.json').exists():
        import numpy as np
        train=table(out/'TRAINING_DATA_VERIFICATION.csv');val=load(out/'VALIDATION_LOADER_ACCEPTANCE.json')['population']
        rng=np.random.default_rng(20261111);a=sorted(rng.choice(len(train),64,replace=False));b=sorted(rng.choice(len(val),16,replace=False))
        save(out/'CANARY_IMAGE_CANDIDATES.json',{'status':'FROZEN_IDS_NOT_RUN','selection_seed':20261111,'training_seed_candidate':42,
            'selection':'Uniform without replacement, no AP access. Same IDs proposed for every admitted canary arm.',
            'train':[train[i] for i in a],'development':[val[i] for i in b]},out)
    save(out/'GPU_CANARY_PLAN.json',{'status':'NOT_AUTHORIZED_NOT_RUN','suggested_hard_seconds':600,
        'required_new_record':['Author task','Finite GPU wall limit','Model/initialization SHA','Approved recipe/protocol hash','Single-use ticket'],
        'engineering_design':{'architecture':'YOLOv8s first; YOLO26s separately after its own admission','legal_training_images':64,
            'development_images':16,'epochs':1,'imgsz':1280,'batch':2,'optimizer':'AdamW','nbs':64,
            'schedule':'Explicitly preserve approved warmup/accumulation; count actual updates, do not assume64/64.',
            'selection':'Deterministic IDs without reading AP, frozen before execution.'},
        'acceptance':['Finite forward/backward/loss/gradients','Actual optimizer and EMA counts','Full model and EMA/BN health',
            'FP32 temporary save, reload and equality before commit','Measured validation, data, training and save time','Allocated/reserved GPU peaks'],
        'full_training_cost':'NOT_MEASURED; do not multiply one mini epoch by100. Measure loading, warmup/steady batches, full validation and saving separately.',
        'prior_inference_budgets_reusable':False,'GPU_started':False},out)


def worker(out):
    runtime(out);began=time.perf_counter();failures=[];protected=[];data=None
    for name,operation in [('sources',lambda:requirements_and_sources(out)),('data',lambda:data_audit(out)),
            ('initialization',lambda:initialization_audit(out)),('validation_loader',lambda:validation_loader(out))]:
        try:
            result=operation()
            if name=='data':data=result;protected.extend(result[2])
            elif name in ('initialization','validation_loader'):protected.extend(result)
        except Exception:
            failures.append(name);save(out/(name+'_failure.json'),{'traceback':traceback.format_exc()},out)
    if data:
        try:loader_validation(out,data[0],data[1])
        except Exception:failures.append('loader');save(out/'loader_failure.json',{'traceback':traceback.format_exc()},out)
    plans(out)
    if not all(sha256(path)==digest for path,digest in protected):raise ValueError('A protected original changed.')
    save(out/'SOURCE_PROTECTION.json',{'checked_files':len(protected),'before_after_SHA_equal':True,'accepted84_rerun':False},out)
    save(out/'TRAINING_ADMISSION.json',{'status':'CPU_PREPARED_TRAINING_BLOCKED' if not failures else 'PARTIAL_WITH_LIMITATIONS',
        'failed_CPU_branches':failures,'training_runs':0,'GPU_tasks':0,'real_GPU_training_cost':'NOT_MEASURED',
        'blockers':['Scientific range/probability/orientation/airlight choices','Initialization strategy and selection/ablation rules',
            'Worker/prefetch protocol','FP32 production training save/health integration and GPU canary','New finite training authorization'],
        'historical_best_epoch':'UNKNOWN','historical_initialization_SHA256':'UNKNOWN','independent_testdev':'DATASET_BLOCKED',
        'CPU_seconds':time.perf_counter()-began},out)


def report(out):
    gate=load(out/'TRAINING_ADMISSION.json');data=load(out/'TRAINING_DATA_ACCEPTANCE.json') if (out/'TRAINING_DATA_ACCEPTANCE.json').exists() else {}
    loader=load(out/'PRODUCTION_DATASET_CPU_ACCEPTANCE.json') if (out/'PRODUCTION_DATASET_CPU_ACCEPTANCE.json').exists() else {}
    candidates=load(out/'CANDIDATE_INITIALIZATION_MANIFEST.json') if (out/'CANDIDATE_INITIALIZATION_MANIFEST.json').exists() else []
    validation=load(out/'VALIDATION_LOADER_ACCEPTANCE.json') if (out/'VALIDATION_LOADER_ACCEPTANCE.json').exists() else {}
    text=['# VM36：RQ4真实训练准备','',
        '## PROBLEM / WHY','',
        '核查目标化增强训练是否具备公平、可追溯的执行条件。原84组只证明已有权重上的开发集响应；不能证明加增强训练会缩小gap。','',
        '## METHOD','',
        '依据导师原计划第11—12页，主模型YOLOv8s、YOLO26s；1280训练，同组控制/增强共享初始化、数据、日程和种子，主比较至少三种子。',
        '本轮全部CPU执行。重新校验6471张train源图/标注SHA、解码、原坐标转换和现有派生标签；没有复制原图或修改原始标签。',
        '技术测试使用seed42、workers0、batch2、1280，每个控制/五增强联合/单家族连接测试16张锚图，两次重复。Mosaic实际会额外读取其他训练图片，访问记录保留。',
        '五增强复用原generator函数，顺序为Haze→Exposure→MotionBlur→Noise→JPEG，作用于原生像素后才进入resize、Mosaic和原几何增强。每次访问重新采样，缓存中保留未增强原图。',
        '技术fixture采用p=1以检测连接，全部具体数值保存在各loader_checks/RECIPE.json；不是科学训练范围、推荐超参数或已获批准的方案。','',
        '## RESULT','',f'最终状态：{gate["status"]}。CPU实际耗时{gate["CPU_seconds"]:.3f}秒。',
        f'完整训练图：{data.get("images","NOT_VERIFIED")}；有效原生十类目标：{data.get("positive_targets","NOT_VERIFIED")}；零面积行3条、去重框4个（以实际审计文件为准）。',
        '异常原值、物理行号见ANNOTATION_EXCEPTIONS.csv；四个重复框的图像与原标注SHA见DUPLICATE_BOX_CASES.csv。',
        '训练仅保留positive native10目标，没有对ignore区域实施loss mask；研究评价仍使用原ignore-union。两种口径明确分开。',
        f'完整val loader验收图数：{validation.get("images","NOT_VERIFIED")}；仅对本地字节不匹配/缺失的图片建立原ZIP字节私有视图，实际复制{validation.get("private_original_byte_image_copies","NOT_VERIFIED")}张，未覆盖原文件。',
        '内部验证使用库默认rect模式，其张量形状与标签口径见VALIDATION_LOADER_ACCEPTANCE.json；它不等于既有square输入、ignore-union研究COCO评价，未生成新AP。',
        '新本地COCO初始化和历史FT权重的SHA、全张量健康、best/last状态比对见HISTORICAL_CHECKPOINT_AUDIT.json与CANDIDATE_INITIALIZATION_MANIFEST.json。',
        '已保存的best epoch=-1，不能用100行训练曲线反推出选中epoch；本地同名COCO文件也不能证明历史使用过这些字节。',
        f'已验收{len(candidates)}份新native10候选初始化（目标为两模型×seed42/43/44）；只计入FP32临时保存、重载逐张量相等的文件。它们用于未来新匹配对照的候选起点，不是历史模型复现。','',
        '| CPU连接测试 | 全池图数 | 锚图/重复 | 两次完全一致 | 实际原图访问/重复 |','|---|---:|---:|---|---:|']
    for row in loader.get('checks',[]):text.append(f'| {row["arm"]} | {row["full_dataset_images"]} | {row["anchors_per_repeat"]} | {row["repeat_exact"]} | {row["native_image_accesses_per_repeat"]} |')
    text+=['','GPU训练0次、optimizer/EMA更新0次；没有新增AP/AR或增强有效性结果。真实训练显存、速度、保存和验证成本均NOT_MEASURED。',
        '现用Ultralytics8.4.61与历史权重记载8.4.33不同。默认训练保存会将EMA及optimizer状态转换为FP16，final_eval还会strip；不能把本轮CPU初始化FP32保存成功当作完整训练保存已验收。','',
        '## CONCLUSION','',
        '数据、候选初始化和CPU训练数据钩子可分别验收；还不能启动完整训练。MATCHED_RQ4_MATRIX_PROPOSAL.csv准备12次主对照和15次YOLOv8s leave-one-out消融，共27次，全部NOT_RUN。最小版本保留全部12次三seed主对照，加seed42的5次消融为17次；另外10次是消融seed重复，不擅自启动或削减主比较seed。',
        '后续新控制臂与新增强臂共享新初始化、派生标签及当前库版本；不能拿历史ck01/ck05直接替代新匹配控制。每个新权重需要独立checkpoint ID和SHA，既有84组不改写、不重标。',
        '三个种子42/43/44、100epochs均为沿用既有提案的候选；日程与增强数值未获新科学批准，不会为了正结果临时缩短训练。',
        'FROZEN_DEVELOPMENT_EVALUATION.json复用84组条件及原548身份、GT、配方和评价源码。Normal、JPEG25/75、Gaussian15和原复合条件均在固定集合中；未重新推理。','',
        '## LIMITATIONS / 待决定事项','',
        'SCIENCE：范围与概率、方向和airlight、fresh retrain或继续FT、单增强或leave-one-out、best-on-val或固定末轮，都在SCIENTIFIC_DECISIONS_REQUIRED.json逐项列出差异，没有默选。',
        'ENGINEERING：本轮仅workers0。历史workers2的预取跨epoch语义尚未准入；不能静默替换。生产训练Health Gate/FP32保存、实际梯度累积与GPU数值链路仍需单独验收。',
        'RESOURCE：这次没有新的有限GPU训练预算；此前推理授权已经结束，不可借用。GPU_CANARY_PLAN.json仅准备建议600秒的独立64训练图/16开发图工程检查，未执行。',
        '数据范围仍是development，test-dev独立准入未解决；不得把训练val选择数据称为最终独立测试。',
        'CPU失败分支：'+json.dumps(gate['failed_CPU_branches'],ensure_ascii=False),'',
        '## 证据与复现','',str(out),
        'CPU入口：'+str(HERE/'rq4_preparation.py')+' --prepare <本轮task.txt>；生成新Run，不启动GPU。',
        'GPU训练没有可执行的授权入口；当前trainer.train()明确拒绝训练。下一轮需先冻结科学选择与有限资源，再另建真实canary入口。']
    note=ROOT/'datasets/reportcn'/('vm36_supervisor_reproduction_RQ4训练准备_'+out.name.split('_')[-1]+'.md')
    write_atomic(note,'\n'.join(text)+'\n',out);save(out/'REPORT_REFERENCE.json',source(note),out)


def figures(out):
    if not (out/'TRAINING_DATA_ACCEPTANCE.json').exists():return
    os.environ['MPLCONFIGDIR']=str(out/'runtime/MPLCONFIGDIR')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from main.labels import VIS_NAMES
    data=load(out/'TRAINING_DATA_ACCEPTANCE.json');fig,ax=plt.subplots(figsize=(11,4.8),layout='constrained')
    ax.bar(VIS_NAMES,data['class_counts_zero_based'],color='#28658c');ax.tick_params(axis='x',rotation=35)
    ax.set_ylabel('Positive training targets after derived-view exclusions');ax.set_title('Verified native10 training labels: 6,471 images')
    fig.savefig(out/'training_class_counts.png',dpi=140);plt.close(fig)
    save(out/'FIGURE_PROVENANCE.json',{'input':source(out/'TRAINING_DATA_ACCEPTANCE.json'),
        'output':source(out/'training_class_counts.png'),'scope':'Training-label counts; not detection accuracy.'},out)


def execute(request):
    from development_guard import bounded_run
    out,_=new_run({'output_root':str(ROOT/'datasets/local_runs'/EXPERIMENT)},EXPERIMENT,'rq4_training_admission_cpu')
    shutil.copyfile(request,out/'author_task.txt');shutil.copyfile(Path(__file__),out/'executed_source.py')
    save(out/'inputs.json',{'author_request':source(out/'author_task.txt'),'accepted84':source(run_path(NORMAL)/'SHA256SUMS.txt'),
        'source':source(Path(__file__))},out);started=time.perf_counter()
    try:
        with (out/'worker.log').open('x',encoding='utf-8') as log:
            code=bounded_run([sys.executable,'-B',str(Path(__file__)),'--worker',out.name],log,HARD_CPU_SECONDS,
                dict(os.environ,CUDA_VISIBLE_DEVICES='',YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2'))
        if code:raise RuntimeError('CPU preparation failed; preserved log contains the evidence.')
    except BaseException:save(out/'failure.json',{'traceback':traceback.format_exc(),'retry':False},out)
    if (out/'TRAINING_ADMISSION.json').exists():report(out);figures(out);gate=load(out/'TRAINING_ADMISSION.json')
    else:gate={'status':'BLOCKED_CPU_PREPARATION','GPU_tasks':0,'training_runs':0}
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'RQ4_TRAINING_PREPARATION',**gate,
        'CPU_supervision_seconds':time.perf_counter()-started,'hard_CPU_seconds':HARD_CPU_SECONDS},out,replace=True)
    write_atomic(out/'diary.md','# RQ4 training admission\n\nRechecked the original training inputs, current trainer source and candidate initialization bytes. CPU effects and dataset checks remain engineering evidence. No accepted inference was repeated and no GPU training allocation was reused.\n',out)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>RQ4 training preparation</h1><p>'+gate['status']+'</p>'+''.join('<p><a href="'+n+'">'+n+'</a></p>' for n in ('REPORT_REFERENCE.json','TRAINING_ADMISSION.json','TRAINING_DATA_ACCEPTANCE.json','PRODUCTION_DATASET_CPU_ACCEPTANCE.json','HISTORICAL_CHECKPOINT_AUDIT.json','CANDIDATE_INITIALIZATION_MANIFEST.json','MATCHED_RQ4_MATRIX_PROPOSAL.csv','SCIENTIFIC_DECISIONS_REQUIRED.json','GPU_CANARY_PLAN.json'))+('<img style="max-width:100%" src="training_class_counts.png">' if (out/'training_class_counts.png').exists() else ''),out)
    seal(out)
    from main.visualization import build_run_index
    build_run_index(out.parent,EXPERIMENT)
    latest=load(out.parent/'latest.json');latest.update(latest_RQ4_preparation_RunID=out.name,RQ4_training_status=gate['status'],
        combined_measured_pairs=84,formal_measured_pairs=0,latest_normal_reference_results_RunID=NORMAL)
    save(out.parent/'latest.json',latest,out,replace=True);print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--prepare',type=Path);choice.add_argument('--worker');args=parser.parse_args()
    if args.prepare:execute(args.prepare)
    else:worker(run_path(args.worker))
