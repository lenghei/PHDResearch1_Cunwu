"""Read original training bytes and test a private derived-label candidate on CPU."""
import argparse
from collections import Counter
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import time
import traceback
from zipfile import ZipFile
import numpy as np
from protocol import ROOT,HERE,EXPERIMENT,load,save,source,sha256,write_atomic
from common import seal
from historical_review import table,csv_file
from main.pipeline import new_run

EXPECTED_ZIP='86a77eba93137bfc16e4993860de9245b0675c0dba0d3ab98fb458699e256f84'
LIMIT=900
VERSION='VISDRONE8_OPTIONAL_TRAILING_DELIMITER_V2'


def native_rows(text,width,height):
    values=[];exceptions=[];crossing=0
    for line_no,line in enumerate(text.splitlines(),1):
        if not line.strip():continue
        fields=line.strip().split(',')
        while fields and not fields[-1].strip():fields.pop()
        if len(fields)!=8 or any(not field.strip() for field in fields):raise ValueError('Malformed eight-field annotation row.')
        row=[float(x.strip()) for x in fields]
        if not np.isfinite(row).all():raise ValueError('Nonfinite annotation row.')
        x,y,w,h,score,category=row[:6]
        if w<=0 or h<=0:
            exceptions.append({'physical_line':line_no,'raw_row':line,'reason':'NONPOSITIVE_AREA',
                'positive_native10':bool(score>0 and 1<=category<=10)})
            continue
        if not (score>0 and category in range(1,11)):continue
        crossing+=bool(x<0 or y<0 or x+w>width or y+h>height)
        values.append([int(category)-1,(x+w/2)/width,(y+h/2)/height,w/width,h/height])
    array=np.asarray(values,dtype=np.float32).reshape(-1,5)
    if len(array) and (array[:,1:].min() < -.01 or array[:,1:].max()>1.01):
        raise ValueError('Labels exceed the installed loader coordinate tolerance; no clipping performed.')
    if len(array):
        _,indices=np.unique(array,axis=0,return_index=True)
        duplicate_count=len(array)-len(indices)
        if duplicate_count:array=array[indices]
    else:duplicate_count=0
    return array,exceptions,crossing,duplicate_count


def loader_smoke(out,manifest,labels,reseed_after_build=False):
    import torch
    from PIL import Image
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.data.utils import verify_image_label
    from ultralytics.utils.torch_utils import init_seeds
    from main.labels import VIS_NAMES
    import inspect
    args=__import__('yaml').safe_load((ROOT/'models/checkpoints/ck05_yolov8s_ft1280/args.yaml').read_text())
    selected=['0000293_03401_d_0000939','9999985_00000_d_0000020','9999999_00590_d_0000267']
    selected+= [r['image_id'] for r in manifest if r['image_id'] not in selected][:13]
    image_dir=out/'loader_smoke/images/train';label_dir=out/'loader_smoke/labels/train'
    image_dir.mkdir(parents=True);label_dir.mkdir(parents=True)
    by_id={r['image_id']:r for r in manifest};archive=ROOT/'datasets/VisDrone/VisDrone2019-DET-train.zip'
    before={}
    with ZipFile(archive) as z:
        for stem in selected:
            record=by_id[stem];p=image_dir/(stem+'.jpg')
            p.write_bytes(z.read(record['image_member']));before[stem]=sha256(p)
            shutil.copyfile(labels/(stem+'.txt'),label_dir/(stem+'.txt'))
    hyp=get_cfg(overrides={k:args[k] for k in ('imgsz','hsv_h','hsv_s','hsv_v','degrees','translate','scale','shear',
        'perspective','flipud','fliplr','bgr','mosaic','mixup','cutmix','copy_paste','copy_paste_mode') if k in args})
    hyp.workers=0;hyp.cache=False;hyp.rect=False
    def one_pass(seed):
        init_seeds(seed,deterministic=True)
        data=YOLODataset(img_path=str(image_dir),imgsz=1280,batch_size=2,augment=True,hyp=hyp,
            rect=False,cache=False,single_cls=False,stride=32,pad=0.,prefix='CPU engineering: ',
            data={'names':dict(enumerate(VIS_NAMES)),'nc':10,'channels':3},task='detect')
        if reseed_after_build:init_seeds(seed,deterministic=True)
        outputs=[]
        for i in range(len(data)):
            result=data[i]
            if tuple(result['img'].shape)!=(3,1280,1280):raise ValueError('Unexpected augmented tensor.')
            boxes=result['bboxes'];classes=result['cls']
            if not torch.isfinite(boxes).all() or not torch.isfinite(classes).all():raise ValueError('Nonfinite augmented labels.')
            if len(boxes) and (bool((boxes[:,2:]<=0).any()) or bool((classes<0).any()) or bool((classes>=10).any())):
                raise ValueError('Invalid augmented target.')
            outputs.append({'id':Path(result['im_file']).stem,'shape':list(result['img'].shape),'targets':len(boxes),
                'pixel_sha256':hashlib.sha256(result['img'].numpy().tobytes()).hexdigest(),
                'targets_sha256':hashlib.sha256(boxes.numpy().tobytes()+classes.numpy().tobytes()).hexdigest()})
        return outputs
    first=one_pass(42);second=one_pass(42)
    unchanged=all(sha256(image_dir/(s+'.jpg'))==digest for s,digest in before.items())
    if not unchanged:raise ValueError('Private loader image copies changed.')
    fixture=out/'zero_height_fixture.jpg';Image.new('RGB',(32,32)).save(fixture)
    fixture_label=out/'zero_height_fixture.txt';fixture_label.write_text('3 0.5 0.5 0.1 0\n')
    verified=verify_image_label((str(fixture),str(fixture_label),'CPU fixture: ',False,10,0,0,False))
    save(out/'LOADER_SOURCE.json',{'verify_image_label':source(Path(inspect.getsourcefile(verify_image_label))),
        'dataset':source(Path(inspect.getsourcefile(YOLODataset))),
        'source_args':source(ROOT/'models/checkpoints/ck05_yolov8s_ft1280/args.yaml')},out)
    return {'status':'CPU_DATA_PATH_SMOKE_TESTED' if first==second else 'SEED_REPEAT_UNVERIFIED','images_per_repeat':16,'repeats':2,'seed':42,'repeat_exact':first==second,
        'original_images_modified':False,'private_image_copies_unchanged':unchanged,'imgsz':1280,'new_model_forward_backward':0,
        'zero_height_fixture_accepted_by_installed_verifier':bool(verified[0] is not None),
        'zero_height_fixture_targets':None if verified[1] is None else verified[1].tolist(),
        'fixture_scope':'Loader validation behavior, not a YOLO detection or training-failure claim.',
        'source':'Unmodified installed Ultralytics, conventional augmentation from archived args; no target augmentation added.',
        'outputs':first,'second_outputs':second,'reseed_after_dataset_construction':reseed_after_build,
        'full6471_library_loader_test':'NOT_RUN; complete custom decode/label audit is separate.'}


def execute():
    started=time.perf_counter();os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['YOLO_AUTOINSTALL']='false'
    out,_=new_run({'output_root':str(ROOT/'datasets/local_runs'/EXPERIMENT)},EXPERIMENT,'training_data_cpu_v2')
    shutil.copyfile(Path(__file__),out/'executed_source.py')
    archive=ROOT/'datasets/VisDrone/VisDrone2019-DET-train.zip';label_dir=out/'derived_labels_native10_v1';label_dir.mkdir()
    records=[];exceptions=[];status='BLOCKED';smoke=None
    try:
        if sha256(archive)!=EXPECTED_ZIP:raise ValueError('Original training ZIP SHA changed.')
        import cv2
        from PIL import Image
        cv2.setNumThreads(2)
        with ZipFile(archive) as z:
            images=sorted(n for n in z.namelist() if '/images/' in n and n.lower().endswith('.jpg'))
            if len(images)!=6471:raise ValueError('Training image population differs.')
            for i,member in enumerate(images):
                if time.perf_counter()-started>LIMIT:raise TimeoutError('CPU data audit budget reached.')
                stem=Path(member).stem;ann=member.replace('/images/','/annotations/').rsplit('.',1)[0]+'.txt'
                raw=z.read(member);annotation=z.read(ann);digest=hashlib.sha256(raw).hexdigest()
                image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
                if image is None or image.dtype!=np.uint8:raise ValueError('Cannot decode original training image: '+stem)
                height,width=image.shape[:2]
                with Image.open(io.BytesIO(raw)) as pil:
                    orientation=pil.getexif().get(274);encoded_size=pil.size
                if encoded_size!=(width,height):raise ValueError('Encoded and decoded geometry differ.')
                if orientation not in (None,1):raise ValueError('Unresolved training EXIF orientation: '+stem)
                values,invalid,crossing,duplicates=native_rows(annotation.decode('utf-8-sig'),width,height)
                output=label_dir/(stem+'.txt')
                output.write_text(''.join(' '.join(format(float(v),'.9g') for v in row)+'\n' for row in values),encoding='utf-8')
                roundtrip=np.loadtxt(output,dtype=np.float32,ndmin=2) if len(values) else np.zeros((0,5),np.float32)
                if not np.array_equal(values,roundtrip):raise ValueError('Derived label serialization differs.')
                current=ROOT/'datasets/VisDrone/VisDrone2019-DET-train/images'/Path(member).name
                same_current=current.exists() and sha256(current)==digest
                records.append({'image_id':stem,'image_member':member,'annotation_member':ann,'width':width,'height':height,
                    'image_SHA256':digest,'annotation_SHA256':hashlib.sha256(annotation).hexdigest(),'orientation':orientation,
                    'current_extracted_image_bytes_match':same_current,'valid_native10_targets':len(values),
                    'duplicates_removed_like_library':duplicates,'boundary_crossing_targets_preserved':crossing,
                    'derived_label_SHA256':sha256(output),'class_counts':json.dumps(dict(Counter(map(int,values[:,0]))))})
                exceptions.extend({'image_id':stem,**row,'annotation_SHA256':hashlib.sha256(annotation).hexdigest()} for row in invalid)
                if (i+1)%500==0:print('TRAIN_DECODE_LABELS',i+1,'/6471',flush=True)
        if len(exceptions)!=3 or sum(r['positive_native10'] for r in exceptions)!=1:
            raise ValueError('Zero-area exception set differs from the prior audit.')
        smoke=loader_smoke(out,records,label_dir)
        status='CPU_DATA_CANDIDATE_VERIFIED_NOT_TRAINING_ADMITTED' if smoke['repeat_exact'] else 'DATA_AUDITED_SEED_PATH_UNVERIFIED'
    except BaseException:
        save(out/'failure.json',{'traceback':traceback.format_exc(),'originals_modified':False,'automatic_retry':False},out)
    csv_file(out/'TRAINING_IMAGE_LABEL_AUDIT.csv',records)
    if exceptions:csv_file(out/'ANNOTATION_EXCLUSIONS.csv',exceptions)
    if smoke:save(out/'LOADER_SMOKE.json',smoke,out)
    result={'status':status,'parser_version':VERSION,'complete_images_audited':len(records),'expected_images':6471,'new_GPU_tasks':0,
        'new_training_runs':0,'zero_area_rows':len(exceptions),'positive_native10_zero_area_rows':sum(r['positive_native10'] for r in exceptions),
        'current_extracted_byte_mismatches':sum(not r['current_extracted_image_bytes_match'] for r in records),
        'valid_native10_targets_after_dedup':sum(r['valid_native10_targets'] for r in records),
        'duplicate_targets_removed':sum(r['duplicates_removed_like_library'] for r in records),
        'boundary_crossing_targets_not_clipped':sum(r['boundary_crossing_targets_preserved'] for r in records),
        'original_zip_unchanged':sha256(archive)==EXPECTED_ZIP,'derived_view':'Candidate only; does not change archived or active training labels.',
        'CPU_wall_seconds':time.perf_counter()-started,'CPU_budget_seconds':LIMIT,
        'remaining':['GPU training canary and measured current full-run cost not performed','Fresh matched initialization SHA required',
            'Target augmentation ranges/orientation/airlight unapproved','No newly authorized training budget','Formal independent evaluation still blocked']}
    save(out/'TRAINING_DATA_GATE.json',result,out)
    save(out/'inputs.json',{'archive':source(archive),'source':source(Path(__file__)),
        'parser_fix':'Eight original fields with an optional trailing CSV delimiter; internal missing fields still reject. Prior4504-image blocked audit retained.'},out)
    write_atomic(out/'diary.md','# Training data engineering audit\n\nRead original ZIP bytes and coordinates. Wrote a separate native10 label candidate. Only private small image copies enter the installed library loader; source images and labels remain unchanged. No detector forward, backward or GPU work.\n',out)
    note=ROOT/'datasets/reportcn'/('vm36_supervisor_reproduction_训练数据CPU验收_'+out.name.split('_')[-1]+'.md')
    text=['# VM36 训练数据CPU验收','', 'PROBLEM：三条零面积标注和实际loader行为是否妨碍匹配训练。',
        'WHY：原标签必须只读，不能让两训练组采用不同过滤规则。',
        'SOLUTION：逐图读取原ZIP并核对尺寸、EXIF、源SHA和native10标签；在本Run写独立候选标签；仅复制16图进行普通增强的数据路径检查。',
        'WHY IS IT A GOOD SOLUTION：过滤记录、重复框处理、像素和张量可核验；没有模型训练或精度结论。','',json.dumps(result,ensure_ascii=False,indent=2),
        '完整日志与错误原值：'+str(out),'候选标签没有接入生产训练。']
    write_atomic(note,'\n'.join(text)+'\n',out);save(out/'REPORT_REFERENCE.json',source(note),out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'TRAINING_DATA_CPU','status':status},out,replace=True)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>Training data CPU audit</h1><p>'+status+'</p><a href="TRAINING_DATA_GATE.json">Gate</a> · <a href="TRAINING_IMAGE_LABEL_AUDIT.csv">Image and label audit</a>',out)
    save(out/'DERIVED_LABEL_MANIFEST.json',{p.name:sha256(p) for p in sorted(label_dir.glob('*.txt'))},out)
    seal(out)
    from joint_analysis import refresh
    from matched_followup import ANALYSIS_ID
    refresh(out.parent/ANALYSIS_ID);print(out,flush=True)


if __name__=='__main__':execute()
