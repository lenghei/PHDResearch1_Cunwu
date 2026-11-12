"""Freeze separate20-epoch and formal100-epoch plans from accepted inputs."""
import csv
import importlib.metadata
import json
from pathlib import Path
import sys

POST=Path(__file__).resolve().parent;BASE=POST.parent
sys.path.insert(0,str(BASE))
from study import *


def relative(path,root):return Path(path).resolve().relative_to(root.resolve()).as_posix()


def execute():
    out=create('postcost_protocol_cpu');runtime(out,cuda=False)
    reference=RUNS/'vm37_yolo26_rq4_full_epoch_yolo26s_control_20261111T232439_301fb25d'
    verify_hashes(reference,('protocol.json','training_acceptance.json'))
    historical=load(PREP/'FROZEN_DEVELOPMENT_EVALUATION.json')
    with (PREP/'TRAINING_DATA_VERIFICATION.csv').open(encoding='utf-8-sig') as stream:train=list(csv.DictReader(stream))
    val=load(PREP/'VALIDATION_LOADER_ACCEPTANCE.json')['population'];lookup=load(PREP/'PRIVATE_LABEL_LOOKUP.json')
    dataset=ROOT/'datasets/VisDrone';portable={'train':[],'val':[]};assets=[]
    for split,rows in (('train',train),('val',val)):
        for row in rows:
            stem=row['image_id'] if split=='train' else row['stem']
            path=Path(row['image_path'] if split=='train' else row['loader_path'])
            expected=row['image_SHA256'] if split=='train' else row['source_image_SHA256']
            image_relative='VisDrone2019-DET-'+split+'/images/'+stem+'.jpg'
            override=None
            if not path.resolve().is_relative_to(dataset.resolve()):
                override='assets/original_val_bytes/'+stem+'.jpg';assets.append({'path':str(path),'asset':override,'sha256':expected})
            label=Path(lookup[stem]) if split=='train' else PREP/'validation_labels'/(stem+'.txt')
            label_asset='assets/labels/'+split+'/'+stem+'.txt'
            assets.append({'path':str(label),'asset':label_asset,'sha256':row['derived_label_SHA256']})
            portable[split].append({'image_id':stem,'image_relative':image_relative,'image_SHA256':expected,'image_override':override,
                'annotation_relative':'VisDrone2019-DET-'+split+'/annotations/'+stem+'.txt',
                'annotation_SHA256':row['annotation_SHA256'] if split=='train' else row['source_annotation_SHA256'],
                'label_asset':label_asset,'label_SHA256':row['derived_label_SHA256']})
    initializations=[]
    for row in load(PREP/'CANDIDATE_INITIALIZATION_MANIFEST.json'):
        asset='assets/initializations/'+row['architecture']+'_seed'+str(row['seed'])+'_native10.pt'
        assets.append({'path':row['native10_candidate']['path'],'asset':asset,'sha256':row['native10_candidate']['sha256']})
        initializations.append({'architecture':row['architecture'],'seed':row['seed'],'asset':asset,
            'sha256':row['native10_candidate']['sha256'],'COCO_source_SHA256':row['source']['sha256'],'role':'UNTRAINED_NATIVE10_INITIALIZATION'})
    gt=load(PREP/'SUPERVISOR_REQUIREMENTS.json')['frozen_source_hashes']['GT']
    assets.append({'path':gt['path'],'asset':'assets/ground_truth.json','sha256':gt['sha256']})
    cases=[]
    for label,selected in [('Normal',{'Normal'}),('JPEG_Q25',{'Normal','JPEG_q25'}),
                            ('JPEG_Q75',{'Normal','JPEG_q75'}),('Gaussian_s15',{'Normal','Gauss_s15'})]:
        ref=next(row for row in historical['conditions'] if row['checkpoint']=='ck01' and row['inference_resolution']==1280 and set(row['leaves'])==selected)
        folder=Path(ref['source_folder']);config=load(folder/'configuration.json')
        if sha256(folder/'configuration.json')!=ref['source_SHA256']['configuration.json']:raise ValueError('Historical recipe changed.')
        manifest='assets/evaluation_pixels/'+label+'.jsonl'
        assets.append({'path':str(folder/'image_manifest.jsonl'),'asset':manifest,'sha256':ref['source_SHA256']['image_manifest.jsonl']})
        cases.append({'id':label,'p':config['p'],'historical_source_pair':ref['pair_id'],'pixel_manifest_asset':manifest,
                      'pixel_manifest_SHA256':ref['source_SHA256']['image_manifest.jsonl']})
    policy=source(BASE/'TARGETED_POLICY.json');assets.append({'path':policy['path'],'asset':'assets/TARGETED_POLICY.json','sha256':policy['sha256']})
    for name in ('fm_p.uvl','fm_d.uvl'):
        path=VM36/'feature_models'/name;assets.append({'path':str(path),'asset':'assets/original_uvl/'+name,'sha256':sha256(path)})
    ordinary={}
    import yaml
    for architecture,model in [('YOLO26s','ck01_yolo26s_ft1280'),('YOLOv8s','ck05_yolov8s_ft1280')]:
        path=ROOT/'models/checkpoints'/model/'args.yaml';ordinary[architecture]=yaml.safe_load(path.read_text())
        ordinary[architecture].pop('save_dir',None)
    evaluation={'scope':'DEVELOPMENT_ONLY','population':548,'resolutions':[640,1280],'contexts':cases,'render_seed':20261108,
        'precision':'FP32','head':'dense one-to-many plus NMS','conf':.001,'NMS_IoU':.7,'max_det':3000,'max_nms':30000,
        'multi_label':True,'class_aware':True,'geometry':'square Letterbox; original bbox coordinates',
        'GT_asset':'assets/ground_truth.json','GT_SHA256':gt['sha256'],'bootstrap_samples':200,'bootstrap_seed':20261112,
        'bootstrap_unit':'Same original-image position indices for all arms, resolutions and contexts; recompute aggregate COCO AP.',
        'difference':'ALL_FIVE minus CONTROL','degradation_loss':'AP_Normal minus AP_condition',
        'loss_change':'loss_ALL_FIVE minus loss_CONTROL; negative indicates smaller loss',
        'not_full21_condition_gap':True,'training_seed_uncertainty':'NOT_IDENTIFIED_BY_ONE_SEED'}
    shared={'imgsz':1280,'batch':2,'nbs':64,'optimizer':'AdamW','lr0':.01,'cos_lr':True,'warmup_epochs':3.,
            'amp':False,'workers':0,'cls_remap':False,'end2end':False,'resume':False,'device':'0'}
    pilot={'id':'PILOT_YOLO26S_CONTROL_VS_ALL_FIVE','version':'VM37_PILOT20_V1','status':'FROZEN_NOT_ADMITTED',
        'architectures':['YOLO26s'],'seeds':[42],'arms':['CONTROL','ALL_FIVE'],'epochs':20,'training_images':6471,'development_images':548,
        'shared_training_args':shared,'schedule_horizon_epochs':20,'scheduler':'Ultralytics8.4.210 cosine with its own20-epoch horizon',
        'planned_counters_per_arm':expected_steps(6471,20),'planned_anchor_exposures_per_arm':6471*20,
        'initialization_rule':'Exact same native10 initialization SHA for both arms; no old cost or historical FT checkpoint.',
        'selection':'Fixed epoch20 EMA','strategy_label':'ALL_FIVE is broad independent stochastic augmentation, not proven Gap-guided training.',
        'policy_SHA256':policy['sha256'],'duration_rationale':'Budget-driven exploratory choice, not an optimal epoch count or substitute for formal100.',
        'evaluation':evaluation,'resource_gate':'Both complete arms, evaluation, saving, generation, validation and safety margin must fit the unchanged allocation and deadline.',
        'execute_default':False}
    formal={'id':'VM37_FORMAL100','status':'RESOURCE_BLOCKED_LOCAL','architectures':['YOLO26s','YOLOv8s'],
        'seeds':[42,43,44],'arms':['CONTROL','ALL_FIVE'],'epochs':100,'training_images':6471,'development_images':548,
        'shared_training_args':shared,'schedule_horizon_epochs':100,'selection':'Fixed epoch100 EMA',
        'planned_counters_per_arm':expected_steps(6471,100),'contexts_per_checkpoint':21,'inference_resolutions':[640,1280],
        'full_matrix_training_arms':12,'evaluation_configurations_per_matched_pair':84,
        'data_boundary':'Development validation only until independent data admission. No test-dev.',
        'execute_default':False,'server_budget':'New server-specific explicit resource authorization and measured admission required.'}
    for name,data in [('pilot20_protocol',pilot),('formal100_protocol',formal),('portable_dataset_manifest',portable),
                      ('portable_initializations',initializations),('evaluation_manifest',evaluation),('ordinary_training_args',ordinary),('private_asset_sources',assets)]:
        save(out/(name+'.json'),data,out)
    observed={name:importlib.metadata.version(name) for name in ('torch','torchvision','ultralytics','numpy','opencv-python','opencv-python-headless','pycocotools','albumentations','albucore','PyYAML','matplotlib','pillow','scipy','psutil')}
    save(out/'environment_observed.json',{'Python':sys.version,'packages':observed,'CUDA_runtime':'12.4','official_Ultralytics_commit':'c6427a7816d42ab4022b5f9a0242fe93e40e1ff8'},out)
    for name,data in [('PILOT20_PROTOCOL.json',pilot),('FORMAL100_PROTOCOL.json',formal)]:
        write_atomic(POST/name,json.dumps(data,indent=2)+'\n',ROOT)
    save(out/'inputs.json',{'accepted_cost':source(reference/'training_acceptance.json'),'accepted84':source(PREP/'FROZEN_DEVELOPMENT_EVALUATION.json'),
                           'policy':policy,'data_manifests':[source(PREP/name) for name in ('TRAINING_DATA_VERIFICATION.csv','VALIDATION_LOADER_ACCEPTANCE.json')]},out)
    write_atomic(out/'diary.md','# Separate pilot and portable formal design\n\nFroze an explicit20-epoch exploratory pair before any new training. Preserved100-epoch design and original initialization/data identities. Four pilot contexts are copied from accepted recipes, with identical pixel-manifest checks. No GPU execution or new budget.\n',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'POSTCOST_PROTOCOL_CPU','status':'PREPARED_NOT_ADMITTED','GPU_seconds':0},out,replace=True)
    seal(out);print(out,flush=True)


if __name__=='__main__':execute()
