"""Bounded source admission and matched-training preparation; no GPU work."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import shutil
import time
from zipfile import ZipFile
import yaml
from protocol import ROOT,HERE,EXPERIMENT,FMP,load,save,source,sha256,write_atomic
from common import seal,verify_hashes
from historical_review import table,csv_file
from main.pipeline import new_run

DATA_RUN='vm36_supervisor_reproduction_training_data_cpu_v2_20261110T110058_057fb01c'
SEED_RUN='vm36_supervisor_reproduction_loader_seed_diagnostic_20261110T110444_70a8cd35'
RECON_RUN='vm36_supervisor_reproduction_uavdt_reconciliation_20261110T004456_299551e8'
TOOLKIT_RUN='vm36_supervisor_reproduction_uavdt_toolkit_source_20261110T004050_44d7a453'


def independent_audit(out):
    root=ROOT/'datasets/research_archive/paper1';records=[]
    configs=sorted((root/'source_configs').glob('*.yaml'))
    configs+=[root/'ieee_results/resolved_config.json',root/'ieee_results_v3/protocol.json']
    for p in configs:
        text=p.read_text(encoding='utf-8')
        records.append({'file':source(p),'record_kind':'SAVED_CONFIG_OR_PROTOCOL',
            'direct_testdev_text_reference':any(s in text.lower() for s in ('test-dev','test_dev','testdev')),
            'interpretation':'A configuration string is not evidence that an evaluation ran or that no other evaluation ran.'})
    training=yaml.safe_load((root/'source_configs/visdrone_10class.yaml').read_text())
    source_index=ROOT/'datasets/research_archive/migration_20261104/file_manifest.csv'
    matches=[]
    for row in table(source_index):
        if row['source'].replace('\\','/').split('/')[-1] in ('yolov8s.pt','yolo26s.pt','yolo11s.pt'):
            matches.append(row)
    model_rows=[]
    for ck,name in (('ck01','ck01_yolo26s_ft1280'),('ck05','ck05_yolov8s_ft1280'),('ck06','ck06_yolo11s_ft1280')):
        args=ROOT/'models/checkpoints'/name/'args.yaml';settings=yaml.safe_load(args.read_text())
        model_rows.append({'checkpoint':ck,'args':source(args),'declared_initial_model':settings['model'],
            'declared_train':training['train'],'declared_val':training['val'],'validation_selection_enabled':settings.get('val'),
            'training_byte_membership_at_creation':'UNKNOWN','selected_epoch_provenance':'UNKNOWN',
            'testdev_prior_evaluation':'UNKNOWN','testdev_method_or_parameter_selection':'UNKNOWN',
            'formal_admission':'DATASET_BLOCKED','reason':'Direct args/YAML support recorded train/val intent; no complete creation-time or decision/use inventory.'})
    save(out/'TESTDEV_EVIDENCE.json',{'checkpoint_rows':model_rows,'direct_saved_sources':records,
        'prior_file_disjointness_evidence':source(ROOT/'datasets/local_runs'/EXPERIMENT/'vm36_supervisor_reproduction_continuation_audit_20261110T002932_d2a7e70c/INDEPENDENT_ADMISSION.json'),
        'official_testdev_images':1610,'new_testdev_inferences':0,'absence_of_hits_proves_no_prior_use':False,
        'status':'DATASET_BLOCKED','required':['Creation-time training and initialization byte manifests','Checkpoint-selection record including selected epoch',
            'Accountable prior test-dev evaluation and research decision inventory for each fixed checkpoint'],
        'independent_plan':'Recover the missing checkpoint-specific history, or prospectively freeze compatible new native10 scenes, source/scene separation, hypotheses and protocol before inspection. No automatic dataset replacement.'},out)
    save(out/'INITIALIZATION_PROVENANCE.json',{'bounded_index':source(source_index),'matching_pretrained_archive_records':matches,
        'search_scope':'Known migration index only; no all-disk absence claim.',
        'historical_initialization_bytes':'NOT_VERIFIED','future_initialization':'Select one explicit healthy COCO source per architecture; freeze its SHA and per-seed native10 initialization before any matched arm.',
        'no_model_download_or_substitution':True},out)


def uavdt_audit(out):
    import cv2
    root=ROOT/'datasets/local_runs'/EXPERIMENT;recon=root/RECON_RUN;toolkit=root/TOOLKIT_RUN
    verify_hashes(recon,('M_FRAME_RECONCILIATION.csv','OFFICIAL_EVALUATION_SEMANTICS.json','OFFICIAL_MEMBER_SHA256.json'))
    verify_hashes(toolkit,('SOURCE.json','UAV-benchmark-MOTD_v1.0.zip'))
    groups=defaultdict(list)
    for row in table(recon/'M_FRAME_RECONCILIATION.csv'):groups[(row['split'],row['sequence'])].append(row)
    examples=[];categories=Counter();ignored=0;start=time.perf_counter()
    with ZipFile(toolkit/'UAV-benchmark-MOTD_v1.0.zip') as z:
        for (split,seq),frames in sorted(groups.items()):
            if time.perf_counter()-start>180:raise TimeoutError('Bounded UAVDT sample audit stopped.')
            frames.sort(key=lambda r:int(r['frame']))
            selected=[frames[i] for i in sorted({0,len(frames)//2,len(frames)-1})]
            gt=z.read('UAV-benchmark-MOTD_v1.0/GT/'+seq+'_gt_whole.txt').decode('utf-8-sig')
            for line in gt.splitlines():
                if line.strip():categories[int(line.split(',')[8])]+=1
            for row in selected:
                image=ROOT/'datasets/uavdt'/split/'img'/row['image_file']
                pixels=cv2.imread(str(image),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
                if pixels is None:raise ValueError('Cannot decode UAVDT source example.')
                height,width=pixels.shape[:2]
                if (width,height)!=(int(row['annotation_width']),int(row['annotation_height'])):
                    raise ValueError('Image and annotation geometry differ.')
                examples.append({'split':split,'sequence':seq,'frame':int(row['frame']),'width':width,'height':height,
                    'source_image':source(image),'converted_annotation_SHA256':row['converted_annotation_SHA256'],
                    'official_ignore_regions':int(row['official_ignore_regions']),
                    'source_byte_identity_to_original_image_release':'UNKNOWN_NO_VERIFIED_ORIGINAL_IMAGE_ARCHIVE'})
    csv_file(out/'UAVDT_IMAGE_SAMPLE.csv',[{**{k:v for k,v in r.items() if k!='source_image'},
        'image_path':r['source_image']['path'],'image_SHA256':r['source_image']['sha256']} for r in examples])
    semantics=load(recon/'OFFICIAL_EVALUATION_SEMANTICS.json')
    save(out/'UAVDT_ADMISSION_UPDATE.json',{'status':'NOT_ADMITTED_FOR_INDEPENDENT_EVALUATION','sequences':len(groups),
        'decoded_fixed_first_middle_last_frames':len(examples),'image_geometry_matches_converted_metadata':True,
        'image_byte_provenance':'UNKNOWN; decoded geometry and names do not prove original image-byte identity.',
        'official_GT_categories':dict(categories),'SOT':'Excluded from DET; no vehicle class substitution.',
        'category_mapping_candidate':{'VisDrone_zero_based_car_3':'UAVDT_car_1','VisDrone_zero_based_truck_5':'UAVDT_truck_2','VisDrone_zero_based_bus_8':'UAVDT_bus_3'},
        'category_semantics':'PROPOSED_NOT_ADMITTED; car/van inclusion needs a verified ontology rule. No silent van merge.',
        'ignore_and_metric':semantics,'pr01_compatibility':'Different official ignore/AP semantics; needs a new separately named protocol, not pr01 concatenation.',
        'prior_model_exposure':'Recorded training is VisDrone; manual tuning/selection and image-release history remain unverified.',
        'source_license':source(ROOT/'datasets/uavdt/LICENSE.md'),'redistribution_permission':'NOT_INFERRED',
        'original_toolkit':source(toolkit/'SOURCE.json'),'new_GPU_inferences':0},out)


def training_matrix(out):
    root=ROOT/'datasets/local_runs'/EXPERIMENT;data=root/DATA_RUN;seed=root/SEED_RUN
    verify_hashes(data,('TRAINING_DATA_GATE.json','ANNOTATION_EXCLUSIONS.csv','DERIVED_LABEL_MANIFEST.json'))
    verify_hashes(seed,('SEED_DIAGNOSTIC.json','nested_sha256.json'))
    audit=load(data/'TRAINING_DATA_GATE.json');diagnostic=load(seed/'SEED_DIAGNOSTIC.json')
    augmentations=['NoiseAug','MotionBlurAug','ExposureAug','JPEGAug','HazeAug'];rows=[]
    for family in ('YOLOv8s','YOLO26s'):
        for training_seed in (42,43,44):
            arms=['DEFAULT_CONTROL','ALL_FIVE']+(['WITHOUT_'+name for name in augmentations] if family=='YOLOv8s' else [])
            for arm in arms:
                active=[] if arm=='DEFAULT_CONTROL' else [name for name in augmentations if arm!='WITHOUT_'+name]
                rows.append({'architecture':family,'seed_candidate':training_seed,'seed_status':'PROPOSED_NOT_EXECUTED',
                    'arm':arm,'added_augmentations':'|'.join(active),'initialization_SHA256':'NOT_FROZEN',
                    'imgsz':1280,'epochs':100,'batch':2,'nbs':64,'optimizer':'AdamW','lr0':.01,
                    'schedule':'cos_lr=true; inherited archived schedule','checkpoint_rule':'Fixed epoch100 EMA; no best-epoch substitution',
                    'train_images':6471,'status':'NOT_RUN','budget':'NOT_AUTHORIZED'})
    csv_file(out/'MATCHED_TRAINING_MATRIX_PROPOSAL.csv',rows)
    params=__import__('supervisor_compliance').numeric_attributes(FMP.read_text())
    save(out/'AUGMENTATION_RANGE_GATE.json',{'status':'NOT_FROZEN','five_families':augmentations,
        'test_attributes':[{'feature':feature,'attribute':attribute,'value':value} for (feature,attribute),value in params.items()],
        'training_ranges':'NOT_SELECTED; ranges must contain test levels while avoiding their exact discrete test settings.',
        'missing_bindings':['Motion orientation/distribution','Haze airlight/geometry','Approved training-only range endpoints and sampling probabilities'],
        'no_new_range_invented':True,'original_UVL_parameters_changed':False},out)
    save(out/'RQ4_GATE.json',{'status':'TRAINING_NOT_READY','complete_training_images_audited':audit['complete_images_audited'],
        'valid_targets':audit['valid_native10_targets_after_dedup'],'three_zero_height_rows':source(data/'ANNOTATION_EXCLUSIONS.csv'),
        'derived_label_view':'Candidate only; full training loader integration is not active. Sixteen private copies tested, not a6471-image optimizer run.',
        'seed_diagnostic':diagnostic,'seed_protocol':'Freeze initialization, data-loader generator, augmentation RNG, cache and worker initialization for all matched arms. CPU reseeding result is not a production-trainer acceptance.',
        'primary_matched_runs':12,'three_seed_leave_one_augmentation_out_runs':15,'full_matrix_runs':len(rows),
        'ablation_definition':'Proposed leave-one-out from all five, YOLOv8s only; not yet executed. The original plan does not uniquely resolve single-add versus leave-one-out.',
        'budget_staging':'Measure one separately authorized training canary; then cost one full run before allocating primary arms. Do not launch27 runs or multiply a mini epoch by100.',
        'historical366_90_hours':'Historical cumulative-time screening only, not a forecast for this laptop.',
        'current_full_run_cost':'NOT_MEASURED','actual_training_steps':0,
        'canary_proposal':{'images':16,'epochs':1,'imgsz':1280,'batch':2,'nbs':64,'worker_hard_limit_seconds':600,
            'status':'NOT_AUTHORIZED_NOT_RUN','purpose':'Forward/backward, actual optimizer/EMA count, health, FP32 save/reload and timing; no efficacy conclusion.'}},out)


def execute():
    start=time.perf_counter();out,_=new_run({'output_root':str(ROOT/'datasets/local_runs'/EXPERIMENT)},EXPERIMENT,'post26_admission_readiness')
    shutil.copyfile(Path(__file__),out/'executed_source.py')
    independent_audit(out);uavdt_audit(out);training_matrix(out)
    save(out/'inputs.json',{'source':source(Path(__file__)),'training_data':source(out.parent/DATA_RUN/'SHA256SUMS.txt'),
        'seed_diagnostic':source(out.parent/SEED_RUN/'SHA256SUMS.txt')},out)
    write_atomic(out/'diary.md','# Admission and training readiness\n\nRead bounded source configs/indexes and original UAVDT metadata. Decoded fixed sequence examples without inference. Prepared, but did not execute, matched three-seed arms. Missing history, mappings, ranges and budgets remain explicit.\n',out)
    note=ROOT/'datasets/reportcn'/('vm36_supervisor_reproduction_独立性与训练准入_'+out.name.split('_')[-1]+'.md')
    body=['# VM36 独立性与训练准入','',
        'PROBLEM：test-dev独立性、UAVDT外部协议和RQ4匹配训练是否具备实际执行条件。',
        'WHY：文件不交叉不能代替历史选模/调参审计，数据全量解码也不能代替真实训练验收。',
        'SOLUTION：检查直接训练args、历史数据YAML及少量明确结果协议；核对50序列各首/中/末图；复用本轮6471图标注和CPU种子诊断。',
        'WHY IS IT A GOOD SOLUTION：记录直接证据、未知事项和未执行矩阵，不把无搜索命中当独立性证明。','',
        'test-dev仍DATASET_BLOCKED；三模型历史训练字节、选中epoch与完整既往使用记录不齐。',
        'UAVDT固定150张M帧几何检查与原converted元数据一致，原GT/ignore来源已有证据；但原图字节身份、car/van语义及历史使用未准入。没有跨数据集AP。',
        '6471图候选标签视图处理3条零面积行和4个重复框，原文件不改；16图增强检查显示标准启动同seed未完全复现，构建dataset后重新固定seed的隔离测试能够复现。生产trainer未改，不宣称GPU训练可复现。',
        'RQ4只准备12次主对照和15次三seed消融的完整方案；不是启动27次训练的指令。当前训练次数0，当前完整训练成本NOT_MEASURED。',
        '消融候选定义为全五种增强逐项移除，单独增强式消融也可能符合原文，必须在真实训练前确认；没有自行假定已获批准。',
        '新范围、方向、airlight、初始化SHA、生产loader/seed方案及有限训练预算仍未冻结。','',
        '证据：'+str(out),'复现：post26_readiness.py（CPU，不训练）']
    write_atomic(note,'\n'.join(body)+'\n',out);save(out/'REPORT_REFERENCE.json',source(note),out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'POST26_ADMISSION_READINESS','status':'COMPLETE_WITH_LIMITATIONS',
        'CPU_wall_seconds':time.perf_counter()-start,'new_GPU_work':0,'new_training_runs':0},out,replace=True)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>Post26 admission readiness</h1><a href="TESTDEV_EVIDENCE.json">Independent test</a> · <a href="UAVDT_ADMISSION_UPDATE.json">UAVDT</a> · <a href="RQ4_GATE.json">Training</a>',out)
    seal(out)
    from joint_analysis import refresh
    from matched_followup import ANALYSIS_ID
    refresh(out.parent/ANALYSIS_ID);print(out,flush=True)


if __name__=='__main__':execute()
