"""Read direct checkpoint metadata and the recovered official UAVDT toolkit."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
from zipfile import ZipFile
from protocol import ROOT,load,save,source,sha256
from historical_review import table,csv_file
from post80_replay import run_path,ANALYSIS
from post80_science import READINESS


def audit(out):
    os.environ.update(CUDA_VISIBLE_DEVICES='',YOLO_AUTOINSTALL='false')
    import torch
    torch.set_num_threads(2)
    design=load(run_path(ANALYSIS)/'MEASURED_DESIGN.json');rows=[];direct=[]
    for ck in ('ck01','ck05'):
        spec=next(s for s in design['detector_settings'].values() if s['checkpoint']==ck);path=Path(spec['path'])
        if sha256(path)!=spec['sha256']:raise ValueError('Checkpoint identity changed.')
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        args=checkpoint.get('train_args') or {};results=checkpoint.get('train_results') or {}
        dimensions={k:len(v) for k,v in results.items() if isinstance(v,(list,tuple))} if isinstance(results,dict) else {}
        history=table(path.parent/'results.csv')
        epoch_values=next((v for k,v in results.items() if k.strip()=='epoch'),[]) if isinstance(results,dict) else []
        direct.append({'checkpoint':ck,'SHA256':spec['sha256'],'checkpoint_epoch_field':checkpoint.get('epoch'),
            'checkpoint_version':checkpoint.get('version'),'checkpoint_date':checkpoint.get('date'),
            'saved_training_result_column_lengths':dimensions,'saved_training_result_epoch_tail':epoch_values[-3:] if isinstance(epoch_values,list) else None,
            'saved_training_args':{k:args.get(k) for k in ('model','data','imgsz','epochs','seed','optimizer','batch','nbs','lr0','val','patience','amp','workers','deterministic')},
            'selected_epoch_status':'UNKNOWN_STRIPPED_OR_UNVERIFIED_SAVE_SEMANTICS' if checkpoint.get('epoch',-1)<0 else 'DIRECT_EPOCH_FIELD_AVAILABLE',
            'interpretation':'Saved curve length is recorded, not promoted to selected-best epoch without verified save/strip provenance.'})
        rows.append({'checkpoint':ck,'checkpoint_SHA256':spec['sha256'],'direct_epoch_field':checkpoint.get('epoch'),
            'recorded_version':checkpoint.get('version'),'training_curve_rows':len(history),
            'curve_last_epoch':next((v for k,v in history[-1].items() if k.strip()=='epoch'),None),
            'args_source':str(path.parent/'args.yaml'),'args_SHA256':sha256(path.parent/'args.yaml'),
            'curve_source':str(path.parent/'results.csv'),'curve_SHA256':sha256(path.parent/'results.csv'),
            'creation_time_data_bytes':'UNKNOWN','testdev_selection_and_method_history':'UNKNOWN','admission':'DATASET_BLOCKED'})
        del checkpoint;gc.collect()
    csv_file(out/'CHECKPOINT_HISTORY_EVIDENCE.csv',rows)
    save(out/'CHECKPOINT_DIRECT_METADATA.json',direct,out)
    old=load(run_path(READINESS)/'TESTDEV_EVIDENCE.json')
    verified=[]
    for r in old['direct_saved_sources']:
        record=r['file'];actual=sha256(Path(record['path']))
        if actual!=record['sha256']:raise ValueError('Historical source config changed.')
        verified.append({'source':record,'current_SHA256':actual,'evidence_kind':'Saved config/protocol, not proof of a completed evaluation or absence of other use.'})
    save(out/'INDEPENDENT_ADMISSION_UPDATE.json',{'status':'DATASET_BLOCKED','official_testdev_images':1610,
        'dataset_disjointness':'Previously verified current membership/bytes; not creation-time provenance.',
        'current_checkpoint_identity':'VERIFIED','checkpoint_history':rows,'direct_sources_rechecked':verified,
        'unknowns':['Creation-time image membership','Complete checkpoint-selection decisions','Manual test-dev evaluations and method tuning'],
        'negative_search_hits_used_as_proof':False,
        'independent_plan':'Recover checkpoint-specific immutable logs and selection inventory, or prospectively admit new native10 scenes with scene separation, labels, rights and hypotheses frozen before prediction/metric inspection. No silent dataset substitution.'},out)
    previous=load(run_path(READINESS)/'UAVDT_ADMISSION_UPDATE.json')
    origin=load(Path(previous['original_toolkit']['path']));archive=Path(origin['local_archive']['path'])
    if sha256(archive)!=origin['local_archive']['sha256']:raise ValueError('Official toolkit archive changed.')
    with ZipFile(archive) as z:
        names=z.namelist();member='UAV-benchmark-MOTD_v1.0/README.md';raw=z.read(member)
        candidates=[n for n in names if ('attribute' in n.lower() or '/attr/' in n.lower() or 'weather' in n.lower()) and not n.endswith('/')]
        reference_files=[n for n in names if n.endswith(('CalculateDetectionsPR.m','CalculateDetectionPR_overall.m'))]
    save(out/'UAVDT_PROTOCOL_UPDATE.json',{'status':'NOT_ADMITTED_FOR_INDEPENDENT_EVALUATION',
        'source_archive':source(archive),'official_README_member':member,'README_SHA256':hashlib.sha256(raw).hexdigest(),
        'official_DET_format':'frame,id,left,top,width,height,out-of-view,occlusion,category;1 car,2 truck,3 bus.',
        'official_ignore_and_AP':previous['ignore_and_metric'],'evaluator_members':reference_files,
        'condition_metadata_filename_candidates_in_this_toolkit':candidates,
        'condition_labels':'NOT_VERIFIED; a negative filename query in this toolkit is not proof that no official attributes exist elsewhere.',
        'conditions_to_UVL':'Night/fog/altitude categories cannot determine exact gain, beta or downscale numeric values without calibration.',
        'image_byte_provenance':previous['image_byte_provenance'],'class_mapping':previous['category_semantics'],
        'SOT':'Kept separate from DET/MOT','prior_model_exposure':previous['prior_model_exposure'],
        'redistribution_permission':'NOT_INFERRED_FROM_TOOLKIT_CODE_LICENSE','new_external_AP':None},out)


def training_update(out,loader):
    previous=run_path(READINESS);accepted=load(loader/'LOADER_ACCEPTANCE.json')
    save(out/'RQ4_READINESS_UPDATE.json',{'status':'TRAINING_NOT_READY','CPU_loader_adapter':source(loader/'SHA256SUMS.txt'),
        'full_library_labels_verified':accepted['full_library_images'],'targets':accepted['native10_targets'],
        'actual_CPU_NCHW':[2,3,1280,1280],'anchors':16,'repeats':2,'repeat_exact':accepted['repeat_exact'],
        'original_images_copied':0,'originals_unchanged':accepted['originals_unchanged'],
        'label_policy':'Three zero-area rows excluded and four duplicate positive boxes removed in a private derived view only.',
        'training_budget':'RESOURCE_BLOCKED; previous54-pair inference approval cannot authorize training.',
        'five_augmentation_families':['NoiseAug','MotionBlurAug','ExposureAug','JPEGAug','HazeAug'],
        'ranges':load(previous/'AUGMENTATION_RANGE_GATE.json'),
        'full_multi_seed_matrix':source(previous/'MATCHED_TRAINING_MATRIX_PROPOSAL.csv'),
        'required_before_canary':['Freeze one healthy initialization SHA per architecture and per-seed native10 initialization',
            'Approve training-only ranges, exact test-value exclusion and missing motion/haze choices',
            'Validate target photometric effects before geometric resize/augmentation with unchanged GT',
            'Freeze actual optimizer accumulation, schedule, EMA, checkpoint selection and new finite compute allocation'],
        'minimal_matched_pilot':{'models':['YOLOv8s','YOLO26s'],'arms':['DEFAULT_CONTROL','ALL_FIVE_TARGETED'],
            'runs':4,'seed_candidate':42,'seed_status':'PROPOSED','initialization_SHA256':'NOT_FROZEN',
            'epochs':'NOT_FROZEN_FOR_PILOT; retain matched schedule and choose only after measured canary cost.',
            'main_study_schedule_candidate':100,'main_study_seeds':3,'evaluation':'Same prospectively frozen P/D conditions, class and size protocol; clean and corrupted outcomes.',
            'purpose':'Feasibility/exploration, not a replacement for three comparable seeds and per-family ablations.'},
        'current_GPU_memory_and_full_run_runtime':'NOT_MEASURED_FOR_THIS_TRAINING_PROTOCOL',
        'new_training_runs':0,'production_trainer_integration':'NOT_VERIFIED'},out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--science-run',required=True);p.add_argument('--loader-run',required=True);a=p.parse_args()
    out=run_path(a.science_run)
    if (out/'SHA256SUMS.txt').exists():raise ValueError('An accepted science run is read-only.')
    audit(out);training_update(out,run_path(a.loader_run))
