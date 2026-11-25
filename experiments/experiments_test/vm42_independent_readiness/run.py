"""CPU-only provenance review and frozen matched evaluation designs."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'vm37_yolo26_rq4'))
import study
ROOT,PUBLIC=study.ROOT,study.PUBLIC
EXPERIMENT=HERE.name;RUNS=ROOT/'datasets/local_runs'/EXPERIMENT
V40=ROOT/'datasets/local_runs/vm40_uvl_gap_independent_validation/vm40_uvl_gap_independent_validation_audit_20261116T011556_955084fb'
V41=ROOT/'datasets/local_runs/vm41_degradation_coverage/vm41_degradation_coverage_coverage_20261116T020508_dfa7879d'
P37=study.RUNS/'vm37_yolo26_rq4_pilot960_execution_20261114T110947_fe2f01b3'
P38=ROOT/'datasets/local_runs/vm38_variability_guided_augmentation/vm38_variability_guided_augmentation_targeted960_20261114T215310_e8bba12b'
P39=ROOT/'datasets/local_runs/vm39_multi_seed_confirmation/vm39_multi_seed_confirmation_confirmation_20261115T095834_9623ace0'
AUTH=study.RUNS/'vm37_yolo26_rq4_pilot10_authorization_cpu_20261113T212036_50286ea7'
MIGRATION=ROOT/'datasets/research_archive/migration_20261104'
SOURCES={};load=study.load


def source(file,role='INPUT'):
    file=Path(file).resolve(strict=True)
    r={'path':str(file),'SHA256':study.sha256(file),'bytes':file.stat().st_size,'role':role};SOURCES[str(file)]=r
    return r


def read(file):source(file);return load(file)


def save(file,value):
    with Path(file).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def csv_read(file):
    source(file)
    with Path(file).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def csv_write(file,values):
    fields=list(dict.fromkeys(k for row in values for k in row))
    with Path(file).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(values)


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def baseline(out):
    for item in read(V41/'checksums.json')['files']:
        if study.sha256(Path(item['path']))!=item['SHA256']:raise ValueError('Accepted VM41 input changed: '+item['path'])
    for name in ('VM40_实验配置与独立验证_科学报告_CN.md','VM41_研究缺口与下一步实验方案_CN.md'):
        source(ROOT/'datasets/reportcn'/name)
    for name in ('VM36_21_measured_conditions.csv','VM36_84_pair_metrics.csv','VM39_three_seed_summary.csv',
                 'MATCHED_56_configuration_metrics.csv','independent_data_admission.json','testdev_CPU_membership.csv'):
        source(V40/name)
    for name in ('fm_p.uvl','fm_d.uvl'):source(HERE.parent/'vm36_supervisor_reproduction/feature_models'/name,'FROZEN_UVL')
    pdf=read(V40/'PDF_pages_fresh.json')
    if source(pdf['source']['path'])['SHA256']!=pdf['source']['sha256']:raise ValueError('Supervisor PDF changed.')
    source(V41/'Future_Gap_protocol.json');source(V41/'Timing_protocol_comparison.csv')
    source(HERE.parent/'vm41_degradation_coverage/timing_protocol.py')
    for path in (P37/'execution_result.json',P38/'final_acceptance.json',P39/'final_acceptance.json',P39/'joint_result_index.json'):source(path)
    save(out/'Accepted_evidence_reuse.json',{'VM36_pairs':84,'VM36_P':21,'VM36_D':4,'VM36_images_per_pair':548,
        'VM39_matched_training_seeds':[42,43,44],'VM39_1280_Gaussian_AP_gain_pp':5.248423724802129,
        'VM39_1280_Normal_AP_change_pp':-.5736956464094362,
        'VM41_observed_leaves':18,'VM41_unmeasured_leaves':7,'VM41_SAT_CPU_candidates':32,'VM41_coordinate_safe':31,
        'scope':'Saved accepted evidence, not new measurements','new_AP':None,'new_GPU_evaluations':0,
        'no_accepted_decode_generator_or_test_suite_repeated':True})


def git_record(out,args,name):
    cmd=['git','-C',str(PUBLIC),*args]
    result=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
    file=out/'logs'/name;file.write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
    return {'command':cmd,'returncode':result.returncode,'output':source(file,'READ_ONLY_GIT_QUERY')}


def provenance(out):
    logs=out/'logs';logs.mkdir()
    git=[]
    git.append(git_record(out,['rev-parse','HEAD'],'git_head.txt'))
    git.append(git_record(out,['status','--short'],'git_status_before.txt'))
    git.append(git_record(out,['log','-30','--date=iso-strict','--format=%H %ad %s'],'git_recent_history.txt'))
    git.append(git_record(out,['log','--all','-40','-G','test.?dev|test_dev|testdev','--format=%H %ad %s','--date=iso-strict','--name-only','--','code','experiments','reports'],'git_testdev_history.txt'))
    # Inspect bounded tracked-source history; commits do not certify execution.
    git.append(git_record(out,['log','-8','-p','--format=%H %ad %s','--date=iso-strict','-G','test.?dev|test_dev|testdev','--',
        'experiments/experiments_test/vm36_supervisor_reproduction/historical_review.py','experiments/experiments_test/cv261207/admission.py',
        'experiments/experiments_test/cv261207/independent_admission.py'],'git_admission_patches.txt'))
    save(out/'Git_provenance_queries.json',git)
    old=read(V41/'Historical_testdev_records.json');usage=[];hits=[];read_records=[]
    for record in old['records']:
        file=Path(record['path']);proof=source(file,'HISTORICAL_DECLARATION')
        if proof['SHA256']!=record['SHA256']:raise ValueError('Historical direct record changed.')
        for i,line in enumerate(file.read_text(encoding='utf-8-sig').splitlines(),1):
            if re.search(r'test.?dev|test_dev|testdev',line,re.I):hits.append({'file':str(file),'line':i,'text':line,'classification':'REFERENCE_NOT_EXECUTION'})
        read_records.append(proof)
    models=csv_read(MIGRATION/'model_catalog.csv');source(MIGRATION/'checkpoint_index.csv');source(MIGRATION/'dataset_index.csv')
    history_rows=[];indexed=[];file_manifest=MIGRATION/'file_manifest.csv';source(file_manifest,'MIGRATION_FILE_INDEX_ONLY')
    with file_manifest.open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            text=row['source']+' '+row['destination']
            if re.search(r'test.?dev|test_dev|testdev',text,re.I):indexed.append(row)
    csv_write(out/'Indexed_testdev_paths.csv',indexed if indexed else [{'status':'NO_MATCH_IN_THIS_SAVED_INDEX; NOT_PROOF_OF_NON_USE'}])
    # Only small provenance records from the saved index are opened.
    inspect=[]
    for row in indexed:
        path=ROOT/row['destination']
        if path.suffix.lower() in ('.json','.yaml','.yml','.txt','.log','.md','.csv') and path.exists() and path.stat().st_size<=1024*1024:
            inspect.append((row,path))
    for row,path in inspect[:24]:
        proof=source(path,'INDEXED_TESTDEV_NAMED_RECORD')
        history_rows.append({'path':str(path),'recorded_SHA256':row['destination_sha256'],'actual_SHA256':proof['SHA256'],
            'identity_matches_migration':proof['SHA256']==row['destination_sha256'],
            'interpretation':'Filename/path evidence only; content or acceptance must establish a completed evaluation.'})
        lines=path.read_text(encoding='utf-8-sig',errors='replace').splitlines()
        for i,line in enumerate(lines,1):
            if re.search(r'test.?dev|test_dev|testdev|complete|validat|checkpoint',line,re.I):
                hits.append({'file':str(path),'line':i,'text':line[:1000],'classification':'INDEXED_CONTENT_REQUIRES_CONTEXT'})
                if sum(h['file']==str(path) for h in hits)>=12:break
    csv_write(out/'Indexed_record_verification.csv',history_rows if history_rows else [{'status':'NO_SMALL_CANDIDATE_IN_BOUNDED_INDEX'}])
    csv_write(out/'Historical_content_hits.csv',hits if hits else [{'status':'NO_MATCH_IN_BOUNDED_RECORDS; NOT_PROOF_OF_NON_USE'}])
    import yaml
    data_config=ROOT/'datasets/research_archive/paper1/source_configs/visdrone_10class.yaml'
    declared=yaml.safe_load(data_config.read_text(encoding='utf-8'));source(data_config)
    for ck in ('ck01_yolo26s_ft1280','ck05_yolov8s_ft1280','ck06_yolo11s_ft1280'):
        folder=ROOT/'models/checkpoints'/ck;args=folder/'args.yaml';source(args);values=yaml.safe_load(args.read_text(encoding='utf-8'))
        history_rows_cfg={'checkpoint':ck,'evidence_status':'CONFIRMED','training_data_config':values.get('data'),
            'declared_train':declared.get('train'),'declared_val':declared.get('val'),'train_resolution':values.get('imgsz'),
            'epochs_configured':values.get('epochs'),'seed':values.get('seed'),'selection':'Historical best.pt; full selection/manual history UNKNOWN',
            'testdev_gradient_use':'UNKNOWN_AT_CREATION_TIME; saved config declares train/val',
            'testdev_eval_or_tuning':'UNKNOWN','scene_independence':'UNKNOWN','source':str(args),
            'scope':'Config declaration is confirmed; execution-time dataset bytes are not reconstructed'}
        usage.append(history_rows_cfg)
    save(out/'Bounded_history_scope.json',{'git_queries':git,'direct_records':read_records,'indexed_testdev_path_count':len(indexed),
        'small_indexed_records_opened':len(history_rows),'small_record_limit':24,'indexed_raw_images_or_predictions_opened':False,
        'inventory_scope':'Saved migration file index and specified Git paths only; not all disks, all former environments or manual actions',
        'completed_testdev_evaluation_proved_by_these_records':False,'negative_keyword_search_used_as_independence':False})
    return usage


def checkpoints(out,usage):
    index=read(P39/'joint_result_index.json');models=[]
    for seed in (42,43,44):
        for strategy in ('CONTROL','VARIABILITY_GUIDED'):
            ref=next(r for r in index if r['seed']==seed and r['arm']==strategy)
            if seed==42:folder=(P37 if strategy=='CONTROL' else P38)/f'YOLO26s_seed42_{strategy}'
            else:folder=P39.parent/ref['source_RunID']/f'YOLO26s_seed{seed}_{strategy}'
            protocol=read(folder/'protocol.json');file=folder/'weights/epoch0010_fp32.pt'
            proof=source(file,'REHASHED_EXISTING_EPOCH10_EMA')
            if proof['SHA256']!=ref['checkpoint_SHA256']:raise ValueError('Existing final EMA SHA differs.')
            models.append({'seed':seed,'strategy':strategy,'checkpoint':proof,'initialization':protocol['initialization'],
                'training_protocol':source(folder/'protocol.json'),'fixed_selection':'EPOCH10_EMA','train_images':len(protocol['train']),
                'development_images':len(protocol['development']),'health_evidence':'Reuse accepted VM37/38/39 FP32 reload and tensor-health evidence; no tensor test rerun',
                'health_source':source(P39/'final_acceptance.json' if seed!=42 else P38/'final_acceptance.json' if strategy!='CONTROL' else P37/'execution_result.json')})
            usage.append({'checkpoint':f'YOLO26s_seed{seed}_{strategy}','evidence_status':'CONFIRMED',
                'training_data_config':str(folder/'protocol.json'),'declared_train':'Frozen6471 train manifest','declared_val':'Frozen548 val manifest',
                'train_resolution':960,'epochs_configured':10,'seed':seed,'selection':'FIXED_EPOCH10_EMA',
                'testdev_gradient_use':'DISPROVED_IN_PRESERVED_TRAINING_MANIFEST; original source-scene equivalence separately UNKNOWN',
                'testdev_eval_or_tuning':'UNKNOWN_COMPLETE_PROJECT_HISTORY; reviewed indexed runs use val only',
                'scene_independence':'UNKNOWN','source':str(folder/'protocol.json'),'scope':'No claim of complete historical project non-use'})
    for seed in (42,43,44):
        same=[r['initialization']['sha256'] for r in models if r['seed']==seed]
        if len(set(same))!=1:raise ValueError('Within-seed initialization mismatch.')
    broad_folder=P37/'YOLO26s_seed42_ALL_FIVE'
    broad=read(broad_folder/'protocol.json')
    usage.append({'checkpoint':'YOLO26s_seed42_ALL_FIVE','evidence_status':'CONFIRMED',
        'training_data_config':str(broad_folder/'protocol.json'),'declared_train':f'Frozen{len(broad["train"])} train manifest',
        'declared_val':f'Frozen{len(broad["development"])} val manifest','train_resolution':960,'epochs_configured':10,'seed':42,
        'selection':'FIXED_EPOCH10_EMA; context arm only, excluded from options A/B',
        'testdev_gradient_use':'DISPROVED_IN_PRESERVED_TRAINING_MANIFEST','testdev_eval_or_tuning':'UNKNOWN_COMPLETE_PROJECT_HISTORY',
        'scene_independence':'UNKNOWN','source':str(broad_folder/'protocol.json'),'scope':'Historical comparator; no extra seeds invented'})
    csv_write(out/'Historical_split_usage_matrix.csv',usage);save(out/'Frozen_checkpoints.json',models)
    return models


def admission(out):
    old=read(V41/'Independent_data_readiness.json');source(V41/'Scene_proxy_overlap.json');source(V40/'testdev_CPU_membership.csv')
    facts=[
      ('source_membership','CONFIRMED','VM40 verified1610 images and1610 source annotations; no decode repeated','VM40/testdev_CPU_membership.csv'),
      ('train_val_ID_or_byte_overlap_present','DISPROVED','No current train/val ID or byte-SHA intersection in accepted VM40 audit','VM40/independent_data_admission.json'),
      ('native10_GT_ignore_compatibility','CONFIRMED','Original VisDrone annotations permit existing native10/ignore-union semantics; actual conversion must retain exact member and code SHA','VM40/independent_data_admission.json'),
      ('old_checkpoint_creation_time_membership','UNKNOWN','Historical copied args declare train/val, not immutable creation-time image bytes','Historical_split_usage_matrix.csv'),
      ('testdev_previous_completed_evaluation','UNKNOWN','Bounded Git/index records cannot establish a complete historical or manual evaluation inventory','Bounded_history_scope.json'),
      ('testdev_performance_used_for_tuning','UNKNOWN','No complete model/policy selection-decision ledger; no search hit cannot prove non-use','Historical_content_hits.csv'),
      ('reliable_scene_video_flight_mapping','UNKNOWN','Filename-prefix proxies are not verified flight identities; no authoritative mapping established by saved records','VM41/Scene_proxy_overlap.json'),
      ('new_epoch10_testdev_gradient_inputs','DISPROVED','Preserved train6471 and val548 manifests contain no testdev input identities; no claim that all prior human exposure is excluded','Frozen_checkpoints.json'),
      ('VM42_testdev_prediction_or_AP','NOT_APPLICABLE','No GPU or new COCO evaluation in this task','Current task scope'),
      ('reuse_legacy_inference_ms_as_cached_pipeline','DISPROVED','VM40/41 established different timing boundaries','VM41/Timing_protocol_comparison.csv')]
    csv_write(out/'Testdev_provenance_audit.csv',[{'question':q,'status':s,'finding':f,'evidence':e} for q,s,f,e in facts])
    blockers=[
      {'issue':'HISTORY_SELECTION','status':'UNKNOWN','evidence_needed':'Checkpoint-specific creation-time data and selection logs, completed testdev evaluation inventory and any downstream tuning decisions',
       'action':'Supervisor decides whether the documented chain is sufficient for a prospective conditional holdout; otherwise retain non-confirmatory scope','owner':'Author / supervisor'},
      {'issue':'SCENE_ISOLATION','status':'UNKNOWN','evidence_needed':'Reliable sequence/flight/source-scene metadata plus review of overlap; filename-prefix absence is insufficient',
       'action':'Document genuine scene identity or state the limitation explicitly; do not infer independence','owner':'Dataset source / author'},
      {'issue':'GPU_SCOPE','status':'NOT_APPLICABLE','evidence_needed':'A separate explicit GPU task authorization naming complete option and budget','action':'No GPU command in VM42','owner':'Author'},
      {'issue':'COST_COMPLETENESS','status':'UNKNOWN','evidence_needed':'Complete-option cost incl data/GT/cache, model loading,200 paired bootstraps, verification and stop margin','action':'Use measured-component ledger and bounded future resource admission; no partial-arm start','owner':'Future executor'},
      {'issue':'THERMAL','status':'UNKNOWN','evidence_needed':'Renewed physical and device thermal admission; historical89C is not clearance','action':'Retain90C /87C30s stops and existing2s monitoring','owner':'Author / future supervisor'},
      {'issue':'MOTION_HAZE_BINDING','status':'UNKNOWN','evidence_needed':'Scientific approval of direction and airlight; CPU proposal values alone insufficient','action':'Keep8-cell branch proposed only','owner':'Supervisor'},
      {'issue':'TIMING_DETAIL_INTEGRATION','status':'UNKNOWN','evidence_needed':'Granular timing recorder integrated with real predictor without changing VM41 primary boundary','action':'Record preprocessing/infer/NMS diagnostics separately and validate later under an authorized task','owner':'Future executor'}]
    csv_write(out/'Scientific_blockers_and_actions.csv',blockers)
    decision={'status':'BLOCKED_INDEPENDENT_ADMISSION','prior_status':old['status'],'source':source(V41/'Independent_data_readiness.json'),
        'disjointness':'CONFIRMED_CURRENT_BYTES_IDS; not scene independence','historical_use':'UNKNOWN_COMPLETE_INVENTORY',
        'scene_grouping':'UNKNOWN','reclassification_as_independent':False,'new_prediction_count':0,
        'supervisor_memo':{'recommendation':'Do not admit confirmatory evaluation yet.',
            'decision_requested':['Supply/review checkpoint and method-selection/testdev-use evidence and scene separation',
                'If adequate evidence cannot be recovered, retain testdev as historically unresolved external evaluation, never relabel it independent',
                'Authorize a genuinely prospective compatible dataset plan separately if strict independence remains required; no dataset invented here'],
            'evidence_review_is_not_GPU_authorization':True,'stop_repeating_same_audit':'This bounded review closes with named missing evidence; reopen only when new records or a supervisor decision arrive'},
        'UAVDT':'Original DET/ignore/split provenance and native10 compatibility remain unresolved; no substitution.'}
    save(out/'Independent_admission_decision.json',decision)


def protocols(out,models):
    timing=read(V41/'Future_Gap_protocol.json');policy=read(P38/'policy.json')
    membership=source(V40/'testdev_CPU_membership.csv')
    members=csv_read(V40/'testdev_CPU_membership.csv')
    if len(members)!=1610 or len({r['image_id'] for r in members})!=1610:raise ValueError('Accepted membership is not1610 unique IDs.')
    selected=[{'image_id':r['image_id'],'source_image_SHA256':r['image_SHA256'],'annotation_SHA256':r['annotation_SHA256'],
        'width':int(r['width']),'height':int(r['height'])} for r in members]
    archive=read(V40/'historical_independence_evidence.json')['prior_dataset_gate']['splits']['test-dev']['archive']
    class_rows=csv_read(V40/'CONTROL_actual_training_args.csv')
    classes=json.loads(next(r['EXACT_VALUE'] for r in class_rows if r['parameter']=='class_order'))
    save(out/'Testdev_frozen_population.json',{'status':'CONDITIONAL_ON_INDEPENDENT_ADMISSION','images':selected,
        'source_ZIP_accepted_certificate':archive,'native10_class_order':classes,
        'members_source':membership,'selection':'ALL1610; no AP-based selection or subsampling','GT_conversion':'Do not change categories, ignore regions, original coordinates or accepted area semantics'})
    population=source(out/'Testdev_frozen_population.json','FROZEN_FUTURE_POPULATION')
    recipe_source=source(P38/'policy.json');cases={r['selected_leaves']:r for r in csv_read(V41/'Candidate_valid_configurations.csv')}
    normal=next(r for r in cases.values() if set(r['selected_leaves'].split('|'))=={'Normal'})
    gaussian=next(r for r in cases.values() if set(r['selected_leaves'].split('|'))=={'Normal','Gauss_s15'})
    result=[]
    for option,seeds,name in [('A',[42],'Pilot_4cell_frozen_protocol.json'),('B',[42,43,44],'ThreeSeed_12cell_frozen_protocol.json')]:
        cells=[]
        for seed in seeds:
            for condition,case in [('Normal',normal),('Gaussian_s15',gaussian)]:
                for strategy in ('CONTROL','VARIABILITY_GUIDED'):
                    identity=next(r for r in models if r['seed']==seed and r['strategy']==strategy)
                    cells.append({'cell_id':f's{seed}_{strategy}_{condition}_R1280','seed':seed,'strategy':strategy,
                        'condition':condition,'FM_P_id':case['configuration_id'],'resolution':1280,'images':1610,
                        'checkpoint':identity['checkpoint'],'initialization_SHA256':identity['initialization']['sha256'],
                        'P_attributes':json.loads(case['complete_renderer_parameters'])})
        spec={'id':'VM42_OPTION_'+option+'_V1','status':'FROZEN_DESIGN_CPU_ONLY_NOT_ADMITTED',
            'option':option,'training_seeds':seeds,'training':'Reuse existing YOLO26s native10 epoch10 EMA from960/10epoch matched training; no retraining',
            'cells':cells,'cells_count':len(cells),'images_per_cell':1610,'image_evaluations':len(cells)*1610,
            'dataset':'VisDrone2019-DET-test-dev ONLY IF admitted; no silent substitute','native10_class_order':classes,
            'population':population,'inference':timing['inference'],'timing_protocol':source(V41/'Future_Gap_protocol.json'),
            'timing_id':'VM41_MATCHED_CACHED_GAP_V1','GT_ignore_mapping':'Original native10 and unchanged accepted raster-union >=50% ignore; do not edit GT',
            'evaluation_code':{f:source(PUBLIC/'code/main'/f)['SHA256'] for f in ('labels.py','statistics.py')},
            'policy':recipe_source,'policy_parameters_retuned':False,'policy_probability':.25,'Gaussian_sigma':[5.5,30.],
            'condition_rng':{'master_seed':20261116,'formula':'SHA256(seed|image_stem|leaf), first8 bytes big-endian -> numpy.default_rng',
                'generator':source(HERE.parent/'cv261109/generator.py'),'image_bytes_shared_by_all_cells':True,'effect_before_Letterbox':True},
            'primary_endpoint':'GUIDED minus CONTROL Gaussian AP50:95 at1280',
            'primary_summary':'One seed42 matched difference' if option=='A' else 'Mean of three matched seed differences plus sample SD(ddof=1)',
            'secondary_endpoints':['Gaussian Tiny AP','Normal AP50:95 clean cost','same-protocol cached pipeline latency','preset Tiny/latency Gap status','AP50 and AR diagnostics'],
            'statistics':timing['statistics'],'gap_rules':timing['Gap'],
            'evidence_scope':'ONE_SEED_INDEPENDENT_PILOT_ONLY_IF_ADMITTED' if option=='A' else 'THREE_SEED_INDEPENDENT_CONFIRMATION_ONLY_IF_ADMITTED',
            'not_interchangeable_with_other_option':True,'new_results':'NOT_EVALUATED',
            'execution_requires':['Independent admission','Separate GPU authorization','Entire matched option cost and unchanged deadline','Fresh thermal gate','Trace granularity/backend integration checked'],
            'GPU_execution_authorized_by_VM42':False}
        save(out/name,spec);result.append(spec)
    csv_write(out/'Timing_protocol_reuse_audit.csv',[
        {'item':k,'status':'UNCHANGED','frozen_value':json.dumps(v),'source':str(V41/'Future_Gap_protocol.json')} for k,v in timing['inference'].items()]+[
        {'item':'primary_timing_boundary','status':'REUSED_CPU_INTERFACE_VALIDATED_BY_VM41','frozen_value':'cached input hash ->read/decode->pixel hash->synchronized predictor->extraction->write+flush','source':str(V41/'Timing_CPU_interface_tests.json')},
        {'item':'granular_stage_recording','status':'REQUIRED_FUTURE_INTEGRATION_NOT_GPU_TESTED','frozen_value':'Separate byte verify/read/decode/pixel verify/preprocess/forward/NMS+sync/extract/write, total worker; existing VM41 groups some sub-stages','source':str(HERE.parent/'vm41_degradation_coverage/timing_protocol.py')},
        {'item':'legacy_latency_conversion','status':'PROHIBITED','frozen_value':'Do not replace cached_pipeline_ms with VM37-39 inference_ms or sums of old means','source':str(V41/'Timing_protocol_comparison.csv')}])
    return result


def costs(out,options):
    import pilot10
    auth=read(AUTH/'authorization.json');charges=[]
    for path in pilot10.charge_files(AUTH):
        record=read(path);charges.append({'source':source(path),'seconds':record['charged_seconds']})
    remaining=pilot10.available(AUTH)
    index=read(P39/'joint_result_index.json');cost_rows=[];summaries=[]
    for option in options:
        phase={'prediction_wall':0.,'COCO':0.,'Bootstrap':0.};refs=[]
        for cell in option['cells']:
            original=next(r for r in index if r['seed']==cell['seed'] and r['arm']==cell['strategy'] and r['resolution']==1280 and r['context']==cell['condition'])
            trace_path=Path(original['source_folder'])/'trace.json';trace=read(trace_path)
            phase['prediction_wall']+=trace['inference_wall_seconds'];phase['COCO']+=trace['COCO_seconds'];phase['Bootstrap']+=trace['bootstrap_seconds']
            refs.append(source(trace_path,'SAVED_MEASURED_PHASE_COST'))
        ratio=1610/548
        matched_linear=sum(phase.values())*ratio
        broad_linear=10808.361848900095/(16*548)*option['image_evaluations']
        linear=max(matched_linear,broad_linear);envelope=linear*1.35+1800
        for component,measured,scope in [('data_GT_PNG_preparation',None,'Not separately measured for1610; no invented estimate'),
            ('model_loading',None,'Global initialization outside per-cell traces; included in old16-cell supervisor aggregate'),
            ('three_warmups_per_cell',None,'Legacy traces have no same three-warmup measurement'),
            ('complete_predictions_plus_decode_effects_outputs',phase['prediction_wall'],'Recorded inference_wall includes I/O/effects/output and source guards; not cached_pipeline latency'),
            ('saved_outputs_separately',None,'Included in prediction wall; do not double count or invent a split'),
            ('COCO',phase['COCO'],'Recorded CPU evaluation seconds'),
            ('paired_image_Bootstrap',phase['Bootstrap'],'Recorded200-resample CPU seconds; independent GT density may change cost'),
            ('verification',None,'Not isolated in all traces; future full verification still required')]:
            cost_rows.append({'option':option['option'],'component':component,'reference_images_per_cell':548,
                'reference_cells':len(option['cells']),'saved_measured_seconds':measured,
                'linear_extrapolated_1610_seconds':measured*ratio if measured is not None else None,
                'status':'RECOVERED_MEASUREMENT_PLUS_EXTRAPOLATION' if measured is not None else 'NOT_SEPARATELY_MEASURED',
                'scope':scope,'source_trace_manifest':str(out/'Cost_trace_sources.json')})
        cost_rows.append({'option':option['option'],'component':'safety_shutdown_margin','saved_measured_seconds':None,
            'linear_extrapolated_1610_seconds':1800,'status':'FIXED_PLANNING_RESERVE_NOT_MEASURED','scope':'Same1800-second stop reserve used in VM40/41 planning'})
        summaries.append({'option':option['option'],'cells':len(option['cells']),'images_per_cell':1610,
            'matched_trace_phase_sum_548_seconds':sum(phase.values()),'matched_trace_linear_1610_seconds':matched_linear,
            'old16_cell_supervisor_linear_1610_seconds':broad_linear,'selected_linear_planning_seconds':linear,
            'uncertainty_multiplier':1.35,'safety_margin_seconds':1800,'complete_planning_envelope_seconds':envelope,
            'remaining_seconds':remaining,'envelope_fits_balance':envelope<=remaining,
            'complete_cost_measured':False,'resource_admission':'NOT_ADMITTED_UNMEASURED_COMPONENTS_AND_OTHER_GATES',
            'interpretation':'Max of saved matched-phase and full16-cell extrapolations, then35% contingency+reserve; heuristic, not guarantee.',
            'trace_sources':refs})
    csv_write(out/'Matched_evaluation_costs.csv',cost_rows)
    save(out/'Cost_trace_sources.json',summaries)
    status={'status':'NOT_ADMITTED_NO_VM42_GPU_AUTHORIZATION','authorization':source(AUTH/'authorization.json'),
        'total_seconds':auth['max_cumulative_GPU_worker_wall_seconds'],'charged_seconds':sum(c['seconds'] for c in charges),
        'remaining_seconds':remaining,'absolute_deadline_UTC':auth['absolute_expiry_UTC'],'new_charge_seconds':0,
        'allocation_reset':False,'charges':charges,'option_cost_summaries':summaries,
        'temperature':{'historical_max_C':89,'current_clearance':'UNKNOWN_NOT_PERFORMED','renewed_admission_required':True,
            'immediate_stop_C':90,'sustained_stop_C':87,'sustained_seconds':30,'existing_monitor_seconds':2},
        'necessary_gates':{'dataset':'BLOCKED','future_scope_authorization':'ABSENT_THIS_TASK_CPU_ONLY',
            'complete_option_cost':'ESTIMATE_ONLY_NOT_MEASURED','thermal':'NOT_RENEWED',
            'expiry':'UNCHANGED; full option and shutdown must finish before this original deadline'},
        'no_partial_arm_start':True}
    save(out/'Thermal_resource_admission.json',status)
    return status


def motion_and_rule(out):
    cases=csv_read(V41/'Candidate_valid_configurations.csv');chosen=[]
    for leaves in ({'Normal'},{'Normal','Motion_k21'},{'Normal','Haze_dense'},{'Normal','Motion_k21','Haze_dense'}):
        r=next(c for c in cases if set(c['selected_leaves'].split('|'))==leaves)
        chosen.append({'FM_P_id':r['configuration_id'],'leaves':sorted(leaves),'parameters':json.loads(r['complete_renderer_parameters']),
            'SAT_and_pixel_evidence':'Reuse VM41 accepted validation','annotation_coordinate_safe':r['annotation_safe']=='True',
            'binding_status':r['scientific_preparation_status']})
    checkpoints=csv_read(V40/'VM36_4_detector_settings.csv')
    spec={'id':'VM42_MOTION_HAZE_PROPOSAL_V1','status':'PROPOSAL_NOT_APPROVED_OR_MEASURED',
        'P_conditions':chosen,'D_settings':[r for r in checkpoints if r['resolution']=='1280'],'cells':8,
        'dataset':'Same frozen548 development IDs if separately admitted; not independent testing',
        'DID':'AP(Motion+Haze)-AP(Motion)-AP(Haze)+AP(Normal), per checkpoint, same metric and same image indices',
        'candidate_bindings_not_supervisor_values':{'Motion_angle':'Uniform[0,180) historical CPU proposal','Haze_airlight':220},
        'RNG':'Reuse VM41 master_seed20261116 and per-image/per-leafSHA seed; reuse exact native pixels across D',
        'requires_review':['Motion direction binding','Haze airlight binding','physical assumptions','pixel/cache/GT/evaluator SHA','unchanged original coordinates and ignore semantics'],
        'RollingShutter':'EXCLUDED_UNVERIFIED_ANNOTATION_WARP','measured_AP':None,'GPU_authorized':False}
    save(out/'Motion_Haze_future_protocol.json',spec)
    evidence=csv_read(P38/'policy_selection_evidence.csv');source(P38/'policy.json')
    code=HERE.parent/'vm38_variability_guided_augmentation/run.py';source(code,'EXISTING_SELECTION_IMPLEMENTATION')
    means={c:sum(float(r['absolute_loss']) for r in evidence if r['context']==c)/2 for c in ('JPEG_Q25','JPEG_Q75','Gaussian_s15')}
    if max(means,key=means.get)!='Gaussian_s15':raise ValueError('Saved selection rationale differs.')
    save(out/'Selection_rule_evidence.json',{'mean_CONTROL_AP95_losses':means,'candidate_contexts':list(means),
        'measured_degradation_families':['JPEG','Gaussian'],'not_scored_families':['MotionBlur','Haze','Exposure','Rain'],
        'actual_code':'baseline_losses computes three context means and asserts Gaussian wins; recipe() then returns a fixed Gaussian policy.',
        'general_controller_implemented':False,'independent_generalization':'NOT_ESTABLISHED',
        'p025_optimized':False,'new_policy_trained':False})
    (out/'Variability_selection_rule_audit.md').write_text('''# Selection rule audit

The existing implementation is a reproducible development-feedback selection check followed by a fixed Gaussian recipe. `baseline_losses` ranks three measured contexts and asserts that Gaussian_s15 is highest. `recipe()` returns Gaussian p=0.25; it does not return arbitrary family-specific policies.

JPEG25, JPEG75 and Gaussian15 are three contexts from two degradation families. Motion/Haze/Exposure/Rain have no comparable baseline scores in this selection step. Missing scores are not zero losses.

## Evidence levels

- Fixed Gaussian augmentation: implemented and evaluated on development data.
- Development-feedback family selection: saved losses and executable rationale guard exist.
- General automated variability controller: not implemented or validated.
- Independent generalization: admission blocked; not demonstrated.

## Prospective interpretable rule

Input: FM_P condition/family identities, measured responses under a common D and dataset protocol, and fixed Tiny/latency constraints.

1. Exclude unsupported geometry, invalid FM_P selections and unapproved generator bindings.
2. Require baseline plus matched response support for at least two degradation families. Preserve unmeasured families as UNKNOWN.
3. Compute predefined weighted Normal-minus-condition AP losses within each family. Freeze condition/resolution weights before any held-out response is queried.
4. Rank positive measured family losses, report support and uncertainty, and emit a decision rationale. If required Tiny/latency evidence is absent, abstain from any Gap-driven claim.
5. Map a selected family only to an already approved frozen policy. If none exists, return POLICY_NOT_DEFINED rather than invent probabilities.
6. Freeze the output policy before a prospectively admitted evaluation. Do not tune it on that evaluation.

This is a proposal, not a newly trained policy or a validated adaptive algorithm. Evidence across multiple families, detector states and held-out environments is required before claiming general cross-environment adaptation. The current Gaussian probability and exclusion semantics remain unchanged.
''',encoding='utf-8')


def validate_design(out,option):
    name='Pilot_4cell_frozen_protocol.json' if option=='A' else 'ThreeSeed_12cell_frozen_protocol.json'
    spec=load(out/name);cells=spec['cells'];population=load(Path(spec['population']['path']))
    expected=4 if option=='A' else 12
    if len(cells)!=expected or len({r['cell_id'] for r in cells})!=expected:raise ValueError('Cell count/identity invalid.')
    if len(population['images'])!=1610 or len({r['image_id'] for r in population['images']})!=1610:raise ValueError('Population differs.')
    for c in cells:
        if c['images']!=1610 or c['resolution']!=1280:raise ValueError('Silent sample/resolution change.')
        if not Path(c['checkpoint']['path']).is_file():raise FileNotFoundError(c['checkpoint']['path'])
    if spec['timing_id']!='VM41_MATCHED_CACHED_GAP_V1' or spec['inference']['precision']!='FP32':raise ValueError('Timing or precision changed.')
    actual=[(c['seed'],c['strategy'],c['condition']) for c in cells]
    required={(s,a,p) for s in spec['training_seeds'] for a in ('CONTROL','VARIABILITY_GUIDED') for p in ('Normal','Gaussian_s15')}
    if set(actual)!=required:raise ValueError('Unmatched cells.')
    if spec['GPU_execution_authorized_by_VM42']:raise ValueError('VM42 cannot authorize GPU work.')
    return {'option':option,'structural_design':'VALID','cells':expected,'images_per_cell':1610,
        'scientific_admission':'BLOCKED','GPU_executed':False,'future_backend_note':'Frozen executable manifest validator; real testdev GT/cache/predictor instrumentation integration still needs an authorized admission stage.'}


def finish(out,start):
    results=[validate_design(out,o) for o in ('A','B')]
    save(out/'Design_validation.json',{'status':'PASS_STRUCTURAL_ONLY','options':results,'GPU':0})
    reporter=module('vm42_reporting',HERE/'reporting.py');names=reporter.figures(out)
    report_paths=reporter.reports(out,ROOT,V40,V41)
    reporter.page(out,names,report_paths)
    for r in SOURCES.values():
        if study.sha256(Path(r['path']))!=r['SHA256']:raise ValueError('Input changed during audit: '+r['path'])
    for relative,digest in read(out/'source_SHA256.json').items():
        if relative.startswith('experiments/'+EXPERIMENT+'/'):continue
        if study.sha256(PUBLIC/relative)!=digest:raise ValueError('Preexisting public source changed: '+relative)
    for file in HERE.glob('*.py'):source(file,'VM42_CODE')
    save(out/'Source_manifest.json',{'files':list(SOURCES.values()),'historical_changes':False,'accepted_raw_predictions_opened':False})
    (out/'diary.md').write_text('# VM42\n\n复用VM36–41，不重复解码或推理。补查有限Git历史、迁移文件索引、checkpoint和训练配置。\n\n'
        '分别冻结seed42四格与三seed十二格，不混淆证据。复用VM41主计时边界，细分计时接入仍需未来实测。\n\n'
        '独立历史/场景证据仍不足，形成具体导师决策清单。所有GPU关卡保持关闭。Gaussian策略是固定策略加开发反馈校验，不是通用自动控制器。\n',encoding='utf-8')
    resource={'CPU_wall_seconds':time.perf_counter()-start,'GPU_seconds':0,'new_training':0,'new_predictions':0,'new_COCO_evaluations':0,
        'full_image_decodes':0,'accepted_CPU_tests_repeated':False}
    save(out/'Resource_usage.json',resource)
    run=load(out/'run.json');run.update(status='COMPLETE_WITH_LIMITATIONS',GPU_seconds=0,
        independent_admission='BLOCKED',finished_UTC=datetime.now(timezone.utc).isoformat())
    (out/'run.json').write_text(json.dumps(run,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save(out/'final_acceptance.json',{'status':'CPU_READINESS_COMPLETE_WITH_LIMITATIONS','RunID':out.name,'new_GPU_evaluations':0,
        'independent_test':'BLOCKED_PENDING_NAMED_EVIDENCE','design_A_cells':4,'design_B_cells':12,'reports':[str(p) for p in report_paths]})
    files=[p for p in out.rglob('*') if p.is_file() and 'runtime' not in p.relative_to(out).parts]+report_paths+list(HERE.glob('*.py'))
    with (out/'SHA256SUMS.txt').open('x',encoding='utf-8') as f:
        for path in sorted(files):f.write(study.sha256(path)+'  '+str(path)+'\n')
    from main.visualization import build_run_index
    build_run_index(out.parent,EXPERIMENT)
    print('COMPLETE_WITH_LIMITATIONS',out,flush=True)


def resolve(identity):
    if Path(identity).name!=identity or not identity.startswith(EXPERIMENT+'_'):raise ValueError('Invalid RunID.')
    return (RUNS/identity).resolve(strict=True)


def audit():
    from main.pipeline import new_run
    out,logger=new_run({'output_root':str(RUNS)},EXPERIMENT,'readiness');study.runtime(out,cuda=False);start=time.perf_counter()
    try:
        baseline(out);usage=provenance(out);logger.info('Bounded history review complete.')
        models=checkpoints(out,usage);admission(out);options=protocols(out,models)
        costs(out,options);motion_and_rule(out);finish(out,start)
    except BaseException as exc:
        save(out/'failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'GPU_seconds':0})
        logger.exception('CPU audit stopped.');raise


def verify(out):
    for line in (out/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
        expected,file=line.split('  ',1)
        if study.sha256(Path(file))!=expected:raise ValueError('SHA differs: '+file)
    print('VERIFIED',out.name,'GPU=0',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='CPU-only. No GPU execution path.')
    choice=parser.add_mutually_exclusive_group(required=True);choice.add_argument('--audit',action='store_true')
    choice.add_argument('--verify');choice.add_argument('--validate-design',metavar='RunID')
    parser.add_argument('--option',choices=('A','B'),default='A');args=parser.parse_args()
    if args.audit:audit()
    elif args.verify:verify(resolve(args.verify))
    else:
        print(json.dumps(validate_design(resolve(args.validate_design),args.option),indent=2))
        raise SystemExit(2)
