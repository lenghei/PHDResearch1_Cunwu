"""Freeze VM46 conditions and rules; perform CPU admission without GPU work."""
import argparse
import ast
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.update(CUDA_VISIBLE_DEVICES='-1', PYTHONDONTWRITEBYTECODE='1', YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false')
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
sys.path.insert(0,str(ROOT/'main/code'))
sys.path.insert(0,str(HERE.parent/'vm43_external_pilot'))
sys.path.insert(0,str(HERE.parent/'vm44_tiny_error_variability_decision'))
from main.pipeline import new_run
from selector import legal, configuration_id
import vm43_common as accepted

RUNS=ROOT/'datasets/local_runs'/HERE.name
VM45=ROOT/'datasets/local_runs/vm45_motion_haze_gap_learning'
SOURCE=VM45/'vm45_motion_haze_gap_learning_execution_20261117T172830_0d9cb1ee'
RULE_SOURCE=VM45/'vm45_motion_haze_gap_learning_cpu_study_20261117T110831_36b2fc57'
V44=ROOT/'datasets/local_runs/vm44_tiny_error_variability_decision/vm44_tiny_error_variability_decision_analysis_20261117T031628_9ebff202'
V41=ROOT/'datasets/local_runs/vm41_degradation_coverage/vm41_degradation_coverage_coverage_20261116T020508_dfa7879d'
V37=ROOT/'datasets/local_runs/vm37_yolo26_rq4'
UVL=HERE.parent/'vm36_supervisor_reproduction/feature_models'
SOURCES={}
PLANS=[('Normal',['Normal']),('Gaussian_s15',['Normal','Gauss_s15']),('JPEG_q25',['Normal','JPEG_q25']),
       ('Gaussian_s15_JPEG_q25',['Normal','Gauss_s15','JPEG_q25']),('Defocus_r6',['Normal','Defocus_r6']),
       ('Downscale_x2',['Normal','Downscale_x2']),('Underexposed_mild',['Underexposed_mild']),('LowLight_Shot_low',['LowLight','Shot_low'])]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def source(path):
    p=Path(path).resolve(strict=True)
    r={'path':str(p),'SHA256':sha(p),'bytes':p.stat().st_size}
    SOURCES[str(p)]=r
    return r


def read(path):
    source(path)
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,value):
    if path.exists():raise FileExistsError(path)
    tmp=path.with_name(path.name+'.partial')
    with tmp.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.flush();os.fsync(f.fileno())
    tmp.rename(path)


def table(path,rows,fields=None):
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields or list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)


def seal_match(path,seal):
    entries={}
    for line in Path(seal).read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1);p=Path(name)
        entries[str(p.resolve() if p.is_absolute() else (Path(seal).parent/p).resolve())]=digest
    assert sha(path)==entries[str(Path(path).resolve())],str(path)
    return source(path)


def attributes(path):
    import re
    result={}
    for name,body in re.findall(r'^\s*(\w+)\s*\{([^{}]*)\}',path.read_text(encoding='utf-8'),re.M):
        try:result[name]={key:float(value) for key,value in (v.strip().split() for v in body.split(','))}
        except ValueError:continue
    return result


def freeze(out):
    seal_match(SOURCE/'experiment_protocol.json',SOURCE/'SHA256SUMS.txt')
    prior=read(SOURCE/'experiment_protocol.json')
    assert read(SOURCE/'final_acceptance.json')['verified_cells']==8
    assert sha(UVL/'fm_p.uvl')=='e6eadd633d1c14e0d6bc425f5e54421d592d766c65a6e5f20c61bc34dd76fb5c'
    attrs=attributes(UVL/'fm_p.uvl');source(UVL/'fm_p.uvl');source(UVL/'fm_d.uvl')
    old_protocol=ROOT/'datasets/local_runs/vm36_supervisor_reproduction/vm36_supervisor_reproduction_joint18_4h_20261110T002121_3ee111ef/EXECUTION_PROTOCOL.json'
    seed=read(old_protocol)['seed'];assert seed==20261108
    configs=[{'ordinal':i,'condition':name,'P_id':configuration_id(leaves),'leaves':leaves,'attributes':{f:attrs[f] for f in leaves}} for i,(name,leaves) in enumerate(PLANS)]
    protocol={**{k:prior[k] for k in ('models','GT','population','native10','resolution','FP32','TF32','batch','head','conf','NMS_IoU','max_det','max_nms','warmups_per_cell','timing_protocol')},
        'id':'VM46_GAUSSIAN_JPEG_HELDOUT_V1','conditions':configs,'images_per_cell':548,'cells':16,'formal_inferences':8768,
        'image_seed':seed,'seed_source':source(old_protocol),'bootstrap_seed':20261117,'bootstrap_samples':200,
        'aliases':{'Gaussian_s15':'Original UVL Gauss_s15; no new feature'},
        'order':'Listed P order, each ck05 then ck01; fixed before outcomes',
        'stage_order':['atmosphere','illumination','blur','rolling_shutter','noise','downscale','compression'],
        'DID':'Gaussian_JPEG-Gaussian-JPEG+Normal','cross_detector_DID':'ck01 minus ck05',
        'statistics':'200 common image-index pooled-COCO bootstrap draws; pointwise percentile95%; no image-AP averaging or seed significance',
        'scope':'DEVELOPMENT_ONLY_SYNTHETIC_CONFIGURATION_HOLDOUT','independent_images':False,
        'task':{'Tiny_AP50_95_min':.10,'cached_pipeline_mean_ms_max':100,'interval_rule':'Accepted VM44 selector.state; NOT_EVALUATED is distinct from UNCERTAIN'},
        'retries':0,'training':False,'full_design_must_not_be_reduced':True,'rule_promotion':False,
        'temperature':{'immediate_C':90,'sustained_C':87,'sustained_seconds':30,'maximum_monitor_interval_seconds':2}}
    save(out/'protocol_freeze.json',protocol)
    table(out/'Frozen_configurations.csv',[{'P_id':c['P_id'],'condition':c['condition'],'D':d['D'],'leaves':json.dumps(c['leaves']),
          'parameters':json.dumps(c['attributes'],sort_keys=True),'image_seed':seed,'images':548,'resolution':1280,'precision':'FP32','batch':1,
          'checkpoint_SHA256':d['checkpoint']['sha256'],'status':'FROZEN_NOT_MEASURED'} for c in configs for d in protocol['models']])
    tree=RULE_SOURCE/'in_sample_tree.txt';seal_match(tree,RULE_SOURCE/'SHA256SUMS.txt')
    expected=['|--- Normal <= 0.50','|   |--- class: 1','|--- Normal >  0.50','|   |--- D_ck05_Infer640 <= 0.50','|   |   |--- class: 0','|   |--- D_ck05_Infer640 >  0.50','|   |   |--- class: 1']
    assert tree.read_text().splitlines()==expected
    source(RULE_SOURCE/'learned_failure_rules.csv')
    rule={'id':'VM45_SHALLOW84_TEXT_REPLAY_V1','source':source(tree),'features':'Original FM_P leaf indicators and D indicators',
          'decision':'FAIL if !Normal or D_ck05_Infer640, otherwise PASS; Normal is illumination leaf',
          'training_population':'Original84 development rows','not_refitted':True,'prediction_is_deployment_authorization':False,
          'safe_controller':'ABSTAIN for every unadmitted prediction','task':protocol['task'],
          'UPDATED90_TREE_RF':'NOT_REPLAYABLE_AS_SAVED_ESTIMATORS; VM45 saved OOF results but no final estimators. No reconstruction/refit.',
          'frozen_UTC':datetime.now(timezone.utc).isoformat(),'future_outcomes_accessed':0}
    save(out/'frozen_rule_protocol.json',rule)
    save(out/'freeze_receipt.json',{'UTC':rule['frozen_UTC'],'configuration_SHA256':sha(out/'Frozen_configurations.csv'),
         'protocol_SHA256':sha(out/'protocol_freeze.json'),'rule_SHA256':sha(out/'frozen_rule_protocol.json'),'candidate_selection':'Current user fixed design; no AP substitutions'})
    return protocol


def cpu_check(out,spec):
    import cv2,numpy as np
    accepted_rules=read(V44/'FM_P_to_FM_D_decision_rules.json');cnf=accepted_rules['models']['fm_p']
    recipes=read(V41/'Candidate_recipes.json');checks=[]
    for c in spec['conditions']:
        assert legal(cnf,c['leaves'])
        existing=[r for r in recipes if set(r['leaves'])==set(c['leaves'])]
        if existing:assert existing[0]['render_attributes']==c['attributes']
        checks.append({'condition':c['condition'],'P_id':c['P_id'],'SAT_valid':True,'numeric_parameters_from_original_UVL':True,
                       'VM41_exact_recipe':existing[0]['id'] if existing else 'New composition of accepted leaf renderers',
                       'GT_coordinates':'Identity; unchanged original dimensions and object/ignore membership'})
    assert not legal(cnf,['LowLight']) and not legal(cnf,['Normal','UNREGISTERED_FEATURE'])
    payload=read(spec['GT']['path']);assert sha(spec['GT']['path'])==spec['GT']['SHA256'] and len(payload['records'])==548
    for d in spec['models']:assert sha(d['checkpoint']['path'])==d['checkpoint']['sha256'];source(d['checkpoint']['path'])
    s=importlib.util.spec_from_file_location('vm46_accepted_backend',HERE.parent/'vm43_external_pilot/backend.py')
    backend=importlib.util.module_from_spec(s);s.loader.exec_module(backend)
    ops=backend.renderer();cv2.setNumThreads(2)
    source(HERE.parent/'cv261109/generator.py');source(HERE.parent/'cv261109/model.py');source(HERE.parent/'vm43_external_pilot/backend.py')
    archive=backend.OriginalZIP(spec['population'],spec['population']['source_ZIP_accepted_certificate']);source(archive.path)
    fixtures=[];preview=[];reference=[]
    try:
        for i in range(6):
            image,record=archive.record(i,True);original=payload['records'][i]
            assert all(record[k]==original[k] for k in ('id','stem','width','height','objects','ignore_regions','other_regions'))
            for c in spec['conditions']:
                pix,trace=ops.generate(image,c['leaves'],c['attributes'],record['stem'],spec['image_seed'])
                again,trace2=ops.generate(image,c['leaves'],c['attributes'],record['stem'],spec['image_seed'])
                assert np.array_equal(pix,again) and trace==trace2 and pix.shape==image.shape and pix.dtype==np.uint8
                ok,png=cv2.imencode('.png',pix);assert ok
                assert np.array_equal(cv2.imdecode(png,cv2.IMREAD_COLOR),pix)
                pixel_sha=hashlib.sha256(pix.tobytes()).hexdigest()
                fixtures.append({'condition':c['condition'],'P_id':c['P_id'],'image_id':record['id'],'pixel_SHA256':pixel_sha,
                      'PNG_bytes_SHA256':hashlib.sha256(png.tobytes()).hexdigest(),'PNG_file_saved':False,'shape':list(pix.shape),
                      'repeatability':True,'effects':trace,'GT_and_ignore_unchanged':True})
                if i==0:
                    tile=cv2.resize(pix,(360,203));cv2.putText(tile,c['condition'],(5,22),cv2.FONT_HERSHEY_SIMPLEX,.48,(0,0,255),1,cv2.LINE_AA);preview.append(tile)
    finally:archive.close()
    # Existing manifests verify the six historical control pixels, without reading AP.
    with (SOURCE/'development_canonical_index.csv').open(encoding='utf-8-sig') as f:old_rows=list(csv.DictReader(f))
    for c in spec['conditions']:
        old=next((r for r in old_rows if r['condition_id']==c['P_id'] and r['D']=='ck01_Infer1280'),None)
        if old:
            path=Path(old['source_folder'])/'image_manifest.jsonl';source(path)
            with path.open(encoding='utf-8') as f:head=[json.loads(next(f)) for _ in range(6)]
            for row in head:
                expected=row.get('pixel_SHA256',row.get('pixel_sha256'))
                actual=next(r['pixel_SHA256'] for r in fixtures if r['P_id']==c['P_id'] and r['image_id']==row['image_id'])
                assert actual==expected,(c['condition'],row['image_id'])
            reference.append({'condition':c['condition'],'first6_pixels':'EXACT_HISTORICAL_MATCH','source':source(path)})
    (out/'plots').mkdir();cv2.imwrite(str(out/'plots/CPU_pixel_preview.png'),np.vstack([np.hstack(preview[:4]),np.hstack(preview[4:])]))
    table(out/'CPU_configuration_checks.csv',checks)
    save(out/'CPU_pixel_fixtures.json',{'checks':fixtures,'OpenCV':cv2.__version__,'numpy':np.__version__,'decode_flag':129,
         'scope':'6 fixed source images x8 conditions; not a548-image cache or detector preflight','historical_pixel_controls':reference})
    indices=np.random.default_rng(spec['bootstrap_seed']).integers(0,548,(200,548),dtype=np.int32)
    np.save(out/'frozen_bootstrap_indices.npy',indices,allow_pickle=False)
    save(out/'CPU_readiness.json',{'status':'PROPOSAL_CPU_PASS','SAT':8,'real_image_pixel_checks':48,'negative_SAT_checks':2,
        'population':'Accepted548 sourceZIP/GT hashes reverified; not a new independent dataset','full548_pixel_cache':'NOT_GENERATED_GPU_BRANCH_BLOCKED',
        'checkpoint_health':'Reused accepted VM45 full-tensor/real-inference health with identical current SHA; no new model loading',
        'GPU_preflight':'NOT_PERFORMED','bootstrap_indices':source(out/'frozen_bootstrap_indices.npy')})
    return old_rows


def rules_and_novelty(out,spec,old_rows):
    source(SOURCE/'development_canonical_index.csv')
    assert sha(SOURCE/'development_canonical_index.csv')==read(SOURCE/'updated_rule_validation.json')['source']['SHA256']
    known={r['condition_id'] for r in old_rows};novelty=[];predictions=[]
    for c in spec['conditions']:
        seen=c['P_id'] in known
        novelty.append({'condition':c['condition'],'P_id':c['P_id'],'full_P_previously_measured':seen,'genuinely_unseen_full_P':not seen,
                        'image_independence':False,'scope':'Previously seen548 development images; whole-P novelty is not image novelty'})
        for d in spec['models']:
            prior=next((r for r in old_rows if r['condition_id']==c['P_id'] and r['D']==d['D']),None)
            predictions.append({'condition':c['condition'],'P_id':c['P_id'],'D':d['D'],'genuinely_unseen_full_P':not seen,
               'method':'VM45_SHALLOW84_TEXT_REPLAY_V1','predicted_state':'FAIL' if 'Normal' not in c['leaves'] or d['D']=='ck05_Infer640' else 'PASS',
               'historical_exact_lookup_state':prior['original_rule_state'] if prior else 'NO_MEASUREMENT',
               'historical_lookup_source':prior['source_folder'] if prior else None,
               'historical_timing_protocol':prior['timing_protocol'] if prior else None,
               'VM46_verified_state':'NOT_EVALUATED','safe_controller':'ABSTAIN','admitted_for_deployment':False})
    table(out/'configuration_novelty.csv',novelty);table(out/'Heldout_rule_predictions.csv',predictions)
    save(out/'heldout_scoring_protocol.json',{'positive_class':'PASS','false_PASS':'actual FAIL predicted PASS / actual FAIL',
       'PASS_precision':'actual PASS predicted PASS / all predicted PASS with determined reference',
       'failure_recall':'actual FAIL predicted FAIL / actual FAIL','balanced_accuracy':'(PASS recall+FAIL recall)/2',
       'UNCERTAIN':'Excluded from binary denominator and separately counted; never converted to FAIL',
       'NOT_EVALUATED':'Missing measurement, not statistical UNCERTAIN',
       'zero_denominator':None,'new_full_P_conditions':sum(not r['full_P_previously_measured'] for r in novelty),
       'heldout_pairs_planned':sum(not r['full_P_previously_measured'] for r in novelty)*2,
       'previously_measured_control_pairs':sum(r['full_P_previously_measured'] for r in novelty)*2,
       'future_outcomes_used_for_training':0,'later_fitting':'Must be labeled in-sample unless untouched confirmation remains'})
    return novelty,predictions


def admission(out,spec,authorization):
    b=accepted.budget();now=datetime.now(timezone.utc)
    assert b['authorization']['SHA256']=='20259f1f92c45eff11721363d4a02b0f01090e6c31c52b93c3e3db0bf3785986'
    for r in b['charges']:source(r['path'])
    source(b['authorization']['path'])
    measured=read(SOURCE/'actual_resource_usage.json');old_plan=read(SOURCE/'full_cost_admission.json')
    costs=[r['cached_pipeline_seconds']+sum(r['CPU_evaluation_cost'].values()) for r in measured['cell_costs']]
    worst=max(costs);margin=old_plan['response_margin_factor'];assert margin==2
    cache_scenario=measured['CPU_cache']['CPU_cache_wall_seconds']*2
    formal_cap=math.ceil((worst*16*margin+cache_scenario+60+600)/60)*60
    total=formal_cap+600+1800
    expiry=datetime.fromisoformat(b['deadline_UTC'])
    auth={'status':'MISSING_NEW_VM46_AUTHORIZATION','historical_VM45_permission_reused':False}
    if authorization:
        a=read(authorization)
        valid=(a.get('ExperimentID')==HERE.name and a.get('RunID')==out.name and a.get('protocol_SHA256')==sha(out/'protocol_freeze.json')
               and a.get('explicit_author_VM46_approval') is True and a.get('cells')==16 and a.get('images_per_cell')==548)
        auth={'status':'SCHEMA_MATCH_REQUIRES_AUTHOR_SOURCE_REVIEW' if valid else 'INVALID_SCOPE','source':source(authorization),'historical_VM45_permission_reused':False}
    result={'status':'GPU_BLOCKED','UTC':now.isoformat(),'clock_before_original_deadline':now<expiry,'actual_ledger':b,
       'new_VM46_GPU_authorization':auth,'full_design_cells':16,'full_cost':{'status':'NOT_ADMITTED' if total>b['remaining_seconds'] else 'COST_ONLY_PASSES',
          'basis':'VM45 actual per-cell cached pipeline plus COCO and200 class/area Bootstrap; worst cell x16 x inherited2 margin',
          'worst_measured_cell_seconds':worst,'margin':margin,'cache_CPU_scenario_seconds_NOT_MEASURED':cache_scenario,
          'startup_allowance_seconds':60,'final_verification_allowance_seconds':600,'formal_worker_cap_seconds_PROPOSED':formal_cap,
          'engineering_preflight_cap_seconds_PROPOSED':600,'shutdown_reserve_seconds':1800,'complete_admission_requirement_seconds':total,
          'fits_current_balance':total<=b['remaining_seconds'],'fits_original_deadline':now+timedelta(seconds=total)<expiry,
          'not_an_ETA':'An empirical planning envelope, not proof actual16 cells cannot finish faster; no optimistic budget bypass'},
       'device_preflight':'NOT_PERFORMED_NO_NEW_AUTHORIZATION','current_thermal_admission':'NOT_PERFORMED; VM45 peak62 is historical inference only',
       'full548_cache':'NOT_GENERATED','GPU_seconds_this_run':0,'automatic_retry':False,'budget_reset':False,
       'blockers':['NEW_VM46_GPU_AUTHORIZATION_NOT_SUPPLIED','CONSERVATIVE_FULL16_COST_EXCEEDS_BALANCE']}
    save(out/'Admission_results.json',result)
    return result


def roadmap(out):
    path=V37/'vm37_yolo26_rq4_measured_cost_analysis_cpu_20261112T215459_ea3fd53c/cost_analysis.json'
    costs=read(path)
    p=RULE_SOURCE/'Formal_RQ4_resource_and_protocol.json';formal=read(p)
    report=ROOT/'datasets/reportcn/vm37_yolo26_rq4_V3_1最终科学准入审查_15c25191.md';source(report)
    timing=V37/'vm37_yolo26_rq4_pilot960_summary_cpu_20261114T202754_a23c920f/resource_summary.csv';source(timing)
    with timing.open(encoding='utf-8-sig') as f:pilot=list(csv.DictReader(f))
    c=costs['CONTROL_training_scope_seconds'];a=costs['ALL_FIVE_training_scope_seconds']
    plan={'status':'FORMAL_RESOURCE_NOT_ADMITTED','formal_protocol_source':source(p),'formal_main_arms':formal['arms'],'five_v8_ablations':formal['ablations'],
      'formal_data':formal['data'],'formal_training':formal['training'],'formal_evaluation':formal['evaluation'],
      'measured_1280_first_epoch_CONTROL_seconds':c,'measured_1280_first_epoch_ALL_FIVE_seconds':a,
      'first_epoch_limit':'Different nominal schedules100 vs20, baseline input pipeline; neither is a completed100-epoch study',
      'historical_960_10epoch_timings':pilot,'960_limit':'Exploratory; close_mosaic10 closes Mosaic fromepoch1. Not a1280/100 cost or scientific substitute.',
      'linear_legacy_26s_100epoch_pair_hours_SCENARIO_ONLY':100*(c+a)/3600,
      'linear_legacy_26s_six_arms_hours_SCENARIO_ONLY':3*100*(c+a)/3600,
      'YOLOv8s_complete_cost':'UNKNOWN_TARGET_BENCHMARK_REQUIRED','new_optimized1280_full_epoch_cost':'UNKNOWN',
      'five_targeted_augmentation_ablations_cost':'UNKNOWN_UNIMPLEMENTED_FROZEN_SPEC',
      'known_26s_checkpoint_bytes':160788774,'six_26s_arms_100_epoch_snapshots_GiB_SCENARIO_ONLY':160788774*600/1024**3,
      'total_study_storage':'UNKNOWN: add v8, five ablations, datasets,504 research cells, raw predictions, logs and reserve',
      'target_planning_not_measured_minimum':{'isolated_GPU_count':1,'GPU_VRAM_GiB':16,'RAM_GiB':32,'logical_CPUs':8},
      'next_target_measurement':'Separately authorized full6471/548 first-epoch CONTROL/ALL_FIVE, nominal100 schedule, same initialization per pair; full health/save/reload/timing. Benchmark v8 separately.',
      'independent_confirmation':'Existing test-dev/UAVDT admission restrictions unchanged; no new independence claim',
      'current_36h_ledger_is_not_formal100_authorization':True}
    save(out/'RQ4_resource_roadmap.json',plan)
    return plan


def reports(out,spec,novelty,admit,plan):
    def md(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(str(x) for x in r)+' |' for r in rows]])
    q=admit['full_cost'];new=[r['condition'] for r in novelty if r['genuinely_unseen_full_P']]
    known=[r['condition'] for r in novelty if r['full_P_previously_measured']]
    body=f'''# VM46 科学实验报告

RunID：`{out.name}`。CPU冻结和必要新增组合检查完成，GPU分支BLOCKED。没有新推理、COCO AP、Bootstrap区间或训练；不存在VM46实测交互与新false-PASS估计。

## 冻结设计与来源

8个P×历史ck05/ck01 Infer1280×548图＝8768张次。FP32/batch1/native10，完整复用原预测器、ignore-union、原图bbox面积、max_det3000、conf.001、IoU.7及VM41缓存计时。Gaussian_s15仅为显示名，原UVL叶是Gauss_s15。参数直接取原UVL，原件未修改。

{md(['条件','原UVL叶','原值'],[[c['condition'],','.join(c['leaves']),json.dumps(c['attributes'],ensure_ascii=False)] for c in spec['conditions']])}

本次48项像素检查为固定前6图×8P，均重复生成、PNG内存往返验证通过。548源ZIP、GT、模型当前SHA与VM45验收匹配；不是重新运行全部548图GPU关卡。没有生成4384份正式PNG。6张Normal/Gaussian/JPEG与历史像素逐图相同。主seed20261108来自历史EXECUTION_PROTOCOL；每图每特征SHA种子保留，Gaussian单项与复合共用同一噪声。未来两个检测器必须复用相同PNG。

顺序为光照→模糊→噪声→下采样后还原尺寸→JPEG（未选阶段跳过）。Gaussian为uint8数字强度标准差15，裁剪/截断沿用源码；Shot为photons100的Poisson代理，不是实测传感器标定。LowLight gain.4/bias−20先于Shot。Defocus为半径6像素归一化圆盘卷积；Downscale2为AREA缩小、LINEAR恢复原尺寸，不改GT坐标。JPEG25为解码后的额外一次编码，Normal不是无损采集源。全部保留原GT和ignore，RollingShutter不参与。CPU结构/像素检查不等于物理真实性证明。

## 留出与规则冻结

已测完整P：{', '.join(known)}。未见完整P：{', '.join(new)}，共{len(new)}P、{2*len(new)}个预定留出P-D。特征叶均有既往开发证据；新组合不等于新图像或新因子。三个已测控制P不能计入未知P恢复成功率。

冻结规则为VM45已导出的84行浅树文本，不重新拟合：无Normal光照叶或D_ck05_Infer640时预测FAIL，否则PASS。它不是检测器放行授权，安全控制器全部ABSTAIN。VM45更新90行的树/RF只有OOF输出，没有保存最终估计器，不能把其OOF标签冒充对新P的预测；本轮不重建这些模型。冻结时未读取任何未来VM46结果，预测见Heldout_rule_predictions.csv。

VM45已有新P迁移PASS精确率20%，更新90行整P留出树/RF为30.43%/35.00%，这些是旧证据，未重算也不冒称VM46结果。未来仅对{2*len(new)}个真正未见P-D计分，UNCERTAIN单列，NOT_EVALUATED是缺测；false-PASS分母为实际FAIL，PASS精确率分母为有确定真值的预测PASS，零分母为null。没有新确定真值，所以本轮二者NOT_EVALUATED。

## GPU准入与实际资源

当前UTC：{admit['UTC']}。真实余额{admit['actual_ledger']['remaining_seconds']:.3f}秒，原截止{admit['actual_ledger']['deadline_UTC']}。未提供新的VM46专项授权；VM45权限没有沿用。设备与当前热预检均未执行，不以VM45最高62°C代替本次通过。

VM45最慢实测单格（缓存管线+COCO+200次Bootstrap）为{q['worst_measured_cell_seconds']:.3f}秒。沿用既有2倍不确定性余量，16格正式worker规划上限{q['formal_worker_cap_seconds_PROPOSED']}秒，另留600秒预检及1800秒安全余量，总需求{q['complete_admission_requirement_seconds']}秒（{q['complete_admission_requirement_seconds']/3600:.2f}小时），超过余额。缓存CPU/启动/验收情景另见Admission_results.json。该包络不是实际ETA，不能证明16格一定跑不完；只是当前没有保守完整成本准入，不为放行而减小余量或缩减设计。

本轮GPU计费0，未写新GPU票据、锁或账本扣费。CPU实耗见Resource_usage.json。原36小时授权与期限不变，没有重试、训练或Git操作。

## 交互、控制器与Gap

Gaussian×JPEG DID＝联合−Gaussian−JPEG+Normal；跨D为ck01−ck05，200次同图索引已保存，但没有计算新AP或区间。正DID可能是AP下界/损失饱和，不能称作有益退化或物理因果。现有开发结果不补作16格新测量。

Actual_metrics.csv和Gaussian_JPEG_DID.csv仅保留空表头，明确零新测量；False_PASS_validation.csv、Controller_comparison.csv、Gap_status.csv记录NOT_EVALUATED，不填0充当AP、不把缺测写成UNCERTAIN。Tiny≥.10、缓存时延≤100ms及原区间判据冻结；未来只给已测两个D范围内的COVERED/GAP/UNCERTAIN，不估计全空间失败率。固定检测器与查表的新增性能比较也未评价。

## RQ4成本路线

正式设计仍为26s/v8s×seed42/43/44×CONTROL/ALL_FIVE，共12臂1280/100轮，固定epoch100 EMA；原五项v8单增强消融另列。6471/548、batch2/nbs64、AdamW lr0.01、warmup3、cosine、workers0、FP32、close_mosaic10不变。

历史1280首轮CONTROL实测{plan['measured_1280_first_epoch_CONTROL_seconds']:.3f}秒、ALL_FIVE {plan['measured_1280_first_epoch_ALL_FIVE_seconds']:.3f}秒，名义调度分别100/20，且为旧输入链路，只能作成本背景。线性外推一对100轮约{plan['linear_legacy_26s_100epoch_pair_hours_SCENARIO_ONLY']:.2f}小时、26s六臂约{plan['linear_legacy_26s_six_arms_hours_SCENARIO_ONLY']:.2f}小时，均非实测或服务器ETA；不与优化CPU比值混作整轮提速。

已有960/10轮实耗另存RQ4_resource_roadmap.json，不能按像素或轮数比例替代1280/100：10轮的close_mosaic10从第一轮关闭Mosaic，正式100轮前90轮仍启用。v8完整成本、优化后1280整轮、504项终点评价及五消融全成本未知。26s单份FP32 checkpoint160788774字节，六臂各100份约89.85GiB只是快照情景，不是整个研究磁盘需求。

下一资源步骤是另行授权目标设备同初始化完整6471/548首轮两臂成本基准，保持100轮调度，记录健康、保存重载、显存和完整阶段时间；v8另测。单卡16GiB显存、32GiB RAM、8逻辑CPU是沿用规划建议，非已测最低需求。正式研究、目标增强与独立确认必须分开；test-dev和UAVDT限制未变，当前余额不授权正式训练。

证据目录：{out}。CPU复现：`.venv_rq4/Scripts/python.exe -B -X utf8 main/experiments/experiments_test/vm46_heldout_variability/run.py`，会分配新Run，不覆盖本Run；该入口没有GPU执行功能。旧VM36–VM45原始文件及权重只读。
'''
    brief=f'''# VM46 导师短汇报

RunID：`{out.name}`。目前新增GPU测量0/16格。

- 八P设计已冻结，SAT8/8、6图×8P像素检查48/48通过。3P已测、5P新完整组合；同548开发图，不是独立图像验证。
- 可回放VM45原导出浅树，已提前保存16条预测；更新树/RF没有最终估计器，不能伪造回放。预测PASS不放行，控制器ABSTAIN。
- 当前余额{admit['actual_ledger']['remaining_seconds']:.3f}秒；沿用保守包络需{q['complete_admission_requirement_seconds']}秒，且没有新VM46专项授权。GPU分支停止，没有预检或扣费。
- Gaussian×JPEG交互、留出false-PASS、局部Gap均未实测。不能从旧控制格补出新DID或把缺测当UNCERTAIN。
- 正式1280/100轮12臂与五消融仍需要目标设备实测成本和独立资源授权。960/10轮探索不能替代该设计。

可汇报的是已冻结、可追溯且明确受阻的下一实验，不是已完成的新科学结论。
'''
    matrix=[{'RQ':'RQ1','state':'CPU_PROPOSAL_VERIFIED','evidence':'8 legal P, numeric source,48 deterministic fixtures; full548 GPU/cache admission absent'},
      {'RQ':'RQ2','state':'NEW_INTERACTION_NOT_EVALUATED','evidence':'Gaussian/JPEG four-corner and frozen5-P holdout planned; no new AP or rule accuracy'},
      {'RQ':'RQ3','state':'NEW_GAP_NOT_EVALUATED','evidence':'Tiny.10/cache100ms frozen; new two-D measurements absent, safe ABSTAIN'},
      {'RQ':'RQ4','state':'FORMAL_RESOURCE_BLOCKED','evidence':'12-arm100epoch and five v8 ablations unchanged; legacy cost scenarios not target ETA'}]
    table(out/'RQ_evidence_matrix.csv',matrix)
    manuscript='# VM46 RQ证据矩阵\n\nRunID：`'+out.name+'`。\n\n'+md(['RQ','本轮状态','证据与限制'],[[r['RQ'],r['state'],r['evidence']] for r in matrix])+'\n\n所有新GPU结果均NOT_EVALUATED。已有VM45结果仍仅DEVELOPMENT_ONLY合成探索，原图独立性、真实天气标定与架构因果均未升级。没有新增规则可部署、Gap关闭或正式RQ4完成声明。\n'
    paths=[]
    for name,text in [('VM46_科学实验报告_CN.md',body),('VM46_导师短汇报_CN.md',brief),('VM46_RQ证据矩阵_CN.md',manuscript)]:
        path=ROOT/'datasets/reportcn'/name
        if path.exists():path=path.with_name(path.stem+'_'+out.name.split('_')[-1]+path.suffix)
        with path.open('x',encoding='utf-8') as f:f.write(text)
        paths.append(path)
    return paths


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--authorization',type=Path,help='Read-only inspection of a separately provided VM46 authorization; not a GPU launch switch')
    args=p.parse_args();started=time.perf_counter()
    out,log=new_run({'output_root':str(RUNS),'GPU':False},HERE.name,'admission_cpu')
    source(__file__)
    spec=freeze(out);log.info('VM46 conditions and exported rule frozen before future outcomes.')
    old_rows=cpu_check(out,spec);novelty,predictions=rules_and_novelty(out,spec,old_rows)
    admit=admission(out,spec,args.authorization);plan=roadmap(out)
    table(out/'Actual_metrics.csv',[],['cell_id','AP50','AP50_95','Tiny_AP','AR','status','raw_prediction_SHA256'])
    table(out/'Gaussian_JPEG_DID.csv',[],['D','metric','DID','CI_low','CI_high','status'])
    table(out/'False_PASS_validation.csv',[{'scope':'VM46 genuinely unseen fullP only','status':'NOT_EVALUATED','scored_pairs':0,'actual_FAIL_denominator':0,
        'false_PASS_count':None,'false_PASS_rate':None,'PASS_precision':None,'failure_recall':None,'balanced_accuracy':None,'reason':'No new measured outcomes'}])
    table(out/'Controller_comparison.csv',[{'method':m,'new_measured_pairs':0,'measured_coverage':None,'status':'NOT_EVALUATED_NEW_OUTCOMES',
        'deployment_action':'ABSTAIN' if m=='SAFE_RULE_CONTROLLER' else 'NOT_AUTHORIZED'} for m in ('SAFE_RULE_CONTROLLER','EXACT_LOOKUP','FIXED_ck05_Infer1280','FIXED_ck01_Infer1280')])
    table(out/'Gap_status.csv',[{'P_id':c['P_id'],'condition':c['condition'],'D_scope':'ck05/ck01 Infer1280','status':'NOT_EVALUATED','measured_D_count':0} for c in spec['conditions']])
    save(out/'Artifact_status.json',{'new_GPU_cells':0,'new_inferences':0,'new_CIs':0,'empty_schema_files':['Actual_metrics.csv','Gaussian_JPEG_DID.csv'],
       'status_only_not_new_measurements':['False_PASS_validation.csv','Controller_comparison.csv','Gap_status.csv'],'no_simulated_AP':True})
    docs=reports(out,spec,novelty,admit,plan)
    after=accepted.budget();assert after==admit['actual_ledger']
    for r in SOURCES.values():assert sha(r['path'])==r['SHA256']
    save(out/'Source_SHA256_manifest.json',{'files':list(SOURCES.values()),'all_registered_inputs_unchanged':True,'historical_results_read_only':True})
    save(out/'Resource_usage.json',{'CPU_wall_seconds_before_seal':time.perf_counter()-started,'GPU_worker_seconds':0,'new_GPU_charge_entries':0,
       'ledger_before':admit['actual_ledger'],'ledger_after':after,'expiry_changed':False,'source_preservation_verified':True})
    save(out/'Final_acceptance.json',{'status':'CPU_COMPLETE_GPU_BLOCKED','CPU_proposal_checks':'PASS','real_cells':0,'formal_cells_required':16,
       'blockers':admit['blockers'],'DID':'NOT_EVALUATED','heldout_false_PASS':'NOT_EVALUATED','new_rule_promotions':0,'reports':[str(p) for p in docs]})
    r=read(out/'run.json');r['status']='BLOCKED';(out/'run.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
    (out/'diary.md').write_text('VM46 CPU admission: fixed8P/2D/548 images, no new outcomes. SAT and48 real-image renderer checks passed. Exported VM45 tree frozen without refit. GPU blocked by missing new authorization and conservative full16 cost. No GPU calls, training, retries, charges or historical writes. Source SHA and three Chinese reports retained.\n',encoding='utf-8')
    links=''.join(f'<li><a href="{p.as_uri()}">{p.name}</a></li>' for p in docs)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>VM46 CPU admission</title><h1>VM46: GPU blocked</h1><p>0/16 measured cells. No new AP or DID.</p><ul>'+links+'</ul><img style="max-width:100%" src="plots/CPU_pixel_preview.png">',encoding='utf-8')
    from main.visualization import build_run_index
    build_run_index(RUNS,HERE.name)
    files=[f for f in out.rglob('*') if f.is_file()]+docs+[Path(__file__).resolve()]
    with (out/'SHA256SUMS.txt').open('x',encoding='utf-8') as f:
        for file in sorted(files):f.write(sha(file)+'  '+str(file)+'\n')
    print(json.dumps({'RunID':out.name,'status':'CPU_COMPLETE_GPU_BLOCKED','GPU_seconds':0,'required_cost_seconds':admit['full_cost']['complete_admission_requirement_seconds'],
       'remaining_seconds':after['remaining_seconds'],'reports':[str(p) for p in docs]},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
