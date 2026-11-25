"""CPU-only degradation coverage and prospective Gap protocol audit."""
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import gzip
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'vm37_yolo26_rq4'))
import study
ROOT,PUBLIC=study.ROOT,study.PUBLIC
EXPERIMENT=HERE.name;RUNS=ROOT/'datasets/local_runs'/EXPERIMENT
OLD=ROOT/'datasets/local_runs/vm40_uvl_gap_independent_validation/vm40_uvl_gap_independent_validation_audit_20261116T011556_955084fb'
VM36=HERE.parent/'vm36_supervisor_reproduction'
V36RUNS=ROOT/'datasets/local_runs/vm36_supervisor_reproduction'
HISTORY=V36RUNS/'vm36_supervisor_reproduction_post80_science_20261110T190438_45783294/INDEPENDENT_ADMISSION_UPDATE.json'
AUTH=study.RUNS/'vm37_yolo26_rq4_pilot10_authorization_cpu_20261113T212036_50286ea7'
SOURCE_FILES={}
load=study.load


def source(file,role='INPUT'):
    file=Path(file).resolve(strict=True)
    result={'path':str(file),'SHA256':study.sha256(file),'bytes':file.stat().st_size,'role':role}
    SOURCE_FILES[str(file)]=result;return result


def read(file):
    source(file);return load(file)


def csv_read(file):
    source(file)
    with Path(file).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def save(file,value):
    with Path(file).open('x',encoding='utf-8') as stream:json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')


def csv_write(file,values):
    if not values:raise ValueError('Empty output table.')
    keys=list(dict.fromkeys(k for row in values for k in row))
    with Path(file).open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader();writer.writerows(values)


def module(name,file):
    spec=importlib.util.spec_from_file_location(name,file);result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result);return result


def inputs(out):
    manifest=read(OLD/'output_checksums.json')
    for item in manifest['files']:
        if study.sha256(Path(item['path']))!=item['sha256']:raise ValueError('Accepted VM40 evidence changed: '+item['path'])
    for name in ('VM40_实验配置与独立验证_科学报告_CN.md','VM40_导师会议详细讲解与问答_CN.md'):
        source(ROOT/'datasets/reportcn'/name)
    for name in ('fm_p.uvl','fm_d.uvl'):source(VM36/'feature_models'/name,'FROZEN_SUPERVISOR_UVL')
    pages=read(OLD/'PDF_pages_fresh.json')
    if source(pages['source']['path'])['SHA256']!=pages['source']['sha256']:raise ValueError('Supervisor PDF changed.')
    sections=[]
    for i,text in enumerate(pages['pages'],1):
        lines=text.splitlines()
        for j,line in enumerate(lines):
            if any(w in line.lower() for w in ('pairwise','uniform','active learning','boundary','rq1','rq2','rq3','rq4')):
                sections.append({'page':i,'text':' '.join(lines[max(0,j-1):j+3])})
    save(out/'Supervisor_plan_excerpts.json',{'PDF':source(pages['source']['path']),'source_date':'2026-10-22 author-identified plan','excerpts':sections})
    conditions=csv_read(OLD/'VM36_21_measured_conditions.csv')
    metrics=csv_read(OLD/'VM36_84_pair_metrics.csv')
    for name in ('MATCHED_56_configuration_metrics.csv','VM39_three_seed_summary.csv','THREE_STRATEGY_comparison.csv',
                 'independent_data_admission.json','gap_summary.json','UVL_validation.json'):
        source(OLD/name)
    sys.path.insert(0,str(HERE.parent/'cv261109'))
    import model
    renderer=module('vm41_frozen_renderer',HERE.parent/'cv261109/generator.py')
    for name in ('generator.py','model.py'):source(HERE.parent/'cv261109'/name,'UNCHANGED_RENDERER')
    binding=read(V36RUNS/'vm36_supervisor_reproduction_protocol_v2_20261109T153241_baec0138/IMPLEMENTATION_BINDINGS_PROPOSAL.json')
    actual_tree=read(OLD/'UVL_validation.json')['FM_P']['tree']
    attributes={r['feature']:json.loads(r['attributes']) for r in actual_tree if r['leaf']}
    if set(attributes)!=model.ALL_LEAVES:raise ValueError('Renderer and frozen UVL leaf set differ.')
    return model,renderer,attributes,binding,conditions,metrics


def propose(out,model,attributes,binding):
    cases={}
    def add(leaves,role):
        leaves=sorted(leaves);identity=model.configuration_id(leaves)
        if not model.valid_leaves(leaves):raise ValueError('Invalid proposed leaves: '+str(leaves))
        if identity not in cases:cases[identity]={'id':identity,'leaves':leaves,'roles':[]}
        cases[identity]['roles'].append(role)
    add(['Normal'],'BASELINE')
    for leaf in sorted(model.ALL_LEAVES-{'Normal','RollingShutter'}):
        leaves={leaf}
        if leaf in model.GROUPS['illumination']:
            if leaf=='LowLight':leaves.add('Shot_low')
        else:
            leaves.add('Underexposed_strong' if leaf=='Shot_high' else 'Normal')
        add(leaves,'SINGLE_FACTOR_CONTROL' if len(leaves)==2 and 'Normal' in leaves or len(leaves)==1 else 'CONSTRAINT_REQUIRED_CONTEXT')
    for a,b in [('Motion_k21','Haze_dense'),('Motion_k21','JPEG_q25'),('Haze_dense','JPEG_q25'),
                ('Motion_k21','Gauss_s15'),('Motion_k21','Defocus_r6'),('Gauss_s15','JPEG_q25'),('Underexposed_strong','Haze_dense')]:
        add(([a,b] if a in model.GROUPS['illumination'] else ['Normal',a,b]),'TWO_FACTOR_CONTRAST')
    add(['Normal','Motion_k7','RollingShutter'],'LOGICAL_WITNESS_ONLY_ANNOTATION_BLOCKED')
    nuisance=binding['proposed_values_from_historical_implementation']
    for item in cases.values():
        item['UVL_attributes']={f:attributes[f] for f in item['leaves']}
        item['implementation_bindings']={f:nuisance[f] for f in item['leaves'] if f in nuisance}
        item['render_attributes']={f:{**attributes[f],**nuisance.get(f,{})} for f in item['leaves']}
        item['binding_status']='PROPOSED_NOT_SUPERVISOR_DEFINED_REQUIRES_REVIEW' if item['implementation_bindings'] else 'NO_EXTRA_BINDING_NEEDED'
    selections=[{'id':r['id'],'leaves':r['leaves']} for r in cases.values()]
    selections.extend([{'id':'NEG_RAIN_OVEREXPOSURE','leaves':['Overexposed_strong','Rain']},
        {'id':'NEG_ROLLING_WITHOUT_MOTION','leaves':['Normal','RollingShutter']},
        {'id':'NEG_SHOT_HIGH_NORMAL','leaves':['Normal','Shot_high']}])
    save(out/'SAT_selections.json',selections)
    command=[str(ROOT/'.venv_flamapy/Scripts/python.exe'),'-B','-X','utf8',str(HERE/'sat_check.py'),
        '--model',str(VM36/'feature_models/fm_p.uvl'),'--input',str(out/'SAT_selections.json'),'--output',str(out/'SAT_results.json')]
    tick=time.perf_counter();completed=subprocess.run(command,cwd=out,capture_output=True,text=True,encoding='utf-8',timeout=120,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
    (out/'SAT.log').write_text(completed.stdout+'\n'+completed.stderr,encoding='utf-8')
    save(out/'SAT_execution.json',{'command':command,'returncode':completed.returncode,'CPU_wall_seconds':time.perf_counter()-tick})
    if completed.returncode:raise RuntimeError('SAT worker failed.')
    verdict={r['id']:r['SAT_valid'] for r in read(out/'SAT_results.json')}
    if not all(verdict[r] for r in cases) or any(verdict[r['id']] for r in selections if r['id'].startswith('NEG_')):
        raise ValueError('SAT candidate or negative control failed.')
    for item in cases.values():item['SAT_valid']=verdict[item['id']]
    save(out/'Candidate_recipes.json',list(cases.values()))
    return list(cases.values())


def audit_coverage(out,model,attributes,conditions,candidates):
    measured=[frozenset(r['selected_leaves'].split('|')) for r in conditions]
    members={s:r['configuration_id'] for s,r in zip(measured,conditions)}
    families={f:g for g,fs in model.GROUPS.items() for f in fs}
    counts=Counter(f for selected in measured for f in selected)
    valid_joint=set();level_pairs={};factor_names=list(model.GROUPS);population_count=0
    for selected in model.combinations():
        population_count+=1;valid_joint.update(tuple(sorted(p)) for p in itertools.combinations(selected,2))
        levels={g:next((f for f in fs if f in selected),'ABSENT') for g,fs in model.GROUPS.items()}
        for a,b in itertools.combinations(factor_names,2):level_pairs.setdefault((a,b),set()).add((levels[a],levels[b]))
    if population_count!=read(OLD/'UVL_validation.json')['FM_P']['valid_count']:raise ValueError('Coverage denominator disagrees with accepted count.')
    joint=Counter(tuple(sorted(p)) for selected in measured for p in itertools.combinations(sorted(selected),2))
    levels_measured={k:set() for k in level_pairs}
    for selected in measured:
        levels={g:next((f for f in fs if f in selected),'ABSENT') for g,fs in model.GROUPS.items()}
        for a,b in level_pairs:levels_measured[a,b].add((levels[a],levels[b]))
    leaf_rows=[]
    for f in sorted(model.ALL_LEAVES):
        related=[p for p in valid_joint if f in p];missing=[p for p in related if not joint[p]]
        exact=next((r for r in candidates if f in r['leaves'] and len(r['leaves'])<=2),None)
        if f=='Normal':single='BASELINE_MEASURED'
        elif f=='LowLight':single='CANNOT_ISOLATE_WITHOUT_SHOT; compare Normal+Shot_low versus LowLight+Shot_low'
        elif f=='Shot_high':single='REQUIRES_LOWLIGHT_OR_STRONG_UNDEREXPOSURE_CONTEXT'
        else:single='MEASURED' if frozenset([f]) in members or frozenset(['Normal',f]) in members else 'MISSING'
        leaf_rows.append({'leaf':f,'family':families[f],'VM36_present':bool(counts[f]),'measured_P_configurations':counts[f],
            'measured_pair_evaluations':counts[f]*4,'UVL_numeric_level':json.dumps(attributes[f],sort_keys=True),
            'actually_measured_numeric_levels':json.dumps(attributes[f],sort_keys=True) if counts[f] else 'NOT_MEASURED',
            'single_factor_contrast':single,'valid_positive_leaf_pairs':len(related),'unmeasured_positive_leaf_pairs':len(missing),
            'missing_interaction_leaves':'|'.join(sorted(set(itertools.chain.from_iterable(missing))-{f})),
            'pixel_generator':'IMPLEMENTED; VM41 CPU fixture validation only',
            'annotation_semantics':'BLOCKED_UNVERIFIED_WARP_MAPPING' if f=='RollingShutter' else 'IDENTITY_COORDINATES; fixed original-object task',
            'binding_review':'REQUIRED' if f in ('Motion_k7','Motion_k15','Motion_k21','Haze_light','Haze_dense','Rain','RollingShutter') else 'NO_EXTRA_BINDING',
            'measurement_state':'MEASURED_IN_VM36' if counts[f] else 'UNMEASURED_NOT_A_FAILURE',
            'source_manifest_SHA256':study.sha256(OLD/'VM36_21_measured_conditions.csv')})
    interaction=[]
    def corners(a,b):
        result=[]
        for on_a,on_b in itertools.product((False,True),repeat=2):
            selected=set()
            if on_a:selected.add(a)
            if on_b:selected.add(b)
            if not selected.intersection(model.GROUPS['illumination']):selected.add('Normal')
            result.append(frozenset(selected))
        return result
    for a,b in itertools.combinations(sorted(model.ALL_LEAVES),2):
        if families[a]==families[b]:continue
        cc=corners(a,b);all_legal=all(model.valid_leaves(c) for c in cc)
        point_ids=[model.configuration_id(c) for c in cc]
        measured_ids=[model.configuration_id(c) for c in cc if c in members]
        interaction.append({'leaf_A':a,'leaf_B':b,'factor_A':families[a],'factor_B':families[b],
            'SAT_cooccurrence_possible':(a,b) in valid_joint,'VM36_joint_P_count':joint[a,b],
            'cooccurrence_status':'MEASURED_JOINT_NOT_CAUSAL_CONTRAST' if joint[a,b] else 'UNMEASURED' if (a,b) in valid_joint else 'NOT_APPLICABLE_CTC_OR_ALTERNATIVE',
            'Normal_anchored_four_corners_legal':all_legal,'four_corner_config_ids':'|'.join(point_ids),
            'four_corner_measured_count':len(set(measured_ids)),'four_corners_distinct':len(set(point_ids))==4,
            'direct_DID_state':'ESTIMABLE_FIXED_NORMAL_CONTEXT' if all_legal and len(set(point_ids))==4 and len(set(measured_ids))==4 else 'NOT_ESTIMABLE',
            'reason':'Normal-anchored support only; invalid corners do not prove all conditional contrasts impossible.',
            'measured_corner_ids':'|'.join(sorted(set(measured_ids)))})
    categorical=[]
    for (a,b),possibilities in level_pairs.items():
        seen=levels_measured[a,b]
        categorical.append({'factor_A':a,'factor_B':b,'SAT_valid_level_pairs':len(possibilities),'VM36_covered_level_pairs':len(seen),
            'coverage':len(seen)/len(possibilities),'ABSENT_included':True})
    csv_write(out/'FM_P_measured_coverage.csv',leaf_rows)
    csv_write(out/'Missing_factor_interactions.csv',interaction)
    csv_write(out/'Factor_pairwise_coverage.csv',categorical)
    save(out/'coverage_summary.json',{'leaves':25,'VM36_measured_leaves':sum(counts[f]>0 for f in model.ALL_LEAVES),
        'unmeasured_leaves':sorted(f for f in model.ALL_LEAVES if not counts[f]),'VM36_P_configurations':21,'VM36_PxD_evaluations':84,
        'full_SAT_valid_population':population_count,'valid_positive_leaf_pairs':len(valid_joint),
        'measured_positive_leaf_pairs':sum(joint[p]>0 for p in valid_joint),
        'categorical_pairwise_numerator':sum(r['VM36_covered_level_pairs'] for r in categorical),
        'categorical_pairwise_denominator':sum(r['SAT_valid_level_pairs'] for r in categorical),
        'direct_Normal_anchored_four_corner_targets':sum(r['direct_DID_state'].startswith('ESTIMABLE') for r in interaction),
        'prevalence_inference':'NOT_PERMITTED; purposeful fixed cohort has no uniform inclusion probabilities'})
    return leaf_rows,interaction,categorical


def validate_pixels(out,renderer,attributes,candidates,conditions):
    import cv2
    import numpy as np
    master=20261116
    yy,xx=np.indices((273,389))
    gradient=np.stack([(xx*255//388),(yy*255//272),((xx+yy)%256)],axis=-1).astype(np.uint8)
    checks=np.where((xx//9+yy//9)%2,230,25).astype(np.uint8)
    checker=np.repeat(checks[:,:,None],3,axis=2)
    archive=ROOT/'datasets/VisDrone/VisDrone2019-DET-val.zip'
    expected=read(OLD/'historical_independence_evidence.json')['prior_dataset_gate']['splits']['val']['archive']['sha256']
    if source(archive,'READ_ONLY_SINGLE_CPU_PREVIEW_SOURCE')['SHA256']!=expected:raise ValueError('Val ZIP SHA changed.')
    with ZipFile(archive) as z:
        name=sorted(n for n in z.namelist() if '/images/' in '/'+n and n.endswith('.jpg'))[0]
        raw=z.read(name);stem=Path(name).stem
        annotation=next(n for n in z.namelist() if '/annotations/' in '/'+n and Path(n).stem==stem)
        gt=z.read(annotation)
    original=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
    if original is None:raise ValueError('Preview source decode failed.')
    fixtures=[('synthetic_gradient',gradient),('synthetic_checker',checker),(stem,original)]
    save(out/'CPU_fixture_sources.json',{'master_seed':master,'OpenCV':cv2.__version__,'numpy':np.__version__,
        'original_ZIP_member':name,'image_bytes_SHA256':hashlib.sha256(raw).hexdigest(),'annotation_member':annotation,
        'annotation_SHA256':hashlib.sha256(gt).hexdigest(),'shape':list(original.shape),'decode_flag':129,
        'scope':'Two synthetic fixtures and one existing development image; no detector forward or AP.'})
    for file in ('generator.py','model.py'):source(HERE.parent/'cv261109'/file,'CPU_EXECUTED_UNCHANGED_GENERATOR')
    trials=[];safety=[];catalog=[];images={};existing={r['configuration_id'] for r in conditions}
    for item in candidates:
        positive=True
        for image_id,image in fixtures:
            local=np.random.get_state();a,trace=renderer.generate(image,item['leaves'],item['render_attributes'],image_id,master)
            b,again=renderer.generate(image,item['leaves'],item['render_attributes'],image_id,master)
            deterministic=np.array_equal(a,b) and trace==again
            if not deterministic or a.shape!=image.shape or a.dtype!=np.uint8:raise ValueError('Candidate pixel validation failed.')
            if not all(np.array_equal(x,y) if isinstance(x,np.ndarray) else x==y for x,y in zip(local,np.random.get_state())):
                raise ValueError('Renderer changed global NumPy RNG.')
            trials.append({'config_id':item['id'],'image_id':image_id,'shape':json.dumps(list(a.shape)),'dtype':str(a.dtype),
                'pixel_SHA256':hashlib.sha256(a.tobytes()).hexdigest(),'repeated_identically':True,'effects':json.dumps(trace,sort_keys=True),
                'master_seed':master,'scope':'CPU_PIXEL_CHECK_NOT_DETECTION_RESULT'})
            if image_id==stem:images[item['id']]=a
        rolling='RollingShutter' in item['leaves']
        annotation='BLOCKED_UNVERIFIED_WARP_MAPPING' if rolling else 'COORDINATE_IDENTITY_VALIDATED_FIXED_OBJECT_TASK'
        safety.append({'config_id':item['id'],'SAT_valid':True,'pixel_generator_validated':positive,'CPU_fixture_count':3,
            'annotation_safe':not rolling,'GT_mapping':'NOT_IMPLEMENTED_FOR_ROW_DEPENDENT_WARP' if rolling else 'IDENTITY: original xywh/classes/areas/ignore regions unchanged',
            'visibility_policy':'Keep original object instances including artificial occlusion; no AP-driven GT deletion',
            'annotation_status':annotation,'physical_realism':'NOT_ESTABLISHED_BY_PIXEL_TEST',
            'GPU_execution_ready':False,'GPU_blocker':'CPU_ONLY_TASK; independent data / scope / cost / thermal gates not passed',
            'nuisance_binding_status':item['binding_status']})
        catalog.append({'configuration_id':item['id'],'selected_leaves':'|'.join(item['leaves']),
            'role':'|'.join(item['roles']),'already_measured_P_in_VM36':item['id'] in existing,
            'UVL_numerical_parameters':json.dumps(item['UVL_attributes'],sort_keys=True),
            'implementation_parameters_proposal':json.dumps(item['implementation_bindings'],sort_keys=True),
            'complete_renderer_parameters':json.dumps(item['render_attributes'],sort_keys=True),'SAT_valid':True,
            'pixel_generator_validated':True,'annotation_safe':not rolling,'scientific_preparation_status':'ANNOTATION_BLOCKED' if rolling else 'BINDING_REVIEW_REQUIRED' if item['implementation_bindings'] else 'PREPARED_FOR_FUTURE_ADMISSION',
            'GPU_execution_ready':False,'master_seed':master,
            'RNG_definition':'SHA256(f"{master_seed}|{image_stem}|{leaf}") first8 bytes big-endian -> numpy.default_rng; independent per leaf; no strategy/resolution term',
            'order':'atmosphere->illumination->blur->rolling_shutter->noise->downscale->compression',
            'generator_SHA256':study.sha256(HERE.parent/'cv261109/generator.py'),
            'UVL_SHA256':study.sha256(VM36/'feature_models/fm_p.uvl')})
    mask=np.zeros((273,389,3),np.uint8);mask[74:108,118:160]=255
    rng=np.random.default_rng(12345)
    warped,phase=renderer.apply_rolling_shutter(mask,.008,2,rng)
    y,x=np.nonzero(warped[:,:,0]>127);new=[int(x.min()),int(y.min()),int(x.max()+1),int(y.max()+1)]
    if new==[118,74,160,108]:raise ValueError('Rolling-shutter negative geometry control did not move the box.')
    save(out/'RollingShutter_geometry_counterexample.json',{'synthetic_original_xyxy':[118,74,160,108],
        'warped_foreground_xyxy':new,'original_bbox_safe':False,'phase':phase,'parameters_proposal':{'max_shift_fraction':.008,'waves':2},
        'scope':'Counterexample, not a validated GT mapping. Fixed boxes are blocked; ignore regions require mapping too.'})
    csv_write(out/'CPU_pixel_validation.csv',trials);csv_write(out/'Candidate_valid_configurations.csv',catalog)
    csv_write(out/'Candidate_annotations_safety.csv',safety)
    save(out/'candidate_validation_summary.json',{'configurations':len(catalog),'SAT_valid':len(catalog),
        'pixel_validated':len(catalog),'annotation_safe':sum(r['annotation_safe'] for r in catalog),
        'GPU_ready':0,'image_fixture_executions_including_repeat':len(trials)*2,
        'nuisance_values':'Historical proposal frozen for CPU testing only; not supervisor-defined or approved for GPU.',
        'GT_rule':'Artificial photometric loss does not select or remove GT instances. RollingShutter excluded.'})
    return catalog,images,original


def timing_protocol(out):
    import cv2
    import numpy as np
    from io import StringIO
    from timing_protocol import cached_image
    old=VM36/'joint_batch.py';new=HERE.parent/'vm37_yolo26_rq4/post_cost/evaluate.py'
    source(old,'TIMING_DEFINITION');source(VM36/'full_development.py','TIMING_DEFINITION');source(new,'TIMING_DEFINITION')
    comparison=[]
    for name,a,b,future in [
        ('cache_generation','OUTSIDE','inside total inference_wall, outside inference_ms','OUTSIDE latency; charged in end-to-end worker'),
        ('cache_file_integrity_hash','INSIDE','OUTSIDE inference_ms; source pixel hash before prediction','INSIDE; verify PNG bytes before decode'),
        ('read_decode','INSIDE','OUTSIDE inference_ms','INSIDE; second read + OpenCV PNG decode'),
        ('pixel_SHA_integrity','INSIDE','OUTSIDE inference_ms','INSIDE'),
        ('preprocess_forward_NMS','INSIDE with CUDA synchronize','INSIDE with CUDA synchronize','INSIDE; FP32 batch1 square Letterbox'),
        ('prediction_extraction_CPU','INSIDE','OUTSIDE inference_ms','INSIDE'),
        ('JSON_gzip_write_flush','INSIDE','OUTSIDE inference_ms; no same per-image flush boundary','INSIDE; prediction and image manifest flush'),
        ('model_load','OUTSIDE per-image latency','OUTSIDE inference_ms','OUTSIDE latency; included worker cost'),
        ('warmup','3 per configuration','No equivalent three explicit warmups recorded','3 per configuration; then CUDA synchronize'),
        ('COCO_bootstrap','OUTSIDE latency','OUTSIDE inference_ms','OUTSIDE latency; included worker cost')]:
        comparison.append({'component':name,'VM36_cached_pipeline':a,'VM37_39_inference_ms':b,'VM41_future_protocol':future,
            'historical_latency_conversion':'NOT_IDENTIFIABLE_FROM_SAVED_AGGREGATES','source_VM36':str(old),'source_VM37_39':str(new)})
    csv_write(out/'Timing_protocol_comparison.csv',comparison)
    protocol={'id':'VM41_MATCHED_CACHED_GAP_V1','status':'FROZEN_PROPOSAL_CPU_INTERFACE_TESTED_GPU_NOT_RUN',
        'task':'native10 VisDrone detection; no substitute label mapping','GT':'Original image coordinates/areas; unchanged accepted ignore-union implementation',
        'inference':{'batch':1,'precision':'FP32','AMP':False,'TF32':False,'Ultralytics':'8.4.210',
            'head':'dense one-to-many','conf':.001,'NMS_IoU':.7,'class_aware':True,'multi_label':True,'max_det':3000,'max_nms':30000,
            'square_letterbox':True,'imgsz_primary':1280,'TTA':False},
        'timing':{'primary':'cached_pipeline_ms','components':comparison,'warmups_per_cell':3,
            'format':'PNG cache, gzip raw prediction JSONL plus per-image manifest; flush per image',
            'clock':'time.perf_counter monotonic','durability':'flush is not fsync; final atomic commit and checksum happen outside per-image latency',
            'cache_policy':'OS-warm cache after generation/validation; no claims of cold-disk latency',
            'hardware':'Same device/driver/runtime/storage and monitoring; record all identities per run',
            'pair_order':'Predefined balanced CONTROL/GUIDED order across seeds; same frozen image ordering and PNG bytes',
            'old_latency':'No retrospective numeric harmonization; new protocol cannot turn old inference_ms into pipeline_ms'},
        'statistics':{'image_bootstrap_samples':200,'seed':20261116,'index_shared_across_all_cells':True,
            'AP':'Recompute pooled COCO AP on each resampled GT/prediction collection; no mean image AP',
            'latency':'Bootstrap mean of per-image cached_pipeline_ms on identical resampling indices',
            'CI':'Pointwise 95% percentile; no simultaneous/familywise Gap guarantee; no flight clustering without reliable IDs',
            'cross_seed':'Three matched training differences; sample SD ddof=1, not image-bootstrap training-seed significance'},
        'Gap':{'Tiny_AP95_min':.10,'pipeline_mean_ms_max':100,'original_rule':'FAIL if latency upper95>100 or Tiny upper95<0.10; PASS if Tiny lower95>=0.10 and latency upper95<=100; otherwise UNCERTAIN',
            'subset':'At least one D PASS -> COVERED; all D FAIL -> GAP; otherwise UNRESOLVED; only declared measured D subset',
            'sensitivity':'Keep separately labelled two-sided latency rule; never silently replace primary rule'},
        'independent_primary':{'cells':12,'seeds':[42,43,44],'strategies':['CONTROL','VARIABILITY_GUIDED'],'conditions':['Normal','Gaussian_s15'],
            'resolution':1280,'checkpoints':'Reuse healthy fixed epoch10 EMA; no new training','primary_endpoint':'Across fixed three training seeds, mean GUIDED minus CONTROL Gaussian AP95; image interval conditional on those seeds',
            'no_retuning':True},
        'safety':{'GPU_budget_new_seconds':0,'thermal_immediate_C':90,'thermal_sustained_C':87,'thermal_sustained_seconds':30,
            'monitor_interval_seconds':2,'retry':False,'requires_new_GPU_task_authorization':True}}
    save(out/'Future_Gap_protocol.json',protocol)
    image=np.arange(64*96*3,dtype=np.uint8).reshape(64,96,3);ok,encoded=cv2.imencode('.png',image)
    if not ok:raise ValueError('CPU test PNG failed.')
    fixture=out/'checks';fixture.mkdir();path=fixture/'synthetic_cache.png';path.write_bytes(encoded.tobytes())
    sync=[];output=StringIO();manifest=StringIO()
    fake_predict=lambda x:([{'image_id':1,'class_id':0,'bbox':[1.,2.,3.,4.],'score':.9}],{'NCHW':[1,3,1280,1280],'scope':'SYNTHETIC_METADATA_NOT_NETWORK_TENSOR'})
    common=(path,study.sha256(path),hashlib.sha256(image.tobytes()).hexdigest(),fake_predict,lambda x,i:x,lambda:sync.append(1))
    timing=cached_image(*common,output,manifest,1)
    if len(sync)!=2 or timing['cached_pipeline_ms']<timing['predict_ms'] or not output.getvalue() or not manifest.getvalue():raise ValueError('Timing boundary test failed.')
    failures=[]
    for kind,bad in [('file_SHA',{'expected_file_sha':'0'*64}),('pixel_SHA',{'expected_pixel_sha':'0'*64}),
                     ('NaN_output',{'extract':lambda r,i:[{'class_id':0,'bbox':[1,2,3,4],'score':float('nan')}]})]:
        args=dict(path=path,expected_file_sha=common[1],expected_pixel_sha=common[2],predict=fake_predict,
            extract=lambda x,i:x,synchronize=lambda:None,prediction_stream=StringIO(),manifest_stream=StringIO(),image_id=1)
        args.update(bad)
        try:cached_image(**args)
        except ValueError:failures.append(kind)
        else:raise ValueError('Corrupt timing input was accepted: '+kind)
    save(out/'Timing_CPU_interface_tests.json',{'status':'PASS','positive_case':True,'rejections':failures,
        'scope':'Synthetic CPU callback only; no actual YOLO tensor, CUDA synchronization, latency or GPU performance was tested.'})


def independent(out):
    gate=read(OLD/'independent_data_admission.json');history=read(HISTORY)
    hits=[];records=[]
    for item in history['direct_sources_rechecked']:
        path=Path(item['source']['path']);proof=source(path,'BOUNDED_HISTORICAL_SOURCE_RECORD')
        if proof['SHA256']!=item['source']['sha256']:raise ValueError('Historical source record SHA differs.')
        text=path.read_text(encoding='utf-8-sig')
        for number,line in enumerate(text.splitlines(),1):
            if any(word in line.lower() for word in ('test-dev','test_dev','testdev')):
                hits.append({'path':str(path),'line':number,'text':line,'kind':'CONFIG_OR_PROTOCOL_REFERENCE_ONLY'})
        records.append(proof)
    metadata=[];prefixes={}
    for split in ('train','val','test-dev'):
        archive=ROOT/'datasets/VisDrone'/('VisDrone2019-DET-'+split+'.zip')
        with ZipFile(archive) as z:
            names=sorted(n for n in z.namelist() if '/images/' in '/'+n and n.endswith('.jpg'))
        # Filename prefixes are proxies, not certified flight identities.
        prefixes[split]={Path(n).stem.split('_')[0] for n in names}
        metadata.append({'split':split,'image_members':len(names),'filename_prefix_count':len(prefixes[split]),
            'source_ZIP':str(archive),'hash_evidence':'Reused accepted VM40 ZIP certificate; no duplicate full archive/image audit',
            'sequence_identity':'UNVERIFIED_FILENAME_PROXY'})
    overlap={'testdev_train_prefix_intersection':sorted(prefixes['test-dev']&prefixes['train']),
        'testdev_val_prefix_intersection':sorted(prefixes['test-dev']&prefixes['val']),
        'interpretation':'Overlap is a review flag, not proof of identical scenes. Zero overlap is not evidence of scene independence.'}
    save(out/'Historical_testdev_records.json',{'scope':'Only the seven direct historical records referenced by the accepted VM36 bounded audit',
        'records':records,'keyword_hits':hits,'absence_of_hits_used_as_independence':False,
        'complete_evaluation_inventory':'UNKNOWN','creation_time_training_membership':'UNKNOWN_FOR_OLD_CK01_CK05',
        'checkpoint_selection_manual_tuning':'UNKNOWN; config paths alone cannot prove evaluation completed or absent'})
    csv_write(out/'Scene_proxy_inventory.csv',metadata)
    save(out/'Scene_proxy_overlap.json',overlap)
    result={'status':'DATASET_BLOCKED','VM40_evidence':source(OLD/'independent_data_admission.json'),
        'new_GPU_evaluation':False,'testdev':{'membership_and_byte_disjointness':'VERIFIED_CURRENT_ARCHIVES_BY_VM40',
            'annotations_and_native10':'VM40_CPU_VALIDATED_1610_IMAGES_AND_GT','research_use':'VM40_SAVED_OFFICIAL_RESEARCH_USE_STATEMENT',
            'new_fixed_epoch10_training':'Preserved train6471/val548 protocols; not cleared by mere absence of historical search hits',
            'old_checkpoint_creation_time_membership':'UNKNOWN','historical_model_and_method_selection':'UNVERIFIED',
            'scene_isolation':'UNVERIFIED','filename_proxy_overlap':overlap,'admission':'BLOCKED'},
        'UAVDT':{'admission':'BLOCKED_ANNOTATION_PROVENANCE_AND_LABEL_SPACE',
            'source':source(OLD/'UAVDT_bounded_inventory.json'),'classes':['bus','car','truck','vehicle'],
            'original_DET_ignore_split':'UNRESOLVED','native10_compatibility':'NOT_ESTABLISHED',
            'new_relabeling_or_GT_invention':False},
        'required_evidence':['Checkpoint-specific creation-time training and selection manifests',
            'Inventory of completed historical test-dev evaluations and subsequent method tuning',
            'Reliable sequence/flight/source-scene mapping and overlap review',
            'Dataset-specific research-use terms and native10/ignore/geometry compatibility'],
        'if_history_cannot_be_recovered':'Submit a prospective genuinely held-out native10 UAV scene plan for supervisor review; freeze scene IDs, labels, rights, endpoints and policy before predictions. No silent dataset substitution.',
        'no_matching_history_is_not_independence':True,'GPU_scope_this_task':'NONE'}
    save(out/'Independent_data_readiness.json',result)
    return result


def cost_and_evidence(out,catalog):
    import pilot10
    gate=read(OLD/'independent_data_admission.json')
    auth=read(AUTH/'authorization.json');charges=[]
    for file in pilot10.charge_files(AUTH):
        row=read(file);charges.append({'file':source(file,'PRESERVED_ORIGINAL_BUDGET_CHARGE'),'charged_seconds':row['charged_seconds']})
    remaining=pilot10.available(AUTH)
    plans=[]
    for name,cells,images,multiplier,reserve in [('INDEPENDENT_PRIMARY',12,1610,1.35,1800),
        ('DEVELOPMENT_TIMING_BRIDGE',12,548,1.35,1800),('MOTION_HAZE_2x2',8,548,1.5,1800)]:
        linear=10808.362/(16*548)*cells*images
        plans.append({'design':name,'cells':cells,'images_per_cell':images,'image_inferences':cells*images,
            'measured_reference':'VM37 accepted16 cells x548 incl initialization, inference, CPU COCO and200 bootstrap',
            'reference_seconds':10808.362,'linear_planning_seconds_NOT_MEASURED':linear,
            'uncertainty_multiplier':multiplier,'shutdown_reserve_seconds':reserve,
            'planning_envelope_seconds_NOT_A_GUARANTEE':linear*multiplier+reserve,
            'fits_remaining_under_envelope':linear*multiplier+reserve<=remaining,
            'cost_status':'ESTIMATE_NOT_RELIABLE_FOR_ADMISSION: different response density, generator cache and full-pipeline I/O unmeasured',
            'GPU_admitted':False,'this_task_GPU_authorization_seconds':0})
    choice={'priority1':'First resolve independent data; then12 matched cells (3seeds x2strategies xNormal/Gauss15) Infer1280 with uniform timing. No policy retuning.',
        'priority2':'If independent data remains blocked, a separately authorized12-cell development timing bridge may establish matched Tiny+latency labels only; does not establish independent generalization.',
        'priority3':'After nuisance binding review, future8-cell design: Normal / Motion21 / Haze_dense / Motion21+Haze_dense x ck01/ck05 Infer1280 on identical548 development IDs. Four-corner support, not global coverage.',
        'priority4':'Keep VM38 rule fixed: choose family by baseline mean AP-loss; p.25 fixed prior. Validate on an admitted holdout before considering further policy selection.',
        'no_parallel_expansion':'Do not launch all plans or all32 candidates automatically; these are alternative conditional future stages.'}
    result={'status':'GPU_NOT_ADMITTED_CPU_TASK','budget_authorization':source(AUTH/'authorization.json'),
        'original_allocation_seconds':auth['max_cumulative_GPU_worker_wall_seconds'],
        'charged_seconds':sum(c['charged_seconds'] for c in charges),'remaining_seconds':remaining,
        'absolute_expiry_UTC':auth['absolute_expiry_UTC'],'new_GPU_charge_seconds':0,'allocation_reset':False,
        'charge_sources':charges,'plans':plans,'ordered_scientific_decision':choice,
        'hard_gates':['Suitable independent data if claiming independent evaluation','Explicit future GPU scope authorization',
            'Complete matched-cell resource admission incl cache, COCO, Bootstrap, shutdown margin and unchanged expiry',
            'Fresh physical/device thermal clearance;90C immediate /87C30s stop /2s telemetry',
            'All nuisance parameters approved and frozen; no RollingShutter with unverified GT',
            'Checkpoint/GT/pixel SHA, evaluator and matched latency protocol compatible'],
        'remaining_balance_is_not_authorization':True}
    save(out/'Complete_pair_cost_plan.json',result)
    evidence=[
        {'RQ':'RQ1','supervisor_goal':'Can the degradation space and the YOLO configuration space be captured as feature models with realistic constraints?',
         'accepted_evidence':'Frozen original FM_P/FM_D; VM36 21P x4D; matched saved effects',
         'coverage_limit':'18/25 leaves; Motion/Haze/Rain/Rolling absent; broad model space not measured',
         'remaining':'Conditional single-factor and complete matched contrasts; retain CTC feasibility',
         'status':'LOGICAL_MODEL_VALIDATED_IMPLEMENTABILITY_PARTIAL','source':str(OLD/'UVL_validation.json')},
        {'RQ':'RQ2','supervisor_goal':'Which degradation features and which detector features interact to cause detection failure?',
         'accepted_evidence':'VM36 four detector settings and84 measured responses; saved restricted factorial contrasts',
         'coverage_limit':'No new held-out learned-constraint accuracy established here; Motion/Haze support missing',
         'remaining':'Learn explicit cross-domain rules and evaluate held-out configuration/context groups; distinguish observed contrast from regression inference',
         'status':'RESTRICTED_DEVELOPMENT_INTERACTIONS_ONLY','source':str(OLD/'VM36_84_pair_metrics.csv')},
        {'RQ':'RQ3','supervisor_goal':'Which regions of the degradation space are not covered by any valid detector configuration, and how large are they?',
         'accepted_evidence':'17 Gap /4 covered among21P against four tested D; Tiny threshold.10 and cached pipeline100ms',
         'coverage_limit':'Purposeful sample and limited D subset; no full-space prevalence or validated full static-gap count',
         'remaining':'Admissible probability sampling, learned static-vs-empirical validation and explicit uncertainty under declared D scope',
         'status':'MEASURED_DETECTOR_SUBSET_ONLY','source':str(OLD/'gap_status_reassessment.csv')},
        {'RQ':'RQ4','supervisor_goal':'Does reconfiguring the detector space to target the gap actually shrink it?',
         'accepted_evidence':'YOLO26s960/10epoch C vs Guided seeds42/43/44; AllFive seed42; Gaussian AP benefit with Normal cost',
         'coverage_limit':'DEVELOPMENT_ONLY / POST_HOC_POLICY_SELECTION; no independent or Gap reduction claim',
         'remaining':'Independent confirmation of fixed rule; original formal study and five YOLOv8s ablations remain pending resources',
         'status':'EXPLORATORY_PARTIAL','source':str(OLD/'VM39_three_seed_summary.csv')}]
    csv_write(out/'Supervisor_RQ1_RQ4_evidence_matrix.csv',evidence)
    sampling=[
        {'method':'PAIRWISE','original_goal':'Cover valid two-way feature/level combinations',
         'measured':'21 purposeful P only; separate21 greedy core/80 logical cohort must not be substituted',
         'future':'Constrained covering sample with declared factor encoding and support; coverage does not imply four-corner estimability'},
        {'method':'UNIFORM_VALID_SPACE','original_goal':'Probability sample of SAT-valid P and/or eligible D with known inclusion probabilities',
         'measured':'VM36 measured cohort not uniform; full-space failure prevalence NOT_ESTIMATED',
         'future':'Freeze eligible population incl annotation exclusions and nuisance distributions; sample without AP; report inclusion probabilities'},
        {'method':'BOUNDARY_ACTIVE','original_goal':'Adapt queries near requirement or predicted Gap boundary',
         'measured':'No accepted VM36 adaptive selection trace establishes active learning; conditions fixed historically',
         'future':'Separate sequential selection log/oracle boundary and stop rule; no population prevalence without valid weighting/design'},
        {'method':'DETECTOR_SAMPLE','original_goal':'Explore legal FM_D choices conditional on compatible implementations',
         'measured':'2 historical small checkpoints xInfer640/1280, FP32 FullFrame NMS;84 evaluations, no retraining',
         'future':'Health/provenance/API-compatible D subset; no extrapolation to all635904 models/settings'}]
    csv_write(out/'Supervisor_sampling_comparison.csv',sampling)
    return result


def finish(out,catalog,leaves,interactions,pairwise,images,original,start):
    reporter=module('vm41_reporting',HERE/'reporting.py')
    figures=reporter.figures(out,leaves,pairwise,catalog,images,original)
    resource={'CPU_wall_seconds_to_reporting':time.perf_counter()-start,'GPU_seconds':0,'new_training':False,
        'new_detector_forward':False,'COCO_recomputed':False,'CPU_validation':'New candidates on3 fixtures including one existing val image; no detector results',
        'accepted_tests_repeated':False}
    reports=reporter.reports(out,ROOT,OLD,resource)
    validation={'candidate_configurations_unique':len({r['configuration_id'] for r in catalog})==len(catalog),
        'all_candidate_SAT_valid':all(r['SAT_valid'] for r in catalog),
        'unmeasured_not_failed':all(r['measurement_state']=='UNMEASURED_NOT_A_FAILURE' for r in leaves if not r['VM36_present']),
        'RollingShutter_not_annotation_safe':all(not r['annotation_safe'] for r in catalog if 'RollingShutter' in r['selected_leaves']),
        'no_GPU_candidates_admitted':all(not r['GPU_execution_ready'] for r in catalog),
        'timing_CPU_interface':load(out/'Timing_CPU_interface_tests.json')['status']=='PASS',
        'independent_gate_not_weakened':load(out/'Independent_data_readiness.json')['status']=='DATASET_BLOCKED'}
    for relative,expected in read(out/'source_SHA256.json').items():
        if relative.startswith('experiments/'+EXPERIMENT+'/'):continue
        if study.sha256(PUBLIC/relative)!=expected:raise ValueError('Historical Python source changed: '+relative)
    for item in SOURCE_FILES.values():
        if study.sha256(Path(item['path']))!=item['SHA256']:raise ValueError('Input changed during audit: '+item['path'])
    validation['all_recorded_input_SHAs_unchanged']=True
    if not all(validation.values()):raise ValueError('VM41 validation failed: '+str(validation))
    save(out/'validation.json',{'status':'PASS','checks':validation})
    for file in HERE.glob('*.py'):source(file,'VM41_SOURCE')
    save(out/'Source_SHA256_manifest.json',{'RunID':out.name,'sources':list(SOURCE_FILES.values()),
        'scope':'Rehashed explicit inputs before/after. VM40 output checksum manifest verified. Raw prediction archives referenced, not reread.',
        'historical_modification':False})
    reporter.page(out,figures,reports)
    resource['CPU_wall_seconds']=time.perf_counter()-start
    save(out/'resource_usage.json',resource)
    (out/'diary.md').write_text('# VM41\n\n目标：核查退化覆盖，冻结未来一致Gap评价与最小方案。\n\n'
        '执行：复用VM40；逐叶/合法二阶覆盖计算；32个结构预定候选SAT及3图CPU确定性检查；RollingShutter几何反例；CPU计时接口测试。\n\n'
        '结果：18/25叶已测，7叶未测；31候选坐标安全，RollingShutter阻塞；任何候选都未获得GPU准入。\n\n'
        '限制：Motion方向/Haze airlight/Rain参数仍属非导师定义的待审核绑定；独立数据、预算和热安全未解除。\n\n'
        '下一步：先恢复独立数据证据，再执行另行授权的完整12格；不能用目的性样本推断全空间失败率。\n',encoding='utf-8')
    record=load(out/'run.json');record.update(status='COMPLETE_WITH_LIMITATIONS',finished_UTC=datetime.now(timezone.utc).isoformat(),
        CPU_audit='PASS',independent_data='BLOCKED',GPU_seconds=0)
    (out/'run.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    save(out/'final_acceptance.json',{'status':'CPU_COVERAGE_AND_PROTOCOL_COMPLETE_WITH_LIMITATIONS','RunID':out.name,
        'GPU_seconds':0,'historical_runs_reexecuted':False,'reports':[str(p) for p in reports],
        'next_experiment':'Conditional independent12-cell matched confirmation with uniform timing; no GPU admission granted'})
    files=[p for p in out.rglob('*') if p.is_file() and 'runtime' not in p.relative_to(out).parts]+reports+list(HERE.glob('*.py'))
    save(out/'checksums.json',{'files':[{'path':str(p),'SHA256':study.sha256(p)} for p in files]})
    from main.visualization import build_run_index
    build_run_index(out.parent,EXPERIMENT)
    print('COMPLETE_WITH_LIMITATIONS',out,flush=True)


def audit():
    from main.pipeline import new_run
    out,logger=new_run({'output_root':str(RUNS)},EXPERIMENT,'coverage');study.runtime(out,cuda=False)
    start=time.perf_counter()
    try:
        model,renderer,attributes,binding,conditions,metrics=inputs(out)
        candidates=propose(out,model,attributes,binding);logger.info('SAT proposal complete: %d configurations',len(candidates))
        leaves,interactions,pairwise=audit_coverage(out,model,attributes,conditions,candidates)
        catalog,images,original=validate_pixels(out,renderer,attributes,candidates,conditions)
        logger.info('CPU pixel and annotation checks complete.')
        timing_protocol(out);independent(out);cost_and_evidence(out,catalog)
        finish(out,catalog,leaves,interactions,pairwise,images,original,start)
    except BaseException as exc:
        save(out/'failure.json',{'error':str(exc),'traceback':traceback.format_exc(),'GPU_seconds':0})
        logger.exception('VM41 stopped; original evidence retained.');raise


def verify(identity):
    if Path(identity).name!=identity or not identity.startswith(EXPERIMENT+'_'):raise ValueError('Invalid RunID.')
    out=(RUNS/identity).resolve(strict=True)
    for r in load(out/'checksums.json')['files']:
        if study.sha256(Path(r['path']))!=r['SHA256']:raise ValueError('Output changed: '+r['path'])
    print('VERIFIED',identity,'GPU=0',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='CPU-only coverage audit. No inference or training endpoint.')
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--audit',action='store_true');group.add_argument('--verify')
    args=parser.parse_args()
    if args.audit:audit()
    else:verify(args.verify)
