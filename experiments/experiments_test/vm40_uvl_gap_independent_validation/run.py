"""Read-only scientific source audit; no inference or training entry point."""
import argparse
from collections import Counter
import csv
from datetime import datetime,timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import urllib.request
from zipfile import ZipFile

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'vm37_yolo26_rq4'))
import study
PUBLIC,ROOT=study.PUBLIC,study.ROOT
EXPERIMENT=HERE.name
RUNS=ROOT/'datasets/local_runs'/EXPERIMENT
VM36=HERE.parent/'vm36_supervisor_reproduction'
V36RUNS=ROOT/'datasets/local_runs/vm36_supervisor_reproduction'
FULL=V36RUNS/'vm36_supervisor_reproduction_full20_results_20261110T161658_46e4aed4'
NORMAL=V36RUNS/'vm36_supervisor_reproduction_normal4_analysis_20261110T234523_1ce40f03'
PRIOR=ROOT/'datasets/local_runs/cv261207/cv261207_supervisor_compliance_20261109T121122_70d953f1'
P37=study.RUNS/'vm37_yolo26_rq4_pilot960_execution_20261114T110947_fe2f01b3'
P38=ROOT/'datasets/local_runs/vm38_variability_guided_augmentation/vm38_variability_guided_augmentation_targeted960_20261114T215310_e8bba12b'
P39=ROOT/'datasets/local_runs/vm39_multi_seed_confirmation/vm39_multi_seed_confirmation_confirmation_20261115T095834_9623ace0'
BUNDLE=study.RUNS/'vm37_yolo26_rq4_portable_v3_build_cpu_20261113T115150_6e2bcebf/bundle'
AUTH=study.RUNS/'vm37_yolo26_rq4_pilot10_authorization_cpu_20261113T212036_50286ea7'
load=study.load
SOURCES={}


def ref(path,role='SOURCE'):
    path=Path(path).resolve(strict=True)
    value={'path':str(path),'sha256':study.sha256(path),'bytes':path.stat().st_size,'role':role}
    SOURCES[str(path)]=value
    return value


def read(path):
    ref(path)
    return load(path)


def save(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')


def csv_out(path,rows):
    if not rows:raise ValueError('No rows: '+str(path))
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def extract_pdf(text,namespace):
    text=text[text.index('namespace '+namespace):]
    clean=[];pending=''
    for line in text.splitlines():
        if line.startswith('Your tasks') or line.startswith('Gap analysis'):break
        if line.startswith('Research plan:') or re.match(r'^Page \d+ of',line):continue
        if pending:
            pending+=' '+line.strip()
            if '}' in pending:clean.append(pending);pending=''
            continue
        if '{' in line and '}' not in line:pending=line;continue
        if re.fullmatch(r'(?:namespace \w+|features|constraints)',line.strip()):clean.append(line.strip())
        elif re.fullmatch(r'\s{4,}\w+(?:\s+\{[^{}]*\})?\s*',line):clean.append(line.rstrip())
        elif '=>' in line or line.strip().startswith(('!','&','|')):
            if clean and clean[-1].rstrip().endswith(('&','|')):clean[-1]+=' '+line.strip()
            else:clean.append(line.rstrip())
    return '\n'.join(clean)+'\n'


def uvl(out):
    from pypdf import PdfReader
    proof=read(PRIOR/'pdf_provenance.json');pdf=Path(proof['canonical']['path'])
    if ref(pdf)['sha256']!=proof['canonical']['sha256']:raise ValueError('Supervisor PDF SHA changed.')
    pages=[p.extract_text() or '' for p in PdfReader(pdf).pages]
    save(out/'PDF_pages_fresh.json',{'source':ref(pdf),'pages':pages})
    for domain,indices,namespace in [('FM_P',(4,6),'UAVDegradation'),('FM_D',(6,9),'YOLODetector')]:
        text=extract_pdf('\n'.join(pages[indices[0]:indices[1]]),namespace)
        (out/('pdf_'+('fm_p.uvl' if domain=='FM_P' else 'fm_d.uvl'))).write_text(text,encoding='utf-8')
    command=[str(ROOT/'.venv_flamapy/Scripts/python.exe'),'-B','-X','utf8',str(HERE/'uvl_audit.py'),
             '--out',str(out),'--experiments',str(HERE.parent)]
    start=time.perf_counter()
    p=subprocess.run(command,cwd=out,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=300,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1'))
    (out/'flamapy.log').write_text(p.stdout+'\n'+p.stderr,encoding='utf-8')
    save(out/'flamapy_execution.json',{'command':command,'returncode':p.returncode,'CPU_wall_seconds':time.perf_counter()-start})
    if p.returncode:raise RuntimeError('FlamaPy audit failed; see flamapy.log.')
    result=load(out/'UVL_validation.json')
    explanations=[
      ('FM_P','LowLight => ShotNoise','选择LowLight必须同时选择ShotNoise；是导师待核验的物理合理性假设，不是已证明物理定律。'),
      ('FM_P','Rain => !Overexposed_strong & !Overexposed_mild','雨不能与两档过曝同选；逻辑可验证，真实场景普遍适用性仍需依据。'),
      ('FM_P','Haze_dense => !Overexposed_strong','浓雾不能与强过曝同选；保留导师草案约束，未将其升级为已证实定律。'),
      ('FM_P','RollingShutter => MotionBlur','RollingShutter要求MotionBlur；渲染改变几何的GT正确性仍单独受限。'),
      ('FM_P','Shot_high => LowLight | Underexposed_strong','高shot噪声仅能与LowLight或强欠曝同选。'),
      ('FM_D','LEAF_YOLO => Nano & VisDrone_FT & Train640 & NMS','LEAF_YOLO限定Nano、VisDrone_FT、Train640及NMS。'),
      ('FM_D','EndToEnd_NMSFree => YOLO26','无NMS端到端路径只允许YOLO26；本次已测路径仍是NMS。'),
      ('FM_D','COCO_ZeroShot => Train640 & !NoiseAug & !MotionBlurAug & !ExposureAug & !JPEGAug & !HazeAug','零样本权重不宣称额外训练增强；COCO来源的初始化经训练后属于VisDrone_FT。'),
      ('FM_D','SAHI_Slicing => !TTA','切片和TTA在该草案中互斥；未用未验证组合增加经验性能结论。')]
    csv_out(out/'UVL_constraints_explained.csv',[{'domain':d,'constraint':c,'explanation_cn':e,
        'origin':'ORIGINAL_PDF_DEFINITION','physical_or_API_validation':'NOT_UNIVERSALLY_ESTABLISHED'} for d,c,e in explanations])
    sys.path.insert(0,str(HERE.parent/'cv261207'))
    from supervisor_compliance import numeric_attributes
    units={'gain':'无量纲强度乘数','bias':'uint8像素强度偏移','kernel':'原生像素边长','radius':'原生像素半径',
        'sigma':'uint8像素强度标准差','photons':'生成器Poisson强度刻度，非实测光子数','quality':'OpenCV JPEG质量参数',
        'beta':'常深度雾化代理，无量纲','streaks':'条纹个数','factor':'缩小倍率','params_m':'百万参数（草案名义值）',
        'imgsz':'网络输入像素边长','hsv_v':'V通道增益幅度','blur_p':'概率','median_blur_p':'概率','clahe_p':'概率'}
    attributes=[]
    for domain,file in [('FM_P','fm_p.uvl'),('FM_D','fm_d.uvl')]:
        repo=VM36/'feature_models'/file;attrs=numeric_attributes(repo.read_text())
        pdfattrs=numeric_attributes((out/('pdf_'+file)).read_text())
        for (feature,name),value in attrs.items():attributes.append({'domain':domain,'feature':feature,'attribute':name,
            'value':value,'unit_interpretation':units.get(name,'UVL未声明单位'),'unit_origin':'GENERATOR_OR_PARAMETER_SEMANTICS; UVL itself has no unit annotation',
            'PDF_value':pdfattrs.get((feature,name)),'status':'ORIGINAL_PDF_DEFINITION' if pdfattrs.get((feature,name))==value else 'UNVERIFIED_DIFFERENCE',
            'source':str(repo),'source_SHA256':ref(repo)['sha256']})
    csv_out(out/'UVL_numeric_attributes.csv',attributes)
    for domain in ('FM_P','FM_D'):
        model=result[domain]
        text=['  '*r['depth']+r['feature']+' ['+r['relation']+'] '+r['attributes'] for r in model['tree']]
        (out/(domain+'_tree.txt')).write_text('\n'.join(text)+'\n',encoding='utf-8')
    return result,pages


def vm36(out):
    import numpy as np
    sys.path.insert(0,str(HERE.parent/'cv261109'))
    from model import valid_leaves
    design=read(FULL/'MEASURED_DESIGN.json');provenance=read(NORMAL/'METRIC_PROVENANCE.json')
    read(FULL/'EVIDENCE_SOURCES.json');read(NORMAL/'NORMAL_ACCEPTANCE.json')
    pairs=[];conditions={};source_rows=[]
    indices=np.random.default_rng(design['bootstrap_seed']).integers(0,548,(200,548))
    sys.path.insert(0,str(HERE.parent/'cv261207'))
    from gap_analysis import pair_outcome
    sys.path.insert(0,str(VM36));from full20_results import latency_interval_state
    for item in provenance['pairs']:
        f=Path(item['source_folder']);spec=read(f/'configuration.json');metrics=read(f/'metrics.json')
        boot=read(f/'bootstrap.json');timing=read(f/'timing.json');verify=read(f/'verification.json')
        for name in ('configuration.json','metrics.json','bootstrap.json','timing.json','verification.json'):
            if study.sha256(f/name)!=item['SHA256'][name]:raise ValueError('VM36 metadata SHA differs: '+str(f/name))
        if metrics['images']!=548 or verify['status']!='VALIDATED':raise ValueError('VM36 pair not accepted.')
        pid=spec['p']['id'];leaves=spec['p']['leaves'];attrs=spec['p']['attributes']
        if not valid_leaves(leaves):raise ValueError('Illegal measured VM36 condition.')
        conditions.setdefault(pid,{'p':spec['p'],'folders':[]});conditions[pid]['folders'].append(str(f))
        interval=next(x['ci95'] for x in boot['intervals'] if x['size']=='tiny' and x['metric']=='map50_95')
        lat=[float(x) for x in np.quantile(np.asarray([r['pipeline_ms'] for r in timing['samples']])[indices].mean(axis=1),[.025,.975])]
        areas={x['size']:x for x in metrics['per_area']}
        row={'population':'VM36_FOUR_DETECTOR_SUBSET','pair_id':spec['id'],'condition_id':pid,
            'checkpoint':spec['d']['checkpoint'],'resolution':spec['d']['resolution'],'AP50':areas['all']['map50'],
            'AP50_95':areas['all']['map50_95'],'Tiny_AP':areas['tiny']['map50_95'],'Tiny_CI_low':interval[0],'Tiny_CI_high':interval[1],
            'cached_pipeline_mean_ms':timing['mean_pipeline_ms'],'latency_CI_low_ms':lat[0],'latency_CI_high_ms':lat[1],
            'original_rule_state':pair_outcome(interval,lat[1],.1,100),
            'two_sided_latency_state':latency_interval_state(interval,lat,.1,100),
            'observed_execution_status':verify['status'],'images':548,'source_RunID':f.parent.parent.name,
            'source_folder':str(f),'metrics_SHA256':study.sha256(f/'metrics.json'),'configuration_SHA256':study.sha256(f/'configuration.json'),
            'prediction_SHA256':item['SHA256']['predictions.jsonl.gz'],'GT_SHA256':item['SHA256']['ground_truth.json']}
        pairs.append(row);source_rows.append({'pair_id':spec['id'],'source_folder':str(f),'SHA256':item['SHA256']})
    if len(pairs)!=84 or len(conditions)!=21:raise ValueError('The measured population is not 21x4.')
    rows=[]
    for pid,item in conditions.items():
        p=item['p'];a=p['attributes'];leaves=p['leaves'];illum=next(x for x in leaves if x in ('Normal','Overexposed_mild','Overexposed_strong','Underexposed_mild','Underexposed_strong','LowLight'))
        extra=[x for x in leaves if x!='Normal'];single=(len(extra)==1)
        fields={'configuration_id':pid,'selected_leaves':'|'.join(leaves),'illumination':illum,'gain':a[illum]['gain'],'bias':a[illum]['bias'],
          'motion_kernel':'NOT_PRESENT','motion_direction':'NOT_APPLICABLE','defocus_radius':next((v['radius'] for v in a.values() if 'radius' in v),'NOT_PRESENT'),
          'Gaussian_sigma':next((v['sigma'] for v in a.values() if 'sigma' in v),'NOT_PRESENT'),
          'Shot_photons':next((v['photons'] for v in a.values() if 'photons' in v),'NOT_PRESENT'),
          'JPEG_quality':next((v['quality'] for v in a.values() if 'quality' in v),'NOT_PRESENT'),
          'haze_beta':'NOT_PRESENT','rain_streaks':'NOT_PRESENT','downscale_factor':next((v['factor'] for v in a.values() if 'factor' in v),1),
          'rolling_shutter':False,'condition_type':'NORMAL_REFERENCE' if leaves==['Normal'] else 'SINGLE_FACTOR' if single else 'COMPOSITE',
          'selection_role':'Later added Normal reference' if leaves==['Normal'] else 'JPEG endpoint diagnostic' if pid in ('p_59915f60d81c6f17','p_5a06deca286f53b2') else 'Gaussian diagnostic' if pid=='p_9031c68f63a468d1' else 'Inherited purposeful historical context; isolated selection rationale not documented',
          'constraint_satisfaction':'VALID','source_folders':json.dumps(item['folders']),'source_design':str(FULL/'MEASURED_DESIGN.json'),
          'source_design_SHA256':study.sha256(FULL/'MEASURED_DESIGN.json'),'configuration_SHA256':study.sha256(Path(item['folders'][0])/'configuration.json')}
        rows.append(fields)
    csv_out(out/'VM36_21_measured_conditions.csv',rows);csv_out(out/'VM36_84_pair_metrics.csv',pairs)
    settings=[]
    for name,spec in design['detector_settings'].items():
        if ref(Path(spec['path']))['sha256']!=spec['sha256']:raise ValueError('Historical checkpoint SHA changed.')
        settings.append({'setting_id':name,**spec,'training_epochs_meaning':'Configured 100 epochs; selected best epoch not proved',
            'class_protocol':'NATIVE10','head_and_postprocessing':'ONE_TO_MANY_STABLE_CLASS_AWARE_MULTILABEL_NMS',
            'conf':.001,'NMS_IoU':.7,'max_det':3000,'max_nms':30000})
    csv_out(out/'VM36_4_detector_settings.csv',settings)
    save(out/'VM36_source_index.json',source_rows)
    gap=[]
    for pid in conditions:
        selected=[x for x in pairs if x['condition_id']==pid]
        for method,column in [('ORIGINAL_UPPER_LATENCY_RULE','original_rule_state'),('TWO_SIDED_LATENCY_SENSITIVITY','two_sided_latency_state')]:
            states=[x[column] for x in selected];state='COVERED_TESTED_SUBSET' if 'PASS' in states else 'GAP_TESTED_SUBSET' if all(x=='FAIL' for x in states) else 'UNRESOLVED'
            gap.append({'population':'VM36_21x4','condition_id':pid,'criterion':'TinyAP95>=0.10 AND cached_pipeline<=100ms',
                'rule':method,'status':state,'measured_D':4,'source':str(NORMAL/'METRIC_PROVENANCE.json')})
    return rows,pairs,settings,gap


def training(out):
    import yaml
    ordinary=read(BUNDLE/'ordinary_training_args.json')['YOLO26s']
    defaults_path=ROOT/'.venv_rq4/Lib/site-packages/ultralytics/cfg/default.yaml'
    ref(defaults_path);defaults=yaml.safe_load(defaults_path.read_text())
    control_folder=P37/'YOLO26s_seed42_CONTROL';plan=read(control_folder/'protocol.json')
    snapshot=control_folder/'trainer/args.yaml'
    if not snapshot.exists():snapshot=control_folder/'args.yaml'
    runtime=yaml.safe_load(snapshot.read_text()) if snapshot.exists() else {}
    if snapshot.exists():ref(snapshot)
    args=plan['args'];rows=[]
    inactive={'auto_augment','erasing','overlap_mask','mask_ratio','pose','kobj','rle','angle','copy_paste_mode'}
    special={
      'mosaic':'配置1.0，但epochs=10且close_mosaic=10，epoch0关闭；实际全程不使用Mosaic。',
      'close_mosaic':'总轮数10，关闭点epochs-close_mosaic=0；worker.log有Closing dataloader mosaic。',
      'nbs':'64/2=32为稳定阶段累积；前9708个batch按warmup逐渐1→32；10轮1816次更新，末17个microbatch不额外flush。',
      'warmup_epochs':'实际9708个warmup batch，非关闭。',
      'warmup_bias_lr':'bias组从0.1而不是lr0=0.01开始；实际健康日志保存各组LR。',
      'momentum':'AdamW使用beta1=0.937、beta2=0.999，不是SGD momentum状态。',
      'warmup_momentum':'0.8保存在参数中；该版本只对含momentum键的参数组插值，AdamW betas不通过此分支warmup。',
      'amp':'False；FP32训练及保存，未切换FP16/TF32。',
      'workers':'DataLoader workers=0；已验收输入实现使用2个CPU算术线程，不是多进程采样worker。',
      'epochs':'实际10轮，固定epoch10 EMA；不是100轮正式结果。',
      'imgsz':'训练960；研究评价另用640/1280。',
      'max_det':'此处300为内部Ultralytics验证配置；研究评价固定max_det=3000，两者不能冒充同一口径。',
      'hsv_v':'V查表乘数范围约0.6–1.4，不提供加性曝光偏移。',
      'seed':'42/43/44分别执行，两个策略在每seed内同初始化、同数据顺序；seed42 ALL_FIVE仅一份。'}
    for key,value in args.items():
        origin='EXPLICIT_PILOT_OVERRIDE' if key in read(study.RUNS/'vm37_yolo26_rq4_pilot10_authorization_cpu_20261113T212036_50286ea7/pilot10_protocol.json')['shared_training_args'] or key in ('epochs','seed','model','data','project','plots') else 'INHERITED_FROZEN_ARGUMENT_SNAPSHOT'
        actual=special.get(key,'按冻结配置执行；未通过本次数据优化。')
        if key in inactive:actual='该字段属于其他任务或关闭分支；不计为本次detect训练的有效新增增强。'
        if key in ('mixup','copy_paste','cutmix','degrees','shear','perspective','flipud','bgr','dropout','multi_scale') and value==0:actual='0，关闭对应扰动/操作。'
        rows.append({'parameter':key,'EXACT_VALUE':json.dumps(value,ensure_ascii=False),
            'SOURCE_FILE':str(control_folder/'protocol.json'),
            'FROZEN_ORDINARY_VALUE':json.dumps(ordinary.get(key,'NOT_IN_SNAPSHOT')),
            'INSTALLED_8_4_210_DEFAULT':json.dumps(defaults.get(key,'NOT_IN_DEFAULTS')),
            'EXPLICIT_OVERRIDE_OR_LIBRARY_DEFAULT':origin,
            'ORIGINAL_CLI_OR_DEFAULT_ORIGIN':'UNKNOWN; saved args equality does not prove CLI origin',
            'ACTUAL_ARGS_SNAPSHOT_VALUE':json.dumps(runtime.get(key,'NOT_PRESERVED_HERE')),
            'EFFECTIVE_RUNTIME_BEHAVIOR':actual,
            'SCIENTIFIC_RATIONALE':'继承历史基线/已批准pilot并保持对照公平；不宣称最优超参数。',
            'UNRESOLVED_ISSUE_IF_ANY':'Teacher source analysis is8.4.33; runtime is8.4.210, not byte-identical source.'})
    extra={'architecture':'YOLO26s','initialization_SHA256':plan['initialization']['sha256'],
        'training_images':len(plan['train']),'development_images':len(plan['development']),
        'class_order':['pedestrian','people','bicycle','car','van','truck','tricycle','awning-tricycle','bus','motor'],
        'EMA_selection':'FIXED_EPOCH10_EMA','DataLoader_labels':'Frozen versioned derived labels; originals unchanged',
        'precision':'FP32; AMP=False; TF32=False','recovery':'No exact resume; no automatic retries'}
    rows += [{'parameter':k,'EXACT_VALUE':json.dumps(v),'SOURCE_FILE':str(control_folder/'protocol.json'),
        'EXPLICIT_OVERRIDE_OR_LIBRARY_DEFAULT':'FROZEN_PROTOCOL','EFFECTIVE_RUNTIME_BEHAVIOR':json.dumps(v),
        'SCIENTIFIC_RATIONALE':'来源与公平性约束','UNRESOLVED_ISSUE_IF_ANY':''} for k,v in extra.items()]
    albu=[{'transform':name,'probability':p,'status':'ACTIVE' if p else 'DISABLED'} for name,p in
          [('Blur',.01),('MedianBlur',.01),('ToGray',.01),('CLAHE',.01),('RandomBrightnessContrast',0.),('RandomGamma',0.),('ImageCompression',0.)]]
    csv_out(out/'CONTROL_actual_training_args.csv',rows);csv_out(out/'CONTROL_Albumentations.csv',albu)
    for file in (ROOT/'.venv_rq4/Lib/site-packages/ultralytics/data/augment.py',ROOT/'.venv_rq4/Lib/site-packages/ultralytics/engine/trainer.py',
                 PUBLIC/'code/main/training_effects.py',PUBLIC/'code/main/training_data.py',PUBLIC/'code/main/training_health.py',
                 HERE.parent/'vm37_yolo26_rq4/post_cost/input_ops.py',HERE.parent/'vm37_yolo26_rq4/training.py'):
        ref(file,'IMPLEMENTATION')
    log=control_folder/'worker.log';ref(log,'ACTUAL_RUNTIME_LOG')
    text=log.read_text(encoding='utf-8')
    if 'Closing dataloader mosaic' not in text:raise ValueError('Missing actual Mosaic close trace.')
    observed=read(BUNDLE/'environment_observed.json')
    save(out/'training_runtime_audit.json',{'actual_args':args,'ordinary_snapshot':ref(BUNDLE/'ordinary_training_args.json'),
        'Mosaic_closed_trace':True,'trainer_admission':read(control_folder/'trainer_admission.json'),
        'environment':observed,'teacher_Ultralytics_version':'8.4.33','actual_Ultralytics_version':'8.4.210',
        'same_version_as_teacher':False,'Albu_transform_probabilities':albu,
        'source_equivalence_8_4_33_vs_8_4_210':'NOT_CLAIMED','original_cli_default_origin':'NOT_RECOVERABLE_FROM_SAVED_ARGS_ALONE'})
    policy=read(HERE.parent/'vm37_yolo26_rq4/TARGETED_POLICY.json');frozen=read(P38/'policy.json')
    if policy!=read(BUNDLE/'assets/TARGETED_POLICY.json'):raise ValueError('ALL_FIVE original/frozen policy differs.')
    family_rows=[]
    all_events=P37/'YOLO26s_seed42_ALL_FIVE/augmentation_events.jsonl'
    ref(all_events,'ACTUAL_AUGMENTATION_TRACE')
    totals=Counter();variants=Counter();bounds={};kernel=Counter();qualities=Counter();joint=Counter()
    with all_events.open() as stream:
        for line in stream:
            row=json.loads(line);active=[]
            if [e['family'] for e in row['effects']]!=policy['order']:raise ValueError('ALL_FIVE application order changed.')
            for e in row['effects']:
                family=e['family'];totals[(family,'total')]+=1
                if e['active']:
                    totals[(family,'active')]+=1;active.append(family)
                    variants[(family,e.get('variant','not_applicable'))]+=1
                    for k,v in e['parameters'].items():
                        if isinstance(v,(int,float)):
                            b=bounds.setdefault((family,k),[float(v),float(v)]);b[0]=min(b[0],v);b[1]=max(b[1],v)
                    if family=='MotionBlurAug':kernel[e['parameters']['kernel']]+=1
                    if family=='JPEGAug':qualities[e['parameters']['quality']]+=1
            joint[len(active)]+=1
    for family,spec in policy['families'].items():family_rows.append({'family':family,'frozen_probability':spec['probability'],
        'frozen_parameters':json.dumps(spec),'events':totals[(family,'total')],'active':totals[(family,'active')],
        'actual_frequency':totals[(family,'active')]/totals[(family,'total')],
        'observed_parameter_ranges':json.dumps({k[1]:v for k,v in bounds.items() if k[0]==family}),
        'source_policy':str(HERE.parent/'vm37_yolo26_rq4/TARGETED_POLICY.json'),'source_trace':str(all_events)})
    if set(kernel)!={9,11,13,17,19} or any(k in qualities for k in (25,50,75)):raise ValueError('Test-level exclusion mismatch.')
    csv_out(out/'ALL_FIVE_policy_verified.csv',family_rows)
    guided_runs={42:P38/'YOLO26s_seed42_VARIABILITY_GUIDED'}
    for identity in read(P39/'execution_result.json')['completed_seed_RunIDs']:
        folder=P39.parent/identity;seed=read(folder/'seed_run.json')['seed']
        guided_runs[seed]=folder/('YOLO26s_seed'+str(seed)+'_VARIABILITY_GUIDED')
    guided_rows=[]
    for seed,folder in guided_runs.items():
        pp=read(folder/'protocol.json')
        if pp['targeted_recipe']!=frozen:raise ValueError('Frozen Gaussian policy changed by seed.')
        file=folder/'augmentation_events.jsonl';ref(file,'ACTUAL_AUGMENTATION_TRACE');count=active=0;lo,hi=100.,0.
        with file.open() as stream:
            for line in stream:
                e=json.loads(line)['effects'];assert len(e)==1 and e[0]['family']=='NoiseAug'
                e=e[0];count+=1
                if e['active']:
                    assert e['variant']=='gaussian';s=e['parameters']['sigma'];assert 5.5<s<30 and s!=15
                    active+=1;lo=min(lo,s);hi=max(hi,s)
        guided_rows.append({'seed':seed,'policy_SHA256':ref(P38/'policy.json')['sha256'],'events':count,'active':active,
            'actual_probability':active/count,'sigma_min':lo,'sigma_max':hi,'other_four_extra_families':'DISABLED','source_trace':str(file)})
    csv_out(out/'GUIDED_realized_frequencies.csv',guided_rows)
    save(out/'policy_realization_details.json',{'ALL_FIVE_joint_active_family_count':dict(joint),
        'Motion_kernels':dict(kernel),'JPEG_qualities':dict(qualities),'noise_variants':{str(k):v for k,v in variants.items()},
        'no_extra_effect_probability_theory':.5**5,'no_extra_effect_count_actual':joint[0],
        'order':policy['order'],'sampling':'Five independent Bernoulli gates, not uniform FM_P configurations'})
    compare=[{'strategy':'CONTROL','ordinary_augmentation':'共同HSV/几何/1% Albu；Mosaic全程关闭','extra_policy':'NONE',
        'not_affected_by_extra_policy_probability':1.,'training_seeds':'42,43,44','sampling_meaning':'Ordinary training, not no augmentation'},
        {'strategy':'ALL_FIVE','ordinary_augmentation':'相同','extra_policy':'五家族独立p=0.5，噪声子类Gaussian/Shot各0.5',
         'not_affected_by_extra_policy_probability':.5**5,'training_seeds':'42_ONLY','sampling_meaning':'Broad random degradation, not all five always active or legal FM_P-uniform'},
        {'strategy':'VARIABILITY_GUIDED','ordinary_augmentation':'相同','extra_policy':'Gaussian only p=0.25; sigma5.5–30 with exact exclusions',
         'not_affected_by_extra_policy_probability':.75,'training_seeds':'42,43,44','sampling_meaning':'Development-selected fixed policy, not online controller'}]
    csv_out(out/'THREE_STRATEGY_comparison.csv',compare)
    return rows,family_rows,guided_rows


def matched_results(out):
    import numpy as np
    index=read(P39/'joint_result_index.json');values={};rows=[];timings=[]
    for row in index:
        folder=Path(row['source_folder']);metrics=read(folder/'metrics.json');boot=read(folder/'bootstrap.json');trace=read(folder/'trace.json')
        if ref(folder/'metrics.json')['sha256']!=row['metrics_SHA256'] or ref(folder/'bootstrap.json')['sha256']!=row['bootstrap_SHA256']:raise ValueError('Matched evidence SHA mismatch.')
        key=(row['seed'],row['arm'],row['resolution'],row['context']);values[key]=(metrics,boot)
        area={a['size']:a for a in metrics['per_area']}
        interval=next(r['ci95'] for r in boot['intervals'] if r['size']=='tiny' and r['metric']=='map50_95')
        rows.append({'seed':row['seed'],'strategy':row['arm'],'resolution':row['resolution'],'condition':row['context'],
            'AP50':area['all']['map50'],'AP50_95':area['all']['map50_95'],'Tiny_AP':area['tiny']['map50_95'],
            'Tiny_CI_low':interval[0],'Tiny_CI_high':interval[1],'source_RunID':row['source_RunID'],'source_folder':str(folder),
            'checkpoint_SHA256':row['checkpoint_SHA256'],'images':metrics['images'],'metrics_SHA256':row['metrics_SHA256']})
        ms=[r['inference_ms'] for r in trace['per_image']]
        timings.append({'seed':row['seed'],'strategy':row['arm'],'resolution':row['resolution'],'condition':row['context'],
            'mean_prediction_ms':float(np.mean(ms)),'definition':'Preprocess+forward+NMS; excludes decode/effect/hash/output, unlike VM36 cached pipeline',
            'COCO_seconds':trace['COCO_seconds'],'bootstrap_seconds':trace['bootstrap_seconds'],
            'source':str(folder/'trace.json')})
    recomputed=[]
    for res in (640,1280):
        for condition in ('Normal','JPEG_Q25','JPEG_Q75','Gaussian_s15'):
            for size in ('all','tiny'):
                differences=[]
                for seed in (42,43,44):
                    c,g=values[(seed,'CONTROL',res,condition)][0],values[(seed,'VARIABILITY_GUIDED',res,condition)][0]
                    point=lambda x:next(a['map50_95'] for a in x['per_area'] if a['size']==size)
                    differences.append(point(g)-point(c))
                recomputed.append({'resolution':res,'condition':condition,'size':size,'seeds':'42,43,44',
                    'seed42_difference':differences[0],'seed43_difference':differences[1],'seed44_difference':differences[2],
                    'mean':float(np.mean(differences)),'sample_seed_SD':float(np.std(differences,ddof=1)),
                    'scope':'DEVELOPMENT_ONLY / POST_HOC_POLICY_SELECTION','training_seed_significance':'NOT_TESTED'})
    csv_out(out/'VM39_three_seed_summary.csv',recomputed);csv_out(out/'MATCHED_56_configuration_metrics.csv',rows)
    csv_out(out/'MATCHED_prediction_timing.csv',timings)
    for file in ('per_seed_comparisons.csv','degradation_losses.csv','final_acceptance.json','execution_result.json'):ref(P39/file)
    matched_gap=[{'population':'VM37_39_MATCHED_ARMS','seed':r['seed'],'strategy':r['strategy'],'resolution':r['resolution'],
        'condition_id':r['condition'],'criterion':'TinyAP95>=0.10 AND VM36-comparable cached_pipeline<=100ms',
        'status':'GAP_CHANGE_NOT_ESTABLISHED','quality_only_state':'PASS' if r['Tiny_CI_low']>=.1 else 'FAIL' if r['Tiny_CI_high']<.1 else 'UNCERTAIN',
        'latency_state':'NOT_COMPARABLE_TIMING_BOUNDARY','AP_status':'AP_IMPROVEMENT_MEASURED' if r['condition']=='Gaussian_s15' and r['strategy']=='VARIABILITY_GUIDED' else 'AP_MEASURED',
        'source':r['source_folder']} for r in rows]
    return rows,recomputed,matched_gap


def independent_gate(out):
    import cv2
    import numpy as np
    sys.path.insert(0,str(HERE.parent/'vm37_yolo26_rq4'))
    import pilot10
    prior=read(V36RUNS/'vm36_supervisor_reproduction_protocol_v2_20261109T153241_baec0138/DATASET_ADMISSION.json')
    old=read(study.RUNS/'vm37_yolo26_rq4_external_provenance_cpu_20261111T230556_4e6de02a/external_dataset_gate.json')
    history=read(V36RUNS/'vm36_supervisor_reproduction_post80_science_20261110T190438_45783294/INDEPENDENT_ADMISSION_UPDATE.json')
    if not (out/'historical_independence_evidence.json').exists():
        save(out/'historical_independence_evidence.json',{'prior_dataset_gate':prior,'prior_UAVDT_gate':old,'bounded_history_update':history})
    official=[]
    for name,url in [('VisDrone','https://raw.githubusercontent.com/VisDrone/VisDrone-Dataset/master/README.md'),
                     ('UAVDT','https://sites.google.com/view/grli-uavdt/')]:
        if (out/'official_sources.json').exists():
            official=load(out/'official_sources.json');break
        try:
            request=urllib.request.Request(url,headers={'User-Agent':'UAV-GAP source audit'})
            with urllib.request.urlopen(request,timeout=30) as response:data=response.read(2*1024*1024);status=response.status
            file=out/(name+'_official_source.txt');file.write_bytes(data)
            official.append({'name':name,'URL':url,'HTTP_status':status,'retrieved_UTC':datetime.now(timezone.utc).isoformat(),
                'SHA256':study.sha256(file),'local_copy':str(file)})
        except Exception as exc:official.append({'name':name,'URL':url,'status':'FETCH_FAILED','reason':str(exc)})
    if not (out/'official_sources.json').exists():save(out/'official_sources.json',official)
    members=[];issues=[];train_ids=set();val_ids=set();train_hashes=set();val_hashes=set()
    for split in ('train','val','test-dev','test-challenge'):
        print('CPU archive audit:',split,flush=True)
        archive=ROOT/'datasets/VisDrone'/('VisDrone2019-DET-'+split+'.zip')
        source=ref(archive,'READ_ONLY_DATA_ARCHIVE')
        if source['sha256']!=prior['splits'][split]['archive']['sha256']:raise ValueError('Source ZIP changed: '+split)
        with ZipFile(archive) as z:
            images=sorted(n for n in z.namelist() if '/images/' in '/'+n.replace('\\','/') and n.lower().endswith('.jpg'))
            labels={Path(n).stem:n for n in z.namelist() if '/annotations/' in '/'+n.replace('\\','/') and n.endswith('.txt')}
            expected=prior['splits'][split]
            print('  images:',len(images),'annotations:',len(labels),flush=True)
            if split=='test-dev' and (len(images)!=1610 or len(labels)!=1610):
                raise ValueError('Test-dev ZIP membership is not 1610 image/annotation pairs.')
            for name in images:
                if split=='test-challenge':continue
                stem=Path(name).stem;data=z.read(name);digest=hashlib.sha256(data).hexdigest()
                if split=='train':train_ids.add(stem);train_hashes.add(digest)
                if split=='val':val_ids.add(stem);val_hashes.add(digest)
                if split!='test-dev':continue
                image=cv2.imdecode(np.frombuffer(data,np.uint8),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
                if image is None:issues.append({'image':stem,'issue':'DECODE_FAILED'});continue
                height,width=image.shape[:2]
                annotation=z.read(labels[stem]) if stem in labels else None
                if annotation is None:issues.append({'image':stem,'issue':'MISSING_GT'})
                count=0;classes=Counter()
                for lineno,line in enumerate(annotation.decode('utf-8-sig').splitlines() if annotation else [],1):
                    values=[v.strip() for v in line.split(',') if v.strip()]
                    try:
                        x,y,w,h=map(float,values[:4]);score,category=int(values[4]),int(values[5]);classes[category]+=1;count+=1
                        if w<=0 or h<=0 or x<0 or y<0 or x+w>width or y+h>height or category not in range(12):
                            issues.append({'image':stem,'line':lineno,'issue':'INVALID_GEOMETRY_OR_CLASS','raw':line})
                    except (ValueError,IndexError):issues.append({'image':stem,'line':lineno,'issue':'PARSE_ERROR','raw':line})
                members.append({'image_id':stem,'ZIP_member':name,'image_SHA256':digest,'annotation_SHA256':hashlib.sha256(annotation).hexdigest() if annotation else '',
                    'width':width,'height':height,'dtype':str(image.dtype),'channels':'BGR','decode_flag':int(cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION),
                    'GT_rows':count,'categories':json.dumps(dict(classes)),'train_ID_overlap':stem in train_ids,'train_byte_overlap':digest in train_hashes,
                    'val_ID_overlap':stem in val_ids,'val_byte_overlap':digest in val_hashes})
    csv_out(out/'testdev_CPU_membership.csv',members)
    save(out/'testdev_annotation_issues.json',issues)
    uav=ROOT/'datasets/uavdt';meta=read(uav/'meta.json')
    for name in ('README.md','LICENSE.md'):ref(uav/name,'LOCAL_DATA_USE_STATEMENT')
    files=[f for f in uav.rglob('*') if f.is_file()]
    candidate_gt=[str(f) for f in files if f.suffix.lower()=='.txt' and any(k in f.name.lower() for k in ('gt_whole','gt_ignore','gt.txt','ignore','groundtruth'))]
    save(out/'UAVDT_bounded_inventory.json',{'root':str(uav),'files_count':len(files),'classes':[x['title'] for x in meta['classes']],
        'original_DET_ignore_GT_candidates':candidate_gt,'annotation_JSON_count':sum(f.suffix=='.json' and 'ann' in f.parts for f in files),
        'scope':'Only this existing converted directory; not a full-disk absence claim.'})
    current=pilot10.available(AUTH);ledger=read(AUTH/'authorization.json')
    charged=[load(p) for p in pilot10.charge_files(AUTH)]
    budget={'authorization':ref(AUTH/'authorization.json'),'total_seconds':ledger['max_cumulative_GPU_worker_wall_seconds'],
        'charged_seconds':sum(x['charged_seconds'] for x in charged),'remaining_seconds':current,'absolute_expiry_UTC':ledger['absolute_expiry_UTC'],
        'new_charge_seconds':0,'allocation_reset':False,'charges_SHA256':[ref(p) for p in pilot10.charge_files(AUTH)]}
    reference_seconds=10808.362
    linear=reference_seconds/(16*548)*(12*1610)
    estimate={'reference_measured_seconds':reference_seconds,'reference_cells':16,'reference_images_per_cell':548,
        'candidate_primary_cells':12,'candidate_images_per_cell':1610,'candidate_image_evaluations':19320,
        'linear_extrapolation_seconds_NOT_MEASURED':linear,'planning_upper_seconds_NOT_MEASURED':linear*1.35+1800,
        'warning':'Different GT density and CPU COCO/Bootstrap cost are unmeasured; no reliable complete-pair admission from simple image scaling.',
        'fits_current_budget_under_this_conservative_envelope':linear*1.35+1800<=current}
    gates=[{'dataset':'VisDrone2019-DET-test-dev','status':'DATASET_BLOCKED',
        'source':'Current ZIP SHA matches the accepted local source certificate; not an independently signed official hash.',
        'research_use':'SUPPORTED_BY_OFFICIAL_PROJECT_TESTDEV_PUBLICATION_STATEMENT; no blanket redistribution claim',
        'annotation_status':'CPU_VALIDATED' if len(members)==1610 and not issues else 'ANNOTATION_ISSUES',
        'images_decoded':len(members),'annotation_issues':len(issues),'native10':'COMPATIBLE_WITH_EXISTING_VISDRONE_RULES',
        'current_ID_and_byte_overlap_train_val':sum(any(x[k] for k in ('train_ID_overlap','train_byte_overlap','val_ID_overlap','val_byte_overlap')) for x in members),
        'new_checkpoint_training_trace':'VM37–39 train6471/val548 manifests, same fixed epoch10; no test-dev gradient inputs in these preserved plans.',
        'historical_selection_and_research_exposure':'UNVERIFIED; prior records explicitly retain missing historical/manual-use evidence',
        'scene_independence':'UNVERIFIED; image SHA disjointness is not scene independence',
        'reason':'Original independent-test gate remains unresolved; no inference.'},
        {'dataset':'UAVDT_existing_conversion','status':'DATASET_BLOCKED','research_use':'SUPPORTED_RESEARCH_ONLY_STATEMENT',
         'native10':'INCOMPATIBLE_VEHICLE_LABEL_SPACE','original_DET_GT':bool(candidate_gt),
         'original_ignore_and_split_semantics':'UNVERIFIED','scene_independence':'UNVERIFIED',
         'reason':'Converted bus/car/truck/vehicle labels do not establish original DET/MOT, ignore or native10 semantics.'},
        {'dataset':'VisDrone-train/val','status':'DATASET_BLOCKED','reason':'Training and policy-development exposure; not an independent image set.'},
        {'dataset':'VisDrone-test-challenge','status':'DATASET_BLOCKED','reason':'Official project states annotations unavailable; cannot invent GT.'},
        {'dataset':'HazyDet_existing','status':'DATASET_BLOCKED','reason':'Existing three-vehicle task is not the frozen VisDrone native10 UAV task; no silent label substitution.'}]
    result={'status':'DATASET_BLOCKED','candidates':gates,'GPU_validation_executed':False,'new_training_executed':False,
        'primary_design':{'seeds':[42,43,44],'strategies':['CONTROL','VARIABILITY_GUIDED'],'resolution':1280,
            'conditions':['Normal','Gaussian_s15'],'primary_endpoint':'Gaussian AP50:95 GUIDED minus CONTROL','cells':12},
        'objective_admission':{'A_independent_dataset':'FAILED','B_existing_scope':'NOT_ACTIVATED; conditional VM40 request does not bypass failed dataset/resource gates',
            'C_complete_cost_and_margin':'NOT_ADMITTED','D_thermal':'NOT_REACHED; VM39 89C peaks are a serious margin concern, not fresh clearance',
            'E_measured_complete_population_cost':'UNMEASURED; extrapolation only'},
        'budget':budget,'cost_screening':estimate,'sources':official,'GPU_seconds':0}
    save(out/'independent_data_admission.json',result)
    return result


def read_csv(file):
    with Path(file).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def gap_table(out):
    pairs=read_csv(out/'VM36_84_pair_metrics.csv');matched=read_csv(out/'MATCHED_56_configuration_metrics.csv')
    result=[];counts={}
    for rule,column in [('ORIGINAL_UPPER_LATENCY_RULE','original_rule_state'),('TWO_SIDED_LATENCY_SENSITIVITY','two_sided_latency_state')]:
        states=[]
        for pid in dict.fromkeys(r['condition_id'] for r in pairs):
            values=[r[column] for r in pairs if r['condition_id']==pid]
            status='COVERED_TESTED_SUBSET' if 'PASS' in values else 'GAP_TESTED_SUBSET' if set(values)=={'FAIL'} else 'UNRESOLVED'
            states.append(status)
            result.append({'population':'VM36_21x4','condition_id':pid,'rule':rule,'status':status,'measured_D':4,
                'criterion':'TinyAP95>=0.10 AND VM36 cached_pipeline<=100ms','source':str(NORMAL/'METRIC_PROVENANCE.json')})
        counts[rule]=dict(Counter(states))
    for r in matched:
        result.append({'population':'VM37_39_MATCHED_ARMS','condition_id':r['condition'],'seed':r['seed'],
            'strategy':r['strategy'],'resolution':r['resolution'],'status':'GAP_CHANGE_NOT_ESTABLISHED',
            'quality_only_state':'PASS' if float(r['Tiny_CI_low'])>=.1 else 'FAIL' if float(r['Tiny_CI_high'])<.1 else 'UNCERTAIN',
            'latency_state':'NOT_COMPARABLE_TIMING_BOUNDARY','criterion':'TinyAP95>=0.10 AND VM36 cached_pipeline<=100ms',
            'source':r['source_folder']})
    result.append({'population':'INDEPENDENT_PRIMARY_12_CELLS','status':'NOT_EVALUATED','source':str(out/'independent_data_admission.json')})
    csv_out(out/'gap_status_reassessment.csv',result)
    save(out/'gap_summary.json',{'VM36_21_conditions':counts,'VM36_20_conditions':{'GAP_TESTED_SUBSET':17,'COVERED_TESTED_SUBSET':3},
        'matched_gap_change':'NOT_ESTABLISHED','independent':'NOT_EVALUATED','full_feature_space':'NOT_MEASURED'})


def provenance(out):
    if (out/'independent_collected_sources.json').exists():SOURCES.update(load(out/'independent_collected_sources.json'))
    for path in (ROOT/'datasets/research_archive/supervisor/Research_plan_Cunwu_UAV_gap.pdf',
            VM36/'feature_models/fm_p.uvl',VM36/'feature_models/fm_d.uvl',VM36/'validate_models.py',
            HERE.parent/'cv261109/flamapy_select.py',HERE.parent/'cv261116/flamapy_count.py',
            FULL/'MEASURED_DESIGN.json',FULL/'EVIDENCE_SOURCES.json',NORMAL/'METRIC_PROVENANCE.json',NORMAL/'NORMAL_ACCEPTANCE.json',
            P39/'per_seed_comparisons.csv',P39/'degradation_losses.csv',P39/'execution_result.json',P39/'final_acceptance.json',
            P39/'joint_result_index.json',BUNDLE/'ordinary_training_args.json',BUNDLE/'environment_observed.json',
            HERE.parent/'vm37_yolo26_rq4/TARGETED_POLICY.json',P38/'policy.json',
            PUBLIC/'code/main/training_effects.py',PUBLIC/'code/main/training_data.py',PUBLIC/'code/main/training_health.py',
            VM36/'full_development.py',VM36/'joint_batch.py',HERE.parent/'vm37_yolo26_rq4/post_cost/evaluate.py'):
        ref(path)
    raw=[]
    for row in load(out/'VM36_source_index.json'):
        folder=Path(row['source_folder'])
        for name,value in row['SHA256'].items():
            if name in ('predictions.jsonl.gz','ground_truth.json'):
                raw.append({'path':str(folder/name),'sha256':value,'status':'ACCEPTED_SOURCE_MANIFEST_REFERENCE; not rehashed in VM40'})
            elif (folder/name).is_file():
                if ref(folder/name)['sha256']!=value:raise ValueError('Historical metadata changed.')
    for row in load(P39/'joint_result_index.json'):
        folder=Path(row['source_folder'])
        for name in ('metrics.json','bootstrap.json','trace.json'):ref(folder/name)
        for name in ('metrics','bootstrap'):
            if SOURCES[str((folder/(name+'.json')).resolve())]['sha256']!=row[name+'_SHA256']:
                raise ValueError('Matched source SHA mismatch.')
    arm_folders=[P37/'YOLO26s_seed42_CONTROL',P37/'YOLO26s_seed42_ALL_FIVE',P38/'YOLO26s_seed42_VARIABILITY_GUIDED']
    for identity in load(P39/'execution_result.json')['completed_seed_RunIDs']:
        root=P39.parent/identity;seed=load(root/'seed_run.json')['seed']
        arm_folders.extend(root/('YOLO26s_seed'+str(seed)+'_'+arm) for arm in ('CONTROL','VARIABILITY_GUIDED'))
    identities=[]
    for folder in arm_folders:
        plan=read(folder/'protocol.json');initial=plan['initialization'];path=Path(initial['path'])
        if ref(path)['sha256']!=initial['sha256']:raise ValueError('Initialization source SHA changed.')
        for name in ('augmentation_events.jsonl','worker.log','trainer_admission.json','trainer/args.yaml','safety_supervisor.json','worker_result.json'):
            if (folder/name).is_file():ref(folder/name,'ACTUAL_RUNTIME_EVIDENCE')
        identities.append({'arm':folder.name,'source_RunID':folder.parent.name,'initialization_path':str(path),
            'initialization_SHA256':initial['sha256'],'actual_protocol_SHA256':study.sha256(folder/'protocol.json'),
            'train':len(plan['train']),'development':len(plan['development']),'expected_counters':json.dumps(plan['expected'])})
    csv_out(out/'initialization_sources.csv',identities)
    for row in read_csv(out/'VM36_4_detector_settings.csv'):ref(Path(row['path']),'REHASHED_CHECKPOINT')
    for path in HERE.glob('*.py'):ref(path,'VM40_SOURCE')
    old=load(out/'source_SHA256.json');changed=[];checked=0
    for relative,value in old.items():
        if relative.startswith('experiments/'+EXPERIMENT+'/'):continue
        path=PUBLIC/relative;checked+=1
        if not path.is_file() or study.sha256(path)!=value:changed.append(relative)
    if changed:raise ValueError('Pre-existing public code changed: '+str(changed))
    save(out/'preservation_check.json',{'preexisting_public_Python_files_checked':checked,'changed':changed,
        'scope':'All preexisting .py files captured by RunStore; VM40 files excluded. Frozen metadata/checkpoint inputs separately rehashed.',
        'historical_raw_prediction_archives':'Manifest SHA references retained; no redundant full reread claimed.',
        'source_UVL_SHA256':{r:study.sha256(VM36/'feature_models'/r) for r in ('fm_p.uvl','fm_d.uvl')},
        'old_SHA_manifests_modified':False,'GPU_calls':0})
    save(out/'source_provenance.json',{'RunID':out.name,'files':list(SOURCES.values()),'accepted_raw_references':raw,
        'source_integrity':'Explicit rehashed inputs distinguished from accepted raw-manifest references',
        'official_sources':load(out/'official_sources.json')})


def validate_tables(out):
    import numpy as np
    checks={}
    pairs=read_csv(out/'VM36_84_pair_metrics.csv');conditions=read_csv(out/'VM36_21_measured_conditions.csv')
    matched=read_csv(out/'MATCHED_56_configuration_metrics.csv');summary=read_csv(out/'VM39_three_seed_summary.csv')
    checks['VM36_84_unique_pairs']=len(pairs)==84 and len({r['pair_id'] for r in pairs})==84
    checks['VM36_all_548_images']=all(int(r['images'])==548 for r in pairs)
    checks['VM36_21_conditions_four_D_each']=len(conditions)==21 and all(v==4 for v in Counter(r['condition_id'] for r in pairs).values())
    checks['exact_condition_breakdown']=Counter(r['condition_type'] for r in conditions)==Counter(COMPOSITE=17,SINGLE_FACTOR=3,NORMAL_REFERENCE=1)
    keys=[(r['seed'],r['strategy'],r['resolution'],r['condition']) for r in matched]
    checks['matched56_unique']=len(matched)==56 and len(set(keys))==56
    checks['ALL_FIVE_seed42_only']=all(r['seed']=='42' for r in matched if r['strategy']=='ALL_FIVE')
    checks['matched_native_development_size']=all(int(r['images'])==548 for r in matched)
    summary_ok=True
    for r in summary:
        dif=[]
        for seed in ('42','43','44'):
            subset={x['strategy']:x for x in matched if x['seed']==seed and x['resolution']==r['resolution'] and x['condition']==r['condition']}
            metric='AP50_95' if r['size']=='all' else 'Tiny_AP'
            dif.append(float(subset['VARIABILITY_GUIDED'][metric])-float(subset['CONTROL'][metric]))
        summary_ok &= np.isclose(float(r['mean']),np.mean(dif),rtol=0,atol=1e-14) and np.isclose(float(r['sample_seed_SD']),np.std(dif,ddof=1),rtol=0,atol=1e-14)
    checks['all_seed_means_and_sample_SD_reproduced']=bool(summary_ok)
    reference=read_csv(P39/'per_seed_comparisons.csv')
    checks['matched_differences_equal_accepted_VM39']=all(
        math.isclose(float(r['GUIDED_minus_CONTROL']),next(float(s['seed'+r['seed']+'_difference']) for s in summary
            if s['resolution']==r['resolution'] and s['condition']==r['context'] and s['size']==r['size']),rel_tol=0,abs_tol=1e-14)
        for r in reference if r['metric']=='map50_95' and r['size'] in ('all','tiny'))
    checks['guided_all_realized_64710']=all(int(r['events'])==64710 for r in read_csv(out/'GUIDED_realized_frequencies.csv'))
    uv=load(out/'UVL_validation.json')
    checks['UVL_PDF_signature_equal']=all(x['PDF_tree_attributes_constraints_equal'] for x in uv.values())
    checks['SAT_no_dead_features']=all(x['satisfiable'] and not x['dead_features'] for x in uv.values())
    gap=read_csv(out/'gap_status_reassessment.csv')
    checks['matched_Gap_not_fabricated']=all(r['status']=='GAP_CHANGE_NOT_ESTABLISHED' for r in gap if r['population']=='VM37_39_MATCHED_ARMS')
    gate=load(out/'independent_data_admission.json')
    checks['no_GPU_on_blocked_gate']=gate['status']=='DATASET_BLOCKED' and not gate['GPU_validation_executed'] and gate['GPU_seconds']==0
    checks['old_authorization_no_reset']=not gate['budget']['allocation_reset'] and gate['budget']['new_charge_seconds']==0
    if not all(checks.values()):raise ValueError('Audit validation failed: '+str(checks))
    save(out/'table_validation.json',{'status':'PASS','checks':checks,'checks_count':len(checks)})


def verify_completed(out):
    manifest=load(out/'output_checksums.json')
    for item in manifest['files']:
        path=Path(item['path'])
        if not path.is_file() or study.sha256(path)!=item['sha256']:raise ValueError('Output SHA mismatch: '+str(path))
    print('VERIFIED',out.name,len(manifest['files']),'files; GPU=0',flush=True)


def finalize(out):
    if (out/'final_acceptance.json').exists():verify_completed(out);return
    import importlib.util
    start=time.perf_counter();study.runtime(out,cuda=False)
    for name in ('vm36','training','matched_results','independent_gate'):
        stage=load(out/(name+'_stage.json'))
        if stage['status']!='COMPLETED':raise ValueError('Stage incomplete: '+name)
    gap_table(out);validate_tables(out);provenance(out)
    measured=sum(load(out/(name+'_stage.json'))['CPU_wall_seconds'] for name in ('vm36','training','matched_results','independent_gate'))
    measured+=load(out/'flamapy_execution.json')['CPU_wall_seconds']
    resource={'measured_successful_CPU_stages_seconds':measured,'new_GPU_seconds':0,'new_GPU_ledger_charge_seconds':0,
        'known_failed_CPU_invocation_wall_seconds':5.0780819,
        'complete_CPU_session_wall_seconds':'NOT_UNIFORMLY_TIMED; no reconstructed total claimed',
        'initial_interrupted_collection':'UVL and three saved stage outputs retained; missing independent result was completed without rerunning accepted stages.',
        'audit_parser_fix':'ZIP paths rooted at images/ and annotations/ now handled; image membership count asserted. No image/GT change.'}
    spec=importlib.util.spec_from_file_location('vm40_reporting',HERE/'reporting.py')
    reporting=importlib.util.module_from_spec(spec);spec.loader.exec_module(reporting)
    names=reporting.charts(out);report_paths=reporting.reports(out,sys.modules[__name__],resource)
    reporting.page(out,names,report_paths)
    budget=load(out/'independent_data_admission.json')['budget']
    for item in budget['charges_SHA256']+[budget['authorization']]:
        if study.sha256(Path(item['path']))!=item['sha256']:raise ValueError('Authorization ledger changed during CPU audit.')
    save(out/'budget_preservation.json',{'status':'UNCHANGED','before':budget,'after_rehashed_same':True,'new_GPU_charge_seconds':0})
    resource['finalization_CPU_wall_seconds']=time.perf_counter()-start
    save(out/'resource_usage.json',resource)
    (out/'diary.md').write_text('# VM40\n\n原PDF/UVL、21条件、四D及三策略来源核验；复算保存的三seed均值/SD，不重跑YOLO或COCO。\n\n'
        'test-dev1610图CPU解码/GT通过，历史使用与场景隔离仍未验证；UAVDT原始协议/类别阻塞。独立GPU分支停止，其余CPU任务完成。\n\n'
        '确认Gaussian收益、Normal代价；新旧计时边界不同，不宣布Gap缩小。ZIP审计修复根路径解析，不改变原图/GT。\n\n'
        '旧36小时授权未重置、未新增计费；来源SHA与历史代码保护已核验。\n',encoding='utf-8')
    run=load(out/'run.json');run.update(status='COMPLETE_WITH_LIMITATIONS',finished_UTC=datetime.now(timezone.utc).isoformat(),
        CPU_audit='COMPLETED',independent_validation='DATASET_BLOCKED',GPU_seconds=0,Gap_change='NOT_ESTABLISHED')
    (out/'run.json').write_text(json.dumps(run,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (out/'execution.log').open('a',encoding='utf-8') as stream:
        stream.write('Finalized saved CPU evidence; tables validated; independent GPU branch blocked; no historical experiment rerun.\n')
    save(out/'final_acceptance.json',{'status':'CPU_AUDIT_COMPLETE_WITH_LIMITATIONS','RunID':out.name,'source_inputs':'SHA_VERIFIED_WITH_SCOPE_IN_PROVENANCE',
        'UVL':'PDF_SIGNATURE_EQUAL_SAT_VALIDATED','VM36_measured_pairs':84,'matched_saved_evaluations':56,
        'independent':'DATASET_BLOCKED_NOT_RUN','Gap_change':'NOT_ESTABLISHED','GPU_seconds':0,
        'reports':[str(p) for p in report_paths],'plots':names,'CPU_resource_record':str(out/'resource_usage.json')})
    files=[*sorted(p for p in out.iterdir() if p.is_file()),*sorted((out/'plots').iterdir()),*report_paths,*HERE.glob('*.py')]
    manifest=[{'path':str(p),'sha256':study.sha256(p),'bytes':p.stat().st_size} for p in files if p.name!='output_checksums.json']
    save(out/'output_checksums.json',{'RunID':out.name,'files':manifest})
    from main.visualization import build_run_index
    build_run_index(out.parent,EXPERIMENT)
    verify_completed(out)
    print('CPU_AUDIT_COMPLETE_WITH_LIMITATIONS',out,flush=True)


def resolve(identity):
    if Path(identity).name!=identity or not identity.startswith(EXPERIMENT+'_'):raise ValueError('Invalid VM40 RunID.')
    return (RUNS/identity).resolve(strict=True)


def audit_new():
    from main.pipeline import new_run
    out,_=new_run({'output_root':str(RUNS)},EXPERIMENT,'audit');study.runtime(out,cuda=False)
    try:
        t=time.perf_counter();uvl(out)
        save(out/'UVL_stage.json',{'CPU_wall_seconds':time.perf_counter()-t,'GPU_seconds':0})
        for name in ('vm36','training','matched_results','independent_gate'):
            t=time.perf_counter();globals()[name](out)
            save(out/(name+'_stage.json'),{'status':'COMPLETED','CPU_wall_seconds':time.perf_counter()-t,'GPU_seconds':0})
        save(out/'independent_collected_sources.json',SOURCES)
        finalize(out)
    except BaseException:
        save(out/'audit_failure.json',{'traceback':traceback.format_exc(),'GPU_seconds':0})
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='CPU-only source audit; no inference/training.')
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--audit',action='store_true')
    choice.add_argument('--complete',metavar='RunID')
    choice.add_argument('--verify',metavar='RunID')
    args=parser.parse_args()
    if args.audit:audit_new()
    elif args.complete:finalize(resolve(args.complete))
    else:verify_completed(resolve(args.verify))
