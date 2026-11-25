"""Assemble one VM48 package and run CPU validation without GPU work."""
import argparse,ast,csv,hashlib,json,os,shutil,subprocess,sys,time,zipfile
from pathlib import Path
from datetime import datetime,timezone

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'main/code'))
RUNS=ROOT/'datasets/local_runs'/HERE.name
BASE_RUN=ROOT/'datasets/local_runs/vm37_yolo26_rq4/vm37_yolo26_rq4_portable_export_cpu_20261113T115344_ade9b69b'
BASE_ZIP=BASE_RUN/'VM37_portable_training.zip'
PDF=ROOT/'datasets/research_archive/supervisor/Research_plan_Cunwu_UAV_gap.pdf'
VM45=ROOT/'datasets/local_runs/vm45_motion_haze_gap_learning/vm45_motion_haze_gap_learning_execution_20261117T172830_0d9cb1ee'
VM47=ROOT/'datasets/local_runs/vm47_gaussian_jpeg_factorial/vm47_gaussian_jpeg_factorial_execution_20261117T190442_7166b669'
SOURCES={}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def ref(path):
    p=Path(path).resolve(strict=True);r={'path':str(p),'SHA256':sha(p),'bytes':p.stat().st_size};SOURCES[str(p)]=r;return r


def read(path):ref(path);return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path,value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)


def csv_write(path,rows):
    with path.open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)


def extract_backend(package):
    original=HERE.parent/'experiments_test/vm43_external_pilot/backend.py';text=original.read_text(encoding='utf-8');tree=ast.parse(text)
    blocks=[];identities={}
    for name in ('validate_predictions','timed_prediction','RealDetector'):
        node=next(n for n in tree.body if getattr(n,'name',None)==name)
        section=ast.get_source_segment(text,node);blocks.append(section)
        identities[name]=hashlib.sha256(section.encode()).hexdigest()
    preamble='"""Accepted VM43 cached boundary; source sections retained verbatim."""\nimport hashlib,json,math,time\nfrom pathlib import Path\nfrom common48 import sha\n\n'
    target=package/'main/experiments/vm48_formal_rq4/cached_backend.py'
    target.write_text(preamble+'\n\n'.join(blocks)+'\n',encoding='utf-8')
    save(package/'cached_backend_provenance.json',{'original':ref(original),'unchanged_section_SHA256':identities,
        'adaptation':'Resolution640/1280 preprocessing wrapper only, in evaluation48.py; predictor/NMS and cached timer sections unchanged'})


def metadata(package,head):
    version='VM48_FORMAL_RQ4_V1';initial=read(package/'portable_initializations.json');by_init={(r['architecture'],r['seed']):r for r in initial}
    rows=[]
    for arch in ('YOLOv8s','YOLO26s'):
        for seed in (42,43,44):
            for strategy in ('CONTROL','ALL_FIVE_TARGETED'):
                rows.append({'arm_id':f'{arch}_seed{seed}_{strategy}','architecture':arch,'seed':seed,'strategy':strategy,'role':'PRIMARY',
                   'initialization_asset':by_init[arch,seed]['asset'],'initialization_SHA256':by_init[arch,seed]['sha256'],
                   'COCO_source_SHA256':by_init[arch,seed]['COCO_source_SHA256'],'recipe_asset':None if strategy=='CONTROL' else 'vm48_recipes/ALL_FIVE_TARGETED.json',
                   'reference_arm':f'{arch}_seed{seed}_CONTROL','epochs':100,'selection':'FIXED_EPOCH100_EMA','status':'NOT_EVALUATED'})
    with (package/'YOLOV8_ABLATIONS_FROZEN.csv').open(encoding='utf-8-sig') as f:old=list(csv.DictReader(f))
    assert len(old)==5 and all(r['seed']=='42' for r in old)
    for r in old:
        strategy=r['arm'];rows.append({'arm_id':'YOLOv8s_seed42_'+strategy,'architecture':'YOLOv8s','seed':42,'strategy':strategy,'role':'ABLATION',
           'initialization_asset':by_init['YOLOv8s',42]['asset'],'initialization_SHA256':by_init['YOLOv8s',42]['sha256'],'COCO_source_SHA256':by_init['YOLOv8s',42]['COCO_source_SHA256'],
           'recipe_asset':'vm48_recipes/'+strategy+'.json','reference_arm':'YOLOv8s_seed42_CONTROL','epochs':100,'selection':'FIXED_EPOCH100_EMA','status':'NOT_EVALUATED'})
    csv_write(package/'training_matrix.csv',rows)
    policy=read(package/'assets/TARGETED_POLICY.json');recipe_root=package/'vm48_recipes';recipe_root.mkdir()
    full={**policy,'version':'VM48_ALL_FIVE_TARGETED_ALIAS_V1','parent_policy_SHA256':sha(package/'assets/TARGETED_POLICY.json'),
          'label_definition':'Same VM37_ALL_FIVE_V1 five-family stochastic strategy; static-gap targeted, not learned adaptive Gap-guided'}
    save(recipe_root/'ALL_FIVE_TARGETED.json',full)
    strategies={'CONTROL':{'recipe_asset':None,'added_families':[]},'ALL_FIVE_TARGETED':{'recipe_asset':'vm48_recipes/ALL_FIVE_TARGETED.json','added_families':list(policy['families'])}}
    for r in old:
        name=r['arm'];family=r['targeted_families'];single={**full,'version':'VM48_'+name+'_V1','families':{family:policy['families'][family]},'parent_strategy':'ALL_FIVE_TARGETED'}
        save(recipe_root/(name+'.json'),single);strategies[name]={'recipe_asset':'vm48_recipes/'+name+'.json','added_families':[family]}
    effects=package/'main/code/main/training_effects.py';tree=ast.parse(effects.read_text(encoding='utf-8'))
    excluded=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TEST_VALUES' for t in n.targets)))
    save(package/'augmentation_protocol.json',{'id':'VM48_AUGMENTATION_V1','parent_policy_SHA256':sha(package/'assets/TARGETED_POLICY.json'),
        'strategies':strategies,'order':policy['order'],'family_specs':policy['families'],'exact_test_values_excluded':excluded,
        'ordinary':'Inherited HSV/geometry/Mosaic and observed installed Albumentations; all arms share them',
        'ordinary_activation':'Observed per-call logging, including after Mosaic closes; CPU logging parity required',
        'native_rng':'SHA256(seed|epoch|image_id|occurrence|family), independent family gates and strengths',
        'scope':'Synthetic training capabilities; no camera/weather calibration or proven empirical benefit'})
    frozen=read(package/'formal_evaluation_manifest.json');assert len(frozen['contexts'])==21 and frozen['resolutions']==[640,1280]
    groups={'NoiseAug':[],'MotionBlurAug':[],'ExposureAug':[],'JPEGAug':[],'HazeAug':[]}
    for c in frozen['contexts']:
        leaves=c['p']['leaves']
        for family,prefixes in [('NoiseAug',('Gauss_','Shot_')),('MotionBlurAug',('Motion_',)),('ExposureAug',('Overexposed_','Underexposed_','LowLight')),('JPEGAug',('JPEG_',)),('HazeAug',('Haze_',))]:
            if any(any(f.startswith(pre) for pre in prefixes) for f in leaves):groups[family].append(c['id'])
    evaluation={**frozen,'version':'VM48_EVALUATION_V1','primary_P_sample':'Original21 purposeful contexts unchanged; no outcome-based selection',
        'timing_protocol':'VM41_MATCHED_CACHED_GAP_V1','legacy_latency_incompatible':'Old portable.evaluate.py prediction-only timing must not be used for Gap comparisons',
        'primary_training_cells':504,'ablation_cells':210,'all_cells':714,'family_context_groups':groups,
        'primary_endpoint':{'definition':'CONTROL empirical Gap count minus ALL_FIVE_TARGETED Gap count on21 frozen P, per training seed',
           'D_set_per_seed_strategy':'YOLOv8s/YOLO26s × Infer640/1280; four measured candidates',
           'Tiny_AP50_95_min':.10,'cached_pipeline_mean_ms_max':100,'three_state':'PASS/FAIL/UNCERTAIN from fixed interval rule',
           'bounds':'Gap count [knownGAP,knownGAP+UNCERTAIN]; paired reduction bounds and200 shared-image draw intervals',
           'population_change':'NOT_ESTIMATED_PURPOSEFUL_SAMPLE','full_FM_P_gap':'UNKNOWN','training_seed_inference':'Three matched seed differences; image bootstrap is not training-seed significance'},
        'family_interpretation':'Configuration-subset summaries, not isolated family causal effects; Motion/Haze contrasts not estimable in primary21 sample',
        'independent_confirmation':'Blocked until independent dataset/source/scene/selection admission; development results cannot satisfy supervisor final-test requirement'}
    save(package/'frozen_evaluation_protocol.json',evaluation)
    save(package/'vm48_protocol.json',{'version':version,'training_matrix':rows,'parent_formal_protocol_SHA256':sha(package/'formal100_protocol.json'),
        'data_manifest_SHA256':sha(package/'portable_dataset_manifest.json'),'initializations_SHA256':sha(package/'portable_initializations.json'),
        'primary_arms':12,'ablation_arms':5,'ablation_seed_definition':'Original YOLOV8_ABLATIONS_FROZEN.csv, seed42 only',
        'formal_epochs':100,'training_resolution':1280,'workers':0,'exact_resume_validated':False,'source_commit':head,
        'author_review':'Engineering package and factual evidence only; not an approved manuscript'})
    args=read(package/'ordinary_training_args.json');shared=read(package/'formal100_protocol.json')['shared_training_args'];manifest=[]
    for r in rows:
        a={**args[r['architecture']],**shared,'epochs':100,'seed':r['seed'],'model':r['initialization_asset'],'project':'RUNTIME_PRIVATE_ARM_DIRECTORY','data':'RUNTIME_PRIVATE_DATASET_YAML'}
        manifest.append({**r,'full_args_JSON':json.dumps(a,sort_keys=True),'training_images':6471,'development_images':548,'planned_anchor_exposures':647100,
          'planned_batches':323600,'planned_optimizer_EMA_updates':10917,'pending_microbatches':25,'FP32_epoch_snapshots':100,'actual_training_exposures':None,'actual_GPU_seconds':None,
          'actual_Albumentations':'NOT_EVALUATED_TRAINING; observed CPU pipeline and runtime activation logs required'})
    csv_write(package/'training_config_manifest.csv',manifest)
    benchmark={'id':'VM48_TARGET_FULL_EPOCH_BENCHMARK_V1','architectures':['YOLO26s','YOLOv8s'],'separate_per_architecture':True,'seed':42,
       'arms':['CONTROL','ALL_FIVE_TARGETED'],'dataset':[6471,548],'nominal_schedule_epochs':100,'stop_after_completed_epochs':1,'resolution':1280,'batch':2,'FP32':True,'workers':0,
       'expected':{'anchor_exposures':6471,'forward_batches':3236,'backward_batches':3236,'optimizer_EMA_updates':784,'FP32_snapshots':1},
       'preserve':'Initialization, augmentation, warmup100 horizon and Mosaic semantics; full epoch includes validation/save/reload',
       'not_scientific_results':True,'GPU_status':'NOT_EXECUTED','target_host':'NOT_ASSUMED','authorization':'New arm-specific approval; both-arm cap plus reserve; separate target CPU receipt',
       'cost_scope':'Init/data checks/loader/train/health/internal validation/FP32 save/reload. Full research COCO/bootstrap cost measured separately.'}
    save(package/'resource_benchmark_protocol.json',benchmark)
    template={'approved':False,'authorization_id':'REPLACE_WITH_NEW_AUTHOR_ID','protocol_id':version,'allocation_id':'REPLACE_WITH_NEW_ALLOCATION_ID',
       'allocation_scope':'NEW_ARM_SPECIFIC_ALLOCATION','historical_VM37_ledger_reuse':False,'mode':'benchmark','arm_ids':[],'arm_authorizations':{},
       'package_manifest_SHA256':'FILL_AFTER_PACKAGE_ASSEMBLY','CPU_receipt_SHA256':'SHA256_OF_CANONICAL_SORTED_RECEIPT_JSON',
       'GPU_uuid':'REPLACE_AFTER_TARGET_ADMISSION','cooling_confirmed':False,'power_confirmed':False,'machine_admitted':False,'total_seconds':0,
       'expires_UTC':'REPLACE_WITH_EXPLICIT_NEW_UTC_EXPIRY','resource_admission':{'status':'NOT_ADMITTED','safety_reserve_seconds':0,'evaluation_hard_seconds':0,'required_complete_group_seconds':0}}
    save(package/'authorization_vm48.example.json',template)
    return rows,evaluation


def evidence(out):
    result={}
    paths={
      'VM38':ROOT/'datasets/local_runs/vm38_variability_guided_augmentation/vm38_variability_guided_augmentation_targeted960_20261114T215310_e8bba12b/final_acceptance.json',
      'VM39':ROOT/'datasets/local_runs/vm39_multi_seed_confirmation/vm39_multi_seed_confirmation_confirmation_20261115T095834_9623ace0/final_acceptance.json',
      'VM40':ROOT/'datasets/local_runs/vm40_uvl_gap_independent_validation/vm40_uvl_gap_independent_validation_audit_20261116T011556_955084fb/final_acceptance.json',
      'VM41':ROOT/'datasets/local_runs/vm41_degradation_coverage/vm41_degradation_coverage_coverage_20261116T020508_dfa7879d/final_acceptance.json',
      'VM42':ROOT/'datasets/local_runs/vm42_independent_readiness/vm42_independent_readiness_readiness_20261116T111422_8c956229/final_acceptance.json',
      'VM43':ROOT/'datasets/local_runs/vm43_external_pilot/vm43_external_pilot_execution_20261116T222929_46c7e0ba/final_acceptance.json',
      'VM44':ROOT/'datasets/local_runs/vm44_tiny_error_variability_decision/vm44_tiny_error_variability_decision_analysis_20261117T031628_9ebff202/FM_P_to_FM_D_decision_rules.json',
      'VM45':VM45/'final_acceptance.json',
      'VM46':ROOT/'datasets/local_runs/vm46_heldout_variability/vm46_heldout_variability_admission_cpu_20261117T184741_6d462e31/Final_acceptance.json',
      'VM47':VM47/'final_acceptance.json'}
    for label,path in paths.items():
        value=read(path);result[label]={'source':ref(path),'status':value.get('status','SOURCE_BOUND_DEVELOPMENT_RULE_INPUTS'),
            'scope':value.get('scope',value.get('scientific_scope','See source; not upgraded by this package')),
            'formal100_completed':False}
    interactions=[];gaps=[]
    for label,path in [('VM45',VM45),('VM47',VM47)]:
        filename='factorial_interactions.csv' if label=='VM45' else 'factorial_effects.csv'
        source=path/filename;ref(source)
        with source.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
        for r in rows:
            if r['size']=='all' and not r['category_id'] and r['metric']=='map50_95' and r['effect']=='Interaction_DID':
                interactions.append({'study':label,'D':r['D'],'DID_raw':float(r.get('effect_raw',r.get('estimate'))),
                    'CI_low_raw':float(r.get('interval_lower',r.get('CI_low'))),'CI_high_raw':float(r.get('interval_upper',r.get('CI_high'))),
                    'scope':'DEVELOPMENT_ONLY_SYNTHETIC_EXPLORATORY','source':str(source),'source_SHA256':sha(source)})
        source=path/('local_P_gap_status.csv' if label=='VM45' else 'Gap_status.csv');ref(source)
        with source.open(encoding='utf-8-sig') as f:
            for r in csv.DictReader(f):gaps.append({'study':label,**r,'source_SHA256':sha(source)})
    csv_write(out/'accepted_interactions.csv',interactions);csv_write(out/'accepted_local_Gap.csv',gaps)
    save(out/'accepted_evidence_index.json',result)
    return interactions,gaps,result


def render_plan(out):
    from pypdf import PdfReader
    reader=PdfReader(PDF);assert len(reader.pages)==17
    text=[page.extract_text() for page in reader.pages]
    (out/'supervisor_plan_text.txt').write_text('\n\n'.join('PAGE '+str(i+1)+'\n'+s for i,s in enumerate(text)),encoding='utf-8')
    ref(PDF);save(out/'supervisor_plan_read.json',{'source':ref(PDF),'pages':17,'read_all_pages':True,'key_pages':[4,5,6,7,8,9,10,11,12,15],
      'requirements':'Formal same schedule/seeds, five individual v8 ablations, training ranges not fixed test points, independent final test, author manuscript review'})
    (out/'plots').mkdir()
    exe=shutil.which('pdftoppm')
    if exe:
        subprocess.run([exe,'-f','12','-l','12','-singlefile','-r','85','-png',str(PDF),str(out/'plots/supervisor_page12')],check=True,capture_output=True,timeout=60)


def resource(out):
    sys.path.insert(0,str(HERE.parent/'experiments_test/vm43_external_pilot'));import vm43_common
    ledger=vm43_common.budget();ref(vm43_common.AUTH/'authorization.json')
    costs=read(ROOT/'datasets/local_runs/vm37_yolo26_rq4/vm37_yolo26_rq4_measured_cost_analysis_cpu_20261112T215459_ea3fd53c/cost_analysis.json')
    old=read(VM47/'actual_resource_usage.json')
    value={'status':'FORMAL_GPU_NOT_ADMITTED','current_historical_ledger':ledger,'new_GPU_seconds':0,'formal_arms_completed':0,
      'measured_legacy1280_CONTROL_epoch1_seconds':costs['CONTROL_training_scope_seconds'],
      'measured_legacy1280_ALL_FIVE_epoch1_seconds':costs['ALL_FIVE_training_scope_seconds'],
      'legacy_schedule_limit':'CONTROL nominal100 vs ALL_FIVE nominal20; original input pipeline; first epoch only',
      'legacy_linear26s_100epoch_pair_hours_SCENARIO_ONLY':100*(costs['CONTROL_training_scope_seconds']+costs['ALL_FIVE_training_scope_seconds'])/3600,
      'VM47_eightcell_worker_seconds_MEASURED':old['GPU_worker_wall_charged_seconds'],
      'simple_714cell_VM47_mean_scaling_hours_NOT_ADMISSION':old['GPU_worker_wall_charged_seconds']*(714/8)/3600,
      'evaluation_scaling_limit':'Different models/contexts/resolutions and CPU costs; neither target ETA nor a guaranteed bound',
      'CPU_speedup_is_not_full_epoch_speedup':True,'960_10epoch_is_not_1280_100epoch_ETA':True,
      'target_machine':'UNKNOWN_NOT_CONNECTED_OR_TESTED','target_training_cost':'UNKNOWN','YOLOv8s_full_epoch_cost':'UNKNOWN',
      'minimum_known26s_snapshot_scenario_GiB':160788774*600/1024**3,'all17_arm_storage':'UNKNOWN_UNTIL_TARGET_BENCHMARK; add v8/ablations, shared PNG cache,714 predictions and logs',
      'planning_recommendation_not_measured_minimum':{'isolated_GPU':1,'VRAM_GiB':16,'RAM_GiB':32,'logical_CPUs':8},
      'remaining_requirements':['Target CPU receipt','New arm-specific allocation','Both-strategy target full epoch measurements','Complete train/evaluation/Bootstrap/storage cost admission','Current device/thermal admission','Independent confirmation gate for final-paper claims']}
    save(out/'resource_feasibility.json',value)
    return value


def reports(out,package,rows,evaluation,tests,resource,interactions,gaps,package_zip):
    def md(headers,rs):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(str(x) for x in r)+' |' for r in rs]])
    state='CPU验证通过，正式GPU训练未准入' if tests['status']=='CPU_SOFTWARE_PASS_NOT_GPU_VALIDATED' else 'CPU验证失败，NO_GO'
    observed=md(['研究','D','AP95 DID百分点','配对95%区间百分点'],[[r['study'],r['D'],f"{r['DID_raw']*100:+.3f}",f"[{r['CI_low_raw']*100:.3f},{r['CI_high_raw']*100:.3f}]"] for r in interactions])
    checks=md(['CPU检查','状态','秒'],[[r['test'],r['status'],f"{r['wall_seconds']:.3f}"] for r in tests['checks']])
    matrix=[{'RQ':'RQ1','status':'PARTIAL','evidence':'原17页PDF和双UVL来源保留；SAT/生成器既有证据复用，未证明全空间物理真实性'},
      {'RQ':'RQ2','status':'MEASURED_LOCAL_INTERACTIONS_UNSAFE_RULE_TRANSFER','evidence':'VM45 Motion×Haze与VM47 Gaussian×JPEG真实DID；规则误放行保留，非可靠独立泛化'},
      {'RQ':'RQ3','status':'LOCAL_MEASURED_GAP_ONLY','evidence':'VM45局部2 COVERED/2 GAP；VM47局部3 COVERED/1 UNCERTAIN，均仅两个Infer1280历史D；不能混合不同时延边界推全域'},
      {'RQ':'RQ4','status':'FORMAL_NOT_EVALUATED','evidence':'17臂软件、冻结21P/714评价和Gap主终点已实现；本轮0训练臂；960/10轮三seed探索不是正式100轮'}]
    csv_write(out/'RQ1_RQ4_evidence_matrix.csv',matrix)
    body=f'''# VM48 RQ4正式实施与资源计划

RunID：`{out.name}`。{state}。本轮实际正式训练0/12、单增强消融0/5、新GPU评价0；没有训练结果、AP收益或Gap缩小声明。

## 已实施组件

新包VM48_FORMAL_RQ4_V1复用未修改V3.1训练器、健康保护、优化输入、原始标签视图和六份native10初始化。12主臂为v8s/26s×42/43/44×CONTROL/ALL_FIVE_TARGETED；5个v8单增强沿用原定义seed42。ALL_FIVE_TARGETED只是原VM37_ALL_FIVE_V1静态缺口五类能力的明确命名，不是VM38单Gaussian经验引导策略。

完整6471/548、1280/100轮、batch2/nbs64、AdamW lr0.01、cosine/warmup3、workers0、FP32、普通增强、close_mosaic10和固定epoch100 EMA不变。每臂计划647100次anchor曝光、323600 F/B batch、10917 optimizer/EMA更新；最后不足2张的一批仍为1张，跨epoch累积和末尾25个pending microbatch不补步。

五增强使用原数值范围与独立p.5门控、原次序和每图/epoch/occurrence/family哈希RNG。所有测试精确强度被抽样器排除，整数JPEG/模糊核也不落在固定测试值。单增强仅保留该族，不改变其参数/RNG语义。普通Albumentations由实际2.0.8运行观察，不能把导师8.4.33说明当成当前实测。新增日志记录真实调用及实际参数，并在Mosaic关闭重建后继续安装；CPU像素一致性测试检查其不改采样。

新执行入口默认拒绝GPU，需要每臂批准、完整组成本、目标CPU receipt、新分配ID/期限、当前GPU UUID、供电/散热、90°C立即/87°C30秒停止。遥测间隔超过2秒或失败即停止所属进程，不影响其他任务。不能复用原36小时账本。每epoch原子FP32保存和重载保留，固定epoch0100 EMA才可进入正式评价；best、960/10轮及合成测试checkpoint均拒绝。CPU模型重载通过不代表GPU精确resume，自动恢复/重试被禁止。

## 冻结评价与主终点

主P样本逐字保留原formal_evaluation_manifest.json的21个有目的条件，640/1280、548图。12主臂504格，5消融210格，共714格。每P共享PNG，原图GT/ignore/面积/native10不变，COCO AP50/AP95/AR与200次同图Bootstrap保留。新评价接入VM41完整缓存计时，不再使用旧portable.evaluate.py的预测段时延作Gap。

主终点按seed比较CONTROL与ALL_FIVE_TARGETED两个检测器集合，各含两架构×两推理分辨率。Gap缩小=CONTROL已证实GAP数−TARGETED已证实GAP数；同时报告UNCERTAIN数、Gap计数上下界、配对缩小上下界及200共享图像索引的条件区间。Tiny阈值.10、缓存均值100ms冻结。另报AP/Tiny变化、Normal代价、逐族配置子集均值及seed42消融；图像区间不作为训练seed显著性。

这是已测21P子集的变化，没有均匀有效空间抽样，不能估计总体Gap变化。原21P没有Motion/Haze单项或包含它们的配置，相关逐族终点必须NOT_ESTIMABLE_PRIMARY21，不能用已训练前VM45数据补出训练后结果。扩展样本需独立冻结和资源准入；本轮不静默改变原正式样本。

导师第11页要求最终报告test-dev而不是val。本包仍为DEVELOPMENT_ONLY：test-dev历史暴露/场景独立性未通过，VM43只有限外部探索；UAVDT标注/类别问题保留。软件与开发训练可准备，独立确认前不能把最终论文证据写成完整满足原计划。

## 实际CPU验证

{checks}

合成测试只检查代码。另用VM47已保存完整548图真实预测复算COCO，逐类别/尺度AP和AR与原指标比较，没有新GPU推理。模型重载、BN异常、保存冲突、epoch100拒绝规则、种子初始化、配对Gap计数和预算/热停止都留存实际测试日志。GPU矩阵未执行，目标服务器未连接或测试。

## 资源：测量与外推分开

原账本余额{resource['current_historical_ledger']['remaining_seconds']:.3f}秒，原截止{resource['current_historical_ledger']['deadline_UTC']}不变，本轮扣费0。该余额不准入12臂100轮。

旧1280首轮CONTROL实测{resource['measured_legacy1280_CONTROL_epoch1_seconds']:.3f}秒、ALL_FIVE {resource['measured_legacy1280_ALL_FIVE_epoch1_seconds']:.3f}秒，但名义调度不同、旧输入链路且只完成首轮。线性100轮一对情景约{resource['legacy_linear26s_100epoch_pair_hours_SCENARIO_ONLY']:.2f}小时，不是目标机器ETA。VM47八格实际worker墙钟{resource['VM47_eightcell_worker_seconds_MEASURED']:.3f}秒，714格按其均值简单缩放约{resource['simple_714cell_VM47_mean_scaling_hours_NOT_ADMISSION']:.2f}小时，也不作为成本准入。960/10轮及CPU输入加速都不能替代1280/100实测。

新resource_benchmark_protocol.json要求先在实际目标机，以100轮调度完整执行每架构seed42 CONTROL/TARGETED各首轮，记录初始化、加载、健康、6471/548、保存重载和显存；不改变Mosaic/warmup。首轮预计784次更新不是100轮10917次。两策略、两架构、研究评价/Bootstrap、存储与余量都有实测来源后才可准入正式组。单GPU16GiB、32GiB RAM、8逻辑CPU只是沿用资源规划建议。26s六臂每epoch快照约89.85GiB为已有单文件大小情景；v8、五消融、PNG与714格预测总存储未知，需目标基准补齐。

## 路径与使用

包目录：`{package}`。
独立ZIP：`{package_zip}`（私有，含初始化与派生标签，不上传Git）。仅复制约百MiB既有包资产，未复制145GiB归档或原始train/val图像。

目标机先安装冻结环境并解包；不要从工作区覆盖包内文件。CPU验收：

```powershell
python -B -X utf8 main/experiments/vm48_formal_rq4/validate48.py --package <解包根目录> --data-root <VisDrone根目录> --output <包内datasets/local_runs/新的CPU检查目录>
```

仅预演：

```powershell
.\run_training.ps1 -Package <解包根目录> -DataRoot <VisDrone根目录> -Output <包内datasets/local_runs/新Run目录> -Mode formal -DryRun
.\run_evaluation.ps1 -Package <解包根目录> -DataRoot <VisDrone根目录> -TrainingParent <训练父Run目录> -DryRun
```

未来真实成本基准改为Mode benchmark、Architecture YOLO26s或YOLOv8s，提供新Authorization、CPUReceipt、Execute；无有效批准会拒绝。正式Mode formal执行12臂，Mode ablations执行原5消融；消融评价还需同包同数据seed42 CONTROL参考。原快照不是精确训练恢复点，失败保留并停止，不自动续跑。

所有可执行文件、完整命令、CPU日志、源commit与逐文件SHA在本Run。Git HEAD只标识基线提交，当前未提交差异另存，不能谎称全部新代码已提交。本轮没有Git网络操作或提交。三份中文材料是操作/事实记录，未经作者与导师审核，不是批准的论文结果或分析正文。
'''
    brief=f'''# VM48 导师会议研究进展

RunID：`{out.name}`。{state}。正式训练实际0/12，五消融0/5。

{observed}

VM45/47均为548开发图、固定历史checkpoint、合成退化配对结果，不是独立确认或架构因果。VM47冻结规则在唯一新P产生1个false-PASS，另1个UNCERTAIN，保持ABSTAIN。

RQ4已完成17臂可执行包、原21P/714格评价清单、真实增强日志、CPU验证和目标机首轮成本协议。未证明增强提高AP或缩小Gap。当前剩余{resource['current_historical_ledger']['remaining_seconds']:.3f}秒不用于正式100轮；需新的目标资源与逐臂授权。

下一步是目标机器CPU验收及获准后的完整首轮两臂成本测量，不是立即开始100轮。Motion/Haze不在原正式21P，独立测试准入与正式训练效果仍缺。本文是可追溯会议事实摘要，需作者审核，不是论文正文。
'''
    matrix_text='# VM48 RQ1–RQ4证据矩阵\n\nRunID：`'+out.name+'`。\n\n'+md(['RQ','状态','可用证据与边界'],[[r['RQ'],r['status'],r['evidence']] for r in matrix])+'\n\n'+observed+'\n\n开发集结果、VM43有限外部探索和真正独立确认分别登记；第三类目前未建立。正式RQ4一律NOT_EVALUATED。已测交互数值由原CSV和SHA读取，不手填研究数值，不补造训练结果或引用。作者审核要求和原论文写作规则不变。\n'
    docs=[]
    for name,text in [('VM48_RQ4正式实施与资源计划_CN.md',body),('VM48_导师会议研究进展_CN.md',brief),('VM48_RQ1_RQ4证据矩阵_CN.md',matrix_text)]:
        path=ROOT/'datasets/reportcn'/name
        if path.exists():path=path.with_name(path.stem+'_'+out.name.split('_')[-1]+path.suffix)
        with path.open('x',encoding='utf-8') as f:f.write(text)
        docs.append(path)
    return docs


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',type=Path,default=ROOT/'datasets/VisDrone');a=p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',PYTHONDONTWRITEBYTECODE='1')
    from main.pipeline import new_run
    out,logger=new_run({'output_root':str(RUNS),'GPU':False},HERE.name,'package_cpu');started=time.perf_counter()
    assert sha(BASE_ZIP)=='394c7739ced42c7b366f43493dbbfa2f88065f6af1fac68577ddd57e672bf0b9';ref(BASE_ZIP)
    render_plan(out)
    head=subprocess.run(['git','-C',str(ROOT/'main'),'rev-parse','HEAD'],capture_output=True,text=True,check=True).stdout.strip()
    dirty=subprocess.run(['git','-C',str(ROOT/'main'),'status','--porcelain'],capture_output=True,text=True,check=True).stdout
    (out/'git_worktree_status.txt').write_text(dirty,encoding='utf-8')
    package=out/'package';package.mkdir()
    with zipfile.ZipFile(BASE_ZIP) as z:
        for member in z.infolist():
            target=(package/member.filename).resolve()
            if not target.is_relative_to(package.resolve()):raise ValueError('Unsafe base archive member.')
        z.extractall(package)
    legacy=read(package/'bundle_manifest.json')
    for relative,digest in legacy['files'].items():assert sha(package/relative)==digest,relative
    dest=package/'main/experiments'/HERE.name;dest.mkdir()
    for file in HERE.iterdir():
        if file.is_file() and file.suffix in ('.py','.ps1','.md'):shutil.copyfile(file,dest/file.name);ref(file)
    for name in ('run_training.ps1','run_evaluation.ps1'):shutil.copyfile(HERE/name,package/name);shutil.copyfile(HERE/name,out/name)
    shutil.copyfile(HERE/'README.md',package/'README_VM48.md')
    extract_backend(package);rows,evaluation=metadata(package,head)
    old_files=set(legacy['files'])|{'bundle_manifest.json'}
    new_files={f.relative_to(package).as_posix() for f in package.rglob('*') if f.is_file()}-old_files
    files={name:sha(package/name) for name in sorted(old_files|new_files)}
    save(package/'package_manifest.json',{'version':'VM48_FORMAL_RQ4_V1','source_commit':head,'worktree_dirty':bool(dirty),'files':files,
         'base_V3_1_archive_SHA256':sha(BASE_ZIP),'historical_base_files_unchanged':True,'new_scientific_training_arms_completed':0,'GPU_seconds':0})
    test_out=package/'datasets/local_runs/vm48_cpu_validation'
    command=[sys.executable,'-B','-X','utf8',str(dest/'validate48.py'),'--package',str(package),'--data-root',str(a.data_root.resolve()),'--output',str(test_out),
             '--real-regression',str(VM47/'Gauss_s15_JPEG_q25__ck05_Infer1280')]
    save(out/'CPU_validation_command.json',{'command':command})
    logger.info('Running new VM48 CPU validation; no GPU execution.')
    with (out/'cpu_validation.log').open('x',encoding='utf-8') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=3600)
    tests=read(test_out/'cpu_test_results.json')
    shutil.copyfile(test_out/'cpu_test_results.json',out/'cpu_test_results.json')
    shell=shutil.which('pwsh') or shutil.which('powershell')
    entries=[]
    if result.returncode==0 and shell:
        dry=package/'datasets/local_runs/vm48_documented_dry_run'
        commands=[ [shell,'-NoProfile','-ExecutionPolicy','Bypass','-File',str(package/'run_training.ps1'),'-Package',str(package),'-DataRoot',str(a.data_root.resolve()),'-Output',str(dry),'-Mode','formal','-Python',sys.executable,'-DryRun'],
                   [shell,'-NoProfile','-ExecutionPolicy','Bypass','-File',str(package/'run_evaluation.ps1'),'-Package',str(package),'-DataRoot',str(a.data_root.resolve()),'-TrainingParent',str(dry),'-Python',sys.executable,'-DryRun'] ]
        for i,cmd in enumerate(commands):
            r=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=240)
            (out/f'entrypoint_{i+1}.log').write_text(r.stdout+'\n'+r.stderr,encoding='utf-8');entries.append({'command':cmd,'returncode':r.returncode,'GPU':False})
    save(out/'entrypoint_tests.json',entries)
    cpu_pass=result.returncode==0 and len(entries)==2 and all(r['returncode']==0 for r in entries)
    interactions,gaps,accepted=evidence(out);resources=resource(out)
    archive=out/'VM48_formal_rq4_package.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name in sorted(files.keys()|{'package_manifest.json'}):
            info=zipfile.ZipInfo(name,date_time=(2026,11,17,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(info,(package/name).read_bytes())
    save(out/'package_result.json',{'version':'VM48_FORMAL_RQ4_V1','ZIP':ref(archive),'manifest_SHA256':sha(package/'package_manifest.json'),
        'files':len(files)+1,'original_raw_dataset_copies':0,'private_initializations':6,'private_labels':7019,'current_target_GPU_tested':False,'CPU_pass':cpu_pass})
    for name in ('training_matrix.csv','augmentation_protocol.json','frozen_evaluation_protocol.json','training_config_manifest.csv','resource_benchmark_protocol.json'):
        shutil.copyfile(package/name,out/name)
    docs=reports(out,package,rows,evaluation,tests,resources,interactions,gaps,archive)
    save(out/'source_provenance.json',{'input_files':list(SOURCES.values()),'source_commit':head,'worktree_status':str(out/'git_worktree_status.txt'),
        'package_manifest_SHA256':sha(package/'package_manifest.json'),'new_code_committed':False,'all_registered_inputs_unchanged':all(sha(r['path'])==r['SHA256'] for r in SOURCES.values()),
        'GPU_seconds':0,'training_arms_run':0,'author_approved_manuscript':False})
    status='CPU_PACKAGE_COMPLETE_FORMAL_GPU_BLOCKED' if cpu_pass else 'PARTIAL_CPU_VALIDATION_FAILED'
    save(out/'Final_acceptance.json',{'status':status,'primary_arms_implemented':12,'ablations_implemented':5,'CPU_tests_status':tests['status'],'entrypoints_pass':len(entries)==2 and all(r['returncode']==0 for r in entries),
        'actual_formal_training_arms':0,'actual_ablation_training_arms':0,'new_GPU_evaluations':0,'new_GPU_seconds':0,'formal_RQ4':'NOT_EVALUATED','package':str(archive),'package_SHA256':sha(archive),'reports':[str(p) for p in docs]})
    (out/'diary.md').write_text('VM48 assembled a new17-arm formal package from accepted V3.1 assets, preserving historical files and original21P sample. New logging, resolution-aware cached evaluation and Gap endpoints are CPU tested. Saved real VM47 predictions verify metric compatibility. No GPU, training, remote access, Git operations or manuscript approval. Formal100 and independent confirmation remain unmeasured.\n',encoding='utf-8')
    run=read(out/'run.json');run['status']=status;(out/'run.json').write_text(json.dumps(run,indent=2),encoding='utf-8')
    links=''.join(f'<li><a href="{p.as_uri()}">{p.name}</a></li>' for p in docs)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>VM48 formal RQ4 package</title><h1>VM48 formal RQ4 package</h1><p>'+status+'</p><p>0 formal training arms. No GPU work.</p><ul>'+links+'</ul><p><a href="VM48_formal_rq4_package.zip">Private executable package</a></p>',encoding='utf-8')
    from main.visualization import build_run_index
    build_run_index(RUNS,HERE.name)
    save(out/'CPU_resource_usage.json',{'wall_seconds':time.perf_counter()-started,'GPU_seconds':0,'archive_bytes':archive.stat().st_size,'original_145GiB_archive_copies':0})
    finalfiles=[f for f in out.rglob('*') if f.is_file() and 'runtime' not in f.relative_to(out).parts]+docs
    with (out/'SHA256SUMS.txt').open('x',encoding='utf-8') as f:
        for path in sorted(finalfiles):f.write(sha(path)+'  '+str(path)+'\n')
    print(json.dumps({'RunID':out.name,'status':status,'CPU_tests':tests['status'],'package':str(archive),'SHA256':sha(archive),'reports':[str(p) for p in docs]},ensure_ascii=False),flush=True)


if __name__=='__main__':main()
