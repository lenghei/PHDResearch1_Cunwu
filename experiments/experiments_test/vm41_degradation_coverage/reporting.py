"""Local figures and Chinese explanations for the coverage audit."""
import csv
import html
import json
from pathlib import Path


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))


def table(headers,values):
    clean=lambda x:str(x).replace('|',' / ').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
        *['| '+' | '.join(clean(x) for x in row)+' |' for row in values]])


def figures(out,leaves,pairwise,catalog,images,original):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager,FontProperties
    import numpy as np
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=font).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'axes.spines.right':False,'axes.spines.top':False,'font.size':10})
    folder=out/'plots';folder.mkdir();names=[]
    def finish(fig,name):
        for ext in ('png','pdf'):fig.savefig(folder/(name+'.'+ext),dpi=170,bbox_inches='tight',facecolor='white')
        plt.close(fig);names.append(name)
    fig,ax=plt.subplots(figsize=(10,10));y=np.arange(len(leaves))
    ax.barh(y,[r['measured_P_configurations'] for r in leaves],color=['#327f72' if r['VM36_present'] else '#c3cbd2' for r in leaves])
    for i,r in enumerate(leaves):
        n=r['measured_P_configurations'];ax.text(n+.15,i,str(n) if n else '未测',va='center',fontsize=9)
    ax.set_yticks(y,[r['leaf'] for r in leaves]);ax.invert_yaxis();ax.set_xlim(0,max(r['measured_P_configurations'] for r in leaves)+2)
    ax.set_xlabel('包含该叶的已测P条件数（分母21；不是84次训练）')
    ax.set_title('VM36：18/25叶有测量，7叶未测\n灰色不代表性能失败；非均匀样本不能估计总体失败率')
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True);finish(fig,'01_measured_leaf_coverage')
    factors=list(dict.fromkeys([r['factor_A'] for r in pairwise]+[r['factor_B'] for r in pairwise]))
    matrix=np.full((len(factors),len(factors)),np.nan);labels={}
    for r in pairwise:
        i,j=factors.index(r['factor_A']),factors.index(r['factor_B'])
        matrix[i,j]=matrix[j,i]=r['coverage'];labels[i,j]=labels[j,i]=f'{r["VM36_covered_level_pairs"]}/{r["SAT_valid_level_pairs"]}'
    fig,ax=plt.subplots(figsize=(10,8));c=ax.imshow(matrix,cmap='YlGnBu',vmin=0,vmax=1)
    for (i,j),label in labels.items():ax.text(j,i,label,ha='center',va='center',fontsize=8)
    ax.set_xticks(range(len(factors)),factors,rotation=45,ha='right');ax.set_yticks(range(len(factors)),factors)
    ax.set_title('VM36合法二阶水平覆盖：包含ABSENT\n逻辑覆盖率不等于四点完整配对或可估计交互')
    fig.colorbar(c,ax=ax,shrink=.6,label='已测水平对 / SAT允许水平对');finish(fig,'02_factor_pairwise_coverage')
    fig,ax=plt.subplots(figsize=(12,5));ax.axis('off')
    for x,y,text in [(.21,.83,'FM_P：原约束合法\n→ 生成参数绑定审核\n→ 像素确定性与标注坐标'),
                     (.79,.83,'FM_D：权重/类别/来源健康\n→ 固定head、NMS、FP32\n→ 明确已测检测器子集'),
                     (.5,.49,'同一图像、同一退化像素、同一GT\nTiny AP95区间 + 统一缓存管线时延区间'),
                     (.5,.15,'Tiny ≥ .10 且缓存管线 ≤100ms\n单D：PASS / FAIL / UNCERTAIN\n子集：存在PASS则covered；全FAIL则Gap')]:
        ax.text(x,y,text,transform=ax.transAxes,ha='center',va='center',bbox=dict(boxstyle='round,pad=.6',fc='#edf4f8',ec='#5c8294'))
    for start,end in [((.25,.66),(.44,.59)),((.75,.66),(.56,.59)),((.5,.37),(.5,.28))]:
        ax.annotate('',xy=end,xytext=start,xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#5c8294'))
    fig.suptitle('未来统一Gap协议：配置合法、标注安全和独立性是不同关卡',fontsize=14)
    finish(fig,'03_features_to_gap')
    fig,ax=plt.subplots(figsize=(12,6));ax.axis('off')
    blocks=[(.5,.87,'优先1：独立native10数据准入\n当前DATASET_BLOCKED；不能以无历史命中代替证明'),
            (.5,.62,'通过全部授权/预算/温度关卡后\n12格：3seed × CONTROL/GUIDED × Normal/Gauss15，1280\n同时获得统一Tiny与时延；本轮未运行'),
            (.26,.30,'独立数据仍阻塞：仅另行授权后\n开发集12格统一计时桥接\n能补Gap判据，不能补独立泛化'),
            (.77,.30,'优先3：审核Motion/Haze绑定后\n4P × 2历史模型 × Infer1280 = 8格\nNormal / Motion21 / Haze / 两者组合'),
            (.5,.04,'优先4：保持Gaussian选择规则、p=.25及范围不变；独立确认前不再开发集反复调参')]
    for x,y,text in blocks:ax.text(x,y,text,transform=ax.transAxes,ha='center',va='center',fontsize=10,
        bbox=dict(boxstyle='round,pad=.7',fc='#fff0dd' if y>.8 else '#edf4f8',ec='#a38151' if y>.8 else '#5c8294'))
    for start,end in [((.5,.77),(.5,.71)),((.43,.51),(.26,.42)),((.57,.51),(.77,.42))]:
        ax.annotate('',xy=end,xytext=start,xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#5c8294'))
    fig.suptitle('下一步的条件式路线：不是自动执行全部候选',fontsize=14);finish(fig,'04_next_experiment_gates')
    preview=[('Normal',{'Normal'}),('Motion21',{'Normal','Motion_k21'}),('Haze dense · A220 proposal',{'Normal','Haze_dense'}),
        ('Motion21 + Haze',{'Normal','Motion_k21','Haze_dense'}),('Rain · proposal',{'Normal','Rain'}),('Gaussian15',{'Normal','Gauss_s15'})]
    fig,axes=plt.subplots(2,3,figsize=(12,6))
    for ax,(name,selection) in zip(axes.flat,preview):
        item=next(r for r in catalog if set(r['selected_leaves'].split('|'))==selection)
        ax.imshow(images[item['configuration_id']][:,:,::-1]);ax.axis('off');ax.set_title(name,fontsize=10)
    fig.suptitle('现有一张val图的CPU像素检查；无YOLO预测、无AP\n显示缩略图，生成验证使用原生尺寸；参数绑定仍是提案',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.91));finish(fig,'05_CPU_pixel_preview_only')
    return names


def reports(out,root,old,resource):
    cov=load(out/'coverage_summary.json');candidate=load(out/'candidate_validation_summary.json')
    gate=load(out/'Independent_data_readiness.json');cost=load(out/'Complete_pair_cost_plan.json')
    leaves=rows(out/'FM_P_measured_coverage.csv');catalog=rows(out/'Candidate_valid_configurations.csv')
    bbox=load(out/'RollingShutter_geometry_counterexample.json')
    protocol=load(out/'Future_Gap_protocol.json');history=load(out/'Historical_testdev_records.json')
    files=root/'datasets/reportcn';report=files/'VM41_研究缺口与下一步实验方案_CN.md'
    beginner=files/'VM41_导师会议详细讲解与问答_CN.md'
    if report.exists() or beginner.exists():
        report=files/('VM41_研究缺口与下一步实验方案_CN_'+out.name+'.md')
        beginner=files/('VM41_导师会议详细讲解与问答_CN_'+out.name+'.md')
    if report.exists() or beginner.exists():raise FileExistsError('This Run already has reports.')
    def link(name):return f'[{name}]({(out/name).as_posix()})'
    def image(name):return f'![{name}]({(out/"plots"/(name+".png")).as_posix()})'
    coverage_table=table(['叶特征','已测P数 /21','实测数值','单因素情况','生成/标注限制'],
        [[r['leaf'],r['measured_P_configurations'],r['actually_measured_numeric_levels'],r['single_factor_contrast'],r['binding_review']+'; '+r['annotation_semantics']] for r in leaves])
    candidate_table=table(['配置ID','叶集合','VM36已有','标注安全','绑定状态'],
        [[r['configuration_id'],r['selected_leaves'],r['already_measured_P_in_VM36'],r['annotation_safe'],r['scientific_preparation_status']] for r in catalog])
    costs=table(['未来方案','格数×图数','线性外推h','含余量规划h','能否准入'],
        [[r['design'],f'{r["cells"]}×{r["images_per_cell"]}',f'{r["linear_planning_seconds_NOT_MEASURED"]/3600:.3f}',
          f'{r["planning_envelope_seconds_NOT_A_GUARANTEE"]/3600:.3f}','否：数据/授权/实测/热准入未完成'] for r in cost['plans']])
    measured=rows(old/'VM39_three_seed_summary.csv')
    seeds=table(['尺寸','条件','GUIDED−CONTROL AP95均值pp','样本SD pp'],
        [[r['resolution'],r['condition'],f'{float(r["mean"])*100:+.4f}',f'{float(r["sample_seed_SD"])*100:.4f}']
         for r in measured if r['size']=='all' and r['condition'] in ('Normal','Gaussian_s15')])
    text=f'''# VM41 研究缺口与下一步实验方案

RunID：`{out.name}`。日期：2026-11-16。

结论：**CPU_COVERAGE_AND_PROTOCOL_COMPLETE_WITH_LIMITATIONS**。完成覆盖与准入设计；独立数据仍阻塞，未执行GPU、训练或COCO重算。保存的像素检查不是新精度结果。

## 1. 导师计划与当前已知证据

复用导师2026-10-22研究计划原PDF、冻结FM_P/FM_D及VM36–VM40验收证据。实际模型仍是25叶/5约束/46,368个FM_P配置，34叶/4约束/635,904个FM_D配置。本轮不更改UVL。PDF相关页摘录与原文件SHA见{link('Supervisor_plan_excerpts.json')}。

VM36是21个P×4个D=84评价对，每对548图，总46,032张次，并非84次训练。新训练结果为YOLO26s960/10epoch：CONTROL/GUIDED各seed42/43/44，ALL_FIVE仅seed42。保留DEVELOPMENT_ONLY与POST_HOC_POLICY_SELECTION。

{seeds}

Gaussian AP改善有三个seed支持，独立泛化及完整Gap缩小仍未建立；这些数值直接引用VM40已核验表，没有重复训练、推理或AP重算。

## 2. 每个FM_P叶实际测了什么

**18/25叶出现过，7叶未测**：Motion_k7/k15/k21、Haze_light/dense、Rain、RollingShutter。未测不是失败，不能计入FAIL。

{coverage_table}

{image('01_measured_leaf_coverage')}

Gaussian有5.5/15/30，JPEG有25/50/75，Defocus有3/6，Downscale有1.5/2，所有光照叶及Shot100/20均在组合中出现。但“数值出现”不等于有独立效应对照；原21项只有Normal、JPEG25、JPEG75、Gaussian15构成单因素参考。JPEG50、Gaussian5.5/30、Defocus、Downscale和普通曝光的多数独立角点缺失。

LowLight必须带ShotNoise，Shot_high必须处于LowLight或强欠曝；不能为了凑“单因素”生成非法Normal+Shot_high或独立LowLight。合理条件对照包括Normal+Shot_low与LowLight+Shot_low，以及强欠曝有/无Shot_high；这不是无条件主效应。

完整逐叶数据：{link('FM_P_measured_coverage.csv')}。Normal是必选光照分组的一档基准，其出现次数不能解释为该数量的clean图像配置。

## 3. 缺少哪些交互

在合法全空间中，正叶两两同现的允许组合共{cov['valid_positive_leaf_pairs']}种，VM36实际同现{cov['measured_positive_leaf_pairs']}种。按九个分类因素、包含可选因子ABSENT的二阶水平对，覆盖{cov['categorical_pairwise_numerator']}/{cov['categorical_pairwise_denominator']}。这些是逻辑覆盖，**不是统计功效或交互识别率**。

{image('02_factor_pairwise_coverage')}

{link('Missing_factor_interactions.csv')}保存跨因子叶对的合法同现、实际联合次数、以Normal为锚的四个角点ID、已测角点数及直接可估状态。完整四点且合法才给直接DID；本定义下可直接估计目标数为{cov['direct_Normal_anchored_four_corner_targets']}。非法Normal角点不证明所有其他条件上下文都不可估，表中明确限定该上下文。

Motion×Haze、Motion×Gaussian/JPEG/Defocus、Haze×JPEG/曝光在VM36没有相应支持；复合条件里某些Defocus/噪声/JPEG同时出现，也不能自动分离其效应。跨分辨率/检测器效应与FM_P内部交互是不同维度。

## 4. 候选覆盖目录与四层准入

冻结{candidate['configurations']}个结构预定候选：24个baseline/单因素/约束必要上下文、7个双因素补点、1个RollingShutter逻辑见证。目录不是下一次自动跑满的实验矩阵，也不按已知AP挑选。完整属性及RNG见{link('Candidate_valid_configurations.csv')}与{link('Candidate_recipes.json')}。

{candidate_table}

四层结果：

{table(['层','实际完成','含义'],[['A SAT合法',f'{candidate["SAT_valid"]}/{candidate["configurations"]}','固定所有叶真假，经实际FlamaPy→CNF→SAT求解；三项非法负对照被拒绝'],
 ['B 像素验证',f'{candidate["pixel_validated"]}/{candidate["configurations"]}','2张合成图+1张现有val原图，各重复生成；shape/dtype/pixel SHA/轨迹与全局RNG检查'],
 ['C 标注坐标安全',f'{candidate["annotation_safe"]}/{candidate["configurations"]}','31项无几何坐标扭曲；RollingShutter阻塞；不宣称自然天气真实性'],
 ['D GPU准入','0','本轮CPU-only；独立数据、绑定、完整资源和新热验收未全部通过']])}

### 参数绑定不能伪称导师已经定义

Motion角度Uniform[0,180)、Haze airlight220、Rain stroke_length14/opacity.25、Rolling max_shift_fraction.008/waves2来自历史实现绑定提案，其原状态NEEDS_SCIENTIFIC_REVIEW。本次只将它们冻结为**CPU候选测试参数**，不改写历史批准记录，也不称为导师原参数或已经取得GPU执行批准。VM37 ALL_FIVE的Haze airlight255是另一训练策略值，不能混成同一测试条件。

RNG固定master_seed=20261116；每图每叶使用`SHA256(master_seed|image_stem|leaf)`前8字节big-endian作为NumPy default_rng种子，不含策略/分辨率，使匹配比较复用同一随机扰动。顺序维持atmosphere→illumination→blur→rolling→noise→downscale→JPEG。同一原图同条件只生成一次，再供所有检测器/推理尺寸复用。

原图坐标、类别、GT面积、ignore区域不随光度噪声/模糊/JPEG重新选目标；人工遮挡仍评价原始对象，不按可见性或AP删框。Downscale缩小再回原画布，只损失细节，不能冒称模拟了真实飞行高度。

### RollingShutter反例

合成原框`{bbox['synthetic_original_xyxy']}`，既有行位移生成器处理后前景包围框为`{bbox['warped_foreground_xyxy']}`。尺寸虽相同，框坐标已变。详情{link('RollingShutter_geometry_counterexample.json')}。没有可靠实例/ignore区域映射，本阶段不将它加入可评价集合，也未通过“框不变”放行。

{image('05_CPU_pixel_preview_only')}

此图为现有一张val图的CPU像素预览，无网络预测或AP。生成检查使用原图尺寸，网页缩略显示不改变实验输入。

## 5. 对照导师采样计划

导师原文第11页提出约200个P与约20个D、约4,000次评价作为初始规模估计，并要求先实测成本。P侧包含约60–100个pairwise条件、约100个均匀合法空间样本及后续边界主动采样；D侧为8–12个有目的训练checkpoint，再配推理特征的pairwise设计。这些是原计划，不是本项目已经完成的数量，也不是本轮执行授权。原文要求val仅用于选模、test-dev用于报告；当前开发结果必须继续明确标为开发证据。

{table(['导师采样目标','已实际做到','尚未做到'],[[r['method'],r['measured'],r['future']] for r in rows(out/'Supervisor_sampling_comparison.csv')])}

21个实际条件是20项历史固定目的性集合加Normal；另一个21点greedy覆盖核心和80逻辑cohort是不同总体。虽然能计算该集合的覆盖率，不能赋予它均匀纳入概率，不能用17/21估计整个46,368空间的失败流行率。

边界/主动采样需保存逐次查询、当前已知响应、选点规则及停止条件；现有21条件不能事后包装成主动学习。未来均匀样本需先声明排除未验证几何后的可评价总体，完整FM_P与该可评价子集的分母必须分开。

## 6. 未来统一Gap评价协议

冻结提案：`{protocol['id']}`，见{link('Future_Gap_protocol.json')}；源码对照见{link('Timing_protocol_comparison.csv')}。

VM36缓存管线包含：缓存文件完整性SHA→读盘/PNG解码→像素SHA→预处理/forward/NMS与CUDA同步→CPU框提取/校验→gzip预测及manifest逐图写出flush。模型加载、缓存初次生成、三次预热、COCO/Bootstrap不在逐图延迟，但均计入总worker预算。

VM37–39 `inference_ms`只覆盖预处理/forward/NMS及同步，排除之前的解码/退化/哈希和之后的框提取/输出。不允许把历史predict均值与其他耗时均值拼起来“恢复”逐图缓存管线，更不能虚造其配对区间。

新协议保留FP32、batch1、native10、dense one-to-many、square Letterbox、conf=.001、NMS IoU=.7、multi-label class-aware、max_det3000/max_nms30000，以及原GT/ignore/evaluator语义。每格3次预热，同PNG字节、同图像顺序、相同硬件/驱动/存储，记录分阶段时间与完整worker墙钟。flush不等于物理fsync；最终提交校验和原子改名另计总时间。

要求仍是Tiny AP95≥.10与缓存管线均值≤100ms；沿用原点区间判据，不偷换宽松规则。200次配对图像Bootstrap、seed20261116，所有格共用索引，重新累计COCO AP，延迟对同索引均值重采样。区间逐点、不是多重比较同时保证；无可靠航次ID时不宣称已控制航次聚类。三个seed样本SD与图像区间分开。

新`timing_protocol.py`以注入式预测/同步接口完成CPU边界测试，错误文件SHA、像素SHA和NaN输出都被拒绝；**没有运行真实YOLO、CUDA同步或网络tensor测试**。这只是可执行协议骨架，不是GPU性能验收。

{image('03_features_to_gap')}

## 7. 独立数据与历史使用复核

复用VM40对test-dev1610图/GT与train/val当前ID/字节不交叉的已接受证据，不再重复解码。对VM36明确引用的{len(history['records'])}份历史配置/协议做有界SHA与内容复核；发现{len(history['keyword_hits'])}条test-dev拼写引用。命中只代表保存文档中的引用，不能证明完成评价；零命中也不能证明没有手工使用。具体路径/行号见{link('Historical_testdev_records.json')}。

{table(['边界','结论'],[['当前ID/字节分离','VM40已验证，重用'],['新epoch10训练来源','固定6471 train /548 val；保留来源记录'],['旧权重创建时成员及完整选模历史','UNKNOWN / UNVERIFIED'],['历史test-dev推理后是否调参','UNVERIFIED'],['真实场景隔离','UNVERIFIED；文件名前缀仅代理'],['UAVDT原始DET/ignore/类别','仍阻塞；bus/car/truck/vehicle不能自动变为native10']])}

本轮仅从ZIP中央成员名统计前缀交集，见{link('Scene_proxy_inventory.csv')}和{link('Scene_proxy_overlap.json')}。有交集是审阅线索，无交集也不是场景隔离证明；没有可靠拍摄/航次身份清单，因此不升级准入。**Independent_data_readiness仍为DATASET_BLOCKED**。

## 8. 下一项最小实验及成本

首选是解决独立数据后，按冻结12格执行：3seed×CONTROL/GUIDED×Normal/Gaussian15，Infer1280，复用健康epoch10 EMA、不训练。主终点Gaussian AP95差，统一Tiny与时延也在同一批格记录；不先试一臂再决定另一臂。

独立数据暂未解决时，开发集12格计时桥接只能填补方法对照的完整Gap判据，不能改称独立确认。Motion/Haze补测是更低优先级：仅4个P（Normal、Motion21、Haze_dense、二者叠加）×ck01/ck05 Infer1280=8格，可完整估计该固定上下文Motion×Haze对比。它不代表全退化空间或不同训练seed。

{image('04_next_experiment_gates')}

{costs}

实测参考是既有16格×548图总10,808.362秒，含相应监督范围的推理/评价/200次Bootstrap；表内新方案均为外推，非实测。Motion/Haze会改变预测密度，缓存生成和统一写盘计时也未实测，不能因外推小于余额就宣告资源通过。

原账本已计费{cost['charged_seconds']:.3f}秒，剩{cost['remaining_seconds']:.3f}秒（{cost['remaining_seconds']/3600:.6f}h），固定期限{cost['absolute_expiry_UTC']}，本轮新增GPU0秒。89°C历史峰值不等于新的散热许可，继续保留90°C即停、87°C持续30秒停、2秒监测。当前任务没有GPU授权；任一数据/绑定/授权/成本/热关卡不通过都不得启动。

## 9. RQ1–RQ4科学状态与导师还需要什么

按导师原文第4页逐项对应，不能交换RQ编号：

{table(['RQ','原始研究问题'],[[r['RQ'],r['supervisor_goal']] for r in rows(out/'Supervisor_RQ1_RQ4_evidence_matrix.csv')])}

{table(['RQ','已接受证据','不足/下一步','状态'],[[r['RQ'],r['accepted_evidence'],r['remaining'],r['status']] for r in rows(out/'Supervisor_RQ1_RQ4_evidence_matrix.csv')])}

导师仍需看到：原始计划的可解释采样覆盖、可比较Tiny与延迟、固定策略的独立确认，以及正式1280/100轮和YOLOv8s五项单增强消融。现有960/10轮结果可作为探索依据，不能代替正式RQ4。

Variability-driven规则保留“按CONTROL两尺寸平均AP损失选家族、固定Gaussian p=.25及原σ范围”。这条规则有解释性，但p不是最优性证明；在同一开发集反复挑更好的参数会继续扩大选择偏差。独立集不用于改规则或选checkpoint。

## 10. 运行、文件与验收

本Run：`{out}`。全部CSV、JSON、像素轨迹、日志与图均在该Run；中文报告保持平铺在`datasets/reportcn/`，由Run索引和校验清单引用，不复制历史训练归档。

{link('FM_P_measured_coverage.csv')}、{link('Missing_factor_interactions.csv')}、{link('Candidate_valid_configurations.csv')}、{link('Candidate_annotations_safety.csv')}、{link('Timing_protocol_comparison.csv')}、{link('Independent_data_readiness.json')}、{link('Complete_pair_cost_plan.json')}、{link('Supervisor_RQ1_RQ4_evidence_matrix.csv')}、{link('Source_SHA256_manifest.json')}。

代码关系：`run.py`读取证据并生成候选/关卡；`sat_check.py`在现有FlamaPy环境核查候选；`timing_protocol.py`定义缓存计时边界；`reporting.py`从结果表画图、生成本地说明。源码均在`{root / 'main/experiments/experiments_test/vm41_degradation_coverage'}`。

```powershell
Set-Location -LiteralPath 'E:\\ForStudy\\code\\uavgap'
.\\.venv_rq4\\Scripts\\python.exe -B -X utf8 .\\main\\experiments\\vm41_degradation_coverage\\run.py --audit
```

上述命令新建CPU Run；不启动GPU。只检查当前已完成文件：

```powershell
.\\.venv_rq4\\Scripts\\python.exe -B -X utf8 .\\main\\experiments\\vm41_degradation_coverage\\run.py --verify {out.name}
```

CPU墙钟见`resource_usage.json`；GPU=0，无新模型AP。原VM36–VM40文件、原UVL、旧SHA和预算记录保持不变。没有Git pull/rebase/commit/push。当前边界到此停止。
'''
    report.write_text(text,encoding='utf-8')
    guide=f'''# VM41 导师会议讲解与问答

RunID：`{out.name}`。本轮只做CPU证据和候选生成，没有新YOLO结果。

## 先讲这次补了什么

“老师，VM40确认原模型没改错，但实际测量没有覆盖所有退化。这次我逐叶数了：25个叶里18个出现过，7个没有。接着把缺少的单因素和双因素对照列出来，检查哪些组合合法、哪些能确定性生成、哪些GT能沿用。最后统一了未来Tiny AP和时延的评价边界。”

## 没测与失败有什么不同

没测就是没有该条件的检测结果。失败必须真实测量后，再按预定精度与时延要求判断。Motion、Haze、Rain、RollingShutter没有出现在VM36的21条件里，所以不能把它们画成性能FAIL。

我们已有的17个Gap，意思是对应条件下四个已测检测设置都没通过要求，不代表所有可能的YOLO设置都失败，更不表示整个46,368空间有17/21会失败。

{image('01_measured_leaf_coverage')}

## 为什么“18叶出现”仍不足

比如JPEG50虽然出现在复合条件里，但没有干净图只加JPEG50的单因素角点，不能把其他噪声/模糊的影响排掉。Defocus和Downscale也类似。

要看Motion与Haze是否相互影响，最直接是同一批图做四个条件：Normal、只Motion、只Haze、Motion+Haze。只测最后一个组合，不能说发现了交互。四个条件都测完才可以算差中之差：两者一起的AP−只Motion的AP−只Haze的AP+Normal的AP。

逻辑pairwise覆盖只要求选项对出现，不要求这四个角点完整，也不说明有统计功效。本次把“同现覆盖”和“直接可估”分开，避免混用。

## 为什么不直接补所有单因素

原模型有依赖。LowLight必须带ShotNoise，所以“LowLight单独出现”本来就不合法。要研究它，可以比较两边都有Shot_low的Normal与LowLight；那是固定噪声条件的曝光效应。

Shot_high也要在LowLight或强欠曝下才合法。不能把不允许的组合补进去，再把它说成覆盖提升。

## 这32个候选是不是32次新实验

不是。它们是32个准备目录，包含24个baseline/单因素或约束必要条件、7个双因素角点，以及1个暂时不能评价的RollingShutter逻辑见证。本轮只在2张合成图和1张现有val图上检查生成，重复后像素SHA与轨迹相同；没有运行检测器。

SAT32个合法，像素32个可以生成，31个可按固定原对象任务保留坐标，GPU准入0个。没有新AP，也不能从像素图看起来合理就说证明了真实天气。

## Motion方向、雾和雨为什么还要审核

导师UVL只给Motion核7/15/21，没有给方向规则；Haze只给β，没有给airlight。历史代码有角度均匀0–180、airlight220等数值，但它们的状态仍是待审核实现绑定。

这次为了CPU检查，明确使用那套历史提案并保存参数，没有说老师原文里写了这些值，也没有把它升级成GPU批准。ALL_FIVE训练的Haze airlight255是另一个策略参数，不能与airlight220的测试条件混为一谈。

## RollingShutter为什么连框都不能直接沿用

光照与JPEG主要改变像素外观；RollingShutter会让不同行横向移动，物体坐标会变。我们用合成框做了反例：原框{bbox['synthetic_original_xyxy']}变为{bbox['warped_foreground_xyxy']}。图宽高没变，框仍然变了。

还没验证实例框与ignore区域怎样一起变换，所以把它挡在标注安全关卡外。不能只看到图像shape一致就批准原框不变。

## 怎样保证同一退化可重复

master seed固定20261116，再用图像ID和每个叶特征名生成独立随机种子。种子不含检测器和推理分辨率，所以同图同条件的噪声与Motion角度可复用。

先按原生图像生成并保存/校验同一份像素，再分别送入640或1280的网络Letterbox。不能为每个分辨率重新生成噪声，否则比较时变了两件事。

## 为什么VM37–39的快时延不能直接判断Gap

VM36测的是读缓存、核对SHA、解码、模型预测、取框和写结果这一整段。VM37–39保存的`inference_ms`只测中间predict，不含前后工作。

原要求是Tiny AP至少.10，且缓存管线均值不超过100ms。这两个时间不是同一把尺；也不能拿几个历史平均耗时相加就造出逐图管线或置信区间。因此旧AP可以继续展示，但“方法让Gap变小”还不能下结论。

{image('03_features_to_gap')}

## 新协议怎样比较

所有策略用同一份PNG缓存与GT，同样的三次预热、FP32、batch1、固定head/NMS，计时到每张预测和manifest写出flush结束。模型加载、生成缓存和COCO不算逐图延迟，但算完整任务预算。flush不是硬盘断电持久化保证。

CPU接口测试使用模拟预测回调，验证了完整性失败和NaN拒绝，不能说真实网络或CUDA已经测试通过。

200次逐图配对Bootstrap使用相同图像位置索引，重新累计AP；三个训练seed仍只有3个。区间没有给全部条件同时覆盖保证，也没解决航次聚类。

## 独立数据现在差在哪里

test-dev原图/标注及当前train/val字节分离在VM40已验过。本轮复核了有直接来源的历史配置，但没有完整的创建时训练成员、选模和人工调参记录；没命中test-dev关键词不等于没用过。

文件名第一段可作排查代理，却不能当已证实的航次ID。有相同前缀值得查，无相同前缀也不能证明不同场景。UAVDT的原DET/ignore来源和类别问题仍没解决，不能把车辆类别自动改成VisDrone十类来放行。

## 下一步到底做哪个

优先补独立数据准入，再做12格：三个seed、CONTROL与GUIDED、Normal与Gaussian15，统一1280推理，不重训、不调整Gaussian策略。这样才能验证现有收益能否迁移到合格新图像，并同时补上Tiny与同口径时延。

如果独立数据仍卡住，可另行授权开发集计时桥接，但必须说它只补Gap判据，不补独立性。Motion/Haze四点8格是后续覆盖实验，不用因为已经有候选目录就全部开跑。

目前余额约{cost['remaining_seconds']/3600:.3f}小时，独立12格含余量外推约9.43小时，而且完整实际成本未测。历史89°C也不等于新热准入。预算、数据、绑定、授权与散热任一不通过，都不启动GPU。

{image('04_next_experiment_gates')}

## 导师可以听到的结论

可以说：树与约束有来源，覆盖缺口已经逐叶量化，新候选有真实SAT和CPU像素证据，RollingShutter没有被错误放行，下一步对照与计时规则已冻结。

不能说：补测完成、未测退化失败、32项有新AP、独立泛化已证明、Gaussian收益等于Gap闭合、完整正式RQ4已完成。

科学报告：[{report.name}]({report.as_posix()})。本地网页：[{out.name}]({(out/'index.html').as_posix()})。

文件说明：`FM_P_measured_coverage.csv`看25叶；`Missing_factor_interactions.csv`看缺失四角点；`Candidate_valid_configurations.csv`看候选ID和具体参数；`Candidate_annotations_safety.csv`看四层关卡；`Future_Gap_protocol.json`看统一规则；`Independent_data_readiness.json`看数据阻塞；`Complete_pair_cost_plan.json`看完整配对成本；`Source_SHA256_manifest.json`查来源。

`run.py --audit`只运行CPU并新建Run，`--verify <RunID>`只核对已生成文件。代码与命令见科学报告，不需要打开或复制旧145GiB研究归档。
'''
    beginner.write_text(guide,encoding='utf-8')
    return [report,beginner]


def page(out,names,reports):
    esc=html.escape
    pics=''.join(f'<figure><img src="plots/{n}.png" alt="{n}"><figcaption>{n}</figcaption></figure>' for n in names)
    docs=''.join(f'<li><a href="{p.as_uri()}">{esc(p.name)}</a></li>' for p in reports)
    data=''.join(f'<li><a href="{p.name}">{esc(p.name)}</a></li>' for p in sorted(out.glob('*.csv')))
    (out/'index.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>VM41 coverage</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#243746;background:#f6f8fb}}figure{{background:white;padding:20px;margin:24px 0;border:1px solid #dae3ea}}img{{width:100%}}a{{color:#226b97}}code{{word-break:break-all}}</style>
<h1>VM41 · Degradation coverage and future Gap protocol</h1><p><code>{esc(out.name)}</code></p>
<p>CPU audit complete with limitations. Independent data blocked. No GPU, training, new detector AP or historical rerun.</p>
<h2>Chinese explanations (local only)</h2><ul>{docs}</ul><h2>Evidence tables</h2><ul>{data}</ul>{pics}</html>''',encoding='utf-8')
