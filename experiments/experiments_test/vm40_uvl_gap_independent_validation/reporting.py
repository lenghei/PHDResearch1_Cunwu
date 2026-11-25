"""Figures and Chinese explanations from saved audit tables."""
import csv
import html
import json
from pathlib import Path


def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        return list(csv.DictReader(stream))


def table(headers,data):
    def cell(x):
        return str(x).replace('|',' / ').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
                      *['| '+' | '.join(cell(x) for x in row)+' |' for row in data]])


def charts(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    import numpy as np
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        from matplotlib.font_manager import fontManager,FontProperties
        fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=font).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    dest=out/'plots';dest.mkdir(exist_ok=True)
    produced=[]
    def finish(fig,name):
        fig.savefig(dest/(name+'.png'),dpi=170,bbox_inches='tight',facecolor='white')
        fig.savefig(dest/(name+'.pdf'),bbox_inches='tight',facecolor='white');plt.close(fig)
        produced.append(name)
    fig,ax=plt.subplots(figsize=(12,5));ax.axis('off')
    boxes=[(.22,.84,'FM_P：图像退化\n25叶 / 5约束 / 46,368合法配置'),
           (.78,.84,'FM_D：检测器选择\n34叶 / 4约束 / 635,904合法配置'),
           (.5,.60,'FlamaPy解析、SAT、无死特征\nPDF树/属性/约束与冻结UVL一致'),
           (.5,.36,'历史有目的选取：20条件 + Normal\n4检测设置 × 548图 = 84对 / 46,032张次'),
           (.24,.10,'开发集AP、Tiny AP + 缓存管线时延\n仅判定已测四检测器子集Gap'),
           (.79,.10,'独立验证：DATASET_BLOCKED\n无新GPU测量；不声称Gap缩小')]
    for x,y,t in boxes:ax.text(x,y,t,ha='center',va='center',transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=.65',fc='#eaf2f8' if y>.1 else '#fff2df',ec='#54778b'))
    for a,b in [((.22,.75),(.45,.68)),((.78,.75),(.55,.68)),((.5,.51),(.5,.44)),((.44,.27),(.25,.18)),((.56,.27),(.78,.18))]:
        ax.annotate('',xy=b,xytext=a,xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#54778b'))
    fig.suptitle('VM40：从导师特征模型到已测经验事实',fontsize=15);finish(fig,'01_feature_model_flow')
    vm=rows(out/'VM36_84_pair_metrics.csv');conditions=rows(out/'VM36_21_measured_conditions.csv')
    ids=[r['configuration_id'] for r in conditions];settings=[('ck05',640),('ck05',1280),('ck01',640),('ck01',1280)]
    lookup={(r['condition_id'],r['checkpoint'],int(r['resolution'])):r for r in vm}
    fig,axes=plt.subplots(1,3,figsize=(12.5,10),sharey=True)
    for ax,metric,title in zip(axes,['AP50','AP50_95','Tiny_AP'],['AP50','AP50:95','Tiny AP50:95']):
        data=np.array([[float(lookup[p,c,r][metric])*100 for c,r in settings] for p in ids])
        im=ax.imshow(data,cmap='Blues',aspect='auto',vmin=0)
        ax.set_xticks(range(4),['v8s 640','v8s 1280','26s 640','26s 1280'],rotation=35,ha='right')
        ax.set_yticks(range(21),[f'C{i+1:02}' for i in range(21)])
        for i in range(21):
            for j in range(4):ax.text(j,i,f'{data[i,j]:.1f}',ha='center',va='center',fontsize=8,color='white' if data[i,j]>data.max()*.65 else '#172b3a')
        ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.4,label='AP (%)')
    fig.suptitle('VM36：84个已接受的开发集评价对；C01–C21见条件清单',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.96));finish(fig,'02_vm36_detector_degradation')
    matched=rows(out/'MATCHED_56_configuration_metrics.csv')
    values={(int(r['seed']),r['strategy'],int(r['resolution']),r['condition']):r for r in matched}
    contexts=['Normal','JPEG_Q25','JPEG_Q75','Gaussian_s15'];arms=['CONTROL','ALL_FIVE','VARIABILITY_GUIDED']
    colors=['#386a9f','#c88736','#258078'];x=np.arange(4)
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for i,res in enumerate((640,1280)):
        for j,metric in enumerate(('AP50_95','Tiny_AP')):
            ax=axes[i,j]
            for k,arm in enumerate(arms):
                ax.bar(x+(k-1)*.25,[float(values[42,arm,res,c][metric])*100 for c in contexts],.24,label=arm,color=colors[k])
            ax.set_xticks(x,['Normal','JPEG25','JPEG75','Gaussian15']);ax.set_ylabel('AP (%)');ax.set_title(f'{res} / {metric}')
            ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0,0].legend(fontsize=8);fig.suptitle('三策略仅比较共同的seed42；ALL_FIVE无seed43/44',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.95));finish(fig,'03_seed42_three_strategies')
    summary=rows(out/'VM39_three_seed_summary.csv')
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for ax,size in zip(axes,('all','tiny')):
        for i,res in enumerate((640,1280)):
            item=next(r for r in summary if int(r['resolution'])==res and r['condition']=='Gaussian_s15' and r['size']==size)
            gains=[float(item[f'seed{s}_difference'])*100 for s in (42,43,44)]
            ax.scatter([i-.08,i,i+.08],gains,s=40,color=colors[0]);ax.errorbar(i,float(item['mean'])*100,yerr=float(item['sample_seed_SD'])*100,
                color=colors[2],fmt='D',capsize=6,label='均值 ± seed样本SD' if i==0 else None)
            for xx,y,s in zip([i-.08,i,i+.08],gains,(42,43,44)):ax.annotate(str(s),(xx,y),xytext=(5,5),textcoords='offset points',fontsize=8)
        ax.axhline(0,color='#888',lw=.7);ax.set_xticks([0,1],['640','1280']);ax.set_ylabel('GUIDED − CONTROL (pp)')
        ax.set_title('Gaussian AP50:95' if size=='all' else 'Gaussian Tiny AP');ax.legend(fontsize=8)
    fig.suptitle('三个训练seed的匹配差；SD不是95%置信区间',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.94));finish(fig,'04_three_seed_gaussian_gains')
    fig,ax=plt.subplots(figsize=(8,5))
    for i,res in enumerate((640,1280)):
        for seed in (42,43,44):
            c,g=values[seed,'CONTROL',res,'Normal'],values[seed,'VARIABILITY_GUIDED',res,'Normal']
            cg,gg=values[seed,'CONTROL',res,'Gaussian_s15'],values[seed,'VARIABILITY_GUIDED',res,'Gaussian_s15']
            cost=(float(c['AP50_95'])-float(g['AP50_95']))*100;gain=(float(gg['Tiny_AP'])-float(cg['Tiny_AP']))*100
            ax.scatter(cost,gain,color=colors[i*2],marker=['o','s'][i],s=65,label=str(res) if seed==42 else None)
            ax.annotate(str(seed),(cost,gain),xytext=(5,5),textcoords='offset points')
    ax.axhline(0,color='#888',lw=.7);ax.axvline(0,color='#888',lw=.7)
    ax.set(xlabel='Normal AP50:95代价：CONTROL − GUIDED (pp)',ylabel='Gaussian Tiny AP增益 (pp)',title='开发集取舍：Gaussian小目标改善与正常图像精度代价')
    ax.legend(title='推理尺寸');finish(fig,'05_tiny_gain_clean_cost')
    codes={'FAIL':0,'UNCERTAIN':1,'PASS':2};data=[]
    for pid in ids:
        line=[codes[lookup[pid,c,r]['original_rule_state']] for c,r in settings]
        subset=2 if 2 in line else 0 if all(v==0 for v in line) else 1;data.append(line+[subset])
    fig,ax=plt.subplots(figsize=(8,9));im=ax.imshow(data,cmap=ListedColormap(['#d8a7a1','#f1d892','#92c5b8']),norm=BoundaryNorm([-.5,.5,1.5,2.5],3),aspect='auto')
    ax.set_xticks(range(5),['v8s640','v8s1280','26s640','26s1280','四设置子集'])
    ax.set_yticks(range(21),[f'C{i+1:02}' for i in range(21)])
    for i,line in enumerate(data):
        for j,v in enumerate(line):ax.text(j,i,['GAP' if j==4 else 'FAIL','UNCERTAIN','COVERED' if j==4 else 'PASS'][v],ha='center',va='center',fontsize=8)
    ax.set_title('VM36原规则：17 GAP / 4 COVERED\n未建立增强前后可比Gap变化；Normal仅为新增参考',pad=15)
    finish(fig,'06_vm36_subset_gap_only')
    return produced


def reports(out,a,resources):
    load=a.load
    uv=load(out/'UVL_validation.json');gate=load(out/'independent_data_admission.json');budget=gate['budget']
    conditions=rows(out/'VM36_21_measured_conditions.csv');vm=rows(out/'VM36_84_pair_metrics.csv')
    summary=rows(out/'VM39_three_seed_summary.csv');matched=rows(out/'MATCHED_56_configuration_metrics.csv')
    arguments=rows(out/'CONTROL_actual_training_args.csv');family=rows(out/'ALL_FIVE_policy_verified.csv')
    guided=rows(out/'GUIDED_realized_frequencies.csv');props=load(out/'policy_realization_details.json')
    per_seed=rows(a.P39/'per_seed_comparisons.csv')
    folder=a.ROOT/'datasets/reportcn'
    science=folder/'VM40_实验配置与独立验证_科学报告_CN.md';guide=folder/'VM40_导师会议详细讲解与问答_CN.md'
    if science.exists() or guide.exists():
        science=folder/('VM40_实验配置与独立验证_科学报告_CN_'+out.name+'.md')
        guide=folder/('VM40_导师会议详细讲解与问答_CN_'+out.name+'.md')
        if science.exists() or guide.exists():raise FileExistsError('Reports already exist for this RunID.')
    def output_link(name):return f'[{name}]({(out/name).as_posix()})'
    def picture(name):return f'![{name}]({(out/"plots"/(name+".png")).as_posix()})'
    def tree(domain):return '```text\n'+(out/(domain+'_tree.txt')).read_text(encoding='utf-8')+'```'
    def percent(x):return f'{float(x)*100:.4f}'
    def brief_delta(size='all'):
        return table(['推理尺寸','条件','seed42 Δpp','seed43 Δpp','seed44 Δpp','平均Δpp','样本SD pp'],
            [[r['resolution'],r['condition'],percent(r['seed42_difference']),percent(r['seed43_difference']),percent(r['seed44_difference']),percent(r['mean']),percent(r['sample_seed_SD'])]
             for r in summary if r['size']==size])
    condition_table=table(['序号 / 配置ID','光照 gain,bias','Defocus r','Gaussian σ','Shot photons','JPEG Q','缩小倍数','选中叶特征'],
        [[f'C{i+1:02} / {r["configuration_id"]}',f'{r["illumination"]} / {r["gain"]},{r["bias"]}',
          r['defocus_radius'],r['Gaussian_sigma'],r['Shot_photons'],r['JPEG_quality'],r['downscale_factor'],r['selected_leaves']] for i,r in enumerate(conditions)])
    parameter_keys=['architecture','initialization_SHA256','training_images','development_images','imgsz','epochs','batch','nbs','optimizer','lr0','lrf','momentum','weight_decay',
        'warmup_epochs','warmup_momentum','warmup_bias_lr','cos_lr','seed','deterministic','workers','amp','hsv_h','hsv_s','hsv_v','degrees','translate','scale','shear','perspective',
        'fliplr','flipud','mosaic','close_mosaic','mixup','copy_paste','cutmix','EMA_selection']
    args_table=table(['参数','实际值','冻结来源类别','有效行为'],[[r['parameter'],r['EXACT_VALUE'],r['EXPLICIT_OVERRIDE_OR_LIBRARY_DEFAULT'],r['EFFECTIVE_RUNTIME_BEHAVIOR']]
        for k in parameter_keys for r in arguments if r['parameter']==k])
    selected_vm=[r for r in vm if r['condition_id'] in ('p_59915f60d81c6f17','p_5a06deca286f53b2','p_9031c68f63a468d1','p_cdbfeb94b010c58d')]
    vm_table=table(['条件','权重','推理尺寸','AP50 %','AP50:95 %','Tiny AP %','缓存管线ms','原规则'],
        [[r['condition_id'],r['checkpoint'],r['resolution'],percent(r['AP50']),percent(r['AP50_95']),percent(r['Tiny_AP']),f'{float(r["cached_pipeline_mean_ms"]):.2f}',r['original_rule_state']] for r in selected_vm])
    strategy_table=table(['尺寸','条件','策略 / seed42','AP50:95 %','Tiny AP %'],
        [[r['resolution'],r['condition'],r['strategy'],percent(r['AP50_95']),percent(r['Tiny_AP'])] for r in matched if r['seed']=='42'])
    ci_table=table(['seed','推理尺寸','Gaussian AP95差 pp','已有逐图配对95%区间 pp'],
        [[r['seed'],r['resolution'],percent(r['GUIDED_minus_CONTROL']),str([round(float(x)*100,4) for x in json.loads(r['difference_CI95'])])]
         for r in per_seed if r['context']=='Gaussian_s15' and r['size']=='all' and r['metric']=='map50_95'])
    constraints=rows(out/'UVL_constraints_explained.csv')
    constraint_text=lambda d: table(['约束','中文解释'],[[r['constraint'],r['explanation_cn']] for r in constraints if r['domain']==d])
    elapsed=f'{resources["measured_successful_CPU_stages_seconds"]:.3f}'
    text=f'''# VM40 实验配置、独立验证与经验Gap审计

日期：2026-11-16。RunID：`{out.name}`。

状态：**CPU_AUDIT_COMPLETE_WITH_LIMITATIONS / DATASET_BLOCKED**。未新增GPU推理或训练，未复跑已接受的VM36–VM39实验，未推送GitHub。

## 01. 研究问题与导师原始要求

本轮检查“问题域退化FM_P × 解决域检测器FM_D → 合法配置 → 实测响应 → 固定要求下的经验Gap”。先恢复导师原文与实际实验，再判断Gaussian定向增强证据能否扩展到独立图像。

原文：`{a.ROOT / 'datasets/research_archive/supervisor/Research_plan_Cunwu_UAV_gap.pdf'}`，SHA256 `2f7620a017843e6f638591e85d373ead99c6e8192ab58fa3153665ed22b124af`。本次重新读取全部17页；PDF提取模型与实际UVL的完整树、属性、约束AST逐项一致。

正式1280/100epoch、YOLO26s与YOLOv8s、seed42/43/44以及导师原定YOLOv8s五种单增强消融均未修改。已完成的960/10epoch只属于已授权探索扩展。

{picture('01_feature_model_flow')}

## 02. FM_P UVL 完整解析

原文件：`{a.VM36 / 'feature_models/fm_p.uvl'}`。SHA256 `{uv['FM_P']['UVL_SHA256']}`。

叶特征25、跨树约束5、合法配置46,368；SAT可满足，死特征为空。使用已安装FlamaPy 2.6.0 / uvlparser 2.5.0及既有已验收计数流程复算。不是新增独立BDD或#SAT验证。

Mandatory表示父被选时子必选；Optional允许不选；Alternative在父有效时恰选一个；OR至少选一个、可多选。没有选中可选父节点时，其子节点均不选。例如Blur的Motion与Defocus可以叠加，并非只能二选一。

{tree('FM_P')}

{constraint_text('FM_P')}

这些物理合理性约束原样来自导师草案，逻辑满足不等于已经证明在所有真实天气中成立。RollingShutter等涉及几何的分支还需要独立GT正确性验收；本21条件不含该分支。

数值与单位见{output_link('UVL_numeric_attributes.csv')}：gain为乘数，bias与Gaussian σ使用uint8强度刻度，kernel/radius使用原生像素，JPEG为编码质量参数，Downscale为缩小倍率。photons是Poisson生成器刻度，Haze β是常深度代理；不是实测传感器光子数、能见度或飞行高度。UVL自身未声明单位，CSV将单位解释与原属性分开保存。

## 03. FM_D UVL 完整解析

原文件：`{a.VM36 / 'feature_models/fm_d.uvl'}`。SHA256 `{uv['FM_D']['UVL_SHA256']}`。

叶特征34、跨树约束4、合法配置635,904，训练变体844；SAT可满足，无死特征。计数采用原有精确因子分解，先验证移除数值属性后的Boolean投影严格相同再转移证明；没有把参考数字直接当计算结果。

{tree('FM_D')}

{constraint_text('FM_D')}

Nano/Small的2.5M/10M是原草案名义属性，并非此次实际checkpoint参数量测量。原FM_D有Infer960，**没有Train960**；本次960训练属于APPROVED_EXTENSION，不伪称原模型叶。Albu_Default原属性只有Blur/Median/CLAHE，实际库还启用ToGray 0.01，属于运行细节补充，未修改原UVL。类别、数据来源、GT及推理实现兼容性不能只靠一个逻辑有效配置自动获得。

## 04. 21种实际退化条件完整清单

实际集合是**20个历史固定条件 + 后加Normal**，其中17个复合条件、JPEG_q25/JPEG_q75两个单因素诊断、Gauss_s15一个单因素诊断及Normal参考。下面C编号只便于图表阅读，原配置ID不变。

{condition_table}

所有21项的MotionBlur、Haze、Rain、RollingShutter均为NOT_PRESENT；Motion方向为NOT_APPLICABLE，不能写UNKNOWN角度或补造退化。Downscale先按冻结生成器缩小再回到原坐标画布；实际网络Letterbox随后执行，GT尺寸层仍按原图定义。

完整每项属性、约束状态、四个来源目录、设计SHA见{output_link('VM36_21_measured_conditions.csv')}。17个复合条件继承历史固定cohort，逐项为何选择该组合的独立说明未保留，记为UNKNOWN，不能追认成均匀随机设计。JPEG端点、Gaussian诊断和Normal参考的用途可恢复。

另一个21点pairwise greedy core和80配置逻辑覆盖cohort是不同总体，不能替换这21项。它们的配置覆盖率不能拿来描述本21项的随机代表性；VM36的80已测对是20条件×4设置，也不是80个FM_P条件。

## 05. 四种检测设置的实际定义

{table(['设置','训练尺寸','推理尺寸','配置训练轮数','选择边界'],[[r['setting_id'],r['training_resolution'],r['resolution'],r['training_epochs'],r['training_epochs_meaning']] for r in rows(out/'VM36_4_detector_settings.csv')])}

ck01 YOLO26s SHA256：`1b3861902e01416f199f2242cb43618f7680848cabf5f5df9546f2b404f0c488`。

ck05 YOLOv8s SHA256：`96c84ece325eccf07edefd231f088f758ba5fa6a75a4b577e8db32a94bfeeb8b`。

两份实际文件已重新计算SHA，与冻结设计一致。历史args配置100轮、1280、seed42，但best的实际选中epoch及当时Albu具体启用日志并未由本次证明。不能用“配置100轮”代替best来源的全部历史信息。

完整模型路径、来源身份和推理参数见{output_link('VM36_4_detector_settings.csv')}。本次FP32、one-to-many路径、class-aware multi-label NMS，conf=0.001、IoU=0.7、max_det=3000、max_nms=30000；不混用端到端head。

## 06. VM36实验方法及已测结果

21×4=84个评价对，每对548张，共46,032张次；不是84次训练。每个已接受对的配置、指标、Bootstrap、计时和verification重新校验保存SHA，原预测/GT保留原始路径和SHA引用；本轮未重复解压全部历史预测或重新做COCO。

评价为native10、原图bbox与面积、既有ignore-union过滤及COCO IoU .50:.05:.95、101 recall points。Tiny为原始面积小于16²，Small为16²至32²的冻结分层；库内默认small口径不能混入。完整84行AP50/AP95/Tiny/延迟/标签见{output_link('VM36_84_pair_metrics.csv')}。四个诊断条件如下。

{vm_table}

{picture('02_vm36_detector_degradation')}

缓存管线计时包含缓存SHA、读盘/解码、像素SHA、模型预处理/forward/NMS、CPU框提取与预测/manifest写出flush。每条件有3次预测预热；不包含退化缓存初次生成、模型加载、COCO和Bootstrap。该边界是100ms要求的适用范围，不等于部署传感器总延迟。

## 07. CONTROL实际训练参数及来源

参数从`ordinary_training_args.json`、实际pilot protocol、可用trainer args及运行日志恢复。每个字段同时列冻结值、安装版默认值、来源、有效行为和未知事项：{output_link('CONTROL_actual_training_args.csv')}。值等于库默认不证明原命令没有显式传入；原CLI/default来源无法仅凭args反推，已标UNKNOWN。

{args_table}

普通增强：HSV h=.015/s=.7/v=.4，translate=.1、scale=.5、fliplr=.5，其余表中零值关闭。Albumentations 2.0.8：Blur/MedianBlur/ToGray/CLAHE各0.01，BrightnessContrast/Gamma/ImageCompression为0。auto_augment=randaugment与erasing=.4是分类任务字段，不作为本detect实验有效增强。

**Mosaic配置为1.0，但10−close_mosaic10=0，epoch0关闭，日志确有Closing dataloader mosaic。** 不能把Mosaic称为10轮中实际启用；正式100轮仍按原计划，不受此解释修改。

warmup实际9708个batch，累积从1渐增到32；不是固定32从头训练，也不是warmup关闭。每臂64,710曝光、32,360训练batch、1,816次optimizer/EMA更新，末17个microbatch未额外flush。AdamW beta1=.937/beta2=.999；warmup_momentum=.8虽在args中，但该库分支仅插值带momentum键的参数组，不能说AdamW beta1也按该字段warmup。bias组从0.1开始，常规lr0=.01。固定epoch10 EMA，不按B集挑best。

导师代码分析版是Ultralytics8.4.33；实际运行8.4.210、Torch2.6.0+cu124、FP32、AMP=False、TF32=False。版本不同已记录，没有声称逐字节等价。Workers=0，两个CPU算术线程不等于两个DataLoader进程。内部YOLO验证max_det=300，研究评价max_det=3000，结果分开。

## 08. ALL_FIVE五类随机增强实现

固定顺序Haze→Exposure→MotionBlur→Noise→JPEG，逐家族独立p=.5。

{table(['家族','冻结参数','实际激活 / 64710','实际比例'],[[r['family'],r['frozen_parameters'],r['active'],f'{float(r["actual_frequency"])*100:.3f}%'] for r in family])}

Motion实际核仅9/11/13/17/19，角度0–180°；Gaussian σ5.5–30、Shot photons20–100，两噪声子类各0.5。拒绝精确冻结测试参数：Motion7/15/21、Gaussian5.5/15/30、Shot20/100、JPEG25/50/75。JPEG实际整数26–49、51–74；连续分布排除端点/精确值不等于留出邻域。

本次读取64,710条真实增强轨迹，验证顺序、核集合、JPEG排除和实际噪声子类。无额外家族激活实际{props['no_extra_effect_count_actual']}次（{props['no_extra_effect_count_actual']/64710*100:.3f}%），五家族全激活实际{props['ALL_FIVE_joint_active_family_count']['5']}次；理论全不激活为0.5⁵=3.125%。名称不是“每图五种都做”，也不是从合法FM_P空间均匀采样。普通增强仍继续。

## 09. GUIDED策略与三个seed结果

GUIDED仅Gaussian，p=.25，σ5.5–30，精确测试值排除语义不变，另四家族关闭。p=.25是预先固定保留75%输入的先验，没有验证其最优。家族依据已看到的CONTROL两分辨率平均AP损失选择，属于POST_HOC_POLICY_SELECTION，不是在线控制器。Noise schema保留photons字段，但Gaussian概率1时不会采样Shot。

{table(['策略','共同普通增强','额外退化','已有seed'],[['CONTROL','相同；Mosaic全关','无','42/43/44'],['ALL_FIVE','相同','五独立门各0.5','仅42'],['GUIDED','相同','仅Gaussian门0.25','42/43/44']])}

{table(['seed','激活数 / 64710','实测概率','σ实测范围'],[[r['seed'],r['active'],f'{float(r["actual_probability"]):.6f}',f'{float(r["sigma_min"]):.6f}–{float(r["sigma_max"]):.6f}'] for r in guided])}

以下为GUIDED−CONTROL，所有AP原值保存在CSV，展示乘100为pp；SD采用ddof=1。

{brief_delta()}

Gaussian Tiny AP的640均值+0.3533 pp、1280均值+2.5056 pp；1280样本SD0.0667 pp。1280 Normal AP平均代价0.5737 pp。JPEG_Q25在640符号不稳定，JPEG_Q75的总体AP三seed均下降，不能称为全面鲁棒性改善。

{ci_table}

区间来自保存的配对逐图Bootstrap：每次同索引重采样GT/预测、重新累计AP，不平均单图AP。是固定一个训练seed、固定开发图像总体下的逐点区间，未做多条件同时覆盖校正，不能当训练seed显著性或真实航次聚类控制。

三策略共同seed42的绝对结果：

{strategy_table}

{picture('03_seed42_three_strategies')}

{picture('04_three_seed_gaussian_gains')}

{picture('05_tiny_gain_clean_cost')}

## 10. 训练与评价成本说明

本轮GPU使用与GPU账本新增计费均为0秒。可计时的成功CPU采集阶段合计{elapsed}秒；后续绘图/校验计时见{output_link('resource_usage.json')}。这是分阶段CPU墙钟，不是完整聊天耗时或CPU核时。未统一计时的实现/中断诊断不补造总秒数；另一次ZIP根成员路径解析失败的已知调用墙钟5.078秒单独记录，修复仅影响VM40审计代码。

原36小时账本当前已计费{budget['charged_seconds']:.3f}秒，剩余{budget['remaining_seconds']:.3f}秒（{budget['remaining_seconds']/3600:.6f}小时），期限{budget['absolute_expiry_UTC']}。VM39新增worker实测53,009.908秒；这些worker墙钟含监督范围中的初始化、训练、CPU评价等，并非CUDA kernel占用时间。

独立12格×1610图共19,320张次。仅按既有16格×548图、10,808.362秒线性推算约{gate['cost_screening']['linear_extrapolation_seconds_NOT_MEASURED']/3600:.3f}小时；加35%不确定性及1800秒停止余量约{gate['cost_screening']['planning_upper_seconds_NOT_MEASURED']/3600:.3f}小时，超过当前余额。此为规划包络，未实测1610图完整COCO/Bootstrap成本，不能宣称一定需要或一定能在该时长完成。

## 11. 独立数据准入审查

test-dev ZIP重新计算SHA并对全部1610图执行OpenCV `IMREAD_COLOR | IMREAD_IGNORE_ORIENTATION`（129）解码，uint8 BGR；1610份原标注坐标/类别解析检查通过，无异常。相对当前train6471/val548 ZIP，图像ID及原始字节SHA交集均为0。成员与标签SHA见{output_link('testdev_CPU_membership.csv')}。

但是历史选模、手动调参及研究使用记录仍有缺口，场景级隔离UNVERIFIED；字节不重合不能证明场景独立。新VM37–39训练计划只使用train/val、固定epoch10的证据保留，但不自动清除其他历史使用疑问。

VisDrone官方项目说明test-dev标注公开并支持科研发表；这支持该split的科研评价用途，不证明本项目的独立性，也不是任意重新发布数据的授权。[VisDrone官方项目](https://github.com/VisDrone/VisDrone-Dataset)

UAVDT本地转换目录仍为bus/car/truck/vehicle，不是VisDrone原生十类；原始DET/MOT与ignore、split语义未恢复。官方页面支持研究用途并区分DET/MOT与SOT，但不能把本地转换直接视为完整官方DET协议。[UAVDT官方说明](https://sites.google.com/view/grli-uavdt/)

{table(['候选','准入','阻塞'],[['test-dev','DATASET_BLOCKED','历史选模/研究使用及场景隔离未证实'],['UAVDT转换','DATASET_BLOCKED','原始DET/ignore/划分来源不足；native10不兼容'],['train / val','DATASET_BLOCKED','训练或开发暴露'],['test-challenge','DATASET_BLOCKED','无可用官方GT'],['HazyDet既有车辆协议','DATASET_BLOCKED','任务与类别不兼容，不静默代换']])}

来源网页已保存URL、获取时间与SHA。ZIP SHA匹配本地已接受证书，不冒充官方签名。完整关卡：{output_link('independent_data_admission.json')}。

## 12. 新评价结果（若真实完成）

**NOT_RUN**。独立数据关卡A失败，完整配对成本C/E未准入，热安全D未进入。GPU预算余额不是越过关卡的许可。VM39峰值89°C仍是热余量风险，不作为本轮已通过散热验收。

冻结候选12格：seed42/43/44 × CONTROL/GUIDED × Normal/Gaussian15，Infer1280，主终点Gaussian AP95差；本轮没有其中任何新预测或AP。未启动640/JPEG二级扩展，未训练新模型。

## 13. Empirical Gap重新判定

固定要求：Tiny AP50:95≥0.10，且VM36同边界缓存管线均值延迟≤100ms。原规则使用Tiny配对Bootstrap区间和延迟均值Bootstrap上界：延迟上界>100或Tiny上界<.10则FAIL；两项通过且Tiny下界≥.10则PASS，否则UNCERTAIN。另保留延迟上下界两侧敏感性规则，不能事后替换原规则。

每个条件只在实测4个D中判定：至少一个PASS→COVERED_TESTED_SUBSET；全部FAIL→GAP_TESTED_SUBSET；其余UNRESOLVED。本次复算原20条件17 Gap/3 covered；后加Normal后21条件17 Gap/4 covered。Normal添加不代表训练方法消除了Gap。没有评完46,368个P或635,904个D。

VM37–39只保存较窄`inference_ms`：在解码/退化/哈希之后开始，到predict及同步结束，排除CPU框转换和输出写盘，也没有VM36相同三次预热边界。不能用它替代VM36 `pipeline_ms`直接套100ms。56格保留Tiny质量单项状态，但完整标签统一为**GAP_CHANGE_NOT_ESTABLISHED / NOT_COMPARABLE_TIMING_BOUNDARY**。独立总体未运行，NOT_EVALUATED。

{picture('06_vm36_subset_gap_only')}

此图只重画原VM36分类，不生成不受支持的增强前后Gap缩小图。完整重判表：{output_link('gap_status_reassessment.csv')}。

## 14. 图表解读与科研局限

六类图均源于本Run的CSV，不含插值或预测AP。CONTROL/ALL_FIVE/GUIDED三策略图只用共同seed42；三seed图只比较CONTROL/GUIDED。ALL_FIVE没有seed43/44结果。

支持的结论是固定开发集上Gaussian定向策略在三个训练seed均改善Gaussian AP，存在正常图像代价。尚不支持独立图像泛化、全退化优势、最优概率、全FM空间Gap闭合或完整RQ4。连续参数排除精确测试点并未留出整个退化家族。训练版本与导师源分析不同，960/10epoch不是1280/100epoch截断，Mosaic与调度行为也不同。

## 15. 下周建议与导师问答

先向导师展示原UVL一致性、21条件来源、Gaussian增益及clean代价和计时不可比性。下一次独立验证需解决test-dev历史使用与场景隔离，或另提有合法native10标注、未参与开发的UAV数据方案供审核；不能把标签不兼容的UAVDT直接拼表。

数据、授权范围、完整配对预算及新散热准入全部满足后，才执行冻结12格，保持90°C即停、87°C连续30秒停与既有2秒监测；不自动增时或改precision/batch。预算估计不通过时另由作者决定资源，不能消耗余额启动不完整对照。

问：为什么AP升了却不说Gap缩小？答：Gap同时要求Tiny与同口径时延，后者缺失。问：为什么17/20与17/21不同？答：多了已覆盖的Normal参考，原17个困难条件并未因此消失。问：为何p=.25？答：先验折中，不是概率搜索得到的最优值。

## 16. 原始证据、路径、SHA及RunID

本Run：`{out}`。

VM36完整20条件来源：`{a.FULL}`；Normal来源：`{a.NORMAL}`。

seed42 CONTROL/ALL_FIVE：`{a.P37.name}`；GUIDED：`{a.P38.name}`。

seed43：`vm39_multi_seed_confirmation_seed43_20261115T100031_9d1155fa`；seed44：`vm39_multi_seed_confirmation_seed44_20261115T173603_db26fb50`；联合来源：`{a.P39.name}`。

每个输入路径、SHA、角色见{output_link('source_provenance.json')}。历史代码、两UVL、checkpoint与旧指标保持只读，保护核验见{output_link('preservation_check.json')}。本轮只有VM40新增代码和私有产物，没有修改旧SHA清单。

复现入口（CPU-only，新建Run，不覆盖既有报告）：

```powershell
Set-Location -LiteralPath 'E:\\ForStudy\\code\\uavgap'
.\\.venv_rq4\\Scripts\\python.exe -B -X utf8 .\\main\\experiments\\vm40_uvl_gap_independent_validation\\run.py --audit
```

只校验本次已完成产物，不重做计算：

```powershell
.\\.venv_rq4\\Scripts\\python.exe -B -X utf8 .\\main\\experiments\\vm40_uvl_gap_independent_validation\\run.py --verify {out.name}
```

`run.py`负责只读采集/校验与分支准入；`uvl_audit.py`使用FlamaPy环境解析；`reporting.py`读取表格生成图与中文解释。本入口没有GPU执行路径。
'''
    science.write_text(text,encoding='utf-8')
    spoken=f'''# VM40 导师会议讲解与问答

RunID：`{out.name}`。日期：2026-11-16。所有精度证据仍是DEVELOPMENT_ONLY，GUIDED属于POST_HOC_POLICY_SELECTION。

## 先用一分钟说明做了什么

“老师，我先把两棵特征树和您的原PDF重新核对。树、数值属性和约束是一致的，FM_P有46,368个合法组合，FM_D有635,904个。然后恢复了实际测过的20个退化条件和后来补的Normal。它们配4个检测设置，共84次评价，每次548图，没有为84个配置重新训练。”

“定向Gaussian策略在三个训练seed都有Gaussian精度收益，1280下平均+5.248个百分点，但正常图像平均下降0.574个百分点。独立验证还没过数据关卡，完整Gap变化也没有建立。”

## FM_P、FM_D和UVL是什么

FM_P是输入问题怎么变化：亮度、模糊、噪声、JPEG等。FM_D是检测器怎么选择：架构、权重、训练/推理尺寸、精度、后处理。UVL就是把这些选项与依赖写成程序能读的树。

必须选的节点是Mandatory，可以不选的是Optional；Alternative在一个分组内恰选一个，OR允许至少选一个。例如选了Blur以后，可以同时有Motion和Defocus。LowLight⇒ShotNoise表示低光被选时必须带ShotNoise。它是研究计划的约束假设，程序验证了逻辑，但没有替导师证明所有真实夜间场景都必须服从它。

数字是模型允许的配置数，不是实验次数。FlamaPy实际完成SAT与无死特征检查，使用既有精确计数方法；没有声称另做独立BDD证明。

{picture('01_feature_model_flow')}

## 为什么是这21种，而不是另一份21点覆盖设计

原20项是历史固定目的性选择：17种复合退化、两种JPEG端点及一个Gaussian15。后来加一项Normal，就变21。它们不是均匀随机抽样；17项逐一为什么选这套组合，原记录没有完整解释，所以不能补写成科学设计理由。

本21项没有Motion、Haze、Rain、RollingShutter。比如C01只有JPEG25；C20为正常光照加Gaussian σ15；C21完全Normal。C04则叠加强欠曝、Defocus6、Gaussian15、Shot20、JPEG50与Downscale2，数值均在科学报告清单。

另外的21点greedy覆盖核心、80配置逻辑cohort与20×4=80个已测对是不同东西，不能因为数量相近就互换。

## 21×4为什么不是84次训练

四个检测设置是历史YOLOv8s和YOLO26s，各用640或1280推理。训练时的checkpoint不变，只改输入推理尺寸。所以21×4=84次整批评价；84×548=46,032张次推理。历史权重配置训练1280/100轮，但旧best实际来自哪一轮没有在这次完全证明。

## CONTROL为什么叫普通训练

CONTROL不是无增强。它有HSV h=.015、s=.7、v=.4，平移.1、缩放.5、水平翻转.5，Blur/Median/Gray/CLAHE各1%。它没有额外五家族退化。

实际新训练是960、10轮、batch2、AdamW、lr0=.01、nbs64、FP32、workers0、warmup3轮。这些来自保存protocol和运行日志，不是套常见默认。字段等于库默认也不表示当时没显式传它，原命令来源不明时就写UNKNOWN。

Mosaic虽然args写1.0，close_mosaic=10又恰好等于总轮数10，所以第0轮已经关闭。warmup累计9708个batch，累积由1渐增到32；每臂实际1816次optimizer/EMA更新。最后使用epoch10 EMA，不挑开发AP最高的best。导师源分析8.4.33，实际8.4.210，不能说两个版本相同。

## ALL_FIVE到底怎样随机

每张图依次经过五个独立的抛硬币门，各50%：雾、曝光、运动模糊、噪声、JPEG。五门不是每次全部打开，也不是从UVL合法配置均匀抽样。

64,710次曝光里全没打开实际{props['no_extra_effect_count_actual']}次，理论比例3.125%；全五家族实际{props['ALL_FIVE_joint_active_family_count']['5']}次。Motion用9/11/13/17/19，避开测试7/15/21；JPEG用26–49与51–74，避开25/50/75。只是排除精确点，不代表从来没见过相邻强度或该类退化。

## 为什么Gaussian概率是0.25

先在已看过的开发结果中，按CONTROL两尺寸平均精度损失选择Gaussian家族；然后固定一个25%的先验门，保留75%的额外退化前输入。没有搜索最优概率，所以不能说0.25最优。σ5.5到30来自原范围，不是根据seed43/44重新调的。

三个seed实际激活概率为24.848%、25.041%、25.146%，接近固定25%。另四额外家族完全关闭，普通训练增强仍存在。因此它是development-guided定向策略，不是运行时看GT作决策的控制器。

## AP、Tiny和两个“95”怎样解释

AP50是在IoU=.50下累计的精度；AP50:95是.50到.95十个IoU阈值的平均。IoU是预测框与GT框的重叠度。“95% Bootstrap区间”的95是重复抽图估计区间的覆盖水平，与AP名字中的.95阈值不是一回事。

Tiny按原图框面积<16²分组，Small按16²至32²。不能把缩放后的框重新分组，否则640和1280比较时目标成员变了。AP按整个重采样集合累计，不能先算“单图AP”再取平均。

## 三seed实际说明了什么

640 Gaussian改善依次+1.846、+2.402、+1.440 pp，平均+1.896、样本SD .483 pp。1280依次+5.114、+5.526、+5.105 pp，平均+5.248、SD .241 pp。1280 Gaussian Tiny平均+2.506 pp。

seed42在1280 Gaussian下CONTROL AP95为16.3105%，GUIDED为21.4244%；但Normal从24.4468%降到23.8213%。三seed Normal平均下降.574 pp，所以应汇报清洁精度代价，而不是只放Gaussian最好看的图。

样本SD是在三个独立训练seed差值之间算的，只有n=3，不等于已经证明训练seed统计显著。ALL_FIVE只有seed42，不能给它画三seed误差条。

{picture('04_three_seed_gaussian_gains')}

## 配对Bootstrap区间表示什么

每次在548张图的位置上有放回抽样，同一索引用于CONTROL与GUIDED，重建预测与GT并重新累计AP。这样保持“在同一批图上比较”的配对关系。

{ci_table}

这些是已有单seed图像不确定性区间，不能把200次Bootstrap说成200次训练。没有可靠航次ID时，也不能说处理了航次间相关性。多个条件是逐点区间，没有整体多重比较保证。

## 为什么现在不能宣称Gap缩小

本任务要求同时满足Tiny AP≥.10以及同口径缓存管线≤100ms，并用保存的置信界作PASS/FAIL/UNCERTAIN。仅总体AP增加还不够。

例如seed42 GUIDED的1280 Gaussian Tiny点估计.102256，刚超过.10，并不自动表示其区间下界过线。更关键是新训练实验只计较窄的predict阶段，而原VM36包含缓存校验、读盘、解码与结果写出；不能拿较短新时延套原100ms门槛。

原VM36的20条件是17 Gap/3 covered，加Normal后是17 Gap/4 covered。这是加了一个正常参考，不是方法让原17个Gap消失。完整FM空间也没有穷尽测完。现在准确说法是AP_IMPROVEMENT_MEASURED，但GAP_CHANGE_NOT_ESTABLISHED。

## 为什么独立验证没有运行

test-dev的1610图和标注全部通过本轮CPU读取与几何检查，与当前训练/验证ZIP没有ID或字节重复。但历史选模、人工调参与研究使用及场景隔离仍不全，不能把“没搜到记录”写成“确定没用过”。UAVDT原始DET/ignore和native10兼容性也没通过，不能偷换数据。

还有预算：36小时原账本剩{budget['remaining_seconds']/3600:.3f}小时，固定期限不变。1610图12格只有规划外推，加入余量约9.43小时，没通过完整配对资源准入。89°C历史峰值也不是新的热安全许可。所以本轮新GPU为0秒。

## 会议上可以说与不能说

可以说：原UVL一致，21×4开发实验有来源，固定Gaussian策略在三个seed上都有相应退化收益，clean代价和局限已量化。

不能说：独立测试已成功、所有退化都改善、全空间Gap闭合、0.25最优、已经完成正式1280/100轮RQ4、ALL_FIVE三seed都失败。

## 从哪些文件看

科学报告：[{science.name}]({science.as_posix()})。

本次图表和CSV：`{out}`；打开`index.html`查看，不必记长RunID。84行原指标在`VM36_84_pair_metrics.csv`，56行策略结果在`MATCHED_56_configuration_metrics.csv`，三seed均值/SD在`VM39_three_seed_summary.csv`，独立数据阻塞在`independent_data_admission.json`，所有来源SHA在`source_provenance.json`。

运行`run.py --verify <本RunID>`只检查保存文件。`--audit`建立新的CPU审计，不会训练或推理；已有中文报告不覆盖，新Run会使用RunID后缀。代码关系是采集run.py→UVL子进程uvl_audit.py→图文reporting.py。报告与证据都在本地私有目录，不上传原始数据。
'''
    guide.write_text(spoken,encoding='utf-8')
    return [science,guide]


def page(out,names,report_paths):
    esc=html.escape
    figures=''.join(f'<figure><img src="plots/{name}.png" alt="{name}"><figcaption>{esc(name)}</figcaption></figure>' for name in names)
    artifacts=''.join(f'<li><a href="{p.name}">{esc(p.name)}</a></li>' for p in sorted(out.glob('*.csv')))
    reports=''.join(f'<li><a href="{p.as_uri()}">{esc(p.name)}</a></li>' for p in report_paths)
    (out/'index.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>VM40 results</title>
<style>body{{font:16px system-ui;max-width:1150px;margin:32px auto;padding:0 20px;color:#233746;background:#f8fafc}}img{{width:100%;height:auto}}figure{{background:white;padding:20px;margin:24px 0;border:1px solid #d9e2ea}}code{{word-break:break-all}}a{{color:#286b98}}h2{{margin-top:36px}}</style>
<h1>VM40 · CPU evidence audit</h1><p><code>{esc(out.name)}</code></p>
<p>CPU_AUDIT_COMPLETE_WITH_LIMITATIONS · Independent validation: DATASET_BLOCKED · GPU: 0 seconds</p>
<p>Gaussian AP improved on development data. Comparable empirical Gap change is not established.</p>
<h2>Chinese explanations (local)</h2><ul>{reports}</ul><h2>Verified tables</h2><ul>{artifacts}</ul><h2>Figures</h2>{figures}</html>''',encoding='utf-8')
