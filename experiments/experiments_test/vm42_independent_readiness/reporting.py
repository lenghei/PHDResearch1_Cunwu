"""Source-driven local charts and Chinese decision notes."""
import csv
import html
import json
from pathlib import Path


def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def table(headers,data):
    clean=lambda x:str(x).replace('|',' / ').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
        *['| '+' | '.join(clean(v) for v in r)+' |' for r in data]])


def figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import fontManager,FontProperties
    import numpy as np
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=font).get_name()
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    folder=out/'plots';folder.mkdir();names=[]
    def finish(fig,name):
        for ext in ('png','pdf'):fig.savefig(folder/(name+'.'+ext),dpi=170,bbox_inches='tight',facecolor='white')
        plt.close(fig);names.append(name)
    fig,ax=plt.subplots(figsize=(12,5));ax.axis('off')
    blocks=[(.23,.76,'已有VM36：21P ×4D =84对\n每对548开发图；未重跑'),
        (.77,.76,'已有VM37–39：三seed C/G\nGaussian1280 +5.2484 pp\nNormal1280 −0.5737 pp'),
        (.23,.30,'VM40/41：来源、覆盖、计时审计\n18/25叶已测；32候选CPU通过\n独立性仍未证明'),
        (.77,.30,'VM42新工作：历史证据增查\nA四格 / B十二格分别冻结\n没有新增GPU、图像解码或AP')]
    for x,y,t in blocks:ax.text(x,y,t,transform=ax.transAxes,ha='center',va='center',
        bbox=dict(boxstyle='round,pad=.8',fc='#e8f2ef' if y>.5 else '#edf2fa',ec='#567b88'))
    for p,q in [((.23,.61),(.23,.45)),((.77,.61),(.77,.45)),((.4,.3),(.59,.3))]:
        ax.annotate('',xy=q,xytext=p,xycoords='axes fraction',arrowprops=dict(arrowstyle='->',color='#567b88'))
    fig.suptitle('已接受测量与本轮CPU准备分开',fontsize=15);finish(fig,'01_past_evidence_and_new_work')
    fig,ax=plt.subplots(figsize=(11,6));ax.axis('off')
    for y,t,fc in [(.87,'当前1610图/GT及train-val ID/字节分离：复用VM40已验证证据','#e8f2ef'),
        (.62,'创建时来源、完整test-dev选模/调参记录、真实场景分组：UNKNOWN','#fff1df'),
        (.37,'独立准入：BLOCKED\n交付具体缺失证据与导师审阅决定；不由零搜索命中自动放行','#fff1df'),
        (.10,'未来：数据 + 完整选项预算 + 新授权 + 新热准入全部通过，才可执行','#edf2fa')]:
        ax.text(.5,y,t,transform=ax.transAxes,ha='center',va='center',bbox=dict(boxstyle='round,pad=.8',fc=fc,ec='#567b88'))
    for y,z in [(.79,.71),(.53,.46),(.27,.18)]:ax.annotate('',xy=(.5,z),xytext=(.5,y),xycoords='axes fraction',arrowprops=dict(arrowstyle='->'))
    ax.set_title('独立数据准入：文件分离不是完整科学独立性');finish(fig,'02_independent_admission')
    fig,axes=plt.subplots(1,2,figsize=(12,4.6))
    for ax,opt,seeds in [(axes[0],'A',[42]),(axes[1],'B',[42,43,44])]:
        ax.axis('off');rows_text=[]
        for seed in seeds:
            for arm in ('CONTROL','GUIDED'):rows_text.append([str(seed),arm,'Normal','Gaussian15','1610 /格'])
        tab=ax.table(cellText=rows_text,colLabels=['seed','策略','条件1','条件2','同一图像'],loc='center',cellLoc='center',colWidths=[.12,.23,.18,.22,.25])
        tab.auto_set_font_size(False);tab.set_fontsize(9);tab.scale(1,1.8)
        ax.set_title(f'未来Option {opt}：{len(seeds)*4}格 / Infer1280',pad=20)
        ax.text(.5,.06,'一seed独立Pilot，不能当三seed确认' if opt=='A' else '三seed独立确认；复用epoch10 EMA，不重训',ha='center',transform=ax.transAxes,fontsize=10)
    fig.suptitle('两项独立冻结提案；尚未执行，均不允许按AP选图或改策略',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.87))
    finish(fig,'03_four_vs_twelve_cells')
    resource=load(out/'Thermal_resource_admission.json');summary=resource['option_cost_summaries']
    fig,ax=plt.subplots(figsize=(9,5));x=np.arange(2)
    linear=[s['selected_linear_planning_seconds']/3600 for s in summary]
    extra=[s['complete_planning_envelope_seconds']/3600-linear[i] for i,s in enumerate(summary)]
    ax.bar(x,linear,label='外推基础（不是实测）',color='#3b779e');ax.bar(x,extra,bottom=linear,label='35%不确定性 +1800秒余量',color='#cb994c')
    for i,s in enumerate(summary):ax.text(i,s['complete_planning_envelope_seconds']/3600+.1,f'{s["complete_planning_envelope_seconds"]/3600:.2f} h',ha='center')
    ax.axhline(resource['remaining_seconds']/3600,color='#913c44',ls='--',label=f'当前余额{resource["remaining_seconds"]/3600:.3f} h')
    ax.set_xticks(x,['A ·4格','B ·12格']);ax.set_ylabel('规划小时');ax.set_title('完整选项成本：未知细分项仍在，包络内不等于准入')
    ax.legend(fontsize=9,loc='upper left');ax.set_ylim(0,max(s['complete_planning_envelope_seconds']/3600 for s in summary)*1.3)
    finish(fig,'04_estimated_complete_costs')
    motion=load(out/'Motion_Haze_future_protocol.json');fig,ax=plt.subplots(figsize=(11,5));ax.axis('off')
    for x,y,title in [(.23,.76,'Normal\n新协议AP未测'),(.77,.76,'Motion_k21\n新协议AP未测'),(.23,.32,'Haze_dense\n新协议AP未测'),(.77,.32,'Motion + Haze\n新协议AP未测')]:
        ax.text(x,y,title,ha='center',va='center',transform=ax.transAxes,bbox=dict(boxstyle='round,pad=.8',fc='#edf2fa',ec='#567b88'))
    ax.text(.5,.04,'每checkpoint：DID = AP(M+H) − AP(M) − AP(H) + AP(Normal)',transform=ax.transAxes,ha='center',fontsize=11)
    fig.suptitle('未来8格：4P ×2历史checkpoint ×Infer1280；绑定待审核',fontsize=13);finish(fig,'05_motion_haze_future_contrast')
    fig,ax=plt.subplots(figsize=(12,4.6));ax.axis('off')
    data=[['RQ1','双特征模型与约束','已对齐；不等于所有配置已可运行'],['RQ2','因素×检测器交互/规则','需留出配置准确率和新退化支持'],
        ['RQ3','静态/实测Gap区域及大小','目前仅21P×4D子集；不能推总体'],['RQ4','针对Gap重配置是否缩小Gap','Gaussian开发收益；独立及同计时Gap待验证']]
    t=ax.table(cellText=data,colLabels=['原编号','导师目标','仍需证据'],loc='center',cellLoc='left',colWidths=[.09,.3,.61]);t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1,2.2)
    ax.set_title('RQ编号沿用原PDF第4页；VM42不新增科学成效结论');finish(fig,'06_remaining_RQ_evidence')
    return names


def reports(out,root,v40,v41):
    costs=load(out/'Thermal_resource_admission.json');history=load(out/'Bounded_history_scope.json')
    audit=rows(out/'Testdev_provenance_audit.csv');usage=rows(out/'Historical_split_usage_matrix.csv')
    rule=load(out/'Selection_rule_evidence.json');motion=load(out/'Motion_Haze_future_protocol.json')
    a=load(out/'Pilot_4cell_frozen_protocol.json');b=load(out/'ThreeSeed_12cell_frozen_protocol.json')
    folder=root/'datasets/reportcn';report=folder/'VM42_独立验证准入与最小实验方案_CN.md';guide=folder/'VM42_导师会议讲解与问答_CN.md'
    if report.exists() or guide.exists():
        report=folder/('VM42_独立验证准入与最小实验方案_CN_'+out.name+'.md');guide=folder/('VM42_导师会议讲解与问答_CN_'+out.name+'.md')
    if report.exists() or guide.exists():raise FileExistsError('Reports for this RunID exist.')
    def link(name):return f'[{name}]({(out/name).as_posix()})'
    def pic(name):return f'![{name}]({(out/"plots"/(name+".png")).as_posix()})'
    cost_table=table(['方案','格数/张次','外推基础h','含余量包络h','包络是否小于余额','真实资源准入'],
        [[s['option'],f'{s["cells"]}/{s["cells"]*1610}',f'{s["selected_linear_planning_seconds"]/3600:.3f}',f'{s["complete_planning_envelope_seconds"]/3600:.3f}',
          str(s['envelope_fits_balance']),'未通过'] for s in costs['option_cost_summaries']])
    checkpoints=load(out/'Frozen_checkpoints.json')
    text=f'''# VM42 独立验证准入与最小实验方案

日期：2026-11-16。RunID：`{out.name}`。

**CPU_READINESS_COMPLETE_WITH_LIMITATIONS**。完成历史来源有界增查、四格/十二格分别冻结、完整配对成本分析和导师决策备忘。独立准入仍为BLOCKED；本轮无训练、GPU预测、图像全量解码或新AP。

## 1. 已接受证据与本轮新增工作

复用导师原PDF、冻结FM_P/FM_D、VM36二十一条件、VM37/38/39训练与评价、VM40来源审计及VM41候选/计时协议。FM_P25叶/5约束/46,368；FM_D34叶/4约束/635,904；VM36为21P×4D=84评价对，不是84次训练。

三seed Gaussian1280 AP95平均+5.2484 pp，Normal1280平均−0.5737 pp是保存的DEVELOPMENT_ONLY / POST_HOC_POLICY_SELECTION结果，不是本轮测量。VM41的18/25叶、32个SAT/CPU候选、31个坐标安全也直接复用，没有重新跑其测试。

{pic('01_past_evidence_and_new_work')}

## 2. test-dev证据边界

{table(['问题','状态','依据'],[[r['question'],r['status'],r['finding']] for r in audit])}

CONFIRMED指该命题有保存证据；DISPROVED指表中明确的相反命题已被反证，例如“当前字节有交叉”被VM40反证。UNKNOWN不能当作无使用，NOT_APPLICABLE表示本轮未执行相应活动。

本轮读取Git最近30条记录、最多40条test-dev变动记录以及限定准入源码路径的最多8条patch；命令、commit身份和原输出保存于`logs/`与{link('Git_provenance_queries.json')}。Git证明代码/规则曾被记录，不证明每次脚本是否实际运行，更不能记录未提交的私人操作。

另查迁移文件索引，test-dev命名路径{history['indexed_testdev_path_count']}项，按预设限制打开其中{history['small_indexed_records_opened']}个小记录；没有打开原图或大预测文件。原文件/迁移SHA核验与上下文命中见{link('Indexed_record_verification.csv')}、{link('Historical_content_hits.csv')}。目录名、命中行和终态日志的证据强度不同，不能将脚本路径直接当作完成的模型选用记录。

{table(['checkpoint/arm','已记录训练/选择','test-dev历史用途','场景'],[[r['checkpoint'],r['selection'],r['testdev_eval_or_tuning'],r['scene_independence']] for r in usage])}

旧ck01/ck05/ck06的复制args与native10数据YAML可确认声明的train/val路径，但不能重建创建时图像字节及所有手工选模决策。新六臂的6471train/548val、固定epoch10及同seed初始化有直接protocol与验收记录；这比旧best来源完整，但不自动证明整个项目历史上没有test-dev响应用于方法选择。

可靠拍摄/航次身份仍未从已保存材料建立。VM41前缀交集为零只是一项代理检查，不能升级成场景隔离证明。UAVDT仍缺原始DET/ignore/划分来源且标签不兼容native10，未替换数据。

{pic('02_independent_admission')}

## 3. 导师需要作出的具体决定

建议目前**不批准严格独立确认性评价**。需要补的是：

1. checkpoint创建时训练/选择材料、完成过的test-dev评价清单，以及这些结果之后是否修改模型/方法的记录。
2. 可靠序列/航次/来源场景信息及交叉审查；图像SHA不重复不能代替它。
3. 若这些材料无法恢复，明确保留“历史使用未证实”的外部评价范围，不把导师同意当成对事实独立性的证明；严格独立验证则另行审核真正前瞻、类别兼容的数据方案。

这是有明确待补证据的终态备忘，不再重复同一关键词审计。只有新材料或具体导师决定出现时重开。科学审阅不等于GPU资源授权。

## 4. Option A：最小四格独立Pilot

固定seed42，CONTROL/GUIDED，Normal/Gaussian_s15，Infer1280，同一完整1610图。四格和checkpoint已逐项冻结：{link('Pilot_4cell_frozen_protocol.json')}。

{table(['格ID','已有权重SHA256'],[[c['cell_id'],c['checkpoint']['SHA256']] for c in a['cells']])}

主终点为Gaussian AP50:95的GUIDED−CONTROL；次终点为Gaussian Tiny、Normal AP95代价、统一缓存管线与固定Gap标签。它只有一个训练seed，不能宣称新增三seed确认；seed42由本次任务预先指定，不按最优AP选择。

## 5. Option B：十二格三seed确认

seed42/43/44，各CONTROL/GUIDED×Normal/Gaussian_s15，Infer1280，同一1610图，共12格、19,320张次。主终点仍为Gaussian AP95差，汇总三个匹配差值均值与ddof=1样本SD；逐图Bootstrap不是新增训练seed。

{link('ThreeSeed_12cell_frozen_protocol.json')}保存全部12格，{link('Frozen_checkpoints.json')}保存6个既有epoch10 EMA的实际文件SHA、初始化与已接受健康来源。本轮只进行来源/字节完整性检查，未重复模型张量健康测试或载入GPU。

{pic('03_four_vs_twelve_cells')}

两方案均冻结全部1610成员及图像/标注SHA引用；不减图、不改seed、不因预算把A冒称B。未构建新的GPU缓存或擅自重新定义GT；真正准入后才能按原native10、ignore-union和原图面积构建同一评价输入。未准入时所有新指标都是NOT_EVALUATED。

## 6. 计时与评价协议准备程度

原样引用`VM41_MATCHED_CACHED_GAP_V1`及文件SHA，不更改冻结源。FP32、batch1、1280、native10、相同PNG、相同GT/ignore、class-aware multi-label NMS、conf=.001、IoU=.7、max_det3000/max_nms30000，三次预热。Tiny按原图面积，不按缩放后重新分组。

缓存字节验证、读取/解码、像素验证、预处理、forward/NMS与同步、取框、输出/manifest写入必须分别留记录，完整worker另计。VM41 CPU接口的主边界已验证，但其部分阶段仍合并计时，真实预测器的细粒度接入未作GPU验收。{link('Timing_protocol_reuse_audit.csv')}明确这个差别；不能称为后端所有分阶段都已实测。

固定Tiny AP95≥.10、可比缓存管线≤100ms，沿用原区间规则。200次、seed20261116的共用逐图索引重建GT/预测并累计AP，延迟对同索引均值重采样。区间是逐点、不是同时覆盖；无可靠航次ID时不宣称已做真实航次聚类。旧VM37–39 `inference_ms`不能转换成VM36 `pipeline_ms`或以旧均值相加伪造逐图时延。

## 7. 完整配对成本与热准入

{cost_table}

预算来自当前原始36小时账本，已用{costs['charged_seconds']:.3f}秒，剩{costs['remaining_seconds']:.3f}秒（{costs['remaining_seconds']/3600:.6f}小时），期限{costs['absolute_deadline_UTC']}。无重置、延长或新计费。

两种估计来源：相应4/12个历史548图配置的保存阶段耗时，以及旧16格完整监督器10,808.362秒。各自按1610/548线性外推，取较大者，再加35%不确定性与1800秒停止余量。模型加载、三次预热、GT/PNG准备、独立验证成本有些未单独记录；{link('Matched_evaluation_costs.csv')}使用null/NOT_SEPARATELY_MEASURED，未造细分秒数，也没重复计入已包含的写盘。

规划包络内不等于完整成本已验证，更不等于科学准入。COCO/Bootstrap成本随GT和预测密度变化，统一缓存写盘也不同于旧时延记录。必须对完整选项及原截止时间重新资源准入，不得只启动一臂。

{pic('04_estimated_complete_costs')}

保留90°C即停、87°C持续30秒停、既有2秒监测。历史89°C峰值要求新的现场/设备热准入，本轮未执行该GPU验证。本轮没有GPU授权，两方案当前均不能启动。

## 8. Variability选择规则目前是什么

实际代码先计算三个已测条件在640/1280的平均CONTROL精度损失，再检查Gaussian_s15是否最高；随后`recipe()`返回固定Gaussian p=.25。保存均值为：

{table(['已测条件','两分辨率平均AP95损失pp'],[[k,f'{v*100:.4f}'] for k,v in rule['mean_CONTROL_AP95_losses'].items()])}

这是三种条件、两个退化家族（JPEG与Gaussian）；Motion/Haze/Exposure/Rain没有该步骤的可比评分，不能补0。它有可复现开发反馈和执行保护，但不是输入任意环境后输出任意增强家族的通用控制器；p=.25是固定先验，不是算法搜索得到的最优值。

未来可解释提案：输入FM_P特征、同协议实测响应、固定Tiny/时延约束；先排除非法/标注不安全/绑定未审核条件，要求多家族有匹配数据；按事先冻结的权重计算家族损失并记录不确定性，输出已批准策略或UNKNOWN/POLICY_NOT_DEFINED。缺完整Tiny/时延时应拒绝Gap-driven结论。没有训练新策略或搜索概率。

完整审计：{link('Variability_selection_rule_audit.md')}。当前贡献是把变异模型、证据支持与固定策略选择关联起来的可追溯流程；通用自适应、独立泛化及全Gap缩小未证明。

## 9. Motion/Haze未来分支

仅Normal、Motion21、Haze_dense、二者叠加四条件×ck01/ck05 Infer1280=8格，同一548开发图。每模型DID=AP(M+H)−AP(M)−AP(H)+AP(Normal)。这是提案，没有新AP。

{table(['P ID','叶集合','参数（CPU提案）'],[[r['FM_P_id'],' / '.join(r['leaves']),json.dumps(r['parameters'],ensure_ascii=False)] for r in motion['P_conditions']])}

方向Uniform[0,180)、airlight220仍是历史CPU绑定提案，不是导师原UVL指定值，也没有在VM42批准。需要科学审核、统一RNG/像素、标注/ignore、缓存/评价SHA后才能进一步准入。RollingShutter排除。

{pic('05_motion_haze_future_contrast')}

## 10. 决策与操作

先解决独立数据，再选完整可承担的A或B，随后补齐统一Tiny/时延并作Gap判定；低覆盖家族与更一般的选择规则在后。没有按最好AP推荐方案。

{pic('06_remaining_RQ_evidence')}

本Run：`{out}`。独立问题与行动见{link('Scientific_blockers_and_actions.csv')}；历史使用矩阵见{link('Historical_split_usage_matrix.csv')}；数据决策见{link('Independent_admission_decision.json')}；资源见{link('Thermal_resource_admission.json')}。`SHA256SUMS.txt`记录输出、报告与源码校验。

```powershell
Set-Location -LiteralPath 'E:\\ForStudy\\code\\uavgap'
.\\.venv_rq4\\Scripts\\python.exe -B -X utf8 .\\main\\experiments\\vm42_independent_readiness\\run.py --verify {out.name}
.\\.venv_rq4\\Scripts\\python.exe -B -X utf8 .\\main\\experiments\\vm42_independent_readiness\\run.py --validate-design {out.name} --option A
```

`--verify`只查保存文件SHA；`--validate-design`检查完整格点、1610成员、权重引用与协议，输出VALID设计和BLOCKED准入并以退出码2表示不能执行；B用`--option B`。这是可执行的CPU清单验证，不是已验收的test-dev GPU后端。真正GT/PNG/计时接入仍需后续明确授权与验收，本入口没有GPU执行命令。

`--audit`会建立新CPU Run，不重复已接受的解码、生成器测试、AP或训练。本次所有旧文件保持只读；无Git pull/rebase/commit/push。缺证分支到此记录，不无休止重复相同审计。
'''
    report.write_text(text,encoding='utf-8')
    guide_text=f'''# VM42 导师会议讲解与问答

RunID：`{out.name}`。这次没有跑检测器，主要把“能不能独立测、最少怎么测、还差什么”写清楚。

## 一分钟说明

“老师，已有三seed在Gaussian下改善，1280平均+5.2484个百分点，但Normal平均下降.5737个百分点。它们还是开发结果。我没有重复实验，先把独立来源查证和最小评价方案准备好。小方案是seed42四格，大方案是三seed十二格，不能把小方案当成三seed确认。”

{pic('03_four_vs_twelve_cells')}

## 四格到底是什么

同一1610图，seed42 CONTROL正常、CONTROL Gaussian15、GUIDED正常、GUIDED Gaussian15，全部1280推理。训练都已经做完，现在只是未来复用两个epoch10 EMA。

主问题是Gaussian条件下GUIDED比CONTROL的AP95高多少，同时保留Normal精度代价、Tiny和完整管线时延。不能只展示噪声增益，不看正常图损失。它只有一个训练seed，所以即使独立数据结果好，也没有新增三个seed的重复性证据。

## 十二格为什么多

把同样四格扩展到seed43和44，总共十二格。两策略在每个seed内同初始化和训练设置，比较每seed差值，再算三个差值的均值和样本SD。Bootstrap抽图200次仍然不是训练200次，单seed图像区间不能代替训练seed变异。

## 为什么有1610张合法图还不能说独立

VM40已经确认原图/GT能读、当前train/val没有ID或字节重复。这解决了文件身份和标注问题，却没有自动解决以前有没有看过test-dev AP后改方法、有没有人工挑模型、以及不同文件是否来自相同场景。

这次补查了保存Git、迁移索引和配置记录。Git里有脚本不等于脚本确实执行过，也不记录所有手工操作。没有搜到test-dev不等于肯定没用过。场景前缀不是官方航次身份。因此结论仍是历史独立性未证实，而不是程序坏了。

## 老师需要决定什么

请老师审阅具体缺失材料：创建时训练/选模数据、完成过的test-dev评价及之后的调整记录、可靠场景/航次隔离。若能恢复，就按这些证据判断，不凭口头“应该没用过”。

若不能恢复，可以明确保留历史使用不明的外部评价范围，但不能叫严格独立确认。若论文必须严格独立，就另审真正前瞻、标签兼容的数据计划。这次没有编造或偷偷换一个数据集。

## 时间上哪个更现实

{cost_table}

当前旧账本余额约{costs['remaining_seconds']/3600:.3f}小时。图里的时间是保存测量的外推，未知的GT/缓存/加载/预热等不是新实测。A即使包络能放进余额，也不能自动开跑：数据、热准入、完整成本和新GPU授权都还需要通过。B需要更完整的资源，但不能为了省时间把A说成B。

{pic('04_estimated_complete_costs')}

## 计时协议准备好了没有

原VM41的主边界已经冻结，并有CPU模拟接口测试：核对缓存、读图、解码、像素SHA、predict、取框、写结果。真实GPU细分阶段接入还没验收，不说已经全通过。

未来每格三次预热、FP32、batch1、同PNG/GT/ignore/NMS。旧`inference_ms`只测中间predict，不能冒充包含读写的缓存管线。Tiny至少.10且同口径管线最多100ms，两项和区间都满足才谈PASS。仅Gaussian AP上升，不能推出Gap缩小。

## Variability方法现在算算法吗

目前可以说有一个可复现的开发反馈选择检查：把JPEG25、JPEG75、Gaussian15在640/1280的CONTROL精度损失算出来，确认Gaussian最大，然后用事先固定p=.25的Gaussian策略。

源码会检查Gaussian是否最高，再返回固定策略；还不会针对任意输入自动选择其他家族。三种条件其实只有JPEG和Gaussian两个家族，Motion/Haze没有该步骤的评分，也不能填0。因此“固定增强”“开发反馈选择”“通用控制器”“独立泛化”是四件事，前两件有证据，后两件没有。

未来规则要有可解释输入、合法/安全约束、匹配响应和拒绝机制，选不到已批准策略就返回未知。还要在多个退化家族、不同环境上验证，不能只用Gaussian一次结果证明自动跨环境适应。这里没有训练新策略，也没有搜索概率。

## Motion/Haze为什么还没开始

最小四点是Normal、Motion21、Haze_dense、两者组合；乘两个历史模型是8格。完整四点才能把组合影响与单因素影响分开。

但方向分布和airlight220只是VM41历史CPU提案，不是老师原UVL已经指定或批准。先审核参数、物理假设、像素与GT/ignore来源，再考虑GPU。RollingShutter连坐标映射还没验证，不能加入。

## 哪些话可以向老师说

可以说：已有Gaussian开发收益跨三个seed方向一致，有正常图代价；独立问题已整理成具体证据清单；四格/十二格清单、模型SHA、计时要求和成本已分别冻结。

不能说：独立验证已通过、四格是三seed确认、通用控制器已完成、所有退化都改善、Gap已经闭合、原100轮正式RQ4已经完成。

科学报告：[{report.name}]({report.as_posix()})。

证据网页：[{out.name}]({(out/'index.html').as_posix()})。看方案打开`Pilot_4cell_frozen_protocol.json`或`ThreeSeed_12cell_frozen_protocol.json`；看未知事项打开`Scientific_blockers_and_actions.csv`；看模型数据来源打开`Historical_split_usage_matrix.csv`；看预算打开`Matched_evaluation_costs.csv`和`Thermal_resource_admission.json`。

运行`run.py --verify <RunID>`只检查文件。`--validate-design <RunID> --option A/B`只检查清单，当前会说明准入仍BLOCKED，不会跑GPU。两个中文文件都在`datasets/reportcn/`，没有放进Git公开目录。
'''
    guide.write_text(guide_text,encoding='utf-8');return [report,guide]


def page(out,names,reports):
    links=''.join(f'<li><a href="{p.as_uri()}">{html.escape(p.name)}</a></li>' for p in reports)
    tables=''.join(f'<li><a href="{p.name}">{p.name}</a></li>' for p in sorted(out.glob('*.csv')))
    figures=''.join(f'<figure><img src="plots/{n}.png" alt="{n}"><figcaption>{n}</figcaption></figure>' for n in names)
    (out/'index.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>VM42 readiness</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;color:#263b4d;background:#f6f8fb}}figure{{background:white;padding:20px;margin:24px 0;border:1px solid #dae2ea}}img{{width:100%}}a{{color:#226e99}}code{{word-break:break-all}}</style>
<h1>VM42 · Independent readiness</h1><p><code>{html.escape(out.name)}</code></p>
<p>CPU preparation complete with limitations. Independent admission BLOCKED. No GPU, training or new AP.</p>
<h2>Chinese guides · local only</h2><ul>{links}</ul><h2>Evidence tables</h2><ul>{tables}</ul>{figures}</html>''',encoding='utf-8')
