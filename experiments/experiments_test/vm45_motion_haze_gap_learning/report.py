"""Figures and factual summaries from measured CPU evidence."""
from vm45_common import *
def deliver(out,rules,learner,uvl,gate):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 rows=rules['development_rows'];groups=sorted({r['condition_id'] for r in rows});ds=sorted(rules['candidates']);plots=[];captions=[]
 def finish(fig,name,caption):
  fig.text(.01,.01,'VM45 CPU study | VM36 development only | Motion/Haze GPU NOT_MEASURED',fontsize=7);fig.tight_layout(rect=(0,.05,1,1))
  for ext in ('png','pdf'):
   p=out/'plots'/(name+'.'+ext);fig.savefig(p,dpi=170);plots.append(p)
  plt.close(fig);captions.append({'figure':name,'caption_CN':caption})
 def message(name,title,body,caption):
  fig,ax=plt.subplots(figsize=(10,4));ax.axis('off');ax.set_title(title);ax.text(.03,.78,body,va='top',transform=ax.transAxes,fontsize=12);finish(fig,name,caption)
 fig,ax=plt.subplots();ax.bar(['FM_P leaves','FM_D leaves'],[25,34]);ax.set_title('01 | Original supervisor feature models');ax.set_ylabel('Leaf count');finish(fig,'01_Feature_models','原导师UVL：25/34叶、5/4约束、46368/635904合法配置；不是全部可执行/已测证明。')
 import csv
 coverage=list(csv.DictReader((V41/'FM_P_measured_coverage.csv').open(encoding='utf-8-sig')));ref(V41/'FM_P_measured_coverage.csv')
 fig,ax=plt.subplots(figsize=(10,8));ax.barh([r['leaf'] for r in coverage],[int(r['measured_P_configurations']) for r in coverage]);ax.set_title('02 | Actual development feature coverage');ax.set_xlabel('Measured P conditions, not failure frequency');finish(fig,'02_Coverage','未测Motion/Haze不是失败或非法；21个有目的条件不代表全空间随机样本。')
 message('03_Factorial_design','03 | PROPOSED / NOT_MEASURED','Normal / Motion21 / Haze_dense / Motion21+Haze_dense\nck01 and ck05, Infer1280 FP32; 548 images per cell\n8 cells / 4,384 image inferences\nBindings and VM45 GPU authorization are NOT admitted.','四角点设计已明确，但新预测0；角度与airlight仍为未准入候选。')
 message('04_AP_changes','04 | Motion/Haze AP: NOT_EVALUATED','No admitted VM45 GPU run.\nNo measured Motion or Haze single-factor AP changes.\nHistorical Normal AP cannot fill the missing corners.','不生成不存在的单因素AP柱图或区间。')
 message('05_DID','05 | Detector-dependent DID: NOT_EVALUATED','DID = AP_both - AP_motion - AP_haze + AP_normal\nAll four matched corners and shared image draws are required.\nNo new DID, cross-detector DID or interaction interval exists.','DID公式可验收；没有完整实测角点就没有交互结论。')
 tiny=np.array([[next(r['Tiny_AP'] for r in rows if r['condition_id']==p and r['D']==d) for d in ds] for p in groups])
 fig,ax=plt.subplots(figsize=(8,8));im=ax.imshow(tiny*100,aspect='auto');ax.set(xticks=range(4),xticklabels=ds,yticks=range(21),yticklabels=[f'P{i+1:02d}' for i in range(21)],title='06 | Saved VM36 Tiny AP (%)');ax.tick_params(axis='x',rotation=25);fig.colorbar(im,ax=ax);finish(fig,'06_Tiny_AP','仅复用VM36原始开发指标，没有Motion/Haze新AP。P编号与结果表condition_id排序一致。')
 fig,ax=plt.subplots()
 for d in ds:
  rs=[r for r in rows if r['D']==d];ax.scatter([r['cached_pipeline_mean_ms'] for r in rs],[r['Tiny_AP']*100 for r in rs],label=d)
 ax.axhline(10,color='gray',ls='--');ax.axvline(100,color='gray',ls='--');ax.set(xlabel='VM36 cached pipeline ms',ylabel='Tiny AP (%)',title='07 | Development measured tradeoffs');ax.legend(fontsize=7);finish(fig,'07_Tradeoff','不同P点不构造全局Pareto优势；旧时延不转换为VM43管线时延。')
 states=np.array([[{'FAIL':0,'UNCERTAIN':1,'PASS':2}[next(r['original_rule_state'] for r in rows if r['condition_id']==p and r['D']==d)] for d in ds] for p in groups])
 from matplotlib.colors import ListedColormap
 fig,ax=plt.subplots(figsize=(8,8));ax.imshow(states,cmap=ListedColormap(['#a85863','#d3bd76','#288b7e']),vmin=0,vmax=2,aspect='auto');ax.set(xticks=range(4),xticklabels=ds,yticks=range(21),yticklabels=[f'P{i+1:02d}' for i in range(21)],title='08 | FAIL / UNCERTAIN / PASS (red / yellow / green)');ax.tick_params(axis='x',rotation=25);finish(fig,'08_States','76 FAIL、6 PASS、2 UNCERTAIN按原区间规则保留；UNCERTAIN不当作FAIL训练。')
 message('09_Candidate_tree','09 | In-sample candidate tree: NOT_ADMITTED',(out/'in_sample_tree.txt').read_text()[:1500],'浅树由完整已标注开发行拟合用于展示；树路径不是已验证约束，未写入原UVL。')
 models=learner['models'];fig,ax=plt.subplots(figsize=(9,4));ax.barh([m['method'] for m in models],[m['balanced_accuracy'] for m in models]);ax.set(xlim=(0,1),xlabel='Whole-P held-out balanced accuracy',title='10 | Fixed classifiers; not independent-image generalization');finish(fig,'10_Heldout_performance','21折整P留出；模型参数事先固定。完整混淆矩阵、失败precision/recall、PASS召回和组Bootstrap区间见JSON；只4个含PASS组，不认可普适规则。')
 fig,axes=plt.subplots(1,2,figsize=(11,4));s=uvl['static_sensitivity'];axes[0].bar(range(len(s)),[r['covered'] for r in s]);axes[0].set(xticks=range(3),xticklabels=['Strict','Gain/scale','Weak Albu'],ylabel='Covered symbolic P /46,368',title='11 | Static mapping assumptions');axes[1].bar(['Covered','Gap'],[4,17]);axes[1].set(ylabel='Purposeful measured P /21',title='Empirical: four tested D only');finish(fig,'11_Static_empirical','静态覆盖数依赖映射假设；4/17经验结果只针对21×4子集，两个分母不同，不能推成全空间失败率。')
 message('12_Controller','12 | Variability Controller V2','Measured exact lookup:4 selected /17 abstained\nMotion/Haze3 unmeasured conditions:3 UNKNOWN\nAdmitted learned-rule predictions:0\nNo coverage improvement over fixed YOLO26s/1280.','V2保留实测/预测/未知的分层及安全弃权，不让候选树路径冒充实测AP。')
 message('13_Derived_constraints','13 | Original versus candidate UVL','Original P count:46,368\nDerived P count:46,368; identical CNF; dead features:0\nNew empirical exclusions admitted:0\nObserved failures are not physical impossibilities.','候选派生模型保留原模型，未对未验证经验规则做空间裁剪；四种规则语义分开。')
 matrix=[{'RQ':'RQ1','status':'PARTIAL','completed':'Original models and scoped SAT/generator checks','missing':'Full executable-space and calibrated physical constraints'},
 {'RQ':'RQ2','status':'PARTIAL','completed':'84 prior measurements; new grouped tree/RF diagnostic','missing':'Motion/Haze GPU, reliable heldout rules, independent confirmation'},
 {'RQ':'RQ3','status':'PARTIAL','completed':'Static sensitivity and21-P empirical subset','missing':'46347 unknown P; no valid probability-space estimator'},
 {'RQ':'RQ4','status':'NOT_EVALUATED_FORMAL','completed':'960/10epoch Gaussian exploratory benefit, Tiny criterion still FAIL','missing':'12 matched100epoch arms plus5 v8 augmentation ablations,504 evaluations and repeated training seeds'}]
 csv_write(out/'rq_evidence_matrix.csv',matrix)
 message('14_RQ_status','14 | Original RQ1-RQ4 evidence status','RQ1 PARTIAL: models valid, executable/physical coverage limited\nRQ2 PARTIAL: grouped diagnostics; no admitted general failure rule\nRQ3 PARTIAL: symbolic sensitivity and measured subset only\nRQ4 formal100epoch study NOT_EVALUATED','不得把960/10轮Gaussian探索、图像Bootstrap或小样本查表称为导师原正式研究完成。')
 save(out/'figure_captions_CN.json',captions)
 table=lambda rs:'\n'.join(f"- {m['method']}: accuracy={m['accuracy']:.4f}, balanced_accuracy={m['balanced_accuracy']:.4f}, failure precision={m['failure_precision']:.4f}, recall={m['failure_recall']:.4f}, PASS recall={m['PASS_recall']:.4f}; confusion(PASS,FAIL)={m['confusion_PASS_FAIL']}" for m in rs)
 text=f'''# VM45 复合退化与Gap学习

RunID：`{out.name}`。本轮新增真实实验为CPU整P留出分类器诊断、派生UVL/SAT/计数/死特征检查、静态覆盖敏感性和控制器V2。未执行新GPU、未训练YOLO、未生成未准入参数的548图缓存。

## 原导师要求与冲突

实际重读2026-10-22原PDF17页。原计划约200P×20D、4000评价、多训练；当前21P×4D并不等价。PDF第15页限制AI撰写分析，当前作者明确要求自动技术分析：本产物是机器辅助技术证据，需人工核验与重写后用于论文，不冒充导师认可的论文稿。PDF每日push与当前禁止push冲突，遵守当前指令；test-dev公开标签不证明历史独立性。

## 新GPU和交互结果

Motion单因素AP、Haze单因素AP、复合DID、跨模型DID及其区间全部NOT_EVALUATED。没有适用于VM45的专项GPU许可，且角度Uniform[0,180)与airlight220只有历史工程提案、没有科学准入。kernel21、beta1.5来自原UVL；Haze公式是常深度光学厚度近似，不是自然天气标定。VM44合成像素验证已复用；它不能代替科学批准。Stage B全缓存因此BLOCKED，成员548清单和共享缓存/GT契约已经准备。没有必要为被阻塞分支重复生成约6GB候选图片。

8格设计、4384张次、FP32/batch1/native10、原图GT/ignore、共享像素和200次同图索引Bootstrap保持提案。Normal历史口径没有证明与未来全流程完全相同，未混入新四角点。最新账本只作报告；余额不等于VM45许可。没有GPU设备/热负载预检，也没有预算计费。

## 新规则学习实测

仅VM36开发84行：76 FAIL、6 PASS、2 UNCERTAIN。UNCERTAIN从训练及标签评分排除，但分类器对全部84行仍给预测，未利用真标签伪造部署弃权。固定浅树depth2/leaf4/类别平衡；RF100树/depth3/leaf4/单CPU，参数不根据留出成绩调整。每折保留完整P的4行，21折，未拆散同一P。沿用已有FlamaPy环境的sklearn，不安装或修改rq4环境。

{table(models)}

组Bootstrap区间条件于已得到的OOF预测，不重新训练、不等于独立图像或训练seed置信度，见`rule_learning_validation.json`。数据只有4个含PASS的P组、共享548图、历史选模/政策暴露，最终仍INSUFFICIENT_DATA_FOR_RELIABLE_RULE。完整树路径是数据内EMPIRICAL_FAILURE_RULE候选，不是物理约束；0条获准推广，未写入导师UVL。

## Static与Empirical Gap

静态覆盖假设计数：{[(s['assumption'],s['covered']) for s in uvl['static_sensitivity']]}，均以46368合法P为总体。这些是TRAINING_COVERAGE_RULE，不是实测失败率。严格identity、gain/scale近似、弱Albu近似分别区分，不能把约等于增强支持当实际训练来源证明。

经验结果沿用21个P和4个D：4个P至少一个PASS，17个P四个D均FAIL。没有合法概率采样或可靠全空间模型，不能把17/21当46368空间失败率；未知P数46347。原FM_D635904配置不是全部已测或可部署证明。

派生`fm_p_empirical_candidate.uvl`和`derived_uvl_constraints.uvl`保留原约束，CNF相同、P配置数46368不变、无死特征。分别记录PHYSICAL_VALIDITY_CONSTRAINT、TRAINING_COVERAGE_RULE、EMPIRICAL_FAILURE_RULE、DEPLOYMENT_REQUIREMENT，未把性能差改成非法输入。

## 控制器与科学结论

V2支持MEASURED_EXACT_LOOKUP、RULE_PREDICTION、UNKNOWN及NO_CONFIRMED_FEASIBLE_DETECTOR的语义区分；当前0条可靠规则，因此未测Motion/Haze的3个条件全部未知弃权。已测21条件仍4选择、17弃权。固定YOLO26s/1280也覆盖4个P，控制器没有提高可行覆盖；优势限于合法性、权重来源、区间筛选及解释/弃权。Oracle有同条件实测标签，只是数据内参考，不是部署学习优势。不存在从图像自动识别退化的模型。

## 正式RQ4差距

原1280/100epoch、YOLOv8s/YOLO26s、seed42/43/44、CONTROL/ALL_FIVE共12臂以及五项YOLOv8s单增强消融保持不变。当前正式匹配臂0/12；每臂647100 anchor曝光，12臂7765200，另五消融需要3235500。正式终点21P×2分辨率×12臂=504配置、276192张次评价。旧best或960/10轮checkpoint不能替代固定epoch100 EMA的匹配臂。已有包与CPU证据可复用，现无可靠目标机器100轮总成本，ESTIMATE_NOT_RELIABLE，不能塞入约5小时余额。

VM43 Gaussian AP确实提高4.955pp、Tiny提高1.450pp，但四格Tiny判据仍FAIL，完整Gap没关闭。外部范围仍LIMITED_EXTERNAL_EXPLORATORY/独立性UNKNOWN，绝未用于本轮规则训练。

## 下一项GPU与论文可写范围

优先完成科学参数绑定审阅和新的VM45专项许可，之后才是Motion×Haze8格开发验证；现有12227秒/3.396小时仅外推，需要真实预检后完整预算/新热安全准入，不复用VM43许可。RollingShutter继续阻塞。可提供给导师的是原UVL复现、可追溯子集结果、负向/有限的整组规则评价、静态假设敏感性和安全弃权；新Motion/Haze交互、可靠全空间Gap和正式RQ4效果都不能写成已完成。

## 文件与复现

实际命令`command.txt`；来源`source_provenance.json`；资源`actual_resource_usage.json`；完整留出预测`held_out_rule_results.csv`、每折IDs`rule_group_folds.csv`；候选路径`learned_failure_rules.csv`；原值/区间/混淆矩阵`rule_learning_validation.json`；静态表`static_empirical_gap_analysis.csv`；图表`plots/`；验收`final_acceptance.json`和SHA256SUMS.txt。所有旧Run、导师UVL及权重只读，没有Git操作。Run：`{out}`。
'''
 report_dir=ROOT/'datasets/reportcn';paths=[]
 for name,body in [('VM45_复合退化交互与Gap学习_科学报告_CN.md',text),('VM45_导师会议讲解与问答_CN.md',text.replace('# VM45 复合退化与Gap学习','# VM45 导师讲解与问答\n\n先说明：本轮真正新增的是CPU规则诊断，不是Motion/Haze GPU结果。逐项证据如下。')),('VM45_RQ1_RQ4_论文进展_CN.md',text)]:
  p=report_dir/name
  if p.exists():p=p.with_name(p.stem+'_'+out.name+'.md')
  p.write_text(body,encoding='utf-8');paths.append(p)
 en=out/'VM45_Research_Update_EN.md';en.write_text('VM45 technical evidence brief, requiring human review under the supervisor AI-use policy.\n\nNew GPU evaluations:0. Motion/Haze AP and DID:NOT_EVALUATED. Grouped development diagnostics and symbolic sensitivity were run. No empirical rule was admitted for UVL pruning. Exact lookup retains4 selected and17 abstained conditions. Formal100-epoch RQ4 remains unmeasured. See source-bound JSON/CSV for actual scores and limitations.\n'+table(models),encoding='utf-8')
 links=''.join(f'<li><a href="{p.as_uri()}">{p.name}</a></li>' for p in paths)
 pics=''.join(f'<figure><img src="plots/{c["figure"]}.png" style="max-width:100%"><figcaption>{c["caption_CN"]}</figcaption></figure>' for c in captions)
 (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>VM45 CPU evidence</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto}figure{margin:30px 0}</style><h1>VM45 · CPU completed / GPU blocked</h1><p>'+out.name+'</p><ul>'+links+'</ul>'+pics,encoding='utf-8')
 return paths,plots
