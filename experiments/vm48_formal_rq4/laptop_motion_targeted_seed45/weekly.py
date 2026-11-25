"""Weekly supervisor update from accepted, saved evidence."""
import csv,json,hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
TRANSFER=ROOT/'datasets/local_runs/vm48_formal_rq4/laptop_transfer_pilot_execution_20261118T100506_d3fa1688'
FORMAL=ROOT/'datasets/local_runs/vm48_formal_rq4/vm48_formal_rq4_package_cpu_20261117T210445_36001a64'

def load(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def create(out):
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plots=out/'weekly_plots';plots.mkdir()
    with (FORMAL/'accepted_interactions.csv').open(encoding='utf-8-sig') as f:effects=list(csv.DictReader(f))
    for r in effects:assert sha(r['source'])==r['source_SHA256']
    fig,ax=plt.subplots(figsize=(7,4));x=range(len(effects));v=[100*float(r['DID_raw']) for r in effects]
    errors=[[v[i]-100*float(r['CI_low_raw']) for i,r in enumerate(effects)],[100*float(r['CI_high_raw'])-v[i] for i,r in enumerate(effects)]]
    ax.bar(x,v,color=['#31688e','#35b779','#31688e','#35b779']);ax.errorbar(x,v,yerr=errors,fmt='none',color='black',capsize=4)
    ax.set(xticks=list(x),xticklabels=[r['study']+' '+r['D'].split('_')[0] for r in effects],ylabel='AP50:95 DID (percentage points)',title='Measured development interactions; paired image CI')
    fig.tight_layout();fig.savefig(plots/'accepted_interactions.png',dpi=180);fig.savefig(plots/'accepted_interactions.pdf');plt.close(fig)
    figrefs=[TRANSFER/'plots'/name for name in ['condition_AP_comparison.png','seed_strategy_comparison.png','Tiny_AP_threshold.png','Normal_accuracy_cost.png']]+[plots/'accepted_interactions.png']
    assert all(p.is_file() for p in figrefs)
    progress=[('RQ1','PARTIAL','Original UVL identity preserved; Train960 is an explicit extension outside original FM_D domain.'),('RQ2','MEASURED_DEVELOPMENT','VM45 Motion x Haze and VM47 Gaussian x JPEG; learned rule transfer remains unsafe.'),('RQ3','LOCAL_SCOPE_ONLY','Frozen Tiny/latency criterion; no full-space prevalence or independent safety proof.'),('RQ4','EXPLORATORY_ONLY','Three-seed saved-model transfer completed; formal1280/100epoch not evaluated; newseed45 training not started at this report freeze.')]
    with (out/'weekly_RQ_progress.csv').open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(['RQ','status','evidence']);w.writerows(progress)
    numeric='| Normal | -0.574 | +0.160 |\n| Motion | +0.218 | +0.083 |\n| Haze | -0.333 | -0.560 |\n| Motion+Haze | +0.017 | -0.012 |'
    did='\n'.join('| '+r['study']+' | '+r['D']+' | '+f"{float(r['DID_raw'])*100:+.3f}"+' | '+f"[{float(r['CI_low_raw'])*100:.3f},{float(r['CI_high_raw'])*100:.3f}]"+' |' for r in effects)
    cn=f'''# 2026-11-16—2026-11-22 导师研究成果汇报

截至2026-11-18已验收证据；不是对未来日期实验的完成声明。本报告在新的seed45 GPU训练之前生成。

## 本周实际进展

- VM43完成4格、每格1610图，范围LIMITED_EXTERNAL_EXPLORATORY，不能称严格独立确认；没有将其标签用于调参或规则训练。
- VM44将原UVL、测得Tiny/时延和可行候选关联；精确证据查询与未知P规则分开，不能将查表写成泛化成功。
- VM45完成Motion×Haze八格，VM47完成Gaussian×JPEG八格；均为548开发图、合成退化、共享200次图像Bootstrap。
- VM48已有960/10轮CONTROL/Gaussian_GUIDED三seed42/43/44迁移评价：24格、13152张次，没有新训练。

| 条件 | GUIDED−CONTROL整体AP50:95平均差pp | Tiny AP平均差pp |
| --- | --- | --- |
{numeric}

Motion整体AP三个配对区间均为正，但Motion Tiny区间均跨0；Haze/复合整体AP方向不一致。Normal整体AP三个seed均下降，平均代价0.574pp。复合条件三seed均FAIL→FAIL，未关闭Gap。

该迁移入口错误地按模型预热、未逐格预热；原始AP与图像Bootstrap可用，时延没有完整VM41准入，不释放PASS。不是正式1280/100轮结果，也不是训练seed显著性。

## 真实交互

| 来源 | 检测器 | AP95 DID pp | 配对95%区间pp |
| --- | --- | --- | --- |
{did}

DID是受控合成实验的非加性表现，可能受AP地板效应影响，不解释为物理因果或架构因果。

## 规则安全与RQ状态

VM45冻结树在新条件可评分5个预测PASS中只有1个真PASS：precision1/5=20%；4个实际FAIL均被预测PASS，false-PASS4/4=100%。VM47新复合P中又出现一个实际FAIL被预测PASS，另一个结果UNCERTAIN；部署放行0。没有可靠未知P安全证据时继续ABSTAIN。

| RQ | 状态 | 当前边界 |
| --- | --- | --- |
| RQ1 | 部分实现 | 双UVL来源保留；960训练为明确扩展，不伪装成原Train1280 |
| RQ2 | 开发集交互已测 | Motion×Haze、Gaussian×JPEG；规则迁移不可靠 |
| RQ3 | 仅局部经验Gap | 不外推全空间失败比例；迁移试验时延未准入 |
| RQ4 | 探索性证据 | 旧Gaussian跨退化收益有限；正式100轮未完成 |

## 下一项已获授权实验

VM48_MOTION_TARGETED_SEED45：YOLO26s、960/10轮、seed45、6471/548、FP32、batch2/nbs64、AdamW及原普通增强。CONTROL对Motion_TARGETED，唯一干预为p.25、kernel{{5,9,13,17,25}}、angle Uniform[0,180)，不使用测试kernel21。先执行两臂各首轮并保留内存中的训练/RNG状态，再仅按完整剩余成本准入，不看早期AP挑选。新独立GPU-worker预算36000秒，截止2026-11-23T00:00:00+01:00；旧36小时账本不改。

首要终点是Motion Tiny AP95差及200次同图配对区间；单seed45不能与旧Gaussian seed42–44混合成同策略重复。当前本报告冻结时新训练尚未开始，不填写新AP。最终8格必须每格3次预热，严格复用VM41完整缓存边界。

## 图与来源

'''
    en=f'''# Research update: 16–22 November 2026

Evidence accepted by18 November. Prepared before the new seed45 GPU attempt.

VM43 completed four1610-image cells, labelled LIMITED_EXTERNAL_EXPLORATORY rather than independent confirmation. VM44 links UVL features to measured Tiny/latency evidence; exact lookup is distinct from prediction on unknown conditions. VM45 and VM47 each completed eight548-image development cells with200 paired image-bootstrap draws.

The saved-model transfer study used seeds42/43/44,24 cells and13152 image evaluations:

| Condition | Mean overall AP95 delta, pp | Mean Tiny AP delta, pp |
| --- | --- | --- |
{numeric}

Motion overall AP improves slightly in all three seeds; its Tiny intervals cross zero. Haze/composite directions are inconsistent. Normal overall AP drops by0.574pp on average. Composite remains FAIL to FAIL. The transfer executor used per-model instead of per-cell warmups: AP evidence remains usable, but full VM41 latency admission and PASS release are withdrawn.

| Study | Detector | AP95 DID, pp | Paired95% interval, pp |
| --- | --- | --- | --- |
{did}

These are development-only synthetic interactions, not physical or architectural causality. AP floor effects can affect DID. The VM45 frozen tree had one true PASS among five predicted PASS outcomes; four of four actual failures were falsely released. VM47 added another false PASS on its new composite condition. Unknown-condition deployment stays ABSTAIN.

| Question | Status |
| --- | --- |
| RQ1 | UVL provenance retained; Train960 is an explicit extension |
| RQ2 | Measured local interactions; unsafe rule transfer |
| RQ3 | Local empirical Gap only; no full-space prevalence estimate |
| RQ4 | Exploratory augmentation evidence; formal1280/100epoch not evaluated |

Next authorized study: YOLO26s, seed45,960/10 epochs,6471/548 images, FP32,batch2/nbs64; CONTROL versus Motion targeting with p=.25, kernels{{5,9,13,17,25}}, angle Uniform[0,180), excluding test kernel21. A new36000-second allocation expires at2026-11-23T00:00:00+01:00; the old ledger is unchanged. Two complete first epochs precede cost-only admission for the remaining training. The primary endpoint is paired Motion Tiny AP95. New outcomes are not available at this report freeze. One new training seed cannot establish multi-seed stability. Each final evaluation cell must receive three warmups.

## Evidence figures and provenance

'''
    for i,p in enumerate(figrefs,1):
        cn+=f'图{i}：`{p}`\n\n';en+=f'Figure{i}: `{p}`\n\n'
    sources=[ROOT/'datasets/local_runs/vm43_external_pilot/vm43_external_pilot_execution_20261116T222929_46c7e0ba/final_acceptance.json',ROOT/'datasets/local_runs/vm44_tiny_error_variability_decision/vm44_tiny_error_variability_decision_analysis_20261117T031628_9ebff202/FM_P_to_FM_D_decision_rules.json',ROOT/'datasets/local_runs/vm45_motion_haze_gap_learning/vm45_motion_haze_gap_learning_execution_20261117T172830_0d9cb1ee/rule_and_controller_update.json',ROOT/'datasets/local_runs/vm47_gaussian_jpeg_factorial/vm47_gaussian_jpeg_factorial_execution_20261117T190442_7166b669/final_acceptance.json',FORMAL/'accepted_interactions.csv',TRANSFER/'scientific_summary.json',TRANSFER/'timing_protocol_deviation.json']
    rows=[{'path':str(p),'SHA256':sha(p)} for p in sources+figrefs]
    with (out/'weekly_sources.json').open('x',encoding='utf-8') as f:json.dump(rows,f,ensure_ascii=False,indent=2)
    tail='\n\nSource paths and SHA256: `'+str(out/'weekly_sources.json')+'`. Figures are existing accepted plots or regenerated from verified CSV. This is a factual supervisor update, not approved manuscript text.\n'
    paths=[ROOT/'datasets/reportcn/WEEK_20261116_20261122_导师研究成果汇报_CN.md',ROOT/'datasets/reportcn/WEEK_20261116_20261122_Research_Update_EN.md']
    for p,text in zip(paths,[cn,en]):
        if p.exists():raise FileExistsError('Preserve existing weekly report: '+str(p))
        with p.open('x',encoding='utf-8') as f:f.write(text+tail)
    return paths
