"""Source-driven VM44 figures and private Chinese interpretation."""
import html
import json
from pathlib import Path
import numpy as np

from vm44_io import ROOT, HERE, V43, V40, V41, read, csv_read, save
from main.labels import VIS_NAMES


def table(headers, rows):
    clean = lambda x: str(x).replace('|', '/').replace('\n', ' ')
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |', *['| '+' | '.join(clean(x) for x in r)+' |' for r in rows]])


def metric(classes, strategy, condition, category, size='tiny', name='map50_95'):
    row = next(r for r in classes if r['strategy'] == strategy and r['condition'] == condition and r['category_id'] == category and r['size'] == size)
    return row[name]


def deliver(out, spec, fn, fp, sizes, classes, transitions, dev, conditions, validation, selections, baselines, proposal, cost):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    folder = out/'plots'; folder.mkdir()
    plots, captions = [], []
    colors = ['#556a80', '#249385', '#b27e48', '#9b5968']
    scope = 'External: LIMITED_EXTERNAL_EXPLORATORY, independence UNKNOWN | Development: VM36 only | No new YOLO run'
    def finish(fig, name, caption):
        fig.text(.01, .01, scope, fontsize=7)
        fig.tight_layout(rect=(0, .04, 1, 1))
        for suffix in ('png', 'pdf'):
            path = folder/(name+'.'+suffix)
            if path.exists(): raise FileExistsError(path)
            fig.savefig(path, dpi=185); plots.append(path)
        plt.close(fig); captions.append({'figure': name, 'caption_CN': caption})
    combinations = [('CONTROL', 'Normal'), ('VARIABILITY_GUIDED', 'Normal'), ('CONTROL', 'Gaussian_s15'), ('VARIABILITY_GUIDED', 'Gaussian_s15')]
    x = np.arange(10)
    fig, ax = plt.subplots(figsize=(13, 5.6))
    for j, (strategy, condition) in enumerate(combinations):
        vals = [metric(classes, strategy, condition, i) for i in range(1, 11)]
        ax.bar(x+(j-1.5)*.19, [v*100 if v is not None else np.nan for v in vals], .18, color=colors[j], label=strategy.replace('VARIABILITY_', '')+' / '+condition)
    ax.set(xticks=x, xticklabels=VIS_NAMES, ylabel='Saved COCO Tiny AP50:95 (%)', title='01 | Tiny AP by native class and strategy')
    ax.tick_params(axis='x', rotation=25); ax.legend(ncol=2, fontsize=8)
    finish(fig, '01_Tiny_AP_by_class', 'VM43原保存COCO Tiny AP，原生十类、原图面积<256像素²。四组共享GT；无GT类别保持null而非填0。柱图没有新增或虚构逐类置信区间。')
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, area in zip(axes, ('all', 'tiny')):
        for j, condition in enumerate(('Normal', 'Gaussian_s15')):
            values = [(metric(classes, 'VARIABILITY_GUIDED', condition, i, area)-metric(classes, 'CONTROL', condition, i, area))*100
                      if metric(classes, 'CONTROL', condition, i, area) is not None and metric(classes, 'VARIABILITY_GUIDED', condition, i, area) is not None else np.nan for i in range(1, 11)]
            ax.bar(x+(j-.5)*.35, values, .33, label=condition, color=colors[j])
        ax.axhline(0, color='gray', linewidth=.7); ax.set(xticks=x, xticklabels=VIS_NAMES, ylabel='GUIDED - CONTROL AP (pp)', title=area.upper()+' original-pixel area'); ax.tick_params(axis='x', rotation=45); ax.legend(fontsize=8)
    finish(fig, '02_Gaussian_Normal_AP_changes', '总体/微小目标按类AP差值来自已保存指标。Gaussian训练策略的收益、正常图像代价均保留；AP为类别宏平均，不可把某类漏检总量当成总体AP的精确分解。')
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ['BOTH_DETECT', 'CONTROL_ONLY_LOST', 'GUIDED_ONLY_IMPROVED', 'BOTH_MISS']
    for ax, condition in zip(axes, ('Normal', 'Gaussian_s15')):
        rows = [next(r for r in transitions if r['condition'] == condition and r['category_id'] == 0 and r['size'] == area) for area in ('tiny', 'small', 'medium', 'large')]
        base = np.zeros(4)
        for j, label in enumerate(labels):
            values = np.array([r[label]/r['GT']*100 for r in rows]); ax.bar(range(4), values, bottom=base, color=colors[j], label=label); base += values
        ax.set(xticks=range(4), xticklabels=['Tiny', 'Small', 'Medium', 'Large'], ylabel='Paired valid GT (%)', title=condition)
    axes[1].legend(fontsize=7, loc='center left', bbox_to_anchor=(1, .5))
    finish(fig, '03_Paired_object_transitions', '对每个原始GT比较四种确定性匹配转移，置信度≥.25、同类IoU≥.50。分母为该尺度有效GT，使用同一批1610图和原GT ID；不是COCO AP95的数学分解或因果归因。')
    fig, ax = plt.subplots(figsize=(10, 5.3))
    for j, (strategy, condition) in enumerate(combinations):
        rows = [r for r in sizes if r['strategy'] == strategy and r['condition'] == condition and r['size'] == 'FINE_BIN']
        ax.plot(range(7), [r['recall']*100 if r['recall'] is not None else np.nan for r in rows], marker='o', color=colors[j], label=strategy.replace('VARIABILITY_', '')+'/'+condition)
    ax.set(xticks=range(7), xticklabels=['<16', '16–64', '64–144', '144–256', '256–1024', '1024–9216', '>=9216'], xlabel='GT bbox area in original pixels² (lower inclusive)', ylabel='Diagnostic detection recall (%)', title='04 | Target area and detection success'); ax.legend(fontsize=8)
    finish(fig, '04_Size_detection_success', '目标按原图bbox面积固定分组，不随分辨率或预测框改变成员。曲线是score=.25、IoU=.50下的TP/GT，非单图AP。尺度与检测成功率的关联不能单独证明源图像不可恢复或特征网络丢失机制。')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, data, key, area, title in [(axes[0], fp, 'FP', 'all', 'FP: all predicted-box sizes'), (axes[1], fn, 'FN', 'tiny', 'FN: Tiny GT only')]:
        delta = []
        for cat in range(1, 11):
            lookup = lambda s: next(r[key] for r in data if r['condition'] == 'Gaussian_s15' and r['strategy'] == s and r['category_id'] == cat and r['size'] == area)
            delta.append(lookup('VARIABILITY_GUIDED')-lookup('CONTROL'))
        ax.barh(VIS_NAMES, delta, color=[colors[1] if n < 0 else colors[3] for n in delta]); ax.axvline(0, color='gray', linewidth=.7); ax.set(xlabel='GUIDED - CONTROL count', title=title)
    finish(fig, '05_FP_FN_class_changes', 'Gaussian条件下按类比较FP和Tiny FN。FP按预测类别、全部预测框面积计数；Tiny FN按GT类别及GT面积计数，两个面板成员定义不同。正FP差意味着固定score阈值下误报负担增加，不能与AP95一一对应。')
    coverage = csv_read(V41/'FM_P_measured_coverage.csv')
    leaves = [r['leaf'] for r in coverage]
    matrix = np.array([[leaf in c['selected_leaves'].split('|') for c in conditions] for leaf in leaves], dtype=int)
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.imshow(matrix, aspect='auto', cmap=ListedColormap(['#eeeeee', '#249385']), vmin=0, vmax=1)
    ax.set(yticks=range(len(leaves)), yticklabels=leaves, xticks=range(21), xticklabels=[f'P{i+1:02d}' for i in range(21)], xlabel='21 measured development conditions (IDs in condition index)', title='06 | Actual FM_P leaf coverage; gray = not selected')
    ax.tick_params(axis='x', rotation=70)
    finish(fig, '06_FM_P_coverage', '只表示VM36有目的选取的21种退化中各叶子的实际出现情况。灰色不是该特征失败；MotionBlur、Haze、Rain等未测条件不能据此标为非法或性能差。46368个合法配置并未全部评价。')
    fig, ax = plt.subplots(figsize=(12, 6)); ax.set(xlim=(0, 12), ylim=(0, 6)); ax.axis('off')
    nodes = [(1.8, 4.8, 'Externally specified FM_P\nNo image-label input'), (6, 4.8, 'Original UVL SAT\nPixel + annotation support'), (10.1, 4.8, 'Development lookup\nExact measured P-D only'),
             (10.1, 1.6, 'Eligible fixed checkpoints\nSHA / FP32 / provenance'), (6, 1.6, 'Tiny / latency CI criteria\nRank interval PASS only'), (1.8, 1.6, 'Select + source trace\nor explicit abstention')]
    for nx, ny, text in nodes:
        ax.add_patch(FancyBboxPatch((nx-1.55, ny-.65), 3.1, 1.3, boxstyle='round,pad=.04', facecolor='#e5efed', edgecolor='#258477')); ax.text(nx, ny, text, ha='center', va='center', fontsize=10)
    for a, b in [((3.4, 4.8), (4.4, 4.8)), ((7.6, 4.8), (8.5, 4.8)), ((10.1, 4.1), (10.1, 2.3)), ((8.5, 1.6), (7.6, 1.6)), ((4.4, 1.6), (3.4, 1.6))]: ax.annotate('', b, a, arrowprops={'arrowstyle': '->', 'color': '#536a79', 'lw': 2})
    ax.set_title('07 | Runnable measured-evidence decision flow')
    finish(fig, '07_Variability_decision_flow', '控制器接收外部明确给定的FM_P叶子；使用真实UVL CNF检查合法性，再检查已测像素/GT语义、历史D资格、Tiny/时延区间，按预定规则选取或弃权。它不读取部署图像GT，不识别未知退化，不为未测条件生成AP。')
    selected_map = {r['P']: r['D'] for r in selections}
    fig, ax = plt.subplots(figsize=(12, 9)); ax.axis('off')
    display = [[f'P{i+1:02d}', c['selected_leaves'][:72], selected_map.get(c['configuration_id'], 'ABSTAIN')] for i, c in enumerate(conditions)]
    tab = ax.table(cellText=display, colLabels=['P', 'Measured leaves (full IDs in CSV)', 'Choice'], loc='center', colWidths=[.07, .70, .23]); tab.auto_set_font_size(False); tab.set_fontsize(7.5); tab.scale(1, 1.6)
    ax.set_title('08 | Development in-sample choices and abstentions')
    finish(fig, '08_Controller_choices', '表中仅列VM36已测条件的查表结果。根据Tiny下界≥.10及时延上界≤100ms保守选择；没有候选通过则弃权。与同信息查表oracle一致是构造结果，不是学到跨环境规律的证据。')
    fig, ax = plt.subplots(figsize=(9.5, 6))
    for j, detector in enumerate(sorted({r['D'] for r in dev})):
        subset = [r for r in dev if r['D'] == detector]
        ax.scatter([r['cached_pipeline_mean_ms'] for r in subset], [r['Tiny_AP']*100 for r in subset], color=colors[j], label=detector, alpha=.7)
    for condition in conditions:
        subset = [r for r in dev if r['condition_id'] == condition['configuration_id']]
        frontier = [r for r in subset if not any(s['cached_pipeline_mean_ms'] <= r['cached_pipeline_mean_ms'] and s['Tiny_AP'] >= r['Tiny_AP'] and (s['cached_pipeline_mean_ms'] < r['cached_pipeline_mean_ms'] or s['Tiny_AP'] > r['Tiny_AP']) for s in subset)]
        frontier.sort(key=lambda r: r['cached_pipeline_mean_ms'])
        ax.plot([r['cached_pipeline_mean_ms'] for r in frontier], [r['Tiny_AP']*100 for r in frontier], color='gray', alpha=.2, linewidth=.8)
    ax.axhline(10, color='#b04d46', linestyle='--'); ax.axvline(100, color='#b04d46', linestyle='--')
    ax.set(xlabel='Measured VM36 cached pipeline mean (ms)', ylabel='Saved Tiny AP50:95 (%)', title='09 | Detector tradeoffs; Pareto links only within the same P'); ax.legend(fontsize=8)
    finish(fig, '09_Development_detector_Pareto', '84个点为21个P×4个D的开发测量。灰线只连接同一P内的非支配点，不在不同退化条件间构造虚假的全局Pareto前沿。图展示点值，控制器仍按实际置信区间判断；旧checkpoint训练和选模史局限保留。')
    fig, ax = plt.subplots(figsize=(11, 4.5)); ax.axis('off')
    tab = ax.table(cellText=[['ck01 /1280']+['PROPOSED']*4, ['ck05 /1280']+['PROPOSED']*4], colLabels=['Detector', 'Normal', 'Motion k21', 'Haze beta1.5', 'Motion + Haze'], loc='center'); tab.auto_set_font_size(False); tab.set_fontsize(10); tab.scale(1, 2)
    ax.set_title('10 | PROPOSED / NOT_MEASURED: missing Motion x Haze contrast')
    ax.text(.02, .1, '548 development images per cell; 8 cells / 4,384 image inferences.\nAngles Uniform[0,180), airlight220: historical proposals, NOT supervisor-approved.\nFresh scientific/resource/thermal authorization required; RollingShutter blocked.', transform=ax.transAxes, fontsize=9)
    finish(fig, '10_Next_minimal_experiment', '未来设计不是本轮结果：两历史模型×Normal/Motion21/Haze_dense/二者复合，共8格4384张次。kernel21和beta1.5来自导师UVL；角度均匀0–180与airlight220只作为历史实现候选，尚未被导师定义或批准，不因CPU像素测试通过而自动准入GPU。')
    save(out/'figure_captions_CN.json', captions)
    interpretation = interpret(out, fn, fp, transitions, classes, sizes)
    reports = chinese_reports(out, spec, fn, fp, classes, transitions, validation, selections, baselines, proposal, cost, interpretation, captions)
    image_html = ''.join(f'<figure><img src="plots/{r["figure"]}.png" style="max-width:100%"><figcaption>{html.escape(r["caption_CN"])}</figcaption></figure>' for r in captions)
    doc_links = ''.join(f'<li><a href="{p.as_uri()}">{html.escape(p.name)}</a></li>' for p in reports)
    csv_links = ''.join(f'<li><a href="{p.name}">{html.escape(p.name)}</a></li>' for p in sorted(out.glob('*.csv')))
    with (out/'index.html').open('x', encoding='utf-8') as f:
        f.write('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>VM44 evidence</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:0 20px;color:#243a4a}figure{margin:35px 0}figcaption{line-height:1.6}code{overflow-wrap:anywhere}</style><h1>VM44 · CPU evidence and decision prototype</h1><p><code>'+out.name+'</code></p><p>External descriptive analysis only; development-only measured lookup. No GPU.</p><ul>'+doc_links+csv_links+'</ul>'+image_html+'</html>')
    return reports, plots, interpretation


def interpret(out, fn, fp, transitions, classes, sizes):
    mechanisms = []
    for strategy in ('CONTROL', 'VARIABILITY_GUIDED'):
        row = next(r for r in fn if r['strategy'] == strategy and r['condition'] == 'Gaussian_s15' and r['category_id'] == 0 and r['size'] == 'tiny')
        mechanisms.append({'claim': 'Fixed-threshold Tiny recall and FN burden', 'status': 'DIRECTLY_MEASURED', 'strategy': strategy, 'GT': row['GT'], 'TP': row['TP'], 'FN': row['FN'], 'recall': row['recall_at_score025_IoU050'],
            'candidate_mechanisms': {k: v for k, v in row.items() if k.endswith('_CANDIDATE')}, 'scope': 'Diagnostic score=.25 andIoU=.50; not AP95 decomposition'})
    mechanisms.extend([
        {'claim': 'Detection success associated with original bbox size', 'status': 'SUPPORTED_ASSOCIATION', 'evidence': 'VM43_size_stratified_diagnostics.csv; no image or training selection'},
        {'claim': 'Class imbalance affects missed-object counts', 'status': 'DIRECTLY_MEASURED', 'evidence': 'GT counts by class; COCO AP remains category macro-average'},
        {'claim': 'Small source targets or Gaussian perturbation may remove useful detail', 'status': 'PLAUSIBLE_NOT_PROVEN', 'reason': 'No sensor/optics/feature-map intervention or visibility ground truth'},
        {'claim': 'Architecture-specific feature loss or insufficient training epochs causes the Tiny failure', 'status': 'NOT_EVALUATED', 'reason': 'No new training, feature probe, resolution intervention or matched architecture experiment'},
        {'claim': 'GUIDED crosses the frozen10% Tiny requirement', 'status': 'DIRECTLY_MEASURED_FALSE', 'evidence': 'VM43 Tiny upper95 CI<.10 for all4 cells; FAIL remains FAIL'}])
    save(out/'mechanism_evidence.json', {'findings': mechanisms, 'causality': 'NOT_ESTABLISHED', 'external_tuning': False})
    return mechanisms


def chinese_reports(out, spec, fn, fp, classes, transitions, validation, selections, baselines, proposal, cost, interpretation, captions):
    report = ROOT/'datasets/reportcn/VM44_Tiny失败原因与自动决策_科学报告_CN.md'
    guide = ROOT/'datasets/reportcn/VM44_导师会议讲解与问答_CN.md'
    if report.exists() or guide.exists():
        report = report.with_name(report.stem+'_'+out.name+'.md'); guide = guide.with_name(guide.stem+'_'+out.name+'.md')
    totals = [next(r for r in transitions if r['condition'] == condition and r['category_id'] == 0 and r['size'] == area) for condition in ('Normal', 'Gaussian_s15') for area in ('tiny', 'small', 'medium', 'large')]
    gaussian_classes = [r for r in transitions if r['condition'] == 'Gaussian_s15' and r['category_id'] > 0 and r['size'] == 'tiny']
    gaussian_classes.sort(key=lambda r: r['BOTH_MISS']+r['CONTROL_ONLY_LOST'], reverse=True)
    class_table = []
    for i, name in enumerate(VIS_NAMES, 1):
        values = [metric(classes, s, c, i) for s, c in [('CONTROL', 'Normal'), ('VARIABILITY_GUIDED', 'Normal'), ('CONTROL', 'Gaussian_s15'), ('VARIABILITY_GUIDED', 'Gaussian_s15')]]
        class_table.append([name, *[f'{v*100:.3f}' if v is not None else 'null' for v in values]])
    fn_table = []
    for r in fn:
        if r['category_id'] == 0 and r['size'] == 'tiny':
            fn_table.append([r['condition'], r['strategy'], r['GT'], r['TP'], r['FN'], f"{r['recall_at_score025_IoU050']*100:.3f}", r.get('LOW_CONFIDENCE_MATCH_CANDIDATE', 0), r.get('WRONG_CLASS_OVERLAP_CANDIDATE', 0), r.get('LOCALIZATION_NEAR_MISS_CANDIDATE', 0)])
    summary = []
    for method in dict.fromkeys(r['method'] for r in baselines):
        rows = [r for r in baselines if r['method'] == method]
        summary.append([method, sum(r['task_state'] == 'PASS' for r in rows), sum(r['abstain'] for r in rows), rows[0]['evaluation']])
    text = f'''# VM44 Tiny失败诊断与Variability决策原型

RunID：`{out.name}`。本轮CPU-only，真实结果来自已保存预测，没有重新运行YOLO、训练、生成test-dev退化或调整Gaussian策略。

## 1. 两条证据边界

外部分支A：VM43、1610张、74687个有效GT、native10、seed42 CONTROL/GUIDED、Normal/Gaussian15。范围始终为LIMITED_EXTERNAL_EXPLORATORY，历史独立性UNKNOWN。只描述错误，不把这些GT用于控制器拟合、排名、阈值校准或新增强策略。

开发分支B：VM36的21个P×4个历史D，共84对、每对548张开发图。只用此分支构建选择器。两个分支不混算总体AP。VM37–39的开发策略和三seed结论仅作已验收背景，未重跑。

## 2. 诊断匹配与官方AP

官方AP/AR、逐类和尺度指标原样引用VM43的COCO文件。诊断另行预先固定：score≥.25、同类IoU≥.50，按score降序，分数并列按原预测index；每个预测匹配尚未占用的最大IoU GT，IoU并列按GT ID。重复预测不去重，其余未匹配副本计FP；无GT图的候选计FP；无GT类别的AP保持null。

先复用原rasterized ignore-union（覆盖≥50%排除，原始floor/ceil规则），然后诊断。保留原图坐标、不改框，原ignore剔除数量逐图记录；诊断输入的保留预测总数与原COCO输入mapped_count完全一致。外部逐类AP、原预测gzip、GT引用、manifest、指标和Bootstrap均按原封存SHA核验。

TP/FP/FN是这个固定工作点的描述，**不是COCO AP50:95的精确数学分解**。FN候选原因采用预定互斥顺序：竞争匹配→低置信度同类重叠→高分异类重叠→同类定位近失→无高分定位候选。它们是可观察候选关联，不证明物理成因。

## 3. 十类Tiny AP与固定10%门槛

下表百分数，原值0–1在CSV。全部面积/类别AP50、AP95、AR见`VM43_tiny_class_metrics.csv`。

{table(['类别','Normal C','Normal G','Gaussian C','Gaussian G'], class_table)}

VM43总体Gaussian AP95由12.338%到17.293%，差+4.955pp，[4.648,5.290]pp；Tiny由5.004%到6.454%，差+1.450pp，[1.070,2.114]pp；Normal整体AP差−.463pp。200次配对图像区间仍条件于一个训练seed，不代表训练seed显著性。

Tiny要求未跨越：四格Tiny区间上界均低于10%。缓存时延都低于100ms，四格固定状态仍FAIL。**改善Tiny**与**达到Tiny任务门槛**是不同判断；本轮没有更改门槛。

## 4. Tiny召回、漏检和可观察候选

{table(['条件','策略','Tiny GT','TP','FN','召回%','低分候选','异类重叠候选','定位近失候选'], fn_table)}

低分候选表示存在score<.25但同类IoU≥.50的已保存预测；异类重叠表示高分其他类框与该GT相交达到.50；定位近失表示同类高分框最佳IoU在[.1,.5)。竞争匹配和无高分定位候选另存完整FN表。它们不对应重新优化后的工作点，不能据此改变test-dev阈值。

## 5. 同GT配对转移

{table(['条件','尺度','GT','双方检出','C有G无/丢失','C无G有/改善','双方漏检','净检出变化'], [[r['condition'],r['size'],r['GT'],r['BOTH_DETECT'],r['CONTROL_ONLY_LOST'],r['GUIDED_ONLY_IMPROVED'],r['BOTH_MISS'],r['net_detected_change']] for r in totals])}

Gaussian Tiny各类详细转移：

{table(['类别','GT','改善','丢失','双方漏检','净变化'], [[r['category_name'],r['GT'],r['GUIDED_ONLY_IMPROVED'],r['CONTROL_ONLY_LOST'],r['BOTH_MISS'],r['net_detected_change']] for r in gaussian_classes])}

按漏检数量排序最多的类别是{', '.join(r['category_name'] for r in gaussian_classes[:3])}。这是目标数量/召回的归属，不是总体AP贡献的因果分解；AP对有GT类别宏平均，不按GT频数简单加权。FP按预测类别与预测框面积记录，FN按GT类别与GT面积记录；两者的Tiny成员不能混为同一集合。

## 6. 机制证据等级

{table(['机制/主张','结论等级'], [[r['claim'], r['status']] for r in interpretation])}

可以直接观察固定工作点的召回不足、误报数量、低分/定位/异类重叠候选和目标尺度分布。原图Tiny面积小、Gaussian损伤细节是合理解释候选，但没有光学/传感器、特征图或受控机制干预，不能将其写成已证明原因。缺训练轮数、架构特征损失等没有在VM44评价。

## 7. 实际可运行的FM_P→FM_D选择器

直接解析导师原UVL并由现有FlamaPy导出CNF，保留原FM_P 25叶/5约束/46368合法配置及FM_D 34叶/4约束/635904合法配置的已验收计数；本轮不重复全空间计数。对输入完整叶子赋值做CNF求解，不把低AP当非法。

API输入：外部给定P叶子、Tiny质量阈值、时延预算、允许的已测D及开发证据。依次检查SAT、像素/GT支持、精确P证据、D资格、固定区间规则，再在PASS中按平均缓存时延最小选择，Tiny下界和D ID破同分。未知P返回UNKNOWN_UNMEASURED_CONDITION；无合格D返回NO_CONFIRMED_FEASIBLE_DETECTOR；没有对应训练权重时不把增强开关当新模型。

支持历史YOLOv8s/YOLO26s × Infer640/1280，FP32、native10、已有权重SHA和CPU张量健康通过。原args记录1280/100轮/seed42/AdamW/VisDrone；但best具体epoch、完整初始化字节来源、历史Albu实际激活仍UNKNOWN。因此这是固定历史checkpoint的经验选择，不是完整训练特征均获证明的FM_D；`--strict-training`会对全部D弃权。选择结果保留来源P/D ID、权重身份、质量区间、时延和解释。

## 8. 基线与组保留验证

{table(['方法','21个P中选到PASS','弃权','评价范围'], summary)}

有效区间PASS对{validation['interval_feasible_pairs']}/84，控制器选择{validation['P_selected']}/21，弃权{validation['P_abstained']}/21。查表oracle与控制器使用同一已测响应、同一判据和排序，故一致是构造结果，不是学习优越性或新算法。

21折每次保留完整P的4行，P不会同时进入训练和评价。精确查表对保留P全部返回未知，覆盖率{validation['group_holdout']['coverage']:.1%}，预测准确率null；这仅证明正确弃权，不是学习泛化。只有4个可行P组、21个有目的条件、共享548图和先前选模/研究暴露，不足以可靠学习普适特征规则，返回INSUFFICIENT_DATA_FOR_GENERALIZABLE_RULE，没有拟合复杂模型制造高准确率。外部test-dev用于拟合的行数0。

## 9. 与训练增强选择的边界

本轮实现的是在已有训练权重间选推理配置。VM38的Gaussian p=.25、sigma5.5–30保持冻结；它源自开发退化损失的事后策略选择，不是自动多族增强控制器。训练配方开关没有已训练权重就不能被选成新D。未来训练增强仅在独立开发发现/确认划分上预注册规则，不从本次test-dev逐例错误改数据或配方。

## 10. 下一次最小GPU设计

仅提出MotionBlur×Haze四角点：Normal、Motion_k21、Haze_dense、两者复合；ck01/ck05 Infer1280，共8格×548图=4384张次，仍是开发评价。主要目标是每模型内的四点DID，再比较checkpoint DID差，不以最高AP选条件。

kernel21与beta1.5来自原UVL。角度Uniform[0,180)和airlight220仅为历史实现候选，**没有导师批准**；本轮显式标注PROPOSED，CPU合成像素可重复/坐标不变不等于科学准入。Haze为常深度传输模型，不是天然真实雾或三维重建。实际图像生成前须审阅并批准这些绑定，完整源图SHA/像素缓存再冻结；RollingShutter在bbox/ignore映射验证前继续BLOCKED。

按VM43真实7658.531秒/6440张次缩放，线性估计{cost['linear_seconds_NOT_MEASURED']:.1f}秒；×2不确定性及1800秒安全退出余量后规划{cost['planning_seconds_NOT_GUARANTEED']:.1f}秒（{cost['planning_seconds_NOT_GUARANTEED']/3600:.3f}小时），不是实测或保证上界。历史VM41另有方案记录，均不能代替新设备/响应密度/缓存/CPU统计成本准入。科学绑定、资源、GPU授权和热安全分开：本轮没有GPU授权，也不挪用VM43余额。

## 11. 图表与中文图注

'''
    for row in captions:
        text += f"\n**{row['figure']}**：{row['caption_CN']}\n\n![{row['figure']}]({(out/'plots'/(row['figure']+'.png')).as_posix()})\n"
    text += f'''
## 12. 运行、测试与证据

代码：`{HERE}`。`run.py`串联只读来源验收、开发控制器、CPU测试、流式外部诊断、配对转移及报告；`diagnostics.py`负责原坐标/ignore/匹配；`selector.py`负责原CNF和查表选择；`uvl_bridge.py`用原FlamaPy环境解析；`tests_cpu.py`检查边界；`reporting.py`只从实际结果画图。

本轮实际命令在`command.txt`。完整分析默认会创建新Run，不应用来覆盖本Run；仅使用已封存选择器可运行：

```powershell
& 'E:\\ForStudy\\code\\uavgap\\.venv_rq4\\Scripts\\python.exe' -B -X utf8 '{HERE/'run.py'}' --select '{out/'FM_P_to_FM_D_decision_rules.json'}' --leaves Normal
```

`--strict-training`演示训练特征证据不完整时弃权。没有部署当前图像GT输入，也没有自动图像退化识别。P未测就不推测其AP。

CPU测试与48个固定随机图像的独立标量匹配/原ignore实现对照见`CPU_test_results.json`、`CPU_tests.log`、`diagnostic_sample_replay.json`。诊断gzip按图流式读取，内存只保留当前图的IoU矩阵与累计表，不复制大型图像缓存。官方COCO AP没有重算，更没有GPU forward。

结果网页：`{out/'index.html'}`。全部CSV、GT级配对明细、来源与结果SHA在同Run；`source_provenance.json`标注每个输入真实SHA与作用，`resource_usage.json`记录CPU/RAM、GPU0及仅报告的原预算余额，`SHA256SUMS.txt`封存输出。VM36–43原始结果、原权重与账本不修改，无Git pull/rebase/reset/push。
'''
    with report.open('x', encoding='utf-8') as f: f.write(text)
    tiny = [r for r in transitions if r['condition'] == 'Gaussian_s15' and r['category_id'] == 0 and r['size'] == 'tiny'][0]
    spoken = f'''# VM44 导师会议讲解与问答

RunID：`{out.name}`。

## 这次最重要的结果是什么

“Gaussian增强让外部探索的Tiny AP从5.004%提高到6.454%，但还没达到10%。这次没有再训练或调test-dev参数，而是把原预测按GT逐例配对，看看哪些目标从漏检变成检出、哪些反而丢失。”

在score≥.25、IoU≥.50的单独诊断工作点，Gaussian Tiny有{tiny['GUIDED_ONLY_IMPROVED']}个目标只被GUIDED检出、{tiny['CONTROL_ONLY_LOST']}个只被CONTROL检出、{tiny['BOTH_MISS']}个双方都漏检，净检出变化{tiny['net_detected_change']:+d}。这能定位可观察召回问题，不是AP95的精确分解。

## 为什么还没达到门槛

所有四格Tiny区间上界都低于10%。仍有大量Tiny没有高分、同类且位置合格的匹配框；低分、定位近失、异类重叠等在报告有具体计数。原图目标面积小与成功率相关，但没有传感器或特征图干预，不能说已证明某一物理或网络原因。

## 有没有用test-dev做新决策

没有。test-dev只做只读描述，标签没有进入控制器、排序、校准或新增强选择。控制器的数据入口只接受VM36开发来源，并有拒绝外部行的测试。外部科学范围仍LIMITED_EXTERNAL_EXPLORATORY，独立性UNKNOWN。

## 自动选择器真正能做什么

给它明确的合法P条件、Tiny要求、时延预算和允许D，它会检查原UVL、像素/GT支持和历史权重，再查询开发证据；只在区间PASS中选最快候选，否则明确弃权。{validation['P_selected']}个已测P有选择，{validation['P_abstained']}个弃权。支持4个历史执行设置，不代表支持635904个D或46368个P的响应。

## 它是不是新的学习算法

不是。当前是带来源/语义/资格/区间检查的可运行证据选择器，和同信息查表oracle一致。完整P分组保留后，精确查表对未测条件全部弃权，没有模型推断去填AP。只有4个可行P组，可靠通用规则学习仍不足。

## 原训练参数是否都已完全证明

没有。权重SHA、原生类别、CPU张量健康、已测FP32推理和args来源存在；但历史Albu实际激活、best具体epoch、完整初始化字节证据仍不全。默认只做固定历史checkpoint经验选择；严格完整训练特征模式会弃权。不能把声明的FM_D叶子误当训练过程的完整证明。

## 推理选择和训练增强有什么区别

选择已有640/1280检测器不需要训练新模型。选择一个新增强开关并不会产生训练好的权重；本轮没有实现自动多族增强训练，也没有改旧Gaussian概率或sigma范围。

## 下一次最小实验是什么

先审阅Motion角度与Haze airlight的非导师定义绑定，再考虑Normal/Motion21/Haze_dense/复合×两历史检测器的8格开发对照。4384张次，规划约{cost['planning_seconds_NOT_GUARANTEED']/3600:.3f}小时只是估算。airlight220不是导师批准值，合成CPU测试也不是GPU准入。新GPU授权、完整预算和新热安全验收都还需要，RollingShutter继续封锁。

## 文件在哪里，怎样看

先打开[{out.name}]({(out/'index.html').as_posix()})；图1–5解释外部错误，图6–9解释开发证据选择，图10只画未来方案。科学报告：[{report.name}]({report.as_posix()})。

CSV是汇总，`matched_GT_transitions.csv.gz`是每个GT的真实转移，`diagnostic_records/`留每图诊断，原YOLO预测仍引用VM43原文件。`FM_P_to_FM_D_decision_rules.json`可直接用于`run.py --select ... --leaves Normal`，不读新图片、不调用GPU。具体CPU/RAM和SHA看`resource_usage.json`、`source_provenance.json`与`SHA256SUMS.txt`。
'''
    with guide.open('x', encoding='utf-8') as f: f.write(spoken)
    return [report, guide]
