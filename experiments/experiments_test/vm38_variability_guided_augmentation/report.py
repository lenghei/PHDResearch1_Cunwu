"""Render the verified VM38 comparison without model execution."""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import run as experiment


def review(out, verify=False):
    load, sha = experiment.load, experiment.study.sha256
    record = out / 'report_review.json'
    if verify:
        value = load(record)
        for item in value['artifacts']:
            if sha(Path(item['path'])) != item['sha256']:
                raise ValueError('A reviewed artifact changed.')
        experiment.check(out)
        print('REPORT_VERIFIED', out)
        return
    if record.exists():
        raise FileExistsError('The final report is already reviewed; use --verify.')
    experiment.study.runtime(out, cuda=False)
    result = load(out / 'execution_result.json')
    acceptance = load(out / 'final_acceptance.json')
    if result['status'] != 'VM38_TARGETED_PILOT_COMPLETE_DEVELOPMENT_ONLY' or acceptance['status'] != 'PASS':
        raise ValueError('The complete measured comparison has not passed acceptance.')
    experiment.check(out)
    with (out / 'configuration_metrics.csv').open(encoding='utf-8', newline='') as stream:
        metrics = list(csv.DictReader(stream))
    comparisons = load(out / 'paired_comparisons.json')
    with (out / 'degradation_losses.csv').open(encoding='utf-8', newline='') as stream:
        losses = list(csv.DictReader(stream))
    def ap(arm, res, context, size='all'):
        return float(next(r['map50_95'] for r in metrics if r['arm'] == arm and
                          int(r['resolution']) == res and r['condition'] == context and r['size'] == size))
    contexts = ('Normal', 'JPEG_Q25', 'JPEG_Q75', 'Gaussian_s15')
    arms = ('CONTROL', 'ALL_FIVE', experiment.ARM)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plots = out / 'plots'
    plots.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.8), constrained_layout=True)
    colors = ('#54677d', '#e6a24c', '#278b8a')
    for column, resolution in enumerate((640, 1280)):
        for row, size in enumerate(('all', 'tiny')):
            ax = axes[row, column]
            for group, arm in enumerate(arms):
                ax.bar(np.arange(4) + (group - 1) * .25,
                       [100 * ap(arm, resolution, c, size) for c in contexts],
                       width=.24, color=colors[group], label=arm)
            ax.set_xticks(range(4), ('Normal', 'JPEG25', 'JPEG75', 'Gauss15'))
            ax.set_ylabel(('AP50:95' if size == 'all' else 'Tiny AP') + ' (%)')
            ax.set_title(f'Inference {resolution}; training 960')
            ax.spines[['right', 'top']].set_visible(False)
            ax.grid(axis='y', alpha=.2)
            ax.set_axisbelow(True)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle('YOLO26s: one development-selected Gaussian policy, seed42')
    figure = plots / 'ap_comparison.png'
    fig.savefig(figure, dpi=180)
    plt.close(fig)
    old_report = load(out / 'final_report.json')
    report = Path(old_report['path'])
    snapshot = out / 'generated_snapshot'
    snapshot.mkdir()
    for path in (report, out / 'final_report.json', out / 'SHA256SUMS.txt'):
        shutil.copy2(path, snapshot / path.name)
    realization = load(out / 'augmentation_realization.json')
    training = acceptance['thermal']['YOLO26s_seed42_VARIABILITY_GUIDED']
    evaluation = acceptance['thermal']['evaluation']
    target = out / ('YOLO26s_seed42_' + experiment.ARM)
    selected = load(target / 'training_acceptance.json')['selected_checkpoint']
    budget = result['budget']
    lines = ['# VM38：定向Gaussian增强的收益与clean代价', '',
        '**执行完成；观察到噪声条件绝对AP提高，同时存在clean AP下降。不是全面优于CONTROL。**', '',
        f"RunID：`{out.name}`。对照：`{experiment.REFERENCE_ID}`。", '',
        '范围：**DEVELOPMENT_ONLY / POST_HOC_DEVELOPMENT_POLICY_SELECTION / 单训练seed42**。不构成独立确认或完整RQ4。', '',
        '## 1. 决策及冻结', '',
        '既有CONTROL的Gaussian sigma15 AP95损失在640/1280下分别为3.340/8.136个百分点，均大于相应JPEG Q25损失1.275/5.404个百分点；JPEG Q75损失接近0。',
        '原ALL_FIVE的clean AP95下降5.431/7.709个百分点。因此只选择Gaussian NoiseAug：概率0.25，sigma均匀5.5–30，沿用既有精确测试值排除规则。其余四个新增增强家族关闭，普通增强保持原样。',
        '0.25是保留75%不加定向扰动输入的固定先验，没有逐次搜索最优概率或强度；本轮只训练这一种策略。训练实际触发Gaussian '+str(realization['counts']['gaussian'])+' / '+str(realization['events'])+f"次（{100*realization['observed_probability']:.3f}%）。其余输入仍接受普通增强。",
        'FM_D允许VisDrone_FT结果单独启用NoiseAug，FM_P允许Normal+Gaussian。COCO只是共同初始化，不把已训练结果称为COCO_ZeroShot。photons仅为兼容原schema保留，Gaussian概率为1，实际未使用ShotNoise。',
        '**960训练是已批准VM37探索扩展，不是原FM_D的Train640/Train1280叶节点。原UVL不变；连续训练强度也不冒充离散FM_P叶配置。**',
        '本策略属于开发反馈引导的固定家族选择，不是在线自适应控制器；也不是新采样算法。', '',
        f"策略SHA256：`{sha(out / 'policy.json')}`。",
        f"冻结时间：`{load(out / 'execution_freeze.json')['frozen_UTC']}`；早于本轮训练。", '',
        '## 2. 匹配条件与真实训练', '',
        '三组均从同一健康YOLO26s native10初始化开始：6471训练图、960、10轮、seed42、FP32、batch2、nbs64、workers0、AdamW、lr0=0.01、warmup3、相同cosine日程与普通增强。固定第10轮EMA。',
        'close_mosaic=10且总训练10轮，因此从epoch0关闭Mosaic。原1280/100轮正式设计和YOLOv8s五项消融均未改动。',
        '本轮64,710张次曝光、32,360次真实forward/backward、1816次optimizer及EMA更新全部完成。与两个旧对照逐batch核对：图像顺序、实际NCHW、累积规则、更新计数完全一致；逐更新实际学习率也完全一致。',
        '沿用原计划，最后17个microbatch不额外flush；三组相同。10份FP32权重均保存、重载并核对一致，没有half strip或精确断点续训声明。', '',
        '## 3. 绝对AP结果', '',
        '表中为百分数，单元格为 **AP50:95 / Tiny AP**。原始CSV保留0–1值。Tiny按原图框面积<256像素²分组，三组及两档推理分辨率共用原始GT/ignore规则。', '',
        '|推理尺寸|条件|CONTROL|ALL_FIVE|VARIABILITY_GUIDED|', '|---|---|---:|---:|---:|']
    for res in (640, 1280):
        for context in contexts:
            cells = [f'{100*ap(a,res,context):.3f} / {100*ap(a,res,context,"tiny"):.3f}' for a in arms]
            lines.append('|'+ '|'.join([str(res), context, *cells])+'|')
    lines += ['', f'![Measured AP comparison]({figure.as_posix()})', '',
        '## 4. 相对CONTROL的配对差异', '',
        '单位为百分点，括号内为95%图像配对Bootstrap区间。200次重采样，seed20261112；与旧两组使用完全相同的548图位置索引，并重新累计COCO AP，不平均单图AP。', '',
        '|尺寸|条件|AP95差 [95%CI]|Tiny AP差 [95%CI]|', '|---|---|---:|---:|']
    def contrast(res, context, size):
        value = next(r for r in comparisons if r['comparator']=='CONTROL' and r['resolution']==res
                     and r['context']==context and r['size']==size and r['metric']=='map50_95')
        low, high = value['difference_CI95']
        return f"{100*value['strategy_minus_comparator']:+.3f} [{100*low:+.3f}, {100*high:+.3f}]"
    for res in (640, 1280):
        for context in contexts:
            lines.append(f'|{res}|{context}|{contrast(res,context,"all")}|{contrast(res,context,"tiny")}|')
    lines += ['', '**主要发现：**', '',
        '- Gaussian sigma15：整体AP95提高1.846/5.114个百分点，Tiny提高0.471/2.541个百分点；对应逐项配对区间均不含0。',
        '- Normal：整体AP95下降0.362/0.625个百分点，逐项区间不含0。Normal Tiny差异区间均含0，不能据点估计宣称clean Tiny改善。',
        '- JPEG Q25：整体AP差异区间均含0，未建立可靠的跨退化迁移收益。JPEG Q75整体AP下降，与clean代价相近。',
        '- 定向策略的八项整体AP95均高于ALL_FIVE；完整对比及区间保留在paired_comparisons.csv。ALL_FIVE在部分AR/大目标指标上的既有收益仍保留，不把“整体AP较低”扩大为所有指标无收益。', '',
        '上述是逐项、条件于单个训练seed的区间，未作多重比较校正，不是跨seed显著性结论。', '',
        '## 5. 退化损失与clean代价', '',
        '退化损失=同策略Normal AP95−退化AP95。以下为百分点；损失变小本身不等于绝对AP变高。', '',
        '|尺寸|条件|CONTROL损失|ALL_FIVE损失|定向损失|', '|---|---|---:|---:|---:|']
    for res in (640, 1280):
        for context in contexts[1:]:
            cells = [100*float(next(x['degradation_loss'] for x in losses if x['arm']==a and int(x['resolution'])==res
                                   and x['condition']==context and x['size']=='all')) for a in arms]
            lines.append(f'|{res}|{context}|'+'|'.join(f'{x:.3f}' for x in cells)+'|')
    lines += ['', '本轮Gaussian收益是实际绝对AP提高，并非只因clean下降而使“保留率”变好；但clean下降仍是真实代价。没有把四个上下文外推为完整empirical Gap。', '',
        '## 6. 实际成本与健康', '',
        '|阶段|worker墙钟秒|最高记录温度|NVML显存峰值MiB|', '|---|---:|---:|---:|',
        f"|定向训练|{training['supervisor']['wall_seconds']:.3f}|{training['maximum_temperature_C']}°C|{training['peak_NVML_used_MiB']}|",
        f"|8项推理、COCO及Bootstrap|{evaluation['supervisor']['wall_seconds']:.3f}|{evaluation['maximum_temperature_C']}°C|{evaluation['peak_NVML_used_MiB']}|", '',
        f"本轮计费worker墙钟共 **{budget['GPU_seconds_this_run']:.3f}秒（{budget['GPU_seconds_this_run']/3600:.3f}小时）**，包含worker内CPU评价和健康检查，不冒充纯CUDA内核时间。原16项评价只读复用，没有重复GPU推理或COCO计算。",
        'CPU准备34.693秒；完整训练轨迹比对约1秒，最终CPU核验另行保存。事前上限为训练5小时+评价3小时+停止余量0.5小时，实际没有延长。',
        f"原36小时总账剩余 **{budget['remaining_seconds']/3600:.3f}小时**；固定截止仍为2026-11-20 21:20:36.593408 UTC。原授权和账本未重置，只追加实际扣费。",
        '最高记录87°C，未达到持续87°C 30秒或90°C硬停止条件；未发生OOM/NaN/Inf或自动降配置。1816次model/EMA更新健康检查通过，梯度范数有限，BN检查和FP32原子保存/重载保护全程保留。不据单次成功宣称设备长期稳定已验证。',
        '14,039条训练/验证/初始化输入引用SHA不变，55项历史保护核验通过；原VM36/VM37结果及旧权重保留。新8项原始预测共4384图记录，GT字节与旧VM37完全一致。', '',
        '## 7. 科学边界与下一步', '',
        '家族选择使用了同一548图开发集的既有损失反馈，虽然概率和范围没有反复调参，仍存在开发选择偏差。这不是独立确认。图像Bootstrap也未控制可靠航次聚类或训练seed变化。',
        '策略以整体AP95最大损失为依据，不是Tiny损失的最优解；例如JPEG Q25对1280 Tiny的损失也很大。本轮不补试概率或组合来追求更好数字。',
        '**结论：该固定Variability引导策略在本开发实验中建立了Gaussian条件的绝对检测收益，并明显优于广泛ALL_FIVE的整体AP；代价是少量clean AP下降。它没有证明全面优势、最优参数、独立泛化或完整RQ4完成。**',
        '后续需要另行授权的多seed确认、独立数据准入及原正式日程检验。本轮已停止；未训练100轮，未触碰test-dev/UAVDT边界，未提交或推送Git。', '',
        '## 8. 文件、SHA与命令', '',
        f"- 执行源码：`{experiment.SOURCE}`，SHA256 `{sha(experiment.SOURCE)}`。",
        f"- 初始化SHA256：`{load(target/'protocol.json')['initialization']['sha256']}`。",
        f"- 第10轮EMA：`{selected['path']}`，SHA256 `{selected['sha256']}`。",
        f"- 原始预测、GT、逐配置指标、200次Bootstrap及输入tensor记录：`{out/'evaluation'}`。",
        f"- 参数/源输入/SHA：`{out/'policy.json'}`、`{out/'scientific_protocol.json'}`、`{out/'execution_freeze.json'}`。",
        f"- 指标与差异：`{out/'configuration_metrics.csv'}`、`{out/'degradation_losses.csv'}`、`{out/'paired_comparisons.csv'}`。",
        f"- 公平性、健康与历史保护：`{out/'full_training_comparability.json'}`、`{out/'augmentation_realization.json'}`、`{out/'final_acceptance.json'}`。",
        f"- 全量训练/安全日志及10份权重：`{target}`。",
        f"- 报告生成源码：`{Path(__file__).resolve()}`，SHA256 `{sha(Path(__file__).resolve())}`。",
        '- 实际训练启动命令（本轮已执行一次，不应再次运行）：', '', '```powershell',
        (out/'execution_command.txt').read_text().strip(), '```', '',
        '- 无GPU的报告核验：', '', '```powershell',
        f"& '{experiment.study.ROOT / '.venv_rq4/Scripts/python.exe'}' -B -X utf8 '{Path(__file__).resolve()}' --run '{out.name}' --verify", '```']
    report.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (out/'final_report.json').write_text(json.dumps({'path':str(report),'sha256':sha(report),
        'generated_snapshot':str(snapshot), 'generated_report_SHA256':old_report['sha256']},indent=2)+'\n', encoding='utf-8')
    artifacts = [report, figure, out/'final_acceptance.json', out/'configuration_metrics.csv',
                 out/'paired_comparisons.json', out/'policy.json', out/'scientific_protocol.json', Path(__file__).resolve()]
    value = {'status':'PASS','reviewed_UTC':datetime.now(timezone.utc).isoformat(),
             'scientific_decision':'GAUSSIAN_AP_GAIN_WITH_CLEAN_TRADEOFF_DEVELOPMENT_ONLY',
             'artifacts':[{'path':str(p),'sha256':sha(p)} for p in artifacts],
             'historical_runs_modified':False,'GPU_tasks_in_report':0}
    record.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')
    (out/'diary.md').write_text((out/'diary.md').read_text(encoding='utf-8')+
        '\nCompleted ten epochs and eight evaluations. Gaussian AP improved at both inference resolutions; clean AP decreased. No parameter search or automatic follow-up. See final_acceptance.json and report_review.json.\n',encoding='utf-8')
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>VM38 comparison</title>'
        '<h1>VM38: Gaussian gain with clean tradeoff</h1><p>Development only; one training seed.</p>'
        '<img src="plots/ap_comparison.png" style="max-width:1100px;width:100%">'
        '<p><a href="configuration_metrics.csv">Metrics</a> | <a href="paired_comparisons.csv">Paired intervals</a> | '
        '<a href="final_acceptance.json">Acceptance</a> | <a href="execution_result.json">Resources</a></p>',encoding='utf-8')
    files = sorted(p for p in out.iterdir() if p.is_file() and p.name!='SHA256SUMS.txt')
    files += sorted(p for p in plots.iterdir() if p.is_file())
    files += sorted(p for p in snapshot.iterdir() if p.is_file())
    (out/'SHA256SUMS.txt').write_text(''.join(sha(p)+'  '+p.relative_to(out).as_posix()+'\n' for p in files),encoding='utf-8')
    from main.visualization import build_run_index
    build_run_index(experiment.RUNS, experiment.EXPERIMENT)
    print('VM38_REPORT_REVIEWED', report)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    review(experiment.run_path(args.run),args.verify)
