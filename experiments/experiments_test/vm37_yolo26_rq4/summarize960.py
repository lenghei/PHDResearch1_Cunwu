"""Export verified pilot metrics and a Chinese local report. CPU only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pilot960 as pilot

load, save, study = pilot.load, pilot.save, pilot.study


def csv_file(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def export(identity):
    source = study.run_path(identity)
    result = load(source / 'execution_result.json')
    if result['status'] != 'PILOT960_MATCHED_COMPLETE_DEVELOPMENT_ONLY':
        raise ValueError('A complete matched pilot is required; do not fill missing results.')
    index = load(source / 'evaluation/result_index.json')
    if len(index) != 16 or len({row['configuration'] for row in index}) != 16:
        raise ValueError('The evaluation matrix is incomplete or duplicated.')
    out = study.create('pilot960_summary_cpu')
    study.runtime(out, cuda=False)
    measurements, classes, timing, provenance, bootstrap = [], [], [], [], None
    models = {}
    for arm in ('CONTROL', 'ALL_FIVE'):
        accepted = load(source / ('YOLO26s_seed42_' + arm) / 'training_acceptance.json')
        if any(accepted[k] != v for k, v in {'completed_epochs': 10, 'anchor_exposures': 64710,
            'forward_batches': 32360, 'backward_batches': 32360, 'optimizer_updates': 1816, 'EMA_updates': 1816}.items()):
            raise ValueError('Training counts differ from the frozen pilot.')
        if study.sha256(Path(accepted['selected_checkpoint']['path'])) != accepted['selected_checkpoint']['sha256']:
            raise ValueError('Final checkpoint changed.')
        models[arm] = accepted
    if models['CONTROL']['initialization'] != models['ALL_FIVE']['initialization']:
        raise ValueError('Initial checkpoints differ.')
    for row in index:
        folder = source / 'evaluation' / row['folder']
        for name, key in (('metrics.json', 'metrics_SHA256'), ('bootstrap.json', 'bootstrap_SHA256'),
                           ('predictions.jsonl.gz', 'predictions_SHA256')):
            if study.sha256(folder / name) != row[key]:
                raise ValueError('Result checksum mismatch: ' + str(folder / name))
        metrics, boot, trace = (load(folder / name) for name in ('metrics.json', 'bootstrap.json', 'trace.json'))
        signature = {k: boot['draws'][k] for k in ('image_ids', 'samples', 'seed', 'sampling')}
        if bootstrap is None:
            bootstrap = signature
        if signature != bootstrap or signature['samples'] != 200 or len(signature['image_ids']) != 548 or metrics['images'] != 548:
            raise ValueError('Bootstrap pairing or image count differs.')
        if row['checkpoint_SHA256'] != models[row['arm']]['selected_checkpoint']['sha256']:
            raise ValueError('Evaluation used a different checkpoint.')
        for area in metrics['per_area']:
            for values in boot['draws']['values'][area['size']].values():
                if len(values) != 200:
                    raise ValueError('Incomplete bootstrap draws.')
            measurements.append({'configuration': row['configuration'], 'arm': row['arm'],
                'resolution': row['resolution'], 'condition': row['context'], **area,
                'source_metrics': str(folder / 'metrics.json'), 'source_SHA256': row['metrics_SHA256']})
        for area in metrics['per_class']:
            classes.append({'configuration': row['configuration'], 'arm': row['arm'],
                'resolution': row['resolution'], 'condition': row['context'], **area})
        timing.append({'configuration': row['configuration'], 'arm': row['arm'], 'resolution': row['resolution'],
            'condition': row['context'], 'mean_prediction_ms': statistics.mean(x['inference_ms'] for x in trace['per_image']),
            'raw_predictions': metrics['raw_count'], 'mapped_predictions': metrics['mapped_count'],
            **{k: v for k, v in trace.items() if k != 'per_image'}})
        provenance.append({'configuration': row['configuration'], 'source_folder': str(folder), **row})
    expected = {(a, r, c) for a in models for r in (640, 1280) for c in {v['context'] for v in index}}
    if len(expected) != 16 or {(v['arm'], v['resolution'], v['context']) for v in index} != expected:
        raise ValueError('Unmatched evaluation factors.')
    effects = load(source / 'evaluation/paired_contrasts.json')
    csv_file(out / 'configuration_metrics.csv', measurements)
    csv_file(out / 'per_class_metrics.csv', classes)
    csv_file(out / 'timing.csv', timing)
    csv_file(out / 'paired_differences.csv', effects)
    csv_file(out / 'results_provenance.csv', provenance)
    overall = [v for v in measurements if v['size'] == 'all']
    losses = []
    for row in overall:
        normal = next(v for v in overall if v['arm'] == row['arm'] and v['resolution'] == row['resolution'] and v['condition'] == 'Normal')
        losses.append({'arm': row['arm'], 'resolution': row['resolution'], 'condition': row['condition'],
            'Normal_AP50': normal['map50'], 'condition_AP50': row['map50'],
            'AP50_loss': normal['map50'] - row['map50'], 'Normal_AP50_95': normal['map50_95'],
            'condition_AP50_95': row['map50_95'], 'AP50_95_loss': normal['map50_95'] - row['map50_95'],
            'source_metrics': row['source_metrics'], 'source_Normal_metrics': normal['source_metrics']})
    csv_file(out / 'degradation_losses.csv', losses)
    robust = []
    for arm in models:
        for resolution in (640, 1280):
            rows = [v for v in overall if v['arm'] == arm and v['resolution'] == resolution]
            clean = next(v for v in rows if v['condition'] == 'Normal')['map50_95']
            mean = statistics.mean(v['map50_95'] for v in rows if v['condition'] != 'Normal')
            robust.append({'arm': arm, 'resolution': resolution, 'clean_AP50_95': clean,
                'mean_three_corrupted_AP50_95': mean, 'retention': mean / clean if clean else None,
                'definition': 'Equal mean over JPEG Q25/Q75 and Gaussian sigma15; not AP8 or full Gap.'})
    csv_file(out / 'robustness_summary.csv', robust)
    resources = []
    for stage in result['stages']:
        supervisor = stage.get('safety_supervisor.json', {})
        resources.append({'stage': stage['stage'], 'worker_wall_seconds': supervisor.get('wall_seconds'),
            'max_recorded_temperature_C': stage.get('maximum_temperature_C'),
            'peak_NVML_used_MiB': stage.get('peak_NVML_used_MiB'), 'stop_reason': supervisor.get('stop_reason')})
    csv_file(out / 'resource_summary.csv', resources)
    charges = [load(p) for p in pilot.old.charge_files(study.run_path(pilot.recovery.AUTH)) if load(p).get('RunID') == identity]
    charged = sum(v['charged_seconds'] for v in charges)
    def percent(x):
        return 'NA' if x is None else f'{100 * x:.3f}'
    text = ['# VM37 YOLO26s 960配对实验最终报告', '',
        '**完成：两臂各10轮、16项548图评价、每项200次配对图像Bootstrap。**', '',
        '执行RunID：`' + identity + '`；CPU汇总RunID：`' + out.name + '`。',
        '固定seed42，同一健康初始化、6471图训练池及548图VisDrone2019-DET-val开发池。',
        '共同初始化SHA256：`' + models['CONTROL']['initialization']['sha256'] + '`。',
        '每臂64710次anchor曝光、32360次forward/backward、1816次optimizer/EMA更新均通过原训练验收。', '',
        '独立单轮成本检查不计入两臂训练，其权重未用于任何一臂初始化；CONTROL与ALL_FIVE均从同一原始健康初始化重新开始。', '',
        '## AP结果', '',
        '下表AP均为百分数；AP50仅使用IoU=0.50，AP50:95平均IoU=0.50:0.05:0.95。',
        '原生十类、ignore-union COCO、maxDet=3000；与库内逐epoch验证分开。', '',
        '|推理输入|条件|策略|AP50|AP50:95|Tiny AP|Small AP|Medium AP|Large AP|AR50:95|',
        '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for resolution in (640, 1280):
        for condition in dict.fromkeys(v['context'] for v in index):
            for arm in models:
                rows = {v['size']: v for v in measurements if v['arm'] == arm and v['resolution'] == resolution and v['condition'] == condition}
                values = [rows['all']['map50'], rows['all']['map50_95'], *[rows[s]['map50_95'] for s in ('tiny', 'small', 'medium', 'large')], rows['all']['ar50_95']]
                text.append('|' + '|'.join([str(resolution), condition, arm, *map(percent, values)]) + '|')
    text += ['', 'Tiny/Small/Medium/Large使用原图bbox面积，区间依次为<256、[256,1024)、[1024,9216)、≥9216像素²。Small是16²至32²的互斥分层；标准COCO small另存为coco_small。', '',
        '## 配对差异与退化损失', '',
        '差值=ALL_FIVE−CONTROL，单位为百分点。退化损失变化=(Normal−退化)ALL_FIVE−(Normal−退化)CONTROL，负值表示损失缩小；仍须同时看绝对退化AP。', '',
        '|推理输入|条件|AP50:95差值|95%配对区间|Tiny AP差值|CONTROL损失|ALL_FIVE损失|损失变化|其95%区间|',
        '|---|---|---:|---|---:|---:|---:|---:|---|']
    for resolution in (640, 1280):
        for condition in dict.fromkeys(v['context'] for v in index):
            rows = [v for v in effects if v['resolution'] == resolution and v['context'] == condition and v['metric'] == 'map50_95']
            all_row = next(v for v in rows if v['size'] == 'all')
            tiny = next(v for v in rows if v['size'] == 'tiny')
            if all_row['status'] != 'ESTIMATED':
                text.append(f'|{resolution}|{condition}|NOT_ESTIMABLE|||||||')
                continue
            interval = lambda key: '[' + ', '.join(percent(x) for x in all_row[key]) + ']'
            per_arm = {v['arm']: v['AP50_95_loss'] for v in losses if v['resolution'] == resolution and v['condition'] == condition}
            text.append(f"|{resolution}|{condition}|{percent(all_row['ALL_FIVE_minus_CONTROL'])}|{interval('difference_CI95')}|{percent(tiny.get('ALL_FIVE_minus_CONTROL'))}|{percent(per_arm['CONTROL'])}|{percent(per_arm['ALL_FIVE'])}|{percent(all_row['degradation_loss_change'])}|{interval('loss_change_CI95')}|")
    text += ['', '区间使用冻结的200次同图索引重采样，重新累计COCO AP；不是单图AP平均，也不是训练seed间区间。点态区间不代表全部比较的联合显著性。', '',
        '## 绝对退化AP与保留率', '', '|推理输入|策略|Normal AP50:95|三退化平均AP50:95|保留率%|', '|---|---|---:|---:|---:|']
    for row in robust:
        text.append(f"|{row['resolution']}|{row['arm']}|{percent(row['clean_AP50_95'])}|{percent(row['mean_three_corrupted_AP50_95'])}|{percent(row['retention'])}|")
    text += ['', '三退化平均仅包含JPEG Q25、JPEG Q75及Gaussian sigma15；保留率=该均值/Normal AP。此汇总不等于完整empirical Gap。', '', '## 本次观察', '']
    for resolution in (640, 1280):
        rows = [v for v in effects if v['resolution'] == resolution and v['size'] == 'all' and v['metric'] == 'map50_95' and v['status'] == 'ESTIMATED']
        positive = sum(v['ALL_FIVE_minus_CONTROL'] > 0 for v in rows)
        above = sum(v['difference_CI95'][0] > 0 for v in rows)
        below = sum(v['difference_CI95'][1] < 0 for v in rows)
        text.append(f'{resolution}推理：四个预设条件中，AP50:95点估计提高{positive}项；点态95%区间完全高于0为{above}项，完全低于0为{below}项，其余包含0。')
        text.append('各条件差值（百分点）：' + '；'.join(v['context'] + ' ' + percent(v['ALL_FIVE_minus_CONTROL']) for v in rows) + '。')
        tiny = [v for v in effects if v['resolution'] == resolution and v['size'] == 'tiny' and v['metric'] == 'map50_95' and v['status'] == 'ESTIMATED']
        text.append('Tiny AP差值（百分点）：' + '；'.join(v['context'] + ' ' + percent(v['ALL_FIVE_minus_CONTROL']) for v in tiny) + '。')
    primary = [v for v in effects if v['size'] == 'all' and v['metric'] == 'map50_95' and v['status'] == 'ESTIMATED']
    if all(v['ALL_FIVE_minus_CONTROL'] <= 0 for v in primary):
        text.append('本次Pilot不支持ALL_FIVE提高这些预设条件下的绝对AP50:95；相对保留率或较小退化损失不能替代绝对精度。')
    elif all(v['ALL_FIVE_minus_CONTROL'] > 0 for v in primary):
        text.append('本次四条件、两档推理尺寸的绝对AP50:95点估计均提高；幅度和不确定性见上表，不能直接推广到正式100轮、多seed或独立数据。')
    else:
        text.append('结果存在条件或推理尺寸依赖的收益与代价；不作ALL_FIVE普遍提高检测或鲁棒性的结论。')
    text += ['',
        '## 实际资源与模型', '', f'本执行Run计费worker墙钟合计{charged:.3f}秒（{charged/3600:.3f}小时）；包括初始化、训练、内部验证、Health Gate、保存及科学评价/Bootstrap，不是GPU纯kernel时间。', '',
        '|阶段|worker墙钟秒|最高记录°C|NVML显存峰值MiB|', '|---|---:|---:|---:|']
    for row in resources:
        text.append(f"|{row['stage']}|{row['worker_wall_seconds']:.3f}|{row['max_recorded_temperature_C']}|{row['peak_NVML_used_MiB']}|")
    text += ['', '推理延迟逐配置见timing.csv：来自实际per-image记录，包含predict调用，不含原图解码/退化生成；配置wall time包含这些步骤。CUDA allocated/reserved峰值另存，不能与NVML设备占用混作同一指标。', '',
             '两臂epoch10 EMA来源权重：']
    for arm, model in models.items():
        text.append('- ' + arm + '：`' + model['selected_checkpoint']['sha256'] + '`；' + model['selected_checkpoint']['path'])
    text += ['', '## 解释范围与下一步', '',
        '这是单seed、960训练、10轮的开发集探索性配对。CONTROL保留相同常规增强；ALL_FIVE仅额外应用冻结的五类随机扰动。close_mosaic=10令本Pilot从epoch0关闭Mosaic，不能直接代表原1280/100轮方案。',
        '以上保留全部正负差异；仅相对保留率提高不能证明绝对检测性能更好。Tiny结果必须按条件和推理尺寸解释。',
        'ALL_FIVE是广义随机增强，不是已证明的Gap-guided策略。没有独立测试、跨训练seed显著性或完整Gap缩减结论；未控制真实航次聚类相关性。',
        '下一步应依据上述绝对AP、干净精度代价及配对区间，决定是否投入已冻结的正式多seed方案；本次不自动追加训练、调参、test-dev/UAVDT评价或云上传。原正式1280/100轮协议未改，原36小时额度和2026-11-20 21:20:36 UTC截止未重置。', '',
        '## 原始证据', '',
        '- 执行与授权：' + str(source),
        '- 原始预测、GT、COCO指标与Bootstrap：' + str(source / 'evaluation'),
        '- 汇总、逐类AP、配对区间、时延、资源及来源SHA：' + str(out),
        '- 数字来源索引：results_provenance.csv；未复制原图、原始预测或模型。',
        '- 汇总代码：' + str(Path(__file__).resolve())]
    report = study.ROOT / 'datasets/reportcn' / ('vm37_yolo26_rq4_960配对最终报告_' + identity[-8:] + '.md')
    with report.open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(text) + '\n')
    save(out / 'inputs.json', {'execution_RunID': identity, 'execution_result': study.source(source / 'execution_result.json'),
        'result_index': study.source(source / 'evaluation/result_index.json'), 'script': study.source(Path(__file__)),
        'ground_truth': study.source(source / 'evaluation/ground_truth.json'),
        'initialization': models['CONTROL']['initialization'],
        'paired_contrasts': study.source(source / 'evaluation/paired_contrasts.json')})
    save(out / 'summary_result.json', {'status': 'VERIFIED_SUMMARY_COMPLETE', 'configurations': 16,
        'images_per_configuration': 548, 'inference_image_records': 8768, 'bootstrap_samples': 200,
        'new_GPU_work': False, 'report': str(report), 'report_SHA256': study.sha256(report), 'charged_execution_seconds': charged})
    meta = load(out / 'run.json'); meta['status'] = 'VERIFIED_SUMMARY_COMPLETE'
    study.save(out / 'run.json', meta, out, replace=True)
    (out / 'diary.md').write_text('# Verified pilot summary\n\nReused accepted evaluation artifacts. Verified checksums, complete matrix, image pairing and training counters. Exported original metrics, fixed paired contrasts and descriptive three-condition retention. No AP recomputation, GPU work or new bootstrap.\n', encoding='utf-8')
    (out / 'index.html').write_text('<!doctype html><h1>Verified 960 matched pilot</h1><a href="configuration_metrics.csv">Metrics</a> | <a href="paired_differences.csv">Paired contrasts</a> | <a href="results_provenance.csv">Sources</a>', encoding='utf-8')
    study.seal(out)
    print(report)
    print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    export(parser.parse_args().run)
