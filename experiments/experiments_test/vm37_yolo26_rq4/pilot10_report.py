"""Summarize measured pilot artifacts without rerunning an evaluator."""
import argparse
from collections import Counter, defaultdict
import csv
import json
import hashlib
import statistics
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import study
PARSE_ERRORS = []


def read(path, default=None):
    return study.load(path) if path.exists() else default


def lines(path):
    if not path.exists():
        return []
    rows = []
    for index, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            PARSE_ERRORS.append({'path': str(path), 'line': index, 'error': str(error)})
    return rows


def evaluation_index(directory):
    if (directory / 'result_index.json').exists():
        return read(directory / 'result_index.json')
    stage = read(directory / 'evaluation_stage.json', {})
    rows = []
    for case in stage.get('contexts', []):
        for arm in ('CONTROL', 'ALL_FIVE'):
            for resolution in (640, 1280):
                identity = f'YOLO26s_seed42_{arm}_{case["id"]}_R{resolution}'
                folder = directory / ('e_' + hashlib.sha256(identity.encode()).hexdigest()[:12])
                receipt = read(folder / 'acceptance.json')
                if receipt and receipt['status'] == 'COMPLETE':
                    for name, key in (('metrics.json', 'metrics_SHA256'), ('bootstrap.json', 'bootstrap_SHA256')):
                        if study.sha256(folder / name) != receipt[key]:
                            raise ValueError('A saved completed evaluation changed.')
                    rows.append({'folder': folder.name, 'arm': arm, 'resolution': resolution,
                        'context': case['id'], 'architecture': 'YOLO26s', 'seed': 42, **receipt})
    return rows


def csv_save(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def f(value):
    return 'NOT_EVALUATED' if value is None else f'{value:.6f}'


def execute(identity, authorization_id):
    parent = study.run_path(identity)
    authorization = study.run_path(authorization_id)
    terminal = read(parent / 'execution_result.json')
    if terminal is None:
        raise RuntimeError('The pilot has not reached a saved terminal state.')
    out = study.create('pilot10_results_cpu')
    study.runtime(out, cuda=False)
    sources = {}
    def remember(path):
        if path.exists():
            sources[str(path)] = study.sha256(path)
    metrics, classes, effects, epochs, telemetry, checkpoints, latency, batches = [], [], [], [], [], [], [], []
    summary = []
    for arm in ('CONTROL', 'ALL_FIVE'):
        folder = parent / ('YOLO26s_seed42_' + arm)
        acceptance = read(folder / 'training_acceptance.json')
        completed = lines(folder / 'epoch_progress.jsonl')
        by_epoch = defaultdict(list)
        recorded_batches = lines(folder / 'training_batches.jsonl')
        for row in recorded_batches:
            by_epoch[row['epoch']].append(row['loss'])
            batches.append({'arm': arm, **{k: v for k, v in row.items() if k != 'loss_items'}, **row['loss_items']})
        for row in completed:
            loss = by_epoch.get(row['epoch'], [])
            epochs.append({'arm': arm, **row, 'mean_batch_loss': sum(loss) / len(loss) if loss else None,
                           'loss_aggregation': 'Arithmetic mean of recorded batch loss; not AP.'})
        current_telemetry = lines(folder / 'gpu_telemetry.jsonl')
        telemetry.extend({'arm': arm, **row} for row in current_telemetry)
        maxima = {'temperature_C': max((r['temperature_C'] for r in current_telemetry), default=None),
                  'used_MiB': max((r['used_MiB'] for r in current_telemetry), default=None)}
        summary.append({'arm': arm, 'completed_epochs': acceptance['completed_epochs'] if acceptance else
                        max((r['epoch'] for r in completed), default=0),
                        'accepted_complete': acceptance is not None,
                        'completed_batches': sum(len(r) for r in by_epoch.values()),
                        'anchor_exposures_recorded': sum(len(r['image_ids']) for r in recorded_batches),
                        'optimizer_updates': acceptance['optimizer_updates'] if acceptance else
                            (recorded_batches[-1]['optimizer_updates'] if recorded_batches else None),
                        'EMA_updates': acceptance['EMA_updates'] if acceptance else
                            (recorded_batches[-1]['EMA_updates'] if recorded_batches else None), **maxima})
        for path in sorted((folder / 'weights').glob('epoch*_fp32.pt')):
            checkpoints.append({'arm': arm, 'path': str(path), 'SHA256': study.sha256(path), 'bytes': path.stat().st_size})
        for name in ('protocol.json', 'training_acceptance.json', 'epoch_progress.jsonl', 'health_events.jsonl',
                     'training_batches.jsonl', 'internal_validation.json', 'gpu_telemetry.jsonl', 'worker_result.json', 'failure.json'):
            remember(folder / name)
    evaluations = [parent / 'evaluation'] + sorted(parent.glob('extra_context_*/evaluation'))
    for directory in evaluations:
        index = evaluation_index(directory)
        for row in index:
            folder = directory / row['folder']
            payload = read(folder / 'metrics.json')
            for size in payload['per_area']:
                metrics.append({'arm': row['arm'], 'resolution': row['resolution'], 'context': row['context'],
                    **size, 'source_metrics': str(folder / 'metrics.json'),
                    'checkpoint_SHA256': row['checkpoint_SHA256'], 'predictions_SHA256': row['predictions_SHA256']})
            for category in payload['per_class']:
                classes.append({'arm': row['arm'], 'resolution': row['resolution'], 'context': row['context'], **category})
            trace = read(folder / 'trace.json')
            ms = [r['inference_ms'] for r in trace['per_image']]
            latency.append({'arm': row['arm'], 'resolution': row['resolution'], 'context': row['context'],
                'images': len(ms), 'mean_ms': statistics.mean(ms), 'median_ms': statistics.median(ms),
                'scope': 'Recorded model.predict wall time; source decode and degradation excluded.',
                'peak_allocated_bytes': trace['peak_allocated_bytes'], 'peak_reserved_bytes': trace['peak_reserved_bytes']})
            for name in ('metrics.json', 'acceptance.json', 'bootstrap.json', 'trace.json'):
                remember(folder / name)
        for row in read(directory / 'paired_contrasts.json', []):
            if directory != parent / 'evaluation' and row['context'] == 'Normal':
                continue
            effects.append({**row, 'source': str(directory / 'paired_contrasts.json')})
        remember(directory / 'result_index.json')
        remember(directory / 'paired_contrasts.json')
    csv_save(out / 'metrics.csv', metrics)
    csv_save(out / 'per_class.csv', classes)
    csv_save(out / 'latency.csv', latency)
    csv_save(out / 'paired_differences.csv', effects)
    csv_save(out / 'epoch_loss_and_cost.csv', epochs)
    csv_save(out / 'checkpoint_SHA256.csv', checkpoints)
    engineering = read(parent / 'entrypoint_smoke/training_acceptance.json')
    if engineering:
        checkpoint = engineering['selected_checkpoint']
        if study.sha256(Path(checkpoint['path'])) != checkpoint['sha256']:
            raise ValueError('Engineering checkpoint changed.')
        csv_save(out / 'engineering_checkpoint_SHA256.csv', [{'role': 'ENGINEERING_ONLY_NEVER_INITIALIZE_RESEARCH', **checkpoint}])
        remember(parent / 'entrypoint_smoke/training_acceptance.json')
        remember(parent / 'entrypoint_smoke/health_events.jsonl')
    csv_save(out / 'recorded_batch_loss.csv', batches)
    charges = [study.load(p) for p in sorted((authorization / 'charges').glob('*.json'))]
    stops = [r for r in charges if r.get('supervisor') and r['supervisor'].get('stop_reason')]
    study.save(out / 'supervisor_stops.json', stops, out)
    csv_save(out / 'worker_cost.csv', [{k: v for k, v in r.items() if k != 'supervisor'} for r in charges])
    activation = []
    for arm in ('CONTROL', 'ALL_FIVE'):
        total = 0
        counts = Counter()
        path = parent / ('YOLO26s_seed42_' + arm) / 'augmentation_events.jsonl'
        if path.exists():
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as error:
                        PARSE_ERRORS.append({'path': str(path), 'error': str(error)})
                        continue
                    total += 1
                    for event in row.get('effects', row.get('trace', [])):
                        if event.get('active'):
                            counts[event['family']] += 1
            remember(path)
        activation.append({'arm': arm, 'native_events': total, 'active_family_counts': dict(counts)})
    study.save(out / 'activation_statistics.json', activation, out)
    study.save(out / 'actual_training_summary.json', summary, out)
    for p in (Path(__file__), authorization / 'authorization.json', authorization / 'pilot10_protocol.json',
              authorization / 'execution_freeze.json', parent / 'execution_result.json'):
        remember(p)
    study.save(out / 'source_SHA256.json', sources, out, replace=True)
    study.save(out / 'parse_errors.json', PARSE_ERRORS, out)
    initial_count = len(evaluation_index(parent / 'evaluation'))
    config_count = len({(r['arm'], r['resolution'], r['context']) for r in metrics})
    arm_rows = '\n'.join(f"| {r['arm']} | {r['completed_epochs']} | {r['completed_batches']} | "
                          f"{r['optimizer_updates']} | {r['temperature_C']} | {r['used_MiB']} |" for r in summary)
    primary = [r for r in metrics if r['size'] == 'all']
    table = '\n'.join(f"| {r['context']} | {r['resolution']} | {r['arm']} | {f(r.get('map50'))} | "
                       f"{f(r.get('map50_95'))} | {f(r.get('ar50_95'))} |" for r in primary)
    differences = [r for r in effects if r['size'] in ('all', 'tiny') and r['metric'] == 'map50_95']
    delta_table = '\n'.join(f"| {r['context']} | {r['resolution']} | {r['size']} | "
        f"{f(r.get('ALL_FIVE_minus_CONTROL'))} | {r.get('difference_CI95', 'NOT_ESTIMABLE')} | "
        f"{f(r.get('degradation_loss_change'))} | {r.get('loss_change_CI95', 'NOT_ESTIMABLE')} |" for r in differences)
    plots = []
    if epochs:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        for arm in ('CONTROL', 'ALL_FIVE'):
            selected = [r for r in epochs if r['arm'] == arm and r['mean_batch_loss'] is not None]
            if selected:
                ax.plot([r['epoch'] for r in selected], [r['mean_batch_loss'] for r in selected], marker='o', label=arm)
        ax.set(xlabel='Completed epoch', ylabel='Mean recorded batch loss', title='Single-seed ten-epoch pilot')
        ax.legend(frameon=False)
        fig.savefig(out / 'training_loss.png', dpi=180)
        fig.savefig(out / 'training_loss.pdf')
        plt.close(fig)
        plots.append('training_loss.png')
    elif batches:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        for arm in ('CONTROL', 'ALL_FIVE'):
            selected = [r for r in batches if r['arm'] == arm]
            if selected:
                axes[0].plot([r['position'] + 1 for r in selected], [r['loss'] for r in selected], lw=.6, label=arm)
        axes[0].set(xlabel='Completed forward/backward batch', ylabel='Recorded batch loss', title='Incomplete epoch; no convergence claim')
        axes[0].legend(frameon=False)
        for arm in ('CONTROL', 'ALL_FIVE'):
            selected = [r for r in telemetry if r['arm'] == arm]
            if selected:
                axes[1].plot([r['monotonic_seconds'] / 60 for r in selected], [r['temperature_C'] for r in selected], lw=.7, label=arm)
        axes[1].axhline(90, color='#ac3543', ls='--', label='Hard stop: 90 C')
        axes[1].set(xlabel='Worker elapsed minutes', ylabel='Sampled GPU temperature (C)', title='Recorded temperature protection')
        axes[1].legend(frameon=False)
        fig.savefig(out / 'partial_loss_and_temperature.png', dpi=180)
        fig.savefig(out / 'partial_loss_and_temperature.pdf')
        plt.close(fig)
        plots.append('partial_loss_and_temperature.png')
    report = study.ROOT / 'datasets/reportcn' / ('vm37_yolo26_rq4_10轮配对Pilot结果_' + out.name.rsplit('_', 1)[-1] + '.md')
    if report.exists():
        raise FileExistsError(report)
    new_auth = study.load(authorization / 'authorization.json')
    total_seconds = sum(r['charged_seconds'] for r in charges)
    failure = read(parent / 'failure.json')
    thermal_stop = any(r['supervisor']['stop_reason'] in ('TEMPERATURE_HARD_STOP', 'SUSTAINED_TEMPERATURE_STOP') for r in stops)
    termination_note = ('本次CONTROL温度保护触发于首轮完成之前。顶层worker_result.json缺失是被安全终止后未写出终态造成的二级记录异常，不是停止根因；以监督器和原始温度遥测为准。没有据此宣称模型数值损坏。'
                        if thermal_stop else '终态按真实execution_result与各worker验收记录判定，不以退出码0或部分日志代替完整结果。')
    completion_note = ('CONTROL已记录完整F/B batch和已确认更新计数见上表，停止时另一个开始但未完成的batch不计入完成量。ALL_FIVE未启动，两个研究臂均无完整epoch或研究权重。'
                       if not any(r['completed_epochs'] for r in summary) else '各臂仅将通过完整epoch验收的计数列为完成，未完成的记录保留为部分数据。')
    approval_note = ('作者后来同意重分配子预算并从头重启一次，原36小时总额及7天期限不变；该确认针对预算问题。期间已发生真实温停，因此本次没有使用该重新开始许可，也没有把预算确认解释为散热问题已经解决。授权记录另存，不修改原授权、冻结参数、旧Run或累计已用时间。当前GPU分支停止，不自动重试。'
                     if thermal_stop and (authorization / 'resource_reallocation_author_reply.json').exists() else '没有重置预算、擅自修改训练参数或精确续训。')
    metrics_link = ('[真实指标](' + (out / 'metrics.csv').as_posix() + ')') if metrics else '研究AP/AR为NOT_EVALUATED，未生成空指标文件'
    differences_link = ('[逐条件配对结果](' + (out / 'paired_differences.csv').as_posix() + ')') if effects else '配对差与区间为NOT_EVALUATED'
    checkpoint_link = ('[研究权重SHA](' + (out / 'checkpoint_SHA256.csv').as_posix() + ')') if checkpoints else '无已完成研究epoch的权重'
    if engineering:
        checkpoint_link += '；[工程smoke权重SHA](' + (out / 'engineering_checkpoint_SHA256.csv').as_posix() + ')'
    report.write_text(f'''# YOLO26s十轮配对Pilot结果

状态：{terminal['status']}。实验RunID：{identity}；汇总RunID：{out.name}。

监督器停止记录：{json.dumps([{'stage': r['stage'], 'reason': r['supervisor']['stop_reason'], 'worker_seconds': r['supervisor']['wall_seconds']} for r in stops], ensure_ascii=False)}。

{termination_note}

这是单seed42、10轮、VisDrone development试验，不是导师完整RQ4，不是独立测试。原100轮12臂、五项YOLOv8s消融及84对历史结果未修改。

## 执行和资源

新授权RunID：{authorization_id}。累计worker墙钟{total_seconds:.3f}秒（{total_seconds/3600:.3f}小时），上限36小时；绝对到期{new_auth['absolute_expiry_UTC']}。包含最小工程smoke、训练和评价worker内CPU工作；不是纯GPU kernel计时。未复用旧预算、没有续训或自动调参。

| 组 | 已完成epoch | 实际F/B batch | optimizer更新 | 最高采样温度°C | 最高采样显存MiB |
|---|---:|---:|---:|---:|---:|
{arm_rows}

两组同一健康native10初始化SHA d6de64f001a0657ac04b88340fea9baa5441998f78eba46704542585c034e674，1280、batch2、FP32、workers0、nbs64、AdamW lr0=0.01。真正10轮cosine周期，warmup3轮；原close_mosaic=10保留，所以两组从首轮关闭Mosaic。固定最终epoch10 EMA，不选择中间best。普通增强相同，只有ALL_FIVE增加冻结五类独立0.5门控；它不是已经成立的Gap-guided方法。

完整每臂应有64710次anchor曝光、32360次F/B、1816次optimizer/EMA更新及10份健康FP32快照。未完成时不以部分记录冒充完整训练。

{completion_note}最小smoke的16图/4图、8batch、1次optimizer/EMA更新及健康FP32快照仅为工程证据，不用于研究AP或后续初始化。

## 检测结果

初始16配置实际完成{initial_count}项；连同额外冻结条件共{config_count}项。AP和AR均为0到1原值。只列真实保存并通过索引的评价；无表格内容表示NOT_EVALUATED，不填0。

{('CONTROL与ALL_FIVE的AP50、AP50:95、AR、各尺寸AP、各退化差和配对区间全部NOT_EVALUATED；没有移用4图smoke指标。' if not metrics else '所有数值逐项引用下表所列真实指标文件。')}

| 条件 | 输入 | 组 | AP50 | AP50:95 | AR50:95 |
|---|---:|---|---:|---:|---:|
{table}

原生十类、dense one-to-many、固定NMS、原GT坐标与ignore-union处理保持原评价器。Tiny/Small/Medium/Large及逐类原值见metrics.csv和per_class.csv，不改变GT尺度成员。库内逐epoch验证不混入上述研究口径。latency.csv记录实际predict时间及显存；该时延不包含原图解码和退化生成，不作为端到端部署或实时性声明。

## 配对差异和区间

差值为ALL_FIVE−CONTROL；degradation-loss为Normal AP−退化AP，loss-change为两组loss之差，负值表示退化损失更小。

| 条件 | 输入 | 尺寸 | AP50:95差 | 配对95%区间 | loss-change | loss-change 95%区间 |
|---|---:|---|---:|---|---:|---|
{delta_table}

区间使用冻结200次、相同原图位置索引重新累计COCO AP，不平均单图AP。它们是条件于一个训练seed的点态区间，没有控制可靠航次聚类或给出跨训练seed显著性。NOT_ESTIMABLE原样保留。负面差值和含0区间不删除，不据新AP调整配方。

## Loss与收敛限制

epoch_loss_and_cost.csv记录实际完整epoch、训练/验证/保存墙钟、LR、更新计数及逐batch loss均值。loss曲线只展示已完成epoch，不能证明100轮收敛，不能把增强导致的训练分布差异直接解读为更好或更差检测器。GPU遥测、FP32保存/重载健康事件和逐epoch权重SHA保留在源Run，见checkpoint_SHA256.csv。

## 科学结论与后续

当前可陈述的证据是表内真实对照及区间；未完成配对或评价时，增强收益为NOT_EVALUATED。即使差异为正，也只支持本10轮、seed42、开发集和所测条件。四条件不等于21条件经验Gap；额外条件只扩充实测覆盖，不自动构成完整质量+时延Gap结论。不声称独立泛化、工业标准、SOTA或Gap缩小。

正式YOLO26s/YOLOv8s×三seed×双组×100轮及原五消融仍未被本Pilot完成，需要另行科学决定和资源准入。test-dev和UAVDT边界不变。没有云上传或GitHub推送。

## 异常和追溯

执行异常：{json.dumps(failure, ensure_ascii=False) if failure else '无保存的顶层执行异常；各worker原始日志仍需按源记录查看。'}

{approval_note}

日志解析异常：{json.dumps(PARSE_ERRORS, ensure_ascii=False)}。原文件未删除或改写；损坏行不用于生成已完成epoch结论。

首次CPU准备触发过Ultralytics用户配置schema迁移提示，原记录保留；后续入口在导入前固定私有YOLO_CONFIG_DIR，未安装或升级依赖，没有猜测恢复旧配置。完整说明在授权Run的preparation_note.json。

执行命令：

```powershell
& 'E:/ForStudy/code/uavgap/.venv_rq4/Scripts/python.exe' -B -X utf8 'E:/ForStudy/code/uavgap/main/experiments/experiments_test/vm37_yolo26_rq4/pilot10.py' --run '{authorization_id}' --execute
```

该票据已消耗，不是重跑或续训命令。冻结科学协议、授权、源文件SHA见授权Run的pilot10_protocol.json、authorization.json、execution_freeze.json。输入和评价源SHA见本汇总source_SHA256.json。

[原始实验]({parent.as_posix()})；{metrics_link}；{differences_link}；{checkpoint_link}；[运行成本]({(out/'worker_cost.csv').as_posix()})。
''', encoding='utf-8')
    study.save(out / 'report_reference.json', study.source(report), out)
    study.save(out / 'run.json', {'RunID': out.name, 'ExperimentID': study.EXPERIMENT,
        'status': terminal['status'], 'source_execution_RunID': identity, 'GPU_seconds': 0}, out, replace=True)
    (out / 'diary.md').write_text('# Pilot10 measured results\n\nRead saved training, checkpoint, telemetry and research-evaluation artifacts. No new predictions or metric evaluation. Preserved missing and negative results.\n', encoding='utf-8')
    (out / 'index.html').write_text('<!doctype html><meta charset="utf-8"><title>YOLO26s pilot10</title>'
        '<h1>YOLO26s matched ten-epoch pilot</h1><p>' + terminal['status'] + '</p>' +
        ''.join('<img style="max-width:900px" src="' + name + '">' for name in plots) +
        '<p><a href="metrics.csv">Measured metrics</a> · <a href="paired_differences.csv">Paired differences</a></p>', encoding='utf-8')
    study.seal(out)
    from main.visualization import build_run_index
    build_run_index(study.RUNS, study.EXPERIMENT)
    print(json.dumps({'RunID': out.name, 'report': str(report), 'status': terminal['status'],
        'actual_worker_seconds': total_seconds, 'completed_configurations': config_count}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--authorization', required=True)
    args = parser.parse_args()
    execute(args.run, args.authorization)
