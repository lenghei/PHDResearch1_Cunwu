"""Create a source-linked delivery report for CPU-accepted portable v3.1."""
import argparse
import csv
import json
from pathlib import Path
import sys

POST = Path(__file__).resolve().parent
sys.path.insert(0, str(POST.parent))
from study import *
import portable


def execute(validation_id, export_id):
    out = create('portable_v3_delivery_cpu')
    runtime(out, cuda=False)
    validation = run_path(validation_id)
    export = run_path(export_id)
    check = load(validation / 'verification.json')
    package = load(export / 'export_result.json')
    if check['status'] != 'CPU_VERIFIED_GPU_ORCHESTRATOR_NOT_RUN' or package['bundle_manifest_SHA256'] != check['bundle_manifest_SHA256']:
        raise ValueError('Package and CPU acceptance do not match.')
    bundle = Path(check['bundle'])
    manifest = portable.verify_bundle(bundle)
    if manifest['execution_wrapper_version'] != 'VM37_PORTABLE_V3.1':
        raise ValueError('Unexpected portable version.')
    if sha256(Path(package['ZIP']['path'])) != package['ZIP']['sha256']:
        raise ValueError('The exported package changed.')
    comparison = Path(check['comparison_path'])
    with (comparison / 'performance.csv').open(encoding='utf-8') as stream:
        performance = list(csv.DictReader(stream))
    with (comparison / 'timing.csv').open(encoding='utf-8') as stream:
        times = list(csv.DictReader(stream))
    receipt = load(comparison / 'CPU_receipt.json')
    environment = load(bundle / 'SERVER_ENVIRONMENT.json')
    boundaries = load(bundle / 'SCIENTIFIC_BOUNDARIES.json')
    old_pairs = load(PREP / 'FROZEN_DEVELOPMENT_EVALUATION.json')['conditions']
    ledger = load(RUNS / 'compute_allocation.json')
    formal = load(bundle / 'formal100_protocol.json')
    exact = {'package_version': manifest['execution_wrapper_version'], 'ZIP': package['ZIP'],
        'bundle_manifest_SHA256': check['bundle_manifest_SHA256'], 'CPU_verification': source(validation / 'verification.json'),
        'CPU_receipt': source(comparison / 'CPU_receipt.json'), 'original_accepted_pairs_unchanged': len(old_pairs),
        'formal_arms': formal['full_matrix_training_arms'], 'formal_epochs_per_arm': 100,
        'research_evaluation_configurations': 12 * 2 * 21, 'new_GPU_seconds': 0,
        'server_benchmark': 'PREPARED_NOT_RUN', 'formal_training': 'NOT_RUN',
        'AP_or_gap_improvement': 'NOT_EVALUATED', 'scientific_boundaries': boundaries,
        'local_authorization_deadline_UTC_unchanged': ledger['absolute_deadline_UTC']}
    save(out / 'delivery.json', exact, out)
    save(out / 'inputs.json', {'verification': source(validation / 'SHA256SUMS.txt'),
        'export': source(export / 'SHA256SUMS.txt'), 'CPU_performance': source(comparison / 'performance.csv'),
        'CPU_timing': source(comparison / 'timing.csv'), 'environment': source(bundle / 'SERVER_ENVIRONMENT.json'),
        'original_RQ4_design': source(bundle / 'formal100_protocol.json'),
        'original84_source': source(PREP / 'FROZEN_DEVELOPMENT_EVALUATION.json'), 'report_generator': source(Path(__file__))}, out)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    labels = [row['architecture'] + '\n' + row['arm'] for row in performance]
    values = [float(row['speedup']) for row in performance]
    axes[0].bar(labels, values, color=['#3a7f72' if value >= 1 else '#b06d52' for value in values])
    axes[0].axhline(1, color='#475569', lw=1, ls='--')
    axes[0].set(ylabel='Baseline input time / optimized input time', title='CPU replay; all seeds and states retained')
    x = np.arange(3)
    for index, (name, color) in enumerate((('baseline', '#6b7280'), ('optimized', '#368574'))):
        total = [sum(float(row['input_seconds']) for row in times if row['pipeline'] == name and int(row['epoch']) == epoch)
                 for epoch in (0, 1, 90)]
        axes[1].bar(x + (index - .5) * .35, total, width=.35, label=name, color=color)
    axes[1].set(xticks=x, xticklabels=['Epoch 0', 'Epoch 1', 'Epoch 90\nMosaic closed'], ylabel='Summed CPU input seconds',
                title='Fixed probe states; not a 100-epoch forecast')
    axes[1].legend(frameon=False)
    fig.savefig(out / 'portable_v3_cpu_performance.png', dpi=180)
    fig.savefig(out / 'portable_v3_cpu_performance.pdf')
    plt.close(fig)
    table = '\n'.join(f"| {row['architecture']} | {row['arm']} | {float(row['baseline_seconds']):.3f} | "
                      f"{float(row['optimized_seconds']):.3f} | {float(row['speedup']):.3f} |"
                      for row in performance)
    e = environment['reference']['packages']
    data_bytes = sum(row['image_bytes'] + row['annotation_bytes'] for row in environment['dataset_storage'].values())
    report = ROOT / 'datasets/reportcn' / ('vm37_yolo26_rq4_优化便携训练包V3_1_' + out.name.rsplit('_', 1)[-1] + '.md')
    content = f'''# VM37优化便携训练包v3.1

## 实际改动

新入口portable_v3.py将已验收的ArithmeticPrefetchDataset接入原训练器。两组使用同一输入实现，默认optimized，保留baseline选项及未修改的v2入口。模型结构、loss、学习率、累积、EMA、FP32健康检查、GT和评价器保持原实现。workers仍为0。

baseline指旧输入实现；CONTROL指没有额外五类增强的研究对照组，两者不是同一概念。

补充了外置结果目录的受限写入适配、整套预算扣除预检耗时、单次票据、v3评价授权适配和两个服务器成本测量方案。旧报告、RunID、策略和权重不覆盖。初版发现外置目录的旧路径保护问题，失败记录保留；没有更改数值结果使验收通过。

## CPU验收证据

- {check['unit_tests']['tests']}项CPU测试通过。
- 两个架构×三个seed×CONTROL/ALL_FIVE，覆盖epoch0、1、90关闭Mosaic状态。
- 每状态8批、每批2个anchor，两次交替重复；图像、标签、类别、参数、增强开关、Mosaic辅助ID、缓存顺序、occurrence与CPU RNG状态完全一致。
- 六份native10初始化健康、FP32和类别顺序通过；150项已保存评价像素核对通过。没有重新运行84对检测评价。
- 五项v3预演、四项默认拒绝和原v2入口通过。全部6471/548原图、标注和派生标签在验收后重新核对SHA。
- CUDA RNG、v3整轮训练及Linux运行未在本轮验证；GPU=0。

| 模型 | 组 | 原输入秒数 | 优化输入秒数 | 比值 |
|---|---|---:|---:|---:|
{table}

表中是三seed、三个探测状态和两次重复的CPU输入时间汇总，不包含模型训练，三个状态等权且不代表100轮中的真实占比。负面或无明显提速的单元均保留在timing.csv，不能据此宣称整轮加速。采样进程RSS峰值{receipt['sampled_peak_RSS_bytes']/1024**2:.1f}MiB。

这里的架构名仅标识训练参数配置组，没有执行检测模型前向；组间CPU时间差不能归因于架构本身。

历史128批GPU位级回归仍可引用9f633b8c报告，但它不等于本次新编排器已通过整轮GPU验收。没有新增AP、鲁棒性或Gap改善结果。

## 便携包与SHA

版本：**{manifest['execution_wrapper_version']}**。

ZIP SHA256：`{package['ZIP']['sha256']}`。

包清单SHA256：`{check['bundle_manifest_SHA256']}`。

[下载本地资料包]({Path(package['ZIP']['path']).as_posix()})。

包中包含代码、原UVL、冻结协议、六份初始化、派生标签、GT和评价像素清单；没有复制原始图像或145GiB研究档案。该包为本地私有资料，不自动上传GitHub或服务器。

## 环境与资源

参考环境Python3.11.9、Ultralytics {e['ultralytics']}、PyTorch {e['torch']}、torchvision {e['torchvision']}、CUDA runtime12.4、NumPy {e['numpy']}、OpenCV {e['opencv-python']}。完整版本与官方源码SHA在包中锁定。新服务器使用独立环境，不改变本地.venv_rq4。

规划建议8个独占逻辑CPU线程、32GiB RAM、16GiB GPU显存；这不是已测最低要求。保留native线程2、Torch线程2、OpenCV线程2和DataLoader workers0。本地历史整轮最高采样显存7935MiB，不能拿短测5533MiB代替完整训练需求。

所需train/val图像与原标注实测共{data_bytes:,}字节，约{data_bytes/1024**3:.3f}GiB。YOLO26s一份已测FP32训练checkpoint约160.8MB，六臂各保留100份的存储情景约{environment['YOLO26s_six_arms_100_checkpoints_storage_scenario_bytes']/1024**3:.2f}GiB；YOLOv8s、504项原预测、Bootstrap和环境空间必须另计，当前不能给出全矩阵可靠存储或时间总额。

Linux软件轮子、驱动和目标GPU尚未实测。先验证版本、官方源码、六初始化和150黄金像素；不匹配时停止，不能静默更换JPEG编码器。历史GPU授权截止仍为{ledger['absolute_deadline_UTC']}，本轮未使用或重置。服务器必须另有明确、有限的新授权。

当前监控器要求只暴露一个NVIDIA设备，多卡服务器应由容器或调度器分配并隔离单卡，以实际预检设备清单为准。本包没有DDP或多卡并行训练。

## 运行命令

解压后在包根目录运行。以下是服务器示例路径，不是已执行的远程任务。先运行CPU验收与预演：

```sh
python -B main/experiments/vm37_yolo26_rq4/post_cost/test_v3.py --output /work/results/v3_unit
python -B main/experiments/vm37_yolo26_rq4/post_cost/compare_v3.py --bundle . --data-root /data/VisDrone --output-root /work/results/vm37
python -B main/experiments/vm37_yolo26_rq4/post_cost/portable_v3.py --bundle . --data-root /data/VisDrone --protocol benchmark26 --output-root /work/results/vm37
python -B main/experiments/vm37_yolo26_rq4/post_cost/portable_v3.py --bundle . --data-root /data/VisDrone --protocol formal100 --output-root /work/results/vm37
```

首轮成本方案：YOLO26s、seed42，两组都从同一初始化开始；6471/548全量、100轮调度、仅完成epoch1工程测量。候选上限每臂7200秒、总计14400秒，不是已获授权。YOLOv8s对应benchmark8，不能套用YOLO26s速度。

只有另获作者授权、目标机器CPU receipt及预算准入后，才可执行：

```sh
python -B main/experiments/vm37_yolo26_rq4/post_cost/portable_v3.py --bundle . --data-root /data/VisDrone --protocol benchmark26 --output-root /work/results/vm37 --cpu-receipt /work/results/vm37/CPU_RUN/CPU_receipt.json --authorization /work/author_approved_benchmark26.json
python -B main/experiments/vm37_yolo26_rq4/post_cost/portable_v3.py --bundle . --data-root /data/VisDrone --protocol formal100 --output-root /work/results/vm37 --cpu-receipt /work/results/vm37/CPU_RUN/CPU_receipt.json --authorization /work/author_approved_formal100.json
```

模板approved=false，默认拒绝。正式方案仍为两个架构×三seed×两臂×100轮，共12臂；固定epoch100 EMA和504项研究评价，完整训练与评价必须同时准入。五项原YOLOv8s消融另外保留，不混入12臂预算。失败保留证据，不自动重试或精确续训，不修改batch/分辨率/健康检查。

## RQ1—RQ4剩余科学工作

| 问题 | 本轮保持的证据 | 尚未完成 |
|---|---|---|
| RQ1 可变性建模 | 原FM_P、FM_D及参数SHA不变 | 条件强度的真实场景校准与约束依据仍需科学证据 |
| RQ2 跨域关系 | 既有{len(old_pairs)}对已验收测量不变 | 更广配置覆盖、分组留出和独立确认；本轮未产生新关系 |
| RQ3 静态/经验Gap | 保留阈值、同任务评价及Bootstrap定义 | 继续用真实测量验证覆盖与不确定性；输入提速不等于Gap缩小 |
| RQ4 缩小Gap | 正式12臂和五消融方案保留 | 完整配对训练、同条件再评价及Gap变化尚未完成 |

ALL_FIVE仍是五个独立0.5门控的广泛随机增强，不是已证明的Gap-guided策略；haze仍为未校准常深度近似。test-dev保持DATASET_BLOCKED，UAVDT保持BLOCKED_ANNOTATION_PROVENANCE，不替换数据集、不声称独立验证。

## 文件与复现

验收RunID：{validation_id}。交付RunID：{out.name}。

[CPU验收]({(validation/'verification.json').as_posix()})；[逐批CPU计时]({(comparison/'timing.csv').as_posix()})；[性能汇总]({(comparison/'performance.csv').as_posix()})；[可视化]({(out/'index.html').as_posix()})。

包根README给出环境安装及全部入口。每项实测来自所列JSON/CSV，包内协议和所有SHA均可复核。当前结论为CPU工程集成通过；新编排器的服务器整轮GPU验收、100轮研究效果仍未完成。
'''
    write_atomic(report, content, ROOT)
    save(out / 'report_provenance.json', {'report': source(report), 'package': package['ZIP'],
        'performance': source(comparison / 'performance.csv'), 'verification': source(validation / 'verification.json')}, out)
    write_atomic(out / 'index.html', '<!doctype html><meta charset="utf-8"><title>VM37 portable v3.1</title>'
        '<style>body{max-width:1100px;margin:2rem auto;font:17px system-ui;color:#193749}img{width:100%}</style>'
        '<h1>VM37 portable v3.1</h1><p>CPU integration accepted. No GPU work in this stage.</p>'
        '<img src="portable_v3_cpu_performance.png" alt="Source-driven CPU input comparison">'
        '<p>No AP or empirical-gap improvement is claimed.</p><a href="delivery.json">Version and evidence</a>', out)
    write_atomic(out / 'diary.md', '# Portable v3.1 delivery\n\nSummarized actual CPU acceptance, immutable package SHA, '
        'server instructions and remaining scientific work. No new GPU allocation or training.\n', out)
    save(out / 'run.json', {'RunID': out.name, 'ExperimentID': EXPERIMENT,
        'status': 'CPU_INTEGRATION_ACCEPTED_GPU_ORCHESTRATOR_NOT_VALIDATED', 'GPU_seconds': 0}, out, replace=True)
    seal(out)
    from main.visualization import build_run_index
    build_run_index(RUNS, EXPERIMENT)
    print(json.dumps({'RunID': out.name, 'report': str(report), 'ZIP_SHA256': package['ZIP']['sha256'],
        'CPU_tests': check['unit_tests']['tests'], 'GPU_seconds': 0}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--validation-run', required=True)
    parser.add_argument('--export-run', required=True)
    args = parser.parse_args()
    execute(args.validation_run, args.export_run)
