"""Summarize measured input speed, numerical regression and resource limits."""
import argparse
import csv
import html
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

POST = Path(__file__).resolve().parent
sys.path.insert(0, str(POST.parent))
from study import *
from safety import available_seconds

PROFILE = 'vm37_yolo26_rq4_input_profile_cpu_20261113T094531_12ba5298'
THREADS = 'vm37_yolo26_rq4_input_optimization_cpu_20261113T095416_96e9c63b'
WORKERS = 'vm37_yolo26_rq4_loader_workers_cpu_20261113T095722_1b8b66cd'
CONTROL = 'vm37_yolo26_rq4_full_epoch_yolo26s_control_20261111T232439_301fb25d'
FULL = 'vm37_yolo26_rq4_all_five_full_epoch_cost_20261112T200430_2ab983f6'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def execute(input_id, unit_id, regression_id, profile_id=PROFILE):
    global PROFILE
    PROFILE = profile_id
    out = create('input_optimization_summary_cpu')
    runtime(out, cuda=False)
    started = time.perf_counter()
    measured = run_path(input_id)
    units = run_path(unit_id)
    regression = run_path(regression_id)
    for folder, names in ((run_path(PROFILE), ('CPU_PROFILE.csv', 'OPERATOR_TIMINGS.csv', 'profile_result.json')),
                          (measured, ('input_acceptance.json', 'throughput.csv', 'inputs.json')),
                          (units, ('test_results.json',)), (regression, ('regression_result.json',))):
        verify_hashes(folder, names)
    result = load(measured / 'input_acceptance.json')
    gpu = load(regression / 'regression_result.json')
    numerical = load(regression / 'numerical_comparison.json') if (regression / 'numerical_comparison.json').exists() else None
    prior_gpu = run_path('vm37_yolo26_rq4_optimized_input_regression_20261113T100530_a0942530')
    prior_result = load(prior_gpu / 'regression_result.json')
    tensors = [json.loads(line) for line in (regression / 'tensor_checks.jsonl').read_text().splitlines()] if (regression / 'tensor_checks.jsonl').exists() else []
    normalization_difference = max((row['CPU_GPU_normalization_max_abs_difference'] for row in tensors), default=None)
    workers = load(run_path(WORKERS) / 'worker_admission.json')
    control = load(run_path(CONTROL) / 'training_acceptance.json')
    full = load(run_path(FULL) / 'training_acceptance.json')
    ledger = load(RUNS / 'compute_allocation.json')
    available = available_seconds(ledger)
    ratio = result['optimized_seconds'] / result['original_seconds']
    original_scope = full['training_setup_epoch_internal_val_save_seconds']
    optimized_scenario = original_scope - full['loader_seconds'] + full['loader_seconds'] * ratio
    pilot_scenario = 20 * (control['training_setup_epoch_internal_val_save_seconds'] + optimized_scenario)
    budget = {'status': 'PILOT_RESOURCE_BLOCKED', 'calculated_UTC': utc().isoformat(),
              'original_allocation': source(RUNS / 'compute_allocation.json'),
              'absolute_deadline_UTC': ledger['absolute_deadline_UTC'], 'available_seconds': available,
              'charged_worker_seconds': sum(row['charged_seconds'] for row in ledger['completed_workers']),
              'CPU_input_ratio': ratio, 'CONTROL_measured_epoch_seconds': control['training_setup_epoch_internal_val_save_seconds'],
              'ALL_FIVE_original_measured_epoch_seconds': original_scope,
              'ALL_FIVE_optimized_epoch_component_scenario_seconds': optimized_scenario,
              'paired20_component_scenario_seconds': pilot_scenario,
              'paired20_plus25percent_scenario_seconds': pilot_scenario * 1.25,
              'checkpoint_saving_and_internal_validation': 'Included in historical first-epoch training scopes.',
              'research_evaluation_16_configurations': 'NOT_MEASURED_FOR_THE_PAIRED_PILOT',
              'paired_bootstrap_200': 'NOT_MEASURED_FOR_THE_PAIRED_PILOT',
              'source_generation_and_loading': 'Component scenario from 32-batch CPU replays; not a measured optimized full epoch.',
              'formal100_YOLO26s_six_arm_component_scenario_seconds':
                  300 * (control['training_setup_epoch_internal_val_save_seconds'] + optimized_scenario),
              'formal100_YOLOv8s_cost': 'NOT_MEASURED', 'formal100': 'RESOURCE_BLOCKED',
              'formal_completed_arms': 0, 'pilot_completed_pairs': 0, 'budget_reset': False,
              'limitations': 'First warmup epoch and small CPU trace; later mosaic closure, hardware contention and multi-epoch cost remain unmeasured.'}
    save(out / 'resource_decision.json', budget, out)
    sources = [source(run_path(PROFILE) / name) for name in ('CPU_PROFILE.csv', 'OPERATOR_TIMINGS.csv', 'profile_result.json')]
    sources += [source(measured / name) for name in ('input_acceptance.json', 'throughput.csv', 'reference_signatures.json')]
    sources += [source(run_path(WORKERS) / 'worker_admission.json'), source(units / 'test_results.json'),
                source(regression / 'regression_result.json'), source(run_path(FULL) / 'training_acceptance.json'),
                source(run_path(CONTROL) / 'training_acceptance.json'), source(Path(__file__)),
                source(prior_gpu / 'regression_result.json'), source(prior_gpu / 'failure.json')]
    save(out / 'inputs.json', {'evidence': sources}, out)
    implementations = ['profile_input.py', 'profile_decode.py', 'fast_input.py', 'input_ops.py', 'verify_input.py',
                       'test_input.py', 'probe_workers.py', 'regress_input.py', 'finalize_input.py']
    protected = []
    frozen = load(run_path(FULL) / 'protocol.json')['sources']
    for relative in ('code/main/training_data.py', 'code/main/training_effects.py', 'code/main/training_health.py',
                     'experiments/vm37_yolo26_rq4/training.py', 'experiments/vm37_yolo26_rq4/TARGETED_POLICY.json',
                     'experiments/cv261109/generator.py', 'experiments/vm36_supervisor_reproduction/feature_models/fm_p.uvl',
                     'experiments/vm36_supervisor_reproduction/feature_models/fm_d.uvl'):
        matches = [expected for key, expected in frozen.items() if key.replace('\\', '/') == relative]
        expected = matches[0] if matches else None
        protected.append({'path': relative, 'SHA256': sha256(PUBLIC / relative),
                          'historical_SHA256': expected, 'unchanged': expected == sha256(PUBLIC / relative)})
    save(out / 'preservation.json', {'files': protected, 'all_unchanged': all(row['unchanged'] for row in protected)}, out)
    if not all(row['unchanged'] for row in protected):
        raise ValueError('A protected scientific source differs.')
    operators = read_csv(run_path(PROFILE) / 'OPERATOR_TIMINGS.csv')
    native = {row['operation']: row for row in operators if row['scope'] == 'historical_native_replay'}
    passes = read_csv(measured / 'throughput.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), constrained_layout=True)
    selected = [('Shot noise', 'apply_shot_noise'), ('Gaussian noise', 'apply_gaussian_noise'),
                ('Haze', 'apply_haze'), ('Exposure', 'apply_exposure'), ('Motion blur', 'apply_motion_blur'), ('JPEG', 'apply_jpeg')]
    axes[0].barh([name for name, _ in selected][::-1],
                 [float(native[key]['inclusive_seconds']) for _, key in selected][::-1], color='#417aa3')
    axes[0].set(xlabel='Seconds in 512 saved native calls', title='Measured transform cost')
    for kind, color in (('original', '#6b7280'), ('arithmetic2', '#287f62')):
        selected_passes = [row for row in passes if row['implementation'] == kind]
        axes[1].plot([int(row['repetition']) + 1 for row in selected_passes],
                     [float(row['input_seconds']) for row in selected_passes], 'o-', color=color, label=kind)
    axes[1].set(xlabel='Alternating paired repetition', ylabel='32-batch CPU input time (s)',
                 title='Exact input replay; no GPU timing', xticks=[1, 2, 3])
    axes[1].legend(frameon=False)
    fig.savefig(out / 'input_performance.png', dpi=180)
    fig.savefig(out / 'input_performance.pdf')
    plt.close(fig)
    patch_dir = out / 'portable_patch'
    patch_dir.mkdir()
    for name in implementations:
        shutil.copyfile(POST / name, patch_dir / name)
    patch_manifest = {'status': 'SOURCE_PATCH_ONLY_NOT_A_TRAINING_AUTHORIZATION',
         'base': 'Verified VM37 portable executor v2 and frozen RQ4 environment.',
         'GPU_regression': gpu['status'], 'optimized_multi_epoch_GPU_training': 'NOT_VALIDATED',
         'policy_SHA256': sha256(HERE / 'TARGETED_POLICY.json'),
         'entry': 'regress_input.py prepares one gated128-batch regression; it does not execute a paired study.',
         'integration': 'Adopt a fresh RQ4Dataset using ArithmeticPrefetchDataset.adopt; keep workers0. Close its pool on all exit paths.',
         'files': {name: sha256(patch_dir / name) for name in implementations},
         'data_or_weights_included': False, 'new_budget': False}
    save(patch_dir / 'patch_manifest.json', patch_manifest, out)
    with zipfile.ZipFile(out / 'VM37_input_optimization_patch.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(patch_dir.iterdir()):
            archive.write(path, path.name)
    with zipfile.ZipFile(out / 'VM37_input_optimization_patch.zip') as archive:
        for name, expected in patch_manifest['files'].items():
            import hashlib
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError('Patch ZIP member differs.')
    prefix = out.name.rsplit('_', 1)[-1]
    report = ROOT / 'datasets/reportcn' / ('vm37_yolo26_rq4_输入管线性能优化_' + prefix + '.md')
    op_table = '\n'.join(f"| {label} | {native[key]['calls']} | {float(native[key]['inclusive_seconds']):.3f} | "
                         f"{float(native[key]['p50_seconds'])*1000:.2f} | {float(native[key]['p95_seconds'])*1000:.2f} |"
                         for label, key in selected)
    checks = '\n'.join('- ' + key + ': ' + str(value) for key, value in (numerical or {}).get('checks', {}).items())
    worker_table = '\n'.join(f"| {row['workers']} | {row['wall_seconds']:.3f} | "
                             f"{row['sampled_total_RSS_bytes']/1024**2:.1f} | {row['pixel_trace_equal_workers0']} |"
                             for row in workers['measurements'] if row['wall_seconds'] is not None)
    text = f'''# VM37 ALL_FIVE输入性能优化

## PROBLEM

在不改变五类增强、随机语义、1280输入和训练协议的前提下，降低CPU输入处理成本。没有重复完整epoch，没有运行20/100轮实验。

## WHY

旧完整ALL_FIVE训练计时90.500分钟，其中CPU加载与变换69.753分钟；CONTROL对应24.600、5.540分钟。此次先定位实际算子，而不是按计算复杂度猜测。

## METHOD

先精确回放历史512次native处理，再使用固定32批、64个anchor、256次native处理的CPU对照。每种实现重复三次，交替执行顺序。NumPy/OpenCV版本、JPEG编码器、运动模糊算法、原图与标签均保留。GPU回归最多600秒，温度规则不变，使用原48小时台账，未创建新预算。

实现分为两项：Mosaic的三个辅助图按原顺序分配ID和occurrence，只并行执行独立像素处理；高斯与泊松噪声使用128行分块，保留原随机数次序和每步float32算术。Haze/Exposure复用临时数组，像素SHA避免不必要的连续数组复制。没有改变采样概率、强度、顺序、标签或Mosaic选择逻辑。

## RESULT

### 1. CPU实测瓶颈

| 算子 | 实际调用 | 总秒数（含子操作） | 中位ms | P95 ms |
|---|---:|---:|---:|---:|
{op_table}

这里是样本回放占比，不能直接冒充完整epoch逐算子占比。父子inclusive时间不能重复相加；exclusive时间另存CSV。原始解码与增强后的JPEG解码分别记录。进程RSS为抽样观测，未测每个算子的独立内存峰值。

### 2. 优化效果与完全一致性

最终原实现三次输入总时长{result['original_seconds']:.3f}秒，新实现{result['optimized_seconds']:.3f}秒，观测提速{result['speedup_observed']:.3f}倍，输入耗时减少{result['reduction_fraction']*100:.2f}%。抽样进程RSS峰值{result['sampled_peak_RSS_bytes']/1024**2:.1f}MiB。该计时包含输入处理，不包括GPU训练；不能称为整轮训练加速比。

图像uint8张量、bbox、类别、增强开关和参数、原始/退化像素SHA、Mosaic辅助ID、occurrence、缓存顺序及Python/NumPy/Torch/Albumentations RNG状态完全一致。512个历史native事件也与旧轨迹一致。并非全6471图、多epoch数值轨迹均已验证。

初次CPU核对因初始化路径不同而失败，保留失败Run；统一首次导入预热后再固定相同RNG起点，未放宽任何一致性断言。首次误用普通.venv也被环境关卡阻止，后续仅使用.venv_rq4的Ultralytics8.4.210。未安装或升级依赖。

### 3. Windows workers调查

| workers | 16次anchor访问墙钟秒 | 进程树RSS MiB | 与workers0像素/轨迹一致 |
|---|---:|---:|---|
{worker_table}

1/2 worker均能启动并退出，anchor顺序相同，但随机种子、增强轨迹和像素不同；当前epoch通知也没有跨进程一致性证明。保留workers=0。短测含启动成本，不能据此认定多worker在所有规模下都更慢。

### 4. 训练数值回归

状态：**{gpu['status']}**。实际预算计费{gpu['charged_seconds']:.3f}秒，最高采样温度{gpu['maximum_sampled_temperature_C']}°C，最高采样显存{gpu['maximum_sampled_memory_MiB']}MiB。

{checks}

实际网络FP32输入逐批核对；梯度和模型/EMA健康检查保留。GPU回归含串行影子管线，不能用它推算优化后的训练速度。工程state文件不是20/100轮初始化，不承诺精确resume。新增AP、AR、退化鲁棒性和增强收益均NOT_EVALUATED。

前一次GPU核对在首次forward之前因CPU归一化参考与CUDA位级比较失败，0次训练更新，计费{prior_result['charged_seconds']:.3f}秒，原失败记录保留。修正为同GPU上的原版DetectionTrainer预处理后，仅另执行一次有界回归；实际CPU/CUDA归一化最大差值记录为{normalization_difference}。没有放宽同设备位级断言，也没有修改训练计算。

### 5. 资源决定

原截止：{ledger['absolute_deadline_UTC']}，巴黎时间2026-11-13 23:03:21。核算时有效余量约{available/3600:.3f}小时，未重置。

若仅把实测CPU输入比值应用于旧完整首轮的加载分量，得到ALL_FIVE约{optimized_scenario/60:.2f}分钟/轮、20轮双组约{pilot_scenario/3600:.2f}小时；加25%规划余量约{pilot_scenario*1.25/3600:.2f}小时。此为分量外推情景，不是优化后完整epoch实测。库内验证和保存已包含旧计时；16项研究评价、200次配对Bootstrap尚无完整实测成本，须另计。

20轮仍为PILOT_RESOURCE_BLOCKED，完成配对0。正式100轮仍为RESOURCE_BLOCKED：两个架构、三个seed、两臂共12次训练及原五项YOLOv8s消融保持不变。YOLO26s六臂按同一情景约{budget['formal100_YOLO26s_six_arm_component_scenario_seconds']/3600:.1f}训练小时，尚不含研究评价；YOLOv8s训练成本未知，不编造全矩阵时间。没有单独启动CONTROL。

## CONCLUSION

本次支持“在所测CPU输入轨迹上获得完全一致且更快的实现”；GPU数值回归结论以以上实际状态为准。没有完整配对训练或增强收益结论。下一步需要服务器CPU与GPU联合成本测量、完整对照预算和明确准入。ALL_FIVE仍是广泛随机增强，haze仍是未校准的常深度近似。

## LIMITATIONS

CPU性能来自固定小样本和未清空的OS缓存；后续Mosaic关闭、温度、共享资源会影响速度。跨epoch、模型种子以及新机器均未由这次短回归验证。没有降低分辨率、batch、FP32精度、增强概率或健康检查频率；未使用test-dev。旧训练器、策略、UVL和数据保持不变。便携v2包未覆盖；本次仅提供独立源码补丁，不是长训练授权。

### RunID与文件

- 原始算子计时：{PROFILE}
- 仅并行对照：{THREADS}
- 最终CPU对照：{input_id}
- 单元测试：{unit_id}
- workers调查：{WORKERS}
- GPU数值回归：{regression_id}
- 本汇总：{out.name}

原始CSV：[CPU_PROFILE.csv]({(run_path(PROFILE)/'CPU_PROFILE.csv').as_posix()})、[OPERATOR_TIMINGS.csv]({(run_path(PROFILE)/'OPERATOR_TIMINGS.csv').as_posix()})。

[本地可视化]({(out/'index.html').as_posix()})；[数值回归]({(regression/'regression_result.json').as_posix()})；[资源台账]({(out/'resource_decision.json').as_posix()})；[源码补丁]({(out/'VM37_input_optimization_patch.zip').as_posix()})。

复现：使用根目录.venv_rq4解释器，从main执行experiments/vm37_yolo26_rq4/post_cost/profile_input.py、test_input.py、verify_input.py --batches 32 --repeats 3 --implementation arithmetic2。各Run的command.txt保存实际命令。GPU票据已消耗，不得复用RunID重跑；历史授权到期后不得继续使用。
'''
    write_atomic(report, text, ROOT)
    save(out / 'report_provenance.json', {'report': source(report), 'patch': source(out / 'VM37_input_optimization_patch.zip'),
         'source_SHA256': {name: sha256(POST / name) for name in implementations}}, out)
    write_atomic(out / 'index.html', '<!doctype html><meta charset="utf-8"><title>VM37 input performance</title>'
         '<style>body{max-width:1050px;margin:2rem auto;font:17px system-ui;color:#183042}img{width:100%}</style>'
         '<h1>VM37 input performance</h1><p>CPU exact input replay: ' + f"{result['speedup_observed']:.3f}x observed. "
         + html.escape(gpu['status']) + '.</p><p>20/100-epoch paired study remains resource-blocked.</p>'
         '<img src="input_performance.png" alt="Measured CPU operation costs and paired replay timing">'
         '<p><a href="resource_decision.json">Resource evidence</a> · <a href="inputs.json">Source index</a> · '
         '<a href="VM37_input_optimization_patch.zip">Versioned source patch</a></p>', out)
    write_atomic(out / 'diary.md', '# Input optimization summary\n\nCollected exact CPU replay, worker probes and '
         'one finite GPU regression. Preserved negative runs and resource limits. No paired study or new AP.\n', out)
    save(out / 'run.json', {'RunID': out.name, 'ExperimentID': EXPERIMENT,
         'status': 'COMPLETE_WITH_LIMITATIONS', 'GPU_regression': gpu['status'],
         'CPU_wall_seconds': time.perf_counter() - started, 'pilot20': 'PILOT_RESOURCE_BLOCKED'}, out, replace=True)
    for folder, title, files in ((run_path(PROFILE), 'Input operation profile', ['CPU_PROFILE.csv', 'OPERATOR_TIMINGS.csv', 'memory.csv', 'LOADER_BOTTLENECKS.md']),
                                 (measured, 'Exact input replay', ['input_acceptance.json', 'throughput.csv', 'timing.csv']),
                                 (units, 'Native prefetch tests', ['test_results.json', 'tests.log']),
                                 (run_path(WORKERS), 'Windows loader probes', ['worker_admission.json', 'worker_comparison.csv']),
                                 (regression, 'Training numerical regression', ['regression_result.json', 'numerical_comparison.json', 'tensor_checks.jsonl'])):
        if not (folder / 'index.html').exists():
            links = ''.join('<li><a href="' + name + '">' + name + '</a></li>' for name in files if (folder / name).exists())
            write_atomic(folder / 'index.html', '<!doctype html><meta charset="utf-8"><h1>' + title + '</h1><ul>'
                 + links + '</ul><p><a href="../' + out.name + '/index.html">Summary and figures</a></p>', folder)
            save(folder / 'viewer_SHA256.json', {'index': source(folder / 'index.html'),
                 'scientific_outputs_modified': False}, folder)
    seal(out)
    from main.visualization import build_run_index
    build_run_index(RUNS, EXPERIMENT)
    print(json.dumps({'RunID': out.name, 'report': str(report), 'GPU': gpu['status'],
                     'CPU_speedup': result['speedup_observed'], 'pilot20': 'PILOT_RESOURCE_BLOCKED'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-run', required=True)
    parser.add_argument('--unit-run', required=True)
    parser.add_argument('--regression-run', required=True)
    parser.add_argument('--profile-run', default=PROFILE)
    args = parser.parse_args()
    execute(args.input_run, args.unit_run, args.regression_run, args.profile_run)
