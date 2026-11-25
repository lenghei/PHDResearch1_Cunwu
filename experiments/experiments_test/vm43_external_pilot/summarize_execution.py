"""Verify and report a finished VM43 run without inference or COCO reruns."""
import argparse
import ast
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import time

from vm43_common import ROOT, HERE, PUBLIC, RUNS, AUTH, V42, load, save, sha, source, budget

SCOPE = 'LIMITED_EXTERNAL_EXPLORATORY'
DEV = ROOT / 'datasets/local_runs/vm39_multi_seed_confirmation/vm39_multi_seed_confirmation_confirmation_20261115T095834_9623ace0'
AUDIT = ROOT / 'datasets/local_runs/vm40_uvl_gap_independent_validation/vm40_uvl_gap_independent_validation_audit_20261116T011556_955084fb'


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_text(path, text):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        f.write(text)


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('x', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def manifest_check(path):
    count = 0
    with Path(path).open(encoding='utf-8') as f:
        for line in f:
            digest, name = line.rstrip('\n').split('  ', 1)
            if sha(name) != digest:
                raise ValueError('Sealed file changed: ' + name)
            count += 1
    return {'manifest': source(path), 'files_verified': count, 'status': 'PASS'}


def ci_text(value, scale=100):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if value is None or any(x is None for x in value):
        return 'NOT_ESTIMABLE'
    return f'[{value[0]*scale:.3f}, {value[1]*scale:.3f}]'


def table(headers, rows):
    clean = lambda x: str(x).replace('|', '/').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |',
                      *['| ' + ' | '.join(clean(x) for x in row) + ' |' for row in rows]])


def verify_cell(folder, spec, payload, cache):
    acceptance = load(folder/'acceptance.json')
    if acceptance['status'] != 'REAL_CELL_COMPLETED' or acceptance['images'] != 1610 or acceptance['scope'] != SCOPE:
        raise ValueError('Invalid cell acceptance.')
    for key in ('raw_predictions', 'GT', 'metrics', 'bootstrap'):
        item = acceptance[key]
        if sha(item['path']) != item['SHA256']:
            raise ValueError('Cell source changed: ' + key)
    if acceptance['checkpoint'] != spec['checkpoint'] or sha(spec['checkpoint']['path']) != spec['checkpoint']['SHA256']:
        raise ValueError('Checkpoint provenance differs.')
    metrics, boot, timing = [load(folder/name) for name in ('metrics.json', 'bootstrap.json', 'timing.json')]
    expected_ids = [r['id'] for r in payload['records']]
    manifest = [json.loads(line) for line in (folder/'image_manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(manifest) != 1610 or [r['image_id'] for r in manifest] != expected_ids:
        raise ValueError('Image order or count differs.')
    for row, pixel in zip(manifest, cache[spec['condition']]):
        if {k: v for k, v in row.items() if k != 'tensor'} != pixel:
            raise ValueError('Cached pixels differ between cells.')
        if row['tensor'] != {'NCHW': [1, 3, 1280, 1280], 'dtype': 'torch.float32'}:
            raise ValueError('Actual tensor differs.')
    ids, count = [], 0
    with gzip.open(folder/'predictions.jsonl.gz', 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            ids.append(row['image_id'])
            for p in row['predictions']:
                if p['image_id'] != row['image_id'] or not isinstance(p['class_id'], int) or isinstance(p['class_id'], bool) or not 0 <= p['class_id'] < 10:
                    raise ValueError('Prediction identity differs.')
                if not all(math.isfinite(x) for x in [*p['bbox'], p['score']]) or p['bbox'][2] <= 0 or p['bbox'][3] <= 0 or not 0 <= p['score'] <= 1:
                    raise ValueError('Invalid prediction.')
                count += 1
    if ids != expected_ids or metrics['images'] != 1610 or count != metrics['raw_count']:
        raise ValueError('Raw prediction population differs.')
    draws = boot['draws']
    if draws['image_ids'] != expected_ids or draws['samples'] != 200 or draws['seed'] != 20261116:
        raise ValueError('Bootstrap identity differs.')
    if [r['image_id'] for r in timing['per_image']] != expected_ids:
        raise ValueError('Timing population differs.')
    for row in timing['per_image']:
        if row['tensor'] != {'NCHW': [1, 3, 1280, 1280], 'dtype': 'torch.float32'} or not math.isfinite(row['cached_pipeline_ms']):
            raise ValueError('Invalid measured timing/tensor.')
    return metrics, boot, timing, {'cell_id': spec['cell_id'], 'images': len(ids), 'raw_predictions': count,
        'checkpoint': acceptance['checkpoint'], 'raw': acceptance['raw_predictions'], 'GT': acceptance['GT'],
        'NCHW_FP32': 'PASS_ALL_1610', 'pixel_identity': 'PASS', 'status': 'PASS'}


def verify_comparisons(cells, rows):
    import numpy as np
    def area(cell, size):
        return next(x for x in cell[0]['per_area'] if x['size'] == size)
    for row in rows:
        context, metric = row['condition'], row['metric']
        if context in ('Normal', 'Gaussian_s15'):
            c, g = cells['CONTROL', context], cells['VARIABILITY_GUIDED', context]
            if metric == 'cached_pipeline_ms':
                left = np.array([r[metric] for r in c[2]['per_image']])
                right = np.array([r[metric] for r in g[2]['per_image']])
                ids = np.random.default_rng(20261116).integers(0, 1610, (200, 1610))
                point, draws = float((right-left).mean()), (right-left)[ids].mean(axis=1)
            else:
                size = row['size']
                x, y = area(c, size)[metric], area(g, size)[metric]
                if x is None or y is None:
                    if row['status'] != 'NOT_ESTIMABLE':
                        raise ValueError('Unestimable comparison not labelled.')
                    continue
                point = y-x
                draws = np.asarray(g[1]['draws']['values'][size][metric])-np.asarray(c[1]['draws']['values'][size][metric])
        else:
            nc, bc, ng, bg = [cells[key] for key in [('CONTROL', 'Normal'), ('CONTROL', 'Gaussian_s15'), ('VARIABILITY_GUIDED', 'Normal'), ('VARIABILITY_GUIDED', 'Gaussian_s15')]]
            value = lambda c: area(c, 'all')['map50_95']
            dist = lambda c: np.asarray(c[1]['draws']['values']['all']['map50_95'])
            point = (value(ng)-value(bg))-(value(nc)-value(bc))
            draws = (dist(ng)-dist(bg))-(dist(nc)-dist(bc))
        if not np.isclose(point, float(row['difference']), atol=1e-12, rtol=0) or not np.allclose(np.percentile(draws, [2.5, 97.5]), ast.literal_eval(row['ci95']), atol=1e-12, rtol=0):
            raise ValueError('Saved paired contrast does not reproduce.')


def plot_results(out, cells, comparisons, gaps, dev, dev_mean):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    plots = out/'plots'
    plots.mkdir()
    paths = []
    def finish(fig, name):
        fig.text(.01, .01, 'LIMITED_EXTERNAL_EXPLORATORY | seed42 | 200 paired image resamples | independence UNKNOWN', fontsize=7)
        fig.tight_layout(rect=(0, .045, 1, 1))
        for suffix in ('png', 'pdf'):
            path = plots/(name+'.'+suffix)
            if path.exists():
                raise FileExistsError(path)
            fig.savefig(path, dpi=175)
            paths.append(path)
        plt.close(fig)
    def point(strategy, condition, size='all'):
        return next(r['map50_95'] for r in cells[strategy, condition][0]['per_area'] if r['size'] == size)
    def bounds(strategy, condition, size='all'):
        return next(r['ci95'] for r in cells[strategy, condition][1]['intervals'] if r['size'] == size and r['metric'] == 'map50_95')
    def difference(context, size='all', metric='map50_95'):
        return next(r for r in comparisons if r['condition'] == context and r.get('size') == size and r['metric'] == metric)
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    for i, strategy in enumerate(('CONTROL', 'VARIABILITY_GUIDED')):
        x = np.arange(2)+(i-.5)*.3
        values = np.array([point(strategy, c)*100 for c in ('Normal', 'Gaussian_s15')])
        ci = np.array([bounds(strategy, c) for c in ('Normal', 'Gaussian_s15')])*100
        ax.bar(x, values, .28, label=strategy.replace('VARIABILITY_', ''), color=('#526579', '#25877e')[i])
        ax.errorbar(x, values, yerr=np.maximum(0, np.vstack([values-ci[:, 0], ci[:, 1]-values])), fmt='none', color='black', capsize=4)
    ax.set(xticks=[0, 1], xticklabels=['Normal', 'Gaussian sigma15'], ylabel='AP50:95 (%)', title='Absolute detection AP on 1,610 test-dev images')
    ax.legend()
    finish(fig, 'absolute_AP')
    ext = difference('Gaussian_s15')
    values = [float(dev['GUIDED_minus_CONTROL'])*100, float(ext['difference'])*100]
    cis = [ast.literal_eval(dev['difference_CI95']), ast.literal_eval(ext['ci95'])]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for y, (v, ci) in enumerate(zip(values, cis)):
        ax.plot(v, y, 'o', color=('#526579', '#25877e')[y])
        ax.hlines(y, ci[0]*100, ci[1]*100, color=('#526579', '#25877e')[y], linewidth=2)
    ax.axvline(0, color='gray', linewidth=1)
    ax.set(yticks=[0, 1], yticklabels=['VM39 development: 548, seed42', 'VM43 exploratory: 1,610, seed42'], xlabel='GUIDED - CONTROL Gaussian AP (pp)', title='Separate populations; AP is not pooled')
    ax.text(.02, .94, f"Development three-seed mean {float(dev_mean['mean'])*100:+.3f} pp; seed SD {float(dev_mean['sample_seed_SD'])*100:.3f} pp", transform=ax.transAxes, fontsize=8)
    ax.margins(y=.5)
    finish(fig, 'development_external_Gaussian_gain')
    for context, size, name, title in [('Gaussian_s15', 'tiny', 'Gaussian_Tiny_difference', 'Gaussian Tiny AP: GUIDED - CONTROL'), ('Normal', 'all', 'Normal_AP_difference', 'Normal AP cost: GUIDED - CONTROL')]:
        row = difference(context, size)
        ci = ast.literal_eval(row['ci95'])
        fig, ax = plt.subplots(figsize=(6.5, 3.4))
        ax.hlines(0, ci[0]*100, ci[1]*100, color='#25877e', linewidth=3)
        ax.plot(float(row['difference'])*100, 0, 'o', color='#25877e')
        ax.axvline(0, color='gray', linewidth=1)
        ax.set(yticks=[], xlabel='AP50:95 difference (pp)', title=title)
        finish(fig, name)
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    for y, row in enumerate(gaps):
        ci = ast.literal_eval(row['latency_CI_ms'])
        ax.hlines(y, ci[0], ci[1], color='#25877e', linewidth=2)
        ax.plot(float(row['latency_mean_ms']), y, 'o', color='#25877e')
    ax.axvline(100, color='#b04c3b', linestyle='--', label='Frozen 100 ms bound')
    ax.set(yticks=range(4), yticklabels=[r['cell_id'].replace('s42_', '').replace('_R1280', '') for r in gaps], xlabel='Measured mean cached pipeline (ms), pointwise 95% CI', title='Integrity/read/decode through prediction and output flush')
    ax.legend()
    finish(fig, 'cached_pipeline_latency')
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.axis('off')
    rendered = [[r['cell_id'].replace('s42_', '').replace('_R1280', ''), ci_text(r['Tiny_CI']), ci_text(r['latency_CI_ms'], 1), r['state']] for r in gaps]
    tab = ax.table(cellText=rendered, colLabels=['Cell', 'Tiny AP CI (%)', 'Mean latency CI (ms)', 'State'], loc='center', colWidths=[.43, .18, .25, .14])
    tab.auto_set_font_size(False)
    tab.set_fontsize(8)
    tab.scale(1, 1.8)
    ax.set_title('Frozen Tiny/latency criterion; no full-space Gap claim')
    finish(fig, 'measured_Gap_states')
    return paths


def main(identity):
    started = time.perf_counter()
    if Path(identity).name != identity or not identity.startswith('vm43_external_pilot_execution_'):
        raise ValueError('Expected a VM43 execution RunID.')
    out = (RUNS/identity).resolve(strict=True)
    if (out/'scientific_acceptance.json').exists():
        raise FileExistsError('This execution already has a scientific receipt; do not overwrite.')
    final = load(out/'final_acceptance.json')
    supervisor = load(out/'safety_supervisor.json')
    ticket = load(out/'execution_ticket.json')
    prepared = Path(ticket['prepared_run'])
    spec = load(V42/'Pilot_4cell_frozen_protocol.json')
    check = manifest_check(out/'SHA256SUMS.txt')
    old_check = manifest_check(prepared/'SHA256SUMS.txt')
    decision, authorization, thermal = [Path(ticket[k]) for k in ('decision', 'authorization', 'thermal')]
    admission = decision.parent
    admission_check = manifest_check(admission/'SHA256SUMS.txt')
    for name, path in [('scientific_decision.json', decision), ('gpu_authorization.json', authorization), ('thermal_admission.json', thermal)]:
        if (out/name).exists():
            raise FileExistsError(out/name)
        shutil.copyfile(path, out/name)
        if sha(out/name) != sha(path):
            raise ValueError('Admission copy differs.')
    inputs = load(prepared/'execution_inputs.json')['files']
    for item in inputs:
        if sha(item['path']) != item['SHA256']:
            raise ValueError('Frozen input changed: ' + item['path'])
    for name, expected in [('fm_p.uvl', 'e6eadd633d1c14e0d6bc425f5e54421d592d766c65a6e5f20c61bc34dd76fb5c'), ('fm_d.uvl', '4022d674822b97dbbe3712646b9a0f970af95d175ae358e63582029bc107571a')]:
        if sha(HERE.parent/'vm36_supervisor_reproduction/feature_models'/name) != expected:
            raise ValueError('Original supervisor UVL changed.')
    complete = final['status'] == 'REAL_FOUR_CELLS_COMPLETED' and supervisor['returncode'] == 0 and supervisor['stop_reason'] is None
    cache = load(out/'pixel_cache_manifest.json') if (out/'pixel_cache_manifest.json').exists() else {}
    if cache:
        probe = load(admission/'CPU_cache_probe.json')['rows']
        pixels = {(condition, r['stem']): r['pixel_SHA256'] for condition, entries in cache.items() for r in entries}
        for row in probe:
            if pixels[row['condition'], row['image_id']] != row['pixel_SHA256']:
                raise ValueError('Formal pixels differ from the engineering probe.')
    cells, evidence = {}, []
    payload = load(prepared/'ground_truth_input.json')
    for cell in spec['cells']:
        folder = out/cell['cell_id']
        if (folder/'acceptance.json').exists():
            a, b, c, proof = verify_cell(folder, cell, payload, cache)
            cells[cell['strategy'], cell['condition']] = (a, b, c)
            evidence.append(proof)
    if complete and len(cells) != 4:
        raise ValueError('Four-cell terminal status without four valid receipts.')
    telemetry = [json.loads(line) for line in (out/'gpu_telemetry.jsonl').read_text(encoding='utf-8').splitlines()]
    preflight = load(admission/'preflight_completion.json')
    ledger = budget()
    for old in preflight['budget_after']['charges']:
        if sha(old['path']) != old['SHA256']:
            raise ValueError('Historical budget charge changed.')
    charge = AUTH/'charges'/(out.name+'_evaluation.json')
    if not math.isclose(load(charge)['charged_seconds'], final['charged_seconds'], abs_tol=1e-9):
        raise ValueError('Charge receipt differs.')
    resource = {'formal_supervisor_wall_seconds': supervisor['wall_seconds'], 'formal_charged_worker_wall_seconds': final['charged_seconds'],
        'engineering_preflight_charged_wall_seconds': preflight['charged_seconds'], 'this_authorization_total_charged_wall_seconds': final['charged_seconds']+preflight['charged_seconds'],
        'unit': 'Supervised worker wall includes CPU cache/COCO/bootstrap and GPU prediction; not GPU kernel busy time',
        'GPU_kernel_busy_seconds': 'NOT_MEASURED', 'max_temperature_C': max(r['temperature_C'] for r in telemetry),
        'max_NVML_used_MiB': max(r['used_MiB'] for r in telemetry), 'telemetry_samples': len(telemetry),
        'power_and_clocks_time_series': 'NOT_RECORDED_BY_FORMAL_MONITOR', 'budget_after': ledger,
        'actual_cell_costs': {a+'_'+b: {'evaluation': v[2]['evaluation_cost'], 'peak_allocated_bytes': v[2]['peak_allocated_bytes'], 'peak_reserved_bytes': v[2]['peak_reserved_bytes'],
            'cached_pipeline_total_seconds': sum(t['cached_pipeline_ms'] for t in v[2]['per_image'])/1000} for (a, b), v in cells.items()},
        'standalone_cache_generation_seconds': 'NOT_SEPARATELY_LOGGED; included in measured worker wall',
        'model_loading_and_warmup_seconds': 'NOT_SEPARATELY_LOGGED_IN_FORMAL_RUN; included in worker wall', 'new_training': False, 'retry': False}
    save(out/'execution_protocol.json', {'RunID': out.name, 'scope': SCOPE, 'historical_independence': 'UNKNOWN', 'protocol': source(V42/'Pilot_4cell_frozen_protocol.json'),
        'prepared_run': str(prepared), 'GT': source(prepared/'ground_truth_input.json'), 'cells': spec['cells'], 'frozen_inference': spec['inference'],
        'primary': 'Gaussian_s15 GUIDED minus CONTROL AP50:95 at1280', 'bootstrap_samples': 200, 'bootstrap_seed': 20261116,
        'training_seed': 42, 'policy_retuned': False, 'clean_loss_tolerance': 'NOT_NUMERICALLY_FROZEN; report tradeoff only'})
    save(out/'source_provenance.json', {'RunID': out.name, 'frozen_input_verification': inputs, 'raw_receipts': evidence,
        'population': spec['population'], 'pixel_cache': source(out/'pixel_cache_manifest.json') if cache else None,
        'source_ZIP': load(spec['population']['path'])['source_ZIP_accepted_certificate'],
        'original_execution_seal_verified': check, 'preparation_seal_verified': old_check,
        'admission_seal_verified': admission_check,
        'analysis_source': source(__file__), 'scientific_scope': SCOPE, 'historical_independence': 'UNKNOWN',
        'GT_handling': 'Unchanged native10 ignore-union; original-pixel xywh area; no target reselection'})
    comparisons = read_csv(out/'paired_comparisons.csv') if complete else []
    gaps = read_csv(out/'gap_status.csv') if complete else []
    plots = []
    paths = []
    if complete:
        verify_comparisons(cells, comparisons)
        per_class = [{'cell_id': cell['cell_id'], 'strategy': cell['strategy'], 'condition': cell['condition'], 'scope': SCOPE, **r}
                     for cell in spec['cells'] for r in cells[cell['strategy'], cell['condition']][0]['per_class']]
        write_csv(out/'per_class_metrics.csv', per_class)
        dev_rows = read_csv(DEV/'per_seed_comparisons.csv')
        dev = next(r for r in dev_rows if r['seed'] == '42' and r['resolution'] == '1280' and r['context'] == 'Gaussian_s15' and r['size'] == 'all' and r['metric'] == 'map50_95')
        mean = next(r for r in read_csv(AUDIT/'VM39_three_seed_summary.csv') if r['resolution'] == '1280' and r['condition'] == 'Gaussian_s15' and r['size'] == 'all')
        save(out/'development_comparison.json', {'development_source': source(DEV/'per_seed_comparisons.csv'), 'three_seed_source': source(AUDIT/'VM39_three_seed_summary.csv'),
            'seed42_development': dev, 'development_three_seed': mean, 'external_scope': SCOPE, 'populations_pooled': False,
            'latency_comparison_to_old_inference_ms': 'NOT_COMPARABLE'})
        plots = plot_results(out, cells, comparisons, gaps, dev, mean)
    report_dir = ROOT/'datasets/reportcn'
    report = report_dir/'VM43_外部探索真实评价_科学报告_CN.md'
    guide = report_dir/'VM43_外部探索导师讲解与问答_CN.md'
    if report.exists() or guide.exists():
        report = report.with_name(report.stem+'_'+out.name+'.md')
        guide = guide.with_name(guide.stem+'_'+out.name+'.md')
    state = 'REAL_FOUR_CELLS_COMPLETED' if complete else 'INCOMPLETE'
    stop = final.get('supervisor_stop_reason')
    brief = f'''# VM43 外部探索真实评价

RunID：`{out.name}`。实际状态：**{state}**。完整且核验通过的格数：{len(cells)}/4。

科学范围：**LIMITED_EXTERNAL_EXPLORATORY**；历史独立性 **UNKNOWN**；训练seed42，策略为POST_HOC_DEVELOPMENT。用户明确授权不是导师签字或独立性证明。

## 实际执行与保护

复用既有31项CPU验收，没有重跑VM36–42、重新训练或改变GUIDED策略。一次真实预检后，仅通过既有`run.py --execute`启动一次正式任务。执行顺序Normal CONTROL/GUIDED，然后Gaussian CONTROL/GUIDED，不根据中途AP改序。原监督器停止原因：`{stop}`。

原准备Run、导师UVL、冻结协议/类别/GT处理、原checkpoint和历史计费SHA复核通过。原始`final_acceptance.json`、`run.json`与`SHA256SUMS.txt`保留；`run.json`中的RUNNING是入口初始记录，终态以`final_acceptance.json`与新增`scientific_acceptance.json`为准。本报告未回写或伪造旧Run终态。

## 数据、模型和口径

全部1610张test-dev候选图，原生十类，ignore后74687个GT实例；源ZIP/原图/标注SHA和成员取自冻结清单。当前train/val没有ID或字节重复，但完整历史调参与场景/航次隔离仍未证明。不能称严格独立确认。

两份YOLO26s是960训练、10轮、seed42、固定epoch10 EMA；本次1280 FP32 batch1推理。CONTROL SHA `ce07d5dc630f7f5350a472e63ad277880c2532d3e3ca09e1d9bb06f8b8f4890d`；GUIDED SHA `4bf69c0af8b168e9e42e5a2f5329297d210ee5f2b033f2e64cbbf5938727db10`。

Normal及Gaussian_s15各生成一套共享PNG，两模型复用同一像素；Gaussian测试固定σ15，不使用训练p=.25再次随机抽增强。逐图NCHW=[1,3,1280,1280]、torch.float32验收，dense one-to-many，conf=.001，IoU=.7，class-aware multi-label，max_det3000/max_nms30000，TF32关闭。

COCO AP采用原图xywh面积、IoU .50:.05:.95、101 recall points及既有ignore-union；Tiny<256，Small[256,1024)，Medium[1024,9216)，Large≥9216像素²；另留COCO-small≤1024，不能混称。原生十类各面积指标见`per_class_metrics.csv`及各格`metrics.json`。缺GT保持null。
'''
    if complete:
        area_rows = []
        for cell in spec['cells']:
            metrics = cells[cell['strategy'], cell['condition']][0]
            by_area = {r['size']: r for r in metrics['per_area']}
            vals = [by_area['all']['map50'], *[by_area[k]['map50_95'] for k in ('all', 'tiny', 'small', 'medium', 'large')], by_area['all']['ar50_95']]
            area_rows.append([cell['strategy'], cell['condition'], *[f'{x*100:.3f}' if x is not None else 'null' for x in vals]])
        brief += '\n## 四格实测精度\n\n全部6440张次完成，另有12次预热，不记作新增评价图。下表为百分数；CSV/JSON保留0–1原值。\n\n'
        brief += table(['策略', '条件', 'AP50', 'AP95', 'Tiny', 'Small', 'Medium', 'Large', 'AR95'], area_rows)+'\n'
        selected = [r for r in comparisons if (r['metric'] == 'map50_95' and r.get('size') in ('all', 'tiny')) or r['metric'] == 'cached_pipeline_ms']
        brief += '\n## 配对差与统计限制\n\n'+table(['条件', '面积/指标', 'GUIDED−CONTROL / 损失变化', '95%区间'],
            [[r['condition'], r.get('size', '')+'/'+r['metric'], f"{float(r['difference'])*(1 if r['metric']=='cached_pipeline_ms' else 100):+.3f}"+( ' ms' if r['metric']=='cached_pipeline_ms' else ' pp'), ci_text(r['ci95'], 1 if r['metric']=='cached_pipeline_ms' else 100)] for r in selected])+'\n'
        brief += '\nGaussian_degradation_loss定义为(GUIDED Normal−Gaussian)−(CONTROL Normal−Gaussian)，负值表示退化损失减小。200次同图像位置配对重采样、seed20261116，每次重累计COCO AP，不是平均单图AP。保存的差值和区间已由同一组Bootstrap draws重新核算通过；没有重跑GPU或COCO。\n\n仅一个训练seed，点态95%区间不等于训练seed显著性；没有多重比较家族校正，也没有可靠真实航次聚类。Normal可接受损失未冻结数值，不能临时将某下降判为可接受。\n'
        brief += '\n## 开发与外部探索分别报告\n\n'
        ext = next(r for r in comparisons if r['condition'] == 'Gaussian_s15' and r.get('size') == 'all' and r['metric'] == 'map50_95')
        brief += f"VM39的548张开发图seed42 Gaussian收益{float(dev['GUIDED_minus_CONTROL'])*100:+.3f}pp，区间{ci_text(dev['difference_CI95'])}pp；三seed开发均值{float(mean['mean'])*100:+.3f}pp、训练seed样本SD{float(mean['sample_seed_SD'])*100:.3f}pp。VM43的1610图为{float(ext['difference'])*100:+.3f}pp，区间{ci_text(ext['ci95'])}pp。两集合AP不混合；外部探索有收益也不能升级为独立确认，负面代价同样保留。\n"
        brief += '\n## 同口径时延与Gap\n\n'+table(['格', 'Tiny区间(%)', '缓存管线均值ms', '均值区间ms', '状态'],
            [[r['cell_id'], ci_text(r['Tiny_CI']), f"{float(r['latency_mean_ms']):.3f}", ci_text(r['latency_CI_ms'], 1), r['state']] for r in gaps])+'\n'
        transitions = []
        for context in ('Normal', 'Gaussian_s15'):
            c, g = [next(r['state'] for r in gaps if r['cell_id'] == f's42_{strategy}_{context}_R1280') for strategy in ('CONTROL', 'VARIABILITY_GUIDED')]
            transitions.append(f'{context}: {c} → {g}')
        brief += '\n固定判据的CONTROL→GUIDED实测转移：'+'；'.join(transitions)+'。这不是完整FM_D空间的覆盖率。\n'
        brief += '\n缓存管线包含PNG完整性、读取、解码、像素SHA、预处理/forward/NMS、取框、预测与manifest写入flush；不含离线PNG生成/模型加载/预热/COCO/Bootstrap。这些仍计入worker预算。库内三个profile是诊断分量，不用于拼接总延迟；旧inference_ms不能替代此口径。顺序固定但并未随机交叉，OS缓存、时钟及热状态可能影响差值，不能解释为训练策略导致的纯算法提速。flush不等于fsync。\n\n冻结规则：Tiny上界<.10或时延均值上界>100ms为FAIL；Tiny下界≥.10且时延上界≤100ms为PASS；否则UNCERTAIN。仅描述两个实测P条件、两份指定D权重的状态和转移；不估计完整FM空间Gap比例，整体AP增加不等于Gap关闭。\n'
    else:
        brief += '\n## 未完成原因\n\n正式四格没有全部通过验收，停止原因见原`final_acceptance.json`/`worker_failure.json`。仅保留真实完成格与partial输出，不发布主配对比较，不补跑、不降低图数或改配置。未完成AP、Tiny差、时延差、Gap变化均为NOT_EVALUATED。\n'
    brief += f'''
## 热安全、真实成本和余额

真实短预检监测最高61°C、计费{preflight['charged_seconds']:.3f}秒；正式监测最高{resource['max_temperature_C']}°C，NVML已用显存峰值{resource['max_NVML_used_MiB']}MiB，监测{len(telemetry)}条。90°C即停、87°C连续30秒停及2秒监测保持原样。使用仍在有效期内的11月14日用户物理检查记录，并非本次重新现场确认；室温、实际风扇RPM、OEM模式仍UNKNOWN。一次预检/推理通过不能证明未来长训练Windows或散热稳定。

正式监督墙钟{supervisor['wall_seconds']:.3f}秒，正式计费{final['charged_seconds']:.3f}秒；含预检本次合计{resource['this_authorization_total_charged_wall_seconds']:.3f}秒（{resource['this_authorization_total_charged_wall_seconds']/3600:.6f}小时）。这是含CPU缓存/COCO/Bootstrap的GPU任务worker墙钟，不是CUDA kernel忙碌时间。最终封存后CPU验收/报告时间另记，不虚称GPU训练时长。

原36小时账本当前剩{ledger['remaining_seconds']:.3f}秒（{ledger['remaining_seconds']/3600:.6f}小时），原截止`{ledger['deadline_UTC']}`不变。无重试、扩额或重置。正式逐项加载/缓存时间未单列日志，只能列为总墙钟包含项；不能以预检时间冒充正式测量。逐格COCO/Bootstrap/管线与显存实测见`resource_usage.json`。

## 证据与复现

运行命令保留`command.txt`；原始终态`final_acceptance.json`，后验核验`scientific_acceptance.json`，原封存`SHA256SUMS.txt`与新增产物封存`analysis_SHA256SUMS.txt`。每格保留1610行原预测gzip、image_manifest、GT引用、原值metrics、200次draws和来源SHA。

仅CPU复核/制图入口：`{HERE/'summarize_execution.py'}`；本Run已封存后再次调用会拒绝覆盖。需要完整重演时保留原Run并单独授权，不能把此命令理解为允许再次GPU执行。

结果目录：`{out}`。图表位于`plots/`，汇总页面`index.html`，各指标原值CSV在Run根目录，私有原图/权重/完整日志不进入Git。没有pull/rebase/reset/commit/push。下一步仅根据本次有限探索证据向导师报告；不调整策略，不自动扩大seed或训练100轮，正式RQ4仍未完成。
'''
    write_text(report, brief)
    guide_text = f'''# VM43 导师讲解与问答

RunID：`{out.name}`；{state}；{SCOPE}；历史独立性UNKNOWN。

## 一句话说明

这次真正运行GPU，核验完成{len(cells)}/4格，用同一批1610图、同一份Gaussian像素比较旧CONTROL与GUIDED权重，没有再训练。所有数值以[科学报告]({report.as_posix()})和原始CSV为准。

## 为什么不叫独立验证

图像ID和字节与当前train/val不重叠是已验证事实；旧人工调参、test-dev使用与真实场景来源没有完整证据。作者允许有限探索，不是导师签字，也不是独立性证明。

## 怎样理解四格

先分别看Normal/Gaussian的绝对AP，再看GUIDED−CONTROL和配对区间；Normal下降就是clean代价，不能略去。Tiny、AR、逐类结果另看，不能只看总体AP。训练p=.25、sigma5.5–30保持历史配置；测试是每张图固定sigma15。

## Bootstrap能证明什么

200次按同一图像索引重新累计COCO AP，只反映固定seed42模型在这些图像上的抽样变化。它不是200次训练，也不能替代三个训练seed或严格外部确认。图像可能有场景相关，点态区间没有家族多重校正。

## Gap为什么要另算

导师条件是Tiny至少.10、同口径缓存管线最多100ms；必须同时检查区间规则。整体AP收益不能代替Tiny/时延达标。这里只观察两个P条件和两份D，不能推断全UVL空间的Gap率。

## 运行结构与文件

`run.py`负责唯一执行、原账本计费及安全监督；`backend.py`负责共享PNG、真实预测、COCO和配对统计；`summarize_execution.py`只读取终态做核验/制图/解释。原始预测按格保留，GT共享引用，无重复训练。

打开[{out.name}]({(out/'index.html').as_posix()})查看表图。`configuration_metrics.csv`是面积汇总，`per_class_metrics.csv`是十类分层，`paired_comparisons.csv`是差值与区间，`timing_by_image.csv`是逐图阶段时间，`gap_status.csv`是固定规则结果。`source_provenance.json`记录SHA，`resource_usage.json`区分实测与未知成本。

## 可以和VM39放一起讲吗

可以并排展示开发548图与外部探索1610图的收益方向和区间，但不能混成同一总体AP，也不能把旧inference_ms拼成新缓存时延。seed42与三seed开发均值/SD也要分开。

## 接下来是否继续训练

本任务到此停止，不用这次test-dev结果改p、sigma、权重或epoch。未来独立确认、更多训练seed及正式1280/100epoch仍需要独立设计、资源与授权。
'''
    if complete:
        headline = []
        for context, size in [('Gaussian_s15', 'all'), ('Gaussian_s15', 'tiny'), ('Normal', 'all')]:
            row = next(r for r in comparisons if r['condition'] == context and r.get('size') == size and r['metric'] == 'map50_95')
            headline.append([context, size, f"{float(row['difference'])*100:+.3f} pp", ci_text(row['ci95'])+' pp'])
        summary = '\n## 本次实测应当怎样讲\n\n'+table(['条件', '目标', 'GUIDED−CONTROL', '图像95%区间'], headline)
        summary += '\n\n上述差值是这批1610图上的真实测量。先讲方向和幅度，再讲区间及Normal代价；不能把图像区间升级为训练seed显著性或独立泛化。Gap状态另看固定Tiny/时延表，不用AP增益代替。\n'
        guide_text += summary
    write_text(guide, guide_text)
    paths.extend([report, guide])
    resource['CPU_postprocessing_seconds_before_final_receipt'] = time.perf_counter()-started
    save(out/'resource_usage.json', resource)
    save(out/'scientific_acceptance.json', {'status': state, 'checked_UTC': datetime.now(timezone.utc).isoformat(), 'scope': SCOPE,
        'historical_independence': 'UNKNOWN', 'training_seed': 42, 'verified_cells': len(cells), 'formal_measured_images': sum(r['images'] for r in evidence),
        'cell_verification': evidence, 'original_execution_seal': check, 'original_preparation_seal': old_check,
        'paired_differences_reproduced': complete, 'new_COCO_or_GPU_run': False, 'reports': [source(p) for p in paths],
        'resource': source(out/'resource_usage.json'), 'original_run_json_initial_status_preserved': True, 'retry': False})
    diary = f'''# VM43 execution diary

- RunID: {out.name}
- Scope: {SCOPE}; historical independence UNKNOWN.
- Goal: frozen seed42 Normal/Gaussian CONTROL/GUIDED evaluation on1610 images.
- Result: {state}; {len(cells)}/4 verified cells. No retraining or retuning.
- Evidence: final_acceptance.json, scientific_acceptance.json, raw cell files and both checksum manifests.
- Cost: {resource['this_authorization_total_charged_wall_seconds']:.6f} supervised worker seconds including preflight; CPU report generation separate.
- Limits: one seed, post-hoc development policy, pointwise image intervals, no reliable flight clusters.
- Next: report evidence; no automatic new GPU workload.
'''
    write_text(out/'diary.md', diary)
    picture_html = ''.join(f'<figure><img src="plots/{p.name}" style="max-width:100%"><figcaption>{html.escape(p.stem)}</figcaption></figure>' for p in plots if p.suffix == '.png')
    docs = ''.join(f'<li><a href="{p.as_uri()}">{html.escape(p.name)}</a></li>' for p in paths)
    links = ''.join(f'<li><a href="{p.name}">{html.escape(p.name)}</a></li>' for p in sorted(out.glob('*.csv')))
    write_text(out/'index.html', f'<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>VM43 actual evaluation</title><style>body{{font:16px system-ui;color:#243b4d;max-width:1050px;margin:30px auto;padding:0 20px}}code{{overflow-wrap:anywhere}}a{{color:#187b91}}figure{{margin:30px 0}}</style><h1>VM43 · {state}</h1><p><code>{out.name}</code></p><p>{len(cells)}/4 verified cells · {SCOPE} · independence UNKNOWN · seed42</p><p>Terminal status comes from final_acceptance.json and scientific_acceptance.json; original launch metadata is preserved.</p><ul>{docs}{links}<li><a href="scientific_acceptance.json">Verified receipt</a></li><li><a href="source_provenance.json">Source SHA and raw evidence</a></li><li><a href="resource_usage.json">Measured resources and ledger</a></li></ul>{picture_html}</html>')
    covered = set()
    with (out/'SHA256SUMS.txt').open(encoding='utf-8') as f:
        covered = {Path(line.rstrip('\n').split('  ', 1)[1]).resolve() for line in f}
    additions = [p for p in out.rglob('*') if p.is_file() and 'runtime' not in p.relative_to(out).parts and p.resolve() not in covered and p.name != 'analysis_SHA256SUMS.txt']+paths
    write_text(out/'analysis_SHA256SUMS.txt', ''.join(sha(p)+'  '+str(p)+'\n' for p in sorted(additions)))
    manifest_check(out/'analysis_SHA256SUMS.txt')
    print(json.dumps({'RunID': out.name, 'status': state, 'verified_cells': len(cells), 'report': str(report), 'index': str(out/'index.html'),
                      'budget_remaining_seconds': ledger['remaining_seconds'], 'CPU_postprocessing_wall_seconds': time.perf_counter()-started}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    main(args.run)
