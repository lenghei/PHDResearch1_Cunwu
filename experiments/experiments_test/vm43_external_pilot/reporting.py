"""Chinese CPU acceptance and blocked-execution report; no placeholder metrics."""
import html
import json
from pathlib import Path

from vm43_common import ROOT,HERE,V42,V41,load


def table(headers,values):
    clean=lambda v:str(v).replace('|',' / ').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',
        *['| '+' | '.join(clean(v) for v in r)+' |' for r in values]])


def prepare_report(out):
    tests=load(out/'real_evaluator_test_results.json');admission=load(out/'gpu_resource_admission.json')
    dataset=load(out/'dataset_membership.json');proof=load(out/'pilot_protocol_verified.json')
    folder=ROOT/'datasets/reportcn';report=folder/'VM43_最小外部验证_科学报告_CN.md';guide=folder/'VM43_导师会议讲解与问答_CN.md'
    if report.exists() or guide.exists():
        report=folder/('VM43_最小外部验证_科学报告_CN_'+out.name+'.md');guide=folder/('VM43_导师会议讲解与问答_CN_'+out.name+'.md')
    if report.exists() or guide.exists():raise FileExistsError('Reports for this RunID already exist.')
    def link(name):return f'[{name}]({(out/name).as_posix()})'
    cells=proof['cells'];aliases={('CONTROL','Normal'):'A',('CONTROL','Gaussian_s15'):'B',('VARIABILITY_GUIDED','Normal'):'C',('VARIABILITY_GUIDED','Gaussian_s15'):'D'}
    cell_table=table(['格','策略','条件','Infer / 图数','已有epoch10 EMA SHA256'],
        [[aliases[c['strategy'],c['condition']],c['strategy'],c['condition'],'1280 /1610',c['checkpoint']['SHA256']] for c in cells])
    budget=admission['budget'];gt=load(out/'ground_truth_input.json')['coco'];counts={c['id']:0 for c in gt['categories']}
    for ann in gt['annotations']:counts[ann['category_id']]+=1
    test_table=table(['CPU检查','结论'],[[r['test'],r['status']] for r in tests['tests']])
    text=f'''# VM43 最小外部验证实现与科学准入

RunID：`{out.name}`。状态：**CPU_IMPLEMENTED_AND_UNIT_TESTED_GPU_BLOCKED**。

## 1. 本轮实际执行

完成真实四格评价后端及CPU集成验收，未运行GPU预测、未训练、未计算test-dev AP。没有重复VM40–42历史搜索或全量图像解码。新产物只写VM43本地Run和本目录中文报告，旧UVL/训练/评价输出不修改。

原VM42 Option A文件及SHA原样核对；保持seed42、两策略、Normal/Gaussian15、Infer1280、FP32、batch1、全部1610图、原epoch10 EMA。p=.25及训练σ范围/排除语义没有改动。A/B/C/D是任务中的格标记；执行按原VM42存储顺序固定为Normal的C/G，再Gaussian的C/G，不根据早期AP改变顺序，两个模型各加载一次。

{cell_table}

## 2. 已实现代码

- `backend.py`：按冻结成员直接读取原ZIP；GT解析、确定性退化和共享PNG缓存；实际YOLO26s one-to-many/稳定NMS适配器；原坐标预测与分阶段计时；真实COCO/200次Bootstrap与四格配对比较。
- `vm43_common.py`：来源、输出边界、原账本及科学/资源/热关卡。
- `run.py`：CPU准备与验收、一次性执行票据、受监督worker和原账本追加计费；关卡失败不访问GPU。
- `tests_cpu.py`：明确隔离的合成预测与实际数据接口测试；不是YOLO精度试验。

复用未修改的`main.labels`、`main.statistics`、`StableNMSPredictor`、原Gaussian生成器、已验收COCO重累计实现和既有安全监督器。

## 3. 数据与GT

1610个冻结成员及原ZIP SHA验证通过；读取全部原标注并构建原生十类GT，ignore过滤后有效实例{dataset['GT_valid_instances_after_ignore']}个。此文件是评价输入，不是预测/AP结果。

{table(['原生类别','有效GT数'],[[c['name'],counts[c['id']]] for c in gt['categories']])}

沿用cat0或score0→ignore、cat11→other ignore、cat1–10→native class0–9的原规则；原图坐标和面积不变，原rasterized ignore-union覆盖≥50%过滤GT与预测。没有修补、删除困难图像或重新选择目标成员。

全量图像解码证明复用VM40；新后端只对第一张原图作解析器对照与Gaussian像素集成检查，并重复调用参考解析器核对，实际CPU解码2次，不是重跑1610图审计。GT来源与成员见{link('dataset_membership.json')}、{link('ground_truth_input.json')}。

## 4. CPU验收

**{tests['passed']}项通过，{tests['failed']}项失败**，CPU测试墙钟{tests['CPU_wall_seconds']:.3f}秒。

{test_table}

使用3张明确合成图、合成预测验证真实COCO计算与200次图像Bootstrap。重复图像索引下，已有重累计结果与显式复制GT/预测后重新评价一致；不是单图AP平均。无GT类别保留null。

合成预测只保存在`checks/`并标记SYNTHETIC_CPU_SOFTWARE_TEST_ONLY。其理想预测AP只用于软件断言，不能报告成YOLO/test-dev AP。没有实际网络tensor或CUDA精度验收；真正GPU运行时才逐图强制检查[1,3,1280,1280]和FP32。

## 5. 计时实现

复用`VM41_MATCHED_CACHED_GAP_V1`主边界，从缓存文件完整性开始，到预测及manifest逐图flush结束。单独记录：文件SHA、读取、解码、像素SHA、预预测同步、推理总段、后预测同步、框提取、输出写入、缓存管线总段。

预处理/forward/NMS采用实际Ultralytics三个阶段profile读数，另存原始predict墙钟与CUDA同步墙钟。它们是诊断分量，不能相加伪造总管线。模型加载、PNG缓存生成、三次预热、COCO/Bootstrap及完整校验计入总worker预算，不混入逐图主延迟。

GPU情况下每格固定三次预热，Square Letterbox，dense native10，conf=.001、NMS IoU=.7、class-aware multi-label、max_det3000/max_nms30000。未切换FP16、TF32或head。

固定要求为Tiny AP95≥.10与缓存管线≤100ms，保留原区间规则；不使用旧VM37–39 `inference_ms`或历史均值拼接。只有真实四格完成后才生成每格判定与配对时延区间。

## 6. 统计与输出

主终点：1280 Gaussian AP95的GUIDED−CONTROL。次终点：Gaussian Tiny、Normal AP95差、Gaussian退化损失差、时延差及固定Gap判定。200次、seed20261116，同图像位置配对重采样并重新累计COCO AP；只条件于一个训练seed，不能当三seed训练重复。

真实worker保存各格不可覆盖的原预测、GT来源、NCHW、计时、指标、Bootstrap和SHA，完成全部四格后才生成根目录联合表。不完整/安全停止时保留已完成格与partial文件，主比较标INCOMPLETE，不重试或自动减图。

当前实际AP、Tiny、AR、时延差和Gap变化均为**NOT_EVALUATED**。因此根目录未生成`configuration_metrics.csv`、`paired_comparisons.csv`、`bootstrap_intervals.csv`、`timing_by_image.csv`或`gap_status.csv`，也未画虚构结果图。

“可接受clean损失”未在VM42定义额外数值容忍度；未来只报告真实clean代价，不能临时发明一个阈值将结果判成可接受。Tiny/.10与时延100ms阈值原样保留。

## 7. 科学与资源准入

未收到新的test-dev历史使用、选模/策略暴露或可靠场景来源证据，也没有单独明确的LIMITED_EXTERNAL_EXPLORATORY用途决定。VM42的阻塞继续有效；未把UNKNOWN改成CONFIRMED。本任务的条件式执行要求不等于已经获得那个独立的科学用途决定。

当前未通过的执行条件：

{table(['关卡原因'],[[x] for x in admission['failures']])}

原36小时账本剩{budget['remaining_seconds']:.3f}秒（{budget['remaining_seconds']/3600:.6f}小时），固定截止{budget['deadline_UTC']}。四格规划包络约{admission['planning_envelope_seconds_ESTIMATE']/3600:.3f}小时，仅是估计；完整生成/写盘/GT密度和200次评价成本仍需资源准入，不因包络内就直接开一臂。

保留原90°C即停、87°C连续30秒停、2秒监测、预检温度/剩余显存和所属进程树终止策略。历史89°C不是当前散热验收，本轮没有进行GPU设备试运行或热负载测试。

新GPU使用及计费均0秒，无重置、延长或占用新配额。账本与来源保护核验保存于`execution_inputs.json`及`final_acceptance.json`。

## 8. 未来启动的具体前提与命令

需要一份由导师/作者确认、绑定本准备Run和冻结协议SHA的科学决定：严格独立准入需新primary evidence；若仍不确定，可另外明确允许有限外部探索并保留独立性未知，不能伪称独立确认。还需完整四格成本/原预算下的有效执行授权，以及新的现场/设备热准入。

程序不生成批准文件、不替作者填CONFIRMED。通过关卡后会新建独立execution Run，准备Run保持只读；一次授权只允许一次尝试。以下命令是未来接口，本轮未执行：

```powershell
Set-Location -LiteralPath 'E:\\ForStudy\\code\\uavgap'
.\\.venv_rq4\\Scripts\\python.exe -B -X utf8 .\\main\\experiments\\vm43_external_pilot\\run.py `
  --execute {out.name} `
  --scientific-decision '<documented-scientific-decision.json>' `
  --gpu-authorization '<valid-four-cell-gpu-authorization.json>' `
  --thermal-admission '<current-thermal-admission.json>'
```

三文件字段约束在`vm43_common.gates()`：RunID、冻结协议SHA、明确scope/证据/确认人；GPU cap+停止余量需覆盖完整选项且不超当前余额和原截止；热确认需有有效期限及真实证据。占位路径不是可执行批准。

当前可安全运行的校验：

```powershell
.\\.venv_rq4\\Scripts\\python.exe -B -X utf8 .\\main\\experiments\\vm43_external_pilot\\run.py --verify {out.name}
```

## 9. 下一步科学决定

先决定是否有足够新证据支持INDEPENDENT_ADMITTED，或是否明确接受有限外部探索范围；不要重复同一关键词审计。数据用途、完整预算、授权、热关卡全部通过后才执行四格。完成后比较Gaussian收益方向、绝对AP、Tiny、clean代价和同口径时延与VM39是否一致，不混算开发与外部AP。

即便四格有收益，也只有seed42；可提出后续三seed设计，但不自动执行，不更新Gaussian策略，不宣称正式1280/100epoch RQ4完成。

证据：{link('pilot_protocol_verified.json')}、{link('checkpoint_provenance.json')}、{link('dataset_membership.json')}、{link('real_evaluator_test_results.json')}、{link('scientific_admission.json')}、{link('gpu_resource_admission.json')}、{link('final_acceptance.json')}、`SHA256SUMS.txt`。

本Run：`{out}`。代码：`{HERE}`。未Git pull/rebase/reset/commit/push，未修改旧科学结果。
'''
    report.write_text(text,encoding='utf-8')
    spoken=f'''# VM43 导师会议讲解与问答

RunID：`{out.name}`。

## 这次实际做了什么

“这次把四格验证从一份清单变成了真实评价程序，并做了CPU验收。但没有跑GPU，没有test-dev AP。原因不是软件把开发集结果替过去，而是数据用途和热/资源准入还没满足。”

四格仍是seed42、两策略、Normal与Gaussian15，1280推理，每格1610图；旧epoch10 EMA不变，不重新训练。程序共享同一份PNG像素，避免两模型看到不同噪声。

## CPU通过能说明什么

{tests['passed']}项通过，包含完整标注成员、第一张原图与旧解析器一致、Gaussian可重复、类别/ignore规则、真实COCO计算、200次Bootstrap、损坏SHA和NaN拒绝、四格配对和不覆盖输出。

合成预测能证明接口和公式接通，不能证明YOLO检测能力。检查里的理想AP来自合成框，不是模型在test-dev上的精度。实际网络NCHW/FP32与CUDA只有未来真实运行时才可验收。

全部1610份GT输入已准备好，有效实例{dataset['GT_valid_instances_after_ignore']}个；不重复全图解码，也不生成假预测来填表。

## 旧解析器怎样保留

类0或score0是ignore，类11是other ignore，类1–10变成检测类别0–9。GT和预测都用原来的ignore区域并集覆盖规则，原图坐标与面积不变。没有根据检测难度删图或删目标。

## 为什么计时现在能分开

程序分别留缓存核验、读取、解码、像素SHA、模型predict、取框、写预测/manifest，以及整个管线。预处理、forward、NMS读取实际库的profile，CUDA同步另有墙钟。profile只是诊断分量，不能加几个旧平均值冒充完整延迟。

固定Tiny至少.10，缓存管线最多100ms。只有真实的AP、时延和区间都出来，才给完整PASS/FAIL/UNCERTAIN；不是AP多了5个百分点就说Gap关掉了。

## 为什么GPU还没跑

VM42说的历史使用和场景独立性没有新证据。现在可以选择的科学范围有两个：有足够证据才叫严格独立；否则需要作者/导师单独明确允许“有限外部探索”，并继续说明历史不确定。这个决定不是一句“脚本实现好了”能代替的。

还需要完整四格成本和原预算内的有效执行授权、新的热准入。旧账本目前剩约{budget['remaining_seconds']/3600:.3f}小时，原截止不变；3.477小时只是规划，不是保证完成。历史89°C也不能当这次散热已经过关。

## 现在有没有AP图和CSV

没有新真实预测，就不生成根目录精度结果表或结果图。`checks/`的合成测试数据有清楚标记，不能拿给老师当检测成绩。`scientific_admission.json`和`gpu_resource_admission.json`写出为何还不能执行。

## 以后四格能回答什么

它能回答固定seed42下，外部指定图像上Gaussian AP是否改善、Tiny如何、Normal代价多少、完整时延是否可接受。200次抽图还是一个训练seed，不能说等于新三seed实验。

如果结果与VM39不同，也原样保留，不能用test-dev再改p或sigma。不明确clean容忍度时，只报告代价，不临时划线说“这点下降可接受”。正式100轮研究也没有因此完成。

## 文件和运行关系

`run.py`组织准备、验收与受控执行；`backend.py`负责原ZIP、缓存、模型与统计；`vm43_common.py`负责冻结来源和关卡；`tests_cpu.py`只做软件测试。中文科学说明：[{report.name}]({report.as_posix()})。

打开本Run的[index.html]({(out/'index.html').as_posix()})查看验收和阻塞，不需要记长ID。运行`run.py --verify <RunID>`只核验文件；`--prepare`会新建CPU准备Run。没有三个真实有效的准入文件，`--execute`会拒绝，程序不会帮作者伪造批准。
'''
    guide.write_text(spoken,encoding='utf-8');return [report,guide]


def page(out,paths):
    tests=load(out/'real_evaluator_test_results.json')
    docs=''.join(f'<li><a href="{p.as_uri()}">{html.escape(p.name)}</a></li>' for p in paths)
    checks=''.join(f'<tr><td>{html.escape(r["test"])}</td><td>{r["status"]}</td></tr>' for r in tests['tests'])
    (out/'index.html').write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>VM43 acceptance</title>
<style>body{{font:16px system-ui;max-width:1050px;margin:32px auto;padding:0 20px;color:#243b4d}}td,th{{padding:8px 12px;text-align:left;border-bottom:1px solid #ddd}}a{{color:#226d98}}code{{word-break:break-all}}</style>
<h1>VM43 · CPU implementation acceptance</h1><p><code>{html.escape(out.name)}</code></p><p>CPU: {tests['passed']} passed. GPU: BLOCKED / NOT RUN. External AP: NOT_EVALUATED.</p>
<p>No scientific result charts or empty success tables were generated.</p><ul>{docs}</ul>
<p><a href="scientific_admission.json">Scientific admission</a> · <a href="gpu_resource_admission.json">Resource and thermal gates</a> · <a href="real_evaluator_test_results.json">CPU tests</a></p>
<table><tr><th>CPU check</th><th>Status</th></tr>{checks}</table></html>''',encoding='utf-8')
