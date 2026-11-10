"""Factual reports for each complete joint-coverage measurement."""
import json
from protocol import ROOT,load,save,source,write_atomic


def report_pair(out,folder,spec,number,timing):
    metrics=load(folder/'metrics.json');boot=load(folder/'bootstrap.json');plan=load(out/'EXECUTION_PROTOCOL.json')
    headline=next(r for r in metrics['per_area'] if r['size']=='all')
    ci=next(r['ci95'] for r in boot['intervals'] if r['size']=='all' and r['metric']=='map50_95')
    lines=['# VM36联合覆盖：完整开发测量','',
        'PROBLEM：固定checkpoint在所选复合退化下是否仍满足既定阈值场景；本组独立测量不能识别单个因素因果。',
        'WHY IS IT A PROBLEM：该点来自冻结贪心覆盖清单，补充两域投影中的状态见证，不按AP挑选。',
        'SOLUTION：完整548图、同原始标注及ignore规则、原七阶段退化、FP32、既有dense NMS与COCO评价；原预测复算一致。',
        'WHY IS IT A GOOD SOLUTION：实际AP/AR、配对图像区间和成本可追溯；单组成功或失败均保留，不推断独立测试或架构因果。','',
        'Dataset: VisDrone2019-DET validation；548张；DEVELOPMENT_ONLY。',
        'Checkpoint: '+spec['d']['checkpoint']+' / '+spec['d']['sha256'],
        'Training: FT1280；历史seed42；选中epoch及部分创建期来源仍未知。',
        'Inference: '+str(spec['d']['resolution'])+'；FP32；batch1；pr01原生十类，原坐标面积分箱。',
        'Features: '+', '.join(spec['p']['leaves']),
        'Parameters: '+json.dumps(spec['p']['attributes'],ensure_ascii=False,sort_keys=True),
        'Seed: '+str(plan['seed'])+'；Bootstrap: 200次相同原图位置重采样，点态95%区间。','',
        '| 尺寸 | AP50 | AP50:95 | AR50:95 | AP95%区间 |','|---|---:|---:|---:|---|']
    for row in metrics['per_area']:
        interval=next(r['ci95'] for r in boot['intervals'] if r['size']==row['size'] and r['metric']=='map50_95')
        lines.append(f'|{row["size"]}|{row["map50"]}|{row["map50_95"]}|{row["ar50_95"]}|{interval}|')
    lines+=['',f'Complete-pair wall before report/seal: {timing["wall_seconds"]:.3f}秒，包含该配方全548图生成、推理、评价、原预测复算及Bootstrap。',
        f'Cached-input pipeline mean: {timing["mean_pipeline_ms"]:.3f} ms；Torch峰值allocated/reserved: {timing["peak_allocated_bytes"]}/{timing["peak_reserved_bytes"]} bytes。',
        'Timing boundary: 缓存PNG核验/解码、同步predictor/NMS及原始预测写入；不是标准化机载部署时延，也不是总GPU占用。',
        'Limitations: 共享历史选模val、固定权重单训练seed、场景相关和重复字节；无训练seed或航次聚类推断。',
        'Evidence: '+str(folder),
        'Reproduction: CPU从该目录predictions.jsonl.gz及ground_truth.json调用原evaluation.evaluate_saved；GPU命令记录在本Run command.txt，不能据此自动重复执行。']
    cn=ROOT/'datasets/reportcn'/f'vm36_supervisor_reproduction_joint_{out.name.split("_")[-1]}_pair{number:02d}.md'
    write_atomic(cn,'\n'.join(lines)+'\n',out)
    save(folder/'REPORT_REFERENCE.json',source(cn),out)
    english='# Full548 joint-coverage measurement\n\nPROBLEM: Characterize a coverage-selected compound condition for a fixed checkpoint.\n\nWHY: The frozen feature-only design needs this state witness.\n\nSOLUTION: All548 validation images; original GT/ignore and pr01; saved predictions replay exactly.\n\nEVIDENCE: AP50='+str(headline['map50'])+', AP50:95='+str(headline['map50_95'])+', AR50:95='+str(headline['ar50_95'])+', pointwise image CI='+str(ci)+'.\n\nDevelopment only; no architecture, training-seed or independent-scene claim. See configuration.json, metrics.json, bootstrap.json and timing.json.\n'
    write_atomic(folder/'report.md',english,out)
