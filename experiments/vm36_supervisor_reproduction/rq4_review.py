"""Seal nested CPU evidence and prepare the remaining RQ4 decisions."""
import argparse
from collections import defaultdict
import copy
import gc
import hashlib
import importlib.metadata
from pathlib import Path
import shutil
import time
from protocol import ROOT,HERE,EXPERIMENT,load,save,source,sha256,write_atomic
from common import seal,verify_hashes
from main.pipeline import new_run
from post80_replay import run_path
from full_development import PREP


def state_digest(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(name.encode());digest.update(str(value.dtype).encode());digest.update(str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def review(prepared):
    verify_hashes(prepared,('TRAINING_ADMISSION.json','CANDIDATE_INITIALIZATION_MANIFEST.json','TRAINING_DATA_ACCEPTANCE.json',
        'PRODUCTION_DATASET_CPU_ACCEPTANCE.json','VALIDATION_LOADER_ACCEPTANCE.json','SOURCE_PROTECTION.json'))
    if load(prepared/'TRAINING_ADMISSION.json')['failed_CPU_branches']:raise ValueError('CPU preparation has failed branches.')
    dependencies=load(prepared/'source_SHA256.json')
    for name in ('rq4_preparation.py','rq4_dataset.py','rq4_augmentation.py'):
        relative='experiments/vm36_supervisor_reproduction/'+name
        if sha256(HERE/name)!=dependencies[relative]:raise ValueError('Executed CPU implementation changed: '+name)
    out,_=new_run({'output_root':str(prepared.parent)},EXPERIMENT,'rq4_training_review_cpu')
    shutil.copyfile(Path(__file__),out/'executed_source.py');started=time.perf_counter()
    from rq4_preparation import runtime,state_health
    runtime(out)
    import torch
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.torch_utils import init_seeds
    from supervisor_compliance import numeric_attributes
    from protocol import FMP
    from rq4_augmentation import TEST_VALUES
    numeric=numeric_attributes(FMP.read_text(encoding='utf-8'))
    for field,values in TEST_VALUES.items():
        if set(values)!={value for (feature,name),value in numeric.items() if name==field}:
            raise ValueError('Training exclusions differ from the PDF-aligned test attributes: '+field)
    groups=defaultdict(list)
    for row in load(prepared/'CANDIDATE_INITIALIZATION_MANIFEST.json'):groups[row['architecture']].append(row)
    checks=[]
    for architecture,rows in groups.items():
        if {r['seed'] for r in rows}!={42,43,44}:raise ValueError('Three-seed initialization membership differs.')
        states=[]
        for row in sorted(rows,key=lambda r:r['seed']):
            candidate=row['native10_candidate'];path=Path(candidate['path'])
            if sha256(path)!=candidate['sha256']:raise ValueError('Candidate initialization changed.')
            model=torch.load(path,map_location='cpu',weights_only=False)['model'];state_health(model)
            states.append({'seed':row['seed'],'state_SHA256':state_digest(model),'file':candidate})
            del model;gc.collect()
        if len({r['state_SHA256'] for r in states})!=3:raise ValueError('Seeds produced identical complete tensor states.')
        src=Path(rows[0]['source']['path'])
        if sha256(src)!=rows[0]['source']['sha256']:raise ValueError('COCO source changed.')
        payload=torch.load(src,map_location='cpu',weights_only=False);pretrained=payload.get('ema') or payload['model']
        init_seeds(42,deterministic=True)
        model=DetectionModel(cfg=copy.deepcopy(pretrained.yaml),ch=3,nc=10,verbose=False);model.load(pretrained,verbose=False);model.cpu().float()
        repeated=state_digest(model);expected=next(r['state_SHA256'] for r in states if r['seed']==42)
        if repeated!=expected:raise ValueError('Repeated native10 initialization is not tensor-identical.')
        checks.append({'architecture':architecture,'seeds':states,'three_distinct_tensor_states':True,
            'seed42_repeat_state_equal':True,'source':rows[0]['source'],'purpose':'CPU_INITIALIZATION_ONLY_NOT_TRAINING'})
        del model,pretrained,payload;gc.collect()
    save(out/'INITIALIZATION_REPRODUCIBILITY.json',checks,out)
    bindings=PREP/'IMPLEMENTATION_BINDINGS_PROPOSAL.json';verify_hashes(PREP,('IMPLEMENTATION_BINDINGS_PROPOSAL.json',))
    legacy=load(bindings)['proposed_values_from_historical_implementation']
    versions={}
    for name in ('torch','ultralytics','numpy','opencv-python','albumentations','albucore'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='NOT_INSTALLED'
    packet={'status':'SCIENTIFIC_DECISIONS_REQUIRED_NOT_AUTHORIZED','test_exclusions_verified_against_UVL':True,
        'training_range_candidate_A':{name:{'low':min(values),'high':max(values),'exclude_exact_test_values':list(values),
            'status':'CANDIDATE_CLOSED_HULL_NOT_SELECTED'} for name,values in TEST_VALUES.items()},
        'range_candidate_B':'Broader ranges can place each test severity inside the sampled range; endpoints require a documented decision before training.',
        'probabilities':{name:None for name in ('NoiseAug','MotionBlurAug','ExposureAug','JPEGAug','HazeAug')},
        'noise_mixture_probability':None,'source_bindings':source(bindings),
        'unapproved_historical_motion':legacy['Motion_k21'],'unapproved_historical_haze':legacy['Haze_light'],
        'haze_semantics':'Existing constant-depth proxy, not a calibrated weather/sensor model.',
        'fixture_values':'The CPU fixture airlight210 and motion11..169deg are connectivity tests only; they are not the historical220/0..180 proposal or an approved replacement.',
        'conventional_augmentation':'Installed Albumentations wrapper also logs Blur/MedianBlur/ToGray/CLAHE p=.01. HSV and conventional geometry remain active after target effects.',
        'numeric_holdout_scope':'Exact target-effect parameter exclusion does not guarantee disjoint output pixels after quantization, clipping and conventional HSV composition.',
        'installed_versions':versions,'worker_epoch_callback':'Not integrated for actual training; prototype train() rejects execution.',
        'full_GPU_training_cost':'NOT_MEASURED','new_GPU_allocation':'REQUIRED','scientific_training_results':'NOT_EVALUATED'}
    save(out/'SCIENTIFIC_PARAMETER_PACKET.json',packet,out)
    nested={str(path.relative_to(prepared)):sha256(path) for path in prepared.rglob('*') if path.is_file()}
    save(out/'PREPARATION_OUTPUT_SHA256.json',nested,out)
    save(out/'inputs.json',{'preparation':source(prepared/'SHA256SUMS.txt'),'source':source(Path(__file__)),
        'accepted84':load(prepared/'inputs.json')['accepted84']},out)
    plan=load(prepared/'OPTIMIZER_EXPOSURE_PLAN.json')
    text=['# VM36：RQ4准备验收与参数决策','',
        'PROBLEM / WHY：确认CPU准备证据完整，列出进入真实训练前仍需决定的内容。','',
        'METHOD：仅复核已保存CPU证据；两模型各重建一次seed42初始化，不进行训练、GPU推理或新数据选择。','',
        'RESULT：六份native10初始化按tensor内容计算SHA，同模型三seed状态不同；seed42重建与已保存状态完全相同。文件SHA与tensor状态SHA分别保存，避免仅由序列化文件名差异造成假重复性。',
        f'候选100epoch日程：647100张次、323600个batch、{plan["optimizer_and_EMA_updates"]}次optimizer/EMA更新，warmup9708个batch、稳态累积32。它是8.4.61源码推导计划，不是实际执行计数；不能用323600/32替代warmup阶段计数。',
        '训练数据、候选权重、原始字节val视图和逐次增强记录的完整输出SHA见PREPARATION_OUTPUT_SHA256.json；原6471训练图、548val来源及84组实验保持。','',
        'CONCLUSION：CPU数据和初始化准备通过；实际训练仍BLOCKED。建议先决定最小17次方案（12次三seed主比较＋5次单seed消融），或完整27次方案；两者均未运行，不能省略主比较三个seed。','',
        'LIMITATIONS / 需要作者决定：','',
        '- 范围A是由PDF测试值的最小/最大值构成的候选范围，并排除精确测试值；范围B是更宽的范围。两者均未选择，概率全部为null。',
        '- 旧实现提案：motion angle_min=0、angle_max=180；haze airlight=220。这些不在导师PDF中，历史提案也标为未批准，不能默用。',
        '- CPU技术测试的airlight210、角度11..169只用于确认代码连接，不能带入正式训练配置。',
        '- 需决定fresh matched retrain还是从FT继续；best-on-val还是固定epoch100；leave-one-out还是单增强；workers2的预取/epoch验证或新协议workers0。',
        '- 现有训练器钩子仅CPU验收，epoch回调和完整Health Gate/FP32训练保存尚未接通。默认FP16保存不等于已验收FP32保存。',
        '- 需要新的有限GPU训练授权后，才能按已列64训练/16开发图计划准备真实canary；当前GPU训练成本、显存、更新次数及AP全部NOT_MEASURED/NOT_EVALUATED。',
        '- val仍development，test-dev仍未准入；历史best具体epoch和原始初始化SHA仍UNKNOWN。','',
        '完整CPU报告：'+load(prepared/'REPORT_REFERENCE.json')['path'],'本轮复核：'+str(out),
        '复现：'+str(HERE/'rq4_review.py')+' --prepared-run '+prepared.name+'（CPU，不训练）。']
    note=ROOT/'datasets/reportcn'/('vm36_supervisor_reproduction_RQ4参数与验收_'+out.name.split('_')[-1]+'.md')
    write_atomic(note,'\n'.join(text)+'\n',out);save(out/'REPORT_REFERENCE.json',source(note),out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'RQ4_CPU_REVIEW','status':'CPU_ACCEPTED_TRAINING_BLOCKED',
        'CPU_seconds':time.perf_counter()-started,'GPU_tasks':0,'training_runs':0},out,replace=True)
    write_atomic(out/'diary.md','# RQ4 decision packet\n\nVerified candidate tensor identity independently of file serialization and preserved the complete nested CPU evidence hashes. Missing scientific choices and GPU measurements remain explicit.\n',out)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>RQ4 CPU review</h1><p>CPU_ACCEPTED_TRAINING_BLOCKED</p><p><a href="../'+prepared.name+'/index.html">Data and loader evidence</a></p><p><a href="SCIENTIFIC_PARAMETER_PACKET.json">Parameter decisions</a></p><p><a href="INITIALIZATION_REPRODUCIBILITY.json">Initialization reproducibility</a></p><img style="max-width:100%" src="../'+prepared.name+'/training_class_counts.png">',out)
    seal(out)
    from main.visualization import build_run_index
    build_run_index(out.parent,EXPERIMENT)
    latest=load(out.parent/'latest.json');latest.update(latest_RQ4_preparation_RunID=prepared.name,latest_RQ4_review_RunID=out.name,
        RQ4_training_status='BLOCKED',combined_measured_pairs=84,formal_measured_pairs=0)
    save(out.parent/'latest.json',latest,out,replace=True);print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepared-run',required=True);args=parser.parse_args()
    review(run_path(args.prepared_run))
