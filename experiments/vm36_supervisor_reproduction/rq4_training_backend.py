"""CPU-verifiable health and FP32 snapshot hooks; training stays disabled."""
import copy
import argparse
import io
import json
import math
from pathlib import Path
import time
import torch
from protocol import ROOT,sha256


def finite_tree(value):
    if isinstance(value,torch.Tensor):return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite_tree(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite_tree(v) for v in value)
    if isinstance(value,float):return math.isfinite(value)
    return True


def model_health(model,gradients=False):
    for name,value in model.state_dict().items():
        if not finite_tree(value) or name.endswith('running_var') and bool((value<0).any()):
            raise FloatingPointError('Unhealthy model tensor: '+name)
    if gradients:
        for name,param in model.named_parameters():
            if param.grad is not None and not finite_tree(param.grad):raise FloatingPointError('Nonfinite gradient: '+name)


def cpu_state(model):
    return {name:value.detach().cpu().clone() for name,value in model.state_dict().items()}


def same_state(model,expected):
    state=model.state_dict()
    return state.keys()==expected.keys() and all(state[k].dtype==v.dtype and torch.equal(state[k].cpu(),v) for k,v in expected.items())


def same_tree(left,right):
    if isinstance(left,torch.Tensor):return isinstance(right,torch.Tensor) and left.dtype==right.dtype and torch.equal(left.cpu(),right.cpu())
    if isinstance(left,dict):return isinstance(right,dict) and left.keys()==right.keys() and all(same_tree(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):return type(left)==type(right) and len(left)==len(right) and all(same_tree(a,b) for a,b in zip(left,right))
    return left==right


class RQ4HealthHooks:
    def event(self,value):
        out=Path(self.rq4_context['output']).resolve()
        if not out.is_relative_to((ROOT/'datasets/local_runs').resolve()):raise PermissionError('Health evidence must remain local.')
        with (out/'health_events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(value)+'\n')

    def check_batch(self):
        if not finite_tree(self.loss):raise FloatingPointError('Nonfinite training loss.')
        model_health(self.model,gradients=True)

    def optimizer_step(self):
        started=time.perf_counter();self.check_batch()
        model_health(self.ema.ema)
        if not finite_tree(self.optimizer.state_dict()):raise FloatingPointError('Nonfinite optimizer state before update.')
        if any(not math.isfinite(group['lr']) or group['lr']<0 for group in self.optimizer.param_groups):raise FloatingPointError('Invalid learning rate.')
        before=int(self.ema.updates)
        self.scaler.unscale_(self.optimizer)
        norm=torch.nn.utils.clip_grad_norm_(self.model.parameters(),max_norm=10.0)
        if not bool(torch.isfinite(norm)):raise FloatingPointError('Nonfinite gradient norm.')
        actual={'calls':0}
        def counted(*args,**kwargs):
            actual['calls']+=1;self.rq4_optimizer_updates+=1
        handle=self.optimizer.register_step_post_hook(counted)
        try:self.scaler.step(self.optimizer)
        finally:handle.remove()
        if actual['calls']!=1:raise RuntimeError('The scaler did not execute exactly one optimizer step.')
        self.event({'event':'optimizer_step_performed','optimizer_updates':self.rq4_optimizer_updates})
        self.scaler.update();self.optimizer.zero_grad()
        self.ema.update(self.model)
        model_health(self.model);model_health(self.ema.ema)
        if not finite_tree(self.optimizer.state_dict()):raise FloatingPointError('Nonfinite optimizer state after update.')
        if int(self.ema.updates)!=before+1:raise RuntimeError('EMA update count differs.')
        self.event({'event':'optimizer_and_EMA_healthy','optimizer_updates':self.rq4_optimizer_updates,
            'EMA_updates':self.ema.updates,'gradient_norm':float(norm),'learning_rates':[group['lr'] for group in self.optimizer.param_groups],
            'check_and_step_seconds':time.perf_counter()-started})

    def save_model(self):
        started=time.perf_counter();model_health(self.model,gradients=True);model_health(self.ema.ema)
        if not finite_tree(self.optimizer.state_dict()):raise FloatingPointError('Cannot save an unhealthy optimizer.')
        out=Path(self.rq4_context['output']).resolve();weights=out/'weights'
        if not out.is_relative_to((ROOT/'datasets/local_runs').resolve()):raise PermissionError('Checkpoint output is not local.')
        selection=self.rq4_context['selection']
        if selection not in ('FIXED_LAST','BEST_ON_VAL'):raise ValueError('Checkpoint selection is not frozen.')
        weights.mkdir(exist_ok=True)
        target=weights/f'epoch{self.epoch+1:04d}_fp32.pt'
        if target.exists():raise FileExistsError('Refusing to overwrite an epoch checkpoint.')
        temporary=target.with_suffix('.partial')
        if temporary.exists():raise FileExistsError('Refusing to overwrite a partial checkpoint.')
        online=copy.deepcopy(self.model).cpu().float();ema=copy.deepcopy(self.ema.ema).cpu().float()
        online_state=cpu_state(online);ema_state=cpu_state(ema)
        payload={'model':None,'online_model':online,'ema':ema,'epoch':self.epoch,'updates':self.ema.updates,
            'optimizer':copy.deepcopy(self.optimizer.state_dict()),'scheduler':copy.deepcopy(self.scheduler.state_dict()),
            'scaler':copy.deepcopy(self.scaler.state_dict()),'train_args':vars(self.args),
            'train_metrics':{**self.metrics,'fitness':self.fitness},'best_fitness':self.best_fitness,
            'train_results':self.read_results_csv(),'rq4_provenance':self.rq4_context,
            'pending_gradient_tensors':sum(p.grad is not None for p in self.model.parameters()),
            'exact_resume_validated':False,'snapshot_scope':'Weights/optimizer snapshot; RNG, sampler and pending gradients are not an exact-resume protocol.'}
        torch.save(payload,temporary)
        loaded=torch.load(temporary,map_location='cpu',weights_only=False)
        for key,expected in (('online_model',online_state),('ema',ema_state)):
            model_health(loaded[key])
            if any(v.is_floating_point() and v.dtype!=torch.float32 for v in loaded[key].state_dict().values()):raise ValueError('Snapshot is not FP32.')
            if not same_state(loaded[key],expected):raise ValueError('Saved model state differs after reload.')
        if not finite_tree(loaded['optimizer']):raise FloatingPointError('Saved optimizer is nonfinite.')
        for key in ('optimizer','scheduler','scaler'):
            if not same_tree(loaded[key],payload[key]):raise ValueError('Saved training state differs: '+key)
        if loaded['rq4_provenance']!=self.rq4_context:raise ValueError('Snapshot provenance changed.')
        temporary.rename(target);self.last=target
        if selection=='FIXED_LAST' or self.fitness==self.best_fitness:self.rq4_selected=target
        self.event({'event':'FP32_snapshot_reloaded_equal','epoch':self.epoch+1,'checkpoint':str(target),
            'SHA256':sha256(target),'save_reload_seconds':time.perf_counter()-started,'exact_resume_validated':False})
        return True

    def final_eval(self):
        selected=getattr(self,'rq4_selected',None)
        if selected is None:raise RuntimeError('No selected healthy checkpoint.')
        payload=torch.load(selected,map_location='cpu',weights_only=False)
        self.metrics={k:v for k,v in payload['train_metrics'].items() if k!='fitness'}
        self.event({'event':'selected_snapshot_kept_FP32','checkpoint':str(selected),
            'strip_optimizer_called':False,'extra_validation_run':False,'research_COCO_evaluation':'NOT_EVALUATED'})


def epoch_start(trainer):
    if trainer.args.workers!=0:raise RuntimeError('Multiprocess epoch/prefetch semantics have not been admitted.')
    trainer.train_loader.dataset.set_epoch(trainer.epoch)


def register_health_callbacks(trainer):
    trainer.add_callback('on_train_epoch_start',epoch_start)
    trainer.add_callback('on_train_batch_end',lambda active:active.check_batch())


def backend_class(context):
    from rq4_dataset import trainer_class
    parent=trainer_class(context)

    class RQ4Backend(RQ4HealthHooks,parent):
        def __init__(self,*args,**kwargs):
            raise RuntimeError('Backend methods are CPU-tested only; no admitted GPU constructor or training entry exists.')

    return RQ4Backend


def cpu_check(source_run):
    from protocol import EXPERIMENT,HERE,load,save,source,write_atomic
    from common import verify_hashes,seal
    from main.pipeline import new_run
    from rq4_preparation import runtime
    import sys,unittest,shutil
    verify_hashes(source_run,('FINAL_RQ4_STATUS.json','INITIALIZATION_SOURCE_GATE.json'))
    out,_=new_run({'output_root':str(source_run.parent)},EXPERIMENT,'rq4_backend_cpu')
    runtime(out);started=time.perf_counter();shutil.copyfile(Path(__file__),out/'executed_source.py')
    sys.path.insert(0,str(HERE/'tests'));import test_rq4_training_backend as test_module
    stream=io.StringIO();tests=unittest.defaultTestLoader.loadTestsFromTestCase(test_module.BackendTests)
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(tests)
    write_atomic(out/'unit_tests.log',stream.getvalue(),out)
    if not result.wasSuccessful():raise RuntimeError('CPU backend tests failed; preserved log contains details.')
    harness=test_module.BackendTests().harness(out);harness.optimizer_step();harness.save_model();harness.final_eval()
    report={'status':'BACKEND_CPU_VERIFIED_GPU_NOT_ADMITTED','unit_tests':result.testsRun,'unit_failures':len(result.failures),
        'unit_errors':len(result.errors),'persisted_synthetic_CPU_optimizer_steps':harness.rq4_optimizer_updates,
        'persisted_synthetic_CPU_EMA_updates':harness.ema.updates,'FP32_snapshot':source(harness.last),
        'scope':'Synthetic CPU Linear/BatchNorm model, not a YOLO training result.',
        'optimizer_math':'Exact model/EMA state parity with the installed optimizer_step on the same synthetic input.',
        'loss_gradient_BN_failure_tests':'PASS','exact_resume_validated':False,
        'real_YOLO_training_runs':0,'GPU_tasks':0,'real_GPU_training_cost':'NOT_MEASURED','new_AP_AR':'NOT_EVALUATED',
        'GPU_constructor':'DENIED','remaining':['Scientific recipe and selection decisions','Admitted worker policy',
            'Explicit finite GPU authorization','Real YOLO integration/health/save and resource canary']}
    save(out/'BACKEND_CPU_ACCEPTANCE.json',report,out)
    save(out/'inputs.json',{'source_admission':source(source_run/'SHA256SUMS.txt'),'source':source(Path(__file__)),
        'tests':source(Path(test_module.__file__))},out)
    text=['# VM36：RQ4数值健康与FP32保存补充验收','',
        'PROBLEM / WHY：GPU预算未准入时，先验证训练健康保护和保存逻辑，避免把默认FP16保存或失效EMA留到长训练时才发现。','',
        'METHOD：微型CPU Linear＋BatchNorm模型；真实执行合成损失反向传播、AdamW更新、ModelEMA更新、临时保存和重新加载。它不是YOLO训练，不评价检测精度。','',
        f'RESULT：{result.testsRun}项CPU测试通过；与已安装8.4.61原optimizer_step的在线模型和EMA权重逐张量一致。',
        '加入loss/梯度/梯度范数、完整model/EMA、BN running_var、optimizer状态和学习率有限性检查。实际optimizer.step通过post hook计数，不把scaler.step调用自动当作更新成功。',
        '一份持久化合成模型快照：optimizer更新1次、EMA更新1次；在线模型和EMA保存为FP32，临时文件完整重载、状态和来源一致后才提交。NaN梯度、负BN方差和同epoch覆盖均按测试拒绝。',
        'final_eval保留FP32快照，不调用strip_optimizer，也不伪造追加研究评价。epoch回调仅允许workers0；workers2仍未准入。','',
        'CONCLUSION：CPU健康保护与FP32保存方法已实现并验收，推进了训练基础设施。当前GPU构造和train入口仍拒绝执行，不能报告真实YOLO反向传播、显存、速度、收敛或增强收益。','',
        'LIMITATIONS：合成CPU正确性不等于真实YOLO/GPU验收；保存包含model/EMA/optimizer/scheduler/scaler，但不构成精确resume（RNG、sampler及未提交梯度未完整验证）。',
        '在源报告的科学参数、选择规则、worker与新有限GPU预算冻结后，才可另建真实canary入口；本轮没有借用原推理预算。','',
        '完整RQ4数据、初始化、科学决策汇总：'+load(source_run/'REPORT_REFERENCE.json')['path'],
        '本轮证据：'+str(out),'快照：'+str(harness.last),
        '复现：'+str(HERE/'rq4_training_backend.py')+' --cpu-check --source-run '+source_run.name]
    note=ROOT/'datasets/reportcn'/('vm36_supervisor_reproduction_RQ4数值与保存验收_'+out.name.split('_')[-1]+'.md')
    write_atomic(note,'\n'.join(text)+'\n',out);save(out/'REPORT_REFERENCE.json',source(note),out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'RQ4_BACKEND_CPU',**report,
        'CPU_seconds':time.perf_counter()-started},out,replace=True)
    save(out/'NESTED_SHA256.json',{str(path.relative_to(out)):sha256(path) for path in (out/'weights').glob('*') if path.is_file()},out)
    write_atomic(out/'diary.md','# RQ4 CPU backend\n\nSynthetic optimizer/EMA and atomic FP32 snapshot checks only. No YOLO training or GPU use. Real integration and resource validation remain unperformed.\n',out)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>RQ4 CPU health and snapshot checks</h1><p>BACKEND_CPU_VERIFIED_GPU_NOT_ADMITTED</p><p><a href="../'+source_run.name+'/index.html">Complete RQ4 preparation</a></p><p><a href="BACKEND_CPU_ACCEPTANCE.json">Actual CPU acceptance</a></p><p><a href="health_events.jsonl">Health and update records</a></p>',out)
    seal(out)
    from main.visualization import build_run_index
    build_run_index(out.parent,EXPERIMENT)
    latest=load(out.parent/'latest.json');latest.update(latest_RQ4_backend_RunID=out.name,latest_RQ4_results_RunID=source_run.name,
        RQ4_training_status=report['status'],combined_measured_pairs=84,formal_measured_pairs=0)
    save(out.parent/'latest.json',latest,out,replace=True);print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cpu-check',required=True,action='store_true');parser.add_argument('--source-run',required=True)
    args=parser.parse_args()
    from post80_replay import run_path
    cpu_check(run_path(args.source_run))
