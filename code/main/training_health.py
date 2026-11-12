"""Strict numerical health checks and atomic FP32 training snapshots."""
import copy
import argparse
import io
import json
import math
from pathlib import Path
import time
import torch
from .config import sha256
ROOT=Path(__file__).resolve().parents[3]


def tensor_groups(values):
    groups={}
    for value in values:
        if value.is_floating_point():groups.setdefault((value.device,value.dtype),[]).append(value.detach().reshape(-1))
    return groups.values()


def tensors_finite(values):
    return all(bool(torch.isfinite(torch.cat(group)).all()) for group in tensor_groups(values))


def finite_tree(value):
    tensors=[];finite=True
    def collect(item):
        nonlocal finite
        if isinstance(item,torch.Tensor):tensors.append(item)
        elif isinstance(item,dict):
            for child in item.values():collect(child)
        elif isinstance(item,(list,tuple)):
            for child in item:collect(child)
        elif isinstance(item,float):finite=finite and math.isfinite(item)
    collect(value)
    return finite and tensors_finite(tensors)


def model_health(model,gradients=False):
    state=model.state_dict()
    if not tensors_finite(state.values()):
        for name,value in state.items():
            if not finite_tree(value):raise FloatingPointError('Unhealthy model tensor: '+name)
    variances=[value for name,value in state.items() if name.endswith('running_var')]
    if any(bool((torch.cat(group)<0).any()) for group in tensor_groups(variances)):
        for name,value in state.items():
            if name.endswith('running_var') and bool((value<0).any()):raise FloatingPointError('Unhealthy model tensor: '+name)
    if gradients:
        if not tensors_finite(param.grad for param in model.parameters() if param.grad is not None):
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
