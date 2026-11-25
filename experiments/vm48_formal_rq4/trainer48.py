"""Accepted trainer with non-sampling Albumentations activation telemetry."""
import json
from pathlib import Path
import numpy as np
from common48 import v3
_BASE_FACTORY=v3.trainer_factory


def serial(value):
    if isinstance(value,np.ndarray):
        return value.tolist() if value.size<=64 else {'shape':list(value.shape),'dtype':str(value.dtype)}
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):serial(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [serial(v) for v in value]
    return value if value is None or isinstance(value,(str,int,float,bool)) else repr(value)


class RecordingCompose:
    def __init__(self,compose,events):
        self.compose=compose;self.events=events
        if not hasattr(compose,'save_applied_params'):raise RuntimeError('Albumentations activation tracing is unavailable.')
        compose.save_applied_params=True
    def __getattr__(self,name):return getattr(self.compose,name)
    def __call__(self,*args,**kwargs):
        result=self.compose(*args,**kwargs)
        self.events.append({'applied_transforms':serial(result.get('applied_transforms',[]))})
        return result


def install_albu(transform,events,seen=None):
    from ultralytics.data.augment import Albumentations
    seen=set() if seen is None else seen
    if id(transform) in seen:return []
    seen.add(id(transform));config=[]
    if isinstance(transform,Albumentations):
        active=transform.transform
        config.append({'class':'Albumentations','available':active is not None,'gate_probability':transform.p,
                       'transforms':[{'class':type(x).__name__,'probability':x.p} for x in active.transforms] if active is not None else []})
        if active is not None and not isinstance(active,RecordingCompose):transform.transform=RecordingCompose(active,events)
    for child in getattr(transform,'transforms',()):config.extend(install_albu(child,events,seen))
    child=getattr(transform,'pre_transform',None)
    if child is not None:config.extend(install_albu(child,events,seen))
    return config


def trainer_factory(plan,out):
    Parent=_BASE_FACTORY(plan,out)
    class Trainer(Parent):
        def build_dataset(self,*args,**kwargs):
            data=super().build_dataset(*args,**kwargs)
            data.vm48_albu_events=[]
            config=install_albu(data.transforms,data.vm48_albu_events)
            path=Path(out)/('ordinary_augmentation_'+('train' if data.augment else 'val')+'.json')
            if path.exists():raise FileExistsError('Ordinary augmentation configuration cannot be silently rebuilt.')
            path.write_text(json.dumps({'actual_pipeline':config,'custom_recipe':plan['targeted_recipe'],'scope':'Observed pipeline, not assumed activation'},indent=2),encoding='utf-8')
            return data
        def _close_dataloader_mosaic(self):
            super()._close_dataloader_mosaic()
            data=self.train_loader.dataset
            config=install_albu(data.transforms,data.vm48_albu_events)
            path=Path(out)/('ordinary_augmentation_mosaic_closed_epoch'+str(self.epoch+1)+'.json')
            with path.open('x',encoding='utf-8') as f:json.dump({'actual_pipeline':config,'mosaic_closed':True,'epoch':self.epoch+1},f,indent=2)
        def batch_end(self,trainer):
            super().batch_end(trainer)
            data=self.train_loader.dataset
            with (Path(out)/'ordinary_albu_events.jsonl').open('a',encoding='utf-8') as f:
                for event in data.vm48_albu_events:f.write(json.dumps({'epoch':self.epoch+1,'batch':len(self.rows),**event})+'\n')
            data.vm48_albu_events.clear()
    return Trainer
