"""Native-pixel effects and a private label view for the installed trainer."""
from collections import OrderedDict
import hashlib
import math
from pathlib import Path
import cv2
import numpy as np
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import load_dataset_cache_file,get_hash
from ultralytics.models.yolo.detect.train import DetectionTrainer
from .training_effects import NativeEffects


def resize_native(image,size,rect_mode=True,resize_short=False):
    h0,w0=image.shape[:2]
    if rect_mode:
        r=size/(min(h0,w0) if resize_short else max(h0,w0))
        if r!=1:
            if resize_short:w,h=(math.ceil(w0*r),size) if h0<w0 else (size,math.ceil(h0*r))
            else:w,h=min(math.ceil(w0*r),size),min(math.ceil(h0*r),size)
            image=cv2.resize(image,(w,h),interpolation=cv2.INTER_LINEAR)
    elif (h0,w0)!=(size,size):image=cv2.resize(image,(size,size),interpolation=cv2.INTER_LINEAR)
    return image,(h0,w0),image.shape[:2]


class RQ4Dataset(YOLODataset):
    def __init__(self,*args,label_lookup,evidence,recipe,seed,engineering=False,label_cache=None,**kwargs):
        self.private_labels=label_lookup;self.evidence=Path(evidence);self.evidence.mkdir(parents=True,exist_ok=True)
        self.private_cache=Path(label_cache) if label_cache else self.evidence/'labels.cache'
        self.effects=NativeEffects(recipe,seed,engineering);self.native_cache=OrderedDict();self.native_hashes={}
        self.epoch_number=0;self.occurrences={};self.effect_events=[]
        if kwargs.get('cache'):raise ValueError('The native-effect adapter requires cache=False.')
        super().__init__(*args,**kwargs)
        self.npy_files=[self.evidence/'unused_npy'/(Path(f).stem+'.npy') for f in self.im_files]

    def get_labels(self):
        self.label_files=[self.private_labels[Path(f).stem] for f in self.im_files]
        if self.private_cache.exists():
            cache=load_dataset_cache_file(self.private_cache)
            if cache['hash']!=self.get_cache_hash():raise ValueError('Private label-cache identity differs.')
        else:cache=self.cache_labels(self.private_cache)
        labels=cache['labels']
        if len(labels)!=len(self.im_files):raise ValueError('The library rejected an image or label.')
        self.im_files=[label['im_file'] for label in labels]
        return labels

    def set_epoch(self,epoch):
        self.epoch_number=int(epoch);self.occurrences={}

    def load_image(self,i,rect_mode=True,resize_short=False):
        if i not in self.native_cache:
            path=Path(self.im_files[i]);image=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),self.cv2_flag)
            if image is None:raise ValueError('Cannot decode native training pixels: '+str(path))
            self.native_hashes[i]=hashlib.sha256(image.tobytes()).hexdigest()
            image.setflags(write=False)
            self.native_cache[i]=image
            if self.augment:
                self.buffer.append(i)
                if 1<len(self.buffer)>=self.max_buffer_length:
                    j=self.buffer.pop(0);self.native_cache.pop(j,None);self.native_hashes.pop(j,None)
            elif len(self.native_cache)>1:
                j,_=self.native_cache.popitem(last=False);self.native_hashes.pop(j,None)
        image=self.native_cache.get(i)
        if image is None:raise ValueError('Native cache evicted the current image.')
        count=self.occurrences.get(i,0);self.occurrences[i]=count+1
        before=self.native_hashes[i]
        changed,trace=self.effects.apply(image,Path(self.im_files[i]).stem,self.epoch_number,count)
        if image.flags.writeable:raise ValueError('Cached original must remain read-only.')
        result=resize_native(changed,self.imgsz,rect_mode,resize_short)
        effect_hash=hashlib.sha256(changed.tobytes()).hexdigest() if any(e['active'] for e in trace) else before
        self.effect_events.append({'id':Path(self.im_files[i]).stem,'epoch':self.epoch_number,'occurrence':count,
            'native_shape':list(image.shape),'native_pixel_SHA256':before,'effect_pixel_SHA256':effect_hash,
            'resized_shape':list(result[0].shape),'effects':trace})
        return result


def trainer_class(context):
    class RQ4DetectionTrainer(DetectionTrainer):
        def build_dataset(self,img_path,mode='train',batch=None):
            if self.args.workers!=0:raise ValueError('Worker/prefetch epoch semantics are not admitted; CPU engineering uses workers=0.')
            stride=max(int(self.model.stride.max()),32)
            return RQ4Dataset(img_path=img_path,imgsz=self.args.imgsz,batch_size=batch,augment=mode=='train',
                hyp=self.args,rect=self.args.rect or mode=='val',cache=False,single_cls=self.args.single_cls,
                stride=stride,pad=0. if mode=='train' else .5,task=self.args.task,classes=self.args.classes,
                data=self.data,fraction=self.args.fraction if mode=='train' else 1.,
                label_lookup=context['labels'],evidence=context['evidence'],recipe=context['recipe'],
                seed=context['seed'],engineering=context['engineering'],label_cache=context.get('label_cache'))

        def train(self):
            raise RuntimeError('RQ4 training is not admitted. This version validates the production dataset hook on CPU only.')

    return RQ4DetectionTrainer
