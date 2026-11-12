"""Measure ALL_FIVE CPU loading and image effects without a YOLO forward pass."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from study import *


def execute(limit):
    out=create('all_five_loader_cost_cpu');runtime(out,cuda=False);began=time.perf_counter()
    import torch,cv2
    from torch.utils.data import DataLoader,RandomSampler
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.torch_utils import init_seeds
    from main.training_data import RQ4Dataset
    from main.labels import VIS_NAMES
    torch.set_num_threads(2);cv2.setNumThreads(2)
    with (PREP/'TRAINING_DATA_VERIFICATION.csv').open(encoding='utf-8-sig') as stream:population=list(csv.DictReader(stream))
    labels=load(PREP/'PRIVATE_LABEL_LOOKUP.json');policy=load(BASE/'TARGETED_POLICY.json')
    reference=RUNS/'vm37_yolo26_rq4_full_epoch_yolo26s_control_20261111T232439_301fb25d'
    args=load(reference/'protocol.json')['args'];args['epochs']=20
    plan={'scope':'CPU_LOADER_ONLY_NOT_YOLO_TRAINING','training_population':6471,'batch':2,'imgsz':1280,
        'horizon_epochs':20,'measured_epoch':0,'augmentation_seed':42,'order_seed':20261112,'CPU_hard_seconds':limit,
        'dataset_policy':source(PREP/'TRAINING_DATA_VERIFICATION.csv'),'frozen_policy':source(BASE/'TARGETED_POLICY.json'),
        'cost_limit':'This measures CPU loading/transforms, not full ALL_FIVE GPU epoch cost. Order is a separate deterministic benchmark sampler.',
        'GPU_forward_backward':0}
    save(out/'protocol.json',plan,out)
    for row in population:
        if sha256(Path(row['image_path']))!=row['image_SHA256'] or sha256(Path(labels[row['image_id']]))!=row['derived_label_SHA256']:
            raise ValueError('Frozen training input changed.')
    write_atomic(out/'images.txt','\n'.join(row['image_path'] for row in population)+'\n',out)
    init_seeds(42,deterministic=True)
    data=RQ4Dataset(img_path=str(out/'images.txt'),imgsz=1280,batch_size=2,augment=True,hyp=get_cfg(overrides=args),
        rect=False,cache=False,stride=32,pad=0.,task='detect',data={'names':dict(enumerate(VIS_NAMES)),'nc':10,'channels':3},
        label_lookup=labels,evidence=out/'labels_cache',recipe=policy,seed=42,engineering=False)
    data.set_epoch(0)
    loader=DataLoader(data,batch_size=2,num_workers=0,pin_memory=False,collate_fn=RQ4Dataset.collate_fn,
        sampler=RandomSampler(data,generator=torch.Generator().manual_seed(20261112)))
    records=[];ids=[];native_calls=0;status='PARTIAL';iterator=iter(loader)
    with (out/'loader_batches.jsonl').open('x',encoding='utf-8') as log,(out/'augmentation_events.jsonl').open('x',encoding='utf-8') as events:
        for position in range(len(loader)):
            if time.perf_counter()-began>=limit:break
            tick=time.perf_counter();batch=next(iterator);seconds=time.perf_counter()-tick
            shape=list(batch['img'].shape)
            if shape[1:]!=[3,1280,1280] or batch['img'].dtype!=torch.uint8:raise ValueError('CPU loader tensor changed.')
            selected=[Path(path).stem for path in batch['im_file']];ids.extend(selected)
            row={'position':position,'anchor_ids':selected,'NCHW':shape,'loader_seconds':seconds,'native_calls':len(data.effect_events)}
            native_calls+=len(data.effect_events);records.append(row);log.write(json.dumps(row)+'\n');log.flush()
            for event in data.effect_events:events.write(json.dumps(event)+'\n')
            data.effect_events.clear()
            if len(records)%100==0:print('CPU_LOADER_BATCH',len(records),'/',len(loader),flush=True)
        if len(records)==len(loader) and len(ids)==6471 and len(set(ids))==6471:status='COMPLETE'
    save(out/'loader_cost.json',{'status':status,'completed_batches':len(records),'anchor_exposures':len(ids),
        'unique_anchors':len(set(ids)),'native_pixel_load_calls':native_calls,'loader_seconds':sum(row['loader_seconds'] for row in records),
        'worker_seconds':time.perf_counter()-began,'GPU_seconds':0,'training_forward_backward':0,
        'full_GPU_ALL_FIVE_cost':'NOT_MEASURED','source_images_unchanged':all(sha256(Path(row['image_path']))==row['image_SHA256'] for row in population),
        'extrapolation':'Any20-epoch projection is a component estimate, not a measured training result.'},out)
    write_atomic(out/'diary.md','# ALL_FIVE CPU loader cost\n\nMeasured native-pixel stochastic effects and the shared ordinary loader on frozen train images. No detector, GPU forward, loss, optimizer or AP evaluation. Incomplete budgets remain PARTIAL.\n',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'ALL_FIVE_LOADER_COST_CPU','status':status},out,replace=True)
    seal(out);print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--max-seconds',type=int,default=3600);args=parser.parse_args()
    if not 60<=args.max_seconds<=7200:parser.error('CPU bound must be60..7200seconds.')
    execute(args.max_seconds)
