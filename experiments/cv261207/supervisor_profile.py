"""CPU-only THOP operation counts for existing measured detector settings."""
import argparse
import copy
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES']=''


def profile(manifest,output):
    import torch
    from thop import profile as thop_profile
    from ultralytics import YOLO,__version__
    from common import load,sha256
    from main.labels import VIS_NAMES
    torch.set_num_threads(2)
    request=load(manifest);rows=[]
    for spec in request['checkpoints']:
        path=Path(spec['path'])
        if sha256(path)!=spec['sha256']:raise ValueError('Checkpoint SHA changed.')
        original=YOLO(path).model.cpu().float().eval()
        if list(original.names.values())!=VIS_NAMES:raise ValueError('Native10 class order differs.')
        parameters=sum(p.numel() for p in original.parameters())
        for resolution in spec['resolutions']:
            net=copy.deepcopy(original)
            if hasattr(net,'end2end'):net.end2end=False
            net.fuse(verbose=False);net.eval()
            if any(p.device.type!='cpu' for p in net.parameters()):raise ValueError('CPU profiling only.')
            tensor=torch.zeros(1,3,resolution,resolution,dtype=torch.float32)
            started=time.perf_counter()
            with torch.inference_mode():macs,_=thop_profile(net,inputs=(tensor,),verbose=False)
            rows.append({'checkpoint':spec['id'],'checkpoint_sha256':spec['sha256'],
                'resolution':resolution,'precision':'FP32','device':'CPU','input_NCHW':list(tensor.shape),
                'checkpoint_parameters':parameters,'fused_registered_parameters':sum(p.numel() for p in net.parameters()),
                'MACs':float(macs),'GFLOPs_2xMAC':2*float(macs)/1e9,
                'profile_wall_seconds':time.perf_counter()-started,'ultralytics':__version__,
                'THOP':importlib.metadata.version('ultralytics-thop'),
                'method':'Actual-size fused dense-head CPU forward; THOP operation convention, 2*MAC. No scaling from a smaller input.',
                'limitations':'Unsupported THOP operators may be omitted; NMS, decode, file I/O and GPU throughput are excluded.',
                'status':'VALIDATED_OPERATION_PROFILE'})
            del net,tensor
        if sha256(path)!=spec['sha256']:raise ValueError('Source checkpoint changed during profiling.')
    temporary=output.with_suffix('.partial')
    with temporary.open('x',encoding='utf-8') as stream:
        stream.write(json.dumps({'status':'PASS','GPU_seconds':0,'new_image_evaluation_pairs':0,
                     'profiles':rows},indent=2,allow_nan=False));stream.flush();os.fsync(stream.fileno())
    if output.exists():raise FileExistsError(output)
    os.link(temporary,output);temporary.unlink()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();profile(args.manifest,args.output)
