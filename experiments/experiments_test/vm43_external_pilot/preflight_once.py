"""One authorized engineering probe; never publishes scientific AP."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time

from vm43_common import ROOT,HERE,RUNS,load,save,sha,source,study,frozen
sys.path.insert(0,str(HERE))
import backend


def probe(out):
    plan=load(out/'preflight_authorization.json')
    if plan['scope']!='AUTHORIZED_VM43_ENGINEERING_PREFLIGHT_ONLY' or plan['GPU_cap_seconds']!=600:
        raise PermissionError('Wrong preflight scope.')
    if not os.environ.get('VM43_PREFLIGHT_TOKEN') or hashlib.sha256(os.environ['VM43_PREFLIGHT_TOKEN'].encode()).hexdigest()!=plan['worker_token_SHA256']:
        raise PermissionError('Owned supervisor token required.')
    if sha(plan['user_instruction']['path'])!=plan['user_instruction']['SHA256']:raise ValueError('Author instruction changed.')
    study.runtime(out,cuda=True)
    import cv2,numpy as np,torch
    spec=frozen();population=load(spec['population']['path']);ops=backend.renderer()
    torch.set_num_threads(2);cv2.setNumThreads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    chosen=sorted(range(len(population['images'])),key=lambda i:(-population['images'][i]['width']*population['images'][i]['height'],population['images'][i]['image_id']))[:6]
    archive=backend.OriginalZIP(population,population['source_ZIP_accepted_certificate']);images={};cpu=[]
    folder=out/'CPU_cache_probe';folder.mkdir();began=time.perf_counter()
    try:
        for i in chosen:
            image,record=archive.record(i,decode=True)
            for name,attrs in {r['condition']:r['P_attributes'] for r in spec['cells']}.items():
                start=time.perf_counter();pixels,trace=ops.generate(image,list(attrs),attrs,record['stem'],20261116)
                repeated,again=ops.generate(image,list(attrs),attrs,record['stem'],20261116)
                if not np.array_equal(pixels,repeated) or trace!=again:raise ValueError('Nondeterministic prospective pixels.')
                ok,encoded=cv2.imencode('.png',pixels)
                if not ok:raise RuntimeError('PNG encode failure.')
                file=folder/(record['stem']+'_'+name+'.png');file.write_bytes(encoded.tobytes())
                digest=sha(file);readback=cv2.imdecode(np.frombuffer(file.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
                if not np.array_equal(pixels,readback):raise ValueError('PNG changed pixels.')
                images[record['stem'],name]=(pixels,record)
                cpu.append({'image_id':record['stem'],'condition':name,'native_shape':list(pixels.shape),
                    'seconds':time.perf_counter()-start,'encoded_bytes':len(encoded),'file_SHA256':digest,
                    'pixel_SHA256':hashlib.sha256(pixels.tobytes()).hexdigest(),'effects':trace})
    finally:archive.close()
    save(out/'CPU_cache_probe.json',{'scope':'ENGINEERING_ONLY_NO_AP','selection':'Six largest native pixel areas; deterministic ties by ID, no AP read',
        'rows':cpu,'wall_seconds':time.perf_counter()-began,'same_pixels_used_for_both_models':True})
    models={};loads=[];timings=[]
    for strategy in ('CONTROL','VARIABILITY_GUIDED'):
        cell=next(r for r in spec['cells'] if r['strategy']==strategy)
        if sha(cell['checkpoint']['path'])!=cell['checkpoint']['SHA256']:raise ValueError('Checkpoint changed.')
        start=time.perf_counter();models[strategy]=backend.RealDetector(cell['checkpoint']['path'])
        loads.append({'strategy':strategy,'model_initialization_seconds':time.perf_counter()-start,'checkpoint':cell['checkpoint']})
    for strategy,adapter in models.items():
        first=next(iter(images.values()))[0];warm=time.perf_counter()
        for _ in range(3):adapter.predict(first)
        adapter.synchronize();loads.append({'strategy':strategy,'three_warmup_seconds':time.perf_counter()-warm})
        for (stem,condition),(pixels,record) in images.items():
            adapter.synchronize();start=time.perf_counter();result=adapter.predict(pixels);adapter.synchronize()
            raw,tensor=adapter.extract(result,record['id']);backend.validate_predictions([{'image_id':record['id'],'predictions':raw}],{record['id']})
            timings.append({'strategy':strategy,'image_id':stem,'condition':condition,'predict_extract_seconds':time.perf_counter()-start,
                'predictions':len(raw),'tensor':tensor,'profiles':adapter.profiles(result),'scope':'ENGINEERING_ONLY_NOT_FORMAL_CELL'})
    save(out/'preflight_worker_result.json',{'status':'ENGINEERING_PROBE_COMPLETED','scope':'THERMAL_AND_COST_ADMISSION_ONLY',
        'model_loads_and_warmups':loads,'prediction_checks':timings,'new_scientific_cells':0,'new_AP':'NOT_EVALUATED',
        'CUDA_runtime':torch.version.cuda,'torch':torch.__version__,'GPU':torch.cuda.get_device_name(0),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
    print('ENGINEERING_PREFLIGHT_COMPLETED',len(timings),'real prediction checks; no AP',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--worker',required=True);a=p.parse_args()
    if Path(a.worker).name!=a.worker or not a.worker.startswith('vm43_external_pilot_'):raise ValueError('Invalid RunID.')
    probe((RUNS/a.worker).resolve(strict=True))
