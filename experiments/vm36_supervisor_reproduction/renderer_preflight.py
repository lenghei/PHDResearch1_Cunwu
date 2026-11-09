"""CPU rendering preflight for the selected joint-coverage candidates."""
import argparse
import hashlib
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
from protocol import ROOT,HERE,EXPERIMENT,FMP,load,save,source,write_atomic
from common import seal,verify_hashes
from historical_review import table,csv_file
from full_development import PREP
from main.pipeline import new_run


def worker(out):
    from archives import OriginalZIP
    from development_v1 import generator
    from alignment import parameters,require_runtime_bindings
    started=time.perf_counter();request=load(out/'request.json')
    design=Path(request['design']);verify_hashes(design,('JOINT_COVERAGE_PLAN.csv','DESIGN_ACCEPTANCE.json'))
    rows=[r for r in table(design/'JOINT_COVERAGE_PLAN.csv') if r['role']=='PROPOSED_NOT_RUN']
    recipes={r['p_id']:r['leaves'].split('|') for r in rows}
    proposal=load(PREP/'IMPLEMENTATION_BINDINGS_PROPOSAL.json')
    archive=OriginalZIP(ROOT/'datasets/VisDrone/VisDrone2019-DET-val.zip',548);renderer=generator();results=[]
    try:
        for index in request['source_indices']:
            image,record,proof=archive.record(index)
            for identity,leaves in recipes.items():
                attributes=require_runtime_bindings(parameters(FMP,leaves),proposal)
                if 'RollingShutter' in leaves:raise ValueError('Unadmitted geometry.')
                t=time.perf_counter();pixels,trace=renderer.generate(image,leaves,attributes,record['stem'],20261108);elapsed=time.perf_counter()-t
                if pixels.shape!=image.shape or pixels.dtype!=np.uint8:raise ValueError('Image frame/dtype changed.')
                if index==request['source_indices'][0]:
                    repeated,again=renderer.generate(image,leaves,attributes,record['stem'],20261108)
                    if not np.array_equal(pixels,repeated) or trace!=again:raise ValueError('Recipe is not deterministic.')
                results.append({'p_id':identity,'image_id':record['id'],'source':proof,'parameters':attributes,
                    'pixel_SHA256':hashlib.sha256(pixels.tobytes()).hexdigest(),'shape':list(pixels.shape),
                    'generation_seconds':elapsed,'status':'CPU_RENDER_ONLY_NO_AP'})
    finally:archive.close()
    csv_file(out/'RENDER_TIMING.csv',[{k:v for k,v in r.items() if k not in ('source','parameters')} for r in results])
    save(out/'PIXEL_PROVENANCE.json',results,out)
    save(out/'result.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'status':'CPU_PREFLIGHT_PASS',
        'planned_new_detector_pairs':len(rows),'unique_degradation_recipes':len(recipes),'source_images':len(request['source_indices']),
        'rendered_image_conditions':len(results),'new_GPU_forwards':0,'new_AP_values':0,
        'deterministic_repeat':'First fixed source image for each recipe','worker_seconds':time.perf_counter()-started,
        'limitations':'Six fixed image positions are engineering checks, not full548 AP evaluation or proof of every scene/geometry case.',
        'next_gate':'A separate finite GPU budget and first complete-pair cost check for the new recipe family remain required.'},out)


def supervise(design):
    from development_guard import bounded_run
    out,_=new_run({'output_root':str(design.parent)},EXPERIMENT,'renderer_preflight')
    save(out/'request.json',{'design':str(design),'source_indices':[0,100,200,300,400,500],
        'selection':'Fixed positions before rendering; no AP used.','CPU_hard_seconds':300},out)
    save(out/'inputs.json',{'plan':source(design/'SHA256SUMS.txt'),'FMP':source(FMP),
        'generator':source(HERE.parent/'cv261109/generator.py'),'worker':source(Path(__file__))},out)
    status='BLOCKED'
    try:
        with (out/'worker.log').open('x',encoding='utf-8') as stream:
            code=bounded_run([sys.executable,'-B',str(Path(__file__)),'--worker',str(out)],stream,300,
                dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',PYTHONIOENCODING='utf-8'))
        if code:raise RuntimeError('CPU worker failed; no retry.')
        status=load(out/'result.json')['status']
    except BaseException:save(out/'failure.json',{'traceback':traceback.format_exc()},out)
    write_atomic(out/'diary.md','# Rendering preflight\n\nFixed image positions and original UVL attributes; CPU pixel generation only. No GPU/AP evaluation or model training. No full-split scientific conclusion.\n',out)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>VM36 rendering preflight</h1><p>'+status+'</p><a href="result.json">Result</a> · <a href="RENDER_TIMING.csv">CPU generation costs</a>',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'CPU_RENDERER_PREFLIGHT','status':status},out,replace=True)
    seal(out);print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--design-run');parser.add_argument('--worker',type=Path);args=parser.parse_args()
    if args.worker:worker(args.worker)
    elif args.design_run and Path(args.design_run).name==args.design_run:
        supervise(ROOT/'datasets/local_runs'/EXPERIMENT/args.design_run)
    else:parser.error('A plain design RunID is required.')
