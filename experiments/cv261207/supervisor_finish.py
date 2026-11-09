"""Complete the bounded CPU evidence audit; never start GPU work."""
import argparse
from datetime import datetime,timezone
import hashlib
import html
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time

from common import ROOT,HERE,PUBLIC,FMP,FMD,VAL,load,source,sha256,resolve_run,import_file,seal
from supervisor_compliance import VERSION,BASE,DEV,STUDY,matrix,save,write_atomic
from historical_review import table,csv_file
from development_guard import bounded_run


def remaining(out):
    stamp=re.search(r'_(\d{8}T\d{6})_',out.name)[1]
    start=datetime.strptime(stamp,'%Y%m%dT%H%M%S').replace(tzinfo=timezone.utc)
    return max(0,load(out/'execution_ticket.json')['wall_budget_seconds']-(datetime.now(timezone.utc)-start).total_seconds())


def demo(out):
    import cv2
    import numpy as np
    from archives import OriginalZIP
    from sampling import models
    fmp,_,_=models();configs=list(fmp.combinations())
    seed=20261109;selected=random.Random(seed).choice(configs)
    attributes=fmp.read_uvl_attributes(FMP)
    sys.path.insert(0,str(FMP.parent))
    try:renderer=import_file('supervisor_demo_renderer',FMP.with_name('generator.py'))
    finally:sys.path.remove(str(FMP.parent))
    cache=ROOT/'datasets/cache/cv261207'/out.name;cache.mkdir(parents=True)
    images=load(STUDY/'image_members.json')
    if isinstance(images,dict):images=images['images']
    rows=[];views=[];started=time.perf_counter();archive=OriginalZIP(VAL,548)
    try:
        for entry in images[:3]:
            index=int(entry.get('ZIP_index',entry.get('index',int(entry.get('image_id',entry.get('id',1)))-1)))
            image,record,provenance=archive.record(index)
            if provenance['image_SHA256']!=entry['image_SHA256'] or provenance['annotation_SHA256']!=entry['annotation_SHA256']:
                raise ValueError('Frozen demo source changed.')
            # Match the selected source ID, not an assumed list position.
            expected=entry.get('stem',entry.get('file_name'))
            if expected and Path(str(expected)).stem!=record['stem']:
                index=next(i for i,n in enumerate(archive.images) if Path(n).stem==Path(str(expected)).stem)
                image,record,provenance=archive.record(index)
            first,trace=renderer.generate(image,selected,attributes,record['stem'],seed)
            second,trace2=renderer.generate(image,selected,attributes,record['stem'],seed)
            if not np.array_equal(first,second) or trace!=trace2:raise ValueError('Demo regeneration differs.')
            target=cache/(record['stem']+'.png')
            success,encoded=cv2.imencode('.png',first)
            if not success:raise ValueError('PNG encoding failed.')
            with target.open('xb') as stream:stream.write(encoded.tobytes())
            preview=np.concatenate((image,first),axis=1)
            preview=cv2.resize(preview,(1200,max(1,round(preview.shape[0]*1200/preview.shape[1]))))
            view=cache/(record['stem']+'_preview.png')
            if not cv2.imwrite(str(view),preview):raise ValueError('Preview save failed.')
            rows.append({'source':provenance,'UVL':source(FMP),'selected_features':list(selected),
                'numeric_attributes':fmp.recipe(selected,attributes),'master_seed':seed,'effect_order':list(renderer.ORDER),
                'generator':source(FMP.with_name('generator.py')),'output':source(target),
                'pixel_sha256':hashlib.sha256(first.tobytes()).hexdigest(),'transformation_trace':trace,
                'determinism':'PASS','annotation_geometry':'NOT_USED_FOR_EVALUATION',
                'preview':source(view),'evaluated':False})
            views.append(view)
    finally:archive.close()
    report={'status':'CPU_READINESS_PASS','seed':seed,'selection':'Exact uniform draw from all 46368 legal FM_P configurations.',
        'images':rows,'wall_seconds':time.perf_counter()-started,'GPU_seconds':0,'AP':'NOT_EVALUATED',
        'supervisor_witnessed_acceptance':'UNRECORDED','cache_directory':str(cache)}
    save(out/'demo_manifest.json',report,out)
    return report


def profiles(out):
    command=[str(ROOT/'.venv/Scripts/python.exe'),'-B',str(HERE/'supervisor_profile.py'),
             '--manifest',str(out/'profile_request.json'),'--output',str(out/'operation_profiles.json')]
    specs=load(STUDY/'protocol.json')['detectors'];seen={};items=[]
    for spec in specs:
        key=spec['sha256']
        if key in seen:continue
        seen[key]=True;items.append({**spec,'id':spec['checkpoint'],'resolutions':[640,1280]})
    save(out/'profile_request.json',{'checkpoints':items},out)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
    started=time.perf_counter()
    try:
        with (out/'operation_profile.log').open('x',encoding='utf-8') as stream:
            code=bounded_run(command,stream,min(300,remaining(out)-15),env)
        if code:raise ValueError('CPU profile worker exit '+str(code))
        result=load(out/'operation_profiles.json')
        if result['GPU_seconds']!=0 or len(result['profiles'])!=6:raise ValueError('Incomplete CPU profiles.')
        status='VALIDATED_CPU_OPERATION_PROFILE'
    except Exception as error:status='BLOCKED';result={'error':str(error)}
    save(out/'operation_profile_acceptance.json',{'status':status,'result':result,
         'wall_seconds':time.perf_counter()-started,'command':command,'source':source(HERE/'supervisor_profile.py')},out)
    return status


def documents(out,demo_status,profile_status):
    rows=table(out/'static_empirical_diagnostics.csv')
    diagnostics=[r for r in rows if r['size']=='tiny' and float(r['tau'])==.1 and float(r['latency_limit_ms']) in (50,100)]
    csv_file(out/'headline_static_diagnostics.csv',diagnostics)
    milestones=[('Week 1','2026-11-02','PARTIAL','Raw replay verified; supervisor live sign-off unrecorded.'),
      ('Week 2','2026-11-09','PARTIAL',demo_status+'; supervisor live sign-off unrecorded.'),
      ('Week 3','2026-11-16','PARTIAL',profile_status+'; physical and draft-attribute review pending.'),
      ('Weeks 4-6','2026-12-07','BLOCKED','Formal 0/500; independent test-dev admission unresolved.'),
      ('Weeks 7-8','2026-12-21','PARTIAL','Development regression exists; validated failure classifier/full-space gap do not.'),
      ('Venue decision','2027-01-04','NOT_EVALUATED','Supervisor decision.'),
      ('Weeks 9-11','2027-01-11','NOT_EVALUATED','Matched augmentation training and UAVDT validation missing.'),
      ('Writing starts','2027-01-11','NOT_EVALUATED','Student and supervisors own manuscript authorship.'),
      ('Weeks 12-15','2027-02-07','NOT_EVALUATED','Evidence-based paper preparation.')]
    csv_file(out/'milestones.csv',[dict(milestone=a,date=b,status=c,evidence=d) for a,b,c,d in milestones])
    policy='''# Evidence release policy for supervisor review

Status: REVIEW_REQUIRED. This document grants no release permission.

The plan requests raw results in Git. The current project keeps raw predictions,
matching GT, images and checkpoints outside public Git. Neither rule is silently waived.

| Material | Current handling | Decision required |
|---|---|---|
| Source, UVLs, small source-linked summaries | Local Git commits | Review before public push |
| Raw predictions and derived matching GT | Immutable local RunIDs and SHA256 | Check data terms and privacy before release |
| Source images and model weights | External local storage | Confirm redistribution rights |
| Detailed environment and local logs | Private evidence | Remove credentials and private paths before release |

Two options remain for supervisor approval: an access-controlled/LFS raw repository,
or an immutable external evidence archive with version identifiers, SHA256 and Git-commit
links. The latter needs an explicit exception to the plan's raw-in-repository requirement.
No external archive has been published and no remote push was performed.

For each released Run, include its manifest, exact command, source commit, environment,
input hashes, raw predictions, matching GT, metric and bootstrap replay commands,
figure script and output checksums. Verify hashes before replay. Missing archives
must produce a clear error, never a fallback metric.
'''
    write_atomic(HERE/'SUPERVISOR_REPRODUCIBILITY_POLICY.md',policy,out)
    replay=[
      {'result':'Existing raw COCO metric replay','command':'python main/experiments/cv261207/run.py verify --help','scope':'Inspect the accepted replay arguments before running; no new inference.'},
      {'result':'Audit stage 1-2','command':'python -B main/experiments/cv261207/supervisor_compliance.py --pdf <plan.pdf> --pdf-python <pypdf-python>','scope':'New immutable CPU audit Run; no GPU.'},
      {'result':'Finish this CPU audit','command':f'python -B main/experiments/cv261207/supervisor_finish.py --run {out.name} --pdf-python <pypdf-python>','scope':'Refuses sealed Runs; retains the original 1800-second budget.'},
      {'result':'CPU operation profiles','command':'python -B main/experiments/cv261207/supervisor_profile.py --manifest <profile_request.json> --output <new-output.json>','scope':'Actual-size CPU forward only; new output required.'},
      {'result':'Original accepted figures','command':'python main/experiments/cv261207/development_figures.py --help','scope':'Source-linked figure entry; preserve original figure versions.'}]
    csv_file(out/'replay_commands.csv',replay)
    # Join extraction line wraps only; this is not bibliographic verification.
    pages=load(out/'pdf_pages.json')['pages'];text='\n'.join(pages[12:14]);text=re.sub(r'(?<=-)\s*\n\s*(?=\w)','',text)
    save(out/'reading_list_gate_v2.json',{'status':'PARTIAL','identifiers':re.findall(r'(?:doi:|arXiv:)\s*([^\s]+)',text),
        'source':'Plan pages 13-14; extraction wraps joined mechanically.',
        'original_publications_verified':False,'human_reading':'UNVERIFIED','scientific_summaries_authored':False},out)


def finish(args):
    out=resolve_run(args.run)
    if (out/'SHA256SUMS.txt').exists():raise ValueError('Run is sealed; no overwrite.')
    original=load(out/'execution_ticket.json')['source']
    snapshot=out/'sources/runner_stage12.py'
    if not snapshot.exists() or sha256(snapshot)!=original['sha256']:raise ValueError('Missing original executed runner snapshot.')
    save(out/'continuation_source.json',{'source':source(Path(__file__)),'original_budget_remaining':remaining(out),
          'GPU_seconds':0,'no_budget_reset':True,'started_utc':datetime.now(timezone.utc).isoformat()},out)
    if remaining(out)<60:raise TimeoutError('No CPU-stage budget remains.')
    started=time.perf_counter()
    try:d=demo(out);demo_status=d['status']
    except Exception as error:
        demo_status='BLOCKED';save(out/'demo_failure.json',{'error':str(error),'GPU_seconds':0},out)
    profile_status=profiles(out) if remaining(out)>90 else 'NOT_EVALUATED_BUDGET'
    context=load(out/'context.json');plan=load(out/'pdf_pages.json')['pages']
    matrix(out,plan,context['source_evidence'],3,profile_status,demo_status)
    documents(out,demo_status,profile_status)
    save(out/'continuation_result.json',{'status':'PARTIAL','CPU_demo':demo_status,'operation_profiles':profile_status,
         'wall_seconds':time.perf_counter()-started,'remaining_budget_seconds':remaining(out),'GPU_seconds':0},out)
    print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--pdf-python',type=Path,required=True)
    finish(p.parse_args())
