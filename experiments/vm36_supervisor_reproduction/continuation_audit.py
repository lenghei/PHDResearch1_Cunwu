"""Bounded read-only provenance review; no detector or training execution."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import urllib.request
from zipfile import ZipFile
from protocol import ROOT,EXPERIMENT,load,save,source,write_atomic
from common import seal,verify_hashes
from historical_review import table,csv_file
from full_development import PREP
from main.pipeline import new_run


def public_source(url,limit=8*1024**2):
    start=time.perf_counter()
    try:
        with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'UAV-GAP provenance review'}),timeout=20) as response:
            blob=response.read(limit+1)
            return {'URL':url,'final_URL':response.url,'status':response.status,'bytes_read':len(blob),
                'bounded':len(blob)<=limit,'SHA256':hashlib.sha256(blob).hexdigest(),'seconds':time.perf_counter()-start},blob
    except Exception as error:return {'URL':url,'status':'ACCESS_NOT_ESTABLISHED','error':str(error),'seconds':time.perf_counter()-start},None


def execute():
    import yaml
    started=time.perf_counter();out,_=new_run({'output_root':str(ROOT/'datasets/local_runs'/EXPERIMENT)},EXPERIMENT,'continuation_audit')
    archived=ROOT/'datasets/research_archive/variability/experiments'
    prior=archived/'vm11_holdout_confirmation/runs/vm11_r002/results'
    verify_hashes(PREP,('DATASET_ADMISSION.json','CHECKPOINT_ADMISSION.csv','INDEPENDENT_EVALUATION_ADMISSION.json'))
    catalog=table(ROOT/'datasets/research_archive/migration_20261104/model_catalog.csv');rows=[]
    for identity in ('ck01','ck05','ck06'):
        row=next(r for r in catalog if r['model_id'].startswith(identity+'_'))
        args_path=ROOT/row['training_config'];args=yaml.safe_load(args_path.read_text(encoding='utf-8'))
        rows.append({'checkpoint':identity,'checkpoint_SHA256':row['sha256'],'training_args':source(args_path),
            'declared_data_path':args['data'],'validation_enabled':args.get('val'),'selection_split':args.get('split'),
            'recorded_training_resolution':args['imgsz'],'recorded_seed':args['seed'],
            'testdev_training_membership':'NO_DIRECT_TESTDEV_PATH_IN_ARGS; creation-time bytes unverified',
            'historical_checkpoint_selection_testdev':'UNKNOWN','historical_method_tuning_testdev':'UNKNOWN',
            'formal_admission':'DATASET_BLOCKED'})
    save(out/'CHECKPOINT_PROVENANCE.json',rows,out)
    data=load(PREP/'DATASET_ADMISSION.json')
    save(out/'INDEPENDENT_ADMISSION.json',{'status':'DATASET_BLOCKED','file_disjointness':data['testdev_disjointness'],
        'source':source(PREP/'DATASET_ADMISSION.json'),'previous_history_audit':source(prior/'checkpoint_exposure_audit.csv'),
        'prior_use_inventory':'UNKNOWN; saved code references and literature mentions do not prove actual model evaluation.',
        'no_search_hit_inference':False,'new_testdev_inferences':0,
        'missing':['Creation-time byte manifests for training and initialization','Selected epoch/checkpoint provenance',
            'Accountable inventory of manual/external test-dev evaluations and decisions'],
        'safe_plan':'Freeze compatible new native10 scenes and hypotheses prospectively, or recover adequate checkpoint-specific history before all1610-image formal evaluation. No automatic dataset substitution.'},out)
    dataset=ROOT/'datasets/uavdt';summary=[];samples=[]
    for split in ('train','test'):
        annotations=sorted((dataset/split/'ann').glob('*.json'));images=list((dataset/split/'img').iterdir())
        prefixes={p.name.split('_')[0] for p in annotations}
        summary.append({'split':split,'annotation_files':len(annotations),'image_files':sum(p.is_file() for p in images),
            'sequence_prefixes':sorted(prefixes),'M_sequences':sum(s.startswith('M') for s in prefixes),
            'S_sequences':sum(s.startswith('S') for s in prefixes)})
        for p in annotations[:2]+annotations[-2:]:
            item=load(p);samples.append({'source':source(p),'classes':sorted({r.get('classTitle') for r in item.get('objects',[]) if r.get('classTitle')}),
                'tags':[t.get('name') for t in item.get('tags',[])],'objects':len(item.get('objects',[]))})
    search=subprocess.run(['rg','--files','--hidden','--no-ignore','-g','*.txt','-g','*.mat','-g','*gt_whole*','-g','*gt_ignore*',str(dataset)],
        capture_output=True,text=True,timeout=30)
    save(out/'UAVDT_LOCAL.json',{'dataset':str(dataset),'splits':summary,'metadata':source(dataset/'meta.json'),
        'license':source(dataset/'LICENSE.md'),'sample_annotations':samples,'raw_annotation_search_matches':search.stdout.splitlines(),
        'search_scope':'Current known UAVDT root only; not an exhaustive search of all disks.',
        'correction':'A shared catalog containing vehicle does not prove the M-prefixed train/test data are SOT. DET/MOT ignore provenance is a separate issue.'},out)
    official='https://sites.google.com/view/grli-uavdt/%E9%A6%96%E9%A1%B5'
    fetches=[]
    for name,url in [('homepage',official),('converter','https://raw.githubusercontent.com/dataset-ninja/uavdt/main/src/convert.py'),
        ('toolkit','https://drive.google.com/uc?export=download&id=19498uJd7T9w4quwnQEy62nibt3uyT9pq')]:
        meta,blob=public_source(url);fetches.append({'name':name,**meta})
        if name=='converter' and blob and meta['bounded']:
            text=blob.decode('utf-8');save(out/'CONVERTER_SEMANTICS.json',{'source':meta,
                'input_whole_GT_declared':'_gt_whole.txt' in text,'ignore_file_reference_found':'_gt_ignore' in text,
                'class_mapping':'1=car,2=truck,3=bus; SOT branch uses vehicle',
                'limit':'Read-only converter source review; not verification of the historical conversion version or recovery of missing ignore regions.'},out)
        if name=='toolkit' and blob and meta['bounded'] and blob.startswith(b'PK'):
            path=out/'official_toolkit.zip';path.write_bytes(blob)
            with ZipFile(path) as z:save(out/'TOOLKIT_MEMBERS.json',[{'name':i.filename,'bytes':i.file_size,'compressed':i.compress_size} for i in z.infolist()],out)
    save(out/'OFFICIAL_SOURCE_ACCESS.json',{'checked_UTC':datetime.now(timezone.utc).isoformat(),'sources':fetches,
        'official_URL':official,'research_use':'Official page states research purpose only; no redistribution permission inferred.',
        'official_scope':'Benchmark-M DET/MOT and Benchmark-S SOT are distinct; original ignore regions matter.'},out)
    status='BLOCKED_ANNOTATION_PROVENANCE'
    save(out/'UAVDT_ADMISSION.json',{'status':status,'official_GT_restored':False,'ignore_regions_restored':False,
        'converted_labels_not_changed':True,'new_external_AP':0,
        'reason':'A converted M-frame catalog and source-code mapping do not establish original GT/ignore completeness, historical converter identity or negative-frame semantics.'},out)
    write_atomic(out/'diary.md','# Bounded provenance review\n\nReviewed direct checkpoint args, saved exposure evidence and current UAVDT metadata. Official source access was attempted without credentials or bypass. Missing history and original ignore semantics remain explicit. No model inference or dataset conversion.\n',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'PROVENANCE_AUDIT','status':'COMPLETE_WITH_LIMITATIONS',
        'wall_seconds':time.perf_counter()-started,'new_GPU_experiments':0},out,replace=True)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>VM36 provenance review</h1><p>Formal test-dev and UAVDT gates remain blocked.</p><a href="INDEPENDENT_ADMISSION.json">Test-dev</a> · <a href="UAVDT_ADMISSION.json">UAVDT</a>',out)
    seal(out);print(out,flush=True)


if __name__=='__main__':execute()
