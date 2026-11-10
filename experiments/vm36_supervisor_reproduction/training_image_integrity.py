"""Check all original JPEGs without invoking the loader's repair path."""
import hashlib
import io
from pathlib import Path
import shutil
import time
from zipfile import ZipFile
from PIL import Image
from protocol import ROOT,EXPERIMENT,load,save,source,sha256,write_atomic
from common import seal,verify_hashes
from historical_review import table,csv_file
from main.pipeline import new_run

DATA='vm36_supervisor_reproduction_training_data_cpu_v2_20261110T110058_057fb01c'


def execute():
    start=time.perf_counter();root=ROOT/'datasets/local_runs'/EXPERIMENT;previous=root/DATA
    verify_hashes(previous,('TRAINING_IMAGE_LABEL_AUDIT.csv','TRAINING_DATA_GATE.json'))
    records=table(previous/'TRAINING_IMAGE_LABEL_AUDIT.csv')
    if len(records)!=6471:raise ValueError('Original population is incomplete.')
    archive=ROOT/'datasets/VisDrone/VisDrone2019-DET-train.zip'
    if sha256(archive)!=load(previous/'inputs.json')['archive']['sha256']:raise ValueError('Original archive changed.')
    out,_=new_run({'output_root':str(root)},EXPERIMENT,'training_image_integrity')
    shutil.copyfile(Path(__file__),out/'executed_source.py');rows=[]
    with ZipFile(archive) as z:
        for record in records:
            if time.perf_counter()-start>300:break
            blob=z.read(record['image_member']);digest=hashlib.sha256(blob).hexdigest()
            if digest!=record['image_SHA256']:raise ValueError('Source image bytes changed.')
            valid=True;error=''
            try:
                with Image.open(io.BytesIO(blob)) as image:image.verify()
            except Exception as e:valid=False;error=str(e)
            rows.append({'image_id':record['image_id'],'image_SHA256':digest,'PIL_verify':valid,
                'JPEG_EOI_at_end':blob[-2:]==b'\xff\xd9','would_trigger_library_EOI_repair':blob[-2:]!=b'\xff\xd9',
                'error':error,'source_modified':False})
            if len(rows)%1000==0:print('READ_ONLY_JPEG_CHECK',len(rows),'/6471',flush=True)
    csv_file(out/'JPEG_INTEGRITY.csv',rows)
    result={'status':'READ_ONLY_IMAGE_INTEGRITY_VERIFIED' if len(rows)==6471 and all(r['PIL_verify'] and r['JPEG_EOI_at_end'] for r in rows) else 'IMAGE_INTEGRITY_NOT_ADMITTED',
        'images_checked':len(rows),'expected_images':6471,'PIL_verify_failures':sum(not r['PIL_verify'] for r in rows),
        'would_trigger_library_EOI_repair':sum(r['would_trigger_library_EOI_repair'] for r in rows),
        'source_images_modified':False,'new_GPU_tasks':0,'CPU_seconds':time.perf_counter()-start,'CPU_budget_seconds':300,
        'scope':'Read-only encoding check. This does not execute a6471-image training loader, optimizer or trajectory test.'}
    save(out/'IMAGE_INTEGRITY_GATE.json',result,out);save(out/'inputs.json',{'source_data':source(previous/'SHA256SUMS.txt'),
        'archive':source(archive),'source':source(Path(__file__))},out)
    write_atomic(out/'diary.md','# Read-only image integrity\n\nVerified original image hashes, JPEG end markers and Pillow parsing in memory. Did not call a repair function or write source images.\n',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'TRAINING_IMAGE_INTEGRITY','status':result['status']},out,replace=True)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>Read-only JPEG integrity</h1><a href="IMAGE_INTEGRITY_GATE.json">Gate</a> · <a href="JPEG_INTEGRITY.csv">Image checks</a>',out)
    seal(out);print(out,flush=True)


if __name__=='__main__':execute()
