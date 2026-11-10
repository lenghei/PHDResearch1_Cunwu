"""Materialize the official public toolkit for read-only annotation review."""
import hashlib
import os
from pathlib import Path
import time
import urllib.request
from zipfile import ZipFile
from protocol import ROOT,EXPERIMENT,save,source,write_atomic
from common import seal
from historical_review import csv_file
from main.pipeline import new_run

FILE_ID='19498uJd7T9w4quwnQEy62nibt3uyT9pq'
EXPECTED_BYTES=245719325


def execute():
    start=time.perf_counter();url=os.environ['UAVGAP_TOOLKIT_DOWNLOAD_URL']
    out,_=new_run({'output_root':str(ROOT/'datasets/local_runs'/EXPERIMENT)},EXPERIMENT,'uavdt_toolkit_source')
    path=out/'UAV-benchmark-MOTD_v1.0.zip';partial=out/(path.name+'.partial')
    h=hashlib.sha256();count=0
    with urllib.request.urlopen(url,timeout=25) as response,partial.open('xb') as stream:
        while True:
            block=response.read(1024**2)
            if not block:break
            count+=len(block)
            if count>256*1024**2 or time.perf_counter()-start>120:raise RuntimeError('Bounded toolkit transfer stopped.')
            stream.write(block);h.update(block)
    if count!=EXPECTED_BYTES:raise ValueError('Toolkit size differs from provider metadata.')
    partial.rename(path)
    with ZipFile(path) as archive:
        members=[{'name':i.filename,'bytes':i.file_size,'compressed_bytes':i.compress_size,'CRC':i.CRC} for i in archive.infolist()]
        selected=[i for i in archive.infolist() if ('gt_whole' in i.filename.lower() or 'gt_ignore' in i.filename.lower()) and i.file_size<4*1024**2]
        samples=[]
        for item in selected:
            blob=archive.read(item.filename)
            lines=blob.decode('utf-8-sig',errors='replace').splitlines()
            samples.append({'member':item.filename,'SHA256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob),
                'rows':len([s for s in lines if s.strip()]),'first_rows':lines[:3]})
    save(out/'TOOLKIT_INVENTORY.json',members,out);save(out/'ANNOTATION_SAMPLES.json',samples,out)
    save(out/'SOURCE.json',{'official_homepage':'https://sites.google.com/view/grli-uavdt/%E9%A6%96%E9%A1%B5',
        'provider_file_ID':FILE_ID,'provider_mime':'application/zip','provider_modified_UTC':'2018-08-13T15:51:58.883Z',
        'provider_bytes':EXPECTED_BYTES,'local_archive':source(path),'source_chain':'Official DET/MOT toolkit link -> provider metadata -> authenticated raw-file reference.',
        'untrusted_code_execution':False,'image_dataset_downloaded':False,
        'license':'Research-use statement retained; no redistribution permission inferred.','signed_transfer_URL_persisted':False},out)
    write_atomic(out/'diary.md','# Original toolkit source\n\nDownloaded the exact official DET/MOT toolkit file through the read-only connector. Inspected ZIP entries and bounded annotation text only; no downloaded code executed. Original local dataset unchanged. Evaluation admission remains separate.\n',out)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>UAVDT official toolkit source</h1><a href="SOURCE.json">Source</a> · <a href="TOOLKIT_INVENTORY.json">Inventory</a> · <a href="ANNOTATION_SAMPLES.json">GT samples</a>',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'UAVDT_TOOLKIT_SOURCE','status':'SOURCE_MATERIALIZED',
        'seconds':time.perf_counter()-start,'new_GPU_experiments':0},out,replace=True)
    seal(out);print(out,flush=True)


if __name__=='__main__':execute()
