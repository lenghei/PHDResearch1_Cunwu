"""Bounded, read-only follow-up of the UAVDT admission blocker."""
import argparse
import json
import ssl
import time
from urllib.request import Request, urlopen
from study import *


def execute():
    tick=time.perf_counter();out=create('external_provenance_cpu');runtime(out,cuda=False)
    previous=ROOT/'datasets/research_archive/variability/experiments/vm25_final_dataset_audit/runs/vm25_r001'
    dataset=ROOT/'datasets/uavdt'
    references=[previous/'results/dataset_gate.json',previous/'results/official_source.json',
                dataset/'meta.json',dataset/'README.md',dataset/'LICENSE.md']
    inputs=[source(path) for path in references]
    # Search this converted dataset only; no global absence claim.
    candidates=[]
    for path in dataset.rglob('*'):
        if path.is_file() and (path.suffix.lower() in ('.txt','.zip') or 'ignore' in path.name.lower()):
            candidates.append(source(path))
    urls={
        'official_homepage':'https://sites.google.com/view/grli-uavdt/',
        'official_toolkit_metadata':'https://drive.google.com/file/d/19498uJd7T9w4quwnQEy62nibt3uyT9pq/view',
    }
    fetched={}
    import certifi
    context=ssl.create_default_context(cafile=certifi.where())
    for name,url in urls.items():
        try:
            with urlopen(Request(url,headers={'User-Agent':'UAV-GAP provenance audit'}),timeout=20,context=context) as response:
                body=response.read(2_000_001)
                if len(body)>2_000_000:raise ValueError('Bounded metadata response exceeded2MB.')
                path=out/(name+'.html');path.write_bytes(body)
                fetched[name]={'url':url,'final_url':response.url,'status':response.status,'evidence':source(path)}
        except Exception as error:fetched[name]={'url':url,'error':str(error)}
    metadata=load(dataset/'meta.json')
    gate={'status':'BLOCKED_ANNOTATION_PROVENANCE','scope':'BOUNDED_CONVERTED_DATASET_AUDIT',
          'dataset_path_now_exists':dataset.exists(),'metadata_classes':[r['title'] for r in metadata['classes']],
          'original_DET_GT_verified':False,'original_ignore_regions_verified':False,
          'DET_split_verified':False,'converted_empty_frame_semantics':'UNVERIFIED',
          'native10_mapping':'NOT_ADMITTED','candidates_in_bounded_directory':candidates,
          'historical_gate':load(previous/'results/dataset_gate.json'),
          'test_dev_status':'DATASET_BLOCKED','new_inference':False,
          'reason':'Relocation restores the converted directory, not the missing original DET/ignore semantics. Official toolkit metadata is not original annotation evidence.',
          'independent_plan':'Acquire the original DET/MOT annotation/toolkit with ignore definitions and split provenance; freeze a compatible vehicle task and checkpoint exposure audit before any new inference.'}
    save(out/'inputs.json',{'read_only_sources':inputs,'new_source_queries':fetched},out)
    save(out/'external_dataset_gate.json',gate,out)
    if any(sha256(Path(r['path']))!=r['sha256'] for r in inputs):raise ValueError('Read-only provenance input changed.')
    write_atomic(out/'diary.md','# UAVDT provenance follow-up\n\nThe converted directory exists at the new location. Original DET labels, ignore handling and split semantics remain unverified. No image inference or class substitution was performed.\n',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'status':'COMPLETE_WITH_LIMITATIONS',
                       'CPU_seconds':time.perf_counter()-tick,'GPU_seconds':0},out,replace=True)
    seal(out);print(out,flush=True)


if __name__=='__main__':execute()
