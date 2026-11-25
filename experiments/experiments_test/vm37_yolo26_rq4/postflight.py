"""Verify frozen input preservation after the owned GPU worker has stopped."""
import argparse
import csv
import time
from study import *


def execute(cost_id):
    if (RUNS/'.gpu.lock').exists():raise RuntimeError('Wait for the owned GPU lease to finish.')
    tick=time.perf_counter();out=create('preservation_cpu');runtime(out,cuda=False)
    cost=run_path(cost_id);verify_hashes(cost,('protocol.json','training_acceptance.json','protected_inputs.json'))
    protected=load(cost/'protected_inputs.json')
    if not protected['unchanged'] or len(protected['inputs'])!=2*(6471+548):raise ValueError('Incomplete data preservation evidence.')
    annotations=[]
    with (PREP/'TRAINING_DATA_VERIFICATION.csv').open(encoding='utf-8-sig') as stream:
        training=list(csv.DictReader(stream))
    population=load(PREP/'VALIDATION_LOADER_ACCEPTANCE.json')['population']
    for split,rows in (('train',training),('val',population)):
        for row in rows:
            stem=row['image_id'] if split=='train' else row['stem']
            path=ROOT/'datasets/VisDrone'/('VisDrone2019-DET-'+split)/'annotations'/(stem+'.txt')
            expected=row['annotation_SHA256'] if split=='train' else row['source_annotation_SHA256']
            actual=sha256(path)
            annotations.append({'split':split,'image_id':stem,'path':str(path),'expected_SHA256':expected,
                                'actual_SHA256':actual,'equal':actual==expected})
    weights={}
    for row in load(PREP/'CANDIDATE_INITIALIZATION_MANIFEST.json'):
        for key in ('source','native10_candidate'):
            ref=row[key];path=Path(ref['path']);actual=sha256(path)
            weights[str(path)]={'expected_SHA256':ref['sha256'],'actual_SHA256':actual,'equal':actual==ref['sha256']}
    historical=[]
    for row in load(PREP/'FROZEN_DEVELOPMENT_EVALUATION.json')['conditions']:
        path=Path(row['source_folder'])/'metrics.json'
        historical.append({'pair_id':row['pair_id'],'expected_SHA256':row['source_SHA256']['metrics.json'],
                           'actual_SHA256':sha256(path)})
    installed=[]
    version=RUNS/'vm37_yolo26_rq4_version_regression_cpu_20261111T223751_94186fca'
    for row in load(version/'version_acceptance.json')['critical_source_files']:
        actual=sha256(ROOT/'.venv_rq4/Lib/site-packages/ultralytics'/row['file'])
        installed.append({'file':row['file'],'equal':actual==row['official_SHA256'],'actual_SHA256':actual})
    admitted_sources=load(cost/'protocol.json')['sources']
    shared=[{'file':name,'equal':sha256(PUBLIC/name)==expected} for name,expected in admitted_sources.items()]
    valid=all(r['equal'] for r in annotations) and all(r['equal'] for r in weights.values()) and all(r['equal'] for r in installed+shared)
    valid=valid and len(historical)==84 and all(r['actual_SHA256']==r['expected_SHA256'] for r in historical)
    with (out/'annotation_integrity.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=annotations[0]);writer.writeheader();writer.writerows(annotations)
    save(out/'preservation.json',{'status':'PASS' if valid else 'FAILED','source_cost_RunID':cost_id,
        'training_images':len(training),'validation_images':len(population),'raw_annotations_checked':len(annotations),
        'training_native10_targets':sum(int(r['native10_targets']) for r in training),
        'image_and_derived_label_posthash_evidence':source(cost/'protected_inputs.json'),
        'original_and_initialization_weights':weights,'historical84_metric_files':historical,
        'installed_official_source':installed,'admitted_project_sources':shared,
        'GPU_seconds':0,'CPU_seconds':time.perf_counter()-tick,
        'scope':'Frozen training/validation inputs, initialization weights,84 saved metric files and admitted source. The entire145GiB archive was not recopied or rehashed.'},out)
    write_atomic(out/'diary.md','# Input preservation\n\nChecked raw annotations, source/initialization weights, accepted84 metric files and current admitted source hashes. Reused the worker posthash for source images and private labels. No GPU execution or historical metric recalculation.\n',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'status':'PASS' if valid else 'FAILED'},out,replace=True)
    seal(out);print(out,flush=True)
    if not valid:raise SystemExit(1)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cost',required=True);execute(parser.parse_args().cost)
