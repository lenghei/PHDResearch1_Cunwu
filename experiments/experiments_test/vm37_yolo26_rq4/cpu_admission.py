"""Check source identity and native10 construction before the first GPU worker."""
import copy
import csv
import json
from pathlib import Path
import urllib.request
from study import *


def prepare():
    out=run_path(load(RUNS/'source_environment.json')['RunID']);runtime(out,cuda=False)
    from pypdf import PdfReader
    pdf=ROOT/'datasets/research_archive/supervisor/Research_plan_Cunwu_UAV_gap.pdf'
    pages={str(i):page.extract_text() for i,page in enumerate(PdfReader(pdf).pages,1)}
    save(out/'supervisor_pages.json',pages,out)
    for n in range(11,17):print('PAGE',n,pages[str(n)])
    tag=load(out/'tag.json')
    with urllib.request.urlopen(urllib.request.Request(tag['object']['url'],headers={'User-Agent':'UAV-GAP-source-audit'}),timeout=30) as response:
        commit=json.loads(response.read())
    save(out/'tag_object.json',commit,out)
    if not (RUNS/'compute_allocation.json').exists():
        save(RUNS/'compute_allocation.json',{'total_seconds':172800,'admission_seconds':7200,
            'author_SHA256':sha256(out/'author_task.txt'),'authorization':source(out/'author_task.txt'),
            'first_worker_UTC':None,'absolute_deadline_UTC':None,'completed_workers':[],'budget_reset':False},RUNS)
    protocol='''# YOLO26-first matched RQ4 study

Primary: YOLO26s. Comparison: YOLOv8s. Preserve the supervisor RQ1-RQ4 and original VM36 feature models.

First run native10 FP32 training at1280, batch2, workers0, seed42 with the fixed64/16 engineering membership. Then measure a complete6471-image epoch and548-image development validation under the candidate100-epoch schedule. Stop the cost study after its first completed epoch. Neither is a completed scientific arm.

Main candidate: two architectures x three seeds42/43/44 x CONTROL/ALL_FIVE;100epochs,1280,batch2,nbs64,AdamW lr0=.01,cosine,warmup3,FP32,workers0. Each seed-matched pair shares initialization and the final-epoch100EMA rule. Historical best checkpoints are not matched controls.

CONTROL retains ordinary HSV/geometry/Albumentations. Targeted families need source-backed ranges and explicit probabilities. Engineering fixtures cannot admit ALL_FIVE. Unsupported families remain BLOCKED_DESIGN.

Research evaluation retains native10, FP32, square Letterbox, one-to-many plus class-aware multi-label NMS, conf.001,IoU.7,max_det3000,max_nms30000 and the original ignore-union/COCO implementation. Internal library validation has a separate metric definition.

One new48-hour allocation includes at most2hours initial admission. One GPU worker, no budget reset. A complete100-epoch pair must fit measured remaining compute/storage before either arm starts. Test-dev remains DATASET_BLOCKED. Validation results are DEVELOPMENT_ONLY.
'''
    write_atomic(HERE/'RESEARCH_PROTOCOL.md',protocol,ROOT)
    write_atomic(HERE/'EXECUTION_BUDGET.json',json.dumps({'total_worker_seconds':172800,'initial_admission_seconds':7200,
        'new_allocation':True,'reset_allowed':False,'single_GPU_worker':True,'runtime_ledger':'Local RunStore compute_allocation.json',
        'old_VM36_budgets_reused':False},indent=2),ROOT)
    rows=[]
    for architecture in ('YOLO26s','YOLOv8s'):
        for seed in (42,43,44):
            for arm in ('CONTROL','ALL_FIVE'):
                rows.append({'architecture':architecture,'seed':seed,'arm':arm,'epochs':100,'imgsz':1280,'batch':2,'nbs':64,
                    'optimizer':'AdamW','lr0':.01,'precision':'FP32','workers':0,'selection':'EPOCH100_EMA',
                    'status':'NOT_RUN_PENDING_MEASURED_ADMISSION'})
    with (HERE/'TRAINING_MATRIX.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    with (HERE/'DECISION_REGISTER.csv').open('x',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream);writer.writerow(['decision','choice','alternative','reason','status'])
        writer.writerows([
            ['Initialization','Verified COCO-derived native10 per-seed states','Historical best FT','Historical selected epochs and creation-time initialization are incompletely auditable.','FROZEN'],
            ['Library','8.4.210 candidate','8.4.61 verified legacy','Official stable release; actual compatibility gates required.','PENDING_REAL_ADMISSION'],
            ['Checkpoint selection','Final epoch100 EMA','Best-on-val','Same fixed rule without degraded-AP selection; explicit design choice.','FROZEN'],
            ['Targeted recipe','Source-supported policy only','CPU fixtures','No favorable-AP tuning or implicit fixture reuse.','PENDING_SOURCE_REVIEW']])
    manifest={'supervisor_pdf_SHA256':sha256(pdf),'ultralytics_version':'8.4.210','official_commit':commit['object'],
        'official_release_URL':'https://github.com/ultralytics/ultralytics/releases/tag/v8.4.210',
        'FMP_SHA256':sha256(VM36/'feature_models/fm_p.uvl'),'FMD_SHA256':sha256(VM36/'feature_models/fm_d.uvl'),
        'shared_adapter_origin':{n:sha256(VM36/n) for n in ('rq4_training_backend.py','rq4_dataset.py','rq4_augmentation.py')},
        'historical_pairs':84,'historical_pairs_policy':'Read-only reference; no rerun.'}
    write_atomic(HERE/'SOURCE_MANIFEST.json',json.dumps(manifest,indent=2),ROOT)
    import torch
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.cfg import DEFAULT_CFG
    from main.labels import VIS_NAMES
    from main.training_health import same_state,cpu_state,model_health
    initial=next(r for r in load(PREP/'CANDIDATE_INITIALIZATION_MANIFEST.json') if r['architecture']=='YOLO26s' and r['seed']==42)
    model=torch.load(initial['native10_candidate']['path'],map_location='cpu',weights_only=False)['model']
    trainer=DetectionTrainer.__new__(DetectionTrainer)
    trainer.data={'nc':10,'channels':3,'names':dict(enumerate(VIS_NAMES))};trainer.args=copy.copy(DEFAULT_CFG);trainer.args.cls_remap=False
    new=trainer.get_model(cfg=copy.deepcopy(model.yaml),weights=model,verbose=False);model_health(new)
    equal=same_state(new,cpu_state(model))
    save(out/'CPU_model_compatibility.json',{'version':'8.4.210','native10_state_equal':equal,
        'parameters':sum(p.numel() for p in new.parameters()),'class_order':list(trainer.data['names'].values()),
        'head_class':type(new.model[-1]).__name__,'source':initial,'GPU_training':'NOT_RUN'},out)
    if not equal:raise ValueError('Native10 constructor incompatibility.')
    print('OFFICIAL_COMMIT',commit['object']);print('CANARY_EXPECTED',expected_steps(64,1))
    print('COST_EPOCH_EXPECTED',expected_steps(6471,100,stop_epochs=1));print(out)


if __name__=='__main__':prepare()
