"""Reconcile named historical-use records without relaxing admission."""
import argparse
from datetime import datetime, timezone
import re
import time

from common import ROOT, load, new_run, save, seal, sha256, source, verify_hashes
from historical_review import VM, table, csv_file

ADMISSION = ROOT/'datasets/local_runs/cv261207/cv261207_admission_20261104T211249_32d83def'


def review():
    started=time.perf_counter()
    out=new_run('historyreview')
    verify_hashes(ADMISSION, ['admission.json','summary.json','run.json'])
    prior=load(ADMISSION/'admission.json')
    if prior['status']!='DATASET_BLOCKED':
        raise ValueError('Expected the existing blocked gate.')
    manifest_path=ROOT/'datasets/research_archive/migration_20261104/file_manifest.csv'
    manifest={r['destination']:r for r in table(manifest_path)}
    paths=[
        ROOT/'datasets/research_archive/degradation_adaptive/reports/phase2_testdev_history_audit.json',
        VM/'experiments/vm11_holdout_confirmation/runs/vm11_r001/acceptance.json',
        VM/'experiments/vm11_holdout_confirmation/runs/vm11_r002/acceptance.json',
        VM/'experiments/vm11_holdout_confirmation/runs/vm11_r002/results/history_search.json',
        VM/'experiments/vm11_holdout_confirmation/runs/vm11_r002/results/checkpoint_exposure_audit.csv',
        VM/'experiments/vm11_holdout_confirmation/runs/vm11_r002/results/dataset_gate.json',
        VM/'experiments/vm15_external_datasets/runs/vm15_r001/results/checkpoint_exposure_evidence.json',
        VM/'experiments/vm34_gpu_route/runs/vm34_r001/results/checkpoint_exposure.json',
        ROOT/'datasets/research_archive/paper1/source_configs/visdrone_10class.yaml']
    for folder in ('ck01_yolo26s_ft1280','ck05_yolov8s_ft1280','ck06_yolo11s_ft1280'):
        paths.append(ROOT/'models/checkpoints'/folder/'args.yaml')
    files=[]; mentions=[]
    for path in paths:
        if path.stat().st_size>2097152:
            raise ValueError('Named evidence exceeds bounded read limit.')
        proof=source(path)
        relative=path.relative_to(ROOT).as_posix()
        if relative in manifest and proof['sha256']!=manifest[relative]['destination_sha256']:
            raise ValueError('Migrated historical record changed.')
        proof['migration_verified']=relative in manifest
        files.append(proof)
        for line_number,line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(),1):
            if re.search(r'test.?dev|UNVERIFIED|UNKNOWN|NOT_QUALIFIED|BLOCKED_INDEPENDENT',line,re.I):
                mentions.append({'source':relative,'source_sha256':proof['sha256'],'line':line_number,
                                 'text':line[:1200], 'classification':'RECORDED_REFERENCE_OR_UNCERTAINTY_NOT_PROOF_OF_EVALUATION'})
    csv_file(out/'bounded_evidence_lines.csv',mentions)
    save(out/'history_review.json',{
        'status':'DATASET_BLOCKED','prior_admission':source(ADMISSION/'admission.json'),
        'manifest':source(manifest_path),'files':files,'max_file_bytes':2097152,
        'scope':'Explicit named evidence only; no recursive archive search.',
        'verified_dataset_disjointness':prior['archive_overlap'],
        'verified_checkpoint_provenance':'SHA, native10, logged training arguments and numeric health verified in the sealed prior admission.',
        'recorded_prior_testdev_use':'Named records do not establish a completed test-dev evaluation for these SHA identities. Historical searches and protocol mentions are not inference evidence.',
        'missing_evidence':['Creation-time complete membership for the three historical FT1280 checkpoints',
                            'Complete checkpoint-selection and manual/external tuning history',
                            'Scene-level acquisition independence'],
        'absence_of_search_hits_proves_independence':False,
        'checkpoints':[{'id':c['id'],'sha256':c['checkpoint_sha256'],
                        'independent_testdev_status':'UNVERIFIED'} for c in prior['candidates']],
        'formal_pairs_measured':0,'GPU_executions':0})
    plan='''# Independent validation plan for supervisor review

Status: PROPOSED; no dataset substitution or inference is authorized by this plan.

## Existing test-dev branch

Keep DATASET_BLOCKED until each exact checkpoint SHA has reviewable training-membership,
checkpoint-selection and historical evaluation/tuning evidence. Current train/val/test-dev
byte disjointness is established; creation-time records and manual/external use are not.
Do not convert a bounded negative search into an independence certificate.

## If the historical record cannot be recovered

Ask the supervisor to approve a separately registered prospective validation cohort.
Use genuinely unseen acquisition scenes, documented permission and aerial ground-object
labels compatible with native10. Confirm class definitions, ignore regions, image geometry
and annotation quality before admitting any model. Do not silently use COMMON3 data.
Keep VisDrone test-dev blocked under the current protocol. Any exploratory use needs a
separate authorization and must not be called independent confirmation.

## Freeze before access to validation responses

- Lock checkpoint, prediction/COCO protocol, class order, JPEG recipe, sample IDs and hashes.
- Register the eight endpoints for each R(640/1280) × B(0/2) × J(None/Q50) contrast.
- Freeze the depth-2 descriptive rule and the practical 0.01 AP threshold without tuning on new responses.
- Split by reliable acquisition scenes; keep all degradations/resolutions of an image together.
- Separate development and held-out configuration groups as well as image groups.
- Fit only on development groups; compute held-out COCO AP from held-out GT and predictions.
- Use common image/scene bootstrap draws across endpoints and a predeclared comparison family.
- Report predictive MAE, direction errors and calibration separately from direct DID and intervals.

## Resource admission

Do not start 500 pairs. First authorize one bounded full pair, regenerate its metrics,
repeat the identical pair and measure end-to-end latency, evaluation time and storage.
Set the next execution wave from those measurements. The eight-endpoint relation plan
is separate from the current single-checkpoint 500-pair manifest and needs explicit approval.
No compute budget is inferred from the historical small pilot.
'''
    (out/'independent_validation_plan.md').write_text(plan,encoding='utf-8')
    save(out/'run.json',{'RunID':out.name,'status':'DATASET_BLOCKED','mode':'bounded_historical_evidence_review',
                        'ended_utc':datetime.now(timezone.utc).isoformat(),
                        'CPU_seconds':time.perf_counter()-started,'GPU_executions':0})
    seal(out)
    print(out)


if __name__=='__main__':
    review()
