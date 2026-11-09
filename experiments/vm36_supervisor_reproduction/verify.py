"""Seal a factual VM36 preparation package without granting scientific approval."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import subprocess
import sys
from datetime import datetime,timezone
from protocol import ROOT,PUBLIC,HERE,EXPERIMENT,load,save,source,sha256,write_atomic,FMP,FMD,LEGACY_FMP,LEGACY_FMD
from common import seal,verify_hashes


def verify(out):
    if (out/'SHA256SUMS.txt').exists():raise ValueError('Run is already sealed.')
    required=['EXPERIMENT_PROTOCOL.md','EXPERIMENT_PROTOCOL.json','DECISION_REGISTER.csv','DATASET_ADMISSION.json',
        'FEATURE_MODEL_VALIDATION.json','SAMPLING_MANIFEST.csv','SAMPLING_RATIONALE.md','CHECKPOINT_ADMISSION.csv',
        'CONFIGURATION_METRICS.csv','PREDICTION_EVIDENCE_MANIFEST.json','STATISTICAL_METHODS.json','RQ1_RQ4_EVIDENCE.json',
        'RESULT_VERIFICATION.json','RESOURCE_BUDGET.json','summary.json','result.json']
    for name in required:
        if not (out/name).is_file():raise ValueError('Missing required artifact: '+name)
    if sha256(out/'EXPERIMENT_PROTOCOL.json')!=(out/'PROTOCOL_SHA256.txt').read_text().split()[0]:raise ValueError('Protocol hash changed.')
    proposal=load(out/'EXPERIMENT_PROTOCOL.json')
    if sha256(FMP)!=proposal['original_FM_P_sha256'] or sha256(FMD)!=proposal['original_FM_D_sha256']:raise ValueError('Active UVL changed.')
    if sha256(LEGACY_FMP)!=proposal['historical_FM_P_sha256'] or sha256(LEGACY_FMD)!=proposal['historical_FM_D_sha256']:raise ValueError('Historical UVL changed.')
    if proposal['GPU_execution_authorized'] or proposal['training_authorized']:raise ValueError('Preparation must not grant execution approval.')
    from historical_review import table
    if table(out/'CONFIGURATION_METRICS.csv'):raise ValueError('No new detector metrics were authorized or produced.')
    if load(out/'PREDICTION_EVIDENCE_MANIFEST.json')['new_predictions']:raise ValueError('Unexpected new predictions.')
    if load(out/'summary.json')['new_measured_pairs']!=0:raise ValueError('New-pair count is incorrect.')
    if load(out/'DATASET_ADMISSION.json')['development_validation']['images']!=548:raise ValueError('Incomplete population.')
    checks={}
    for key,record in load(out/'inputs.json').items():
        if sha256(Path(record['path']))!=record['sha256']:raise ValueError('Source input changed: '+key)
        checks[key]=record['sha256']
    save(out/'PRESERVED_INPUTS.json',{'status':'VALIDATED','checked_input_hashes':checks,
         'scope':'Referenced source files and model identities, not a new whole-archive scan.'},out)
    # Clarify an overbroad draft sentence without editing the frozen proposal.
    clarification={'status':'VALIDATED_FACTUAL_CLARIFICATION','affected_decision':'D03',
       'source':source(out/'CHECKPOINT_ADMISSION.csv'),'observed_training_library_version':'8.4.33 for all three primary checkpoints',
       'observed_shared_settings':'Recorded 1280 / 100 configured epochs / seed42 / AdamW lr0 .01 / batch2 / nbs64 / AMP=True.',
       'correction':'The proposal sentence about differing training/library provenance is overbroad. Recorded values match. Selected epochs and creation-time data/initialization byte identities remain unknown.',
       'scientific_selection_changed':False,'experiment_executed':False,'original_proposal_modified':False}
    save(out/'PROTOCOL_FACTUAL_CLARIFICATION.json',clarification,out)
    save(out/'PROTOCOL_REVIEW_BUNDLE.json',{'status':'NEEDS_SCIENTIFIC_REVIEW',
       'base_protocol':source(out/'EXPERIMENT_PROTOCOL.json'),'factual_clarification':source(out/'PROTOCOL_FACTUAL_CLARIFICATION.json'),
       'implementation_bindings':source(out/'IMPLEMENTATION_BINDINGS_PROPOSAL.json'),
       'student_directive':source(out/'STUDENT_ALIGNMENT_DIRECTIVE.json'),'execution_approval':None},out)
    save(out/'source_versions.json',{str(p.relative_to(PUBLIC)).replace('\\','/'):sha256(p)
       for p in sorted(HERE.rglob('*')) if p.is_file() and p.suffix in ('.py','.uvl','.md')},out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'SUPERVISOR_ALIGNED_PREPARATION',
        'status':'PARTIAL','engineering_scope':'Source alignment, full548 dataset audit reuse, saved-evidence CPU validation.',
        'GPU_seconds':0,'new_measured_pairs':0,'formal_measured_pairs':0,'ended_at_utc':datetime.now(timezone.utc).isoformat(),
        'scientific_approval':'NOT_GRANTED','source_model_alignment':'VALIDATED'},out,replace=True)
    save(out/'nested_manifest.json',{str(p.relative_to(out)).replace('\\','/'):sha256(p)
        for p in sorted(out.rglob('*')) if p.is_file() and not p.is_relative_to(out/'runtime') and not p.is_relative_to(out/'staging')},out)
    seal(out)
    root=out.parent;items=[]
    folders=sorted(root.iterdir(),key=lambda f:(re.search(r'_(\d{8}T\d{6})_',f.name).group(1) if re.search(r'_(\d{8}T\d{6})_',f.name) else '',f.name),reverse=True)
    for folder in folders:
        if not folder.is_dir() or not (folder/'run.json').is_file():continue
        r=load(folder/'run.json');target='index.html' if (folder/'index.html').is_file() else 'run.json'
        items.append(f'<li><a href="{folder.name}/{target}">{folder.name}</a> · {r.get("status")}</li>')
    write_atomic(root/'index.html','<!doctype html><meta charset="utf-8"><title>VM36 history</title><style>body{font:17px Arial;max-width:1100px;margin:40px auto}li{margin:12px 0}</style><h1>VM36 · Evidence history</h1>'+f'<p><a href="{out.name}/index.html">Latest source-aligned preparation</a></p><p>No new detector pair or training was executed.</p><ul>'+''.join(items)+'</ul>',out,replace=(root/'index.html').exists())
    save(root/'latest.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'status':'PARTIAL','mode':'SUPERVISOR_ALIGNED_PREPARATION',
        'summary_sha256':sha256(out/'summary.json'),'new_measured_pairs':0},out,replace=(root/'latest.json').exists())
    print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    if Path(args.run).name!=args.run:raise ValueError('Expected a plain RunID.')
    verify(ROOT/'datasets/local_runs'/EXPERIMENT/args.run)
