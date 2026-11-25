"""Audit the supervisor plan without changing historical experiments."""
import argparse
import ast
from collections import Counter
from datetime import datetime,timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import urllib.request
from uuid import uuid4

from common import ROOT,PUBLIC,HERE,RUNS,FMP,FMD,VAL,FLAMA,load,source,sha256,new_run,resolve_run,seal,verify_hashes
from historical_review import csv_file,table
from supervisor_requirements import requirements,FIGURES
from development_guard import bounded_run

VERSION='CV261207_SUPERVISOR_COMPLIANCE_V1'
BASE=RUNS/'cv261207_rq_acceptance_20261107T220632_8271fd92'
DEV=RUNS/'cv261207_development_analysis_20261108T194456_6e430eb3'
STUDY=RUNS/'cv261207_development_20261107T235344_5fcd2d79'
DELIVERY=RUNS/'cv261207_development_delivery_20261108T194939_4be5d3c4'
ADMIT=RUNS/'cv261207_admission_20261104T211249_32d83def'
ARCHIVE=ROOT/'datasets/research_archive'


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC',text).split())


def numeric_attributes(text):
    rows={}
    for match in re.finditer(r'\b([A-Za-z_]\w*)\s*\{([^{}]*)\}',text,re.S):
        for key,value in re.findall(r'\b([A-Za-z_]\w*)\s+(-?(?:\d+(?:\.\d*)?|\.\d+))\b',match[2]):
            number=float(value)
            rows[match[1],key]=number
    return rows


def attribute_diff(pdf_attributes,uvl_attributes):
    result=[]
    for feature,key in sorted(pdf_attributes.keys() | uvl_attributes.keys()):
        a,b=pdf_attributes.get((feature,key)),uvl_attributes.get((feature,key))
        status='MATCH' if a==b else 'MISSING_FROM_REPOSITORY_UVL' if b is None else 'IMPLEMENTATION_EXTENSION' if a is None else 'VALUE_MISMATCH'
        result.append({'feature':feature,'attribute':key,'PDF_value':a,'repository_value':b,'status':status})
    return result


def write_atomic(path,text,out,replace=False):
    path=Path(path).resolve();allowed=ROOT.resolve()
    if not path.is_relative_to(allowed):raise ValueError('Write outside the project.')
    path.parent.mkdir(parents=True,exist_ok=True)
    staging=out/'staging';staging.mkdir(exist_ok=True)
    temporary=staging/(uuid4().hex+'.tmp')
    expected=sha256(path) if path.exists() else None
    if expected and not replace:raise FileExistsError(path)
    try:
        with temporary.open('xb') as stream:
            stream.write(text.encode('utf-8'));stream.flush();os.fsync(stream.fileno())
        if expected and sha256(path)!=expected:raise ValueError('Concurrent output change.')
        if expected:os.replace(temporary,path)
        else:
            os.link(temporary,path)
            temporary.unlink()
    finally:
        if temporary.exists():temporary.unlink()


def save(path,value,out,replace=False):
    write_atomic(path,json.dumps(value,indent=2,ensure_ascii=True,allow_nan=False),out,replace)


def pdf_source(pdf,pdf_python,out):
    command=[str(pdf_python),'-c',
        'import json,sys,importlib.metadata;from pypdf import PdfReader;r=PdfReader(sys.argv[1]);print(json.dumps({"reader":importlib.metadata.version("pypdf"),"pages":[p.extract_text() for p in r.pages]},ensure_ascii=True))',str(pdf)]
    p=subprocess.run(command,capture_output=True,text=True,check=True,timeout=60)
    result=json.loads(p.stdout)
    if len(result['pages'])!=17 or any(not s.strip() for s in result['pages']):raise ValueError('Expected 17 readable plan pages.')
    canonical=ARCHIVE/'supervisor/Research_plan_Cunwu_UAV_gap.pdf'
    copied=False
    if canonical.exists():
        if sha256(canonical)!=sha256(pdf):raise ValueError('Conflicting supervisor-plan snapshot.')
    else:
        canonical.parent.mkdir(parents=True,exist_ok=True)
        with canonical.open('xb') as destination,pdf.open('rb') as original:shutil.copyfileobj(original,destination)
        copied=True
    if sha256(canonical)!=sha256(pdf):raise ValueError('PDF snapshot copy differs.')
    save(out/'pdf_pages.json',result,out)
    save(out/'pdf_provenance.json',{'original':source(pdf),'canonical':source(canonical),
        'single_missing_snapshot_created':copied,'page_count':17,'archive_recopied':False},out)
    return result['pages'],canonical


def references(plan,out):
    # Inventory only. The student's reading and summaries are not inferred.
    text='\n'.join(plan[12:14])
    identifiers=re.findall(r'(?:doi:|arXiv:)\s*([^\s]+)',text)
    save(out/'reading_list_gate.json',{'status':'PARTIAL','source_pages':[13,14],
         'plan_identifiers_as_extracted':identifiers,'verification':'PLAN_REPORTED; not newly verified against original papers',
         'original_PDF_availability':'NOT_AUDITED_FROM_FILENAME_ALONE','student_authored_summaries':'UNVERIFIED',
         'human_reading':'UNVERIFIED','automatic_scientific_summaries_written':False},out)


def audit_parameters(plan,out):
    rows=[]
    for name,pages,path in (('FM_P',plan[4:6],FMP),('FM_D',plan[6:9],FMD)):
        a=numeric_attributes('\n'.join(pages));b=numeric_attributes(path.read_text(encoding='utf-8'))
        for row in attribute_diff(a,b):rows.append({'domain':name,**row,'UVL_sha256':sha256(path)})
    csv_file(out/'parameter_origin_audit.csv',rows)
    summary={name:dict(Counter(r['status'] for r in rows if r['domain']==name)) for name in ('FM_P','FM_D')}
    save(out/'parameter_gate.json',{'status':'PARTIAL','counts':summary,
       'repository_UVLs_changed':False,'PDF_defined_values_silently_replaced':False,
       'proposed_FM_D_action':'Restore draft attribute metadata only in a separately reviewed version; retain generic params_m as draft values, not real checkpoint counts.',
       'runtime_action':'Current measured settings retain their frozen explicit runtime bindings; absent UVL attributes are not invented.',
       'new_development_conditions':'Downscale_x1_5, Normal and JPEG_q25 use matching PDF-defined values.',
       'extension_action':'Keep the 12 existing implementation attributes separate from the supervisor PDF definitions until reviewed.'},out)
    constraints=table(BASE/'uvl_constraints.csv')
    physics=[{'constraint':r['constraint'],'PDF_pages':'5;6','evidence':'Plan calls the initial constraints provisional.',
         'status':'PARTIAL','physical_reference':None,'human_justification':None,
         'next_action':'Student/supervisors review a direct physical reference or revise a new model version.'}
         for r in constraints if r['domain']=='FM_P']
    csv_file(out/'physical_justification_matrix.csv',physics)
    return summary


def pinned_training_source(out):
    folder=out/'sources';folder.mkdir()
    url='https://raw.githubusercontent.com/ultralytics/ultralytics/v8.4.33/ultralytics/data/augment.py'
    try:
        with urllib.request.urlopen(url,timeout=25) as response:data=response.read()
        path=folder/'ultralytics_8.4.33_augment.py'
        path.write_bytes(data);tree=ast.parse(data.decode('utf-8'))
        scope=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Albumentations')
        calls=[]
        for node in ast.walk(scope):
            if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute):continue
            if not isinstance(node.func.value,ast.Name) or node.func.value.id!='A':continue
            for kw in node.keywords:
                if kw.arg=='p' and isinstance(kw.value,ast.Constant):
                    calls.append({'transform':node.func.attr,'probability':kw.value.value,'source_line':node.lineno})
        report={'status':'VALIDATED_SOURCE_ONLY','URL':url,'source':source(path),'probabilities':calls,
             'execution':'Downloaded source was parsed, never imported or executed.',
             'historical_activation':'UNKNOWN unless a checkpoint-specific execution record proves it.'}
    except Exception as error:report={'status':'BLOCKED_SOURCE_FETCH','URL':url,'error':str(error)}
    save(out/'historical_training_source.json',report,out)
    return report


def evidence_audit(out):
    inputs={
      'logic':(BASE,'rq1_validation.json'),'static':(BASE,'static_coverage_by_checkpoint.csv'),
      'sampling':(RUNS/'cv261207_audit_20261104T193649_d1b63492','selection_manifest.json'),
      'design_v2':(BASE,'cross_architecture_v2.json'),'historical':(BASE,'historical_response_union.json'),
      'learning':(DEV,'learning_acceptance.json'),'development':(DEV,'configuration_metrics.csv'),
      'uncertainty':(DEV,'confidence_intervals.csv'),'states':(DEV,'pair_states.csv'),
      'gaps':(DEV,'subset_gaps.csv'),'comparison':(DEV,'static_empirical_diagnostics.csv'),
      'independence':(ADMIT,'admission.json'),
      'canary':(RUNS/'cv261207_clean_canary_20261108T194319_0384e775','result.json'),
      'rehearsal':(RUNS/'cv261207_rehearsal_20261108T191430_d2a88118','result.json'),
      'figures':(RUNS/'cv261207_development_figures_20261108T194742_2b59ab1d','figure_evidence.json')}
    public={};private={}
    for name,(folder,file) in inputs.items():
        checked=verify_hashes(folder,(file,'run.json'))
        public[name]={'RunID':folder.name,'artifact':file,'sha256':checked[file]}
        private[name]=source(folder/file)
    logic=load(BASE/'rq1_validation.json');gate=load(ADMIT/'admission.json')
    source0=load(RUNS/'cv261207_audit_20261104T193649_d1b63492/model_validation.json')
    if sha256(FMP)!=source0['FM_P']['UVL_SHA256'] or sha256(FMD)!=source0['FM_D']['UVL_SHA256']:raise ValueError('UVL identity changed.')
    data=table(DEV/'configuration_metrics.csv');seen={r['RunID'] for r in data}
    if len(seen)!=12:raise ValueError('Expected the accepted 12 unique development pairs.')
    for name in seen:
        folder=resolve_run(name)
        verify_hashes(folder,('predictions.jsonl.gz','ground_truth.json','metrics.json','verification.json','bootstrap.json'))
        if not load(folder/'verification.json')['COCO_recomputed_from_raw']:raise ValueError('No raw replay certificate.')
    for item in load(STUDY/'protocol.json')['detectors']:
        if sha256(Path(item['path']))!=item['sha256']:raise ValueError('Checkpoint changed.')
    uniform=load(inputs['sampling'][0]/inputs['sampling'][1]);v2=load(BASE/'cross_architecture_v2.json')
    fmp=__import__('sampling').models()[0]
    full_uniform=uniform['full_space_uniform_candidates'];pairwise=uniform['pairwise']['cohort']
    selected={p['degradation']['degradation_id']:p['degradation']['leaves'] for p in uniform['pairs']}
    for leaves in selected.values():
        if not fmp.valid_leaves(leaves) or 'RollingShutter' in leaves:raise ValueError('Invalid frozen executable sample.')
    reconciliation={'status':'PARTIAL','supervisor_approximate_pairs':4000,'supervisor_approximate_P':200,
         'supervisor_approximate_D':20,'supervisor_checkpoint_target':[8,12],
         'v1_frozen_pairs':len(uniform['pairs']),'v2_prepared_pairs':len(v2['pairs']),
         'formal_completed_pairs':gate['formal_pairs_measured'],'historical_unique_responses':load(BASE/'historical_response_union.json')['unique_responses'],
         'new_development_pairs':len(seen),'new_development_image_IDs':248,'new_unique_image_hashes':247,
         'pairwise_full_population_configurations':len(pairwise),'pairwise_semantics':uniform['pairwise']['definition'],
         'pairwise_greedy_core_configurations':uniform['pairwise']['greedy_cover_size'],
         'pairwise_padding_configurations':len(pairwise)-uniform['pairwise']['greedy_cover_size'],
         'feasible_pair_states':uniform['pairwise']['feasible'],'achieved_pair_states':uniform['pairwise']['achieved'],
         'impossible_pair_states':uniform['pairwise']['impossible'],
         'full_space_uniform_count':len(full_uniform),'full_space_population':uniform['full_FM_P'],
         'eligible_uniform_count':len(selected),'eligible_population':uniform['eligible_FM_P'],
         'inclusion_probability':uniform['per_configuration_inclusion_probability'],
         'active_count':None,'active_status':'NOT_EVALUATED','BDD_sampling':False,
         'approximate_60_100':'The existing frozen cohort has 80 configurations: a 21-configuration greedy core plus 59 padding selections. Neither is changed.',
         'pooling':'Historical 300 and new 248 image cohorts are not merged by averaging AP.'}
    save(out/'sampling_reconciliation.json',reconciliation,out)
    save(out/'evidence_sources.json',private,out)
    save(out/'verified_counts.json',{'logic':logic,'independence':gate['status'],
         'historical_unique_responses':279,'new_development_pairs':12,'formal_pairs':0,
         'not_new_experiments':'Existing raw hashes and replay certificates were checked; YOLO and accepted COCO evaluations were not rerun.'},out)
    csv_file(out/'threshold_sensitivity.csv',table(DEV/'subset_gaps.csv'))
    csv_file(out/'static_empirical_diagnostics.csv',table(DEV/'static_empirical_diagnostics.csv'))
    return public,reconciliation


def model_and_coverage_audit(out,training_source):
    catalog=table(ARCHIVE/'migration_20261104/model_catalog.csv');health=load(ARCHIVE/'migration_20261104/core_model_cpu_health.json')
    health_lookup={r['sha256']:r for r in health['models']};rows=[]
    for entry in catalog:
        path=ROOT/entry['local_checkpoint'];exists=path.is_file()
        match=exists and sha256(path)==entry['sha256']
        if not match:raise ValueError('Catalogue checkpoint is missing or changed: '+entry['model_id'])
        h=health_lookup.get(entry['sha256'])
        rows.append({'model_id':entry['model_id'],'checkpoint_sha256':entry['sha256'],'file_verified':True,
          'health':h['status'] if h else entry['current_cpu_health'],'parameters':h.get('parameters') if h else None,
          'imgsz':entry['imgsz'],'configured_epochs':entry['epochs_configured'],'seed':entry['seed'],
          'optimizer':entry['optimizer'],'lr0':entry['lr0'],'batch':entry['batch'],
          'training_args_sha256':sha256(ROOT/entry['training_config']) if entry['training_config'] and (ROOT/entry['training_config']).exists() else None,
          'class_order':h.get('classes') if h else None,'native10_verified':bool(h and len(h.get('classes',[]))==10),
          'usage':entry['usage'],'formal_admission':'BLOCKED_INDEPENDENCE_OR_PROTOCOL',
          'independent_training_replicate':False,'review':'Do not count untrained initialization, quarantined tensors or external models as admitted baseline replicates.'})
    csv_file(out/'checkpoint_readiness.csv',rows)
    indexed=table(ARCHIVE/'migration_20261104/checkpoint_index.csv');variants=[]
    for row in indexed:
        if row['model_id'] not in ('y26_B_seed43','y26_B_seed44','B1_seed43','B1_seed44'):continue
        manifest=ROOT/row['training_and_results']/'manifest.json'
        if not manifest.is_file():continue
        m=load(manifest)
        variants.append({'model_id':row['model_id'],'checkpoint_sha256':row['sha256'],'seed':m.get('seed'),
             'additional_epochs':m.get('epochs'),'parent_sha256':m.get('parent_sha256'),
             'protocol_sha256':m.get('protocol_sha256'),'manifest_sha256':sha256(manifest),
             'status':'OTHER_METHOD_CONTINUATION_NOT_PRIMARY_BASELINE_REPEAT','CPU_health':row['current_cpu_health']})
    csv_file(out/'other_seed_variants.csv',variants)
    candidates=load(ADMIT/'admission.json')['candidates'];baseline=load(BASE/'static_baseline.json')
    static=[]
    for c in candidates:
        for leaf,record in baseline['leaves'].items():
            level=record['classification']
            if c['checkpoint_version']!=record['source_version'] and level=='WEAK_PROXY':level='UNKNOWN'
            static.append({'checkpoint':c['id'],'checkpoint_sha256':c['checkpoint_sha256'],'leaf':leaf,
                'classification':level,'training_transform':record['transform'],
                'parameters_from_existing_audit':json.dumps(record['parameter_range']),
                'probability_from_existing_audit':json.dumps(record['probability']),
                'checkpoint_version':c['checkpoint_version'],'source_audit_version':record['source_version'],
                'training_hsv_v':c['training_protocol'].get('hsv_v'),'training_scale':c['training_protocol'].get('scale'),
                'historical_runtime_activation':'UNKNOWN','pinned_8_4_33_source':training_source['status'],
                'policy':'Existing strict source-version policy preserved; pinned source availability is not activation evidence.'})
    if len(static)!=100:raise ValueError('Expected 25 leaves x four candidate checkpoints.')
    csv_file(out/'leaf_checkpoint_coverage.csv',static)
    csv_file(out/'static_coverage_policies.csv',table(BASE/'static_coverage_by_checkpoint.csv'))
    matrix=table(BASE/'fmd_compatibility.csv')
    tested={'YOLOv8','YOLO11','YOLO26','Small','VisDrone_FT','Infer640','Infer1280','FP32','FullFrame','NMS'}
    for row in matrix:
        f=row['feature'];row['LOGICALLY_VALID']=True
        row['GPU_TESTED_CURRENT_248_PROTOCOL']=f in tested
        row['current_evidence_scope']='Feature occurrence in three fixed FT1280 checkpoints, not every feature combination.'
        row['IMPLEMENTABLE_CURRENT_248_PROTOCOL']=f in tested or f in ('Train1280','HSV_Default','Geometric_Default')
        if f in ('YOLO11','YOLO26','Train1280'):row['note']='Current development_v1 backend and accepted 248-image evidence supersede the older YOLOv8-only implementation note.'
        if f.endswith('Aug'):row['note']='Pixel hook unit-tested; targeted-arm GPU efficacy not evaluated.'
    csv_file(out/'detector_compatibility.csv',matrix)
    return rows


def external_gate(out):
    path=ROOT/'datasets/UAVDT';meta=path/'meta.json'
    old=ARCHIVE/'variability/experiments/vm25_final_dataset_audit/runs/vm25_r001/results/dataset_gate.json'
    original=load(old)
    report={'status':'BLOCKED_ANNOTATION_PROVENANCE','local_root_exists':path.is_dir(),
         'metadata':source(meta) if meta.is_file() else None,'prior_gate':source(old),
         'prior_source_status':original['status'],'schema':'Converted Supervisely folders; native DET/MOT ignore semantics not restored.',
         'conditions_in_metadata':[r['name'] for r in load(meta).get('tags',[])] if meta.is_file() else [],
         'classes_in_metadata':[r['title'] for r in load(meta).get('classes',[])] if meta.is_file() else [],
         'checked_paths':{},'independent_protocol_created':False,'images_inferred':0}
    for split in ('train','test'):
        report['checked_paths'][split]={name:(path/split/name).is_dir() for name in ('img','ann','meta')}
    report['original_annotation_files_at_root']=[p.name for p in path.iterdir() if p.is_file() and p.suffix.lower() in ('.txt','.mat','.csv')]
    report['search_limitation']='Bounded named-path check, not proof that no original annotation exists elsewhere. Prior unresolved gate is retained.'
    save(out/'uavdt_gate.json',report,out)


def matrix(out,plan,evidence,stage,profile_status='NOT_EVALUATED',demo_status='NOT_EVALUATED'):
    statuses={'overall':'PARTIAL','rq1':'PARTIAL','rq2':'PARTIAL','rq3':'PARTIAL','release':'PARTIAL',
      'logic':'VALIDATED','generator':'VALIDATED','attributes':'PARTIAL','physics':'PARTIAL','calibration':'NOT_EVALUATED',
      'cache':'PARTIAL','metrics':'VALIDATED','source_version':'PARTIAL','d_attributes':'PARTIAL','deployment':'PARTIAL',
      'parameters':'VALIDATED','compatibility':'PARTIAL','static':'PARTIAL','oracle':'PARTIAL','sensitivity':'VALIDATED',
      'subset':'VALIDATED','full_gap':'NOT_EVALUATED','pruning':'NOT_EVALUATED','learned_constraints':'BLOCKED',
      'design':'PARTIAL','cost':'VALIDATED','independence':'BLOCKED','sampling':'VALIDATED','active':'BLOCKED','catalog':'PARTIAL',
      'uncertainty':'VALIDATED','seeds':'NOT_EVALUATED','learning':'PARTIAL','uavdt':'BLOCKED','training':'NOT_EVALUATED',
      'hooks':'VALIDATED','reading':'PARTIAL','scope':'VALIDATED','git':'PARTIAL','meeting':'PARTIAL','milestones':'PARTIAL',
      'week1':'PARTIAL','week2':'PARTIAL','week3':'PARTIAL','formal':'BLOCKED','venue':'NOT_EVALUATED','writing':'NOT_EVALUATED'}
    facts={
      'logic':'Unchanged SHA certificates: FM_P 25/5/46368; FM_D 34/4/635904; 844 training variants. SAT/dead-feature proofs reused; no new BDD claim.',
      'attributes':'PDF-defined FM_P values match; 12 implementation attributes are extra. FM_D is missing ten numeric draft attributes.',
      'generator':'Existing 25-leaf determinism tests and seven-stage order are verified. New readiness demo: '+demo_status+'. RollingShutter remains excluded from evaluation.',
      'd_attributes':'Missing draft metadata is reported, not inserted into frozen v1. Draft params_m is not a checkpoint measurement.',
      'independence':'Current file disjointness is verified; checkpoint-specific prior-use and creation history remain incomplete. Formal pairs 0/500.',
      'learning':'279 historical responses fitted the fixed regression baselines; 12 image/configuration-disjoint development responses tested them. This is not validated PASS/FAIL constraint learning.',
      'training':'One 64/16 clean-control engineering canary passed one optimizer/EMA update. Targeted arms, ablations, before/after gap and seed variability remain unmeasured.',
      'sampling':'The frozen pairwise cohort contains 80 configurations: a 21-configuration greedy cover plus 59 padding selections. It covers 1163 feasible binary states; 37 are impossible. Uniform cohorts have different populations.',
      'design':'500 v1 pairs and 900 v2 pairs remain plans; approximately 4000 in the supervisor plan is a larger target, not an executed manifest.',
      'static':'100 leaf/checkpoint source rows. Existing 1/3/18 reference policies remain separate; historical activation is unknown.',
      'oracle':'Existing decisions use cached-PNG pipeline timing. A deployment latency(d) protocol is not yet admitted; tau and L remain explicit.',
      'cost':'Saved 12-pair worker, forward, CPU evaluation, replay and bootstrap costs are distinct. No new GPU budget is inferred.',
      'parameters':'Saved checkpoint SHA and real parameter-count certificates verified; operation-profile status: '+profile_status+'.',
      'deployment':'GPU latency/memory already measured for a fixed subset. CPU structural operation counts are not GPU throughput or deployment latency.',
      'uavdt':'Converted data and condition tags exist, but original DET/ignore provenance and a compatible class protocol remain unresolved.',
      'release':'Local raw evidence and committed source are available; no external immutable raw release or approved exception to the raw-in-repo rule exists.',
      'meeting':'CPU readiness or saved outputs do not establish witnessed supervisor acceptance.',
      'rq1':'Logical validation is distinct from physical justification, numeric-origin completeness and universal executability.',
      'rq2':'Historical 279 responses plus 12 new development pairs are separate cohorts; formal test-dev evaluation is blocked.',
      'rq3':'Observed six-detector subset states and threshold sensitivity exist; population gap, learned gap and full-space count do not.',
      'seeds':'Admitted primary baselines have seed42. Indexed seed43/44 Paper2 continuations use different 15-epoch protocols and are not baseline repeats.'}
    output=[]
    for row in requirements():
        if normalize(row['exact_supervisor_requirement']) not in normalize(plan[row['pdf_page']-1]):
            raise ValueError('Requirement quote not found on its declared PDF page: '+row['id'])
        key=row['evidence_key'];refs=[evidence[n] for n in (key,) if n in evidence]
        if not refs:refs=[evidence['logic' if key in ('rq1','generator','attributes','d_attributes','physics') else 'development']]
        output.append({**row,'required_parameters':'See parameter_origin_audit.csv / frozen runtime bindings; no second parameter table.',
            'current_implementation':facts.get(key,'See the checked source artifacts and explicit missing acceptance condition.'),
            'existing_evidence':refs,'missing_evidence':None if statuses[key]=='VALIDATED' else row['acceptance_condition'],
            'current_status':statuses[key],'status_scope':'Only the stated evidence scope; no whole-study completion implied.',
            'supervisor_signoff':'UNRECORDED'})
    figure_status={i:'PARTIAL' for i in FIGURES}
    for i in (1,3,4,5,6):figure_status[i]='VALIDATED'
    figure_status[12]='NOT_EVALUATED'
    report={'schema':VERSION,'RunID':out.name,'stage':stage,'updated_utc':datetime.now(timezone.utc).isoformat(),
       'status':'PARTIAL','source_pdf_sha256':sha256(ARCHIVE/'supervisor/Research_plan_Cunwu_UAV_gap.pdf'),
       'source_pdf_pages':17,'requirements':output,'status_counts':dict(Counter(r['current_status'] for r in output)),
       'RQ_status':{q:s for q,s in (('RQ1','PARTIAL'),('RQ2','PARTIAL'),('RQ3','PARTIAL'),('RQ4','NOT_EVALUATED'))},
       'supervisor_contributions':{'SUP_C1':'PARTIAL','SUP_C2':'PARTIAL','SUP_C3':'PARTIAL','SUP_C4':'PARTIAL'},
       'contribution_mapping':'PDF p4 contribution numbers are not aliases for RQ numbers. In particular SUP_C4 is the replication package.',
       'figure_acceptance':[{'figure':f'Fig{i:02}','requirement':text,'status':figure_status[i]} for i,text in FIGURES.items()],
       'formal_completed_pairs':0,'historical_development_responses':279,'new_development_pairs':12,
       'new_GPU_seconds':0,'new_training_runs':0,'profile_status':profile_status,'CPU_demo_status':demo_status,
       'scientific_prose_authored':False,'public_paths':'Evidence is identified by RunID, artifact basename and SHA; full private paths remain local.'}
    save(out/f'compliance_stage_{stage:02}.json',report,out)
    lines=['# Supervisor plan compliance','',f'RunID: `{out.name}`. Status: **PARTIAL**. Source PDF: 17 pages; SHA256 `{report["source_pdf_sha256"]}`.',
       '', 'This is a requirements and engineering evidence record, not manuscript analysis or supervisor sign-off.',
       'PDF contribution numbers are distinct from RQ numbers; contribution 4 is the replication package.',
       '', '| Requirement | PDF page | Exact supervisor excerpt | Status | Checked implementation/evidence | Missing acceptance | Next action |',
       '|---|---:|---|---|---|---|---|']
    for r in output:
        values=[r['id'],str(r['pdf_page']),r['exact_supervisor_requirement'],r['current_status'],r['current_implementation'],r['missing_evidence'] or 'None within stated scope.',r['next_executable_action']]
        lines.append('| '+' | '.join(str(x).replace('|','\\|').replace('\n',' ') for x in values)+' |')
    lines+=['','## Current gates','', '- Formal test-dev: DATASET_BLOCKED; 0/500. No new GPU budget is inferred from the 70–100 hour planning estimate.',
      '- Existing development evidence is retained; no accepted YOLO/COCO experiment is rerun merely to reproduce it.',
      '- FM_P numeric extensions and missing FM_D draft attributes require explicit review. The original UVLs are unchanged.',
      '- Paired image bootstrap is distinct from training-seed variability. Duplicate-image/annotation and scene limits remain.',
      '- The original raw-in-repository requirement conflicts with the local-only publication policy. A review policy is provided.',
      '- RQ4 efficacy and full-gap shrinkage remain NOT_EVALUATED; the engineering canary is not a matched augmentation study.',
      '', '## Source evidence','']
    for key,ref in evidence.items():lines.append(f'- {key}: `{ref["RunID"]}/{ref["artifact"]}`, SHA256 `{ref["sha256"]}`.')
    lines+=['','## Reproduce the audit','', '```powershell',
      'python -B main/experiments/experiments_test/cv261207/supervisor_compliance.py --pdf <supervisor-plan.pdf> --pdf-python <python-with-pypdf>',
      '```','', 'This entry is CPU-only. It does not admit test-dev, start YOLO training or push Git.']
    md='\n'.join(lines)+'\n'
    write_atomic(out/f'compliance_stage_{stage:02}.md',md,out)
    return report


def run(args):
    started=time.perf_counter();out=new_run('supervisor_compliance')
    save(out/'execution_ticket.json',{'scope':'CPU_ONLY_SUPERVISOR_AUDIT','wall_budget_seconds':1800,
          'new_GPU_seconds':0,'training_authorized':False,'source':source(Path(__file__)),
          'reason':'The new request authorizes continued evidence work; a GPU-hour planning estimate is not an execution authorization.'},out)
    plan,_=pdf_source(args.pdf.resolve(strict=True),args.pdf_python,out)
    parameters=audit_parameters(plan,out);references(plan,out)
    evidence,reconciliation=evidence_audit(out)
    external_gate(out)
    matrix(out,plan,evidence,1)
    src=pinned_training_source(out);model_and_coverage_audit(out,src)
    matrix(out,plan,evidence,2)
    # Later optional engineering stages write their own actual status.
    save(out/'context.json',{'source_evidence':evidence,'sampling':reconciliation,'parameters':parameters,
                           'PDF_pages_artifact':'pdf_pages.json'},out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':'cv261207','mode':'SUPERVISOR_COMPLIANCE',
           'status':'PARTIAL','scope':'CPU_AUDIT','wall_seconds':time.perf_counter()-started,'GPU_seconds':0},out)
    print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pdf',type=Path,required=True)
    parser.add_argument('--pdf-python',type=Path,default=Path(os.environ.get('UAVGAP_PDF_PYTHON',sys.executable)))
    run(parser.parse_args())
