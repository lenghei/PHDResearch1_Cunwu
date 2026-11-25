"""Frozen inputs, local writes and fail-closed VM43 admission."""
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'vm37_yolo26_rq4'))
import study
ROOT,PUBLIC=study.ROOT,study.PUBLIC
EXPERIMENT=HERE.name;RUNS=ROOT/'datasets/local_runs'/EXPERIMENT
V42=ROOT/'datasets/local_runs/vm42_independent_readiness/vm42_independent_readiness_readiness_20261116T111422_8c956229'
V41=ROOT/'datasets/local_runs/vm41_degradation_coverage/vm41_degradation_coverage_coverage_20261116T020508_dfa7879d'
AUTH=study.RUNS/'vm37_yolo26_rq4_pilot10_authorization_cpu_20261113T212036_50286ea7'
BUNDLE=study.RUNS/'vm37_yolo26_rq4_portable_v3_build_cpu_20261113T115150_6e2bcebf/bundle'
load=study.load;sha=study.sha256


def module(name,file):
    spec=importlib.util.spec_from_file_location(name,file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def source(file):
    file=Path(file).resolve(strict=True)
    return {'path':str(file),'SHA256':sha(file),'bytes':file.stat().st_size}


def output(path):
    path=Path(path).resolve()
    if not path.is_relative_to(RUNS.resolve()):raise ValueError('Write outside VM43 run root.')
    return path


def save(path,value):
    path=output(path)
    if path.exists():raise FileExistsError(path)
    temporary=path.with_name(path.name+'.partial')
    with temporary.open('x',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    if path.exists():raise FileExistsError(path)
    temporary.rename(path)


def frozen():
    spec=load(V42/'Pilot_4cell_frozen_protocol.json')
    if spec['id']!='VM42_OPTION_A_V1' or spec['training_seeds']!=[42] or spec['cells_count']!=4 or spec['images_per_cell']!=1610:
        raise ValueError('VM42 Option A scope differs.')
    expected={(42,a,c) for a in ('CONTROL','VARIABILITY_GUIDED') for c in ('Normal','Gaussian_s15')}
    if {(r['seed'],r['strategy'],r['condition']) for r in spec['cells']}!=expected:raise ValueError('Unmatched four-cell scope.')
    for key,value in {'batch':1,'precision':'FP32','AMP':False,'TF32':False,'conf':.001,'NMS_IoU':.7,'max_det':3000,'max_nms':30000,'square_letterbox':True}.items():
        if spec['inference'][key]!=value:raise ValueError('Inference setting changed: '+key)
    if sha(Path(spec['population']['path']))!=spec['population']['SHA256']:raise ValueError('Frozen population changed.')
    if sha(Path(spec['timing_protocol']['path']))!=spec['timing_protocol']['SHA256']:raise ValueError('VM41 timing protocol changed.')
    return spec


def budget():
    import pilot10
    auth=load(AUTH/'authorization.json');files=pilot10.charge_files(AUTH)
    return {'authorization':source(AUTH/'authorization.json'),'total_seconds':auth['max_cumulative_GPU_worker_wall_seconds'],
        'charged_seconds':sum(load(f)['charged_seconds'] for f in files),'remaining_seconds':pilot10.available(AUTH),
        'deadline_UTC':auth['absolute_expiry_UTC'],'charges':[source(f) for f in files],
        'new_charge_seconds':0,'reset':False}


def gates(out,decision=None,authorization=None,thermal=None):
    now=datetime.now(timezone.utc);spec=frozen();b=budget();failures=[];scope=None
    scientific=load(decision) if decision else None
    gpu=load(authorization) if authorization else None
    cooling=load(thermal) if thermal else None
    if not scientific:failures.append('DOCUMENTED_SCIENTIFIC_DECISION_MISSING')
    else:
        scope=scientific.get('evaluation_scope')
        common_ok=scientific.get('RunID')==out.name and scientific.get('protocol_SHA256')==sha(V42/'Pilot_4cell_frozen_protocol.json') and bool(scientific.get('approved_by')) and bool(scientific.get('primary_evidence'))
        if scope=='INDEPENDENT_ADMITTED':
            common_ok=common_ok and all(scientific.get(k)=='CONFIRMED' for k in ('historical_use_review','selection_exposure_review','scene_provenance_review','annotation_ignore_compatibility'))
        elif scope=='LIMITED_EXTERNAL_EXPLORATORY':
            common_ok=common_ok and scientific.get('explicitly_allows_historical_independence_uncertainty') is True and scientific.get('annotation_ignore_compatibility')=='CONFIRMED'
        else:common_ok=False
        if not common_ok:failures.append('SCIENTIFIC_SCOPE_OR_EVIDENCE_INVALID')
    minimum=load(V42/'Thermal_resource_admission.json')['option_cost_summaries'][0]['complete_planning_envelope_seconds']
    if not gpu:failures.append('EXPLICIT_GPU_AUTHORIZATION_MISSING')
    else:
        valid=(gpu.get('RunID')==out.name and gpu.get('scope')=='VM43_OPTION_A_FOUR_CELL_EVALUATION_ONLY'
            and gpu.get('protocol_SHA256')==sha(V42/'Pilot_4cell_frozen_protocol.json') and gpu.get('original_authorization_SHA256')==b['authorization']['SHA256']
            and gpu.get('complete_four_cell_cost_admitted') is True and bool(gpu.get('cost_evidence')) and bool(gpu.get('approved_by')))
        if not valid:failures.append('GPU_SCOPE_OR_COMPLETE_COST_NOT_ADMITTED')
        cap=gpu.get('max_worker_seconds',0);reserve=gpu.get('shutdown_reserve_seconds',0)
        if cap<=0 or reserve<1800 or cap+reserve<minimum or cap+reserve>b['remaining_seconds']:
            failures.append('COMPLETE_FOUR_CELL_CAP_AND_RESERVE_DO_NOT_FIT')
    if not cooling:failures.append('CURRENT_THERMAL_ADMISSION_MISSING')
    else:
        try:
            valid_until=datetime.fromisoformat(cooling['valid_until_UTC'].replace('Z','+00:00'))
            valid=(cooling.get('RunID')==out.name and valid_until>now and all(cooling.get(k) is True for k in
                ('hard_ventilated_surface','unobstructed_intake_exhaust','fans_normal','reliable_AC_adapter','bounded_thermal_admission_passed'))
                and bool(cooling.get('evidence')) and bool(cooling.get('confirmed_by')))
        except (ValueError,KeyError,TypeError):valid=False
        if not valid:failures.append('CURRENT_THERMAL_ADMISSION_INVALID')
    if now>=datetime.fromisoformat(b['deadline_UTC']):failures.append('ORIGINAL_DEADLINE_EXPIRED')
    policy=load(BUNDLE/'safety_policy.json')
    if (policy['immediate_stop_temperature_C'],policy['sustained_stop_temperature_C'],policy['sustained_seconds'],policy['poll_seconds'])!=(90,87,30,2):
        failures.append('ORIGINAL_SAFETY_THRESHOLDS_CHANGED')
    return {'all_gates_pass':not failures,'status':'ADMITTED' if not failures else 'BLOCKED','failures':failures,
        'evaluation_scope':scope or 'NOT_ADMITTED','independence':'NOT_ESTABLISHED' if scope!='INDEPENDENT_ADMITTED' else 'DOCUMENTED_PRIMARY_EVIDENCE_REVIEW_REQUIRED',
        'budget':b,'planning_envelope_seconds_ESTIMATE':minimum,'new_GPU_executed':False,'safety_policy':source(BUNDLE/'safety_policy.json')}
