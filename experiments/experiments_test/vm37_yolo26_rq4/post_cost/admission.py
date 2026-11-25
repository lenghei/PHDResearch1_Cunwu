"""Read-only resource and thermal admission after the accepted cost study."""
import ctypes
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
from study import ROOT,PUBLIC,RUNS,PREP,VM36,EXPERIMENT,create,runtime,load,save,source,sha256,seal,write_atomic


def execute():
    out=create('postcost_admission_cpu');runtime(out,cuda=False);started=time.perf_counter()
    authorization=Path(r'C:\Users\w\.codex\attachments\8cf4551a-28e0-44ea-8c6c-461b6b6b2d14\已粘贴的文本.txt')
    shutil.copyfile(authorization,out/'author_task.txt')
    ledger_path=RUNS/'compute_allocation.json';ledger_before=sha256(ledger_path);allocation=load(ledger_path)
    now=datetime.now(timezone.utc);deadline=datetime.fromisoformat(allocation['absolute_deadline_UTC'])
    spent=sum(row['charged_seconds'] for row in allocation['completed_workers'])
    resource={'UTC':now.isoformat(),'absolute_deadline_UTC':deadline.isoformat(),'total_authorized_seconds':allocation['total_seconds'],
        'charged_worker_seconds':spent,'unconsumed_worker_seconds':max(0,allocation['total_seconds']-spent),
        'deadline_remaining_seconds':max(0,(deadline-now).total_seconds()),
        'available_seconds':max(0,min(allocation['total_seconds']-spent,(deadline-now).total_seconds())),
        'ledger_SHA256':ledger_before,'disk_free_bytes':shutil.disk_usage(ROOT).free,'new_allocation_created':False}
    diagnostics={}
    for name,args in [('nvidia_full',['nvidia-smi','-q','-d','TEMPERATURE,PERFORMANCE,MEMORY,POWER']),
                      ('nvidia_compact',['nvidia-smi','--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu,driver_version','--format=csv,noheader,nounits'])]:
        result=subprocess.run(args,capture_output=True,text=True,timeout=15)
        write_atomic(out/(name+'.txt'),result.stdout+result.stderr,out)
        diagnostics[name]={'returncode':result.returncode,'evidence':source(out/(name+'.txt'))}
    class Power(ctypes.Structure):
        _fields_=[('ACLineStatus',ctypes.c_ubyte),('BatteryFlag',ctypes.c_ubyte),('BatteryLifePercent',ctypes.c_ubyte),
            ('SystemStatusFlag',ctypes.c_ubyte),('BatteryLifeTime',ctypes.c_ulong),('BatteryFullLifeTime',ctypes.c_ulong)]
    power=Power();ok=bool(ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(power)))
    physical={'AC_status_query_ok':ok,'AC_line_status':power.ACLineStatus if ok else None,
              'battery_percent':power.BatteryLifePercent if ok else None,
              'ventilation_clear':'UNKNOWN','hard_ventilated_surface':'UNKNOWN','fans_physical_health':'UNKNOWN',
              'adapter_rating_and_reliability':'UNKNOWN','system_settings_changed':False}
    traces={}
    for identity in ('vm37_yolo26_rq4_full_epoch_yolo26s_20261111T220632_79bf512c',
                     'vm37_yolo26_rq4_full_epoch_yolo26s_control_20261111T232439_301fb25d'):
        path=RUNS/identity/'gpu_telemetry.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        intervals=[];start=None;previous=None
        for row in rows+[{'UTC':rows[-1]['UTC'],'temperature_C':-999}]:
            when=datetime.fromisoformat(row['UTC'])
            if row['temperature_C']>=87:
                if start is None:start=when
                previous=when
            elif start is not None:
                intervals.append({'first_hot_sample_UTC':start.isoformat(),'last_hot_sample_UTC':previous.isoformat(),
                                  'sampled_span_seconds':(previous-start).total_seconds()});start=None
        traces[identity]={'input':source(path),'samples':len(rows),'maximum_temperature_C':max(r['temperature_C'] for r in rows),
            'samples_at_least87':sum(r['temperature_C']>=87 for r in rows),
            'samples_at_least90':sum(r['temperature_C']>=90 for r in rows),
            'maximum_sampled_memory_used_MiB':max(r['used_MiB'] for r in rows),
            'hot_sample_spans':intervals,'limitations':'Samples every10s across training and CPU evaluation. Sample spans do not prove continuous temperature between observations.'}
    safety={'version':'VM37_POSTCOST_SAFETY_V1','poll_seconds':2,'preflight_maximum_temperature_C':65,
        'preflight_minimum_free_MiB':7168,'immediate_stop_temperature_C':90,
        'sustained_stop_temperature_C':87,'sustained_seconds':30,'stop_on_telemetry_failure':True,
        'stop_on_cuda_or_numerical_error':True,'stop_only_owned_process_tree':True,'automatic_retry':False,
        'meaning':'Conservative research-task stop rules, not manufacturer damage limits.',
        'power_requirement':'AC connected and reliable adapter confirmed','cooling_requirement':'Physical ventilation/fan confirmation plus bounded monitored load admission.'}
    gate={'GPU_training':'BLOCKED_THERMAL_READINESS','multi_hour_unattended_training':'NO_GO',
        'bounded_ALL_FIVE_cost':'NOT_ADMITTED_PENDING_COOLING_AND_MONITORED_SAFETY',
        'formal100':'RESOURCE_BLOCKED','pilot20':'PILOT_RESOURCE_BLOCKED_PENDING_THERMAL_AND_ALL_FIVE_COST',
        'reason':'Idle temperature is not sustained-load validation. Prior optimized run sampled92C and near-capacity memory; physical cooling is unverified and ALL_FIVE GPU cost is unmeasured.',
        'independent_CPU_work':'CONTINUE','budget_reset':False}
    from pypdf import PdfReader
    pdf=ROOT/'datasets/research_archive/supervisor/Research_plan_Cunwu_UAV_gap.pdf'
    pages={str(i):page.extract_text() for i,page in enumerate(PdfReader(pdf).pages,1)}
    save(out/'original_supervisor_text.json',pages,out)
    inputs={'author_request':source(out/'author_task.txt'),'original_plan':source(pdf),
        'original_FMP':source(VM36/'feature_models/fm_p.uvl'),'original_FMD':source(VM36/'feature_models/fm_d.uvl'),
        'accepted84':source(PREP/'FROZEN_DEVELOPMENT_EVALUATION.json'),'policy':source(BASE/'TARGETED_POLICY.json'),
        'current_ledger':source(ledger_path),'diagnostics':diagnostics}
    for name,value in [('resource',resource),('physical_conditions',physical),('thermal_history',traces),('safety_policy',safety),('admission',gate),('inputs',inputs)]:
        save(out/(name+'.json'),value,out)
    assert sha256(ledger_path)==ledger_before
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'POSTCOST_ADMISSION_CPU',
        'status':'COMPLETE_WITH_LIMITATIONS','CPU_seconds':time.perf_counter()-started,'GPU_seconds':0},out,replace=True)
    write_atomic(out/'diary.md','# Post-cost admission\n\nRead the original plan and existing evidence. Recomputed the unchanged deadline/ledger and inspected current GPU/power plus historical thermal samples. No training, inference or settings changes. Physical cooling and ALL_FIVE cost remain gates.\n',out)
    seal(out);print(out,flush=True)
    print(json.dumps({'available_hours':resource['available_seconds']/3600,'thermal_gate':gate['GPU_training'],
                      'old_peak_C':traces['vm37_yolo26_rq4_full_epoch_yolo26s_control_20261111T232439_301fb25d']['maximum_temperature_C']}))


if __name__=='__main__':execute()
