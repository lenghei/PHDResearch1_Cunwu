"""Run the focused compatibility and calculation-parity tests on CPU."""
import os
import subprocess
import sys
import time
from study import *


def execute():
    started=time.perf_counter();out=create('focused_regression_cpu');runtime(out,cuda=False)
    command=[sys.executable,'-B','-m','unittest','discover','-s',str(HERE/'tests'),'-v']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',
             PYTHONDONTWRITEBYTECODE='1',YOLO_AUTOINSTALL='false',YOLO_OFFLINE='true')
    with (out/'tests.log').open('x',encoding='utf-8') as log:
        completed=subprocess.run(command,cwd=HERE,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
    log=(out/'tests.log').read_text(encoding='utf-8')
    passed=completed.returncode==0 and '\nOK\n' in log and 'Ran 8 tests' in log
    save(out/'acceptance.json',{'status':'PASS' if passed else 'FAILED','test_count':8,
        'CPU_seconds':time.perf_counter()-started,'GPU_seconds':0,'returncode':completed.returncode,
        'scope':'Compatibility and parity tests, not GPU training stability or augmentation effectiveness.',
        'sources':{str(path.relative_to(PUBLIC)):sha256(path) for path in [HERE/'training.py',HERE/'tests/test_admission.py']+
                   [PUBLIC/'code/main'/n for n in ('training_health.py','training_data.py','training_effects.py','predictor_stable.py')]}},out)
    write_atomic(out/'diary.md','# Focused CPU regression\n\nChecked the installed warmup rule, NMS parity, private labels, pixel/target parity, strict finite checks and optimizer/EMA equality. See tests.log for actual outcomes. No CUDA model execution.\n',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'status':'PASS' if passed else 'FAILED'},out,replace=True)
    seal(out);print(out,flush=True)
    if not passed:raise SystemExit(1)
    return out


if __name__=='__main__':execute()
