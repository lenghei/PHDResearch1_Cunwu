"""Version the portable orchestration fix without altering the v1 bundle."""
from pathlib import Path
import shutil
import sys
import time

POST=Path(__file__).resolve().parent
sys.path.insert(0,str(POST.parent))
from study import *
from portable import verify_bundle


def execute():
    out=create('portable_v2_cpu');runtime(out,cuda=False);started=time.perf_counter()
    prior=run_path('vm37_yolo26_rq4_portable_bundle_cpu_20261112T140353_793d9ad0')/'portable_bundle'
    manifest=verify_bundle(prior)
    bundle=out/'bundle';bundle.mkdir()
    for relative in manifest['files']:
        target=bundle/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(prior/relative,target)
    changed={}
    for name in ('portable.py','test_post_cost.py','verify_portable.py'):
        relative='main/experiments/vm37_yolo26_rq4/post_cost/'+name
        changed[relative]={'previous_SHA256':manifest['files'][relative],'new_SHA256':sha256(POST/name)}
        shutil.copyfile(POST/name,bundle/relative)
    note='\n## Execution wrapper v2\n\nThe complete matrix must pass measured resource admission before its first worker. Later workers validate that initial receipt and the actual remaining budget; they do not demand the original complete-matrix budget again. Historical and current-execution charges both count. Initialization and frozen bundle hashes are rechecked per training arm. Training protocols and numerical algorithms are unchanged. CPU validation does not establish GPU matrix readiness.\n'
    with (bundle/'README.md').open('a',encoding='utf-8') as stream:stream.write(note)
    revision={'version':'VM37_PORTABLE_V2','prior_bundle_manifest_SHA256':sha256(prior/'bundle_manifest.json'),
        'changes':changed,'training_parameters_changed':False,'scientific_results_changed':False,
        'reason':['Avoid re-admitting the original whole matrix before every later arm.',
                  'Subtract historical and current execution charges together.',
                  'Recheck frozen assets and initialization identity per training arm.'],
        'GPU_matrix_execution':'NOT_RUN'}
    save(bundle/'V2_REVISION.json',revision,out)
    updated={**manifest,'execution_wrapper_version':'VM37_PORTABLE_V2',
        'files':{p.relative_to(bundle).as_posix():sha256(p) for p in sorted(bundle.rglob('*')) if p.is_file()}}
    save(bundle/'bundle_manifest.json',updated,out)
    save(out/'bundle_result.json',{'bundle':str(bundle),'bundle_manifest_SHA256':sha256(bundle/'bundle_manifest.json'),
        'status':'ASSEMBLED_V2_NOT_GPU_VALIDATED','prior_bundle':str(prior),'revision':revision,
        'CPU_seconds':time.perf_counter()-started,'GPU_seconds':0,'external_uploads':0},out)
    write_atomic(out/'diary.md','# Portable v2\n\nPreserved the original bundle. Versioned two budget-accounting fixes and per-arm source checks. Numerical training/evaluation code and protocols are unchanged. No GPU execution or upload.\n',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'PORTABLE_V2_CPU',
        'status':'ASSEMBLED_V2_NOT_GPU_VALIDATED','GPU_seconds':0},out,replace=True)
    seal(out);print(out,flush=True)


if __name__=='__main__':execute()
