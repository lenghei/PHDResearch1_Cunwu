"""Validate an assembled bundle on CPU, including relocated dry-runs."""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
import traceback

POST = Path(__file__).resolve().parent
sys.path.insert(0, str(POST.parent))
from study import *


def execute(bundle, data_root):
    out = create('portable_verification_cpu')
    runtime(out, cuda=False)
    started = time.perf_counter()
    statuses = []
    entry = bundle / 'main/experiments/vm37_yolo26_rq4/post_cost'
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='', OMP_NUM_THREADS='2')
    commands = [
        [sys.executable, '-B', '-X', 'utf8', str(entry / 'test_post_cost.py')],
        *[[sys.executable, '-B', '-X', 'utf8', str(entry / 'portable.py'), '--bundle', str(bundle),
           '--data-root', str(data_root), '--protocol', name] for name in ('pilot20', 'formal100')],
    ]
    for index, command in enumerate(commands):
        result = subprocess.run(command, cwd=bundle, env=env, capture_output=True, text=True, encoding='utf-8', timeout=600)
        write_atomic(out / ('test_' + str(index) + '.log'), result.stdout + '\n' + result.stderr, out)
        statuses.append({'command': command, 'returncode': result.returncode})
        if result.returncode:
            save(out / 'failure.json', {'status': 'CPU_VERIFICATION_FAILED', 'command': command, 'returncode': result.returncode}, out)
            save(out / 'run.json', {'RunID': out.name, 'ExperimentID': EXPERIMENT, 'status': 'CPU_VERIFICATION_FAILED', 'GPU_seconds': 0}, out, replace=True)
            seal(out)
            raise RuntimeError('Portable CPU test failed; preserve this run and fix a separate bundle version.')
    check_code = '''import sys,json,hashlib,gc
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from portable import *
bundle=Path(sys.argv[2]);data_root=Path(sys.argv[3]);out=Path(sys.argv[4])
runtime(out,gpu=False)
import torch,cv2,numpy as np
torch.set_num_threads(2);cv2.setNumThreads(2)
from main.training_health import model_health
from main.labels import VIS_NAMES,ground_truth
from training import trainer_class
environment_check(bundle);verify_bundle(bundle)
population=resolve_data(bundle,data_root)
health=[]
for row in load(bundle/'portable_initializations.json'):
    model=torch.load(within(bundle,row['asset']),map_location='cpu',weights_only=False)['model']
    model_health(model)
    assert list(model.names.values())==VIS_NAMES
    assert all(not v.is_floating_point() or v.dtype==torch.float32 for v in model.state_dict().values())
    health.append({'architecture':row['architecture'],'seed':row['seed'],'sha256':row['sha256'],'finite':True,'FP32':True,'classes':VIS_NAMES})
    del model;gc.collect()
records=load(bundle/'assets/ground_truth.json')['records']
assert {r['stem'] for r in records}=={r['image_id'] for r in population['val']}
gt,regions=ground_truth(records,'native10')
assert len(gt['images'])==548 and len(gt['categories'])==10
ops=renderer();checks=[]
for spec_name in ('evaluation_manifest.json','formal_evaluation_manifest.json'):
    spec=load(bundle/spec_name)
    for case in spec['contexts']:
        expected={Path(r['source']['ZIP_member']).stem:r for r in map(json.loads,within(bundle,case['pixel_manifest_asset']).read_text().splitlines())}
        assert len(expected)==548
        for source in population['val'][:6]:
            image=cv2.imdecode(np.frombuffer(Path(source['image_path']).read_bytes(),np.uint8),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            pixels,trace=ops.generate(image,case['p']['leaves'],case['p']['attributes'],source['image_id'],spec['render_seed'])
            actual=hashlib.sha256(pixels.tobytes()).hexdigest()
            assert actual==expected[source['image_id']]['pixel_sha256']
            assert list(pixels.shape)==expected[source['image_id']]['shape']
            checks.append({'manifest':spec_name,'condition':case['id'],'image':source['image_id'],'pixel_SHA256':actual})
save(out/'health_and_pixels.json',{'status':'PASS_CPU_ONLY','initializations':health,'pixel_checks':checks,'GT_images':len(gt['images']),'GT_instances':len(gt['annotations']),'GPU_forward_backward':0,'trainer_factory_importable':callable(trainer_class)})
'''
    result = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-c', check_code, str(entry), str(bundle), str(data_root), str(out)],
                            cwd=bundle, env=env, capture_output=True, text=True, encoding='utf-8', timeout=600)
    write_atomic(out / 'health_and_pixels.log', result.stdout + '\n' + result.stderr, out)
    if result.returncode:
        save(out / 'failure.json', {'returncode': result.returncode, 'status': 'CPU_VERIFICATION_FAILED'}, out)
        save(out / 'run.json', {'RunID': out.name, 'ExperimentID': EXPERIMENT, 'status': 'CPU_VERIFICATION_FAILED', 'GPU_seconds': 0}, out, replace=True)
        seal(out)
        raise RuntimeError('Portable source/health/pixel verification failed.')
    save(out / 'verification.json', {'status': 'CPU_VERIFIED_GPU_NOT_VALIDATED', 'bundle': str(bundle),
        'bundle_manifest_SHA256': sha256(bundle / 'bundle_manifest.json'), 'commands': statuses,
        'unit_tests': int(re.search(r'Ran (\d+) tests', (out/'test_0.log').read_text()).group(1)),
        'relocated_dry_runs': ['pilot20', 'formal100'], 'GPU_seconds': 0,
        'wall_seconds': time.perf_counter() - started, 'formal_training': 'NOT_RUN', 'pilot_training': 'NOT_RUN'}, out)
    write_atomic(out / 'diary.md', '# Portable CPU verification\n\nRan isolated commands from the assembled bundle. Verified relocated source/data paths, default execution refusal, initialization health and accepted evaluation pixels. No YOLO forward/backward or GPU test.\n', out)
    save(out / 'run.json', {'RunID': out.name, 'ExperimentID': EXPERIMENT, 'mode': 'PORTABLE_VERIFICATION_CPU',
                          'status': 'CPU_VERIFIED_GPU_NOT_VALIDATED', 'GPU_seconds': 0}, out, replace=True)
    seal(out)
    print(out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--data-root', required=True, type=Path)
    args = parser.parse_args()
    execute(args.bundle.resolve(), args.data_root.resolve())
