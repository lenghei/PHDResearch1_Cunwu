"""CV261116: reproducible detector FM_D and coverage proxy audit."""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import html
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
PUBLIC = HERE.parents[1]
OUTER = PUBLIC.parent
sys.path.insert(0, str(HERE))
import model
from source_audit import audit_training


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def finalize_run(folder, run_id, status, mode):
    payload = {'RunID':run_id,'ExperimentID':'cv261116',
               'status':status, 'mode':mode}
    tmp = folder/'run.json.partial'
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    tmp.replace(folder/'run.json')


def run_dir(mode):
    root = (OUTER / 'datasets/local_runs/cv261116').resolve()
    if root.is_relative_to(PUBLIC.resolve()):
        raise RuntimeError('Local runs cannot be inside the GitHub root.')
    root.mkdir(parents=True, exist_ok=True)
    key = f'cv261116_{mode}_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}'
    folder = root / key
    folder.mkdir(exist_ok=False)
    return folder, key


def read_fmp_model(path):
    if not path.is_file():
        raise FileNotFoundError('Missing CV261109 FM_P implementation: ' + str(path))
    spec = importlib.util.spec_from_file_location('cv261109_fmp_source', path)
    if spec is None or spec.loader is None:
        raise RuntimeError('Cannot load verified FM_P source.')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, 'combinations') or not hasattr(module, 'valid_leaves'):
        raise ValueError('CV261109 source is incompatible.')
    return module


def flav_python(args):
    if args.flamapy_python:
        return Path(args.flamapy_python).resolve()
    folder = '.venv_flamapy/Scripts/python.exe' if os.name == 'nt' else '.venv_flamapy/bin/python'
    return (OUTER / folder).resolve()


def evaluate_model(args):
    independent = model.enumerate_count()
    if independent['independent_configurations'] != model.SUPERVISOR_COUNT or independent['independent_training_variants'] != model.SUPERVISOR_TRAIN_VARIANTS:
        raise ValueError('Independent FM_D count does not match supervisor draft.')
    text = args.uvl.read_text(encoding='utf-8')
    if text.count('\n    LEAF_YOLO =>') != 1 or text.count('\n    EndToEnd_NMSFree =>') != 1 or text.count('\n    COCO_ZeroShot =>') != 1 or text.count('\n    SAHI_Slicing =>') != 1:
        raise ValueError('Required four UVL constraints missing.')
    fpython = flav_python(args)
    if not fpython.is_file():
        raise FileNotFoundError('FlamaPy interpreter missing; use --flamapy-python: ' + str(fpython))
    try:
        process = subprocess.run([str(fpython), '-B', str(HERE/'flamapy_count.py'), '--model', str(args.uvl)],
                                 capture_output=True, text=True, encoding='utf-8', timeout=120,
                                 env={**os.environ, 'PYTHONDONTWRITEBYTECODE':'1'})
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('FM_D SAT admission exceeded 120 seconds. '
                           'The previous expensive #SAT operation has been removed; '
                           'inspect local FlamaPy installation before retrying.') from exc
    if process.returncode:
        raise RuntimeError('FlamaPy count failed: ' + (process.stderr or process.stdout)[-1800:])
    try:
        evidence = json.loads(process.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError) as exc:
        raise ValueError('FlamaPy process returned no JSON count.') from exc
    if (evidence['valid_count'] != model.SUPERVISOR_COUNT or
            evidence.get('training_variants') != model.SUPERVISOR_TRAIN_VARIANTS or
            evidence['constraint_count'] != 4 or evidence['leaf_count'] != 34 or
            evidence.get('dead_feature_count') != 0 or evidence.get('satisfiable') is not True or
            evidence.get('count_backend') != 'EXACT_FACTORIZED_COUNT_OF_STRICT_UVL_V1' or
            evidence.get('UVL_SHA256') != model.sha256(args.uvl)):
        raise ValueError('FM_D audited UVL / PySAT checks / factorized count mismatch.')
    return {'status':'PASS', 'FlamaPy':evidence, 'independent':independent,
            'UVL_SHA256': model.sha256(args.uvl)}


def write_coverage_tables(folder, gap, rules):
    with (folder/'coverage_by_feature.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('feature', 'present_total', 'covered_when_present'))
        writer.writeheader()
        for feature, d in sorted(gap['per_leaf'].items()):
            writer.writerow({'feature': feature, **d})
    with (folder/'coverage_rules.csv').open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=('feature_family','default_status','evidence_needed','rationale'))
        writer.writeheader()
        writer.writerows(rules['rules'])


def save_html(folder, result, gap, strict, rules):
    rows = '\n'.join('<tr><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
        html.escape(r['feature_family']), html.escape(r['default_status']),
        html.escape(r['rationale'])) for r in rules['rules'])
    body = f'''<!doctype html><meta charset="utf-8"><title>CV261116 | FM_D and Static Gap</title>
<style>body{{font:16px system-ui;max-width:1100px;margin:34px auto;padding:0 20px;color:#243342}}
header{{background:#102d43;color:white;padding:23px 28px;border-radius:12px}}h1{{margin:0}}section{{padding:12px 0}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}}
.card{{background:#edf4f8;border-radius:10px;padding:15px}}.value{{font-size:28px;font-weight:700}}
table{{width:100%;border-collapse:collapse}}td,th{{text-align:left;border-bottom:1px solid #d9e2e8;padding:10px}}
small{{color:#617888}}@media(max-width:700px){{.cards{{grid-template-columns:repeat(2,1fr)}}}}</style>
<header><h1>CV261116 · FM_D and static coverage</h1><p>Supervisor Week 3 · Read-only saved model analysis</p></header>
<section class="cards"><div class="card">FM_D configurations<div class="value">{result['FM_D']['FlamaPy']['valid_count']:,}</div></div>
<div class="card">Training variants<div class="value">{result['FM_D']['independent']['independent_training_variants']:,}</div></div>
<div class="card">FM_P proxy covered<div class="value">{gap['proxy_covered']:,}/{gap['FM_P_configurations']:,}</div></div>
<div class="card">Strict identity<div class="value">{strict['covered']:,}/{gap['FM_P_configurations']:,}</div></div></section>
<section><h2>Coverage assumptions</h2><p>These are documented proxies, not pixel-exact equality and not empirical detection gap.</p>
<table><tr><th>Family</th><th>Status</th><th>Reason</th></tr>{rows}</table></section>
<section><h2>Measurement gate</h2><p>GPU detector profiling: {html.escape(result['gpu_measurements'])}.</p>
<p>Training augmentation source and saved training arguments require review. Treat the 18 covered configurations as provisional.</p>
<small>RunID: {html.escape(result['RunID'])} | Model SHA256: {html.escape(result['FM_D']['UVL_SHA256'])}</small></section>'''
    (folder/'static_gap_report.html').write_text(body, encoding='utf-8')


def do_check(args):
    folder, run_id = run_dir('static')
    started = time.perf_counter()
    save(folder/'run.json', {'RunID':run_id,'ExperimentID':'cv261116',
                             'status':'RUNNING','mode':'FM_D_STATIC_GAP'})
    try:
        checked = evaluate_model(args)
        fmp = read_fmp_model(args.fmp_model)
        rules = model.load_rules(args.rules)
        gap = model.gap_analysis(fmp, rules)
        strict = model.strict_identity_gap(fmp)
        if gap['proxy_covered'] != 18 or strict['covered'] != 1:
            raise ValueError('Static-gap proxy does not match the documented supervisor assumption set.')
        source = audit_training(args.train_args)
        save(folder/'model_count.json', checked)
        save(folder/'coverage_rules.json', rules)
        save(folder/'static_gap.json', {'proxy':gap, 'strict_identity':strict})
        save(folder/'training_source_audit.json', source)
        write_coverage_tables(folder, gap, rules)
        finalize_run(folder, run_id, 'PARTIAL', 'FM_D_STATIC_GAP')
        result = {'RunID':run_id, 'ExperimentID':'cv261116', 'status':'PARTIAL',
            'mode':'FM_D_STATIC_GAP', 'scope':'SUPERVISOR_PROXY_GAP_NEEDS_TRAINING_SOURCE_REVIEW',
            'model_validation':'PASS', 'static_gap_proxy_reproduction':'PASS_PROVISIONAL_RULES',
            'FM_D':checked, 'FM_P_configurations':gap['FM_P_configurations'],
            'proxy_covered':gap['proxy_covered'], 'proxy_gap':gap['proxy_gap'],
            'strict_identity_covered':strict['covered'],
            'training_source_audit_status':source['status'],
            'gpu_measurements':'NOT_COLLECTED', 'AP_metrics':'NOT_EVALUATED',
            'FM_P_UVL_SHA256':model.sha256(args.fmp_uvl),
            'FM_P_code_SHA256':model.sha256(args.fmp_model),
            'coverage_rules_SHA256':model.sha256(args.rules),
            'elapsed_seconds':round(time.perf_counter()-started,3),
            'files_SHA256': {p.name:model.sha256(p) for p in sorted(folder.iterdir()) if p.is_file()}}
        save(folder/'result.json', result)
        save_html(folder, result, gap, strict, rules)
        print('[CV261116] FM_D PASS: FlamaPy UVL parsing + PySAT satisfiability/dead-feature checks.')
        print('[CV261116] Exact factorized count (strict UVL): 635904 configurations; 844 training variants; 34 leaves; 4 constraints.')
        print(f'[CV261116] Static proxy: 18 covered / 46368; strict identity: 1 / 46368; provisional assumptions.')
        print(f'[CV261116] Training augmentation coverage: {source["coverage_claim"]}; GPU: NOT_COLLECTED.')
        print('[CV261116] Run:', folder)
        print('[CV261116] Meeting report:', folder/'static_gap_report.html')
        return
    except BaseException as exc:
        finalize_run(folder, run_id, 'FAILED', 'FM_D_STATIC_GAP')
        save(folder/'failure.json',{'RunID':run_id,'status':'FAILED','error':repr(exc)})
        raise


def do_profile(args):
    if not args.checkpoint or not args.val_zip:
        raise ValueError('--profile requires --checkpoint and --val-zip.')
    folder, run_id = run_dir('gpu')
    started = time.perf_counter()
    save(folder/'run.json',{'RunID':run_id,'ExperimentID':'cv261116','status':'RUNNING','mode':'FM_D_GPU_PILOT'})
    try:
        declared = model.declared_profile_configuration(
            args.architecture, args.model_scale, args.weights_kind, args.train_resolution,
            args.imgsz, args.precision, args.strategy, args.postprocessing,
            tta=args.tta, preprocessing=args.preprocessing)
        from gpu_profile import run_profile
        measured = run_profile(args.checkpoint, args.val_zip, folder, args)
        measured['FM_D_declared_configuration'] = declared
        provenance = audit_training(args.train_args)
        save(folder/'source_audit.json', provenance)
        finalize_run(folder, run_id, 'PASS', 'FM_D_GPU_PILOT')
        result = {'RunID':run_id, 'ExperimentID':'cv261116', 'status':'PASS',
            'mode':'FM_D_GPU_PILOT', 'scope': measured['scope'],
            'model_validation':'NOT_EXECUTED_IN_THIS_RUN',
            'static_gap':'NOT_EVALUATED_IN_THIS_RUN', 'gpu_measurements':measured,
            'AP_metrics':'NOT_EVALUATED', 'training_source_audit_status':provenance['status'],
            'elapsed_seconds':round(time.perf_counter()-started,3),
            'measurements_csv_SHA256':model.sha256(folder/'measurements.csv')}
        save(folder/'result.json',result)
        print('[CV261116] GPU pilot PASS; original images:', measured['source_image_count'])
        print('[CV261116] Mean inference wall latency (ms/image):',round(measured['latency_ms_per_image_mean'],3))
        print('[CV261116] Peak PyTorch allocated bytes:',measured['peak_pytorch_allocated_bytes'])
        print('[CV261116] Run:',folder)
    except BaseException as exc:
        finalize_run(folder, run_id, 'FAILED', 'FM_D_GPU_PILOT')
        save(folder/'failure.json',{'RunID':run_id,'status':'FAILED','error':repr(exc)})
        raise


def do_publish(args):
    path = Path(args.publish_run).resolve(strict=True)
    expected = (OUTER / 'datasets/local_runs/cv261116').resolve()
    if not path.is_dir() or path.parent != expected or path.is_symlink():
        raise ValueError('Only an actual local CV261116 run may be published.')
    meta = json.loads((path/'result.json').read_text(encoding='utf-8'))
    if meta.get('RunID') != path.name or meta.get('ExperimentID') != 'cv261116':
        raise ValueError('Run identity mismatch.')
    run_state = json.loads((path/'run.json').read_text(encoding='utf-8'))
    if run_state.get('RunID') != meta['RunID'] or run_state.get('status') != meta['status']:
        raise ValueError('Local run status and identity differ.')
    for name, expected_sha in meta.get('files_SHA256', {}).items():
        if not name.endswith(('.json','.csv')) or '/' in name or '\\' in name:
            raise ValueError('Invalid evidence filename in run.')
        if model.sha256(path/name) != expected_sha:
            raise ValueError('Local run evidence hash mismatch: ' + name)
    if meta.get('mode') == 'FM_D_STATIC_GAP':
        if meta.get('model_validation') != 'PASS' or meta.get('proxy_covered') != 18:
            raise ValueError('Cannot publish an unverified static model run.')
        public = {'mode':'FM_D_STATIC_GAP','status':'PARTIAL',
                  'FM_D_valid_count':meta['FM_D']['FlamaPy']['valid_count'],
                  'FM_D_training_variants':meta['FM_D']['independent']['independent_training_variants'],
                  'FM_P_total':meta['FM_P_configurations'],'proxy_covered':meta['proxy_covered'],
                  'proxy_gap':meta['proxy_gap'],
                  'scope':'PROVISIONAL_SUPERVISOR_PROXY_NO_AP'}
    elif meta.get('mode') == 'FM_D_GPU_PILOT':
        digest = meta['measurements_csv_SHA256']
        if model.sha256(path/'measurements.csv') != digest:
            raise ValueError('GPU measurement table hash mismatch.')
        m = meta['gpu_measurements']
        public = {'mode':'FM_D_GPU_PILOT','status':'PASS',
                  'sample_images':m['source_image_count'],
                  'mean_ms_per_image':m['latency_ms_per_image_mean'],
                  'peak_pytorch_allocated_bytes':m['peak_pytorch_allocated_bytes'],
                  'parameters':m['checkpoint_parameters'], 'GFLOPs_estimate':m['GFLOPs_estimate'],
                  'scope':'INFERENCE_LATENCY_PILOT_NO_AP'}
    else:
        raise ValueError('Unknown experiment run mode.')
    import datetime
    snapshot = {'RunID':meta['RunID'],'ExperimentID':'cv261116', 'mode':meta['mode'],
                'status':meta['status'], 'files': {'run.json':{'RunID':meta['RunID'],
                  'ExperimentID':'cv261116','status':meta['status'],'mode':meta['mode']},
                  'result.json': public,
                  'summary.json': {'ExperimentID':'cv261116','RunID':meta['RunID'],
                    'status':meta['status'],'scope':public['scope']}}}
    location = PUBLIC/'reports/results/cv261116'/path.name
    if location.exists():
        raise FileExistsError('Public result already exists; snapshots are immutable.')
    location.mkdir(parents=True)
    save(location/'snapshot.json',snapshot)
    sha=model.sha256(location/'snapshot.json')
    save(location/'manifest.json',{'schema':2,'RunID':meta['RunID'],
       'ExperimentID':'cv261116','mode':meta['mode'],'status':meta['status'],
       'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
       'bundle':'snapshot.json','SHA256':sha,'files':{'snapshot.json':sha}})
    print('[CV261116] Published summary:',location)
    print('[CV261116] No private source images, logs or weights were exported.')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    action=p.add_mutually_exclusive_group()
    action.add_argument('--check',action='store_true',help='Validate FM_D and reproduce static-gap proxy on CPU.')
    action.add_argument('--profile',action='store_true',help='Measure real CUDA latency, memory and model size.')
    action.add_argument('--publish-run',type=Path,help='Publish a sanitized read-only snapshot of an audited local run.')
    p.add_argument('--uvl',type=Path,default=HERE/'fm_d.uvl')
    p.add_argument('--rules',type=Path,default=HERE/'coverage_rules.json')
    p.add_argument('--fmp-uvl',type=Path,default=PUBLIC/'experiments/cv261109/fm_p.uvl')
    p.add_argument('--fmp-model',type=Path,default=PUBLIC/'experiments/cv261109/model.py')
    p.add_argument('--flamapy-python',type=Path)
    p.add_argument('--train-args',type=Path,help='Saved Ultralytics training args.yaml to audit; optional.')
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--val-zip',type=Path)
    p.add_argument('--imgsz',type=int,choices=(640,960,1280,1536),default=1280)
    p.add_argument('--precision',choices=('FP32','FP16'),default='FP32')
    p.add_argument('--architecture',choices=model.ARCH,default='YOLOv8')
    p.add_argument('--model-scale',choices=model.SCALE,default='Small')
    p.add_argument('--weights-kind',choices=model.WEIGHTS,default='VisDrone_FT')
    p.add_argument('--train-resolution',choices=model.TRAIN_RES,default='Train640')
    p.add_argument('--postprocessing',choices=('NMS','EndToEnd_NMSFree'),default='NMS')
    p.add_argument('--strategy',choices=('FullFrame','SAHI_Slicing'),default='FullFrame')
    p.add_argument('--preprocessing',choices=('None','CLAHE_Pre','Gamma_Pre','Denoise_Pre','Deblur_Pre'),default='None')
    p.add_argument('--tta',action='store_true')
    p.add_argument('--device',default='0')
    p.add_argument('--warmup',type=int,default=4)
    p.add_argument('--repeat',type=int,default=2)
    p.add_argument('--sample-count',type=int,default=16)
    return p


if __name__=='__main__':
    args=parser().parse_args()
    if args.publish_run: do_publish(args)
    elif args.profile: do_profile(args)
    else: do_check(args)
