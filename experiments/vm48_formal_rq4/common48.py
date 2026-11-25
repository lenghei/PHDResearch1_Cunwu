"""VM48 immutable package, matrix and admission contracts."""
from datetime import datetime,timezone
import hashlib,json,math,os,platform,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
POST=HERE.parent/'experiments_test/vm37_yolo26_rq4/post_cost'
if not POST.is_dir():
    POST=HERE.parent/'vm37_yolo26_rq4/post_cost'
sys.path.insert(0,str(ROOT/'main/code'));sys.path.insert(0,str(POST))
import portable_v3 as v3
base=v3.base
VERSION='VM48_FORMAL_RQ4_V1'


def load(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return base.digest(path)
def save(path,value):return base.save(path,value)


def within(root,value):return base.within(root,value)


def verify_package(package):
    package=Path(package).resolve(strict=True)
    if package!=ROOT:raise PermissionError('Launch the entrypoint inside the assembled package; mixed workspace sources are not admitted.')
    manifest=load(package/'package_manifest.json')
    if manifest['version']!=VERSION:raise ValueError('Wrong package version.')
    for relative,digest in manifest['files'].items():
        if sha(within(package,relative))!=digest:raise ValueError('Frozen package changed: '+relative)
    base.verify_bundle(package)
    return manifest


def matrix(package,mode,architecture=None):
    rows=load(package/'vm48_protocol.json')['training_matrix']
    if mode=='formal':return [r for r in rows if r['role']=='PRIMARY']
    if mode=='ablations':return [r for r in rows if r['role']=='ABLATION']
    if mode=='benchmark':
        if architecture not in ('YOLO26s','YOLOv8s'):raise ValueError('Choose one benchmark architecture.')
        return [r for r in rows if r['role']=='PRIMARY' and r['seed']==42 and r['architecture']==architecture]
    raise ValueError('Unknown execution mode.')


def plan_arm(package,population,row,out,mode):
    protocol=load(package/'formal100_protocol.json')
    alias='CONTROL' if row['strategy']=='CONTROL' else 'ALL_FIVE'
    plan=base.plan_arm(package,population,protocol,row['architecture'],row['seed'],alias,out)
    completed=1 if mode=='benchmark' else 100
    from study import expected_steps
    recipe=load(package/row['recipe_asset']) if row['recipe_asset'] else None
    plan.update(arm=row['strategy'],arm_id=row['arm_id'],targeted_recipe=recipe,scope='DEVELOPMENT_ONLY',
        executor_version=VERSION,input_pipeline='optimized',v3_protocol='formal100',protocol_name='VM48',
        benchmark_only=mode=='benchmark',execution_epochs=completed,stop_after_completed_epochs=completed,
        expected=expected_steps(6471,100,stop_epochs=completed),input_implementation_SHA256=v3.implementation_files(package),
        augmentation_label='Static-gap family targeting, broad stochastic ranges; not learned Gap-guided augmentation',
        selection='FIXED_EPOCH100_EMA' if completed==100 else 'EPOCH1_ENGINEERING_ONLY')
    if sha(plan['initialization']['path'])!=row['initialization_SHA256']:raise ValueError('Wrong seed initialization.')
    if plan['args']['epochs']!=100 or plan['args']['imgsz']!=1280 or plan['args']['workers']!=0 or plan['args']['resume']:
        raise ValueError('Formal training contract changed.')
    return plan


def host_identity():
    import importlib.metadata
    return {'hostname':platform.node(),'system':platform.system(),'machine':platform.machine(),'Python':platform.python_version(),
        'packages':{name:importlib.metadata.version(name) for name in ('torch','ultralytics','numpy','opencv-python','albumentations','pycocotools')}}


def remaining(auth,charges=(),now=None):
    now=now or datetime.now(timezone.utc)
    total=auth['total_seconds']
    if isinstance(total,bool) or not isinstance(total,(int,float)) or not math.isfinite(total) or total<=0:raise ValueError('Invalid allocation.')
    used=sum(r['charged_seconds'] for r in charges)
    return max(0.,min(total-used,(datetime.fromisoformat(auth['expires_UTC'])-now).total_seconds()))


def validate_authorization(package,mode,rows,auth,receipt,*,evaluation=False):
    if auth.get('approved') is not True or auth.get('protocol_id')!=VERSION:raise PermissionError('New explicit VM48 approval is required.')
    identity=auth.get('authorization_id','')
    if not identity or Path(identity).name!=identity or any(v in identity.upper() for v in ('REPLACE','EXAMPLE','SYNTHETIC')):raise PermissionError('A new immutable authorization ID is required.')
    if auth.get('allocation_scope')!='NEW_ARM_SPECIFIC_ALLOCATION' or auth.get('historical_VM37_ledger_reuse') is not False:
        raise PermissionError('The historical36-hour laptop allocation is not valid here.')
    if auth.get('package_manifest_SHA256')!=sha(package/'package_manifest.json'):raise PermissionError('Wrong package authorization.')
    if auth.get('mode')!=mode or auth.get('arm_ids')!=[r['arm_id'] for r in rows]:raise PermissionError('Complete requested group must be explicitly authorized.')
    individual=auth.get('arm_authorizations',{})
    if any(individual.get(r['arm_id'],{}).get('approved') is not True for r in rows):raise PermissionError('Missing arm-specific approval.')
    if not all(auth.get(k) is True for k in ('cooling_confirmed','power_confirmed','machine_admitted')):raise PermissionError('Physical admission is absent.')
    if receipt.get('status')!='CPU_SOFTWARE_PASS_NOT_GPU_VALIDATED' or receipt.get('package_manifest_SHA256')!=sha(package/'package_manifest.json'):
        raise PermissionError('A complete target-host CPU validation receipt is required.')
    if receipt.get('host')!=host_identity():raise PermissionError('The target host or software differs from its CPU receipt.')
    if auth.get('CPU_receipt_SHA256')!=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':')).encode()).hexdigest():
        raise PermissionError('Authorization does not bind the canonical target receipt.')
    if remaining(auth)<60:raise PermissionError('Expired allocation.')
    caps=[individual[r['arm_id']].get('hard_seconds') for r in rows]
    if any(isinstance(c,bool) or not isinstance(c,(int,float)) or not math.isfinite(c) or c<60 for c in caps):raise PermissionError('Each arm needs a finite hard cap.')
    cost=auth.get('resource_admission',{})
    if cost.get('status')!='AUTHOR_ADMITTED_COMPLETE_GROUP':raise PermissionError('Complete cost admission is absent.')
    if mode!='benchmark':
        evidence=Path(cost['evidence_path']).resolve(strict=True)
        if sha(evidence)!=cost['evidence_SHA256']:raise PermissionError('Target cost evidence changed.')
        measured=load(evidence)
        if measured.get('host')!=receipt['host'] or measured.get('full_epoch_semantics') is not True:
            raise PermissionError('Target full-epoch cost evidence is missing.')
        if not {r['architecture'] for r in rows}.issubset(measured.get('measured_architectures',[])):raise PermissionError('Architecture cost evidence is incomplete.')
        if measured.get('CONTROL_and_ALL_FIVE_measured') is not True or cost.get('evaluation_and_bootstrap_included') is not True:
            raise PermissionError('Both strategies, evaluation and bootstrap must be costed.')
    reserve=cost.get('safety_reserve_seconds',0);evaluation_cap=cost.get('evaluation_hard_seconds',0)
    required=cost.get('required_complete_group_seconds',0)
    if any(isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or x<0 for x in (reserve,evaluation_cap,required)):
        raise PermissionError('Invalid cost bound.')
    if reserve<60 or required<sum(caps)+evaluation_cap+reserve or required>remaining(auth):raise PermissionError('Full group including reserve does not fit.')
    if evaluation and mode=='benchmark':raise PermissionError('Engineering benchmark cannot authorize scientific evaluation.')
    return remaining(auth)


def epoch100_contract(receipt,checkpoint,expected_initialization):
    expected={'completed_epochs':100,'anchor_exposures':647100,'forward_batches':323600,'backward_batches':323600,
              'optimizer_updates':10917,'EMA_updates':10917}
    if any(receipt.get(k)!=v for k,v in expected.items()):raise ValueError('Incomplete formal training counters.')
    if checkpoint.get('epoch')!=99 or checkpoint.get('ema') is None:raise ValueError('Expected fixed epoch100 EMA.')
    if checkpoint.get('rq4_provenance',{}).get('initialization_SHA256')!=expected_initialization:raise ValueError('Initialization provenance differs.')
    if checkpoint.get('rq4_provenance',{}).get('scope')!='DEVELOPMENT_ONLY':raise ValueError('Synthetic or engineering fixtures cannot be formal checkpoints.')
    if checkpoint.get('exact_resume_validated') is not False:raise ValueError('Unsupported resume claim.')
    if checkpoint.get('snapshot_scope','').startswith('SYNTHETIC'):raise ValueError('Synthetic fixtures cannot be scientific checkpoints.')
    from main.training_health import model_health
    import torch
    model_health(checkpoint['ema'])
    from main.labels import VIS_NAMES
    if list(checkpoint['ema'].names.values())!=VIS_NAMES:raise ValueError('Expected native10 EMA class order.')
    if any(t.is_floating_point() and t.dtype!=torch.float32 for t in checkpoint['ema'].state_dict().values()):raise ValueError('Expected FP32 EMA.')
    return checkpoint['ema']
