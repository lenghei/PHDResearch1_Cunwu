"""Reconcile checkpoint, split and historical evaluation evidence without inference."""
from collections import Counter
import csv
import inspect
from hashlib import sha256 as hash_bytes
from pathlib import Path, PurePosixPath
import json
from zipfile import ZipFile

from common import (ARCHIVES, BASELINE, CHECKPOINT, ROOT, TEST, load, new_run, save,
                    seal, sha256, source, verify_hashes)


ARCHIVE = ROOT / 'datasets/research_archive'
MIGRATION = ARCHIVE / 'migration_20261104'
VM = ARCHIVE / 'variability'
HISTORY = VM / 'experiments/vm11_holdout_confirmation/runs/vm11_r002/results'
PAPER1 = ARCHIVE / 'paper1/ieee_results_v3'
CHECKPOINTS = ROOT / 'models/checkpoints'
CANDIDATES = {
    'ck01': 'ck01_yolo26s_ft1280',
    'ck05': 'ck05_yolov8s_ft1280',
    'ck06': 'ck06_yolo11s_ft1280',
    'cv261102': CHECKPOINT.parent.name,
}
EVIDENCE_DOMAINS = ('training_membership', 'checkpoint_selection', 'historical_evaluation_and_tuning')
PATH_KEYS = {'model', 'project', 'name', 'save_dir'}


def table(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def archive_images(path, members=None):
    with ZipFile(path) as archive:
        selected = [name for name in archive.namelist()
                    if PurePosixPath(name).parent.name == 'images'
                    and PurePosixPath(name).suffix.lower() in {'.jpg', '.jpeg', '.png'}
                    and '__MACOSX' not in name and not PurePosixPath(name).name.startswith('._')]
        if members is not None:
            return {PurePosixPath(name).stem: members[name] for name in selected}
        result = {PurePosixPath(name).stem: hash_bytes(archive.read(name)).hexdigest() for name in selected}
        if len(result) != len(selected):
            raise ValueError('Duplicate image IDs in archive.')
        return result


def current_folder(path, expected):
    if not path.is_dir():
        return {'status': 'UNAVAILABLE'}
    actual = {file.stem: sha256(file) for file in path.iterdir()
              if file.is_file() and file.suffix.lower() in {'.jpg', '.jpeg', '.png'}}
    return {'status': 'CURRENT_FILES_ONLY', 'images': len(actual),
            'missing_ids': len(expected.keys()-actual.keys()),
            'extra_ids': len(actual.keys()-expected.keys()),
            'different_bytes': sorted(k for k in expected.keys() & actual.keys() if expected[k] != actual[k])}


def compare_splits(audit):
    gate = load(audit / 'dataset_gate.json')
    if gate['archive_status'] != 'PASS':
        raise ValueError('Original dataset audit did not pass.')
    local = {}
    for name, expected in gate['archives'].items():
        p = ARCHIVES / expected['name']
        if sha256(p) != expected['sha256']:
            raise ValueError(f'Archive changed: {name}')
        local[name] = p
    old_zip = load(HISTORY / 'local_source_audit.json')
    if old_zip['ZIP_SHA256'] != gate['archives']['test-dev']['sha256']:
        raise ValueError('Historical test-dev ZIP differs.')
    prior = {r['image_id']:r['SHA256'] for r in table(HISTORY / 'testdev_zip_members.csv') if r['kind']=='image'}
    parsed = {}
    members = {}
    for line in (audit / 'test_dev_members.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        parsed[row['image_stem']] = row['image_SHA256']
        members[row['ZIP_member']] = row['image_SHA256']
    if len(parsed) != 1610 or parsed != prior or archive_images(TEST, members) != parsed:
        raise ValueError('Test-dev image ledgers disagree.')
    recorded = {name:archive_images(local[name]) for name in ('train','val')}
    overlap = {k:{'images':len(v), 'matching_ids':len(v.keys() & parsed.keys()),
                  'matching_image_sha256':len(set(v.values()) & set(parsed.values()))}
               for k,v in recorded.items()}
    for name, row in overlap.items():
        if row['images'] != gate['archives'][name]['images'] or row['matching_ids'] or row['matching_image_sha256']:
            raise ValueError(f'Unexpected split membership or overlap: {name}')
    return gate, recorded, overlap


def models(audit, out):
    import yaml

    gate, split_images, overlap = compare_splits(audit)
    cfg = ARCHIVE / 'paper1/source_configs/visdrone_10class.yaml'
    split_config = yaml.safe_load(cfg.read_text(encoding='utf-8-sig'))
    data_doc = load(BASELINE / 'data_preparation.json')
    base_manifest = load(BASELINE / 'manifest.json')
    base_ckpt = load(BASELINE / 'training/checkpoint.json')
    history = {r['CheckpointID']:r for r in table(HISTORY / 'checkpoint_exposure_audit.csv')}
    catalog = {r['model_id']:r for r in table(MIGRATION / 'model_catalog.csv')}
    from evaluation import runtime
    runtime(out)

    import torch
    import ultralytics
    from main.labels import VIS_NAMES

    torch.set_num_threads(2)
    if list(split_config['names'].values()) != VIS_NAMES:
        raise ValueError('Historical data YAML class order differs.')
    clean = load(PAPER1 / 'dataset_manifest.json')['conditions']['CLEAN']['files']
    paper_members = {PurePosixPath(r['file_name']).stem:r['sha256'] for r in clean}
    if paper_members.keys() != split_images['val'].keys():
        raise ValueError('Historical paper CLEAN membership differs from val IDs.')
    paper_byte_diffs = sorted(k for k in paper_members if paper_members[k] != split_images['val'][k])
    eval_rows = table(PAPER1 / 'metrics.csv')
    vm09_path = VM / 'experiments/vm09_research_synthesis/runs/vm09_r001/results/full_configuration_metrics.csv'
    vm09 = Counter(r['CheckpointID'] for r in table(vm09_path))
    hist = load(HISTORY / 'history_search.json')
    candidates = []
    protocols = {}
    files = [cfg, BASELINE/'data_preparation.json', BASELINE/'manifest.json', BASELINE/'training/checkpoint.json',
             BASELINE/'prepared_dataset/visdrone10.yaml', BASELINE/'coco_regeneration.json',
             HISTORY/'checkpoint_exposure_audit.csv', HISTORY/'history_search.json', HISTORY/'history_matches.csv',
             HISTORY/'testdev_zip_members.csv', HISTORY/'dataset_gate.json', HISTORY/'local_source_audit.json',
             PAPER1/'dataset_manifest.json', PAPER1/'metrics.csv', PAPER1/'protocol.json', vm09_path,
             MIGRATION/'model_catalog.csv', MIGRATION/'result_integrity.json']
    for cid, model_id in CANDIDATES.items():
        local = CHECKPOINTS/model_id
        path = local/'best.pt'
        args = local/'args.yaml'
        curve = local/'results.csv'
        if cid == 'cv261102':
            args = BASELINE/'training/yolov8s/args.yaml'
            curve = BASELINE/'training/yolov8s/results.csv'
        files.extend([args, curve])
        entry = catalog[model_id]
        weight = sha256(path)
        expected = base_ckpt if cid == 'cv261102' else history[cid]
        expected_weight = expected['best_checkpoint_SHA256'] if cid == 'cv261102' else expected['SHA256']
        expected_args = expected['ultralytics_args_SHA256'] if cid == 'cv261102' else expected['training_args_SHA256']
        if weight != expected_weight or weight != entry['sha256'] or sha256(args) != expected_args:
            raise ValueError(f'{cid}: weight or args differ from the historical ledger.')
        if cid == 'cv261102' and sha256(curve) != base_ckpt['ultralytics_results_SHA256']:
            raise ValueError('CV261102 training curve differs from its sealed receipt.')
        saved = yaml.safe_load(args.read_text(encoding='utf-8-sig'))
        model = ultralytics.YOLO(str(path))
        checkpoint = model.ckpt
        embedded = checkpoint['train_args']
        compared = {k:v for k,v in saved.items() if k not in PATH_KEYS}
        differences = {k:{'saved_args':v, 'embedded':embedded.get(k)}
                       for k,v in compared.items() if embedded.get(k) != v}
        explained = {}
        if (cid == 'cv261102' and checkpoint.get('version') == ultralytics.__version__
                and saved.get('optimizer') == 'auto'
                and differences.get('warmup_bias_lr') == {'saved_args':0.1,'embedded':0.0}):
            from ultralytics.engine.trainer import BaseTrainer
            if 'self.args.warmup_bias_lr = 0.0' in inspect.getsource(BaseTrainer.build_optimizer):
                files.append(Path(inspect.getfile(BaseTrainer)))
                explained['warmup_bias_lr'] = 'optimizer=auto updates the runtime argument in build_optimizer'
        unexplained = sorted(differences.keys()-explained.keys())
        state_errors = []
        for kind, module in [('loaded',model.model),('saved_model',checkpoint.get('model')),('saved_ema',checkpoint.get('ema'))]:
            if isinstance(module,torch.nn.Module):
                for key,t in module.state_dict().items():
                    if t.is_floating_point() and (not torch.isfinite(t).all() or
                                                  key.endswith('running_var') and (t < 0).any()):
                        state_errors.append([kind,key])
        native = list(model.names.values()) == VIS_NAMES
        train_curve = table(curve)
        epochs = [int(next(v for k,v in r.items() if k.strip()=='epoch')) for r in train_curve]
        protocol_ok = (not unexplained and native and not state_errors and
                       saved.get('split')=='val' and saved.get('val') is True and epochs == list(range(1,101)))
        if cid == 'cv261102':
            training_status = 'BOUND_TO_RECORDED_ARCHIVES'
            for split in ('train','val'):
                if (data_doc[split]['source_zip_SHA256'] != gate['archives'][split]['sha256']
                        or base_manifest['system'][f'{split}_zip_SHA256'] != gate['archives'][split]['sha256']):
                    raise ValueError('CV261102 historical training archive hash mismatch.')
            selection_status = 'RECORDED_VAL_SELECTION_WITH_ARCHIVE_SHA'
        else:
            training_status = 'RECORDED_TRAIN_VAL_PATHS_CREATION_BYTES_UNVERIFIED'
            if PurePosixPath(saved['data'].replace('\\','/')).name != cfg.name:
                raise ValueError(f'{cid}: unexpected historical training data path.')
            selection_status = 'RECORDED_VAL_SELECTION_SELECTED_EPOCH_UNKNOWN'
            protocols[cid] = compared
        head = model.model.model[-1]
        candidates.append({'id':cid, 'model':PurePosixPath(str(saved['model']).replace('\\','/')).stem,
                           'checkpoint_sha256':weight, 'checkpoint':source(path),
                           'checkpoint_version':checkpoint.get('version'), 'checkpoint_epoch':checkpoint.get('epoch'),
                           'native10':native, 'parameters':sum(t.numel() for t in model.model.parameters()),
                           'head_end2end':bool(getattr(head,'end2end',False)),
                           'has_dense_branch':hasattr(head,'cv2') and hasattr(head,'cv3'),
                           'numerical_issues':state_errors, 'args_differences':differences,
                           'explained_args_differences':explained,
                           'epochs_logged':len(epochs), 'training_protocol':compared,
                           'training_membership':training_status, 'checkpoint_selection':selection_status,
                           'historical_testdev_evaluation_or_tuning':'UNVERIFIED',
                           'recorded_paper1_native10_cells':sum(r['checkpoint_sha256']==weight and
                                 r['protocol']=='native10' and r['size']=='all' for r in eval_rows),
                           'recorded_vm09_development_cells':vm09[cid],
                           'candidate_status':'CPU_READY_DEVELOPMENT_ONLY' if protocol_ok else 'CHECKPOINT_BLOCKED',
                           'formal_admission':'DATASET_BLOCKED'})
        del model, checkpoint

    keys = set().union(*(v.keys() for v in protocols.values()))
    differences = {key:{cid:p.get(key) for cid,p in protocols.items()}
                   for key in sorted(keys) if len({repr(p.get(key)) for p in protocols.values()})>1}
    folder_checks = {name:current_folder(Path(split_config[name]), split_images[name]) for name in ('train','val')}
    prepared = {name:current_folder(BASELINE/'prepared_dataset/images'/name, split_images[name])
                for name in ('train','val')}
    return {'status':'DATASET_BLOCKED','dataset':'VisDrone2019-DET-test-dev',
            'dataset_sha256':gate['archives']['test-dev']['sha256'],'source_audit_RunID':audit.name,
            'testdev_images':1610,'formal_pairs_measured':0,'archive_overlap':overlap,
            'legacy_current_training_folders':folder_checks,'cv261102_prepared_current':prepared,
            'paper1_val_byte_differences':paper_byte_diffs,
            'historical_search':{'files_checked':hist['files_checked'],
                                 'match_classes':Counter(r['classification'] for r in table(HISTORY/'history_matches.csv'))},
            'comparability':{'recorded_train_argument_differences':differences,
                             'checkpoint_versions':{r['id']:r['checkpoint_version'] for r in candidates},
                             'scope':'Historical checkpoint comparison on development data; no architecture-only causal attribution.'},
            'candidates':candidates,'evidence':[source(p) for p in files]}


def run(audit):
    from outputs import summarize
    verify_hashes(audit, ('dataset_gate.json','test_dev_members.jsonl','selection_manifest.json'))
    out = new_run('admission')
    print('Admission evidence:',out,flush=True)
    report = models(audit,out)
    save(out/'admission.json',report)
    public = public_summary(report)
    summary = summarize(audit,[])
    summary['admission'] = public
    summary['source_hashes']['admission.json'] = sha256(out/'admission.json')
    save(out/'summary.json',summary)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':'cv261207','mode':'admission',
                         'status':report['status'],'scope':'CPU_CHECKPOINT_AND_SPLIT_AUDIT'})
    seal(out)
    print(report['status'],flush=True)


def public_summary(report):
    return {**{k:report[k] for k in ('status','dataset','dataset_sha256','testdev_images','formal_pairs_measured','archive_overlap')},
            'candidates':[{k:r[k] for k in ('id','model','checkpoint_sha256','native10','epochs_logged',
                'training_membership','checkpoint_selection','historical_testdev_evaluation_or_tuning',
                'candidate_status','formal_admission')} for r in report['candidates']]}

def verify_review(decision, checkpoint_shas, root):
    """Require checkpoint-specific review of byte-addressed private evidence."""
    rows = decision.get('checkpoint_reviews')
    if not isinstance(rows, dict) or not checkpoint_shas <= rows.keys():
        raise ValueError('DATASET_BLOCKED: checkpoint-specific review is missing.')
    private = (root / 'datasets').resolve()
    for checkpoint in checkpoint_shas:
        entry = rows[checkpoint]
        if not isinstance(entry, dict):
            raise ValueError('DATASET_BLOCKED: invalid checkpoint review.')
        for domain in EVIDENCE_DOMAINS:
            review = entry.get(domain)
            if (not isinstance(review, dict) or review.get('status') != 'VERIFIED_INDEPENDENT'
                    or not isinstance(review.get('evidence'), list) or not review['evidence']):
                raise ValueError(f'DATASET_BLOCKED: {domain} is unresolved for {checkpoint}.')
            for evidence in review['evidence']:
                if not isinstance(evidence, dict) or not isinstance(evidence.get('path'), str):
                    raise ValueError(f'DATASET_BLOCKED: invalid evidence for {domain}.')
                path = (root / evidence['path']).resolve()
                if (not path.is_relative_to(private) or not path.is_file()
                        or not path.stat().st_size or sha256(path) != evidence.get('sha256')):
                    raise ValueError(f'DATASET_BLOCKED: {domain} evidence path or SHA is invalid.')
