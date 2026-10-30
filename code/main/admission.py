"""CPU-only dataset and checkpoint admission."""
import json
from pathlib import Path

from .config import sha256
from .dataset import VisDroneZIP
from .labels import VIS_NAMES


def dataset_gate(cfg):
    archive_path = Path(cfg['dataset_zip'])
    actual_sha = sha256(archive_path)
    if actual_sha != cfg['dataset_zip_sha256']:
        raise ValueError('Relocated dataset ZIP SHA mismatch.')
    source = VisDroneZIP(archive_path, cfg['expected_images'])
    manifest = []
    try:
        for index in range(len(source.images)):
            pixels, record, proof = source.record(index)
            if pixels.dtype.name != 'uint8' or pixels.shape != (record['height'], record['width'], 3):
                raise ValueError('Image geometry mismatch.')
            manifest.append({**proof, 'file_name': record['file_name'],
                             'object_count_before_ignore': len(record['objects'])})
    finally:
        source.close()
    return {'status': 'PASS', 'source': str(archive_path), 'ZIP_SHA256': actual_sha,
            'images': len(manifest), 'decode': 'IMREAD_COLOR|IMREAD_IGNORE_ORIENTATION',
            'scope': 'DEVELOPMENT_VALIDATION', 'manifest': manifest}


def checkpoint_gate(cfg, return_detector=False):
    import torch
    import ultralytics
    from ultralytics import YOLO
    if ultralytics.__version__ != cfg['ultralytics_version']:
        raise ValueError('Ultralytics version differs from the frozen protocol.')
    if cfg['checkpoint_id'] != 'ck01' or sha256(cfg['checkpoint']) != cfg['checkpoint_sha256']:
        raise ValueError('YOLO26s ck01 identity mismatch.')
    detector = YOLO(cfg['checkpoint'])
    model = detector.model.cpu()
    names = list(model.names.values()) if isinstance(model.names, dict) else list(model.names)
    if names != VIS_NAMES:
        raise ValueError('Checkpoint must use native VisDrone ten-class order.')
    architecture = str(model.yaml.get('yaml_file', ''))
    if 'yolo26s' not in architecture.lower():
        raise ValueError('Checkpoint architecture is not YOLO26s.')
    for name, tensor in model.state_dict().items():
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError('Nonfinite checkpoint tensor: ' + name)
        if name.endswith('running_var') and (tensor < 0).any():
            raise ValueError('Negative BN variance: ' + name)
    head = model.model[-1]
    if head.nc != 10 or not all(hasattr(head, name) for name in ['cv2', 'cv3']):
        raise ValueError('One-to-many detection head is incompatible.')
    report = {'status': 'PASS', 'checkpoint_id': 'ck01', 'SHA256': cfg['checkpoint_sha256'],
            'architecture': architecture, 'class_order': names,
            'parameters': sum(p.numel() for p in model.parameters()),
            'head': type(head).__name__, 'one_to_many': 'end2end=False; existing predictor performs NMS once',
            'device': 'cpu', 'forward': 'NOT_RUN'}
    if cfg.get('training_args'):
        import yaml
        source = Path(cfg['training_args'])
        args = yaml.safe_load(source.read_text(encoding='utf-8'))
        report['training_provenance'] = {
            'source': str(source), 'SHA256': sha256(source),
            'configured': {key: args.get(key) for key in ['model', 'data', 'epochs', 'imgsz', 'batch',
                                                         'optimizer', 'seed', 'amp', 'cos_lr', 'warmup_epochs']},
            'selected_best_epoch': 'NOT_VERIFIED',
            'exposure': 'VisDrone train; val used for checkpoint selection; DEVELOPMENT_VALIDATION',
            'augmentation_coverage': 'NOT_INFERRED_FROM_TRAINING_ARGUMENTS'}
    return (report, detector) if return_detector else report


def verify_relocated_extracted(cfg, manifest):
    folder = Path(cfg['dataset_zip']).with_suffix('')
    images, labels = folder / 'images', folder / 'annotations'
    if not images.is_dir() or not labels.is_dir():
        return {'status': 'NOT_PRESENT', 'selected_source': 'original ZIP'}
    members = {Path(row['file_name']).stem for row in manifest}
    image_members = {p.stem for p in images.iterdir() if p.suffix.lower() in ['.jpg', '.jpeg']}
    label_members = {p.stem for p in labels.glob('*.txt')}
    if members != image_members or members != label_members:
        raise ValueError('Extracted val image/annotation membership mismatch.')
    return {'status': 'PASS_MEMBERSHIP', 'image_count': len(image_members),
            'annotation_count': len(label_members), 'selected_source': 'SHA-verified original ZIP',
            'note': 'Extracted JPEG bytes are not substituted for the frozen ZIP source.'}
