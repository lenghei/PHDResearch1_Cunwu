"""Collect installed augmentation source and training arguments for human audit."""
from __future__ import annotations
import hashlib
import importlib.metadata
import inspect
from pathlib import Path


TRAIN_KEYS = ('model', 'data', 'epochs', 'imgsz', 'batch', 'seed', 'hsv_v', 'hsv_h',
              'hsv_s', 'scale', 'shear', 'perspective', 'mosaic', 'mixup', 'copy_paste',
              'augment', 'fliplr', 'flipud', 'auto_augment', 'erasing')


def audit_training(args_file=None):
    result = {'status': 'REQUIRES_HUMAN_REVIEW',
              'coverage_claim': 'NOT_VERIFIED', 'actual_training_args': 'NOT_PROVIDED',
              'installed_source': 'NOT_INSPECTED'}
    if args_file:
        import yaml
        file = Path(args_file).resolve(strict=True)
        data = yaml.safe_load(file.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            raise ValueError('Training args must be a mapping.')
        result['actual_training_args'] = {'SHA256': hashlib.sha256(file.read_bytes()).hexdigest(),
                                          'values': {key: data.get(key) for key in TRAIN_KEYS}}
    try:
        import ultralytics
        import ultralytics.data.augment as augment
        source_file = Path(inspect.getfile(augment)).resolve()
        src = source_file.read_bytes()
        snippets = {}
        for item in ('Albumentations', 'RandomHSV'):
            klass = getattr(augment, item, None)
            if klass is not None:
                try:
                    snippet = inspect.getsource(klass)
                except (OSError, TypeError):
                    snippet = ''
                snippets[item] = {'source_available': bool(snippet),
                                  'class_source_SHA256': hashlib.sha256(snippet.encode()).hexdigest() if snippet else None,
                                  'selected_keywords': [w for w in ('Blur', 'MedianBlur', 'ImageCompression',
                                    'RandomBrightnessContrast', 'RandomGamma', 'CLAHE', 'MotionBlur', 'GaussNoise',
                                    'hsv_v') if w in snippet]}
        result['installed_source'] = {
            'ultralytics_version': ultralytics.__version__,
            'augment_py_SHA256': hashlib.sha256(src).hexdigest(),
            'source_classes': snippets}
    except ImportError:
        result['installed_source'] = 'ULTRALYTICS_NOT_INSTALLED'
    return result
