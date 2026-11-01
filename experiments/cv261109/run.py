"""CV261109: FlamaPy FM_P verification, reproducible 548-image generation, and audit.

Run from the outer uavgap/ project root. No model inference or GPU is required.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import html
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import uuid
import zipfile
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent
PUBLIC = HERE.parents[1]
OUTER = PUBLIC.parent
sys.path.insert(0, str(HERE))
from model import ORDER, combinations, configuration_id, read_uvl_attributes, recipe, sha256
from generator import generate

MODEL = HERE / 'fm_p.uvl'
DEFAULT_CONFIG = HERE / 'config.local.yaml'
VM1_CONFIG = PUBLIC / 'experiments/vm1_flamapy_yolo26s_validation/config.local.yaml'
PUBLIC_RESULTS = PUBLIC / 'reports/results'
STUDY = 'cv261109'


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.write('\n')


def expand_path(value):
    value = os.path.expandvars(str(value))
    if not value or '${' in value or '%' in value or value.strip() == 'CHANGEME':
        raise ValueError('Unresolved path variable; edit config.local.yaml or set environment variable.')
    return Path(value).expanduser().resolve()


def input_config(args):
    import yaml
    cfgfile = args.config or (DEFAULT_CONFIG if DEFAULT_CONFIG.is_file() else VM1_CONFIG)
    if not cfgfile.is_file() and not args.dataset_zip:
        raise FileNotFoundError('Create config.local.yaml from config.example.yaml, or retain the existing VM1 config.local.yaml.')
    values = yaml.safe_load(cfgfile.read_text(encoding='utf-8')) if cfgfile.is_file() else {}
    if not isinstance(values, dict):
        raise ValueError('Invalid configuration dictionary.')
    dataset = expand_path(args.dataset_zip or values.get('dataset_zip', ''))
    if not dataset.is_file():
        raise FileNotFoundError('Original annotated validation ZIP: ' + str(dataset))
    digest = sha256(dataset)
    expected = values.get('dataset_zip_sha256')
    if expected and digest.lower() != str(expected).lower():
        raise ValueError('Validation ZIP checksum differs from the configured reference.')
    raw_interpreter = args.flamapy_python or values.get('flamapy_python') or (
        OUTER / '.venv_flamapy/Scripts/python.exe' if os.name == 'nt' else sys.executable)
    interpreter = expand_path(raw_interpreter)
    if not interpreter.is_file():
        raise FileNotFoundError('FlamaPy interpreter: ' + str(interpreter))
    expected_images = int(values.get('expected_images', 548))
    if expected_images != 548:
        raise ValueError('The supervisor full validation requires exactly 548 source images.')
    output = expand_path(args.output_root or (OUTER / 'datasets/local_runs/cv261109'))
    if output.is_relative_to(PUBLIC.resolve()):
        raise ValueError('Generated images must be kept outside the public Git repository.')
    return {'dataset_zip': dataset, 'dataset_zip_sha256': digest,
            'flamapy_python': interpreter, 'expected_images': 548,
            'output_root': output, 'config_file': cfgfile if cfgfile.is_file() else None,
            'degradation_seed': int(args.degradation_seed if args.degradation_seed is not None
                                    else values.get('degradation_seed', 20261028))}


def flamapy_select(cfg, seed, requested=None):
    command = [str(cfg['flamapy_python']), '-B', str(HERE / 'flamapy_select.py'),
               '--model', str(MODEL), '--seed', str(seed)]
    if requested:
        command.extend(['--configuration-id', requested])
    process = subprocess.run(command, text=True, capture_output=True,
                             encoding='utf-8', errors='replace', timeout=360)
    if process.returncode:
        raise RuntimeError('FlamaPy selection failed: ' + (process.stderr or process.stdout)[-3000:])
    selection = json.loads(process.stdout.strip())
    if selection['valid_count'] != 46368 or selection['constraint_count'] != 5:
        raise ValueError('FlamaPy model count/constraints differ from the supervisor FM_P.')
    if selection['UVL_SHA256'] != sha256(MODEL):
        raise ValueError('FM_P UVL source changed during selection.')
    if selection['configuration_id'] != configuration_id(selection['selected_leaves']):
        raise ValueError('Selected feature identity mismatch.')
    return selection


class ValidationArchive:
    def __init__(self, path, expected=548):
        import cv2
        self.archive = zipfile.ZipFile(path)
        files = self.archive.namelist()
        if len(files) != len(set(files)):
            raise ValueError('Duplicate source ZIP entries.')
        images = sorted(n for n in files if '/images/' in n.lower() and
                        n.lower().endswith(('.jpg', '.jpeg', '.png')) and
                        not PurePosixPath(n).name.startswith('._') and '__MACOSX' not in n)
        annotations = {PurePosixPath(n).stem: n for n in files if '/annotations/' in n.lower()
                       and n.lower().endswith('.txt') and not PurePosixPath(n).name.startswith('._')
                       and '__MACOSX' not in n}
        if len(images) != expected or {PurePosixPath(n).stem for n in images} != set(annotations):
            raise ValueError('VisDrone validation split must contain 548 matched image/annotation pairs.')
        if len(annotations) != expected:
            raise ValueError('Duplicate VisDrone annotation stems.')
        self.images, self.labels = images, annotations

    def record(self, i):
        import cv2
        import numpy as np
        member = self.images[i]
        stem = PurePosixPath(member).stem
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', stem):
            raise ValueError('Unsafe source image name: ' + repr(stem))
        source = self.archive.read(member)
        label = self.archive.read(self.labels[stem])
        image = cv2.imdecode(np.frombuffer(source, dtype=np.uint8),
                             cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError('Cannot decode source image ' + member)
        line_count = ignored = zero_area = 0
        for lineno, line in enumerate(label.decode('utf-8-sig').splitlines(), 1):
            if not line.strip():
                continue
            fields = [part.strip() for part in line.split(',')]
            while fields and not fields[-1]:
                fields.pop()
            if len(fields) != 8:
                raise ValueError(f'{self.labels[stem]}:{lineno}: expected eight annotation fields.')
            try:
                values = [float(n) for n in fields]
            except ValueError as exc:
                raise ValueError(f'{self.labels[stem]}:{lineno}: nonnumeric annotation.') from exc
            if not all(map(math.isfinite, values)):
                raise ValueError(f'{self.labels[stem]}:{lineno}: nonfinite annotation.')
            if values[2] < 0 or values[3] < 0 or values[5] != int(values[5]) or not 0 <= values[5] <= 11:
                raise ValueError(f'{self.labels[stem]}:{lineno}: invalid category or size.')
            if values[2] == 0 or values[3] == 0:
                zero_area += 1
            if values[5] == 0 or values[4] == 0:
                ignored += 1
            line_count += 1
        proof = {'source_member': member, 'source_image_SHA256': hashlib.sha256(source).hexdigest(),
                 'source_annotation_member': self.labels[stem],
                 'source_annotation_SHA256': hashlib.sha256(label).hexdigest(),
                 'source_pixel_SHA256': hashlib.sha256(image.tobytes()).hexdigest(),
                 'annotation_rows': line_count, 'ignored_annotation_rows': ignored,
                 'zero_area_annotation_rows': zero_area}
        return image, label, stem, proof

    def close(self):
        self.archive.close()


def preflight(cfg, seed, requested):
    attr = read_uvl_attributes(MODEL)
    count = sum(1 for _ in combinations())
    if count != 46368:
        raise ValueError('Independent FM_P count differs from the supervisor model.')
    selected = flamapy_select(cfg, seed, requested)
    archive = ValidationArchive(cfg['dataset_zip'], cfg['expected_images'])
    try:
        count = len(archive.images)
    finally:
        archive.close()
    return selected, attr, count


def preview_page(path, run_id, feature_list, cards):
    content = ['<!doctype html><html lang="en"><meta charset="utf-8">',
               '<title>CV261109 - FM_P generation preview</title>',
               '<style>body{font:16px Arial;margin:24px auto;max-width:1300px;color:#26364c;background:#f7f9fa}',
               '.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px}',
               'img{width:100%;height:auto}figure{margin:0 0 22px}figcaption{margin:8px 0}',
               '@media(max-width:700px){.pair{grid-template-columns:1fr}}</style>',
               '<h1>CV261109: deterministic FM_P configuration</h1>',
               '<p>RunID: ' + html.escape(run_id) + '</p>',
               '<p>Features: ' + html.escape(', '.join(feature_list)) + '</p>']
    for index, clean, degraded in cards:
        content.extend([f'<h3>Source image {index}</h3>', '<div class="pair">',
                        f'<figure><img src="{html.escape(clean)}"><figcaption>Original</figcaption></figure>',
                        f'<figure><img src="{html.escape(degraded)}"><figcaption>Generated</figcaption></figure>',
                        '</div>'])
    content.extend(['<p>Images are for local demonstration only. Source: VisDrone2019-DET.</p>', '</html>'])
    path.write_text('\n'.join(content) + '\n', encoding='utf-8')


def create_run(cfg, selection, attributes, count, preview, degradation_seed):
    import cv2
    import numpy as np
    if not 1 <= count <= 548 or not 0 <= preview <= count:
        raise ValueError('Image count or preview count is invalid.')
    cfg_id = selection['configuration_id']
    run_id = f'cv261109_fmp_{cfg_id}_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}'
    root = cfg['output_root'] / run_id
    root.mkdir(parents=True, exist_ok=False)
    for child in ('images', 'annotations', 'preview'):
        (root / child).mkdir(exist_ok=False)
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    start = time.perf_counter()
    try:
        setup = {'RunID': run_id, 'ExperimentID': STUDY, 'status': 'RUNNING',
                 'started_at_UTC': now, 'selection': selection,
                 'parameters_from_UVL': recipe(selection['selected_leaves'], attributes),
                 'effect_order': list(ORDER), 'degradation_seed': degradation_seed,
                 'dataset_ZIP_SHA256': cfg['dataset_zip_sha256'],
                 'generator_SHA256': sha256(HERE / 'generator.py'),
                 'model_code_SHA256': sha256(HERE / 'model.py'),
                 'runner_SHA256': sha256(HERE / 'run.py'),
                 'selection_code_SHA256': sha256(HERE / 'flamapy_select.py'),
                 'source_image_count': 548, 'requested_image_count': count,
                 'environment': {'python':sys.version, 'platform':platform.platform(),
                                 'opencv':cv2.__version__, 'numpy':np.__version__},
                 'image_encoding': 'lossless PNG',
                 'image_geometry': 'original dimensions; altitude effect uses downscale then resize back',
                 'annotation_coordinates': 'unchanged original-image coordinates',
                 'scope': 'FULL_548_FM_P_GENERATION' if count == 548 else 'DEMONSTRATION_SUBSET'}
        write_json(root / 'selection.json', setup)
        (root / 'command.txt').write_text(' '.join(sys.argv) + '\n', encoding='utf-8')
        records, cards = [], []
        archive = ValidationArchive(cfg['dataset_zip'])
        try:
            with (root / 'image_manifest.jsonl').open('x', encoding='utf-8') as manifest:
                for i in range(count):
                    image, label, stem, proof = archive.record(i)
                    transformed, trace = generate(image, selection['selected_leaves'], attributes,
                                                   stem, degradation_seed)
                    suffix = f'{i + 1:04d}_{stem}'
                    target = root / 'images' / (suffix + '.png')
                    success, buffer = cv2.imencode('.png', transformed,
                                                   [cv2.IMWRITE_PNG_COMPRESSION, 3])
                    if not success:
                        raise RuntimeError('Could not encode generated image.')
                    target.write_bytes(buffer.tobytes())
                    check = cv2.imread(str(target), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                    transformed_sha = hashlib.sha256(transformed.tobytes()).hexdigest()
                    if check is None or hashlib.sha256(check.tobytes()).hexdigest() != transformed_sha:
                        raise ValueError('Generated PNG pixels differ from memory before saving.')
                    label_file = root / 'annotations' / (stem + '.txt')
                    label_file.write_bytes(label)
                    if sha256(label_file) != proof['source_annotation_SHA256']:
                        raise ValueError('Saved original annotations differ from the source ZIP.')
                    record = {'image_index': i + 1, 'image_stem': stem, **proof,
                              'output_file': target.relative_to(root).as_posix(),
                              'output_file_SHA256': sha256(target),
                              'output_pixel_SHA256': transformed_sha,
                              'annotation_file': label_file.relative_to(root).as_posix(),
                              'image_height': image.shape[0], 'image_width': image.shape[1],
                              'transformation_trace': trace}
                    records.append(record)
                    manifest.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + '\n')
                    manifest.flush()
                    if i < preview:
                        names = []
                        for kind, pixels in (('original', image), ('degraded', transformed)):
                            dst = root / 'preview' / f'{kind}_{i + 1:04d}.jpg'
                            success, encoded = cv2.imencode('.jpg', pixels,
                                                            [cv2.IMWRITE_JPEG_QUALITY, 92])
                            if not success:
                                raise RuntimeError('Could not encode demonstration JPEG.')
                            dst.write_bytes(encoded.tobytes())
                            names.append(dst.name)
                        cards.append((i + 1, *names))
                    if (i + 1) % 50 == 0 or i + 1 == count:
                        print(f'[CV261109] Generated {i + 1}/{count} images', flush=True)
        finally:
            archive.close()
        preview_page(root / 'preview/index.html', run_id, selection['selected_leaves'], cards)
        result = {'RunID': run_id, 'ExperimentID': STUDY,
                  'status': 'PASS' if count == 548 else 'DEMO',
                  'scope': setup['scope'], 'generated_images': count, 'source_images': 548,
                  'configuration_id': cfg_id, 'FlamaPy_valid_configurations': selection['valid_count'],
                  'annotation_rows': sum(r['annotation_rows'] for r in records),
                  'ignored_annotation_rows': sum(r['ignored_annotation_rows'] for r in records),
                  'zero_area_annotation_rows': sum(r['zero_area_annotation_rows'] for r in records),
                  'image_manifest_SHA256': sha256(root / 'image_manifest.jsonl'),
                  'source_annotations_unchanged': True, 'image_geometry_unchanged': True,
                  'elapsed_seconds': round(time.perf_counter() - start, 3),
                  'finished_at_UTC': dt.datetime.now(dt.timezone.utc).isoformat(),
                  'model_inference': 'NOT_RUN', 'AP_metrics': 'NOT_EVALUATED'}
        write_json(root / 'result.json', result)
        print('[CV261109] ' + result['status'] + ': ' + str(root), flush=True)
        print('[CV261109] Preview: ' + str(root / 'preview/index.html'), flush=True)
        return root
    except BaseException as exc:
        write_json(root / 'failure.json', {'RunID': run_id, 'status': 'FAILED',
                                          'reason': repr(exc),
                                          'traceback': traceback.format_exc(),
                                          'elapsed_seconds': round(time.perf_counter() - start, 3)})
        raise


def verify_run(root, cfg, replay_count=3):
    import cv2
    if replay_count < 0:
        raise ValueError('Replay count must be nonnegative.')
    root = root.resolve()
    setup = json.loads((root / 'selection.json').read_text(encoding='utf-8'))
    result = json.loads((root / 'result.json').read_text(encoding='utf-8'))
    if setup['RunID'] != root.name or result['RunID'] != root.name:
        raise ValueError('Run identity mismatch.')
    if setup['selection']['UVL_SHA256'] != sha256(MODEL):
        raise ValueError('FM_P UVL changed since the recorded run.')
    for key, name in (('generator_SHA256','generator.py'),
                      ('model_code_SHA256','model.py'), ('runner_SHA256','run.py'),
                      ('selection_code_SHA256','flamapy_select.py')):
        if setup[key] != sha256(HERE / name):
            raise ValueError('Experiment code changed since the recorded run: ' + name)
    if setup['dataset_ZIP_SHA256'] != cfg['dataset_zip_sha256']:
        raise ValueError('Source dataset SHA does not match the recorded run.')
    if setup['selection']['configuration_id'] != configuration_id(setup['selection']['selected_leaves']):
        raise ValueError('Selected configuration ID mismatch.')
    attributes = read_uvl_attributes(MODEL)
    if setup['parameters_from_UVL'] != recipe(setup['selection']['selected_leaves'], attributes):
        raise ValueError('Numeric UVL attributes changed since the run.')
    path = root / 'image_manifest.jsonl'
    if sha256(path) != result['image_manifest_SHA256']:
        raise ValueError('Saved image manifest checksum mismatch.')
    records = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
    if len(records) != result['generated_images']:
        raise ValueError('Saved image count differs from the manifest.')
    replay = len(records) if replay_count == 0 else min(replay_count, len(records))
    source = ValidationArchive(cfg['dataset_zip'])
    try:
        for i, record in enumerate(records):
            if record['image_index'] != i + 1:
                raise ValueError('Source image ordering mismatch.')
            image, label, stem, proof = source.record(i)
            if stem != record['image_stem']:
                raise ValueError('Source image stem mismatch.')
            for key in ('source_image_SHA256', 'source_annotation_SHA256', 'source_pixel_SHA256'):
                if proof[key] != record[key]:
                    raise ValueError('Source evidence differs from saved record: ' + key)
            target = (root / record['output_file']).resolve()
            original_label = (root / record['annotation_file']).resolve()
            if not target.is_relative_to(root) or not original_label.is_relative_to(root):
                raise ValueError('Saved output path escapes the Run directory.')
            if sha256(target) != record['output_file_SHA256'] or sha256(original_label) != record['source_annotation_SHA256']:
                raise ValueError('Generated image or label checksum mismatch.')
            pixels = cv2.imread(str(target), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if pixels is None or hashlib.sha256(pixels.tobytes()).hexdigest() != record['output_pixel_SHA256']:
                raise ValueError('Saved generated pixels do not match the evidence.')
            if i < replay:
                again, trace = generate(image, setup['selection']['selected_leaves'], attributes,
                                        stem, setup['degradation_seed'])
                if hashlib.sha256(again.tobytes()).hexdigest() != record['output_pixel_SHA256']:
                    raise ValueError('Independent deterministic replay differs on image ' + str(i + 1))
                if trace != record['transformation_trace']:
                    raise ValueError('Recorded random effects differ during replay.')
    finally:
        source.close()
    report = {'RunID': root.name, 'status': 'PASS', 'files_checked': len(records),
              'images_regenerated': replay, 'source_ZIP_SHA256': cfg['dataset_zip_sha256'],
              'image_manifest_SHA256': sha256(path)}
    print('[CV261109] Audit PASS: ' + json.dumps(report, sort_keys=True), flush=True)
    return report


def publish_run(root, cfg):
    import hashlib
    audit = verify_run(root, cfg, replay_count=3)
    selection = json.loads((root / 'selection.json').read_text(encoding='utf-8'))
    result = json.loads((root / 'result.json').read_text(encoding='utf-8'))
    if result['status'] != 'PASS' or result['generated_images'] != 548:
        raise ValueError('Only complete 548-image validated runs may be published.')
    provenance = {'source_RunID': root.name, 'UVL_SHA256': selection['selection']['UVL_SHA256'],
                  'image_manifest_SHA256': result['image_manifest_SHA256'],
                  'source_dataset_ZIP_SHA256': selection['dataset_ZIP_SHA256'],
                  'generator_SHA256': selection['generator_SHA256'], 'audit': audit}
    public_result = {key: result[key] for key in ('status','scope','generated_images',
                    'configuration_id','FlamaPy_valid_configurations','annotation_rows',
                    'ignored_annotation_rows','zero_area_annotation_rows','elapsed_seconds')}
    summary = {'RunID': root.name, 'ExperimentID': STUDY, 'status':'PASS',
               'mode':'FM_P_GENERATION', 'dataset': {'images':548, 'source':'VisDrone2019-DET-val'},
               'configuration': {'id':selection['selection']['configuration_id'],
                                 'features':selection['selection']['selected_leaves'],
                                 'numeric_parameters':selection['parameters_from_UVL'],
                                 'effect_order':selection['effect_order']},
               'timing': {'wall_seconds': result['elapsed_seconds']},
               'metrics': {'AP': 'NOT_EVALUATED', 'model_inference':'NOT_RUN'}}
    started = selection['started_at_UTC']
    snapshot = {'RunID':root.name, 'ExperimentID':STUDY,'mode':'FM_P_GENERATION',
                'status':'PASS', 'files':{'run.json':{'RunID':root.name, 'ExperimentID':STUDY,
                'status':'PASS','mode':'FM_P_GENERATION'},'result.json':public_result,
                'summary.json':summary,'provenance.json':provenance}}
    raw = (json.dumps(snapshot, indent=2, allow_nan=False, ensure_ascii=True) + '\n').encode()
    if b':\\' in raw or len(raw) > 2_000_000:
        raise ValueError('Private path or unexpectedly large public snapshot.')
    public = PUBLIC_RESULTS / STUDY / root.name
    if public.exists():
        raise FileExistsError('Published RunID already exists: ' + str(public))
    public.mkdir(parents=True)
    (public/'snapshot.json').write_bytes(raw)
    manifest = {'schema': 2, 'RunID':root.name, 'ExperimentID':STUDY,
                'mode':'FM_P_GENERATION', 'status':'PASS', 'started_at':started,
                'scope': result['scope'], 'bundle':'snapshot.json',
                'SHA256': hashlib.sha256(raw).hexdigest(),
                'files':{'snapshot.json':hashlib.sha256(raw).hexdigest()}}
    write_json(public/'manifest.json',manifest)
    print('[CV261109] Publish-ready metadata: ' + str(public),flush=True)
    return public


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='New CV configuration; defaults to existing VM1 local config.')
    parser.add_argument('--dataset-zip', type=Path, help='Original VisDrone2019-DET-val ZIP.')
    parser.add_argument('--flamapy-python', type=Path, help='Interpreter for the dedicated FlamaPy environment.')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--seed', type=int, default=1031, help='Supervisor-provided selection seed.')
    parser.add_argument('--degradation-seed', type=int)
    parser.add_argument('--configuration-id', help='Re-select an exact previously verified FM_P configuration.')
    parser.add_argument('--count', type=int, default=548, help='Generate 548 by default; 3 for a live quick demonstration.')
    parser.add_argument('--preview', type=int, default=3)
    parser.add_argument('--check', action='store_true', help='Check model and dataset without creating a Run.')
    parser.add_argument('--verify-run', type=Path, help='Audit saved image and annotation evidence.')
    parser.add_argument('--replay-count', type=int, default=3, help='Re-render first N images; 0 re-renders all.')
    parser.add_argument('--publish-run', type=Path, help='Write a small GitHub-ready verified result snapshot.')
    args = parser.parse_args(argv)
    if args.verify_run and args.publish_run:
        parser.error('Select only one audit/publish action.')
    cfg = input_config(args)
    if args.verify_run:
        verify_run(args.verify_run, cfg, args.replay_count)
        return
    if args.publish_run:
        publish_run(args.publish_run.resolve(), cfg)
        return
    tick = time.perf_counter()
    selection, attributes, count = preflight(cfg, args.seed, args.configuration_id)
    print(f'[CV261109] Preflight PASS: {count} VisDrone images; '
          f'{selection["valid_count"]} FlamaPy-valid configurations; '
          f'five constraints; selected {selection["configuration_id"]}; '
          f'{time.perf_counter()-tick:.2f} s.', flush=True)
    print('[CV261109] Selected features: ' + ', '.join(selection['selected_leaves']), flush=True)
    if args.check:
        print('[CV261109] Preflight only; no images or Run directory created.',flush=True)
        return
    create_run(cfg, selection, attributes, args.count, min(args.preview, args.count),
               cfg['degradation_seed'])


if __name__ == '__main__':
    main()
