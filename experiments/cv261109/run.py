"""CV261109: render a FlamaPy-valid degradation over the complete VisDrone validation set.

Uses the existing 27-configuration baseline UVL. This is NOT the final FM_P.
No detector inference, metrics, or training is performed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import sys
import time
import uuid
from pathlib import Path

PUBLIC = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PUBLIC / 'code'))


def dump(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def pixel_sha(image) -> str:
    return hashlib.sha256(image.tobytes(order='C')).hexdigest()


def transform(image, stem: str, indices, degradation_seed: int):
    """Delegate to the validated VM1 degradation recipe; don't duplicate maths."""
    from main.degradation import degrade, field_for
    if len(indices) != 3 or any(not isinstance(v, int) or v not in (0, 1, 2) for v in indices):
        raise ValueError('Only the original I/B/N UVL baseline is supported here.')
    if list(indices) == [0, 0, 0]:
        raise ValueError('Expected a non-CLEAN selection.')
    return degrade(image, indices, field_for(image, stem, degradation_seed))


def render(archive, out: Path, selection: dict, count: int, preview_count: int, degradation_seed: int) -> dict:
    """Render a complete local set and keep source/provenance per image."""
    import cv2
    if not 1 <= count <= len(archive.images):
        raise ValueError('Invalid image count.')
    if not 0 <= preview_count <= count:
        raise ValueError('Invalid preview count.')
    indices = selection['indices']
    images_dir = out / 'images'
    images_dir.mkdir(parents=True, exist_ok=False)
    labels_dir = out / 'annotations'
    labels_dir.mkdir(parents=True, exist_ok=False)
    preview_dir = out / 'preview'
    preview_dir.mkdir(parents=True, exist_ok=False)
    records = []
    recipe = None
    cards = []
    for index in range(count):
        original, item, source_proof = archive.record(index)
        transformed, current_recipe = transform(original, item['stem'], indices, degradation_seed)
        if transformed.shape != original.shape or transformed.dtype != original.dtype:
            raise ValueError('Degradation changed image geometry or dtype.')
        if recipe is None:
            recipe = {k:v for k,v in current_recipe.items() if k in ('alpha','beta','kernel','noise_variance','noise_std')}
        filename = f'{index+1:04d}_{item["stem"]}.png'
        success, encoded = cv2.imencode('.png', transformed)
        if not success:
            raise RuntimeError('Could not encode PNG.')
        path = images_dir / filename
        path.write_bytes(encoded.tobytes())
        actual = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if actual is None or actual.shape != transformed.shape or pixel_sha(actual) != pixel_sha(transformed):
            raise ValueError('Saved PNG differs from generated pixels.')
        label_path = labels_dir / (item['stem'] + '.txt')
        label_bytes = archive.archive.read(archive.labels[item['stem']])
        if hashlib.sha256(label_bytes).hexdigest() != source_proof['annotation_SHA256']:
            raise ValueError('Source annotation SHA mismatch.')
        label_path.write_bytes(label_bytes)
        records.append({
            'annotation_file': 'annotations/' + label_path.name,
            'annotation_SHA256': hashlib.sha256(label_bytes).hexdigest(),
            'image_id': item['id'], 'original_file': item['file_name'],
            'source_image_SHA256': source_proof['image_SHA256'],
            'source_annotation_SHA256': source_proof['annotation_SHA256'],
            'source_pixel_SHA256': pixel_sha(original),
            'output_file': f'images/{filename}', 'output_SHA256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'output_pixel_SHA256': pixel_sha(transformed),
            'image_width': item['width'], 'image_height': item['height'],
            'gt_objects': len(item['objects']), 'ignore_regions': len(item['ignore_regions']),
            'transformation': current_recipe,
        })
        if index < preview_count:
            clean_name, deg_name = f'clean_{index+1:04d}.jpg', f'degraded_{index+1:04d}.jpg'
            for frame, name in ((original, clean_name), (transformed, deg_name)):
                ok, arr = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 94])
                if not ok:
                    raise RuntimeError('Could not encode preview.')
                (preview_dir / name).write_bytes(arr.tobytes())
            cards.append('<h3>Image {}</h3><div class="pair"><figure><img src="{}"><figcaption>CLEAN</figcaption></figure>'
                         '<figure><img src="{}"><figcaption>DEGRADED</figcaption></figure></div>'.format(
                             item['id'], html.escape(clean_name), html.escape(deg_name)))
        if (index+1) % 50 == 0 or (index+1) == count:
            print(f'Generated {index+1}/{count} | config={selection["configuration_id"]}', flush=True)
    with (out / 'image_manifest.jsonl').open('x', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, allow_nan=False) + '\n')
    (preview_dir / 'index.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>FM_P preview</title>'
        '<style>body{{font:16px Arial;margin:24px;background:#f5f6f9;color:#243448}}'
        '.pair{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}img{{width:100%}}figure{{margin:0}}'
        '@media(max-width:750px){{.pair{{grid-template-columns:1fr}}}}</style>'
        '<h1>FM_P — valid configuration preview</h1><p>{}</p>{}'
        '<p>Image source: <a href="https://github.com/VisDrone/VisDrone-Dataset">VisDrone2019-DET</a></p>'
        '</html>'.format(html.escape(selection['configuration_id']), ''.join(cards)),
        encoding='utf-8')
    return {'generated_images': count, 'images_for_preview': preview_count,
            'source_gt_objects': sum(v['gt_objects'] for v in records),
            'ignored_regions': sum(v['ignore_regions'] for v in records),
            'effective_parameter_levels': recipe,
            'manifest_SHA256': hashlib.sha256((out / 'image_manifest.jsonl').read_bytes()).hexdigest()}


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=PUBLIC / 'experiments/vm1_flamapy_yolo26s_validation/config.local.yaml')
    parser.add_argument('--seed', type=int, default=None, help='Select a valid UVL configuration using this seed.')
    parser.add_argument('--preview', type=int, default=3, help='Number of images shown locally (not the generation count).')
    parser.add_argument('--count', type=int, default=None, help='Defaults to ALL validation images (548).')
    args = parser.parse_args(argv)
    from main.pipeline import bridge, settings
    from main.dataset import VisDroneZIP
    from main.config import sha256
    cfg = settings(args.config)
    cfg['sampling_seed'] = cfg['sampling_seed'] if args.seed is None else args.seed
    count = cfg['expected_images'] if args.count is None else args.count
    outroot = PUBLIC.parent / 'datasets/local_runs/cv261109'
    if outroot.resolve().is_relative_to(PUBLIC.resolve()):
        parser.error('Full image sets cannot be stored inside the GitHub repository.')
    outroot.mkdir(parents=True, exist_ok=True)
    run_id = f'cv261109_generate_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}'
    out = outroot / run_id
    out.mkdir(exist_ok=False)
    start = time.perf_counter()
    try:
        if sha256(cfg['dataset_zip']) != cfg['dataset_zip_sha256']:
            raise ValueError('Dataset ZIP SHA does not match configured source.')
        value = bridge(cfg, out)
        if value['status'] != 'PASS' or value['dataset']['valid_count'] != 27:
            raise ValueError('Baseline FlamaPy configuration check failed.')
        chosen = next(c for c in value['conditions'] if c['name'] == 'DEGRADED')
        dump(out / 'selection.json', {
            'RunID': run_id, 'mode':'FM_P_GENERATION_ONLY', 'config_file_SHA256': sha256(args.config),
            'FlamaPy_valid_configurations': value['dataset']['valid_count'],
            'FlamaPy_constraints': value['dataset']['constraint_count'],
            'UVL_SHA256': value['UVL_SHA256'], 'seed': cfg['sampling_seed'],
            'degradation_seed': cfg['degradation_seed'], 'selected': chosen,
            'order': value['order'], 'dataset_ZIP_SHA256': cfg['dataset_zip_sha256'],
            'scope':'Original 27-config baseline; not the supervisor final FM_P',
            'generator_SHA256':sha256(Path(__file__)),
            'degradation_code_SHA256':sha256(PUBLIC / 'code/main/degradation.py'),
        })
        archive = VisDroneZIP(cfg['dataset_zip'], cfg['expected_images'])
        try:
            report = render(archive, out, chosen, count, args.preview, cfg['degradation_seed'])
        finally:
            archive.close()
        result = {'RunID':run_id, 'status':'PASS', 'mode':'FM_P_GENERATION_ONLY',
                  'model_inference':'NOT_RUN', 'evaluation_metrics':'NOT_EVALUATED',
                  'elapsed_seconds':round(time.perf_counter()-start, 3), **report}
        dump(out / 'result.json', result)
        print('PASS: '+str(out), flush=True)
        print('Preview: '+str(out / 'preview/index.html'), flush=True)
    except BaseException as exc:
        dump(out/'failure.json', {'RunID':run_id, 'status':'FAILED',
                                  'elapsed_seconds':round(time.perf_counter()-start, 3), 'error':str(exc)})
        raise


if __name__ == '__main__':
    cli()
