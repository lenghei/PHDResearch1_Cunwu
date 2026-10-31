"""Render audited, paired VM1 images from existing predictions. No inference."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / 'code'))
STUDY = 'vm1_flamapy_yolo26s_validation'
IMAGE_NAMES = ('original.jpg', 'degraded.jpg', 'clean_predictions.jpg', 'degraded_predictions.jpg')
LABEL_COLORS = ((22, 160, 133), (45, 127, 211), (222, 133, 50), (178, 70, 92),
                (114, 75, 171), (46, 172, 180), (191, 99, 155), (195, 134, 44),
                (73, 128, 65), (77, 101, 192))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def safe_child(parent: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute() or '..' in Path(relative).parts or '\\' in relative:
        raise ValueError('Unsafe artifact path: ' + relative)
    child = (parent / relative).resolve()
    if not child.is_relative_to(parent.resolve()) or child.is_symlink():
        raise ValueError('Artifact escapes run: ' + relative)
    return child


def verify_condition(path):
    summary = read(path / 'summary.json')
    if summary.get('Status') != 'PASS' or summary.get('images') != 548:
        raise ValueError('Requires a successful 548-image condition: ' + str(path))
    for filename, key in [('ground_truth.json', 'GT_SHA256'),
                          ('predictions_coco.json', 'predictions_coco_SHA256'),
                          ('metrics.json', 'metrics_SHA256'),
                          (summary.get('raw_prediction_file', 'predictions_raw.jsonl'), 'predictions_SHA256')]:
        if sha256(path / filename) != summary[key]:
            raise ValueError('Original evidence hash mismatch: ' + str(path / filename))
    return summary


def draw_predictions(image, predictions, names, confidence):
    from PIL import ImageDraw, ImageFont
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for box in predictions:
        if float(box['score']) < confidence:
            continue
        cid = int(box['category_id'])
        if cid not in names:
            raise ValueError('Unknown category id')
        x, y, w, h = map(float, box['bbox'])
        if w <= 0 or h <= 0:
            continue
        color = LABEL_COLORS[(cid - 1) % len(LABEL_COLORS)]
        draw.rectangle((x, y, x + w, y + h), outline=color, width=2)
        label = f"{names[cid]} {float(box['score']):.2f}"
        tx, ty = max(0, int(x)), max(0, int(y) - 14)
        bbox = draw.textbbox((tx, ty), label, font=font)
        draw.rectangle(bbox, fill=(15, 25, 30))
        draw.text((tx, ty), label, font=font, fill=color)
    return canvas


def output_jpeg(image, dest, max_width):
    image = image.copy()
    if image.width > max_width:
        new_height = max(1, round(image.height * max_width / image.width))
        from PIL import Image
        image = image.resize((max_width, new_height), Image.Resampling.LANCZOS)
    image.save(dest, 'JPEG', quality=88, optimize=True, subsampling=0)


def publish_verified_preview(local: Path, run: Path):
    """Publish an existing derived preview without altering original run evidence."""
    index_file = local / 'manifest.json'
    if not index_file.is_file():
        raise ValueError('Missing locally generated comparison manifest.')
    index = read(index_file)
    if index.get('schema') != 1 or index.get('RunID') != run.name:
        raise ValueError('Comparison identity mismatch.')
    for row in index.get('images', []):
        image_id = row.get('image_id')
        if type(image_id) is not int or not 1 <= image_id <= 548:
            raise ValueError('Invalid comparison image ID.')
        for name in IMAGE_NAMES:
            path = local / f'{image_id:04d}' / name
            if not path.is_file() or path.is_symlink() or sha256(path) != row.get('files', {}).get(name):
                raise ValueError('Comparison image SHA mismatch: ' + str(path))
    dest = PROJECT / 'reports' / 'results' / STUDY / run.name
    public_manifest_path = dest / 'manifest.json'
    public_snapshot_path = dest / 'snapshot.json'
    if not public_manifest_path.is_file() or not public_snapshot_path.is_file():
        raise ValueError('Publish the verified VM1 result snapshot before publishing images.')
    public_meta = read(public_manifest_path)
    if (public_meta.get('RunID') != run.name or public_meta.get('schema') != 2 or
            sha256(public_snapshot_path) != public_meta.get('SHA256')):
        raise ValueError('Published snapshot identity/SHA check failed.')
    target = dest / 'image_comparison'
    if target.exists():
        raise FileExistsError('Public image comparison already exists: ' + str(target))
    shutil.copytree(local, target)
    # Augment published artifact checksums without modifying the result snapshot.
    public_meta.setdefault('files', {})
    for path in sorted(target.rglob('*')):
        if path.is_file():
            public_meta['files'][path.relative_to(dest).as_posix()] = sha256(path)
    public_manifest_path.write_text(json.dumps(public_meta, indent=2) + '\n', encoding='utf-8')


def create_comparisons(run, ids=(1, 2, 3), dataset_zip=None, confidence=.25,
                       max_width=1280, publish=False, acknowledge_license=False):
    import cv2
    import numpy as np
    from PIL import Image
    from main.dataset import VisDroneZIP
    run = Path(run).resolve()
    if not run.is_dir() or not (run / 'run.json').is_file():
        raise ValueError('Expected local VM1 RunID directory.')
    run_meta = read(run / 'run.json')
    result = read(run / 'result.json')
    if (run_meta.get('RunID') != run.name or run_meta.get('ExperimentID') != STUDY or
            run_meta.get('status') != 'PASS' or run_meta.get('mode') != 'full-paired' or
            result.get('status') != 'PASS' or result.get('images_per_condition') != 548 or
            result.get('conditions') != 2):
        raise ValueError('Only completed VM1 548-image paired runs are supported.')
    clean, degraded = [run / name for name in ('CLEAN', 'DEGRADED')]
    s_clean, s_deg = verify_condition(clean), verify_condition(degraded)
    if s_clean['GT_SHA256'] != s_deg['GT_SHA256']:
        raise ValueError('CLEAN and DEGRADED GT differ.')
    cfg = read(run / 'configuration.json')
    dataset_zip = Path(dataset_zip or cfg['dataset_zip']).resolve()
    if not dataset_zip.is_file() or sha256(dataset_zip) != cfg['dataset_zip_sha256']:
        raise ValueError('The original validation ZIP is missing or hash-mismatched.')
    ids = list(ids)
    if not ids or len(set(ids)) != len(ids) or len(ids) > 5 or any(not 1 <= i <= 548 for i in ids):
        raise ValueError('Choose 1-5 distinct image IDs in range 1..548.')
    if not (0 <= confidence <= 1 and 480 <= max_width <= 1920):
        raise ValueError('Invalid rendering settings.')
    if publish and not acknowledge_license:
        raise ValueError('Publishing VisDrone imagery requires --acknowledge-license after reviewing rights.')
    local = run / 'image_comparison'
    if local.exists():
        if not publish:
            raise FileExistsError('Comparison already generated: ' + str(local))
        old = read(local / 'manifest.json')
        if ([row['image_id'] for row in old.get('images', [])] != ids or
                old.get('display_confidence') != confidence):
            raise ValueError('Existing comparison uses different IDs or confidence; review it before publication.')
        publish_verified_preview(local, run)
        return local

    gt = read(clean / 'ground_truth.json')
    names = {int(x['id']): x['name'] for x in gt['categories']}
    frames = {int(x['id']): x for x in gt['images']}
    clean_pred = defaultdict(list)
    degraded_pred = defaultdict(list)
    for folder, index in [(clean, clean_pred), (degraded, degraded_pred)]:
        for entry in read(folder / 'predictions_coco.json'):
            if int(entry['image_id']) in ids:
                index[int(entry['image_id'])].append(entry)
    traces = {name: {int(x['image_id']): x for x in read(folder / 'trace.json')}
              for name, folder in [('CLEAN', clean), ('DEGRADED', degraded)]}
    temp = run / 'image_comparison.incomplete'
    if temp.exists():
        raise FileExistsError('Previous incomplete export needs manual review: ' + str(temp))
    temp.mkdir()
    manifest = {
        'schema': 1, 'RunID': run.name, 'images': [],
        'display_confidence': confidence,
        'description': 'Matched CLEAN / DEGRADED inputs and predictions, display only',
        'dataset': 'VisDrone2019-DET-val',
        'dataset_source': 'https://github.com/VisDrone/VisDrone-Dataset',
        'attribution': 'VisDrone2019-DET, AISKYEYE team, Tianjin University',
        'licensing_note': 'Verify the terms for the exact dataset release before public redistribution.',
        'source_summary_SHA256': {'CLEAN': sha256(clean / 'summary.json'),
                                  'DEGRADED': sha256(degraded / 'summary.json')},
        'preview_only': True,
    }
    archive = VisDroneZIP(dataset_zip, 548)
    try:
        for image_id in ids:
            original_bgr, record, source_proof = archive.record(image_id - 1)
            a, b = traces['CLEAN'][image_id], traces['DEGRADED'][image_id]
            if (record['id'] != image_id or frames[image_id]['file_name'] != record['file_name'] or
                    source_proof['image_SHA256'] != a['image_SHA256'] or
                    source_proof['image_SHA256'] != b['image_SHA256']):
                raise ValueError('Source image provenance mismatch: ' + str(image_id))
            original_pixels_sha = hashlib.sha256(original_bgr.tobytes()).hexdigest()
            if original_pixels_sha != a['output_pixel_SHA256']:
                raise ValueError('CLEAN pixels do not match the source.')
            transformed_name = b.get('transformed_file')
            if not transformed_name:
                raise ValueError('Missing saved transformed image for image ID ' + str(image_id))
            transformed_file = safe_child(degraded, transformed_name)
            if sha256(transformed_file) != b['transformed_file_SHA256']:
                raise ValueError('Transformed image hash mismatch.')
            degraded_bgr = cv2.imread(str(transformed_file), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if (degraded_bgr is None or degraded_bgr.shape != original_bgr.shape or
                    hashlib.sha256(degraded_bgr.tobytes()).hexdigest() != b['output_pixel_SHA256']):
                raise ValueError('Transformed image pixel mismatch.')
            if np.array_equal(original_bgr, degraded_bgr):
                raise ValueError('DEGRADED image is identical to CLEAN: ' + str(image_id))
            original = Image.fromarray(cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB))
            degraded_image = Image.fromarray(cv2.cvtColor(degraded_bgr, cv2.COLOR_BGR2RGB))
            sample = temp / f'{image_id:04d}'
            sample.mkdir()
            rendered = [original, degraded_image,
                        draw_predictions(original, clean_pred[image_id], names, confidence),
                        draw_predictions(degraded_image, degraded_pred[image_id], names, confidence)]
            image_hashes = {}
            for name, frame in zip(IMAGE_NAMES, rendered):
                output_jpeg(frame, sample / name, max_width)
                image_hashes[name] = sha256(sample / name)
            manifest['images'].append({
                'image_id': image_id,
                'source_file': record['file_name'],
                'source_image_SHA256': source_proof['image_SHA256'],
                'degraded_pixels_SHA256': b['output_pixel_SHA256'],
                'configuration_id': b['configuration_id'],
                'recipe': b['recipe'],
                'files': image_hashes,
                'displayed_clean_boxes': sum(float(p['score']) >= confidence for p in clean_pred[image_id]),
                'displayed_degraded_boxes': sum(float(p['score']) >= confidence for p in degraded_pred[image_id]),
            })
        (temp / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')
        blocks = []
        for sample in manifest['images']:
            folder_name = f"{sample['image_id']:04d}"
            caption = html.escape(sample['source_file'])
            items = ''.join(
                f'<figure><figcaption>{title}</figcaption><img src="{folder_name}/{filename}"></figure>'
                for title, filename in [('CLEAN input', IMAGE_NAMES[0]), ('DEGRADED input', IMAGE_NAMES[1]),
                                        ('CLEAN predictions', IMAGE_NAMES[2]), ('DEGRADED predictions', IMAGE_NAMES[3])]
            )
            blocks.append(f'<section><h2>Image {folder_name} · {caption}</h2><div class="grid">{items}</div></section>')
        page = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>VM1 matched images</title>'
                '<style>body{font:16px Arial;margin:24px auto;padding:0 18px;max-width:1450px;background:#f5f7fa;color:#243449}'
                'section,figure{background:white;border-radius:8px;padding:12px}section{margin:20px 0}'
                '.grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}figure{margin:0}'
                'figcaption{font-weight:bold;margin-bottom:8px}img{width:100%;height:auto}'
                '@media(max-width:720px){.grid{grid-template-columns:1fr}}</style>'
                '<h1>VM1 · Matched CLEAN / DEGRADED images</h1><p>Display-only previews from saved predictions; '
                'evaluation AP is unchanged.</p>' + ''.join(blocks) +
                '<p>Images: VisDrone2019-DET, AISKYEYE team, Tianjin University. Verify dataset terms before redistribution.</p></html>')
        (temp / 'index.html').write_text(page, encoding='utf-8')
        temp.rename(local)
    finally:
        archive.close()
    if publish:
        publish_verified_preview(local, run)
    return local


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--ids', type=int, nargs='+', default=[1, 2, 3], help='1-5 image IDs, no GPU')
    parser.add_argument('--dataset-zip', type=Path, help='Override relocated original VisDrone ZIP path')
    parser.add_argument('--confidence', type=float, default=.25)
    parser.add_argument('--max-width', type=int, default=1280)
    parser.add_argument('--publish', action='store_true', help='Copy verified thumbnails into the public Run directory')
    parser.add_argument('--acknowledge-license', action='store_true', help='Confirm rights/attribution for public preview images')
    args = parser.parse_args()
    folder = create_comparisons(args.run, args.ids, args.dataset_zip, args.confidence,
                                args.max_width, args.publish, args.acknowledge_license)
    print('Image comparisons saved: ' + str(folder), flush=True)
    if args.publish:
        print('Public preview copied. Review files, then commit and push manually.', flush=True)


if __name__ == '__main__':
    main()
