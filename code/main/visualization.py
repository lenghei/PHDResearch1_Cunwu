"""Local detection previews from saved predictions and metrics."""
import html
import json
import webbrowser
from collections import defaultdict
from pathlib import Path

from .config import sha256, save


def create_preview(run_path, open_browser=True, limit=5, display_confidence=0.25):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    from .dataset import VisDroneZIP

    run = Path(run_path).resolve()
    project = Path(__file__).resolve().parents[2]
    if any(run.is_relative_to(project / name) for name in ['code', 'experiments', 'reports']):
        raise ValueError('Image previews must stay outside public directories.')
    if limit < 1 or not 0 <= display_confidence <= 1:
        raise ValueError('Invalid preview limit or display confidence.')
    read = lambda name: json.loads((run / name).read_text(encoding='utf-8'))
    summary, metrics, gt = read('summary.json'), read('metrics.json'), read('ground_truth.json')
    cfg, trace = read('configuration.json'), read('trace.json')
    for name, key in [('metrics.json', 'metrics_SHA256'), ('ground_truth.json', 'GT_SHA256'),
                      (summary.get('raw_prediction_file', 'predictions_raw.json'), 'predictions_SHA256')]:
        if sha256(run / name) != summary[key]:
            raise ValueError('Saved artifact SHA mismatch: ' + name)
    predictions = read('predictions_coco.json')
    folder = run / 'preview'
    folder.mkdir(exist_ok=False)
    names = {cat['id']: cat['name'] for cat in gt['categories']}
    by_gt, by_pred = defaultdict(list), defaultdict(list)
    for ann in gt['annotations']:
        by_gt[ann['image_id']].append(ann)
    for pred in predictions:
        if pred['score'] >= display_confidence:
            by_pred[pred['image_id']].append(pred)
    provenance = {record['image_id']: record for record in trace}
    palette = [tuple(int(c * 255) for c in color[:3]) for color in plt.get_cmap('tab10').colors]

    def annotated(image, boxes, predicted=False):
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        for box in boxes:
            x, y, w, h = box['bbox']
            color = palette[(box['category_id'] - 1) % len(palette)]
            draw.rectangle((x, y, x + w, y + h), outline=color, width=2)
            label = names[box['category_id']]
            if predicted:
                label += f" {box['score']:.2f}"
            draw.text((max(0, x), max(0, y - 11)), label, fill=color,
                      stroke_width=1, stroke_fill='black')
        return canvas

    cards, image_proofs = [], []
    archive = VisDroneZIP(cfg['dataset_zip'], cfg['expected_images'])
    try:
        for image_info in gt['images'][:limit]:
            image_id = image_info['id']
            pixels, record, proof = archive.record(image_id - 1)
            if (record['file_name'] != image_info['file_name'] or
                    proof['image_SHA256'] != provenance[image_id]['image_SHA256']):
                raise ValueError('Preview image provenance mismatch.')
            transformed = provenance[image_id].get('transformed_file')
            if transformed:
                import cv2
                import hashlib
                source = (run / transformed).resolve()
                if not source.is_relative_to(run) or sha256(source) != provenance[image_id]['transformed_file_SHA256']:
                    raise ValueError('Transformed preview source mismatch.')
                pixels = cv2.imread(str(source), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
                if pixels is None or pixels.shape != (record['height'], record['width'], 3):
                    raise ValueError('Transformed preview geometry mismatch.')
                if hashlib.sha256(pixels.tobytes()).hexdigest() != provenance[image_id]['output_pixel_SHA256']:
                    raise ValueError('Transformed preview pixels changed.')
            image = Image.fromarray(pixels[:, :, ::-1])
            a, b = annotated(image, by_gt[image_id]), annotated(image, by_pred[image_id], True)
            pair = Image.new('RGB', (2 * image.width, image.height + 28), 'white')
            pair.paste(a, (0, 28))
            pair.paste(b, (image.width, 28))
            draw = ImageDraw.Draw(pair)
            draw.text((8, 8), 'Ground truth', fill='black')
            draw.text((image.width + 8, 8), f'Predictions >= {display_confidence:.2f}', fill='black')
            filename = f'image_{image_id:04d}.jpg'
            pair.save(folder / filename, quality=92)
            cards.append(f'<figure><figcaption>{html.escape(image_info["file_name"])}</figcaption>'
                         f'<a href="{filename}"><img src="{filename}" alt="GT and predictions"></a></figure>')
            image_proofs.append({'image_id': image_id, 'image_SHA256': proof['image_SHA256']})
    finally:
        archive.close()

    area_rows = [r for r in metrics['per_area'] if r['size'] != 'coco_small']
    class_rows = [r for r in metrics['per_class'] if r['size'] == 'all']

    def chart(rows, label_key, filename, title):
        fig, ax = plt.subplots(figsize=(8, max(3, len(rows) * .35)))
        valid = [r for r in rows if r['map50_95'] is not None]
        ax.barh([r[label_key] for r in valid], [100 * r['map50_95'] for r in valid], color='#3575ad')
        ax.invert_yaxis()
        ax.set(xlim=(0, 100), xlabel='AP50:95 (%)', title=title)
        ax.grid(axis='x', alpha=.2)
        fig.tight_layout()
        fig.savefig(folder / filename, dpi=160)
        plt.close(fig)

    chart(area_rows, 'size', 'size_ap.png', 'AP by original-pixel object size')
    chart(class_rows, 'category_name', 'class_ap.png', 'AP by class (classes with GT)')
    fmt = lambda value: 'N/A' if value is None else f'{value * 100:.2f}%'
    rows = ''.join(f'<tr><td>{html.escape(r["size"])}</td><td>{fmt(r["map50"])}</td>'
                   f'<td>{fmt(r["map50_95"])}</td><td>{fmt(r["ar50_95"])}</td></tr>' for r in area_rows)
    aggregate = summary['aggregate']
    content = f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>Detection results</title>
<style>body{{max-width:1300px;margin:24px auto;padding:0 18px;font:16px Arial;color:#203040;background:#f5f7fa}}
img{{max-width:100%;height:auto}}figure,section{{margin:18px 0;padding:16px;background:white;border-radius:8px}}
table{{border-collapse:collapse}}th,td{{padding:8px 20px;border-bottom:1px solid #ddd;text-align:left}}
.charts{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}@media(max-width:700px){{.charts{{grid-template-columns:1fr}}}}</style>
<h1>YOLO detection results</h1><p>{html.escape(summary['RunID'])}</p>
<section><b>{summary['images']} images | {html.escape(summary['checkpoint_id'])} | {html.escape(summary['scope'])}</b>
<p>AP50: {fmt(aggregate['map50'])} &nbsp; AP50:95: {fmt(aggregate['map50_95'])} &nbsp;
AR50:95: {fmt(aggregate['ar50_95'])}</p><p>Display confidence: {display_confidence:.2f}. Evaluation metrics use all saved predictions.
Previews show the first {len(cards)} images in the evaluated set. Click an image to enlarge.</p>
<table><tr><th>Size</th><th>AP50</th><th>AP50:95</th><th>AR50:95</th></tr>{rows}</table></section>
<section class="charts"><img src="size_ap.png" alt="Size AP"><img src="class_ap.png" alt="Class AP"></section>
{''.join(cards)}</html>'''
    page = folder / 'index.html'
    page.write_text(content, encoding='utf-8')
    save(folder / 'manifest.json', {'run_id': summary['RunID'], 'metrics_SHA256': summary['metrics_SHA256'],
                                  'predictions_coco_SHA256': sha256(run / 'predictions_coco.json'),
                                  'display_confidence': display_confidence, 'images': image_proofs,
                                  'files': {p.name: sha256(p) for p in folder.iterdir() if p.is_file()}})
    if open_browser:
        webbrowser.open(page.as_uri())
    return page


def create_status_preview(run_path):
    run = Path(run_path).resolve()
    result = json.loads((run / 'result.json').read_text(encoding='utf-8'))
    page = run / 'index.html'
    body = html.escape(json.dumps(result, indent=2))
    page.write_text(f'''<!doctype html><meta charset="utf-8"><title>CPU run results</title>
<style>body{{max-width:1000px;margin:32px auto;font:16px Arial;padding:0 20px}}
pre{{white-space:pre-wrap;background:#f3f6f9;padding:20px}}</style>
<h1>CPU run results</h1><p>{html.escape(run.name)}</p>
<p>No new detector inference was performed.</p><pre>{body}</pre>
<p><a href="result.json">Result JSON</a> · <a href="execution.log">Execution log</a></p>''', encoding='utf-8')
    return page


def _history_record(folder, experiment_id):
    import datetime
    import re
    from urllib.parse import quote

    metadata = next((folder / name for name in ['run.json', 'summary.json'] if (folder / name).is_file()), None)
    if metadata is None or not metadata.resolve().is_relative_to(folder):
        return None
    try:
        record = json.loads(metadata.read_text(encoding='utf-8'))
        if not isinstance(record, dict):
            raise ValueError('Expected an object')
        status = str(record.get('status', record.get('Status', 'UNKNOWN'))).upper()
    except (OSError, ValueError):
        record, status = {}, 'UNREADABLE'
    if record.get('ExperimentID', experiment_id) != experiment_id:
        status = 'EXPERIMENT_MISMATCH'
    mode = record.get('mode')
    if not mode:
        mode = ('full-validation' if str(record.get('scope', '')).startswith('FULL') else 'smoke') if metadata.name == 'summary.json' else 'unknown'
    match = re.search(r'(?:^|_)(\d{8}T\d{6})(?:_|$)', folder.name)
    try:
        started = datetime.datetime.strptime(match.group(1), '%Y%m%dT%H%M%S').replace(tzinfo=datetime.timezone.utc) if match else None
    except ValueError:
        started = None
    time_source = 'RunID_UTC'
    if started is None:
        started = datetime.datetime.fromtimestamp(metadata.stat().st_mtime, datetime.timezone.utc)
        time_source = 'metadata_mtime'
    choices = ['index.html', 'preview/index.html', 'result.json', 'summary.json', 'run.json']
    if status != 'PASS':
        choices = ['failure.json', 'run.json', 'summary.json', 'result.json', 'index.html']
    target = next((folder / name for name in choices if (folder / name).is_file()
                   and (folder / name).resolve().is_relative_to(folder)), metadata)
    return {'RunID': folder.name, 'mode': str(mode), 'status': status,
            'started_at': started.isoformat(timespec='microseconds').replace('+00:00', 'Z'),
            'time_source': time_source, 'page': quote(target.relative_to(folder.parent).as_posix(), safe='/'),
            'page_kind': 'Results' if target.suffix == '.html' else 'Metadata'}


def build_run_index(group_path, experiment_id=None):
    group = Path(group_path).resolve()
    public = Path(__file__).resolve().parents[2]
    if group.is_relative_to(public):
        raise ValueError('Run indexes must stay outside the publication root.')
    experiment_id = experiment_id or group.name
    history = []
    for folder in group.iterdir():
        if folder.is_dir() and folder.resolve().is_relative_to(group):
            row = _history_record(folder.resolve(), experiment_id)
            if row is not None:
                history.append(row)
    history.sort(key=lambda row: (row['started_at'], row['RunID']), reverse=True)
    modes = {}
    for row in history:
        if row['status'] == 'PASS' and row['mode'] not in modes:
            modes[row['mode']] = {'RunID': row['RunID'], 'page': row['page'], 'page_kind': row['page_kind']}
    save(group / 'latest.json', {'ExperimentID': experiment_id, 'modes': modes, 'history': history})
    esc = html.escape
    cards = ''.join(f'<a class="card" href="{esc(item["page"], quote=True)}"><b>{esc(mode)}</b><br>'
                    f'Latest successful · {esc(item["page_kind"])}</a>' for mode, item in sorted(modes.items()))
    rows = []
    for row in history:
        status_class = 'pass' if row['status'] == 'PASS' else 'running' if row['status'] == 'RUNNING' else 'other'
        if row['status'] in ['FAILED', 'PARTIAL', 'BLOCKED', 'UNREADABLE', 'EXPERIMENT_MISMATCH']:
            status_class = 'failed'
        when = row['started_at'].split('.')[0].replace('T', ' ') + ' UTC'
        if row['time_source'] != 'RunID_UTC':
            when += ' (file time)'
        rows.append(f'<tr data-mode="{esc(row["mode"], quote=True)}" data-status="{esc(row["status"], quote=True)}">'
                    f'<td>{esc(when)}</td><td>{esc(row["mode"])}</td>'
                    f'<td><span class="badge {status_class}">{esc(row["status"])}</span></td>'
                    f'<td><a href="{esc(row["page"], quote=True)}">{esc(row["page_kind"])}</a>'
                    f'<details><summary>Run ID</summary><code>{esc(row["RunID"])}</code></details></td></tr>')
    options = lambda values: ''.join(f'<option value="{esc(value, quote=True)}">{esc(value)}</option>' for value in sorted(values))
    content = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Experiment results</title><style>
body{{max-width:1280px;margin:32px auto;padding:0 20px;font:16px Arial;color:#243449;background:#f5f7fa}}
h1{{font-size:24px;overflow-wrap:anywhere}}.cards{{display:flex;flex-wrap:wrap;gap:12px;margin:20px 0}}
a{{color:#205f9b}}.card{{background:white;border:1px solid #dce3eb;border-radius:8px;padding:16px;text-decoration:none;line-height:1.8}}
.filters{{display:flex;flex-wrap:wrap;gap:10px;margin:20px 0}}select,input{{padding:10px;border:1px solid #ccd6e0;border-radius:5px;font:inherit}}
.table{{overflow-x:auto;background:white;border-radius:8px}}table{{width:100%;border-collapse:collapse;text-align:left}}
th,td{{padding:14px;border-bottom:1px solid #e5eaf0;vertical-align:top}}th{{background:#eaf0f6}}td:first-child{{white-space:nowrap}}
.badge{{padding:4px 8px;border-radius:4px;font-size:13px}}.pass{{background:#d9f2e2;color:#21613b}}
.failed{{background:#fce2df;color:#8f3025}}.running{{background:#fff0c7;color:#795600}}.other{{background:#e8ebee}}
details{{margin-top:8px;font-size:13px;color:#66788a}}code{{overflow-wrap:anywhere}}.muted{{color:#66788a}}
</style><h1>{esc(experiment_id)}</h1><p class="muted">{len(history)} recorded runs · newest first · timestamps in UTC</p>
<div class="cards">{cards}</div><h2>Run history</h2>
<div class="filters"><select id="mode"><option value="">All modes</option>{options({r['mode'] for r in history})}</select>
<select id="status"><option value="">All statuses</option>{options({r['status'] for r in history})}</select>
<input id="search" type="search" placeholder="Search run history" aria-label="Search run history"></div>
<div class="table"><table><thead><tr><th>Time</th><th>Mode</th><th>Saved status</th><th>Open</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div><p id="count" class="muted"></p>
<script>
const mode=document.getElementById('mode'),status=document.getElementById('status'),search=document.getElementById('search');
function filterRows(){{let shown=0;document.querySelectorAll('tbody tr').forEach(row=>{{
const visible=(!mode.value||row.dataset.mode===mode.value)&&(!status.value||row.dataset.status===status.value)&&row.textContent.toLowerCase().includes(search.value.toLowerCase());
row.hidden=!visible;if(visible)shown++;}});document.getElementById('count').textContent=shown+' runs shown';}}
[mode,status,search].forEach(control=>control.addEventListener('input',filterRows));filterRows();
</script></html>'''
    page = group / 'index.html'
    temporary = group / 'index.html.partial'
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write(content)
    temporary.replace(page)
    return page


def refresh_config_index(config_file, experiment_id):
    import os
    import yaml
    cfg = yaml.safe_load(Path(config_file).read_text(encoding='utf-8'))
    value = os.path.expandvars(cfg['output_root'])
    if '$' in value or '%' in value:
        raise ValueError('Set the environment variable for output_root')
    return build_run_index(Path(value).expanduser(), experiment_id)


def update_run_index(run_path):
    run = Path(run_path).resolve()
    record = json.loads((run / 'run.json').read_text(encoding='utf-8'))
    if record['status'] != 'PASS' or not (run / 'index.html').is_file():
        raise ValueError('Only completed runs with a result page can be indexed.')
    return build_run_index(run.parent, record['ExperimentID'])


def create_interaction_preview(run_path):
    import csv
    import statistics
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    run = Path(run_path).resolve()
    read = lambda name: json.loads((run / name).read_text(encoding='utf-8'))
    rows, effects, result = read('measured_grids.json'), read('interaction_replay.json'), read('result.json')
    folder = run / 'preview'
    folder.mkdir(exist_ok=False)
    grid = [r for r in rows if r['measurement_kind'] == 'grid']
    checkpoints = sorted({r['checkpoint'] for r in grid})
    fig, axes = plt.subplots(1, len(checkpoints), figsize=(5 * len(checkpoints), 4), squeeze=False)
    points = []
    for ax, checkpoint in zip(axes[0], checkpoints):
        for resolution in [640, 960, 1280]:
            values = []
            for blur in [0, 1, 2]:
                selected = [r['AP50_95'] for r in grid if r['checkpoint'] == checkpoint
                            and r['resolution'] == resolution and r['blur'] == blur and r['AP50_95'] is not None]
                value = statistics.mean(selected)
                values.append(value)
                points.append({'checkpoint': checkpoint, 'resolution': resolution, 'blur': blur,
                               'mean_configuration_AP50_95': value, 'configurations': len(selected)})
            ax.plot([0, 1, 2], values, marker='o', label=str(resolution))
        ax.set(title=checkpoint, xlabel='Motion blur level', ylabel='Mean configuration AP50:95',
               xticks=[0, 1, 2], ylim=(0, 1))
        ax.grid(alpha=.2)
        ax.legend(title='Input')
    fig.tight_layout()
    fig.savefig(folder / 'resolution_blur.png', dpi=160)
    plt.close(fig)
    with (folder / 'plot_data.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(points[0]))
        writer.writeheader()
        writer.writerows(points)
    grouped = defaultdict(list)
    for effect in effects:
        if effect['status'] == 'PASS':
            checkpoint = effect.get('checkpoint') or effect['context']['checkpoint']
            grouped[(checkpoint, effect['relation'])].append(effect['estimate'])
    labels, values = [], []
    for (checkpoint, relation), estimates in sorted(grouped.items()):
        labels.append(checkpoint + ': ' + relation)
        values.append(statistics.mean(estimates))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(labels, values, color='#3575ad')
    ax.axvline(0, color='black', linewidth=.8)
    ax.invert_yaxis()
    ax.set_xlabel('Mean matched finite-difference contrast (AP50:95)')
    fig.tight_layout()
    fig.savefig(folder / 'interaction_effects.png', dpi=160)
    plt.close(fig)
    page = run / 'index.html'
    page.write_text(f'''<!doctype html><meta charset="utf-8"><title>Degradation interactions</title>
<style>body{{max-width:1250px;margin:28px auto;font:16px Arial;padding:0 20px}}img{{max-width:100%}}</style>
<h1>Degradation interactions</h1><p>{html.escape(run.name)}</p>
<p>{result['measurement_rows']} historical configuration records. Development validation; no new inference.</p>
<p>Each line point averages matched illumination/noise configuration APs, not pooled-image AP.</p>
<img src="preview/resolution_blur.png" alt="Resolution and blur">
<p>Contrasts require observed four/eight-corner sets. These are descriptive effects, without significance claims.
JPEG and the original grid retain separate cohorts.</p>
<img src="preview/interaction_effects.png" alt="Matched interaction contrasts">
<p>Static coverage: {html.escape(result['static_gap'])}. Empirical evaluation: {html.escape(result['empirical_gap'])}.</p>
<p><a href="configuration_metrics.csv">Configuration metrics</a> ·
<a href="interaction_replay.json">Matched contrasts</a> · <a href="preview/plot_data.csv">Plot data</a> ·
<a href="provenance.json">Sources</a></p>''', encoding='utf-8')
    save(folder / 'manifest.json', {'source_SHA256': {name: sha256(run / name) for name in
         ['measured_grids.json', 'interaction_replay.json']},
         'files': {p.name: sha256(p) for p in folder.iterdir() if p.is_file()}})
    return page
