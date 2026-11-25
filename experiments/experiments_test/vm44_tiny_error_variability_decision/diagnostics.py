"""Streaming descriptive diagnostics on frozen VM43 predictions."""
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
import math
import numpy as np

from vm44_io import V43, read, save, csv_write, source
from main.labels import VIS_NAMES, keep_outside_ignore, mapped_predictions

SCORE = .25
IOU = .50
SIZES = ('all', 'tiny', 'small', 'medium', 'large')
SCORE_EDGES = [0, .001, .01, .05, .10, .25, .50, .75, 1.0]
IOU_EDGES = [0, .1, .25, .5, .75, .9, 1.0]
AREA_EDGES = [0, 16, 64, 144, 256, 1024, 9216, float('inf')]


def size(area):
    return 'tiny' if area < 256 else 'small' if area < 1024 else 'medium' if area < 9216 else 'large'


def iou_matrix(pred, gt):
    a = np.asarray([p['bbox'] for p in pred], dtype=np.float64).reshape(-1, 4)
    b = np.asarray([g['bbox'] for g in gt], dtype=np.float64).reshape(-1, 4)
    left = np.maximum(a[:, None, :2], b[None, :, :2])
    right = np.minimum(a[:, None, :2]+a[:, None, 2:], b[None, :, :2]+b[None, :, 2:])
    inter = np.maximum(0, right-left).prod(axis=2)
    return inter/np.maximum(a[:, None, 2:].prod(axis=2)+b[None, :, 2:].prod(axis=2)-inter, 1e-30)


def retained(raw, record):
    for p in raw:
        if p['image_id'] != record['id'] or not isinstance(p['class_id'], int) or isinstance(p['class_id'], bool) or not 0 <= p['class_id'] < 10:
            raise ValueError('Unexpected image/class in raw predictions.')
        if not all(math.isfinite(float(x)) for x in [*p['bbox'], p['score']]) or p['bbox'][2] <= 0 or p['bbox'][3] <= 0 or not 0 <= p['score'] <= 1:
            raise ValueError('Invalid saved box/score.')
    ordered = sorted(enumerate(raw), key=lambda pair: -pair[1]['score'])[:3000]
    keep = keep_outside_ignore([p['bbox'] for _, p in ordered], record['ignore_regions']+record['other_regions'], record['width'], record['height'])
    return [{'index': i, 'category_id': p['class_id']+1, 'bbox': p['bbox'], 'score': p['score']} for (i, p), valid in zip(ordered, keep) if valid], len(ordered)-int(keep.sum())


def match(pred, gt):
    overlaps = iou_matrix(pred, gt)
    used = np.zeros(len(gt), dtype=bool)
    pred_match, gt_match = {}, {}
    for i in sorted(range(len(pred)), key=lambda j: (-pred[j]['score'], pred[j]['index'])):
        if pred[i]['score'] < SCORE:
            continue
        choices = [j for j, g in enumerate(gt) if not used[j] and g['category_id'] == pred[i]['category_id'] and overlaps[i, j] >= IOU]
        if choices:
            j = min(choices, key=lambda j: (-overlaps[i, j], gt[j]['id']))
            used[j] = True
            pred_match[i] = j
            gt_match[j] = i
    return pred_match, gt_match, overlaps


def reference_match(pred, gt):
    used, result = set(), {}
    def one(a, b):
        ax, ay, aw, ah = a; bx, by, bw, bh = b
        inter = max(0, min(ax+aw, bx+bw)-max(ax, bx))*max(0, min(ay+ah, by+bh)-max(ay, by))
        return inter/(aw*ah+bw*bh-inter)
    for i in sorted(range(len(pred)), key=lambda j: (-pred[j]['score'], pred[j]['index'])):
        if pred[i]['score'] < SCORE:
            continue
        candidates = [(one(pred[i]['bbox'], g['bbox']), g['id'], j) for j, g in enumerate(gt) if j not in used and g['category_id'] == pred[i]['category_id']]
        candidates = [v for v in candidates if v[0] >= IOU]
        if candidates:
            _, _, j = min(candidates, key=lambda v: (-v[0], v[1]))
            result[i] = j; used.add(j)
    return result


def groups(category, area):
    return [(c, a) for c in (0, category) for a in ('all', size(area))]


def category_name(category):
    return 'ALL' if category == 0 else VIS_NAMES[category-1]


def analyze(out, spec, payload, logger):
    records = payload['records']
    by_image = defaultdict(list)
    for ann in payload['coco']['annotations']:
        by_image[ann['image_id']].append(ann)
    for gt in by_image.values():
        gt.sort(key=lambda g: g['id'])
    folder = out/'diagnostic_records'; folder.mkdir()
    sums, fp_sums, hist, iou_hist, fine = defaultdict(Counter), defaultdict(Counter), Counter(), Counter(), defaultdict(Counter)
    per_image, replay, class_metrics = [], [], []
    sampled = set(np.random.default_rng(20261117).choice(len(records), 12, replace=False).tolist())
    for cell in spec['cells']:
        key = cell['cell_id']; original = V43/key
        metrics = read(original/'metrics.json')
        for row in metrics['per_class']:
            class_metrics.append({'cell_id': key, 'condition': cell['condition'], 'strategy': cell['strategy'], 'metric_role': 'UNCHANGED_SAVED_COCO', **row})
        raw_count = ignored_count = kept_count = 0
        dest = folder/(key+'.jsonl.gz')
        with gzip.open(original/'predictions.jsonl.gz', 'rt', encoding='utf-8') as stream, gzip.open(dest, 'xt', encoding='utf-8') as saved:
            for index, rec in enumerate(records):
                line = stream.readline()
                if not line:
                    raise ValueError('Truncated raw prediction stream.')
                row = json.loads(line)
                if row['image_id'] != rec['id']:
                    raise ValueError('Unmatched or reordered image IDs.')
                pred, ignored = retained(row['predictions'], rec)
                gt = by_image[rec['id']]
                pm, gm, overlaps = match(pred, gt)
                same_class = np.equal(np.asarray([p['category_id'] for p in pred])[:, None], np.asarray([g['category_id'] for g in gt])[None, :])
                high_score = np.asarray([p['score'] >= SCORE for p in pred], dtype=bool)[:, None]
                high_same = overlaps.max(axis=0, where=same_class & high_score, initial=0)
                low_same = overlaps.max(axis=0, where=same_class & ~high_score, initial=0)
                high_wrong = overlaps.max(axis=0, where=~same_class & high_score, initial=0)
                pred_same = overlaps.max(axis=1, where=same_class, initial=0)
                pred_wrong = overlaps.max(axis=1, where=~same_class, initial=0)
                if index in sampled:
                    independent = reference_match(pred, gt)
                    if pm != independent:
                        raise AssertionError('Vectorized and scalar matching differ.')
                    mapped = mapped_predictions(row['predictions'], [rec], {rec['id']: rec['ignore_regions']+rec['other_regions']}, 'native10', 'finetuned', 3000)
                    ours = [{'image_id': rec['id'], **{k: p[k] for k in ('category_id', 'bbox', 'score')}} for p in pred]
                    if mapped != ours:
                        raise AssertionError('Ignore-union mapping differs from saved evaluator.')
                    replay.append({'cell': key, 'image_id': rec['id'], 'stem': rec['stem'], 'source_row_SHA256': hashlib.sha256(line.encode()).hexdigest(),
                                   'raw': len(row['predictions']), 'retained': len(pred), 'TP': len(pm), 'FP': sum(p['score'] >= SCORE for p in pred)-len(pm),
                                   'FN': len(gt)-len(gm), 'independent_scalar_match': 'PASS', 'official_ignore_mapping': 'PASS'})
                outcomes, selected = [], []
                for j, g in enumerate(gt):
                    i = gm.get(j)
                    if i is not None:
                        reason = 'DETECTED'
                    elif high_same[j] >= .5:
                        reason = 'COMPETING_MATCH_CANDIDATE'
                    elif low_same[j] >= .5:
                        reason = 'LOW_CONFIDENCE_MATCH_CANDIDATE'
                    elif high_wrong[j] >= .5:
                        reason = 'WRONG_CLASS_OVERLAP_CANDIDATE'
                    elif high_same[j] >= .1:
                        reason = 'LOCALIZATION_NEAR_MISS_CANDIDATE'
                    else:
                        reason = 'NO_HIGH_SCORE_LOCALIZED_CANDIDATE'
                    outcome = {'GT_id': g['id'], 'category_id': g['category_id'], 'bbox': g['bbox'], 'area': g['area'], 'size': size(g['area']),
                               'detected': i is not None, 'prediction_index': pred[i]['index'] if i is not None else None,
                               'IoU': float(overlaps[i, j]) if i is not None else None, 'confidence': pred[i]['score'] if i is not None else None, 'reason': reason}
                    outcomes.append(outcome)
                    for group in groups(g['category_id'], g['area']):
                        acc = sums[key, *group]; acc['GT'] += 1; acc['TP'] += int(i is not None); acc['FN'] += int(i is None); acc[reason] += 1
                    area_bin = int(np.searchsorted(AREA_EDGES, g['area'], side='right')-1)
                    fine[key, area_bin]['GT'] += 1; fine[key, area_bin]['TP'] += int(i is not None)
                    if i is not None:
                        for category in (0, g['category_id']):
                            bin_id = min(len(IOU_EDGES)-2, int(np.searchsorted(IOU_EDGES, overlaps[i, j], side='right')-1))
                            iou_hist[key, category, size(g['area']), bin_id] += 1
                for i, p in enumerate(pred):
                    band = min(len(SCORE_EDGES)-2, max(0, int(np.searchsorted(SCORE_EDGES, p['score'], side='right')-1)))
                    for category in (0, p['category_id']):
                        hist[key, category, 'ALL_RETAINED', band] += 1
                    if p['score'] < SCORE:
                        continue
                    if i in pm:
                        reason, status = 'MATCHED', 'TP'
                    else:
                        reason = ('DUPLICATE_OR_COMPETING_GT' if pred_same[i] >= .5 else 'WRONG_CLASS_OVERLAP_CANDIDATE' if pred_wrong[i] >= .5 else 'LOCALIZATION_NEAR_MISS_CANDIDATE' if pred_same[i] >= .1 else 'BACKGROUND_OR_UNLOCALIZED')
                        status = 'FP'
                    selected.append({**p, 'status': status, 'reason': reason, 'matched_GT_id': gt[pm[i]]['id'] if i in pm else None})
                    for group in groups(p['category_id'], p['bbox'][2]*p['bbox'][3]):
                        fp_sums[key, *group][status] += 1; fp_sums[key, *group][reason] += 1
                    for category in (0, p['category_id']):
                        hist[key, category, status, band] += 1
                raw_count += len(row['predictions']); ignored_count += ignored; kept_count += len(pred)
                per_image.append({'cell_id': key, 'image_id': rec['id'], 'stem': rec['stem'], 'raw_predictions': len(row['predictions']), 'ignored_predictions': ignored,
                                  'retained_predictions': len(pred), 'selected_predictions': len(selected), 'TP': len(pm), 'FP': len(selected)-len(pm), 'FN': len(gt)-len(gm), 'valid_GT': len(gt),
                                  'source_objects_before_ignore': len(rec['objects']), 'GT_removed_by_original_ignore': len(rec['objects'])-len(gt), 'diagnostic_score': SCORE, 'diagnostic_IoU': IOU})
                saved.write(json.dumps({'image_id': rec['id'], 'stem': rec['stem'], 'GT_outcomes': outcomes, 'selected_predictions': selected}, separators=(',', ':'))+'\n')
                if (index+1) % 400 == 0:
                    logger.info('%s: %d/1610 images diagnosed', key, index+1)
            if stream.readline():
                raise ValueError('Extra/duplicate prediction image record.')
        if raw_count != metrics['raw_count'] or kept_count != metrics['mapped_count']:
            raise ValueError('Diagnostic retained population differs from official COCO input.')
        logger.info('%s complete; raw=%d ignored=%d retained=%d', key, raw_count, ignored_count, kept_count)
    fn_rows, fp_rows, size_rows = [], [], []
    for cell in spec['cells']:
        key = cell['cell_id']
        for cat in range(11):
            for area in SIZES:
                s, f = sums[key, cat, area], fp_sums[key, cat, area]
                base = {'cell_id': key, 'condition': cell['condition'], 'strategy': cell['strategy'], 'category_id': cat, 'category_name': category_name(cat), 'size': area}
                fn_rows.append({**base, **dict(s), 'GT': s['GT'], 'TP': s['TP'], 'FN': s['FN'], 'recall_at_score025_IoU050': s['TP']/s['GT'] if s['GT'] else None, 'area_reference': 'GT_original_bbox'})
                fp_rows.append({**base, **dict(f), 'TP': f['TP'], 'FP': f['FP'], 'FP_per_image': f['FP']/1610, 'precision': f['TP']/(f['TP']+f['FP']) if f['TP']+f['FP'] else None, 'area_reference': 'PREDICTED_original_bbox'})
                size_rows.append({**base, 'GT': s['GT'], 'TP': s['TP'], 'FN': s['FN'], 'recall': s['TP']/s['GT'] if s['GT'] else None})
        for b in range(len(AREA_EDGES)-1):
            acc = fine[key, b]
            size_rows.append({'cell_id': key, 'condition': cell['condition'], 'strategy': cell['strategy'], 'category_id': 0, 'category_name': 'ALL', 'size': 'FINE_BIN',
                              'area_low': AREA_EDGES[b], 'area_high': AREA_EDGES[b+1] if math.isfinite(AREA_EDGES[b+1]) else None,
                              'GT': acc['GT'], 'TP': acc['TP'], 'FN': acc['GT']-acc['TP'], 'recall': acc['TP']/acc['GT'] if acc['GT'] else None})
    csv_write(out/'VM43_tiny_class_metrics.csv', class_metrics)
    csv_write(out/'VM43_false_negative_analysis.csv', fn_rows)
    csv_write(out/'VM43_false_positive_analysis.csv', fp_rows)
    csv_write(out/'VM43_size_stratified_diagnostics.csv', size_rows)
    csv_write(out/'VM43_per_image_diagnostics.csv', per_image)
    confidence = [{'cell_id': k, 'category_id': cat, 'category_name': category_name(cat), 'status': status, 'score_low': SCORE_EDGES[b], 'score_high': SCORE_EDGES[b+1], 'count': count,
                   'bin_convention': 'left_closed_right_open; final bin includes1.0'} for (k, cat, status, b), count in sorted(hist.items())]
    csv_write(out/'VM43_detection_confidence_analysis.csv', confidence)
    csv_write(out/'VM43_IoU_distribution.csv', [{'cell_id': k, 'category_id': cat, 'category_name': category_name(cat), 'GT_size': area, 'IoU_low': IOU_EDGES[b], 'IoU_high': IOU_EDGES[b+1], 'count': n}
                                                for (k, cat, area, b), n in sorted(iou_hist.items())])
    save(out/'diagnostic_sample_replay.json', {'seed': 20261117, 'images_per_cell': 12, 'checks': replay, 'status': 'PASS', 'new_COCO_or_YOLO': False})
    return fn_rows, fp_rows, size_rows, class_metrics


def transitions(out, spec):
    counts = defaultdict(Counter)
    detailed = out/'matched_GT_transitions.csv.gz'
    with gzip.open(detailed, 'xt', encoding='utf-8', newline='') as stream:
        fields = ['condition', 'image_id', 'stem', 'GT_id', 'category_id', 'category_name', 'size', 'area', 'bbox', 'transition', 'CONTROL_IoU', 'GUIDED_IoU', 'CONTROL_confidence', 'GUIDED_confidence']
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for condition in ('Normal', 'Gaussian_s15'):
            files = [out/'diagnostic_records'/(next(c['cell_id'] for c in spec['cells'] if c['strategy'] == s and c['condition'] == condition)+'.jsonl.gz') for s in ('CONTROL', 'VARIABILITY_GUIDED')]
            with gzip.open(files[0], 'rt', encoding='utf-8') as c, gzip.open(files[1], 'rt', encoding='utf-8') as g:
                n = 0
                for left, right in zip(c, g):
                    a, b = json.loads(left), json.loads(right)
                    if a['image_id'] != b['image_id'] or len(a['GT_outcomes']) != len(b['GT_outcomes']):
                        raise ValueError('Paired diagnostic image/GT differs.')
                    n += 1
                    for x, y in zip(a['GT_outcomes'], b['GT_outcomes']):
                        if any(x[k] != y[k] for k in ('GT_id', 'category_id', 'area', 'bbox')):
                            raise ValueError('Paired original GT differs.')
                        label = 'BOTH_DETECT' if x['detected'] and y['detected'] else 'CONTROL_ONLY_LOST' if x['detected'] else 'GUIDED_ONLY_IMPROVED' if y['detected'] else 'BOTH_MISS'
                        for group in groups(x['category_id'], x['area']):
                            counts[condition, *group][label] += 1
                        writer.writerow({'condition': condition, 'image_id': a['image_id'], 'stem': a['stem'], 'GT_id': x['GT_id'], 'category_id': x['category_id'], 'category_name': category_name(x['category_id']),
                                         'size': x['size'], 'area': x['area'], 'bbox': x['bbox'], 'transition': label, 'CONTROL_IoU': x['IoU'], 'GUIDED_IoU': y['IoU'],
                                         'CONTROL_confidence': x['confidence'], 'GUIDED_confidence': y['confidence']})
                if n != 1610 or c.readline() or g.readline():
                    raise ValueError('Incomplete diagnostic pairing.')
    rows = []
    for (condition, cat, area), count in sorted(counts.items()):
        row = {'condition': condition, 'category_id': cat, 'category_name': category_name(cat), 'size': area, **{k: count[k] for k in ('BOTH_DETECT', 'CONTROL_ONLY_LOST', 'GUIDED_ONLY_IMPROVED', 'BOTH_MISS')}}
        row['GT'] = sum(count.values()); row['net_detected_change'] = count['GUIDED_ONLY_IMPROVED']-count['CONTROL_ONLY_LOST']
        rows.append(row)
    csv_write(out/'VM43_object_match_transitions.csv', rows)
    return rows
