"""Original-pixel label mapping and rasterized ignore-union rules."""
from collections import defaultdict
import cv2
import numpy as np

VIS_NAMES = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
             'tricycle', 'awning-tricycle', 'bus', 'motor']

COMMON_NAMES = ['person', 'car', 'bus', 'truck', 'motorcycle', 'bicycle']

VIS_TO_COMMON = {0: 1, 3: 2, 8: 3, 5: 4, 9: 5, 2: 6}

COCO_TO_COMMON = {0: 1, 2: 2, 5: 3, 7: 4, 3: 5, 1: 6}

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

VERSION = 'ieee_v3_original_pixels_strict_common6_ignore_union'

def keep_outside_ignore(boxes, regions, width, height, threshold=0.5):
    """Filter boxes by ignored-region union coverage (default >=50%).

    Uses floor/ceil rasterization and a custom COCO-style protocol."""
    if not boxes:
        return np.zeros(0, dtype=bool)
    if not regions:
        return np.ones(len(boxes), dtype=bool)
    def raster(values):
        v = np.asarray(values, dtype=np.float64).reshape(-1, 4)
        a = np.floor(v[:, :2]).astype(int)
        b = np.ceil(v[:, :2] + v[:, 2:]).astype(int)
        a = np.clip(a, [0, 0], [width, height])
        b = np.clip(b, [0, 0], [width, height])
        return a, b
    mask = np.zeros((height, width), np.uint8)
    left, right = raster(regions)
    for (x1, y1), (x2, y2) in zip(left, right):
        mask[y1:y2, x1:x2] = 1
    integral = cv2.integral(mask, sdepth=cv2.CV_32S)
    left, right = raster(boxes)
    x1, y1 = left.T
    x2, y2 = right.T
    covered = integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
    area = (x2 - x1) * (y2 - y1)
    return (area > 0) & (covered / np.maximum(area, 1) < threshold)

def ground_truth(records, protocol):
    common = protocol == 'common6'
    categories = COMMON_NAMES if common else VIS_NAMES
    gt = {'info': {'protocol': protocol, 'version': VERSION, 'area_reference': 'original_image'},
          'licenses': [], 'images': [], 'annotations': [],
          'categories': [{'id': i+1, 'name': n} for i, n in enumerate(categories)]}
    region_map = {}
    for r in records:
        gt['images'].append({k: r[k] for k in ('id', 'file_name', 'width', 'height')})
        # Unsupported categories are ignored spatially in the matched common subset.
        regions = r['ignore_regions'] + r['other_regions']
        if common:
            regions = regions + [o['bbox'] for o in r['objects'] if o['class_id'] not in VIS_TO_COMMON]
        objects = [o for o in r['objects'] if not common or o['class_id'] in VIS_TO_COMMON]
        keep = keep_outside_ignore([o['bbox'] for o in objects], regions, r['width'], r['height'])
        region_map[r['id']] = regions
        for o, valid in zip(objects, keep):
            if not valid:
                continue
            cat = VIS_TO_COMMON[o['class_id']] if common else o['class_id']+1
            gt['annotations'].append({'id': len(gt['annotations'])+1, 'image_id': r['id'],
                                      'category_id': cat, 'bbox': o['bbox'],
                                      'area': o['bbox'][2]*o['bbox'][3], 'iscrowd': 0})
    return gt, region_map

def mapped_predictions(raw, records, regions, protocol, state, max_det=3000):
    lookup = {r['id']: r for r in records}
    groups = defaultdict(list)
    common = protocol == 'common6'
    mapper = COCO_TO_COMMON if state == 'zero_shot' else VIS_TO_COMMON
    for p in raw:
        cls = int(p['class_id'])
        if common and cls not in mapper:
            continue
        cat = mapper[cls] if common else cls+1
        groups[p['image_id']].append({'image_id': p['image_id'], 'category_id': cat,
                                      'bbox': p['bbox'], 'score': p['score']})
    output = []
    for iid in sorted(groups):
        group = sorted(groups[iid], key=lambda p: p['score'], reverse=True)[:max_det]
        rec = lookup[iid]
        keep = keep_outside_ignore([p['bbox'] for p in group], regions[iid], rec['width'], rec['height'])
        output.extend(p for p, valid in zip(group, keep) if valid)
    return output
