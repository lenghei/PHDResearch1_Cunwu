"""CPU replay of separately identified historical measurement tables."""
import csv
import itertools
import json
import math
import os
import time
import datetime
from collections import defaultdict
from pathlib import Path

from .config import save, sha256
from .pipeline import new_run

EXPERIMENT = 'vm2_degradation_gap_analysis'


def number(value):
    if value is None or value == '':
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def empirical_gap(ap, latency, min_ap=None, max_latency=None):
    tests = []
    if min_ap is not None:
        tests.append(None if ap is None else ap >= min_ap)
    if max_latency is not None:
        tests.append(None if latency is None else latency <= max_latency)
    if not tests:
        return 'NOT_EVALUATED'
    if False in tests:
        return 'FAIL'
    if None in tests:
        return 'UNCERTAIN'
    return 'PASS'


def static_gap(row, rules):
    if not rules:
        return 'NOT_EVALUATED'
    uncertain = False
    for rule in rules:
        if row[rule['factor']] not in rule['covered_levels']:
            if rule['exhaustive']:
                return 'FAIL'
            uncertain = True
    required = {key for key in ['resolution', 'illumination', 'blur', 'noise', 'jpeg'] if key in row}
    documented = {rule['factor'] for rule in rules}
    return 'UNCERTAIN' if uncertain or not required.issubset(documented) else 'PASS'


def finite_difference(rows, factors, levels):
    keys = {tuple(row[f] for f in factors): row['AP50_95'] for row in rows}
    required = list(itertools.product(*levels))
    if any(key not in keys or keys[key] is None for key in required):
        return {'status': 'NOT_EVALUATED', 'reason': 'NOT_ESTIMABLE: required corners missing', 'required': required}
    effect = 0.
    for bits in itertools.product([0, 1], repeat=len(factors)):
        key = tuple(levels[j][bit] for j, bit in enumerate(bits))
        effect += (-1 if (len(factors) - sum(bits)) % 2 else 1) * keys[key]
    return {'status': 'PASS', 'estimate': effect, 'required_corners': len(required),
            'interpretation': 'DESCRIPTIVE_FINITE_GRID_CONTRAST; not a significance test'}


def verify_measurement(row, measured):
    if 'per_area' in measured:
        areas = {item['size']: item for item in measured['per_area']}
        values = {'AP50': areas['all']['map50'], 'AP50_95': areas['all']['map50_95'],
                  'AR': areas['all']['ar50_95']}
        for field, area in [('Tiny_AP', 'tiny'), ('Small_AP', 'small'),
                            ('Medium_AP', 'medium'), ('Large_AP', 'large')]:
            values[field] = areas[area]['map50_95']
    else:
        values = measured
    for key in ['AP50', 'AP50_95', 'AR', 'Tiny_AP', 'Small_AP', 'Medium_AP', 'Large_AP']:
        expected, actual = number(row[key]), number(values[key])
        if (expected is None) != (actual is None) or (
                expected is not None and not math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError('Historical metric extraction differs: ' + key)


def replay(config_file):
    import yaml
    started = datetime.datetime.now(datetime.timezone.utc)
    clock = time.perf_counter()
    cfg = yaml.safe_load(Path(config_file).read_text(encoding='utf-8'))
    for key in ['history_root', 'output_root']:
        value = os.path.expandvars(cfg[key])
        if '$' in value:
            raise ValueError('Set environment variables for ' + key)
        cfg[key] = str(Path(value).resolve())
    out, logger = new_run(cfg, EXPERIMENT, 'replay')
    logger.info('Started: %s', started.astimezone().isoformat(timespec='seconds'))
    logger.info('Mode: CPU historical replay | no YOLO inference or training')
    history = Path(cfg['history_root'])
    metric_index = None
    if cfg.get('source_index'):
        index_file = Path(os.path.expandvars(cfg['source_index'])).resolve()
        if sha256(index_file) != cfg['source_index_SHA256']:
            raise ValueError('Local research source index SHA mismatch.')
        metric_index = json.loads(index_file.read_text(encoding='utf-8'))['files']
    groups, evidence, summaries = [], [], []
    try:
        for item in cfg['sources']:
            source = history / item['path']
            if sha256(source) != item['SHA256']:
                raise ValueError('Historical source SHA mismatch: ' + item['path'])
            with source.open(encoding='utf-8-sig', newline='') as stream:
                rows = list(csv.DictReader(stream))
            evidence.append({'old_run_id': item['run_id'], 'component': item['kind'],
                             'relative_source': item['path'], 'SHA256': item['SHA256'], 'rows': len(rows)})
            logger.info('%s: %s rows', item['run_id'], len(rows))
            if item['kind'] in ['grid', 'jpeg']:
                clean = []
                for row in rows:
                    original_path = row.get('source_result_file') or row['source_metric']
                    digest = row.get('source_result_SHA256') or row['source_metric_SHA256']
                    if metric_index is None:
                        metric_file = Path(original_path)
                    else:
                        indexed = metric_index[original_path]
                        metric_file = (history / indexed['relative_path']).resolve()
                        if not metric_file.is_relative_to(history) or indexed['SHA256'] != digest:
                            raise ValueError('Local research metric identity mismatch.')
                    if sha256(metric_file) != digest:
                        raise ValueError('Underlying measured metric SHA mismatch.')
                    measured = json.loads(metric_file.read_text(encoding='utf-8'))
                    verify_measurement(row, measured)
                    count = int(row['image_count'])
                    cuda_time = measured.get('CUDA_event_seconds')
                    clean.append({'cohort': item['cohort'], 'measurement_kind': item['kind'], 'checkpoint': row['CheckpointID'],
                                  'configuration_id': row['config_id'], 'resolution': int(row['resolution']),
                                  'illumination': int(row['illumination']), 'blur': int(row['blur']),
                                  'noise': int(row['noise']), 'jpeg': row.get('jpeg', 'None'),
                                  'AP50': number(row['AP50']), 'AP50_95': number(row['AP50_95']),
                                  'Tiny_AP': number(row['Tiny_AP']), 'Small_AP': number(row['Small_AP']),
                                  'Medium_AP': number(row['Medium_AP']), 'Large_AP': number(row['Large_AP']),
                                  'AR': number(row['AR']), 'images': count,
                                  'gpu_event_seconds_per_image': cuda_time / count if cuda_time is not None else None,
                                  'metric_source': metric_file.relative_to(history).as_posix(), 'metric_source_SHA256': digest})
                groups.append((item['kind'], clean))
            else:
                # Preserve method/scale summaries separately; never pool their metrics with detector grids.
                safe_rows = [{k: v for k, v in row.items() if not any(s in k.lower() for s in ['source', 'path'])} for row in rows]
                save(out / (item['kind'] + '_historical_summary.json'), safe_rows)
                summaries.append({'component': item['kind'], 'rows': len(rows), 'scope': item['scope']})
        effects = []
        gap_rows = []
        for kind, rows in groups:
            identities = [(r['checkpoint'], r['configuration_id']) for r in rows]
            if len(identities) != len(set(identities)):
                raise ValueError('Duplicate configurations within one historical cohort.')
            rules = cfg.get('training_coverage_rules', [])
            for rule in rules:
                required = ['factor', 'covered_levels', 'exhaustive', 'evidence_path', 'evidence_SHA256', 'checkpoint_id']
                if any(key not in rule for key in required) or sha256(history / rule['evidence_path']) != rule['evidence_SHA256']:
                    raise ValueError('Training coverage rule lacks verified evidence.')
            for row in rows:
                selected_rules = [r for r in rules if r['checkpoint_id'] == row['checkpoint']]
                thresholds = cfg.get('thresholds', {})
                gap_rows.append({**row, 'static_status': static_gap(row, selected_rules),
                                 'empirical_status': empirical_gap(row['AP50_95'], row['gpu_event_seconds_per_image'],
                                                                   thresholds.get('min_AP50_95'),
                                                                   thresholds.get('max_gpu_event_seconds_per_image'))})
            if kind == 'grid':
                for factor in ['blur', 'illumination', 'noise']:
                    context = [key for key in ['checkpoint', 'illumination', 'blur', 'noise'] if key != factor]
                    grouped = defaultdict(list)
                    for row in rows:
                        grouped[tuple(row[key] for key in context)].append(row)
                    for key, subset in grouped.items():
                        value = finite_difference(subset, ['resolution', factor], [[640, 1280], [0, 2]])
                        effects.append({'cohort': subset[0]['cohort'], 'relation': 'Resolution x ' + factor,
                                        'context': dict(zip(context, key)), **value})
            else:
                for checkpoint in sorted({r['checkpoint'] for r in rows}):
                    subset = [r for r in rows if r['checkpoint'] == checkpoint]
                    value = finite_difference(subset, ['resolution', 'blur', 'jpeg'], [[640, 1280], [0, 2], ['None', 'Q50']])
                    effects.append({'cohort': subset[0]['cohort'], 'relation': 'Resolution x Blur x JPEG',
                                    'checkpoint': checkpoint, **value})
        save(out / 'measured_grids.json', gap_rows)
        with (out / 'configuration_metrics.csv').open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(gap_rows[0]))
            writer.writeheader()
            writer.writerows(gap_rows)
        save(out / 'interaction_replay.json', effects)
        save(out / 'provenance.json', evidence)
        result = {'status': 'PASS', 'admission': 'PARTIAL', 'scope': 'HISTORICAL_DEVELOPMENT_REPLAY',
                  'VM1_new_measurements': 'NOT_INCLUDED', 'sources': len(evidence),
                  'measurement_rows': len(gap_rows), 'separate_summaries': summaries,
                  'static_gap': 'READY' if cfg.get('training_coverage_rules') else 'NOT_READY: no verified exhaustive coverage rules',
                  'empirical_gap': 'READY' if any(v is not None for v in cfg.get('thresholds', {}).values()) else 'NOT_EVALUATED: task thresholds unset',
                  'rule_learning': 'FROZEN_HISTORICAL_SUMMARIES_ONLY; no new model fitted',
                  'cross_cohort_pooling': False, 'GPU_inference': 'NOT_RUN'}
        save(out / 'result.json', result)
        save(out / 'run.json', {'RunID': out.name, 'ExperimentID': EXPERIMENT, 'status': 'PASS', 'mode': 'replay'})
        from .visualization import create_interaction_preview, update_run_index
        page = create_interaction_preview(out)
        logger.info('Preview: %s', page.as_uri())
        logger.info('Experiment results: %s', update_run_index(out).as_uri())
        finished = datetime.datetime.now(datetime.timezone.utc)
        timing = {'started_at_UTC': started.isoformat(), 'finished_at_UTC': finished.isoformat(),
                  'wall_seconds': time.perf_counter() - clock}
        save(out / 'timing.json', timing)
        result['timing'] = timing
        save(out / 'result.json', result)
        logger.info('Summary: %s verified sources | %s configuration records | %s/%s estimable contrasts',
                    len(evidence), len(gap_rows), sum(e['status'] == 'PASS' for e in effects), len(effects))
        logger.info('Scientific gap admission: %s', result['admission'])
        logger.info('Finished: %s', finished.astimezone().isoformat(timespec='seconds'))
        logger.info('Total wall time: %.2f s', timing['wall_seconds'])
        logger.info('PASS: %s', out)
        return out
    except Exception as exc:
        save(out / 'failure.json', {'status': 'FAILED', 'error': repr(exc),
                                   'started_at_UTC': started.isoformat(),
                                   'stopped_at_UTC': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                   'wall_seconds': time.perf_counter() - clock})
        save(out / 'run.json', {'RunID': out.name, 'ExperimentID': EXPERIMENT, 'status': 'FAILED', 'mode': 'replay'})
        try:
            from .visualization import build_run_index
            build_run_index(out.parent, EXPERIMENT)
        except (OSError, ValueError) as index_error:
            logger.warning('Result index update failed: %s', index_error)
        raise
