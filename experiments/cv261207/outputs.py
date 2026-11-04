"""Build checked summaries and dispatch reproducible figure generation."""
from collections import defaultdict
from datetime import datetime, timezone
import csv

from common import PUBLIC, load, new_run, resolve_run, save, seal, sha256, verify_hashes
from evaluation import verify
from gap_analysis import pair_outcome, subset_outcome
from learning import evaluate


def records_for(names):
    records = []
    for name in names:
        folder = resolve_run(name)
        run = verify(folder)
        data = {'run': run, **{stem: load(folder / (stem + '.json'))
                               for stem in ('configuration', 'metrics', 'timing', 'environment')},
                'hashes': verify_hashes(folder, ('run.json', 'metrics.json', 'timing.json'))}
        if run['mode'] == 'FORMAL':
            verify_hashes(folder, ('bootstrap.json',))
            data['bootstrap'] = load(folder / 'bootstrap.json')
        records.append(data)
    return records


def summarize(audit, pairs):
    checks = verify_hashes(audit, ('static_gap.json', 'model_validation.json', 'selection_manifest.json'))
    model = load(audit / 'model_validation.json')
    design = load(audit / 'selection_manifest.json')
    static = load(audit / 'static_gap.json')
    joined = []
    seen = set()
    for pair in pairs:
        run, cfg = pair['run'], pair['configuration']
        if run['RunID'] in seen:
            raise ValueError('Duplicate RunID.')
        seen.add(run['RunID'])
        areas = {r['size']: {'AP50': r['map50'], 'AP50_95': r['map50_95'], 'AR50_95': r['ar50_95']}
                 for r in pair['metrics']['per_area']}
        timing = pair['timing']
        joined.append({'RunID': run['RunID'], 'scope': run['scope'], 'images': run['images'],
                       'p': cfg['degradation']['degradation_id'], 'd': cfg['detector']['detector_id'],
                       'degradation': cfg['degradation']['leaves'], 'detector': cfg['detector']['checkpoint'],
                       'checkpoint_sha256': cfg['model_hash'], 'resolution': cfg['detector']['resolution'],
                       'precision': cfg['detector']['precision'], 'protocol': cfg['protocol'],
                       'source_metrics_sha256': pair['hashes']['metrics.json'],
                       'prediction_sha256': pair['hashes']['predictions.jsonl.gz'],
                       'mean_pipeline_ms': timing['mean_pipeline_ms'],
                       'peak_allocated_bytes': timing['peak_allocated_bytes'],
                       'peak_reserved_bytes': timing['peak_reserved_bytes'],
                       'pair_wall_seconds': timing['pair_wall_seconds'],
                       'compressed_prediction_bytes': (resolve_run(run['RunID']) / 'predictions.jsonl.gz').stat().st_size,
                       'area': areas})
    return {'source_audit_RunID': audit.name, 'source_hashes': {k: checks[k] for k in
            ('model_validation.json', 'selection_manifest.json', 'static_gap.json')},
            'FM_P': {'count': model['FM_P']['valid_count'], 'sha256': model['FM_P']['UVL_SHA256']},
            'FM_D': {k: model['FM_D'][k] for k in ('valid_count','training_variants','UVL_SHA256')},
            'sampling': {k: design[k] for k in ('seed','eligible_FM_P','full_FM_P',
                                                'per_configuration_inclusion_probability','rolling_shutter_excluded')},
            'pairwise_coverage': {k: design['pairwise'][k] for k in ('achieved','feasible','impossible')},
            'planned_pairs': len(design['pairs']), 'static': {k: v['covered'] for k, v in static['definitions'].items()},
            'formal_evaluation': 'DATASET_BLOCKED' if not any(r['scope']=='TEST_DEV' for r in joined) else 'MEASURED_SUBSET',
            'pair_runs': joined}


def empirical(pairs, manifest, static):
    measured = {}
    frozen = {pair['pair_id']: pair for pair in manifest['pairs']}
    for record in pairs:
        if record['run']['mode'] != 'FORMAL':
            continue
        config = record['configuration']
        expected = frozen.get(config['pair_id'])
        if (expected is None or config['degradation'] != expected['degradation']
                or config['detector'] != expected['detector'] or config['images'] != 1610):
            raise ValueError('Formal run is outside the frozen comparison.')
        if config['pair_id'] in measured:
            raise ValueError('Select one formal measurement per pair.')
        measured[config['pair_id']] = record
    if not measured:
        return {'status': 'NOT_EVALUATED', 'reason': 'No verified formal pairs.'}
    split_hashes = {r['configuration']['dataset_sha256'] for r in measured.values()}
    if len(split_hashes) != 1:
        raise ValueError('Mixed dataset source identities.')
    rows = []
    for tau in (.05, .10, .15):
        for limit in (25., 50., 100.):
            groups = defaultdict(list)
            for spec in manifest['pairs']:
                key = spec['pair_id']
                record = measured.get(key)
                status = 'NOT_EVALUATED'
                if record:
                    ci = record['bootstrap']['intervals']['tiny']['ci95']
                    status = pair_outcome(ci, record['timing']['mean_pipeline_ms'], tau, limit)
                groups[spec['degradation']['degradation_id']].append((spec, status, record))
            for pid, group in groups.items():
                leaves = set(group[0][0]['degradation']['leaves'])
                row = {'p': pid, 'tau_tiny_AP': tau, 'pipeline_limit_ms': limit,
                       'tested_subset_status': subset_outcome([s for _, s, _ in group]),
                       'counts': {s: sum(x[1] == s for x in group) for s in ('PASS','FAIL','UNCERTAIN','NOT_EVALUATED')},
                       'static_covered': {k: leaves.issubset(v['covered_leaves']) for k, v in static['definitions'].items()},
                       'pairs': [{'d': spec['detector']['detector_id'], 'status': status,
                                  'RunID': rec['run']['RunID'] if rec else None} for spec, status, rec in group]}
                rows.append(row)
    return {'status': 'MEASURED_SUBSET', 'metric': 'tiny AP50:95',
            'scope': 'Five bound inference configurations; no full FM_D gap inference.', 'rows': rows}


def execute(args):
    if not args.audit:
        raise ValueError('--audit is required.')
    audit = resolve_run(args.audit)
    pairs = records_for(args.run)
    report = summarize(audit, pairs)
    if args.mode == 'publish':
        if any(r['scope'] != 'TEST_DEV' for r in report['pair_runs']):
            raise ValueError('Engineering AP remains local; publish the CPU summary separately.')
        out = PUBLIC / 'reports/results/cv261207' / audit.name
        out.mkdir(parents=True)
        brief = {'RunID': audit.name, 'ExperimentID': 'cv261207', 'status': 'PASS_CPU_AUDIT',
                 'scope': 'DUAL_VARIABILITY_EVIDENCE', 'mode': 'CV261207_DUAL_VARIABILITY', **report}
        bundle = {'RunID': audit.name, 'ExperimentID': 'cv261207', 'mode': brief['mode'],
                  'status': brief['status'], 'files': {'summary.json': brief, 'result.json': brief}}
        save(out / 'snapshot.json', bundle)
        checksum = sha256(out / 'snapshot.json')
        save(out / 'manifest.json', {'schema': 2, 'RunID': audit.name, 'ExperimentID': 'cv261207',
                                     'status': brief['status'], 'mode': brief['mode'],
                                     'started_at': datetime.now(timezone.utc).isoformat(), 'bundle': 'snapshot.json',
                                     'SHA256': checksum, 'files': {'snapshot.json': checksum}})
    else:
        out = new_run(args.mode)
        static = load(audit / 'static_gap.json')
        save(out / 'summary.json', report)
        save(out / 'empirical_gap.json', empirical(pairs, load(audit / 'selection_manifest.json'), static))
        save(out / 'learning.json', evaluate(pairs))
        if args.mode == 'figures':
            from figures import render
            render(out, audit, report)
        save(out / 'run.json', {'RunID': out.name, 'ExperimentID': 'cv261207', 'status': 'PASS',
                                'mode': args.mode, 'scope': 'EVIDENCE_REVIEW'})
        seal(out)
    print(out)
