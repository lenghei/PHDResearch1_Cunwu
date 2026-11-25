"""Conservative measured-evidence selector; no image labels or AP predictor."""
import hashlib
import json
import math
from pathlib import Path


def configuration_id(leaves):
    payload = json.dumps(sorted(leaves), separators=(',', ':'), ensure_ascii=True)
    return 'p_'+hashlib.sha256(payload.encode()).hexdigest()[:16]


def satisfiable(clauses):
    clauses = [tuple(c) for c in clauses]
    while True:
        if not clauses:
            return True
        if any(not c for c in clauses):
            return False
        units = {c[0] for c in clauses if len(c) == 1}
        if not units:
            literal = clauses[0][0]
            return satisfiable(clauses+[(literal,)]) or satisfiable(clauses+[(-literal,)])
        if any(-x in units for x in units):
            return False
        clauses = [tuple(x for x in c if -x not in units) for c in clauses if not any(x in units for x in c)]


def legal(model, leaves):
    chosen = list(leaves)
    if len(chosen) != len(set(chosen)) or not set(chosen).issubset(model['leaves']):
        return False
    assumptions = [[model['variables'][f] if f in chosen else -model['variables'][f]] for f in model['leaves']]
    return satisfiable(model['clauses']+assumptions)


def state(row, quality=.10, latency=100):
    values = [row[k] for k in ('Tiny_CI_low', 'Tiny_CI_high', 'latency_CI_low_ms', 'latency_CI_high_ms')]
    if any(x is None for x in values):
        return 'NOT_ESTIMABLE'
    q0, q1, l0, l1 = values
    if l1 > latency or q1 < quality:
        return 'FAIL'
    return 'PASS' if q0 >= quality else 'UNCERTAIN'


class Controller:
    def __init__(self, cnf, rows, candidates, semantic_support):
        if any(r.get('evidence_scope') != 'DEVELOPMENT_ONLY' or 'vm36_supervisor_reproduction' not in r.get('source_folder', '') for r in rows):
            raise ValueError('Controller accepts only source-bound VM36 development evidence.')
        keys = [(r['condition_id'], r['D']) for r in rows]
        if len(keys) != len(set(keys)):
            raise ValueError('Duplicate measured P-D evidence.')
        self.cnf, self.rows, self.candidates, self.support = cnf, rows, candidates, semantic_support
        self.available = set()
        checked = {}
        for name, value in candidates.items():
            ck = value['checkpoint']; path = Path(ck['path'])
            identity = (str(path), ck['sha256'])
            if identity not in checked:
                h = hashlib.sha256()
                if path.is_file():
                    with path.open('rb') as f:
                        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
                    checked[identity] = h.hexdigest() == ck['sha256']
                else: checked[identity] = False
            if checked[identity]: self.available.add(name)

    def select(self, leaves, quality=.10, latency=100, eligible=None, strict_training=False):
        if not 0 <= quality <= 1 or not math.isfinite(latency) or latency <= 0:
            raise ValueError('Invalid task threshold.')
        trace = []
        def result(status, reason, selected=None):
            return {'P': configuration_id(leaves), 'leaves': sorted(leaves), 'status': status, 'reason': reason,
                    'selected': selected, 'quality_threshold': quality, 'latency_budget_ms': latency, 'trace': trace,
                    'information': 'DEVELOPMENT_ONLY_MEASURED_LOOKUP', 'degradation_recognition': 'EXTERNALLY_SPECIFIED'}
        if not legal(self.cnf['fm_p'], leaves):
            return result('INVALID_FM_P_CONFIGURATION', 'Original UVL CNF rejects the full leaf assignment.')
        trace.append({'step': 1, 'UVL': 'SAT_VALID'})
        pid = configuration_id(leaves)
        semantics = self.support.get(pid)
        trace.append({'step': 2, 'image_GT_semantics': semantics or 'UNSUPPORTED_UNMEASURED'})
        measured = [r for r in self.rows if r['condition_id'] == pid]
        if not semantics or not measured:
            return result('UNKNOWN_UNMEASURED_CONDITION', 'No admitted measured image/GT condition; no AP interpolation.')
        trace.append({'step': 3, 'measured_pairs': len(measured)})
        allowed = set(self.candidates) if eligible is None else set(eligible)
        supported = {k for k, v in self.candidates.items() if k in allowed and k in self.available and v['eligible_historical_checkpoint'] and legal(self.cnf['fm_d'], v['declared_leaves']) and (not strict_training or v['full_training_feature_provenance'])}
        trace.append({'step': 4, 'eligible_D': sorted(supported), 'strict_training': strict_training})
        if not supported:
            return result('NO_CONFIRMED_FEASIBLE_DETECTOR', 'NO_TRAINING_AVAILABLE_OR_UNSUPPORTED_D: no eligible measured trained checkpoint.')
        tested = [r for r in measured if r['D'] in supported]
        assessed = [{**r, 'task_state': state(r, quality, latency)} for r in tested]
        trace.append({'step': 5, 'states': {r['D']: r['task_state'] for r in assessed}})
        passing = [r for r in assessed if r['task_state'] == 'PASS']
        if not passing:
            return result('NO_CONFIRMED_FEASIBLE_DETECTOR', 'No eligible D passes frozen Tiny/latency interval criteria.')
        choice = min(passing, key=lambda r: (r['cached_pipeline_mean_ms'], -r['Tiny_CI_low'], r['D']))
        trace.append({'step': 6, 'ranking': 'Lowest measured mean latency; then highest Tiny lower CI; then D ID.'})
        return result('SELECTED_MEASURED_DETECTOR', 'Interval-feasible historical checkpoint; provenance limitations retained.', choice)


def disjoint_groups(train, test):
    a, b = {r['condition_id'] for r in train}, {r['condition_id'] for r in test}
    if a & b:
        raise ValueError('P configuration leaks across folds.')
    return True
