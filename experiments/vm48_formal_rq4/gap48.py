"""Frozen measured-subset Gap endpoint and paired image uncertainty."""
import numpy as np


def state(tiny,latency):
    if any(v is None for v in [*tiny,*latency]):return 'NOT_EVALUATED'
    if any(not np.isfinite([lo, hi]).all() or not lower <= lo <= hi <= upper
           for (lo, hi), lower, upper in ((tiny, 0, 1), (latency, 0, np.inf))):
        raise ValueError('Invalid Gap confidence interval.')
    if tiny[1]<.10 or latency[0]>100:return 'FAIL'
    return 'PASS' if tiny[0]>=.10 and latency[1]<=100 else 'UNCERTAIN'


def p_state(states):
    if not states or 'NOT_EVALUATED' in states:return 'NOT_EVALUATED'
    if 'PASS' in states:return 'COVERED'
    return 'GAP' if all(s=='FAIL' for s in states) else 'UNCERTAIN'


def count(states):
    if any(s not in ('COVERED','GAP','UNCERTAIN') for s in states):raise ValueError('Incomplete P set.')
    g=states.count('GAP');u=states.count('UNCERTAIN')
    return {'P':len(states),'COVERED':states.count('COVERED'),'GAP':g,'UNCERTAIN':u,'gap_count_bounds':[g,g+u]}


def paired_gap(control,target):
    c=count(control);t=count(target)
    if c['P']!=t['P']:raise ValueError('P populations differ.')
    return {'CONTROL':c,'TARGETED':t,'known_Gap_count_reduction':c['GAP']-t['GAP'],
       'three_state_reduction_bounds':[c['gap_count_bounds'][0]-t['gap_count_bounds'][1],c['gap_count_bounds'][1]-t['gap_count_bounds'][0]],
       'scope':'Frozen purposeful P sample only; no population prevalence estimator'}


def ablation_gap(control, target, contexts, resolutions):
    expected = {(p, resolution) for p in contexts for resolution in resolutions}
    if control.keys() != expected or target.keys() != expected:
        raise ValueError('Ablation requires matched CONTROL and treatment cells.')

    rows = []
    for p in contexts:
        baseline = {r: control[p, r] for r in resolutions}
        treatment = {r: target[p, r] for r in resolutions}
        rows.append({
            'P_id': p, 'CONTROL_by_resolution': baseline, 'ABLATION_by_resolution': treatment,
            'CONTROL': p_state(list(baseline.values())),
            'ABLATION': p_state(list(treatment.values())),
        })
    summary = paired_gap([r['CONTROL'] for r in rows], [r['ABLATION'] for r in rows])
    return {
        **summary, 'seed': 42, 'architecture': 'YOLOv8s', 'resolutions': resolutions,
        'confirmed_closure_P_ids': [r['P_id'] for r in rows
                                    if (r['CONTROL'], r['ABLATION']) == ('GAP', 'COVERED')],
        'per_P': rows,
    }


def bootstrap_gap_draws(cells,contexts,seeds,strategies=('CONTROL','ALL_FIVE_TARGETED'),samples=200):
    output=[]
    for seed in seeds:
        counts={s:np.zeros(samples,dtype=int) for s in strategies}
        for s in strategies:
            for p in contexts:
                candidates=[v for key,v in cells.items() if key[1]==seed and key[2]==s and key[4]==p]
                if len(candidates)!=4:raise ValueError('Both architectures and resolutions are required per seed.')
                if any(not np.isfinite(np.asarray(v[k],dtype=float)).all() for v in candidates for k in ('tiny_draws','latency_draws')):raise ValueError('Nonfinite or missing Gap draws.')
                passes=np.array([(np.asarray(v['tiny_draws'])>=.10)&(np.asarray(v['latency_draws'])<=100) for v in candidates])
                if passes.shape!=(4,samples):raise ValueError('Paired draw dimensions differ.')
                counts[s]+=~passes.any(axis=0)
        reduction=counts[strategies[0]]-counts[strategies[1]]
        output.append({'seed':seed,'reduction_draws':reduction.tolist(),'CI95_count':np.percentile(reduction,[2.5,97.5]).tolist(),
          'unit':'Paired pooled-AP image resampling, conditional on fixed training seed','P':len(contexts)})
    return output
