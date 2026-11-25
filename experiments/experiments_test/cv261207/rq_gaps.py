"""Evaluate conditional development gaps without inventing latency or population coverage."""
from collections import Counter
import numpy as np

from common import load, save, sha256, source
from historical_review import V26, csv_file, table, checked
from gap_analysis import pair_outcome, subset_outcome


def joint_intervals(point, draws):
    point=np.asarray(point,dtype=float)
    draws=np.asarray(draws,dtype=float)
    if draws.ndim!=2 or draws.shape[1]!=len(point) or not np.isfinite(draws).all():
        raise ValueError('Invalid paired response matrix.')
    sd=draws.std(axis=0,ddof=1)
    z=np.divide(abs(draws-point),sd,out=np.zeros_like(draws),where=sd>1e-15)
    critical=float(np.quantile(z.max(axis=1),.95))
    return point-critical*sd,point+critical*sd,critical


def analyze(out, review, static):
    proofs=load(review/'provenance.json')
    membership={r['config_id']:r for r in proofs['configurations']}
    sources=load(V26/'results/bootstrap/source_manifest.json')
    points=[]; columns=[]
    for cell in sources:
        cid=cell['config_id']
        complete=load(V26/f'results/bootstrap/{cid}/complete.json')
        values=np.load(checked(V26/f'results/bootstrap/{cid}/draws.npy',complete['SHA256']),allow_pickle=False)
        metric_file=checked(__import__('pathlib').Path(membership[cid]['metric_file']),membership[cid]['metric_sha256'])
        areas={r['size']:r for r in load(metric_file)['per_area']}
        for ai,size in enumerate(('all','tiny','small')):
            points.append({'config_id':cid,'checkpoint':cell['checkpoint'],'blur':cell['blur'],
                  'jpeg':cell['jpeg'],'resolution':cell['resolution'],'size':size,
                  'AP50_95':areas[size]['map50_95'],'AP50':areas[size]['map50'],
                  'point_source_sha256':membership[cid]['metric_sha256'],
                  'draws_sha256':complete['SHA256'],'source_RunID':'vm26_r001'})
            columns.append(values[:,ai,1])
    low,high,critical=joint_intervals([r['AP50_95'] for r in points],np.column_stack(columns))
    for i,row in enumerate(points):
        row.update(ci95_low=float(low[i]),ci95_high=float(high[i]),family=72,resamples=200,
                   latency_ms=None,latency_status='END_TO_END_NOT_AVAILABLE_FOR_THIS_SCOPE',
                   scope='DEVELOPMENT_ONLY_POST_HOC')
    csv_file(out/'endpoint_confidence_intervals.csv',points)
    classified=[]
    for row in points:
        leaves={'Normal'} | ({'Motion_k21'} if row['blur']==2 else set()) | ({'JPEG_q50'} if row['jpeg']=='Q50' else set())
        for tau in (.05,.10,.15):
            lo,hi=row['ci95_low'],row['ci95_high']
            ap_status='FAIL' if hi<tau else 'PASS' if lo>=tau else 'UNCERTAIN'
            for policy,rule in static['definitions'].items():
                classified.append({**row,'tau':tau,'L_ms':50.,'policy':policy,
                    'static_gap':not leaves.issubset(set(rule['covered_leaves'])),
                    'AP_only_state':ap_status,'joint_AP_latency_state':pair_outcome([lo,hi],None,tau,50.),
                    'feature_analogue':'|'.join(sorted(leaves)),
                    'pixel_recipe':'VM26_dv03; feature analogue is not identity with the CV261109 renderer'})
    csv_file(out/'development_pair_states.csv',classified)
    gaps=[];confusion=[]
    for size in ('all','tiny','small'):
        for tau in (.05,.10,.15):
            rows=[r for r in classified if r['size']==size and r['tau']==tau and r['policy']=='exact_identity']
            states=[]
            for b in (0,2):
                for j in ('None','Q50'):
                    cell=[r for r in rows if r['blur']==b and r['jpeg']==j]
                    if len(cell)!=6:
                        raise ValueError('Each context needs six measured detector endpoints.')
                    state=subset_outcome([r['joint_AP_latency_state'] for r in cell])
                    states.append(state)
                    gaps.append({'size':size,'tau':tau,'blur':b,'jpeg':j,'tested_detectors':6,
                                 'subset_state':state,'scope':'DEVELOPMENT_ONLY_FINITE_FOUR_CONTEXT_GRID',
                                 'full_space_gap':'NOT_EVALUATED','sampling_uncertainty':'NOT_ESTIMABLE_FROM_PURPOSIVE_GRID'})
            for policy in static['definitions']:
                rows=[r for r in classified if r['size']==size and r['tau']==tau and r['policy']==policy]
                observed=[r for r in rows if r['joint_AP_latency_state'] in ('PASS','FAIL')]
                counts=Counter((r['static_gap'],r['joint_AP_latency_state']=='FAIL') for r in observed)
                tp,fp,fn,tn=(counts[True,True],counts[True,False],counts[False,True],counts[False,False])
                confusion.append({'size':size,'tau':tau,'policy':policy,'TP':tp,'FP':fp,'FN':fn,'TN':tn,
                      'complete_cases':len(observed),'missing_or_uncertain':len(rows)-len(observed),
                      'sensitivity':tp/(tp+fn) if tp+fn else None,
                      'specificity':tn/(tn+fp) if tn+fp else None,
                      'complete_case_agreement':(tp+tn)/len(observed) if observed else None,
                      'status':'INCOMPLETE_OUTCOME_ASCERTAINMENT_NOT_A_VALIDATED_PREDICTOR',
                      'limitation':'Missing end-to-end latency prevents PASS labels; complete cases are selected failures.'})
    csv_file(out/'development_subset_gaps.csv',gaps)
    csv_file(out/'static_empirical_diagnostics.csv',confusion)
    save(out/'empirical_gap_acceptance.json',{'status':'PARTIAL','scope':'DEVELOPMENT_ONLY',
        'measured_configurations':24,'degradation_contexts':4,'detector_subset':6,'images':300,
        'new_forward':0,'confidence_family':72,'critical':critical,'resamples':200,
        'interval_method':'Max-standardized centered paired bootstrap; post-hoc family of 24 endpoints x3 AP sizes.',
        'original_DID_family_36_unchanged':True,
        'shared_indices':source(V26/'results/bootstrap/indices.npy'),
        'end_to_end_latency':'NOT_AVAILABLE','medium_large_intervals':'NOT_EVALUATED',
        'population_gap':'NOT_EVALUATED','learned_gap':'NOT_EVALUATED','full_space_gap':'NOT_EVALUATED',
        'no_learned_UVL_constraint_added':True,
        'reason':'No independently validated rules. Conditional AP-only failure can establish subset failure, but cannot admit a full-space constraint.'})
    return gaps
