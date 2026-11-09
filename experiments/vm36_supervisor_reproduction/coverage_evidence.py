"""Expand the verified binary pairwise proof without selecting new GPU work."""
from itertools import combinations
import json
from protocol import load,source,FMP,FMD
from historical_review import csv_file,table
from common import verify_hashes


def export(out,preparation):
    verify_hashes(preparation,('sampling_verification.json','SAMPLING_MANIFEST.csv','FEATURE_MODEL_VALIDATION.json'))
    proof=load(preparation/'sampling_verification.json')
    from sampling import models
    fmp,_,prior=models()
    population=prior.enumerate_configs(fmp)
    masks,feasible=prior.compute_binary_pair_masks(population,sorted(fmp.ALL_LEAVES))
    core=[r for r in table(preparation/'SAMPLING_MANIFEST.csv')
          if r['cohort']=='historical_pairwise' and r['role']=='greedy_core']
    selected=[json.loads(r['selected_leaves']) for r in core]
    core_masks,_=prior.compute_binary_pair_masks(selected,sorted(fmp.ALL_LEAVES))
    coverage=0
    for mask in core_masks:coverage|=mask
    if len(core)!=21 or coverage!=feasible or feasible.bit_count()!=1163:
        raise ValueError('The saved core does not cover the legal state universe.')
    csv_file(out/'PAIRWISE_CORE.csv',core)
    states=[]
    for k,(left,right) in enumerate(combinations(sorted(fmp.ALL_LEAVES),2)):
        for a,b in ((0,0),(0,1),(1,0),(1,1)):
            bit=1<<(4*k+2*a+b)
            witnesses=[row['configuration_id'] for row,mask in zip(core,core_masks) if mask&bit]
            states.append({'left':left,'right':right,'left_state':a,'right_state':b,
                'feasible':bool(feasible&bit),'core_covered':bool(coverage&bit),
                'core_configuration_ids':'|'.join(witnesses)})
    csv_file(out/'PAIRWISE_STATES.csv',states)
    return {'source':source(preparation/'sampling_verification.json'),
        'aligned_FMP':source(FMP),'aligned_FMD':source(FMD),'core_size':len(core),
        'feasible_states':feasible.bit_count(),'infeasible_states':len(states)-feasible.bit_count(),
        'scope':'Binary FM_P presence/absence only; no full FM_P x FM_D coverage claim.',
        'additional_59_required_for_this_objective':False,
        'new_GPU_evaluations':0}
