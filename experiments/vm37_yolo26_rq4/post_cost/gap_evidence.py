"""Extract measured development deficits without changing the broad policy."""
import csv
import json
from pathlib import Path
import sys
import numpy as np

BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE))
from study import *


def families(leaves):
    supported=set();outside=set()
    for leaf in leaves:
        if leaf=='Normal':continue
        if leaf.startswith(('Overexposed','Underexposed','LowLight')):supported.add('ExposureAug')
        elif leaf.startswith(('Gauss_','Shot_')):supported.add('NoiseAug')
        elif leaf.startswith('Motion_'):supported.add('MotionBlurAug')
        elif leaf.startswith('JPEG_'):supported.add('JPEGAug')
        elif leaf.startswith('Haze_'):supported.add('HazeAug')
        else:outside.add(leaf)
    return sorted(supported),sorted(outside)


def execute():
    out=create('historical_gap_evidence_cpu');runtime(out,cuda=False)
    source_manifest=PREP/'FROZEN_DEVELOPMENT_EVALUATION.json';accepted=load(source_manifest)['conditions']
    if len(accepted)!=84 or len({r['pair_id'] for r in accepted})!=84:raise ValueError('Expected accepted84 pairs.')
    loaded={};sources=[]
    for row in accepted:
        folder=Path(row['source_folder'])
        for name in ('metrics.json','bootstrap.json','configuration.json'):
            if sha256(folder/name)!=row['source_SHA256'][name]:raise ValueError('Historical evidence changed.')
        loaded[row['pair_id']]={'record':row,'metrics':load(folder/'metrics.json'),'bootstrap':load(folder/'bootstrap.json')}
        sources.append({'pair_id':row['pair_id'],'metrics_SHA256':row['source_SHA256']['metrics.json'],
                        'bootstrap_SHA256':row['source_SHA256']['bootstrap.json'],'GT_SHA256':row['source_SHA256']['ground_truth.json']})
    tau_source=ROOT/'datasets/local_runs/vm36_supervisor_reproduction/vm36_supervisor_reproduction_full20_results_20261110T161658_46e4aed4/FINITE20_GAPS.csv'
    with tau_source.open(encoding='utf-8-sig') as stream:thresholds=sorted({float(r['tau']) for r in csv.DictReader(stream)})
    contrasts=[];states=[]
    for value in loaded.values():
        row=value['record']
        normal=next(v for v in loaded.values() if v['record']['checkpoint']==row['checkpoint'] and
                    v['record']['inference_resolution']==row['inference_resolution'] and v['record']['leaves']==['Normal'])
        base_draws=normal['bootstrap']['draws'];draws=value['bootstrap']['draws']
        paired=all(draws[k]==base_draws[k] for k in ('image_ids','samples','seed','sampling'))
        paired=paired and row['source_SHA256']['ground_truth.json']==normal['record']['source_SHA256']['ground_truth.json']
        cover,uncovered=families(row['leaves'])
        for size in ('all','tiny','small','medium','large'):
            current=next(r for r in value['metrics']['per_area'] if r['size']==size)
            clean=next(r for r in normal['metrics']['per_area'] if r['size']==size)
            for metric in ('map50','map50_95','ar50_95'):
                delta=clean[metric]-current[metric]
                interval=None
                if paired:
                    left=base_draws['values'][size][metric];right=draws['values'][size][metric]
                    if len(left)!=200 or len(right)!=200:raise ValueError('Unexpected saved bootstrap count.')
                    if all(v is not None for v in left+right):interval=np.percentile(np.asarray(left)-np.asarray(right),[2.5,97.5]).tolist()
                contrasts.append({'checkpoint':row['checkpoint'],'resolution':row['inference_resolution'],'pair_id':row['pair_id'],
                    'leaves':'|'.join(row['leaves']),'size':size,'metric':metric,'Normal':clean[metric],'degraded':current[metric],
                    'Normal_minus_degraded':delta,'paired_CI_low':interval[0] if interval else None,'paired_CI_high':interval[1] if interval else None,
                    'paired_draw_metadata_equal':paired,'interval_scope':'Pointwise exploratory image CI; no training-seed or simultaneous coverage.',
                    'covered_families':'|'.join(cover),'outside_ALL_FIVE':'|'.join(uncovered),
                    'metrics_SHA256':row['source_SHA256']['metrics.json'],'bootstrap_SHA256':row['source_SHA256']['bootstrap.json']})
        absolute=next(r for r in value['bootstrap']['intervals'] if r['size']=='all' and r['metric']=='map50_95')['ci95']
        for tau in thresholds:
            state='PASS' if absolute[0]>=tau else 'FAIL' if absolute[1]<tau else 'UNCERTAIN'
            states.append({'checkpoint':row['checkpoint'],'resolution':row['inference_resolution'],'pair_id':row['pair_id'],
                'leaves':'|'.join(row['leaves']),'tau':tau,'state':state,'AP_CI_low':absolute[0],'AP_CI_high':absolute[1],
                'scope':'DEVELOPMENT_ONLY_QUALITY_THRESHOLD; not full quality-and-latency gap.'})
    for name,rows in [('historical_deficits.csv',contrasts),('historical_quality_states.csv',states)]:
        with (out/name).open('x',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=rows[0],lineterminator='\n');writer.writeheader();writer.writerows(rows)
    proposal={'status':'DESIGN_ONLY_NOT_IMPLEMENTED','accepted_policy':'VM37_ALL_FIVE_V1 remains unchanged broad stochastic augmentation.',
        'measured_evidence_source':source(source_manifest),'threshold_source':source(tau_source),'historical_thresholds':thresholds,
        'future_rule':['Define a calibration population separate from final evaluation.',
            'Use original static-gap regions intersected with legally valid FM_P configurations.',
            'Retain measured development failures and matched Normal deficits with explicit detector/resolution identity.',
            'Freeze the quality/latency target and legal-region selection before final responses.',
            'Sample legal regions with a fixed seed and predeclared weights; do not tune p on final evaluation.',
            'Define continuous strength jitter and exact test-value exclusion before training; do not infer physical calibration.',
            'Match initialization, schedule, source-image exposure and measured compute budget.'],
        'required_FMP_constraints':['LowLight=>ShotNoise','Rain=>!Overexposed_strong & !Overexposed_mild',
            'Haze_dense=>!Overexposed_strong','RollingShutter=>MotionBlur','Shot_high=>LowLight | Underexposed_strong'],
        'limits':['Current five independent gates do not enforce these discrete FM_P conditional regions.',
            'Five families do not implement every rain/defocus/downscale/rolling-shutter training capability.',
            'Historical val evidence is exposed development data, not independent final confirmation.',
            'Paired deficit CIs use stored AP bootstrap draws with checked shared seed, image order, sampling convention and GT.',
            'No multiplicity-corrected significance or causal architecture attribution is claimed.'],
        'new_COCO_evaluations':0,'new_GPU_predictions':0,'augmentation_benefit':'NOT_EVALUATED'}
    save(out/'gap_guided_design.json',proposal,out);save(out/'sources.json',sources,out)
    write_atomic(out/'diary.md','# Historical gap evidence\n\nRead accepted84 metrics and saved bootstrap draws. Computed matched Normal deficits only when draw metadata and GT match. Retained original thresholds. This does not evaluate augmentation or change its independent0.5 gates. No COCO replay or inference.\n',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'HISTORICAL_GAP_EVIDENCE_CPU','status':'COMPLETE_WITH_LIMITATIONS','GPU_seconds':0},out,replace=True)
    seal(out);print(out,flush=True)


if __name__=='__main__':execute()
