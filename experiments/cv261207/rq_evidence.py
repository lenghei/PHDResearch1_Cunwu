"""Continue accepted RQ evidence; blocked branches do not authorize inference or training."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from common import (ROOT,PUBLIC,HERE,RUNS,FMP,FMD,FLAMA,load,save,source,sha256,
                    new_run,seal,resolve_run,verify_hashes)
from historical_review import csv_file,table
from evaluation import runtime,verify
from rq_models import rq1,checkpoint_static,freeze_v2
from rq_learning import prepare as learning_prepare
from rq_gaps import analyze as gap_analyze
from rq4_protocol import prepare as augmentation_prepare

AUDIT='cv261207_audit_20261104T193649_d1b63492'
ADMISSION='cv261207_admission_20261104T211249_32d83def'
REVIEW='cv261207_vm26review_20261106T212756_fdf80ed8'
HISTORY='cv261207_historyreview_20261106T212926_539e1709'
ENGINEERING='cv261207_acceptance_20261104T194630_793d1f93'


def checked_inputs(out):
    paths={key:resolve_run(name) for key,name in (
          ('audit',AUDIT),('admission',ADMISSION),('review',REVIEW),('history',HISTORY),('engineering',ENGINEERING))}
    evidence=[]
    for key,folder in paths.items():
        verify_hashes(folder,('run.json',))
        evidence.append(source(folder/'SHA256SUMS.txt'))
    if load(paths['admission']/'admission.json')['status']!='DATASET_BLOCKED':
        raise ValueError('Expected the current blocked admission state.')
    documents=[]
    prior=load(paths['audit']/'input_audit.json')
    for record in prior['sources']:
        if record['name'].lower().endswith(('.pdf','.jpg')):
            p=Path(record['path'])
            if sha256(p)!=record['sha256']:
                raise ValueError('Supervisor source changed.')
            documents.append(source(p))
    # The meeting digest is a secondary account; the original PDF controls this audit.
    parent=Path(documents[0]['path']).parent
    notes=parent/'recording2026年10月26日_artifact.md'
    if notes.is_file():
        notes.read_text(encoding='utf-8');documents.append(source(notes))
    for exp in ('cv261109','cv261116','cv261123','cv261207'):
        for p in sorted((HERE.parent/exp).glob('*.md')):
            p.read_text(encoding='utf-8-sig');documents.append(source(p))
    for p in sorted((ROOT/'datasets/reportcn').glob('cv261*.md')):
        p.read_text(encoding='utf-8-sig');documents.append(source(p))
    p=ROOT/'datasets/reportcn/project_历史模型与实验结果索引.md'
    p.read_text(encoding='utf-8');documents.append(source(p))
    save(out/'inputs.json',{'accepted_runs':evidence,'reviewed_documents':documents,
                          'source_code':[source(p) for p in sorted(HERE.glob('*.py'))]})
    return paths


def command(out,name,args,timeout):
    started=time.perf_counter()
    with (out/(name+'.log')).open('x',encoding='utf-8') as stream:
        result=subprocess.run(args,stdout=stream,stderr=subprocess.STDOUT,timeout=timeout,
                              cwd=ROOT,env={**__import__('os').environ,'PYTHONDONTWRITEBYTECODE':'1','PYTHONIOENCODING':'utf-8'})
    return {'command':args,'exit_code':result.returncode,'wall_seconds':time.perf_counter()-started,
            'log_sha256':sha256(out/(name+'.log'))}


def engineering_evidence(out,paths,live):
    from evaluation import evaluate_saved
    costs=load(paths['engineering']/'execution_budget.json')
    names=list(costs['all_verified_engineering_runs'])
    new=[];demo={'status':'REUSED_SAVED_ENGINEERING_EVIDENCE','new_GPU_worker_seconds':0.,'new_GPU_forward_seconds':0.}
    if live:
        before={p.name for p in RUNS.glob('cv261207_smoke_*')}
        receipt=command(out,'live_prediction_demo',[sys.executable,'-B',str(HERE/'run.py'),'demo',
                  '--audit',AUDIT,'--seed','1031','--budget-seconds','120'],150)
        save(out/'demo_supervisor.json',receipt)
        if receipt['exit_code']==0:
            new=[p.name for p in RUNS.glob('cv261207_smoke_*') if p.name not in before]
            if len(new)!=1:raise ValueError('Expected one new engineering prediction run.')
            verify(resolve_run(new[0]))
            names.extend(new)
            timing=load(resolve_run(new[0])/'timing.json')
            demo={'status':'REAL_DEVELOPMENT_DEMO_COMPLETED','new_runs':new,
                  'new_GPU_worker_seconds':receipt['wall_seconds'],
                  'new_GPU_forward_seconds':sum(r['network_forward_ms'] for r in timing['samples'])/1000,
                  'scope':'ONE_IMAGE_ENGINEERING_ONLY','not_GPU_occupancy_measurement':True}
        else:
            demo={'status':'DEMO_BLOCKED_NO_RETRY','new_GPU_worker_seconds':receipt['wall_seconds'],
                  'new_GPU_forward_seconds':None,'reason':'Inspect bounded worker log; no further GPU command is issued.'}
    rows=[]
    for name in names:
        folder=resolve_run(name);record=verify(folder)
        cfg=load(folder/'configuration.json');metric=load(folder/'metrics.json');timing=load(folder/'timing.json')
        all_area=next(r for r in metric['per_area'] if r['size']=='all')
        rows.append({'RunID':name,'scope':record['scope'],'images':cfg['images'],
             'pair_id':cfg['pair_id'],'resolution':cfg['detector']['resolution'],'precision':cfg['detector']['precision'],
             'AP50_95':all_area['map50_95'],'AP50':all_area['map50'],
             'mean_pipeline_ms':timing['mean_pipeline_ms'],'peak_allocated_bytes':timing['peak_allocated_bytes'],
             'metric_sha256':sha256(folder/'metrics.json'),'timing_sha256':sha256(folder/'timing.json')})
    # Replay saved evidence, not a second detector forward.
    chosen=resolve_run(new[0] if new else names[0])
    started=time.perf_counter()
    regenerated,_=evaluate_saved(chosen)
    if regenerated!=load(chosen/'metrics.json'):
        raise ValueError('Saved raw prediction replay differs.')
    save(out/'saved_metric_replay.json',{'status':'PASS','RunID':chosen.name,
           'metric_sha256':sha256(chosen/'metrics.json'),'seconds':time.perf_counter()-started})
    csv_file(out/'engineering_pairs.csv',rows)
    save(out/'execution_cost.json',{'historical_accepted_cost':costs,'this_run':demo,
          'full_pair_budget_calibrated':False,'large_batches_started':False,
          'engineering_rows':len(rows),'engineering_image_inferences':sum(r['images'] for r in rows),
          'distinct_engineering_pairs':len({r['pair_id'] for r in rows})})
    return rows,demo


def report(out,paths,engineering,demo):
    info=load(out/'rq1_validation.json');learn=load(out/'grouped_learning_acceptance.json')
    gaps=table(out/'development_subset_gaps.csv')
    subset=[r for r in gaps if r['size']=='tiny' and float(r['tau'])==.1]
    gap_count=sum(r['subset_state']=='GAP_TESTED_SUBSET' for r in subset)
    static=table(out/'static_coverage_by_checkpoint.csv')
    union=load(out/'historical_response_union.json')
    rq=[{'RQ':'RQ1','question':'Are the variability models correct and executable?',
         'status':'PARTIAL','tested_hypothesis':'Preserved UVL logic, counts, deterministic generator and declared feature bindings agree.',
         'result':'Logical models validated; physical calibration and full detector executability remain incomplete.',
         'configurations':'46368 FM_P; 635904 FM_D; 844 training variants',
         'scope':'EXACT_LOGIC_AND_IMPLEMENTED_SUBSET','blocker':'Unsupported features and uncalibrated scenario constraints.'},
        {'RQ':'RQ2','question':'Which detector and degradation features interact?',
         'status':'PARTIAL','tested_hypothesis':'Resolution gain depends on blur and additional JPEG under the fixed measured grid.',
         'result':f'{union["unique_responses"]} unique historical responses; verified VM26 three-way effects; descriptive tree/stump/forest fits only.',
         'configurations':'243 VM09 +54 VM26 -18 reused =279',
         'scope':'DEVELOPMENT_ONLY','blocker':'Shared image population; no image/configuration-disjoint held-out scores.'},
        {'RQ':'RQ3','question':'How large are static and empirical gaps?',
         'status':'PARTIAL','tested_hypothesis':'Static coverage is distinguishable from measured AP/latency failure.',
         'result':f'{gap_count}/4 selected VM26 contexts have tiny-AP-supported failure across six measured endpoints at tau=0.10. Other contexts are unresolved.',
         'configurations':'24 endpoints x3 size bins; 4 purposive contexts; 6 detector settings',
         'scope':'STATIC_POLICIES_AND_DEVELOPMENT_SUBSET_BOUND','blocker':'No compatible complete latency, random population sample or independently validated gap predictor.'},
        {'RQ':'RQ4','question':'Does targeted augmentation close a measured gap?',
         'status':'NOT_EVALUATED','tested_hypothesis':'Matched clean-control versus targeted continuation improves the frozen gap without unacceptable clean loss.',
         'result':'Versioned matched protocol and CPU validator prepared; no training or gap-closure measurement.',
         'configurations':'2 candidate architectures x6 candidate arms; all planned, zero executed',
         'scope':'PROTOCOL_ONLY','blocker':'Representative baseline gap, independent admission, augmentation implementation/health canary and measured training budget.'}]
    csv_file(out/'rq_acceptance.csv',rq)
    accepted={'status':'PARTIAL','RunID':out.name,'ExperimentID':'cv261207','mode':'RQ_EVIDENCE_ACCEPTANCE',
          'scope':'LOGIC_PLUS_DEVELOPMENT_AND_ENGINEERING','research_questions':rq,
          'formal_measured_pairs':0,'independent_test_status':'DATASET_BLOCKED',
          'historical_unique_responses':279,'historical_VM26_configurations':54,
          'development_image_population':300,'engineering_runs':len(engineering),
          'this_run_GPU_worker_seconds':demo['new_GPU_worker_seconds'],
          'this_run_GPU_forward_seconds':demo['new_GPU_forward_seconds'],
          'GPU_training_seconds':0,'C1':'PARTIAL_FULL_SCOPE; VALIDATED_LOGIC',
          'C2':'PARTIAL_DEVELOPMENT','C3':'PARTIAL_STATIC_AND_SUBSET_BOUND','C4':'NOT_EVALUATED',
          'held_out_metrics_status':learn['status'],'static_coverage':static,
          'tiny_AP_tau_0_10_finite_grid_gap_lower_bound':{'failed_contexts':gap_count,'selected_contexts':4,
                'tested_detectors':6,'population_extrapolation':False},
          'source_RunIDs':{k:v.name for k,v in paths.items()},
          'scientific_interpretation_requires_student_supervisor_review':True}
    save(out/'acceptance.json',accepted)
    lines=['# RQ1–RQ4 evidence acceptance','',
        'This is an evidence and execution report, not a manuscript results section.',
        'Student and supervisor scientific review remains required.','',
        f'RunID: `{out.name}`. Formal test-dev pairs: **0/500**. Admission remains **DATASET_BLOCKED**.',
        'No historical experiment, original UVL, dataset or checkpoint was overwritten.', '',
        '| RQ | Status | Actual evidence | Remaining blocker |','|---|---|---|---|']
    lines += [f"| {r['RQ']} | {r['status']} | {r['result']} | {r['blocker']} |" for r in rq]
    lines+=['','## RQ1 boundaries','',
         'FM_P: 25 leaves, five constraints, 46,368 valid configurations and no dead leaves.',
         'FM_D: 34 leaves, four constraints, 635,904 valid configurations, 844 training variants, SAT and no dead leaves.',
         'The FM_D count is guarded exact factorization checked against independent structural enumeration, not a BDD/#SAT count.',
         'All numeric FM_P attributes and configuration IDs are checked. The original FM_D v1 has no numeric attribute blocks;',
         'per-checkpoint measurements are external evidence, and generic parameter counts are not substituted.',
         'The compatibility matrix covers every FM_D leaf. Feature support is not support for every combination.',
         'The preserved SAHI/TTA exclusion and physical FM_P restrictions remain study assumptions.',
         'RollingShutter is generated in renderer tests but excluded from evaluation until GT/ignore geometry is validated.', '',
         '## RQ2 data and inference','',
         'VM09 has 243 configurations; VM26 has 54 with 18 reused None-JPEG configurations. Their union has 279 unique responses.',
         'All share the same 300 ds04 development images. VM09 CSV and cell result hashes were checked; no new raw rematching of all 243 is claimed.',
         'The accepted VM26 raw/GT replay and 200-draw bootstrap evidence is reused without rerunning it.',
         'JPEG Q85 is outside the original FM_P; it remains a measured numeric factor. Historical noise variance 15/30 is not UVL sigma 15/30.',
         'Feature analogues are not asserted to be pixel-identical to the newer UVL renderer.',
         'Tree, rule-stump and forest fits are descriptive, in-sample diagnostics. Fitting seeds are not detector training seeds.',
         'Balanced accuracy, macro F1, failure precision/recall and held-out confusion remain null: ML_INSUFFICIENT_EVIDENCE.',
         'The learning admission now checks both degradation and exact image membership before accepting a fold.', '',
         '## RQ3 uncertainty and missing outcomes','',
         'Three policies remain separate: exact identity, conservative source-supported proxy and supervisor provisional proxy.',
         'For CV261102 the coverage counts are 1/3/18; the three older version-8.4.33 checkpoints have conservative coverage 1 under the stricter source-version evidence policy, not proof that brightness variation was absent.',
         'The AP predicate uses the original-coordinate AP50:95 intervals; a verified AP failure is sufficient to fail AP AND latency even when latency is unavailable.',
         'A PASS requires both a sufficient AP lower bound and compatible measured latency. Unknown outcomes are not failures.',
         'The development subset analysis uses a newly declared post-hoc simultaneous family of 72 endpoint/size AP intervals from the saved 200 paired image draws.',
         'It is separate from the original VM26 36-contrast DID family. There is no flight-cluster or training-seed uncertainty.',
         'Only 640/1280 and JPEG None/Q50 have these intervals. Medium/large AP points exist, but their intervals are NOT_EVALUATED.',
         'The four purposive contexts cannot estimate the 46,368-configuration population. No full-space gap or learned UVL constraint is claimed.',
         'Confusion counts are shown with missing outcomes. Complete-case agreement is selected-failure diagnostics; specificity may be undefined.', '',
         '## Sampling and future training','',
         'The frozen v1 500-pair manifest is unchanged. Cross-architecture v2 is a separate 900-pair plan:',
         'the same uniform 100 of 26,496 geometry-eligible FM_P settings crossed with nine purposively bound checkpoint/resolution settings.',
         'It is not a uniform sample of all 635,904 detector configurations. Albu activation, independent admission and full-pair cost are unresolved.',
         'The RQ4 JSON freezes proposed matched continuation and ablation arms. Its validator runs on CPU.',
         'A new augmentation training executor and health canary are not implemented or validated for this protocol; no training is marked complete.',
         'Future execution requires one measured pair and identical repeat, a one-seed training feasibility pilot and an explicit bounded budget.', '',
         '## Reproduction','', '```powershell',
         r'.\.venv\Scripts\python.exe -B main/experiments/cv261207/rq_evidence.py',
         r'.\.venv\Scripts\python.exe -B main/experiments/cv261207/rq_evidence.py --live-demo',
         r'.\.venv\Scripts\python.exe -B main/experiments/cv261207/rq4_protocol.py --check <RunDir>/rq4_matched_protocol.json',
         r'.\.venv\Scripts\python.exe -B -m unittest discover -s main/experiments/cv261207/tests -v', '```','',
         'The default reuses saved GPU evidence. --live-demo authorizes one bounded 120-second validation-image prediction, not a formal pair or training.',
         'Both modes create a new RunID and verify inputs. See inputs.json, stage timings, per-figure CSVs and SHA256SUMS.txt.', '',
         'The independent validation plan is linked through the accepted history review. No incompatible external dataset is substituted.']
    (out/'acceptance_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    oral='''# Short oral explanation

I use two feature models. One describes image changes. The other describes detector settings.
The model counts are checked, but many detector settings are not implemented yet.

We have 279 historical development results from three checkpoints. The JPEG study shows that
higher resolution helps without blur, but this advantage reverses under severe blur in this grid.
These images were used in development. I do not call this an independent test.

Static gap means missing training coverage under a stated rule. Empirical gap means that no
tested detector meets the AP and time requirements. These are different measurements.
Some AP failures are clear, but complete latency and independent validation are still missing.

The test-dev gate stays blocked. I have not run the 500 formal pairs or the augmentation study.
The next step is to agree on independent data and measure one complete pair before scaling up.
'''
    (out/'oral_explanation_en.md').write_text(oral,encoding='utf-8')
    return accepted


def execute(args):
    out=new_run('rq_acceptance');start=time.perf_counter()
    print('[RQ evidence]',out,flush=True)
    runtime(out)
    import platform
    import importlib.metadata
    save(out/'environment.json',{'python':sys.version,'platform':platform.platform(),
            'versions':{name:importlib.metadata.version(name) for name in ('numpy','ultralytics','torch','pycocotools','matplotlib')},
            'training':False})
    (out/'command.txt').write_text(' '.join([sys.executable,'-B',str(Path(__file__)),*sys.argv[1:]])+'\n',encoding='utf-8')
    save(out/'protocol.json',{'version':'CV261207_RQ_EVIDENCE_V1','scope':'CPU_REVIEW_AND_OPTIONAL_SINGLE_GPU_DEMO',
          'expensive_training_authorized':False,'formal_gate_bypass':False,
          'GPU_demo_requested':args.live_demo,'GPU_demo_hard_seconds':120,
          'thresholds':[.05,.10,.15],'latency_limit_ms':50,
          'thresholds_source':'Existing CV261207 outputs.py empirical sensitivity grid; not a calibrated mission requirement.',
          'new_gap_CI_family':'POST_HOC_72_ENDPOINT_AP_SIZES','new_training_runs':0})
    try:
        paths=checked_inputs(out)
        checkpoint=sha256(paths['audit']/'selection_manifest.json')
        configs,fmp,fmd=rq1(out,paths['audit'])
        print('RQ1 logical counts and compatibility matrix checked.',flush=True)
        base=checkpoint_static(out,paths['admission'],configs,fmp)
        freeze_v2(out,paths['audit'],paths['admission'],fmd)
        learning_prepare(out,paths['review'],paths['admission'])
        gap_analyze(out,paths['review'],base)
        augmentation_prepare(out,paths['admission'],paths['audit'])
        print('Historical union, descriptive baselines and conditional subset gaps checked.',flush=True)
        sat=command(out,'SAT_demo',[str(FLAMA),'-B',str(FMD.with_name('flamapy_count.py')),'--model',str(FMD)],120)
        if sat['exit_code']:raise RuntimeError('SAT demonstration failed.')
        save(out/'SAT_demo_receipt.json',sat)
        engineering,demo=engineering_evidence(out,paths,args.live_demo)
        if sha256(paths['audit']/'selection_manifest.json')!=checkpoint:
            raise ValueError('Frozen v1 manifest changed.')
        report(out,paths,engineering,demo)
        from rq_figures import render
        render(out,paths['audit'],paths['review'])
        save(out/'run.json',{'RunID':out.name,'ExperimentID':'cv261207','mode':'RQ_EVIDENCE_ACCEPTANCE',
             'status':'PARTIAL','scope':'DEVELOPMENT_AND_ENGINEERING_WITH_LOGIC_VALIDATION',
             'wall_seconds':time.perf_counter()-start,'ended_utc':datetime.now(timezone.utc).isoformat(),
             'formal_pairs':0,'new_training_runs':0})
        seal(out)
    except BaseException:
        save(out/'failure.json',{'status':'PARTIAL','traceback':traceback.format_exc(),
                                'wall_seconds':time.perf_counter()-start})
        print('Partial evidence retained:',out,flush=True)
        raise
    print('[PARTIAL: scientific blockers retained]',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live-demo',action='store_true',help='One bounded development prediction; no formal pair or training.')
    execute(parser.parse_args())
