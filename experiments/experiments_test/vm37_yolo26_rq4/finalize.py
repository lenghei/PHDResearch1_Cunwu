"""Freeze the verified local evidence and small public result exports."""
import argparse
import csv
import shutil
from study import *


def execute(summary_id):
    if (RUNS/'.gpu.lock').exists():raise RuntimeError('A GPU worker is still active.')
    out=create('release_cpu');runtime(out,cuda=False)
    summary=run_path(summary_id)
    gate=RUNS/'vm37_yolo26_rq4_matched_budget_cpu_20261111T235845_34c7f6da'
    cpu=RUNS/'vm37_yolo26_rq4_focused_regression_cpu_20261111T235941_ce168cdb'
    parity=RUNS/'vm37_yolo26_rq4_optimization_parity_cpu_20261111T235905_5455807b'
    preservation=RUNS/'vm37_yolo26_rq4_preservation_cpu_20261111T235905_93e9b8cc'
    for folder,required in ((summary,('training_measurements.csv','results_provenance.csv')),
                            (gate,('budget_admission.json',)),(cpu,('acceptance.json',)),
                            (parity,('state_comparison.json',)),(preservation,('preservation.json',))):
        verify_hashes(folder,required)
    if load(gate/'budget_admission.json')['status']!='RESOURCE_BLOCKED':raise ValueError('Unexpected resource decision; review before finalizing.')
    if load(cpu/'acceptance.json')['status']!='PASS' or load(parity/'state_comparison.json')['status']!='NUMERICAL_PARITY_PASS' or load(preservation/'preservation.json')['status']!='PASS':
        raise ValueError('Evidence failed final acceptance.')
    source_run=run_path(load(RUNS/'source_environment.json')['RunID'])
    review=RUNS/'vm37_yolo26_rq4_augmentation_source_review_20261111T221639_fa2dab24'
    for folder,description in ((source_run,'Verified stable source and native10 engineering compatibility.'),
                               (review,'Verified augmentation sources with explicit uncalibrated-proxy limits.')):
        if (folder/'SHA256SUMS.txt').exists():raise FileExistsError('Preparation evidence was already finalized.')
        save(folder/'final_acceptance.json',{'status':'COMPLETE_WITH_LIMITATIONS','description':description,
            'version_audit_RunID':'vm37_yolo26_rq4_version_regression_cpu_20261111T223751_94186fca',
            'optimized_GPU_RunID':'vm37_yolo26_rq4_full_epoch_yolo26s_control_20261111T232439_301fb25d',
            'formal100_arms_completed':0,'matched_pairs':0},folder)
        save(folder/'nested_SHA256.json',{str(path.relative_to(folder)):sha256(path) for path in folder.rglob('*') if path.is_file() and path.parent!=folder},folder)
        write_atomic(folder/'diary.md','# Source preparation\n\n'+description+' Raw source retrievals and failed requests are retained. No completed formal matched training is claimed.\n',folder)
        save(folder/'run.json',{'RunID':folder.name,'ExperimentID':EXPERIMENT,'status':'COMPLETE_WITH_LIMITATIONS'},folder,replace=True)
        seal(folder)
    with (HERE/'TRAINING_MATRIX.csv').open(encoding='utf-8',newline='') as stream:matrix=list(csv.DictReader(stream))
    if len(matrix)!=12 or len({(r['architecture'],r['seed'],r['arm']) for r in matrix})!=12:raise ValueError('Invalid planned matrix.')
    for row in matrix:
        row['status']='RESOURCE_BLOCKED' if row['architecture']=='YOLO26s' and row['seed']=='42' else 'NOT_RUN_PRIMARY_PAIR_RESOURCE_BLOCKED'
    with (HERE/'TRAINING_MATRIX.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=matrix[0]);writer.writeheader();writer.writerows(matrix)
    with (HERE/'DECISION_REGISTER.csv').open(encoding='utf-8',newline='') as stream:decisions=list(csv.DictReader(stream))
    for row in decisions:
        if row['decision']=='Budget admission':row['status']='RESOURCE_BLOCKED_MEASURED_COST'
    decisions.append({'decision':'Post-training input SHA','choice':'Raise after preserving failed posthash evidence',
        'alternative':'Record unchanged=False without failing','reason':'Correct a postcondition gate; fault-injection CPU test passed; measured inputs were unchanged.','status':'CPU_TESTED_AFTER_GPU_MEASUREMENTS'})
    with (HERE/'DECISION_REGISTER.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=('decision','choice','alternative','reason','status'));writer.writeheader();writer.writerows(decisions)
    exports=HERE/'results';exports.mkdir(exist_ok=False)
    names={'training_measurements.csv':'engineering_summary.csv','research_metrics.csv':'research_metrics.csv',
        'per_class_metrics.csv':'per_class_metrics.csv','results_provenance.csv':'results_provenance.csv',
        'epoch_cost_and_loss.png':'epoch_cost_and_loss.png','epoch_cost_and_loss.pdf':'epoch_cost_and_loss.pdf'}
    for origin,target in names.items():shutil.copyfile(summary/origin,exports/target)
    manifest=load(HERE/'SOURCE_MANIFEST.json')
    manifest['actual_status']={'full100_arms_completed':0,'matched_scientific_pairs':0,'decision':'RESOURCE_BLOCKED',
        'augmentation_effectiveness':'NOT_EVALUATED','historical_VM36_pairs_rerun':0,'independent_test_dev':'DATASET_BLOCKED'}
    manifest['acceptance_RunIDs']={'evidence':summary_id,'resource':gate.name,'CPU':cpu.name,'state_parity':parity.name,'preservation':preservation.name}
    manifest['source_initializations']=[{'architecture':r['architecture'],'seed':r['seed'],'COCO_source_SHA256':r['source']['sha256'],
        'native10_initialization_SHA256':r['native10_candidate']['sha256']} for r in load(PREP/'CANDIDATE_INITIALIZATION_MANIFEST.json')]
    manifest['post_measurement_patch']={'scope':'Input-integrity failure assertion only; no numerical training or metric change.',
        'CPU_test_RunID':cpu.name,'GPU_runs_after_patch':0,'measured_source':'GPU Runs retain protocol.json and exact source_snapshots.'}
    manifest['public_exports']={name:sha256(exports/name) for name in names.values()}
    shared=[PUBLIC/'code/main'/name for name in ('training_health.py','training_data.py','training_effects.py','predictor_stable.py','predictor.py','labels.py','statistics.py')]
    manifest['current_project_sources']={str(path.relative_to(PUBLIC)).replace('\\','/'):sha256(path) for path in [*HERE.rglob('*.py'),*shared]}
    write_atomic(HERE/'SOURCE_MANIFEST.json',__import__('json').dumps(manifest,indent=2)+'\n',ROOT,replace=True)
    budget=load(gate/'budget_admission.json');measured=budget['arms'][0]['measurement']
    with (HERE/'README.md').open('a',encoding='utf-8') as stream:
        stream.write(f'''\n## Measured status

Two complete first-epoch CONTROL cost studies and two engineering Canaries were run. The optimized full epoch took {measured['training_setup_epoch_internal_val_save_seconds']/60:.3f} minutes for {measured['training_images']} train / {measured['validation_images']} development images, {measured['forward_batches']} forward/backward batches and {measured['optimizer_updates']} optimizer/EMA updates. Saved model, EMA, optimizer and scheduler states match the original implementation exactly.

The 100-epoch CONTROL forecast alone is {budget['conservative_arm_seconds'][0]/3600:.3f} hours with the frozen 25% planning margin; the minimum single-resolution research-evaluation reserve is {budget['evaluation_reserve_seconds']/3600:.3f} hours. This is a forecast, not measured 100-epoch training. The complete pair failed resource admission. Zero formal 100-epoch arms and zero matched augmentation pairs are complete.

See results/engineering_summary.csv, results/research_metrics.csv and results/results_provenance.csv. All reported AP values are engineering/development evidence.
''')
    paths=sorted(path for path in HERE.rglob('*') if path.is_file() and path.name!='SHA256SUMS.txt' and '__pycache__' not in path.parts)
    with (HERE/'SHA256SUMS.txt').open('x',encoding='utf-8') as stream:
        for path in paths:stream.write(sha256(path)+'  '+str(path.relative_to(HERE)).replace('\\','/')+'\n')
        for path in shared:stream.write(sha256(path)+'  ../../'+str(path.relative_to(PUBLIC)).replace('\\','/')+'\n')
    state=load(RUNS/'continuation_state.json');state.update(current_stage='GPU branch stopped: RESOURCE_BLOCKED. Final CPU checks and local commit only.',active_RunID=None,
        focused_CPU=cpu.name,latest_summary=summary_id,resource_gate=gate.name,full_state_parity=parity.name,preservation=preservation.name)
    save(RUNS/'continuation_state.json',state,RUNS,replace=True)
    save(out/'release_acceptance.json',{'status':'PARTIAL_RESOURCE_BLOCKED','sources':[source(folder/'SHA256SUMS.txt') for folder in (summary,gate,cpu,parity,preservation)],
        'published_checksums':source(HERE/'SHA256SUMS.txt'),'GPU_seconds':0,'public_exports':manifest['public_exports'],
        'historical_runs_modified':False,'raw_data_published':False,'checkpoints_published':False,'private_configs_published':False},out)
    write_atomic(out/'diary.md','# Verified public export\n\nExported small accepted engineering summaries and source-driven figures. Closed preparation stages and froze sources/checksums. Formal training remains resource-blocked. No new GPU work.\n',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'status':'COMPLETE_WITH_LIMITATIONS'},out,replace=True)
    seal(out);print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--summary',required=True);execute(parser.parse_args().summary)
