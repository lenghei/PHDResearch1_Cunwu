"""Assemble the twelve requested acceptance fields without rerunning experiments."""
import argparse
import json
from common import load,new_run,resolve_run,save,seal,sha256,source,verify_hashes
from historical_review import csv_file


def build(run,figures):
    verify_hashes(run,('acceptance.json','run.json'))
    verify_hashes(figures,('figure_evidence.json','rq1_rq4_figures.pdf'))
    a=load(run/'acceptance.json');out=new_run('rq_delivery')
    rows=[]
    designs={
      'RQ1':'Hash-check original UVLs, verify exact counts and attribute IDs, reconcile solver/generator receipts and installed API source; run explicit constraint examples.',
      'RQ2':'Join VM09/VM26 with exact overlap removal; preserve measured numeric factors; reuse paired DID inference and fit descriptive tree/stump/forest baselines.',
      'RQ3':'Separate three static policies; compute post-hoc joint72 endpoint/size intervals and four-state AP/latency outcomes on a finite measured subset.',
      'RQ4':'Prepare matched clean-control and targeted continuation arms; run CPU protocol validator only.'}
    uncertainty={
      'RQ1':'Exact guarded logical counts, not a statistical estimate. Physical realism and all-configuration executability remain unverified.',
      'RQ2':'Original 200-draw paired image bootstrap with joint36 contrast family. No training-seed or flight-cluster inference. Held-out classifier metrics are not evaluated.',
      'RQ3':'Post-hoc joint72 image-bootstrap intervals; four purposive contexts are not a probability sample of full FM_P. Missing latency prevents complete outcome ascertainment.',
      'RQ4':'No before/after response, gap-reduction interval or training-seed variability measured.'}
    source_names={
      'RQ1':['rq1_validation.json','fmd_compatibility.csv','uvl_constraints.csv','SAT_demo_receipt.json'],
      'RQ2':['historical_response_union.json','measured_learning_features.csv','grouped_learning_acceptance.json'],
      'RQ3':['endpoint_confidence_intervals.csv','development_subset_gaps.csv','static_empirical_diagnostics.csv'],
      'RQ4':['rq4_matched_protocol.json']}
    for q in a['research_questions']:
        name=q['RQ']
        command='python -B main/experiments/experiments_test/cv261207/rq_evidence.py'
        if name=='RQ4':command=f'python -B main/experiments/experiments_test/cv261207/rq4_protocol.py --check datasets/local_runs/cv261207/{run.name}/rq4_matched_protocol.json'
        dataset=('Original UVL v1 and accepted synthetic generator tests; no AP dataset population.' if name=='RQ1' else
                 'No training/evaluation data executed in RQ4; proposed inputs retain explicit gates.' if name=='RQ4' else
                 'ds04, 300 VisDrone2019-DET-val images; native10; pr01; DEVELOPMENT_ONLY; checkpoint-selection exposure.')
        rows.append({'RQ':name,'research_question':q['question'],'tested_hypothesis':q['tested_hypothesis'],
             'experiment_design':designs[name],'actual_configurations_evaluated':q['configurations'],
             'actual_dataset_and_split':dataset,'measured_results':q['result'],
             'statistical_uncertainty':uncertainty[name],'reproduction_commands':command,
             'evidence_RunIDs':'|'.join([run.name,a['source_RunIDs']['audit'],a['source_RunIDs']['review']]),
             'supported_conclusion':q['result'],'unresolved_limitations':q['blocker'],
             'acceptance_status':q['status'],
             'source_files_SHA256':json.dumps({n:sha256(run/n) for n in source_names[name]},sort_keys=True)})
    csv_file(out/'rq_acceptance_details.csv',rows)
    save(out/'rq_acceptance_details.json',{'source_acceptance_RunID':run.name,'questions':rows,
          'figures_RunID':figures.name,'formal_pairs':0,'independent_test_gate':'DATASET_BLOCKED',
          'scientific_review':'Student/supervisor review remains required.'})
    text=['# RQ1–RQ4 acceptance details','',f'Source acceptance: `{run.name}`. Figure run: `{figures.name}`.',
          'Formal pairs: 0. Independent testing: DATASET_BLOCKED.','']
    fields=('research_question','tested_hypothesis','experiment_design','actual_configurations_evaluated',
            'actual_dataset_and_split','measured_results','statistical_uncertainty','reproduction_commands',
            'evidence_RunIDs','supported_conclusion','unresolved_limitations','acceptance_status')
    for row in rows:
        text+=['## '+row['RQ'],'']
        text += [f'{i}. **{key.replace("_"," ")}**: {row[key]}' for i,key in enumerate(fields,1)]
        text+=['','Source files and SHA256:','', '```json',row['source_files_SHA256'],'```','']
    (out/'acceptance_details.md').write_text('\n'.join(text),encoding='utf-8')
    save(out/'inputs.json',{'acceptance':source(run/'SHA256SUMS.txt'),'figures':source(figures/'SHA256SUMS.txt'),
                            'generator':source(__import__('pathlib').Path(__file__))})
    save(out/'run.json',{'RunID':out.name,'ExperimentID':'cv261207','status':'PARTIAL',
                         'mode':'RQ_ACCEPTANCE_DELIVERY','source_acceptance_RunID':run.name,
                         'new_forward':0,'new_training':0})
    seal(out);print(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--figures',required=True)
    args=parser.parse_args();build(resolve_run(args.run),resolve_run(args.figures))
