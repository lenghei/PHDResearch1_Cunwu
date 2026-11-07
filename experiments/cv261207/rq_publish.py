"""Export a verified RQ acceptance summary without datasets or private paths."""
import argparse
from common import RUNS,load,resolve_run,verify_hashes,save,sha256
from historical_review import table
from outputs import publish_snapshot


def publish(run):
    checks=verify_hashes(run,('acceptance.json','run.json','rq_acceptance.csv',
          'rq1_validation.json','static_coverage_by_checkpoint.csv','development_subset_gaps.csv',
          'grouped_learning_acceptance.json','engineering_pairs.csv'))
    a=load(run/'acceptance.json')
    if a['formal_measured_pairs']!=0 or a['independent_test_status']!='DATASET_BLOCKED':
        raise ValueError('Expected the accepted development-only state.')
    counts=load(run/'rq1_validation.json')
    body={'RunID':run.name,'status':a['status'],'scope':a['scope'],
          'research_questions':a['research_questions'],'FM_P':{k:counts['FM_P'][k] for k in ('leaves','constraints','count','dead_features')},
          'FM_D':counts['FM_D'],'formal_pairs':0,'test_gate':'DATASET_BLOCKED',
          'historical_unique_responses':a['historical_unique_responses'],
          'shared_development_images':a['development_image_population'],
          'historical_VM26_configurations':54,'static':table(run/'static_coverage_by_checkpoint.csv'),
          'subset_gaps':table(run/'development_subset_gaps.csv'),
          'confusion_diagnostics':table(run/'static_empirical_diagnostics.csv'),
          'learning_status':'ML_INSUFFICIENT_EVIDENCE',
          'held_out_metrics':{k:None for k in ('balanced_accuracy','macro_F1','failure_precision','failure_recall')},
          'descriptive_model_fits':table(run/'descriptive_model_fits.csv'),
          'feature_compatibility':table(run/'fmd_compatibility.csv'),
          'new_GPU_worker_seconds':a['this_run_GPU_worker_seconds'],
          'new_GPU_forward_seconds':a['this_run_GPU_forward_seconds'],
          'new_training_seconds':0,'source_RunIDs':a['source_RunIDs'],'source_hashes':checks,
          'original_frozen_pairs':500,'v2_planned_pairs':900,'v2_executed_pairs':0,
          'gap_closure':'NOT_EVALUATED','learned_constraints_added':False}
    print(publish_snapshot(run.name,'PARTIAL','LOGIC_AND_DEVELOPMENT_ONLY',{'rq_acceptance':body}))


def local_index(run,figures,details=None):
    verify_hashes(run,('acceptance.json','run.json'))
    verify_hashes(figures,('figure_evidence.json','rq1_rq4_figures.pdf'))
    if load(figures/'run.json').get('source_acceptance_RunID')!=run.name:
        raise ValueError('Figure source differs from the acceptance run.')
    pointer=RUNS/'latest_rq_evidence.json'
    if pointer.exists() and load(pointer).get('format')!='CV261207_RQ_INDEX_V1':
        raise ValueError('Unknown local pointer; do not overwrite.')
    target=RUNS/'rq_index.html'
    marker='<!-- cv261207-rq-index-v1 -->'
    if target.exists() and marker not in target.read_text(encoding='utf-8'):
        raise ValueError('Unknown local index; do not overwrite.')
    extra=''
    if details:
        verify_hashes(details,('rq_acceptance_details.json','run.json'))
        if load(details/'run.json')['source_acceptance_RunID']!=run.name:
            raise ValueError('Detailed report source differs.')
        extra=f'<li><a href="{details.name}/acceptance_details.md">Twelve-field acceptance for each research question</a></li>'
    save(pointer,{'format':'CV261207_RQ_INDEX_V1','RunID':run.name,'figures_RunID':figures.name,
          'details_RunID':details.name if details else None,
          'status':'PARTIAL','test_gate':'DATASET_BLOCKED','acceptance_sha256':sha256(run/'acceptance.json'),
          'figure_manifest_sha256':sha256(figures/'figure_evidence.json')})
    target.write_text(marker+'<!doctype html><meta charset="utf-8"><title>CV261207 RQ evidence</title>'
       '<style>body{font:18px Arial;max-width:1000px;margin:40px auto}li{margin:15px}</style>'
       '<h1>Latest RQ1–RQ4 evidence</h1><p>PARTIAL · Independent test-dev remains blocked · Formal pairs: 0</p><ul>'
       f'<li><a href="{run.name}/acceptance_report.md">Acceptance report</a></li>'
       f'<li><a href="{figures.name}/index.html">Twelve verified figures and source tables</a></li>'
       f'<li><a href="{figures.name}/rq1_rq4_figures.pdf">PDF figure book</a></li>'
       f'<li><a href="{run.name}/oral_explanation_en.md">Short English oral explanation</a></li>'
       f'<li><a href="{run.name}/rq4_matched_protocol.json">Unexecuted matched-augmentation protocol</a></li>'+extra+'</ul>',encoding='utf-8')
    print(target)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',required=True)
    parser.add_argument('--link-local',action='store_true');parser.add_argument('--figures');parser.add_argument('--details')
    args=parser.parse_args()
    if args.link_local:
        if not args.figures:parser.error('--figures is required for --link-local.')
        local_index(resolve_run(args.run),resolve_run(args.figures),resolve_run(args.details) if args.details else None)
    else:publish(resolve_run(args.run))
