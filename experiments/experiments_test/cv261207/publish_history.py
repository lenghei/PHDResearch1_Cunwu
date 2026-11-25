"""Export small verified historical summaries without private paths or raw data."""
import argparse
from common import load, resolve_run, verify_hashes
from historical_review import table
from outputs import publish_snapshot

NAMES={'ck01':'YOLO26s','ck05':'YOLOv8s','ck06':'YOLO11s'}


def publish(review, gate):
    hashes=verify_hashes(review,('run.json','provenance.json','configuration_metrics.csv',
           'bootstrap_joint36.csv','resolution_differences.csv','rule_validation.json','learned_rules.txt'))
    gate_hashes=verify_hashes(gate,('history_review.json','independent_validation_plan.md'))
    run=load(review/'run.json')
    if run['status']!='COMPLETE_WITH_LIMITATIONS' or load(gate/'history_review.json')['status']!='DATASET_BLOCKED':
        raise ValueError('Expected verified development evidence and a blocked test gate.')
    provenance=load(review/'provenance.json')
    cells=[]
    for row in table(review/'configuration_metrics.csv'):
        cells.append({'config_id':row['config_id'],'checkpoint':row['CheckpointID'],'model':NAMES[row['CheckpointID']],
                      'blur':int(row['blur']),'jpeg':row['jpeg'],'resolution':int(row['resolution']),
                      **{k:float(row[k]) for k in ('AP50_95','AP50','Tiny_AP','Small_AP','Medium_AP','Large_AP','AR')},
                      'metric_sha256':row['source_metric_SHA256'],'raw_prediction_sha256':row['raw_SHA256'],
                      'checkpoint_sha256':row['checkpoint_SHA256']})
    gains=[{**r,'model':NAMES[r['checkpoint']],'blur':int(r['blur']),'estimate':float(r['estimate'])}
           for r in table(review/'resolution_differences.csv')
           if r['metric']=='AP50_95' and r['high']=='1280' and r['low']=='640']
    intervals=[{**r,'model':NAMES[r['checkpoint']],
                **{k:float(r[k]) for k in ('estimate','joint36_low','joint36_high','critical')}}
               for r in table(review/'bootstrap_joint36.csv') if r['metric']=='map50_95' and r['size']=='all']
    contributions=[
        {'contribution':'C1','status':'VALIDATED_LOGIC_AND_GENERATOR','boundary':'Existing UVL v1 validation; not all configurations executed.'},
        {'contribution':'C2','status':'PARTIAL_DEVELOPMENT_EVIDENCE','boundary':'54 historical configurations support measured interactions; image-disjoint held-out confirmation is missing.'},
        {'contribution':'C3','status':'PARTIAL_STATIC_ONLY','boundary':'No independent empirical gap or validated static-to-empirical predictor.'},
        {'contribution':'C4','status':'NOT_EVALUATED','boundary':'No targeted augmentation experiment.'}]
    report={'historical_vm26':{'source_RunID':'vm26_r001','review_RunID':review.name,
             'gate_review_RunID':gate.name,'scope':'DEVELOPMENT_ONLY','configurations':54,'images':300,
             'dataset':'ds04','variant':'dv03','protocol':'pr01','checkpoints':list(NAMES),
             'GT_sha256':provenance['configurations'][0]['GT_sha256'],
             'cells':cells,'gains':gains,'headline_intervals':intervals,
             'rules':(review/'learned_rules.txt').read_text(encoding='utf-8'),
             'validation':load(review/'rule_validation.json'),
             'contributions':contributions,
             'source_hashes':{**hashes,'history_review.json':gate_hashes['history_review.json']}},
             'formal_evaluation':'DATASET_BLOCKED','formal_pairs_measured':0}
    # The rule receipt contains a local source path; publish only its identity.
    report['historical_vm26']['validation']['source'].pop('path',None)
    print(publish_snapshot(review.name,'COMPLETE_WITH_LIMITATIONS','DEVELOPMENT_ONLY',report))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--gate',required=True)
    args=parser.parse_args()
    publish(resolve_run(args.run),resolve_run(args.gate))
