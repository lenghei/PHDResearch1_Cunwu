"""Write source-linked findings and contribution boundaries from reviewed files."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import platform
import sys

from common import HERE, load, new_run, resolve_run, save, seal, source, verify_hashes
from historical_review import table, csv_file

NAMES={'ck01':'YOLO26s','ck05':'YOLOv8s','ck06':'YOLO11s'}


def summarize(review, gate):
    verify_hashes(review,('run.json','provenance.json','bootstrap_joint36.csv','rule_validation.json'))
    verify_hashes(gate,('history_review.json','independent_validation_plan.md'))
    if load(review/'run.json')['status']!='COMPLETE_WITH_LIMITATIONS':
        raise ValueError('Historical review incomplete.')
    out=new_run('synthesis')
    joint=table(review/'bootstrap_joint36.csv')
    gains=table(review/'resolution_differences.csv')
    headline=[]
    for ck in NAMES:
        r=next(r for r in joint if r['checkpoint']==ck and r['metric']=='map50_95'
               and r['size']=='all' and r['effect']=='R_B_J')
        headline.append({**r,'model':NAMES[ck]})
    csv_file(out/'headline_effects.csv',headline)
    text=['# CV261207: historical cross-model interaction review','',
          'Status: COMPLETE_WITH_LIMITATIONS. Independent test admission: DATASET_BLOCKED.',
          'Formal measured pairs: 0/500. No new GPU inference, training or archive copying.', '',
          '## Evidence', '',
          f'Historical RunID: vm26_r001. Review: `{review.name}`. Gate review: `{gate.name}`.',
          'The 54 cells comprise three checkpoints × three resolutions (640/960/1280) ×',
          'two blur settings (B0/B2) × three extra-JPEG settings (None/Q85/Q50).',
          'All use the same 300 ds04 development images, native10 labels and pr01 FP32 evaluation.',
          'Illumination and noise are fixed at I0/N0. This is not the full degradation space.',
          'JPEG None means no additional encode/decode; the source images are already JPEGs.', '',
          'All 54 metric and raw-prediction SHA256 identities and 300-record memberships passed.',
          'Original GT contains 21,756 effective instances. All 24 endpoint bootstrap arrays',
          '(200 shared image draws each) and their paired index/source receipts passed.',
          'Two configurations were independently replayed from raw predictions through ignore-union',
          'filtering and COCO matching. Their full-data AP/AR and six selected aggregate bootstrap',
          'draws matched exactly. The remaining 4,794 saved endpoint draws were integrity-checked,',
          'not freshly rematched. No per-image AP averaging was used.', '',
          '## AP50:95 resolution differences', '',
          'Values below are percentage points, 1280 minus 640. AP itself remains in [0,1] in CSVs.', '',
          '| Checkpoint | B0 None | B0 Q85 | B0 Q50 | B2 None | B2 Q85 | B2 Q50 |',
          '|---|---:|---:|---:|---:|---:|---:|']
    for ck,name in NAMES.items():
        values=[next(float(r['estimate'])*100 for r in gains if r['checkpoint']==ck and int(r['blur'])==b
                  and r['jpeg']==j and r['metric']=='AP50_95' and r['high']=='1280' and r['low']=='640')
                for b in (0,2) for j in ('None','Q85','Q50')]
        text.append('| '+name+' | '+' | '.join(f'{v:+.3f}' for v in values)+' |')
    text+=['', 'Let G(b,j) = AP1280(b,j) − AP640(b,j). The three-way contrast is',
           '[G(B2,Q50) − G(B2,None)] − [G(B0,Q50) − G(B0,None)].', '',
           '| Checkpoint | Three-way contrast (pp) | 95% simultaneous interval (pp) |',
           '|---|---:|---:|']
    for r in headline:
        text.append(f"| {r['model']} | {100*float(r['estimate']):+.3f} | [{100*float(r['joint36_low']):+.3f}, {100*float(r['joint36_high']):+.3f}] |")
    text+=['', 'The original max-standardized centered bootstrap family has 36 contrasts.',
           'Its critical value was reproduced from the 200 paired draws. These are conditional',
           'image-bootstrap intervals, not training-seed or independent-dataset intervals.',
           'There is no reliable flight clustering. Q85 and 960 confidence intervals remain NOT_EVALUATED.', '',
           'The no-blur JPEG effect on the resolution gain is negative for every checkpoint,',
           'with its joint interval below zero. The severe-blur JPEG effect has an interval',
           'containing zero for every checkpoint. The positive three-way contrast describes',
           'this difference between conditions; it does not mean JPEG repairs blur.',
           'The signs of the 1280−640 gains are positive in all nine B0 cases and negative',
           'in all nine B2 cases. Only two blur levels and fixed I0/N0 were tested here.', '',
           '## Interpretable rules and validation', '',
           'A depth-2 decision tree fitted 18 measured resolution-gain rows using categorical',
           'checkpoint, blur and JPEG indicators. Its source, predictions and fitted rules are saved.',
           'It describes this finite observed grid; it is not a new sampling algorithm.',
           'Held-out image-and-configuration validation: NOT_EVALUATED. Every aggregate response',
           'shares the same 300 images. Random configuration splits, checkpoint splits and bootstrap',
           'replicas cannot establish image-disjoint validation. No such score is reported.',
           'Differences between three selected checkpoints do not establish architecture causation.', '',
           '## Independent testing and C1–C4', '',
           '| Contribution | Status | Remaining evidence |', '|---|---|---|',
           '| C1 | VALIDATED for existing UVL v1 logic/generator | Executed configurations do not cover the full formal space. |',
           '| C2 | PARTIAL: development interaction evidence | Unseen image/scene and configuration groups; frozen-rule confirmation; exact model provenance. |',
           '| C3 | PARTIAL: static definitions only | Measured AP/cost failures, fixed task thresholds and independent static-to-empirical prediction validation. |',
           '| C4 | NOT_EVALUATED | Targeted augmentation remains untested. |', '',
           'Current train/val/test-dev SHA and ID disjointness is verified in the prior sealed',
           'admission. The bounded follow-up reconciles named Phase2/VM11/VM15/VM34 records.',
           'It does not establish a completed historical test-dev evaluation, nor prove none occurred.',
           'Creation-time training membership and complete manual/external selection/tuning records',
           'remain missing. The independent gate is unchanged.',
           'The supervisor plan proposes a separately admitted prospective native10 aerial cohort',
           'if those records cannot be recovered. No dataset is substituted and no inference is started.', '',
           '## Sources and reproduction', '',
           f'Local evidence: `{review.name}` and `{gate.name}`, each with SHA256SUMS.txt.',
           'Source-to-cell identities are in provenance.json; metric and prediction files stay local.',
           'Figures: resolution_blur_jpeg, checkpoint_resolution_effects, contrast_confidence_intervals,',
           'rules_and_validation (PNG/SVG), with figure_sources.json. Open the review index.html.', '',
           '```powershell',
           r'.\.venv\Scripts\python.exe -B main/experiments/cv261207/historical_review.py review',
           r'.\.venv\Scripts\python.exe -B main/experiments/cv261207/history_gate_review.py',
           r'.\.venv\Scripts\python.exe -B -m unittest discover -s main/experiments/cv261207/tests -v',
           '```', '', 'Each review creates a new immutable RunID. Existing results are not overwritten.']
    (out/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    save(out/'inputs.json',{'review':source(review/'SHA256SUMS.txt'),'gate':source(gate/'SHA256SUMS.txt'),
                           'source_code':[source(p) for p in sorted(HERE.glob('*histor*.py'))]})
    versions={p:importlib.metadata.version(p) for p in ('numpy','pycocotools','matplotlib')}
    save(out/'environment.json',{'python':sys.version,'platform':platform.platform(),'versions':versions,
          'rule_environment':'Existing .venv_flamapy; version in rule_validation.json',
          'raw_replay_seconds':[load(p)['seconds'] for p in sorted(review.glob('replay_*.json'))],
          'total_review_wall_seconds':'NOT_INSTRUMENTED; raw replay times are measured separately.'})
    (out/'index.html').write_text(f'<!doctype html><meta charset="utf-8"><h1>Historical evidence review</h1>'
       f'<p><a href="../{review.name}/index.html">Verified figures</a></p>'
       f'<p><a href="report.md">Findings</a></p><p><a href="../{gate.name}/independent_validation_plan.md">Independent validation plan</a></p>',encoding='utf-8')
    save(out/'run.json',{'RunID':out.name,'ExperimentID':'cv261207','status':'COMPLETE_WITH_LIMITATIONS',
         'scope':'DEVELOPMENT_ONLY','formal_pairs_measured':0,'independent_test_status':'DATASET_BLOCKED',
         'ended_utc':datetime.now(timezone.utc).isoformat()})
    seal(out)
    print(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True);parser.add_argument('--gate',required=True)
    args=parser.parse_args()
    summarize(resolve_run(args.run),resolve_run(args.gate))
