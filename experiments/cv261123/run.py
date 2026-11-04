"""CV261123: evidence-based figures and first-500-pair sampling plan.

Does not run YOLO, estimate an empirical gap or fabricate any AP scores.
"""
from __future__ import annotations
import argparse
import csv
import datetime as dt
import hashlib
import html
import json
import os
import platform
import sys
import uuid
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
PUBLIC_ROOT = HERE.parents[1]
PROJECT_ROOT = PUBLIC_ROOT.parent

from sampling import load_fmp_module, enumerate_configs, make_plan, write_plan_files


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def locate_static_run(root):
    root = Path(root)
    candidates = []
    if not root.is_dir():
        raise FileNotFoundError('No local CV261116 run root: ' + str(root))
    for folder in root.iterdir():
        if not folder.is_dir() or folder.is_symlink():
            continue
        meta = folder / 'result.json'
        if not meta.is_file() or meta.is_symlink():
            continue
        try:
            record = load(meta)
            if (record.get('ExperimentID') == 'cv261116'
                    and record.get('FM_D') and record.get('RunID') == folder.name
                    and (folder / 'static_gap.json').is_file()):
                candidates.append(folder)
        except (OSError, ValueError, TypeError):
            continue
    if not candidates:
        raise FileNotFoundError('No saved CV261116 FM_D static result with required files.')
    return sorted(candidates)[-1]


def read_evidence(folder, fmp_uvl, fmd_uvl):
    folder = Path(folder)
    record=load(folder/'result.json')
    if (record.get('RunID') != folder.name or record.get('ExperimentID') != 'cv261116'
            or record.get('model_validation') != 'PASS'):
        raise ValueError('Not an accepted CV261116 static model analysis.')
    expected=record.get('files_SHA256') or {}
    for name in ('model_count.json','static_gap.json','coverage_by_feature.csv','coverage_rules.csv'):
        if name not in expected or sha(folder/name)!=expected[name]:
            raise ValueError('Saved CV261116 evidence SHA256 mismatch: '+name)
    if sha(fmp_uvl) != record['FM_P_UVL_SHA256'] or sha(fmd_uvl) != record['FM_D']['UVL_SHA256']:
        raise ValueError('The installed UVL differs from the saved source evidence.')
    model=load(folder/'model_count.json')
    gap=load(folder/'static_gap.json')
    with (folder/'coverage_by_feature.csv').open(encoding='utf-8-sig',newline='') as stream:
        features=list(csv.DictReader(stream))
    with (folder/'coverage_rules.csv').open(encoding='utf-8-sig',newline='') as stream:
        rules=list(csv.DictReader(stream))
    fmd=model['independent']['independent_configurations']
    variants=model['independent']['independent_training_variants']
    if (fmd != 635904 or variants != 844 or model['FlamaPy']['satisfiable'] is not True
            or model['FlamaPy']['dead_feature_count'] != 0 or
            model['FlamaPy']['leaf_count'] != 34 or model['FlamaPy']['constraint_count'] != 4):
        raise ValueError('FM_D result is inconsistent with the established UVL model.')
    base=gap['proxy']['FM_P_configurations']
    covered=gap['proxy']['proxy_covered']
    strict=gap['strict_identity']['covered']
    if (base!=46368 or covered!=18 or strict!=1 or len(features)!=25 or len(rules)!=9
        or gap['proxy']['proxy_gap'] != base-covered):
        raise ValueError('The saved Static Gap evidence is incomplete or inconsistent.')
    if any(not row.get('feature') or not row.get('present_total') or
           int(row['covered_when_present'])>int(row['present_total']) for row in features):
        raise ValueError('Feature-level coverage table is malformed.')
    return {
        'source_run':record['RunID'], 'fmp_count':base,'fmd_count':fmd,
        'fmd_training_variants':variants, 'fmd_leaves':34,'fmd_constraints':4,
        'fmp_leaves':25,'fmp_constraints':5,
        'proxy':covered,'strict':strict,
        'training_coverage_status':record['training_source_audit_status'],
        'GPU_measurements':record['gpu_measurements'],
        'AP_metrics':record['AP_metrics'],
        'count_method':model['FlamaPy']['count_backend'],
        'count_limit':model['FlamaPy']['count_limitation'],
        'FM_P_UVL_SHA256':record['FM_P_UVL_SHA256'],
        'FM_D_UVL_SHA256':record['FM_D']['UVL_SHA256'],
        'static_result_SHA256':sha(folder/'result.json'),
        'static_gap_SHA256':sha(folder/'static_gap.json'),
        'features':features,'rules':rules,
    }


def build_html(target, figure_ids, model, plan):
    sections=[]
    titles={
        'fig01_two_variability_spaces':'The advisor\'s three-column system',
        'fig02_feature_model_groups':'Two UVL feature models',
        'fig03_configuration_space':'Why exhaustive evaluation is impossible',
        'fig04_exact_fmp_distribution':'Valid configuration distribution',
        'fig05_static_gap':'Static gap and coverage assumptions',
        'fig06_feature_level_coverage':'Which leaves lack training proxies?',
        'fig07_sampling_pipeline':'Pairwise, uniform and active sampling',
        'fig08_rq_evidence_status':'RQ1–RQ4 evidence status',
        'fig09_interaction_data_required':'The interaction matrix to measure next',
    }
    for name in figure_ids:
        sections.append(f'<section><h2>{html.escape(titles[name])}</h2><img src="{name}.svg" alt="{name}"><p><a href="{name}.png">PNG</a> · <a href="{name}.svg">Editable vector SVG</a></p></section>')
    document='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CV261123 research figures</title>
<style>body{font:16px system-ui;background:#edf2f5;color:#183449;margin:0}header{background:#102d43;color:white;padding:38px max(24px,calc((100vw - 1200px)/2))}h1{margin:0}main{max-width:1200px;margin:24px auto;padding:0 18px}section{background:white;margin:24px 0;padding:20px 24px;border-radius:12px;box-shadow:0 1px 6px #d6e0e8}img{max-width:100%;height:auto;display:block;margin:16px auto}small{color:#5e7282}a{color:#117782}.notice{background:#fff2dc;padding:15px;border-radius:9px}</style>
<header><h1>CV261123 · Supervisor figures and sampling design</h1><p>Evidence-driven visualizations · RQ1–RQ4 · VisDrone/FlamaPy/YOLO</p></header><main>
<p class="notice"><b>Scientific boundary:</b> model enumeration and static coverage are saved evidence. Pairwise/uniform sampling is a design, not detector evaluation. No empirical gaps, trained ML rules, AP scores or RQ4 claims have been generated.</p>
<p><b>Source run:</b> ''' +html.escape(model['source_run'])+''' · <b>FM_P:</b> '''+f"{model['fmp_count']:,}"+''' · <b>FM_D:</b> '''+f"{model['fmd_count']:,}"+''' · <b>Pairwise design:</b> '''+str(plan['pairwise_budget'])+''' · <b>Uniform design:</b> '''+str(plan['uniform_budget'])+'''</p>
<p><a href="cv261123_supervisor_figures.pdf">Download all nine vector figures as a PDF</a> · <a href="first_500_pairs_PLAN_ONLY.csv">500 pair design (not executed)</a> · <a href="sampling_summary.json">Sampling provenance</a></p>
'''+''.join(sections)+'''<p><small>All numeric figures are regenerated by committed Python source from the saved experiment files and the exact FM_P configuration generator.</small></p></main></html>'''
    (target/'index.html').write_text(document,encoding='utf-8')


def execute(args):
    fmp=PUBLIC_ROOT/'experiments/cv261109/fm_p.uvl'
    fmp_source=PUBLIC_ROOT/'experiments/cv261109/model.py'
    fmd=PUBLIC_ROOT/'experiments/cv261116/fm_d.uvl'
    source=Path(args.static_run).resolve() if args.static_run else locate_static_run(
        PROJECT_ROOT/'datasets/local_runs/cv261116')
    evidence=read_evidence(source,fmp,fmd)
    module=load_fmp_module(fmp_source)
    configs=enumerate_configs(module)
    if evidence['fmp_count']!=len(configs):
        raise ValueError('FM_P model count differs from saved static audit.')
    plan=make_plan(module,configs,seed=args.seed)
    print(f"[CV261123] Evidence PASS: FM_P {len(configs):,}; FM_D {evidence['fmd_count']:,}; "
          f"proxy {evidence['proxy']}/{len(configs)}; feasible t=2 states {plan['feasible_pairwise_states']}.",flush=True)
    if args.check:
        print('[CV261123] Check only: no files or model inference.',flush=True)
        return None
    name=f'cv261123_figures_{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:8]}'
    destination=Path(args.output).resolve() if args.output else PROJECT_ROOT/'datasets/local_runs/cv261123'/name
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Refusing to overwrite saved output: '+str(destination))
    destination.mkdir(parents=True,exist_ok=True)
    write_plan_files(plan,destination)
    from figures import draw_all
    ids=draw_all(destination,module,fmd,evidence,evidence['features'],plan)
    build_html(destination,ids,evidence,plan)
    public={key:v for key,v in evidence.items() if key not in ('features','rules')}
    public.update({'status':'FIGURES_GENERATED','scope':'MODEL_AND_STATIC_EVIDENCE_PLUS_SAMPLING_DESIGN',
                   'figures':ids,'seed':args.seed,'python':platform.python_version(),
                   'pair_plan_status':'NOT_EXECUTED','empirical_gap':'NOT_MEASURED',
                   'learned_rules':'NOT_LEARNED','targeted_retraining':'NOT_RUN'})
    (destination/'chart_data.json').write_text(json.dumps(public,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    hashes={p.name:sha(p) for p in sorted(destination.iterdir()) if p.is_file()}
    manifest={'status':'PASS_FIGURES_ONLY','scope':'EVIDENCE_BASED_CHART_GENERATION',
              'source_RunID':evidence['source_run'],'sample_seed':args.seed,
              'files_SHA256':hashes,'figure_count':len(ids),
              'pair_design': '500_PLANNED_NOT_EXECUTED',
              'input_evidence_SHA256':{
                    'result.json':evidence['static_result_SHA256'],
                    'static_gap.json':evidence['static_gap_SHA256'],
                    'fm_p.uvl':evidence['FM_P_UVL_SHA256'],
                    'fm_d.uvl':evidence['FM_D_UVL_SHA256']}}
    (destination/'figures_manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print('[CV261123] Generated',len(ids),'figures and 500 PLANNED pairs (NO GPU evaluation).',flush=True)
    print('[CV261123] Open:',destination/'index.html',flush=True)
    return destination


def main():
    parser=argparse.ArgumentParser(description='Build figures from existing validated local evidence; no detector inference')
    parser.add_argument('--static-run',type=Path,help='Path to an existing CV261116 static run folder')
    parser.add_argument('--output',type=Path,help='Fresh output directory; default datasets/local_runs/cv261123/<RunID>')
    parser.add_argument('--seed',type=int,default=20261104)
    parser.add_argument('--check',action='store_true',help='Validate evidence and sampling without writing files')
    execute(parser.parse_args())

if __name__=='__main__':main()
