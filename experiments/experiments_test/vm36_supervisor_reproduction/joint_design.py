"""Feature-only joint coverage design, seeded with accepted measurements."""
import argparse
import json
from pathlib import Path
import time
from protocol import ROOT,HERE,EXPERIMENT,load,save,source,write_atomic
from common import seal,verify_hashes
from historical_review import csv_file
from main.pipeline import new_run


def design(study):
    begin=time.perf_counter();verify_hashes(study,('EVIDENCE_SOURCES.json','RQ1_RQ4_NEXT_GATES.json'))
    refs=load(study/'EVIDENCE_SOURCES.json');original=load(Path(refs['original_protocol']['path']))
    from sampling import models
    from full_development import PREP
    fmp,_,prior=models();blocked=set(load(PREP/'IMPLEMENTATION_BINDINGS_PROPOSAL.json')['proposed_values_from_historical_implementation'])
    population=[c for c in prior.enumerate_configs(fmp) if not blocked.intersection(c)]
    detectors={s['d']['id']:s['d'] for s in original['pairs']}
    if len(detectors)!=4:raise ValueError('Expected four actual checkpoint/resolution settings.')
    records=[];configurations=[]
    for p in population:
        pid=fmp.configuration_id(p)
        for did,d in sorted(detectors.items()):
            features=list(p)+(['D_Checkpoint_ck01'] if d['checkpoint']=='ck01' else [])+(['D_Infer1280'] if d['resolution']==1280 else [])
            configurations.append(features)
            records.append({'pair_id':pid+'__'+did,'p_id':pid,'leaves':'|'.join(p),'d_id':did,
                'checkpoint':d['checkpoint'],'resolution':d['resolution']})
    leaves=sorted(fmp.ALL_LEAVES)+['D_Checkpoint_ck01','D_Infer1280']
    masks,feasible=prior.compute_binary_pair_masks(configurations,leaves)
    measured={r['pair_id'] for r in refs['conditions']}
    selected=[i for i,r in enumerate(records) if r['pair_id'] in measured]
    if len(selected)!=8:raise ValueError('Existing eight measurements do not match the design population.')
    uncovered=feasible
    for i in selected:uncovered&=~masks[i]
    after_reuse=feasible.bit_count()-uncovered.bit_count()
    while uncovered:
        gains=[(mask&uncovered).bit_count() for mask in masks]
        best=gains.index(max(gains))
        if max(gains)==0 or best in selected:raise ValueError('Joint coverage cannot progress.')
        selected.append(best);uncovered&=~masks[best]
    out,_=new_run({'output_root':str(study.parent)},EXPERIMENT,'joint_design')
    csv_file(out/'JOINT_COVERAGE_PLAN.csv',[{**records[i],'role':'REUSE_ACCEPTED' if records[i]['pair_id'] in measured else 'PROPOSED_NOT_RUN'} for i in selected])
    # Costs are read only after the feature-only selection is frozen.
    times=[load(Path(r['pair_folder'])/'timing.json')['wall_seconds'] for r in refs['conditions']]
    additional=len(selected)-8
    result={'status':'CPU_DESIGN_VERIFIED_GPU_RESOURCE_BLOCKED','full_legal_FMP':46368,'restricted_FMP':len(population),
        'actual_detector_settings':4,'restricted_joint_population':len(records),'binary_features':len(leaves),
        'feature_names':leaves,'feasible_pairwise_states':feasible.bit_count(),'states_covered_by_existing_eight':after_reuse,
        'reused_pairs':8,'new_pairs_required_by_greedy_design':additional,'total_design_pairs':len(selected),
        'coverage_complete_for_stated_projection':uncovered==0,'optimization':'Deterministic greedy after seeding accepted pairs; global minimum not claimed.',
        'scope':'FM_P admitted stratum crossed with four realized checkpoint/resolution settings; not all theoretical FM_D configurations.',
        'checkpoint_feature':'Identity contrast, not an architecture-causality intervention.',
        'AP_used_for_selection':False,'new_GPU_evaluations':0,'new_full548_image_inferences_if_authorized':additional*548,
        'screening_new_worker_hours':additional*sum(times)/len(times)/3600,
        'conservative_screening_hours':additional*max(times)*1.25/3600,
        'budget_admission':'No new GPU budget allocated; the two-corner worker has ended. Renderer combination admission and learning-label plan still required.'}
    save(out/'DESIGN_ACCEPTANCE.json',result,out)
    save(out/'protocol.json',{'RQ':'RQ1 coverage; RQ2 future measured interactions',
        'necessity':'Derive cross-domain coverage rather than multiplying the FM_P core by every detector setting.',
        'sampling':'Constraint-aware feature-only greedy binary t=2, seeded by eight accepted pairs.',
        'acceptance':'All feasible states of the precisely restricted projection covered; no repeated pair IDs; no AP-driven selection.',
        'dataset_if_executed':'All548 VisDrone validation images per new pair; development only.',
        'not_executed':'GPU measurements, gap prevalence estimation, grouped learned-rule validation and augmentation training.'},out)
    save(out/'inputs.json',{'source':source(study/'SHA256SUMS.txt'),'model':source(HERE/'feature_models/fm_p.uvl'),
        'method_source':source(Path(__file__))},out)
    write_atomic(out/'diary.md','# Joint coverage preparation\n\nSelected only from legal parameter-admitted configurations and four actual detector settings. Eight accepted pairs seed coverage. AP responses did not guide selection. No new GPU execution or global-minimum claim.\n',out)
    text=['# VM36跨域二阶覆盖设计','',
        '1–2. Research problem/necessity: RQ1/RQ2；在明确的两域投影中补足二阶状态，而不是机械执行21×4组。',
        '3–7. Dataset/models/parameters: 若执行，每组完整548张val，原生十类；仍使用现有ck01/ck05、640/1280和原始UVL参数；本次CPU不测AP。',
        '8–9. Sampling/controls: 仅依据配置特征；已验收八组作为免费覆盖起点；确定性贪心补点，未读取AP选点。',
        f'10. Actual design: {len(records)}个候选联合配置；{feasible.bit_count()}个可行二阶状态；复用8组，新增候选{additional}组，合计{len(selected)}组。',
        '11–12. Metrics/intervals: NOT_EVALUATED；这是设计覆盖验收，不是新增检测结果。',
        f'13. CPU time: {time.perf_counter()-begin:.3f}秒。新增完整worker成本粗估{result["screening_new_worker_hours"]:.2f}小时，不是GPU占用实测。',
        '14. Supported: 该受限投影的状态覆盖完整；不保证故障发现率，不声称贪心为全局最小。',
        '15. Limits: 不是整个FM_P×FM_D；未分配下一批GPU预算，组合渲染与学习标签关卡仍保留。',
        '16. Evidence: '+str(out)+'；joint_design.py --analysis-run '+study.name,
        '', 'PROBLEM：只覆盖FM_P不能冒称已覆盖两域。',
        'WHY IS IT A PROBLEM?：固定乘法会掩盖真实覆盖缺口或引入无必要实验。',
        'SOLUTION：显式枚举可实现的联合投影，复用已测角点后补充状态见证。',
        'WHY IS IT A GOOD SOLUTION?：覆盖目标、点集和成本可核查，未把设计当成执行结果。']
    path=ROOT/'datasets/reportcn'/('vm36_supervisor_reproduction_跨域覆盖设计_'+out.name.split('_')[-1]+'.md')
    write_atomic(path,'\n'.join(text)+'\n',out);save(out/'REPORT_REFERENCE.json',source(path),out)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>VM36 joint coverage design</h1><p>CPU design verified; GPU not run.</p><a href="DESIGN_ACCEPTANCE.json">Coverage and resources</a>',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'JOINT_COVERAGE_DESIGN',
        'status':'GPU_RESOURCE_BLOCKED','CPU_seconds':time.perf_counter()-begin},out,replace=True)
    seal(out)
    from factorial_analysis import refresh_history
    refresh_history(study)
    print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--analysis-run',required=True);args=parser.parse_args()
    if Path(args.analysis_run).name!=args.analysis_run:raise ValueError('Expected a plain RunID.')
    design(ROOT/'datasets/local_runs'/EXPERIMENT/args.analysis_run)
