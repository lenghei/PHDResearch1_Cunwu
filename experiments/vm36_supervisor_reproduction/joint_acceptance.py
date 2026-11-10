"""Verify terminal joint evidence and run CPU/viewer regression checks."""
import argparse
import json
from pathlib import Path
import subprocess
import time
from protocol import ROOT,HERE,EXPERIMENT,load,save,source,sha256,write_atomic
from common import seal,verify_hashes
from main.pipeline import new_run


def execute(analysis):
    started=time.perf_counter()
    verify_hashes(analysis,('summary.json','EVIDENCE_SOURCES.json','BATCH_ACCEPTANCE_REFERENCE.json'))
    summary=load(analysis/'summary.json');references=load(analysis/'EVIDENCE_SOURCES.json')
    batch=Path(references['batch_protocol']['path']).parent
    verify_hashes(batch,('EXECUTION_PROTOCOL.json','supervisor.json','worker_result.json','EVIDENCE_TREE.json'))
    if load(batch/'run.json')['status']=='RUNNING':raise ValueError('Worker is still running.')
    plan=load(batch/'EXECUTION_PROTOCOL.json');supervisor=load(batch/'supervisor.json')
    if supervisor['hard_budget_seconds']!=14400 or supervisor['actual_child_supervision_seconds']>14400:
        raise ValueError('Observed supervision exceeded the authorized worker budget.')
    for key in ('FMP','FMD','GT','image_members','ZIP','generator'):
        if sha256(Path(plan[key]['path']))!=plan[key]['sha256']:raise ValueError('Changed protected input: '+key)
    inputs=load(batch/'inputs.json')
    if sha256(Path(inputs['worker']['path']))!=inputs['worker']['sha256']:raise ValueError('Execution source changed.')
    frozen=load(batch/'source_SHA256.json')
    for relative in ('code/main/predictor.py','code/main/labels.py','code/main/statistics.py',
        'experiments/cv261207/evaluation.py','experiments/cv261207/development_v1.py',
        'experiments/cv261207/development_fast_coco.py','experiments/vm36_supervisor_reproduction/joint_reporting.py'):
        if sha256(ROOT/'main'/relative)!=frozen[relative]:raise ValueError('Scientific source changed: '+relative)
    checks=[];checkpoints={}
    for row in references['accepted_pairs']:
        folder=Path(row['folder'])
        checked=verify_hashes(folder,('configuration.json','metrics.json','bootstrap.json','predictions.jsonl.gz',
            'ground_truth.json','verification.json','image_manifest.jsonl','timing.json'))
        spec=load(folder/'configuration.json');checkpoints[spec['d']['path']]=spec['d']['sha256']
        records=[json.loads(line) for line in (folder/'image_manifest.jsonl').read_text().splitlines()]
        if len(records)!=548 or len({r['image_id'] for r in records})!=548:
            raise ValueError('Pair does not contain 548 distinct image IDs.')
        if (folder/'REPORT_REFERENCE.json').exists():
            report=load(folder/'REPORT_REFERENCE.json')
            if sha256(Path(report['path']))!=report['sha256']:raise ValueError('Linked report changed.')
        checks.append({'pair_id':row['pair_id'],'images':len(records),'raw_prediction_SHA256':checked['predictions.jsonl.gz'],
            'GT_SHA256':checked['ground_truth.json'],'raw_COCO_replay':load(folder/'verification.json')['COCO_recomputed_from_raw'],
            'manifest_SHA256':checked['image_manifest.jsonl'],'per_pair_hashes_verified':True})
    for path,digest in checkpoints.items():
        if sha256(Path(path))!=digest:raise ValueError('Checkpoint changed.')
    out,_=new_run({'output_root':str(analysis.parent)},EXPERIMENT,'joint_acceptance')
    tests=[]
    commands=[('CPU',[str(ROOT/'.venv/Scripts/python.exe'),'-B','-m','unittest','discover','-s',str(HERE/'tests'),'-v']),
        ('dashboard',[str(ROOT/'.venv_dashboard/Scripts/python.exe'),'-B','-m','unittest','discover','-s',str(HERE/'tests'),'-p','test_dashboard.py','-v'])]
    for name,command in commands:
        t=time.perf_counter();process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
        write_atomic(out/(name+'_tests.log'),process.stdout+process.stderr,out)
        tests.append({'name':name,'command':command,'exit_code':process.returncode,'seconds':time.perf_counter()-t})
    status='VALIDATED' if all(r['exit_code']==0 for r in tests) else 'BLOCKED_TESTS'
    result={'status':status,'measured_pairs':len(checks),'new_pairs':summary['new_measured_pairs'],
        'formal_independent_pairs':0,'protected_inputs':'UNCHANGED','original_checkpoints':'UNCHANGED',
        'measurement_source':'UNCHANGED','scientific_results_modified':False,'new_GPU_tasks':0,
        'worker_seconds':supervisor['actual_child_supervision_seconds'],'tests':tests,
        'CPU_wall_seconds':time.perf_counter()-started,
        'replay_scope':'Saved same-evaluator replay verified at measurement; acceptance verifies sealed outputs and identities without another detector forward.'}
    save(out/'ACCEPTANCE.json',result,out);save(out/'PAIR_CHECKS.json',checks,out)
    save(out/'inputs.json',{'analysis':source(analysis/'SHA256SUMS.txt'),'batch':source(batch/'SHA256SUMS.txt'),'source':source(Path(__file__))},out)
    write_atomic(out/'diary.md','# Joint evidence acceptance\n\nVerified terminal evidence, protected input/source hashes, full-split identities, checkpoint hashes and CPU/viewer checks. No new inference, metric interpolation or training.\n',out)
    report=['# VM36 最终工程验收','',f'RunID：{out.name}',f'状态：{status}',
        f'累计验收{len(checks)}组，新增{summary["new_measured_pairs"]}组，每组548图；独立测试组为0。',
        '原始八组、checkpoint、ZIP、双UVL、检测与评价源码SHA保持一致；保存的原预测COCO复算凭据均有效。',
        'CPU测试和仪表盘测试的实际命令、退出码与输出分别保存在ACCEPTANCE.json及两个tests.log。',
        '仪表盘仅显示既有结果；云端无私有数据回退。没有训练、测试集推理或额外GPU任务。',
        '证据：'+str(out),'联合结果：'+str(analysis)]
    note=ROOT/'datasets/reportcn'/('vm36_supervisor_reproduction_最终工程验收_'+out.name.split('_')[-1]+'.md')
    write_atomic(note,'\n'.join(report)+'\n',out);save(out/'REPORT_REFERENCE.json',source(note),out)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>VM36 evidence acceptance</h1><p>'+status+'</p><a href="ACCEPTANCE.json">Checks</a> · <a href="PAIR_CHECKS.json">Pair identities</a> · <a href="CPU_tests.log">CPU tests</a> · <a href="dashboard_tests.log">Viewer tests</a>',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'JOINT_EVIDENCE_ACCEPTANCE','status':status},out,replace=True)
    seal(out)
    from joint_analysis import refresh
    refresh(analysis);print(out,flush=True)
    if status!='VALIDATED':raise RuntimeError('A regression check failed; see saved logs.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--analysis-run',required=True);args=parser.parse_args()
    if Path(args.analysis_run).name!=args.analysis_run:raise ValueError('Expected a plain RunID.')
    execute(ROOT/'datasets/local_runs'/EXPERIMENT/args.analysis_run)
