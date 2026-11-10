"""Check complete-grid evidence, source protection and read-only UI tests."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from protocol import ROOT,HERE,EXPERIMENT,load,save,source,sha256,write_atomic
from common import seal,verify_hashes
from full20_completion import path_for
from matched_followup import ANALYSIS_ID
from main.pipeline import new_run


def execute(analysis):
    start=time.perf_counter();verify_hashes(analysis,('summary.json','EVIDENCE_SOURCES.json','RESOURCE_RESULT.json','result.json'))
    summary=load(analysis/'summary.json');sources=load(analysis/'EVIDENCE_SOURCES.json')
    digest=hashlib.sha256(json.dumps(summary,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if digest!=load(analysis/'result.json')['summary_canonical_sha256']:raise ValueError('Summary digest differs.')
    worker=Path(sources['remainder_protocol']['path']).parent;plan=load(worker/'EXECUTION_PROTOCOL.json')
    verify_hashes(worker,('worker_result.json','supervisor.json','EVIDENCE_TREE.json'))
    execution_source=load(worker/'inputs.json')['source']
    if sha256(Path(execution_source['path']))!=execution_source['sha256'] or sha256(worker/'executed_source.py')!=execution_source['sha256']:
        raise ValueError('The executed worker source changed.')
    expected=sorted(r['id'] for r in load(Path(plan['GT']['path']))['records']);signatures={};checks=[]
    for row in sources['conditions']:
        folder=Path(row['folder']);hashes=verify_hashes(folder,('image_manifest.jsonl','metrics.json','bootstrap.json',
            'predictions.jsonl.gz','ground_truth.json','verification.json','configuration.json','timing.json'))
        spec=load(folder/'configuration.json');timing=load(folder/'timing.json')
        manifest=[json.loads(line) for line in (folder/'image_manifest.jsonl').read_text().splitlines()]
        if [r['image_id'] for r in manifest]!=expected or any(r['tensor']!={'NCHW':[1,3,spec['d']['resolution'],spec['d']['resolution']],'dtype':'torch.float32'} for r in manifest):
            raise ValueError('Actual input population or precision differs.')
        if not timing['FP32'] or timing['TF32']:raise ValueError('Inference precision changed.')
        signature=hashlib.sha256(json.dumps([(r['image_id'],r['pixel_sha256'],r['file_sha256']) for r in manifest]).encode()).hexdigest()
        pid=spec['p']['id']
        if pid in signatures and signatures[pid]!=signature:raise ValueError('One P used different pixels across settings.')
        signatures[pid]=signature
        if hashes['ground_truth.json']!=plan['GT']['sha256'] or sha256(Path(spec['d']['path']))!=spec['d']['sha256']:
            raise ValueError('GT or source checkpoint differs.')
        checks.append({'pair_id':spec['id'],'p_id':pid,'d_id':spec['d']['id'],'images':548,'actual_FP32_NCHW_verified':True,
            'prediction_SHA256':hashes['predictions.jsonl.gz'],'GT_SHA256':hashes['ground_truth.json'],
            'input_pixel_signature':signature,'metrics_SHA256':hashes['metrics.json']})
    if len(checks)!=summary['combined_measured_pairs'] or len({r['pair_id'] for r in checks})!=len(checks):
        raise ValueError('Repeated or missing measurement identity.')
    protected={}
    for key in ('FMP','FMD','GT','image_members','ZIP','generator','measure_pair_source'):
        record=plan[key];actual=sha256(Path(record['path']))
        if actual!=record['sha256']:raise ValueError('Protected input changed: '+key)
        protected[key]={'source':record,'current_SHA256':actual,'unchanged':True}
    out,_=new_run({'output_root':str(analysis.parent)},EXPERIMENT,'full20_acceptance')
    tests=[]
    for name,python,pattern in (('CPU',ROOT/'.venv/Scripts/python.exe','test_*.py'),
        ('dashboard',ROOT/'.venv_dashboard/Scripts/python.exe','test_dashboard.py')):
        command=[str(python),'-B','-m','unittest','discover','-s',str(HERE/'tests'),'-p',pattern,'-v']
        t=time.perf_counter();process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
        write_atomic(out/(name+'_tests.log'),process.stdout+process.stderr,out)
        tests.append({'name':name,'command':command,'exit_code':process.returncode,'seconds':time.perf_counter()-t})
    resource=load(analysis/'RESOURCE_RESULT.json');within=resource['global_elapsed_seconds_at_analysis']<=86400
    status='ENGINEERING_ACCEPTED_RESEARCH_PARTIAL' if all(r['exit_code']==0 for r in tests) and within else 'BLOCKED_ACCEPTANCE'
    save(out/'ACCEPTANCE.json',{'status':status,'measured_pairs':len(checks),'new_pairs':summary['new_measured_pairs'],
        'target_pairs':80,'images_per_pair':548,'all80_measured':len(checks)==80,'matched_original_P_pixels_verified':True,
        'unique_P':len(signatures),'formal_independent_pairs':0,'protected_inputs_unchanged':True,
        'original26_manifest':source(analysis.parent/ANALYSIS_ID/'SHA256SUMS.txt'),
        'within_total_day_budget_at_analysis':within,'tests':tests,'CPU_wall_seconds':time.perf_counter()-start,
        'scientific_scope':'Development-only finite20 subset; independent test, validated learner and augmentation efficacy remain unproven.'},out)
    save(out/'PAIR_CHECKS.json',checks,out);save(out/'PROTECTED_INPUTS.json',protected,out)
    save(out/'inputs.json',{'analysis':source(analysis/'SHA256SUMS.txt'),'source':source(Path(__file__))},out)
    write_atomic(out/'diary.md','# Complete-grid acceptance\n\nVerified per-P pixel identity across settings, actual tensors, raw-output/GT hashes, source protection and CPU/read-only UI behavior. Engineering acceptance does not complete the scientific questions.\n',out)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>Full20 evidence acceptance</h1><p>'+status+'</p><a href="ACCEPTANCE.json">Acceptance</a> · <a href="PAIR_CHECKS.json">Pixel and prediction identities</a>',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'FULL20_ACCEPTANCE','status':status},out,replace=True)
    seal(out)
    from full20_results import refresh_results
    refresh_results(analysis);print(out,flush=True)
    if status=='BLOCKED_ACCEPTANCE':raise RuntimeError('Acceptance failed; see preserved evidence.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analysis-run',required=True);a=p.parse_args();execute(path_for(a.analysis_run))
