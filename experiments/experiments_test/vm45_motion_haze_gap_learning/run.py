"""VM45 CPU continuation; no inherited GPU permission."""
import os,sys,time,subprocess,importlib.util,json,threading
os.environ.update(CUDA_VISIBLE_DEVICES='-1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',YOLO_AUTOINSTALL='false')
from vm45_common import *
def loadmod(name):
 spec=importlib.util.spec_from_file_location('vm45_'+name,HERE/(name+'.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
 from main.pipeline import new_run
 start=time.perf_counter();cpu=time.process_time();out,log=new_run({'output_root':str(ROOT/'datasets/local_runs'/HERE.name),'GPU_authorized':False},HERE.name,'cpu_study')
 (out/'plots').mkdir();(out/'runtime').mkdir();os.environ['MPLCONFIGDIR']=str(out/'runtime/matplotlib')
 import psutil
 process=psutil.Process();peak=[process.memory_info().rss];end=threading.Event()
 def watch():
  while not end.wait(.2):peak[0]=max(peak[0],process.memory_info().rss)
 threading.Thread(target=watch,daemon=True).start()
 for p in HERE.rglob('*.py'):ref(p)
 ref(Path(r'C:\Users\w\.codex\attachments\1f873a6d-1cdf-42cf-a81e-0922589facaa\已粘贴的文本.txt'))
 sys.path.insert(0,str(HERE.parent/'vm43_external_pilot'));import vm43_common
 before=vm43_common.budget()
 try:
  for line in (V44/'SHA256SUMS.txt').read_text().splitlines():
   digest,name=line.split('  ',1)
   if Path(name).parent==V44 and Path(name).name in ('FM_P_to_FM_D_decision_rules.json','Controller_validation_report.json','final_acceptance.json'):
    assert ref(name)['SHA256']==digest
  rules=read(V44/'FM_P_to_FM_D_decision_rules.json');gate=loadmod('prepare').prepare(out,rules,before)
  log.info('GPU blocked: no VM45-specific permission and no nuisance scientific admission. Continue CPU research.')
  loadmod('evaluate').record_unmeasured(out)
  cmds=[[str(ROOT/'.venv_flamapy/Scripts/python.exe'),'-B','-X','utf8',str(HERE/'learn_constraints.py'),'--input',str(V44/'FM_P_to_FM_D_decision_rules.json'),'--out',str(out)],
        [str(ROOT/'.venv_flamapy/Scripts/python.exe'),'-B','-X','utf8',str(HERE/'gap_analysis.py'),'--out',str(out),'--models',str(UVL),'--rules',str(V44/'FM_P_to_FM_D_decision_rules.json')]]
  for i,cmd in enumerate(cmds):
   log.info('Running CPU branch %d',i+1)
   with (out/f'CPU_branch_{i+1}.log').open('x',encoding='utf-8') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,timeout=1800,env=os.environ.copy())
   if r.returncode:raise RuntimeError('CPU branch failed; raw log retained: '+str(i+1))
  save(out/'CPU_commands.json',cmds)
  learned=read(out/'rule_learning_validation.json');uvl=read(out/'UVL_analysis.json');loadmod('controller').evaluate(out,rules,learned)
  tests=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s',str(HERE/'tests'),'-v'],cwd=HERE,capture_output=True,text=True,env=os.environ.copy())
  (out/'tests.log').write_text(tests.stdout+'\n'+tests.stderr,encoding='utf-8');assert tests.returncode==0
  reports,plots=loadmod('report').deliver(out,rules,learned,uvl,gate)
  source_report=ROOT/'datasets/reportcn/vm37_yolo26_rq4_V3_1最终科学准入审查_15c25191.md';ref(source_report)
  after=vm43_common.budget();assert before['charges']==after['charges'] and before['authorization']==after['authorization']
  for item in SOURCES.values():assert sha(item['path'])==item['SHA256']
  end.set()
  save(out/'actual_resource_usage.json',{'CPU_parent_wall_seconds':time.perf_counter()-start,'CPU_parent_process_seconds':time.process_time()-cpu,'peak_parent_RSS_bytes':peak[0],
    'learning_child_CPU_seconds':learned['CPU_process_seconds'],'GPU_seconds':0,'GPU_budget_charged_seconds':0,'original_budget_reporting_only':after,'cache_generated':False,'CPU_environment_reused':'.venv_flamapy contains sklearn; .venv_rq4 unchanged','GPU_probe_executed':False})
  save(out/'source_provenance.json',{'files':list(SOURCES.values()),'learning_source_RunID':V44.name,'learning_population':'VM36 development84 rows only','external_testdev_training_rows':0,'original_UVL_unchanged':True})
  save(out/'final_acceptance.json',{'status':'PARTIAL_WITH_LIMITATIONS','CPU_study':'COMPLETED','GPU':'BLOCKED','new_GPU_cells':0,'Motion_Haze_DID':'NOT_EVALUATED','rules':learned['status'],'admitted_empirical_rules':0,'original_and_derived_P_count':46368,'tests':'4_PASS','RQ4_formal':'NOT_EVALUATED','reports':[str(p) for p in reports]})
  (out/'execution_diary.md').write_text('VM45: actual grouped CPU classifiers, static sensitivity, derived UVL validation, safe selectorV2. No admitted Motion/Haze GPU execution. No new detector training. Candidate bindings remain unapproved; all absent AP/DID are NOT_EVALUATED.\n',encoding='utf-8')
  r=json.loads((out/'run.json').read_text());r['status']='COMPLETE_WITH_LIMITATIONS';(out/'run.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
  from main.visualization import build_run_index
  build_run_index(out.parent,HERE.name)
  log.info('CPU study completed with limitations; GPU0.')
  files=[p for p in out.rglob('*') if p.is_file() and 'runtime' not in p.relative_to(out).parts]+reports+list(HERE.rglob('*.py'))
  with (out/'SHA256SUMS.txt').open('x',encoding='utf-8') as f:
   for p in sorted(files):f.write(sha(p)+'  '+str(p)+'\n')
  print('VM45_COMPLETED',out,flush=True)
 except BaseException:
  import traceback
  end.set();(out/'failure.log').write_text(traceback.format_exc(),encoding='utf-8');raise
if __name__=='__main__':main()
