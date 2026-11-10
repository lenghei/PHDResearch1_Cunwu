"""Separate label validity from CPU data-path seed reproducibility."""
import os
from pathlib import Path
import shutil
import time
from protocol import ROOT,EXPERIMENT,load,save,source,sha256,write_atomic
from common import seal,verify_hashes
from historical_review import table
from training_data_readiness import loader_smoke
from main.pipeline import new_run

AUDIT='vm36_supervisor_reproduction_training_data_cpu_v2_20261110T110058_057fb01c'


def execute():
    os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['YOLO_AUTOINSTALL']='false'
    start=time.perf_counter();root=ROOT/'datasets/local_runs'/EXPERIMENT;previous=root/AUDIT
    verify_hashes(previous,('TRAINING_DATA_GATE.json','DERIVED_LABEL_MANIFEST.json','TRAINING_IMAGE_LABEL_AUDIT.csv'))
    if load(previous/'TRAINING_DATA_GATE.json')['complete_images_audited']!=6471:raise ValueError('Full data audit unavailable.')
    labels=previous/'derived_labels_native10_v1'
    for name,digest in load(previous/'DERIVED_LABEL_MANIFEST.json').items():
        if sha256(labels/name)!=digest:raise ValueError('Derived label bytes changed.')
    out,_=new_run({'output_root':str(root)},EXPERIMENT,'loader_seed_diagnostic')
    shutil.copyfile(Path(__file__),out/'executed_source.py')
    shutil.copyfile(Path(__file__).with_name('training_data_readiness.py'),out/'data_path_source.py')
    records=table(previous/'TRAINING_IMAGE_LABEL_AUDIT.csv');results={}
    for mode,reseed in (('standard_startup',False),('reseed_after_dataset',True)):
        folder=out/mode;folder.mkdir()
        results[mode]=loader_smoke(folder,records,labels,reseed_after_build=reseed)
        save(folder/'RESULT.json',results[mode],out)
    strict=results['reseed_after_dataset']['repeat_exact']
    save(out/'SEED_DIAGNOSTIC.json',{'status':'CPU_SEED_ENGINEERING_VERIFIED_WITH_LIMITATIONS' if strict else 'SEED_REPEAT_UNVERIFIED',
        'standard_startup_repeat_exact':results['standard_startup']['repeat_exact'],'reseed_after_dataset_repeat_exact':strict,
        'configuration_change':'An isolated CPU harness reseeds after dataset construction; the production trainer and frozen protocol were not changed.',
        'cause':'Any startup/import/cache-state cause remains an engineering hypothesis; results record the actual sequence and pixel/target hashes.',
        'data_gate':'All6471 original images and candidate labels audited; original bytes unchanged.',
        'training_gate':'NOT_ADMITTED: no GPU optimizer/checkpoint test, initialization provenance or measured full-run budget.',
        'zero_area_verifier_accepts':results['reseed_after_dataset']['zero_height_fixture_accepted_by_installed_verifier'],
        'new_model_forward_backward':0,'new_GPU_tasks':0,'CPU_wall_seconds':time.perf_counter()-start},out)
    save(out/'inputs.json',{'full_data_audit':source(previous/'SHA256SUMS.txt'),'source':source(Path(__file__))},out)
    save(out/'nested_sha256.json',{str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.parent!=out},out)
    note=ROOT/'datasets/reportcn'/('vm36_supervisor_reproduction_加载种子诊断_'+out.name.split('_')[-1]+'.md')
    write_atomic(note,'# VM36 CPU数据路径种子诊断\n\nPROBLEM：同seed增强输出是否一致。\nWHY：全量标签通过不能代替训练路径可复现性。\nSOLUTION：只用16张私有图像副本，分别检查标准启动顺序与构建dataset后重新固定种子的顺序；原生产协议不改。\nWHY IS IT A GOOD SOLUTION：保存两次像素/标签SHA，而非只报通过。\n\n'+__import__('json').dumps(load(out/'SEED_DIAGNOSTIC.json'),ensure_ascii=False,indent=2)+'\n\n证据：'+str(out)+'\n',out)
    save(out/'REPORT_REFERENCE.json',source(note),out)
    write_atomic(out/'diary.md','# CPU seed diagnostic\n\nReused the frozen complete label audit. Separate small image copies expose standard and reseeded data-path behavior. This is not YOLO training or independent performance evidence.\n',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'LOADER_SEED_DIAGNOSTIC','status':load(out/'SEED_DIAGNOSTIC.json')['status']},out,replace=True)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>CPU loader seed diagnostic</h1><a href="SEED_DIAGNOSTIC.json">Result</a>',out)
    seal(out)
    from joint_analysis import refresh
    from matched_followup import ANALYSIS_ID
    refresh(root/ANALYSIS_ID);print(out,flush=True)


if __name__=='__main__':execute()
