"""Bounded CPU checks for follow-up geometry, labels and research gates."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from zipfile import ZipFile
import numpy as np
from protocol import ROOT,HERE,EXPERIMENT,load,save,source,sha256,write_atomic
from full_development import PREP
from common import seal,verify_hashes
from historical_review import csv_file,table
from main.pipeline import new_run


def execute(study):
    started=time.perf_counter()
    verify_hashes(study,('EVIDENCE_SOURCES.json','EVIDENCE_ACCEPTANCE.json','RQ1_RQ4_NEXT_GATES.json'))
    if load(study/'EVIDENCE_ACCEPTANCE.json')['status']!='VALIDATED':raise ValueError('Factorial acceptance failed.')
    out,_=new_run({'output_root':str(study.parent)},EXPERIMENT,'research_acceptance')
    refs=load(PREP/'full548_data_references.json');manifest_path=Path(refs['complete_split_members.csv']['path'])
    if sha256(manifest_path)!=refs['complete_split_members.csv']['sha256']:raise ValueError('Training source manifest changed.')
    members=[r for r in table(manifest_path) if r['split']=='train']
    if len(members)!=6471:raise ValueError('Incomplete training membership.')
    certificate=load(PREP/'DATASET_ADMISSION.json')['splits']['train']['archive']
    archive_path=Path(certificate['path'])
    if sha256(archive_path)!=certificate['sha256']:raise ValueError('Training archive changed.')
    audit=[];changes=[]
    with ZipFile(archive_path) as archive:
        for member in members:
            if time.perf_counter()-started>240:raise TimeoutError('CPU audit budget reached; no training was started.')
            raw=archive.read(member['ZIP_annotation_member'])
            if hashlib.sha256(raw).hexdigest()!=member['annotation_SHA256']:raise ValueError('Annotation source differs.')
            keep=[];bad=[]
            for line_number,line in enumerate(raw.splitlines(keepends=True),1):
                text=line.decode('utf-8-sig').strip()
                if not text:keep.append(line);continue
                values=[float(x) for x in text.split(',') if x.strip()]
                if len(values)<8 or not np.isfinite(values).all():raise ValueError('Unexpected annotation format.')
                if values[2]<=0 or values[3]<=0:
                    bad.append(line_number)
                    changes.append({'image':member['image_id'],'physical_line':line_number,'original':text,
                        'width':values[2],'height':values[3],'score':values[4],'VisDrone_class_id':values[5],
                        'proposal':'EXCLUDE_NONPOSITIVE_AREA_IN_A_NEW_VERSION_ONLY','applied_to_original':False})
                else:keep.append(line)
            filtered=b''.join(keep)
            audit.append({'image':member['image_id'],'original_SHA256':member['annotation_SHA256'],
                'proposed_view_SHA256':hashlib.sha256(filtered).hexdigest(),'excluded_rows':len(bad),
                'image_ID_retained':True,'original_modified':False})
    if len(changes)!=3:raise ValueError('The expected three-row exception inventory changed.')
    csv_file(out/'TRAINING_ANNOTATION_AUDIT.csv',audit);csv_file(out/'PROPOSED_LABEL_EXCLUSIONS.csv',changes)
    save(out/'TRAINING_LABEL_POLICY_DRAFT.json',{'status':'CPU_VERIFIED_PROPOSAL_NOT_ACTIVE','training_images':6471,
        'annotations_hashed':6471,'files_with_proposed_changes':sum(r['excluded_rows']>0 for r in audit),
        'excluded_nonpositive_area_rows':3,'score_positive_native_class_rows':sum(r['score']>0 and 1<=r['VisDrone_class_id']<=10 for r in changes),
        'source_archive':certificate,'originals_modified':False,'derived_training_labels_written':False,
        'same_policy_required_for_all_arms':True,'training_admission':'NOT_GRANTED',
        'remaining':['Freeze the loader/derived-view version and verify full training image decoding.',
            'Match initialization, optimizer, schedule, image IDs, seeds and fixed checkpoint selection for all arms.',
            'Resolve augmentation range/orientation/airlight definitions and record a measured finite training budget.']},out)
    geometry=[]
    for width,height in ((960,540),(1360,765),(1920,1080),(2000,1500)):
        for factor in (1.5,2.0):
            low_w,low_h=max(1,round(width/factor)),max(1,round(height/factor))
            points=np.array([[0.,0.],[width-1.,height-1.],[width/3,height/3]])
            scale=np.array([low_w/width,low_h/height])
            restored=((points+.5)*scale-.5+.5)/scale-.5
            error=float(np.max(np.abs(restored-points)))
            if error>1e-10:raise ValueError('Resize coordinate-composition check failed.')
            geometry.append({'width':width,'height':height,'factor':factor,'intermediate_width':low_w,
                'intermediate_height':low_h,'coordinate_composition_max_error':error})
    csv_file(out/'DOWNSCALE_COORDINATE_CHECK.csv',geometry)
    save(out/'GEOMETRY_SCOPE.json',{'status':'CPU_COORDINATE_PROPERTIES_VERIFIED','source':source(HERE.parent/'cv261109/generator.py'),
        'downscale':'The existing INTER_AREA downsample and INTER_LINEAR return to the original canvas; centre-coordinate scale composition is identity.',
        'limitations':'This verifies the coordinate convention, not preservation of localization information, physical realism or every combined recipe.',
        'rolling_shutter':'BLOCKED_FOR_AP; no validated annotation/ignore warp',
        'missing_motion_orientation':'UNCHANGED_BLOCKER; no direction invented',
        'renderer_smoke':source(study/'RENDERER_SMOKE.json')},out)
    tests=[]
    for name,python,pattern in [('CPU',ROOT/'.venv/Scripts/python.exe','test_*.py'),
        ('DASHBOARD',ROOT/'.venv_dashboard/Scripts/python.exe','test_dashboard.py')]:
        p=subprocess.run([str(python),'-B','-m','unittest','discover','-s',str(HERE/'tests'),'-p',pattern,'-v'],
            capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
        write_atomic(out/(name+'_TESTS.log'),p.stdout+'\n'+p.stderr,out)
        if p.returncode:raise ValueError('Regression failed: '+name)
        tests.append({'test_suite':name,'returncode':p.returncode})
    gates=load(study/'RQ1_RQ4_NEXT_GATES.json')
    save(out/'RESEARCH_DECISION.json',{'status':'COMPLETE_WITH_LIMITATIONS','source_RunID':study.name,
        'factorial_conditions_accepted':8,'new_GPU_experiments':0,'new_training_runs':0,
        'training_annotation_proposal':'6471 hashes checked; three zero-height rows isolated; original data unchanged; proposal not activated.',
        'geometry':'Down/up coordinate-frame composition checked; rolling GT remains blocked.',
        'research_gates':gates,'tests':tests,'CPU_wall_seconds':time.perf_counter()-started,
        'next_executable_branch':'Freeze a scientifically accepted mission threshold/precision and separate finite resources before a uniform gap study; resolve augmentation bindings/label-view admission before matched training.',
        'independent_testdev':'DATASET_BLOCKED; no new source evidence establishes historical independence.'},out)
    save(out/'inputs.json',{'source_analysis':source(study/'SHA256SUMS.txt'),'training_members':source(manifest_path),
        'source_code':source(Path(__file__))},out)
    report=['# VM36后续研究准入','',
        '1. Research problem: RQ1的坐标约定、RQ4的训练标签异常与后续准入。',
        '2. Why necessary: 避免将未准入的渲染配置或零面积训练框直接用于后续实验。',
        '3–4. Dataset/split/count: VisDrone2019-DET train的6471份原始标注全部核对SHA；未重新解码训练图片。',
        '5–7. Models/resolution/degradation: 本CPU分支未加载检测器、未做推理；所有模型和退化参数保持已有冻结版本。',
        '8–9. Method/controls: 对照原始完整成员清单；仅读取原ZIP；仅计算派生标签草案的SHA，不写入或覆盖标签。',
        '10. Actual checks: 6471份标注；定位3个零高度框，其中1个score>0原生目标、2个ignore行。8项down/up坐标组合检查通过。',
        '11–12. AP/intervals: NOT_EVALUATED，本分支不产生新的AP或统计区间。',
        f'13. CPU runtime: {time.perf_counter()-started:.3f}秒。测试日志见本Run。GPU=0。',
        '14. Supported: 非正面积排除草案可复现；原图坐标的缩放复合约定已检查。',
        '15. Limitations: 草案不是训练准入；未证明物理传感器或所有组合有效。独立test-dev、参数绑定、三seed增强和总体gap预算仍有阻塞。',
        '16. Evidence/reproduction: '+str(out)+'；research_acceptance.py --analysis-run '+study.name,
        '', 'PROBLEM：数据和几何边界影响后续实验有效性。',
        'WHY IS IT A PROBLEM?：零高度框及未定义的rolling标签变换不能被忽略。',
        'SOLUTION：只读核对全部标注、记录异常、验证坐标约定并保留版本边界。',
        'WHY IS IT A GOOD SOLUTION?：修正方案可检查，未通过新训练或改原始数据来掩盖异常。']
    path=ROOT/'datasets/reportcn'/('vm36_supervisor_reproduction_后续研究准入_'+out.name.split('_')[-1]+'.md')
    write_atomic(path,'\n'.join(report)+'\n',out);save(out/'REPORT_REFERENCE.json',source(path),out)
    write_atomic(out/'diary.md','# Follow-up CPU acceptance\n\nHashed all6471 training annotations, isolated three nonpositive-area rows and checked resize-coordinate composition. No original labels changed. Training, independent testing and population-level gap admission remain separate.\n',out)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>VM36 follow-up CPU evidence</h1><p>Complete with limitations; no GPU or training.</p><a href="RESEARCH_DECISION.json">Decision</a> · <a href="TRAINING_LABEL_POLICY_DRAFT.json">Label proposal</a>',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'RESEARCH_CPU_ACCEPTANCE',
        'status':'COMPLETE_WITH_LIMITATIONS','ended_UTC':datetime.now(timezone.utc).isoformat()},out,replace=True)
    seal(out)
    from factorial_analysis import refresh_history
    refresh_history(study)
    print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--analysis-run',required=True);args=parser.parse_args()
    if Path(args.analysis_run).name!=args.analysis_run:raise ValueError('Expected a plain RunID.')
    execute(ROOT/'datasets/local_runs'/EXPERIMENT/args.analysis_run)
