"""CPU archive checks after the supervised process and its logs have closed."""
import argparse,csv,gzip,json,math,shutil,time
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from motion_common import *
from motion_evaluation import csv_write,area

def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def archive(out):
    started=time.perf_counter();check(out)
    if (out/'archive_acceptance.json').exists():raise FileExistsError('Already archived.')
    accepted=load(out/'final_acceptance.json');assert accepted['both_arms_completed'] and accepted['complete_evaluation_cells']==8
    import psutil
    for stage in ['training','evaluation']:
        r=load(out/(stage+'_supervision/process.json'))
        assert not psutil.pid_exists(r['supervisor_PID']),'Wait for the parent process to exit.'
        s=load(out/(stage+'_supervision/supervisor.json'));assert s['returncode']==0 and s['stop_reason'] is None and s['maximum_observed_sample_interval_seconds']<=2
    old=load(out/'old_budget_READ_ONLY_snapshot.json');assert all(sha(r['path'])==r['SHA256'] for r in old['files'])
    old_paths={str(Path(r['path']).resolve()) for r in old['files']};auth=next(Path(r['path']) for r in old['files'] if Path(r['path']).name=='authorization.json')
    assert {str(p.resolve()) for p in (auth.parent/'charges').glob('*.json')}=={p for p in old_paths if Path(p).parent.name=='charges'}
    order={};traces={};epoch_counts={}
    for arm in ['CONTROL','MOTION_TARGETED']:
        with (out/arm/'training_batches.jsonl').open(encoding='utf-8') as f:batch=[json.loads(s) for s in f]
        order[arm]=[(r['epoch'],r['image_ids']) for r in batch];epoch_counts[arm]={ep:sum(len(r['image_ids']) for r in batch if r['epoch']==ep) for ep in range(1,11)}
        assert len(batch)==32360 and all(v==6471 for v in epoch_counts[arm].values())
        traces[arm]=sha(out/arm/'ordinary_augmentation_events.jsonl')
    assert order['CONTROL']==order['MOTION_TARGETED'] and traces['CONTROL']==traces['MOTION_TARGETED']
    matching={'all10_epoch_sample_order_equal':True,'all10_epoch_ordinary_Albumentations_trace_equal':True,'ordinary_trace_SHA256':traces,'epoch_image_counts':epoch_counts,
        'live_pause':load(out/'CONTROL_live_pause_preservation.json'),'seed':45,'no_new_GPU_or_COCO':True}
    save(out/'matched_sequence_acceptance.json',matching)
    aug=read_csv(out/'augmentation_event_log.csv');active=[r for r in aug if r['active']=='True']
    assert all(r['strategy']=='MOTION_TARGETED' and int(r['kernel']) in [5,9,13,17,25] and r['pixel_changed']=='True' and 0<=float(r['angle_deg'])<180 for r in active)
    assert len(aug)==129420 and all(sum(r['strategy']==a for r in aug)==64710 for a in ['CONTROL','MOTION_TARGETED'])
    angle_counts,edges=np.histogram([float(r['angle_deg']) for r in active],bins=np.arange(0,181,10))
    csv_write(out/'augmentation_angle_histogram.csv',[{'seed':45,'low_degrees':int(edges[i]),'high_exclusive_degrees':int(edges[i+1]),'count':int(v)} for i,v in enumerate(angle_counts)])
    rows=read_csv(out/'configuration_metrics.csv');comparisons=read_csv(out/'paired_AP_comparisons.csv');transitions=read_csv(out/'tiny_gap_transitions.csv');assert len(rows)==8
    reference_ids=load(out/'paired_bootstrap_indices.json')['image_ids'];bindings=[]
    for r in rows:
        folder=out/'raw_predictions'/(r['strategy']+'__'+r['condition']);a=load(folder/'acceptance.json');m=load(folder/'metrics.json')
        assert int(r['images'])==548 and float(r['AP50_95'])==area(m,'all','map50_95') and float(r['Tiny_AP50_95'])==area(m,'tiny','map50_95')
        receipt=load(out/r['strategy']/'training_acceptance.json');assert r['checkpoint_SHA256']==receipt['selected_checkpoint']['SHA256']
        assert load(folder/'warmup.json')['count']==3
        for key in ['raw','metrics','bootstrap','timing']:assert sha(a[key]['path'])==a[key]['SHA256']
        with gzip.open(a['raw']['path'],'rt',encoding='utf-8') as f:ids=[json.loads(s)['image_id'] for s in f]
        assert ids==reference_ids and len(set(ids))==548
        bindings.append({'strategy':r['strategy'],'condition':r['condition'],'checkpoint':receipt['selected_checkpoint'],'source_metrics':source(folder/'metrics.json'),'P_id':r['P_id']})
    clarification=load(out/'evaluation_identity_clarification.json')
    assert datetime.fromisoformat(clarification['created_UTC'])<datetime.fromisoformat(load(out/'evaluation_supervision/process.json')['started_UTC'])
    models=[{'strategy':a,'checkpoint':load(out/a/'training_acceptance.json')['selected_checkpoint'],'seed':45,'architecture':'YOLO26s','selection':'FIXED_EPOCH10_EMA'} for a in ['CONTROL','MOTION_TARGETED']]
    original=load(out/'evaluation_protocol.json');resolved={**original,'models':models,'cell_bindings':bindings,'order':clarification['actual_order'],'original_frozen_protocol':source(out/'evaluation_protocol.json'),
        'metadata_clarification':source(out/'evaluation_identity_clarification.json'),'view_only_not_a_new_experiment_or_posthoc_selection':True}
    save(out/'evaluation_protocol_resolved.json',resolved)
    small=[r for r in comparisons if r['size'] in ['all','tiny'] and r['metric']=='map50_95']
    header='|Condition|Area|CONTROL (%)|TARGETED (%)|Delta (pp)|95% image CI (pp)|\n|---|---|---|---|---|---|\n'
    human=header+'\n'.join('|'+ '|'.join([r['condition'],r['size'],f"{100*float(r['CONTROL']):.3f}",f"{100*float(r['MOTION_TARGETED']):.3f}",f"{100*float(r['difference']):+.3f}",f"[{100*float(r['CI95_low']):+.3f}, {100*float(r['CI95_high']):+.3f}]"])+'|' for r in small)
    primary=next(r for r in small if r['condition']=='Motion_k21' and r['size']=='tiny')
    resources=load(out/'resource_usage.json');gap='; '.join(r['condition']+': '+r['transition'] for r in transitions)
    reports=load(out/'report_paths.json');full=Path(reports['reports'][0]);shutil.copyfile(full,out/('before_archive_'+full.name))
    with full.open('a',encoding='utf-8') as f:
        f.write('\n\n## 实测结果解读与归档核验\n\n'+human+'\n\n两臂全部10轮的图像顺序与普通Albumentations激活/参数轨迹完全相同；每臂64710个图像锚点，新增Motion实际激活'+str(len(active))+'次，均有像素变化，训练未使用核21。角度分布见 `augmentation_angle_histogram.csv`。\n\n')
        f.write('主要终点为预先固定的Motion Tiny差值。其余对比为探索性、逐项区间，未进行多重比较校正。仅一个训练seed，不能由图像Bootstrap推断训练seed稳定性。\n\n')
        f.write('元数据说明：冻结评价文件继承的 `models/order` 是VM45缓存来源字段；新冻结worker从训练验收记录加载本次CONTROL/MOTION_TARGETED第10轮EMA，并不读取这两个历史字段。该问题已在评价开始前记录于 `evaluation_identity_clarification.json`。`evaluation_protocol_resolved.json` 将实际执行绑定整理为可读视图，没有更改历史冻结文件、模型选择或评价口径。\n\n')
        f.write('Gap转移：'+gap+'。\n\n')
        for p in reports['figures']:
            if p.endswith('.png'):f.write('!['+Path(p).stem+']('+str(Path(p)).replace('\\','/')+')\n\n')
    en=next(Path(p) for p in reports['weekly_updated'] if p.endswith('_EN.md'));shutil.copyfile(en,out/('before_numeric_update_'+en.name))
    with en.open('a',encoding='utf-8') as f:f.write('\n\n### Accepted seed45 measurements\n\n'+human+'\n\nGap states: '+gap+'. Single-seed development experiment; pointwise paired-image intervals do not establish training-seed significance.\n')
    initial=load(out/'run.json');save(out/'run_initial.json',initial);initial['status']=accepted['status'];initial['completed_UTC']=datetime.now(timezone.utc).isoformat();save(out/'run.json',initial,replace=True)
    (out/'diary.md').write_text('Seed45 Motion-targeted matched pilot\n\nCONTROL and MOTION_TARGETED completed10 epochs each on6471 training images. Both fixed epoch10 EMAs evaluated on548 development images, four frozen conditions, eight cells.\n\n'+human+'\n\n'+gap+'\n\nSupervised worker seconds: '+str(resources['GPU_worker_supervision_charged_seconds'])+'. New allocation balance: '+str(resources['new_remaining_seconds'])+'. Old ledger unchanged.\n\nCommands: training_supervision/command.txt; evaluation_supervision/command.txt. No retry, new seed, test-dev, full100epoch training or Git operation.\n',encoding='utf-8')
    sources=out/'execution_source';(sources/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    save(out/'report_SHA256.json',[source(p) for p in reports['reports']+reports['weekly_updated']])
    old_seal=out/'SHA256SUMS.txt';shutil.copyfile(old_seal,out/'before_log_closure.sha256')
    save(out/'archive_acceptance.json',{'status':'PASS','scope':accepted['scope'],'original_result_status':accepted['status'],'source':source(__file__),'CPU_seconds':time.perf_counter()-started,
        'old_ledger_files_and_membership_unchanged':True,'metrics_checked_against_saved_sources':True,'raw_prediction_images':4384,'new_GPU_calls':0,'new_COCO_evaluations':0,
        'purpose':'Seal after parent execution.log closes and add actual matching/metadata evidence. No scientific result changed.','prior_manifest_preserved':'before_log_closure.sha256'})
    old_seal.write_text(''.join(sha(p)+'  '+str(p.relative_to(out))+'\n' for p in sorted(out.rglob('*')) if p.is_file() and p.name!='SHA256SUMS.txt'),encoding='utf-8')
    for line in old_seal.read_text(encoding='utf-8').splitlines():
        digest,name=line.split('  ',1);assert sha(out/name)==digest
    print(json.dumps({'RunID':out.name,'archive':'PASS','primary_endpoint':primary,'Gap':transitions,'resource':resources},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);a=p.parse_args();archive(a.run)
