"""Fixed epoch100 evaluation with the accepted cached-pipeline boundary."""
import argparse,csv,gzip,hashlib,json,os,sys,time,traceback
from datetime import datetime,timezone,timedelta
from pathlib import Path
import numpy as np
from common48 import *
from gap48 import state,p_state,paired_gap,bootstrap_gap_draws


def sized_predictor(parent,resolution):
    import torch
    from main.predictor_stable import StableNMSPredictor
    class Sized(parent):
        def preprocess(self,images):
            tensor=StableNMSPredictor.preprocess(self,images)
            if list(tensor.shape)!=[1,3,resolution,resolution] or tensor.dtype!=torch.float32:raise ValueError('Inference tensor differs.')
            self.input_tensor={'NCHW':list(tensor.shape),'dtype':str(tensor.dtype)}
            return tensor
    return Sized


def cached_adapter(checkpoint,resolution):
    if resolution not in (640,1280):raise ValueError('Inference resolution is outside the frozen study.')
    import torch
    from cached_backend import RealDetector
    from main.predictor_stable import StableNMSPredictor
    class Detector(RealDetector):
        def __init__(self,path):
            super().__init__(path)
            self.predictor=sized_predictor(self.predictor,resolution)
        def predict(self,image):
            self.last_shape=image.shape[:2]
            with torch.inference_mode():
                return self.model.predict(image,predictor=self.predictor,imgsz=resolution,batch=1,device='0',quantize=None,nms=None,
                    conf=.001,iou=.7,max_det=3000,rect=False,agnostic_nms=False,classes=None,augment=False,save=False,verbose=False)[0]
    return Detector(checkpoint)


def saved_metrics(raw,records,spec):
    from cached_backend import validate_predictions
    from main.labels import ground_truth,mapped_predictions
    from main.statistics import evaluate_coco,bootstrap_coco
    with gzip.open(raw,'rt',encoding='utf-8') as f:rows=[json.loads(line) for line in f]
    validate_predictions(rows,{r['id'] for r in records})
    gt,regions=ground_truth(records,'native10');predictions=[p for r in rows for p in r['predictions']]
    mapped=mapped_predictions(predictions,records,regions,'native10','finetuned',3000)
    tick=time.perf_counter();areas,classes,evaluator=evaluate_coco(gt,mapped,3000);coco=time.perf_counter()-tick
    base.install_accumulator();tick=time.perf_counter()
    intervals,draws=bootstrap_coco(evaluator,samples=200,seed=spec['bootstrap_seed'])
    return {'per_area':areas,'per_class':classes,'images':len(records),'raw_count':len(predictions),'mapped_count':len(mapped)}, {'intervals':intervals,'draws':draws}, {'COCO_seconds':coco,'bootstrap_seconds':time.perf_counter()-tick}


def cache_conditions(package,data_root,out,spec):
    import cv2,shutil
    population=base.resolve_data(package,data_root);records=load(package/spec['GT_asset'])['records'];ops=base.renderer()
    root=out/'pixel_cache';root.mkdir();cases={}
    if shutil.disk_usage(out).free<sum(r['width']*r['height']*3 for r in records)*len(spec['contexts'])+8*1024**3:
        raise RuntimeError('Full shared PNG cache and output reserve do not fit.')
    for case in spec['contexts']:
        directory=root/case['id'];directory.mkdir()
        expected=[json.loads(s) for s in (package/case['pixel_manifest_asset']).read_text().splitlines()]
        by_id={Path(r['source']['ZIP_member']).stem:r for r in expected};entries=[]
        by_stem={r['image_id']:r for r in population['val']}
        if set(by_stem)!={r['stem'] for r in records}:raise ValueError('GT/image membership differs.')
        for record in records:
            row=by_stem[record['stem']]
            raw=Path(row['image_path']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['image_SHA256']:raise ValueError('Source image changed.')
            image=cv2.imdecode(np.frombuffer(raw,np.uint8),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
            pixel,trace=ops.generate(image,case['p']['leaves'],case['p']['attributes'],record['stem'],spec['render_seed'])
            digest=hashlib.sha256(pixel.tobytes()).hexdigest()
            if digest!=by_id[record['stem']]['pixel_sha256']:raise ValueError('Frozen evaluation pixels differ.')
            ok,png=cv2.imencode('.png',pixel)
            if not ok:raise RuntimeError('PNG encoding failed.')
            target=directory/(record['stem']+'.png')
            with target.open('xb') as f:f.write(png.tobytes())
            entries.append({'image_id':record['id'],'stem':record['stem'],'path':str(target),'file_SHA256':sha(target),'pixel_SHA256':digest,
                            'source_image_SHA256':row['image_SHA256'],'effects':trace})
        cases[case['id']]=entries
    save(out/'pixel_cache_manifest.json',cases)
    return records,cases


def evaluate_worker(package,data_root,parent):
    from runtime import own_ticket
    out=parent/'evaluation';ticket=load(parent/'execution_ticket.json');auth=load(parent/'authorization.json')
    token=os.environ.get('VM48_EVALUATION_TOKEN','')
    if not token or hashlib.sha256(token.encode()).hexdigest()!=load(parent/'evaluation_token.json')['token_SHA256']:raise PermissionError('Owned evaluation token required.')
    if sha(parent/'authorization.json')!=ticket['authorization_SHA256']:raise PermissionError('Authorization changed.')
    if datetime.now(timezone.utc)>=datetime.fromisoformat(load(out/'WORKER_LEASE.json')['deadline_UTC']):raise PermissionError('Evaluation lease expired.')
    verify_package(package);base.runtime(out,gpu=True)
    for name,digest in ticket['plan_SHA256'].items():
        if sha(parent/name/'protocol.json')!=digest:raise PermissionError('Training plan changed before evaluation.')
    import cv2,torch
    from cached_backend import timed_prediction,validate_predictions
    torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    spec=load(package/'frozen_evaluation_protocol.json');rows=ticket['rows'];records,cache=cache_conditions(package,data_root,out,spec)
    indices=np.random.default_rng(spec['bootstrap_seed']).integers(0,548,(200,548));values={};index=[]
    for row in rows:
        train_out=parent/row['arm_id'];receipt=load(train_out/'training_acceptance.json')
        checkpoint=Path(receipt['selected_checkpoint']['path'])
        if checkpoint.name!='epoch0100_fp32.pt' or sha(checkpoint)!=receipt['selected_checkpoint']['sha256']:raise ValueError('Fixed epoch100 checkpoint missing or changed.')
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False);epoch100_contract(receipt,payload,row['initialization_SHA256']);del payload
        for resolution in spec['resolutions']:
            adapter=cached_adapter(str(checkpoint),resolution)
            for case in spec['contexts']:
                cid=row['arm_id']+'__R'+str(resolution)+'__'+case['id'];folder=out/cid;folder.mkdir()
                warm=cv2.imread(cache[case['id']][0]['path'])
                for _ in range(3):adapter.predict(warm)
                adapter.synchronize();torch.cuda.reset_peak_memory_stats();times=[];raw=folder/'predictions.jsonl.gz';partial=folder/'predictions.partial.gz'
                with gzip.open(partial,'xt',encoding='utf-8') as stream,(folder/'image_manifest.jsonl').open('x',encoding='utf-8') as manifest:
                    for entry in cache[case['id']]:times.append(timed_prediction(entry,adapter,stream,manifest))
                partial.rename(raw)
                metrics,boot,cost=saved_metrics(raw,records,spec)
                save(folder/'metrics.json',metrics);save(folder/'bootstrap.json',boot)
                latency=np.array([r['cached_pipeline_ms'] for r in times]);lat_draws=latency[indices].mean(axis=1)
                tiny=next(r for r in boot['intervals'] if r['size']=='tiny' and r['metric']=='map50_95')
                ci=np.percentile(lat_draws,[2.5,97.5]).tolist();status=state(tiny['ci95'],ci)
                save(folder/'timing.json',{'per_image':times,'evaluation_cost':cost,'cached_pipeline_mean_ms':float(latency.mean()),'latency_CI95':ci,
                    'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'timing_protocol':'VM41_MATCHED_CACHED_GAP_V1'})
                key=(row['architecture'],row['seed'],row['strategy'],resolution,case['id'])
                values[key]={'metrics':metrics,'draws':boot['draws'],'tiny_draws':boot['draws']['values']['tiny']['map50_95'],'latency_draws':lat_draws.tolist(),'state':status}
                evidence={'cell_id':cid,'architecture':row['architecture'],'seed':row['seed'],'strategy':row['strategy'],'resolution':resolution,'P_id':case['id'],
                    'state':status,'checkpoint_SHA256':sha(checkpoint),'raw_SHA256':sha(raw),'GT_SHA256':sha(package/spec['GT_asset']),
                    'metrics_SHA256':sha(folder/'metrics.json'),'bootstrap_SHA256':sha(folder/'bootstrap.json'),'timing_SHA256':sha(folder/'timing.json'),'scope':'DEVELOPMENT_ONLY','images':548}
                save(folder/'acceptance.json',evidence);index.append(evidence);print('COMPLETE_CELL',len(index),flush=True)
            del adapter;torch.cuda.empty_cache()
    expected=len(rows)*len(spec['contexts'])*2
    if len(index)!=expected:raise ValueError('Evaluation matrix incomplete.')
    save(out/'evaluation_index.json',index)
    if ticket['mode']=='formal':
        primary_gap(out,values,spec)
        # The accepted paired AP contrast code is adapted only for the strategy alias.
        import evaluate as previous
        old={(a,s,('ALL_FIVE' if arm=='ALL_FIVE_TARGETED' else arm),r,p):v for (a,s,arm,r,p),v in values.items()}
        normal_id=next(c['id'] for c in spec['contexts'] if c['p']['leaves']==['Normal'])
        renamed={(a,s,arm,r,('Normal' if p==normal_id else p)):v for (a,s,arm,r,p),v in old.items()}
        save(out/'paired_metric_contrasts.json',previous.paired_contrasts(renamed))
        summarize_families(out,values,spec)
    else:
        control_parent=Path(load(out/'reference_control.json')['parent'])
        reference_index=load(control_parent/'evaluation/evaluation_index.json')
        if load(control_parent/'execution_ticket.json')['package_manifest_SHA256']!=sha(package/'package_manifest.json'):raise PermissionError('Ablation reference uses another package.')
        control={}
        for r in reference_index:
            if (r['architecture'],r['seed'],r['strategy'])!=('YOLOv8s',42,'CONTROL'):continue
            folder=control_parent/'evaluation'/r['cell_id']
            for name,key in [('metrics.json','metrics_SHA256'),('bootstrap.json','bootstrap_SHA256'),('timing.json','timing_SHA256')]:
                if sha(folder/name)!=r[key]:raise ValueError('Ablation reference evidence changed.')
            control[('YOLOv8s',42,'CONTROL',r['resolution'],r['P_id'])]={'metrics':load(folder/'metrics.json'),'draws':load(folder/'bootstrap.json')['draws']}
        if len(control)!=42:raise ValueError('Complete42-cell seed42 control evaluation is required.')
        import evaluate as previous
        normal_id=next(c['id'] for c in spec['contexts'] if c['p']['leaves']==['Normal'])
        contrasts={}
        for strategy in sorted({r['strategy'] for r in rows}):
            selected={**control,**{k:v for k,v in values.items() if k[2]==strategy}}
            renamed={(a,s,('ALL_FIVE' if arm==strategy else arm),r,('Normal' if p==normal_id else p)):v for (a,s,arm,r,p),v in selected.items()}
            contrasts[strategy]=previous.paired_contrasts(renamed)
        save(out/'matched_ablation_contrasts.json',{'seed':42,'three_seed_replication':False,'reference_control':str(control_parent),'comparisons':contrasts})
    save(out/'evaluation_result.json',{'status':'COMPLETE_MEASURED_SUBSET_DEVELOPMENT_ONLY','cells':expected,'full_FM_P_gap':'UNKNOWN','formal_RQ4_complete_claim':False})


def primary_gap(out,values,spec):
    reference=next(iter(values.values()))['draws']
    for value in values.values():
        if any(value['draws'][key]!=reference[key] for key in ('image_ids','samples','seed','sampling')):raise ValueError('Global paired Bootstrap indices differ.')
    contexts=[r['id'] for r in spec['contexts']];rows=[]
    for seed in (42,43,44):
        states={s:[] for s in ('CONTROL','ALL_FIVE_TARGETED')}
        for s in states:
            for p in contexts:
                labels=[v['state'] for k,v in values.items() if k[1]==seed and k[2]==s and k[4]==p]
                if len(labels)!=4:raise ValueError('Incomplete seed-specific detector set.')
                states[s].append(p_state(labels))
        rows.append({'seed':seed,**paired_gap(states['CONTROL'],states['ALL_FIVE_TARGETED'])})
    save(out/'primary_Gap_endpoint.json',{'definition':'CONTROL Gap count minus TARGETED Gap count; positive is reduction','per_seed':rows,
        'across_seed_known_reduction_mean':float(np.mean([r['known_Gap_count_reduction'] for r in rows])),
        'across_seed_known_reduction_range':[min(r['known_Gap_count_reduction'] for r in rows),max(r['known_Gap_count_reduction'] for r in rows)],
        'image_uncertainty':bootstrap_gap_draws(values,contexts,[42,43,44]),'full_space_gap':'UNKNOWN','population_gap_change':'NOT_ESTIMATED_PURPOSEFUL_SAMPLE'})


def summarize_families(out,values,spec):
    normal=next(c['id'] for c in spec['contexts'] if c['p']['leaves']==['Normal']);rows=[];clean=[]
    point=lambda v,size,metric:next(r[metric] for r in v['metrics']['per_area'] if r['size']==size)
    for architecture in ('YOLOv8s','YOLO26s'):
        for seed in (42,43,44):
            for resolution in (640,1280):
                c=values[architecture,seed,'CONTROL',resolution,normal];t=values[architecture,seed,'ALL_FIVE_TARGETED',resolution,normal]
                for size in ('all','tiny','small','medium','large'):
                    delta=point(t,size,'map50_95')-point(c,size,'map50_95')
                    draws=np.asarray(t['draws']['values'][size]['map50_95'])-np.asarray(c['draws']['values'][size]['map50_95'])
                    clean.append({'architecture':architecture,'seed':seed,'resolution':resolution,'size':size,'clean_accuracy_cost_CONTROL_minus_TARGETED':-delta,
                        'clean_cost_CI95':np.percentile(-draws,[2.5,97.5]).tolist(),'latency_change_ms_TARGETED_minus_CONTROL':float(np.mean(t['latency_draws'])-np.mean(c['latency_draws']))})
                for family,contexts in spec['family_context_groups'].items():
                    for size in ('all','tiny'):
                        if not contexts:
                            rows.append({'architecture':architecture,'seed':seed,'resolution':resolution,'family':family,'size':size,'status':'NOT_ESTIMABLE_PRIMARY21','contexts':0});continue
                        changes=[];draws=[]
                        for p in contexts:
                            c=values[architecture,seed,'CONTROL',resolution,p];t=values[architecture,seed,'ALL_FIVE_TARGETED',resolution,p]
                            changes.append(point(t,size,'map50_95')-point(c,size,'map50_95'))
                            draws.append(np.asarray(t['draws']['values'][size]['map50_95'])-np.asarray(c['draws']['values'][size]['map50_95']))
                        rows.append({'architecture':architecture,'seed':seed,'resolution':resolution,'family':family,'size':size,'status':'MEASURED_CONFIGURATION_SUBSET',
                           'contexts':len(contexts),'mean_configuration_AP_change':float(np.mean(changes)),'CI95':np.percentile(np.mean(draws,axis=0),[2.5,97.5]).tolist(),
                           'interpretation':'Mean paired configuration AP differences, not per-image AP averaging or isolated family causality'})
    save(out/'clean_accuracy_and_latency_cost.json',clean);save(out/'per_degradation_family_changes.json',rows)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);p.add_argument('--training-parent',type=Path,required=True)
    p.add_argument('--data-root',type=Path,required=True);p.add_argument('--authorization',type=Path);p.add_argument('--cpu-receipt',type=Path)
    p.add_argument('--execute',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--worker',action='store_true');p.add_argument('--control-parent',type=Path)
    a=p.parse_args();package=a.package.resolve(strict=True);parent=a.training_parent.resolve(strict=True)
    verify_package(package);spec=load(package/'frozen_evaluation_protocol.json')
    if a.worker:
        try:evaluate_worker(package,a.data_root.resolve(strict=True),parent)
        except BaseException:save(parent/'evaluation/failure.json',{'traceback':traceback.format_exc(),'retry':False});raise
        return
    if not a.execute:
        print(json.dumps({'status':'EVALUATION_DRY_RUN_ONLY','primary_cells':12*21*2,'ablation_cells':5*21*2,'GPU_seconds':0,'requires':'All requested fixed epoch100 receipts, new authorization, complete budget and current safety admission'}));return
    from runtime import validate_authorization,read_ledger,remaining,charge,supervised
    if not a.authorization or not a.cpu_receipt:raise PermissionError('New evaluation authorization and CPU receipt required.')
    ticket=load(parent/'execution_ticket.json');auth=load(a.authorization);receipt=load(a.cpu_receipt)
    if sha(a.authorization)!=ticket['authorization_SHA256']:raise PermissionError('Evaluation must use the source-bound training-group authorization.')
    validate_authorization(package,ticket['mode'],ticket['rows'],auth,receipt,evaluation=True)
    cap=auth['resource_admission']['evaluation_hard_seconds']
    if cap<60 or cap+auth['resource_admission']['safety_reserve_seconds']>remaining(auth,read_ledger(package,auth)['charges']):raise PermissionError('Complete evaluation budget is absent.')
    import torch
    for row in ticket['rows']:
        folder=parent/row['arm_id'];accepted=load(folder/'training_acceptance.json');path=Path(accepted['selected_checkpoint']['path'])
        if sha(path)!=accepted['selected_checkpoint']['sha256']:raise ValueError('Training checkpoint changed.')
        epoch100_contract(accepted,torch.load(path,map_location='cpu',weights_only=False),row['initialization_SHA256'])
    if ticket['mode']=='ablations' and (a.control_parent is None or not (a.control_parent/'evaluation/evaluation_result.json').exists()):raise PermissionError('Ablation evaluation needs the completed matched seed42 CONTROL reference.')
    out=parent/'evaluation';out.mkdir();now=datetime.now(timezone.utc)
    if ticket['mode']=='ablations':save(out/'reference_control.json',{'parent':str(a.control_parent.resolve(strict=True))})
    save(out/'WORKER_LEASE.json',{'deadline_UTC':min(now+timedelta(seconds=cap),datetime.fromisoformat(auth['expires_UTC'])).isoformat(),'cap_seconds':cap})
    import secrets
    # A separate evaluation token is source-bound to a new immutable ticket copy.
    token_path=parent/'evaluation_token.json'
    token=secrets.token_hex(24);save(token_path,{'token_SHA256':hashlib.sha256(token.encode()).hexdigest()})
    command=[sys.executable,'-B','-X','utf8',str(Path(__file__).resolve()),'--package',str(package),'--data-root',str(a.data_root),
             '--training-parent',str(parent),'--worker']
    lock=package/'datasets/local_runs/.vm48_gpu.lock'
    with lock.open('x',encoding='utf-8') as f:json.dump({'RunID':parent.name,'evaluation':True},f)
    started=time.perf_counter();result=None
    try:result=supervised(command,out,cap,dict(os.environ,VM48_EVALUATION_TOKEN=token,CUDA_VISIBLE_DEVICES='0'),load(package/'safety_policy.json'),auth['GPU_uuid'])
    finally:
        charge(package,auth,parent.name,'evaluation',time.perf_counter()-started,result)
        if lock.exists() and load(lock).get('RunID')==parent.name:lock.unlink()
    if result['stop_reason'] or result['returncode']:raise RuntimeError('Evaluation stopped; see supervisor reason before missing-result diagnostics.')


if __name__=='__main__':main()
