"""Eight matched cells using the accepted detector, timer and COCO accumulator."""
import csv,gc,gzip,json,time
import numpy as np
from motion_common import *

engine=module('seed45_accepted_vm45_evaluator',ROOT/'main/experiments/experiments_test/vm45_motion_haze_gap_learning/execute_vm45.py')
engine.RUNS=RUNS
backend=engine.backend

def csv_write(path,rows):
    if not rows:return
    with Path(path).open('x',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def interval(values):
    v=np.asarray(values,dtype=float);v=v[np.isfinite(v)]
    return np.percentile(v,[2.5,97.5]).tolist() if len(v) else [None,None]

def area(metrics,size,metric):return next(r[metric] for r in metrics['per_area'] if r['size']==size)

def worker(out):
    check(out);base.runtime(out/'evaluation_runtime',gpu=True)
    import torch,cv2
    from main.training_health import model_health
    from main.labels import VIS_NAMES
    torch.set_num_threads(2);cv2.setNumThreads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    spec=load(out/'evaluation_protocol.json');cache=load(spec['pixel_cache_source']['path'])
    assert sha(spec['GT']['path'])==spec['GT']['SHA256'] and sha(spec['pixel_cache_source']['path'])==spec['pixel_cache_source']['SHA256']
    records=load(spec['GT']['path'])['records'];assert len(records)==548
    started=time.perf_counter();models=[];checkpoints=[];cells=[]
    for arm in ['CONTROL','MOTION_TARGETED']:
        accepted=load(out/arm/'training_acceptance.json');ck=accepted['selected_checkpoint'];p=Path(ck['path'])
        assert accepted['completed_epochs']==10 and p.name=='epoch0010_fp32.pt' and sha(p)==ck['SHA256']
        data=torch.load(p,map_location='cpu',weights_only=False);assert data['epoch']==9
        model_health(data['ema']);assert list(data['ema'].names.values())==VIS_NAMES
        assert all(t.dtype==torch.float32 for t in data['ema'].state_dict().values() if t.is_floating_point())
        assert data['rq4_provenance']['initialization_SHA256']==load(out/'initialization_SHA.json')['initialization']['SHA256']
        del data;gc.collect()
        models.append((arm,ck,backend.RealDetector(str(p))))
        checkpoints.append({'strategy':arm,'seed':45,'epoch':10,'selection':'FIXED_EPOCH10_EMA','training_imgsz':960,'evaluation_imgsz':1280,**ck})
    csv_write(out/'checkpoint_manifest.csv',checkpoints)
    for cond in spec['conditions']:
        for arm,ck,adapter in models:
            name=arm+'__'+cond['condition'];folder=out/'raw_predictions'/name;folder.mkdir(parents=True)
            print('CELL_START',name,flush=True)
            entry=cache[cond['condition']][0];assert sha(entry['path'])==entry['file_SHA256']
            image=cv2.imread(entry['path']);assert image is not None
            t=time.perf_counter()
            for _ in range(3):adapter.predict(image)
            adapter.synchronize();save(folder/'warmup.json',{'condition':cond['condition'],'image_id':entry['image_id'],'count':3,'seconds':time.perf_counter()-t,'immediately_before_this_timed_cell':True})
            torch.cuda.reset_peak_memory_stats();times=[];partial=folder/'predictions.jsonl.gz.partial';raw=folder/'predictions.jsonl.gz'
            with gzip.open(partial,'xt',encoding='utf-8') as f,(folder/'image_manifest.jsonl').open('x',encoding='utf-8') as mf:
                for i,e in enumerate(cache[cond['condition']]):
                    times.append(backend.timed_prediction(e,adapter,f,mf))
                    if (i+1)%100==0:print('INFERENCE',name,i+1,548,flush=True)
            with gzip.open(partial,'rt',encoding='utf-8') as f:rows=[json.loads(s) for s in f]
            assert len(rows)==548 and [r['image_id'] for r in rows]==[r['id'] for r in records]
            backend.validate_predictions(rows,{r['id'] for r in records});partial.rename(raw);del rows
            metrics,boot,cost=engine.evaluation(raw,records,spec['bootstrap_seed'])
            save(folder/'metrics.json',metrics);save(folder/'bootstrap.json',boot)
            save(folder/'timing.json',{'per_image':times,'evaluation_cost':cost,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'timing_protocol':'VM41_MATCHED_CACHED_GAP_V1','warmups_per_cell':3})
            save(folder/'acceptance.json',{'status':'REAL_CELL_COMPLETED','seed':45,'strategy':arm,'condition':cond['condition'],'P_id':cond['P_id'],'images':548,
                'checkpoint_SHA256':ck['SHA256'],'raw':source(raw),'metrics':source(folder/'metrics.json'),'bootstrap':source(folder/'bootstrap.json'),'timing':source(folder/'timing.json'),
                'GT':spec['GT'],'pixel_manifest':spec['pixel_cache_source'],'scope':'DEVELOPMENT_ONLY / SINGLE_SEED45'})
            cells.append(name);print('CELL_COMPLETE',name,flush=True)
    save(out/'evaluation_result.json',{'status':'REAL_EIGHT_CELLS_COMPLETED','cells':cells,'formal_images':4384,'warmup_images':24,'wall_seconds':time.perf_counter()-started})
    summarize(out)

def summarize(out):
    spec=load(out/'evaluation_protocol.json');result=load(out/'evaluation_result.json');assert len(result['cells'])==8
    table=[];classes=[];cells={};comparisons=[];perclass=[];transitions=[]
    indices=np.random.default_rng(spec['bootstrap_seed']).integers(0,548,(200,548));image_ids=None
    for name in result['cells']:
        folder=out/'raw_predictions'/name;a=load(folder/'acceptance.json')
        for k in ['raw','metrics','bootstrap','timing']:assert sha(a[k]['path'])==a[k]['SHA256']
        m=load(a['metrics']['path']);b=load(a['bootstrap']['path']);t=load(a['timing']['path']);d=b['draws']
        assert d['samples']==200 and d['seed']==spec['bootstrap_seed'] and load(folder/'warmup.json')['count']==3
        if image_ids is None:image_ids=d['image_ids']
        assert d['image_ids']==image_ids
        latency=np.asarray([r['cached_pipeline_ms'] for r in t['per_image']]);assert len(latency)==548
        ld=latency[indices].mean(axis=1);lc=interval(ld)
        tiny=next(r['ci95'] for r in b['intervals'] if r['kind']=='area' and r['size']=='tiny' and r['metric']=='map50_95')
        state=backend.gap_state(tiny,lc);key=(a['strategy'],a['condition']);cells[key]=(m,d,latency,ld,state,tiny,a)
        table.append({'strategy':a['strategy'],'seed':45,'condition':a['condition'],'P_id':a['P_id'],'images':548,'AP50':area(m,'all','map50'),'AP50_95':area(m,'all','map50_95'),
            **{s.title()+'_AP50_95':area(m,s,'map50_95') for s in ['tiny','small','medium','large']},'AR50_95':area(m,'all','ar50_95'),'cached_pipeline_ms':float(latency.mean()),
            'cached_latency_CI_low':lc[0],'cached_latency_CI_high':lc[1],'Tiny_CI_low':tiny[0],'Tiny_CI_high':tiny[1],'state':state,'prediction_count':m['raw_count'],
            'checkpoint_SHA256':a['checkpoint_SHA256'],'GT_SHA256':a['GT']['SHA256'],'metrics_SHA256':a['metrics']['SHA256'],'metrics_path':a['metrics']['path'],
            'GPU_peak_allocated_bytes':t['peak_allocated_bytes'],'COCO_seconds':t['evaluation_cost']['COCO_seconds'],'bootstrap_seconds':t['evaluation_cost']['bootstrap_seconds'],'warmups':3})
        classes.extend({'strategy':a['strategy'],'condition':a['condition'],**r} for r in m['per_class'])
    save(out/'paired_bootstrap_indices.json',{'seed':spec['bootstrap_seed'],'image_ids':image_ids,'samples':200,'indices':indices.tolist(),'same_across_all_cells':True})
    for cond in spec['conditions']:
        name=cond['condition'];c,g=[cells[a,name] for a in ['CONTROL','MOTION_TARGETED']]
        for size in c[1]['values']:
            for metric in c[1]['values'][size]:
                left,right=area(c[0],size,metric),area(g[0],size,metric)
                draws=np.asarray(g[1]['values'][size][metric],float)-np.asarray(c[1]['values'][size][metric],float);ci=interval(draws)
                comparisons.append({'condition':name,'size':size,'metric':metric,'CONTROL':left,'MOTION_TARGETED':right,'difference':right-left if left is not None and right is not None else None,
                    'CI95_low':ci[0],'CI95_high':ci[1],'n_valid':int(np.isfinite(draws).sum()),'samples':200,'interval_scope':'POINTWISE_PAIRED_IMAGE_BOOTSTRAP_SINGLE_TRAINING_SEED'})
        ci=interval(g[3]-c[3]);comparisons.append({'condition':name,'size':'all','metric':'cached_pipeline_ms','CONTROL':float(c[2].mean()),'MOTION_TARGETED':float(g[2].mean()),'difference':float(g[2].mean()-c[2].mean()),'CI95_low':ci[0],'CI95_high':ci[1],'n_valid':200,'samples':200})
        for r in c[0]['per_class']:
            rr=next(z for z in g[0]['per_class'] if z['size']==r['size'] and z['category_id']==r['category_id']);key=r['size']+':'+str(r['category_id'])
            for metric in c[1]['class_values'][key]:
                draws=np.asarray(g[1]['class_values'][key][metric],float)-np.asarray(c[1]['class_values'][key][metric],float);ci=interval(draws)
                perclass.append({'condition':name,'size':r['size'],'category_id':r['category_id'],'metric':metric,'CONTROL':r[metric],'MOTION_TARGETED':rr[metric],
                    'difference':rr[metric]-r[metric] if r[metric] is not None and rr[metric] is not None else None,'CI95_low':ci[0],'CI95_high':ci[1],'n_valid':int(np.isfinite(draws).sum())})
        transitions.append({'condition':name,'P_id':cond['P_id'],'CONTROL_state':c[4],'MOTION_TARGETED_state':g[4],'transition':c[4]+' -> '+g[4],'confirmed_Gap_closed':c[4]=='FAIL' and g[4]=='PASS',
            'CONTROL_Tiny_CI':json.dumps(c[5]),'MOTION_TARGETED_Tiny_CI':json.dumps(g[5]),'thresholds':'Tiny lower95>=.10 and cached latency upper95<=100ms'})
    csv_write(out/'configuration_metrics.csv',table);csv_write(out/'per_class_metrics.csv',classes);csv_write(out/'paired_AP_comparisons.csv',comparisons)
    csv_write(out/'paired_per_class_comparisons.csv',perclass);csv_write(out/'tiny_gap_transitions.csv',transitions)
    save(out/'analysis_acceptance.json',{'cells':8,'formal_image_records':4384,'seed':45,'per_cell_warmups':3,'paired_bootstrap_samples':200,'primary_endpoint_frozen':True,
        'independent_validation':False,'training_seed_significance':False,'formal_1280_100epoch_RQ4_complete':False})
