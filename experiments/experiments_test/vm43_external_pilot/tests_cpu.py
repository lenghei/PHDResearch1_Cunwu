"""Integration tests use synthetic predictions only under checks/."""
import copy
import gzip
import hashlib
from io import StringIO
import json
from pathlib import Path
import time
from zipfile import ZipFile

import backend as b
from vm43_common import save,sha,frozen,module,HERE


def run(out,records):
    import cv2,numpy as np
    from main.labels import ground_truth,mapped_predictions,keep_outside_ignore,VIS_NAMES
    from main.dataset import VisDroneZIP
    from main.statistics import evaluate_coco,resample_evaluator,_metric_rows
    tests=[];checks=out/'checks';checks.mkdir();start=time.perf_counter()
    def check(name,condition):
        if not condition:raise AssertionError(name)
        tests.append({'test':name,'status':'PASS'})
    def rejects(name,fn,expected=(ValueError,RuntimeError,FileExistsError)):
        try:fn()
        except expected:check(name,True)
        else:check(name,False)
    spec=frozen();check('exact_four_cells',len(spec['cells'])==4)
    population=b.load(spec['population']['path']);check('full1610_GT_membership',len(records)==1610 and len({r['stem'] for r in records})==1610)
    check('native10_order',population['native10_class_order']==VIS_NAMES)
    archive=b.OriginalZIP(population,population['source_ZIP_accepted_certificate'],verify_archive=False)
    try:
        image,actual=archive.record(0,decode=True)
        reference=VisDroneZIP.__new__(VisDroneZIP);reference.archive=archive.archive
        reference.images=[archive.images[r['image_id']] for r in population['images']];reference.labels=archive.labels
        second,legacy,_=reference.record(0)
        check('one_original_image_parser_parity',actual==legacy and np.array_equal(image,second))
        check('one_original_GT_matches_full_payload',actual==records[0])
    finally:archive.close()
    ops=b.renderer();cell=next(r for r in spec['cells'] if r['condition']=='Gaussian_s15')
    state=np.random.get_state();pix,trace=ops.generate(image,list(cell['P_attributes']),cell['P_attributes'],actual['stem'],20261116)
    repeated,again=ops.generate(image,list(cell['P_attributes']),cell['P_attributes'],actual['stem'],20261116)
    check('actual_Gaussian_pixel_determinism',np.array_equal(pix,repeated) and trace==again)
    check('global_numpy_RNG_unchanged',all(np.array_equal(x,y) if isinstance(x,np.ndarray) else x==y for x,y in zip(state,np.random.get_state())))
    check('Gaussian_no_geometry_change',pix.shape==image.shape and pix.dtype==np.uint8)
    ok,encoded=cv2.imencode('.png',pix);check('PNG_encode',ok);path=checks/'one_Gaussian_cache.png';path.write_bytes(encoded.tobytes())
    entry={'path':str(path),'file_SHA256':sha(path),'pixel_SHA256':hashlib.sha256(pix.tobytes()).hexdigest(),'image_id':1}
    check('PNG_pixel_roundtrip',np.array_equal(pix,cv2.imdecode(encoded,cv2.IMREAD_COLOR)))
    synthetic=[{'id':1,'stem':'synthetic1','file_name':'synthetic1.png','width':128,'height':128,
        'objects':[{'class_id':0,'bbox':[60,2,8,8]},{'class_id':1,'bbox':[60,25,20,20]},{'class_id':2,'bbox':[1,1,10,10]}],
        'ignore_regions':[[0,0,20,20]],'other_regions':[[20,0,20,20]]},
        {'id':2,'stem':'synthetic2','file_name':'synthetic2.png','width':128,'height':128,
         'objects':[{'class_id':0,'bbox':[10,60,40,40]}],'ignore_regions':[],'other_regions':[]},
        {'id':3,'stem':'synthetic3','file_name':'synthetic3.png','width':128,'height':128,
         'objects':[{'class_id':1,'bbox':[0,0,100,100]}],'ignore_regions':[],'other_regions':[]}]
    gt,regions=ground_truth(synthetic,'native10')
    check('ignore_union_filters_GT',len(gt['annotations'])==4)
    check('union_not_sum_or_single_rectangle',not bool(keep_outside_ignore([[0,0,40,20]],regions[1],128,128)[0]))
    raw=[{'image_id':r['id'],'predictions':[{'image_id':r['id'],'class_id':o['class_id'],'bbox':o['bbox'],'score':.9} for o in r['objects']]} for r in synthetic]
    fake=checks/'synthetic_predictions.jsonl.gz'
    with gzip.open(fake,'xt',encoding='utf-8') as f:
        for r in raw:f.write(json.dumps(r)+'\n')
    metrics,intervals,draws,cost=b.evaluate_saved(fake,synthetic,200,20261116)
    check('synthetic_perfect_COCO_AP',abs(metrics['per_area'][0]['map50_95']-1)<1e-12)
    check('original_area_bins',all(next(x['gt_count'] for x in metrics['per_area'] if x['size']==size)==1 for size in ('tiny','small','medium','large')))
    check('absent_class_AP_null',all(x['map50_95'] is None for x in metrics['per_class'] if x['category_id']>=3))
    check('exact200_bootstrap_draws',draws['samples']==200 and len(draws['values']['all']['map50_95'])==200)
    mapped=mapped_predictions([p for r in raw for p in r['predictions']],synthetic,regions,'native10','finetuned',3000)
    _,_,ev=evaluate_coco(gt,mapped,3000);sample=[0,0,2]
    resampled=resample_evaluator(ev,np.array(sample));actual_rows,_=_metric_rows(resampled)
    explicit=copy.deepcopy(gt);explicit['images']=[];explicit['annotations']=[];newpred=[]
    for iid,position in enumerate(sample,1):
        old=gt['images'][position]['id'];explicit['images'].append({**gt['images'][position],'id':iid})
        for ann in gt['annotations']:
            if ann['image_id']==old:explicit['annotations'].append({**ann,'id':len(explicit['annotations'])+1,'image_id':iid})
        newpred.extend({**p,'image_id':iid} for p in mapped if p['image_id']==old)
    expected,_,_=evaluate_coco(explicit,newpred,3000)
    check('bootstrap_reaccumulation_equals_explicit_GT_pred_resample',all(a['map50_95'] is c['map50_95'] if a['map50_95'] is None or c['map50_95'] is None else abs(a['map50_95']-c['map50_95'])<1e-12 for a,c in zip(actual_rows,expected)))
    check('shared_bootstrap_metadata',draws['image_ids']==[1,2,3])
    rejects('duplicate_or_truncated_output_rejected',lambda:b.validate_predictions(raw[:-1],{1,2,3}))
    bad=copy.deepcopy(raw);bad[0]['predictions'][0]['score']=float('nan')
    rejects('NaN_score_rejected',lambda:b.validate_predictions(bad,{1,2,3}))
    bad=copy.deepcopy(raw);bad[0]['predictions'][0]['bbox'][0]=float('inf')
    rejects('Inf_box_rejected',lambda:b.validate_predictions(bad,{1,2,3}))
    bad=copy.deepcopy(raw);bad[0]['predictions'][0]['class_id']=10
    rejects('out_of_native10_class_rejected',lambda:b.validate_predictions(bad,{1,2,3}))
    rejects('invalid_zero_area_GT_rejected',lambda:b.parse_annotation(b'1,2,0,4,1,1,0,0',{'width':128,'height':128}))
    class CPUAdapter:
        def synchronize(self):pass
        def predict(self,image):return None
        def profiles(self,result):return {'preprocessing_profile_ms':0.,'forward_profile_ms':0.,'NMS_profile_ms':0.}
        def extract(self,result,iid):return [{'image_id':iid,'class_id':0,'bbox':[1.,2.,8.,8.],'score':.9}],{'NCHW':[1,3,1280,1280],'dtype':'SYNTHETIC_INTERFACE_ONLY'}
    times=b.timed_prediction(entry,CPUAdapter(),StringIO(),StringIO())
    required={'cached_file_verification_ms','image_read_ms','decode_ms','pixel_verification_ms','preprocessing_profile_ms','forward_profile_ms','NMS_profile_ms',
        'inference_only_ms','prediction_extraction_ms','output_and_manifest_write_flush_ms','cached_pipeline_ms','pre_predict_sync_ms','post_predict_sync_ms'}
    check('all_timing_fields_present',required.issubset(times) and all(times[k]>=0 for k in required))
    check('primary_cached_boundary_covers_predict',times['cached_pipeline_ms']>=times['inference_only_ms'])
    rejects('cache_byte_corruption_rejected',lambda:b.timed_prediction({**entry,'file_SHA256':'0'*64},CPUAdapter(),StringIO(),StringIO()))
    rejects('cache_pixel_corruption_rejected',lambda:b.timed_prediction({**entry,'pixel_SHA256':'0'*64},CPUAdapter(),StringIO(),StringIO()))
    save(checks/'atomic.json',{'scope':'SYNTHETIC_CPU_TEST_ONLY'})
    rejects('existing_output_cannot_be_overwritten',lambda:save(checks/'atomic.json',{}))
    cells={(a,c):{'metrics':metrics,'draws':draws} for a in ('CONTROL','VARIABILITY_GUIDED') for c in ('Normal','Gaussian_s15')}
    comparisons=b.compare_cells(cells,synthetic,20261116,200)
    check('matched_synthetic_differences_zero',all(r['difference'] in (0,None) for r in comparisons))
    cells.pop(('CONTROL','Normal'));rejects('unmatched_comparison_rejected',lambda:b.compare_cells(cells,synthetic,20261116,200))
    check('fixed_Gap_rules',b.gap_state([.11,.12],[50,60])=='PASS' and b.gap_state([.08,.09],[50,60])=='FAIL' and b.gap_state([.09,.11],[50,60])=='UNCERTAIN')
    save(checks/'synthetic_metric_assertions.json',{'scope':'SYNTHETIC_CPU_SOFTWARE_TEST_ONLY_NOT_YOLO_ACCURACY','perfect_AP':metrics['per_area'][0]['map50_95'],
        'bootstrap_samples':200,'images':3,'source_image_CPU_decode_calls':2,'actual_network_tensors_tested':False})
    return {'status':'PASS','tests':tests,'passed':len(tests),'failed':0,'CPU_wall_seconds':time.perf_counter()-start,
        'GPU_access':False,'real_YOLO_accuracy_tested':False,'scientific_results_generated':False,
        'scope':'Synthetic predictions only under checks; real original-data parser/pixels and fullGT membership checked separately'}
