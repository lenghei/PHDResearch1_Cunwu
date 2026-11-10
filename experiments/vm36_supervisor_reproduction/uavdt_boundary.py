"""Document conversion-edge differences without modifying either annotation source."""
import argparse
from collections import Counter
import hashlib
from pathlib import Path
import time
import cv2
from protocol import ROOT,EXPERIMENT,load,save,source,write_atomic
from common import seal,verify_hashes
from historical_review import csv_file
from main.pipeline import new_run


def clipped(value,width,height):
    identifier,x,y,w,h,o,c,category=value
    left=max(0,min(width-1,x));top=max(0,min(height-1,y))
    right=max(0,min(width-1,x+w));bottom=max(0,min(height-1,y+h))
    return identifier,left,top,right-left,bottom-top,o,c,category


def execute(reconciliation):
    started=time.perf_counter();verify_hashes(reconciliation,('MISMATCHES.json','RECONCILIATION_RESULT.json','OFFICIAL_MEMBER_SHA256.json'))
    out,_=new_run({'output_root':str(reconciliation.parent)},EXPERIMENT,'uavdt_boundary_review')
    findings=[]
    for row in load(reconciliation/'MISMATCHES.json'):
        name=f'{row["sequence"]}_img{row["frame"]:06d}.jpg';root=ROOT/'datasets/uavdt'/row['split']
        image_path=root/'img'/name;annotation=root/'ann'/(name+'.json')
        image=cv2.imread(str(image_path),cv2.IMREAD_COLOR|cv2.IMREAD_IGNORE_ORIENTATION)
        if image is None:raise ValueError('Cannot decode boundary example.')
        height,width=image.shape[:2];metadata=load(annotation)
        if metadata['size']!={'width':width,'height':height}:raise ValueError('Actual pixels and annotation size differ.')
        before=Counter({tuple(value):count for value,count in row['missing']})
        after=Counter({tuple(value):count for value,count in row['extra']})
        transformed=Counter()
        for value,count in before.items():transformed[clipped(value,width,height)]+=count
        findings.append({'split':row['split'],'sequence':row['sequence'],'frame':row['frame'],
            'width':width,'height':height,'original_difference':row['missing'],'converted_difference':row['extra'],
            'consistent_with_inclusive_edge_clipping':transformed==after,
            'image':source(image_path),'converted_annotation':source(annotation),
            'inputs_modified':False})
    consistent=all(r['consistent_with_inclusive_edge_clipping'] for r in findings)
    save(out/'BOUNDARY_FINDINGS.json',findings,out)
    result={'status':'ORIGINAL_SOURCE_RECOVERED_EDGE_DIFFERENCES_DOCUMENTED' if consistent else 'BOUNDARY_DIFFERENCES_UNRESOLVED',
        'reviewed_frames':len(findings),'all_consistent_with_right_bottom_W_minus_1_H_minus_1_clipping':consistent,
        'historical_converter_cause':'NOT_PROVEN; observed coordinate correspondence only.',
        'source_choice_proposal':'Use original official GT widths/heights and original ignore records directly; do not silently repair either source.',
        'evaluation_admission':'NOT_ADMITTED','official_GT_ignore_source':source(reconciliation/'inputs.json'),
        'remaining':['Image-byte/geometry qualification for the intended external subset','Versioned category and metric/ignore protocol',
            'Checkpoint exposure and independent-scene evidence'],
        'new_GPU_experiments':0,'wall_seconds':time.perf_counter()-started}
    save(out/'ANNOTATION_SOURCE_DECISION.json',result,out)
    save(out/'inputs.json',{'reconciliation':source(reconciliation/'SHA256SUMS.txt'),'worker':source(Path(__file__))},out)
    write_atomic(out/'diary.md','# Boundary review\n\nDecoded the actual differing frames and compared original and converted geometry. No pixel or label repair. A matching clipping formula does not prove the historical conversion cause or make one-pixel differences harmless for tiny-object evaluation.\n',out)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>UAVDT conversion-edge review</h1><a href="ANNOTATION_SOURCE_DECISION.json">Decision</a> · <a href="BOUNDARY_FINDINGS.json">Examples</a>',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'UAVDT_BOUNDARY_REVIEW','status':result['status']},out,replace=True)
    seal(out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reconciliation-run',required=True);args=p.parse_args()
    if Path(args.reconciliation_run).name!=args.reconciliation_run:raise ValueError('Expected a plain RunID.')
    execute(ROOT/'datasets/local_runs'/EXPERIMENT/args.reconciliation_run)
