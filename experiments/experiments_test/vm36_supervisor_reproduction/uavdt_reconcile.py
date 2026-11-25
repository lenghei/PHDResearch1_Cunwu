"""Compare the converted M sequences against official DET GT and ignore files."""
import argparse
from collections import Counter,defaultdict
import hashlib
from pathlib import Path
import re
import time
from zipfile import ZipFile
from protocol import ROOT,EXPERIMENT,load,save,source,write_atomic
from common import seal,verify_hashes
from historical_review import csv_file
from main.pipeline import new_run


def execute(toolkit):
    started=time.perf_counter();verify_hashes(toolkit,('SOURCE.json','UAV-benchmark-MOTD_v1.0.zip'))
    archive_path=toolkit/'UAV-benchmark-MOTD_v1.0.zip'
    out,_=new_run({'output_root':str(toolkit.parent)},EXPERIMENT,'uavdt_reconciliation')
    dataset=ROOT/'datasets/uavdt';frames=[];issues=[];source_members=[];sequence_rows=[];ignored=[]
    category={'car':1,'truck':2,'bus':3};out_codes={'no out':1,'medium out':2,'small out':3}
    occ_codes={'no occlusion':1,'lagre occlusion':2,'medium occlusion':3,'small occlusion':4}
    with ZipFile(archive_path) as archive:
        prefix='UAV-benchmark-MOTD_v1.0/'
        evaluator=archive.read(prefix+'utils/CalculateDetectionPR_overall.m')
        text=evaluator.decode('utf-8','replace')
        test_seq=re.findall(r"'([^']+)'",re.search(r'seqDirs\s*=\s*\{([^}]+)\}',text)[1])
        lengths=[int(x) for x in re.search(r'seqLens\s*=\s*\[([^]]+)\]',text)[1].split(',')]
        official_test=dict(zip(test_seq,lengths))
        names=[n for n in archive.namelist() if n.endswith('_gt_whole.txt')]
        official_sequences={Path(n).name.split('_')[0] for n in names}
        for split in ('train','test'):
            files=sorted((dataset/split/'ann').glob('*.json'))
            excluded=[p for p in files if p.name.startswith('S')]
            ignored.append({'split':split,'SOT_files_excluded':len(excluded),'reason':'Not DET ground truth; no vehicle-to-DET class substitution.'})
            by_sequence=defaultdict(list)
            for p in files:
                if not p.name.startswith('M'):continue
                match=re.fullmatch(r'(M\d+)_img(\d+)\.jpg\.json',p.name)
                if not match:raise ValueError('Unexpected M-frame name.')
                by_sequence[match[1]].append((int(match[2]),p))
            for seq,items in by_sequence.items():
                if time.perf_counter()-started>240:raise TimeoutError('Bounded reconciliation stopped.')
                if seq not in official_sequences:raise ValueError('Sequence absent from official GT.')
                raw=archive.read(prefix+'GT/'+seq+'_gt_whole.txt');ignore=archive.read(prefix+'GT/'+seq+'_gt_ignore.txt')
                expected=defaultdict(Counter);ignore_by_frame=Counter()
                for line in raw.decode('utf-8-sig').splitlines():
                    if not line.strip():continue
                    v=tuple(int(x) for x in line.split(','))
                    if len(v)!=9:raise ValueError('Unexpected official DET row format.')
                    expected[v[0]][v[1:]]+=1
                for line in ignore.decode('utf-8-sig').splitlines():
                    if line.strip():ignore_by_frame[int(line.split(',')[0])]+=1
                source_members.append({'sequence':seq,'GT_SHA256':hashlib.sha256(raw).hexdigest(),
                    'ignore_SHA256':hashlib.sha256(ignore).hexdigest(),'GT_rows':sum(sum(c.values()) for c in expected.values()),
                    'ignore_rows':sum(ignore_by_frame.values())})
                mismatch=0;objects=0
                for frame,p in items:
                    blob=p.read_bytes();item=__import__('json').loads(blob)
                    actual=Counter()
                    for obj in item.get('objects',[]):
                        tags={t['name']:t.get('value') for t in obj['tags']}
                        a,b=obj['points']['exterior'];left,top=a;right,bottom=b
                        o=next((code for name,code in out_codes.items() if name in tags),None)
                        c=next((code for name,code in occ_codes.items() if name in tags),None)
                        actual[(tags.get('target id'),left,top,right-left,bottom-top,o,c,category.get(obj['classTitle']))]+=1
                    equal=actual==expected[frame]
                    objects+=sum(actual.values());mismatch+=not equal
                    if not equal and len(issues)<200:issues.append({'split':split,'sequence':seq,'frame':frame,
                        'missing':list((expected[frame]-actual).items())[:3],'extra':list((actual-expected[frame]).items())[:3]})
                    frames.append({'split':split,'sequence':seq,'frame':frame,'image_file':p.name.removesuffix('.json'),
                        'converted_annotation_SHA256':hashlib.sha256(blob).hexdigest(),'matched_official_GT':equal,
                        'GT_instances':sum(expected[frame].values()),'official_ignore_regions':ignore_by_frame[frame],
                        'annotation_width':item['size']['width'],'annotation_height':item['size']['height']})
                correct_split=(seq in official_test)==(split=='test')
                valid_frames={f for f,_ in items}==set(range(1,max(f for f,_ in items)+1))
                length_ok=seq not in official_test or len(items)==official_test[seq]
                sequence_rows.append({'split':split,'sequence':seq,'frames':len(items),'objects':objects,
                    'mismatch_frames':mismatch,'official_test_membership_matches':correct_split,
                    'contiguous_frames':valid_frames,'official_test_length_matches':length_ok})
        save(out/'OFFICIAL_EVALUATION_SEMANTICS.json',{'member':'utils/CalculateDetectionPR_overall.m',
            'SHA256':hashlib.sha256(evaluator).hexdigest(),'test_sequences':official_test,
            'ignore_rule':'The official routine filters detections strictly contained in ignore rectangles for its listed sequences.',
            'metric':'The supplied overall DET routine uses VOC-style PR/AP; do not call a custom COCO ignore-union result official UAVDT AP.',
            'readme_GT_format':'frame,id,left,top,width,height,out-of-view,occlusion,category; categories1 car/2 truck/3 bus.',
            'licensing':'Toolkit README states research purpose and GNU-GPL for the library. Dataset redistribution is not inferred.'},out)
    csv_file(out/'M_FRAME_RECONCILIATION.csv',frames);csv_file(out/'SEQUENCE_RECONCILIATION.csv',sequence_rows)
    save(out/'MISMATCHES.json',issues,out);save(out/'OFFICIAL_MEMBER_SHA256.json',source_members,out)
    ok=all(r['mismatch_frames']==0 and r['official_test_membership_matches'] and r['contiguous_frames'] and r['official_test_length_matches'] for r in sequence_rows)
    result={'status':'ANNOTATION_SOURCE_MATCHED_EVALUATION_NOT_ADMITTED' if ok else 'ANNOTATION_MISMATCH_UNRESOLVED',
        'M_sequences':len(sequence_rows),'M_frames_compared':len(frames),'GT_instances':sum(r['GT_instances'] for r in frames),
        'mismatched_frames':sum(r['mismatch_frames'] for r in sequence_rows),'ignore_rows_recovered':sum(r['ignore_rows'] for r in source_members),
        'SOT_exclusion':ignored,'original_inputs_modified':False,'new_GPU_inferences':0,
        'remaining':['Image-byte provenance/actual image geometry audit','Versioned compatible label/evaluator protocol',
            'Checkpoint training/selection/method exposure and independent-scene boundaries'],
        'wall_seconds':time.perf_counter()-started}
    save(out/'RECONCILIATION_RESULT.json',result,out)
    save(out/'inputs.json',{'official_toolkit':source(archive_path),'source':source(Path(__file__))},out)
    write_atomic(out/'diary.md','# Official annotation reconciliation\n\nCompared only official M sequences. SOT frames remain excluded. Recovered source ignore records without filling or modifying converted labels. Source recovery does not admit GPU evaluation or prove independent model exposure.\n',out)
    write_atomic(out/'index.html','<!doctype html><meta charset="utf-8"><h1>UAVDT original GT reconciliation</h1><a href="RECONCILIATION_RESULT.json">Result</a> · <a href="SEQUENCE_RECONCILIATION.csv">Sequences</a>',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'UAVDT_ANNOTATION_RECONCILIATION','status':result['status']},out,replace=True)
    seal(out);print(out,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--toolkit-run',required=True);args=p.parse_args()
    if Path(args.toolkit_run).name!=args.toolkit_run:raise ValueError('Expected a plain RunID.')
    execute(ROOT/'datasets/local_runs'/EXPERIMENT/args.toolkit_run)
