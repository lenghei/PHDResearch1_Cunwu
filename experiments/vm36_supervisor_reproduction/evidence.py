"""Read-only source, dataset and sampling checks for VM36."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path,PurePosixPath
import re
import subprocess
import sys
import time
from zipfile import ZipFile

from protocol import ROOT,PUBLIC,HERE,PRIOR,BASE,DEV,STUDY,ADMIT,EXPERIMENT,FMP,FMD,LEGACY_FMP,LEGACY_FMD,load,save,source,sha256,table,csv_file,write_atomic
from common import FLAMA,verify_hashes
from archives import OriginalZIP,members
from main.labels import VIS_NAMES,ground_truth


def transcript_gate(out,meeting):
    audio=meeting/'Standard recording 51.mp3';legacy=meeting/'20261109.md'
    if not audio.is_file() or not legacy.is_file():raise FileNotFoundError('Meeting audio or legacy text not found.')
    script="""$a=$env:UAVGAP_AUDIO_METADATA_SOURCE
$s=New-Object -ComObject Shell.Application
$f=$s.NameSpace((Split-Path -LiteralPath $a))
$i=$f.ParseName([IO.Path]::GetFileName($a))
@{duration_100ns=$i.ExtendedProperty('System.Media.Duration');bitrate=$i.ExtendedProperty('System.Audio.EncodingBitrate');channels=$i.ExtendedProperty('System.Audio.ChannelCount');sample_rate=$i.ExtendedProperty('System.Audio.SampleRate')} | ConvertTo-Json
"""
    process=subprocess.run(['powershell','-NoProfile','-NonInteractive','-Command',script],
        env=dict(os.environ,UAVGAP_AUDIO_METADATA_SOURCE=str(audio)),capture_output=True,text=True,timeout=25,check=True)
    metadata=json.loads(process.stdout);duration=int(metadata['duration_100ns'])/10_000_000
    text=legacy.read_text(encoding='utf-8');start=text.index('> **[');stop=text.index('## 二、')
    partial=text[start:stop].strip().removesuffix('---').strip()
    headers=list(re.finditer(r'> \*\*\[(\d+:\d+)\s*-\s*(\d+:\d+)\]\*\*',partial))
    seconds=lambda s:sum(int(x)*m for x,m in zip(s.split(':'),(60,1)))
    segments=[]
    for i,m in enumerate(headers):
        body=partial[m.end():headers[i+1].start() if i+1<len(headers) else len(partial)].strip()
        segments.append({'start_seconds':seconds(m[1]),'end_seconds':seconds(m[2]),
            'source_text_sha256':hashlib.sha256(body.encode()).hexdigest(),'audio_verified':False,
            'speaker_identity':'UNVERIFIED_LEGACY_LABELS','original_language':'English as written; not checked against audio'})
    local={n:bool(importlib.util.find_spec(n)) for n in ('whisper','faster_whisper','vosk','transformers','speech_recognition')}
    gate={'status':'FULL_TRANSCRIPT_BLOCKED','audio':source(audio),'legacy_text':source(legacy),
        'metadata_method':'Read-only Windows Shell System.Media.Duration; no audio upload or recognition.',
        'metadata':metadata,'duration_seconds':duration,'expected_duration_seconds':2545.93,
        'duration_difference_seconds':duration-2545.93,'local_ASR_packages':local,
        'local_model_cache':'No Hugging Face hub/model directories found in the inspected user cache.',
        'search_scope':'Selected project/runtime environments and user model cache; not an exhaustive search of all installed applications.',
        'verified_transcribed_seconds':0,'legacy_claimed_end_seconds':max(r['end_seconds'] for r in segments),
        'legacy_text_missing_interval_seconds':[870,duration],
        'audio_verification_missing_interval_seconds':[0,duration],
        'actual_meeting_end':'UNVERIFIED','later_discussion':'UNVERIFIED','unintelligible_spans':'NOT_EVALUATED',
        'speaker_identification':'UNVERIFIED','external_upload':False,'student_review':'REQUIRED','segments':segments}
    save(out/'audio_alignment_manifest.json',gate,out)
    cn=ROOT/'datasets/reportcn'
    prefix='<!-- VM36_MEETING_STATUS_V1 -->\n'
    full=prefix+'# 2026-11-09录音转写状态\n\n**FULL_TRANSCRIPT_BLOCKED：这不是完整录音逐字稿。**\n\n'
    full+=f'原始录音时长：{duration:.6f}秒，约42分25.93秒；SHA256：`{gate["audio"]["sha256"]}`。没有上传外部服务。\n\n'
    full+='已核对音频的转写时长为0秒。旧文档仅声称覆盖00:00—14:30，尚未逐段听核；其说话人标签也未经确认。14:30—42:25.93没有可用转写；不能据旧稿的告别语认定会议在14:30结束。\n\n'
    full+='当前检查的环境没有可用本地ASR及模型，因此没有补写剩余内容，也没有把未听取内容标为“听不清”。需要本地转写工具及模型，或作者批准外部处理后，才能继续完整转写。\n\n'
    full+='## 旧文档文字摘录（仅供学生核对，不是核验后的导师引文）\n\n'+partial+'\n'
    write_atomic(cn/'meeting_20261109_transcript_full.md',full,out)
    translated=prefix+'# 2026-11-09录音中文翻译状态\n\n**FULL_TRANSLATION_BLOCKED**\n\n'
    translated+='原语言完整转写尚未完成，因此没有生成或推断全录音中文译文。旧00:00—14:30英文稿也未经音频核验，不能把它的分析段、人物归属或强烈措辞当成已经确认的导师原话。\n\n'
    translated+='原稿与缺口见同目录meeting_20261109_transcript_full.md；完整时间轴和文件SHA见本次Run的audio_alignment_manifest.json。学生完成听核后再翻译。\n'
    write_atomic(cn/'meeting_20261109_chinese_translation.md',translated,out)
    requirements=prefix+'# 会议决策要求与来源边界\n\n'
    requirements+='已直接查看两张手稿。白纸可辨内容为：Problem；Why is a problem；Solution；Why is a good solution，以及Reason behind decision。蓝纸可辨内容包含YOLO variability、Dataset variability、blur/motion/lights、640/1280。未把图中不明数值解释为真实相机姿态或飞行参数。\n\n'
    requirements+='原17页计划及作者本轮任务是当前可用要求来源；完整会议音频内容仍待核验。每个重要选择须列出替代方案、依据、验证方法和未知项。DECISION_REGISTER.csv中的NEEDS_SCIENTIFIC_REVIEW不会由程序自动改为已批准。\n\n'
    requirements+='旧文档中547张、pairwise保证90%失效发现、AI只参与排版、15 m/s与5—10°物理解释等说法不再作为实验依据。具体原文行号和修正见本Run的unsupported_notes.csv。\n'
    write_atomic(cn/'meeting_20261109_decision_requirements.md',requirements,out)
    findings=[]
    patterns=[('wrong_val_count',r'547','Official validation has 548 images.'),
       ('fixed_fault_percentage',r'90%|90％','Feature-state coverage does not guarantee a fault-discovery percentage.'),
       ('AI_formatting_only',r'AI was only used|completely written by myself','Do not claim formatting-only assistance or sole manual coding contrary to the development record.'),
       ('unverified_physical_parameters',r'15m/s|5-10°|5–10°','No verified camera/flight calibration supports this mapping.'),
       ('venue_conflict',r'AAAI','The original plan leaves the venue decision to supervisors; the sketch does not grant a venue acceptance.')]
    for number,line in enumerate(text.splitlines(),1):
        for code,pattern,correction in patterns:
            if re.search(pattern,line,re.I):findings.append({'source':'20261109.md','source_sha256':sha256(legacy),
                'line':number,'issue':code,'original_text':line,'correction':correction,'old_file_modified':False})
    csv_file(out/'unsupported_notes.csv',findings)
    return gate


def dataset_check(out):
    from PIL import Image
    start=time.perf_counter();root=ROOT/'datasets/VisDrone';registry={};all_members=[];issues=[];duplicate_groups=defaultdict(list)
    for split,expected in [('train',6471),('val',548),('test-dev',1610),('test-challenge',1580)]:
        archive_path=root/f'VisDrone2019-DET-{split}.zip';entry={'archive':source(archive_path)}
        with ZipFile(archive_path) as archive:
            images=sorted(members(archive,'images',('.jpg','.jpeg','.png')))
            annotations={PurePosixPath(n).stem:n for n in members(archive,'annotations',('.txt',))}
            ids=[PurePosixPath(n).stem for n in images]
            if len(ids)!=expected or len(ids)!=len(set(ids)) or len(archive.namelist())!=len(set(archive.namelist())):
                raise ValueError('Dataset membership mismatch: '+split)
            if split!='test-challenge' and set(ids)!=set(annotations):raise ValueError('Annotation membership differs: '+split)
            image_values=[];invalid=0
            for image_name in images:
                stem=PurePosixPath(image_name).stem;blob=archive.read(image_name);digest=hashlib.sha256(blob).hexdigest()
                with Image.open(io.BytesIO(blob)) as image:
                    width,height=image.size;orientation=image.getexif().get(274)
                label=archive.read(annotations[stem]) if stem in annotations else None
                label_sha=hashlib.sha256(label).hexdigest() if label is not None else None
                count=0
                if label is not None:
                    for lineno,line in enumerate(label.decode('utf-8-sig').splitlines(),1):
                        if not line.strip():continue
                        try:
                            values=[float(v) for v in line.split(',') if v.strip()]
                            valid=len(values)>=8 and all(math.isfinite(v) for v in values) and values[2]>0 and values[3]>0 and values[5]==int(values[5]) and 0<=values[5]<=11
                        except (ValueError,IndexError):valid=False
                        if not valid:
                            invalid+=1;issues.append({'split':split,'image':stem,'annotation_line':lineno,'original_value':line,'action':'PRESERVED; no filtering approved'})
                        count+=1
                row={'split':split,'image_id':stem,'image_SHA256':digest,'annotation_SHA256':label_sha,
                    'ZIP_image_member':image_name,'ZIP_annotation_member':annotations.get(stem),
                    'encoded_width':width,'encoded_height':height,'EXIF_orientation':orientation,'raw_annotation_rows':count}
                image_values.append(row);all_members.append(row);duplicate_groups[digest].append({'split':split,'image_id':stem,'annotation_SHA256':label_sha})
            entry.update(images=len(images),annotations=len(annotations),invalid_annotation_rows=invalid,
                membership_status='VALIDATED')
            entry['image_member_manifest_sha256']=hashlib.sha256(json.dumps(image_values,sort_keys=True).encode()).hexdigest()
            registry[split]=entry
    csv_file(out/'complete_split_members.csv',all_members)
    csv_file(out/'annotation_issues.csv',issues or [{'split':'None','image':None,'annotation_line':None,'original_value':None,'action':'No issue detected'}])
    duplicates={k:v for k,v in duplicate_groups.items() if len(v)>1}
    save(out/'duplicate_image_bytes.json',duplicates,out)
    records=[];provenance=[];zip_reader=OriginalZIP(root/'VisDrone2019-DET-val.zip',548);classes=Counter()
    try:
        for index in range(548):
            image,rec,src=zip_reader.record(index)
            if list(image.shape)!=src['shape']:raise ValueError('Decoded shape mismatch.')
            records.append(rec);provenance.append(src);classes.update(o['class_id'] for o in rec['objects'])
    finally:zip_reader.close()
    coco,_=ground_truth(records,'native10')
    save(out/'full548_ground_truth.json',{'records':records,'coco':coco,'source_archive':registry['val']['archive']},out)
    save(out/'full548_image_manifest.json',provenance,out)
    area=Counter();post=Counter()
    for a in coco['annotations']:
        n=a['bbox'][2]*a['bbox'][3]
        area['tiny' if n<256 else 'small' if n<1024 else 'medium' if n<9216 else 'large']+=1
        post[a['category_id']-1]+=1
    cross={}
    for split in ('train','val'):
        left={r['image_id'] for r in all_members if r['split']==split};right={r['image_id'] for r in all_members if r['split']=='test-dev'}
        cross[split]={'image_ID_overlap':len(left&right),'byte_overlap_groups':sum(any(a['split']==split for a in v) and any(a['split']=='test-dev' for a in v) for v in duplicates.values())}
    report={'status':'PARTIAL','splits':registry,'development_validation':{'status':'VALIDATED','images':548,
        'actual_OpenCV_decodes':548,'excluded_images':[],'class_order':VIS_NAMES,'classes_before_ignore':dict(classes),
        'classes_after_ignore':dict(post),'size_counts_after_ignore':dict(area),'GT_after_ignore':len(coco['annotations']),
        'ignore_policy':'Original native10 rasterized ignore-union; original raw records retained.',
        'duplicate_byte_groups':sum(all(a['split']=='val' for a in v) for v in duplicates.values()),'scope':'DEVELOPMENT_ONLY'},
        'formal_testdev':'DATASET_BLOCKED','prior_use':'UNKNOWN_PRIOR_USE','testdev_disjointness':cross,
        'training_gate':'DATASET_INVALID_FOR_UNFILTERED_TRAINING' if registry['train']['invalid_annotation_rows'] else 'VALIDATED_SOURCE_ONLY',
        'official_source':'https://github.com/VisDrone/VisDrone-Dataset','official_archive_identity':'Matches local saved archive SHA certificates where available; local filename alone is not official provenance.',
        'wall_seconds':time.perf_counter()-start,'GPU_seconds':0}
    if registry['val']['invalid_annotation_rows']:report['development_validation']['status']='DATASET_INVALID'
    save(out/'DATASET_ADMISSION.json',report,out)
    return report


def model_and_sampling(out):
    start=time.perf_counter();outputs={}
    for name,script,model in [('FM_P','flamapy_select.py',FMP),('FM_D','flamapy_count.py',FMD)]:
        command=[str(FLAMA),'-B',str(HERE/'validate_models.py'),'--model',str(model),'--domain',name]
        if name=='FM_P':command+=['--seed','20261109']
        p=subprocess.run(command,cwd=out,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120,check=True,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        outputs[name]={'command':command,'response':json.loads(p.stdout.splitlines()[-1]),'UVL':source(model)}
    if outputs['FM_P']['response']['valid_count']!=46368 or outputs['FM_D']['response']['valid_count']!=635904:raise ValueError('Count mismatch.')
    outputs.update(status='PARTIAL',logic_status='VALIDATED',realistic_constraints='NEEDS_SCIENTIFIC_REVIEW',
        count_method='FlamaPy SAT/dead-feature checks and recorded exact counts; FM_D factorization is not an independent BDD/#SAT count.',
        prior_numeric_audit=source(PRIOR/'parameter_origin_audit.csv'),wall_seconds=time.perf_counter()-start,GPU_seconds=0)
    save(out/'FEATURE_MODEL_VALIDATION.json',outputs,out)
    from sampling import models
    fmp,_,prior=models();frozen=ROOT/'datasets/local_runs/cv261207/cv261207_audit_20261104T193649_d1b63492/selection_manifest.json'
    manifest=load(frozen);rows=[]
    for i,r in enumerate(manifest['pairwise']['cohort']):
        rows.append({'cohort':'historical_pairwise','position':i,'configuration_id':r['configuration_id'],
            'selected_leaves':json.dumps(r['selected_leaves']),'population_size':46368,'inclusion_probability':None,
            'seed':manifest['seed'],'role':'greedy_core' if i<manifest['pairwise']['greedy_cover_size'] else 'padding',
            'status':'FROZEN_REFERENCE_NOT_NEW_MEASUREMENT'})
    for r in manifest['full_space_uniform_candidates']:
        rows.append({'cohort':'historical_full_space_uniform','position':r['valid_space_index'],
            'configuration_id':r['configuration_id'],'selected_leaves':json.dumps(r['selected_leaves']),
            'population_size':46368,'inclusion_probability':100/46368,'seed':manifest['seed'],'role':'uniform',
            'status':'FROZEN_REFERENCE_NOT_NEW_MEASUREMENT'})
    selected={r['degradation']['degradation_id']:r['degradation']['leaves'] for r in manifest['pairs']}
    for identity,leaves in selected.items():
        rows.append({'cohort':'historical_geometry_eligible_uniform','position':None,'configuration_id':identity,
            'selected_leaves':json.dumps(leaves),'population_size':26496,'inclusion_probability':100/26496,
            'seed':manifest['seed'],'role':'uniform','status':'FROZEN_REFERENCE_NOT_NEW_MEASUREMENT'})
    csv_file(out/'SAMPLING_MANIFEST.csv',rows)
    selections=[r['selected_leaves'] for r in manifest['pairwise']['cohort']]
    masks,_=prior.compute_binary_pair_masks(selections,sorted(fmp.ALL_LEAVES));cover=0;core=0
    for i,mask in enumerate(masks):
        cover|=mask
        if i<manifest['pairwise']['greedy_cover_size']:core|=mask
    if cover!=core or cover.bit_count()!=1163:raise ValueError('Pairwise coverage mismatch.')
    verify={'status':'VALIDATED','frozen_source':source(frozen),'pairwise_cohort':len(selections),'greedy_core':21,
         'padding':59,'covered_feasible_states':cover.bit_count(),'infeasible_states':4*300-cover.bit_count(),
         'uniform_design':'Python random.Random.sample without replacement over exact enumerated legal population.',
         'uniform_probability_proof':'Each n-subset has probability 1/C(N,n); each member has inclusion probability n/N under an ideal uniform PRNG draw.',
         'PRNG_limit':'Pseudorandom finite-state implementation, reproducibility is exact; not cryptographic randomness or a BDD sampler.',
         'tie_break':'Greedy uses earliest maximum in stable enumeration order; padding uses the frozen PRNG state.',
         'first_uniform_draws':'Full-space uniform follows padding draws in the historical PRNG stream. Replaying only Random(seed).sample is not the same cohort.',
         'V1_pairs':len(manifest['pairs']),'V2_pairs':len(load(BASE/'cross_architecture_v2.json')['pairs']),
         'active':'NOT_EVALUATED','full_product':46368*635904,'new_measurements':0}
    save(out/'sampling_verification.json',verify,out)
    write_atomic(out/'SAMPLING_RATIONALE.md',sampling_rationale(verify),out)
    save(out/'metadata_status.json',{'status':'PDF_ALIGNED_NEW_VERSION','models':[source(FMP),source(FMD)],
       'alignment':source(out/'MODEL_ALIGNMENT.json'),'warning':'Draft params_m are not actual checkpoint counts; runtime extensions still need review.'},out)
    return outputs,verify


def sampling_rationale(v):
    return '''# Sampling rationale — factual worksheet for student review

The full logical product is 46,368 × 635,904 = 29,485,596,672 pairs. It is not an executed experiment count.

Pairwise covers feasible binary presence/absence states over the 25 FM_P leaves. There are 300 leaf pairs and four states per pair: 1,163 feasible states and 37 impossible states. The deterministic greedy core uses 21 legal configurations; the existing frozen cohort adds 59 PRNG-selected configurations, totalling 80. It does not cover the whole two-model Cartesian product and guarantees no fixed fault-discovery rate.

Uniform sampling answers a different question. The existing full-space cohort draws 100 from 46,368 legal configurations. The executable cohort draws 100 from 26,496 legal configurations without rolling shutter. Their member inclusion probabilities are 100/46,368 and 100/26,496 respectively; estimates cannot share the wrong denominator. The saved seed, algorithm order and selected IDs are part of the sample identity.

The PRNG sample is without replacement over an explicitly enumerated finite population. Under the uniform sampling abstraction, every n-subset has probability 1/C(N,n), and every member has probability n/N. Small-population exhaustive tests and deterministic replay check the implementation; they do not prove a scientific fault-detection rate.

Active sampling remains NOT_EVALUATED. It requires a validated measured-label predictor and a frozen acquisition rule. Its selected responses cannot enter an unweighted prevalence estimate.

V1's 500 pairs and V2's 900 pairs remain unexecuted formal plans. The approximately 4,000-pair supervisor target is larger. The proposed 36-condition full-validation factorial is a separate development design awaiting student review; it replaces neither frozen manifest.

Student to explain and approve: which population is relevant to the mission, which interactions matter, and which failure/timing criterion is defensible. No scientific approval is inferred from these engineering checks.
'''


def audit(out,meeting,source_run=None):
    if (out/'SHA256SUMS.txt').exists():raise ValueError('Run is sealed.')
    if not meeting:raise ValueError('Supply the source meeting directory.')
    protocol=load(out/'EXPERIMENT_PROTOCOL.json')
    expected=(out/'PROTOCOL_SHA256.txt').read_text().split()[0]
    if sha256(out/'EXPERIMENT_PROTOCOL.json')!=expected:raise ValueError('Proposal hash changed.')
    started=time.perf_counter()
    original_inputs=[FMP,FMD,PRIOR/'compliance_stage_04.json',DEV/'configuration_metrics.csv',
          STUDY/'protocol.json',ADMIT/'admission.json',meeting/'Standard recording 51.mp3',meeting/'20261109.md',
          meeting/'IMG20261109115938.jpg',meeting.parent/'20261026/图像 2026年10月26日 12_25_40.jpg']
    before={str(p):sha256(p) for p in original_inputs}
    save(out/'inputs.json',{str(p):source(p) for p in original_inputs},out)
    if source_run:
        if Path(source_run).name!=source_run:raise ValueError('Expected a plain source RunID.')
        previous=ROOT/'datasets/local_runs'/EXPERIMENT/source_run
        verify_hashes(previous,('DATASET_ADMISSION.json','audio_alignment_manifest.json','full548_ground_truth.json','full548_image_manifest.json','complete_split_members.csv'))
        dataset=load(previous/'DATASET_ADMISSION.json');transcript=load(previous/'audio_alignment_manifest.json')
        for split,entry in dataset['splits'].items():
            if sha256(Path(entry['archive']['path']))!=entry['archive']['sha256']:raise ValueError('Source dataset changed: '+split)
        for key in ('audio','legacy_text'):
            if sha256(Path(transcript[key]['path']))!=transcript[key]['sha256']:raise ValueError('Meeting source changed.')
        dataset['reused_evidence']={'RunID':source_run,'source':source(previous/'DATASET_ADMISSION.json'),'new_OpenCV_decodes':0}
        save(out/'DATASET_ADMISSION.json',dataset,out)
        save(out/'audio_alignment_manifest.json',{'status':transcript['status'],'source':source(previous/'audio_alignment_manifest.json'),
             'verified_transcribed_seconds':0,'external_upload':False},out)
        save(out/'full548_data_references.json',{name:source(previous/name) for name in ('full548_ground_truth.json','full548_image_manifest.json','complete_split_members.csv','duplicate_image_bytes.json','annotation_issues.csv')},out)
    else:
        transcript=transcript_gate(out,meeting)
        dataset=dataset_check(out)
    logic,sampling=model_and_sampling(out)
    for path,digest in before.items():
        if sha256(Path(path))!=digest:raise ValueError('Original input changed: '+path)
    save(out/'cpu_audit.json',{'status':'PARTIAL','full548_data_gate':dataset['development_validation']['status'],
       'transcript':transcript['status'],'logic':logic['logic_status'],'sampling':sampling['status'],
       'wall_seconds':time.perf_counter()-started,'GPU_seconds':0,'new_inference_pairs':0,
       'original_inputs_unchanged':True,'scientific_approval':'NEEDS_SCIENTIFIC_REVIEW'},out)
    print(out,flush=True)
