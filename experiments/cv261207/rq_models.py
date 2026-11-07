"""Reconcile UVL validity, numeric attributes and executable detector features."""
from collections import Counter
import copy
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path

from common import ROOT, FMP, FMD, BASELINE, EXPERIMENTS, load, save, sha256, source
from historical_review import csv_file
from sampling import models


def installed_sources():
    import ultralytics
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.engine.exporter import Exporter
    from ultralytics.data import augment
    objects={'TTA':DetectionModel._predict_augment,'INT8_export':Exporter,
             'training_augment':augment.v8_transforms,'RandomHSV':augment.RandomHSV,
             'Albumentations':augment.Albumentations}
    records={}
    for key,obj in objects.items():
        text=inspect.getsource(obj)
        records[key]={'file':source(Path(inspect.getfile(obj))),
                      'relevant_lines':[line.strip() for line in text.splitlines()
                           if any(word in line for word in ('end2end','augment','INT8','int8','MotionBlur',
                               'ImageCompression','Blur(', 'CLAHE(', 'lut_val','hsv_v'))][:30]}
    return {'ultralytics':ultralytics.__version__,
            'sahi_installed':importlib.util.find_spec('sahi') is not None,
            'sources':records,
            'SAHI_TTA':'A preserved study exclusion, not an established universal API incompatibility.',
            'NMS_free':'Available for compatible end-to-end heads; not interchangeable with pr01 dense-head NMS.',
            'TTA':'Installed DetectionModel._predict_augment checks end-to-end models; source support alone is not a protocol test.',
            'INT8':'Export support does not provide a calibrated compatible checkpoint/runtime.'}


def rq1(out, audit):
    fmp,fmd,_=models()
    verified=load(audit/'model_validation.json')
    if sha256(FMP)!=verified['FM_P']['UVL_SHA256'] or sha256(FMD)!=verified['FM_D']['UVL_SHA256']:
        raise ValueError('Original UVLs differ from accepted evidence.')
    configs=list(fmp.combinations())
    independent=fmd.enumerate_count()
    attrs=fmp.read_uvl_attributes(FMP)
    if len(configs)!=46368 or independent['independent_configurations']!=635904 or independent['independent_training_variants']!=844:
        raise ValueError('Exact structural counts disagree.')
    if len({fmp.configuration_id(c) for c in configs})!=len(configs):
        raise ValueError('Configuration ID collision.')
    csv_file(out/'fmp_attributes.csv',[{'feature':f,'attribute':k,'value':v,'source_sha256':sha256(FMP)}
             for f,values in attrs.items() for k,v in values.items()])
    constraints=[]
    rationales=[
        'Low light can increase photon-noise effects; it does not universally require visible shot noise.',
        'Rain and overexposure can coexist. This is a study scenario exclusion.',
        'Dense haze and exposure gain can coexist. This is a study scenario exclusion.',
        'Rolling shutter need not produce visible motion blur. This is a shared-motion scenario assumption.',
        'Severe shot noise depends on photon statistics and processing, not these labels alone.']
    for domain,file in (('FM_P',FMP),('FM_D',FMD)):
        body=file.read_text(encoding='utf-8').split('\nconstraints\n',1)[1]
        for i,line in enumerate(x.strip() for x in body.splitlines() if x.strip()):
            constraints.append({'domain':domain,'constraint':line,'source_sha256':sha256(file),
                    'status':'PRESERVED_STUDY_ASSUMPTION',
                    'rationale':rationales[i] if domain=='FM_P' else
                       ('Preserved checkpoint/domain restriction; not a hardware impossibility.' if i!=3 else
                        'SAHI/TTA exclusion lacks a demonstrated universal API prohibition.')})
    csv_file(out/'uvl_constraints.csv',constraints)
    csv_file(out/'configuration_distribution.csv',[{'selected_leaves':k,'configurations':v}
             for k,v in sorted(Counter(map(len,configs)).items())])
    library=installed_sources();save(out/'library_compatibility.json',library)
    matrix=[]
    implemented={'YOLOv8','Small','VisDrone_FT','Train640','HSV_Default','Geometric_Default',
                 'Infer640','Infer960','Infer1280','FullFrame','NMS','FP32','FP16'}
    historical=implemented|{'YOLO11','YOLO26','Train1280'}
    notes={
        'Albu_Default':'Historical activation unknown; installed source is not a per-run activation log.',
        'COCO_ZeroShot':'COCO80 does not implement the native10 task; requires a separate common-label protocol.',
        'Nano':'Healthy nano assets exist, but the current CV261207 worker binds YOLOv8s only.',
        'LEAF_YOLO':'External weights require a separate backend and training provenance.',
        'Infer1536':'Not accepted by the current fixed input-shape worker.',
        'SAHI_Slicing':'Not integrated into the fixed protocol; installed package availability is recorded separately.',
        'TTA':'No TTA validation under pr01; YOLO26 end-to-end source has an augmentation restriction.',
        'INT8':'No admitted calibrated artifact for these checkpoints and this native10 protocol.',
        'EndToEnd_NMSFree':'Would change the fixed dense one-to-many/NMS prediction path.',
        'YOLO11':'Historical pr01 VM26 support; current CV261207 GPU entry binds YOLOv8s.',
        'YOLO26':'Historical pr01 VM26 dense-head support; current CV261207 GPU entry binds YOLOv8s.',
        'Train1280':'Recorded historical checkpoint training attribute, not a newly executed training option.'}
    for feature in sorted(fmd.LEAVES):
        current=feature in implemented
        evidence=feature in historical
        matrix.append({'feature':feature,'logically_valid':True,
            'current_runner_implemented':current,'prior_GPU_evidence':evidence,
            'scope':'FEATURE_OCCURRENCE_NOT_ALL_COMBINATIONS',
            'status':'IMPLEMENTED_SUBSET' if current else 'HISTORICAL_BACKEND_ONLY' if evidence else 'UNSUPPORTED_CURRENT_PROTOCOL',
            'note':notes.get(feature,'Recorded contract feature; exact historical augmentation activation remains separately audited.'
                 if feature in ('HSV_Default','Geometric_Default') else
                 'Matched augmentation or preprocessing execution is not implemented in the current worker.' if not current else
                 'Accepted only within the fixed checkpoint, shape, precision and postprocessing cohort.')})
    if len(matrix)!=34:
        raise ValueError('Incomplete FM_D compatibility matrix.')
    csv_file(out/'fmd_compatibility.csv',matrix)
    examples=[({'Normal'},True),({'LowLight'},False),({'LowLight','Shot_low'},True),
              ({'Normal','RollingShutter'},False),({'Normal','Motion_k7','RollingShutter'},True),
              ({'Overexposed_mild','Rain'},False)]
    csv_file(out/'constraint_examples.csv',[{'selected_leaves':'|'.join(sorted(c)),
              'expected':ok,'actual':fmp.valid_leaves(c),'geometry_executable':'RollingShutter' not in c}
              for c,ok in examples])
    if any(fmp.valid_leaves(c)!=ok for c,ok in examples):
        raise ValueError('Constraint demonstration failed.')
    status={'status':'PARTIAL','logical_models':'VALIDATED','FM_P':{'leaves':25,'constraints':5,'count':46368,
            'dead_features':0,'attributes':len(attrs),'configuration_ids':len(configs),
            'solver_evidence':source(audit/'model_validation.json')},
            'FM_D':{'leaves':34,'constraints':4,'count':635904,'training_variants':844,
                    'SAT':True,'dead_features':0,'count_method':'EXACT_FACTORIZATION_PLUS_INDEPENDENT_STRUCTURAL_COUNT',
                    'BDD_count':'NOT_PERFORMED'},
            'all_configurations_GPU_tested':False,'physical_calibration':'NOT_VALIDATED',
            'FM_D_numeric_attribute_note':'Original v1 has no numeric attribute blocks. Measured per-checkpoint parameters and settings are retained externally; no generic model-size parameter count is invented.',
            'generator_evidence':source(audit/'generator_validation.json'),
            'unsupported_feature_count':sum(not r['current_runner_implemented'] for r in matrix)}
    save(out/'rq1_validation.json',status)
    return configs,fmp,fmd


def checkpoint_static(out, admission, configs, fmp):
    from gap_analysis import static_audit
    base=static_audit(fmp,configs)
    save(out/'static_baseline.json',base)
    rows=[]
    for c in load(admission/'admission.json')['candidates']:
        # Source-version agreement is required before using the brightness proxy.
        version=c['checkpoint_version']
        conservative={'Normal','Overexposed_mild','Underexposed_mild'} if version==base['leaves']['Normal']['source_version'] else {'Normal'}
        for policy,allowed in (('exact_identity',{'Normal'}),('conservative',conservative),
                                ('supervisor_proxy',set(base['definitions']['supervisor_proxy']['covered_leaves']))):
            count=sum(set(cfg).issubset(allowed) for cfg in configs)
            rows.append({'checkpoint':c['id'],'checkpoint_sha256':c['checkpoint_sha256'],
                         'recorded_version':version,'installed_source_version':base['leaves']['Normal']['source_version'],
                         'policy':policy,'covered':count,'gap':len(configs)-count,'population':len(configs),
                         'covered_leaves':'|'.join(sorted(allowed)),
                         'scope':'SOURCE_EVIDENCE_POLICY_NOT_PROOF_OF_PIXEL_REACHABILITY',
                         'historical_Albu_activation':'UNKNOWN'})
    csv_file(out/'static_coverage_by_checkpoint.csv',rows)
    return base


def freeze_v2(out, audit, admission, fmd):
    original=load(audit/'selection_manifest.json')
    cohorts=list({p['degradation']['degradation_id']:p['degradation'] for p in original['pairs']}.values())
    candidates=[c for c in load(admission/'admission.json')['candidates'] if c['id'] in ('ck01','ck05','ck06')]
    architecture={'ck01':'YOLO26','ck05':'YOLOv8','ck06':'YOLO11'}
    detectors=[]
    for c in candidates:
        if sha256(Path(c['checkpoint']['path']))!=c['checkpoint_sha256'] or c['numerical_issues'] or not c['native10']:
            raise ValueError('Historical candidate changed or is unhealthy.')
        for resolution in (640,960,1280):
            declared=fmd.declared_profile_configuration(architecture[c['id']],'Small','VisDrone_FT','Train1280',
                                       resolution,'FP32','FullFrame','NMS')
            detectors.append({'id':c['id']+'_R'+str(resolution),'checkpoint':c['id'],
                    'checkpoint_sha256':c['checkpoint_sha256'],'resolution':resolution,'precision':'FP32',
                    'declared_leaves':declared['selected_features'],
                    'complete_historical_training_feature_binding':'UNVERIFIED_ALBU_ACTIVATION',
                    'evaluation_protocol':'pr01_native10_dense_head_nms'})
    pairs=[{'pair_id':p['degradation_id']+'__'+d['id'],'degradation':p,'detector':d}
           for p in cohorts for d in detectors]
    if len(cohorts)!=100 or len(pairs)!=900 or len({p['pair_id'] for p in pairs})!=900:
        raise ValueError('Unexpected v2 sampling cohort.')
    save(out/'cross_architecture_v2.json',{'version':'CV261207_CROSS_ARCH_V2_PREPARED',
        'status':'PLANNED_NOT_EXECUTED','formal_admission':'DATASET_BLOCKED',
        'original_v1':source(audit/'selection_manifest.json'),'v1_modified':False,
        'detectors':detectors,'pairs':pairs,'eligible_FM_P':original['eligible_FM_P'],
        'sampling':'Same frozen uniform-without-replacement 100 of 26496 geometry-eligible FM_P configurations',
        'detector_sampling':'Purposive nine checkpoint/resolution settings; not a uniform sample of all FM_D.',
        'pair_population':original['eligible_FM_P']*len(detectors),
        'per_pair_inclusion_probability':100/original['eligible_FM_P'],
        'RollingShutter':'EXCLUDED_PENDING_GT_AND_IGNORE_TRANSFORMS',
        'full_training_feature_binding_required_before_execution':True,
        'resource_gate':'NOT_FULL_PAIR_CALIBRATED','model_count_not_equal_to_training_runs':True})
