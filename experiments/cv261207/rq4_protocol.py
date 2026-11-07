"""Freeze and validate a matched augmentation protocol; this entry never trains."""
import argparse
import json
from pathlib import Path

from common import load, save, source


def prepare(out, admission, audit):
    candidates={c['id']:c for c in load(admission/'admission.json')['candidates']}
    protocol={
        'version':'CV261207_RQ4_MATCHED_ADAPTATION_V1','status':'NOT_EVALUATED',
        'execution_gate':'BLOCKED_SCIENTIFIC_AND_RESOURCE_ADMISSION',
        'checkpoint_initialization':{c:candidates[c]['checkpoint_sha256'] for c in ('ck01','ck05')},
        'initialization_policy':'Within each architecture, clean and augmentation arms start from the same healthy FT1280 checkpoint. This is additional adaptation, not a repeat of the historical 100-epoch training.',
        'groups':['clean_control','NoiseAug','MotionBlurAug','ExposureAug','JPEGAug','HazeAug'],
        'training_seed_candidates':[42,73,101],'primary_feasibility_seed':42,
        'pilot':{'train_images':64,'development_images':16,'epochs':1,'status':'NOT_RUN'},
        'matched_fields':['checkpoint_SHA256','image_IDs','epochs','image_exposures','optimizer','LR_schedule',
                          'imgsz','batch','seed','routine_augmentation','development_evaluation','checkpoint_selection'],
        'candidate_settings':{'imgsz':1280,'batch':2,'workers':0,'optimizer':'AdamW',
                              'amp':False,'additional_epochs':12,'lr0':0.0001,'warmup_epochs':0},
        'settings_status':'PROPOSED_NOT_OPTIMIZED_NOT_EXECUTED',
        'additional_augmentation':{
            'NoiseAug':{'sigma_range':[3,35],'sampling':'continuous','excluded_test_sigma':[5.5,15,30]},
            'MotionBlurAug':{'kernel_candidates':[9,11,13,17,19,23],'angle_range':[0,180],
                            'note':'Kernel width is discrete; angle is continuous. Do not claim continuously valued kernel pixels.'},
            'ExposureAug':{'gain_range':[.35,1.4],'bias_range':[-70,90],'sampling':'continuous paired parameters'},
            'JPEGAug':{'quality_range':[35,95],'excluded_quality':[25,50,75,85],
                       'note':'Codec quality is integer-valued; sample a range excluding frozen test levels.'},
            'HazeAug':{'beta_range':[.15,1.8],'airlight_range':[190,240],'excluded_beta':[.5,1.5]}},
        'evaluation':{'frozen_sample':source(audit/'selection_manifest.json'),
                      'protocol':'Native10 same-checkpoint settings; admitted scene/image-disjoint data required for independent claims.',
                      'metrics':['clean_AP50_95','corrupted_AP50_95','tiny_AP50_95','empirical_gap',
                                 'paired_gap_reduction_CI','latency','memory','training_seed_variability']},
        'budget':{'measured_training_pilot_seconds':None,'approved_worker_seconds':None,
                  'status':'NOT_CALIBRATED','automatic_long_training':False},
        'remaining_gates':['Representative measurable baseline gap and compatible frozen task latency',
                           'Full training-label admission and immutable derived label view if required',
                           'Validated pixel-only augmentation hook and loss/optimizer/checkpoint health canary',
                           'Measured one-seed pilot cost, independent evaluation admission and bounded worker authorization'],
        'training_executor':'NOT_IMPLEMENTED_OR_VALIDATED_FOR_THIS_NEW_PROTOCOL',
        'protocol_validator':'rq4_protocol.py --check <protocol.json>',
        'no_historical_results_relabelled_as_RQ4':True}
    validate(protocol)
    save(out/'rq4_matched_protocol.json',protocol)
    return protocol


def validate(protocol):
    if (protocol['status']!='NOT_EVALUATED' or len(protocol['checkpoint_initialization'])!=2
            or 'clean_control' not in protocol['groups'] or len(set(protocol['groups']))!=6
            or protocol['budget']['approved_worker_seconds'] is not None
            or protocol['budget']['automatic_long_training']):
        raise ValueError('Protocol is not an unexecuted matched, gated design.')
    if any(len(v)!=64 for v in protocol['checkpoint_initialization'].values()):
        raise ValueError('Checkpoint identity missing.')
    return {'status':'VALID_PROTOCOL_PREPARATION','training_status':'NOT_EVALUATED',
            'training_execution_ready':False,'worker_started':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',type=Path,required=True)
    args=parser.parse_args();print(json.dumps(validate(load(args.check))))
