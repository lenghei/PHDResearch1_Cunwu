"""Source-bound continuous policy; no evaluation responses select parameters."""
from study import *


def uniform(low,high,integer=False):
    return {'distribution':'integer_uniform' if integer else 'uniform','low':low,'high':high}


def candidate():
    return {'version':'VM37_ALL_FIVE_V1','status':'APPROVED_TRAINING_RECIPE',
        'scientific_approval':{'basis':'Author-delegated source-backed protocol design; no manual parameter sign-off claimed.',
            'performance_validation':'NOT_EVALUATED','data_calibration':'NOT_CALIBRATED'},
        'order':['HazeAug','ExposureAug','MotionBlurAug','NoiseAug','JPEGAug'],
        'families':{
            'HazeAug':{'probability':.5,'beta':uniform(.5,1.5),'airlight':255.},
            'ExposureAug':{'probability':.5,'gain':uniform(.4,1.3),'bias':uniform(-50.,80.)},
            'MotionBlurAug':{'probability':.5,'kernel':uniform(7,21,True),'angle_min':0.,'angle_max':180.},
            'NoiseAug':{'probability':.5,'sigma':uniform(5.5,30.),'photons':uniform(20.,100.),'gaussian_probability':.5},
            'JPEGAug':{'probability':.5,'quality':uniform(25,75,True)}},
        'sampling':'Independent Bernoulli family gates; independent uniform strengths; per-image/epoch/occurrence/family SHA-derived RNG.',
        'conditional_dependencies':'Noise type is drawn only when NoiseAug is active. No other conditional gates are imposed.',
        'units':{'beta':'dimensionless constant-depth optical-thickness proxy','airlight':'uint8 intensity',
            'gain':'unitless intensity multiplier','bias':'uint8 intensity offset','kernel':'odd native-pixel side length',
            'angle':'degrees; symmetric centered kernel has180-degree periodicity','sigma':'uint8 standard deviation',
            'photons':'existing generator Poisson intensity scale, not measured sensor photons','quality':'OpenCV JPEG quality'},
        'UVL_scope':'The selected capabilities are a legal FM_D TrainAugmentation branch. Continuous training draws are not enumerated FM_P leaf states; FM_P cross-tree rules continue to govern frozen evaluation configurations.',
        'train_evaluation_separation':'Reject the exact discrete UVL test parameter values. This is numeric separation, not proof of disjoint output pixels or independent image populations.',
        'geometry':'No target geometry change before shared ordinary geometry/mosaic. Original images and annotations are read-only.',
        'haze_limit':'A full-image constant-depth stress proxy. DehazeNet supports the equation and white A=1 convention on patches; full-frame use is an explicit simplification, not calibrated UAV weather.',
        'activation_rationale':'Use the common documented0.5 transform default as a fixed prior, not an optimal probability.',
        'alternatives':{'activation':'A single OneOf block retains more undegraded inputs but does not exercise compound combinations.',
            'motion':'A0..360-degree range repeats orientations of this symmetric centered line kernel.',
            'haze':'Legacy airlight220 lacks an independent calibration record; the cited white-airlight convention is explicit and reproducible.',
            'strengths':'Broader arbitrary bounds increase unseen extremes without calibration; use the supervisor UVL numeric hulls.'},
        'sources':[
            {'URL':'https://albumentations.ai/docs/2-core-concepts/probabilities/','supports':'Bernoulli application gates and normalized OneOf weights;0.5 is a common documented starting point.'},
            {'URL':'https://albumentations.ai/docs/api-reference/albumentations/augmentations/blur/transforms/','supports':'Random motion orientation; the180-degree reduction is specific to the existing symmetric centered kernel.'},
            {'URL':'https://arxiv.org/abs/1601.07661','DOI':'10.1109/TIP.2016.2598681','supports':'Haze blending equation and white atmospheric-light convention in SectionIII-C; not this full-frame UAV calibration.'},
            {'local':'VM36 feature_models/fm_p.uvl','supports':'Test-level numeric hulls and exact-value exclusion requested by the supervisor.'}]}


def verify_cpu():
    import numpy as np
    from main.training_effects import NativeEffects,validate_recipe,TEST_VALUES
    out=create('targeted_policy_cpu');runtime(out,cuda=False)
    recipe=candidate();validate_recipe(recipe)
    image=np.random.default_rng(42).integers(0,256,(96,128,3),dtype=np.uint8)
    effects=NativeEffects(recipe,42);repeat=NativeEffects(recipe,42);events=[]
    for i in range(200):
        before=image.copy();pixels,trace=effects.apply(image,'CPU_'+str(i),0,0)
        other,same=repeat.apply(image,'CPU_'+str(i),0,0)
        if not np.array_equal(before,image) or not np.array_equal(pixels,other) or trace!=same:raise ValueError('Policy determinism or source immutability failed.')
        if pixels.shape!=image.shape or pixels.dtype!=np.uint8:raise ValueError('Policy geometry changed.')
        for event in trace:
            for key,value in event['parameters'].items():
                if key in TEST_VALUES and any(abs(value-v)<1e-9 for v in TEST_VALUES[key]):raise ValueError('Exact evaluation strength leaked into training.')
        events.append({'id':'CPU_'+str(i),'events':trace,'pixel_SHA256':__import__('hashlib').sha256(pixels.tobytes()).hexdigest()})
    save(out/'policy.json',recipe,out)
    save(out/'sampling_events.json',events,out)
    counts={family:sum(e['active'] for row in events for e in row['events'] if e['family']==family) for family in recipe['families']}
    save(out/'CPU_policy_acceptance.json',{'status':'PASS','samples':200,'deterministic_repeat':True,'shape_dtype_preserved':True,
        'source_pixels_unchanged':True,'exact_UVL_test_values_excluded':True,'actual_family_activation_counts':counts,
        'GPU_training':'NOT_RUN','augmentation_effectiveness':'NOT_EVALUATED','UVL_scope':recipe['UVL_scope']},out)
    write_atomic(out/'diary.md','# Source-backed policy CPU checks\n\nFixed all five families before new robustness responses. Tested deterministic draws, geometry and original-pixel preservation; no detection-effect claim.\n',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'TARGETED_POLICY_CPU','status':'PASS'},out,replace=True)
    write_atomic(HERE/'TARGETED_POLICY.json',__import__('json').dumps(recipe,indent=2),ROOT)
    save(RUNS/'policy_admission.json',{'RunID':out.name,'status':'PASS','policy_SHA256':sha256(out/'policy.json'),
        'effects_implementation_SHA256':sha256(PUBLIC/'code/main/training_effects.py'),
        'generator_SHA256':sha256(HERE.parent/'cv261109/generator.py')},RUNS,
        replace=(RUNS/'policy_admission.json').exists())
    seal(out);print(out)


if __name__=='__main__':verify_cpu()
