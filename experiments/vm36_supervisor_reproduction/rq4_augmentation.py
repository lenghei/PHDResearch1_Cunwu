"""Explicit training recipes using the existing image-effect implementations."""
import hashlib
import math
import numpy as np

FAMILIES=('HazeAug','ExposureAug','MotionBlurAug','NoiseAug','JPEGAug')
TEST_VALUES={'gain':(.4,.8,.9,1.,1.15,1.3),'bias':(-50.,-20.,0.,30.,80.),
    'sigma':(5.5,15.,30.),'photons':(20.,100.),'kernel':(7,15,21),'quality':(25,50,75),'beta':(.5,1.5)}
_renderer=None


def renderer():
    global _renderer
    if _renderer is None:
        from development_v1 import generator
        _renderer=generator()
    return _renderer


def validate_recipe(recipe,engineering=False):
    expected='ENGINEERING_FIXTURE_ONLY' if engineering else 'APPROVED_TRAINING_RECIPE'
    if recipe.get('status')!=expected:raise ValueError('The recipe has not been admitted for this use.')
    if tuple(recipe.get('order',()))!=FAMILIES:raise ValueError('Effect order differs from the existing generator.')
    if not engineering and not recipe.get('scientific_approval'):raise ValueError('Scientific approval is required.')
    for family,spec in recipe['families'].items():
        if family not in FAMILIES or not 0<=spec['probability']<=1:raise ValueError('Invalid family or probability.')
        required={'HazeAug':('beta','airlight'),'ExposureAug':('gain','bias'),
            'MotionBlurAug':('kernel','angle_min','angle_max'),
            'NoiseAug':('sigma','photons','gaussian_probability'),'JPEGAug':('quality',)}[family]
        if not set(required).issubset(spec):raise ValueError('Missing explicit augmentation parameters.')
        for name in required:
            value=spec[name]
            if isinstance(value,dict):
                if value.get('distribution') not in ('uniform','integer_uniform') or not math.isfinite(value['low']) or not math.isfinite(value['high']) or value['low']>=value['high']:
                    raise ValueError('Invalid parameter distribution.')
                if not engineering and name in TEST_VALUES and not (value['low']<=min(TEST_VALUES[name]) and value['high']>=max(TEST_VALUES[name])):
                    raise ValueError('The training range does not contain the test levels.')
            elif not math.isfinite(value):raise ValueError('Nonfinite parameter.')
            elif not engineering and name in TEST_VALUES:raise ValueError('Scientific training requires declared ranges.')
            lo,hi=(value['low'],value['high']) if isinstance(value,dict) else (value,value)
            if name in ('sigma','beta','gain') and lo<0:raise ValueError('Negative effect parameter.')
            if name=='photons' and lo<=0:raise ValueError('Photon scale must be positive.')
            if name=='quality' and not 1<=lo<=hi<=100:raise ValueError('JPEG quality is outside [1,100].')
            if name=='airlight' and not 0<=lo<=hi<=255:raise ValueError('Airlight is outside [0,255].')
            if name=='kernel' and (lo<3 or not isinstance(value,dict) and (value%2!=1 or int(value)!=value)):
                raise ValueError('Motion kernel must be an odd integer >=3.')
        if family=='NoiseAug' and not 0<=spec['gaussian_probability']<=1:raise ValueError('Invalid noise mixture probability.')
        if family=='MotionBlurAug' and spec['angle_min']>spec['angle_max']:raise ValueError('Invalid motion orientation interval.')
    return recipe


def draw(spec,name,rng):
    value=spec[name]
    if not isinstance(value,dict):return value
    for _ in range(1000):
        if value['distribution']=='integer_uniform':
            result=int(rng.integers(math.ceil(value['low']),math.floor(value['high'])+1))
        else:result=float(rng.uniform(value['low'],value['high']))
        if name=='kernel' and (result<3 or result%2!=1):continue
        if any(abs(result-x)<1e-9 for x in TEST_VALUES.get(name,())):continue
        return result
    raise ValueError('The range has no admissible non-test draw.')


class NativeEffects:
    def __init__(self,recipe,seed,engineering=False):
        self.recipe=validate_recipe(recipe,engineering);self.seed=seed

    def apply(self,image,image_id,epoch,occurrence):
        if image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3:raise ValueError('Expected native BGR uint8 pixels.')
        output=image.copy();trace=[];ops=renderer()
        for family in FAMILIES:
            if family not in self.recipe['families']:continue
            spec=self.recipe['families'][family]
            seed=int.from_bytes(hashlib.sha256(f'{self.seed}|{epoch}|{image_id}|{occurrence}|{family}'.encode()).digest()[:8],'big')
            rng=np.random.default_rng(seed);active=bool(rng.random()<spec['probability'])
            event={'family':family,'seed':seed,'probability':spec['probability'],'active':active,'parameters':{}}
            if active:
                names={'HazeAug':('beta','airlight'),'ExposureAug':('gain','bias'),
                    'MotionBlurAug':('kernel','angle_min','angle_max'),'JPEGAug':('quality',)}
                if family=='NoiseAug':
                    gaussian=bool(rng.random()<spec['gaussian_probability']);name='sigma' if gaussian else 'photons'
                    p={name:draw(spec,name,rng)};event['variant']='gaussian' if gaussian else 'shot'
                    output=ops.apply_gaussian_noise(output,p[name],rng) if gaussian else ops.apply_shot_noise(output,p[name],rng)
                else:
                    p={name:draw(spec,name,rng) for name in names[family]}
                    if family=='HazeAug':output=ops.apply_haze(output,**p)
                    elif family=='ExposureAug':output=ops.apply_exposure(output,**p)
                    elif family=='MotionBlurAug':output,extra=ops.apply_motion_blur(output,p['kernel'],p['angle_min'],p['angle_max'],rng);event.update(extra)
                    else:output=ops.apply_jpeg(output,**p)
                event['parameters']=p
            trace.append(event)
        if output.shape!=image.shape or output.dtype!=np.uint8:raise ValueError('A photometric effect changed geometry.')
        return output,trace


def engineering_fixture(active):
    """Technical fixtures only; these values cannot authorize scientific training."""
    values={'HazeAug':{'beta':.8,'airlight':210.},'ExposureAug':{'gain':.53,'bias':-13.},
        'MotionBlurAug':{'kernel':9,'angle_min':11.,'angle_max':169.},
        'NoiseAug':{'sigma':11.,'photons':60.,'gaussian_probability':.5},'JPEGAug':{'quality':61}}
    return {'status':'ENGINEERING_FIXTURE_ONLY','order':list(FAMILIES),
        'families':{name:{'probability':1.,**values[name]} for name in active}}
