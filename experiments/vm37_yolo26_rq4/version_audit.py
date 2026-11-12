"""Match the installed stable wheel to its official source commit."""
import copy
import importlib.metadata
import urllib.request
from study import *


def audit():
    out=create('version_regression_cpu');runtime(out,cuda=False)
    import torch,ultralytics
    torch.set_num_threads(2)
    from ultralytics.models.yolo.detect.train import DetectionTrainer
    from ultralytics.cfg import DEFAULT_CFG
    from main.labels import VIS_NAMES
    from main.training_health import model_health,same_state,cpu_state
    environment=run_path(load(RUNS/'source_environment.json')['RunID'])
    commit=load(environment/'tag_object.json')['object']['sha'];checks=[]
    for relative in ('engine/trainer.py','engine/predictor.py','nn/tasks.py','nn/modules/head.py','utils/loss.py','utils/nms.py','data/augment.py','data/dataset.py'):
        url='https://raw.githubusercontent.com/ultralytics/ultralytics/'+commit+'/ultralytics/'+relative
        with urllib.request.urlopen(url,timeout=30) as response:data=response.read()
        path=out/('official_'+relative.replace('/','__'));path.write_bytes(data)
        actual=Path(ultralytics.__file__).parent/relative
        checks.append({'file':relative,'official_URL':url,'official_SHA256':sha256(path),'installed_SHA256':sha256(actual),
            'equal':data==actual.read_bytes()})
    if not all(r['equal'] for r in checks):raise ValueError('The stable wheel differs from its claimed source commit.')
    initialization=next(r for r in load(PREP/'CANDIDATE_INITIALIZATION_MANIFEST.json') if r['architecture']=='YOLOv8s' and r['seed']==42)
    source_model=torch.load(initialization['native10_candidate']['path'],map_location='cpu',weights_only=False)['model']
    trainer=DetectionTrainer.__new__(DetectionTrainer);trainer.data={'nc':10,'channels':3,'names':dict(enumerate(VIS_NAMES))}
    trainer.args=copy.copy(DEFAULT_CFG);trainer.args.cls_remap=False
    model=trainer.get_model(cfg=copy.deepcopy(source_model.yaml),weights=source_model,verbose=False);model_health(model)
    equal=same_state(model,cpu_state(source_model))
    if not equal:raise ValueError('YOLOv8s native10 state changed in the stable constructor.')
    result={'status':'PASS','version':ultralytics.__version__,'official_commit':commit,'critical_source_files':checks,
        'YOLOv8s_native10_constructor_equal':equal,'YOLO26s_constructor':source(environment/'CPU_model_compatibility.json'),
        'environment':{name:importlib.metadata.version(name) for name in ('torch','torchvision','ultralytics','numpy','opencv-python','albumentations','pycocotools','pypdf')},
        'CUDA_runtime':torch.version.cuda,'Python':sys.version,'interpreter':sys.executable,
        'source_stack':'Isolated Ultralytics andpypdf; read-only shared Torch/dependency directory from the existing .venv.',
        'initial_import_side_effect':'Before runtime isolation was set, the initial library import reported a global Ultralytics settings-schema update. No global settings content was copied or exposed; subsequent imports use private runtime directories.',
        'GPU_tests_started':0,'research_evaluator':'Original VM36 COCO/ignore-union code retained; stable API adapter changes setup argument names only.'}
    save(out/'version_acceptance.json',result,out)
    write_atomic(out/'diary.md','# Stable version regression\n\nMatched critical installed files to the official commit and checked the comparison-model native10 constructor. No GPU work.\n',out)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':EXPERIMENT,'mode':'VERSION_REGRESSION_CPU','status':'PASS'},out,replace=True)
    seal(out);print(out)


if __name__=='__main__':audit()
