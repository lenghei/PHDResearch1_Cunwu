"""CV261102 fresh train + full 548-image validation. No historical CSV input."""
import argparse,csv,datetime as dt,json,math,os,subprocess,sys,time,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'code'))
from prepare_data import NAMES,digest,prepare

def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def to_csv(path,rows):
    if not rows:raise ValueError('Empty CSV')
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def git_state():
    a=subprocess.run(['git','-C',str(ROOT),'rev-parse','HEAD'],capture_output=True,text=True)
    b=subprocess.run(['git','-C',str(ROOT),'status','--porcelain'],capture_output=True,text=True)
    return dict(commit=a.stdout.strip() if a.returncode==0 else None,clean=b.returncode==0 and not b.stdout.strip())

def cli():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,help='Existing native10 YOLO data YAML')
    p.add_argument('--train-zip',type=Path,help='Original annotated VisDrone train ZIP')
    p.add_argument('--val-zip',type=Path,required=True,help='Original annotated VisDrone 548-image val ZIP')
    p.add_argument('--weights',type=Path,required=True,help='COCO-pretrained yolov8s.pt')
    p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--train-imgsz',type=int,default=640)
    p.add_argument('--eval-imgsz',type=int,default=1280)
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--eval-batch',type=int,default=2)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--preflight',action='store_true')
    a=p.parse_args()
    for key in ('data','train_zip','val_zip','weights'):
        v=getattr(a,key)
        if v is not None:setattr(a,key,v.expanduser().resolve())
    if (a.data is None)==(a.train_zip is None):p.error('Exactly one of --data or --train-zip')
    if a.epochs<1 or a.batch<1 or a.eval_batch<1 or a.train_imgsz<320 or a.eval_imgsz<320:p.error('Invalid sizes or epochs')
    if not a.val_zip.is_file() or not a.weights.is_file():p.error('Missing val ZIP or pretrained weights')
    if 'yolov8s' not in a.weights.stem.lower():p.error('Use local pretrained yolov8s.pt')
    import torch,yaml,zipfile
    from prepare_data import list_members
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU unavailable')
    with zipfile.ZipFile(a.val_zip) as z:
        images,labels=list_members(z)
        if len(images)!=548:raise ValueError('VisDrone validation must contain 548 image/label pairs')
    if a.train_zip:
        with zipfile.ZipFile(a.train_zip) as z:
            images,labels=list_members(z)
            if len(images)<100:raise ValueError('Expected complete training dataset')
    else:
        if not a.data.is_file():raise FileNotFoundError(a.data)
        data=yaml.safe_load(a.data.read_text(encoding='utf-8'))
        names=data.get('names')
        if isinstance(names,dict):names=[v for k,v in sorted(names.items(),key=lambda kv:int(kv[0]))]
        if names!=NAMES or not data.get('train') or not data.get('val'):raise ValueError('Invalid native10 dataset YAML')
    provenance=dict(weights_SHA256=digest(a.weights),val_zip_SHA256=digest(a.val_zip),
                    train_zip_SHA256=digest(a.train_zip) if a.train_zip else None,
                    data_yaml_SHA256=digest(a.data) if a.data else None,
                    gpu=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda)
    print('[CV261102] preflight PASS',provenance['gpu'],flush=True)
    if a.preflight:return
    from ultralytics import YOLO
    from main.evaluation import evaluate,regenerate
    outroot=(ROOT.parent/'datasets/local_runs/cv261102').resolve()
    if outroot.is_relative_to(ROOT):raise ValueError('Private data must remain outside GitHub root')
    outroot.mkdir(parents=True,exist_ok=True)
    out=outroot/('cv261102_full_'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:8])
    out.mkdir(exist_ok=False)
    manifest=dict(RunID=out.name,ExperimentID='cv261102',status='RUNNING',
                  scope='NEW_FINE_TUNING_AND_DEVELOPMENT_VALIDATION_NOT_INDEPENDENT_TEST',
                  provenance=provenance,git=git_state(),
                  args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
                  source_SHA256={f:digest(Path(__file__).with_name(f)) for f in ('run.py','prepare_data.py')})
    save(out/'manifest.json',manifest)
    started=time.perf_counter()
    try:
        data_yaml=a.data
        if a.train_zip:
            data_yaml,record=prepare(a.train_zip,a.val_zip,out/'prepared_dataset')
            save(out/'data_preparation.json',record)
        trainroot=out/'training';trainroot.mkdir()
        timing=trainroot/'epoch_times.csv'
        with timing.open('w',encoding='utf-8-sig',newline='') as f:
            csv.DictWriter(f,fieldnames=['epoch','train_seconds','train_plus_val_seconds','clock']).writeheader()
        clocks={};observed=set()
        model=YOLO(str(a.weights))
        def start(t):clocks[t.epoch+1]={'start':time.perf_counter()}
        def train_end(t):
            if t.epoch+1 in clocks:clocks[t.epoch+1]['train_end']=time.perf_counter()
        def fit_end(t):
            n=t.epoch+1;c=clocks.get(n)
            if n in observed or not c or 'train_end' not in c:return
            train=c['train_end']-c['start'];total=time.perf_counter()-c['start']
            if not 0<train<=total:raise RuntimeError('Invalid epoch clock')
            with timing.open('a',encoding='utf-8-sig',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['epoch','train_seconds','train_plus_val_seconds','clock'])
                w.writerow(dict(epoch=n,train_seconds=train,train_plus_val_seconds=total,clock='time.perf_counter'))
                f.flush();os.fsync(f.fileno())
            observed.add(n);print(f'[CV261102] epoch {n}: train {train:.2f}s, train+val {total:.2f}s',flush=True)
        model.add_callback('on_train_epoch_start',start)
        model.add_callback('on_train_epoch_end',train_end)
        model.add_callback('on_fit_epoch_end',fit_end)
        trainstart=time.perf_counter()
        requested=dict(data=str(data_yaml),epochs=a.epochs,imgsz=a.train_imgsz,batch=a.batch,
                       workers=a.workers,seed=a.seed,device='0',amp=True,deterministic=True,val=True,
                       save=True,plots=False,cache=False,patience=a.epochs,
                       project=str(trainroot),name='yolov8s',exist_ok=False)
        save(trainroot/'requested_args.json',requested)
        model.train(**requested)
        train_seconds=time.perf_counter()-trainstart
        best=trainroot/'yolov8s/weights/best.pt'
        if not best.is_file():raise RuntimeError('Training checkpoint best.pt missing')
        if len(observed)!=a.epochs:raise RuntimeError(f'Measured {len(observed)}/{a.epochs} epochs')
        cfg=dict(dataset_zip=str(a.val_zip),checkpoint=str(best),checkpoint_sha256=digest(best),
                 output_root=str(out/'_eval_tmp'),expected_images=548,smoke_images=5,
                 imgsz=a.eval_imgsz,batch=a.eval_batch,device='0',allow_cpu_fallback=False)
        cfgpath=out/'evaluation_config.yaml';cfgpath.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
        evstart=time.perf_counter();old=evaluate(str(cfgpath),full=True)
        validation=out/'validation';old.rename(validation);(out/'_eval_tmp').rmdir()
        evseconds=time.perf_counter()-evstart
        summary=json.loads((validation/'summary.json').read_text(encoding='utf-8'))
        summary.update(RunID=out.name,ExperimentID='cv261102',checkpoint_id='new_best')
        save(validation/'summary.json',summary)
        current=json.loads((validation/'metrics.json').read_text(encoding='utf-8'))
        repeated=regenerate(validation)
        for field in ('per_area','per_class'):
            for x,y in zip(current[field],repeated[field]):
                for k,v in x.items():
                    w=y[k]
                    if isinstance(v,(int,float)) and not isinstance(v,bool) and isinstance(w,(int,float)):
                        if not math.isclose(v,w,rel_tol=1e-10,abs_tol=1e-10):raise RuntimeError('COCO replay mismatch '+k)
                    elif v!=w:raise RuntimeError('COCO replay mismatch '+k)
            if len(current[field])!=len(repeated[field]):raise RuntimeError('COCO replay length mismatch')
        csvroot=out/'csv'
        overall=next(x for x in current['per_area'] if x['size']=='all')
        to_csv(csvroot/'metrics_overall.csv',[overall])
        to_csv(csvroot/'metrics_by_area.csv',current['per_area'])
        to_csv(csvroot/'metrics_by_class.csv',current['per_class'])
        to_csv(csvroot/'evaluation_timing.csv',[dict(images=548,network_seconds=summary['network_seconds'],
                coco_seconds=summary['evaluation_seconds'],pipeline_seconds=summary['total_seconds'],
                evaluation_wall_seconds=evseconds,training_wall_seconds=train_seconds)])
        with timing.open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
        to_csv(csvroot/'training_epoch_times.csv',rows)
        manifest.update(status='PASS',best_checkpoint_SHA256=digest(best),COCO_regeneration='PASS',
                        csv_SHA256={p.name:digest(p) for p in sorted(csvroot.glob('*.csv'))},
                        metrics_SHA256=digest(validation/'metrics.json'),raw_predictions_SHA256=digest(validation/'predictions_raw.json'),
                        epoch_count=len(observed),AP50=overall['map50'],AP50_95=overall['map50_95'],
                        train_wall_seconds=train_seconds,eval_wall_seconds=evseconds,
                        total_wall_seconds=time.perf_counter()-started)
        save(out/'manifest.json',manifest)
        print('[CV261102] PASS',out,'AP50',overall['map50'],'AP50:95',overall['map50_95'],flush=True)
    except BaseException as exc:
        manifest.update(status='PARTIAL',error=repr(exc));save(out/'manifest.json',manifest)
        raise

if __name__=='__main__':cli()
