"""Measured feature bridge and descriptive baselines; shared images block holdout claims."""
import argparse
import json
import subprocess
from pathlib import Path
import sys

from common import ROOT, FLAMA, load, save, source
from historical_review import VM, table, csv_file, checked, relocated


def measured_grid(out, review):
    vm09=VM/'experiments/vm09_research_synthesis/runs/vm09_r001/results/full_configuration_metrics.csv'
    integrity=load(ROOT/'datasets/research_archive/migration_20261104/result_integrity.json')
    checked(vm09,integrity['checks']['vm09_r001']['source_csv_sha256'])
    grid={}
    for row in table(vm09):
        result=checked(relocated(row['source_result_file']),row['source_result_SHA256'])
        record=load(result)
        if record['image_count']!=300 or record['GT_count']!=21756 or any(
                abs(float(row[k])-record[k])>1e-14 for k in ('AP50_95','AP50','Tiny_AP','Small_AP')):
            raise ValueError('VM09 response source differs.')
        if record['GT_source_SHA256']!='c0da1ac75b55ea9910933a8f5cc03d358fc5adf2230846955bcae27e41e8fab5':
            raise ValueError('Mixed historical GT populations.')
        key=(row['CheckpointID'],int(row['illumination']),int(row['blur']),int(row['noise']),int(row['resolution']),'None')
        grid[key]={**row,'jpeg':'None','metric_sha256':row['source_result_SHA256'],
                   'raw_sha256':record.get('raw_predictions_SHA256'),
                   'source_RunID':'vm09_r001','checkpoint_sha256':record['checkpoint_SHA256']}
    reused=0
    for row in table(review/'configuration_metrics.csv'):
        key=(row['CheckpointID'],int(row['illumination']),int(row['blur']),int(row['noise']),int(row['resolution']),row['jpeg'])
        if key in grid:
            if abs(float(grid[key]['AP50_95'])-float(row['AP50_95']))>1e-14:
                raise ValueError('Overlapping VM09/VM26 responses disagree.')
            reused+=1
            continue
        grid[key]={**row,'metric_sha256':row['source_metric_SHA256'],'raw_sha256':row['raw_SHA256'],
                   'source_RunID':'vm26_r001','checkpoint_sha256':row['checkpoint_SHA256']}
    if len(grid)!=279 or reused!=18:
        raise ValueError('Unexpected unique historical response count.')
    rows=[]
    for key,row in sorted(grid.items()):
        c,i,b,n,r,j=key
        rows.append({'CheckpointID':c,'config_id':f'{c}_I{i}B{b}N{n}_J{j}_R{r}',
                     'illumination':i,'blur':b,'noise':n,'resolution':r,'jpeg':j,
                     **{k:float(row[k]) for k in ('AP50_95','AP50','Tiny_AP','Small_AP','Medium_AP','Large_AP','AR')},
                     'metric_sha256':row['metric_sha256'],'raw_sha256':row['raw_sha256'],
                     'checkpoint_sha256':row['checkpoint_sha256'],'source_RunID':row['source_RunID']})
    csv_file(out/'measured_configuration_metrics.csv',rows)
    save(out/'historical_response_union.json',{'VM09_rows':243,'VM26_rows':54,'overlap':18,'unique_responses':279,
         'scope':'DEVELOPMENT_ONLY','images':300,'contexts':31,'source':source(vm09),
         'VM09_verification':'CSV and original cell result SHA/metrics; raw predictions not newly rematched.',
         'VM26_verification':'Accepted raw/GT review is reused, not restarted.',
         'VM26_review':source(review/'SHA256SUMS.txt')})
    return rows


def prepare(out, review, admission):
    models={r['id']:r for r in load(admission/'admission.json')['candidates']}
    registry=next(r for r in table(VM/'datasets/registry.csv') if r['DatasetID']=='dv02')
    recipe=load(checked(VM/registry['ManifestPath'],registry['ManifestSHA256']))['parameters']
    identity={'ck01':'YOLO26','ck05':'YOLOv8','ck06':'YOLO11'}
    rows=[]
    for cell in measured_grid(out,review):
        cid=cell['CheckpointID'];model=models[cid]
        illumination=int(cell['illumination']);blur=int(cell['blur']);noise=int(cell['noise'])
        gain,bias=recipe['illumination'][illumination]
        variance=recipe['noise_variances'][noise]
        features={'P:motion_kernel':recipe['blur_kernels'][blur],
                  'P:illumination_gain':gain,'P:illumination_bias':bias,'P:noise_variance':variance,
                  'P:noise_sigma':variance**.5,
                  'P:extra_JPEG':int(cell['jpeg']!='None'),'P:JPEG_quality_missing':int(cell['jpeg']=='None'),
                  'P:JPEG_quality':int(cell['jpeg'][1:]) if cell['jpeg']!='None' else 0,
                  'D:architecture':identity[cid],'D:model_scale':'Small','D:training_resolution':1280,
                  'D:inference_resolution':int(cell['resolution']),'D:precision':'FP32',
                  'D:strategy':'FullFrame','D:postprocessing':'native10_dense_NMS',
                  'D:training_epochs':100,'D:optimizer':model['training_protocol']['optimizer'],
                  'D:training_seed':model['training_protocol']['seed'],'D:weights':'VisDrone_FT',
                  'D:TTA':0,'D:Albu_activation':'UNKNOWN'}
        features['P:'+('Normal','Overexposed_strong','Underexposed_strong')[illumination]]=1
        if blur:features['P:'+('','Motion_k7','Motion_k21')[blur]]=1
        if cell['jpeg']=='Q50':features['P:JPEG_q50']=1
        if cell['jpeg']=='Q85':features['P:JPEG_q85_OUTSIDE_FMP_V1']=1
        features['X:resolution_x_blur']=int(cell['resolution'])*int(blur>0)
        rows.append({'config_id':cell['config_id'],'checkpoint':cid,'features':json.dumps(features,sort_keys=True),
                     'AP50_95':float(cell['AP50_95']),'Tiny_AP':float(cell['Tiny_AP']),
                     'raw_prediction_sha256':cell['raw_sha256'],'metric_sha256':cell['metric_sha256'],
                     'image_group':'ds04_same_300_images','configuration_group':f'I{illumination}B{blur}N{noise}_{cell["jpeg"]}',
                     'scope':'DEVELOPMENT_ONLY','FM_P_binding':'FEATURE_ANALOGUE_NOT_PIXEL_IDENTITY',
                     'known_outside_v1':cell['jpeg']=='Q85'})
    csv_file(out/'measured_learning_features.csv',rows)
    with (out/'learning_cpu.log').open('x',encoding='utf-8') as stream:
        subprocess.run([str(FLAMA),'-B',str(Path(__file__)),'--run',str(out)],check=True,
                       timeout=120,stdout=stream,stderr=subprocess.STDOUT)
    return rows


def fit(out):
    import numpy as np
    import sklearn
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.tree import DecisionTreeRegressor,export_text
    from sklearn.ensemble import RandomForestRegressor
    rows=table(out/'measured_learning_features.csv')
    features=[json.loads(r['features']) for r in rows]
    y=np.array([float(r['AP50_95']) for r in rows])
    vector=DictVectorizer(sparse=False);x=vector.fit_transform(features)
    results=[];predictions=[];rules=[]
    for seed in (42,73,101):
        methods={'rule_stump':DecisionTreeRegressor(max_depth=1,min_samples_leaf=3,random_state=seed),
                 'tree':DecisionTreeRegressor(max_depth=3,min_samples_leaf=3,random_state=seed),
                 'forest':RandomForestRegressor(n_estimators=100,max_depth=5,min_samples_leaf=3,
                                               random_state=seed,n_jobs=1)}
        for name,model in methods.items():
            model.fit(x,y);pred=model.predict(x)
            complexity=model.get_n_leaves() if name!='forest' else sum(t.get_n_leaves() for t in model.estimators_)
            results.append({'model':name,'seed':seed,'in_sample_MAE':float(abs(pred-y).mean()),
                           'leaf_count':complexity,'evaluation':'IN_SAMPLE_DESCRIPTIVE_NOT_HELD_OUT'})
            predictions.extend({'config_id':r['config_id'],'model':name,'seed':seed,
                                'measured_AP':float(y[i]),'fitted_AP':float(pred[i])} for i,r in enumerate(rows))
            if name!='forest':
                rules.append(f'{name}, seed={seed}\n'+export_text(model,feature_names=list(vector.get_feature_names_out()),decimals=6))
    csv_file(out/'descriptive_model_fits.csv',results)
    csv_file(out/'descriptive_fitted_responses.csv',predictions)
    (out/'descriptive_rule_baselines.txt').write_text('\n'.join(rules),encoding='utf-8')
    save(out/'grouped_learning_acceptance.json',{
         'status':'ML_INSUFFICIENT_EVIDENCE','scope':'DEVELOPMENT_ONLY',
         'measured_rows':len(rows),'degradation_groups':len({r['configuration_group'] for r in rows}),
         'image_groups':1,'shared_images':300,'checkpoints':3,'training_seeds_per_checkpoint':1,
         'descriptive_fits':['rule_stump','tree','forest'],'sklearn':sklearn.__version__,
         'training_seed_stability':'NOT_EVALUATED; fit seeds above are model-fitting seeds only.',
         'held_out_balanced_accuracy':None,'held_out_macro_F1':None,
         'held_out_failure_precision':None,'held_out_failure_recall':None,
         'held_out_confusion_matrix':None,'held_out_rule_stability':None,
         'reason':'One shared image population cannot support image-and-configuration-disjoint validation. Joint AP/latency PASS labels are incomplete.',
         'Q85':'Retained as a measured numeric factor; not silently inserted into original FM_P.',
         'noise':'Historical variances 15/30 imply sigma=sqrt(15)/sqrt(30); they are not the UVL sigma=15/30 leaves.',
         'source':source(out/'measured_learning_features.csv')})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    args=parser.parse_args();fit(Path(args.run))
