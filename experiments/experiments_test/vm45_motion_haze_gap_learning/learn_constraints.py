"""Fixed shallow classifiers with whole-condition holdout; no external data."""
import argparse,json,time
from pathlib import Path
def learn(path,out):
 import numpy as np, sklearn
 from sklearn.tree import DecisionTreeClassifier,export_text
 from sklearn.ensemble import RandomForestClassifier
 from sklearn.metrics import confusion_matrix,balanced_accuracy_score,accuracy_score,precision_score,recall_score
 from vm45_common import save,csv_write
 t=time.perf_counter();rules=json.loads(path.read_text());rows=rules['development_rows'];leaves=rules['models']['fm_p']['leaves'];ds=sorted(rules['candidates'])
 assert len(rows)==84 and all(r['evidence_scope']=='DEVELOPMENT_ONLY' and 'vm36_supervisor_reproduction' in r['source_folder'] for r in rows)
 names=leaves+['D_'+d for d in ds]
 X=np.array([[int(f in r['leaves']) for f in leaves]+[int(r['D']==d) for d in ds] for r in rows])
 labels=np.array([1 if r['original_rule_state']=='FAIL' else 0 if r['original_rule_state']=='PASS' else -1 for r in rows]);groups=np.array([r['condition_id'] for r in rows])
 definitions={'SHALLOW_TREE':lambda:DecisionTreeClassifier(max_depth=2,min_samples_leaf=4,class_weight='balanced',random_state=20261117),
 'RANDOM_FOREST_REFERENCE':lambda:RandomForestClassifier(n_estimators=100,max_depth=3,min_samples_leaf=4,class_weight='balanced',random_state=20261117,n_jobs=1)}
 output=[];folds=[];scores=[];candidate=[]
 def score(y,p):
  return {'accuracy':float(accuracy_score(y,p)),'balanced_accuracy':float(balanced_accuracy_score(y,p)),'failure_precision':float(precision_score(y,p,zero_division=0)),
  'failure_recall':float(recall_score(y,p,zero_division=0)),'PASS_recall':float(recall_score(y,p,pos_label=0,zero_division=0)),'confusion_PASS_FAIL':confusion_matrix(y,p,labels=[0,1]).tolist()}
 for name,factory in definitions.items():
  prediction=np.full(84,-1);prob=np.zeros(84)
  for group in sorted(set(groups)):
   train=(groups!=group)&(labels>=0);test=groups==group
   assert not set(groups[train])&set(groups[test])
   m=factory().fit(X[train],labels[train]);prediction[test]=m.predict(X[test]);prob[test]=m.predict_proba(X[test])[:,list(m.classes_).index(1)]
   folds.append({'method':name,'heldout_P':group,'train_P_ids':json.dumps(sorted(set(groups[train]))),'validation_P_ids':json.dumps([group]),'train_PASS':int((labels[train]==0).sum()),'train_FAIL':int((labels[train]==1).sum()),'test_rows':int(test.sum()),'leakage':False})
  valid=labels>=0;result=score(labels[valid],prediction[valid]);rng=np.random.default_rng(20261117);boots=[];unique=sorted(set(groups))
  for _ in range(200):
   draw=rng.choice(unique,len(unique),replace=True);inds=np.concatenate([np.where((groups==g)&valid)[0] for g in draw])
   if len(set(labels[inds]))==2:boots.append(balanced_accuracy_score(labels[inds],prediction[inds]))
  result.update(method=name,scored_rows=int(valid.sum()),prediction_coverage=1.,abstention_rate=0.,uncertain_reference_excluded_from_scoring=int((~valid).sum()),
    balanced_accuracy_group_bootstrap_CI95=np.percentile(boots,[2.5,97.5]).tolist(),valid_group_draws=len(boots),interval_scope='Conditional on fixed OOF predictions, not refitting models or independent images',scientific_scope='DEVELOPMENT_ONLY_POST_HOC_DIAGNOSTIC',admitted_for_constraint_generalization=False)
  scores.append(result)
  for i,r in enumerate(rows):output.append({'method':name,'P':r['condition_id'],'D':r['D'],'true_state':r['original_rule_state'],'prediction':'FAIL' if prediction[i] else 'PASS','failure_probability':float(prob[i]),'scored':bool(valid[i]),'same_P_in_train':False})
  if name=='SHALLOW_TREE':
   m=factory().fit(X[valid],labels[valid]);(out/'in_sample_tree.txt').write_text(export_text(m,feature_names=names),encoding='utf-8')
   def walk(node,terms):
    if m.tree_.children_left[node]<0:
     if int(m.classes_[np.argmax(m.tree_.value[node])])==1:
      candidate.append({'expression':' & '.join(terms)+' => FAILURE','rule_kind':'EMPIRICAL_FAILURE_RULE','detector_scope':'Joint P-D indicators; not a physical P validity constraint','training_rows':int(m.tree_.n_node_samples[node]),'heldout_balanced_accuracy_MODEL_LEVEL':result['balanced_accuracy'],'heldout_rule_specific':'NOT_VALIDATED','evidence_grade':'IN_SAMPLE_CANDIDATE','admitted':False,'safe_to_generalize':False,'source':'VM36 fixed84 development pairs'})
    else:
     f=names[m.tree_.feature[node]];walk(m.tree_.children_left[node],terms+['!'+f]);walk(m.tree_.children_right[node],terms+[f])
   walk(0,[])
 baseline=score(labels[labels>=0],np.ones((labels>=0).sum(),dtype=int));baseline.update(method='ALWAYS_FAIL_REFERENCE',scored_rows=82,admitted_for_constraint_generalization=False);scores.append(baseline)
 csv_write(out/'held_out_rule_results.csv',output);csv_write(out/'rule_group_folds.csv',folds);csv_write(out/'learned_failure_rules.csv',candidate, list(candidate[0]) if candidate else ['expression','admitted'])
 save(out/'rule_learning_validation.json',{'status':'INSUFFICIENT_DATA_FOR_RELIABLE_RULE','method':'Fixed-depth tree and RF diagnostic, no tuning;21 whole-P folds','sklearn_version':sklearn.__version__,'interpreter':__import__('sys').executable,
 'classes':{'PASS':int((labels==0).sum()),'FAIL':int((labels==1).sum()),'UNCERTAIN':int((labels<0).sum())},'feasible_P_groups':len(set(groups[labels==0])),'features':names,'models':scores,'CPU_wall_seconds':time.perf_counter()-t,'CPU_process_seconds':time.process_time(),'external_testdev_rows':0,'admitted_rules':0})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();learn(a.input,a.out)
