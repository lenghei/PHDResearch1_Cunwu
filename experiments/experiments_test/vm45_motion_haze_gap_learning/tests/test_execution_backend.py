"""CPU checks for the new classwise bootstrap orchestration."""
import gzip,json,tempfile,unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import execute_vm45 as e

class Checks(unittest.TestCase):
 def test_classwise_bootstrap_matches_accepted_area_oracle(self):
  from main.labels import ground_truth,mapped_predictions
  from main.statistics import evaluate_coco,bootstrap_coco
  records=[{'id':i,'stem':str(i),'file_name':str(i)+'.jpg','width':100,'height':100,'objects':[{'class_id':0,'bbox':[10.,10.,10.,10.]}],'ignore_regions':[],'other_regions':[]} for i in (1,2,3)]
  raw=[{'image_id':i,'predictions':[{'image_id':i,'class_id':0,'bbox':[10.,10.,10.,10.],'score':.9}]} for i in (1,2,3)]
  with tempfile.TemporaryDirectory() as folder:
   path=Path(folder)/'synthetic_predictions.jsonl.gz'
   with gzip.open(path,'wt',encoding='utf-8') as f:
    for r in raw:f.write(json.dumps(r)+'\n')
   metrics,boot,cost=e.evaluation(path,records,20261117)
  gt,regions=ground_truth(records,'native10');mapped=mapped_predictions([p for r in raw for p in r['predictions']],records,regions,'native10','finetuned',3000)
  _,_,oracle=evaluate_coco(gt,mapped,3000);e.backend.install_accumulator();_,draws=bootstrap_coco(oracle,200,20261117)
  self.assertEqual(boot['draws']['values'],draws['values'])
  self.assertEqual(boot['draws']['class_values']['tiny:1']['map50_95'],draws['values']['tiny']['map50_95'])
  self.assertTrue(all(v is None for v in boot['draws']['class_values']['tiny:2']['map50_95']))
  self.assertEqual(metrics['images'],3)

if __name__=='__main__':unittest.main()
