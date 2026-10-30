"""CPU acceptance tests."""
import unittest,sys,os
from pathlib import Path
import numpy as np
from main.config import load_config
from main.dataset import VisDroneZIP
from main.labels import ground_truth,mapped_predictions,keep_outside_ignore
from main.statistics import evaluate_coco
from main.degradation import degrade,field_for
class Tests(unittest.TestCase):
 def test_real_config_and_zip(self):
  cfg=load_config(os.environ['UAVGAP_TEST_CONFIG']);z=VisDroneZIP(cfg['dataset_zip']);im,rec,proof=z.record(0);self.assertEqual(im.shape[:2],(rec['height'],rec['width']));self.assertEqual(len(z.images),548);z.close()
 def test_ignore_union(self):
  self.assertFalse(keep_outside_ignore([[0,0,10,10]],[[0,0,3,10],[3,0,3,10]],10,10)[0])
 def test_metrics(self):
  rec=dict(id=1,file_name='synthetic',width=100,height=100,objects=[dict(class_id=3,bbox=[10,10,20,20])],ignore_regions=[],other_regions=[]);gt,regions=ground_truth([rec],'native10');pred=mapped_predictions([dict(image_id=1,class_id=3,bbox=[10,10,20,20],score=.9)],[rec],regions,'native10','finetuned');areas,classes,_=evaluate_coco(gt,pred,3000);self.assertAlmostEqual(next(r for r in areas if r['size']=='all')['map50_95'],1)
 def test_degradation(self):
  image=np.arange(30*40*3,dtype=np.uint8).reshape(30,40,3);field=field_for(image,'synthetic');a,_=degrade(image,(2,2,2),field);b,_=degrade(image,(2,2,2),field);self.assertTrue(np.array_equal(a,b));self.assertEqual(a.shape,image.shape)
 def test_imports(self):
  import main.predictor,main.evaluation,main.uvl
  package=Path(main.evaluation.__file__).resolve().parents[1]
  self.assertEqual(package,Path(__file__).resolve().parents[1])
 def test_entry_modes(self):
  import importlib.util
  path=Path(__file__).resolve().parents[2]/'experiments/experiments_test/vm03_gpu_pilot/run.py'
  spec=importlib.util.spec_from_file_location('vm03_entry_test',path);entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
  direct=entry.parse_args([]);self.assertTrue(direct.full);self.assertEqual(Path(direct.config),path.with_name('config.local.yaml'))
  self.assertFalse(entry.parse_args(['--config','local.yaml']).full)
  self.assertFalse(entry.parse_args(['--smoke','--no-open']).full)
  self.assertTrue(entry.parse_args(['--config','local.yaml','--full']).full)
 def test_regeneration_requires_output(self):
  import importlib.util,contextlib,io
  path=Path(__file__).resolve().parents[2]/'experiments/experiments_test/vm03_gpu_pilot/run.py'
  spec=importlib.util.spec_from_file_location('vm03_regeneration_test',path);entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
  with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):entry.parse_args(['--regenerate','saved-run'])
if __name__=='__main__':unittest.main()
