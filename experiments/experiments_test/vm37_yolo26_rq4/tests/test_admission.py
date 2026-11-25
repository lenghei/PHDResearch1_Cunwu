import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from study import *


class AdmissionTests(unittest.TestCase):
    def test_changed_input_rejects_acceptance_and_preserves_failure_evidence(self):
        from training import verify_input_preservation
        folder=RUNS/'cpu_test_runtime';folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as name:
            out=Path(name);original=out/'source.bin';original.write_bytes(b'frozen input')
            reference=source(original);good=out/'good';bad=out/'changed';good.mkdir();bad.mkdir()
            self.assertTrue(verify_input_preservation(good,[reference])['unchanged'])
            original.write_bytes(b'changed input')
            with self.assertRaises(ValueError):verify_input_preservation(bad,[reference])
            self.assertFalse(load(bad/'protected_inputs.json')['unchanged'])

    def test_current_warmup_and_partial_batch_plan(self):
        self.assertEqual(expected_steps(64,1)['updates'],1)
        plan=expected_steps(6471,100,stop_epochs=1)
        self.assertEqual((plan['batches'],plan['updates'],plan['warmup_batches']),(3236,784,9708))
        self.assertEqual(plan['pending_microbatches'],1)

    def test_current_nms_matches_frozen_implementation(self):
        import torch
        from ultralytics.utils.nms import non_max_suppression
        path=ROOT/'.venv/Lib/site-packages/ultralytics/utils/nms.py'
        spec=importlib.util.spec_from_file_location('vm36_reference_nms',path)
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        generator=torch.Generator().manual_seed(42)
        predictions=torch.rand(2,14,50,generator=generator);predictions[:,:4]*=100
        kwargs=dict(conf_thres=.001,iou_thres=.7,multi_label=True,agnostic=False,nc=10,max_det=3000,max_nms=30000,max_time_img=float('inf'),end2end=False)
        actual=non_max_suppression(predictions.clone(),**kwargs);expected=old.non_max_suppression(predictions.clone(),**kwargs)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(actual,expected)))

    def test_private_loader_reopens_without_source_writes(self):
        from ultralytics.cfg import DEFAULT_CFG
        from main.training_data import RQ4Dataset
        from main.training_effects import engineering_fixture
        from main.labels import VIS_NAMES
        row=load(PREP/'CANARY_IMAGE_CANDIDATES.json')['train'][0]
        labels=load(PREP/'PRIVATE_LABEL_LOOKUP.json')
        folder=RUNS/'cpu_test_runtime';folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as name:
            out=Path(name);(out/'images.txt').write_text(row['image_path']+'\n',encoding='utf-8')
            kwargs=dict(img_path=str(out/'images.txt'),imgsz=1280,batch_size=2,augment=False,hyp=copy.copy(DEFAULT_CFG),rect=False,
                cache=False,stride=32,pad=0.,task='detect',data={'names':dict(enumerate(VIS_NAMES)),'nc':10,'channels':3},
                label_lookup={row['image_id']:labels[row['image_id']]},evidence=out/'cache',recipe=engineering_fixture(()),seed=42,engineering=True)
            one=RQ4Dataset(**kwargs);two=RQ4Dataset(**kwargs)
            self.assertEqual(len(one),1);self.assertEqual(len(two),1)
            self.assertEqual(one[0]['img'].shape[1:],(1280,1280))
            self.assertFalse(one.native_cache[0].flags.writeable)
            self.assertEqual(sha256(Path(row['image_path'])),row['image_SHA256'])

    def test_readonly_cache_matches_original_adapter_pixels_and_targets(self):
        import torch
        from ultralytics.cfg import DEFAULT_CFG
        from ultralytics.utils.torch_utils import init_seeds
        from main.training_data import RQ4Dataset
        from rq4_dataset import RQ4Dataset as Reference
        from main.training_effects import engineering_fixture
        from main.labels import VIS_NAMES
        from main.training_effects import renderer as current_renderer
        from rq4_augmentation import renderer as reference_renderer
        current_renderer();reference_renderer()
        from ultralytics.data.augment import Albumentations
        Albumentations()
        row=load(PREP/'CANARY_IMAGE_CANDIDATES.json')['train'][0];labels=load(PREP/'PRIVATE_LABEL_LOOKUP.json')
        folder=RUNS/'cpu_test_runtime';folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as name:
            out=Path(name);(out/'images.txt').write_text(row['image_path']+'\n',encoding='utf-8');observed=[]
            for i,kind in enumerate((Reference,RQ4Dataset)):
                init_seeds(42,deterministic=False)
                data=kind(img_path=str(out/'images.txt'),imgsz=1280,batch_size=2,augment=True,hyp=copy.copy(DEFAULT_CFG),
                    rect=False,cache=False,stride=32,pad=0.,task='detect',data={'names':dict(enumerate(VIS_NAMES)),'nc':10,'channels':3},
                    label_lookup={row['image_id']:labels[row['image_id']]},evidence=out/str(i),recipe=engineering_fixture(()),seed=42,engineering=True)
                observed.append(data[0])
            for key in ('img','cls','bboxes','batch_idx'):self.assertTrue(torch.equal(observed[0][key],observed[1][key]),key)

    def test_stable_adapter_rejects_nms_free_and_non_fp32(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from ultralytics.models.yolo.detect.predict import DetectionPredictor
        from main.predictor_stable import StableNMSPredictor
        obj=StableNMSPredictor.__new__(StableNMSPredictor)
        obj.args=SimpleNamespace(nms=False,quantize=None)
        with self.assertRaises(ValueError):obj.setup_model(None)
        def base(self,**kwargs):self.model=SimpleNamespace(end2end=False)
        with patch.object(DetectionPredictor,'setup_model',base):
            obj.args.nms=None;obj.setup_model(None)
            obj.args.quantize=16
            with self.assertRaises(ValueError):obj.setup_model(None)

    def test_grouped_health_checks_match_reference_and_reject_faults(self):
        import torch
        from main.training_health import finite_tree,model_health
        from rq4_training_backend import finite_tree as reference
        for dtype in (torch.float16,torch.float32,torch.float64):
            for value in (0.,1.,float('nan'),float('inf'),-float('inf')):
                payload={'a':[torch.tensor([value],dtype=dtype),torch.tensor([2])],'b':3.}
                self.assertEqual(finite_tree(payload),reference(payload))
        model=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.BatchNorm1d(2))
        model_health(model,gradients=True)
        model[1].running_var[0]=-1
        with self.assertRaises(FloatingPointError):model_health(model)
        model[1].running_var.fill_(1);model[0].weight.grad=torch.full_like(model[0].weight,float('nan'))
        with self.assertRaises(FloatingPointError):model_health(model,gradients=True)

    def test_vectorized_checks_preserve_optimizer_ema_and_snapshot(self):
        import torch
        from types import SimpleNamespace
        from ultralytics.utils.torch_utils import ModelEMA
        from main.training_health import RQ4HealthHooks,same_tree
        from rq4_training_backend import RQ4HealthHooks as Reference
        folder=RUNS/'cpu_test_runtime';folder.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=folder) as name:
            out=Path(name);left=None;right=None
            for kind in (Reference,RQ4HealthHooks):
                obj=type('Harness',(kind,),{})();torch.manual_seed(42)
                obj.model=torch.nn.Sequential(torch.nn.Linear(3,10),torch.nn.BatchNorm1d(10));obj.ema=ModelEMA(obj.model)
                obj.optimizer=torch.optim.AdamW(obj.model.parameters(),lr=.01)
                obj.scaler=torch.amp.GradScaler('cuda',enabled=False);obj.rq4_optimizer_updates=0
                obj.rq4_context={'output':str(out),'selection':'FIXED_LAST'}
                obj.loss=obj.model(torch.arange(12,dtype=torch.float32).reshape(4,3)).square().mean();obj.loss.backward()
                obj.optimizer_step()
                if left is None:left=obj
                else:right=obj
            self.assertTrue(same_tree(left.model.state_dict(),right.model.state_dict()))
            self.assertTrue(same_tree(left.ema.ema.state_dict(),right.ema.ema.state_dict()))
            self.assertTrue(same_tree(left.optimizer.state_dict(),right.optimizer.state_dict()))
            self.assertEqual((right.rq4_optimizer_updates,right.ema.updates),(1,1))


if __name__=='__main__':unittest.main()
