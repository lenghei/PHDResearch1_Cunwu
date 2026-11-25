import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from protocol import ROOT


class BackendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os
        cls.folder=ROOT/'datasets/local_runs/checks/rq4_backend_unit_runtime';cls.folder.mkdir(exist_ok=True)
        os.environ['YOLO_CONFIG_DIR']=str(cls.folder);os.environ['CUDA_VISIBLE_DEVICES']=''
        import torch
        cls.torch=torch

    def harness(self,out):
        import torch
        from ultralytics.utils.torch_utils import ModelEMA
        from rq4_training_backend import RQ4HealthHooks
        class Harness(RQ4HealthHooks):pass
        torch.manual_seed(31);obj=Harness();obj.model=torch.nn.Sequential(torch.nn.Linear(3,10),torch.nn.BatchNorm1d(10))
        obj.ema=ModelEMA(obj.model);obj.optimizer=torch.optim.AdamW(obj.model.parameters(),lr=.001)
        obj.scheduler=torch.optim.lr_scheduler.LambdaLR(obj.optimizer,lambda step:1.)
        obj.scaler=torch.amp.GradScaler('cuda',enabled=False);obj.rq4_optimizer_updates=0;obj.epoch=0
        obj.rq4_context={'output':str(out),'selection':'FIXED_LAST','scope':'SYNTHETIC_CPU_ONLY','initialization_SHA256':'UNIT_TEST_ONLY'}
        obj.args=SimpleNamespace(workers=0);obj.metrics={};obj.fitness=.1;obj.best_fitness=.1;obj.read_results_csv=lambda:{}
        obj.loss=obj.model(torch.arange(12,dtype=torch.float32).reshape(4,3)/12).square().mean();obj.loss.backward()
        return obj

    def test_update_matches_installed_optimizer_and_ema_math(self):
        import tempfile
        from ultralytics.engine.trainer import BaseTrainer
        with tempfile.TemporaryDirectory(dir=self.folder) as folder:
            ours=self.harness(Path(folder));reference=self.harness(Path(folder))
            ours.optimizer_step();BaseTrainer.optimizer_step(reference)
            self.assertEqual(ours.rq4_optimizer_updates,1);self.assertEqual(ours.ema.updates,1)
            for left,right in ((ours.model,reference.model),(ours.ema.ema,reference.ema.ema)):
                self.assertTrue(all(self.torch.equal(v,right.state_dict()[k]) for k,v in left.state_dict().items()))

    def test_fp32_epoch_snapshot_reloads_and_is_not_overwritten(self):
        import tempfile
        with tempfile.TemporaryDirectory(dir=self.folder) as folder:
            obj=self.harness(Path(folder));obj.optimizer_step();self.assertTrue(obj.save_model())
            payload=self.torch.load(obj.last,map_location='cpu',weights_only=False)
            self.assertFalse(payload['exact_resume_validated']);self.assertEqual(payload['updates'],1)
            from rq4_training_backend import same_tree
            self.assertTrue(same_tree(payload['optimizer'],obj.optimizer.state_dict()))
            self.assertTrue(all(not v.is_floating_point() or v.dtype==self.torch.float32 for v in payload['ema'].state_dict().values()))
            obj.final_eval()
            with self.assertRaises(FileExistsError):obj.save_model()

    def test_bad_gradient_and_bn_fail_before_valid_save(self):
        import tempfile
        from rq4_training_backend import model_health
        with tempfile.TemporaryDirectory(dir=self.folder) as folder:
            obj=self.harness(Path(folder));next(obj.model.parameters()).grad.fill_(float('nan'))
            with self.assertRaises(FloatingPointError):obj.optimizer_step()
            self.assertEqual(obj.rq4_optimizer_updates,0);self.assertEqual(obj.ema.updates,0)
            obj.model[1].running_var.fill_(-1)
            with self.assertRaises(FloatingPointError):model_health(obj.model)
            with self.assertRaises(FloatingPointError):obj.save_model()
            self.assertFalse((Path(folder)/'weights').exists())

    def test_backend_cannot_start_gpu_and_epoch_hook_checks_workers(self):
        from rq4_training_backend import backend_class,epoch_start
        with self.assertRaises(RuntimeError):backend_class({})()
        seen=[];obj=SimpleNamespace(args=SimpleNamespace(workers=0),epoch=4,
            train_loader=SimpleNamespace(dataset=SimpleNamespace(set_epoch=seen.append)))
        epoch_start(obj);self.assertEqual(seen,[4]);obj.args.workers=2
        with self.assertRaises(RuntimeError):epoch_start(obj)


if __name__=='__main__':unittest.main()
