from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from development_canary_v1 import augment_pixels,healthy,finite_state,epoch_start


class CanaryTests(unittest.TestCase):
    def test_hooks_preserve_geometry_dtype_and_repeatability(self):
        image=np.random.default_rng(2).integers(0,256,(75,91,3),dtype=np.uint8)
        for group in ('clean_control','NoiseAug','MotionBlurAug','ExposureAug','JPEGAug','HazeAug'):
            a,pa=augment_pixels(image,group,42);b,pb=augment_pixels(image,group,42)
            self.assertEqual(a.shape,image.shape);self.assertEqual(a.dtype,np.uint8)
            np.testing.assert_array_equal(a,b);self.assertEqual(pa,pb)
            if group=='clean_control':np.testing.assert_array_equal(a,image)
            if group=='MotionBlurAug':self.assertNotIn(pa['kernel'],(7,15,21))
            if group=='JPEGAug':self.assertNotIn(pa['quality'],(25,50,75,85))

    def test_health_gate_rejects_nonfinite_and_negative_BN_variance(self):
        import torch
        layer=torch.nn.BatchNorm2d(3);healthy(layer)
        layer.running_var[0]=-1
        with self.assertRaises(ValueError):healthy(layer)
        layer.running_var[0]=float('inf')
        with self.assertRaises(ValueError):healthy(layer)

    def test_optimizer_health_and_no_automatic_batch_reduction(self):
        import torch
        from types import SimpleNamespace
        finite_state({'x':[torch.tensor([1.0])]})
        with self.assertRaises(ValueError):finite_state({'x':[torch.tensor([float('nan')])]})
        trainer=SimpleNamespace(epoch=0,_oom_retries=0,train_loader=SimpleNamespace(dataset=SimpleNamespace(epoch=-1)))
        epoch_start(trainer)
        self.assertEqual(trainer._oom_retries,3)
        self.assertEqual(trainer.train_loader.dataset.epoch,0)

    def test_continuous_hooks_reject_exact_heldout_levels(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        image=np.zeros((8,8,3),dtype=np.uint8)
        fake=SimpleNamespace(uniform=lambda *args:15.,normal=lambda *args:np.zeros(image.shape))
        with patch('numpy.random.default_rng',return_value=fake):
            _,params=augment_pixels(image,'NoiseAug',42)
            self.assertNotIn(params['sigma'],(5.5,15,30))
        values=iter((.5,200.))
        with patch('numpy.random.default_rng',return_value=SimpleNamespace(uniform=lambda *args:next(values))):
            _,params=augment_pixels(image,'HazeAug',42)
            self.assertNotIn(params['beta'],(.5,1.5))


if __name__=='__main__':unittest.main()
