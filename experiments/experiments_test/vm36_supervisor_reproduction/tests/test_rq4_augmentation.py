from pathlib import Path
import copy
import random
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import protocol
from rq4_augmentation import NativeEffects,FAMILIES,TEST_VALUES,engineering_fixture,validate_recipe,draw
from rq4_preparation import optimizer_exposure_plan


class RQ4EffectsTests(unittest.TestCase):
    def image(self):
        return np.random.default_rng(91).integers(0,256,(79,113,3),dtype=np.uint8)

    def test_fixture_cannot_authorize_training(self):
        with self.assertRaises(ValueError):validate_recipe(engineering_fixture(FAMILIES))
        recipe=engineering_fixture(FAMILIES);recipe['status']='APPROVED_TRAINING_RECIPE'
        with self.assertRaises(ValueError):validate_recipe(recipe)

    def test_exclusion_levels_match_the_pdf_aligned_uvl(self):
        from supervisor_compliance import numeric_attributes
        values=numeric_attributes(protocol.FMP.read_text(encoding='utf-8'))
        for field,levels in TEST_VALUES.items():
            self.assertEqual(set(levels),{v for (feature,name),v in values.items() if name==field})

    def test_native_effects_preserve_geometry_source_and_global_rng(self):
        image=self.image();original=image.copy();py_state=random.getstate();np_state=np.random.get_state()
        engine=NativeEffects(engineering_fixture(FAMILIES),42,engineering=True)
        first,trace=engine.apply(image,'fixture',0,0);second,again=engine.apply(image,'fixture',0,0)
        self.assertTrue(np.array_equal(first,second));self.assertEqual(trace,again)
        self.assertTrue(np.array_equal(image,original));self.assertEqual(first.shape,image.shape)
        self.assertEqual(first.dtype,np.uint8);self.assertEqual(random.getstate(),py_state)
        self.assertTrue(np.array_equal(np.random.get_state()[1],np_state[1]))
        self.assertEqual([r['family'] for r in trace],list(FAMILIES))

    def test_cached_image_does_not_freeze_noise_across_occurrences(self):
        engine=NativeEffects(engineering_fixture(('NoiseAug',)),42,engineering=True);image=self.image()
        first,_=engine.apply(image,'fixture',0,0);next_access,_=engine.apply(image,'fixture',0,1)
        next_epoch,_=engine.apply(image,'fixture',1,0)
        self.assertFalse(np.array_equal(first,next_access));self.assertFalse(np.array_equal(first,next_epoch))

    def test_noop_and_disabled_family_are_pixel_identities(self):
        image=self.image()
        recipe=engineering_fixture(FAMILIES)
        for value in recipe['families'].values():value['probability']=0.
        output,events=NativeEffects(recipe,42,engineering=True).apply(image,'fixture',0,0)
        self.assertTrue(np.array_equal(image,output));self.assertTrue(all(not r['active'] for r in events))
        output,_=NativeEffects(engineering_fixture(()),42,engineering=True).apply(image,'fixture',0,0)
        self.assertTrue(np.array_equal(image,output))

    def test_training_integer_draws_exclude_test_levels(self):
        rng=np.random.default_rng(12)
        for field,limits in (('kernel',(7,21)),('quality',(25,75))):
            spec={field:{'distribution':'integer_uniform','low':limits[0],'high':limits[1]}}
            values=[draw(spec,field,rng) for _ in range(200)]
            self.assertFalse(set(values)&set(TEST_VALUES[field]))
            if field=='kernel':self.assertTrue(all(v>=3 and v%2==1 for v in values))
        with self.assertRaises(ValueError):draw({'quality':{'distribution':'integer_uniform','low':25,'high':25}},'quality',rng)

    def test_invalid_physical_domains_fail_closed(self):
        for family,key,value in (('HazeAug','airlight',256),('JPEGAug','quality',101),
                ('MotionBlurAug','kernel',8),('NoiseAug','photons',0)):
            recipe=engineering_fixture((family,));recipe['families'][family][key]=value
            with self.assertRaises(ValueError):validate_recipe(recipe,engineering=True)

    def test_accumulation_crosses_epoch_boundaries_with_partial_batches(self):
        plan=optimizer_exposure_plan(5,2,2,4,0)
        self.assertEqual(plan['image_exposures'],10);self.assertEqual(plan['forward_backward_batches'],6)
        self.assertEqual(plan['optimizer_and_EMA_updates'],3)
        self.assertEqual([r['planned_optimizer_and_EMA_updates'] for r in plan['epochs_detail']],[1,2])
        self.assertTrue(all(r['last_batch_images']==1 for r in plan['epochs_detail']))

    def test_short_canary_warmup_is_not_steady_accumulation(self):
        warm=optimizer_exposure_plan(5,2,2,4,1)
        self.assertEqual(warm['warmup_batches'],100)
        self.assertEqual(warm['optimizer_and_EMA_updates'],6)


if __name__=='__main__':unittest.main()
