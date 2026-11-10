from pathlib import Path
import copy
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from normal_reference import contract,estimate_pair,SCIENCE,BUDGET
from post80_replay import run_path
from protocol import load


class NormalReferenceTests(unittest.TestCase):
    def fixture(self):return load(run_path(SCIENCE)/'FROZEN_MINIMUM_GPU_PLAN.json')

    def test_only_four_original_identity_pairs_are_authorized(self):
        plan=self.fixture();self.assertEqual(len(contract(plan)),4);self.assertEqual(BUDGET,7200)
        bad=copy.deepcopy(plan);bad['new_pairs'][0]['p']['leaves'].append('JPEG_q75')
        with self.assertRaises(ValueError):contract(bad)
        bad=copy.deepcopy(plan);bad['new_pairs'][0]['p']['attributes']['Normal']['gain']=.9
        with self.assertRaises(ValueError):contract(bad)

    def test_population_precision_and_unique_settings_are_fixed(self):
        for field,value in (('images',20),):
            bad=self.fixture();bad['new_pairs'][0][field]=value
            with self.assertRaises(ValueError):contract(bad)
        bad=self.fixture();bad['new_pairs'][0]['d']['precision']='FP16'
        with self.assertRaises(ValueError):contract(bad)
        bad=self.fixture();bad['new_pairs'][1]=copy.deepcopy(bad['new_pairs'][0])
        with self.assertRaises(ValueError):contract(bad)
        bad=self.fixture();bad['bootstrap_draws']=100
        with self.assertRaises(ValueError):contract(bad)

    def test_complete_pair_cost_cannot_drop_after_a_slower_measurement(self):
        first=estimate_pair(750,[]);later=estimate_pair(750,[{'complete_seconds':1000}])
        self.assertGreater(later,first);self.assertGreater(first,750)


if __name__=='__main__':unittest.main()
