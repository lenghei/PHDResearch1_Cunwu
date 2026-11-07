from pathlib import Path
import copy
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gap_analysis import pair_outcome,subset_outcome
from learning import fold_membership
from rq_gaps import joint_intervals
from rq4_protocol import validate


class RQEvidenceTests(unittest.TestCase):
    def test_failure_of_one_conjunct_does_not_require_the_other(self):
        self.assertEqual(pair_outcome([.01,.03],None,.1,50),'FAIL')
        self.assertEqual(pair_outcome(None,60,.1,50),'FAIL')
        self.assertEqual(pair_outcome([.2,.3],None,.1,50),'NOT_EVALUATED')
        self.assertEqual(pair_outcome([.09,.11],40,.1,50),'UNCERTAIN')
        self.assertEqual(pair_outcome([.1,.2],50,.1,50),'PASS')
        self.assertEqual(subset_outcome(['FAIL','NOT_EVALUATED']),'UNRESOLVED')

    def test_disjoint_configurations_are_not_disjoint_images(self):
        a={'group':'a','image_sha256':['a'*64]}
        b={'group':'b','image_sha256':['a'*64]}
        self.assertEqual(fold_membership([a],[b]),'IMAGE_LEAKAGE')
        b['image_sha256']=['b'*64]
        self.assertEqual(fold_membership([a],[b]),'PASS')
        b['group']='a'
        self.assertEqual(fold_membership([a],[b]),'DEGRADATION_LEAKAGE')
        b['group']='b';b['image_sha256']='b'*64
        self.assertEqual(fold_membership([a],[b]),'IMAGE_MEMBERSHIP_UNVERIFIED')

    def test_joint_intervals_preserve_pairing_and_reject_missing_data(self):
        draws=np.array([[.1,.2],[.12,.23],[.08,.17],[.1,.2]])
        low,high,c=joint_intervals([.1,.2],draws)
        self.assertTrue(np.all(low<=np.array([.1,.2])))
        self.assertTrue(np.all(high>=np.array([.1,.2])))
        self.assertGreater(c,0)
        draws[0,0]=np.nan
        with self.assertRaises(ValueError):joint_intervals([.1,.2],draws)

    def test_training_preparation_never_grants_a_budget(self):
        protocol={'status':'NOT_EVALUATED','checkpoint_initialization':{'a':'a'*64,'b':'b'*64},
             'groups':['clean_control','NoiseAug','MotionBlurAug','ExposureAug','JPEGAug','HazeAug'],
             'budget':{'approved_worker_seconds':None,'automatic_long_training':False}}
        self.assertFalse(validate(protocol)['training_execution_ready'])
        protocol['budget']['approved_worker_seconds']=3600
        with self.assertRaises(ValueError):validate(protocol)


if __name__=='__main__':unittest.main()
