from pathlib import Path
import copy
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from full_analysis import binary_sample_size,matched_contrasts,paired_effect,tested_subset_state


def evidence(value,seed=42):
    sizes=('all','tiny','small','medium','large')
    measured={'per_area':[{'size':size,'map50_95':value} for size in sizes]}
    bootstrap={'draws':{'image_ids':[1,2,3], 'samples':3, 'seed':seed,'sampling':'fixture',
        'values':{size:{'map50_95':[value-.01,value,value+.01]} for size in sizes}}}
    return {},measured,bootstrap


class FullAnalysisTests(unittest.TestCase):
    def test_missing_corners_do_not_create_a_did(self):
        data={('ck05',640,25):evidence(.1),('ck05',1280,25):evidence(.2)}
        result=matched_contrasts(data)
        self.assertEqual(len(result),5)
        self.assertTrue(all(row['contrast']=='1280-640 at Q25' for row in result))
        self.assertAlmostEqual(result[0]['estimate'],.1)

    def test_four_corners_use_the_frozen_did_sign(self):
        data={('ck05',640,25):evidence(.1),('ck05',1280,25):evidence(.2),
              ('ck05',640,75):evidence(.15),('ck05',1280,75):evidence(.3)}
        rows=matched_contrasts(data)
        did=next(r for r in rows if r['size']=='all' and r['contrast'].startswith('('))
        self.assertAlmostEqual(did['estimate'],-.05)
        self.assertAlmostEqual(did['CI95_low'],-.05)
        self.assertEqual(len(rows),15)

    def test_different_resampling_seed_is_rejected(self):
        a=evidence(.1);b=evidence(.2,43)
        with self.assertRaises(ValueError):paired_effect([(1,a[1],a[2]),(-1,b[1],b[2])],'all')

    def test_different_image_identity_is_rejected(self):
        a=evidence(.1);b=evidence(.2)
        b[2]['draws']['image_ids']=[1,2,4]
        with self.assertRaises(ValueError):paired_effect([(1,a[1],a[2]),(-1,b[1],b[2])],'all')

    def test_missing_draw_is_not_zero_filled(self):
        a=evidence(.1);a[2]['draws']['values']['all']['map50_95'].pop()
        with self.assertRaises(ValueError):paired_effect([(1,a[1],a[2])],'all')

    def test_unmeasured_detector_prevents_a_gap_claim(self):
        self.assertEqual(tested_subset_state(['FAIL']*3,4),'UNRESOLVED')
        self.assertEqual(tested_subset_state(['FAIL']*4,4),'GAP_TESTED_SUBSET')
        self.assertEqual(tested_subset_state(['FAIL','PASS'],4),'COVERED_TESTED_SUBSET')
        self.assertEqual(tested_subset_state(['FAIL']*3+['UNCERTAIN'],4),'UNRESOLVED')

    def test_finite_population_precision_is_a_sensitivity_not_a_fixed_target(self):
        self.assertEqual(binary_sample_size(46368,.1),96)
        self.assertGreater(binary_sample_size(46368,.05),binary_sample_size(46368,.1))
        self.assertLessEqual(binary_sample_size(10,.01),10)
        with self.assertRaises(ValueError):binary_sample_size(100,0)


if __name__=='__main__':unittest.main()
