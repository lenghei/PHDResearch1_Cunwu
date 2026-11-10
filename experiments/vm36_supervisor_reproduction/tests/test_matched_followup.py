from pathlib import Path
import hashlib
import json
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from matched_followup import missing_product,gap_bounds,resumable_pairs


class MatchedPlanningTests(unittest.TestCase):
    def test_complement_does_not_repeat_accepted_pairs(self):
        missing=missing_product({'a','b'},{'x','y'},{('a','x'):True})
        self.assertEqual(missing,[('a','y'),('b','x'),('b','y')])

    def test_unknown_is_not_a_failed_setting(self):
        self.assertEqual(gap_bounds(['FAIL']*3+['NOT_EVALUATED']),'UNRESOLVED')
        self.assertEqual(gap_bounds(['FAIL']*3+['UNCERTAIN']),'UNRESOLVED')
        self.assertEqual(gap_bounds(['FAIL']*4),'GAP_TESTED_SUBSET')
        self.assertEqual(gap_bounds(['PASS']+['NOT_EVALUATED']*3),'COVERED_TESTED_SUBSET')

    def test_resume_requires_sealed_full_split_and_same_configuration(self):
        plan=[{'id':'a','images':548},{'id':'b','images':548}]
        done={'pair_id':'a','status':'SEALED_FULL548','images':548,
            'configuration_sha256':hashlib.sha256(json.dumps(plan[0],sort_keys=True).encode()).hexdigest()}
        self.assertEqual(resumable_pairs(plan,[done]),[plan[1]])
        with self.assertRaises(ValueError):resumable_pairs(plan,[dict(done,images=547)])
        with self.assertRaises(ValueError):resumable_pairs(plan,[dict(done,configuration_sha256='changed')])
        with self.assertRaises(ValueError):resumable_pairs(plan,[done,done])


if __name__=='__main__':unittest.main()
