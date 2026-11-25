from pathlib import Path
import json
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from post80_science import legal_design,classification_scores,check_numeric_table
from post80_replay import run_path,ANALYSIS
from post80_loader import read_only_sources


class Post80Tests(unittest.TestCase):
    def fixtures(self):
        design=json.loads((run_path(ANALYSIS)/'MEASURED_DESIGN.json').read_text())
        return {p:{'spec':{'p':v}} for p,v in design['degradation_configurations'].items()}

    def test_design_requires_no_responses_and_has_one_shared_minimal_reference(self):
        found=self.fixtures();design=legal_design(found)
        for r in found.values():r['metrics']={'map50_95':'RESPONSE_MUST_NOT_BE_READ'}
        self.assertEqual(legal_design(found),design)
        self.assertEqual(design['primary_minimal_plan']['new_pairs'],4)
        self.assertEqual(sum(r['name'].startswith('NORMAL_REFERENCE__') for r in design['contrasts']),3)
        self.assertEqual(len(design['new_P_configurations']),8)

    def test_illegal_single_change_is_not_relabelled_as_legal(self):
        from sampling import models
        pm,_,_=models();design=legal_design(self.fixtures())
        for case in design['candidate_checks']:self.assertEqual(pm.valid_leaves(case['leaves']),case['expected_legal'])
        illegal=next(r for r in design['candidate_checks'] if r['name']=='ILLEGAL_LOWLIGHT_TO_NORMAL')
        joint=next(r for r in design['candidate_checks'] if r['name']=='NORMAL_AND_SHOT_LOW')
        self.assertFalse(illegal['expected_legal']);self.assertTrue(joint['expected_legal'])
        self.assertIn('two-family',next(r['reason'] for r in design['candidate_checks'] if r['name']=='NORMAL_AND_SHOT_LOW'))

    def test_majority_accuracy_does_not_imply_balanced_performance(self):
        r=classification_scores(['FAIL']*4+['PASS'],['FAIL']*5)
        self.assertEqual(r['accuracy'],.8);self.assertEqual(r['balanced_accuracy'],.5)
        self.assertAlmostEqual(r['macro_F1'],4/9)
        self.assertIsNone(classification_scores(['FAIL'],['FAIL'])['balanced_accuracy'])

    def test_unknown_numeric_output_cannot_be_filled_with_zero(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'x.csv';p.write_text('id,value\nA,0\n')
            with self.assertRaises(ValueError):check_numeric_table(p,[{'id':'A','value':None}],('id',),('value',))

    def test_source_write_guard_blocks_modification_outside_candidate(self):
        with tempfile.TemporaryDirectory() as folder:
            parent=Path(folder);allowed=parent/'candidate';allowed.mkdir();original=parent/'original.txt';original.write_text('unchanged')
            with read_only_sources(allowed):
                (allowed/'cache.txt').write_text('private')
                with self.assertRaises(PermissionError):original.write_text('bad')
                with self.assertRaises(PermissionError):original.unlink()
            self.assertEqual(original.read_text(),'unchanged')


if __name__=='__main__':unittest.main()
