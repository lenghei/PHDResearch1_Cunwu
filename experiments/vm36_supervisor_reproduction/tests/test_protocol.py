from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from protocol import FMP,FMD,LEGACY_FMP,LEGACY_FMD,PRIOR,load,candidate_design,RQ,CONTRIBUTIONS
from alignment import aligned_text,parameters,require_runtime_bindings
from supervisor_compliance import numeric_attributes,normalize
from validate_models import boolean_text


class ProtocolTests(unittest.TestCase):
    def test_original_questions_and_contributions_are_not_rewritten(self):
        page=normalize(load(PRIOR/'pdf_pages.json')['pages'][3])
        for line in RQ+CONTRIBUTIONS:self.assertIn(normalize(line),page)

    def test_both_numeric_models_match_pdf_with_no_extra_values(self):
        pages=load(PRIOR/'pdf_pages.json')['pages']
        self.assertEqual(numeric_attributes(FMP.read_text(encoding='utf-8')),numeric_attributes('\n'.join(pages[4:6])))
        self.assertEqual(numeric_attributes(FMD.read_text(encoding='utf-8')),numeric_attributes('\n'.join(pages[6:9])))

    def test_boolean_projection_preserves_the_strict_original_tree(self):
        for aligned,old in ((FMP,LEGACY_FMP),(FMD,LEGACY_FMD)):
            self.assertEqual(boolean_text(aligned.read_text(encoding='utf-8')),boolean_text(old.read_text(encoding='utf-8')))
        changed=FMD.read_text(encoding='utf-8').replace('SAHI_Slicing => !TTA','SAHI_Slicing => TTA')
        self.assertNotEqual(boolean_text(changed),boolean_text(LEGACY_FMD.read_text(encoding='utf-8')))

    def test_projection_does_not_conceal_unexpected_numeric_drift(self):
        with self.assertRaises(ValueError):aligned_text('A {sigma 6}\n',{('A','sigma'):5.5})

    def test_original_motion_kernel_has_no_invented_orientation(self):
        recipe=parameters(FMP,['Normal','Motion_k21','JPEG_q25'])
        self.assertEqual(set(recipe['Motion_k21']),{'kernel'})
        self.assertEqual(recipe['Motion_k21']['kernel'],21)
        self.assertEqual(parameters(FMP,['Normal','Motion_k21','RollingShutter'])['RollingShutter'],{})

    def test_unreviewed_runtime_bindings_fail_closed(self):
        recipe={'Motion_k21':{'kernel':21}}
        proposal={'proposed_values_from_historical_implementation':{'Motion_k21':{'angle_min':0,'angle_max':180}}}
        with self.assertRaises(ValueError):require_runtime_bindings(recipe,proposal)
        approval={'status':'APPROVED','protocol_sha256':'a'*64,'approved_values':proposal['proposed_values_from_historical_implementation']}
        with self.assertRaises(ValueError):require_runtime_bindings(recipe,proposal,approval,'a'*64)
        approval.update(reviewer='unit-test fixture only',reviewed_at_utc='2000-01-01T00:00:00Z')
        with self.assertRaises(ValueError):require_runtime_bindings(recipe,proposal,approval,'b'*64)
        self.assertEqual(require_runtime_bindings(recipe,proposal,approval,'a'*64)['Motion_k21']['kernel'],21)

    def test_candidate_factorial_is_legal_unique_and_not_approved(self):
        rows=candidate_design();self.assertEqual(len(rows),36)
        self.assertEqual(len({r['pair_id'] for r in rows}),36)
        for row in rows:
            self.assertEqual(row['images'],548);self.assertEqual(row['status'],'PLANNED')
            self.assertEqual(row['student_review'],'NEEDS_SCIENTIFIC_REVIEW')
            self.assertNotIn('Gauss_s5',row['selected_FM_P'])
            self.assertNotIn('RollingShutter',row['selected_FM_P'])

    def test_jpeg_only_generation_is_deterministic_without_extensions(self):
        import numpy as np
        from common import import_file
        sys.path.insert(0,str(LEGACY_FMP.parent))
        try:renderer=import_file('vm36_renderer_fixture',LEGACY_FMP.with_name('generator.py'))
        finally:sys.path.remove(str(LEGACY_FMP.parent))
        image=np.arange(64*96*3,dtype=np.uint8).reshape(64,96,3)
        for leaf in ('JPEG_q25','JPEG_q50','JPEG_q75'):
            leaves=['Normal',leaf];attrs=parameters(FMP,leaves)
            a,trace=renderer.generate(image,leaves,attrs,'unit_fixture',42)
            b,again=renderer.generate(image,leaves,attrs,'unit_fixture',42)
            self.assertTrue(np.array_equal(a,b));self.assertEqual(trace,again);self.assertEqual(a.shape,image.shape)


if __name__=='__main__':unittest.main()
