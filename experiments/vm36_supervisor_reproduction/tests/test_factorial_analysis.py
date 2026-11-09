from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from factorial_analysis import contrast_terms
from full_analysis import paired_effect


class FactorialTests(unittest.TestCase):
    def test_checkpoint_did_difference_uses_all_eight_distinct_corners(self):
        definition=contrast_terms('ck01')+[(-s,k) for s,k in contrast_terms('ck05')]
        self.assertEqual(len(set(k for _,k in definition)),8)
        values={('ck01',1280,25):.4,('ck01',640,25):.2,('ck01',1280,75):.5,('ck01',640,75):.25,
            ('ck05',1280,25):.3,('ck05',640,25):.1,('ck05',1280,75):.4,('ck05',640,75):.15}
        terms=[]
        for sign,key in definition:
            value=values[key]
            terms.append((sign,{'per_area':[{'size':'all','map50_95':value}]},
                {'draws':{'image_ids':[1,2],'samples':2,'seed':42,'sampling':'fixture',
                          'values':{'all':{'map50_95':[value,value]}}}}))
        result=paired_effect(terms,'all')
        self.assertAlmostEqual(result['estimate'],0)
        self.assertAlmostEqual(result['CI95_low'],0)

    def test_resolution_contrast_holds_quality_fixed(self):
        self.assertEqual(contrast_terms('ck01',25),[(1,('ck01',1280,25)),(-1,('ck01',640,25))])


if __name__=='__main__':unittest.main()
