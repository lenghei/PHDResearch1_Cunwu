from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from historical_review import effects, checked, csv_file, table
from common import sha256


class HistoricalReviewTests(unittest.TestCase):
    def test_eight_endpoint_contrast_sign_and_cancellation(self):
        v={(b,j,r):.2+.1*(r==1280)-.05*(b==2)-.03*(j=='Q50')
           for b in (0,2) for j in ('None','Q50') for r in (640,1280)}
        for value in effects(v).values():
            self.assertAlmostEqual(value,0)
        v[2,'Q50',1280]+=.04
        result=effects(v)
        self.assertAlmostEqual(result['JPEG_DID_B0'],0)
        self.assertAlmostEqual(result['JPEG_DID_B2'],.04)
        self.assertAlmostEqual(result['R_B_J'],.04)

    def test_paired_draws_are_contrasted_before_intervals(self):
        shared=np.array([0.,.1,.2])
        v={(b,j,r):shared+.2 for b in (0,2) for j in ('None','Q50') for r in (640,1280)}
        v[2,'Q50',1280]=v[2,'Q50',1280]+np.array([.01,.02,.03])
        np.testing.assert_allclose(effects(v)['R_B_J'],[.01,.02,.03])
        del v[0,'None',640]
        with self.assertRaises(KeyError):
            effects(v)

    def test_tampered_sources_and_output_overwrite_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/'cell.csv'
            csv_file(p,[{'estimate':.123}])
            expected=sha256(p)
            self.assertEqual(len(table(checked(p,expected))),1)
            with self.assertRaises(FileExistsError):
                csv_file(p,[{'estimate':.9}])
            p.write_text('changed',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'SHA mismatch'):
                checked(p,expected)


if __name__=='__main__':
    unittest.main()
