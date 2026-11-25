import copy
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import ROOT
from development_fast_coco import accumulate_fast
from main.statistics import evaluate_coco


class DevelopmentTests(unittest.TestCase):
    def test_vectorized_accumulator_matches_reference_with_repeated_images(self):
        gt={'images':[{'id':i,'width':100,'height':100} for i in (1,3,7)],
            'categories':[{'id':1,'name':'object'}],
            'annotations':[{'id':i,'image_id':i,'category_id':1,'bbox':[0,0,8,8],'area':64,'iscrowd':0} for i in (1,3,7)]}
        pred=[{'image_id':i,'category_id':1,'bbox':[0,0,8,8],'score':.5} for i in (1,3)]
        ev=evaluate_coco(gt,pred)[2]
        n=3;groups=len(ev.params.catIds)*len(ev.params.areaRng)
        ev.evalImgs=[ev.evalImgs[g*n+i] for g in range(groups) for i in (2,0,0)]
        ev.params.imgIds=[1,2,3];ev._paramsEval=copy.deepcopy(ev.params)
        with redirect_stdout(StringIO()):ev.accumulate()
        original={k:ev.eval[k].copy() for k in ('precision','recall','scores')}
        accumulate_fast(ev)
        for k in original:np.testing.assert_allclose(ev.eval[k],original[k],rtol=0,atol=1e-14)

    def test_new_degradations_are_valid_and_geometry_preserving(self):
        from development_v1 import generator
        from sampling import models
        from common import FMP
        fmp,_,_=models();gen=generator();attrs=fmp.read_uvl_attributes(FMP)
        image=np.random.default_rng(2).integers(0,256,(55,87,3),dtype=np.uint8)
        for leaves in (['Normal','Downscale_x1_5'],['Normal','JPEG_q25']):
            self.assertTrue(fmp.valid_leaves(leaves));self.assertNotIn('RollingShutter',leaves)
            a,_=gen.generate(image,leaves,attrs,'test',20261108)
            b,_=gen.generate(image,leaves,attrs,'test',20261108)
            self.assertEqual(a.shape,image.shape);np.testing.assert_array_equal(a,b)


if __name__=='__main__':unittest.main()
