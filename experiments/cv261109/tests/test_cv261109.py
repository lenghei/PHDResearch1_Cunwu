"""CPU regression tests; FlamaPy installation and real VisDrone require local checks."""
import hashlib
import json
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from model import (ALL_LEAVES, GROUPS, ORDER, combinations, configuration_id,
                   read_uvl_attributes, recipe, valid_leaves)
from generator import generate, effect_seed, LEAF_RENDERERS
import run


class FeatureModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.attributes = read_uvl_attributes(ROOT / 'fm_p.uvl')

    def test_exact_supervisor_valid_configuration_count(self):
        configs = list(combinations())
        self.assertEqual(len(configs), 46368)
        self.assertEqual(len(set(configs)), 46368)

    def test_feature_attributes_are_read_from_uvl(self):
        self.assertEqual(set(self.attributes), ALL_LEAVES)
        self.assertEqual(self.attributes['Gauss_s5']['sigma'], 5.5)
        self.assertEqual(self.attributes['Overexposed_strong'], {'gain': 1.3, 'bias': 80})
        self.assertEqual(self.attributes['RollingShutter']['waves'], 2)

    def test_one_renderer_for_every_supervisor_leaf(self):
        self.assertEqual(set(LEAF_RENDERERS), ALL_LEAVES)
        self.assertEqual(len(LEAF_RENDERERS), 25)
        self.assertTrue(all(callable(handler) for handler in LEAF_RENDERERS.values()))

    def test_constraint_rejections(self):
        invalid = [('LowLight',), ('Normal', 'Rain', 'Overexposed_mild'),
                   ('Overexposed_strong', 'Haze_dense'), ('Normal', 'RollingShutter'),
                   ('Normal', 'Shot_high'), ('Normal', 'Gauss_s5', 'Gauss_s15'),
                   ('Normal', 'Motion_k7', 'Motion_k15')]
        for selected in invalid:
            with self.subTest(selected=selected):
                self.assertFalse(valid_leaves(selected))

    def test_two_blurs_and_two_noises_are_valid(self):
        selected = ('Underexposed_strong','Motion_k15','Defocus_r6',
                    'Gauss_s15','Shot_high','Rain','Downscale_x2','RollingShutter','JPEG_q25')
        self.assertTrue(valid_leaves(selected))
        self.assertEqual(recipe(selected, self.attributes)['Downscale_x2']['factor'], 2.0)

    def test_computed_id_does_not_depend_on_leaf_order(self):
        self.assertEqual(configuration_id(['Normal','JPEG_q25']),
                         configuration_id(['JPEG_q25','Normal']))

    def test_parameter_change_in_uvl_is_visible(self):
        text = (ROOT/'fm_p.uvl').read_text(encoding='utf-8').replace('sigma 5.5', 'sigma 6.5')
        with tempfile.TemporaryDirectory() as folder:
            candidate = Path(folder)/'other.uvl'
            candidate.write_text(text, encoding='utf-8')
            self.assertEqual(read_uvl_attributes(candidate)['Gauss_s5']['sigma'], 6.5)


@unittest.skipUnless(__import__('importlib').util.find_spec('cv2') and
                     __import__('importlib').util.find_spec('numpy'), 'OpenCV and numpy required')
class ImageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        cls.image = np.random.default_rng(7).integers(0, 256, (48, 64, 3), dtype=np.uint8)
        cls.attributes = read_uvl_attributes(ROOT / 'fm_p.uvl')

    def test_identity_does_not_modify_original(self):
        import numpy as np
        copy = self.image.copy()
        generated, trace = generate(self.image, ['Normal'], self.attributes, 'image001', 55)
        np.testing.assert_array_equal(copy, generated)
        np.testing.assert_array_equal(copy, self.image)
        self.assertEqual(len(trace), 1)

    def test_all_effects_fixed_order_and_determinism(self):
        import numpy as np
        features = ('Underexposed_strong','Haze_light','Motion_k21', 'Defocus_r3',
                    'RollingShutter', 'Gauss_s30', 'Shot_high', 'Downscale_x2','JPEG_q25')
        a, ta = generate(self.image, features, self.attributes, 'file-01', 123)
        b, tb = generate(self.image, tuple(reversed(features)), self.attributes, 'file-01', 123)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(ta, tb)
        self.assertEqual(list(dict.fromkeys(t['stage'] for t in ta)), list(ORDER))
        self.assertEqual(a.shape, self.image.shape)
        self.assertEqual(a.dtype, self.image.dtype)

    def test_seed_derivation_is_stable_and_independent(self):
        self.assertEqual(effect_seed(5, 'frame1', 'Rain'), effect_seed(5,'frame1','Rain'))
        self.assertNotEqual(effect_seed(5, 'frame1', 'Rain'), effect_seed(5,'frame2','Rain'))
        self.assertNotEqual(effect_seed(5, 'frame1', 'Rain'), effect_seed(5,'frame1','Gauss_s5'))

    def test_seed_changes_noise(self):
        import numpy as np
        features = ('Normal','Gauss_s15')
        a, _ = generate(self.image, features, self.attributes, 'f', 100)
        b, _ = generate(self.image, features, self.attributes, 'f', 101)
        self.assertFalse(np.array_equal(a, b))

    def test_rain_and_haze_are_distinct(self):
        import numpy as np
        rain, _ = generate(self.image, ('Normal','Rain'), self.attributes, 'a', 5)
        haze, _ = generate(self.image, ('Normal','Haze_light'), self.attributes, 'a', 5)
        self.assertFalse(np.array_equal(rain, haze))

    def test_invalid_combinations_refused(self):
        with self.assertRaises(ValueError):
            generate(self.image, ('LowLight',), self.attributes, 'a', 5)


@unittest.skipUnless(__import__('importlib').util.find_spec('cv2') and
                     __import__('importlib').util.find_spec('numpy'), 'OpenCV and numpy required')
class BundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cv2
        import numpy as np
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.zip = cls.root/'sample.zip'
        with zipfile.ZipFile(cls.zip,'w',compression=zipfile.ZIP_STORED) as archive:
            for i in range(548):
                frame = np.full((24, 32, 3), (i*17) % 255, dtype=np.uint8)
                ok, arr = cv2.imencode('.jpg', frame)
                assert ok
                stem=f'{i+1:07d}_00000_d_0000001'
                archive.writestr('VisDrone2019-DET-val/images/'+stem+'.jpg', arr.tobytes())
                archive.writestr('VisDrone2019-DET-val/annotations/'+stem+'.txt',
                                 '1,2,3,4,1,1,0,0,\n')
        cls.cfg={'dataset_zip':cls.zip,'dataset_zip_sha256':run.sha256(cls.zip),
                 'output_root':cls.root/'runs','expected_images':548,
                 'degradation_seed':123,'flamapy_python':Path(sys.executable)}
        cls.attr=read_uvl_attributes(ROOT/'fm_p.uvl')
        cls.features=('Underexposed_strong','Motion_k7', 'Gauss_s5','JPEG_q75')
        cls.selection={'valid_count':46368,'constraint_count':5,'UVL_SHA256':run.sha256(ROOT/'fm_p.uvl'),
                       'selected_leaves':list(cls.features),'configuration_id':configuration_id(cls.features),
                       'selection_seed':1031,'status':'PASS'}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_dataset_annotation_trailing_comma(self):
        archive=run.ValidationArchive(self.zip)
        try:
            _,label,_,proof=archive.record(0)
            self.assertTrue(label.endswith(b',\n'))
            self.assertEqual(proof['annotation_rows'],1)
        finally:
            archive.close()

    def test_demo_run_and_replay(self):
        root=run.create_run(self.cfg,self.selection,self.attr,3,2,123)
        self.assertEqual(run.verify_run(root,self.cfg,3)['images_regenerated'],3)
        self.assertEqual(json.loads((root/'result.json').read_text())['status'],'DEMO')
        self.assertTrue((root/'preview/index.html').is_file())
        with self.assertRaises(ValueError):
            run.publish_run(root,self.cfg)

    def test_full_run_export_and_snapshot_schema(self):
        root=run.create_run(self.cfg,self.selection,self.attr,548,0,123)
        self.assertEqual(run.verify_run(root,self.cfg,3)['files_checked'],548)
        with patch.object(run,'PUBLIC_RESULTS',self.root/'public'):
            published=run.publish_run(root,self.cfg)
        manifest=json.loads((published/'manifest.json').read_text())
        self.assertEqual(manifest['schema'],2)
        self.assertEqual(manifest['bundle'],'snapshot.json')
        self.assertEqual(manifest['SHA256'],run.sha256(published/'snapshot.json'))
        self.assertEqual(json.loads((published/'snapshot.json').read_text())['files']['summary.json']['dataset']['images'],548)

    def test_verify_rejects_tampered_generated_images(self):
        root=run.create_run(self.cfg,self.selection,self.attr,1,0,123)
        file=next((root/'images').glob('*.png'))
        file.write_bytes(file.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            run.verify_run(root,self.cfg,1)

    def test_verify_rejects_changed_source_archive(self):
        root=run.create_run(self.cfg,self.selection,self.attr,1,0,123)
        changed=dict(self.cfg)
        changed['dataset_zip_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'Source dataset SHA'):
            run.verify_run(root,changed,1)


class SelectorContractTests(unittest.TestCase):
    def test_adapter_refuses_wrong_counts(self):
        sample={'valid_count':27,'constraint_count':5,'UVL_SHA256':run.sha256(ROOT/'fm_p.uvl'),
                'configuration_id':'x','selected_leaves':['Normal']}
        fake=__import__('types').SimpleNamespace(returncode=0, stdout=json.dumps(sample), stderr='')
        with patch.object(run.subprocess, 'run',return_value=fake):
            with self.assertRaisesRegex(ValueError,'model count'):
                run.flamapy_select({'flamapy_python':Path(sys.executable)},1031)


if __name__ == '__main__':
    unittest.main()
