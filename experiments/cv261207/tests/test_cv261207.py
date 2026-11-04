from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import file_id, save, seal, verify_hashes
from archives import OriginalZIP
from gap_analysis import pair_outcome, subset_outcome
from sampling import models, detector_design, uniform


class ProtocolTests(unittest.TestCase):
    def test_uniform_repeatability_and_exact_small_population(self):
        import itertools
        from collections import Counter
        from fractions import Fraction

        population = list(range(4))
        samples = list(itertools.combinations(population, 2))
        counts = Counter(item for sample in samples for item in sample)
        self.assertEqual({Fraction(n, len(samples)) for n in counts.values()}, {Fraction(1, 2)})
        self.assertEqual(uniform(population, 2, 42), uniform(population, 2, 42))
        self.assertEqual(len(set(uniform(population, 2, 42))), 2)
        with self.assertRaises(ValueError):
            uniform(population, 5, 42)

    def test_detector_design(self):
        _, fmd, _ = models()
        rows = detector_design(fmd, 'a'*64)
        self.assertEqual(len({r['detector_id'] for r in rows}), 5)
        for row in rows:
            self.assertEqual(row['detector_id'], file_id(row['leaves'], 'd_'))
            self.assertIn('Train640', row['leaves'])

    def test_archive_layouts(self):
        with TemporaryDirectory() as temp:
            for index, prefix in enumerate(('', 'VisDrone2019-DET-test-dev/')):
                path = Path(temp) / f'{index}.zip'
                with ZipFile(path, 'w') as archive:
                    archive.writestr(f'{prefix}images/frame.jpg', b'jpeg')
                    archive.writestr(f'{prefix}annotations/frame.txt', b'')
                archive = OriginalZIP(path, 1)
                self.assertEqual(len(archive.images), 1)
                archive.close()
                with self.assertRaises(ValueError):
                    OriginalZIP(path, 2)

    def test_missing_and_tampered_evidence(self):
        with TemporaryDirectory() as temp:
            out = Path(temp)
            save(out / 'run.json', {'status': 'PASS'})
            seal(out)
            verify_hashes(out, ('run.json',))
            with self.assertRaises(ValueError):
                verify_hashes(out, ('metrics.json',))
            (out / 'run.json').write_text('changed')
            with self.assertRaises(ValueError):
                verify_hashes(out, ('run.json',))

    def test_oracle_boundaries(self):
        self.assertEqual(pair_outcome([.1,.2], 50, .1, 50), 'PASS')
        self.assertEqual(pair_outcome([.09,.1], 50, .1, 50), 'UNCERTAIN')
        self.assertEqual(pair_outcome([.01,.09], 50, .1, 50), 'FAIL')
        self.assertEqual(pair_outcome([.2,.3], 51, .1, 50), 'FAIL')
        self.assertEqual(pair_outcome(None, 51, .1, 50), 'FAIL')
        self.assertEqual(pair_outcome(None, 50, .1, 50), 'NOT_EVALUATED')
        self.assertEqual(subset_outcome(['FAIL', 'NOT_EVALUATED']), 'UNRESOLVED')
        self.assertEqual(subset_outcome(['FAIL', 'FAIL']), 'GAP_TESTED_SUBSET')

    def test_bootstrap_resamples_real_coco_matches(self):
        from evaluation import bootstrap
        from main.statistics import evaluate_coco

        gt = {'images': [{'id': 1, 'width': 64, 'height': 64}, {'id': 2, 'width': 64, 'height': 64}],
              'categories': [{'id': 1, 'name': 'object'}],
              'annotations': [{'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [0,0,8,8], 'area': 64, 'iscrowd': 0},
                              {'id': 2, 'image_id': 2, 'category_id': 1, 'bbox': [0,0,8,8], 'area': 64, 'iscrowd': 0}]}
        pred = [{'image_id': 1, 'category_id': 1, 'bbox': [0,0,8,8], 'score': .95}]
        evaluator = evaluate_coco(gt, pred)[2]
        results = bootstrap(evaluator, count=12)
        self.assertEqual(results, bootstrap(evaluator, count=12))
        self.assertEqual(results['intervals']['all'], results['intervals']['tiny'])
        values = results['intervals']['all']['draws']
        self.assertTrue(any(v < .1 for v in values))
        self.assertTrue(any(v > .9 for v in values))
        self.assertIsNone(results['intervals']['large']['ci95'])
