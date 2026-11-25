"""VM44 tests use synthetic boxes, not synthetic scientific responses."""
import copy
import io
import unittest
import numpy as np

from diagnostics import retained, match, reference_match, iou_matrix, size
from main.labels import keep_outside_ignore
from main.statistics import _mean_valid
from selector import Controller, legal, state, disjoint_groups
from vm44_io import UVL, sha, source, save


def execute_tests(out, cnf, rows, candidates, support):
    class Checks(unittest.TestCase):
        def setUp(self):
            self.rec = {'id': 1, 'width': 100, 'height': 100, 'ignore_regions': [], 'other_regions': []}
            self.gt = [{'id': 1, 'category_id': 1, 'bbox': [1., 2., 10., 10.]}]
            self.pred = [{'index': 0, 'category_id': 1, 'bbox': [1., 2., 10., 10.], 'score': .25}]
            self.controller = Controller(cnf, rows, candidates, support)
            self.leaves = ['Normal']

        def test_exact_source_SHA(self):
            self.assertEqual(sha(UVL/'fm_p.uvl'), 'e6eadd633d1c14e0d6bc425f5e54421d592d766c65a6e5f20c61bc34dd76fb5c')
            with self.assertRaises(ValueError): source(UVL/'fm_p.uvl', '0'*64)

        def test_class_mapping_and_original_coordinates(self):
            p = {'image_id': 1, 'class_id': 0, 'bbox': [1.25, 2.5, 10.5, 8.5], 'score': .3}
            result, _ = retained([p], self.rec)
            self.assertEqual(result[0]['category_id'], 1)
            self.assertEqual(result[0]['bbox'], p['bbox'])

        def test_ignore_union(self):
            regions = [[0, 0, 3, 10], [2, 0, 3, 10]]
            self.assertEqual(keep_outside_ignore([[0, 0, 10, 10]], regions, 100, 100).tolist(), [False])
            self.assertEqual(keep_outside_ignore([[0, 0, 10, 10]], [[0, 0, 4, 10]], 100, 100).tolist(), [True])

        def test_deterministic_matching(self):
            a = match(self.pred, self.gt)[0]
            self.assertEqual(a, reference_match(self.pred, self.gt))
            self.assertEqual(a, match(self.pred, self.gt)[0])

        def test_duplicate_predictions_are_FP(self):
            duplicate = [{**self.pred[0], 'index': i} for i in range(2)]
            self.assertEqual(match(duplicate, self.gt)[0], {0: 0})

        def test_wrong_image_ID(self):
            with self.assertRaises(ValueError): retained([{'image_id': 2, 'class_id': 0, 'bbox': [0, 0, 1, 1], 'score': .5}], self.rec)

        def test_wrong_native_class(self):
            with self.assertRaises(ValueError): retained([{'image_id': 1, 'class_id': 10, 'bbox': [0, 0, 1, 1], 'score': .5}], self.rec)

        def test_missing_class_metric_null(self):
            self.assertIsNone(_mean_valid(np.asarray([-1., -1.])))

        def test_no_GT_image(self):
            self.assertEqual(match(self.pred, [])[0], {})
            self.assertEqual(iou_matrix(self.pred, []).shape, (1, 0))

        def test_no_prediction_image(self):
            self.assertEqual(match([], self.gt)[1], {})

        def test_confidence_edge(self):
            self.assertEqual(len(match(self.pred, self.gt)[0]), 1)
            self.assertEqual(len(match([{**self.pred[0], 'score': np.nextafter(.25, 0)}], self.gt)[0]), 0)

        def test_IoU_edge(self):
            pred = [{**self.pred[0], 'bbox': [1., 2., 5., 10.]}]
            self.assertEqual(iou_matrix(pred, self.gt)[0, 0], .5)
            self.assertEqual(len(match(pred, self.gt)[0]), 1)

        def test_class_aware_assignment(self):
            self.assertEqual(match([{**self.pred[0], 'category_id': 2}], self.gt)[0], {})

        def test_GT_tie_by_ID(self):
            gt = [{**self.gt[0], 'id': n} for n in (3, 1)]
            self.assertEqual(match(self.pred, gt)[0], {0: 1})

        def test_size_edges(self):
            self.assertEqual([size(x) for x in (255.9, 256, 1024, 9216)], ['tiny', 'small', 'medium', 'large'])

        def test_original_UVL_valid(self):
            self.assertTrue(legal(cnf['fm_p'], ['Normal']))
            self.assertTrue(legal(cnf['fm_p'], ['LowLight', 'Shot_low']))
            self.assertTrue(legal(cnf['fm_p'], ['Normal', 'Motion_k21', 'RollingShutter']))

        def test_SAT_against_independent_PySAT(self):
            for name in ('fm_p', 'fm_d'):
                for case in cnf[name]['reference_cases']:
                    self.assertEqual(legal(cnf[name], case['leaves']), case['PySAT_reference'])

        def test_original_UVL_illegal_combinations(self):
            for leaves in (['Normal', 'Rain', 'Overexposed_mild'], ['LowLight'], ['Normal', 'Shot_high'], ['Normal', 'RollingShutter'], ['Overexposed_strong', 'Haze_dense'], ['Normal', 'Normal'], ['Unknown']):
                self.assertFalse(legal(cnf['fm_p'], leaves), leaves)

        def test_D_constraints(self):
            for c in candidates.values(): self.assertTrue(legal(cnf['fm_d'], c['declared_leaves']))
            original = next(iter(candidates.values()))['declared_leaves']
            changed = [x for x in original if x not in ('YOLOv8', 'YOLO26', 'NMS')]+['YOLOv8', 'EndToEnd_NMSFree']
            self.assertFalse(legal(cnf['fm_d'], changed))

        def test_unsupported_D(self):
            r = self.controller.select(self.leaves, eligible=['UNTRAINED_CUSTOM_DETECTOR'])
            self.assertEqual(r['status'], 'NO_CONFIRMED_FEASIBLE_DETECTOR')

        def test_no_feasible_detector(self):
            self.assertEqual(self.controller.select(self.leaves, quality=1)['status'], 'NO_CONFIRMED_FEASIBLE_DETECTOR')

        def test_no_training_available(self):
            self.assertIn('NO_TRAINING_AVAILABLE', self.controller.select(self.leaves, eligible=[])['reason'])

        def test_strict_training_provenance_abstention(self):
            self.assertEqual(self.controller.select(self.leaves, strict_training=True)['status'], 'NO_CONFIRMED_FEASIBLE_DETECTOR')

        def test_unmeasured_condition(self):
            self.assertEqual(self.controller.select(['Normal', 'Motion_k21'])['status'], 'UNKNOWN_UNMEASURED_CONDITION')

        def test_heldout_group_leakage(self):
            with self.assertRaises(ValueError): disjoint_groups(rows[:1], rows[:1])
            train, test = [r for r in rows if r['leaves'] != ['Normal']], [r for r in rows if r['leaves'] == ['Normal']]
            self.assertTrue(disjoint_groups(train, test))

        def test_no_external_data_for_fitting(self):
            external = [{**rows[0], 'evidence_scope': 'LIMITED_EXTERNAL_EXPLORATORY', 'source_folder': 'vm43_external_pilot'}]
            with self.assertRaises(ValueError): Controller(cnf, external, candidates, support)

        def test_invalid_input_abstention(self):
            self.assertEqual(self.controller.select(['LowLight'])['status'], 'INVALID_FM_P_CONFIGURATION')

        def test_interval_rule_edges(self):
            r = {'Tiny_CI_low': .1, 'Tiny_CI_high': .1, 'latency_CI_low_ms': 90, 'latency_CI_high_ms': 100}
            self.assertEqual(state(r), 'PASS')
            self.assertEqual(state({**r, 'latency_CI_high_ms': 100.01}), 'FAIL')
            self.assertEqual(state({**r, 'Tiny_CI_low': .09}), 'UNCERTAIN')

        def test_all_selected_are_interval_feasible(self):
            for group in {r['condition_id'] for r in rows}:
                leaves = next(r['leaves'] for r in rows if r['condition_id'] == group)
                r = self.controller.select(leaves)
                if r['selected']: self.assertEqual(state(r['selected']), 'PASS')

    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Checks))
    (out/'CPU_tests.log').write_text(stream.getvalue(), encoding='utf-8')
    report = {'tests': result.testsRun, 'passed': result.testsRun-len(result.failures)-len(result.errors), 'failed': len(result.failures), 'errors': len(result.errors),
              'synthetic_inputs': 'SOFTWARE_TEST_ONLY_NOT_YOLO_RESULTS', 'GPU': 0}
    save(out/'CPU_test_results.json', report)
    return report
