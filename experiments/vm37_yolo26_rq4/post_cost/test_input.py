"""CPU regression tests for native prefetch ordering and failure handling."""
from collections import OrderedDict, defaultdict
import hashlib
from pathlib import Path
import pickle
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

POST = Path(__file__).resolve().parent
sys.path.insert(0, str(POST))
from profile_input import setup
from study import *


class NativePrefetchTests(unittest.TestCase):
    def setUp(self):
        import cv2
        import numpy as np
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)
        self.paths = []
        for i in range(4):
            image = np.random.default_rng(i).integers(0, 256, (64, 96, 3), dtype=np.uint8)
            path = self.folder / (str(i) + '.jpg')
            cv2.imwrite(str(path), image)
            self.paths.append(str(path))

    def tearDown(self):
        self.temp.cleanup()

    def bare(self, optimized):
        from fast_input import PrefetchDataset
        from main.training_data import RQ4Dataset
        from main.training_effects import NativeEffects, engineering_fixture
        cls = PrefetchDataset if optimized else RQ4Dataset
        data = cls.__new__(cls)
        data.im_files = self.paths
        data.cv2_flag = 1
        data.imgsz = 64
        data.augment = True
        data.buffer = []
        data.max_buffer_length = 4
        data.native_cache = OrderedDict()
        data.native_hashes = {}
        data.epoch_number = 0
        data.occurrences = {}
        data.effect_events = []
        data.effects = NativeEffects(engineering_fixture(('HazeAug', 'ExposureAug', 'MotionBlurAug',
                                                          'NoiseAug', 'JPEGAug')), 42, True)
        if optimized:
            from collections import deque
            data.native_threads = 2
            data.pool = None
            data.pending = defaultdict(deque)
            data.cache_counts = {'decode_misses': 0, 'native_hits': 0, 'prefetch_groups': 0}
        return data

    def test_zero_copy_hash_equals_bytes(self):
        import numpy as np
        from fast_input import pixel_sha
        image = np.arange(600, dtype=np.uint8).reshape(10, 20, 3)
        for value in (image, image[:, ::2]):
            self.assertEqual(pixel_sha(value), hashlib.sha256(value.tobytes()).hexdigest())

    def test_parallel_duplicate_ids_exact(self):
        import numpy as np
        baseline, fast = self.bare(False), self.bare(True)
        try:
            for indexes in ([0], [0, 0, 2], [1], [2, 3, 0]):
                expected = [baseline.load_image(i) for i in indexes]
                if len(indexes) > 1:
                    fast.prefetch_native(indexes)
                actual = [fast.load_image(i) for i in indexes]
                for left, right in zip(expected, actual):
                    self.assertTrue(np.array_equal(left[0], right[0]))
                    self.assertEqual(left[1:], right[1:])
                self.assertEqual(baseline.effect_events, fast.effect_events)
                self.assertEqual(baseline.buffer, fast.buffer)
                self.assertEqual(baseline.occurrences, fast.occurrences)
                self.assertFalse(fast.pending)
        finally:
            fast.close()

    def test_original_cache_remains_read_only(self):
        fast = self.bare(True)
        try:
            fast.prefetch_native([0, 1, 2])
            for i in (0, 1, 2):
                fast.load_image(i)
            for image in fast.native_cache.values():
                self.assertFalse(image.flags.writeable)
        finally:
            fast.close()

    def test_wrong_pending_identity_rejected(self):
        fast = self.bare(True)
        try:
            fast.prefetch_native([0, 1])
            with self.assertRaises(ValueError):
                fast.load_image(3)
        finally:
            fast.close()

    def test_pending_epoch_rejected(self):
        fast = self.bare(True)
        try:
            fast.prefetch_native([0, 1])
            with self.assertRaises(RuntimeError):
                fast.set_epoch(1)
        finally:
            fast.close()

    def test_pending_resize_rejected(self):
        fast = self.bare(True)
        try:
            fast.prefetch_native([0])
            with self.assertRaises(ValueError):
                fast.load_image(0, rect_mode=False)
        finally:
            fast.close()

    def test_multiprocessing_not_silently_enabled(self):
        with self.assertRaises(TypeError):
            pickle.dumps(self.bare(True))

    def test_effect_failure_closes_pool(self):
        fast = self.bare(True)
        with patch.object(fast.effects, 'apply', side_effect=FloatingPointError('injected effect failure')):
            with self.assertRaises(FloatingPointError):
                fast.prefetch_native([0, 1, 2])
        self.assertIsNone(fast.pool)
        self.assertFalse(any(thread.name.startswith('native-effects') for thread in threading.enumerate()))

    def test_decode_failure_stops(self):
        fast = self.bare(True)
        Path(self.paths[0]).write_bytes(b'invalid-jpeg')
        with self.assertRaises(ValueError):
            fast.load_image(0)
        self.assertIsNone(fast.pool)

    def test_arithmetic_and_random_state_exact(self):
        import numpy as np
        import input_ops
        from main.training_effects import renderer
        original = renderer()
        images = [np.zeros((1, 3, 3), dtype=np.uint8), np.full((129, 97, 3), 255, dtype=np.uint8),
                  np.random.default_rng(91).integers(0, 256, (259, 131, 3), dtype=np.uint8)]
        for image in images:
            for seed in (0, 42, 20261113):
                for photons in (20., 43.123, 100.):
                    a, b = np.random.default_rng(seed), np.random.default_rng(seed)
                    self.assertTrue(np.array_equal(original.apply_shot_noise(image, photons, a), input_ops.shot(image, photons, b)))
                    self.assertEqual(a.bit_generator.state, b.bit_generator.state)
                for sigma in (5.5, 13.456, 30.):
                    a, b = np.random.default_rng(seed), np.random.default_rng(seed)
                    self.assertTrue(np.array_equal(original.apply_gaussian_noise(image, sigma, a), input_ops.gaussian(image, sigma, b)))
                    self.assertEqual(a.bit_generator.state, b.bit_generator.state)
                for gain, bias in ((.4, -50.), (.93, 17.3), (1.3, 80.)):
                    self.assertTrue(np.array_equal(original.apply_exposure(image, gain, bias), input_ops.exposure(image, gain, bias)))
                for beta in (.5, .913, 1.5):
                    self.assertTrue(np.array_equal(original.apply_haze(image, beta, 255.), input_ops.haze(image, beta, 255.)))

    def test_adopt_preserves_initialized_source(self):
        import numpy as np
        from fast_input import ArithmeticPrefetchDataset
        original, candidate = self.bare(False), self.bare(False)
        candidate.transforms = object()
        fast = ArithmeticPrefetchDataset.adopt(candidate)
        try:
            expected = [original.load_image(i) for i in (0, 1, 0)]
            fast.prefetch_native([0, 1, 0])
            actual = [fast.load_image(i) for i in (0, 1, 0)]
            self.assertTrue(all(np.array_equal(a[0], b[0]) for a, b in zip(expected, actual)))
            self.assertEqual(original.effect_events, fast.effect_events)
        finally:
            fast.close()


if __name__ == '__main__':
    out = create('input_unit_tests_cpu')
    setup(out)
    with (out / 'tests.log').open('x', encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(NativePrefetchTests))
    save(out / 'test_results.json', {'status': 'PASS' if result.wasSuccessful() else 'FAIL',
         'tests': result.testsRun, 'errors': len(result.errors), 'failures': len(result.failures), 'GPU_seconds': 0}, out)
    save(out / 'run.json', {'RunID': out.name, 'ExperimentID': EXPERIMENT,
         'status': 'PASS' if result.wasSuccessful() else 'FAIL', 'GPU_seconds': 0}, out, replace=True)
    save(out / 'inputs.json', {'sources': [source(Path(__file__)), source(POST / 'fast_input.py')]}, out)
    write_atomic(out / 'diary.md', '# Native prefetch CPU tests\n\nChecked duplicate occurrences, cache order, '
         'read-only pixels, shutdown, decode/effect failure and forbidden worker serialization. GPU=0.\n', out)
    seal(out)
    print(json.dumps({'RunID': out.name, 'tests': result.testsRun, 'PASS': result.wasSuccessful()}))
    raise SystemExit(0 if result.wasSuccessful() else 1)
