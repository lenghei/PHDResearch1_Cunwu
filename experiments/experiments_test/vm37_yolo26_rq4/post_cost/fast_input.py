"""Versioned native-image prefetch; the scientific transforms stay unchanged."""
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
from types import MethodType
import cv2
import numpy as np
from main.training_data import RQ4Dataset, resize_native

VERSION = 'VM37_NATIVE_PREFETCH_V1'


def pixel_sha(image):
    return hashlib.sha256(memoryview(image) if image.flags.c_contiguous else image.tobytes()).hexdigest()


def prefetched_indexes(mosaic):
    indexes = mosaic._original_indexes()
    mosaic.dataset.prefetch_native(indexes)
    return indexes


class PrefetchDataset(RQ4Dataset):
    @classmethod
    def adopt(cls, data, native_threads=2):
        if type(data) is not RQ4Dataset or data.native_cache or data.occurrences or data.buffer:
            raise ValueError('Adopt only a fresh accepted dataset.')
        if native_threads not in (1, 2, 3):
            raise ValueError('Native prefetch is limited to 1..3 threads.')
        data.__class__ = cls
        data.native_threads = native_threads
        data.pool = None
        data.pending = defaultdict(deque)
        data.cache_counts = {'decode_misses': 0, 'native_hits': 0, 'prefetch_groups': 0}
        data._install_prefetch(data.transforms, set())
        if issubclass(cls, ArithmeticPrefetchDataset):
            from input_ops import ArithmeticEffects
            data.effects = ArithmeticEffects(data.effects.recipe, data.effects.seed,
                data.effects.recipe['status'] == 'ENGINEERING_FIXTURE_ONLY')
        return data

    def __init__(self, *args, native_threads=2, **kwargs):
        if native_threads not in (1, 2, 3):
            raise ValueError('Native prefetch is limited to 1..3 threads.')
        self.native_threads = native_threads
        self.pool = None
        self.pending = defaultdict(deque)
        self.cache_counts = {'decode_misses': 0, 'native_hits': 0, 'prefetch_groups': 0}
        super().__init__(*args, **kwargs)
        self._install_prefetch(self.transforms, set())

    def _install_prefetch(self, transform, seen):
        from ultralytics.data.augment import Mosaic
        if id(transform) in seen:
            return
        seen.add(id(transform))
        if isinstance(transform, Mosaic):
            transform._original_indexes = transform.get_indexes
            transform.get_indexes = MethodType(prefetched_indexes, transform)
        for child in getattr(transform, 'transforms', ()):
            self._install_prefetch(child, seen)
        child = getattr(transform, 'pre_transform', None)
        if child is not None:
            self._install_prefetch(child, seen)

    def _prepare_native(self, i, rect_mode=True, resize_short=False):
        if i not in self.native_cache:
            path = Path(self.im_files[i])
            image = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), self.cv2_flag)
            if image is None:
                raise ValueError('Cannot decode native training pixels: ' + str(path))
            self.native_hashes[i] = pixel_sha(image)
            image.setflags(write=False)
            self.native_cache[i] = image
            self.cache_counts['decode_misses'] += 1
            if self.augment:
                self.buffer.append(i)
                if 1 < len(self.buffer) >= self.max_buffer_length:
                    j = self.buffer.pop(0)
                    self.native_cache.pop(j, None)
                    self.native_hashes.pop(j, None)
            elif len(self.native_cache) > 1:
                j, _ = self.native_cache.popitem(last=False)
                self.native_hashes.pop(j, None)
        else:
            self.cache_counts['native_hits'] += 1
        image = self.native_cache.get(i)
        if image is None:
            raise ValueError('Native cache evicted the current image.')
        count = self.occurrences.get(i, 0)
        self.occurrences[i] = count + 1
        return (image, Path(self.im_files[i]).stem, self.epoch_number, count,
                self.native_hashes[i], rect_mode, resize_short)

    def _render_native(self, prepared):
        image, identity, epoch, count, before, rect_mode, resize_short = prepared
        changed, trace = self.effects.apply(image, identity, epoch, count)
        if image.flags.writeable:
            raise ValueError('Cached original must remain read-only.')
        result = resize_native(changed, self.imgsz, rect_mode, resize_short)
        event = {'id': identity, 'epoch': epoch, 'occurrence': count,
                 'native_shape': list(image.shape), 'native_pixel_SHA256': before,
                 'effect_pixel_SHA256': pixel_sha(changed) if any(e['active'] for e in trace) else before,
                 'resized_shape': list(result[0].shape), 'effects': trace}
        return result, event

    def prefetch_native(self, indexes):
        if self.pending:
            raise RuntimeError('Unconsumed native results before the next mosaic.')
        indexes = list(indexes)
        if not indexes:
            return
        prepared = [self._prepare_native(i) for i in indexes]
        if self.pool is None:
            self.pool = ThreadPoolExecutor(max_workers=self.native_threads, thread_name_prefix='native-effects')
        futures = [self.pool.submit(self._render_native, item) for item in prepared]
        try:
            results = [future.result() for future in futures]
        except BaseException:
            for future in futures:
                future.cancel()
            self.close()
            raise
        for i, value in zip(indexes, results):
            self.pending[i].append(value)
        self.cache_counts['prefetch_groups'] += 1

    def load_image(self, i, rect_mode=True, resize_short=False):
        if self.pending.get(i):
            if not rect_mode or resize_short:
                raise ValueError('Prefetched resize mode differs.')
            result, event = self.pending[i].popleft()
            if not self.pending[i]:
                del self.pending[i]
        else:
            if self.pending:
                raise ValueError('Mosaic consumed a different native image order.')
            result, event = self._render_native(self._prepare_native(i, rect_mode, resize_short))
        self.effect_events.append(event)
        return result

    def set_epoch(self, epoch):
        if self.pending:
            raise RuntimeError('Cannot change epoch with pending native results.')
        super().set_epoch(epoch)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None

    def __getstate__(self):
        raise TypeError('Native prefetch requires workers0; multiprocessing semantics are not admitted.')


class ArithmeticPrefetchDataset(PrefetchDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from input_ops import ArithmeticEffects
        self.effects = ArithmeticEffects(self.effects.recipe, self.effects.seed,
            self.effects.recipe['status'] == 'ENGINEERING_FIXTURE_ONLY')
