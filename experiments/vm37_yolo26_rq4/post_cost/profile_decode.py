"""Separate source decoding from JPEG decoding in the accepted CPU profiler."""
from pathlib import Path
import argparse
import sys

POST = Path(__file__).resolve().parent
sys.path.insert(0, str(POST))
import profile_input


class DecodeTimings(profile_input.Timings):
    def wrap(self, function, name):
        if name != 'cv2.imdecode':
            return super().wrap(function, name)
        def timed(*args, **kwargs):
            label = 'JPEG_decode' if any(frame[0] == 'apply_jpeg' for frame in self.stack) else 'source_decode'
            with self.measure(label):
                return function(*args, **kwargs)
        return timed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request-file', type=Path)
    args = parser.parse_args()
    original_create = profile_input.create
    def create(mode):
        out = original_create(mode + '_decode_detail')
        profile_input.save(out / 'decode_instrumentation.json', {
            'driver': profile_input.source(Path(__file__)),
            'decoder_scope': 'Parent stack distinguishes source JPEG decode from post-augmentation JPEG decode.',
            'metric_or_transform_changed': False}, out)
        (out / 'decode_profiler_source.py').write_bytes(Path(__file__).read_bytes())
        return out
    profile_input.create = create
    profile_input.Timings = DecodeTimings
    profile_input.execute(512, 32, args.request_file)
