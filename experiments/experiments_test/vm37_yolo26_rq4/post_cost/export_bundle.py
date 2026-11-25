"""Export only frozen bundle members to a verified local ZIP."""
import argparse
import hashlib
from pathlib import Path
import sys
import time
import zipfile

POST = Path(__file__).resolve().parent
sys.path.insert(0, str(POST.parent))
from study import *
from portable import verify_bundle, within


def execute(bundle):
    out = create('portable_export_cpu')
    runtime(out, cuda=False)
    started = time.perf_counter()
    manifest = verify_bundle(bundle)
    files = {**manifest['files'], 'bundle_manifest.json': sha256(bundle / 'bundle_manifest.json')}
    target = out / 'VM37_portable_training.zip'
    temporary = target.with_suffix('.partial')
    with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in sorted(files):
            info = zipfile.ZipInfo(relative, (2026, 11, 12, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            with within(bundle, relative).open('rb') as source_stream, archive.open(info, 'w') as destination:
                while block := source_stream.read(1024 * 1024):
                    destination.write(block)
    with zipfile.ZipFile(temporary) as archive:
        if len(archive.namelist()) != len(files) or set(archive.namelist()) != set(files):
            raise ValueError('ZIP membership differs.')
        for relative, expected in files.items():
            value = hashlib.sha256()
            with archive.open(relative) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    value.update(block)
            if value.hexdigest() != expected:
                raise ValueError('ZIP member differs: ' + relative)
    temporary.rename(target)
    result = {'status': 'LOCAL_EXPORT_VERIFIED_GPU_NOT_VALIDATED', 'bundle': str(bundle),
        'bundle_manifest_SHA256': files['bundle_manifest.json'], 'ZIP': source(target),
        'members': len(files), 'CPU_wall_seconds': time.perf_counter() - started,
        'GPU_seconds': 0, 'external_uploads': 0, 'raw_dataset_archives_copied': 0,
        'scope': 'Private local transfer artifact; no execution authorization or new compute budget.'}
    save(out / 'export_result.json', result, out)
    write_atomic(out / 'diary.md', '# Portable local export\n\nArchived frozen members only, excluding dry-run outputs and runtime directories. Re-read every compressed member and checked its original SHA. No upload or GPU execution.\n', out)
    write_atomic(out / 'index.html', '<!doctype html><meta charset="utf-8"><h1>VM37 portable export</h1>'
        '<p>Local export verified. GPU execution remains unvalidated and requires explicit resource admission.</p>'
        '<a href="VM37_portable_training.zip">Training package</a><br><a href="export_result.json">SHA and provenance</a>', out)
    save(out / 'run.json', {'RunID': out.name, 'ExperimentID': EXPERIMENT, 'mode': 'PORTABLE_EXPORT_CPU',
        'status': result['status'], 'GPU_seconds': 0}, out, replace=True)
    seal(out)
    from main.visualization import build_run_index
    build_run_index(RUNS, EXPERIMENT)
    print(out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bundle', required=True, type=Path)
    execute(parser.parse_args().bundle.resolve(strict=True))
