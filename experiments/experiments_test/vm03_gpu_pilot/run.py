import argparse
import json
import os
import sys
from pathlib import Path

PUBLIC_ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'code/main').is_dir())
sys.path.insert(0, str(PUBLIC_ROOT / 'code'))
from main.config import save
from main.evaluation import evaluate, regenerate


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--full', action='store_true')
    mode.add_argument('--smoke', action='store_true')
    parser.add_argument('--regenerate')
    parser.add_argument('--output')
    parser.add_argument('--preview-run')
    parser.add_argument('--no-preview', action='store_true')
    parser.add_argument('--no-open', action='store_true')
    args = parser.parse_args(argv)
    # Explicit configurations retain the previous five-image default.
    args.full = args.full or (args.config is None and not args.smoke)
    args.config = args.config or str(Path(__file__).with_name('config.local.yaml'))
    if args.regenerate and not args.output:
        parser.error('--output is required for regeneration')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.regenerate:
        save(args.output, regenerate(args.regenerate))
        return
    if args.preview_run:
        from main.visualization import create_preview
        print(create_preview(args.preview_run, open_browser=not args.no_open), flush=True)
        return
    if not Path(args.config).is_file():
        raise FileNotFoundError('Create config.local.yaml beside this script from config.example.yaml.')
    print('Mode: ' + ('full validation' if args.full else 'five-image smoke'), flush=True)
    caller_temp = {key: os.environ.get(key) for key in ['TMP', 'TEMP']}
    try:
        output = evaluate(args.config, args.full)
    finally:
        # External browsers should not inherit experiment temp directories.
        for key, value in caller_temp.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    metrics = summary['aggregate']
    print(f"Images: {summary['images']} | AP50: {metrics['map50']:.4f} | AP50:95: {metrics['map50_95']:.4f}", flush=True)
    print(f"Model calls: {summary['network_seconds']:.2f}s | COCO evaluation: {summary['evaluation_seconds']:.2f}s", flush=True)
    if not args.no_preview:
        try:
            from main.visualization import create_preview
            print('Preview: ' + str(create_preview(output, open_browser=not args.no_open)), flush=True)
            print('Preview: ground truth on the left, predictions on the right. Charts show AP50:95.', flush=True)
        except Exception as exc:
            save(output / 'preview_failure.json', {'error': repr(exc)})
            print(f'Preview failed; evaluation files are saved: {exc}', flush=True)


if __name__ == '__main__':
    main()
