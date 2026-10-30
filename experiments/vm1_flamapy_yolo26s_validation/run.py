import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))


def main(argv=None):
    p = argparse.ArgumentParser()
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--demo', action='store_true')
    modes.add_argument('--full-clean', action='store_true')
    modes.add_argument('--full-paired', action='store_true', help='548-image CLEAN plus DEGRADED evaluation')
    modes.add_argument('--regenerate', type=Path)
    modes.add_argument('--refresh-index', action='store_true')
    modes.add_argument('--uvl-only', action='store_true', help=argparse.SUPPRESS)
    modes.add_argument('--bridge-request', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('config.local.yaml'))
    p.add_argument('--output', type=Path)
    p.add_argument('--no-open', action='store_true')
    args = p.parse_args(argv)
    if args.refresh_index:
        from main.visualization import refresh_config_index
        print(refresh_config_index(args.config, 'vm1_flamapy_yolo26s_validation'), flush=True)
        return
    if args.uvl_only or args.bridge_request:
        from main.bridge import build_bridge
        if args.output is None:
            p.error('--output is required for the UVL bridge')
        request = json.loads(args.bridge_request.read_text(encoding='utf-8')) if args.bridge_request else {}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        value = build_bridge(Path(__file__).resolve().parent, args.output, **request)
        print(json.dumps(value))
        return
    mode = ('check' if args.check else 'demo' if args.demo else 'full-clean' if args.full_clean
            else 'regenerate' if args.regenerate else 'full-paired')
    if mode in ['check', 'regenerate']:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    from main.pipeline import execute
    out = execute(args.config, mode, args.regenerate)
    if mode in ['check', 'regenerate']:
        from main.visualization import create_status_preview
        print('Preview: ' + str(create_status_preview(out)), flush=True)
    if mode in ['demo', 'full-clean', 'full-paired']:
        from main.visualization import create_preview
        import webbrowser
        conditions = json.loads((out / 'conditions.json').read_text())
        links = []
        for item in conditions:
            page = create_preview(out / item['directory'], open_browser=False)
            links.append(f'<p><a href="{page.relative_to(out).as_posix()}">{item["name"]}</a></p>')
        page = out / 'index.html'
        extras = ('<p><a href="paired_comparison.csv">Full paired comparison CSV</a> · '
                  '<a href="paired_comparison.json">JSON and source hashes</a></p>'
                  if mode == 'full-paired' else '')
        page.write_text('<!doctype html><meta charset="utf-8"><h1>YOLO26s validation</h1>' + ''.join(links) + extras,
                        encoding='utf-8')
        print('Preview: ' + str(page), flush=True)
        if not args.no_open:
            webbrowser.open(page.as_uri())
    from main.visualization import update_run_index
    print('Experiment results: ' + str(update_run_index(out)), flush=True)


if __name__ == '__main__':
    main()
