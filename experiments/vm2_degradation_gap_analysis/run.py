import argparse
import sys
import webbrowser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=Path(__file__).with_name('config.local.yaml'))
    modes = p.add_mutually_exclusive_group()
    modes.add_argument('--replay', action='store_true')
    modes.add_argument('--check', action='store_true')
    modes.add_argument('--refresh-index', action='store_true')
    p.add_argument('--no-open', action='store_true', help='Do not open the result page in a browser.')
    args = p.parse_args()
    if args.refresh_index:
        from main.visualization import refresh_config_index
        page = refresh_config_index(args.config, 'vm2_degradation_gap_analysis')
        print('Experiment results: ' + page.as_uri(), flush=True)
    else:
        from main.gap_analysis import replay
        page = replay(args.config) / 'index.html'
    if not args.no_open:
        try:
            if not webbrowser.open(page.as_uri()):
                print('Browser was not opened. Use the page URL above.', flush=True)
        except OSError as exc:
            print('Browser could not be opened: ' + str(exc), flush=True)
