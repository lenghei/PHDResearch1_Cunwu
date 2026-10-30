"""Start the local viewer with its separate environment."""
import socket
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PYTHON = ROOT / ('.venv_dashboard/Scripts/python.exe' if sys.platform == 'win32' else '.venv_dashboard/bin/python')
URL = 'http://127.0.0.1:8501'

if __name__ == '__main__':
    if not PYTHON.is_file():
        raise FileNotFoundError('Create .venv_dashboard and install code/tools/requirements.txt.')
    with socket.socket() as probe:
        occupied = probe.connect_ex(('127.0.0.1', 8501)) == 0
    if occupied:
        print('Port 8501 is already in use. Open ' + URL + ' if this viewer is already running.', flush=True)
        raise SystemExit(0)
    command = [str(PYTHON), '-m', 'streamlit', 'run', str(Path(__file__).with_name('dashboard.py')),
               '--server.address=127.0.0.1', '--server.port=8501', '--server.headless=true',
               '--browser.gatherUsageStats=false', '--server.fileWatcherType=none',
               '--theme.base=light', '--theme.primaryColor=#087f8c']
    environment = os.environ.copy()
    environment['UAVGAP_DASHBOARD_MODE'] = 'local'
    process = subprocess.Popen(command, cwd=ROOT, env=environment)
    try:
        for _ in range(60):
            if process.poll() is not None:
                raise RuntimeError('Dashboard server exited during startup.')
            try:
                with urllib.request.urlopen(URL + '/_stcore/health', timeout=1) as response:
                    if response.status == 200:
                        print('Dashboard: ' + URL, flush=True)
                        webbrowser.open(URL)
                        break
            except OSError:
                time.sleep(.5)
        else:
            process.terminate()
            raise TimeoutError('Dashboard startup exceeded 30 seconds.')
        process.wait()
    except KeyboardInterrupt:
        process.terminate()
