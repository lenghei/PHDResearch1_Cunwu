"""Install automatic study discovery into the existing Streamlit dashboard.

Run from the Git repository root after copying both files into code/tools/:
    python -B code/tools/install_dashboard_autodiscovery.py

The original dashboard is backed up outside the Git repository.
The installer is idempotent and refuses unexpected source layouts.
"""
from __future__ import annotations

from pathlib import Path


TARGET = Path(__file__).resolve().parent / 'dashboard.py'
BACKUP = TARGET.parents[3] / 'datasets/research_archive/dashboard_backups/dashboard_before_autodiscovery.py'


def patch(source: str) -> str:
    imports = 'from dashboard_data import DashboardStore, DataError\n'
    replacement = (imports +
                   'from dashboard_autodiscovery import (\n'
                   '    discover_studies, render_discovered_study, LEGACY_STUDIES,\n'
                   ')\n')
    if 'from dashboard_autodiscovery import' in source:
        if ('STUDIES = discover_studies(store)' in source and
                'render_discovered_study(store, STUDIES[label])' in source):
            return source
        raise ValueError('Dashboard contains a partial or different auto-discovery installation.')
    if source.count(imports) != 1:
        raise ValueError('Dashboard data import has changed; refusing to modify the file.')
    source = source.replace(imports, replacement, 1)

    start = source.find('STUDIES = {')
    end = source.find('\nMODELS =', start)
    if start < 0 or end < 0 or source.count('STUDIES = {') != 1:
        raise ValueError('Static study registry was not found exactly once.')
    original = source[start:end]
    if not all(name in original for name in
               ('vm1_flamapy_yolo26s_validation', 'vm2_degradation_gap_analysis', 'vm03_gpu_pilot')):
        raise ValueError('Unexpected static study registry; refusing to modify the file.')
    source = source[:start] + 'STUDIES = discover_studies(store)' + source[end:]

    old = "options = list(STUDIES) if store.mode == 'local' else list(STUDIES)[:2]\n    label = st.selectbox('Experiment', options)"
    new = ("options = list(STUDIES)\n"
           "    if not options:\n"
           "        st.info('No published experiments found.')\n"
           "        st.stop()\n"
           "    label = st.selectbox('Experiment', options)")
    if source.count(old) != 1:
        raise ValueError('Cloud experiment menu has changed; refusing to modify the file.')
    source = source.replace(old, new, 1)

    anchor = 'try:\n    records = store.history(STUDIES[label])\n'
    branch = ("if STUDIES[label] not in LEGACY_STUDIES:\n"
              "    render_discovered_study(store, STUDIES[label])\n"
              "    st.stop()\n\n"
              + anchor)
    if source.count(anchor) != 1:
        raise ValueError('Run history entry point has changed; refusing to modify the file.')
    source = source.replace(anchor, branch, 1)
    compile(source, str(TARGET), 'exec')
    return source


def main() -> None:
    source = TARGET.read_text(encoding='utf-8')
    updated = patch(source)
    if updated == source:
        print('Dashboard automatic discovery is already installed.')
        return
    if BACKUP.exists():
        raise FileExistsError('Backup already exists; review before patching: ' + str(BACKUP))
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    BACKUP.write_text(source, encoding='utf-8')
    TARGET.write_text(updated, encoding='utf-8')
    print('Installed automatic study discovery in ' + str(TARGET))
    print('Original dashboard backup: ' + str(BACKUP))


if __name__ == '__main__':
    main()
