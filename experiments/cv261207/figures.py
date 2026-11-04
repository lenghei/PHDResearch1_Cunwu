"""Regenerate CV261207 figures from sealed data."""
from collections import Counter
import csv
import html

from common import EXPERIMENTS, FMD, import_file, load, save, sha256
from sampling import models


def csv_table(path, rows):
    with path.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def render(out, audit, summary):
    from evaluation import runtime
    runtime(out)
    old = import_file('cv261123_figures', EXPERIMENTS / 'cv261123/figures.py')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    fmp, _, _ = models()
    model = load(audit / 'model_validation.json')
    static = load(audit / 'static_gap.json')
    selection = load(audit / 'selection_manifest.json')
    gaps = load(out / 'empirical_gap.json')
    learned = load(out / 'learning.json')
    p, d = model['FM_P']['valid_count'], model['FM_D']['valid_count']
    figures = {}

    def add(number, figure, sources):
        figures[f'fig{number:02}'] = (figure, sources)

    def panel(number, title, state, reason, sources):
        fig, ax = plt.subplots(figsize=(15, 8.5))
        old.heading(fig, title, f'Figure {number:02} | research status')
        ax.axis('off')
        ax.text(.08, .57, state, transform=ax.transAxes, fontsize=24, color=old.TEAL, weight='bold')
        ax.text(.08, .45, reason, transform=ax.transAxes, fontsize=13, wrap=True)
        add(number, fig, sources)

    uvls = ['model_validation.json', 'fm_p.uvl', 'fm_d.uvl']
    add(1, old.page01_two_trees(), uvls)
    add(2, old.page02_feature_models(fmp, old.read_uvl_top_groups(FMD), {'fmd_leaves': 34}), uvls)
    fig, ax = plt.subplots(figsize=(15, 8.5))
    old.heading(fig, 'Design space and measured pairs', 'Figure 03 | scope')
    labels = ['Valid FM_P', 'Valid FM_D', 'Theoretical product', 'Frozen pair plan']
    counts = [p, d, p*d, summary['planned_pairs']]
    ax.barh(labels, counts, color=[old.TEAL, old.BLUE, old.NAVY, old.AMBER])
    ax.set_xscale('log')
    ax.set_position([.25, .28, .62, .46])
    ax.invert_yaxis()
    for label, count in zip(labels, counts):
        ax.text(count, label, f'  {count:,}', va='center')
    measured = Counter(row['scope'] for row in summary['pair_runs'])
    old.foot(fig, f"Saved pair runs: {measured['TEST_DEV']} test-dev; {measured['VAL_ENGINEERING_ONLY']} validation engineering.")
    add(3, fig, ['summary.json'])
    csv_table(out / 'fig03_counts.csv', [{'quantity': x, 'count': y} for x, y in zip(labels, counts)]
              + [{'quantity': key, 'count': measured[key]} for key in ('TEST_DEV','VAL_ENGINEERING_ONLY')])
    hist = selection['full_space_histogram']
    csv_table(out / 'fig04_histogram.csv', [{'leaf_count': key, 'configurations': value} for key, value in hist.items()])
    add(4, old.page04_distribution(hist), ['fig04_histogram.csv', 'selection_manifest.json'])
    gap_rows = [{'definition': k, 'covered': v['covered'], 'gap': v['gap'], 'population': v['population']}
                for k, v in static['definitions'].items()]
    csv_table(out / 'fig05_static_gap.csv', gap_rows)
    fig, ax = plt.subplots(figsize=(15, 8.5))
    old.heading(fig, 'Static coverage sensitivity', 'Figure 05 | definition matters',
                'Rule-based coverage in the complete FM_P. Pixel equivalence is assumed only for the identity recipe.')
    ax.bar([r['definition'].replace('_',' ') for r in gap_rows], [r['covered'] for r in gap_rows],
           color=[old.TEAL, old.BLUE, old.AMBER], width=.56)
    ax.set_ylabel(f'Covered configurations (out of {p:,})')
    ax.set_position([.14,.20,.77,.56])
    for index, row in enumerate(gap_rows):
        ax.text(index, row['covered']+.3, str(row['covered']), ha='center')
    ax.set_ylim(0, max(r['covered'] for r in gap_rows)*1.2)
    add(5, fig, ['fig05_static_gap.csv', 'static_gap.json'])
    leaves = [{'leaf': k, **v} for k,v in static['leaves'].items()]
    csv_table(out / 'fig06_leaf_coverage.csv', leaves)
    colors = {'EXACT':old.TEAL,'WEAK_PROXY':old.BLUE,'UNKNOWN':old.AMBER,'UNSUPPORTED':old.GRAY}
    fig, ax = plt.subplots(figsize=(15, 10))
    old.heading(fig, 'Baseline augmentation evidence by degradation feature', 'Figure 06 | CV261102')
    y = list(range(len(leaves)))
    ax.set_position([.30,.10,.62,.70])
    ax.scatter([list(colors).index(row['classification']) for row in leaves], y,
               c=[colors[row['classification']] for row in leaves], s=58)
    ax.set_xticks(list(range(len(colors))), labels=list(colors))
    ax.set_yticks(y, labels=[row['leaf'] for row in leaves], fontsize=8)
    ax.set_xlim(-.5,3.5)
    ax.invert_yaxis()
    ax.grid(alpha=.15)
    add(6, fig, ['fig06_leaf_coverage.csv', 'static_gap.json'])
    pairs = selection['pairwise']
    panel(7, 'Sampling workflow', 'Frozen design',
          f"Pairwise: {pairs['achieved']:,}/{pairs['feasible']:,} feasible binary states.  "
          f"Uniform: 100 of {selection['eligible_FM_P']:,} geometry-eligible configurations.  "
          'Active: requires measured responses.', ['selection_manifest.json'])
    measured_rows = summary['pair_runs']
    if measured_rows:
        rows = [{'RunID': r['RunID'], 'scope': r['scope'], 'images': r['images'], 'precision': r['precision'],
                 'resolution': r['resolution'], 'mean_pipeline_ms': r['mean_pipeline_ms'],
                 'AP50_95': r['area']['all']['AP50_95']} for r in measured_rows]
        csv_table(out / 'fig08_pairs.csv', rows)
        fig, ax = plt.subplots(figsize=(15,8.5))
        old.heading(fig, 'Saved pair AP versus latency', 'Figure 08 | measured pairs',
                    'Validation engineering runs and formal test runs are shown separately.')
        ax.set_position([.13,.19,.76,.57])
        for scope in sorted({r['scope'] for r in rows}):
            group = [r for r in rows if r['scope']==scope and r['AP50_95'] is not None]
            ax.scatter([r['mean_pipeline_ms'] for r in group], [r['AP50_95'] for r in group], label=scope)
        ax.legend()
        ax.set(xlabel='Mean image pipeline latency (ms)', ylabel='COCO AP50:95')
        add(8, fig, ['fig08_pairs.csv', 'summary.json'])
    else:
        panel(8, 'Measured AP versus latency', 'NOT_EVALUATED', 'No verified pair results were selected.', ['summary.json'])
    panel(9, 'Held-out inter-model interactions', learned['status'],
          'A heatmap requires a separately verified held-out interaction analysis.', ['learning.json'])
    for number, title in ((10, 'Static and empirical agreement'), (11, 'Failure rate by degradation')):
        panel(number, title, 'NOT_EVALUATED',
              'The comparison requires a validated formal response set and complete detector observations.',
              ['empirical_gap.json'])
    panel(12, 'Targeted augmentation and gap closure', 'NOT_EVALUATED',
          'Matched retraining and a paired before/after evaluation are pending.', ['summary.json'])

    manifest = {}
    with PdfPages(out / 'cv261207_figures.pdf') as pdf:
        for name, (fig, sources) in figures.items():
            files = {}
            for ext in ('png','svg'):
                path = out / f'{name}.{ext}'
                fig.savefig(path, dpi=150)
                files[path.name] = sha256(path)
            pdf.savefig(fig)
            plt.close(fig)
            evidence = {key: sha256(out/key if (out/key).is_file() else audit/key)
                        for key in sources if (out/key).is_file() or (audit/key).is_file()}
            evidence.update({k: v for k, v in {'fm_p.uvl': model['FM_P']['UVL_SHA256'],
                                             'fm_d.uvl': model['FM_D']['UVL_SHA256']}.items() if k in sources})
            manifest[name] = {'files': files, 'source_hashes': evidence}
    save(out / 'figure_evidence.json', {'figures': manifest, 'script_sha256': sha256(__import__('pathlib').Path(__file__)),
                                       'reused_plotting_sha256': sha256(EXPERIMENTS/'cv261123/figures.py'),
                                       'audit_RunID': audit.name})
    gallery = ''.join(f'<section><h2>{name.upper()}</h2><img src="{name}.svg" alt="{name}"></section>' for name in figures)
    (out / 'index.html').write_text(f"""<!doctype html><html lang="en"><meta charset="utf-8">
<title>CV261207 evidence</title><style>body{{font:16px system-ui;max-width:1100px;margin:auto;padding:1rem}}
img{{width:100%}}section{{border-bottom:1px solid #ddd}}</style>
<h1>CV261207 evidence</h1><p>Source audit: {html.escape(audit.name)}</p>
<p><a href="cv261207_figures.pdf">PDF</a> · <a href="figure_evidence.json">Source hashes</a></p>{gallery}</html>""",
                                  encoding='utf-8')
