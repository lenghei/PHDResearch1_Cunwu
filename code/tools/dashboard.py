"""Read-only views of saved UAV-GAP runs."""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dashboard_data import DashboardStore, DataError
from dashboard_autodiscovery import (
    discover_studies, render_discovered_study, LEGACY_STUDIES,
)

store = DashboardStore()
STUDIES = discover_studies(store)
MODELS = {'ck01': 'YOLO26s', 'ck05': 'YOLOv8s', 'ck06': 'YOLO11s'}
COLORS = ['#087f8c', '#4463c7', '#a25caf']


def chart(fig):
    fig.update_layout(template='plotly_white', font=dict(family='Arial', color='#243449'),
                      margin=dict(l=20, r=20, t=45, b=20), legend_title_text='', height=380)
    st.plotly_chart(fig, width='stretch')


st.set_page_config(page_title='UAV-GAP · Experiment Studio', page_icon='📊', layout='wide')
st.markdown('''<style>
.stApp{background:#f5f7fb}h1,h2,h3{letter-spacing:-.025em}
[data-testid="stMetric"]{background:white;border:1px solid #e1e7ee;padding:18px;border-radius:12px}
[data-testid="stSidebar"]{background:#edf2f7}
.eyebrow{color:#087f8c;letter-spacing:.15em;font-size:12px;font-weight:700}
.hero{background:linear-gradient(110deg,#102d43,#155463);padding:26px 30px;border-radius:14px;color:white;margin-bottom:24px}
.hero h1{color:white;margin:0;font-size:32px}.hero p{color:#d4e5ec;margin:8px 0 0}
</style>''', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('## UAV-GAP')
    st.caption('RESEARCH RESULTS')
    options = list(STUDIES)
    if not options:
        st.info('No published experiments found.')
        st.stop()
    label = st.selectbox('Experiment', options)
    st.caption(store.mode.title() + ' · Read only · Saved evidence')
    if store.mode == 'cloud':
        st.caption('Published verified summaries. Full predictions and datasets remain local.')
    if st.button('Refresh', width='stretch'):
        st.rerun()
    st.divider()
    st.caption('Saved summaries only. No inference, training or AP recalculation.' if store.mode == 'cloud' else 'Complete local evidence. No inference, training or AP recalculation.')

if STUDIES[label] not in LEGACY_STUDIES:
    render_discovered_study(store, STUDIES[label])
    st.stop()

try:
    records = store.history(STUDIES[label])
except (DataError, OSError) as exc:
    st.error(str(exc))
    st.stop()
st.markdown('<div class="hero"><div class="eyebrow" style="color:#8ce1d4">UAV OBJECT DETECTION</div>'
            '<h1>UAV-GAP Experiments</h1><p>Saved measurements, configurations and reproducibility records.</p></div>', unsafe_allow_html=True)
if not records:
    st.info('No saved runs for this study yet.')
    st.stop()

run_ids = [r['RunID'] for r in records]
lookup = {r['RunID']: r for r in records}
selection = st.selectbox('Run · newest first', run_ids,
                         format_func=lambda key: f"{lookup[key]['started_at'][:19].replace('T', ' ')} UTC · {lookup[key]['mode']} · {lookup[key]['status']}")
try:
    run = store.view(STUDIES[label], selection)
    result = run.read('result.json') or run.read('summary.json', {})
    detailed_summary = run.read('summary.json', {})
except (DataError, OSError) as exc:
    st.error(str(exc))
    st.stop()
record = lookup[selection]
timing = detailed_summary.get('timing', {}) or result.get('timing', {})
elapsed = timing.get('whole_run_wall_seconds') or timing.get('pipeline_seconds') or timing.get('wall_seconds') or result.get('total_seconds')
columns = st.columns(4)
for col, name, value in zip(columns, ['Status', 'Mode', 'Runs', 'Recorded time'],
                           [record['status'], record['mode'], str(len(records)), f'{elapsed:.2f} s' if elapsed is not None else 'Not recorded']):
    col.metric(name, value)
st.caption(f"Recorded start: {record['started_at'][:19].replace('T', ' ')} UTC · Dataset boundaries remain separate.")

overview, comparisons_tab, runs_tab, configurations, interactions, evidence = st.tabs(['Overview', 'Image comparisons', 'Run history', 'Configurations', 'Interactions', 'Evidence'])
data = pd.DataFrame(run.read('measured_grids.json', []))

with overview:
    st.subheader(label.split(' · ', 1)[1])
    if detailed_summary.get('conditions') and detailed_summary.get('paired_comparison'):
        details = detailed_summary
        observed = details['conditions']
        by_name = {c['name']: c for c in observed}
        clean, degraded = by_name.get('CLEAN'), by_name.get('DEGRADED')
        if clean and degraded:
            st.caption('VisDrone2019-DET-val · 548 original images per condition · development validation, not independent test-dev')
            a, b, c, d = st.columns(4)
            a.metric('CLEAN AP50:95', f"{100 * clean['AP50_95']:.2f}%")
            b.metric('DEGRADED AP50:95', f"{100 * degraded['AP50_95']:.2f}%", delta=f"{100 * (degraded['AP50_95'] - clean['AP50_95']):+.2f} pp")
            c.metric('DEGRADED AP50', f"{100 * degraded['AP50']:.2f}%")
            d.metric('DEGRADED AR50:95', f"{100 * degraded['AR50_95']:.2f}%")
            compare = pd.DataFrame([{'Condition': row['name'],
                                     'AP50 (%)': 100 * row['AP50'],
                                     'AP50:95 (%)': 100 * row['AP50_95'],
                                     'AR50:95 (%)': 100 * row['AR50_95'],
                                     'Inference calls (s)': row.get('inference_call_seconds')}
                                    for row in observed])
            st.dataframe(compare, hide_index=True, width='stretch')
            tidy = compare.melt(id_vars='Condition', value_vars=['AP50 (%)', 'AP50:95 (%)', 'AR50:95 (%)'],
                                var_name='Metric', value_name='Value (%)')
            chart(px.bar(tidy, x='Metric', y='Value (%)', color='Condition', barmode='group',
                         color_discrete_sequence=COLORS, title='CLEAN vs DEGRADED · full validation'))
            pair = details.get('paired_comparison', {}).get('rows', [])
            tiny = next((v for v in pair if v.get('size') == 'tiny' and v.get('metric') == 'map50_95'), None)
            if tiny:
                st.caption(f"Tiny AP50:95: {100 * tiny['clean']:.2f}% → {100 * tiny['degraded']:.2f}% "
                           f"({100 * tiny['delta_degraded_minus_clean']:+.2f} percentage points).")
            ds = details.get('dataset', {})
            model = details.get('model', {})
            run_timing = details.get('timing', {})
            m1, m2, m3, m4 = st.columns(4)
            m1.metric('Images / condition', ds.get('images_per_condition', '—'))
            m2.metric('Evaluated GT objects', ds.get('GT_after_protocol', '—'))
            m3.metric('Model parameters', f"{model['parameter_count'] / 1e6:.2f} M" if model.get('parameter_count') else '—')
            seconds = run_timing.get('whole_run_wall_seconds') or run_timing.get('pipeline_seconds')
            m4.metric('Pipeline time', f'{seconds:.1f} s' if seconds is not None else 'Not recorded')
            st.caption('Degradation: ' + ', '.join(f'{key}={value}' for key, value in degraded.get('degradation_parameters', {}).items()))
            with st.expander('Model, dataset and protocol'):
                st.json({'dataset': ds, 'model': model, 'inference': details.get('inference'),
                         'FlamaPy': details.get('flamapy'), 'timing': run_timing}, expanded=False)
            st.info('This paired result covers one sampled compound degradation. It does not establish the static or empirical gap across all valid configurations.')
        else:
            st.warning('Paired report is missing one of the conditions.')
    elif not data.empty:
        c1, c2, c3 = st.columns(3)
        c1.metric('Configuration records', len(data))
        c2.metric('Saved checkpoints', data['checkpoint'].nunique())
        c3.metric('Verified source tables', result.get('sources', '—'))
        st.info('Historical 300-image development evaluation. Grid and JPEG cohorts are shown separately; these records are not new GPU experiments.')
        base = data[data['measurement_kind'] == 'grid'] if 'measurement_kind' in data else data[data['cohort'].str.contains('/dv02/')]
        mean = base.groupby(['checkpoint', 'resolution', 'blur'], as_index=False)['AP50_95'].mean()
        mean['Model'] = mean['checkpoint'].map(MODELS).fillna(mean['checkpoint'])
        mean['Input'] = mean['resolution'].astype(str)
        chart(px.line(mean, x='blur', y='AP50_95', color='Input', facet_col='Model', markers=True,
                      color_discrete_sequence=COLORS, labels={'blur': 'Blur level', 'AP50_95': 'Mean configuration AP50:95'},
                      title='Resolution responses across blur levels'))
        st.caption('Each point is the arithmetic mean of saved configuration APs over illumination/noise contexts. It is not pooled-image AP or an uncertainty interval.')
    elif 'aggregate' in result:
        aggregate = result['aggregate']
        a, b, c = st.columns(3)
        a.metric('AP50', f"{aggregate['map50']:.4f}")
        b.metric('AP50:95', f"{aggregate['map50_95']:.4f}")
        c.metric('Images', result['images'])
        st.caption(f"Checkpoint: {result['checkpoint_id']} · {result['scope']}")
        st.caption('This historical YOLOv8s result is separate from new YOLO26s evaluation.')
    else:
        st.info('This run records CPU checks or metric regeneration. No new detector AP is inferred from its PASS status.')
        if 'UVL_counts' in result:
            a, b = st.columns(2)
            a.metric('Dataset configurations', result['UVL_counts'][0])
            b.metric('Detector configurations', result['UVL_counts'][1])
        st.json(result, expanded=False)

with comparisons_tab:
    st.subheader('CLEAN vs DEGRADED · matched images')
    st.caption('Original inputs appear without boxes. Prediction panels show the same image ID with a display-only confidence threshold; AP uses all saved detections.')
    try:
        samples = run.comparison_sets()
    except (DataError, OSError) as exc:
        st.error('Saved comparison evidence failed validation: ' + str(exc))
        samples = []
    if not samples:
        st.info('No four-panel previews have been published for this run. Generate them locally from saved predictions; no new inference is required.')
    else:
        options = {f"{x['image_id']:04d} · {Path(x['source_file'] or '').name}": x for x in samples}
        selected_image = st.selectbox('Image ID', list(options))
        sample = options[selected_image]
        st.caption(f"Prediction display confidence ≥ {sample['confidence']:.2f}; CLEAN boxes: {sample['clean_boxes']}; DEGRADED boxes: {sample['degraded_boxes']}.")
        left, right = st.columns(2)
        with left:
            st.markdown('**CLEAN input**')
            st.image(sample['pictures']['original.jpg'], width='stretch')
            st.markdown('**CLEAN predictions**')
            st.image(sample['pictures']['clean_predictions.jpg'], width='stretch')
        with right:
            st.markdown('**DEGRADED input**')
            st.image(sample['pictures']['degraded.jpg'], width='stretch')
            st.markdown('**DEGRADED predictions**')
            st.image(sample['pictures']['degraded_predictions.jpg'], width='stretch')
        st.caption('Selected image recipe: ' + ', '.join(f'{key}={val}' for key, val in sample['recipe'].items() if key != 'seed'))
        st.caption('VisDrone2019-DET · AISKYEYE team, Tianjin University. Images are display-only; consult source dataset terms before redistribution.')

with runs_tab:
    st.subheader('Run history')
    status = st.multiselect('Saved statuses', sorted({r['status'] for r in records}), default=sorted({r['status'] for r in records}))
    frame = pd.DataFrame([r for r in records if r['status'] in status])
    st.dataframe(frame, hide_index=True, width='stretch')
    st.caption('Ordered by recorded RunID UTC time; file-time fallbacks are identified in time_source. Saved RUNNING status does not verify process health.')

with configurations:
    if data.empty:
        st.info('This run has no saved configuration grid. Select a VM2 replay to explore configuration responses.')
    else:
        cohort = st.selectbox('Dataset / protocol cohort', sorted(data['cohort'].unique()))
        scope = data[data['cohort'] == cohort]
        model = st.selectbox('Checkpoint', sorted(scope['checkpoint'].unique()), format_func=lambda k: f'{MODELS.get(k,k)} · {k}')
        metric = st.selectbox('Response', ['AP50_95', 'AP50', 'Tiny_AP', 'Small_AP', 'Medium_AP', 'Large_AP', 'AR'])
        selected = scope[scope['checkpoint'] == model]
        controls = st.columns(3)
        for col, factor in zip(controls, ['illumination', 'noise', 'jpeg']):
            levels = sorted(selected[factor].unique())
            value = col.selectbox(factor.title(), levels, key='factor_' + factor)
            selected = selected[selected[factor] == value]
        st.caption('Illumination 0/1/2 = normal/overexposed/underexposed; levels are categorical. Object-size groups use fixed original-image areas.')
        left, right = st.columns([1.2, 1])
        with left:
            chart(px.line(selected.sort_values('blur'), x='blur', y=metric,
                          color=selected.sort_values('blur')['resolution'].astype(str), markers=True,
                          color_discrete_sequence=COLORS, title='Matched configuration responses'))
        with right:
            matrix = selected.pivot(index='blur', columns='resolution', values=metric)
            chart(px.imshow(matrix, text_auto='.3f', color_continuous_scale='Teal', aspect='auto',
                            title='Resolution × blur', labels={'x':'Input resolution','y':'Blur level','color':metric}))
        st.dataframe(selected[['configuration_id','checkpoint','resolution','illumination','blur','noise','jpeg',metric,'images']], hide_index=True, width='stretch')
        st.download_button('Download selected configurations', selected.to_csv(index=False).encode('utf-8'), 'selected_configurations.csv', 'text/csv')

with interactions:
    if not run.exists('interaction_replay.json'):
        st.info('No saved interaction analysis in this run.')
    else:
        effects = run.read('interaction_replay.json')
        tidy = [{'Checkpoint': e.get('checkpoint') or e.get('context',{}).get('checkpoint'),
                 'Relation':e['relation'], 'Contrast':e.get('estimate'), 'Status':e['status'],
                 'Context': json.dumps(e.get('context',{})), 'Cohort':e['cohort']} for e in effects]
        table = pd.DataFrame(tidy)
        summary = table[table['Status']=='PASS'].groupby(['Checkpoint','Relation'],as_index=False)['Contrast'].mean()
        chart(px.bar(summary, x='Contrast', y='Relation', color='Checkpoint', barmode='group',
                     color_discrete_sequence=COLORS, title='Mean matched finite-difference contrasts', orientation='h'))
        st.caption('Descriptive finite-grid contrasts from observed four/eight-corner sets. Different relation contexts and cohorts are not interchangeable; no statistical significance is implied.')
        st.dataframe(table, hide_index=True, width='stretch')
        a,b=st.columns(2)
        a.warning('Static gap: documented training coverage is not supplied.')
        b.info('Empirical gap: task AP/latency thresholds are not set.')

with evidence:
    st.subheader('Saved evidence')
    st.code(run.display_path, language=None)
    choices = run.metadata_names()
    choice=st.selectbox('Metadata file', choices)
    payload = run.bytes(choice)
    st.json(json.loads(payload), expanded=False)
    st.download_button('Download metadata',payload,choice,'application/json')
    images = run.images()
    if images:
        image=st.selectbox('Saved GT / prediction preview',images,format_func=lambda p:p.name)
        st.image(str(image),caption='Saved preview; display threshold does not change evaluation metrics.',width='stretch')
    logs = run.log_text()
    if logs is not None:
        with st.expander('Execution log'):
            st.code(logs, language=None)
