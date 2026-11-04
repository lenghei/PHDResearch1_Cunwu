"""Read-only CV261207 summary and pair views."""
from dashboard_autodiscovery import public_runs
from dashboard_data import DashboardStore, DataError


def render(store, chart):
    import pandas as pd
    import plotly.express as px
    import streamlit as st

    study = 'cv261207'
    st.title('CV261207 · Dual variability')
    runs = public_runs(store, study)
    if not runs:
        st.info('No verified summaries available.')
        return
    selected = st.selectbox('Saved run', list(range(len(runs))),
                            format_func=lambda i: runs[i]['RunID'])
    run = runs[selected]
    reader = DashboardStore(mode='cloud') if run['source'] == 'published' and store.mode == 'local' else store
    try:
        view = reader.view(study, run['RunID'])
        summary = view.read('summary.json', {}) or {}
        if not summary.get('FM_P'):
            result = view.read('result.json', {}) or {}
            if result.get('FM_P'):
                summary = result
    except (OSError, DataError, ValueError, KeyError) as exc:
        st.warning(f'Verification failed: {exc}')
        return
    st.caption(f"{run['RunID']} · {run['source']} · {run['status']}")
    if not summary.get('FM_P'):
        st.info('This run has no review summary. Select a saved evidence summary.')
        return
    cards = st.columns(4)
    cards[0].metric('Valid FM_P', f"{summary['FM_P']['count']:,}")
    cards[1].metric('Valid FM_D', f"{summary['FM_D']['valid_count']:,}")
    cards[2].metric('Frozen pair plan', summary['planned_pairs'])
    cards[3].metric('Formal pair runs', sum(p['scope']=='TEST_DEV' for p in summary.get('pair_runs', [])))
    st.write('Formal evaluation: ' + summary['formal_evaluation'])
    admission = summary.get('admission')
    if admission:
        st.subheader('Checkpoint admission')
        st.caption('Archive integrity and checkpoint fitness are separate from independent test admission.')
        st.dataframe(pd.DataFrame(admission['candidates'])[
            ['id', 'model', 'candidate_status', 'training_membership', 'checkpoint_selection',
             'historical_testdev_evaluation_or_tuning', 'formal_admission']],
            hide_index=True, width='stretch')
        overlap = pd.DataFrame.from_dict(admission['archive_overlap'], orient='index')
        st.dataframe(overlap, width='stretch')
        st.info('The frozen 500-pair plan binds the CV261102 YOLOv8s checkpoint.')
    st.subheader('Sampling')
    st.json({'uniform': summary['sampling'], 'pairwise': summary['pairwise_coverage']}, expanded=False)
    st.subheader('Static coverage')
    frame = pd.DataFrame(list(summary['static'].items()), columns=['Definition','Covered'])
    chart(px.bar(frame, x='Definition', y='Covered', title='Configurations covered by each rule set'))
    st.subheader('Saved pair evidence')
    rows = summary.get('pair_runs', [])
    if rows:
        frame = pd.DataFrame([{**r, 'AP50:95': r['area']['all']['AP50_95']} for r in rows])
        chart(px.scatter(frame, x='mean_pipeline_ms', y='AP50:95', color='scope',
                         symbol='precision', hover_data=['resolution','RunID']))
        st.dataframe(frame[['RunID','scope','resolution','precision','AP50:95','images']],
                     hide_index=True, width='stretch')
    else:
        st.info('No measured pairs in this snapshot.')
    st.subheader('Evidence hashes')
    st.json(summary['source_hashes'], expanded=False)
    st.caption('Empirical gap and held-out rules are available only after formal evaluation.')
