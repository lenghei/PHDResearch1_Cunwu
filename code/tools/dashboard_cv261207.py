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
    if summary.get('rq_acceptance'):
        render_acceptance(st,pd,px,chart,summary['rq_acceptance'])
        return
    if summary.get('historical_vm26'):
        render_history(st, pd, px, chart, summary['historical_vm26'])
        return
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


def render_history(st, pd, px, chart, history):
    st.warning('Historical development evidence only. Independent test-dev admission remains blocked.')
    with st.container(horizontal=True):
        st.metric('Measured historical configurations', history['configurations'])
        st.metric('Shared development images', history['images'])
        st.metric('Checkpoints', len(history['checkpoints']))
        st.metric('Formal pairs', 0)
    st.caption('VM26 RunID: vm26_r001 · native10 · pr01 · FP32 · no new GPU execution')
    cells=pd.DataFrame(history['cells'])
    cells['AP50:95 (%)']=cells['AP50_95']*100
    cells['Blur']=cells['blur'].map({0:'No added blur',2:'Severe added blur'})
    chart(px.line(cells.sort_values('resolution'), x='resolution', y='AP50:95 (%)',
                  color='jpeg', line_dash='Blur', facet_col='model', markers=True,
                  labels={'jpeg':'Extra JPEG','resolution':'Inference input size'}))
    gains=pd.DataFrame(history['gains'])
    gains['AP difference (pp)']=gains['estimate']*100
    gains['Blur']=gains['blur'].map({0:'No added blur',2:'Severe added blur'})
    chart(px.bar(gains, x='model', y='AP difference (pp)', color='jpeg', facet_col='Blur',
                 barmode='group', title='AP50:95 difference: 1280 minus 640'))
    intervals=pd.DataFrame(history['headline_intervals'])
    intervals['Contrast (pp)']=intervals['estimate']*100
    intervals['upper_error']=(intervals['joint36_high']-intervals['estimate'])*100
    intervals['lower_error']=(intervals['estimate']-intervals['joint36_low'])*100
    chart(px.scatter(intervals,x='model',y='Contrast (pp)',facet_col='effect',
                     error_y='upper_error',error_y_minus='lower_error',
                     title='Original 36-contrast family: 95% simultaneous image-bootstrap intervals'))
    st.caption('200 paired image draws. No flight-cluster or training-seed uncertainty. Q85 and 960 intervals were not evaluated.')
    st.subheader('Descriptive learned rules')
    st.code(history['rules'])
    st.info('Held-out image-and-configuration validation: NOT_EVALUATED. All aggregate responses share the same 300 development images.')
    st.subheader('Contribution evidence')
    st.dataframe(pd.DataFrame(history['contributions']),hide_index=True,width='stretch')
    st.subheader('Source hashes')
    st.json(history['source_hashes'],expanded=False)


def render_acceptance(st,pd,px,chart,summary):
    st.warning('Independent test-dev admission remains blocked. Formal measured pairs: 0.')
    with st.container(horizontal=True):
        st.metric('Valid FM_P',f"{summary['FM_P']['count']:,}")
        st.metric('Valid FM_D',f"{summary['FM_D']['count']:,}")
        st.metric('Historical development responses',summary['historical_unique_responses'])
        st.metric('Shared development images',summary['shared_development_images'])
    st.caption('FM_D: SAT plus exact factorization and structural cross-check; no independent BDD count.')
    st.subheader('RQ1–RQ4 acceptance')
    st.dataframe(pd.DataFrame(summary['research_questions'])[['RQ','status','result','blocker']],
                 hide_index=True,width='stretch')
    st.subheader('Static coverage policies')
    static=pd.DataFrame(summary['static'])
    static['covered']=pd.to_numeric(static['covered'])
    chart(px.bar(static,x='checkpoint',y='covered',color='policy',barmode='group',
                 labels={'covered':'Covered FM_P configurations'}))
    st.caption('Coverage is a source-evidence policy, not measured detector robustness.')
    st.subheader('Measured-subset gap diagnostics')
    gap=pd.DataFrame(summary['subset_gaps'])
    gap=gap[(gap['size']=='tiny') & (pd.to_numeric(gap['tau'])==.1)]
    st.dataframe(gap[['blur','jpeg','tested_detectors','subset_state']],hide_index=True,width='stretch')
    st.info('Four purposive contexts and six measured endpoints per context. Missing latency prevents PASS labels. No population or full-space gap estimate.')
    st.subheader('Learning validation')
    st.info('ML_INSUFFICIENT_EVIDENCE: all responses share the same image population. Held-out scores remain NOT_EVALUATED.')
    st.dataframe(pd.DataFrame(summary['descriptive_model_fits']),hide_index=True,width='stretch')
    st.caption('These are in-sample descriptive fits. Fitting seeds do not measure detector training-seed variability.')
    st.subheader('Detector-feature compatibility')
    st.dataframe(pd.DataFrame(summary['feature_compatibility']),hide_index=True,width='stretch')
    st.caption(f"V1: {summary['original_frozen_pairs']} planned. V2: {summary['v2_planned_pairs']} separately planned, zero executed.")
    st.info('Targeted augmentation and gap closure: NOT_EVALUATED. No YOLO training was started.')
    st.subheader('Execution cost')
    st.write(f"New bounded GPU-demo worker wall time: {summary['new_GPU_worker_seconds']:.3f} s. This is not GPU occupancy or full-pair throughput.")
    st.subheader('Evidence hashes')
    st.json(summary['source_hashes'],expanded=False)
