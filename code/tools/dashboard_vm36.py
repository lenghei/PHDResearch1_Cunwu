"""Read-only protocol and evidence status for VM36."""
from dashboard_autodiscovery import public_runs
from dashboard_data import DashboardStore,DataError
import hashlib
import json


def checked_summary(view):
    data=view.read('summary.json',{})
    if data.get('schema')!='VM36_READ_ONLY_SUMMARY_V1':return None
    metadata=view.read('result.json',{})
    digest=hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if metadata.get('summary_canonical_sha256')!=digest or metadata.get('RunID')!=data.get('RunID'):
        raise DataError('Summary identity or checksum differs.')
    return data


def render(store):
    import pandas as pd
    import streamlit as st
    study='vm36_supervisor_reproduction'
    st.title('VM36 · Supervisor-aligned reproduction')
    st.caption('Saved evidence only. This page cannot approve a protocol or execute experiments.')
    runs=public_runs(store,study)
    if not runs:
        st.info('No reviewed summary is available in this mode. Cloud mode does not read local evidence.')
        return
    preferred=next((i for i,r in enumerate(runs) if r['mode']=='FULL548_R_JPEG' and r['status'] in ('VALIDATED','PARTIAL')),
        next((i for i,r in enumerate(runs) if r['mode']=='SUPERVISOR_ALIGNED_PREPARATION'),0))
    selection=st.selectbox('Saved VM36 run',range(len(runs)),index=preferred,format_func=lambda i:runs[i]['RunID'])
    row=runs[selection]
    reader=DashboardStore(mode='cloud') if row['source']=='published' and store.mode=='local' else store
    try:data=checked_summary(reader.view(study,row['RunID']))
    except (DataError,OSError,ValueError) as error:
        st.error('Saved summary could not be verified: '+str(error));return
    if data is None:
        st.info('This historical preparation run has no final VM36 summary. Select the aligned V2 run.');return
    st.caption(row['RunID']+' · '+data['status'])
    if data.get('measured_study'):
        render_measurements(data)
        return
    with st.container(horizontal=True):
        st.metric('Validation images checked',data['validated_validation_images'])
        st.metric('New measured pairs',data['new_measured_pairs'])
        st.metric('Formal measured pairs',data['formal_measured_pairs'])
        st.metric('New GPU work',str(data['GPU_seconds'])+' s')
    st.warning('Scientific review is pending. The proposed conditions are not completed measurements.')
    st.subheader('Original model alignment')
    st.dataframe(pd.DataFrame(data['model_alignment']),hide_index=True,width='stretch')
    st.caption('Draft Nano/Small parameter attributes are not measured checkpoint parameter counts.')
    st.subheader('Admission and research gates')
    st.dataframe(pd.DataFrame(data['gates']),hide_index=True,width='stretch')
    st.subheader('Student decision register')
    st.dataframe(pd.DataFrame(data['decisions']),hide_index=True,width='stretch')
    st.caption('36 candidate conditions × 548 images = 19,728 planned image-inference operations. No AP is filled in for them.')
    if data.get('CPU_benchmark'):
        st.subheader('Saved CPU equivalence check')
        st.dataframe(pd.DataFrame([data['CPU_benchmark']]),hide_index=True,width='stretch')
        st.caption('Existing 248-image predictions; not a new full-split detector experiment or GPU speed measurement.')
    st.subheader('Evidence identities')
    st.json(data['source_hashes'],expanded=False)


def render_measurements(data):
    import pandas as pd
    import plotly.express as px
    import streamlit as st
    with st.container(horizontal=True):
        st.metric('Validation images per pair',data['validated_validation_images'])
        st.metric('New measured pairs',data['new_measured_pairs'])
        st.metric('Formal measured pairs',data['formal_measured_pairs'])
        elapsed=data.get('worker_wall_seconds')
        st.metric('Worker wall time',f'{elapsed/60:.1f} min' if elapsed is not None else 'Not recorded')
    st.info('Full validation split; development only. JPEG Q75 is not CLEAN. Intervals condition on fixed checkpoints.')
    if data['status']=='PARTIAL':st.warning('The budget ended before all planned pairs were validated. Missing pairs are not estimated.')
    values=pd.DataFrame(data['metrics'])
    if not values.empty:
        values['model']=values['checkpoint'].map({'ck05':'YOLOv8s','ck01':'YOLO26s'})
        values['condition']=values['model']+' · Q'+values['JPEG'].astype(str)
        chart=px.line(values.sort_values('resolution'),x='resolution',y='map50_95',color='condition',markers=True,
            labels={'resolution':'Inference resolution','map50_95':'COCO AP50:95 (0–1)','condition':'Checkpoint / JPEG'})
        st.plotly_chart(chart,width='stretch')
        st.dataframe(values[['model','resolution','JPEG','images','map50','map50_95','AP_CI_low','AP_CI_high','ar50_95']],hide_index=True,width='stretch')
    if data['effects']:
        st.subheader('Matched resolution and JPEG effects')
        st.dataframe(pd.DataFrame(data['effects']),hide_index=True,width='stretch')
        st.caption('200 paired image draws; pointwise 95% intervals. No training-seed or independent-flight inference.')
    st.subheader('Research gates')
    st.dataframe(pd.DataFrame(data['gates']),hide_index=True,width='stretch')
    st.caption(f"Summed forward profiler: {data['forward_profiler_seconds']:.2f} s. This is a component, not total GPU occupancy or deployment latency.")
    st.subheader('Evidence identities')
    st.json(data['source_hashes'],expanded=False)
