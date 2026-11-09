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
    preferred=next((i for i,r in enumerate(runs) if r['mode']=='SUPERVISOR_ALIGNED_PREPARATION'),0)
    selection=st.selectbox('Saved VM36 run',range(len(runs)),index=preferred,format_func=lambda i:runs[i]['RunID'])
    row=runs[selection]
    reader=DashboardStore(mode='cloud') if row['source']=='published' and store.mode=='local' else store
    try:data=checked_summary(reader.view(study,row['RunID']))
    except (DataError,OSError,ValueError) as error:
        st.error('Saved summary could not be verified: '+str(error));return
    if data is None:
        st.info('This historical preparation run has no final VM36 summary. Select the aligned V2 run.');return
    st.caption(row['RunID']+' · '+data['status'])
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
