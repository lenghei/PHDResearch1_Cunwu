"""Source-linked figure update for the new development measurements."""
import argparse
import html
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import ROOT,RUNS,load,save,source,sha256,new_run,resolve_run,seal
from historical_review import table,csv_file

OLD=RUNS/'cv261207_rq_figures_20261107T221015_cf1dcb63'
NAMES={'ck01':'YOLO26s','ck05':'YOLOv8s','ck06':'YOLO11s'}


def render(analysis,canary=None):
    python=Path(os.environ.get('UAVGAP_PDF_PYTHON',sys.executable))
    subprocess.run([str(python),'-c','import pypdf'],check=True,timeout=30)
    out=new_run('development_figures');manifest={};plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    old=load(OLD/'figure_evidence.json')
    for i in range(1,8):
        stem=f'fig{i:02}'
        for ext in ('csv','png','svg','pdf'):shutil.copyfile(OLD/(stem+'.'+ext),out/(stem+'.'+ext))
        manifest[stem]={'status':'REUSED_VERIFIED','source_RunID':OLD.name,'original_provenance':old[stem],
             'source_hashes':{ext:sha256(out/(stem+'.'+ext)) for ext in ('csv','png','svg','pdf')}}
    def begin(number,title,rows,source_file):
        stem=f'fig{number:02}';csv_file(out/(stem+'.csv'),rows)
        fig,ax=plt.subplots(figsize=(12,6));fig.suptitle(title,weight='bold',fontsize=16,y=.96)
        ax.set_position([.10,.24,.84,.60])
        fig.text(.04,.10,'DEVELOPMENT_ONLY | 248 image IDs / 247 distinct image byte hashes | native10 | FP32',fontsize=8)
        fig.text(.04,.065,'Pointwise image bootstrap; no independent test or training-seed inference.',fontsize=8)
        fig.text(.04,.03,analysis.name+' | source '+source_file.name+' SHA '+sha256(source_file)[:12],fontsize=7)
        manifest[stem]={'source_RunID':analysis.name,'source':source(source_file),'CSV_SHA256':sha256(out/(stem+'.csv')),
             'generator':source(Path(__file__)),'scope':'DEVELOPMENT_ONLY'}
        return fig,ax,stem
    def finish(fig,stem):
        for ext in ('png','svg','pdf'):
            fig.savefig(out/(stem+'.'+ext),dpi=170);manifest[stem][ext+'_SHA256']=sha256(out/(stem+'.'+ext))
        plt.close(fig)
    allrows=table(analysis/'configuration_metrics.csv');rows=[r for r in allrows if r['size']=='all']
    fig,ax,stem=begin(8,'Measured AP and cached-input pipeline cost',rows,analysis/'configuration_metrics.csv')
    colors={'ck01':'#087f8c','ck05':'#4463c7','ck06':'#a25caf'}
    for r in rows:
        ax.scatter(float(r['mean_pipeline_ms']),100*float(r['map50_95']),color=colors[r['checkpoint']],
                   marker='o' if r['condition']=='Downscale x1.5' else 's',s=70)
        ax.annotate(NAMES[r['checkpoint']]+' '+r['resolution'],(float(r['mean_pipeline_ms']),100*float(r['map50_95'])),
                    xytext=(4,4),textcoords='offset points',fontsize=8)
    ax.set(xlabel='Mean cached-PNG decode + prediction/NMS + raw-write time (ms/image)',ylabel='AP50:95 (%)')
    ax.grid(alpha=.15);ax.text(.01,.02,'Circle: downscale x1.5 | Square: extra JPEG Q25\nGeneration, cold initialization and GPU occupancy are separate.',transform=ax.transAxes,fontsize=8)
    finish(fig,stem)
    gains=[r for r in table(analysis/'resolution_effects.csv') if r['size']=='all' and r['metric']=='map50_95']
    fig,ax,stem=begin(9,'Same-checkpoint resolution effect: 1280 minus 640',gains,analysis/'resolution_effects.csv')
    for i,r in enumerate(gains):
        point,lo,hi=[100*float(r[k]) for k in ('estimate','ci95_low','ci95_high')]
        ax.errorbar(point,i,xerr=[[max(0,point-lo)],[max(0,hi-point)]],fmt='o',color=colors[r['checkpoint']],capsize=3)
    ax.set_yticks(range(len(gains)),[NAMES[r['checkpoint']]+' | '+r['condition'] for r in gains]);ax.invert_yaxis()
    ax.axvline(0,color='gray',ls='--');ax.set_xlabel('AP50:95 difference (percentage points), pointwise 95% paired interval')
    ax.set_position([.32,.24,.62,.60]);finish(fig,stem)
    states=[r for r in table(analysis/'pair_states.csv') if r['size']=='tiny' and float(r['tau'])==.1 and
            float(r['latency_limit_ms'])==100 and r['policy']=='exact_identity']
    fig,ax,stem=begin(10,'Measured subset: Tiny AP >= 0.10 and mean pipeline <= 100 ms',states,analysis/'pair_states.csv')
    palette={'PASS':'#059669','FAIL':'#dc2626','UNCERTAIN':'#d97706','NOT_EVALUATED':'#94a3b8'}
    for i,r in enumerate(states):
        ax.barh(i,1,color=palette[r['state']]);ax.text(.02,i,r['state'],va='center',color='white',weight='bold')
    ax.set_yticks(range(len(states)),[NAMES[r['checkpoint']]+' '+r['resolution']+' | '+r['condition'] for r in states],fontsize=8)
    ax.invert_yaxis();ax.set_xticks([]);ax.set_xlim(0,1);ax.set_position([.38,.24,.56,.60])
    finish(fig,stem)
    size_rows=[r for r in allrows if r['size'] in ('tiny','small','medium','large')]
    fig,ax,stem=begin(11,'Original-coordinate target-size AP in the two frozen conditions',size_rows,analysis/'configuration_metrics.csv')
    bins=['tiny','small','medium','large']
    for checkpoint in NAMES:
        for condition,style in (('Downscale x1.5','-'),('Extra JPEG Q25','--')):
            selected=[r for r in size_rows if r['checkpoint']==checkpoint and r['condition']==condition and int(r['resolution'])==1280]
            if len(selected)==4:ax.plot(bins,[100*float(next(r for r in selected if r['size']==b)['map50_95']) for b in bins],
               style,marker='o',color=colors[checkpoint],label=NAMES[checkpoint]+' | '+condition)
    ax.set(ylabel='AP50:95 (%)',xlabel='Size bin in original image coordinates | input 1280');ax.legend(fontsize=8,ncol=2);finish(fig,stem)
    canary_data=load(canary/'result.json') if canary and (canary/'result.json').exists() else {'status':'NOT_EVALUATED'}
    rows=[{'group':canary_data.get('group','clean_control'),'status':canary_data['status'],
            'exposures':canary_data.get('network_image_exposures'),'updates':canary_data.get('optimizer_steps'),
            'targeted_arms':'NOT_EVALUATED','gap_closure':'NOT_EVALUATED'}]
    fig,ax,stem=begin(12,'RQ4 engineering readiness; efficacy remains unmeasured',rows,analysis/'learning_acceptance.json')
    ax.axis('off');lines=['Clean-control canary: '+canary_data['status'],
         'Actual network exposures: '+str(canary_data.get('network_image_exposures','not measured')),
         'Actual optimizer updates: '+str(canary_data.get('optimizer_steps','not measured')),
         'Targeted augmentation arms: NOT_EVALUATED','Matched before/after gap closure: NOT_EVALUATED',
         'No efficacy curve is available. No long training was started.']
    ax.text(.05,.9,'\n\n'.join(lines),va='top',fontsize=13)
    if canary:manifest[stem]['canary_source']=source(canary/'result.json') if (canary/'result.json').exists() else source(canary/'failure.json')
    finish(fig,stem)
    pdfs=[str(out/f'fig{i:02}.pdf') for i in range(1,13)];save(out/'pdf_inputs.json',pdfs)
    script='import sys,json;from pypdf import PdfWriter;w=PdfWriter();[w.append(p) for p in json.load(open(sys.argv[1]))];w.write(sys.argv[2])'
    subprocess.run([str(python),'-c',script,str(out/'pdf_inputs.json'),str(out/'rq1_rq4_figures.pdf')],check=True,timeout=60)
    save(out/'figure_evidence.json',manifest)
    links=''.join(f'<article><h2>Figure {i:02}</h2><a href="fig{i:02}.pdf"><img src="fig{i:02}.png"></a><p><a href="fig{i:02}.csv">Source table</a></p></article>' for i in range(1,13))
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>CV261207 development evidence</title><style>body{font:17px Arial;max-width:1200px;margin:30px auto}img{width:100%}article{border-top:1px solid #ddd;margin:35px 0}</style><h1>RQ1–RQ4 evidence update</h1><p>Figures 1–7 reuse their verified historical sources. Figures 8–12 reflect the new development branch.</p><a href="rq1_rq4_figures.pdf">Complete figure book</a>'+links,encoding='utf-8')
    save(out/'run.json',{'RunID':out.name,'ExperimentID':'cv261207','mode':'DEVELOPMENT_FIGURES','status':'PASS',
           'source_analysis':analysis.name,'source_canary':canary.name if canary else None,'figures':12})
    seal(out);print(out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--analysis',required=True);parser.add_argument('--canary')
    args=parser.parse_args();render(resolve_run(args.analysis),resolve_run(args.canary) if args.canary else None)
