"""Versioned supervisor figures from checked evidence; no inference."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import RUNS,load,sha256,source,resolve_run,verify_hashes
from historical_review import table,csv_file
from supervisor_compliance import save

OLD=RUNS/'cv261207_development_figures_20261108T194742_2b59ab1d'


def render(out,pdf_python):
    destination=out/'figures';destination.mkdir()
    verify_hashes(OLD,('figure_evidence.json',))
    manifest={};old=load(OLD/'figure_evidence.json')
    for i in (1,2,3,4,5,6,8,9,11):
        stem=f'fig{i:02}';files={}
        for ext in ('csv','png','svg','pdf'):
            original=OLD/(stem+'.'+ext);copied=destination/original.name
            shutil.copyfile(original,copied)
            if sha256(copied)!=sha256(original):raise ValueError('Figure copy differs.')
            files[ext]=sha256(copied)
        manifest[stem]={'status':'REUSED_VERIFIED','original_RunID':OLD.name,'source':old[stem],'hashes':files}
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    def begin(number,title,rows,evidence):
        keys=list(dict.fromkeys(key for row in rows for key in row))
        rows=[{key:row.get(key) for key in keys} for row in rows]
        stem=f'fig{number:02}';csv_file(destination/(stem+'.csv'),rows)
        fig,ax=plt.subplots(figsize=(12,6));fig.suptitle(title,fontsize=16,weight='bold',y=.95)
        ax.set_position([.10,.25,.84,.55])
        fig.text(.04,.07,'Supervisor readiness audit | no new GPU experiment | no supervisor sign-off inferred',fontsize=8)
        fig.text(.04,.035,out.name+' | source SHA '+sha256(evidence)[:16],fontsize=8)
        manifest[stem]={'status':'SOURCE_DRIVEN','source':source(evidence),'script':source(Path(__file__))}
        return fig,ax,stem
    def finish(fig,stem):
        for ext in ('png','svg','pdf'):
            fig.savefig(destination/(stem+'.'+ext),dpi=160)
        plt.close(fig)
        manifest[stem]['hashes']={ext:sha256(destination/(stem+'.'+ext)) for ext in ('csv','png','svg','pdf')}
    sampling=load(out/'sampling_reconciliation.json')
    rows=[{'method':'Binary-state pairwise','configurations':sampling['pairwise_full_population_configurations'],
          'population':sampling['full_space_population'],'status':'FROZEN_DESIGN','covered_states':sampling['achieved_pair_states'],
          'feasible_states':sampling['feasible_pair_states']},
          {'method':'Uniform: full logical space','configurations':sampling['full_space_uniform_count'],
           'population':sampling['full_space_population'],'status':'FROZEN_DESIGN'},
          {'method':'Uniform: geometry-eligible','configurations':sampling['eligible_uniform_count'],
           'population':sampling['eligible_population'],'status':'FROZEN_FORMAL_MANIFEST'},
          {'method':'Active acquisition','configurations':None,'population':None,'status':'NOT_EVALUATED'}]
    fig,ax,stem=begin(7,'Sampling design: populations and execution are distinct',rows,out/'sampling_reconciliation.json')
    for i,row in enumerate(rows[:3]):
        ax.barh(i,row['configurations'],color=['#146e7a','#426ac4','#9c62ac'][i])
        ax.text(row['configurations']+2,i,f"{row['configurations']} of {row['population']:,}",va='center',fontsize=11)
    ax.text(2,3,'NOT_EVALUATED — no active sample count',va='center',color='#a34f18')
    ax.set_yticks(range(4),[r['method'] for r in rows]);ax.set_ylim(3.5,-.5);ax.set_xlim(0,145)
    ax.set_position([.30,.28,.63,.48]);ax.set_xlabel('Frozen configuration count; not completed formal pairs')
    fig.text(.12,.16,f"Pairwise: {sampling['achieved_pair_states']:,}/{sampling['feasible_pair_states']:,} feasible binary states; {sampling['impossible_pair_states']} impossible states.",fontsize=10)
    finish(fig,stem)
    rows=table(out/'headline_static_diagnostics.csv')
    fig,ax,stem=begin(10,'Static versus measured subset labels at two latency budgets',rows,out/'headline_static_diagnostics.csv')
    labels=[]
    for i,row in enumerate(rows):
        left=0
        for field,color in [('TP','#227a74'),('FP','#d7784a'),('FN','#8c66ac'),('TN','#4b78b8'),('uncertain_or_missing','#b3b8c0')]:
            value=int(row[field]);ax.barh(i,value,left=left,color=color,label=field if i==0 else None)
            if value:ax.text(left+value/2,i,str(value),ha='center',va='center',color='white',weight='bold')
            left+=value
        if left!=12:raise ValueError('Finite-subset diagnostic denominator changed.')
        labels.append(row['policy']+' | '+row['latency_limit_ms']+' ms')
    ax.set_position([.31,.29,.63,.48]);ax.set_yticks(range(len(rows)),labels,fontsize=9);ax.invert_yaxis()
    ax.set_xlim(0,12);ax.set_xlabel('12 measured detector-condition pairs; unknowns retained')
    ax.legend(loc='upper center',bbox_to_anchor=(.45,1.24),ncol=5,fontsize=9)
    fig.text(.06,.17,'Tiny AP threshold = 0.10. Positive label = failure/gap prediction. Cached-input pipeline timing is not admitted deployment latency.',fontsize=8)
    fig.text(.06,.12,'At 50 ms all observed outcomes fail: specificity and balanced accuracy are undefined, not zero.',fontsize=9)
    finish(fig,stem)
    canary=load(RUNS/'cv261207_clean_canary_20261108T194319_0384e775/result.json')
    rows=[{'target':'Matched augmentation gap closure','status':'NOT_EVALUATED','before':None,'after':None},
          {'target':'Clean-control engineering canary','status':canary['status'],
           'exposures':canary['network_image_exposures'],'optimizer_updates':canary['optimizer_steps']}]
    fig,ax,stem=begin(12,'Gap closure after matched retraining: NOT_EVALUATED',rows,out/'verified_counts.json')
    ax.axis('off');ax.text(.03,.78,'No matched before/after efficacy measurements are available.',fontsize=17)
    ax.text(.03,.55,'Targeted arms, ablations and training-seed variability: NOT_EVALUATED',fontsize=12)
    ax.text(.03,.34,f"Separate engineering evidence: {canary['network_image_exposures']} exposures, {canary['optimizer_steps']} optimizer update.",fontsize=12)
    ax.text(.03,.13,'The clean-control canary does not measure augmentation efficacy or gap reduction.',fontsize=12)
    manifest[stem]['canary_source']=source(RUNS/'cv261207_clean_canary_20261108T194319_0384e775/result.json')
    finish(fig,stem)
    manifest['limitations']=['FM_D diagram retains the unchanged v1; missing numeric draft metadata is reported separately.',
        'Historical 300-image and newer 248-image AP are separate cohorts.',
        'Figure 12 is an explicit missing-evidence panel, not a gap-closure result.']
    save(out/'figure_evidence.json',manifest,out)
    script='import sys;from pathlib import Path;from pypdf import PdfWriter;d=Path(sys.argv[1]);w=PdfWriter();[w.append(d/f"fig{i:02}.pdf") for i in range(1,13)];w.write(d/"supervisor_figures.pdf")'
    subprocess.run([str(pdf_python),'-c',script,str(destination)],check=True,timeout=40)
    links=''.join(f'<article><h2>Fig {i:02}</h2><a href="fig{i:02}.pdf"><img src="fig{i:02}.png"></a><p><a href="fig{i:02}.csv">Source CSV</a></p></article>' for i in range(1,13))
    (destination/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Supervisor evidence figures</title><style>body{font:17px Arial;max-width:1200px;margin:30px auto}img{width:100%}article{border-top:1px solid #ddd;margin:30px 0}</style><h1>Source-linked supervisor figures</h1><p>Updated figures 7, 10 and 12. Other figures retain verified source bytes.</p><p><a href="supervisor_figures.pdf">Twelve-page PDF</a></p>'+links,encoding='utf-8')
    print(destination,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--pdf-python',type=Path,required=True)
    args=p.parse_args();render(resolve_run(args.run),args.pdf_python)
