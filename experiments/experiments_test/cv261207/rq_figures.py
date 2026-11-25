"""Render twelve evidence figures with explicit measured and unmeasured scopes."""
import csv
import html
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from common import FMP,FMD,load,save,sha256,source
from historical_review import table,csv_file

NAMES={'ck01':'YOLO26s','ck05':'YOLOv8s','ck06':'YOLO11s','cv261102':'YOLOv8s FT640'}


def render(out, audit, review, data=None):
    data=data or out
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    info=load(data/'rq1_validation.json')
    manifest={};pdf=PdfPages(out/'rq1_rq4_figures.pdf')
    def figure(number,title,rows,sources,scope,units,size=(12,6)):
        stem=f'fig{number:02}'
        csv_file(out/(stem+'.csv'),rows)
        fig,ax=plt.subplots(figsize=size)
        fig.suptitle(title,fontsize=16,weight='bold',y=.97)
        ax.set_position([.12,.23,.82,.61])
        source_map={p.name:sha256(p) for p in sources}
        ids='CV261207 '+out.name+' | vm26_r001' if number in (9,11) else 'CV261207 '+out.name
        fig.text(.04,.10,scope+' | Units: '+units,fontsize=8)
        first=next(iter(source_map.items()))
        fig.text(.04,.065,ids,fontsize=7)
        fig.text(.04,.03,'Source: '+first[0]+' SHA256 '+first[1][:12]+'; full provenance in figure_evidence.json',fontsize=7)
        manifest[stem]={'scope':scope,'units':units,'source_RunIDs':[data.name,audit.name,review.name],
                        'source_hashes':source_map,'CSV_sha256':sha256(out/(stem+'.csv')),
                        'generator_sha256':sha256(Path(__file__))}
        return fig,ax,stem
    def finish(fig,stem):
        for ext in ('png','svg','pdf'):
            p=out/(stem+'.'+ext)
            fig.savefig(p,dpi=170)
            manifest[stem][ext+'_sha256']=sha256(p)
        pdf.savefig(fig);plt.close(fig)

    rows=[{'stage':s,'evidence':e} for s,e in (
        ('FM_P','UVL attributes and constraints'),('FM_D','Training and inference features'),
        ('Sampling','Valid configurations and bound checkpoints'),('Oracle','Saved GT/predictions, COCO AP and cost'),
        ('RQ2','Interpretable development relationships'),('RQ3','Static and measured-subset gaps'),
        ('RQ4','Matched augmentation, pending admission'))]
    fig,ax,stem=figure(1,'Dual variability research workflow',rows,[data/'rq1_validation.json'],
                       'CONCEPTUAL_FRAMEWORK; completion tracked separately','workflow steps')
    ax.axis('off')
    for i,r in enumerate(rows):
        y=.94-i*.145
        ax.text(.12,y,r['stage'],bbox={'boxstyle':'round,pad=.4','fc':'#dbeafe','ec':'#2563eb'},va='center',weight='bold')
        ax.text(.39,y,r['evidence'],va='center')
        if i<len(rows)-1:ax.annotate('',(.17,y-.1),(.17,y-.035),arrowprops={'arrowstyle':'->'})
    finish(fig,stem)

    rows=[];trees=[]
    for name,p in (('FM_P',FMP),('FM_D',FMD)):
        lines=p.read_text(encoding='utf-8').split('features\n',1)[1].split('\nconstraints',1)[0].strip('\n').splitlines()
        cleaned=[line.split(' {')[0].rstrip() for line in lines if line.strip()]
        indent=min(len(x)-len(x.lstrip()) for x in cleaned)
        trees.append('\n'.join(x[indent:] for x in cleaned))
        rows.extend({'model':name,'tree_line':i,'content':line} for i,line in enumerate(cleaned,1))
    fig,ax,stem=figure(2,'Original UVL feature trees',rows,[FMP,FMD],
                      'VALIDATED_LOGIC; not universal implementation support','25 / 34 leaves',size=(15,15))
    ax.axis('off');ax.set_position([.02,.11,.96,.79])
    ax.text(.01,1,'FM_P',va='bottom',weight='bold',fontsize=12)
    ax.text(.52,1,'FM_D',va='bottom',weight='bold',fontsize=12)
    ax.text(.01,.98,trees[0],va='top',family='monospace',fontsize=8)
    ax.text(.52,.98,trees[1],va='top',family='monospace',fontsize=8)
    finish(fig,stem)

    union=load(data/'historical_response_union.json')
    rows=[{'quantity':name,'count':value} for name,value in (
        ('FM_P',info['FM_P']['count']),('FM_D',info['FM_D']['count']),
        ('Theoretical product',info['FM_P']['count']*info['FM_D']['count']),
        ('V1 planned pairs',500),('V2 planned pairs',900),('Historical unique responses',union['unique_responses']))]
    fig,ax,stem=figure(3,'Configuration space and observed scope',rows,
             [data/'rq1_validation.json',data/'cross_architecture_v2.json',data/'historical_response_union.json'],
             'LOGICAL_COUNTS_AND_PLANS; historical responses are development only','configuration count, logarithmic axis')
    ax.barh([r['quantity'] for r in rows],[r['count'] for r in rows],color='#2563eb');ax.set_xscale('log');ax.invert_yaxis()
    ax.set_position([.26,.23,.66,.61]);finish(fig,stem)

    rows=table(data/'configuration_distribution.csv')
    fig,ax,stem=figure(4,'Exact distribution of valid FM_P configurations',rows,
                      [data/'configuration_distribution.csv'],'EXACT_ENUMERATION','configuration count')
    ax.bar([int(r['selected_leaves']) for r in rows],[int(r['configurations']) for r in rows],color='#059669')
    ax.set(xlabel='Selected leaf features',ylabel='Valid configurations');finish(fig,stem)

    rows=table(data/'static_coverage_by_checkpoint.csv')
    fig,ax,stem=figure(5,'Static coverage depends on the evidence policy',rows,
                       [data/'static_coverage_by_checkpoint.csv'],'STATIC_ONLY; not detector failure','covered FM_P configurations out of 46,368')
    ids=list(NAMES)
    for offset,policy,color in zip((-.24,0,.24),('exact_identity','conservative','supervisor_proxy'),('#2563eb','#059669','#d97706')):
        vals=[int(next(r['covered'] for r in rows if r['checkpoint']==c and r['policy']==policy)) for c in ids]
        ax.bar(np.arange(4)+offset,vals,.24,label=policy,color=color)
    ax.set_xticks(range(4),[NAMES[c] for c in ids]);ax.set_ylabel('Covered configurations');ax.legend()
    finish(fig,stem)

    rows=[{'feature':f,**r} for f,r in load(data/'static_baseline.json')['leaves'].items()]
    fig,ax,stem=figure(6,'Training-source coverage by degradation feature',rows,
                       [data/'static_baseline.json'],'CV261102_SOURCE_POLICY; historical activation limits retained','classification, not AP',size=(12,10))
    labels=('EXACT','WEAK_PROXY','UNKNOWN','UNSUPPORTED')
    ax.set_position([.28,.19,.66,.68])
    ax.scatter([labels.index(r['classification']) for r in rows],range(len(rows)),color='#2563eb')
    ax.set_yticks(range(len(rows)),[r['feature'] for r in rows],fontsize=8);ax.invert_yaxis()
    ax.set_xticks(range(4),labels);ax.grid(alpha=.2);finish(fig,stem)

    summary=load(data/'acceptance.json')
    rows=[{'cohort':name,'planned':planned,'formally_evaluated':0} for name,planned in (('V1 frozen',500),('V2 prepared',900))]
    fig,ax,stem=figure(7,'Sampling design and formal execution coverage',rows,
             [data/'cross_architecture_v2.json',audit/'selection_manifest.json',data/'acceptance.json'],
             'PLANNED_ONLY; DATASET_BLOCKED; no formal pairs','pair count')
    ax.bar([r['cohort'] for r in rows],[r['planned'] for r in rows],color='#cbd5e1')
    for i,r in enumerate(rows):ax.text(i,r['planned']/2,f"{r['planned']} planned\n0 formal measured",ha='center')
    ax.set_ylabel('Pairs');finish(fig,stem)

    rows=table(data/'engineering_pairs.csv')
    fig,ax,stem=figure(8,'Saved engineering AP versus end-to-end image cost',rows,
             [data/'engineering_pairs.csv'],'VAL_ENGINEERING_ONLY; image counts differ; not a matched accuracy comparison','AP50:95 (%), ms/image')
    ax.set_position([.12,.23,.60,.61])
    for i,r in enumerate(rows):
        if r['AP50_95']!='':
            ax.scatter(float(r['mean_pipeline_ms']),100*float(r['AP50_95']),s=55,
                       label=f"{i+1}: R{r['resolution']} {r['precision']}, n={r['images']}")
    ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize=8)
    ax.set(xlabel='Decode, degradation, inference and prediction-write mean (ms/image)',ylabel='AP50:95 (%)')
    finish(fig,stem)

    rows=[r for r in table(review/'bootstrap_joint36.csv') if r['effect']=='R_B_J' and r['metric']=='map50_95' and r['size']=='all']
    fig,ax,stem=figure(9,'Resolution × Blur × JPEG development interaction',rows,
             [review/'bootstrap_joint36.csv',review/'provenance.json'],'DEVELOPMENT_ONLY; 200 paired image draws, original joint36 family','AP50:95 contrast (percentage points)')
    y=np.array([100*float(r['estimate']) for r in rows]);lo=np.array([100*float(r['joint36_low']) for r in rows]);hi=np.array([100*float(r['joint36_high']) for r in rows])
    ax.errorbar(range(3),y,yerr=[y-lo,hi-y],fmt='o',capsize=6,color='#2563eb')
    ax.set_xticks(range(3),[NAMES[r['checkpoint']] for r in rows]);ax.axhline(0,color='black',lw=.7)
    ax.set_ylabel('Three-way contrast (pp)');finish(fig,stem)

    rows=[r for r in table(data/'static_empirical_diagnostics.csv') if r['size']=='tiny' and float(r['tau'])==.1]
    fig,ax,stem=figure(10,'Static versus empirical: incomplete outcome diagnostics',rows,
             [data/'static_empirical_diagnostics.csv'],'DEVELOPMENT_ONLY; compatible end-to-end latency is missing','pair counts; tiny AP50:95 threshold 0.10')
    categories=('TP','FP','FN','TN','missing_or_uncertain');bottom=np.zeros(len(rows))
    for key,color in zip(categories,('#2563eb','#d97706','#dc2626','#059669','#cbd5e1')):
        vals=np.array([int(r[key]) for r in rows]);ax.bar(range(len(rows)),vals,bottom=bottom,label=key,color=color);bottom+=vals
    ax.set_xticks(range(len(rows)),[r['policy'] for r in rows]);ax.legend(ncol=5)
    ax.set_ylabel('Pairs; complete cases are selected failures');finish(fig,stem)

    rows=table(review/'configuration_metrics.csv')
    fig,ax,stem=figure(11,'Development robustness across the two measured blur levels',rows,
             [review/'configuration_metrics.csv'],'DEVELOPMENT_ONLY; I0/N0; mean over three measured extra-JPEG settings','AP50:95 (%); no intermediate blur inferred')
    colors=('#2563eb','#d97706','#059669')
    for c,color in zip(('ck01','ck05','ck06'),colors):
        for offset,r in zip((-.15,0,.15),(640,960,1280)):
            vals=[np.mean([float(x['AP50_95'])*100 for x in rows if x['CheckpointID']==c and int(x['resolution'])==r and int(x['blur'])==b]) for b in (0,2)]
            ax.scatter(np.array((0,1))+offset,vals,label=NAMES[c]+f' R{r}',color=color,marker={640:'o',960:'s',1280:'^'}[r])
    ax.set_xticks((0,1),('B0: no added blur','B2: severe added blur'));ax.set_ylabel('AP50:95 (%)');ax.legend(ncol=3,fontsize=8)
    finish(fig,stem)

    rows=[{'metric':m,'status':'NOT_EVALUATED','value':None} for m in ('Clean AP change','Corrupted AP change','Tiny AP change','Gap reduction','Gap reduction CI')]
    fig,ax,stem=figure(12,'Targeted augmentation and gap closure',rows,[data/'rq4_matched_protocol.json'],
                      'NOT_EVALUATED; no matched training or before/after response','no measured gain')
    ax.axis('off');ax.text(.05,.7,'NOT_EVALUATED',fontsize=26,color='#b45309',weight='bold')
    ax.text(.05,.45,'Matched protocol prepared. Training executor and health canary pending.\nRepresentative baseline gap, independent admission and measured budget remain required.',fontsize=12)
    finish(fig,stem)
    pdf.close()
    save(out/'figure_evidence.json',{'figures':manifest,'combined_pdf':source(out/'rq1_rq4_figures.pdf'),
             'numeric_values_are_source_driven':True,'formal_pairs':0})
    gallery=''.join(f'<section><h2>{k}</h2><a href="{k}.pdf">PDF</a> · <a href="{k}.svg">SVG</a> · <a href="{k}.csv">CSV</a><img src="{k}.png" alt="{k}"></section>' for k in manifest)
    link='' if data==out else '../'+data.name+'/'
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>RQ1–RQ4 evidence</title>'
       '<style>body{font:16px Arial;max-width:1200px;margin:25px auto}img{width:100%}section{margin:30px 0}</style>'
       '<h1>RQ1–RQ4 evidence acceptance</h1><p>Independent test: DATASET_BLOCKED. Development and engineering evidence are identified separately.</p>'
       f'<p><a href="{link}acceptance_report.md">Acceptance report</a> · <a href="rq1_rq4_figures.pdf">Figure book</a> · <a href="{link}oral_explanation_en.md">Oral explanation</a></p>'+gallery,encoding='utf-8')


if __name__=='__main__':
    import argparse
    from common import new_run,resolve_run,verify_hashes,seal
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--from-run',required=True)
    args=parser.parse_args();data=resolve_run(args.from_run)
    verify_hashes(data,('acceptance.json','run.json'))
    sources=load(data/'acceptance.json')['source_RunIDs']
    out=new_run('rq_figures')
    render(out,resolve_run(sources['audit']),resolve_run(sources['review']),data)
    save(out/'run.json',{'RunID':out.name,'ExperimentID':'cv261207','status':'PARTIAL',
          'mode':'FIGURE_REGENERATION','source_acceptance_RunID':data.name,'new_forward':0})
    seal(out);print(out)
