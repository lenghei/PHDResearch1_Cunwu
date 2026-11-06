"""Plot verified VM26 cells, contrasts and descriptive rules."""
from pathlib import Path
import html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from common import save, source
from historical_review import table

NAMES = {'ck01':'YOLO26s', 'ck05':'YOLOv8s', 'ck06':'YOLO11s'}
COLORS = {'None':'#2563eb', 'Q85':'#d97706', 'Q50':'#059669'}


def render(out):
    cells = table(out / 'configuration_metrics.csv')
    gains = table(out / 'resolution_differences.csv')
    joint = table(out / 'bootstrap_joint36.csv')
    inputs = [source(out / n) for n in ('configuration_metrics.csv', 'resolution_differences.csv',
                                       'bootstrap_joint36.csv', 'rule_validation.json', 'learned_rules.txt')]
    pictures=[]
    def finish(fig, name):
        fig.savefig(out / (name+'.png'), dpi=180, bbox_inches='tight')
        fig.savefig(out / (name+'.svg'), bbox_inches='tight')
        plt.close(fig)
        pictures.append(name)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2,3, figsize=(12,7), sharex=True, sharey=True)
    for column,(ck,name) in enumerate(NAMES.items()):
        for row,b in enumerate((0,2)):
            ax=axes[row,column]
            for jpeg,color in COLORS.items():
                data=sorted((r for r in cells if r['CheckpointID']==ck and int(r['blur'])==b and r['jpeg']==jpeg),
                            key=lambda r:int(r['resolution']))
                ax.plot([int(r['resolution']) for r in data], [100*float(r['AP50_95']) for r in data],
                        'o-', color=color, label=jpeg)
            ax.set_title(name+' | '+('No added blur' if b==0 else 'Severe added blur'))
            ax.set_xticks((640,960,1280));ax.grid(alpha=.2)
            if column==0:ax.set_ylabel('AP50:95 (%)')
            if row==1:ax.set_xlabel('Inference input size')
    axes[0,0].legend(title='Extra JPEG')
    fig.suptitle('Resolution × Blur × JPEG | 300 shared development images')
    fig.tight_layout();finish(fig,'resolution_blur_jpeg')
    fig, axes=plt.subplots(1,2,figsize=(12,4.6),sharey=True)
    for ax,b in zip(axes,(0,2)):
        for offset,(jpeg,color) in zip((-.22,0,.22),COLORS.items()):
            values=[100*float(next(r['estimate'] for r in gains if r['checkpoint']==ck and int(r['blur'])==b
                         and r['jpeg']==jpeg and r['metric']=='AP50_95' and r['high']=='1280' and r['low']=='640'))
                    for ck in NAMES]
            ax.bar(np.arange(3)+offset,values,.22,color=color,label=jpeg)
        ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(3),NAMES.values())
        ax.set_title('No added blur' if b==0 else 'Severe added blur');ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('AP50:95 difference: 1280 − 640 (pp)');axes[1].legend(title='Extra JPEG')
    fig.tight_layout();finish(fig,'checkpoint_resolution_effects')
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    labels={'JPEG_DID_B0':'JPEG effect on resolution gain\nNo added blur',
            'JPEG_DID_B2':'JPEG effect on resolution gain\nSevere added blur',
            'R_B_J':'Three-way interaction\nSevere minus no blur'}
    for ax,(effect,label) in zip(axes,labels.items()):
        data=[next(r for r in joint if r['checkpoint']==ck and r['size']=='all' and r['metric']=='map50_95'
                   and r['effect']==effect) for ck in NAMES]
        y=np.array([100*float(r['estimate']) for r in data])
        lo=np.array([100*float(r['joint36_low']) for r in data]);hi=np.array([100*float(r['joint36_high']) for r in data])
        ax.errorbar(range(3),y,yerr=[y-lo,hi-y],fmt='o',capsize=5,color='#2563eb')
        ax.axhline(0,color='black',lw=.7);ax.set_xticks(range(3),NAMES.values());ax.set_title(label)
        ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('AP50:95 contrast (pp)')
    fig.suptitle('95% simultaneous intervals: original 36-contrast family, 200 paired image draws',fontsize=11)
    fig.tight_layout();finish(fig,'contrast_confidence_intervals')
    from common import load
    validation=load(out/'rule_validation.json')
    fig,ax=plt.subplots(figsize=(11,4.2));ax.axis('off')
    text=(out/'learned_rules.txt').read_text(encoding='utf-8')
    ax.text(0,1,'Descriptive depth-2 tree: resolution gain',fontsize=14,va='top',weight='bold')
    ax.text(0,.85,text,va='top',family='monospace',fontsize=10)
    ax.text(.62,.8,'18 measured gain rows\n3 checkpoints × 2 blur × 3 JPEG\n300 shared development images\n\nHeld-out validation: NOT EVALUATED\nNo image-disjoint evidence\nNo causal or deployment claim',va='top',fontsize=11)
    fig.tight_layout();finish(fig,'rules_and_validation')
    save(out/'figure_sources.json',{'scope':'DEVELOPMENT_ONLY','inputs':inputs,
          'figures':[source(out/(name+'.png')) for name in pictures]})
    body=''.join(f'<h2>{html.escape(n.replace("_"," "))}</h2><img src="{n}.png" style="max-width:100%">' for n in pictures)
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>VM26 evidence review</title>'
       '<main style="max-width:1200px;margin:30px auto;font-family:Arial"><h1>VM26 historical evidence review</h1>'
       '<p>DEVELOPMENT_ONLY. Historical RunID: vm26_r001. No new GPU execution or formal pairs.</p>'+body+'</main>',encoding='utf-8')
