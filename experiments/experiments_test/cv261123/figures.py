"""Reproducible, source-driven research figures for the supervisor's two-model plan.

Only saved model counts, exact FM_P enumerations and recorded static results
are plotted. No detector AP, learned constraints or empirical gap is invented.
"""
from __future__ import annotations
import csv
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.backends.backend_pdf import PdfPages

NAVY = '#142d44'
TEAL = '#087e83'
BLUE = '#3869b7'
AMBER = '#d99b31'
RED = '#b55454'
GREEN = '#468b6c'
GRAY = '#566b7e'
PALE = '#edf3f7'
BG = '#fbfcfe'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.labelcolor': NAVY, 'text.color': NAVY, 'axes.edgecolor': '#cbd6de',
    'xtick.color': GRAY, 'ytick.color': GRAY, 'svg.fonttype': 'none',
    'savefig.facecolor': 'white', 'figure.facecolor': BG})


def heading(fig, title, kicker, subtitle=None):
    fig.text(.045, .955, kicker.upper(), fontsize=9, fontweight='bold', color=TEAL, va='top')
    fig.text(.045, .915, title, fontsize=24, fontweight='bold', color=NAVY, va='top')
    if subtitle:
        fig.text(.045, .858, subtitle, fontsize=10, color=GRAY, va='top')


def foot(fig, note, y=.035):
    fig.text(.05, y, note, color=GRAY, fontsize=8, va='center')


def rounded(ax, x, y, w, h, text, *, fc='white', ec='#d5dfe7', size=12,
            weight='normal', textcolor=NAVY, alpha=1, ls='solid'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.009,rounding_size=0.015',
        linewidth=1.4, facecolor=fc, edgecolor=ec, alpha=alpha, linestyle=ls))
    ax.text(x+w/2, y+h/2, text, va='center', ha='center', fontsize=size,
            color=textcolor, fontweight=weight, linespacing=1.5)


def arrow(ax, a, b, *, color='#94a9ba', style='-|>', lw=1.9, arc=None):
    kw = {'arrowstyle': style, 'mutation_scale': 15, 'color': color, 'linewidth': lw}
    if arc is not None:
        kw['connectionstyle'] = f'arc3,rad={arc}'
    ax.add_patch(FancyArrowPatch(a, b, **kw))


def page01_two_trees():
    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.set(xlim=(0, 1), ylim=(0, 1)); ax.axis('off')
    heading(fig, 'Two variability spaces, two gaps, one loop', 'Figure 01 | conceptual architecture',
            'Redrawn from the advisor sketch and the 22 October research plan. Not an empirical result.')
    rounded(ax, .045, .29, .255, .50, 'FM_D  |  Detector variability\n\nArchitecture + scale\nTraining weights + augmentation\nInput resolution + precision\nInference strategy + postprocess',
            fc='#e8f0f9', ec=BLUE, size=12, weight='medium')
    rounded(ax, .700, .29, .255, .50, 'FM_P  |  Dataset variability\n\nIllumination + motion blur\nNoise + atmosphere\nCompression + altitude\nRolling shutter',
            fc='#e6f5f3', ec=TEAL, size=12, weight='medium')
    rounded(ax, .385, .640, .23, .125, 'Valid configurations\n+ constraints', fc='white', ec='#7a97a7', size=12)
    rounded(ax, .385, .473, .23, .125, 'Uniform + pairwise sampling\nSelected (p, d) pairs', fc='white', ec='#7a97a7', size=12)
    rounded(ax, .385, .299, .23, .125, 'YOLO inference + COCO AP\nLatency / bootstrap intervals', fc='#fff6e8', ec=AMBER, size=11)
    arrow(ax, (.301, .70), (.382, .70), color=BLUE)
    arrow(ax, (.697, .70), (.618, .70), color=TEAL)
    arrow(ax, (.5,.64),(.5,.60));arrow(ax, (.5,.47),(.5,.426))
    rounded(ax,.155,.080,.294,.11,'Static gap\nAugmentation coverage (proxy)', fc='#f1f4f7', ec=GRAY, size=11)
    rounded(ax,.551,.080,.294,.11,'Empirical gap + learned rules\nRequires measured pair AP',fc='#fff6e8', ec=AMBER,size=11)
    arrow(ax,(.470,.295),(.32,.193),color=GRAY)
    arrow(ax,(.53,.295),(.699,.193),color=AMBER)
    arrow(ax,(.185,.199),(.185,.280),color=TEAL,lw=1.6)
    ax.text(.218,.244,'update FM_D',ha='left',va='center',fontsize=8.5,color=TEAL)
    foot(fig, 'Research design: source = supervisor plan, pp. 1-4, 10-12; blue-paper sketch (26 October 2026).')
    return fig


def page02_feature_models(fmp_model, fmd_categories, evidence):
    fig, ax = plt.subplots(figsize=(15, 8.5))
    heading(fig, 'FM_P and FM_D: actual current feature groups', 'Figure 02 | model structure',
            'Group names from the two UVL implementations; counts verified against the saved model audit.')
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    rounded(ax,.058,.760,.408,.075,f"FM_P  |  {len(fmp_model.ALL_LEAVES)} leaves  |  5 constraints",fc='#e6f5f3',ec=TEAL,weight='bold')
    rounded(ax,.533,.760,.408,.075,f"FM_D  |  {evidence['fmd_leaves']} leaves  |  4 constraints",fc='#e8f0f9',ec=BLUE,weight='bold')
    p_groups=list(fmp_model.GROUPS.items())
    d_groups=list(fmd_categories.items())
    p_y=.692
    for index,(group,values) in enumerate(p_groups):
        y=p_y-index*.054
        label=f"{group.title():<14}    {len(values)} leaf levels"
        rounded(ax,.076,y,.369,.045,label,fc='#f5faf9',ec='#dbece8',size=9.8)
    for index,(group,values) in enumerate(d_groups):
        y=p_y-index*.054
        label=f"{group:<18}    {len(values)} options"
        rounded(ax,.553,y,.369,.045,label,fc='#f5f8fc',ec='#dbe4ee',size=9.8)
    foot(fig, 'FM_P includes optional composition. FM_D contains both training-time and inference-time choices.')
    return fig


def page03_cardinality(evidence):
    fig, ax=plt.subplots(figsize=(15,8.5))
    heading(fig,'The combined design space is enormous','Figure 03 | exact model cardinalities',
            'The Cartesian product is theoretical, not a count of configurations tested on a GPU.')
    categories=['FM_P  |  image degradations','FM_D  |  detector configurations','FM_P x FM_D  |  possible pairs']
    counts=[evidence['fmp_count'],evidence['fmd_count'],evidence['fmp_count']*evidence['fmd_count']]
    bars=ax.barh(range(3),[math.log10(x) for x in counts],height=.51,color=[TEAL,BLUE,NAVY])
    ax.set_yticks(range(3),labels=categories,fontsize=13);ax.invert_yaxis()
    ax.set_xlim(0,13);ax.set_xticks([0,2,4,6,8,10,12],labels=['10^0','10^2','10^4','10^6','10^8','10^10','10^12'])
    ax.set_xlabel('Log10(number of configurations or configuration pairs)',fontsize=11)
    ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
    for b,n in zip(bars,counts):
        ax.text(b.get_width()+.15,b.get_y()+b.get_height()/2,f'{n:,}',va='center',fontsize=15,fontweight='bold')
    ax.set_position([.29,.25,.61,.48])
    fig.text(.29,.15, 'Practical plan: finite checkpoints + sampled inference settings + sampled degradations.',
             fontsize=11,color=GRAY)
    foot(fig,'Source: FM_P source enumeration and FM_D model_count.json. No AP values used.')
    return fig


def page04_distribution(hist):
    fig,ax=plt.subplots(figsize=(15,8.5))
    heading(fig,'Distribution of valid degradations by selected leaf count',
            'Figure 04 | directly enumerated FM_P',
            'The advisor sketched a bell curve. This is the measured discrete distribution, not a fitted Gaussian.')
    x=sorted(int(k) for k in hist)
    y=[hist.get(k,hist.get(str(k))) for k in x]
    bars=ax.bar(x,y,color=[TEAL if a in (6,7) else '#87b1b5' for a in x],width=.76)
    ax.set_xlabel('Number of selected FM_P leaf features',fontsize=12)
    ax.set_ylabel('Number of valid configurations',fontsize=12)
    ax.set_xticks(x)
    ax.grid(axis='y',alpha=.17);ax.set_axisbelow(True)
    for b,v in zip(bars,y):ax.text(b.get_x()+b.get_width()/2,b.get_height()+130,f'{v:,}',ha='center',fontsize=11)
    ax.set_ylim(0,max(y)*1.19);ax.set_position([.14,.20,.78,.58])
    fig.text(.15,.12,f'Full valid space: {sum(y):,}  |  Uniform sampling must be over configurations, not counts of selected features.',fontsize=10,color=GRAY)
    foot(fig,'Computed from cv261109/model.py combinations() with the checked FM_P constraints.')
    return fig


def page05_static_coverage(e):
    fig, axes=plt.subplots(1,2,figsize=(15,8.5),gridspec_kw={'width_ratios':[1.25,1]})
    heading(fig,'Static gap: strict identity vs provisional augmentation proxy',
            'Figure 05 | actual CV261116 static audit',
            'Provisional proxy coverage is not physical equivalence or observed AP-based detector failure.')
    proxy=e['proxy'];total=e['fmp_count'];strict=e['strict']
    ax=axes[0]
    labels=['Supervisor proxy\n(weakly equivalent)','Exact identity\n(clean only)']
    fractions=[proxy/total*100,strict/total*100]
    for i,(lab,num,percentage) in enumerate(zip(labels,[proxy,strict],fractions)):
        ax.barh(i,100,color='#f2d8d8',height=.44)
        ax.barh(i,percentage,color=GREEN,height=.44)
        ax.text(52,i,f'{total-num:,} not covered',ha='center',va='center',fontsize=12,color=NAVY)
        ax.text(101,i,f'{num}/{total:,}\n({percentage:.5f}%)',fontsize=12,va='center',fontweight='bold')
    ax.set_xlim(0,150);ax.set_yticks([0,1],labels=labels);ax.invert_yaxis()
    ax.set_xlabel('Percentage of FM_P configurations (%)')
    ax.set_xticks([0,25,50,75,100]);ax.grid(axis='x',alpha=.16)
    axes[1].axis('off')
    axes[1].text(.05,.79,'Key interpretation',fontsize=19,fontweight='bold',color=NAVY)
    axes[1].text(.05,.61,f'{100*(total-proxy)/total:.5f}%',fontsize=35,fontweight='bold',color=RED)
    axes[1].text(.05,.53,'outside the provisional training proxy',fontsize=12,color=GRAY)
    axes[1].text(.05,.35,'Not yet empirical gap',fontsize=16,color=NAVY,fontweight='bold')
    axes[1].text(.05,.23,'AP, latency budgets and bootstrap intervals\nare required before claiming detector failure.',fontsize=12,color=GRAY)
    for ax in axes:ax.set_position([.12,.26,.45,.44] if ax==axes[0] else [.64,.22,.31,.52])
    foot(fig,'Sources: CV261116 static_gap.json and coverage_rules.json; reference protocol remains provisional.')
    return fig


def page06_leaf_coverage(rows):
    fig,ax=plt.subplots(figsize=(15,8.5))
    heading(fig,'Which degradation leaves appear in proxy-covered configurations?',
            'Figure 06 | feature-level static coverage',
            'Counts conditioned on a feature being selected. They are not detection accuracies.')
    rows=sorted(rows,key=lambda r:(int(r['covered_when_present']),r['feature']),reverse=True)
    vals=[int(r['covered_when_present']) for r in rows]
    labels=[r['feature'] for r in rows]
    bars=ax.barh(range(len(rows)), vals, color=[GREEN if a else '#d6e0e6' for a in vals],height=.70)
    ax.set_yticks(range(len(rows)),labels=labels,fontsize=8.8);ax.invert_yaxis()
    ax.set_xlim(0,max(vals)+3);ax.set_xlabel('Proxy-covered configurations containing this leaf')
    ax.set_xticks(range(0,max(vals)+2,2));ax.grid(axis='x',alpha=.13)
    for b,v in zip(bars,vals):
        if v>0:ax.text(v+.18,b.get_y()+b.get_height()/2,str(v),va='center',fontsize=8.8)
    ax.set_position([.22,.12,.71,.66])
    foot(fig,'Source: CV261116 coverage_by_feature.csv. Each covered configuration contributes to several leaf counts.')
    return fig


def page07_sampling(plan):
    fig,ax=plt.subplots(figsize=(15,8.5))
    heading(fig,'A traceable sampling strategy before expensive GPU evaluation',
            'Figure 07 | proposed experiment design',
            'Design only: selection counts are generated, but AP and empirical-gap results are NOT MEASURED.')
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    rounded(ax,.08,.66,.245,.12, f"Pairwise design\n{plan['pairwise_budget']} valid FM_P configs",fc='#e6f5f3',ec=TEAL,size=12,weight='medium')
    rounded(ax,.375,.66,.245,.12,f"Uniform random\n{plan['uniform_budget']} valid FM_P configs",fc='#e8f0f9',ec=BLUE,size=12,weight='medium')
    rounded(ax,.67,.66,.245,.12,'Active boundary sampling\n20 slots  |  NOT SELECTED',fc='#f6f6f6',ec=GRAY,size=12,ls='dashed')
    rounded(ax,.235,.405,.53,.125,'Initial 100 distinct FM_P configurations\nx 5 detector slots = 500 PLANNED PAIRS',fc='#fff5e5',ec=AMBER,size=14,weight='bold')
    arrow(ax,(.2,.65),(.38,.537));arrow(ax,(.496,.65),(.535,.537));
    rounded(ax,.275,.174,.45,.14,'Bind real checkpoints -> YOLO on test-dev -> save predictions\nAP + latency + 95% image bootstrap -> learned rules',fc='#f4f7fa',ec='#a9bcc9',size=11)
    arrow(ax,(.5,.398),(.5,.316))
    ax.text(.500,.10,'No learning/accuracy claims are made until actual (p,d) evaluations exist.',ha='center',fontsize=11,color=RED)
    foot(fig,f"Pairwise semantics: {plan['feasible_pairwise_states']} reachable binary leaf-pair states, covered by {plan['greedy_cover_count']} greedy configurations before padding.")
    return fig


def page08_status(e):
    fig,ax=plt.subplots(figsize=(15,8.5))
    heading(fig,'Research questions: evidence and remaining work','Figure 08 | supervision checklist',
            'Based on saved local checks; model-only or provisional results must not be treated as detector findings.')
    ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    rows=[
      ('RQ1','Represent FM_P and FM_D','UVL + SAT + audited counts','MODEL CHECKED',GREEN),
      ('RQ2','Learn cross-model failure relations','Real (p,d) AP + held-out ML','NOT MEASURED',AMBER),
      ('RQ3','Count static and empirical gaps','Static proxy; empirical needs oracle','STATIC PROXY',BLUE),
      ('RQ4','Close the gap with retraining','Matched seeds + held-out comparison','NOT MEASURED',AMBER),
    ]
    for i,(rq,title,ev,status,col) in enumerate(rows):
        y=.67-i*.157
        rounded(ax,.08,y,.105,.098,rq,fc=PALE,ec='#d5e3eb',size=15,weight='bold')
        rounded(ax,.209,y,.53,.098,title+'\n'+ev,fc='white',ec='#d7e0e7',size=10)
        rounded(ax,.762,y,.176,.098,status,fc='white',ec=col,size=10,weight='bold',textcolor=col)
    foot(fig,'Sources: supervisor plan pp. 4, 10-12, 16; CV261109 and CV261116 saved evidence.')
    return fig


def page09_heatmap_template():
    fig,ax=plt.subplots(figsize=(15,8.5))
    heading(fig,'Inter-model relationships: the future learned result',
            'Figure 09 | unmeasured matrix template',
            'Each cell requires real (degradation, detector) outcomes; all entries remain intentionally blank.')
    detector=['Infer640 FP32','Infer960 FP32','Infer1280 FP32','Infer1280 FP16']
    degrade=['Illumination','Motion blur','Noise','Atmosphere','Compression','Altitude']
    for i in range(len(degrade)):
        for j in range(len(detector)):
            ax.add_patch(Rectangle((j-.5,i-.5),1,1,facecolor='#edf2f6',edgecolor='white',linewidth=5))
            ax.text(j,i,'NO DATA',va='center',ha='center',fontsize=9,color='#8998a8')
    ax.set_xlim(-.5,len(detector)-.5);ax.set_ylim(-.5,len(degrade)-.5)
    ax.set_xticks(range(len(detector)),labels=detector,fontsize=11);ax.set_yticks(range(len(degrade)),labels=degrade,fontsize=11)
    ax.xaxis.tick_top();ax.invert_yaxis();ax.tick_params(length=0)
    for s in ax.spines.values():s.set_visible(False)
    ax.set_position([.20,.18,.69,.55])
    foot(fig,'Replace only when real AP / bootstrap oracle labels are available. This image is not a result.')
    return fig


def read_uvl_top_groups(path: Path) -> dict:
    """Read named direct detector groups and their leaf-level option names."""
    groups={};current=None
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        val=line.strip()
        if not val or val.startswith('//') or val.startswith('namespace') or val=='features' or val=='constraints':
            if val=='constraints':break
            continue
        indent=len(line)-len(line.lstrip(' '))
        name=val.split(' ',1)[0]
        if indent==12 and name not in ('mandatory','optional','alternative','or'):
            current=name;groups[current]=[]
        elif current and indent>=20 and name not in ('mandatory','optional','alternative','or'):
            if name not in groups[current]:groups[current].append(name)
    if len(groups)<7:
        raise ValueError('Cannot extract detector feature families from FM_D UVL.')
    for name, choices in groups.items():
        if not choices:
            groups[name] = [name]
    return groups


def draw_all(target, fmp_model, fmd_uvl, evidence, feature_rows, plan):
    """Render SVG, PNG and multipage PDF from the same source-driven figures."""
    target=Path(target);target.mkdir(parents=True,exist_ok=True)
    hist=plan['selected_feature_histogram']
    figures=[
      ('fig01_two_variability_spaces',page01_two_trees()),
      ('fig02_feature_model_groups',page02_feature_models(fmp_model,read_uvl_top_groups(fmd_uvl),evidence)),
      ('fig03_configuration_space',page03_cardinality(evidence)),
      ('fig04_exact_fmp_distribution',page04_distribution(hist)),
      ('fig05_static_gap',page05_static_coverage(evidence)),
      ('fig06_feature_level_coverage',page06_leaf_coverage(feature_rows)),
      ('fig07_sampling_pipeline',page07_sampling(plan)),
      ('fig08_rq_evidence_status',page08_status(evidence)),
      ('fig09_interaction_data_required',page09_heatmap_template()),
    ]
    with PdfPages(target/'cv261123_supervisor_figures.pdf') as pdf:
        for name,fig in figures:
            fig.savefig(target/(name+'.svg'),format='svg',bbox_inches='tight',pad_inches=.16)
            fig.savefig(target/(name+'.png'),format='png',dpi=135,bbox_inches='tight',pad_inches=.16)
            pdf.savefig(fig,bbox_inches='tight',pad_inches=.16)
            plt.close(fig)
    return [name for name,_ in figures]
