"""Project PDF-defined attributes onto a separate, unchanged logical model."""
from pathlib import Path
import json
import re
from protocol import ROOT,HERE,PRIOR,load,source,sha256,write_atomic,save,csv_file
from supervisor_compliance import numeric_attributes,attribute_diff,normalize


def strip_attributes(text):
    return normalize(re.sub(r'\{[^{}]*\}','',text))


def aligned_text(original,pdf_values):
    """Retain model layout, replacing only numeric attribute blocks."""
    defined={}
    for (feature,key),value in pdf_values.items():defined.setdefault(feature,{})[key]=value
    seen=set();result=[]
    for line in original.splitlines():
        match=re.fullmatch(r'(\s*)([A-Za-z_]\w*)(?:\s+\{([^{}]*)\})?\s*',line)
        if not match or match[2] in ('namespace','features','constraints'):
            result.append(line);continue
        indent,feature,body=match.groups()
        if body=='abstract':result.append(line);continue
        if feature not in defined and (body is None or not re.search(r'\d',body)):
            result.append(line);continue
        if feature in defined:
            attrs=[]
            old_tokens={x.strip().split()[0]:x.strip() for x in (body or '').split(',') if x.strip()}
            for key,value in defined[feature].items():
                literal=str(int(value)) if key=='imgsz' else str(value)
                attrs.append(old_tokens.get(key,key+' '+literal))
            result.append(indent+feature+' {'+', '.join(attrs)+'}');seen.add(feature)
        else:result.append(indent+feature)
    if seen!=set(defined):raise ValueError('A PDF feature could not be projected onto the model.')
    projected='\n'.join(result)+'\n'
    if numeric_attributes(projected)!=pdf_values:raise ValueError('Projected numeric attributes differ from the PDF.')
    if strip_attributes(projected)!=strip_attributes(original):raise ValueError('Projection changed the logical tree or constraints.')
    return projected


def create(out):
    from common import FMP,FMD
    pages=load(PRIOR/'pdf_pages.json')['pages'];folder=HERE/'feature_models';folder.mkdir(exist_ok=True)
    rows=[];report={};bindings={}
    for domain,path,part in [('FM_P',FMP,pages[4:6]),('FM_D',FMD,pages[6:9])]:
        values=numeric_attributes('\n'.join(part));old=path.read_text(encoding='utf-8')
        new=aligned_text(old,values);target=folder/('fm_p.uvl' if domain=='FM_P' else 'fm_d.uvl')
        if target.exists():
            if target.read_text(encoding='utf-8')!=new:raise FileExistsError('Conflicting aligned model: '+str(target))
        else:write_atomic(target,new,out)
        for row in attribute_diff(values,numeric_attributes(new)):rows.append({'domain':domain,**row})
        differences=attribute_diff(values,numeric_attributes(old))
        if domain=='FM_P':
            for row in differences:
                if row['status']=='IMPLEMENTATION_EXTENSION':bindings.setdefault(row['feature'],{})[row['attribute']]=row['repository_value']
        report[domain]={'source':source(path),'aligned':source(target),'PDF_defined_attributes':len(values),
            'numeric_equality':'EXACT','logical_tree_and_constraints':'UNCHANGED',
            'literal_PDF_bytes':'NOT_CLAIMED; PDF line wraps and UVL formatting differ.',
            'historical_attribute_differences':differences}
    csv_file(out/'MODEL_ALIGNMENT.csv',rows)
    proposal={'schema':'VM36_IMPLEMENTATION_BINDINGS_PROPOSAL_V1','status':'NEEDS_SCIENTIFIC_REVIEW',
        'source_PDF_sha256':load(PRIOR/'compliance_stage_04.json')['source_pdf_sha256'],
        'source_kind':'Existing implementation attributes absent from the supervisor PDF; not supervisor-defined values.',
        'proposed_values_from_historical_implementation':bindings,
        'default_application':False,'approval':None,
        'rationale':{'motion_angles':'A 2D motion PSF needs an orientation rule; kernel length alone does not define it. The previous uniform-angle rule is not calibrated flight physics.',
             'airlight':'The historical haze proxy also needs an airlight intensity.',
             'rain':'Streak count alone does not define line length or opacity.',
             'rolling':'Pixel warp amplitude and frequency are unspecified by the PDF; GT/ignore geometry is still unverified.'}}
    save(out/'IMPLEMENTATION_BINDINGS_PROPOSAL.json',proposal,out)
    save(out/'MODEL_ALIGNMENT.json',{'status':'VALIDATED_NUMERIC_AND_STRUCTURAL_ALIGNMENT','models':report,
        'no_original_files_modified':True,'missing_rendering_choices':'NEEDS_SCIENTIFIC_REVIEW',
        'FM_D_parameter_counts':'Draft params_m values match the PDF; do not substitute them for actual checkpoint measurements.'},out)
    return report


def parameters(path,leaves):
    from sampling import models
    fmp,_,_=models()
    if not fmp.valid_leaves(leaves):raise ValueError('Illegal FM_P selection.')
    attributes=numeric_attributes(Path(path).read_text(encoding='utf-8'))
    return {leaf:{key:value for (feature,key),value in attributes.items() if feature==leaf} for leaf in sorted(leaves)}


def require_runtime_bindings(recipe,proposal,approval=None,protocol_sha256=None):
    needed={leaf:extra for leaf,extra in proposal['proposed_values_from_historical_implementation'].items() if leaf in recipe}
    if not needed:return recipe
    if not approval or approval.get('status')!='APPROVED' or approval.get('protocol_sha256')!=protocol_sha256 or not approval.get('reviewer') or not approval.get('reviewed_at_utc'):
        raise ValueError('IMPLEMENTATION_BINDINGS_REVIEW_REQUIRED: '+', '.join(sorted(needed)))
    if approval.get('approved_values')!=needed:raise ValueError('Approved implementation values differ from the proposal.')
    return {leaf:{**attrs,**needed.get(leaf,{})} for leaf,attrs in recipe.items()}
