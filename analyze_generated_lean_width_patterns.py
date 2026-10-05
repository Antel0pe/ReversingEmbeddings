"""Check shared scalar distance patterns, with fresh width/lean controls."""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from make_generated_lean_width_comparison import BASE, images, norm
from make_generated_lean_width_sweep import OUT, make_figures


def main():
    result=json.loads((OUT/'results.json').read_text())
    offsets=np.array(result['offsets'])
    with (OUT/'profiles.csv').open() as f:
        records=list(csv.DictReader(f))
    profiles=[]
    for off in offsets:
        group=[r for r in records if abs(float(r['increase'])-off)<1e-8]
        good=[r for r in group if r['local_direction_angle']!='']
        if off==0:lean=np.array([float(r['lean_deg']) for r in group])
        profiles.append(dict(from_upright=np.array([float(r['distance_from_normal_upright']) for r in group]),
            same_lean_gap=np.array([float(r['same_lean_gap']) for r in group]),
            aligned_mismatch=np.array([float(r['upright_aligned_divergence']) for r in group]),
            angle_lean=np.array([float(r['lean_deg']) for r in good]),
            direction_angle=np.array([float(r['local_direction_angle']) for r in good])))
    make_figures(offsets,lean,None,result['rows'],profiles)
    metrics={}
    templates={}
    model_profiles={}
    for key in ['same_lean_gap','aligned_mismatch','direction_angle']:
        values=np.array([p[key] for p in profiles[1:]])
        template=np.sum(offsets[1:,None]*values,axis=0)/np.sum(offsets[1:]**2)
        prediction=offsets[1:,None]*template
        corr=np.corrcoef(values)
        upper=corr[np.triu_indices(len(values),1)]
        metrics[key]=dict(relative_frobenius_error=float(norm((values-prediction).ravel())/norm(values.ravel())),
            pairwise_pearson_min=float(upper.min()),pairwise_pearson_median=float(np.median(upper)),pairwise_pearson_max=float(upper.max()))
        templates[key]=template
        model_profiles[key]=values
    # Fresh offsets and half-grid lean values were not used to fit either distance template.
    fresh_offsets=np.array([.15,.35,.55,.75,.95])
    fresh_lean=(lean[:-1]+lean[1:])/2
    normal=images(BASE[3],fresh_lean)
    normal0=images(BASE[3],0)[0]
    fresh_values={k:[] for k in ['same_lean_gap','aligned_mismatch']}
    for off in fresh_offsets:
        wider=images(BASE[3]+off,fresh_lean)
        wider0=images(BASE[3]+off,0)[0]
        fresh_values['same_lean_gap'].append(norm(wider-normal))
        fresh_values['aligned_mismatch'].append(norm((wider-wider0)-(normal-normal0)))
    for key,values in fresh_values.items():
        values=np.array(values)
        prediction=fresh_offsets[:,None]*np.interp(fresh_lean,lean,templates[key])[None,:]
        metrics[key]['held_out_width_and_lean_relative_error']=float(norm((values-prediction).ravel())/norm(values.ravel()))
    analysis=dict(model='distance(increase, lean) approximately equals increase times shared_profile(lean)',
        training_offsets=offsets[1:].tolist(),held_out_offsets=fresh_offsets.tolist(),
        held_out_lean='midpoints of training grid; 0.025-degree shift',metrics=metrics,
        limits='Scalar distance profiles only; does not establish parallel 784D tangents, vector separability, or a decoder of the five-knob family.')
    (OUT/'pattern_analysis.json').write_text(json.dumps(analysis,indent=2)+'\n')
    with (OUT/'shared_distance_templates.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['lean_deg','same_lean_gap_per_width_unit','aligned_mismatch_per_width_unit'])
        writer.writerows(zip(lean,templates['same_lean_gap'],templates['aligned_mismatch']))
    colors=[plt.cm.viridis(v) for v in np.linspace(.12,.88,10)]
    fig,axs=plt.subplots(1,2,figsize=(14,6.4))
    fig.subplots_adjust(left=.07,right=.98,bottom=.25,top=.71,wspace=.3)
    fig.text(.07,.94,'Do the distance curves share a common shape?',fontsize=19,weight='bold')
    fig.text(.07,.86,'Divide each distance by its width increase (+0.1 through +1.0). Curves would coincide if the width effect were exactly proportional.\n'
        'Color matches the width sweep: purple = +0.1; yellow-green = +1.0. Dashed black = one shared profile fitted to all ten widths.',fontsize=11)
    for ax,key,title in zip(axs,['same_lean_gap','aligned_mismatch'],['Same-lean gap, per unit of width increase','Upright-subtracted divergence, per width unit']):
        for off,values,color in zip(offsets[1:],model_profiles[key],colors):
            ax.plot(lean,values/off,color=color,lw=1.4)
        ax.plot(lean,templates[key],color='#20252a',ls='--',lw=2)
        ax.axvline(0,color='#aaa',ls=':');ax.grid(alpha=.15)
        ax.set(title=title,xlabel='Lean (degrees); 0 = upright',ylabel='Image distance / width-knob increase')
    fig.text(.07,.775,
        f'Left: fitted error {100*metrics["same_lean_gap"]["relative_frobenius_error"]:.1f}%; fresh widths/lean {100*metrics["same_lean_gap"]["held_out_width_and_lean_relative_error"]:.1f}%. '
        f'Right: fitted error {100*metrics["aligned_mismatch"]["relative_frobenius_error"]:.1f}%; fresh widths/lean {100*metrics["aligned_mismatch"]["held_out_width_and_lean_relative_error"]:.1f}%.',fontsize=11)
    fig.text(.07,.085,'All measurements use 784 coverage values. Center (14.5, 14.5) px and height 19.75 px are fixed; width knob is measured in horizontal pixels.\n'
        'Fresh controls use width increases +0.15, +0.35, +0.55, +0.75, +0.95 and lean values halfway between the original grid points.\n'
        'Small scalar errors show approximate proportionality over this tested range; they do not mean the image curves have parallel directions.',fontsize=10)
    fig.savefig(OUT/'shared_distance_pattern.png',dpi=150);plt.close(fig)
    assert metrics['same_lean_gap']['held_out_width_and_lean_relative_error']<.03
    assert metrics['aligned_mismatch']['held_out_width_and_lean_relative_error']<.06
    report=OUT.parent/'README.md'
    text=report.read_text()
    heading='## Clarifying upright references and the shared pattern'
    if heading in text:text=text[:text.index(heading)].rstrip()+'\n'
    text+='\n'+heading+'\n\n'
    text+='Upright means lean is zero. It does not identify one unique image: width 3.2, width 3.3, and width 4.2 produce different upright images. Only the width-3.2 lean curve passes through the original normal upright reference. The colored dots mark each curve at lean zero.\n\n'
    text+='The left graph measures distance to the original upright image, not the raw norm of an image. At lean zero, normal width has distance zero; the wider upright images have gaps 0.310 for +0.1 and 3.105 for +1.0.\n\n'
    text+='"Aligning" means subtracting the upright image separately from each curve, then comparing their displacement vectors. It is a change of origin in pixel space, not a change to the rendered strokes. Both displacement vectors become zero at upright, so their difference is zero there. This does not imply the original rendered images were identical or that the original curves cross.\n\n'
    text+='For every curve to contain exactly the same original normal image, width would need to return to 3.2 at that point. That would be a different experiment: width and lean would both vary along each path.\n\n'
    text+='The noticed pattern is supported by the distance profiles. Fitting `distance approximately equals width_increase times shared_profile(lean)` gives relative errors '+f"{100*metrics['same_lean_gap']['relative_frobenius_error']:.2f}% for same-lean gaps and {100*metrics['aligned_mismatch']['relative_frobenius_error']:.2f}% for upright-subtracted divergence. Fresh offsets and shifted lean values give {100*metrics['same_lean_gap']['held_out_width_and_lean_relative_error']:.2f}% and {100*metrics['aligned_mismatch']['held_out_width_and_lean_relative_error']:.2f}% respectively. These percentages are Frobenius residual divided by measured-profile norm; they are not reconstruction errors.\n\n"
    text+='![Shared scalar distance pattern](width_sweep/shared_distance_pattern.png)\n\n'
    text+='A small width increase approximately multiplies the local pixel response to width. That response changes with lean. This explains a shared scalar-distance shape whose amplitude grows with the width step. The same-lean gap is fairly steady but not constant: for +1 it ranges 2.618 to 3.105.\n\n'
    text+='Angle profiles also have repeatable features, but a simple width-proportional profile has '+f"{100*metrics['direction_angle']['relative_frobenius_error']:.1f}% relative error. Pairwise Pearson correlations range {metrics['direction_angle']['pairwise_pearson_min']:.3f} to {metrics['direction_angle']['pairwise_pearson_max']:.3f}, with median {metrics['direction_angle']['pairwise_pearson_median']:.3f}. Some angle peaks shift or change shape with width; avoid claiming every pair has high correlation.\n\n"
    text+='Similarity of scalar distance graphs is distinct from parallelism of the 784D curve tangents. The graph collapses do not provide vector reconstruction, exact separability, or coordinates decoding the full five-knob family.\n'
    report.write_text(text)
    print(json.dumps(analysis,indent=2))

if __name__=='__main__':main()
