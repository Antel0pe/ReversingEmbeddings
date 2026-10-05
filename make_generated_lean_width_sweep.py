"""Increase the width knob in 0.1 steps; measure lean paths directly in 784D.

Run: python make_generated_lean_width_sweep.py
Uses the actual Python renderer and reuses 3-unit chord sampling helpers.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist

from grey_ones import RANGES
from make_generated_lean_width_comparison import BASE, LO, HI, images, norm, angle, stats, sample, closest_paths

OUT = Path(__file__).resolve().parent / 'figures/generated_lean_width_comparison/width_sweep'


def make_figures(offsets, lean, dense, rows, profiles):
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':150})
    colors = ['#20252a'] + [plt.cm.viridis(v) for v in np.linspace(.12,.88,len(offsets)-1)]
    fig = plt.figure(figsize=(17,8.2))
    fig.text(.06,.95,'How far does each wider lean path diverge from normal?',fontsize=19,weight='bold')
    fig.text(.06,.89,'Generated 28 x 28 coverage images: white = no ink (0), black = full ink (1). Normal upright reference is on the left.\n'
             'Only width (horizontal pixels) and lean vary; center (14.5, 14.5) px and height 19.75 px stay fixed. Distances use all 784 pixel values.',fontsize=11)
    for k,i in enumerate([0,1,5,10]):
        ax = fig.add_axes([.08+k*.145,.65,.105,.16])
        ax.imshow(images(BASE[3]+offsets[i],0)[0].reshape(28,28),cmap='gray_r',vmin=0,vmax=1)
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True);spine.set_color(colors[i]);spine.set_linewidth(2)
        ax.text(.5,-.12,f'upright gap: {rows[i]["upright_gap"]:.2f}',ha='center',va='top',transform=ax.transAxes,fontsize=10,color=colors[i])
        ax.set_title(f'Width {BASE[3]+offsets[i]:.1f}\nknob +{offsets[i]:.1f}',fontsize=10,color=colors[i])
    fig.text(.72,.725,'Dots mark each upright image at lean = 0.\nThese are DIFFERENT images: their widths differ.\nBlack dashed = normal width 3.2.',fontsize=12)
    positions=[.06,.325,.59]
    specs=[('from_upright','Distance from the original upright 1'),
           ('same_lean_gap','Gap from normal at the same lean'),
           ('aligned_mismatch','Lean-change mismatch from each upright')]
    for pos,(key,title) in zip(positions,specs):
        ax = fig.add_axes([pos,.205,.225,.345])
        for i,off in enumerate(offsets):
            ax.plot(lean,profiles[i][key],color=colors[i],ls='--' if i==0 else '-',lw=1.8,
                    label='normal (+0)' if i==0 else f'+{off:.1f} (width {BASE[3]+off:.1f})')
        zero=int(np.argmin(abs(lean)))
        for i,off in enumerate(offsets):
            ax.scatter([0],[profiles[i][key][zero]],s=35,color=colors[i],edgecolor='white',linewidth=.7,zorder=4)
        if key=='aligned_mismatch':
            ax.scatter([0],[0],s=65,facecolors='white',edgecolors='#20252a',linewidth=1.5,zorder=5)
            ax.annotate('All zero here by subtraction',xy=(0,0),xytext=(7,.15),fontsize=9,arrowprops=dict(arrowstyle='->',color='#20252a'))
        ax.axvline(0,color='#b0b0b0',ls=':',lw=1)
        ax.set(title=title,xlabel='Lean (degrees); 0 = upright',ylabel='Image distance (L2)')
        ax.grid(alpha=.16)
        if key=='aligned_mismatch':ax.legend(loc='center left',bbox_to_anchor=(1.04,.5),fontsize=9,frameon=False)
    fig.text(.06,.065,'At lean = 0: only the NORMAL width is the original reference image. Wider upright images have nonzero gaps (colored dots).\n'
             'Right: compare [image at this lean - its own upright image] between widths. Every difference is zero at upright by construction.\n'
             'Upright angle (x = 0) does not mean identical image (y = 0). These are scalar distances in 784D, not projected curve locations.',fontsize=11)
    fig.savefig(OUT/'divergence_lines.png');plt.close(fig)

    fig = plt.figure(figsize=(15,7.1))
    fig.text(.065,.94,'How much does a wider stroke change the lean direction?',fontsize=18,weight='bold')
    fig.text(.065,.86,'Normal width 3.2 is the reference. Compare the same small forward lean move at matching lean; normalize vectors to remove speed.\n'
             'Angle is measured in the original 784 pixel coordinates: 0 degrees = identical direction, 90 = perpendicular.',fontsize=11)
    ax = fig.add_axes([.065,.28,.52,.43])
    theta = profiles[0]['angle_lean']
    for i in range(len(offsets)):
        ax.plot(theta,profiles[i]['direction_angle'],color=colors[i],ls='--' if i==0 else '-',lw=1.6,
                label='normal (+0)' if i==0 else f'knob +{offsets[i]:.1f}')
    ax.axvline(0,color='#b0b0b0',ls=':');ax.grid(alpha=.16)
    ax.set(xlabel='Lean (degrees)',ylabel='Angle from normal lean direction (degrees)',ylim=(-2,92),title='Angle along each lean path (0.1-degree centered move)')
    ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize=9,frameon=False)
    ax = fig.add_axes([.77,.28,.20,.43])
    ax.plot(offsets,[r['direction_angle_01deg']['median'] for r in rows],'-o',color='#604493',label='median across lean')
    ax.plot(offsets,[r['direction_angle_01deg']['max'] for r in rows],'--s',color='#b26820',label='maximum across lean')
    ax.set(xlabel='Increase of width knob',ylabel='Angle (degrees)',ylim=(-2,92),title='Does the change grow with width?');ax.grid(alpha=.16)
    ax.legend(fontsize=9,loc='upper left')
    fig.text(.065,.11,'Centered moves cover -9.5 to +34.5 degrees so all tested step sizes remain inside the allowed lean interval.\n'
             'A direction is an ink-change pattern at stroke edges. Increasing width moves these edges; even the same lean operation can affect different pixels.\n'
             'Small-step angles describe local directions; a single angle between whole endpoint chords would hide turning along each curve.',fontsize=11)
    fig.savefig(OUT/'direction_angles.png');plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    offsets=np.round(np.arange(11)*.1,10)
    widths=BASE[3]+offsets
    assert np.all((widths>=RANGES[3,0])&(widths<=RANGES[3,1]))
    lean=np.linspace(LO,HI,901)
    angle_lean=np.linspace(LO+.5,HI-.5,881)
    dense=[images(w,lean) for w in widths]
    upright=[images(w,0)[0] for w in widths]
    baseline_steps={h:images(widths[0],angle_lean+h/2)-images(widths[0],angle_lean-h/2) for h in [.02,.1,.5,1.]}
    paths=[sample(w,3.) for w in widths]
    baseline_directions=np.diff(paths[0][1],axis=0)[np.clip(np.searchsorted(paths[0][0],lean,side='right')-1,0,len(paths[0][0])-2)]
    base_dm=cdist(dense[0][::10],dense[0][::10])
    rows=[];profiles=[]
    for i,(off,w,x,x0,(knots,px,gaps)) in enumerate(zip(offsets,widths,dense,upright,paths)):
        from_upright=norm(x-upright[0])
        same=norm(x-dense[0])
        aligned=norm((x-x0)-(dense[0]-upright[0]))
        angles={h:angle(baseline_steps[h],images(w,angle_lean+h/2)-images(w,angle_lean-h/2)) for h in baseline_steps}
        if i==0:
            angles={h:np.zeros(len(angle_lean)) for h in baseline_steps}
        coarse=np.diff(px,axis=0)[np.clip(np.searchsorted(knots,lean,side='right')-1,0,len(knots)-2)]
        coarse_angle=angle(baseline_directions,coarse) if i else np.zeros(len(lean))
        exact=closest_paths(paths[0][1],px) if i else (0.,0.,0.,0,0)
        dm=cdist(x[::10],x[::10])
        scale=float(np.sum(base_dm*dm)/np.sum(base_dm**2))
        residual=float(norm((dm-scale*base_dm).ravel())/norm(dm.ravel()))
        arc_fine=float(norm(np.diff(images(w,np.linspace(LO,HI,1801)),axis=0)).sum())
        arc_coarse=float(norm(np.diff(x,axis=0)).sum())
        assert abs(arc_fine-arc_coarse)/arc_fine < .0002
        assert np.max(abs(x.sum(axis=1)-w*BASE[2]))<2e-5
        assert np.min(same)+2e-6>=off*BASE[2]/28
        if i==0:assert np.max(same)==0 and np.max(aligned)==0
        rows.append(dict(width_knob=float(w),increase=float(off),upright_gap=float(same[np.argmin(abs(lean))]),
            same_lean_gap=stats(same),distance_from_normal_upright=stats(from_upright),upright_aligned_divergence=stats(aligned),
            direction_angle_01deg=stats(angles[.1]),direction_angle_by_step={str(h):stats(a) for h,a in angles.items()},
            direction_angle_at_upright=float(angles[.1][np.argmin(abs(angle_lean))]),
            polyline_direction_angle=stats(coarse_angle),polyline_minimum_distance=exact[0],
            closest_segments=dict(normal_index=exact[3],comparison_index=exact[4],normal_fraction=exact[1],comparison_fraction=exact[2]),
            point_count=len(knots),lean_points=knots.tolist(),consecutive_distances=gaps.tolist(),
            polyline_length=float(gaps.sum()),arc_length_005deg=arc_coarse,arc_length_0025deg=arc_fine,
            distance_matrix_best_scale=scale,distance_matrix_residual_fraction=residual))
        profiles.append(dict(from_upright=from_upright,same_lean_gap=same,aligned_mismatch=aligned,
            angle_lean=angle_lean,direction_angle=angles[.1],polyline_angle=coarse_angle))
    result=dict(normal_knobs=BASE.tolist(),width_unit='existing raw width knob (horizontal pixels); +0.1 means increase, not absolute width 0.1',
        offsets=offsets.tolist(),lean_range=[float(LO),float(HI)],lean_grid_degrees=.05,distance_definition='L2 over all 784 coverage coordinates',
        rows=rows,sampling_rule='First 3-unit chord crossing from minimum lean; force upright, restart, then force maximum.',
        scope='Fixed center and height; width and lean slice of the full five-parameter generated family.')
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'summary.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['width_knob','increase','upright_gap','median_same_lean_gap','max_aligned_divergence','median_direction_angle','max_direction_angle','minimum_polyline_gap'])
        for r in rows:writer.writerow([r['width_knob'],r['increase'],r['upright_gap'],r['same_lean_gap']['median'],r['upright_aligned_divergence']['max'],r['direction_angle_01deg']['median'],r['direction_angle_01deg']['max'],r['polyline_minimum_distance']])
    with (OUT/'profiles.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['width_knob','increase','lean_deg','distance_from_normal_upright','same_lean_gap','upright_aligned_divergence','local_direction_angle','polyline_direction_angle'])
        for w,off,p in zip(widths,offsets,profiles):
            for j,t in enumerate(lean):
                a=float(np.interp(t,angle_lean,p['direction_angle'])) if angle_lean[0]<=t<=angle_lean[-1] else ''
                writer.writerow([w,off,t,p['from_upright'][j],p['same_lean_gap'][j],p['aligned_mismatch'][j],a,p['polyline_angle'][j]])
    with (OUT/'sampled_points.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['width_knob','increase','cx','cy','height','lean_deg','distance_from_previous','forced_anchor'])
        for w,off,(knots,x,gaps) in zip(widths,offsets,paths):
            for j,t in enumerate(knots):writer.writerow([w,off,*BASE[:3],t,'' if j==0 else gaps[j-1],any(np.isclose(t,z) for z in [LO,0,HI])])
    make_figures(offsets,lean,dense,rows,profiles)
    print(json.dumps([{k:r[k] for k in ['increase','upright_gap','same_lean_gap','upright_aligned_divergence','direction_angle_01deg','polyline_minimum_distance','arc_length_0025deg']} for r in rows],indent=2))

if __name__=='__main__':main()
