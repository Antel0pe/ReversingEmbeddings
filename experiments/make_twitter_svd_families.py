"""Render two 1600×1200 social figures from existing measured experiments.

OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/make_twitter_svd_families.py
The original generated matcher is replayed, with saved-state checks; physical
labels enter evaluation only.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/twitter-svd-mpl')
from pathlib import Path
import sys
import json
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from grey_ones import render
from experiments.svd_cross_path_patterns import observe, candidates, select_bundle, select_unseen, orient_frame, OFFSETS
from analyze_generated_one_manifold_transport import jet, normalized

OUT = REPO/'figures/twitter/svd_families_2026_10_18'
GENERATED = REPO/'figures/generated_one_cross_path_patterns'
MNIST = REPO/'figures/mnist_svd_families'
INK = '#19272f'
MUTED = '#53636c'
BLUE = '#236b91'
RED = '#b84339'
GREEN = '#22775c'
GOLD = '#b97c1b'
BG = '#faf9f5'
FAMILY_COLORS = ['#247996', '#9c691f', '#7260a6', '#428163', '#b05163']
NAMES = ['Lean', 'Width', 'Shift x', 'Shift y', 'Height']


def canvas(title, subtitle, number):
    fig = plt.figure(figsize=(16,12), dpi=100, facecolor=BG)
    fig.text(.045,.957,title,fontsize=30,weight='bold',color=INK,va='top')
    fig.text(.045,.895,subtitle,fontsize=16,color=MUTED,va='top',linespacing=1.5)
    fig.text(.955,.018,f'{number} / 2',fontsize=12,color=MUTED,ha='right')
    return fig


def text(fig,x,y,value,size=15,**kw):
    kw.setdefault('color', INK)
    return fig.text(x,y,value,fontsize=size,**kw)


def panel(fig,rect,values,label=None,vector=False,limit=.55,color=None,labelsize=13):
    ax = fig.add_axes(rect)
    ax.imshow(np.asarray(values).reshape(28,28),cmap='RdBu_r' if vector else 'gray_r',
              vmin=-limit if vector else 0,vmax=limit if vector else 1,interpolation='nearest')
    ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.4);spine.set_color(color or '#cbd1d1')
    if label:
        ax.text(.5,-.11,label,transform=ax.transAxes,ha='center',va='top',fontsize=labelsize,color=INK,linespacing=1.35)
    return ax


def arrow(fig,a,b,color=MUTED,style='->',linewidth=2,curve=0):
    fig.add_artist(FancyArrowPatch(a,b,transform=fig.transFigure,arrowstyle=style,
                   mutation_scale=18,color=color,linewidth=linewidth,connectionstyle=f'arc3,rad={curve}'))


def save(fig,name):
    fig.savefig(OUT/name,dpi=100,facecolor=BG)
    plt.close(fig)


def inputs():
    measured=json.loads((GENERATED/'results.json').read_text())
    g=dict(np.load(GENERATED/'paths_and_profiles.npz'))
    walks=dict(np.load(GENERATED/'repeated_walks.npz'))
    m=dict(np.load(MNIST/'mnist_arrays.npz'))
    finite=json.loads((MNIST/'results.json').read_text())
    # Replay the original width walk, using its matcher, original sampling seed,
    # and saved-state checks. Physical labels never choose a displayed family.
    generated_frames=[g['root_basis']]
    generated_images=[render(g['root_settings'],dtype=np.float64)[0].ravel()]
    cosines=[np.array(measured['root']['matching_cosines'])]
    labels=[4,3,0,1,2]  # KNOBS indices for lean, width, cx, cy, height
    parent=observe(g['root_settings'])
    assert np.allclose(parent['basis'],g['root_basis'],atol=1e-12,rtol=0)
    for hop in range(1,10):
        probes=[observe(parent['p']+s*parent['lifts'][1]) for s in OFFSETS]
        fitted=select_bundle([candidates(parent['curves'],n) for n in probes])
        p=parent['p']+.3*parent['lifts'][1]
        assert np.allclose(p,walks['settings'][1,hop],atol=1e-12,rtol=0)
        child=observe(p)
        obs=candidates(parent['curves'],child)
        order=select_unseen(obs,fitted['template'])
        parent=orient_frame(child,order,obs['signs'])
        if hop in [6,9]:
            b=parent['basis']
            generated_frames.append(b)
            generated_images.append(render(p,dtype=np.float64)[0].ravel())
            true=normalized(jet(p)[3]).T[labels]
            cosines.append(abs(np.sum(b*true,axis=1)))
        print(f'Replayed original width walk: {hop}/9',flush=True)
    valid=np.flatnonzero(np.isfinite(m['grouped_bases']).all((1,2)))
    order=valid[np.argsort(m['root_distance'][valid])]
    chosen=order[np.round(np.linspace(0,len(order)-1,3)).astype(int)]
    return measured,g,walks,m,finite,np.array(generated_frames),np.array(generated_images),np.array(cosines),chosen


def discovery(measured,g,walks,frames):
    fig=canvas('Shared patterns → shared knob families',
               'Same lean sweep, three widths.',1)
    root=g['root_settings'].copy()
    lean=np.linspace(-10,35,361)
    widths=[2.4,3.2,4.6]
    colors=[BLUE,GREEN,GOLD]
    distances=[];angles=[]
    for width in widths:
        p=np.tile(root,(len(lean),1));p[:,3]=width;p[:,4]=lean
        baseline=root.copy();baseline[3]=width
        delta=render(p,dtype=np.float64).reshape(-1,784)-render(baseline,dtype=np.float64).ravel()
        tangent=jet(baseline)[3][:,4];tangent/=np.linalg.norm(tangent)
        norm=np.linalg.norm(delta,axis=1)
        use=norm>1e-12
        angle=np.full(len(lean),np.nan)
        angle[use]=np.rad2deg(np.arccos(np.clip(abs(delta[use]@tangent)/norm[use],0,1)))
        distances.append(norm);angles.append(angle)
    for col,(values,ylabel) in enumerate([(distances,'Distance from upright root (pixel L2)'),
                                        (angles,'Angle from root lean axis (°)')]):
        ax=fig.add_axes([.10+col*.46,.605,.365,.225],facecolor=BG)
        for width,color,curve in zip(widths,colors,values):
            ax.plot(lean,curve,color=color,lw=2.7,label=f'Width {width:g} px')
        ax.axvline(0,color='#a8aeae',ls=':',lw=1)
        ax.set(xlim=(-10,35),xlabel='Lean (degrees)',ylabel=ylabel)
        ax.set_xticks([-10,0,15,35]);ax.tick_params(labelsize=12)
        ax.xaxis.label.set_size(14);ax.yaxis.label.set_size(13)
        ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.13)
        if col==0:ax.legend(fontsize=11,frameon=False,loc='upper left')
        else:ax.set_ylim(0,95)
    text(fig,.10,.550,'Each width has its own upright root. Angle ignores direction sign; zero-lean angle is undefined.',12)
    text(fig,.045,.490,'Matched families',21,weight='bold')
    text(fig,.695,.490,'9 / 9 width steps stayed in-family',16,weight='bold',color=GREEN)
    centers=[.16+.175*i for i in range(5)]
    limit=float(max(abs(frames[0]).max(),abs(frames[2]).max()))
    for f,center in enumerate(centers):
        text(fig,center,.447,NAMES[f],16,weight='bold',ha='center')
        for row,index in enumerate([0,2]):
            y=.318-row*.124
            panel(fig,[center-.044,.0+y,.088,.100],frames[index,f],vector=True,limit=limit,color=FAMILY_COLORS[f])
        arrow(fig,(center,.314),(center,.299),color=FAMILY_COLORS[f],linewidth=1.5)
    text(fig,.025,.361,'Root',12,va='center')
    text(fig,.025,.236,'Wider\nstart',12,va='center')
    examples=[(4,[-10.,35.]),(3,[1.8,4.6]),(0,[9.,19.]),(1,[10.,18.]),(2,[12.,25.])]
    for f,(knob,limits) in enumerate(examples):
        center=centers[f]
        for side,value in enumerate(limits):
            p=root.copy();p[knob]=value
            panel(fig,[center-.064+side*.072,.074,.056,.075],render(p,dtype=np.float64)[0])
        arrow(fig,(center-.004,.112),(center+.004,.112),color=FAMILY_COLORS[f],linewidth=1)
        unit='°' if knob==4 else ' px'
        text(fig,center,.059,f'{limits[0]:g} → {limits[1]:g}{unit}',11,ha='center')
    text(fig,.025,.113,'Larger\nexample',10.5,va='center')
    text(fig,.045,.033,'Red gains ink; blue loses ink. Bottom: larger knob changes for visibility; some extend the default ranges.',11)
    text(fig,.045,.012,'Root-relative curves illustrate the pattern. Matching used cross-path profiles and renderer-assisted probes.',11)
    save(fig,'01_generated_knob_discovery.png')
    return dict(lean_degrees=lean,curve_widths=np.array(widths),distance_from_own_root=np.array(distances),
                angle_from_root_lean_axis=np.array(angles),illustrative_knob_extremes=np.array([v for k,v in examples]))


def families(measured,m,finite,frames,images,cosines,chosen):
    fig=canvas('Generated 1s vs real handwriting',
               'Local SVD directions, grouped across starting images.',2)
    selected_generated=[0,1,4]
    selected_real=[0,1,2]
    realframes=m['grouped_bases'][chosen]
    scale=float(max(abs(frames[[0,2]][:,selected_generated]).max(),
                    abs(realframes[[0,2]][:,selected_real]).max()))
    for block,x0 in enumerate([.045,.535]):
        text(fig,x0,.812,'GENERATED 1s' if block==0 else 'REAL MNIST',22,weight='bold')
        names=['Image','Lean','Width','Height'] if block==0 else ['Image','F1','F2','F3']
        for c,name in enumerate(names):
            text(fig,x0+.0475+c*.11,.735,name,16,ha='center',weight='bold')
        for row,index in enumerate([0,2]):
            y=.550-row*.210
            for c in range(4):
                label=None
                if c==0:
                    values=images[index] if block==0 else m['family_anchor_images'][chosen[index]]
                    label='Root' if row==0 else ('Wider start' if block==0 else 'Another start')
                else:
                    f=(selected_generated if block==0 else selected_real)[c-1]
                    values=frames[index,f] if block==0 else realframes[index,f]
                panel(fig,[x0+c*.11,y,.095,.127],values,label,c>0,scale,
                      FAMILY_COLORS[c-1] if c else None,labelsize=14)
                if c and row==0:
                    arrow(fig,(x0+c*.11+.0475,.536),(x0+c*.11+.0475,.486),
                          color=FAMILY_COLORS[c-1],linewidth=1.5)
    text(fig,.045,.208,'9 / 9',40,weight='bold',color=GREEN)
    text(fig,.045,.165,'width steps kept the expected family',18)
    agreement=finite['datasets'][1]['graphs']['pixel cosine']['triangle_axis_agreement']
    text(fig,.535,.208,f'{100*agreement:.1f}%',40,weight='bold',color=RED)
    text(fig,.535,.165,'labels agree across alternate routes',18)
    text(fig,.045,.082,'Different tests: generated uses rendered probes; MNIST uses observed neighboring images.',14)
    text(fig,.045,.047,'Black = ink. Vector red gains ink; blue loses ink. Shared color scale; 3 of 5 families shown.',13)
    text(fig,.045,.021,'Family labels belong to each dataset separately. Names on generated directions are checked afterward.',12)
    save(fig,'02_generated_vs_mnist_families.png')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    cache=OUT/'plotted_data.npz'
    if cache.exists():
        old=dict(np.load(cache))
        measured=json.loads((GENERATED/'results.json').read_text())
        g=dict(np.load(GENERATED/'paths_and_profiles.npz'))
        walks=dict(np.load(GENERATED/'repeated_walks.npz'))
        m=dict(np.load(MNIST/'mnist_arrays.npz'))
        finite=json.loads((MNIST/'results.json').read_text())
        frames=old['generated_grouped_frames'];images=old['generated_start_images']
        cosines=old['generated_control_cosines'];chosen=old['mnist_selected_anchor_ids']
    else:
        measured,g,walks,m,finite,frames,images,cosines,chosen=inputs()
    curves=discovery(measured,g,walks,frames)
    families(measured,m,finite,frames,images,cosines,chosen)
    dropped=OUT/'03_finite_sampling_control.png'
    if dropped.exists():dropped.unlink()
    np.savez_compressed(cache,generated_grouped_frames=frames,generated_start_images=images,
        generated_control_cosines=cosines,mnist_selected_anchor_ids=chosen,
        mnist_grouped_frames=m['grouped_bases'][chosen],mnist_start_images=m['family_anchor_images'][chosen],
        tracked_generated_walk_images=walks['images'],**curves)
    (OUT/'measurements.json').write_text(json.dumps(dict(image_size=[1600,1200],cards=2,
        generated_displayed_start_contexts=['root','width walk hop 6','width walk hop 9'],
        mnist_displayed_anchors=chosen[[0,2]].tolist(),generated_width_walk_correct_steps=9,
        card2_display=dict(generated_families=['lean','width','height'],mnist_families=['F1','F2','F3'],rows=['root','changed start']),
        root_relative_curves=dict(widths=[2.4,3.2,4.6],lean_range_degrees=[-10,35],samples=361,
            distance='L2 from each width own zero-lean image',angle='acute angle of root-to-image displacement to that width zero-lean derivative; undefined at zero lean',
            purpose='illustration of consistency; original matcher uses cross-path profiles'),
        larger_examples=dict(lean=[-10,35],width=[1.8,4.6],cx=[9,19],cy=[10,18],height=[12,25],
            purpose='known-control illustrations only, not scored family continuation; some extend default ranges'),
        sources=['figures/generated_one_cross_path_patterns','figures/mnist_svd_families']),indent=2)+'\n')
    (OUT/'README.md').write_text("""# SVD knob families: two social figures

Two 1600 × 1200 PNGs, in posting order. The third card was dropped from this set and the download bundle.

1. **Shared patterns to families:** root-relative distance and angle curves for a lean sweep from −10 to 35 degrees at widths 2.4, 3.2, and 4.6 px. Every width uses its own zero-lean root; other settings are the canonical root. Distance is the full 784-pixel L2 displacement. Angle is the acute angle between that displacement and the local zero-lean derivative, ignoring sign; it is undefined at zero lean. These are illustrative known-control curves, not a claim that root-relative curves alone were the assignment objective. Below are actual SVD vectors at the root and original width-walk step 9, assigned by the original path-profile matcher. Names are evaluation labels. The bottom image pairs exaggerate each physical knob so the transformation is visible. Lean: −10→35°; width: 1.8→4.6; cx: 9→19; cy: 10→18; height: 12→25 px. These illustration settings can extend the default box; they are not claimed to be recovered paths.
2. **Generated versus MNIST families:** simplified to two starting images and three representative families per dataset. Generated shows lean, width, and height at root and original width-walk step 9. MNIST shows F1/F2/F3 at the root and farthest root-distance anchor in its graph. The other two families are omitted only for display; the original matcher still assigned all five jointly. Cosine labels were removed to keep attention on the maps and continuation results. The generated width walk was replayed with the original matcher and sampling seed; every state matched saved parameters. Physical labels enter only evaluation. MNIST source indices and all five original frame vectors remain saved in the arrays. The displayed 9/9 and 54.2% describe different tests: signed physical-family continuation for generated data and graph-triangle label closure for MNIST. Vector pattern differences alone do not establish a bad family; route inconsistency supplies the independent failure evidence.

All vector maps show the full 28×28 unit pixel-change arrays, with a common symmetric color scale in each card. Red gains ink and blue loses ink. Source images use white=0 and black=1. The larger example pairs move only the named physical parameter and remain inside the canvas.

The generated matcher compared whole profiles of corresponding-path distance, displacement mismatch, and displacement angle across offset branches. Physical control labels did not enter the assignment score. The renderer and Jacobian supplied valid sampling and probes. Raw MNIST cannot supply those probes and uses an explicitly labeled observed-neighbor adaptation. The omitted finite-cloud control remains documented in `figures/mnist_svd_families/README.md`.

Reproduce from the repository root:
```sh
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/make_twitter_svd_families.py
```

`plotted_data.npz` retains all displayed frames and new curves; `measurements.json` records definitions and larger illustration settings. Figures were inspected with the repository understanding check before delivery. Shareable artifacts contain no machine-specific workspace paths.
""")
    with zipfile.ZipFile(OUT/'twitter_svd_families.zip','w',zipfile.ZIP_DEFLATED) as bundle:
        for name in ['01_generated_knob_discovery.png','02_generated_vs_mnist_families.png','README.md','measurements.json']:
            bundle.write(OUT/name,arcname=name)
    print('Rebuilt the two-card set and download bundle.')


if __name__=='__main__':
    main()
