"""Render readable evidence for the SVD/PCA coordinate bridge experiments."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/svd-state-bridge-mpl')
from pathlib import Path
import sys,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.svd_state_coordinate_bridge import OUT,readers


def one(rows,name,radius,method):
    return next(r for r in rows if r['name']==name and r['radius']==radius and r['method']==method)


def direction_figure(result,a):
    rows=result['cases'];root=one(rows,'root',.001,'svd');prefix='root_0.001_svd'
    x=a[prefix+'_source_image'];b=a[prefix+'_basis'];j=a[prefix+'_true_fields'];jc=a[prefix+'_unmixed_fields']
    _,_,_,dp,_=readers(x)
    scores=np.abs((j/np.linalg.norm(j,axis=0)).T@b.T)
    rr,cc=linear_sum_assignment(-scores);order=cc[np.argsort(rr)]
    raw=b[order[2]]/(dp[2]@b[order[2]])
    norm=np.linalg.norm(j[:,2])
    shown=[x,j[:,2]/norm,raw/norm,(raw-j[:,2])/norm,jc[:,2]/norm,(jc[:,2]-j[:,2])/norm]
    titles=['Starting image','True height direction','One SVD axis, rescaled','Its signed error','Mix all five SVD axes','Remaining signed error']
    labels=['Root: (14,14,19.75,3.2,0°)','Independent evaluation target',
            f"Relative error {root['best_assigned_single_axis']['per_knob_relative'][2]:.3%}",
            'Side changes contaminate height',f"Relative error {root['calibrated_span']['per_knob_relative'][2]:.5%}",
            'Below visible contrast on this scale']
    limit=max(np.abs(v).max() for v in shown[1:])
    fig=plt.figure(figsize=(16,10),facecolor='white')
    fig.text(.04,.97,'Mixed estimated vectors can still produce a clean physical control',fontsize=20,weight='bold',va='top')
    fig.text(.04,.918,'Controlled 28 × 28 generated 1s: each pixel is ink coverage (white 0, black 1). Top row: original root, neighborhood radius 0.001.\n'
        'The conversion reads image geometry, then combines all five estimated axes. True generator derivatives are used only to evaluate the result.',
        fontsize=10.6,va='top',linespacing=1.5)
    grid=fig.add_gridspec(1,6,left=.045,right=.96,bottom=.61,top=.825,wspace=.20)
    for i,v in enumerate(shown):
        ax=fig.add_subplot(grid[i])
        if i==0:ax.imshow(v.reshape(28,28),cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
        else:im=ax.imshow(v.reshape(28,28),cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
        ax.set_xticks([]);ax.set_yticks([]);ax.set_title(titles[i],fontsize=10,pad=10)
        ax.text(.5,-.10,labels[i],transform=ax.transAxes,ha='center',va='top',fontsize=8.8,wrap=True)
    cb=fig.add_axes([.78,.554,.17,.012]);fig.colorbar(im,cax=cb,orientation='horizontal')
    cb.set_xlabel('Shared signed scale',fontsize=8.5)
    fig.text(.045,.55,'Directions: normalized by the same true-height vector length. Red = positive; blue = negative.\n'
        'Direction panels describe ink change. Error panels describe predicted minus true pixel change.',fontsize=9.3,va='top',linespacing=1.5)
    axes=[fig.add_axes([.085,.14,.37,.295]),fig.add_axes([.58,.14,.37,.295])]
    colors=['#6a6a6a','#276c9b','#16846c'];names=['One assigned axis','Unmix the five axes','Fit changing directions']
    selectors=[lambda r:r['best_assigned_single_axis']['worst_relative'],lambda r:r['calibrated_span']['worst_relative'],
               lambda r:r['polynomial']['2']['calibrated_graph_direction']['worst_relative']]
    for method,style in [('svd','-'),('pca','--')]:
        rr=sorted([r for r in rows if r['name']=='root' and r['method']==method],key=lambda r:r['radius'])
        for color,label,select in zip(colors,names,selectors):
            axes[0].plot([r['radius'] for r in rr],[100*select(r) for r in rr],style,color=color,marker='.' if method=='svd' else None,label=label if method=='svd' else None)
        rr=[r for r in rows if r['radius']==.001 and r['method']==method]
        rr=sorted(rr,key=lambda r:r['calibrated_span']['worst_relative'])
        for color,label,select in zip(colors,names,selectors):
            axes[1].plot(np.arange(1,len(rr)+1),[100*select(r) for r in rr],style,color=color)
    axes[0].set_xscale('log');axes[0].set_title('Root: worst direction error across all five controls',fontsize=11)
    axes[0].set_xlabel('Neighborhood radius: Euclidean image distance',fontsize=9.5)
    axes[1].set_title('49 starts at radius 0.001, including boundary cases',fontsize=11)
    axes[1].set_xlabel('Starting images ordered by unmixing error',fontsize=9.5)
    for ax in axes:
        ax.set_yscale('log');ax.set_ylabel('Relative direction error (%)',fontsize=10);ax.grid(alpha=.2);ax.tick_params(labelsize=9)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='center left',bbox_to_anchor=(.045,.485),ncol=3,frameon=False,fontsize=9)
    fig.text(.045,.065,'Solid = SVD; dashed = centered PCA. Their calibrated curves nearly coincide. '
        'Errors compare full 784-pixel derivative vectors, not retained variance.',fontsize=9.4)
    fig.text(.045,.03,'A single associated axis keeps its physical mixture as the radius shrinks. Combining axes removes mixing; '
        'fitted interaction terms reduce local curvature error. Boundary cases still need pieces.',fontsize=9.4)
    fig.savefig(OUT/'unmixing_estimated_vectors.png',dpi=150)
    plt.close(fig)


def chart_figure(result,a):
    models=result['algebraic_calibrated_charts']
    fig=plt.figure(figsize=(14,11),facecolor='white')
    fig.text(.045,.972,'Five imperfect measurements identify images locally; a fixed global frame fails',fontsize=18,weight='bold',va='top')
    fig.text(.045,.924,'Top: fit explicit pixel formulas from 1,024 images in each small region; infer 512 new images using only five SVD/PCA measurements.\n'
        'Top/bottom/intercept/width/slope are derived from fitting images. Query images do not supply hidden knob values to the inverse.',fontsize=10.5,va='top',linespacing=1.5)
    for i,name in enumerate(['root','smooth','tilted_cap_boundary']):
        m=next(x for x in models if x['name']==name and x['method']=='svd')
        pref='algebraic_'+name+'_svd';left=.085+.315*i
        ax=fig.add_axes([left,.65,.19,.21])
        ax.imshow(a[pref+'_source_image'].reshape(28,28),cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
        ax.set_xticks([]);ax.set_yticks([])
        ax.set_title({'root':'Upright root','smooth':'Smooth start, lean 17.3°','tilted_cap_boundary':'Cap boundary, lean 30°'}[name],fontsize=11,pad=10)
        mx=max(x['closed_loop_joint_actions']['worst_pixel'] for x in models if x['name']==name)
        fig.text(left+.095,.625,f"Box half-spans: {m['coordinate_half_spans'][0]:g} px; slope {m['coordinate_half_spans'][4]:g}\n{m['pieces']} polynomial piece(s)\nSVD and PCA: 512/512 unique inverses\nWorst joint-update pixel error: {mx:.1g}",ha='center',va='top',fontsize=10,linespacing=1.6)
    fig.text(.045,.515,'These are deliberately small calibrated regions, not whole-family validation. '
        'The boundary examples use four pieces; the smooth example uses one.',fontsize=10.1)
    fig.text(.045,.468,'Counterexample: reuse the root’s five estimated vectors everywhere',fontsize=14,weight='bold')
    images=a['collision_images'];delta=images[1]-images[0]
    lim=float(np.max(np.abs(delta)))
    for i,(data,title,caption) in enumerate([(images[0],'Valid image A','Height 19.6 px'),(images[1],'Valid image B','Height 19.8 px'),
                                           (delta,'Actual pixel difference','Red = gained ink; blue = lost ink')]):
        left=.085+.315*i
        ax=fig.add_axes([left,.19,.19,.23])
        if i<2:ax.imshow(data.reshape(28,28),cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
        else:im=ax.imshow(data.reshape(28,28),cmap='RdBu_r',vmin=-lim,vmax=lim,interpolation='nearest')
        ax.set_xticks([]);ax.set_yticks([]);ax.set_title(title,fontsize=11,pad=10)
        ax.text(.5,-.08,caption,transform=ax.transAxes,ha='center',va='top',fontsize=10)
    ca=fig.add_axes([.91,.225,.013,.14]);fig.colorbar(im,cax=ca);ca.tick_params(labelsize=9)
    fig.text(.045,.115,'Both images have center (14.5,14.5), width 3.2 px, and lean 30°. '
        'Their five fixed root measurements are exactly identical for both SVD and PCA.',fontsize=10)
    fig.text(.045,.076,f"Yet the full images differ by {result['fixed_frame_collision']['svd']['true_image_distance']:.4f} in 784-pixel Euclidean distance. "
        'No function of those same five fixed numbers can distinguish them.',fontsize=10)
    fig.text(.045,.035,'Coverage images: 28 × 28, white = 0 ink, black = 1. Difference uses signed coverage change. '
        'A fixed root frame fails globally; moving charts or a nonlinear global chart remain possible.',fontsize=9.1)
    fig.savefig(OUT/'local_charts_and_global_limit.png',dpi=150)
    plt.close(fig)


def report(result,a):
    rows=result['cases'];lines=['# From estimated SVD/PCA vectors to interpretable local manifold functions','',
    '**Result: the estimated vectors were clean enough to build accurate local coordinates after calibration and nonlinear correction. They did not have to be pure physical knobs. Centered PCA did not materially improve the calibrated result.**','',
    'Three deliberately calibrated local regions, including a tilted boundary, supported a unique inverse from five estimated measurements and image reconstruction/joint original-knob actions at numerical precision. This is a local construction assisted by the previously derived image measurements. It does not discover those measurements automatically or establish a compact atlas for the entire family.','',
    '## What entered the method','',
    '- Actual estimated basis vectors from ordinary local SVD or centered PCA; no true derivative vectors were substituted.\n'
    '- Observed 784-value images in the neighborhoods.\n'
    '- The explicit state functions from the prior experiment: top, bottom, centerline intercept, width, and slope. They receive pixels, not supplied current physical settings.\n'
    '- The established fact that this renderer is quadratic in those five quantities inside a fixed overlap formula cell, and its known cap/subrow boundaries.','',
    'Known physical settings and analytic renderer derivatives generate observations and evaluate accuracy. Analytic derivatives initialize image-distance sampling; measured image distances determine the actual samples. Neither true derivatives nor original settings are targets for the calibration or polynomial decoder fits. State-derived geometric values are calibration targets.','',
    '## Experiment 1: can mixtures of estimated axes recover original controls?','',
    'At each source image, measure how each estimated axis changes all five observable physical quantities. This gives a small five-by-five conversion matrix. Inverting it gives the combination of the five axes needed for a pure horizontal, vertical, height, width, or lean action. No bucket name or matching permutation is required by this conversion.','',
    'One associated vector still mixes physical effects. A weighted combination can remove the mixture, even though true physical directions are not mutually perpendicular. A correctly assigned bucket is therefore an initial category, not a finished physical knob.','',
    'The numerical check compares the reconstructed pixel vector with an independently calculated true physical derivative. Merely reproducing the desired observable readings would be insufficient because errors could lie outside those measurements.','',
    '| Root, radius 0.001 | SVD | Centered PCA |','| --- | ---: | ---: |']
    svd=one(rows,'root',.001,'svd');pca=one(rows,'root',.001,'pca')
    for label,key in [('One assigned height axis: relative pixel-direction error','best_assigned_single_axis'),
                      ('Combine all five: height-direction error','calibrated_span')]:
        lines.append(f"| {label} | {svd[key]['per_knob_relative'][2]:.6%} | {pca[key]['per_knob_relative'][2]:.6%} |")
    lines += ['', 'The conversion removes mixing. It cannot by itself replace curvature directions missing from the retained five-vector span. The next experiment adds those as functions of the same five coordinates.','',
    '## Experiment 2: explicit functions of the five measured coordinates','',
    'Each SVD/PCA vector becomes a fixed weighted pixel measurement relative to the source image. The five resulting numbers are candidate local coordinates. They are functions of the current state, not new external inputs.','',
    'Fit an explicit image equation using the five numbers, their squares, and their pairwise products. There are twenty nonconstant terms. The fitted coefficients are fixed signed pixel patterns. This decoder fit uses images only; it does not call the known interval-overlap image equation. A separate polynomial maps the measured coordinates to geometric state quantities and then to the original physical settings.','',
    'The inverse control test starts with an image and requested original-knob increments, predicts the required five coordinates, and evaluates the fitted image equation. The held-out target image is used only for checking.','',
    '### Scope','',
    '- 61 saved neighborhoods at 17 starts, including radii 0.1, 0.01, 0.001 and selected smaller radii.\n'
    '- 32 additional starts across all five original ranges, each at radius 0.001.\n'
    '- Total: 93 neighborhoods at 49 distinct starting images, each processed by both methods.\n'
    '- Each frame uses 512 fitting neighbors. Each new chart uses 512 independent filled-neighborhood images and 256 independent checking images with joint changes.\n'
    '- SVD and PCA receive identical images. The PCA axis fit subtracts its training mean. Its ordinary linear reconstruction adds that mean; the polynomial charts use the source image as coordinate zero.\n'
    '- Two-sided local neighborhoods may extend position bounds left/up below 14, as in the source root experiments. The 32 fresh starts lie in the interior of the default five-knob box.','',
    '| Radius 0.001, 49 starts | SVD | Centered PCA |','| --- | ---: | ---: |']
    ss={m:next(s for s in result['summary'] if s['method']==m and s['radius']==.001) for m in ['svd','pca']}
    for label,key,part in [('Median of per-neighborhood worst joint-update errors','quadratic_graph','median'),
                           ('Worst joint-update error across all neighborhoods','quadratic_graph','worst'),
                           ('Median of per-neighborhood worst original-coordinate mapping errors','quadratic_mapping','median')]:
        lines.append(f"| {label} | {ss['svd'][key][part]:.8%} | {ss['pca'][key][part]:.8%} |")
    lines += ['', 'Image errors are relative to the requested image change, not the norm of the whole starting image and not retained variance. Coordinate mapping errors use original-range-normalized changes. Every case is saved, including failures.','',
    'A single quadratic does not handle every boundary. At the tilted cap-boundary start, its worst joint-update error remains around one percent at very small radii. Splitting only at top/bottom subrows fixes some cases but worsens other neighborhoods that also cross horizontal side/pixel boundaries or have too little data per piece. Those negative results remain in the archive.','',
    '## Experiment 3: can a calibrated five-coordinate chart be exact locally?','',
    'Instead of approximating the inverse coordinate conversion with another quadratic, identify the actual quadratic image equation in each local formula piece. Its coefficients come from fitting images and their measured geometric quantities. The original renderer is not used as the predictor.','',
    'Project that equation onto the five estimated SVD/PCA vectors. This produces five explicit quadratic equations relating the estimated coordinates to top, bottom, intercept, width, and slope. Invert those five equations within the calibrated local region. The returned geometric quantities then map to the original physical knobs using the previously derived formulas.','',
    'Held-out inference receives only the five estimated measurements and saved model coefficients. It does not receive the query image’s true geometry, original settings, or formula-piece label. It tries the fitted pieces and retains the unique solution satisfying the region conditions.','',
    'The straight-stroke overlap structure makes the forward equation exactly quadratic per piece. A full-rank quadratic design identifies its coefficients in exact arithmetic. A derivative bound below one after linear normalization establishes a locally one-to-one projection throughout each calibrated convex box, including across its continuous cap joins. The numerical bound checks every box corner for every polynomial piece, since each derivative is affine and its spectral norm is convex. This uses the known quadratic structure, not just finite held-out success.','',
    '| Region | Method | Polynomial pieces | Unique inverses / 512 | Worst reconstructed pixel error | Worst joint-action pixel error |',
    '| --- | --- | ---: | ---: | ---: | ---: |']
    for m in result['algebraic_calibrated_charts']:
        lines.append(f"| {m['name']} | {m['method'].upper()} | {m['pieces']} | 512 | {m['reconstruction']['worst_pixel']:.3g} | {m['closed_loop_joint_actions']['worst_pixel']:.3g} |")
    lines += ['', 'Each region uses 1,024 fitting images and 512 new checking images shared between SVD and PCA. Each method separately performs 256 closed-loop joint original-knob actions per region. The smooth example uses one piece; the root and tilted cap boundary require four. The regions deliberately exclude side/pixel formula crossings while allowing cap crossings. They are small, not the entire default box.','',
    'Original settings recovered through the five-score inverse agree with the true settings to at most about 1.1e-14 in their declared pixel/degree units. Some estimated axes are heavily mixed: the smallest assigned cosine is about 0.42 at the tilted boundary, yet the whole five-coordinate map remains usable. Accuracy of each individual knob label is not the relevant requirement for coordinate invertibility.','',
    '### The actual mapping to original controls','',
    '| Original physical quantity | From the calibrated intermediate quantities |','| --- | --- |',
    '| Vertical center | Midpoint of top and bottom |','| Height | Bottom minus top |','| Width | Same width |',
    '| Lean angle | Arctangent of slope, in degrees |','| Horizontal center | Intercept minus slope times vertical center |','',
    'Infinitesimal conversion is a five-by-five mixing matrix. Finite conversion generally requires the quadratic relations and their locally unique inverse, not one global constant matrix or a one-to-one axis label.','',
    'The exported-model check reproduces inference without hidden query settings, rotates the five basis vectors, and replaces them with a fresh independent 64-neighbor frame without refitting the image coefficients. It also applies all five original controls in two orders inside each region. All these checks pass at numerical precision. This is not proof that arbitrary noisy or rank-deficient vectors will work; the calibrated projection must remain one-to-one and sufficiently conditioned.','',
    '## Experiment 4: compact rules for direction changes','',
    'At 64 independent states inside one certified smooth cell, obtain fresh noisy SVD/PCA frames, unmix them using image observables, and fit each resulting geometric direction as an affine function of the five state-derived quantities. Fit on 32 states and predict true directions at 32 new states.','',
    '| Method | Worst predicted-direction relative error, all five families |','| --- | ---: |']
    for m in result['same_cell_field_compression']:
        e=max(f['affine_field_prediction_worst_relative'] for f in m['family_results'])
        lines.append(f"| {m['method'].upper()} | {e:.6%} |")
    lines += ['', 'Two dominant variation patterns per family survive this estimated-vector route. The tiny residual singular directions include finite-neighborhood estimation error. The affine prediction is checked against held-out true directions; merely projecting a target onto a learned span is reported separately and is not treated as prediction.','',
    '## Global limit demonstrated by an exact collision','',
    'Keep the root’s five measured vectors fixed. Two valid images share center (14.5,14.5), width 3.2 and lean 30 degrees, but have heights 19.6 and 19.8. Their changing pixels lie outside the support observed by the root vectors.','',
    'For both SVD and PCA their five measurements are exactly identical, while their full-image Euclidean distance is about 0.2362. Consequently no decoder or original-knob mapping based only on those same five fixed measurements can represent the whole family. Moving charts, chart transitions, or other full-state-derived measurements are necessary. This counterexample does not rule out a nonlinear global five-coordinate chart.','',
    '## What the experiment establishes','',
    'The earlier estimated vectors are useful starting coordinates. They need not equal the original knobs. Image-derived measurements can identify their mixtures, and explicit nonlinear equations can turn those coordinates into accurate local image/control functions. SVD versus PCA is a secondary choice here; calibrated information content and boundary handling matter more.','',
    'The geometry-assisted local charts are exact in the declared regions under the established piecewise-quadratic renderer structure, apart from numerical evaluation. The broad automatic quadratic fits are approximations and retain the saved failures. No compact, globally complete, automatically discovered manifold representation is claimed. The semantic state readers and renderer-boundary structure were supplied by the preceding geometric analysis, not discovered from vector-angle buckets alone.','',
    '![Unmixing estimated vectors](unmixing_estimated_vectors.png)','',
    '![Local calibrated charts and global limit](local_charts_and_global_limit.png)','',
    '## Saved evidence','',
    '- `results.json`: all 186 method/neighborhood comparisons, six calibrated charts, field-compression checks, and independent verification.\n'
    '- `models_and_examples.npz`: representative frames, conversion matrices, fixed polynomial coefficients, model regions, and held-out predictions.\n'
    '- `root_axis_to_physical.csv`: the actual root five-axis mixing coefficients per original physical knob.','',
    'The analysis, exported-model verification, and figure/report generation are in `experiments/svd_state_coordinate_bridge.py`, `experiments/verify_svd_state_coordinate_bridge.py`, and `experiments/report_svd_state_coordinate_bridge.py`. All paths are repository relative.','']
    (OUT/'README.md').write_text('\n'.join(lines))
    # Directly inspectable five-axis physical conversion, with no derivative target.
    with (OUT/'root_axis_to_physical.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['method','physical_control','unit']+[f'estimated_axis_{i+1}_weight' for i in range(5)])
        for method in ['svd','pca']:
            prefix='root_0.001_'+method;b=a[prefix+'_basis'];_,_,_,dp,_=readers(a[prefix+'_source_image'])
            weights=np.linalg.inv(dp@b.T)
            for i,name in enumerate(['horizontal center','vertical center','height','width','lean']):
                writer.writerow([method,name,'degree' if i==4 else 'pixel']+weights[:,i].tolist())


def main():
    result=json.loads((OUT/'results.json').read_text());a=dict(np.load(OUT/'models_and_examples.npz'))
    direction_figure(result,a);chart_figure(result,a);report(result,a)
    print('Saved two explanatory figures, the English report, and actual conversion coefficients.')


if __name__=='__main__':main()
