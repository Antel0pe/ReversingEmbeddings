"""Explain fitted generated-one principal curves, with figures and a local explorer."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from analyze_generated_one_principal_curves import OUT, project
from grey_ones import KNOBS, RANGES, render

BLUE, ORANGE, INK = '#176da0', '#bc5a22', '#203343'
plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                     'axes.labelcolor':INK,'text.color':INK,'axes.titlecolor':INK})


def image(ax, values, title='', caption=''):
    im=np.asarray(values).reshape(28,28)
    ax.imshow(im,cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
    bad=(im < -1e-6)|(im > 1+1e-6)
    overlay=np.zeros((28,28,4)); overlay[bad]=[.78,.15,.66,.65]
    ax.imshow(overlay,interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_color('#cbd3d9')
    if title: ax.set_title(title,fontsize=10,pad=8)
    if caption: ax.text(.5,-.12,caption,ha='center',va='top',transform=ax.transAxes,fontsize=9)


def arc_at(c,t):
    lengths=np.linalg.norm(np.diff(c,axis=0),axis=1)
    s=np.r_[0,np.cumsum(lengths)]; s/=s[-1]
    return np.column_stack([np.interp(np.atleast_1d(t),s,col) for col in c.T])


def main_curve_figure(s,r,sensitivity):
    mean,c=s['mean'],s['curves'][0]
    closest=s['nearest_renderer_knobs']; valid=render(closest).reshape(-1,784)
    inds=np.linspace(0,64,7).round().astype(int)
    fig=plt.figure(figsize=(15.8,10.2),facecolor='white')
    fig.text(.045,.970,'What images lie along the first principal curve?',fontsize=20,weight='bold',va='top')
    fig.text(.045,.925,'Each column advances along one fitted path in 784 pixel values. White = no ink; black = full ink.',fontsize=11)
    grid=fig.add_gridspec(2,7,left=.175,right=.96,top=.855,bottom=.51,wspace=.35,hspace=.68)
    fig.text(.045,.790,'Allowed stroke\nfitted to the\ncurve point',fontsize=11,va='center',weight='bold')
    fig.text(.045,.588,'Principal curve\n(learned pixel\nfit)',fontsize=11,va='center',weight='bold',color=BLUE)
    for j,i in enumerate(inds):
        p=closest[i]
        image(fig.add_subplot(grid[0,j]),valid[i],f'{i/64:.0%} along path',f'lean {p[4]:.1f}°\nwidth {p[3]:.2f} px')
        image(fig.add_subplot(grid[1,j]),mean+c[i],caption=f'L2 to fitted stroke\n{r["first_curve_nearest_renderer_L2"][i]:.2f}')
    bottom=fig.add_gridspec(1,2,left=.085,right=.95,top=.405,bottom=.13,wspace=.28)
    ax=fig.add_subplot(bottom[0,0])
    test_scores=(s['test_images']-mean)@s['basis'][:2].T
    ax.scatter(test_scores[::3,0],test_scores[::3,1],s=3,color='#aab6bf',alpha=.35,label='Unseen generated 1s')
    for cc,color,label in [(c,BLUE,'Flexible: 89.9% captured'),(sensitivity['curves'][3],ORANGE,'Smoother: 74.1% captured')]:
        xy=cc@s['basis'][:2].T
        ax.plot(xy[:,0],xy[:,1],color=color,lw=1.7,label=label)
    ax.set_xlabel('PC1 score (pixel-space units)');ax.set_ylabel('PC2 score (pixel-space units)')
    ax.set_title('A projected view: this plane shows only 63.8% of variation',fontsize=11,pad=12)
    ax.legend(fontsize=8,loc='lower center');ax.grid(alpha=.15)
    ax=fig.add_subplot(bottom[0,1])
    t=np.linspace(0,1,65)
    for j,color,label in [(0,'#686b73','Horizontal center'),(3,BLUE,'Width'),(4,ORANGE,'Lean')]:
        ax.plot(100*t,(closest[:,j]-RANGES[j,0])/np.ptp(RANGES[j]),color=color,lw=1.8,label=label)
    ax.set_ylim(-.05,1.05);ax.set_xlabel('Position along flexible curve (%)')
    ax.set_ylabel('Knob value as fraction of its allowed range')
    ax.set_title('Overall lean progression, with width / center zigzags',fontsize=11,pad=12)
    ax.legend(fontsize=8,loc='upper left');ax.grid(alpha=.15)
    fig.text(.045,.055,'Fitted strokes are approximate nearest allowed renderer images, not proof of exact curve membership.\n'
             'Magenta marks pixels outside [0, 1] by more than 0.000001; those values are clipped in the grayscale display.',fontsize=10,linespacing=1.6)
    fig.savefig(OUT/'first_curve.png',dpi=145);plt.close(fig)


def variance_figure(r):
    fig=plt.figure(figsize=(14.5,7.3),facecolor='white')
    fig.text(.055,.96,'How much image variation do the curves explain?',fontsize=20,weight='bold',va='top')
    fig.text(.055,.895,'4,096 training + 2,048 validation + 4,096 untouched test images; independent uniform sampling of all five knobs.',fontsize=10.5)
    gs=fig.add_gridspec(1,2,left=.08,right=.96,top=.78,bottom=.27,wspace=.28)
    ax=fig.add_subplot(gs[0,0]);xx=np.arange(1,6)
    pc=[s['explained_percent'] for s in r['pca_cumulative'][:5]]
    cu=[s['explained_percent'] for s in r['curves_cumulative']]
    corner=[s['explained_percent'] for s in r['probes']['all_32_corners']['cumulative']]
    ax.plot(xx,pc,'o-',color='#707a83',lw=2,label='PCA, test images')
    ax.plot(xx,cu,'o-',color=BLUE,lw=2,label='Sequential curves, test images')
    ax.plot(xx,corner,'s--',color=ORANGE,lw=1.8,label='Same curves, all 32 corners')
    for x,y in zip(xx,cu): ax.annotate(f'{y:.1f}%',(x,y),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9,color=BLUE)
    for x,y in [(1,pc[0]),(5,pc[4])]: ax.annotate(f'{y:.1f}%',(x,y),xytext=(0,-17),textcoords='offset points',ha='center',fontsize=9)
    ax.axhline(95,color='#9ba2a9',lw=1,linestyle=':')
    ax.set_ylim(40,105);ax.set_xticks(xx);ax.set_xlabel('Coordinates kept (curve 2 onward fits the residual)')
    ax.set_ylabel('Variation explained by reconstruction (%)');ax.set_title('Flexible curves outperform linear coordinates',fontsize=12,pad=16)
    ax.legend(fontsize=9,loc='lower right');ax.grid(alpha=.15)
    ax=fig.add_subplot(gs[0,1])
    items=sorted(r['smoothing_sensitivity'],key=lambda q:q['length'])
    xs=[q['length'] for q in items];ys=[q['test']['explained_percent'] for q in items]
    ax.plot(xs,ys,'o-',color=BLUE,lw=2)
    for x,y in zip(xs,ys):ax.annotate(f'{y:.1f}%',(x,y),xytext=(0,9),textcoords='offset points',ha='center',fontsize=10)
    ax.axhline(pc[0],color='#707a83',linestyle='--',lw=1,label=f'PC1 baseline: {pc[0]:.1f}%')
    ax.axvline(r['exact_mid_knob_lean_path']['length'],color=ORANGE,linestyle=':',lw=1.5,label='Length of exact lean-only path')
    ax.set_ylim(40,99);ax.set_xlim(0,160);ax.set_xlabel('First-curve length (sum of pixel L2 segment lengths)')
    ax.set_ylabel('Test variation explained (%)');ax.set_title('Longer, more winding curves capture more',fontsize=12,pad=16)
    ax.legend(fontsize=9,loc='lower right');ax.grid(alpha=.15)
    fig.text(.055,.12,'Score = 100 × [1 − squared reconstruction error / squared distance to the training-mean image].\n'
             'Errors use all 784 pixels. Scores are sampling-dependent; one flexible curve has many more shape parameters than one PCA line.',fontsize=10.5,linespacing=1.6)
    fig.text(.055,.035,'Five curves capture 97.3% on test images, but 92.0% at the corners. Neither result establishes exact five-knob coordinates.',fontsize=11,weight='bold')
    fig.savefig(OUT/'variance_capture.png',dpi=150);plt.close(fig)


def residual_figure(s,r):
    positions=np.array([.1,.3,.5,.7,.9]);curves=s['curves'][1:]
    patches=[arc_at(c,positions)-arc_at(c,[.5]) for c in curves]
    vmax=np.ceil(max(np.max(abs(a)) for a in patches)*10)/10
    fig=plt.figure(figsize=(13.4,10.9),facecolor='white')
    fig.text(.055,.965,'What do the later curves change?',fontsize=20,weight='bold',va='top')
    fig.text(.055,.918,'Each panel is a change in residual coverage relative to that curve’s midpoint, not a standalone stroke.',fontsize=11)
    fig.text(.055,.882,'Red adds ink; blue removes ink; white = zero change. All panels share the same scale.',fontsize=10.5)
    gs=fig.add_gridspec(4,5,left=.24,right=.95,top=.81,bottom=.155,hspace=.62,wspace=.3)
    for k,patch in enumerate(patches):
        dominant=int(np.argmax(np.abs(r['curve_coordinate_knob_spearman'][k+1])))
        rho=r['curve_coordinate_knob_spearman'][k+1][dominant]
        y=.737-k*.177
        fig.text(.055,y,f'Curve {k+2}\nadds {r["marginal_percentage_points"][k+1]:.2f} percentage points\n'
                 f'largest knob association:\n{KNOBS[dominant]}, rank correlation {rho:.2f}',fontsize=10.5,va='center',linespacing=1.65)
        for j,im in enumerate(patch):
            ax=fig.add_subplot(gs[k,j]);ax.imshow(im.reshape(28,28),cmap='RdBu_r',vmin=-vmax,vmax=vmax,interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(True);spine.set_color('#cbd3d9')
            ax.set_title(f'{positions[j]:.0%} along curve',fontsize=9,pad=6)
            ax.text(.5,-.1,f'L2 change {np.linalg.norm(im):.2f}',transform=ax.transAxes,ha='center',va='top',fontsize=9)
    cax=fig.add_axes([.37,.092,.49,.016]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-vmax,vmax),cmap='RdBu_r'),cax=cax,orientation='horizontal')
    cb.set_label('Change in ink coverage per pixel (negative = remove; positive = add)',fontsize=10)
    fig.text(.055,.018,'Associations use 4,096 held-out images. The later curves mix knobs; they are not width, center, height, and lean axes.',fontsize=10.5)
    fig.savefig(OUT/'residual_curves.png',dpi=145);plt.close(fig)


def report(r):
    lines=['# Principal curves of the generated-one family','',
        'This experiment fits curves directly to the 784 coverage values from `grey_ones.render`; it does not fit in a PCA projection. All five knobs vary independently and uniformly over `grey_ones.RANGES`.', '',
        '## Results on unseen images','',
        '| Coordinates | Sequential curves | Centered PCA |',
        '| --- | ---: | ---: |']
    for k,(c,p) in enumerate(zip(r['curves_cumulative'],r['pca_cumulative']),1):
        lines.append(f'| {k} | {c["explained_percent"]:.2f}% | {p["explained_percent"]:.2f}% |')
    lines += ['', 'The first curve is strongly associated with lean (held-out Spearman rank correlation 0.971), but it also snakes back and forth through width and horizontal position. Its length is 145.59 pixel-distance units, compared with 28.77 for an exact lean-only sweep at the other knobs’ midpoints. That flexibility is a major source of its high score. Its approximate closest allowed strokes range from thin to thick repeatedly; their heights remain near the midrange.', '',
        'Later curves fit the residual coverage after earlier projections. They mix width, position, and remaining lean-dependent patterns: the largest absolute single-knob correlations are only 0.30, 0.37, 0.21, and 0.07 for curves 2–5. They should not be interpreted as five clean physical knobs.', '',
        '## How this relates to PCA','',
        'PC1 is a straight line through the mean image. A principal curve replaces that straight line with a bendable path through image space. Each location on the path is a complete 28×28 image, and a sample is encoded by its nearest location on the path. The fit alternates nearest-point projection with a smooth fit of the pixel values against path position.', '',
        'The numerical implementation uses 65 polyline nodes and a second-difference smoothing penalty. Each projection searches every segment using all 784 pixels. Smoothing is a penalized linear spline, followed by arc-length reparameterization. These are finite-resolution, regularized principal-curve approximations, not a proof of exact self-consistency or a globally optimal curve.', '',
        'Curve 1 is a principal-curve fit; curves 2–5 are a sequential residual extension, not a canonical ordered list analogous to PC1–PC5. The encoder projects an image onto curve 1, subtracts that predicted image difference, projects the residual onto curve 2, and so on. The decoder adds the training mean and the five selected curve contributions. Independent choices of these coordinates need not give valid strokes. The curves are not required to be orthogonal and the later fit is order-dependent.', '',
        '## What “variance captured” means','',
        '`100 × (1 − sum of squared reconstruction errors / sum of squared distances to the training-mean image)`', '',
        'Every square is a coverage-value error across all 784 pixels. There are 4,096 training images, 2,048 validation images for smoothing/iteration selection, and 4,096 untouched test images, from independently scrambled Sobol sequences with seeds 731–733. Rendering is float32; fitting and scoring are float64. The PCA basis and mean are learned on the same training set. This experiment concerns the full default box, not the earlier radius-4.5 shell.', '',
        'A nonlinear curve has many more fitted shape parameters than a PCA line. Equal numbers of output coordinates do not mean equal model complexity. These nonlinear reconstruction scores do not have PCA’s eigenvalue decomposition or orthogonality guarantee.', '',
        '## Smoothing controls the answer','',
        '| Smoothing penalty | Curve length | Test variation captured |', '| ---: | ---: | ---: |']
    for a in r['smoothing_sensitivity']:
        lines.append(f'| {a["penalty"]:g} | {a["length"]:.2f} | {a["test"]["explained_percent"]:.2f}% |')
    lines += ['', 'All sequential fits selected penalty 1 from the tested values 1, 10, 100, 1000, using validation reconstruction error. Fits are capped at 35 iterations and retain the best validation iterate, so this is an approximate finite computation. The extended sensitivity check shows that a smoother, lean-like path of length 27.15 captures 74.06%, while a highly smoothed 16.67-long path captures 66.92%. There is no unique variance percentage independent of smoothing, node count, initialization, and the knob-sampling distribution.', '',
        f'A separate PC2-initialized first curve captures {r["alternate_PC2_start"]["cumulative"][0]["explained_percent"]:.2f}%, despite a very different coordinate association with lean (rank correlation 0.265 rather than 0.971). This is further evidence against treating the curve as a uniquely identified physical coordinate.', '',
        '## Boundaries, joint changes, and validity','',
        '| Probe set | Images | One curve | Five curves | Worst five-curve image L2 error |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name,probe in r['probes'].items():
        lines.append(f'| {name} | {probe["count"]} | {probe["cumulative"][0]["explained_percent"]:.2f}% | {probe["cumulative"][-1]["explained_percent"]:.2f}% | {probe["cumulative"][-1]["max_image_L2"]:.3f} |')
    lines += ['', 'The 5×5×5×5×5 lattice and the corners include boundary combinations that uniform sampling gives little weight. The joint-midpoint probe renders the parameter midpoint of two independent settings and encodes that image; it is not a test of linearly interpolating the curve coordinates. These probes test sampled reconstruction, not continuous coverage or collision-free coordinates.', '',
        f'The first curve has pixels from −0.1065 to 1.0786; the five-curve test reconstruction has pixels from {r["five_curve_pixel_min"]:.4f} to {r["five_curve_pixel_max"]:.4f}. The fit is unconstrained, so averaged or spline-interpolated images can leave the coverage cube or violate the parallelogram rule. Each plotted curve-point validity error is distance to an approximately fitted allowed renderer image; it is not a certificate of the globally nearest point. Across the 65 first-curve nodes those errors have median 0.373 and maximum 0.570.', '',
        'The five physical parameters plus the renderer remain the exact construction. The fitted curves are useful nonlinear summaries, not a replacement for that complete coordinate system.', '',
        '## Reproduce and inspect','',
        '```sh', '.venv/bin/python analyze_generated_one_principal_curves.py',
        '.venv/bin/python supplement_generated_one_principal_curves.py',
        '.venv/bin/python make_generated_one_principal_curve_figures.py', '```', '',
        '- `index.html`: offline explorer for curve shape, smoothing, and held-out reconstructions.',
        '- `first_curve.png`: actual curve images, approximately fitted valid strokes, and projected curve shape.',
        '- `variance_capture.png`: unseen-image scores, boundary scores, and smoothing tradeoff.',
        '- `residual_curves.png`: later curve changes, with a common signed coverage scale.',
        '- `results.json`: metrics and probe results.',
        '- `fit_history.json`: training/validation fitting history.',
        '- `model_and_samples.npz`: learned mean, PCA basis, curves, sample settings, and reconstructions.',
        '- `smoothing_sensitivity.npz`: first-curve smoothing controls and exact lean-only comparison path.', '',
        'Independent checks confirm that the vectorized nearest-segment search matches a direct segment loop, and that the sum decoder equals stored sequential reconstructions. Figure interpretation checks cover source object, scales, starting state, measurements, clipping, sampled scope, and hidden projection dimensions.', '',
        'Method reference: [Hastie and Stuetzle, Principal Curves (1989)](https://hastie.su.domains/Papers/Principal_Curves.pdf).', '']
    (OUT/'README.md').write_text('\n'.join(lines))


def explorer(s,r,sensitivity):
    mean=s['mean'];curves=s['curves']
    first=mean+curves[0];nearest=render(s['nearest_renderer_knobs']).reshape(-1,784)
    support=np.where(np.any(np.abs(curves)>1e-10,axis=(0,1))|(mean>0)|np.any(s['test_images']>0,axis=0)|np.any(nearest>0,axis=0))[0]
    ids=np.r_[np.arange(40),np.argsort(np.linalg.norm(s['test_images']-s['reconstructed'],axis=1))[-8:]]
    sample_imgs=s['test_images'][ids]
    residual=sample_imgs-mean;predictions=[]
    for c in curves:
        fitted,_,_=project(residual,c);residual-=fitted;predictions.append(sample_imgs-residual)
    def compact(x):return np.round(np.asarray(x),5).tolist()
    scores=(s['test_images'][::8]-mean)@s['basis'][:2].T
    data=dict(support=support.tolist(),mean=compact(mean[support]),
              first=compact(first[:,support]),nearest=compact(nearest[:,support]),knobs=compact(s['nearest_renderer_knobs']),
              errors=r['first_curve_nearest_renderer_L2'],
              samples=compact(sample_imgs[:,support]),predictions=compact(np.array(predictions)[:,:,support]),
              sampleKnobs=compact(s['test_knobs'][ids]),
              sensitivity=compact(np.array([mean+c for c in sensitivity['curves']])[:,:,support]),
              smoothScores=[a['test']['explained_percent'] for a in r['smoothing_sensitivity']],
              smoothLengths=[a['length'] for a in r['smoothing_sensitivity']],
              planeSamples=compact(scores),planeCurves=compact(sensitivity['curves']@s['basis'][:2].T),
              curveVariance=[a['explained_percent'] for a in r['curves_cumulative']],
              pcaVariance=[a['explained_percent'] for a in r['pca_cumulative'][:5]])
    template='''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Generated 1 principal curves</title>
<style>
:root{color-scheme:light dark;--bg:#fafbfc;--fg:#203343;--muted:#52616e;--line:#d9e0e6;--blue:#176da0;--orange:#bc5a22}@media(prefers-color-scheme:dark){:root{--bg:#141a20;--fg:#e8edf1;--muted:#bac6ce;--line:#42515e;--blue:#72b9e8;--orange:#efad7a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}main{max-width:1100px;margin:32px auto;padding:0 24px}h1{font-size:28px;line-height:1.2}h2{font-size:20px;margin:0 0 16px}p{max-width:960px;color:var(--muted)}section{padding:24px 0;border-top:1px solid var(--line)}.controls{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin:16px 0}.controls input{min-width:180px;flex:1;accent-color:var(--blue)}select,button{font:inherit;color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:6px 10px}.images{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px}figure{margin:0}figcaption{font-size:14px;margin:7px 0 0;color:var(--muted)}canvas.pixel{width:100%;max-width:250px;aspect-ratio:1;image-rendering:pixelated;border:1px solid var(--line);display:block}canvas#plane{width:100%;height:330px;border:1px solid var(--line)}.metric{font-variant-numeric:tabular-nums;color:var(--fg)}table{border-collapse:collapse;width:100%;max-width:650px}th,td{padding:9px 14px;text-align:right;border-bottom:1px solid var(--line)}th:first-child,td:first-child{text-align:left}.note{font-size:14px}a{color:var(--blue)}@media(max-width:650px){main{padding:0 16px}.images{gap:10px}h1{font-size:24px}figcaption{font-size:12px}.controls{gap:9px}canvas#plane{height:300px}}
</style></head><body><main>
<h1>Principal curves of the generated 1 family</h1>
<p>One image is 784 ink-coverage values. PCA summarizes the cloud with straight directions; a principal curve is a bendable path through it. These fits sample all five knobs over their default allowed ranges.</p>
<section><h2>Walk the first fitted curve</h2>
<div class="controls"><label for="position">Position along curve</label><input id="position" type="range" min="0" max="64" step="1" value="32"><output id="positionValue" class="metric"></output></div>
<div class="images"><figure><canvas class="pixel" id="valid" width="28" height="28"></canvas><figcaption>Approximately fitted allowed stroke</figcaption></figure><figure><canvas class="pixel" id="curve" width="28" height="28"></canvas><figcaption>Actual fitted principal-curve image</figcaption></figure><figure><canvas class="pixel" id="difference" width="28" height="28"></canvas><figcaption>Curve minus allowed stroke<br>Red adds ink; blue removes ink; scale −1 to +1.</figcaption></figure></div>
<p id="curveDetails" class="metric"></p><p class="note">Coverage: white = 0, black = 1. Magenta marks values outside [0, 1] by more than 0.000001. The allowed stroke is an approximate closest fit, not an exact match.</p>
</section>
<section><h2>Smoothing changes the curve and its score</h2>
<div class="controls"><label for="smooth">Fit</label><select id="smooth"><option value="0">Flexible</option><option value="1">Moderately smoothed</option><option value="2">Smoothed</option><option value="3">Smoother, lean-like</option><option value="4">Highly smoothed</option></select><output id="smoothDetails" class="metric"></output></div>
<canvas id="plane" aria-label="Projection of generated images and a fitted curve onto PC1 and PC2"></canvas>
<p class="note">Gray points: unseen generated images. Blue path: selected smoothing. Orange dot: current position. This PCA plane shows 63.8% of test variation; fitting and scoring used all 784 pixels.</p>
</section>
<section><h2>Reconstruct unseen images with successive curves</h2>
<div class="controls"><label for="sample">Test image</label><input id="sample" type="range" min="0" max="47" step="1" value="0"><label for="count">Curves kept</label><select id="count"><option>1</option><option>2</option><option>3</option><option>4</option><option selected>5</option></select></div>
<div class="images"><figure><canvas class="pixel" id="original" width="28" height="28"></canvas><figcaption>Unseen generated image</figcaption></figure><figure><canvas class="pixel" id="reconstruction" width="28" height="28"></canvas><figcaption>Sequential curve reconstruction</figcaption></figure><figure><canvas class="pixel" id="error" width="28" height="28"></canvas><figcaption>Reconstruction minus original<br>Red adds ink; blue removes ink; scale −1 to +1.</figcaption></figure></div>
<p id="sampleDetails" class="metric"></p><p class="note">Examples 1–40 are the first test samples; 41–48 are the worst five-curve reconstructions in the test set. Curves 2–5 fit residual pixel differences, so they mix physical knobs.</p>
</section>
<section><h2>Variation explained on all 4,096 test images</h2><table><thead><tr><th>Coordinates kept</th><th>Sequential curves</th><th>PCA</th></tr></thead><tbody id="scores"></tbody></table>
<p class="note">Score = 100 × (1 − squared reconstruction error / squared distance to the training-mean image). One flexible curve has many more shape parameters than one PCA line. Five curves capture 97.3% here, but 92.0% at all 32 range-box corners. The curves are summaries, not exact five-knob coordinates.</p>
<p><a href="README.md">Full analysis and method</a> · <a href="results.json">Measured results</a></p></section>
</main><script id="curve-data" type="application/json">DATA_PLACEHOLDER</script>
<script>
(()=>{
'use strict';const D=JSON.parse(document.getElementById('curve-data').textContent);const $=id=>document.getElementById(id);
function ink(canvas,values,signed=false){const ctx=canvas.getContext('2d');const im=ctx.createImageData(28,28);for(let i=0;i<784;i++){im.data[4*i]=255;im.data[4*i+1]=255;im.data[4*i+2]=255;im.data[4*i+3]=255;}D.support.forEach((idx,j)=>{const v=values[j];let rgb;if(signed){const a=Math.min(1,Math.abs(v));rgb=v>=0?[255,Math.round(255*(1-a)),Math.round(255*(1-a))]:[Math.round(255*(1-a)),Math.round(255*(1-a)),255];}else if(v < -1e-6 || v > 1+1e-6){rgb=[199,38,168];}else{const g=Math.round(255*(1-Math.max(0,Math.min(1,v))));rgb=[g,g,g];}rgb.forEach((c,k)=>im.data[4*idx+k]=c);});ctx.putImageData(im,0,0);}
function plane(){const canvas=$('plane');const rect=canvas.getBoundingClientRect();const w=Math.max(320,rect.width),h=Math.max(260,rect.height);const ratio=window.devicePixelRatio||1;canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio);const ctx=canvas.getContext('2d');ctx.scale(ratio,ratio);const style=getComputedStyle(document.documentElement);const fg=style.getPropertyValue('--fg'), blue=style.getPropertyValue('--blue'),orange=style.getPropertyValue('--orange'),line=style.getPropertyValue('--line');ctx.fillStyle=fg;ctx.strokeStyle=line;const all=D.planeSamples.concat(...D.planeCurves);const minX=Math.min(...all.map(p=>p[0]))-.3,maxX=Math.max(...all.map(p=>p[0]))+.3,minY=Math.min(...all.map(p=>p[1]))-.3,maxY=Math.max(...all.map(p=>p[1]))+.3;const x=v=>60+(v-minX)/(maxX-minX)*(w-85),y=v=>h-46-(v-minY)/(maxY-minY)*(h-72);ctx.strokeRect(60,26,w-85,h-72);ctx.font='13px system-ui';ctx.textAlign='center';ctx.fillText('PC1 score (pixel-space units)',w/2,h-8);ctx.save();ctx.translate(16,h/2);ctx.rotate(-Math.PI/2);ctx.fillText('PC2 score (pixel-space units)',0,0);ctx.restore();for(let i=0;i<=3;i++){let xx=minX+i/3*(maxX-minX),yy=minY+i/3*(maxY-minY);ctx.fillText(xx.toFixed(1),x(xx),h-28);ctx.textAlign='right';ctx.fillText(yy.toFixed(1),54,y(yy)+4);ctx.textAlign='center';}ctx.fillStyle=line;D.planeSamples.forEach(p=>{ctx.beginPath();ctx.arc(x(p[0]),y(p[1]),1.7,0,2*Math.PI);ctx.fill();});const c=D.planeCurves[Number($('smooth').value)];ctx.strokeStyle=blue;ctx.lineWidth=2;ctx.beginPath();c.forEach((p,i)=>{if(i)ctx.lineTo(x(p[0]),y(p[1]));else ctx.moveTo(x(p[0]),y(p[1]));});ctx.stroke();const p=c[Number($('position').value)];ctx.fillStyle=orange;ctx.beginPath();ctx.arc(x(p[0]),y(p[1]),5,0,2*Math.PI);ctx.fill();}
function update(){const i=Number($('position').value),sm=Number($('smooth').value),sid=Number($('sample').value),k=Number($('count').value)-1;const p=D.knobs[i];$('positionValue').textContent=(100*i/64).toFixed(0)+'%';ink($('valid'),D.nearest[i]);ink($('curve'),D.first[i]);ink($('difference'),D.first[i].map((v,j)=>v-D.nearest[i][j]),true);$('curveDetails').textContent='Fitted stroke: lean '+p[4].toFixed(1)+'°, width '+p[3].toFixed(2)+' px, center ('+p[0].toFixed(2)+', '+p[1].toFixed(2)+') px. Curve-to-stroke L2 error '+D.errors[i].toFixed(3)+'.';$('smoothDetails').textContent=D.smoothScores[sm].toFixed(1)+'% captured · path length '+D.smoothLengths[sm].toFixed(1);const original=D.samples[sid],rec=D.predictions[k][sid];ink($('original'),original);ink($('reconstruction'),rec);ink($('error'),rec.map((v,j)=>v-original[j]),true);const e=Math.sqrt(rec.reduce((sum,v,j)=>sum+(v-original[j])**2,0));const q=D.sampleKnobs[sid];$('sampleDetails').textContent='Image '+(sid+1)+' of 48 · lean '+q[4].toFixed(1)+'°, width '+q[3].toFixed(2)+' px, height '+q[2].toFixed(2)+' px · '+(k+1)+' curves: image L2 error '+e.toFixed(3);plane();}
$('scores').innerHTML=D.curveVariance.map((v,i)=>'<tr><td>'+(i+1)+'</td><td>'+v.toFixed(2)+'%</td><td>'+D.pcaVariance[i].toFixed(2)+'%</td></tr>').join('');['position','smooth','sample','count'].forEach(id=>$(id).addEventListener('input',update));window.addEventListener('resize',plane);update();
})();
</script></body></html>'''
    (OUT/'index.html').write_text(template.replace('DATA_PLACEHOLDER',json.dumps(data,separators=(',',':'))))


def main():
    s=np.load(OUT/'model_and_samples.npz');r=json.loads((OUT/'results.json').read_text())
    sensitivity=np.load(OUT/'smoothing_sensitivity.npz')
    main_curve_figure(s,r,sensitivity);variance_figure(r);residual_figure(s,r);report(r);explorer(s,r,sensitivity)
    print('Saved 3 explanatory figures, README, and offline explorer.')


if __name__=='__main__':main()
