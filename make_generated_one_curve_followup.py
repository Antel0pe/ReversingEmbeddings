"""Literal curve definitions, residual explanations, and exact knob-path comparison."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from analyze_generated_one_principal_curves import OUT, pixels
from make_generated_one_principal_curve_figures import arc_at, image, BLUE, ORANGE, INK
from grey_ones import KNOBS, RANGES, render

F=OUT/'followup'
BASE=RANGES.mean(axis=1)


def literal_figure(s):
    ts=np.array([0,.25,.5,.75,1]);nodes=s['curves'];vmax=.7
    fig=plt.figure(figsize=(13.5,13.1),facecolor='white')
    fig.text(.055,.969,'The five fitted curves are lists of pixel-valued nodes',fontsize=20,weight='bold',va='top')
    fig.text(.055,.923,'65 stored nodes per curve; each node contains 784 pixel values. Between nodes, use straight interpolation.',fontsize=11)
    fig.text(.055,.89,'Curve 1 below includes the mean image. Curves 2–5 below are signed contributions added to the reconstruction.',fontsize=10.5)
    gs=fig.add_gridspec(5,5,left=.22,right=.95,top=.835,bottom=.155,hspace=.55,wspace=.3)
    for k,c in enumerate(nodes):
        values=arc_at(c,ts)
        fig.text(.055,.777-k*.142,f'Curve {k+1}\n'+('mean + C1(t)\nfull image' if k==0 else f'C{k+1}(t)\npixel correction'),fontsize=11,weight='bold',va='center',linespacing=1.5)
        for j,v in enumerate(values):
            ax=fig.add_subplot(gs[k,j])
            if k==0:
                image(ax,s['mean']+v,f't = {ts[j]:.2f}',f'contribution L2 {np.linalg.norm(v):.2f}')
            else:
                ax.imshow(v.reshape(28,28),cmap='RdBu_r',vmin=-vmax,vmax=vmax,interpolation='nearest')
                ax.set_xticks([]);ax.set_yticks([])
                for spine in ax.spines.values():spine.set_visible(True);spine.set_color('#cbd3d9')
                ax.set_title(f't = {ts[j]:.2f}',fontsize=10,pad=8)
                ax.text(.5,-.12,f'contribution L2 {np.linalg.norm(v):.2f}',ha='center',va='top',transform=ax.transAxes,fontsize=9)
    ca=fig.add_axes([.32,.10,.57,.014]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-vmax,vmax),cmap='RdBu_r'),cax=ca,orientation='horizontal')
    cb.set_label('Curves 2–5: red adds ink; blue removes ink; white = zero. Same scale for every panel.',fontsize=10)
    fig.text(.055,.023,'t is normalized path length, not a physical knob. Curve 1: white = 0 ink; black = 1; magenta marks values outside [0, 1].\n'
             'Decoder: mean image + C1(t1) + C2(t2) + C3(t3) + C4(t4) + C5(t5). This is the original fitted model.',fontsize=10.5,linespacing=1.65)
    fig.savefig(F/'literal_five_curves.png',dpi=145);plt.close(fig)


def missing_figure(s,r):
    err=s['test_images']-s['reconstructed'];norm=np.linalg.norm(err,axis=1)
    idx=np.argsort(norm)[[len(norm)//4,len(norm)//2,-1]]
    fig=plt.figure(figsize=(13.1,10.4),facecolor='white')
    fig.text(.055,.969,'What is in the original missing 2.7%?',fontsize=20,weight='bold',va='top')
    fig.text(.055,.923,'Missing = exact generated image minus its five-curve reconstruction. Positive means the reconstruction missed ink.',fontsize=10.5)
    gs=fig.add_gridspec(3,4,left=.065,right=.95,top=.83,bottom=.19,wspace=.35,hspace=.45)
    for row,i in enumerate(idx):
        titles=['Exact source image','Original five-curve fit','Missing ink pattern'] if row==0 else ['','','']
        image(fig.add_subplot(gs[row,0]),s['test_images'][i],titles[0],f'lean {s["test_knobs"][i,4]:.1f}°, width {s["test_knobs"][i,3]:.2f} px')
        image(fig.add_subplot(gs[row,1]),s['reconstructed'][i],titles[1],f'image L2 error {norm[i]:.3f}')
        ax=fig.add_subplot(gs[row,2]);ax.imshow(err[i].reshape(28,28),cmap='RdBu_r',vmin=-1,vmax=1,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(titles[2],fontsize=10,pad=8)
        ax.text(.5,-.12,f'max pixel error {np.max(abs(err[i])):.3f}',transform=ax.transAxes,ha='center',va='top',fontsize=9)
    ax=fig.add_subplot(gs[0,3]);ax.imshow(np.sqrt(np.mean(err**2,axis=0)).reshape(28,28),cmap='magma',vmin=0,vmax=.12);ax.set_xticks([]);ax.set_yticks([])
    ax.set_title('All 4,096 source images\nRMS pixel error, 0 to 0.12',fontsize=10,pad=8)
    heat_pos=ax.get_position()
    heat_ca=fig.add_axes([heat_pos.x0,heat_pos.y0-.025,heat_pos.width,.011])
    heat_cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0,.12),cmap='magma'),cax=heat_ca,orientation='horizontal',ticks=[0,.06,.12])
    heat_cb.ax.tick_params(labelsize=8)
    ax=fig.add_subplot(gs[1,3]);v=r['original_missing_error']['error_regions_percent'];names=['Top / bottom bands','Side-edge bands','Remaining pixels']
    ax.barh(names,list(v.values()),color=[ORANGE,BLUE,'#808990']);ax.invert_yaxis();ax.set_xlim(0,65);ax.set_xlabel('Share of squared error (%)',fontsize=9)
    ax.tick_params(labelsize=8)
    bar_pos=ax.get_position();ax.set_position([bar_pos.x0+.045,bar_pos.y0,bar_pos.width-.045,bar_pos.height])
    ax.set_xlabel('Squared error share (%)',fontsize=8)
    for i,y in enumerate(v.values()):ax.text(y+1,i,f'{y:.1f}%',va='center',fontsize=9)
    ax=fig.add_subplot(gs[2,3]);ax.axis('off');ax.text(0,.98,'Typical pixel RMS error: 0.027\n\nThe error is concentrated\nalong boundaries, not in\nanother hidden stroke.\n\nRows: lower quartile, median,\nand worst image error.',fontsize=10.5,va='top',linespacing=1.5)
    ca=fig.add_axes([.12,.103,.52,.015]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-1,1),cmap='RdBu_r'),cax=ca,orientation='horizontal');cb.set_label('Missing coverage: blue = excess reconstructed ink; red = missing reconstructed ink',fontsize=9.5)
    fig.text(.055,.022,'Top / bottom bands include each endpoint row and its immediate neighboring rows. Side bands are within 1.5 pixels of an edge.\n'
             'Coverage display: white = no ink, black = full ink; magenta marks fitted values outside [0, 1]. This diagnoses the original model.',fontsize=10,linespacing=1.6)
    fig.savefig(F/'missing_variation.png',dpi=150);plt.close(fig)


def joint_figure():
    p=np.tile(BASE,(4,1));p[1,3]+=1;p[2,4]+=15;p[3,3]+=1;p[3,4]+=15
    x=pixels(p);summed=x[1]+x[2]-x[0];missing=x[3]-summed
    fig=plt.figure(figsize=(12.8,8.1),facecolor='white')
    fig.text(.055,.965,'Why five fixed knob curves cannot simply be added',fontsize=20,weight='bold',va='top')
    fig.text(.055,.906,'Start: center (14.5, 14.5) px, height 19.75 px, width 3.2 px, lean 12.5°. Every panel uses the same pixels.',fontsize=10.5)
    gs=fig.add_gridspec(2,3,left=.10,right=.92,top=.81,bottom=.20,wspace=.45,hspace=.40)
    panels=[(x[0],'Common starting stroke','width 3.2 px; lean 12.5°'),
            (x[1],'Change width only','width 4.2 px; lean 12.5°'),
            (x[2],'Change lean only','width 3.2 px; lean 27.5°'),
            (summed,'Add the two separate pixel changes',f'L2 error {np.linalg.norm(missing):.3f}'),
            (x[3],'Render the actual joint change','width 4.2 px; lean 27.5°; L2 error 0'),
            (missing,'Interaction missing from that sum',f'max pixel difference {np.max(abs(missing)):.3f}')]
    for i,(im,title,caption) in enumerate(panels):
        ax=fig.add_subplot(gs[i//3,i%3])
        if i<5:image(ax,im,title,caption)
        else:
            ax.imshow(im.reshape(28,28),cmap='RdBu_r',vmin=-1,vmax=1,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([]);ax.set_title(title,fontsize=10,pad=8)
            ax.text(.5,-.12,caption,transform=ax.transAxes,ha='center',va='top',fontsize=9)
    fig.text(.055,.095,'Red = joint change needs more ink; blue = joint change needs less ink; scale −1 to +1 coverage.\n'
             'Magenta marks values outside [0, 1]. The missing pattern is an interaction, not a new knob.',fontsize=10.5,linespacing=1.6)
    fig.text(.055,.025,'Exact solution: first change lean, then follow the width curve at that new lean (or reverse the order). The endpoint agrees.',fontsize=11,weight='bold')
    fig.savefig(F/'knob_curve_interaction.png',dpi=150);plt.close(fig)
    return dict(starting_knobs=BASE.tolist(),width_step=1.,lean_step_degrees=15.,
                mixed_pixel_L2=float(np.linalg.norm(missing)),max_pixel_difference=float(np.max(abs(missing))))


def write_report(r,refine,interaction):
    lines=['# What the five curves actually are','',
    'The target is the known renderer family, not a prediction problem. The new fits use every point in the covering cloud: 10,240 previous uniform points, a 3,125-point five-level-per-knob lattice, and 1,024 near-face points. No train/validation/test split is used. The continuous parameter box has infinitely many points; an exact construction uses the renderer equations rather than enumerating points.', '',
    '## Exact definition of the original fitted curves','',
    'Each curve stores 65 vectors `C[k, j]`, each containing all 784 pixels. For curve k, form node positions by cumulative segment length, normalized to [0, 1]:', '',
    '```text', 'length[j] = Euclidean distance between node[j+1] and node[j]',
    'position[j] = sum(lengths before j) / sum(all segment lengths)',
    'alpha = (t - position[j]) / (position[j+1] - position[j])',
    'curve_k(t) = (1 - alpha) * node[j] + alpha * node[j+1]', '```', '',
    'Here j is the interval containing t. This is an exact piecewise-linear mathematical equation for the stored curve, not merely a list of example source images. `original_curve_nodes.csv` contains every node and its path position. The original mean is in `../model_and_samples.npz`. Curve 1 panels show `mean + curve_1(t)`; curves 2–5 are residual contributions and can have positive or negative coverage.', '',
    'The five-coordinate decoder is:', '',
    '```text', 'predicted image = mean + curve_1(t1) + curve_2(t2) + curve_3(t3) + curve_4(t4) + curve_5(t5)', '```', '',
    'The original encoder chooses t1 by projecting image-minus-mean, subtracts that curve contribution, chooses t2 from the residual, and continues. There are five curves, not five points: the model has 325 stored pixel vectors.', '',
    '## Where the starting line comes from','',
    'No pair of images is selected to define PC1. All sample images determine the mean and covariance. The unit eigenvector of the largest covariance eigenvalue is the PC1 direction. The line is `mean + score * PC1_direction`; its initial endpoint scores are the minimum and maximum scores of the source samples. Those endpoint vectors are generally synthesized pixel vectors, not actual generated images. The sign of the direction can flip without changing the line.', '',
    'For every current curve segment from node a to node b, a sample x projects to:', '',
    '```text', 'fraction = clip(dot(x-a, b-a) / dot(b-a, b-a), 0, 1)',
    'candidate = a + fraction * (b-a)',
    'choose the candidate with the smallest squared pixel distance over all segments', '```', '',
    'All images are projected in the same iteration; none is selected as an image that the curve must pass through. A least-squares fit then uses all assigned images, with a second-difference penalty on neighboring nodes. It is a simultaneous matrix solve, not a left-to-right sequence of bends through chosen images. The assigned curve coordinate supplies the ordering. The curve is then reparameterized by arc length and the projection/smoothing process repeats.', '',
    'The smoothing fit minimizes:', '',
    '```text', 'sum over images: squared distance(image, interpolated curve at its assigned t)',
    '  + smoothing strength * sum over interior nodes:',
    '      squared norm(node[j+1] - 2*node[j] + node[j-1])', '```', '',
    'Fixed samples, initial line, smoothing strength, node count, and stopping rule give deterministic results, apart from floating-point differences and irrelevant direction-sign conventions. A new sampling seed changes the finite cloud; a different starting curve can lead to a different local solution. These are separate effects.', '',
    '## The original missing 2.7%','',
    f'The original residual is {r["original_missing_error"]["missing_variation_percent"]:.4f}% of total squared image variation. Its RMS pixel error is 0.02708 and its RMS image L2 error is 0.7583.', '',
    '| Location | Share of the residual squared error |','| --- | ---: |']
    for name,p in r['original_missing_error']['error_regions_percent'].items():lines.append(f'| {name} | {p:.2f}% |')
    lines += ['', 'Top/bottom bands include the endpoint rows and one neighboring row on each side. Side-edge bands are within 1.5 pixels of the geometric edges, excluding the endpoint bands. The residual contains blurred/misplaced edge coverage and endpoint corrections, rather than one distinct missing knob.', '',
    'A univariate spline of any single knob predicts only 1.50–2.88% of this residual energy. These individual prediction fractions overlap and are not a causal partition. A local tangent projection attributes about 32.88% of the error to the first-order span of the five physical knob derivatives and 67.12% to its orthogonal complement; two finite-difference steps agree within 0.001 percentage points. This is a local linear diagnostic, not proof that all normal error is globally off the manifold.', '',
    f'Keeping the original curve nodes fixed and improving all five coordinate choices cyclically raises the score to {refine["original_fixed_curves"]["metrics"]["explained_percent"]:.4f}%, removing {refine["original_fixed_curves"]["fraction_of_original_missing_error_removed_percent"]:.2f}% of the original residual energy. Thus the one-pass coordinate search explains a small part of the original loss. The rest reflects the fitted curve shapes and additive decoder approximation; it is not evidence for a sixth intrinsic knob.', '',
    '## All-point fit and initialization comparison','',
    '| Curves retained | Pooled-cloud fitted score |','| --- | ---: |']
    for k,a in enumerate(r['all_point_curves_cumulative'],1):lines.append(f'| {k} | {a["explained_percent"]:.3f}% |')
    lines += ['', f'Refining the five coordinate choices on the all-point curves, without moving any curve node, raises the pooled-cloud score to {refine["all_point_fixed_curves"]["metrics"]["explained_percent"]:.4f}%.', '',
        'These percentages use the mean and equal weights of the pooled covering cloud. Its inclusion of lattice/boundary points changes the measure, so it must not be treated as the same score distribution as the original uniform sample. The new fits use 65 nodes, smoothing penalty 1, and up to 45 iterations. They retain the best whole-cloud iterate, not a validation-selected iterate.', '',
    '| First-curve initialization | Pooled-cloud score | Correlation with lean | Correlation with width |',
    '| --- | ---: | ---: | ---: |']
    for a in r['initialized_from_physical_knob']:lines.append(f'| Exact {a["initialization"]} path | {a["metrics"]["explained_percent"]:.3f}% | {a["knob_spearman"][4]:.3f} | {a["knob_spearman"][3]:.3f} |')
    lines += ['', 'Starting with an exact knob path does not keep the unconstrained curve aligned to that knob. Projection and smoothing can move it toward a different summary of the cloud.', '',
    '## Aligning curves to the known knobs','',
    'Let F be the renderer and let `(cx, cy, height, width, lean)` be the current state. Exact physical coordinate curves are:', '',
    '```text', 'horizontal-center curve(t) = F(t, cy, height, width, lean)',
    'vertical-center curve(t)   = F(cx, t, height, width, lean)',
    'height curve(t)            = F(cx, cy, t, width, lean)',
    'width curve(t)             = F(cx, cy, height, t, lean)',
    'lean curve(t)              = F(cx, cy, height, width, t)', '```', '',
    'Each is a family indexed by the other four fixed settings. They are known coordinate curves of the renderer, not unconstrained principal curves inferred from pixels.', '',
    '| Known-knob construction | Pooled-cloud variation captured |', '| --- | ---: | ---: |',
    f'| Add five exact midpoint-anchor axis displacements | {r["fixed_midpoint_knob_curves"]["explained_percent"]:.3f}% |']
    for a in r['optimized_knob_aligned_additive_models']:lines.append(f'| Best joint least-squares sum of five knob-only curves, {a["nodes"]} nodes each | {a["metrics"]["explained_percent"]:.3f}% |')
    lines += ['| Follow each physical knob curve at the updated current state | 100.000% |', '',
    'The optimal additive models use the actual known knob values as coordinates, not nearest-curve inference. Their 33/65/129-node comparison isolates the limitation of independent knob-only contributions from crude sampling of the curve. The fixed-midpoint model uses exact renderer values at the requested knob settings, so its error is not polyline-interpolation error.', '',
    f'For a width increase of 1 px and a lean increase of 15° from the midpoint state, the mixed interaction has pixel L2 norm {interaction["mixed_pixel_L2"]:.4f}. The joint image differs from the sum of the two separate image changes. Therefore no sum of five functions depending only on individual physical knob values can represent this renderer exactly: such a sum would have zero mixed finite differences.', '',
    'The exact construction updates one knob, uses that updated state to define the next curve, and repeats. Differences telescope to the endpoint renderer image. On all 14,389 covering points, forward and reverse knob orders agree with the source image bit for bit. The endpoint is order-independent even though intermediate curves depend on state. This exactness comes from the known renderer; it is not a claim that a learned principal-curve algorithm discovered the full manifold.', '',
    '## Artifacts and reproduction','',
    '```sh', '.venv/bin/python analyze_generated_one_curve_followup.py',
    '.venv/bin/python refine_generated_one_curve_coordinates.py',
    '.venv/bin/python make_generated_one_curve_followup.py', '```', '',
    '- `literal_five_curves.png`: the original five literal curve contributions.',
    '- `missing_variation.png`: exact source, original reconstruction, signed missing coverage, and whole-sample controls.',
    '- `knob_curve_interaction.png`: one direct counterexample to adding independent knob changes.',
    '- `original_mean_pixels.csv`: all 784 values of the original mean image.',
    '- `original_curve_nodes.csv`: every original curve node with its exact stored pixel values and normalized arc coordinate.',
    '- `results.json`, `coordinate_refinement.json`, and numerical arrays preserve all scores and definitions.',
    '- The parent `index.html` includes sliders for literal fitted curve nodes and exact physical knob curves at selectable states.', '']
    (F/'README.md').write_text('\n'.join(lines))


def extend_page(s,r):
    # Append a new explainer to the existing page without altering original results.
    page=(OUT/'index.html').read_text();marker='<!-- curve-followup-start -->'
    if marker in page:page=page[:page.index(marker)]+page[page.index('<!-- curve-followup-end -->')+len('<!-- curve-followup-end -->'):]
    support=np.where(np.any(abs(s['curves'])>1e-12,axis=(0,1))|(s['mean']>0))[0]
    arcs=[]
    for c in s['curves']:
        a=np.r_[0,np.cumsum(np.linalg.norm(np.diff(c,axis=0),axis=1))];arcs.append(a/a[-1])
    def compact(x):return np.round(np.asarray(x),7).tolist()
    # Source the PC1 endpoints from all original fitting samples, not two images.
    train=pixels(s['train_knobs']);scores=(train-s['mean'])@s['basis'][0]
    ends=s['mean']+np.array([scores.min(),scores.max()])[:,None]*s['basis'][0]
    # Physical paths are pre-rendered directly for several background states.
    backgrounds=np.tile(BASE,(4,1));backgrounds[1,3]=RANGES[3,0];backgrounds[2,3]=RANGES[3,1];backgrounds[3,4]=RANGES[4,1]
    paths=[];units=['px','px','px','px','degrees']
    for background in backgrounds:
        paths_at_state=[]
        for k in range(5):
            pp=np.tile(background,(129,1));pp[:,k]=np.linspace(*RANGES[k],129)
            paths_at_state.append(pixels(pp))
        paths.append(paths_at_state)
    support=np.union1d(support,np.where(np.any(np.array(paths)>0,axis=(0,1,2)))[0])
    data=dict(mean=compact(s['mean'][support]),curves=compact(s['curves'][:,:,support]),support=support.tolist(),arcs=compact(arcs),
        starts=compact(ends[:,support]),ranges=RANGES.tolist(),knobs=KNOBS,units=units,backgrounds=backgrounds.tolist(),
        physical=compact(np.array(paths)[:,:,:,support]))
    section='''<!-- curve-followup-start -->
<section id="curve-definition"><h2>What exactly are curves 1–5?</h2><p>Each stores 65 nodes. One node is 784 pixel values, not a selected source image. Between neighboring nodes, the point is (1 − fraction) × left node + fraction × right node.</p>
<div class="controls"><label for="literalCurve">Curve</label><select id="literalCurve"><option value="0">1 — image-minus-mean contribution</option><option value="1">2 — residual contribution</option><option value="2">3 — residual contribution</option><option value="3">4 — residual contribution</option><option value="4">5 — residual contribution</option></select><label for="literalPosition">Path position</label><input id="literalPosition" type="range" min="0" max="1000" value="500"></div>
<div class="images"><figure><canvas class="pixel" id="literalLeft" width="28" height="28"></canvas><figcaption id="literalLeftLabel">Left stored node</figcaption></figure><figure><canvas class="pixel" id="literalPoint" width="28" height="28"></canvas><figcaption>Interpolated contribution at selected position</figcaption></figure><figure><canvas class="pixel" id="literalRight" width="28" height="28"></canvas><figcaption id="literalRightLabel">Right stored node</figcaption></figure></div>
<p id="literalValue" class="metric"></p><p class="note">All three panels show literal signed contributions: red adds ink, blue removes ink, white = zero; scale −1 to +1. Curve 1 becomes an image after adding the mean. Reconstruction = mean + C1(t1) + C2(t2) + C3(t3) + C4(t4) + C5(t5). <a href="followup/original_curve_nodes.csv">Download every stored node</a>.</p>
</section>
<section id="physical-curves"><h2>Five exact curves aligned with the known knobs</h2><p>Select one physical knob to vary. The other four stay at the selected background state. A different background gives a different curve for the same knob.</p>
<div class="controls"><label for="physicalKnob">Knob</label><select id="physicalKnob"><option value="0">Horizontal center</option><option value="1">Vertical center</option><option value="2">Height</option><option value="3">Width</option><option value="4" selected>Lean</option></select><label for="physicalBackground">Other settings</label><select id="physicalBackground"><option value="0">Range midpoints</option><option value="1">Thin stroke (width 1.8 px)</option><option value="2">Thick stroke (width 4.6 px)</option><option value="3">Strong lean (35 degrees)</option></select></div>
<div class="controls"><label for="physicalPosition">Knob value</label><input id="physicalPosition" type="range" min="0" max="128" step="1" value="64"><output id="physicalValue" class="metric"></output></div>
<div class="images"><figure><canvas class="pixel" id="physicalStart" width="28" height="28"></canvas><figcaption>Knob at its minimum</figcaption></figure><figure><canvas class="pixel" id="physicalPoint" width="28" height="28"></canvas><figcaption>Exact renderer image at selected value</figcaption></figure><figure><canvas class="pixel" id="physicalEnd" width="28" height="28"></canvas><figcaption>Knob at its maximum</figcaption></figure></div>
<p id="physicalState" class="metric"></p><p class="note">These are exact renderer coordinate paths at the displayed values. Five fixed paths added independently are approximate. Updating the background after each knob move reconstructs every endpoint exactly.</p>
</section>
<section><h2>Follow-up measurements: use every coverage point</h2><p>New fitting uses all 14,389 coverage points together, including the lattice and near-boundary states. No split is reserved. One fitted curve captures 90.0%; five capture 97.43%. A best-fit sum of five independent known-knob curves captures 80.74%; state-dependent physical knob paths reconstruct exactly.</p><p>The original missing error is concentrated in side-edge bands (53.0%) and top/bottom bands (43.4%). It is not a single missing sixth knob. <a href="followup/README.md">Detailed explanation and measured comparisons</a>.</p></section>
<script id="followup-data" type="application/json">FOLLOWUP_DATA</script>
<script>
(()=>{const D=JSON.parse(document.getElementById('followup-data').textContent);const $=id=>document.getElementById(id);
function draw(id,values,signed){const ctx=$(id).getContext('2d'),im=ctx.createImageData(28,28);for(let i=0;i<784;i++){im.data[4*i]=im.data[4*i+1]=im.data[4*i+2]=255;im.data[4*i+3]=255;}D.support.forEach((idx,j)=>{const v=values[j],a=Math.min(1,Math.abs(v));let rgb=signed?(v>=0?[255,Math.round(255*(1-a)),Math.round(255*(1-a))]:[Math.round(255*(1-a)),Math.round(255*(1-a)),255]):[255*(1-v),255*(1-v),255*(1-v)];rgb.forEach((c,k)=>im.data[4*idx+k]=Math.max(0,Math.min(255,Math.round(c))));});ctx.putImageData(im,0,0);}
function update(){const k=Number($('literalCurve').value),t=Number($('literalPosition').value)/1000,a=D.arcs[k];let j=0;while(j<63&&a[j+1]<t)j++;const f=Math.max(0,Math.min(1,(t-a[j])/(a[j+1]-a[j])));const left=D.curves[k][j],right=D.curves[k][j+1],point=left.map((v,i)=>(1-f)*v+f*right[i]);draw('literalLeft',left,true);draw('literalPoint',point,true);draw('literalRight',right,true);$('literalLeftLabel').textContent='Stored node '+j+' at t = '+a[j].toFixed(4);$('literalRightLabel').textContent='Stored node '+(j+1)+' at t = '+a[j+1].toFixed(4);$('literalValue').textContent='C'+(k+1)+'('+t.toFixed(3)+') = '+(1-f).toFixed(4)+' × node '+j+' + '+f.toFixed(4)+' × node '+(j+1);const knob=Number($('physicalKnob').value),bg=Number($('physicalBackground').value),i=Number($('physicalPosition').value),p=D.backgrounds[bg].slice();p[knob]=D.ranges[knob][0]+i/128*(D.ranges[knob][1]-D.ranges[knob][0]);draw('physicalStart',D.physical[bg][knob][0],false);draw('physicalPoint',D.physical[bg][knob][i],false);draw('physicalEnd',D.physical[bg][knob][128],false);$('physicalValue').textContent=D.knobs[knob]+' = '+p[knob].toFixed(3)+' '+D.units[knob];$('physicalState').textContent='Current settings (cx, cy, height, width, lean): '+p.map(v=>v.toFixed(3)).join(', ');}
['literalCurve','literalPosition','physicalKnob','physicalBackground','physicalPosition'].forEach(id=>$(id).addEventListener('input',update));update();})();
</script>
<!-- curve-followup-end -->'''
    section=section.replace('FOLLOWUP_DATA',json.dumps(data,separators=(',',':')))
    page=page.replace('</main>',section+'\n</main>')
    banner='<p><a href="#curve-definition">Inspect all five curve equations</a> · <a href="#physical-curves">Try exact knob-aligned paths</a> · <a href="followup/README.md">New all-point analysis</a></p>'
    if 'Inspect all five curve equations' not in page:page=page.replace('<section>',banner+'\n<section>',1)
    (OUT/'index.html').write_text(page)


def main():
    s=np.load(OUT/'model_and_samples.npz');r=json.loads((F/'results.json').read_text());ref=json.loads((F/'coordinate_refinement.json').read_text())
    literal_figure(s);missing_figure(s,r);interaction=joint_figure();r['explicit_width_lean_interaction']=interaction
    (F/'results.json').write_text(json.dumps(r,indent=2)+'\n')
    write_report(r,ref,interaction);extend_page(s,r)
    print('Saved literal curves, missing-error figure, physical interaction figure, report, and updated explorer.')


if __name__=='__main__':main()
