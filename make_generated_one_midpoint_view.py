"""Measured, interactive reference-pair midpoint comparisons."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_generated_one_midpoint_interpolation import OUT
from make_generated_one_joint_curve_view import ink


def expand(x,active):
    out=np.zeros(784);out[active]=x;return out


def figures(r,d):
    fig=plt.figure(figsize=(15.8,10.2),facecolor='white')
    fig.text(.04,.972,'Does averaging two reference 1s reproduce the true midpoint?',fontsize=20,weight='bold',va='top')
    fig.text(.04,.921,'Read left to right: actual reference endpoints → pixel average → exact image at the mean knob settings → learned fit to that exact image.',fontsize=10.7)
    fig.text(.04,.890,'28 × 28 ink coverage: white = 0; black = 1. Errors are image L2 over all 784 pixels; lower is better. Examples are selected from 952 pairs.',fontsize=10.2)
    ids=[int(d['example_indices'][i]) for i in [0,1,3]]
    gs=fig.add_gridspec(3,7,left=.035,right=.97,top=.82,bottom=.155,wspace=.16,hspace=.53)
    limit=.65
    for row,i in enumerate(ids):
        label=['Typical nearby pair','Typical distance-4 pair','Largest sampled distance-4 error'][row]
        a=d['endpoint_A'][i];b=d['endpoint_B'][i];t=d['truth'][i];v=d['average'][i];f=d['learned_clipped'][i]
        vals=[a,b,v,t,f,t-v,t-f]
        names=['Reference A','Reference B','Average of pixels','True knob midpoint','Learned fit, clipped','Truth − average','Truth − learned']
        distance=d['endpoint_distance'][i]
        for j,value in enumerate(vals):
            ax=fig.add_subplot(gs[row,j])
            title=names[j] if row==0 else ''
            if j<5:
                caption=f'L2 {np.linalg.norm(value-t):.3f}' if j in [2,4] else ('Correct target' if j==3 else '')
                ink(ax,expand(value,d['active']),title,caption)
            else:
                ax.imshow(expand(value,d['active']).reshape(28,28),cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
                ax.set_xticks([]);ax.set_yticks([]);ax.set_title(title,fontsize=10,pad=10)
                ax.text(.5,-.09,f'Max |error| {np.max(abs(value)):.3f}',transform=ax.transAxes,ha='center',va='top',fontsize=9)
            if j==0:ax.text(0,1.23,f'{label}\nEndpoint distance {distance:.3f}',transform=ax.transAxes,ha='left',va='bottom',fontsize=9,weight='bold')
    fig.text(.04,.072,'Error colors: red = missing ink; blue = excess ink. Both error maps use the same scale ±0.65; larger values saturate.',fontsize=10.2)
    fig.text(.04,.039,'These are finite examples, not shortest-path midpoints. The learned grid is unchanged; only its reconstruction of each true midpoint is measured.',fontsize=10.2)
    fig.savefig(OUT/'midpoint_examples.png',dpi=145);plt.close(fig)

    names=['Nearest reference pairs','Distance 1','Distance 2','Distance 3','Distance 4','Distance 5']
    fig,ax=plt.subplots(figsize=(11.7,6.6));fig.subplots_adjust(left=.105,right=.97,bottom=.27,top=.80)
    fig.text(.06,.965,'Averaging error grows with endpoint distance',fontsize=19,weight='bold',va='top')
    fig.text(.06,.912,'192 nearby reference pairs; 128 pairs in each distance band. Exact physical-knob midpoints are the targets.',fontsize=10.8)
    dist=[r['groups'][n]['endpoint_distance']['mean'] for n in names]
    avg=[r['groups'][n]['pixel_average']['rms_image_L2'] for n in names]
    fit=[r['groups'][n]['learned_clipped']['rms_image_L2'] for n in names]
    ax.plot(dist,avg,'o-',color='#bb5e22',label='Average of two exact reference images',linewidth=2)
    ax.plot(dist,fit,'o-',color='#2465a8',label='Learned manifold, output clipped',linewidth=2)
    for x,y in zip(dist,avg):ax.annotate(f'{y:.3f}',(x,y),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9)
    for x,y in zip(dist,fit):ax.annotate(f'{y:.3f}',(x,y),xytext=(0,-18),textcoords='offset points',ha='center',fontsize=9)
    ax.set_ylim(-.10,1.76);ax.set_xlabel('Mean Euclidean pixel distance between the two reference images',fontsize=11)
    ax.set_ylabel('RMS midpoint image L2 error (lower is better)',fontsize=11)
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.legend(loc='upper left',frameon=False,fontsize=10)
    fig.text(.06,.115,'Distance ≈4: averaging error 1.021 vs learned error 0.121 (8.4×). Nearby pairs: 0.051 vs 0.087.',fontsize=11,weight='bold')
    fig.text(.06,.064,'RMS = square each image error, average, then take the square root. Pixel RMS is this value / 28.',fontsize=10.4)
    fig.text(.06,.028,'Distance bands change which pairs are represented. This measures chord error, not a causal share of the model’s fitting error.',fontsize=10.4)
    fig.savefig(OUT/'midpoint_distance.png',dpi=145);plt.close(fig)


HTML='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Reference-image averaging versus learned manifold</title><style>
:root{color-scheme:light dark;--bg:#f4f7fa;--card:#fff;--text:#172a3b;--muted:#546777;--line:#dce4ec;--blue:#2365a8}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.6 system-ui,sans-serif}main{max-width:1320px;margin:auto;padding:30px 24px 70px}h1{font-size:34px;line-height:1.2}h2{font-size:23px;margin-top:0}section{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:24px;margin:24px 0}a{color:var(--blue)}.lead{font-size:21px}.muted{color:var(--muted);font-size:14px}.controls{display:flex;gap:16px;align-items:center;flex-wrap:wrap;margin:18px 0}select,input{font:inherit}select{padding:6px;max-width:100%}.images{display:grid;grid-template-columns:repeat(5,1fr);gap:18px}.errors{display:grid;grid-template-columns:repeat(2,1fr);gap:18px;max-width:600px;margin:24px auto}figure{margin:0;text-align:center}canvas{width:100%;max-width:180px;image-rendering:pixelated;border:1px solid var(--line)}figcaption{font-size:14px}table{width:100%;border-collapse:collapse}td,th{padding:9px;text-align:left;border-bottom:1px solid var(--line);font-variant-numeric:tabular-nums}.scroll{overflow:auto}.metric{font-weight:600}.wide{width:100%;height:auto}details{margin:18px 0}@media(prefers-color-scheme:dark){:root{--bg:#101923;--card:#172431;--text:#e8eef5;--muted:#a8bacb;--line:#324455;--blue:#79b6ed}}@media(max-width:720px){main{padding:20px 12px}section{padding:18px}h1{font-size:28px}.images{grid-template-columns:1fr 1fr}table{font-size:12px}}
</style></head><body><main><p><a href="../failures/">← Failure analysis</a> · <a href="../">Five-dimensional manifold</a> · <a href="../locations/">Error locations and bending</a></p><h1>What happens between two reference images?</h1><p class="lead">At endpoint distance ≈4, averaging has 8.4× the error of the learned manifold. For nearby reference pairs, averaging does better.</p>
<section><h2>The comparison</h2><p>Take two actual images from the 14,389-image fitting cloud. Average their pixels. Separately, average their five physical knob settings and render that exact image: this is our defined true midpoint. Then reconstruct that same true image with the unchanged learned five-dimensional manifold.</p><p>The original fitting cloud used 10,240 uniform samples, 3,125 lattice points and 1,024 near-face samples. No distance-4 cutoff chose the fitting images. The median nearest-reference distance in a 1,024-point probe was 0.557.</p><p class="muted">A mean knob setting selects a definite point on the known family. It does not claim to be a geodesic midpoint or the closest valid stroke to the averaged pixels. The model’s pixel-only projection does not receive the midpoint knobs.</p></section>
<section><h2>Matched midpoint errors</h2><div class="scroll"><table><thead><tr><th>Pair group</th><th>Pairs</th><th>Mean endpoint distance</th><th>Average-image RMS error</th><th>Learned RMS error, clipped</th><th>Error ratio</th></tr></thead><tbody id="scores"></tbody></table></div><p class="muted">Image L2 distance is the square root of summed squared ink-coverage differences across 784 pixels. RMS error aggregates squared image errors before averaging. Divide by 28 for pixel RMS. Distance-4 pairs lie in [3.75, 4.25); the other random bands are [0.5, 1.5), [1.5, 2.5), [2.5, 3.5), [4.75, 5.25). Each conditional band uses 128 random reference pairs, without requiring that pairs be neighbors.</p><img class="wide" src="midpoint_distance.png" alt="RMS error of averaging and learned reconstruction versus mean reference endpoint pixel distance"></section>
<section><h2>Isolate spacing at the same true midpoints</h2><p>For the same 128 distance-4 pairs, keep the true midpoint and physical-knob path direction fixed. Move both endpoints symmetrically closer to that midpoint, render their exact images and average those images. These closer endpoints are extra diagnostic renders; the fitted model stays unchanged.</p><div class="scroll"><table><thead><tr><th>Endpoint knob offsets</th><th>Mean endpoint pixel distance</th><th>Average-image RMS error</th><th>Learned RMS error, clipped</th></tr></thead><tbody id="controlled"></tbody></table></div><p class="muted">This isolates chord length for averaging: the target images and path directions are identical in every row. It does not measure how much the learned model would improve after refitting to a denser reference cloud.</p></section>
<section><h2>Inspect the images</h2><div class="controls"><label>Selected example <select id="example"></select></label><label>Pair group <select id="group"></select></label><label>Pair <input id="pair" type="range" min="0" value="0" step="1"></label></div><div class="controls"><label><input id="clip" type="checkbox" checked> Clip learned pixels to [0, 1]</label><label>Error map scale ± <select id="scale"><option>0.05</option><option>0.2</option><option selected>0.65</option><option>1</option></select></label></div><p id="pairInfo" class="metric"></p><div class="images"><figure><canvas id="A" width="28" height="28"></canvas><figcaption>Reference A</figcaption></figure><figure><canvas id="B" width="28" height="28"></canvas><figcaption>Reference B</figcaption></figure><figure><canvas id="average" width="28" height="28"></canvas><figcaption>Average of reference pixels</figcaption></figure><figure><canvas id="truth" width="28" height="28"></canvas><figcaption>Exact image at mean knobs</figcaption></figure><figure><canvas id="learned" width="28" height="28"></canvas><figcaption>Learned reconstruction</figcaption></figure></div><p id="errors" class="metric"></p><div class="errors"><figure><canvas id="averageError" width="28" height="28"></canvas><figcaption>Truth − average</figcaption></figure><figure><canvas id="learnedError" width="28" height="28"></canvas><figcaption>Truth − learned</figcaption></figure></div><p class="muted">Images: white = no ink, black = full ink; magenta flags raw values outside [0, 1], with amplified tint. Error maps: red = missing ink, blue = excess ink; both maps share the selected scale, and larger errors saturate.</p><div class="scroll"><table><thead><tr><th>Physical knob</th><th>Reference A</th><th>Reference B</th><th>True midpoint</th></tr></thead><tbody id="knobs"></tbody></table></div><p id="chartError" class="muted"></p></section>
<section><h2>Same kind of error?</h2><p>There is a shared geometric failure: averaging moving stroke edges spreads ink across positions the exact intermediate stroke does not cover. A connected, bounded image can look like a 1 and still have incorrect pixel coverage. At distance ≈4, 70.5% of the averaging squared error is on partially covered target pixels, 14.2% on empty target pixels and 15.3% on full-ink target pixels. For the clipped learned image at those same midpoints, the corresponding shares are 88.5%, 6.0% and 5.5%.</p><p>There is also a clear difference: an average of two valid [0, 1] images stays in [0, 1]. It cannot make the model’s negative or above-one magenta pixels. The model interpolates 32 learned grid corners inside a five-dimensional cell; those corners were fitted without pixel bounds. It does not simply average two selected reference images.</p><p>For nearest-reference pairs, the average has RMS error 0.051, while the learned model has 0.087. This is evidence that a short reference chord alone does not account for all learned error. Finite grid resolution, fitted node errors, smoothing, boundary behavior and coordinate projection can also matter. These comparisons do not establish their separate causal shares.</p><p>At distance ≈4 the median cosine similarity between the two signed pixel-error patterns is 0.331 (1 would mean identical direction; 0 means perpendicular). They share an edge-coverage failure class but are not the same pixel-by-pixel error.</p></section>
<section><h2>Single-knob steps in the existing reference lattice</h2><p>Here all other knobs stay fixed, and two adjacent original lattice images are averaged. There are 24 sampled pairs for each knob. Steps are 0.25 px horizontally, 0.25 px vertically, 0.375 px in height, 0.7 px in width, or 11.25° in lean. These deliberately coarse lattice steps are supplemented by the other reference samples; the lean step is especially large.</p><div class="scroll"><table><thead><tr><th>Knob</th><th>Mean endpoint distance</th><th>Average-image RMS error</th><th>Learned RMS error, clipped</th></tr></thead><tbody id="lattice"></tbody></table></div><p class="muted">These are a different, often boundary-heavy cohort. Do not compare their errors with random pair groups as if only the knob name changed.</p></section>
<details><summary>Figures, method and numerical evidence</summary><p><a href="README.md">Method and definitions</a> · <a href="results.json">All summary measurements</a> · <a href="pairs.npz">952 pair comparisons</a></p><img class="wide" src="midpoint_examples.png" alt="Three paired reference examples, their averaged images, exact knob midpoints and learned fits with signed pixel errors"></details></main>
<script>const D=__DATA__;
const $=id=>document.getElementById(id);let current=0;const f=x=>x.toFixed(3);
const norm=a=>Math.sqrt(a.reduce((s,v)=>s+v*v,0));const sub=(a,b)=>a.map((v,i)=>v-b[i]);
function draw(id,values,error=false){const context=$(id).getContext('2d');const im=context.createImageData(28,28);let v=Array(784).fill(0);D.active.forEach((p,k)=>v[p]=values[k]);const limit=Number($('scale').value);v.forEach((x,i)=>{let rgb;if(error){let a=Math.min(1,Math.abs(x)/limit);rgb=x>=0?[255,255*(1-a),255*(1-a)]:[255*(1-a),255*(1-a),255];}else{let shade=255*(1-Math.max(0,Math.min(1,x)));rgb=[shade,shade,shade];if(x<-1e-6||x>1+1e-6){let a=Math.min(1,.1+3*Math.abs(x-Math.max(0,Math.min(1,x))));rgb=rgb.map((v,k)=>(1-a)*v+a*[199,38,168][k]);}}rgb.forEach((c,k)=>im.data[4*i+k]=Math.round(c));im.data[4*i+3]=255;});context.putImageData(im,0,0);}
function update(){const e=D.pairs[current];const learned=$('clip').checked?e.learned.map(v=>Math.max(0,Math.min(1,v))):e.learned;const ae=sub(e.truth,e.average),le=sub(e.truth,learned);['A','B','average','truth'].forEach(id=>draw(id,e[id]));draw('learned',learned);draw('averageError',ae,true);draw('learnedError',le,true);$('pairInfo').textContent=e.group+' · pair '+(Number($('pair').value)+1)+' · reference indices '+e.indices.join(' and ')+' · endpoint distance '+f(e.distance);$('errors').textContent='Midpoint image L2 error: average '+f(norm(ae))+' · learned '+f(norm(le))+' · pixel RMS: '+(norm(ae)/28).toFixed(5)+' vs '+(norm(le)/28).toFixed(5);$('knobs').innerHTML=D.knobNames.map((n,k)=>'<tr><td>'+n+'</td><td>'+f(e.pa[k])+'</td><td>'+f(e.pb[k])+'</td><td>'+f((e.pa[k]+e.pb[k])/2)+'</td></tr>').join('');$('chartError').textContent='Different operation: decode at the mean of the endpoints’ fitted chart coordinates. Error to this physical-knob midpoint: '+f(e.chartError)+'. Fitted chart coordinates can warp relative to physical knobs; this operation has no closest-point refinement.';}
function setPair(i){current=i;const group=D.pairs[i].group;$('group').value=group;const candidates=D.pairs.map((e,k)=>e.group===group?k:-1).filter(k=>k>=0);$('pair').max=candidates.length-1;$('pair').value=candidates.indexOf(i);update();}
$('group').innerHTML=D.groupNames.map(n=>'<option>'+n+'</option>').join('');$('example').innerHTML=D.examples.map((e,i)=>'<option value="'+i+'">'+e.label+'</option>').join('');
$('example').addEventListener('input',()=>setPair(D.examples[Number($('example').value)].index));$('group').addEventListener('input',()=>setPair(D.pairs.findIndex(e=>e.group===$('group').value)));$('pair').addEventListener('input',()=>{const candidates=D.pairs.map((e,k)=>e.group===$('group').value?k:-1).filter(k=>k>=0);current=candidates[Number($('pair').value)];update();});['clip','scale'].forEach(id=>$(id).addEventListener('input',update));
const spacing=D.results.controlled_same_midpoints;$('controlled').innerHTML=Object.values(spacing.fractions).map(g=>'<tr><td>'+g.endpoint_offset_fraction+'</td><td>'+f(g.endpoint_distance.mean)+'</td><td>'+f(g.pixel_average.rms_image_L2)+'</td><td>'+f(spacing.learned_clipped.rms_image_L2)+'</td></tr>').join('');
const names=['Nearest reference pairs','Distance 1','Distance 2','Distance 3','Distance 4','Distance 5'];$('scores').innerHTML=names.map(n=>{const g=D.results.groups[n];return '<tr><td>'+n+'</td><td>'+g.count+'</td><td>'+f(g.endpoint_distance.mean)+'</td><td>'+f(g.pixel_average.rms_image_L2)+'</td><td>'+f(g.learned_clipped.rms_image_L2)+'</td><td>'+g.ratio_average_to_clipped_manifold_rms.toFixed(1)+'×</td></tr>';}).join('');$('lattice').innerHTML=D.knobNames.map(n=>{const g=D.results.groups['Adjacent lattice: '+n];return '<tr><td>'+n+'</td><td>'+f(g.endpoint_distance.mean)+'</td><td>'+f(g.pixel_average.rms_image_L2)+'</td><td>'+f(g.learned_clipped.rms_image_L2)+'</td></tr>';}).join('');setPair(D.examples[1].index);$('example').value=1;
</script></body></html>'''


def main():
    r=json.loads((OUT/'results.json').read_text());d=np.load(OUT/'pairs.npz')
    figures(r,d)
    pairs=[]
    for i in range(len(d['truth'])):
        pairs.append(dict(group=str(d['groups'][i]),indices=d['reference_indices'][i].tolist(),
            distance=float(d['endpoint_distance'][i]),pa=d['endpoint_A_knobs'][i].tolist(),pb=d['endpoint_B_knobs'][i].tolist(),
            A=d['endpoint_A'][i].tolist(),B=d['endpoint_B'][i].tolist(),average=d['average'][i].tolist(),
            truth=d['truth'][i].tolist(),learned=d['learned_raw'][i].tolist(),
            chartError=float(np.linalg.norm(d['chart_midpoint'][i]-d['truth'][i]))))
    data=dict(results=r,pairs=pairs,active=np.flatnonzero(d['active']).tolist(),knobNames=r['knob_order'],
              groupNames=list(r['groups']),examples=[dict(label=str(label),index=int(i))
                for label,i in zip(d['example_labels'],d['example_indices'])])
    (OUT/'index.html').write_text(HTML.replace('__DATA__',json.dumps(data,separators=(',',':'))))
    lines=['# Reference-image midpoint interpolation','','Two endpoints are actual fitting-cloud images. Pixel average = (image A + image B) / 2. True midpoint = render((knobs A + knobs B) / 2). The unchanged fine five-dimensional learned manifold reconstructs this true midpoint with the same pixel-only multistart projection as the failure analysis. No training or model selection occurs.','',
        'Image L2 error is Euclidean distance over 784 coverage pixels. RMS image L2 squares each image error, averages, and takes a square root. Pixel RMS = RMS image L2 / 28. Average-image error is distance to a defined physical-knob midpoint, not the minimum distance to any valid stroke.','',
        'The fitting cloud has 14,389 images: 10,240 uniform, 3,125 five-point-per-axis lattice, 1,024 near-face. There was no fixed distance-4 image selection. All fitting images were used by the prior fit.','',
        'Pair diagnostic: seed 1007, 192 unique nearest-reference pairs sampled from 1,024 uniform reference-index probes; 128 randomly selected reference pairs in each of five conditional pixel-distance bands; 24 adjacent original lattice pairs per knob. Endpoint pairs are sampled from the existing discrete cloud, not an exhaustive pair analysis. Single-knob lattice steps are one quarter of the physical range.','',
        '| Pair group | N | Mean endpoint distance | Average RMS image error | Learned clipped RMS image error |','| --- | ---: | ---: | ---: | ---: |']
    for name,g in r['groups'].items():lines.append(f"| {name} | {g['count']} | {g['endpoint_distance']['mean']:.4f} | {g['pixel_average']['rms_image_L2']:.5f} | {g['learned_clipped']['rms_image_L2']:.5f} |")
    lines+=['','## Fixed-target spacing check','','The same 128 distance-4 true midpoints and physical-knob path directions are held fixed. Shrink both endpoint offsets symmetrically, render exact additional reference endpoints, and average. No fitting cloud or model is changed. At each row the learned clipped RMS error is 0.12095.','', '| Endpoint offset fraction | Mean pixel distance | Average RMS image error |','| ---: | ---: | ---: |']
    for g in r['controlled_same_midpoints']['fractions'].values():lines.append(f"| {g['endpoint_offset_fraction']:g} | {g['endpoint_distance']['mean']:.5f} | {g['pixel_average']['rms_image_L2']:.5f} |")
    lines+=['','Averaging exact endpoint images creates bounded edge blur and incorrect intermediate coverage. It cannot cause negative or above-one pixels. The learned manifold interpolates 32 learned grid nodes per cell, not two original references. Nearby-reference averaging error is smaller than learned error, so short chords alone are insufficient to explain every learned residual. Node approximation, grid resolution, smoothing, boundary behavior and projection are not causally apportioned by this experiment.','',
        'For distance-4 pairs, average error mean 1.0001, RMS 1.0207, median 1.0234, p95 1.2674. Learned clipped RMS on the same true midpoints is 0.1210. The 8.44 ratio refers to RMS image L2; the squared-error ratio is its square.','',
        'Mean fitted endpoint chart-coordinate decoding is separately recorded. Since the chart has been fitted and is warped relative to physical knobs, this is a different target operation and does not include closest-point refinement.','',
        'Display examples include typical nearby and distance-4 averaging errors, smallest and largest distance-4 errors in the finite sample, and typical averaging errors for each single-knob lattice move. Representative means selected by proximity to median image L2 error; no claim about population extrema.','',
        'Reproduce from the repository root: `.venv/bin/python analyze_generated_one_midpoint_interpolation.py`, then `.venv/bin/python make_generated_one_midpoint_view.py`. Read `results.json` for complete metric definitions and `pairs.npz` for all pairs, endpoints, true midpoints, predictions and fitted coordinates.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
