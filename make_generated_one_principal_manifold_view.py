"""Measured comparisons and a renderer-free interactive principal-grid decoder."""
import base64
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from fit_generated_one_principal_manifold import OUT, decode
from make_generated_one_joint_curve_view import ink

JOINT=OUT.parent/'joint'


def comparison(r, probe, previous, indices):
    best=r['best_candidate']; c=r['candidates'][best]
    old=json.loads((JOINT/'results.json').read_text())
    values=[old['candidates'][old['best_candidate']]['metrics']['explained_percent']]
    labels=['Five additive curves\n513 nodes each']
    for name,v in r['candidates'].items():
        labels.append(f'One 5D grid: {name}\n{v["node_count"]:,} nodes')
        values.append(v['metrics']['explained_percent'])
    fig=plt.figure(figsize=(12.5,9.5),facecolor='white')
    fig.text(.055,.97,'One learned manifold, with five coordinates',fontsize=20,weight='bold',va='top')
    fig.text(.055,.921,'Each generated 1 is a 28 × 28 coverage image: white = 0, black = 1. All 14,389 covering images are fitted.',fontsize=10.4)
    ax=fig.add_axes([.09,.645,.83,.20]);missing=100-np.array(values)
    ax.bar(np.arange(len(values)),missing,color=['#bd5a22']+['#2563a6']*(len(values)-1),width=.58)
    ax.set_xticks(np.arange(len(values)),labels,fontsize=9.3)
    ax.set_ylabel('Variation left unreconstructed (%)',fontsize=10)
    ax.set_ylim(0,max(missing)*1.27);ax.spines[['top','right']].set_visible(False)
    for i,(m,v) in enumerate(zip(missing,values)):
        ax.text(i,m+max(missing)*.04,f'{v:.4f}% captured',ha='center',fontsize=9.6,weight='bold')
    gs=fig.add_gridspec(2,4,left=.09,right=.92,top=.49,bottom=.16,wspace=.28,hspace=.52)
    src=probe['images'];new=probe['reconstructed'];limit=float(max(.05,np.ceil(np.max(abs(src[indices]-new[indices]))*20)/20))
    def full(v):
        a=np.zeros(784);a[probe['active']]=v;return a
    for row,i in enumerate(indices):
        label=['Typical remaining error','Worst remaining error'][row]
        for j,(v,title) in enumerate([(src[i],'Exact generated image'),(previous[i],'Best five curves'),(new[i],'One 5D manifold')]):
            caption=label if j==0 else f'Image L2 error {np.linalg.norm(v-src[i]):.4f}'
            ink(fig.add_subplot(gs[row,j]),full(v),title if row==0 else '',caption)
        ax=fig.add_subplot(gs[row,3]);ax.imshow(full(src[i]-new[i]).reshape(28,28),cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
        ax.set_xticks([]);ax.set_yticks([]);ax.set_title('Source − manifold' if row==0 else '',fontsize=10,pad=10)
        ax.text(.5,-.09,f'Max pixel error {np.max(abs(src[i]-new[i])):.4f}',ha='center',va='top',transform=ax.transAxes,fontsize=9)
    dense=r['sampling_checks']['dense_uniform']['pixel_only_projection']['explained_percent']
    fig.text(.055,.065,f'Images below are from 16,384 additional uniform points: manifold capture {dense:.4f}%. Red = missing ink; blue = excess; error scale ±{limit:g}.',fontsize=9.7)
    fig.text(.055,.029,'The learned 5D cells include interactions. Magenta marks image values outside [0, 1]. Finite sampling and approximate projection; no exactness guarantee.',fontsize=9.4)
    fig.savefig(OUT/'manifold_comparison.png',dpi=145);plt.close(fig)


def slice_figure(model, coordinate):
    fig=plt.figure(figsize=(10.2,9.5),facecolor='white')
    fig.text(.07,.962,'A two-coordinate slice of the learned 5D manifold',fontsize=18,weight='bold',va='top')
    fig.text(.07,.915,'Every tile is a full decoded 28 × 28 image, computed only by interpolation among learned grid nodes.',fontsize=10.5)
    fig.text(.07,.88,'The other three learned coordinates are fixed at one fitted example. These are learned positions, not exact physical knob values.',fontsize=9.5)
    gs=fig.add_gridspec(5,5,left=.16,right=.95,bottom=.18,top=.80,wspace=.28,hspace=.25)
    pos=np.linspace(.1,.9,5)
    for row,y in enumerate(pos[::-1]):
        for col,x in enumerate(pos):
            u=coordinate.copy();u[3]=x;u[4]=y
            v=decode(u[None,:],model['nodes'],model['shape'])[0]
            a=np.zeros(784);a[model['active']]=v
            ax=fig.add_subplot(gs[row,col]);ink(ax,a)
            if row==4:ax.set_xlabel(f'{x:.1f}',fontsize=10)
            if col==0:ax.set_ylabel(f'{y:.1f}',fontsize=10,labelpad=14)
    fig.text(.55,.116,'Learned coordinate 4 (initialized from width), fractional position',ha='center',fontsize=11)
    fig.text(.045,.49,'Learned coordinate 5 (initialized from lean)',ha='center',va='center',rotation=90,fontsize=11)
    fig.text(.07,.046,'White = no ink; black = full ink. Magenta tint shows values outside [0, 1]. This fixed-coordinate slice is not the entire manifold.',fontsize=9.3)
    fig.savefig(OUT/'manifold_slice.png',dpi=145);plt.close(fig)


HTML=r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>One five-dimensional principal manifold</title>
<style>:root{color-scheme:light dark;--bg:#f4f7fa;--card:#fff;--text:#172a3b;--muted:#546777;--line:#dce4ec;--blue:#2365a8}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.6 system-ui,sans-serif}main{max-width:1100px;margin:auto;padding:35px 24px 70px}h1{font-size:35px;line-height:1.2}h2{font-size:23px;margin-top:0}section{background:var(--card);border:1px solid var(--line);border-radius:14px;margin:24px 0;padding:25px}a{color:var(--blue)}.lead{font-size:21px}.muted{color:var(--muted);font-size:14px}table{width:100%;border-collapse:collapse}td,th{padding:11px;text-align:left;border-bottom:1px solid var(--line)}.images{display:grid;grid-template-columns:repeat(4,1fr);gap:20px}figure{margin:0;text-align:center}canvas{width:100%;max-width:168px;border:1px solid var(--line);image-rendering:pixelated}figcaption{font-size:14px}select,button,input{font:inherit}select,button{padding:6px}.controls{display:flex;gap:15px;align-items:center;flex-wrap:wrap;margin:16px 0}.sliders{display:grid;grid-template-columns:1fr 1fr;gap:14px}label.slider{display:grid;grid-template-columns:1fr 65px;gap:5px}.slider input{grid-column:1 / 3;width:100%}output{font-variant-numeric:tabular-nums}.equation{background:var(--bg);padding:15px;border-radius:8px;font:15px/1.8 ui-monospace,monospace}.wide{max-width:100%;height:auto}details{margin:18px 0}summary{cursor:pointer}.metric{font-weight:600;font-variant-numeric:tabular-nums}@media(prefers-color-scheme:dark){:root{--bg:#101923;--card:#172431;--text:#e8eef5;--muted:#a8bacb;--line:#324455;--blue:#79b6ed}}@media(max-width:720px){main{padding:20px 12px}section{padding:18px}h1{font-size:28px}.images{grid-template-columns:1fr 1fr}.sliders{grid-template-columns:1fr}table{font-size:12px}td,th{padding:8px 4px}}</style></head>
<body><main><p><a href="../joint/">← Five jointly fitted curves</a></p><h1>One principal manifold with five coordinates</h1><p class="lead" id="headline"></p>
<section><h2>What is being fitted?</h2><p>A flexible five-dimensional grid lives in the 784-dimensional space of images. Every node stores an image’s pixel values. The fitting process moves these nodes and each image’s position on the grid to reduce reconstruction error, with a bending penalty to keep the grid coherent.</p><div class="equation">five curves: mean + C1(t1) + C2(t2) + C3(t3) + C4(t4) + C5(t5)<br>one 5D manifold: F(t1, t2, t3, t4, t5)<br>F = weighted interpolation of the 32 corners of one 5D grid cell</div><p>The joint cells can capture interactions: the effect of changing one coordinate depends on the other four positions. The decoder uses learned pixels and interpolation. It contains no stroke-rendering equations.</p><p class="muted">All 14,389 covering images participate: 10,240 uniform samples, a 3,125-point lattice, and 1,024 near-face samples. Five coordinates can describe the family exactly through its known renderer, but a finite learned grid still approximates its geometry.</p></section>
<section><h2>Measured variation captured</h2><table><thead><tr><th>Representation</th><th>Stored grid nodes</th><th>Covering-cloud capture</th><th>Pixel RMS error</th></tr></thead><tbody id="scores"></tbody></table><p class="muted">Capture = 100 × (1 − total squared pixel reconstruction error / total squared distance to this cloud’s mean). Scores use unclipped pixels. One manifold has five coordinate dimensions; grid resolution controls its complexity.</p><p class="metric" id="gain"></p><h3>Additional sampling-resolution checks</h3><table><thead><tr><th>Points</th><th>Best five curves</th><th>One 5D manifold</th></tr></thead><tbody id="checks"></tbody></table><p class="muted">These renderings check gaps and boundaries; they did not choose the winning grid. All 32 corners also belong to the fitting lattice. Projection starts from nearby images in pixel distance. Difficult cases get eight starts and more steps; no physical knob values enter that search.</p></section>
<section><h2>Compare reconstructed images</h2><div class="controls"><label for="example">Example</label><select id="example"></select></div><div class="images"><figure><canvas id="source" width="28" height="28"></canvas><figcaption>Exact generated image</figcaption></figure><figure><canvas id="curves" width="28" height="28"></canvas><figcaption>Best five additive curves</figcaption></figure><figure><canvas id="manifold" width="28" height="28"></canvas><figcaption>One 5D manifold</figcaption></figure><figure><canvas id="error" width="28" height="28"></canvas><figcaption>Source − manifold</figcaption></figure></div><p id="details" class="metric"></p><p id="scale" class="muted"></p></section>
<section><h2>Move through the learned manifold</h2><p>These are five learned coordinates. They were initialized using horizontal center, vertical center, height, width, and lean, then allowed to move during fitting.</p><div class="sliders" id="sliders"></div><div class="controls"><button id="reset">Use fitted example coordinates</button><button id="center">Center of coordinate box</button></div><div class="images"><figure><canvas id="sweep" width="28" height="28"></canvas><figcaption>Decoded at slider positions</figcaption></figure><figure><canvas id="fitted" width="28" height="28"></canvas><figcaption>Decoded at example positions</figcaption></figure></div><p id="position" class="metric"></p><p class="muted">White = no ink, black = full ink; magenta tint shows values outside [0, 1]. Arbitrary slider combinations are not certified valid strokes. The coordinates can warp away from physical knob values.</p><details><summary>See how two coordinates interact</summary><img class="wide" src="manifold_slice.png" alt="A five by five grid of decoded images across a width and lean initialized coordinate slice"></details></section>
<section><h2>How the known knobs helped</h2><ol><li>They set the dimension to five and supplied normalized initial positions in a bounded coordinate box.</li><li>They gave a meaningful grid orientation. The anisotropic grids allocate more nodes along width and lean, based on their larger observed image changes.</li><li>After that, repeated pixel-space node fits and closest-point searches refine the manifold. Bending penalties are reduced in stages; the decoder does not use the knobs or renderer.</li></ol><p id="rank"></p><p>A five-dimensional description is sufficient in principle. This search finds an approximation; it does not prove a global optimum, recover every possible point exactly, or establish that the entire learned coordinate box is bijective and valid.</p><p class="muted">Method: an elastic principal manifold with multilinear cells and continuous-coordinate projection, inspired by <a href="https://arxiv.org/abs/cond-mat/0405648">Gorban and Zinovyev’s elastic principal manifolds</a>. Node fitting is quadratic for fixed coordinates; the full alternating problem is nonconvex.</p></section>
<details><summary>Figures and reproducible numerical model</summary><p><a href="README.md">Method and limits</a> · <a href="results.json">Measured results</a> · <a id="download" href="">Learned grid NPZ</a> · <a href="decoder.json">Renderer-free browser decoder</a></p><img class="wide" src="manifold_comparison.png" alt="Variance capture across curve and manifold fits, followed by actual pixel reconstruction examples"></details></main>
<script id="data" type="application/json">PAYLOAD</script><script>
const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);let nodes;
function decode(t){const j=t.map((v,k)=>Math.min(Math.floor(Math.max(0,Math.min(1,v))*(D.shape[k]-1)),D.shape[k]-2)),f=t.map((v,k)=>Math.max(0,Math.min(1,v))*(D.shape[k]-1)-j[k]);const out=new Array(D.support.length).fill(0);for(let mask=0;mask<32;mask++){let index=0,w=1;for(let k=0;k<5;k++){const bit=(mask>>(4-k))&1;index=index*D.shape[k]+j[k]+bit;w*=bit?f[k]:1-f[k];}if(w===0)continue;for(let p=0;p<out.length;p++)out[p]+=w*nodes[index*out.length+p];}return out;}
function draw(id,a,limit=0){const canvas=$(id),ctx=canvas.getContext('2d'),im=ctx.createImageData(28,28);for(let p=0;p<784;p++){im.data[p*4]=255;im.data[p*4+1]=255;im.data[p*4+2]=255;im.data[p*4+3]=255;}D.support.forEach((p,i)=>{let rgb;if(limit){const v=Math.max(-1,Math.min(1,a[i]/limit)),base=1-Math.abs(v);rgb=v>=0?[255,255*base,255*base]:[255*base,255*base,255];}else{const v=a[i],c=255*(1-Math.max(0,Math.min(1,v)));rgb=[c,c,c];if(v < -1e-6||v>1+1e-6){const amt=Math.min(1,.10+3*Math.abs(v-Math.max(0,Math.min(1,v))));rgb=[c*(1-amt)+199*amt,c*(1-amt)+38*amt,c*(1-amt)+168*amt];}}rgb.forEach((v,k)=>im.data[4*p+k]=v);});ctx.putImageData(im,0,0);}
function l2(a,b){return Math.sqrt(a.reduce((s,v,j)=>s+(v-b[j])**2,0));}
function updateSweep(){if(!nodes)return;const t=Array.from({length:5},(_,k)=>Number($('t'+k).value));t.forEach((v,k)=>$('o'+k).textContent=v.toFixed(3));draw('sweep',decode(t));$('position').textContent='Learned coordinates: ('+t.map(v=>v.toFixed(3)).join(', ')+')';}
function reset(){const e=D.examples[Number($('example').value)];e.coordinates.forEach((v,k)=>$('t'+k).value=v);updateSweep();}
function updateExample(){if(!nodes)return;const e=D.examples[Number($('example').value)],rec=decode(e.coordinates);draw('source',e.source);draw('curves',e.curves);draw('manifold',rec);draw('fitted',rec);draw('error',e.source.map((v,k)=>v-rec[k]),D.errorLimit);$('details').textContent='Image L2 error: five curves '+l2(e.source,e.curves).toFixed(4)+' → manifold '+l2(e.source,rec).toFixed(4)+'. Physical settings (cx, cy, height, width, lean): '+e.knobs.map(v=>v.toFixed(3)).join(', ');reset();}
$('headline').textContent=D.bestScore.toFixed(4)+'% of covering-cloud variation captured; '+D.denseScore.toFixed(4)+'% on additional uniform points.';
$('scores').innerHTML=D.rows.map(r=>'<tr><td>'+r.name+'</td><td>'+r.nodes.toLocaleString()+'</td><td>'+r.score.toFixed(4)+'%</td><td>'+r.rms.toFixed(6)+'</td></tr>').join('');
$('checks').innerHTML=D.checks.map(r=>'<tr><td>'+r.name+'</td><td>'+r.curves.toFixed(4)+'%</td><td>'+r.manifold.toFixed(4)+'%</td></tr>').join('');
$('gain').textContent=D.gain.toFixed(1)+'% of the five-curve model’s remaining squared error removed on the covering cloud.';
$('scale').textContent='Error colors: red = missing ink; blue = excess ink; fixed scale ±'+D.errorLimit+'. Magenta shows image values outside [0, 1].';
$('rank').textContent='All '+D.rank.points+' checked Jacobians had five independent directions at relative singular-value tolerance 10⁻⁶. This checks local dimension at sampled positions, not global uniqueness.';
$('example').innerHTML=D.examples.map((e,i)=>'<option value="'+i+'">'+e.label+'</option>').join('');
$('sliders').innerHTML=D.names.map((name,k)=>'<label class="slider">'+(k+1)+': initialized from '+name+'<output id="o'+k+'">0.500</output><input id="t'+k+'" type="range" min="0" max="1" step="0.0001" value="0.5"></label>').join('');
for(let k=0;k<5;k++)$('t'+k).addEventListener('input',updateSweep);$('example').addEventListener('input',updateExample);$('reset').addEventListener('click',reset);$('center').addEventListener('click',()=>{for(let k=0;k<5;k++)$('t'+k).value=.5;updateSweep();});$('download').href=D.best+'.npz';
async function load(){const m=await(await fetch('decoder.json')).json(),bytes=Uint8Array.from(atob(m.nodes_float32_base64),c=>c.charCodeAt(0));nodes=new Float32Array(bytes.buffer);if(nodes.length!==D.shape.reduce((a,b)=>a*b,1)*D.support.length)throw new Error('Invalid grid length');updateExample();}
$('position').textContent='Loading the learned pixel grid…';load().catch(e=>{$('position').textContent='Could not load the learned grid: '+e.message;});
</script></body></html>'''


def main():
    r=json.loads((OUT/'results.json').read_text());old=json.loads((JOINT/'results.json').read_text())
    best=r['best_candidate'];stat=r['candidates'][best];model=np.load(OUT/f'{best}.npz')
    probe=np.load(OUT/'dense_uniform_examples.npz');previous=np.load(JOINT/'joint_513_dense.npz')
    assert np.allclose(probe['knobs'],previous['knobs']) and np.array_equal(probe['active'],previous['active'])
    errors=np.linalg.norm(probe['images']-probe['reconstructed'],axis=1)
    indices=[int(np.argsort(errors)[len(errors)//2]),int(np.argmax(errors))]+list(range(12))
    labels=['Typical manifold error','Worst manifold error']+[f'Additional point {i+1}' for i in range(12)]
    comparison(r,probe,previous['reconstructed'],indices[:2]);slice_figure(model,probe['coordinates'][indices[0]])
    rows=[dict(name='Best five additive curves (513 nodes each)',nodes=2565,score=old['candidates'][old['best_candidate']]['metrics']['explained_percent'],rms=old['candidates'][old['best_candidate']]['metrics']['rms_pixel_error'])]
    for name,v in r['candidates'].items():
        rows.append(dict(name=f'One 5D manifold: {name} {tuple(v["shape"])}',nodes=v['node_count'],score=v['metrics']['explained_percent'],rms=v['metrics']['rms_pixel_error']))
    checks=[]
    for name,text in [('dense_uniform','16,384 additional uniform points'),('all_32_corners','All 32 box corners'),('joint_midpoints','2,048 joint midpoints')]:
        checks.append(dict(name=text,curves=old['sampling_checks'][name][old['best_candidate']]['metrics']['explained_percent'],manifold=r['sampling_checks'][name]['pixel_only_projection']['explained_percent']))
    examples=[dict(label=label,source=probe['images'][i].tolist(),curves=previous['reconstructed'][i].tolist(),coordinates=probe['coordinates'][i].tolist(),knobs=probe['knobs'][i].tolist()) for i,label in zip(indices,labels)]
    gain=100*(1-(100-stat['metrics']['explained_percent'])/(100-rows[0]['score']))
    payload=dict(shape=model['shape'].tolist(),support=np.flatnonzero(model['active']).tolist(),examples=examples,rows=rows,checks=checks,best=best,bestScore=stat['metrics']['explained_percent'],denseScore=checks[0]['manifold'],gain=gain,names=['horizontal center','vertical center','height','width','lean'],rank=stat['local_rank'],errorLimit=float(max(.05,np.ceil(np.max(abs(probe['images'][indices]-probe['reconstructed'][indices]))*20)/20)))
    (OUT/'index.html').write_text(HTML.replace('PAYLOAD',json.dumps(payload,separators=(',',':'))))
    browser_nodes=np.asarray(model['nodes'],dtype='<f4')
    (OUT/'decoder.json').write_text(json.dumps(dict(shape=payload['shape'],active_pixel_indices=payload['support'],nodes_float32_base64=base64.b64encode(browser_nodes.tobytes()).decode(),layout='C-order five grid axes, then active pixel; IEEE-754 float32 little-endian'),separators=(',',':')))
    # Record raw range violations at a meaningful tolerance rather than floating-point noise.
    pred=probe['reconstructed'];tol=1e-3;src=probe['images'];square=(src-pred)**2
    r['display_diagnostics']=dict(out_of_range_pixel_fraction_at_tolerance_1e_3=float(np.sum((pred < -tol)|(pred>1+tol))/(len(pred)*784)),maximum_coverage_violation=float(max(0,-pred.min(),pred.max()-1)),browser_float32_max_node_difference=float(np.max(abs(browser_nodes-model['nodes']))),fraction_of_squared_error_on_fractional_coverage_pixels=float(np.sum(square*((src>0)&(src<1)))/np.sum(square)))
    (OUT/'results.json').write_text(json.dumps(r,indent=2)+'\n')
    lines=['# One five-dimensional principal manifold','',
        f'Best fitted grid: **{best}**, shape {tuple(stat["shape"])}, {stat["node_count"]:,} pixel-valued nodes. Exactly one manifold, with five continuous coordinates.','',
        'This is an elastic principal grid with continuous cell interpolation and projection, based on [Gorban and Zinovyev](https://arxiv.org/abs/cond-mat/0405648). It is an implementation variant, rather than a claim to reproduce a particular reference package.','',
        '## Decoder','',
        'The decoder locates the 5D grid cell containing coordinates u in [0,1]^5, then takes a weighted sum of its 32 corner nodes. For each corner, multiply its five one-axis interpolation weights. Weights sum to one. A node stores 283 active pixel values; the other 501 pixels are always zero. This is a joint tensor grid, so it models interactions between coordinates. It does not sum five independent curves. The decoder is continuous across shared cell faces, with possible slope changes at cell boundaries. No renderer, geometric stroke formula, or inverse-rendering solver occurs in the decoder.','',
        '## Fitting and knob information','',
        'All 14,389 covering images receive equal weight. Knob labels initialize u by subtracting each range minimum and dividing by its range width. Known dimension sets five axes. The grids prioritize width and lean based on their observed variation, while coarse center/height axes remain jointly present. These are heuristic allocations, not a proved optimal grid.','',
        'For fixed image coordinates, solve a sparse quadratic node fit: squared pixel reconstruction error + a bending penalty + weak stretching + a tiny numerical ridge. Bending is the squared second difference along each normalized grid axis, with spacing scaling. Then move image coordinates using bounded, damped Gauss–Newton steps and a line search that accepts only lower pixel error. The next node fit uses these updated coordinates.','',
        'Cold-start grids use penalty stages 0.01, 0.001, 0.0001. The fine grid interpolates the completed guided grid as its start, then uses two cycles at 0.001 and five at 0.0001, retaining the starting grid if an update worsens pixel capture. Each node solve uses pixel-column conjugate gradients with target relative residual 3e-6 and a finite iteration cap. The recorded solver history exposes any unconverged columns; this is an approximate numerical solve. Each coordinate refinement takes up to five steps per alternation, followed by twenty at the end. Nodes and coordinates are free after initialization; there is no knob-label loss or physical formula in updates. The box bounds and lattice connections preserve chart organization.','',
        'No samples are withheld from the covering-cloud fit. Additional uniform samples and joint midpoints check approximation between covering points. They are not used for selecting the winning grid; a preliminary guided-grid check was inspected during computation. All 32 corners are already part of the fitting lattice and are a boundary diagnostic. The encoder uses the fitted coordinates of the two nearest covering images in pixel distance as starts, takes up to twenty projection steps each, and chooses the smaller residual. If image L2 error exceeds 0.35, refinement uses eight nearest-image starts and up to sixty steps each, retaining only lower error. This policy also refines difficult covering points without changing learned nodes. Closest-point search is approximate and local.','',
        '## Results','',
        '| Model | Nodes | Covering-cloud capture | Pixel RMS |','|---|---:|---:|---:|']
    lines += [f'| {v["name"]} | {v["nodes"]} | {v["score"]:.6f}% | {v["rms"]:.7f} |' for v in rows]
    lines += ['', '| Additional points | Five curves | One 5D manifold |','|---|---:|---:|']
    lines += [f'| {v["name"]} | {v["curves"]:.6f}% | {v["manifold"]:.6f}% |' for v in checks]
    lines += ['', 'Capture is 100 times one minus total squared reconstruction error divided by total squared distance to the evaluated cloud mean. All 784 unclipped coverage pixels enter the metric. Different clouds have different denominators. Five coordinates do not fix model complexity: grid nodes are separately reported.','',
        '## Limits and validation','',
        f'All {stat["local_rank"]["points"]} checked fitted positions have five independent Jacobian columns at relative singular-value tolerance 1e-6. Extra samples and joint midpoints also have rank checks in results.json. This does not prove global injectivity, complete coverage, or exact agreement with the renderer.','',
        'The learned coordinates correlate with the initial physical knobs but drift during fitting. Feeding true knob fractions directly into the final decoder is a different operation and can have substantially larger error; those scores are recorded separately. The viewer uses learned image positions. Arbitrary slider combinations can leave the valid stroke family or coverage range. No clipping is used to improve reported scores.','',
        'The full fit is nonconvex. Five coordinates suffice in principle because the exact renderer has five inputs; that does not guarantee this finite, smooth grid and local optimizer find an exact manifold or global optimum. More nodes and alternate charts could reduce remaining approximation error.','',
        '## Reproduce','',
        'From the repository root, using the existing Python environment:', '```sh',
        '.venv/bin/python fit_generated_one_principal_manifold.py --phase uniform --cycles 5 --cg-iterations 200',
        '.venv/bin/python fit_generated_one_principal_manifold.py --phase guided --cycles 7 --cg-iterations 200',
        '.venv/bin/python fit_generated_one_principal_manifold.py --phase fine --cycles 5 --cg-iterations 160',
        '.venv/bin/python fit_generated_one_principal_manifold.py --phase evaluate',
        '.venv/bin/python fit_generated_one_principal_manifold.py --phase refine',
        '.venv/bin/python make_generated_one_principal_manifold_view.py','```','',
        'The best NPZ contains nodes, grid shape, fitted image coordinates, active pixels, reference images, and knob labels. Only nodes, shape, and active pixels are needed to decode. decoder.json exports float32 nodes for the browser; measurements use full-precision fitted nodes.','']
    (OUT/'README.md').write_text('\n'.join(lines))
    print(json.dumps(dict(best=best,covering=stat['metrics'],checks=checks,diagnostics=r['display_diagnostics']),indent=2))


if __name__=='__main__':main()
