"""Explain joint curve fitting with measured image comparisons and an HTML view."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from optimize_generated_one_joint_curves import OUT, SOURCE, evaluate, encode
from analyze_generated_one_principal_curves import pixels

BLUE = '#2563a6'
ORANGE = '#bd5a22'


def ink(ax, v, title='', caption=''):
    im = v.reshape(28, 28)
    rgb = np.repeat((1-np.clip(im, 0, 1))[:, :, None], 3, axis=2)
    outside = (im < -1e-6)|(im > 1+1e-6)
    amount = np.minimum(1, .10+3*np.abs(im-np.clip(im, 0, 1)))
    rgb[outside] = ((1-amount[outside, None])*rgb[outside]
                    +amount[outside, None]*np.array([.78, .15, .66]))
    ax.imshow(rgb, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, pad=10)
    if caption:
        ax.text(.5, -.09, caption, ha='center', va='top', fontsize=9, transform=ax.transAxes)


def comparisons(r, data):
    best = r['best_candidate']
    names = ['Existing curves\n65 nodes', 'Joint fitting\n65 nodes',
             'Joint fitting\n129 nodes', 'Joint fitting\n257 nodes', 'Joint fitting\n513 nodes']
    best65 = max((k for k,v in r['candidates'].items() if v['nodes_per_curve']==65),
                 key=lambda k:r['candidates'][k]['metrics']['explained_percent'])
    models = ['baseline', best65, 'joint_129', 'joint_257', 'joint_513']
    scores = [r['baseline']['metrics']['explained_percent']]+[
        r['candidates'][k]['metrics']['explained_percent'] for k in models[1:]]
    fig = plt.figure(figsize=(12.8, 9.4), facecolor='white')
    fig.text(.055, .965, 'Five curves, optimized for their combined reconstruction',
             fontsize=19, weight='bold', va='top')
    fig.text(.055, .919, 'All 14,389 covering images participate in fitting. Each image is 784 pixel ink-coverage values.', fontsize=10.5)
    ax = fig.add_axes([.09, .63, .82, .22])
    missing = 100-np.array(scores)
    ax.bar(np.arange(5), missing, color=[ORANGE]+[BLUE]*4, width=.58)
    ax.set_xticks(np.arange(5), names, fontsize=10)
    ax.set_ylabel('Variation left unreconstructed (%)', fontsize=10)
    ax.set_ylim(0, max(missing)*1.3)
    for i,(v,score) in enumerate(zip(missing,scores)):
        ax.text(i, v+.08, f'{score:.3f}% captured', ha='center', fontsize=10, weight='bold')
    ax.spines[['top','right']].set_visible(False)
    gs = fig.add_gridspec(2, 4, left=.09, right=.91, top=.49, bottom=.15, hspace=.45, wspace=.35)
    error_limit = float(np.ceil(np.max(abs(data['images'][:2]-data['best'][:2]))*10)/10)
    for row, i in enumerate([0, 1]):
        source = data['images'][i]
        old = data['baseline'][i]
        new = data['best'][i]
        title = ['Exact generated image', 'Existing five curves', 'Best joint five curves', 'Remaining pixel error']
        ink(fig.add_subplot(gs[row,0]), source, title[0] if row==0 else '', data['labels'][i])
        ink(fig.add_subplot(gs[row,1]), old, title[1] if row==0 else '', f'L2 error {np.linalg.norm(old-source):.3f}')
        ink(fig.add_subplot(gs[row,2]), new, title[2] if row==0 else '', f'L2 error {np.linalg.norm(new-source):.3f}')
        ax = fig.add_subplot(gs[row,3])
        ax.imshow((source-new).reshape(28,28), cmap='RdBu_r', vmin=-error_limit, vmax=error_limit, interpolation='nearest')
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(title[3] if row==0 else '', fontsize=10, pad=10)
        ax.text(.5,-.09,f'max pixel error {np.max(abs(source-new)):.3f}',transform=ax.transAxes,ha='center',va='top',fontsize=9)
    fig.text(.055,.057,f'Images: white = no ink; black = full ink; magenta tint grows with values outside [0, 1]. Error: red = missing ink; blue = excess ink; scale −{error_limit:g} to +{error_limit:g}.',fontsize=9.3)
    fig.text(.055,.025,'Decoder: intercept + C1(t1) + C2(t2) + C3(t3) + C4(t4) + C5(t5). More nodes give each curve more shape freedom. Best found; no global-optimality proof.',fontsize=9.3)
    fig.savefig(OUT/'joint_comparison.png', dpi=145)
    plt.close(fig)


def curve_figure(model):
    c = model['curves']
    t = model['coordinates']
    fig = plt.figure(figsize=(12.4, 11.8), facecolor='white')
    fig.text(.055,.97,'The five optimized curves: signed pixel contributions',fontsize=19,weight='bold',va='top')
    fig.text(.055,.923,f'Each row is one learned curve with {c.shape[1]} stored nodes. Neighboring nodes connect by straight interpolation.',fontsize=10.5)
    fig.text(.055,.889,'Columns sample the fitted coordinate distribution, from its 5th to 95th percentile. These are contributions, not source images.',fontsize=10)
    gs=fig.add_gridspec(5,5,left=.19,right=.93,top=.84,bottom=.16,hspace=.40,wspace=.25)
    for k in range(5):
        positions=np.quantile(t[:,k],[.05,.25,.5,.75,.95])
        values=evaluate(c[k],positions)
        limit=float(np.ceil(np.max(np.abs(values))*20)/20)
        fig.text(.055,.775-k*.136,f'Curve {k+1}\nscale ±{limit:.2f}',fontsize=10.5,weight='bold',va='center',linespacing=1.7)
        for j,v in enumerate(values):
            ax=fig.add_subplot(gs[k,j])
            ax.imshow(v.reshape(28,28),cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'{[5,25,50,75,95][j]}th percentile\nt = {positions[j]:.3f}',fontsize=9,pad=5)
    ca=fig.add_axes([.27,.103,.55,.014])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-1,1),cmap='RdBu_r'),cax=ca,orientation='horizontal',ticks=[-1,0,1])
    cb.ax.set_xticklabels(['minus row scale','zero','plus row scale'])
    cb.set_label('Signed coverage: blue removes ink; red adds ink. Each row has its labeled scale to reveal small corrections.',fontsize=9.3)
    fig.text(.055,.035,'The five contributions are added to a fitted intercept image. Their shapes are learned from pixels and can mix the physical knobs.\n'
             't is fractional node index, from 0 to 1; it is not a physical knob. This shows curves themselves, with no dimensional projection.',fontsize=10,linespacing=1.6)
    fig.savefig(OUT/'joint_curves.png',dpi=145);plt.close(fig)


HTML = r'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Five curves optimized together</title>
<style>:root{color-scheme:light dark;--bg:#f5f7fa;--card:#fff;--text:#182a3b;--muted:#546577;--line:#dce4ec;--blue:#2365a8}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:16px/1.6 system-ui,sans-serif}main{max-width:1080px;padding:36px 24px 70px;margin:auto}h1{font-size:36px;line-height:1.2}h2{font-size:23px;margin-top:0}section{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:26px;margin:24px 0}a{color:var(--blue)}.lead{font-size:20px}.muted{color:var(--muted);font-size:14px}.controls{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin:20px 0}select,input{font:inherit}input[type=range]{width:260px}.images{display:grid;grid-template-columns:repeat(4,1fr);gap:20px}figure{margin:0;text-align:center}canvas{width:100%;max-width:168px;image-rendering:pixelated;border:1px solid var(--line)}figcaption{font-size:14px;margin-top:8px}table{border-collapse:collapse;width:100%;font-size:15px}td,th{padding:12px 10px;text-align:left;border-bottom:1px solid var(--line)}th{color:var(--muted)}.metric{font-variant-numeric:tabular-nums;font-weight:600}.equation{background:var(--bg);padding:15px;border-radius:8px;font:15px/1.7 ui-monospace,monospace}.wide{max-width:100%;height:auto}details{margin-top:20px}summary{cursor:pointer}@media(prefers-color-scheme:dark){:root{--bg:#101923;--card:#172431;--text:#e8eef5;--muted:#a8bacb;--line:#324455;--blue:#79b6ed}}@media(max-width:700px){main{padding:20px 14px}section{padding:18px}.images{grid-template-columns:repeat(2,1fr)}h1{font-size:29px}table{font-size:13px}td,th{padding:9px 5px}}</style></head>
<body><main><p><a href="../#curve-definition">← Earlier principal-curve experiment</a></p><h1>Five curves optimized together</h1><p class="lead" id="headline"></p><p>The target is the combined five-curve reconstruction. The curves can share and mix the five physical knobs.</p>
<section><h2>What changed?</h2><p>The curves still store pixel-valued nodes joined by straight segments. Each image still gets its closest positions along those curves. We now revisit all five curves: their nodes are fitted together, then image positions are refined, and the process repeats.</p><div class="equation">reconstructed image = fitted intercept<br> + C1(t1) + C2(t2) + C3(t3) + C4(t4) + C5(t5)</div><p>The known knobs supply two alternative starting arrangements. After initialization, curve fitting uses pixel distances alone. The decoder interpolates the stored nodes; it never calls the stroke generator.</p><p class="muted">All 14,389 covering images participate in fitting: 10,240 uniform samples, a 3,125-point lattice, and 1,024 near-face samples. These cover a continuous parameter box approximately; they are not every possible point.</p></section>
<section><h2>Combined variation captured</h2><table><thead><tr><th>Fit</th><th>Nodes per curve</th><th>Capture</th><th>Pixel RMS error</th></tr></thead><tbody id="scoreRows"></tbody></table><p class="muted">Capture = 100 × (1 − total squared reconstruction error / total squared distance to this image cloud’s mean). Higher is better. Each fit has exactly five curves; added nodes increase curve complexity.</p><p id="residualGain" class="metric"></p><p id="tradeoff"></p><details><summary>Initialization comparisons at 65 nodes</summary><table><thead><tr><th>Starting arrangement</th><th>Final capture</th></tr></thead><tbody id="startRows"></tbody></table></details></section>
<section><h2>Compare the actual images</h2><div class="controls"><label for="example">Example</label><select id="example"></select><label for="model">Joint fit</label><select id="model"></select></div><div class="images"><figure><canvas id="source" width="28" height="28"></canvas><figcaption>Exact generated image</figcaption></figure><figure><canvas id="baseline" width="28" height="28"></canvas><figcaption>Existing five curves</figcaption></figure><figure><canvas id="joint" width="28" height="28"></canvas><figcaption>Jointly optimized five curves</figcaption></figure><figure><canvas id="error" width="28" height="28"></canvas><figcaption>Source − joint fit</figcaption></figure></div><p id="exampleDetails" class="metric"></p><p class="muted" id="imageScale"></p></section>
<section><h2>Inspect the five learned curves</h2><p>Move along one curve while the other four contributions stay fixed at the selected example’s fitted positions.</p><div class="controls"><label for="curve">Curve</label><select id="curve"><option value="0">1</option><option value="1">2</option><option value="2">3</option><option value="3">4</option><option value="4">5</option></select><label for="position">Position</label><input id="position" type="range" min="0" max="1000" value="500"></div><div class="images"><figure><canvas id="contribution" width="28" height="28"></canvas><figcaption>Selected curve contribution</figcaption></figure><figure><canvas id="sweep" width="28" height="28"></canvas><figcaption>Sum with the other four fixed</figcaption></figure><figure><canvas id="atExample" width="28" height="28"></canvas><figcaption>Reconstruction at example position</figcaption></figure></div><p id="curveDetails" class="metric"></p><p class="muted" id="contributionScale"></p><p class="muted">Position is fractional node index from 0 to 1. The sum can leave the valid stroke family when one coordinate is changed independently. Each curve is a signed correction, rather than a standalone generated stroke.</p></section>
<section><h2>Check the gaps between sampled points</h2><table><thead><tr><th>Additional points</th><th>Existing curves</th><th>Best joint fit</th></tr></thead><tbody id="probeRows"></tbody></table><p class="muted">Additional rendering checks measure behavior between covering points. They were not used to choose the winning fit. Positions are found from pixels with the same projection procedure; no true knob settings enter the encoder.</p><p>This is the best set found in the reported search, with no proof of global optimality. The five additive curves approximate the manifold; the score does not certify every possible stroke or every coordinate combination.</p></section>
<details><summary>Figures, numerical results, and curve definitions</summary><p><a href="README.md">Method and limits</a> · <a href="results.json">All measured results</a> · <a href="curve_nodes.csv">Best model’s curve nodes</a> · <a href="intercept_pixels.csv">Fitted intercept pixels</a></p><img class="wide" src="joint_comparison.png" alt="Measured reconstruction comparison across curve complexities"><img class="wide" src="joint_curves.png" alt="Signed pixel contributions along all five learned curves"></details>
</main><script id="data" type="application/json">DATA_PLACEHOLDER</script><script>
const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id);
function nodeAt(c,t){const u=Math.max(0,Math.min(1,t))*(c.length-1),j=Math.min(Math.floor(u),c.length-2),f=u-j;return c[j].map((v,i)=>(1-f)*v+f*c[j+1][i]);}
function draw(id,values,scale=0){const ctx=$(id).getContext('2d'),im=ctx.createImageData(28,28);im.data.fill(255);D.support.forEach((idx,i)=>{let rgb;if(scale){const v=values[i],a=Math.min(1,Math.abs(v)/scale);rgb=v>=0?[255,255*(1-a),255*(1-a)]:[255*(1-a),255*(1-a),255];}else{const v=values[i],clipped=Math.max(0,Math.min(1,v)),g=255*(1-clipped);rgb=[g,g,g];if(v<-1e-6||v>1+1e-6){const a=Math.min(1,.1+3*Math.abs(v-clipped));rgb=rgb.map((x,k)=>(1-a)*x+a*[199,38,168][k]);}}for(let k=0;k<3;k++)im.data[4*idx+k]=Math.round(rgb[k]);});ctx.putImageData(im,0,0);}
function reconstructed(model,t){const v=model.mean.slice();model.curves.forEach((c,k)=>{nodeAt(c,t[k]).forEach((x,j)=>v[j]+=x);});return v;}
function l2(a,b){return Math.sqrt(a.reduce((s,v,j)=>s+(v-b[j])**2,0));}
function update(){const i=Number($('example').value),name=$('model').value,m=D.models[name],src=D.images[i],old=D.baseline[i],t=m.coordinates[i],rec=reconstructed(m,t);draw('source',src);draw('baseline',old);draw('joint',rec);draw('error',src.map((v,j)=>v-rec[j]),D.errorLimit);$('exampleDetails').textContent='Image L2 error: existing '+l2(src,old).toFixed(3)+' → joint '+l2(src,rec).toFixed(3)+'. Settings (cx, cy, height, width, lean): '+D.knobs[i].map(v=>v.toFixed(3)).join(', ');const k=Number($('curve').value),pos=Number($('position').value)/1000,part=nodeAt(m.curves[k],pos),q=t.slice();q[k]=pos;const scale=Math.max(.05,Math.ceil(Math.max(...part.map(Math.abs))*20)/20);draw('contribution',part,scale);draw('sweep',reconstructed(m,q));draw('atExample',rec);$('curveDetails').textContent='C'+(k+1)+'('+pos.toFixed(3)+') · example position '+t[k].toFixed(3)+' · '+m.curves[k].length+' nodes';$('contributionScale').textContent='Contribution colors: red adds ink; blue removes ink; scale −'+scale+' to +'+scale+'. This scale adjusts with the selected point so small corrections remain visible.';}
$('headline').textContent=D.bestScore.toFixed(3)+'% of sampled variation captured, compared with '+D.baselineScore.toFixed(3)+'% before joint fitting.';
$('imageScale').textContent='Images: white = no ink; black = full ink; magenta tint grows with the amount outside [0, 1]. Error: red = missing ink; blue = excess ink; fixed scale −'+D.errorLimit+' to +'+D.errorLimit+'. No pixel clipping is used to score the models.';
$('residualGain').textContent=(100*(1-(100-D.bestScore)/(100-D.baselineScore))).toFixed(1)+'% of the previous remaining squared error removed.';
$('tradeoff').textContent='At the same 65-node resolution, curve 1 alone captures '+D.soloBefore.toFixed(2)+'% before joint fitting and '+D.soloAfter.toFixed(2)+'% afterward, while the five-curve total improves. This is a measured case of sacrificing solo capture for a better collection.';
$('scoreRows').innerHTML=D.rows.map(r=>'<tr><td>'+r.name+'</td><td>'+r.nodes+'</td><td>'+r.score.toFixed(3)+'%</td><td>'+r.rms.toFixed(4)+'</td></tr>').join('');
$('startRows').innerHTML=D.starts.map(r=>'<tr><td>'+r.name+'</td><td>'+r.score.toFixed(3)+'%</td></tr>').join('');
$('probeRows').innerHTML=D.probes.map(r=>'<tr><td>'+r.name+'</td><td>'+r.old.toFixed(3)+'%</td><td>'+r.best.toFixed(3)+'%</td></tr>').join('');
$('example').innerHTML=D.labels.map((v,i)=>'<option value="'+i+'">'+v+'</option>').join('');
$('model').innerHTML=Object.keys(D.models).map(k=>'<option value="'+k+'">'+D.models[k].label+'</option>').join('');$('model').value=D.best;
['example','model','curve','position'].forEach(id=>$(id).addEventListener('input',update));update();
</script></body></html>'''


def main():
    r=json.loads((OUT/'results.json').read_text())
    best=r['best_candidate']
    best65=max((k for k,v in r['candidates'].items() if v['nodes_per_curve']==65),key=lambda k:r['candidates'][k]['metrics']['explained_percent'])
    s=np.load(SOURCE); p=s['coverage_knobs']; x=pixels(p)
    original=np.load(OUT/'baseline.npz')
    old=original['mean']+sum(evaluate(c,original['coordinates'][:,k]) for k,c in enumerate(original['curves']))
    winner=np.load(OUT/f'{best}.npz')
    new=winner['mean']+sum(evaluate(c,winner['coordinates'][:,k]) for k,c in enumerate(winner['curves']))
    errors=np.linalg.norm(x-old,axis=1)
    indices=[int(np.argsort(errors)[len(errors)//2]),int(np.argmax(errors)),int(np.argmax(np.linalg.norm(x-new,axis=1)))]+list(range(12))
    labels=['Typical previous error','Worst previous error','Worst remaining joint error']+[f'Covering image {i+1}' for i in range(12)]
    xx=x[indices]; pp=p[indices]
    data=dict(images=xx,baseline=old[indices],best=new[indices],labels=labels)
    comparisons(r,data);curve_figure(winner)
    rows=[dict(name='Existing curves, positions refined',nodes=65,score=r['baseline']['metrics']['explained_percent'],rms=r['baseline']['metrics']['rms_pixel_error'])]
    model_labels={best65:'Joint fitting, 65 nodes','joint_129':'Joint fitting, 129 nodes','joint_257':'Joint fitting, 257 nodes','joint_513':'Joint fitting, 513 nodes'}
    models={}
    for label,text in model_labels.items():
        model=np.load(OUT/f'{label}.npz')
        stat=r['candidates'][label]
        models[label]=dict(mean=model['mean'].tolist(),curves=model['curves'].tolist(),coordinates=model['coordinates'][indices].tolist(),label=text)
        rows.append(dict(name=text,nodes=stat['nodes_per_curve'],score=stat['metrics']['explained_percent'],rms=stat['metrics']['rms_pixel_error']))
    starts=[]
    names={'greedy_65':'Existing greedy fit','knob_additive_65':'Knob-informed additive start','lean_seeded_65':'Lean-informed first curve + residual starts'}
    for label,text in names.items():
        starts.append(dict(name=text,score=r['candidates'][label]['metrics']['explained_percent']))
    probes=[dict(name=text,old=r['sampling_checks'][key]['baseline']['metrics']['explained_percent'],best=r['sampling_checks'][key][best]['metrics']['explained_percent'])
            for key,text in [('dense_uniform','16,384 additional uniform points'),('all_32_corners','All 32 corners'),('joint_midpoints','2,048 joint midpoints')]]
    error_limit=float(np.ceil(max(np.max(abs(xx-(np.array(m['mean'])+sum(evaluate(np.array(c),np.array(m['coordinates'])[:,k]) for k,c in enumerate(m['curves']))))) for m in models.values())*10)/10)
    active=np.any(x!=0,axis=0)
    for m in models.values():
        m['mean']=np.round(np.array(m['mean'])[active],6).tolist()
        m['curves']=np.round(np.array(m['curves'])[:,:,active],6).tolist()
    payload=dict(images=np.round(xx[:,active],6).tolist(),knobs=pp.tolist(),baseline=np.round(old[indices][:,active],6).tolist(),labels=labels,models=models,
                 rows=rows,starts=starts,probes=probes,best=best,bestScore=r['candidates'][best]['metrics']['explained_percent'],
                 baselineScore=r['baseline']['metrics']['explained_percent'],errorLimit=error_limit,support=np.flatnonzero(active).tolist(),
                 soloBefore=r['baseline']['first_curve_alone_explained_percent'],soloAfter=r['candidates'][best65]['first_curve_alone_explained_percent'])
    (OUT/'index.html').write_text(HTML.replace('DATA_PLACEHOLDER',json.dumps(payload,separators=(',',':'))))
    # Export literal numerical curves so the decoder can be reproduced without renderer code.
    import csv
    with (OUT/'curve_nodes.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['curve','node','position']+[f'pixel_{i//28}_{i%28}' for i in range(784)])
        for k,c in enumerate(winner['curves']):
            for i,v in enumerate(c):w.writerow([k+1,i,i/(len(c)-1)]+v.tolist())
    np.savetxt(OUT/'intercept_pixels.csv',winner['mean'].reshape(28,28),delimiter=',')
    np.savez_compressed(OUT/'shown_examples.npz',indices=indices,knobs=pp,images=xx,baseline=old[indices],best=new[indices])
    lines=['# Five curves optimized together','',
        'The goal is minimum combined squared pixel reconstruction error with exactly five learned additive curves. Knob alignment is not required.','',
        'Each curve is a list of pixel-space nodes joined by straight interpolation. Reconstruction is a fitted intercept plus the sum of five curve values. No renderer call or physical stroke equation occurs in this decoder.','',
        'All 14,389 covering points participate in each fit, with equal weights. This is a finite approximation to the continuous five-knob range box, not every possible image. Scores use the mean of the evaluated cloud.','',
        '| Fit | Nodes per curve | Captured variation | Pixel RMS error |',
        '| --- | ---: | ---: | ---: |']
    lines += [f'| {v["name"]} | {v["nodes"]} | {v["score"]:.6f}% | {v["rms"]:.6f} |' for v in rows]
    lines += ['',f'At the matched 65-node resolution, the first curve alone drops from {r["baseline"]["first_curve_alone_explained_percent"]:.6f}% to {r["candidates"][best65]["first_curve_alone_explained_percent"]:.6f}% capture while the combined five-curve score improves. Solo capture projects the image-minus-intercept onto curve 1 with all other curve contributions zero.','']
    lines += ['', '## Fitting choices','',
        'At 65 nodes, compare the previous greedy result, a known-knob additive initial fit, and a lean-informed initial path with residual starts. Knob values only choose starting arrangements; all positions are free during subsequent fitting.','',
        'Each cycle projects images onto the current segments, fits all five node sets jointly by least squares with a second-difference smoothness penalty, and projects again. Unlike the previous routine, it does not resample nodes to uniform arc length after each update; fractional node index defines interpolation. This avoids a lossy resampling step.','',
        'For 65-node fits the penalty schedule is 1, 0.1, 0.01. For higher resolution, insert interpolated nodes into the best previous curves and use 0.01 and 0.001 times the cube of relative node density. Finally polish the 513-node fit with penalties 0.0512 and 0.00512. Keep the lowest full-cloud reconstruction error. Histories record every completed update. The model intercept is learned jointly and curve constants are centered to resolve additive gauge freedom.','',
        'The encoder refines closest-point coordinates from two pixel-only starts: sequential projections and the coordinates of the nearest covering image. The generator is used to supply data and initial paths, then to render additional diagnostic points. It is absent from projection, node fitting, and decoding.','',
        '## Additional sampling checks','',
        '| Additional points | Existing curves | Best joint fit |','| --- | ---: | ---: |']
    lines += [f'| {v["name"]} | {v["old"]:.6f}% | {v["best"]:.6f}% |' for v in probes]
    lines += ['', 'No fitting split is reserved. These additional points diagnose finite sampling resolution and were not used to choose the winning fit. All corners are also part of the covering lattice.','',
        '## Limits','',
        'Best found in the reported search; no certificate of a global optimum. More nodes increase shape freedom, so higher-resolution results are not same-complexity comparisons. Five additive curves remain a restricted decoder. High sampled capture is not an exact representation or a guarantee that arbitrary coordinate combinations yield valid strokes. Pixels outside [0, 1] are not clipped for scoring and are shown in magenta.','',
        'Use index.html for image and curve controls, results.json for measurements, curve_nodes.csv and intercept_pixels.csv for the literal best decoder. Preview nodes are rounded to six decimals for page size; fitting, scoring, and exported curves retain full precision. Run optimize_generated_one_joint_curves.py with phases 65, 129, 257, 513, polish, evaluate, then make_generated_one_joint_curve_view.py from the repository root.','']
    (OUT/'README.md').write_text('\n'.join(lines))
    parent=OUT.parent/'index.html'; text=parent.read_text()
    marker='<!-- joint-curve-result -->'
    if marker not in text:
        section=marker+'<section id="joint-curves"><h2>Optimize the five curves together</h2><p>The new experiment revisits all five learned curves and maximizes their combined reconstruction. <a href="joint/">Open the joint-fit comparison and inspect the curves</a>.</p></section>'
        text=text.replace('<!-- curve-followup-start -->',section+'\n<!-- curve-followup-start -->')
        parent.write_text(text)
    print('Created joint-fit figures, interactive view, literal curves, and report.',flush=True)


if __name__=='__main__':main()
