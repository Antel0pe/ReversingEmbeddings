"""Explain the saved five-curve decoder and measure final pixel clipping."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from grey_ones import render
from optimize_generated_one_joint_curves import evaluate, encode_multistart

OUT=Path('figures/generated_one_principal_curves/decoder')
BASE=OUT.parent


def scores(x,y):
    denominator=float(np.sum((x-x.mean(0))**2))
    raw=float(np.sum((x-y)**2));clipped=float(np.sum((x-np.clip(y,0,1))**2))
    assert clipped<=raw+1e-10
    return dict(images=len(x),raw_capture_percent=100*(1-raw/denominator),
                clipped_capture_percent=100*(1-clipped/denominator),
                denominator_squared_distance_to_mean=denominator,raw_squared_error=raw,
                clipped_squared_error=clipped,residual_squared_error_reduction_percent=100*(1-clipped/raw))


def main():
    OUT.mkdir(exist_ok=True)
    m=np.load(BASE/'joint/joint_513.npz')
    x=render(m['coverage_knobs']).reshape(-1,784).astype(float)
    pred=m['mean']+sum(evaluate(c,m['coordinates'][:,k]) for k,c in enumerate(m['curves']))
    d=np.load(BASE/'joint/joint_513_dense.npz');xd=render(d['knobs']).reshape(-1,784).astype(float)
    yd=np.zeros_like(xd);yd[:,d['active']]=d['reconstructed']
    direct=np.load(BASE/'direct_construction/direct_513_examples.npz')
    support=np.any(abs(m['curves'])>1e-8,axis=1)
    p4=np.array([[14.5,14.5,19.75,3.2,12.5],[14.5,14.5,19.75,4.2,12.5],
                 [14.5,14.5,19.75,3.2,27.5],[14.5,14.5,19.75,4.2,27.5]])
    exact=render(p4).reshape(-1,784).astype(float);active=np.any(x!=0,axis=0)
    yy,t,_=encode_multistart(exact[:,active],m['mean'][active],m['curves'][:,:,active],x[:,active],m['coordinates'])
    parts=np.array([evaluate(c,t[:,k]) for k,c in enumerate(m['curves'])]).transpose(1,0,2)
    reconstruction=m['mean']+parts.sum(1)
    assert np.allclose(reconstruction[:,active],yy,atol=1e-9)
    r=dict(method='Fixed saved curves and positions; clip only the final pixel sum to [0,1]; no refitting or re-encoding for clipping scores',
           score_definition='100 * (1 - sum over images and pixels of (source - prediction)^2 / sum over images and pixels of (source - dataset mean image)^2)',
           freely_fitted_cloud=scores(x,pred),additional_uniform_cloud=scores(xd,yd),
           fixed_knob_cloud=scores(direct['images'],direct['predictions']),
           active_pixels_per_curve=support.sum(1).tolist(),active_pixels_shared_by_all_curves=int(np.all(support,axis=0).sum()),
           support_threshold=1e-8,
           example_knobs=p4.tolist(),example_coordinates=t.tolist(),
           curve_coordinate_knob_spearman=json.loads((BASE/'joint/results.json').read_text())['candidates']['joint_513']['curve_coordinate_knob_spearman'])
    (OUT/'results.json').write_text(json.dumps(r,indent=2)+'\n')
    np.savez_compressed(OUT/'decoder_examples.npz',knobs=p4,coordinates=t,base=m['mean'],parts=parts,
                        raw=reconstruction,clipped=np.clip(reconstruction,0,1),source=exact)
    labels=['Starting stroke','Width +1 px','Lean +15 degrees','Both changes']
    fig,axs=plt.subplots(4,8,figsize=(17,10.2))
    fig.subplots_adjust(left=.105,right=.975,top=.80,bottom=.16,hspace=.45,wspace=.25)
    fig.text(.035,.974,'Five overlapping curve contributions reconstruct each stroke',fontsize=21,weight='bold',va='top')
    fig.text(.035,.921,'Same five learned curves in every row; each selected position returns a 28 x 28 signed pixel contribution.',fontsize=12)
    fig.text(.035,.879,'Base + C1(t1) + C2(t2) + C3(t3) + C4(t4) + C5(t5), then clip the sum to [0, 1]. Positions are inferred from source pixels.',fontsize=11)
    titles=['Base image','+ Curve 1','+ Curve 2','+ Curve 3','+ Curve 4','+ Curve 5','= Clipped sum','Exact source']
    for row in range(4):
        values=[m['mean'],*parts[row],np.clip(reconstruction[row],0,1),exact[row]]
        for col,v in enumerate(values):
            ax=axs[row,col]
            if 1<=col<=5:
                im=ax.imshow(v.reshape(28,28),cmap='RdBu_r',vmin=-.8,vmax=.8,interpolation='nearest')
                ax.set_xlabel(f't{col} = {t[row,col-1]:.3f}',fontsize=10)
            else:ax.imshow(v.reshape(28,28),cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(titles[col],fontsize=11,pad=12)
        axs[row,0].set_ylabel(labels[row],fontsize=11,labelpad=13)
    ca=fig.add_axes([.32,.1,.4,.016]);fig.colorbar(im,cax=ca,orientation='horizontal',label='Curve contributions: blue removes ink; red adds ink. Common coverage scale, no values clipped.')
    fig.text(.035,.025,'Base, sum and source: white = 0 coverage, black = 1. All 784 pixels shown; every curve can change the same 283 active pixels.\nFixed source center (14.5, 14.5), height 19.75 px. Starting width 3.2 px, lean 12.5 degrees. Each curve stores 513 nodes with linear interpolation.',fontsize=10.5)
    fig.savefig(OUT/'decoder_breakdown.png',dpi=145);plt.close(fig)
    data=dict(base=m['mean'].tolist(),parts=parts.tolist(),source=exact.tolist(),coordinates=t.tolist(),knobs=p4.tolist(),labels=labels)
    table=''.join(f'<tr><td>{name}</td><td>{v["images"]:,}</td><td>{v["raw_capture_percent"]:.5f}%</td><td>{v["clipped_capture_percent"]:.5f}%</td></tr>' for name,v in [('Freely fitted curves, fitting cloud',r['freely_fitted_cloud']),('Freely fitted curves, additional uniform images',r['additional_uniform_cloud']),('Knob-indexed curves, same fitting cloud',r['fixed_knob_cloud'])])
    html='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>How the five curves decode</title><style>body{max-width:1300px;margin:30px auto;padding:0 22px;font:17px/1.6 system-ui;color:#172b3a;background:#fafafa}a{color:#2563a6}img{width:100%;height:auto}canvas{width:100%;image-rendering:pixelated;cursor:crosshair;border:1px solid #ccd4dc;display:block}#cards{display:grid;grid-template-columns:repeat(8,minmax(0,1fr));gap:12px;margin:20px 0}article{text-align:center;font-size:14px}article p{line-height:1.4;min-height:42px}td,th{padding:7px 12px;text-align:left}pre{white-space:pre-wrap;background:#e9edf2;padding:15px}select{font:inherit;padding:5px} @media(max-width:800px){#cards{grid-template-columns:repeat(4,1fr)}table{font-size:13px}td,th{padding:5px}}</style></head><body><h1>How five overlapping curves reconstruct a “1”</h1><p><a href="../quality/">Image quality and small moves</a> · <a href="../joint/">Fitting experiment</a> · <a href="results.json">Measurements</a></p><p>The knob-indexed comparison did not start from just one stroke. Its curves were jointly fitted across all 14,389 covering images. A separate complete-grid calculation averages over all combinations of the other four knobs. Its restriction is that each curve position is fixed by exactly one physical knob.</p><p>The freely fitted model chooses five positions together for each image. All five curves contribute to the same 283 active pixels. Their contributions may be positive or negative and can cancel. The curve nodes remain fixed when decoding.</p><h2>Decode an actual image, term by term</h2><label>Source state <select id="state"></select></label><p id="settings"></p><p>Base, sum and source use white = no ink, black = full ink. Contributions use blue = removes ink, red = adds ink, with a shared scale from −0.8 to +0.8 coverage. Click any image to read that pixel's arithmetic.</p><div id="cards"></div><pre id="pixel" aria-live="polite"></pre><img src="decoder_breakdown.png" alt="Base plus five signed curve contributions, clipped sum and exact source at four width and lean settings"><h2>Why the learned positions handle knob interactions</h2><p>Changing width can move several curve positions. Doing that at another lean angle selects different pixel corrections. Nothing requires a width change to produce the same pixel-change vector at every stroke. Each curve is a fixed function of its own coordinate, but that coordinate can depend on all the physical knobs.</p><p>For comparison, the knob-indexed model fixes t1 = normalized horizontal position, t2 = normalized vertical position, and so on. Its width contribution depends only on width. It cannot change its width correction in response to lean. Clipping its final sum adds a nonlinear operation, but does not eliminate the observed gap.</p><h2>What the curves learned</h2><p>Curve 1 strongly tracks lean (rank correlation 0.988 across the fitting cloud) and alone accounts for 91.05% by the reconstruction score. Curves 2–5 supply overlapping corrections that depend on the selected combination: edges, stroke thickness, ends and positioning. They do not have established one-knob meanings. The visualization shows their literal signed pixels; the particular division into curves is not a unique physical decomposition.</p><h2>Captured variance before and after final clipping</h2><table><thead><tr><th>Model and images</th><th>Images</th><th>Raw sum</th><th>Clipped sum</th></tr></thead><tbody>TABLE</tbody></table><p>Same saved curves and same saved positions, no refitting. Clipping maps values below 0 to 0 and above 1 to 1. It leaves positive faint speckles inside [0,1] untouched. Because every source pixel is inside [0,1], clipping cannot increase squared reconstruction error.</p><h2>What “captured variance” measures</h2><pre>mean image = average source pixel values across the dataset
baseline error = sum over every image and pixel of
                 (source pixel − mean-image pixel)²
reconstruction error = sum over every image and pixel of
                       (source pixel − reconstructed pixel)²
captured variance (%) = 100 × (1 − reconstruction error / baseline error)</pre><p>100% means zero pixel reconstruction error; 0% means the same total error as returning the average image every time. Negative scores are possible. This is a reduction in squared error relative to the mean-image baseline, not a count of faithfully captured knobs. Constant white pixels contribute zero to the baseline. Large pixel-space changes influence the score more than small changes. For this nonlinear decoder, residual and reconstruction need not be orthogonal as in PCA, so the remaining percentage is a residual-error ratio, not necessarily a separate independent variance component.</p><script>const D=DATA;const names=['Base','+ Curve 1','+ Curve 2','+ Curve 3','+ Curve 4','+ Curve 5','= Clipped sum','Exact source'];let chosen=0,px=14,py=14;const select=document.getElementById('state');D.labels.forEach((v,i)=>select.add(new Option(v,i)));const cards=document.getElementById('cards');const canvases=names.map((name,j)=>{const a=document.createElement('article');const p=document.createElement('p');p.textContent=name;a.append(p);const c=document.createElement('canvas');c.width=c.height=28;c.setAttribute('aria-label',name);a.append(c);const text=document.createElement('span');text.id='coord'+j;a.append(text);cards.append(a);c.onclick=e=>{const box=c.getBoundingClientRect();px=Math.min(27,Math.floor((e.clientX-box.left)*28/box.width));py=Math.min(27,Math.floor((e.clientY-box.top)*28/box.height));arithmetic()};return c});function raw(){return D.base.map((v,i)=>v+D.parts[chosen].reduce((s,c)=>s+c[i],0))}function arithmetic(){const i=py*28+px;const total=raw()[i];document.getElementById('pixel').textContent='Pixel (row '+py+', column '+px+'), coverage units:\nbase '+D.base[i].toFixed(5)+'\n'+D.parts[chosen].map((p,k)=>'  + C'+(k+1)+' '+p[i].toFixed(5)).join('\n')+'\n= raw '+total.toFixed(5)+'\n= clipped '+Math.max(0,Math.min(1,total)).toFixed(5)+'\nexact source '+D.source[chosen][i].toFixed(5)}function draw(){const sum=raw();const values=[D.base,...D.parts[chosen],sum.map(v=>Math.max(0,Math.min(1,v))),D.source[chosen]];values.forEach((v,k)=>{const ctx=canvases[k].getContext('2d');const image=ctx.createImageData(28,28);v.forEach((x,i)=>{let rgb;if(k>=1&&k<=5){const a=Math.min(1,Math.abs(x)/.8);const color=x>=0?[178,24,43]:[33,102,172];rgb=color.map(c=>Math.round(255*(1-a)+a*c))}else{const c=Math.round(255*(1-Math.max(0,Math.min(1,x))));rgb=[c,c,c]}image.data.set([...rgb,255],4*i)});ctx.putImageData(image,0,0);document.getElementById('coord'+k).textContent=k>=1&&k<=5?'t'+k+' = '+D.coordinates[chosen][k-1].toFixed(5):''});const p=D.knobs[chosen];document.getElementById('settings').textContent='cx '+p[0]+' px; cy '+p[1]+' px; height '+p[2]+' px; width '+p[3]+' px; lean '+p[4]+' degrees.';arithmetic()}select.onchange=()=>{chosen=Number(select.value);draw()};draw();</script></body></html>'''
    html=html.replace('TABLE',table).replace('DATA',json.dumps(data,separators=(',',':')))
    (OUT/'index.html').write_text(html)
    (OUT/'README.md').write_text('# Five-curve decoder explanation\n\nReproduce with `explain_generated_one_curve_decoder.py`. See `results.json` for final-clipping scores and `decoder_examples.npz` for the literal base, selected contributions, coordinates, sums and source pixels. No model refitting. The HTML lets you inspect the arithmetic at any pixel in four source states.\n')
    quality=BASE/'quality/index.html';text=quality.read_text();marker='<!-- decoder-explanation-link -->'
    if marker not in text:text=text.replace('<h1>',marker+'<p><a href="../decoder/">Understand the decoder and clipping scores</a></p><h1>',1);quality.write_text(text)
    print(json.dumps(r,indent=2),flush=True)

if __name__=='__main__':main()
