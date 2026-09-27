"""Create a small standalone canvas view of the generated three-control 1 family.

The 28x28 renderer is defined in make_three_dimensional_ones.py. The displayed
3D charts are interpolated fits to its 784-dimensional pixel distances.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from make_three_dimensional_ones import (
    BOW, LEAN, WIDTH, manifold_view, render, sample_family,
)
from three_control_distance_fit import fit_view, five_image_witness


OUTPUT = Path("figures/three_control_ones_volume_light.html")
PREVIEW = Path("figures/three_control_ones_volume_fit.png")
FINE_W = np.linspace(float(WIDTH[0]), float(WIDTH[-1]), 9)
FINE_B = np.linspace(float(BOW[0]), float(BOW[-1]), 21)
FINE_A = np.linspace(float(LEAN[0]), float(LEAN[-1]), 21)


def chart(interpolator):
    w, b, a = np.meshgrid(FINE_W, FINE_B, FINE_A, indexing="ij")
    xyz = interpolator(np.stack((w, b, a), axis=-1).reshape(-1, 3))
    return np.round(xyz.reshape(9, 21, 21, 3), 4).tolist()


def preview_figure(charts, diagnostics):
    colors = {"lean": "#3f77aa", "bow": "#318590", "width": "#c69243"}
    fig = plt.figure(figsize=(13.5, 6.1), facecolor="white")
    fig.suptitle("One exact three-knob image family, two three-dimensional views",
                 fontsize=16, fontweight="bold", y=.975)
    image_ax = fig.add_axes((.025, .28, .16, .52))
    image_ax.imshow(render(0, 0, 3), cmap="gray_r", vmin=0, vmax=1,
                    interpolation="nearest")
    image_ax.set_title("One member\nlean 0 · bow 0 · width 3", fontsize=10)
    image_ax.axis("off")
    fig.text(.105, .22, "Black = full ink\nWhite = no ink",
             ha="center", fontsize=8, color="#596b77")
    for slot, name in enumerate(("fit", "coords")):
        ax = fig.add_axes((.21 + slot * .39, .22, .35, .63), projection="3d")
        grid = np.asarray(charts[name])
        for side in (0, -1):
            for sheet, color in ((grid[side], colors["width"]),
                                 (grid[:, side], colors["bow"]),
                                 (grid[:, :, side], colors["lean"])):
                ax.plot_surface(sheet[..., 0], sheet[..., 1], sheet[..., 2],
                                color=color, alpha=.28, linewidth=0,
                                shade=False)
        ax.set_box_aspect(np.ptp(grid.reshape(-1, 3), axis=0))
        ax.view_init(elev=19, azim=-55)
        ax.tick_params(labelsize=7)
        if name == "fit":
            ax.set_title("Distance-shaped fit · approximate", fontsize=11,
                         pad=11)
            ax.set(xlabel="display x", ylabel="display y", zlabel="display z")
        else:
            ax.set_title("Exact knob coordinates · rectangular box",
                         fontsize=11, pad=11)
            ax.set(xlabel="lean", ylabel="bow", zlabel="width")
    fig.text(.5, .13,
             "Blue: lean limits   ·   teal: bow limits   ·   gold: width limits. "
             "These transparent faces show the six parameter-boundary families.",
             ha="center", fontsize=9.5)
    fig.text(.5, .075,
             f"Fit: {diagnostics['global_median']:.0%} median random-pair and "
             f"{diagnostics['local_median']:.0%} small-move distance error on "
             "unseen images. The coordinate box is exact for knob settings, "
             "not pixel distances.", ha="center", fontsize=9)
    fig.savefig(PREVIEW, dpi=140, facecolor="white")
    plt.close(fig)


def main():
    np.random.seed(9)  # Fix Isomap's eigensolver start for a repeatable fit.
    settings, images = sample_family()
    original_views, _ = manifold_view(settings, images)
    fitted, diagnostics = fit_view(original_views["local"], images)
    views = {"fit": fitted, "coords": settings}
    interpolators = {
        name: RegularGridInterpolator(
            (WIDTH, BOW, LEAN),
            view.reshape(len(WIDTH), len(BOW), len(LEAN), 3),
            method="linear",
        )
        for name, view in views.items()
    }
    data = json.dumps({
        "charts": {name: chart(ip) for name, ip in interpolators.items()},
        "diagnostics": diagnostics,
        "witness": five_image_witness(),
    }, separators=(",", ":"))
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(TEMPLATE.replace("__DATA__", data), encoding="utf-8")
    preview_figure({name: chart(ip) for name, ip in interpolators.items()},
                   diagnostics)
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size:,} bytes)")
    print("Held-out distance errors and mesh diagnostic:", diagnostics)


TEMPLATE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Three-control generated 1: manifold volume</title>
<style>
:root{color-scheme:light;--ink:#24333d;--muted:#596b77;--line:#dce5e9}
*{box-sizing:border-box}body{margin:0;background:#f4f7f8;color:var(--ink);font:15px/1.4 system-ui,sans-serif}
header{padding:15px 22px;background:white;border-bottom:1px solid var(--line)}h1{font-size:22px;line-height:1.2;margin:0 0 4px}
header p{max-width:1000px;margin:0;color:var(--muted)}main{display:grid;grid-template-columns:minmax(400px,1fr) 300px;gap:12px;padding:12px}
.viewer,.panel{background:white;border:1px solid var(--line);border-radius:8px}.viewer{position:relative;overflow:hidden}
#shape{display:block;width:100%;height:calc(100vh - 113px);min-height:540px;cursor:grab;touch-action:none}#shape.dragging{cursor:grabbing}
.view-note{position:absolute;left:12px;bottom:12px;background:#ffffffdc;border:1px solid var(--line);border-radius:5px;padding:7px 9px;font-size:12px;pointer-events:none}
.panel{padding:15px;max-height:calc(100vh - 113px);overflow:auto}h2{font-size:15px;margin:0 0 7px}h3{font-size:13px;margin:17px 0 6px}
.small{font-size:12px;color:var(--muted);margin:6px 0}.row{display:flex;gap:5px}.row button{flex:1;border:1px solid var(--line);background:#f4f7f8;border-radius:5px;padding:8px 5px;font:12px system-ui;cursor:pointer}
.row button.active{background:var(--ink);color:white}.control{margin:10px 0 13px}.control label{display:flex;justify-content:space-between;font-size:13px;font-weight:600}
input[type=range]{width:100%;accent-color:#d85c4e}select{width:100%;padding:6px;border:1px solid var(--line);border-radius:5px;background:white}
.check{display:flex;gap:7px;align-items:center;font-size:13px;margin:10px 0}#image{width:140px;height:140px;image-rendering:pixelated;border:1px solid var(--line)}
.image-row{display:flex;gap:12px;align-items:center}.metric{padding:8px;background:#eef3f5;border-radius:5px;font-size:12px}
.equation{overflow-wrap:anywhere;background:#f4f7f8;padding:7px;border-radius:5px}
.dot{display:inline-block;width:10px;height:10px;border-radius:2px;margin:0 3px 0 8px;vertical-align:middle}
@media(max-width:850px){main{grid-template-columns:1fr}.panel{max-height:none}#shape{height:57vh;min-height:380px}}
</style></head><body>
<header><h1>The generated “1” manifold in three dimensions</h1>
<p>Every allowed lean, bow, and width gives one exact 28×28 image. The colored faces show the six limits of those settings. The default shape is fitted to pixel distances; “Exact knob coordinates” shows the guaranteed parameter domain. Drag to turn; scroll to zoom.</p></header>
<main><div class="viewer"><canvas id="shape" aria-label="Rotatable three-dimensional view of the generated 1 family"></canvas>
<div class="view-note">Blue faces: lean limits <span class="dot" style="background:#318590"></span>bow limits
<span class="dot" style="background:#c69243"></span>width limits <span class="dot" style="background:#d85c4e"></span>optional interior cut</div></div>
<aside class="panel"><h2>Shape view</h2><div class="row"><button id="fit" class="active" type="button">Distance-shaped fit</button><button id="coords" type="button">Exact knob coordinates</button></div>
<p id="error" class="small"></p>
<div class="control"><label for="alpha">Skin opacity <output id="alpha-out">35%</output></label><input id="alpha" type="range" min="0.1" max="0.8" step="0.01" value="0.35"></div>
<label class="check"><input id="cut-on" type="checkbox">Show an interior sheet</label>
<select id="cut-kind" disabled><option value="width">Fixed width</option><option value="lean">Fixed lean</option><option value="bow">Fixed bow</option></select>
<div class="control"><label for="cut-value">Cut position <output id="cut-out">width 3.00 px</output></label><input id="cut-value" type="range" min="1.8" max="4.2" step="0.01" value="3" disabled></div>
<h3>Choose one generated image</h3>
<div class="control"><label for="lean">Lean <output id="lean-out">0.00 px</output></label><input id="lean" type="range" min="-3" max="3" step="0.01" value="0"></div>
<div class="control"><label for="bow">Bow <output id="bow-out">0.00 px</output></label><input id="bow" type="range" min="-2" max="2" step="0.01" value="0"></div>
<div class="control"><label for="width">Width <output id="width-out">3.00 px</output></label><input id="width" type="range" min="1.8" max="4.2" step="0.01" value="3"></div>
<div class="image-row"><canvas id="image" width="28" height="28" aria-label="Exact generated 1"></canvas><div class="small">The dark marker in the shape view is this image. Black pixels have full ink coverage.</div></div>
<p id="distance" class="metric"></p>
<p id="local-moves" class="metric"></p>
<details><summary class="small">Local pixel-distance rule</summary><p class="small">For a tiny knob move <i>dq</i>, the squared pixel distance is approximately <i>dq</i>ᵀ<i>G</i>(<i>q</i>)<i>dq</i>. This metric is estimated numerically from the exact image generator at the selected 1. At a pixel-boundary kink, the two one-sided directions can differ.</p><pre id="metric" class="small"></pre></details>
<h3>How to read the shape</h3>
<p class="small">The colored faces are six continuous parameter-boundary families, not six separate layers or a finite point cloud. The exact parameter box contains all allowed combinations. The red sheet, when shown, fixes one knob and varies the other two.</p>
<p class="small">The fitted view uses exact images at 405 anchor settings, optimizes image distances and small mixed knob moves, and joins anchors continuously without smoothing across them. This avoids the false folds created by the earlier smooth interpolation. The coordinate view maps each setting to its literal (lean, bow, width) coordinate.</p>
<details><summary class="small">Exact image-family equation</summary>
<p class="small">Let a = lean, b = bow, w = width, y<sub>r,k</sub> = r + (k + 0.5)/16, z(y) = (y − 14.5)/9.875, and x(y) = 14.5 − a z(y) + b(1 − z(y)²).</p>
<p class="small equation">F<sub>r,c</sub>(a,b,w) = (1/16) ∑<sub>k=0</sub><sup>15</sup> v<sub>r,k</sub> max{0, min[c+1, x(y<sub>r,k</sub>)+w/2] − max[c, x(y<sub>r,k</sub>)−w/2]}.</p>
<p class="small">Here v<sub>r,k</sub> is the fraction of subrow [r+k/16, r+(k+1)/16] inside the fixed vertical stroke interval [4.625,24.375]. This defines all 784 pixel values for every allowed setting.</p></details>
<h3>Why the fit cannot be 100%</h3>
<p class="small">Intrinsic dimension three means three numbers identify an image. It does not mean all 784-pixel distances fit in ordinary 3D. Even five legal images at the same width need four independent pixel-space directions: their fourth measured spread is <span id="witness"></span>. No optimization can preserve all pairwise distances exactly in three Euclidean coordinates.</p>
</aside></main>
<script>
const data=__DATA__, $=id=>document.getElementById(id), canvas=$('shape'), ctx=canvas.getContext('2d',{alpha:false});
let layout='fit',yaw=-0.65,pitch=0.37,zoom=1,drag=null,pending=false,dpr=1,W=0,H=0;
let faces=[],edges=[],cutFaces=[],center=[0,0,0],radius=1;
const colors={lean:[63,119,170],bow:[48,144,151],width:[198,151,68],cut:[215,82,69]};
const dims={w:9,b:21,a:21};
function clamp(x,a,b){return Math.max(a,Math.min(b,x))}
function bracket(x,low,high,n){let t=clamp((x-low)*(n-1)/(high-low),0,n-1),i=Math.min(n-2,Math.floor(t));return [i,t-i]}
function point(q){
  if(layout==='coords')return q;
  const chart=data.charts[layout], [wi,wt]=bracket(q[2],1.8,4.2,dims.w),[bi,bt]=bracket(q[1],-2,2,dims.b),[ai,at]=bracket(q[0],-3,3,dims.a);
  let out=[0,0,0];for(let dw=0;dw<2;dw++)for(let db=0;db<2;db++)for(let da=0;da<2;da++){
    const f=(dw?wt:1-wt)*(db?bt:1-bt)*(da?at:1-at),p=chart[wi+dw][bi+db][ai+da];
    for(let k=0;k<3;k++)out[k]+=f*p[k];
  }return out;
}
function quad(p00,p01,p11,p10,color){return {v:[p00,p01,p11,p10],color}}
function makeGrid(rows,color){const out=[];for(let i=0;i<rows.length-1;i++)for(let j=0;j<rows[0].length-1;j++)out.push(quad(rows[i][j],rows[i][j+1],rows[i+1][j+1],rows[i+1][j],color));return out}
function rebuild(){
  const c=data.charts[layout],maxW=dims.w-1,maxB=dims.b-1,maxA=dims.a-1;faces=[];edges=[];
  for(const wi of [0,maxW])faces.push(...makeGrid(c[wi],'width'));
  for(const bi of [0,maxB])faces.push(...makeGrid(c.map(row=>row[bi]),'bow'));
  for(const ai of [0,maxA])faces.push(...makeGrid(c.map(row=>row.map(col=>col[ai])),'lean'));
  for(const bi of [0,maxB])for(const wi of [0,maxW])edges.push(c[wi][bi]);
  for(const ai of [0,maxA])for(const wi of [0,maxW])edges.push(c[wi].map(row=>row[ai]));
  for(const ai of [0,maxA])for(const bi of [0,maxB])edges.push(c.map(row=>row[bi][ai]));
  center=[0,0,0];let count=0;for(const row of c)for(const col of row)for(const p of col){for(let k=0;k<3;k++)center[k]+=p[k];count++}
  center=center.map(x=>x/count);radius=0;for(const row of c)for(const col of row)for(const p of col){radius=Math.max(radius,Math.hypot(...p.map((x,k)=>x-center[k])))}
  rebuildCut();drawSoon();
}
function rebuildCut(){
  cutFaces=[];if(!$('cut-on').checked)return;
  const kind=$('cut-kind').value,val=+$('cut-value').value,n=21,rows=[];
  for(let i=0;i<n;i++){let row=[];for(let j=0;j<n;j++){
    const u=i/(n-1),v=j/(n-1);
    const q=kind==='width'?[-3+6*v,-2+4*u,val]:kind==='lean'?[val,-2+4*v,1.8+2.4*u]:[-3+6*v,val,1.8+2.4*u];
    row.push(point(q));
  }rows.push(row)}cutFaces=makeGrid(rows,'cut');drawSoon();
}
function camera(p){
  const x=(p[0]-center[0])/radius,y=(p[1]-center[1])/radius,z=(p[2]-center[2])/radius;
  const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch);
  const x1=cy*x+sy*z,z1=-sy*x+cy*z,y1=cp*y-sp*z1,z2=sp*y+cp*z1;
  const scale=Math.min(W,H)*0.4*zoom*3.5/(3.5-z2);
  return [W/2+x1*scale,H/2-y1*scale,z2];
}
function polygon(item){
  const v=item.v.map(camera),depth=v.reduce((s,p)=>s+p[2],0)/4;
  return {...item,p:v,depth};
}
function draw(){
  pending=false;if(!W||!H)return;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.fillStyle='#ffffff';ctx.fillRect(0,0,W,H);
  const polys=[...faces,...cutFaces].map(polygon).sort((a,b)=>a.depth-b.depth),alpha=+$('alpha').value;
  for(const poly of polys){
    const p=poly.p,c=colors[poly.color];ctx.beginPath();ctx.moveTo(p[0][0],p[0][1]);for(let i=1;i<4;i++)ctx.lineTo(p[i][0],p[i][1]);ctx.closePath();
    ctx.fillStyle=`rgb(${c[0]},${c[1]},${c[2]})`;ctx.globalAlpha=poly.color==='cut'?0.65:alpha;ctx.fill();
  }ctx.globalAlpha=1;
  ctx.strokeStyle='rgba(34,49,59,.48)';ctx.lineWidth=1.15;
  for(const line of edges){ctx.beginPath();line.forEach((p,i)=>{const v=camera(p);i?ctx.lineTo(v[0],v[1]):ctx.moveTo(v[0],v[1])});ctx.stroke()}
  const marker=camera(point(selectedQ()));ctx.beginPath();ctx.arc(marker[0],marker[1],6,0,Math.PI*2);ctx.fillStyle='#23313b';ctx.fill();ctx.strokeStyle='white';ctx.lineWidth=2;ctx.stroke();
}
function drawSoon(){if(!pending){pending=true;requestAnimationFrame(draw)}}
function resize(){const r=canvas.getBoundingClientRect();W=r.width;H=r.height;dpr=Math.min(window.devicePixelRatio||1,1.5);canvas.width=Math.round(W*dpr);canvas.height=Math.round(H*dpr);drawSoon()}
canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);canvas.classList.add('dragging')});
canvas.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-drag[0])*0.008;pitch=clamp(pitch+(e.clientY-drag[1])*0.008,-1.5,1.5);drag=[e.clientX,e.clientY];drawSoon()});
canvas.addEventListener('pointerup',()=>{drag=null;canvas.classList.remove('dragging')});
canvas.addEventListener('pointercancel',()=>{drag=null;canvas.classList.remove('dragging')});
canvas.addEventListener('wheel',e=>{e.preventDefault();zoom=clamp(zoom*Math.exp(-e.deltaY*0.001),0.55,3.5);drawSoon()},{passive:false});
function image784(q){
  const a=q[0],b=q[1],w=q[2],out=new Float64Array(784),sub=16,top=14.5-19.75/2,bottom=14.5+19.75/2;
  for(let r=0;r<28;r++)for(let k=0;k<sub;k++){
    const lo=r+k/sub,mid=lo+.5/sub,vertical=clamp((Math.min(lo+1/sub,bottom)-Math.max(lo,top))*sub,0,1);
    if(!vertical)continue;const z=(mid-14.5)/(19.75/2),center=14.5-a*z+b*(1-z*z),left=center-w/2,right=center+w/2;
    for(let col=0;col<28;col++)out[r*28+col]+=Math.max(0,Math.min(right,col+1)-Math.max(left,col))*vertical/sub;
  }return out;
}
const base=image784([0,0,3]);function selectedQ(){return [+$('lean').value,+$('bow').value,+$('width').value]}
function localMetric(q){
  const limits=[[-3,3],[-2,2],[1.8,4.2]],h=0.0001,columns=[];
  for(let k=0;k<3;k++){
    const plus=q.slice(),minus=q.slice();plus[k]=Math.min(limits[k][1],q[k]+h);minus[k]=Math.max(limits[k][0],q[k]-h);
    const a=image784(plus),b=image784(minus),step=plus[k]-minus[k],column=new Float64Array(784);
    for(let i=0;i<784;i++)column[i]=(a[i]-b[i])/step;columns.push(column);
  }
  const G=Array.from({length:3},()=>[0,0,0]);
  for(let j=0;j<3;j++)for(let k=0;k<3;k++)for(let i=0;i<784;i++)G[j][k]+=columns[j][i]*columns[k][i];
  return G;
}
function updateImage(){
  const q=selectedQ(),pixels=image784(q),im=$('image').getContext('2d'),bytes=im.createImageData(28,28);let dist=0;
  for(let i=0;i<784;i++){const gray=Math.round(255*(1-pixels[i]));bytes.data.set([gray,gray,gray,255],4*i);dist+=(pixels[i]-base[i])**2}im.putImageData(bytes,0,0);
  for(const [key,val] of [['lean',q[0]],['bow',q[1]],['width',q[2]]])$(key+'-out').textContent=val.toFixed(2)+' px';
  const p=point(q),zero=point([0,0,3]),shown=Math.hypot(...p.map((x,k)=>x-zero[k]));
  $('distance').textContent=`From the central 1: pixel-space distance ${Math.sqrt(dist).toFixed(2)}; drawn 3D distance ${shown.toFixed(2)}.`;drawSoon();
  const G=localMetric(q),rate=G.map((row,k)=>(0.1*Math.sqrt(row[k])).toFixed(2));
  $('local-moves').textContent=`Near this 1, a 0.1 px move changes the 784-pixel image by about: lean ${rate[0]}, bow ${rate[1]}, width ${rate[2]} in pixel distance.`;
  $('metric').textContent='G =\n'+G.map(row=>'  '+row.map(x=>x.toFixed(2).padStart(7)).join(' ')).join('\n')+'\nrows/columns: lean, bow, width';
}
function setLayout(which){layout=which;
  $('fit').classList.toggle('active',which==='fit');$('coords').classList.toggle('active',which==='coords');
  if(which==='fit'){
    const e=data.diagnostics;
    $('error').textContent=`On unseen generated images: median distance error ${Math.round(100*e.global_median)}% for random pairs, ${Math.round(100*e.local_median)}% for small moves. Reversal found in ${e.reversed_fine_cells} of ${e.fine_cell_count} displayed cells (${e.jacobian_samples_per_cell} checks per cell). This does not prove a perfect global embedding.`;
  }else{
    $('error').textContent='Exact lean, bow, and width coordinates: every allowed setting has its proper location in this box. Its ruler is knob units, not 784-pixel distance.';
  }
  rebuild();updateImage();
}
for(const name of ['lean','bow','width'])$(name).addEventListener('input',updateImage);
$('fit').addEventListener('click',()=>setLayout('fit'));$('coords').addEventListener('click',()=>setLayout('coords'));
$('alpha').addEventListener('input',()=>{$('alpha-out').textContent=Math.round(100*(+$('alpha').value))+'%';drawSoon()});
$('cut-on').addEventListener('change',()=>{$('cut-kind').disabled=!$('cut-on').checked;$('cut-value').disabled=!$('cut-on').checked;rebuildCut()});
$('cut-kind').addEventListener('change',()=>{const kind=$('cut-kind').value,range=$('cut-value');
  const bounds=kind==='width'?[1.8,4.2]:kind==='lean'?[-3,3]:[-2,2];range.min=bounds[0];range.max=bounds[1];range.value=kind==='width'?3:0;updateCut()});
function updateCut(){$('cut-out').textContent=$('cut-kind').value+' '+(+$('cut-value').value).toFixed(2)+' px';rebuildCut()}
$('cut-value').addEventListener('input',updateCut);
$('witness').textContent=data.witness[3].toFixed(2)+' (the third is '+data.witness[2].toFixed(2)+')';
window.addEventListener('resize',resize);resize();setLayout('fit');
</script></body></html>'''


if __name__ == "__main__":
    main()
