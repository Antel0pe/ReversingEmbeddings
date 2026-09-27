"""Write a continuous-sheet HTML view of the three-control generated-1 family.

The source image generator is exact for its 16-subrow coverage rule. The 3D
charts are cubic interpolants of sampled Isomap and MDS placements and are
explicitly presented as visual fits, not exact isometric embeddings.
"""

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator


FINE_W = np.linspace(1.8, 4.2, 13)
FINE_B = np.linspace(-2.0, 2.0, 31)
FINE_A = np.linspace(-3.0, 3.0, 31)
FACE_COLORS = {
    "lean low": "#315f94",
    "lean high": "#5c93c0",
    "bow low": "#237e86",
    "bow high": "#62adb1",
    "width low": "#b88234",
    "width high": "#dfb66a",
}
SLICE_COLOR = "#d95a4e"


def model_interpolators(views, lean_values, bow_values, width_values):
    return {
        name: RegularGridInterpolator(
            (width_values, bow_values, lean_values),
            view.reshape(len(width_values), len(bow_values), len(lean_values), 3),
            method="cubic",
        )
        for name, view in views.items()
    }


def parameter_sheet(which, value):
    """Return a 31x31 array of (lean, bow, width) settings."""
    if which == "width":
        b, a = np.meshgrid(FINE_B, FINE_A, indexing="ij")
        w = np.full_like(a, value)
    elif which == "bow":
        w, a = np.meshgrid(FINE_W, FINE_A, indexing="ij")
        b = np.full_like(a, value)
    elif which == "lean":
        w, b = np.meshgrid(FINE_W, FINE_B, indexing="ij")
        a = np.full_like(b, value)
    else:
        raise ValueError(which)
    return np.stack((a, b, w), axis=-1)


def sheet_xyz(interpolator, q):
    query = q[..., [2, 1, 0]].reshape(-1, 3)
    return interpolator(query).reshape(*q.shape[:-1], 3)


def constant_surface(interpolator, q, color, label, opacity):
    xyz = sheet_xyz(interpolator, q)
    return go.Surface(
        x=xyz[..., 0], y=xyz[..., 1], z=xyz[..., 2],
        customdata=q, surfacecolor=np.zeros(q.shape[:2]),
        colorscale=[[0, color], [1, color]], cmin=0, cmax=1,
        opacity=opacity, showscale=False, showlegend=False,
        lighting=dict(ambient=.76, diffuse=.72, specular=.15, roughness=.87),
        hovertemplate=(label + "<br>lean %{customdata[0]:.2f} px"
                       "<br>bow %{customdata[1]:.2f} px"
                       "<br>width %{customdata[2]:.2f} px<extra></extra>"),
        name=label,
    )


def boundary_edges(interpolator, lean, bow, width):
    lines = []
    for b in (bow[0], bow[-1]):
        for w in (width[0], width[-1]):
            lines.append(np.column_stack((FINE_A, np.full(31, b),
                                           np.full(31, w))))
    for a in (lean[0], lean[-1]):
        for w in (width[0], width[-1]):
            lines.append(np.column_stack((np.full(31, a), FINE_B,
                                           np.full(31, w))))
    for a in (lean[0], lean[-1]):
        for b in (bow[0], bow[-1]):
            lines.append(np.column_stack((np.full(13, a), np.full(13, b),
                                           FINE_W)))
    x, y, z = [], [], []
    for q in lines:
        xyz = interpolator(q[:, [2, 1, 0]])
        x.extend([*xyz[:, 0], None])
        y.extend([*xyz[:, 1], None])
        z.extend([*xyz[:, 2], None])
    return go.Scatter3d(
        x=x, y=y, z=z, mode="lines",
        line=dict(color="rgba(31,47,58,.55)", width=2.5),
        hoverinfo="skip", showlegend=False, name="Boundary edges",
    )


def dense_chart(interpolator):
    w, b, a = np.meshgrid(FINE_W, FINE_B, FINE_A, indexing="ij")
    query = np.stack((w, b, a), axis=-1).reshape(-1, 3)
    return np.round(interpolator(query).reshape(len(FINE_W), len(FINE_B),
                                                 len(FINE_A), 3), 5).tolist()


def plot_and_chart(views, lean, bow, width):
    interpolators = model_interpolators(views, lean, bow, width)
    fig = go.Figure()
    groups = {}
    for name in ("local", "global"):
        ip = interpolators[name]
        start = len(fig.data)
        faces = [
            ("lean", lean[0], "lean low"),
            ("lean", lean[-1], "lean high"),
            ("bow", bow[0], "bow low"),
            ("bow", bow[-1], "bow high"),
            ("width", width[0], "width low"),
            ("width", width[-1], "width high"),
        ]
        for axis, value, label in faces:
            fig.add_trace(constant_surface(
                ip, parameter_sheet(axis, value), FACE_COLORS[label],
                f"Boundary · {axis} = {value:g}", .32
            ))
        fig.add_trace(boundary_edges(ip, lean, bow, width))
        fig.add_trace(constant_surface(
            ip, parameter_sheet("width", 3.0), SLICE_COLOR,
            "Interior width slice", .72
        ))
        center = ip([[3.0, 0.0, 0.0]])[0]
        fig.add_trace(go.Scatter3d(
            x=[center[0]], y=[center[1]], z=[center[2]],
            mode="markers", marker=dict(size=8, color="#26333c",
                                          line=dict(color="white", width=2)),
            customdata=[[0.0, 0.0, 3.0]],
            hovertemplate=("Selected generated 1<br>lean %{customdata[0]:.2f} px"
                           "<br>bow %{customdata[1]:.2f} px"
                           "<br>width %{customdata[2]:.2f} px<extra></extra>"),
            showlegend=False, name="Selected image",
        ))
        groups[name] = {
            "outer": list(range(start, start + 6)),
            "edges": start + 6,
            "slice": start + 7,
            "marker": start + 8,
        }
    for index in range(9, len(fig.data)):
        fig.data[index].visible = False
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False), aspectmode="data",
            bgcolor="rgb(250,252,253)",
            camera=dict(eye=dict(x=1.55, y=-1.7, z=1.15)),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="white", showlegend=False,
        uirevision="keep-camera",
    )
    return fig, {name: dense_chart(ip) for name, ip in interpolators.items()}, groups


def write_volume_html(path, views, errors, lean, bow, width):
    """Write the same HTML deliverable as the earlier point-cloud view."""
    fig, dense, groups = plot_and_chart(views, lean, bow, width)
    plot_html = fig.to_html(full_html=False, include_plotlyjs=True,
                            div_id="manifold-plot")
    data_json = json.dumps({
        "charts": dense,
        "groups": groups,
        "ranges": {"lean": [float(lean[0]), float(lean[-1])],
                   "bow": [float(bow[0]), float(bow[-1])],
                   "width": [float(width[0]), float(width[-1])]},
        "errors": errors,
    }, separators=(",", ":"))
    template = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Continuous three-control 1 manifold</title>
<style>
:root{color-scheme:light;--ink:#26333c;--muted:#60717e;--line:#dce4e8;--red:#d95a4e}
*{box-sizing:border-box}body{margin:0;font:15px/1.4 system-ui,sans-serif;color:var(--ink);background:#f4f7f8}
header{background:white;border-bottom:1px solid var(--line);padding:16px 24px}
h1{font-size:22px;line-height:1.2;margin:0 0 5px}p{margin:7px 0}
header p{color:var(--muted);max-width:1050px}
main{display:grid;grid-template-columns:minmax(480px,1fr) 315px;gap:12px;padding:12px;min-height:calc(100vh - 96px)}
.plot-wrap{background:white;border:1px solid var(--line);border-radius:9px;overflow:hidden;position:relative}
#manifold-plot{height:calc(100vh - 125px);min-height:630px}
.plot-note{position:absolute;left:14px;bottom:12px;background:rgba(255,255,255,.88);padding:8px 10px;
border:1px solid var(--line);border-radius:6px;font-size:12px;color:var(--muted);pointer-events:none}
aside{background:white;border:1px solid var(--line);border-radius:9px;padding:15px;overflow:auto;max-height:calc(100vh - 125px)}
h2{font-size:15px;margin:0 0 9px}h3{font-size:13px;margin:17px 0 6px}
.button-row{display:flex;gap:5px}.button-row button{flex:1;border:1px solid var(--line);background:#f4f7f8;
padding:8px 6px;border-radius:5px;font:inherit;font-size:12px;cursor:pointer}
.button-row button.active{background:#26333c;color:white;border-color:#26333c}
.small{font-size:12px;color:var(--muted)}.control{margin:11px 0 14px}
.control label{display:flex;justify-content:space-between;font-weight:600;font-size:13px;margin-bottom:3px}
input[type=range]{width:100%;accent-color:#d95a4e}select{width:100%;padding:7px;border:1px solid var(--line);border-radius:5px;background:white}
.checkbox{display:flex;gap:8px;align-items:center;font-size:13px;margin:10px 0}
#selected-image{width:168px;height:168px;image-rendering:pixelated;border:1px solid var(--line);background:white}
.image-row{display:flex;align-items:center;gap:13px}.metric{background:#eef3f6;padding:9px;border-radius:5px;font-size:12px}
.swatch{display:inline-block;width:9px;height:9px;border-radius:2px;margin:0 3px 0 8px}
@media(max-width:900px){main{grid-template-columns:1fr}aside{max-height:none}#manifold-plot{height:64vh;min-height:450px}}
</style></head><body>
<header><h1>The continuous three-control “1” manifold</h1>
<p>These colored sheets show the six boundary families of the allowed lean, bow, and width settings.
Drag to rotate, scroll to zoom, move the interior slice, and change the controls to generate any image in the family.</p></header>
<main><div class="plot-wrap">__PLOT_HTML__
<div class="plot-note">Blue: lean limit <span class="swatch" style="background:#237e86"></span>bow limit
<span class="swatch" style="background:#b88234"></span>width limit
<span class="swatch" style="background:#d95a4e"></span>interior slice</div></div>
<aside>
<h2>3D placement</h2><div class="button-row">
<button id="layout-local" class="active" type="button">Follow local changes</button>
<button id="layout-global" type="button">Fit overall distances</button></div>
<p id="layout-note" class="small"></p>
<div class="control"><label for="opacity">Outer-sheet opacity <output id="opacity-output">32%</output></label>
<input id="opacity" type="range" min="0.05" max="0.85" step="0.01" value="0.32"></div>
<label class="checkbox"><input id="shell-toggle" type="checkbox" checked>Show the six outer sheets</label>
<h3>Look inside</h3><label for="slice-kind" class="small">Interior sheet</label>
<select id="slice-kind"><option value="width">Fixed width</option><option value="lean">Fixed lean</option>
<option value="bow">Fixed bow</option><option value="none">Hide inner sheet</option></select>
<div class="control"><label for="slice-value">Slice setting <output id="slice-output">width 3.00 px</output></label>
<input id="slice-value" type="range" min="1.8" max="4.2" step="0.01" value="3"></div>
<h3>Generate an exact image</h3>
<div class="control"><label for="lean">Lean <output id="lean-output">0.00 px</output></label>
<input id="lean" type="range" min="-3" max="3" step="0.01" value="0"></div>
<div class="control"><label for="bow">Bow <output id="bow-output">0.00 px</output></label>
<input id="bow" type="range" min="-2" max="2" step="0.01" value="0"></div>
<div class="control"><label for="width">Width <output id="width-output">3.00 px</output></label>
<input id="width" type="range" min="1.8" max="4.2" step="0.01" value="3"></div>
<div class="image-row"><canvas id="selected-image" width="28" height="28" aria-label="Generated 1 image"></canvas>
<div><strong id="selected-coordinates"></strong><p class="small">Black = full ink coverage.<br>White = no ink.</p></div></div>
<p id="distance-readout" class="metric"></p>
<h3>What the sheets mean</h3>
<p class="small">The exact generator continuously maps each allowed (lean, bow, width) to a 784-pixel image.
The colored sheets are the parameter boundaries after a fitted 3D placement. Their smooth display comes from
interpolating that placement; pixel edge events in the generator can still make real direction changes.</p>
<p class="small">Three intrinsic controls do not guarantee an exact distance-preserving 3D shape.
The fit error above measures what this view changes.</p>
</aside></main>
<script>
const volumeData=__VOLUME_DATA__;
const plot=document.getElementById('manifold-plot');
const $=id=>document.getElementById(id);
let layout='local';
const DIM={width:13,bow:31,lean:31};
function bracket(value,low,high,count){
  const t=Math.max(0,Math.min(count-1,(value-low)*(count-1)/(high-low)));
  const i=Math.min(count-2,Math.floor(t));return [i,t-i];
}
function point3(q,which=layout){
  const chart=volumeData.charts[which];
  const [wi,wt]=bracket(q[2],1.8,4.2,DIM.width);
  const [bi,bt]=bracket(q[1],-2,2,DIM.bow);
  const [ai,at]=bracket(q[0],-3,3,DIM.lean);
  const out=[0,0,0];
  for(let dw=0;dw<2;dw++)for(let db=0;db<2;db++)for(let da=0;da<2;da++){
    const weight=(dw?wt:1-wt)*(db?bt:1-bt)*(da?at:1-at);
    const p=chart[wi+dw][bi+db][ai+da];
    for(let k=0;k<3;k++)out[k]+=weight*p[k];
  }
  return out;
}
function image784(q){
  const a=q[0],b=q[1],w=q[2],out=new Float64Array(784),sub=16;
  const top=14.5-19.75/2,bottom=14.5+19.75/2;
  for(let r=0;r<28;r++)for(let k=0;k<sub;k++){
    const ylo=r+k/sub,ymid=ylo+.5/sub;
    const vertical=Math.max(0,Math.min(1,(Math.min(ylo+1/sub,bottom)-Math.max(ylo,top))*sub));
    if(!vertical)continue;
    const z=(ymid-14.5)/(19.75/2);
    const center=14.5-a*z+b*(1-z*z),left=center-w/2,right=center+w/2;
    for(let c=0;c<28;c++){
      const overlap=Math.max(0,Math.min(right,c+1)-Math.max(left,c));
      out[r*28+c]+=overlap*vertical/sub;
    }
  }
  return out;
}
const baseQ=[0,0,3],baseImage=image784(baseQ);
function selectedQ(){return [+$('lean').value,+$('bow').value,+$('width').value]}
function drawImage(pixels){
  const canvas=$('selected-image'),ctx=canvas.getContext('2d');
  const image=ctx.createImageData(28,28);
  for(let i=0;i<784;i++){
    const gray=Math.round(255*(1-pixels[i]));
    image.data.set([gray,gray,gray,255],i*4);
  }
  ctx.putImageData(image,0,0);
}
function updateSelected(){
  const q=selectedQ(),pixels=image784(q),p=point3(q),origin=point3(baseQ);
  drawImage(pixels);
  for(const [key,value] of [['lean',q[0]],['bow',q[1]],['width',q[2]]])
    $(key+'-output').textContent=value.toFixed(2)+' px';
  $('selected-coordinates').textContent=`lean ${q[0].toFixed(2)} · bow ${q[1].toFixed(2)} · width ${q[2].toFixed(2)}`;
  let pixelSquared=0,shownSquared=0;
  for(let i=0;i<784;i++)pixelSquared+=(pixels[i]-baseImage[i])**2;
  for(let i=0;i<3;i++)shownSquared+=(p[i]-origin[i])**2;
  $('distance-readout').textContent=`From the central 1: exact pixel distance ${Math.sqrt(pixelSquared).toFixed(2)}; shown 3D distance ${Math.sqrt(shownSquared).toFixed(2)}.`;
  const marker=volumeData.groups[layout].marker;
  Plotly.restyle(plot,{x:[[p[0]]],y:[[p[1]]],z:[[p[2]]],customdata:[[q]]},[marker]);
}
function sheetQ(kind,value,i,j){
  const u=i/30,v=j/30;
  if(kind==='width')return [-3+6*v,-2+4*u,value];
  if(kind==='lean')return [value,-2+4*v,1.8+2.4*u];
  return [-3+6*v,value,1.8+2.4*u];
}
function updateSlice(){
  const kind=$('slice-kind').value;
  if(kind==='none'){ $('slice-output').textContent='hidden'; updateVisibility();return; }
  const value=+$('slice-value').value;
  $('slice-output').textContent=`${kind} ${value.toFixed(2)} px`;
  const x=[],y=[],z=[],customdata=[];
  for(let i=0;i<31;i++){
    const xr=[],yr=[],zr=[],qr=[];
    for(let j=0;j<31;j++){
      const q=sheetQ(kind,value,i,j),p=point3(q);
      xr.push(p[0]);yr.push(p[1]);zr.push(p[2]);qr.push(q);
    }
    x.push(xr);y.push(yr);z.push(zr);customdata.push(qr);
  }
  const trace=volumeData.groups[layout].slice;
  Plotly.restyle(plot,{x:[x],y:[y],z:[z],customdata:[customdata]},[trace]);
  updateVisibility();
}
function updateVisibility(){
  const group=volumeData.groups[layout],vis=Array(plot.data.length).fill(false);
  if($('shell-toggle').checked){for(const i of group.outer)vis[i]=true;vis[group.edges]=true;}
  if($('slice-kind').value!=='none')vis[group.slice]=true;
  vis[group.marker]=true;
  Plotly.restyle(plot,{visible:vis},Array.from({length:vis.length},(_,i)=>i));
}
function setLayout(which){
  layout=which;
  $('layout-local').classList.toggle('active',which==='local');
  $('layout-global').classList.toggle('active',which==='global');
  const e=volumeData.errors[which];
  $('layout-note').textContent=(which==='local'?'Nearby moves are favored. ':'Overall pair distances are favored. ')+
    `Median error: nearby ${(100*e.local_median).toFixed(0)}%; all pairs ${(100*e.all_median).toFixed(0)}%.`;
  updateVisibility();updateSlice();updateSelected();
}
for(const key of ['lean','bow','width'])$(key).addEventListener('input',updateSelected);
for(const key of ['slice-kind','slice-value'])$(key).addEventListener('input',updateSlice);
$('slice-kind').addEventListener('change',()=>{
  const kind=$('slice-kind').value,range=$('slice-value');
  range.disabled=kind==='none';
  if(kind!=='none'){
    const bounds=volumeData.ranges[kind];
    range.min=bounds[0];range.max=bounds[1];
    range.value=kind==='width'?3:0;
  }
  updateSlice();
});
$('opacity').addEventListener('input',()=>{
  const opacity=+$('opacity').value;
  $('opacity-output').textContent=Math.round(opacity*100)+'%';
  const ids=[...volumeData.groups.local.outer,...volumeData.groups.global.outer];
  Plotly.restyle(plot,{opacity:ids.map(()=>opacity)},ids);
});
$('shell-toggle').addEventListener('change',updateVisibility);
$('layout-local').addEventListener('click',()=>setLayout('local'));
$('layout-global').addEventListener('click',()=>setLayout('global'));
plot.on('plotly_click',event=>{
  const q=event.points[0].customdata;
  if(q&&q.length===3&&q.every(Number.isFinite)){
    $('lean').value=q[0];$('bow').value=q[1];$('width').value=q[2];updateSelected();
  }
});
setLayout('local');
</script></body></html>'''
    html = template.replace("__PLOT_HTML__", plot_html).replace(
        "__VOLUME_DATA__", data_json
    )
    Path(path).write_text(html, encoding="utf-8")
