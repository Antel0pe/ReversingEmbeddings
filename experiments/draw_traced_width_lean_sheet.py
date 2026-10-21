"""Draw the actual crossing families of image curves, with visible distortion.

Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/draw_traced_width_lean_sheet.py
"""
import base64
import gzip
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR","/tmp/traced-sheet-mpl")
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PolyCollection
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import minimize
from scipy.spatial.distance import pdist,squareform

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from grey_ones import render
from experiments.width_vectors_by_lean import norm,angle

OUT=ROOT/"figures/generated_one_width_vectors_by_lean/traced_sheet"
SOURCE=OUT.parent/"sheet_views/sheet_data.npz"
BLUE="#226eaa"
ORANGE="#bd6325"


def faces_for(nt,nw):
    faces=[]
    for t in range(nt-1):
        for w in range(nw-1):
            a=t*nw+w;b=a+1;c=a+nw;d=c+1
            faces.extend([[a,b,d],[a,d,c]])
    return np.array(faces,dtype=int)


def area(y,faces):
    a=y[faces[:,1]]-y[faces[:,0]];b=y[faces[:,2]]-y[faces[:,0]]
    return .5*(a[:,0]*b[:,1]-a[:,1]*b[:,0])


def tick_indices(nt,nw):
    i=np.arange(nt*nw).reshape(nt,nw)
    wt=i.copy();wh=i.copy();lt=i.copy();lh=i.copy()
    wt[:,-1]=i[:,-2];wh[:,:-1]=i[:,1:]
    lt[-1]=i[-2];lh[:-1]=i[1:]
    return [v.ravel() for v in [wt,wh,lt,lh]]


def fit_data(images,distances,shape,faces):
    nt,nw=shape;n=len(images);ea=[];eb=[]
    for dt,dw in [(0,1),(1,0),(1,1),(1,-1),(0,2),(2,0)]:
        for t in range(nt-dt):
            for w in range(nw):
                if 0<=w+dw<nw:
                    ea.append(t*nw+w);eb.append((t+dt)*nw+w+dw)
    ea,eb=np.array(ea),np.array(eb)
    wt,wh,lt,lh=tick_indices(nt,nw)
    u,v=images[wh]-images[wt],images[lh]-images[lt]
    cosine=np.sum(u*v,axis=1)/(norm(u)*norm(v))
    a=images[faces[:,1]]-images[faces[:,0]];b=images[faces[:,2]]-images[faces[:,0]]
    true_area=.5*np.sqrt(np.maximum(0,np.sum(a*a,axis=1)*np.sum(b*b,axis=1)-np.sum(a*b,axis=1)**2))
    assert true_area.min()>0
    return dict(n=n,distances=distances,ea=ea,eb=eb,wt=wt,wh=wh,lt=lt,lh=lh,
                cosine=cosine,true_area=true_area,faces=faces)


def objective(data,path_weight,angle_weight,area_weight=.02):
    n=data["n"];D=data["distances"];ea,eb=data["ea"],data["eb"]
    W=np.full((n,n),1/np.sum(D*D))
    local=path_weight/(2*len(ea)*D[ea,eb]**2)
    W[ea,eb]+=local;W[eb,ea]+=local;np.fill_diagonal(W,0)
    faces=data["faces"];reference=data["true_area"]
    wt,wh,lt,lh=[data[k] for k in ["wt","wh","lt","lh"]]

    def fun(flat):
        y=flat.reshape(n,2);d=squareform(pdist(y));delta=d-D
        value=float(np.sum(W*delta*delta))
        K=W*np.divide(delta,d,out=np.zeros_like(d),where=d>1e-12)
        grad=4*(K.sum(1)[:,None]*y-K@y)
        # Explicitly fit the Gram angle of the actual two available raw ticks.
        u=y[wh]-y[wt];v=y[lh]-y[lt];un=np.maximum(norm(u),1e-12);vn=np.maximum(norm(v),1e-12)
        cosine=np.sum(u*v,axis=1)/(un*vn);error=cosine-data["cosine"]
        value+=angle_weight*np.mean(error*error)
        gu=2*angle_weight/n*error[:,None]*(v/(un*vn)[:,None]-cosine[:,None]*u/(un*un)[:,None])
        gv=2*angle_weight/n*error[:,None]*(u/(un*vn)[:,None]-cosine[:,None]*v/(vn*vn)[:,None])
        np.add.at(grad,wh,gu);np.add.at(grad,wt,-gu);np.add.at(grad,lh,gv);np.add.at(grad,lt,-gv)
        # Actual 784D triangle areas regularize collapsed/inverted triangles.
        # The penalty has a smooth quadratic continuation below area ratio .05;
        # residual inversions are measured, not declared impossible.
        ratio=area(y,faces)/reference;cut=.05;safe=np.maximum(ratio,cut);offset=ratio-cut
        penalty=np.where(ratio>cut,ratio-np.log(safe)-1,
                         cut-np.log(cut)-1+(1-1/cut)*offset+.5*offset*offset/(cut*cut))
        derivative=np.where(ratio>cut,1-1/safe,1-1/cut+offset/(cut*cut))
        value+=area_weight*np.mean(penalty)
        coef=area_weight/len(faces)*derivative/reference
        p,q,r=y[faces[:,0]],y[faces[:,1]],y[faces[:,2]]
        np.add.at(grad,faces[:,0],.5*coef[:,None]*np.c_[q[:,1]-r[:,1],r[:,0]-q[:,0]])
        np.add.at(grad,faces[:,1],.5*coef[:,None]*np.c_[r[:,1]-p[:,1],p[:,0]-r[:,0]])
        np.add.at(grad,faces[:,2],.5*coef[:,None]*np.c_[p[:,1]-q[:,1],q[:,0]-p[:,0]])
        return float(value),grad.ravel()
    return fun


def initialize(raw,unfolded,faces):
    flat=unfolded.copy()-unfolded.mean(0)
    if np.median(area(flat,faces))<0:flat[:,1]*=-1
    raw=raw.copy()-raw.mean(0)
    rotation,_=orthogonal_procrustes(raw,flat);raw=raw@rotation
    dflat=squareform(pdist(flat));draw=squareform(pdist(raw))
    flat*=np.sum(dflat*draw)/np.sum(dflat*dflat)
    alpha=0.
    for value in np.linspace(0,1,101):
        if area((1-value)*flat+value*raw,faces).min()<=1e-6:break
        alpha=value
    alpha=max(0,alpha-.04)
    return (1-alpha)*flat+alpha*raw


def orient(y,nt,nw,widths,leans,root):
    y=y-y[root]
    zero=int(np.argmin(abs(leans)))
    chord=y[zero*nw+nw-1]-y[zero*nw]
    a=np.arctan2(chord[1],chord[0]);c,s=np.cos(a),np.sin(a)
    y=y@np.array([[c,-s],[s,c]])
    wi=int(np.argmin(abs(widths-3.2)))
    if y[(nt-1)*nw+wi,1]<y[wi,1]:y[:,1]*=-1
    return y


def geometry(y,shape,indices):
    nt,nw=shape;wt,wh,lt,lh=indices
    u,v=y[wh]-y[wt],y[lh]-y[lt]
    grid=y.reshape(nt,nw,-1)
    return dict(width_tick_lengths=norm(u),lean_tick_lengths=norm(v),
                crossing_angles=angle(u,v),
                width_curve_lengths=norm(np.diff(grid,axis=1)).sum(1),
                lean_curve_lengths=norm(np.diff(grid,axis=0)).sum(0))


def report(y,source,distances,shape,faces,indices):
    g=geometry(y,shape,indices);actual=geometry(source,shape,indices)
    delta=pdist(y);target=squareform(distances,checks=False)
    areas=area(y,faces);majority=1 if np.median(areas)>=0 else -1
    error=abs(g["crossing_angles"]-actual["crossing_angles"])
    result=dict(image_gap_relative_RMS=float(np.sqrt(np.sum((delta-target)**2)/np.sum(target**2))),
                width_tick_relative_RMS=float(np.sqrt(np.mean((g["width_tick_lengths"]/actual["width_tick_lengths"]-1)**2))),
                lean_tick_relative_RMS=float(np.sqrt(np.mean((g["lean_tick_lengths"]/actual["lean_tick_lengths"]-1)**2))),
                crossing_angle_error_median=float(np.median(error)),crossing_angle_error_p95=float(np.quantile(error,.95)),
                reversed_triangles=int(np.sum(areas*majority<0)),triangle_count=len(faces),
                orientation_sign=majority,
                coordinates="Root-centered; rotated so the root-lean min-to-max width chord points right. These are fitted display positions, not PCA coefficients or raw knob axes.")
    # A good crossing angle does not imply that a whole curve bends correctly.
    # Measure consecutive-segment turning separately for each path family.
    original_grid=source.reshape(*shape,-1);drawn_grid=y.reshape(*shape,2)
    for axis,name in [(1,"width"),(0,"lean")]:
        original_steps=np.diff(original_grid,axis=axis);drawn_steps=np.diff(drawn_grid,axis=axis)
        if axis==1:
            true_turn=angle(original_steps[:,:-1],original_steps[:,1:])
            drawn_turn=angle(drawn_steps[:,:-1],drawn_steps[:,1:])
        else:
            true_turn=angle(original_steps[:-1],original_steps[1:])
            drawn_turn=angle(drawn_steps[:-1],drawn_steps[1:])
        error=abs(true_turn-drawn_turn)
        result[name+"_turn_error_median"]=float(np.median(error))
        result[name+"_turn_error_p95"]=float(np.quantile(error,.95))
    return result,g


def local_cross(ax,width_length,lean_length,theta,limit,title):
    a=np.deg2rad(theta)
    ax.annotate("",(width_length,0),(0,0),arrowprops=dict(arrowstyle="->",color=BLUE,lw=2))
    ax.annotate("",(lean_length*np.cos(a),lean_length*np.sin(a)),(0,0),arrowprops=dict(arrowstyle="->",color=ORANGE,lw=2))
    ax.set(xlim=(-limit*.35,limit*1.1),ylim=(-limit*.15,limit*1.15),aspect="equal",title=title)
    ax.axhline(0,color="#ddd",lw=.7);ax.axvline(0,color="#ddd",lw=.7)
    ax.set_xticks([]);ax.set_yticks([])


def figure(layout,measured,actual,widths,leans,root,faces,metrics):
    nt,nw=len(leans),len(widths);grid=layout.reshape(nt,nw,2)
    fig=plt.figure(figsize=(15,8.3),facecolor="white")
    fig.text(.04,.95,"The sheet traced by width paths and lean paths",fontsize=22,weight="bold")
    fig.text(.04,.891,"Blue: change width at fixed lean. Orange: change lean at fixed width. Every intersection is an actual generated 1.\n"
             "Placement is fitted using all 784 pixel values, including global image gaps and local path geometry. Center (14,14), height 19.75 px fixed.",fontsize=11,linespacing=1.5)
    ax=fig.add_axes([.055,.31,.55,.46])
    for t in range(nt):ax.plot(*grid[t].T,color=BLUE,alpha=.45,lw=.7)
    for w in range(nw):ax.plot(*grid[:,w].T,color=ORANGE,alpha=.45,lw=.7)
    rt,rw=divmod(root,nw)
    ax.plot(*grid[rt].T,color=BLUE,lw=2.6,label="Width path through selected 1")
    ax.plot(*grid[:,rw].T,color=ORANGE,lw=2.6,label="Lean path through selected 1")
    ax.scatter(*layout[root],s=45,color="#151f28",zorder=5)
    sign=metrics["orientation_sign"];bad=area(layout,faces)*sign<0
    if bad.any():ax.add_collection(PolyCollection(layout[faces[bad]],facecolor="#cf384b",alpha=.35,edgecolor="none"))
    for p,label,color in [(grid[rt,0],"width 1.8",BLUE),(grid[rt,-1],"width 4.6",BLUE),
                          (grid[0,rw],"lean −10°",ORANGE),(grid[-1,rw],"lean 35°",ORANGE)]:
        ax.annotate(label,p,xytext=(5,6),textcoords="offset points",fontsize=10,color=color)
    ax.set_aspect("equal",adjustable="box");ax.margins(.13);ax.set_xlabel("Fitted display distance · source path values label the curves")
    ax.set_ylabel("Fitted display coordinate 2");ax.grid(alpha=.12)
    fig.legend(*ax.get_legend_handles_labels(),loc="upper left",bbox_to_anchor=(.05,.835),frameon=False,ncol=2,fontsize=10)
    limit=max(actual["width_tick_lengths"][root],actual["lean_tick_lengths"][root],measured["width_tick_lengths"][root],measured["lean_tick_lengths"][root])
    a=fig.add_axes([.65,.47,.14,.24]);b=fig.add_axes([.82,.47,.14,.24])
    local_cross(a,*[actual[k][root] for k in ["width_tick_lengths","lean_tick_lengths","crossing_angles"]],limit,"Actual 784D crossing")
    local_cross(b,*[measured[k][root] for k in ["width_tick_lengths","lean_tick_lengths","crossing_angles"]],limit,"Crossing in this drawing")
    for left,values in [(.65,actual),(.82,measured)]:
        fig.text(left,.39,f"width tick {values['width_tick_lengths'][root]:.3f}\nlean tick {values['lean_tick_lengths'][root]:.3f}\nangle {values['crossing_angles'][root]:.1f}°",fontsize=10,linespacing=1.5)
    fig.text(.65,.765,"Selected root: width 3.2 px, lean 0°\nCompare a +0.1 px width tick and a +1° lean tick.",fontsize=10.5,linespacing=1.5)
    fig.text(.65,.275,"These two arrows can be drawn exactly in 2D\nusing their actual 784D lengths and angle.\nTheir surrounding sheet may not fit equally well.",fontsize=10.5,linespacing=1.5)
    fig.text(.04,.125,f"All-image distance error: {metrics['image_gap_relative_RMS']:.1%}. Median crossing-angle error: {metrics['crossing_angle_error_median']:.1f}°.\n"
             f"Width-step relative error: {metrics['width_tick_relative_RMS']:.1%}; lean-step relative error: {metrics['lean_tick_relative_RMS']:.1%}.\n"
             f"Median curve-turn errors: width {metrics['width_turn_error_median']:.1f}°, lean {metrics['lean_turn_error_median']:.1f}°.\n"
             f"Red triangles mark {metrics['reversed_triangles']} local orientation reversals in the drawing out of {metrics['triangle_count']:,} triangles.",fontsize=10.5,linespacing=1.5)
    fig.text(.04,.045,"1,334 samples: 29 widths × 46 leans, spanning both allowed ranges. Path lengths and crossing angles use the available finite image steps.\n"
             "Every curve keeps its true width/lean order. Fitted positions approximate the 784D geometry; they are not an exact reconstruction or lossless flattening.",fontsize=10,linespacing=1.5)
    fig.savefig(OUT/"10_traced_sheet_geometry.png",dpi=150);fig.savefig(OUT/"10_traced_sheet_geometry.svg")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    a=np.load(SOURCE);widths,leans=a["widths"],a["leans"]
    nt,nw=len(leans),len(widths);n=nt*nw;root=int(a["root_index"])
    source=a["images"].reshape(n,784).astype(float);D=a["chord_distances"]
    p=np.tile([14.,14.,19.75,3.2,0.],(nt,nw,1));p[:,:,3]=widths;p[:,:,4]=leans[:,None]
    assert np.array_equal(source,render(p.reshape(-1,5)).reshape(n,784))
    faces=faces_for(nt,nw);indices=tick_indices(nt,nw);fdata=fit_data(source,D,(nt,nw),faces)
    init=initialize(a["mds2"],a["unfold2"],faces)
    # Check the nontrivial objective gradient before fitting.
    fun=objective(fdata,.3,.1)
    value,gradient=fun(init.ravel());rng=np.random.default_rng(20261020)
    errors=[]
    for index in rng.choice(init.size,12,replace=False):
        plus=init.ravel().copy();minus=plus.copy();plus[index]+=1e-6;minus[index]-=1e-6
        finite=(fun(plus)[0]-fun(minus)[0])/(2e-6)
        errors.append(abs(finite-gradient[index])/max(1e-7,abs(finite),abs(gradient[index])))
    assert max(errors)<.001
    layouts={"global":orient(a["mds2"],nt,nw,widths,leans,root)}
    optimization={}
    for name,path_weight,angle_weight in [("balanced",.3,.1),("paths",3.,1.)]:
        print("Fitting "+name,flush=True)
        objective_fn=objective(fdata,path_weight,angle_weight)
        result=minimize(objective_fn,init.ravel(),jac=True,method="L-BFGS-B",options=dict(maxiter=1000,ftol=1e-11,gtol=1e-7,maxls=50,maxcor=20))
        layouts[name]=orient(result.x.reshape(n,2),nt,nw,widths,leans,root)
        optimization[name]=dict(path_weight=path_weight,angle_weight=angle_weight,area_weight=.02,
                                iterations=int(result.nit),converged=bool(result.success),stop_reason=str(result.message),
                                objective=float(result.fun),initial_objective=float(objective_fn(init.ravel())[0]))
        assert result.fun<=objective_fn(init.ravel())[0]+1e-8
    actual=geometry(source,(nt,nw),indices)
    metrics={};per_layout={}
    for key,y in layouts.items():
        metrics[key],per_layout[key]=report(y,source,D,(nt,nw),faces,indices)
    # Default follows actual path vectors as well as global image distances.
    default="balanced"
    summary=dict(fixed=dict(cx=14,cy=14,height=19.75),root=[14,14,19.75,3.2,0],
                 vertices=n,width_count=nw,lean_count=nt,width_tick_px=.1,lean_tick_degrees=1,
                 actual_geometry={k:v.tolist() for k,v in actual.items()},layouts=metrics,optimization=optimization,
                 validation=dict(all_images_match_renderer=True,gradient_max_relative_error=float(max(errors))),
                 default=default,
                 definitions=dict(global_error="Relative RMS error of all distinct image-pair Euclidean distances in the original 784 values.",
                                  tick_error="RMS relative error of available increasing width or lean ticks, with the final backward interval reused at upper bounds.",
                                  crossing_angle="Angle between those actual finite width and lean vectors, compared with the same two vectors in the drawing.",
                                  turn_error="Absolute difference between original and drawn angles of consecutive increasing segments along the same path; median and 95th percentile reported separately for width and lean.",
                                  path_length="Sum of consecutive mesh-edge image distances along a fixed-width or fixed-lean curve; sampled polygonal length.",
                                  regularization="Image-pair stress plus relative local-distance stress, cosine-angle error, and an original-triangle-area penalty. It does not certify optimality, prevent every inversion, or eliminate global self-overlap."),
                 understanding_check=dict(object="The connected two families of actual image curves; not pixel-coordinate coloring",
                                          axes="Blue fixed-lean width paths and orange fixed-width lean paths; display positions fit the 784D metric",
                                          baseline="Root point and its two complete paths highlighted",
                                          evidence="Actual versus drawn crossing vectors, sampled path lengths, all-pair error, and orientation reversals",
                                          scope="Finite fixed-center fixed-height sheet; measured approximation, no exact global flattening claim"))
    (OUT/"results.json").write_text(json.dumps(summary,indent=2)+"\n")
    np.savez_compressed(OUT/"traced_sheet_data.npz",images=source.astype(np.float32),widths=widths,leans=leans,faces=faces,**layouts)
    figure(layouts[default],per_layout[default],actual,widths,leans,root,faces,metrics[default])
    payload=dict(widths=widths.tolist(),leans=leans.tolist(),root=root,faces=faces.tolist(),
                 layouts={k:v.tolist() for k,v in layouts.items()},actual={k:v.tolist() for k,v in actual.items()},
                 geometry={name:{k:v.tolist() for k,v in values.items()} for name,values in per_layout.items()},
                 summary=summary,images=base64.b64encode(gzip.compress(source.astype("<f4").tobytes())).decode())
    html=(ROOT/"experiments/draw_traced_width_lean_sheet.html").read_text().replace("__TRACED_SHEET__",json.dumps(payload,separators=(",",":")))
    (OUT/"index.html").write_text(html)
    # Replace the misleading primary inspector view, preserving its old page.
    old=OUT.parent/"original_coordinates/index.html"
    backup=old.with_name("pixel_inspector.html")
    if old.exists() and not backup.exists():backup.write_text(old.read_text())
    if old.exists():old.write_text(html)
    (OUT/"README.md").write_text("""# The sheet traced by the two curve families

Run `OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/draw_traced_width_lean_sheet.py` after the preceding sheet-view experiment.

The main object is the connected grid of actual rendered images. Blue curves vary width at fixed lean; orange curves vary lean at fixed width.
The 1,334 images span 29 widths (1.8-4.6 px) by 46 leans (-10 to 35 degrees), with center (14,14) and height 19.75 px fixed.
Every source image is checked against `grey_ones.render`. No PCA or pixel-based coloring is used for the main drawing.

The global-distance view reuses the preceding metric MDS fit to all raw image-pair distances.
Two new fits use those raw pair distances, relative short-path distances (immediate edges, cell diagonals, and two-tick path chords), actual width/lean crossing cosines, and triangle areas measured from 784D edge Gram matrices.
The objective has an analytic gradient checked against 12 central finite differences before fitting.
Each new fit has a 1,000-iteration limit; `results.json` records convergence and the objective weights. Finite optimization and initialization do not guarantee a global optimum.
An area regularizer discourages collapsed and inverted cells; residual orientation reversals are still counted and drawn red. The count does not detect all global self-overlap.

All source and drawn local crossing angles use the same available +0.1 px width and +1 degree lean intervals.
At upper boundaries the last increasing interval is translated to the same arrow origin for comparison. These are finite differences, not infinitesimal derivatives.
The two actual 784D vectors can be represented exactly as a pair of 2D arrows using their norms and angle. This local two-vector drawing does not claim the surrounding sheet is planar.
Sampled curve length sums the original consecutive image distances and compares them with the same curve's drawn segment lengths.
Curve bending is checked separately: median and 95th-percentile errors compare the angles between consecutive segments of each actual and drawn path. Small crossing-angle errors do not imply that the curve bends are preserved.
Map units use the original Euclidean coverage-distance scale. Source knob values label the paths, while fitted screen positions are approximate.

The old primary pixel inspector is preserved as `original_coordinates/pixel_inspector.html`. Its previous main URL now shows this geometric mesh.
The old inspector generator can overwrite that URL; rerun this generator to restore the main mesh if needed.
`traced_sheet_data.npz` preserves the original sampled images and all new display coordinates. No continuous decoder or exact lossless flattening is claimed.
""")
    print(json.dumps(dict(layouts=metrics,optimization=optimization),indent=2),flush=True)


if __name__=="__main__":
    main()
