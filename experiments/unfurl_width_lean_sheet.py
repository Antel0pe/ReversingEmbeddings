"""Compare extrinsic projections and intrinsic unfoldings of one width/lean sheet.

Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/unfurl_width_lean_sheet.py
"""
import base64
import csv
import gzip
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/unfurl-width-lean-mpl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
from scipy.linalg import eigh, svd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial.distance import cdist, pdist, squareform

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.compare_width_lean_distances import f
from experiments.width_vectors_by_lean import norm

OUT = ROOT / "figures/generated_one_width_vectors_by_lean/sheet_views"
OFFSETS = [(0,1),(1,0),(1,1),(1,-1),(1,2),(1,-2),(2,1),(2,-1)]


def log(message):
    print(message, flush=True)


def graph(widths, leans, images):
    """Only parameter-local edges; half-edge rendering measures bent short paths."""
    nt, nw = len(leans), len(widths)
    ids = np.arange(nt*nw).reshape(nt,nw)
    rows=[]; cols=[]; weights=[]
    for dt, dw in OFFSETS:
        ti, wi = np.meshgrid(np.arange(nt),np.arange(nw),indexing="ij")
        valid=(ti+dt<nt)&(wi+dw>=0)&(wi+dw<nw)
        ta,wa=ti[valid],wi[valid]
        tb,wb=ta+dt,wa+dw
        a,b=ids[ta,wa],ids[tb,wb]
        mid=f((widths[wa]+widths[wb])/2,(leans[ta]+leans[tb])/2)
        length=norm(mid-images[ta,wa])+norm(images[tb,wb]-mid)
        rows.extend([a,b]); cols.extend([b,a]); weights.extend([length,length])
    return coo_matrix((np.concatenate(weights),(np.concatenate(rows),np.concatenate(cols))),shape=(nt*nw,nt*nw)).tocsr()


def classical_mds(distances):
    square=distances**2
    gram=-.5*(square-square.mean(0)[None]-square.mean(1)[:,None]+square.mean())
    values,vectors=eigh(gram,subset_by_index=[len(gram)-3,len(gram)-1])
    return vectors[:,-2:][:,::-1]*np.sqrt(np.maximum(values[-2:][::-1],0))


def stress(distances, coordinates):
    delta=pdist(coordinates)
    target=squareform(distances,checks=False)
    return float(np.sqrt(np.sum((delta-target)**2)/np.sum(target**2)))


def smacof(distances, initial, iterations=700, tolerance=1e-8):
    """Metric MDS, equal weights on every distinct pair; monotone majorization."""
    y=initial.copy()-initial.mean(0)
    old=np.inf
    target_sq=float(np.sum(distances**2))
    history=[]
    for iteration in range(iterations):
        current=squareform(pdist(y))
        ratio=np.divide(distances,current,out=np.zeros_like(current),where=current>1e-12)
        np.fill_diagonal(ratio,0)
        new=(ratio.sum(axis=1)[:,None]*y-ratio@y)/len(y)
        value=float(np.sum((squareform(pdist(new))-distances)**2)/target_sq)
        history.append(value)
        assert value <= old+1e-10
        improvement=(old-value)/max(old,1e-12) if np.isfinite(old) else np.inf
        y=new; old=value
        if improvement<tolerance:
            break
    return y,dict(iterations=iteration+1,relative_squared_stress=value,last_relative_improvement=float(improvement),monotone=True)


def fit_mds(distances, initial, seed):
    # Two starts expose optimization dependence; the result is not called a
    # globally optimal nonlinear map.
    rng=np.random.default_rng(seed)
    candidates=[]
    for init in [initial, initial+rng.normal(0,.15,initial.shape)]:
        coords,details=smacof(distances,init)
        candidates.append((stress(distances,coords),coords,details))
    best=min(candidates,key=lambda z:z[0])
    return best[1],dict(best[2],initialization_stresses=[z[0] for z in candidates])


def orientation_and_ratios(y, shape, images):
    nt,nw=shape
    z=y.reshape(nt,nw,-1)
    lengths=np.concatenate([norm(np.diff(z,axis=0)).ravel(),norm(np.diff(z,axis=1)).ravel()])
    original=np.concatenate([norm(np.diff(images,axis=0)).ravel(),norm(np.diff(images,axis=1)).ravel()])
    ratio=lengths/original
    out=dict(local_edge_distance_ratio_quantiles=np.quantile(ratio,[0,.05,.5,.95,1]).tolist())
    if y.shape[1]==2:
        cross=lambda a,b:a[...,0]*b[...,1]-a[...,1]*b[...,0]
        a,b,c,d=z[:-1,:-1],z[:-1,1:],z[1:,:-1],z[1:,1:]
        signed=np.r_[cross(b-a,d-a).ravel(),cross(d-a,c-a).ravel()]
        positive=int(np.sum(signed>1e-10)); negative=int(np.sum(signed < -1e-10))
        out.update(triangle_orientation_reversal_fraction=min(positive,negative)/len(signed),
                   nearly_zero_area_triangle_count=int(np.sum(abs(signed)<=1e-10)),
                   orientation_note="Opposite orientation triangles reveal local foldovers in this drawing; this does not count all global overlaps.")
    return out


def diagnosis():
    # Equal 5-degree intervals are used explicitly here, independently of line
    # selection or palette choices in the earlier figures.
    t=np.linspace(0,35,701)
    images=f(3.2,t)
    root=f(3.2,0)
    radial=norm(images-root)
    cumulative=np.r_[0,np.cumsum(norm(np.diff(images,axis=0)))]
    dense=f(3.2,np.linspace(0,35,1401))
    arc_fine=float(norm(np.diff(dense,axis=0)).sum())
    starts=np.arange(35.)
    tick=norm(f(3.2,starts+1)-f(3.2,starts))
    d_integer=norm(f(3.2,np.arange(36.))-root)
    radial_tick=np.diff(d_integer)
    # Direct endpoint-distance identity measures changing overlap of the actual
    # grey coverage arrays, without replacing it by a binary-shape metric.
    squared_norm=np.sum(images**2,axis=1)
    overlap=images@root
    residual=radial**2-(squared_norm+root@root-2*overlap)
    assert abs(residual).max()<1e-12
    # A finer integration control separates the overall effect from the
    # renderer's finite vertical quadrature. Restore the source configuration.
    import grey_ones
    original_sub=grey_ones.SUB
    try:
        grey_ones.SUB=128
        high=f(3.2,np.arange(36.))
    finally:
        grey_ones.SUB=original_sub
    high_radial=norm(high-high[0])
    high_tick=norm(np.diff(high,axis=0))
    result=dict(width_px=3.2, lean_range=[0,35],
                five_degree_radial_distance_increments=np.diff(d_integer[::5]).tolist(),
                actual_1deg_step_at_0_and_34=tick[[0,-1]].tolist(),
                chord_at_35=float(radial[-1]),arc_005deg=float(cumulative[-1]),arc_0025deg=arc_fine,
                arc_relative_refinement_change=float(abs(arc_fine-cumulative[-1])/arc_fine),
                distance_squared_identity_max_error=float(abs(residual).max()),
                code_uses_exponential=False,
                lean_code="center = cx - tan(radians(lean)) * (subrow_midpoint - cy)",
                coverage_code="clip(min(right_edge, pixel_right) - max(left_edge, pixel_left), 0, 1)",
                subrow_integration_control=dict(original_subrows=original_sub,control_subrows=128,
                                                max_radial_distance_change=float(abs(high_radial-d_integer).max()),
                                                max_1deg_step_distance_change=float(abs(high_tick-tick).max()),
                                                control_1deg_step_at_0_and_34=high_tick[[0,-1]].tolist()),
                reason="Distance to a fixed upright reference levels off while the curve keeps moving. Equal degrees also give unequal tangent/shear increments; pixel boundary crossings change slopes.")
    fig=plt.figure(figsize=(14,7.5),facecolor="white")
    fig.text(.055,.95,"Why do the plotted distances bunch together at larger lean?",fontsize=19,weight="bold")
    fig.text(.055,.887,"Same generated 1: width 3.2 px, center (14,14), height 19.75 px. Lean increases in equal degree steps.\n"
             "Distances use all 784 coverage values. The upright image is the fixed reference; there is no exponential spacing rule.",fontsize=11,linespacing=1.5)
    ax=fig.add_axes([.075,.31,.355,.43])
    ax.plot(t,radial,color="#236f9c",lw=2,label="Straight image distance from upright")
    ax.plot(t,cumulative,color="#b26025",lw=2,label="Distance travelled along the lean curve")
    ax.set(xlabel="Lean (degrees)",ylabel="Distance (coverage L2)",title="The curve keeps travelling as its endpoint gap levels off",xlim=(0,35),ylim=(0,None))
    ax.grid(alpha=.16)
    ax=fig.add_axes([.57,.31,.355,.43])
    ax.plot(starts+.5,tick,color="#b26025",lw=2,label="Actual distance between consecutive 1° images")
    ax.plot(starts+.5,radial_tick,color="#236f9c",lw=2,label="Increase in distance-from-upright reading")
    ax.set(xlabel="Midpoint of the 1° lean interval (degrees)",ylabel="Distance over one degree (coverage L2)",title="Closer line heights do not mean closer consecutive images",xlim=(0,35),ylim=(0,None))
    ax.grid(alpha=.16)
    fig.legend([plt.Line2D([],[],color="#b26025",lw=2),plt.Line2D([],[],color="#236f9c",lw=2)],
               ["Brown: motion along the curve (accumulated on left; actual 1° step on right)",
                "Blue: distance-from-upright reading (total on left; its increase on right)"],
               loc="upper center",bbox_to_anchor=(.5,.845),frameon=False,fontsize=10)
    fig.text(.075,.17,f"At 35°: straight distance from upright = {radial[-1]:.3f}; sampled distance travelled = {cumulative[-1]:.3f}.\n"
             f"Actual 1° step: {tick[0]:.3f} near upright and {tick[-1]:.3f} at 34° → 35°. Its motion does not exponentially shrink.\n"
             "The distance-from-upright reading squares pixel differences, sums them and takes a square root; overlap with the upright stroke matters.",fontsize=10.5,linespacing=1.6)
    fig.text(.075,.055,"Lean uses tan(angle); coverage is clipped at pixel edges. Coverage is continuous but changes formula at edge crossings.\n"
             "The earlier legend also selected lean values: its original lean-path plot was a subset, not every consecutive angular increment.",fontsize=10,linespacing=1.5)
    fig.savefig(OUT/"07_why_curves_bunch.png",dpi=150)
    fig.savefig(OUT/"07_why_curves_bunch.svg")
    plt.close(fig)
    return result


def comparison_figure(layouts, report, widths, leans, root_index):
    nt,nw=len(leans),len(widths)
    fig=plt.figure(figsize=(15,11.8),facecolor="white")
    fig.text(.045,.963,"The same connected width/lean sheet, drawn four ways",fontsize=21,weight="bold")
    fig.text(.045,.918,"Each vertex is one generated image. Crossed lines connect fixed-width and fixed-lean curves; color = lean (degrees).\n"
             "Center (14,14), height 19.75 px fixed. Shapes are measured-coordinate drawings, not the ordered width/lean rectangle.",fontsize=11,linespacing=1.5)
    positions=[[.075,.59,.365,.26],[.565,.59,.365,.26],[.075,.19,.365,.26],[.565,.19,.365,.26]]
    titles={"pca2":"2D linear view · maximizes retained variance", "mds2":"2D distance fit · targets image-to-image gaps",
            "unfold2":"2D unfolding · targets distances along the sheet", "pca3":"3D linear view · a bent 2D sheet in 3D"}
    for p,key in zip(positions,layouts):
        y=layouts[key];d=y.shape[1];z=y.reshape(nt,nw,d)
        ax=fig.add_axes(p,projection="3d" if d==3 else None)
        segments=[];colors=[]
        for i in range(nt):
            segments.append(z[i]);colors.append(plt.cm.viridis((leans[i]+10)/45))
        for j in range(nw):
            for i in range(nt-1):
                segments.append(z[i:i+2,j]);colors.append(plt.cm.viridis(((leans[i]+leans[i+1])/2+10)/45))
        collection=(Line3DCollection if d==3 else LineCollection)(segments,colors=colors,linewidths=.65,alpha=.85)
        if d==3:
            ax.add_collection3d(collection)
            for k,method in enumerate([ax.set_xlim,ax.set_ylim,ax.set_zlim]):
                lo,hi=y[:,k].min(),y[:,k].max();method(lo-.05*(hi-lo),hi+.05*(hi-lo))
            ax.set_box_aspect(np.ptp(y,axis=0));ax.view_init(elev=25,azim=-55)
            ax.scatter(*y[root_index],color="black",s=28)
            ax.set(xlabel="PC1",ylabel="PC2",zlabel="PC3")
            ax.tick_params(labelsize=8,pad=1)
        else:
            ax.add_collection(collection);ax.autoscale();ax.margins(.08);ax.set_aspect("equal",adjustable="box")
            ax.scatter(*y[root_index],color="black",s=28,zorder=5)
            ax.set(xlabel="Measured coordinate 1 (coverage L2 scale)",ylabel="Measured coordinate 2")
            ax.grid(alpha=.12)
        ax.set_title(titles[key],fontsize=12,pad=12)
        m=report[key]
        variance=(f"Retained variance {m['retained_variance']:.1%}; " if "retained_variance" in m else "")
        text=variance+f"image-gap error {m['chord_stress']:.1%}; sheet-distance error {m['sheet_stress']:.1%}"
        fig.text(p[0],p[1]-.060,text,fontsize=10)
        if d==2:
            fig.text(p[0],p[1]-.080,f"Opposite-orientation triangles: {m['triangle_orientation_reversal_fraction']:.1%} (local foldovers)",fontsize=9.5,color="#666")
    fig.text(.045,.035,"Black dot = root (width 3.2, lean 0°). Drawings use equal coordinate scales; overlapping parts can hide grid lines.\n"
             "Errors are relative RMS distance errors across all sampled pairs. Sheet distances are local-graph approximations, checked at finer resolution.\n"
             "Variance means centered pixel variance under uniform width/degree sampling. None of these drawings is a lossless record of the whole continuous sheet.",fontsize=10,linespacing=1.5)
    cax=fig.add_axes([.947,.59,.012,.26]);fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-10,35),cmap="viridis"),cax=cax,label="Lean (degrees)")
    fig.savefig(OUT/"08_same_sheet_four_views.png",dpi=150)
    fig.savefig(OUT/"08_same_sheet_four_views.svg")
    plt.close(fig)


def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({"font.size":10.5,"axes.spines.top":False,"axes.spines.right":False})
    diag=diagnosis();log("Curve-spacing diagnosis complete")
    full_w=np.linspace(1.8,4.6,57);full_t=np.linspace(-10,35,91)
    full=f(full_w[None],full_t[:,None])
    flat=full.reshape(-1,784)
    active=np.flatnonzero(np.ptp(flat,axis=0)>0)
    mean=flat[:,active].mean(0)
    centered=flat[:,active]-mean
    _,singular,basis=svd(centered,full_matrices=False)
    all_scores=centered@basis[:3].T
    tw=np.arange(0,57,2);tt=np.arange(0,91,2)
    anchor_ids=(tt[:,None]*57+tw[None]).ravel()
    widths,leans=full_w[tw],full_t[tt]
    shape=(len(leans),len(widths))
    images=full[tt[:,None],tw[None]]
    sampled=images.reshape(-1,784)
    scores=all_scores[anchor_ids]
    root_index=int(np.argmin(abs(leans))) * len(widths)+int(np.argmin(abs(widths-3.2)))
    chord=squareform(pdist(sampled))
    log(f"PCA complete: {len(sampled)} mesh vertices; building local path distances")
    g=graph(full_w,full_t,full)
    distances=dijkstra(g,directed=False,indices=anchor_ids)
    sheet=distances[:,anchor_ids]
    assert np.isfinite(sheet).all() and np.max(abs(sheet-sheet.T))<1e-10
    assert np.min(sheet-chord)>-1e-10
    log("Fitting the image-distance map")
    mds,mds_details=fit_mds(chord,scores[:,:2],20261020)
    log("Fitting the sheet-distance unfolding")
    initial=classical_mds(sheet)
    unfolded,unfold_details=fit_mds(sheet,initial,20261021)
    layouts=dict(pca2=scores[:,:2],mds2=mds,unfold2=unfolded,pca3=scores)
    layouts={key:values-values[root_index] for key,values in layouts.items()}
    log("Checking path-distance stability at half the parameter spacing")
    fine_w=np.linspace(1.8,4.6,113);fine_t=np.linspace(-10,35,181)
    fine=f(fine_w[None],fine_t[:,None])
    assert np.array_equal(fine[::2,::2],full)
    fine_g=graph(fine_w,fine_t,fine)
    fine_anchor_ids=(tt[:,None]*2*113+tw[None]*2).ravel()
    landmark=np.unique(np.r_[root_index,np.linspace(0,len(sampled)-1,48).astype(int)])
    fine_landmark=dijkstra(fine_g,directed=False,indices=fine_anchor_ids[landmark])[:,fine_anchor_ids]
    original_landmark=sheet[landmark]
    mask=original_landmark>1e-8
    path_relative=(fine_landmark[mask]-original_landmark[mask])/fine_landmark[mask]
    path_check=dict(landmarks=len(landmark),targets=len(sampled),
                    relative_change_quantiles=np.quantile(abs(path_relative),[.5,.95,1]).tolist(),
                    rms_relative_distance_change=float(norm((fine_landmark-original_landmark).ravel())/norm(fine_landmark.ravel())),
                    note="Both graphs retain a finite set of local travel directions. Refining step size tests discretization but does not remove directional graph bias or certify exact geodesics.")
    total_variance=float(np.sum(singular**2))
    report={}
    rng=np.random.default_rng(20261020)
    hold_w=rng.uniform(1.8,4.6,512);hold_t=rng.uniform(-10,35,512)
    hold=f(hold_w,hold_t)[:,active]-mean
    for key,values in layouts.items():
        measured=orientation_and_ratios(values,shape,images)
        measured.update(chord_stress=stress(chord,values),sheet_stress=stress(sheet,values))
        if key.startswith("pca"):
            k=values.shape[1]
            reconstruction=centered-(centered@basis[:k].T)@basis[:k]
            hold_error=hold-(hold@basis[:k].T)@basis[:k]
            measured.update(retained_variance=float(np.sum(singular[:k]**2)/total_variance),
                            grid_reconstruction_RMS_L2=float(np.sqrt(np.mean(norm(reconstruction)**2))),
                            grid_reconstruction_max_L2=float(norm(reconstruction).max()),
                            held_out_count=512,held_out_reconstruction_RMS_L2=float(np.sqrt(np.mean(norm(hold_error)**2))),
                            held_out_reconstruction_max_L2=float(norm(hold_error).max()))
        report[key]=measured
    summary=dict(fixed=dict(cx=14,cy=14,height=19.75),root=[14,14,19.75,3.2,0],
                 mesh=dict(width_count=len(widths),lean_count=len(leans),vertices=len(sampled),
                           width_range=[1.8,4.6],lean_range=[-10,35],
                           coordinate_sampling="uniform width and lean angle; not uniform surface area"),
                 pca_fit=dict(images=len(flat),varying_pixel_coordinates=len(active),centered=True,
                              variance_cumulative=[float(np.sum(singular[:k]**2)/total_variance) for k in [1,2,3,5,10]]),
                 diagnosis=diag,layouts=report,optimization=dict(mds2=mds_details,unfold2=unfold_details),
                 path_distance_check=path_check,
                 caveats=["Known controls supply grid connectivity. These are sampled drawings, not a recovered unknown manifold.",
                          "Linear PCA optimizes reconstruction variance. MDS fits distances and is not a globally optimal guarantee.",
                          "Geodesic unfolding uses local parameter paths, including edge midpoints; it approximates along-sheet distances.",
                          "2D foldovers can create false visual neighbors. A point's image can be retrieved from stored samples, but that is not a learned decoder.",
                          "No 2D view here is claimed to preserve all 784-dimensional geometry or reconstruct every continuous image exactly."],
                 understanding_check=dict(object="Connected image samples from one width/lean sheet, not knob coordinates used as plot positions",
                                          axes="PCA coefficients or fitted metric coordinates; colors are lean; visible edge connections retain the actual grid",
                                          reference="Root image marked and centered at zero in all layouts",
                                          evidence="Distance errors, local edge distortion, foldovers, and held-out PCA reconstruction quantify missing geometry",
                                          scope="Finite fixed-center fixed-height slice; geodesic approximation and sampling distribution explicitly stated"))
    (OUT/"results.json").write_text(json.dumps(summary,indent=2)+"\n")
    np.savez_compressed(OUT/"sheet_data.npz",images=images.reshape(*shape,28,28),widths=widths,leans=leans,root_index=root_index,
                        **layouts,chord_distances=chord,sheet_distances=sheet,active_pixels=active,pca_mean=mean,pca_basis=basis[:3])
    with (OUT/"view_coordinates.csv").open("w",newline="") as handle:
        writer=csv.writer(handle);writer.writerow(["width_px","lean_deg","view","coordinate_1","coordinate_2","coordinate_3"])
        for key,values in layouts.items():
            for i,(width,lean) in enumerate([(w,t) for t in leans for w in widths]):
                writer.writerow([width,lean,key,*values[i],*([""] if values.shape[1]==2 else [])])
    comparison_figure(layouts,report,widths,leans,root_index)
    payload=dict(widths=widths.tolist(),leans=leans.tolist(),rootIndex=root_index,
                 layouts={k:v.tolist() for k,v in layouts.items()},summary=summary,
                 images=base64.b64encode(gzip.compress(images.astype("<f4").tobytes())).decode(),
                 sheetDistances=base64.b64encode(gzip.compress(sheet.astype("<f4").tobytes())).decode())
    template=(ROOT/"experiments/unfurl_width_lean_sheet.html").read_text()
    (OUT/"index.html").write_text(template.replace("__SHEET_DATA__",json.dumps(payload,separators=(",",":"))))
    (OUT/"README.md").write_text("""# Four drawings of the width/lean sheet

Run `OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/unfurl_width_lean_sheet.py` from the repository root.
Center (14,14) and height 19.75 px are fixed. Width 1.8-4.6 px and lean -10 to 35 degrees span the sheet.

The ordered parameter rectangle supplies connectivity, but its knob values are not used as display positions in these maps.
PCA2 and PCA3 use actual centered pixel projection coefficients. PCA is fitted on 5,187 equally spaced width/angle samples.
The displayed mesh has 1,334 vertices (29 widths by 46 leans). Uniform controls do not imply uniform surface-area weighting.
Metric MDS2 minimizes the sum of squared differences between displayed pair distances and the original image distances.
The unfolding uses the same stress objective with graph path distances as its target.
Each MDS objective uses two initializations and up to 700 monotone-majorization iterations; it is a fitted result, not a certified global optimum.

Graph edges connect only local parameter offsets (16 directions) on a 57 by 91 grid.
An edge's weight includes a newly rendered midpoint, measuring two short image-space segments instead of one chord.
Shortest graph distances approximate travel constrained to the sheet. A 113 by 181 grid checks resolution at 49 or fewer landmarks against every displayed vertex.
This check leaves finite-direction graph bias unresolved; these distances are not certified exact continuous geodesics.

Reported stress is sqrt(sum((displayed_distance-target_distance)^2)/sum(target_distance^2)) across all distinct pairs, in the original distance units without rescaling.
The local edge ratios compare displayed mesh edges to original pixel chords, not to a nonlinear decoder metric.
Opposite-orientation triangles flag local planar foldovers; they do not enumerate all global self-overlap.
PCA reconstruction uses its linear decoder and 512 fresh joint held-out points. MDS maps have no decoder here.
Exact source images are retrieved from stored samples in the viewer; that lookup does not validate continuous reconstruction.

`07_why_curves_bunch.png` compares radial distance to a fixed upright image with distance travelled along the lean curve.
It also compares actual consecutive one-degree gaps with increments in the scalar distance-from-upright reading.
The renderer uses tan(lean) and clipped horizontal overlap, not an exponential spacing schedule. Coverage is continuous and piecewise affine in width and tan(lean) at fixed vertical extent.

See `results.json` for measurements, `sheet_data.npz` for arrays and distances, and `view_coordinates.csv` for all plotted coordinates.
""")
    log(json.dumps(dict(elapsed_seconds=time.monotonic()-started,layouts=report,path_check=path_check),indent=2))


if __name__=="__main__":
    main()
