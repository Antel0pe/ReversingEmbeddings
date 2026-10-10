"""Inspect full local bases, sampling bias, observable matching and scale laws.

Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python analyze_generated_one_basis_followup.py
Existing samples supply most analyses. New samples only test sampling-seed
stability and a held-out smaller radius. Knob labels are evaluation data only.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/generated-one-mpl")

import base64
import csv
import html
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from grey_ones import KNOBS
from analyze_root_one_local_svd import directions, pixels, shell
from analyze_generated_one_local_bases_across_starts import BASE, LOW, HIGH, sign_basis

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "figures/generated_one_local_bases_across_starts"
OUT = HERE / "figures/generated_one_basis_followup"
N_TRAIN = 512


def assignment(a):
    row, col = linear_sum_assignment(-np.abs(a))
    return col[np.argsort(row)]


def energy_fractions(singular):
    e = np.asarray(singular) ** 2
    return e / e.sum()


def full_match(reference, candidate):
    """Match leading five and smaller directions separately, preserving their roles."""
    first = assignment(reference[:5] @ candidate[:5].T)
    rest = assignment(reference[5:9] @ candidate[5:].T) + 5
    used = [*first, *rest]
    extra = [i for i in range(len(candidate)) if i not in used]
    ordered = np.array([*used, *extra])
    aligned = candidate[ordered].copy()
    products = np.sum(reference[:9] * aligned[:9], axis=1)
    signs = np.sign(products)
    signs[signs == 0] = 1
    aligned[:9] *= signs[:, None]
    return ordered, aligned, np.abs(products)


def image_data_url(values, signed=False, limit=1):
    stream = io.BytesIO()
    plt.imsave(stream, np.asarray(values).reshape(28,28), format="png",
               cmap="RdBu_r" if signed else "gray_r",
               vmin=-limit if signed else 0, vmax=limit if signed else 1)
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def full_basis_views(rows, archive):
    reference = archive["root_r_0p001_basis"]
    full = {}
    for r in rows:
        name = r["name"]
        b = archive[name+"_r_0p001_basis"]
        order, aligned, similarity = full_match(reference,b)
        fraction = energy_fractions(r["singular_values"])
        full[name] = dict(order=order, aligned=aligned, similarity=similarity,
                          fraction=fraction, row=r)
    names = ["root","right_0p3","width_2p2","width_4p2","lean_minus7","lean_12p5","lean_30"]
    labels = ["Root", "Right +0.3 px", "Width 2.2 px", "Width 4.2 px", "Lean −7°", "Lean 12.5°", "Lean 30°"]
    limit = max(np.abs(v["aligned"]).max() for v in full.values())
    fig = plt.figure(figsize=(21,13.3),facecolor="white")
    fig.text(.025,.98,"All nine local vectors, including the small residual directions",fontsize=22,weight="bold",va="top")
    fig.text(.025,.942,
             "Generated 1s; root settings (cx, cy, height, width, lean) = (14, 14, 19.75, 3.2, 0°). 512 fitting + 256 independent neighbors per start; radius 0.001.\n"
             "Start images: white = no ink, black = full ink. Signed vectors: red gains ink; blue loses ink. Unit-length vectors share a color scale, including tiny energy contributions.\n"
             "Columns 1–5 match only among the leading five; 6–9 match separately among the residual vectors. Matching uses pixel cosine, sign and permutation, not knob labels.",
             fontsize=11,va="top",linespacing=1.45)
    grid = fig.add_gridspec(7,10,left=.13,right=.985,bottom=.115,top=.84,hspace=.52,wspace=.24)
    for i,(name,label) in enumerate(zip(names,labels)):
        entry=full[name];r=entry["row"]
        for j in range(10):
            ax=fig.add_subplot(grid[i,j])
            if j==0:
                ax.imshow(archive[name+"_start_image"].reshape(28,28),cmap="gray_r",vmin=0,vmax=1,interpolation="nearest")
                miss=100*np.sum(entry["fraction"][5:])
                text=f"5-vector residual energy\n{miss:.2e}%"
            else:
                shown=ax.imshow(entry["aligned"][j-1].reshape(28,28),cmap="RdBu_r",vmin=-limit,vmax=limit,interpolation="nearest")
                original=entry["order"][j-1]
                text=f"SVD {original+1} · E {100*entry['fraction'][original]:.2g}%\nroot cosine {entry['similarity'][j-1]:.3f}"
            if i==0:
                ax.set_title("Starting 1" if j==0 else f"Root vector {j}",fontsize=10,pad=10)
            ax.set_xticks([]);ax.set_yticks([])
            ax.text(.5,-.10,text,transform=ax.transAxes,ha="center",va="top",fontsize=8.2)
        y=.115+.725*(1-(i+.5)/7)
        fig.text(.018,y,label+f"\n5-vector worst error {r['five_vectors_test_worst_percent']:.3g}%",
                 fontsize=10,va="center",linespacing=1.5)
    cax=fig.add_axes([.73,.057,.25,.015])
    fig.colorbar(shown,cax=cax,orientation="horizontal").set_label("Pixel change per unit coefficient (shared scale)",fontsize=9)
    fig.text(.025,.07,
             "E = contribution to total squared fitting displacement; these fits subtract each start, not the sample mean.\n"
             "All nine-vector fits shown reconstruct held-out changes to < 1.5 × 10⁻¹⁴ absolute image distance.\n"
             "The all-start HTML includes the tenth vector where needed. Rounded cosines of 1.000 do not imply exact equality.",
             fontsize=10,va="top",linespacing=1.5)
    fig.savefig(OUT/"matched_nine_vectors.png",dpi=150)
    plt.close(fig)

    cards=[]
    for name,entry in full.items():
        r=entry["row"]
        cells=[f'<div class="cell"><b>Starting 1</b><img src="{image_data_url(archive[name+"_start_image"])}"><span>Rank {len(entry["order"])}</span></div>']
        for j,original in enumerate(entry["order"]):
            match=f"Root cosine {entry['similarity'][j]:.6f}" if j<9 else "Extra vector; no root counterpart"
            cells.append(f'<div class="cell"><b>{"Root slot "+str(j+1) if j<9 else "Extra 10"}</b>'
                         f'<img src="{image_data_url(entry["aligned"][j],True,limit)}">'
                         f'<span>SVD {original+1}<br>Energy {100*entry["fraction"][original]:.6g}%<br>{match}</span></div>')
        captured=100*(1-r["test_relative_rms_error_by_k"][5]**2)
        caption=(f'{html.escape(name)} · settings {html.escape(str(r["settings"]))}<br>'
                 f'First five held-out energy captured: {captured:.12f}% · worst relative error: {r["five_vectors_test_worst_percent"]:.6g}% · '
                 f'full basis maximum absolute error: {r["full_basis_test_max_absolute_L2_error"]:.3g}')
        cards.append(f'<section><h2>{caption}</h2><div class="strip">{"".join(cells)}</div></section>')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Full local bases for generated 1s</title><style>body{font:15px system-ui;margin:24px;color:#172a3a;background:#fafafa}h1{font-size:27px}h2{font-size:14px;font-weight:500;line-height:1.6}.strip{display:flex;gap:14px;overflow-x:auto;padding-bottom:16px}.cell{min-width:142px;max-width:142px;text-align:center;font-size:12px;line-height:1.5}.cell img{display:block;width:140px;height:140px;image-rendering:pixelated;border:1px solid #aaa;margin:8px 0}.cell:nth-child(7){border-left:3px solid #ddd;padding-left:14px}section{background:white;padding:16px;margin:20px 0;border:1px solid #ddd;border-radius:8px}.intro{max-width:1150px;line-height:1.65}button{padding:8px 12px;cursor:pointer}.only-five .cell:nth-child(n+7){display:none}</style>
<h1>All local basis vectors at all 17 starting 1s</h1><p class="intro">Each neighborhood has 512 fitting and 256 held-out neighbors at image distance 0.001. Images are 28×28 ink coverage: white = 0; black = 1. Signed unit vectors use red for ink gain and blue for ink loss, with a shared color range ±LIMIT. Each vector is enlarged to unit length; energy labels distinguish important and tiny directions. First five and residual vectors are matched separately by pixel cosine, allowing signs and permutations. These columns are comparisons, not claims of identical knob meanings. Ninth/tenth vectors are retained when numerically active.</p>
<button onclick="document.body.classList.toggle('only-five')">Toggle first five / all vectors</button>CARDS</html>'''
    (OUT/"all_starts_full_basis.html").write_text(page.replace("LIMIT",f"{limit:.5f}").replace("CARDS","".join(cards)))
    return full


def image_features_and_gradients(image):
    yy,xx=np.mgrid[:28,:28];x=xx.ravel()+.5;y=yy.ravel()+.5
    ink=np.asarray(image).ravel();mass=ink.sum();cx=ink@x/mass;cy=ink@y/mass
    dx=x-cx;dy=y-cy;vx=ink@(dx*dx)/mass;vy=ink@(dy*dy)/mass;cov=ink@(dx*dy)/mass
    f=np.array([mass,cx,cy,vx,vy,cov])
    g=np.array([np.ones(784),dx/mass,dy/mass,(dx*dx-vx)/mass,
                (dy*dy-vy)/mass,(dx*dy-cov)/mass])
    return f,g


def observable_matching(rows,archive):
    features=[];gradients=[]
    for r in rows:
        f,g=image_features_and_gradients(archive[r["name"]+"_start_image"])
        features.append(f);gradients.append(g)
    scales=np.std(features,axis=0)
    assert np.all(scales>0)
    root_basis=archive["root_r_0p001_basis"][:5]
    root_profile=root_basis@gradients[0].T/scales
    root_profile/=np.linalg.norm(root_profile,axis=1)[:,None]
    root_knob=assignment(root_basis@archive["root_r_0p001_unit_knob_derivatives"].T)
    rng=np.random.default_rng(20261013)
    result=[]
    for r,g in zip(rows,gradients):
        name=r["name"];b=archive[name+"_r_0p001_basis"][:5]
        # Scramble every candidate frame so matching cannot exploit column order/sign.
        permutation=rng.permutation(5);sign=rng.choice([-1,1],5)
        candidate=b[permutation]*sign[:,None]
        profile=candidate@g.T/scales
        profile/=np.linalg.norm(profile,axis=1)[:,None]
        pixel_order=assignment(root_basis@candidate.T)
        feature_order=assignment(root_profile@profile.T)
        # Labels enter only here, to evaluate the unsupervised image-based assignments.
        knob=assignment(candidate@archive[name+"_r_0p001_unit_knob_derivatives"].T)
        truth=np.array([np.flatnonzero(knob==k)[0] for k in root_knob])
        cos_to_knob=np.abs(candidate@archive[name+"_r_0p001_unit_knob_derivatives"].T)
        result.append(dict(name=name,candidate_permutation=permutation.tolist(),candidate_signs=sign.tolist(),
                           pixel_assignment=pixel_order.tolist(),observable_assignment=feature_order.tolist(),
                           evaluation_assignment=truth.tolist(),pixel_correct=int(np.sum(pixel_order==truth)),
                           observable_correct=int(np.sum(feature_order==truth)),
                           assigned_knob_cosines=cos_to_knob[np.arange(5),knob].tolist()))
    nonroot=[r for r in result if r["name"]!="root"]
    return dict(observables=["ink_mass","centroid_x","centroid_y","variance_x","variance_y","covariance_xy"],
                feature_scales=scales.tolist(),normalization="standard deviation over the 17 source images",
                total_nonroot_assignments=len(nonroot)*5,pixel_correct=sum(r["pixel_correct"] for r in nonroot),
                observable_correct=sum(r["observable_correct"] for r in nonroot),results=result)


def fit_basis(vectors,weights=None):
    matrix=vectors if weights is None else vectors*np.sqrt(weights[:,None])
    _,s,b=np.linalg.svd(matrix,full_matrices=False)
    rank=int(np.sum(s>s[0]*1e-10))
    return s,sign_basis(b[:rank])


def bias_experiment(archive):
    source=archive["root_start_image"];images=archive["root_r_0p001_images"]
    train=images[:512]-source;test=images[512:]-source
    original=archive["root_r_0p001_basis"]
    tangent=archive["root_r_0p001_unit_knob_derivatives"]
    rows=[];saved={};variants=[]
    def record(label,b,s,weights=None,**extra):
        a=np.abs(b[:5]@tangent.T);knob=assignment(a)
        error=np.linalg.norm(test-(test@b.T)@b,axis=1).max()
        assert error<1e-12
        row=dict(label=label,rank=len(b),known_knob_assignment=knob.tolist(),
                 assigned_knob_cosines=a[np.arange(5),knob].tolist(),
                 best_knob_cosine_by_vector=a.max(axis=1).tolist(),full_basis_test_max_absolute_L2_error=float(error),
                 singular_values=s.tolist(),**extra)
        if weights is not None:
            row["weight_ratio"]=float(weights.max()/weights.min())
            row["effective_sample_count"]=float(weights.sum()**2/(weights@weights))
        rows.append(row);saved[label+"_basis"]=b;saved[label+"_test_coefficients"]=test@b.T
        return row
    s,b=fit_basis(train);record("original",b,s);variants.append(("Original sampling",b,rows[-1]))
    for i,seed in enumerate([20261020,20261021,20261022]):
        p,x,_,_=shell(directions(512,seed),.001,source,center=BASE,low=LOW,high=HIGH)
        s,b=fit_basis(x-source);record("fresh_seed_"+str(seed),b,s,seed=seed)
        saved["fresh_seed_"+str(seed)+"_settings"]=p
        saved["fresh_seed_"+str(seed)+"_images"]=x
        if i==0:variants.append((f"Fresh sampling seed\n{seed}",b,rows[-1]))
    diagonal=(original[1]+original[2])/np.sqrt(2)
    score=(train@diagonal/np.linalg.norm(train,axis=1))**2
    for strength in [2.,6.]:
        weights=np.exp(strength*(score-score.max()))
        s,b=fit_basis(train,weights)
        label="weighted_"+str(int(strength));record(label,b,s,weights,strength=strength)
        saved[label+"_weights"]=weights
        variants.append((f"Same images, reweighted\nstrength {strength:g}",b,rows[-1]))
    rotation=np.eye(len(original));rotation[1,1]=rotation[1,2]=rotation[2,2]=1/np.sqrt(2);rotation[2,1]=-1/np.sqrt(2)
    rotated=rotation@original
    assert np.allclose(rotated@rotated.T,np.eye(len(rotated)),atol=2e-12)
    assert np.max(np.abs((test@rotated.T)@rotated-(test@original.T)@original))<1e-15
    saved["rotated_basis"]=rotated;saved["rotation_matrix"]=rotation
    saved["rotated_test_coefficients"]=test@rotated.T
    np.savez_compressed(OUT/"alternative_bases.npz",**saved)

    limit=max(np.abs(b[:5]).max() for _,b,_ in variants)
    fig=plt.figure(figsize=(14,10),facecolor="white")
    fig.text(.035,.97,"Does SVD have to return the physical knob directions?",fontsize=19,weight="bold",va="top")
    fig.text(.035,.91,
             "Root (14, 14, 19.75, 3.2, 0°); image distance 0.001. Each row uses SVD on pixel displacements only.\n"
             "First five unit vectors shown; all nine retained for precision. Knob names below are an evaluation against generator derivatives.\n"
             "Start: white = no ink, black = full ink. Vectors: red gains ink, blue loses ink; one shared color scale.",fontsize=10.5,va="top",linespacing=1.5)
    grid=fig.add_gridspec(4,6,left=.2,right=.97,bottom=.16,top=.78,hspace=.6,wspace=.25)
    for i,(label,b,row) in enumerate(variants):
        for j in range(6):
            ax=fig.add_subplot(grid[i,j])
            if j==0:
                ax.imshow(source.reshape(28,28),cmap="gray_r",vmin=0,vmax=1,interpolation="nearest")
                text=f"Full error {row['full_basis_test_max_absolute_L2_error']:.1e}"
            else:
                shown=ax.imshow(b[j-1].reshape(28,28),cmap="RdBu_r",vmin=-limit,vmax=limit,interpolation="nearest")
                a=np.abs(b[j-1]@tangent.T);idx=int(a.argmax())
                text=f"Closest: {KNOBS[idx]}\ncosine {a[idx]:.3f}"
            if i==0:ax.set_title("Starting 1" if j==0 else f"SVD vector {j}",fontsize=10,pad=10)
            ax.set_xticks([]);ax.set_yticks([])
            ax.text(.5,-.1,text,transform=ax.transAxes,ha="center",va="top",fontsize=9)
        fig.text(.025,.16+.62*(1-(i+.5)/4),label,fontsize=10.5,va="center",linespacing=1.5)
    cax=fig.add_axes([.64,.08,.30,.018])
    fig.colorbar(shown,cax=cax,orientation="horizontal").set_label("Pixel change per unit coefficient",fontsize=9)
    fig.text(.035,.09,
             "Weighting favors a diagonal between two learned pixel directions.\n"
             "Strength 2 changes relative weights by at most e² = 7.39.\n"
             "All full bases reconstruct the same 256 held-out changes to < 10⁻¹².",fontsize=10,va="top",linespacing=1.5)
    fig.savefig(OUT/"sampling_bias.png",dpi=150);plt.close(fig)
    return dict(results=rows,weight_definition="exp(strength * squared cosine to (original SVD2+SVD3)/sqrt(2)), normalized by maximum",
                rotated_basis="45-degree rotation of original SVD2/SVD3; entire subspace and reconstruction unchanged")


def tail_fractions(singular):
    e=np.asarray(singular)**2
    return np.array([e[k:].sum()/e.sum() for k in range(1,9)])


def scale_experiment():
    root_meta=json.loads((HERE/"figures/root_one_local_svd/results.json").read_text())
    root_rows=[r for r in root_meta["results"] if r["radius"]<=.03]
    boundary=json.loads((SOURCE/"subrow_boundary_diagnostic.json").read_text())
    cases={"upright_root":root_rows}
    for name in ["tilted_on_subrow_boundary","tilted_shifted_down_0p023"]:
        cases[name]=[r for r in boundary if r["name"]==name]
    results={}
    for name,rows in cases.items():
        rows=sorted(rows,key=lambda r:r["radius"])
        radii=np.array([r["radius"] for r in rows]);tails=np.array([tail_fractions(r["singular_values"]) for r in rows])
        fits=[]
        for j in range(8):
            slope,intercept=np.polyfit(np.log(radii),np.log(tails[:,j]),1)
            xr=radii**2;coef=np.linalg.lstsq(np.column_stack([np.ones(len(xr)),xr/xr.max()]),tails[:,j],rcond=None)[0]
            fits.append(dict(k=j+1,power_exponent=float(slope),power_coefficient=float(np.exp(intercept)),
                             free_intercept_a=float(coef[0]),quadratic_coefficient_b=float(coef[1]/xr.max())))
        candidate=next((f["k"] for f in fits if f["power_exponent"]>1.5),None)
        results[name]=dict(radii=radii.tolist(),tail_energy_fractions_by_k=tails.tolist(),fits=fits,
                           first_tail_with_power_exponent_above_1p5=candidate)
    # Predict one unseen smaller scale before measuring its spectrum.
    fit=results["upright_root"]["fits"][4]
    predicted_radius=.0003;predicted=fit["power_coefficient"]*predicted_radius**fit["power_exponent"]
    source=pixels(BASE)[0]
    p,x,_,_=shell(directions(512,20261010),predicted_radius,source,center=BASE,low=LOW,high=HIGH)
    singular,_=fit_basis(x-source);measured=tail_fractions(singular)[4]
    prediction=dict(radius=predicted_radius,k=5,predicted_tail_energy_fraction=float(predicted),
                    measured_tail_energy_fraction=float(measured),relative_prediction_error=float(abs(predicted-measured)/measured))
    np.savez_compressed(OUT/"smaller_radius_check.npz",settings=p,images=x,singular_values=singular)

    fig=plt.figure(figsize=(12,7.8),facecolor="white")
    fig.text(.055,.965,"Can scale dependence distinguish tangent variation from residual directions?",fontsize=17,weight="bold",va="top")
    fig.text(.055,.90,
             "Generated 1s; each mark is a 512-image equal-distance shell. Rₖ = squared fitting displacement left after k SVD vectors / total.\n"
             "Power exponents are fitted freely; separate fits Rₖ = a + b r² allow any intercept a. All k = 1…8 are checked.",fontsize=10.5,va="top",linespacing=1.5)
    ax=fig.add_axes([.11,.26,.63,.52])
    curves=[("upright_root",4,"Root: 4 vectors","#777777"),("upright_root",5,"Root: 5 vectors","#0077aa"),
            ("tilted_on_subrow_boundary",5,"Tilted boundary: 5","#cc5533"),
            ("tilted_on_subrow_boundary",7,"Tilted boundary: 7","#9b55aa"),
            ("tilted_shifted_down_0p023",5,"Tilted, shifted: 5","#228855")]
    for name,k,label,color in curves:
        case=results[name];r=np.array(case["radii"]);t=np.array(case["tail_energy_fractions_by_k"])[:,k-1];fit=case["fits"][k-1]
        ax.loglog(r,t,"o-",color=color,label=f"{label}\npower {fit['power_exponent']:.2f}",markersize=5)
        extrapolated=np.geomspace(r.min()/10,r.min(),50)
        ax.loglog(extrapolated,fit["power_coefficient"]*extrapolated**fit["power_exponent"],"--",color=color,lw=1)
    ax.scatter([predicted_radius],[measured],marker="*",s=160,color="#0077aa",edgecolor="black",zorder=5)
    ax.set_xlabel("Image-space radius r (log scale)",fontsize=11)
    ax.set_ylabel("Uncaptured squared-displacement fraction Rₖ (log scale)",fontsize=10.5)
    ax.legend(loc="upper left",bbox_to_anchor=(1.02,1),fontsize=9)
    ax.grid(alpha=.15,which="both")
    fig.text(.78,.425,"Starting images: white = 0 ink; black = 1",fontsize=8)
    for left,settings,label in [(.78,BASE,"Root"),(.855,[14,14,19.75,3.2,30],"Tilt 30°\ncy = 14"),
                                (.93,[14,14.023,19.75,3.2,30],"Tilt 30°\ncy = 14.023")]:
        image_ax=fig.add_axes([left,.29,.06,.11])
        image_ax.imshow(pixels(settings)[0].reshape(28,28),cmap="gray_r",vmin=0,vmax=1,interpolation="nearest")
        image_ax.set_xticks([]);image_ax.set_yticks([])
        image_ax.text(.5,-.1,label,transform=image_ax.transAxes,ha="center",va="top",fontsize=8)
    fig.text(.055,.16,
             "Circles / solid lines: measured. Dashed lines: extrapolation, not measured reconstruction. Star: held-out smaller-radius check.\n"
             "A power near 2 means relative residual energy falls as r²; a power near 0 means it plateaus. This diagnoses local linear structure, not knob names.",fontsize=10,va="top",linespacing=1.5)
    fig.text(.055,.075,
             f"Root five-vector prediction at r = 0.0003 differs from measurement by {100*prediction['relative_prediction_error']:.3g}%.\n"
             "Root (cx, cy, height, width, lean) = (14, 14, 19.75, 3.2, 0°); tilted cases change only the displayed settings. Float64 coverage.\n"
             "Boundary case has renderer derivative kinks. A vanishing fitted tail does not establish exact reconstruction at any finite radius.",fontsize=9.5,va="top",linespacing=1.5)
    fig.savefig(OUT/"radius_scale_laws.png",dpi=150);plt.close(fig)
    return dict(cases=results,smaller_radius_prediction_check=prediction,
                selection_rule="First k among 1..8 whose measured tail log-log power exceeds 1.5; exploratory, not a general calibrated estimator")


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    metadata=json.loads((SOURCE/"results.json").read_text())
    archive=np.load(SOURCE/"bases_and_samples.npz")
    rows=[r for r in metadata["results"] if r["radius"]==.001]
    full=full_basis_views(rows,archive)
    print("Full basis PNG and all-start gallery generated.",flush=True)
    matching=observable_matching(rows,archive)
    print("Observable matching:",matching["observable_correct"],"/",matching["total_nonroot_assignments"],flush=True)
    bias=bias_experiment(archive)
    print("Sampling and basis counterexamples complete.",flush=True)
    scale=scale_experiment()
    result=dict(method="Existing canonical-renderer data; no knob labels in SVD, weighting or observable matching",
                radius=.001,observable_matching=matching,sampling_bias=bias,scale_laws=scale,
                full_basis_summary=[dict(name=r["name"],rank=r["numerical_rank_relative_1e_10"],
                    first_five_fitting_energy_percent=float(100*energy_fractions(r["singular_values"])[:5].sum()),
                    first_five_heldout_energy_percent=float(100*(1-r["test_relative_rms_error_by_k"][5]**2)),
                    first_five_heldout_worst_error_percent=r["five_vectors_test_worst_percent"],
                    full_basis_max_absolute_error=r["full_basis_test_max_absolute_L2_error"],
                    displayed_original_indices=full[r["name"]]["order"].tolist()) for r in rows])
    (OUT/"results.json").write_text(json.dumps(result,indent=2)+"\n")
    with (OUT/"energy_and_errors.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(result["full_basis_summary"][0]));writer.writeheader();writer.writerows(result["full_basis_summary"])
    print(json.dumps(dict(scale_candidate_counts={n:c["first_tail_with_power_exponent_above_1p5"] for n,c in scale["cases"].items()},
                          smaller_radius_check=scale["smaller_radius_prediction_check"])),flush=True)


if __name__=="__main__":
    main()
