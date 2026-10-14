"""Known-target verification: can local SVD spans reconstruct true knob vectors?

Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/verify_true_knobs_in_svd.py
This measures representability, not blind identification of physical knobs.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/true-knobs-svd-mpl')
from pathlib import Path
import sys
import json
import csv
import time
import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grey_ones import render, SUB, KNOBS
from analyze_root_one_local_svd import directions, SCALE
from analyze_generated_one_manifold_transport import jet
from analyze_generated_one_local_bases_across_starts import starts

OUT = ROOT/'figures'/'true_knobs_in_svd'
RADII = [1e-3, 1e-4, 1e-5]
SEED = 20261012
N = 256
RANK_RTOL = 1e-9
BASE = np.array([14., 14., 19.75, 3.2, 0.])


def pixels(p):
    return render(p, dtype=np.float64).reshape(-1, 784)


def min_derivative(a, b, da, db):
    return np.where(a < b, da, np.where(a > b, db, np.minimum(da, db)))


def max_derivative(a, b, da, db):
    return np.where(a > b, da, np.where(a < b, db, np.maximum(da, db)))


def clip_derivative(value, derivative, upper):
    return np.where((value > 0) & (value < upper), derivative,
                    np.where(value == 0, np.maximum(derivative, 0),
                             np.where(value == upper, np.minimum(derivative, 0), 0)))


def directional_derivative(p, velocity):
    """Exact forward directional derivative of grey_ones' coverage formula.

    Differentiate min/max/clip with their directional tie rules, so tilted
    cap boundaries and integer edge boundaries receive genuine one-sided
    derivatives, not an assumed common two-sided tangent.
    """
    cx, cy, height, width, lean = p
    dx, dy, dh, dw, dl = velocity
    lo = np.arange(28*SUB)/SUB; hi = lo+1/SUB; mid = lo+.5/SUB
    cols = np.arange(28)
    top, bot = cy-height/2, cy+height/2
    dtop, dbot = dy-dh/2, dy+dh/2
    raw_v = np.minimum(hi, bot)-np.maximum(lo, top)
    d_v = min_derivative(hi, bot, 0., dbot)-max_derivative(lo, top, 0., dtop)
    d_v = clip_derivative(raw_v, d_v, 1/SUB)
    v = np.clip(raw_v, 0, 1/SUB)
    slope = np.tan(np.deg2rad(lean))
    slope_prime = np.pi/180*(1+slope*slope)
    center = cx-slope*(mid-cy)
    dcenter = dx+slope*dy-slope_prime*(mid-cy)*dl
    left = (center-width/2)[:, None]; right = (center+width/2)[:, None]
    dleft = (dcenter-dw/2)[:, None]; dright = (dcenter+dw/2)[:, None]
    raw_h = np.minimum(right, cols+1)-np.maximum(left, cols)
    d_h = min_derivative(right, cols+1, dright, 0.)-max_derivative(left, cols, dleft, 0.)
    d_h = clip_derivative(raw_h, d_h, 1.)
    h = np.clip(raw_h, 0, 1)
    derivative = d_v[:, None]*h+v[:, None]*d_h
    return derivative.reshape(28, SUB, 28).sum(1).ravel()


def targets(p):
    eye = np.eye(5)
    plus = np.array([directional_derivative(p, v) for v in eye])
    minus = np.array([-directional_derivative(p, -v) for v in eye])
    average = (plus+minus)/2
    raw = np.concatenate([average, plus, minus])
    norms = np.linalg.norm(raw, axis=1)
    assert np.all(norms > 0)
    separation = np.linalg.norm(plus-minus, axis=1)/np.linalg.norm(average, axis=1)
    return dict(unit=raw/norms[:, None], raw=raw, norms=norms,
                kinds=['average']*5+['plus']*5+['minus']*5,
                knobs=KNOBS*3, separation=separation,
                kink=bool(np.max(separation) > 1e-10))


def full_svd(v):
    active = np.any(v != 0, axis=0)
    _, s, small = np.linalg.svd(v[:, active], full_matrices=False)
    b = np.zeros((len(s), 784)); b[:, active] = small
    for row in b:
        if row[np.argmax(np.abs(row))] < 0:
            row *= -1
    rank = int(np.sum(s > s[0]*RANK_RTOL))
    assert rank >= 5
    assert np.max(np.abs(b[:rank]@b[:rank].T-np.eye(rank))) < 1e-12
    return dict(basis=b, singular=s, rank=rank, active=active)


def shell_fit(p, radius, seed=SEED):
    base = pixels(p)[0]
    velocity = directions(N, seed)*SCALE
    j = jet(p)[3]
    amounts = radius/np.linalg.norm(velocity@j.T, axis=1)
    for _ in range(8):
        settings = p+amounts[:, None]*velocity
        x = pixels(settings)
        distances = np.linalg.norm(x-base, axis=1)
        if np.max(np.abs(distances/radius-1)) < 2e-6:
            break
        amounts *= radius/distances
    else:
        for i in range(N):
            def objective(t):
                return np.linalg.norm(pixels(p+t*velocity[i])[0]-base)-radius
            hi = amounts[i]*2
            while objective(hi) < 0:
                hi *= 2
            amounts[i] = brentq(objective, 0, hi, xtol=1e-14)
        settings = p+amounts[:, None]*velocity
        x = pixels(settings)
        distances = np.linalg.norm(x-base, axis=1)
    error = float(np.max(np.abs(distances/radius-1)))
    assert error < 3e-6
    fit = full_svd(x-base)
    fit.update(settings=settings, images=x, shell_error=error)
    return fit


def reconstruct(unit_targets, fit):
    b = fit['basis']; rank = fit['rank']
    coeff = unit_targets@b.T
    one_indices = np.argmax(np.abs(coeff[:, :5]), axis=1)
    single = coeff[np.arange(len(coeff)), one_indices, None]*b[one_indices]
    five = coeff[:, :5]@b[:5]
    full = coeff[:, :rank]@b[:rank]
    return dict(coeff=coeff, one_indices=one_indices,
                single=single, five=five, full=full,
                single_error=np.linalg.norm(unit_targets-single, axis=1),
                five_error=np.linalg.norm(unit_targets-five, axis=1),
                full_error=np.linalg.norm(unit_targets-full, axis=1))


def state_list():
    result = [(name, p, 'reference_starts') for name,p in starts()]
    archive = ROOT/'figures/generated_one_cross_path_patterns/repeated_walks.npz'
    if archive.exists():
        paths = np.load(archive)['settings']
        for f,path in enumerate(paths):
            for hop,p in enumerate(path[1:], 1):
                if not any(np.max(np.abs(p-q)) < 1e-12 for _,q,_ in result):
                    result.append((f'walk_F{f+1}_hop_{hop}', p, 'successful_walks'))
    return result


def verify_derivatives(states):
    rows = []
    for name,p,_ in states:
        t = targets(p)
        f0 = pixels(p)[0]
        max_error = 0.
        for k in range(5):
            for sign, index in [(1, 5+k), (-1, 10+k)]:
                h = SCALE[k]*1e-4
                best = np.inf
                for _ in range(7):
                    delta = np.eye(5)[k]*sign*h
                    f1,f2 = pixels(np.array([p+delta,p+2*delta]))
                    finite = sign*(-3*f0+4*f1-f2)/(2*h)
                    error = np.linalg.norm(finite-t['raw'][index])/t['norms'][index]
                    best = min(best, error)
                    h /= 3
                max_error = max(max_error, best)
        j_error = np.linalg.norm(t['raw'][:5].T-jet(p)[3])/np.linalg.norm(t['raw'][:5])
        rows.append(dict(name=name, best_one_sided_finite_difference_relative_error=float(max_error),
                         averaged_derivative_vs_existing_jet_relative_error=float(j_error)))
    assert max(r['best_one_sided_finite_difference_relative_error'] for r in rows) < 1e-7
    assert max(r['averaged_derivative_vs_existing_jet_relative_error'] for r in rows) < 1e-10
    return rows


def render_extended(settings):
    """Same coverage formula, extended arithmetic before differencing."""
    p = np.atleast_2d(np.asarray(settings, np.longdouble))
    out = np.empty((len(p), 784), np.longdouble)
    lo = np.arange(28*SUB, dtype=np.longdouble)/SUB
    mid = lo+np.longdouble(.5)/SUB
    cols = np.arange(28, dtype=np.longdouble)
    for i in range(0, len(p), 64):
        cx,cy,h,w,a = [p[i:i+64,k,None] for k in range(5)]
        top,bot = cy-h/2,cy+h/2
        vertical = np.clip(np.minimum(lo+np.longdouble(1)/SUB,bot)-np.maximum(lo,top),0,np.longdouble(1)/SUB)
        center = cx-np.tan(np.deg2rad(a))*(mid-cy)
        left,right = center-w/2,center+w/2
        overlap = np.clip(np.minimum(right[:,:,None],cols+1)-np.maximum(left[:,:,None],cols),0,1)
        out[i:i+64] = (vertical[:,:,None]*overlap).reshape(-1,28,SUB,28).sum(2).reshape(-1,784)
    return out


def extended_precision_checks(states):
    rows = []
    assert np.finfo(np.longdouble).eps < np.finfo(float).eps
    for name in ['root','spread_02','lean_30']:
        p = next(p for n,p,_ in states if n == name)
        fit = shell_fit(p,1e-4)
        v = render_extended(fit['settings'])-render_extended(p)[0]
        ext = full_svd(np.asarray(v,float))
        rec = reconstruct(targets(p)['unit'],ext)
        rows.append(dict(name=name,rank=ext['rank'],max_full_error=float(rec['full_error'].max()),
                         float64_epsilon=float(np.finfo(float).eps),
                         extended_epsilon=float(np.finfo(np.longdouble).eps),
                         same_formula_coverage_disagreement=float(np.max(np.abs(render_extended(p)[0]-pixels(p)[0])))))
    return rows


def make_figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    d = json.loads((OUT/'results.json').read_text())
    z = np.load(OUT/'vectors_and_weights.npz')
    target = z['root_targets'][:5]
    single = z['root_single'][:5]
    five = z['root_five'][:5]
    full = z['root_full'][:5]
    rank = len(z['root_basis'])
    names = ['Horizontal position', 'Vertical position', 'Height', 'Width', 'Lean']
    errors = [np.linalg.norm(target-x,axis=1) for x in [single,five,full]]
    colors = ['#bc6a35','#3675a6','#299a80']
    fig = plt.figure(figsize=(15,13.1),facecolor='#fafaf7')
    fig.text(.035,.975,'Can SVD mixtures reproduce the true knob directions?',fontsize=21,weight='bold',va='top')
    fig.text(.035,.933,'Root generated 1: (14, 14, 19.75, 3.2, 0°). Fit: 256 neighbors at pixel L2 distance 0.0001; no mean subtraction.',fontsize=10.8)
    ax0=fig.add_axes([.04,.797,.085,.105])
    ax0.imshow(pixels(BASE)[0].reshape(28,28),cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
    ax0.set_xticks([]);ax0.set_yticks([])
    ax0.text(.5,-.12,'Starting image',transform=ax0.transAxes,ha='center',fontsize=9)
    fig.text(.16,.883,'Image: white = no ink, black = full ink. Direction panels: red adds ink; blue removes ink.',fontsize=11)
    fig.text(.16,.854,'Each true direction has length 1. Other panels show the best reconstruction, including its amplitude.',fontsize=11)
    fig.text(.16,.825,'Known true directions supply the target weights. This verifies what the basis contains; it does not discover knob labels.',fontsize=10.8)
    grid=fig.add_gridspec(5,5,left=.16,right=.97,bottom=.16,top=.735,
                          width_ratios=[1,1,1,1,1.6],wspace=.40,hspace=.64)
    limit=max(np.abs(x).max() for x in [target,single,five,full])
    for i in range(5):
        first=None
        for j,x in enumerate([target,single,five,full]):
            ax=fig.add_subplot(grid[i,j])
            if first is None:first=ax
            shown=ax.imshow(x[i].reshape(28,28),cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            if i==0:ax.set_title(['True knob','Best scaled\nsingle vector','Best mixture\nof five vectors',f'Best mixture\nof all {rank} vectors'][j],fontsize=10.5,pad=12)
            label='Unit length = 1' if j==0 else f'Error {errors[j-1][i]:.2e}'
            ax.text(.5,-.12,label,transform=ax.transAxes,ha='center',fontsize=9)
        pos=first.get_position()
        fig.text(.025,(pos.y0+pos.y1)/2,names[i].replace(' ','\n'),fontsize=11,va='center')
        ax=fig.add_subplot(grid[i,4])
        for j in range(3):
            ax.scatter(errors[j][i],2-j,color=colors[j],s=35,zorder=4)
        ax.set_xscale('log');ax.set_xlim(1e-13,1);ax.set_ylim(-.55,2.55)
        ax.set_yticks([2,1,0],['One','Five','All nine'],fontsize=9)
        ax.set_xticks([1e-12,1e-8,1e-4,1]);ax.tick_params(axis='x',labelsize=8)
        ax.grid(axis='x',alpha=.22);ax.set_xlabel('Relative vector error',fontsize=9)
        if i==0:ax.set_title('How much error remains?\nLeft is more accurate',fontsize=11,pad=12)
    cax=fig.add_axes([.19,.082,.32,.014])
    fig.colorbar(shown,cax=cax,orientation='horizontal')
    cax.set_xlabel('Signed pixel change for a unit true direction (one scale for all panels)',fontsize=9)
    fig.text(.56,.093,f'Worst errors across the five root knobs:\nOne vector: {max(errors[0]):.3g}   |   Five: {max(errors[1]):.3g}   |   Nine: {max(errors[2]):.3g}',fontsize=10.5,va='top',linespacing=1.5)
    fig.text(.035,.025,'Error = length of (true unit direction − reconstruction). The full nine-vector residual is dominated by finite arithmetic, checked separately.',fontsize=10)
    fig.savefig(OUT/'01_root_reconstruction.png',dpi=150,facecolor=fig.get_facecolor());plt.close(fig)

    fig=plt.figure(figsize=(15,9.1),facecolor='#fafaf7')
    fig.text(.04,.96,'Do weighted combinations recover the true directions across states?',fontsize=20,weight='bold',va='top')
    fig.text(.04,.909,'62 starting images: the root, 16 other reference starts, and 45 states from the successful walks. All five knobs checked in both directions.',fontsize=10.8)
    fig.text(.04,.883,'Each point is the WORST relative vector error in that group. Targets and SVD vectors use all 784 pixels; true directions are unit length.',fontsize=10.8)
    axes=[fig.add_axes([.09,.30,.36,.48]),fig.add_axes([.57,.30,.36,.48])]
    for ax,kink in zip(axes,[False,True]):
        rows=[next(r for r in d['aggregates'] if r['radius']==radius and r['kink']==kink) for radius in [*RADII,0.]]
        for key,color,label in [('worst_single',colors[0],'Best scaled single vector'),('worst_five',colors[1],'Mixture of five'),('worst_full',colors[2],'Mixture of full retained basis')]:
            ax.plot(range(4),[r[key] for r in rows],color=color,marker='o',lw=1.8,label=label)
        ax.set_yscale('log');ax.set_ylim(5e-16,1.5);ax.set_xlim(-.15,3.15)
        ax.set_xticks(range(4),['0.001','0.0001','0.00001','Exact derivative\ncontrol'])
        ax.tick_params(labelsize=9);ax.grid(alpha=.2)
        ax.set_xlabel('Neighborhood radius; final column uses analytic infinitesimal changes',fontsize=9)
        ax.set_ylabel('Worst relative vector error (lower = better)',fontsize=10)
        title='3 tilted cap-boundary starts' if kink else '59 starts with a common two-sided derivative'
        info='Finite retained basis: 9 vectors. Infinitesimal span: 7.' if kink else 'Finite retained basis: 9–10 vectors. Infinitesimal span: 5.'
        ax.set_title(title+'\n'+info,fontsize=11.5,pad=14)
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,fontsize=9.5,frameon=False,loc='center',bbox_to_anchor=(.5,.852),ncol=3)
    fig.text(.04,.205,'The exact-derivative control uses known analytic derivatives to construct the limiting SVD. It is a geometry check, not a new data-only recovery method.',fontsize=10.2)
    ext=d['extended_precision_controls']
    fig.text(.04,.166,'Arithmetic check at radius 0.0001: using higher-precision coverage differences, nine-vector errors fall to',fontsize=10.5)
    fig.text(.04,.132,'root '+f'{ext[0]["max_full_error"]:.2e}'+', smooth spread start '+f'{ext[1]["max_full_error"]:.2e}'+', and lean 30° boundary '+f'{ext[2]["max_full_error"]:.2e}'+'. Ordinary double precision is used for SVD.',fontsize=10.5)
    fig.text(.04,.087,'At the three boundary starts, the two sides of height/position changes need seven linear directions in total. This does not mean seven independent knobs.',fontsize=10.2)
    fig.text(.04,.045,'Finite full-basis errors increase as tiny neighborhoods magnify renderer subtraction noise. Only three selected starts receive the higher-precision check.',fontsize=10.2)
    fig.savefig(OUT/'02_across_states.png',dpi=155,facecolor=fig.get_facecolor());plt.close(fig)
    with (OUT/'root_weights.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['knob','basis_vector','unit_direction_weight','physical_unit_derivative_weight'])
        norm_lookup={r['knob']:r['raw_target_norm'] for r in csv.DictReader((OUT/'reconstruction_errors.csv').open()) if r['state']=='root' and r['radius']=='0.0001' and r['target_kind']=='average'}
        for i,knob in enumerate(KNOBS):
            for k,c in enumerate(z['root_coefficients'][i]):
                writer.writerow([knob,k+1,c,c*float(norm_lookup[knob])])



def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    states = state_list()
    derivative_checks = verify_derivatives(states)
    rows = []; arrays = {}; summaries = []
    for name,p,source in states:
        t = targets(p)
        for radius in RADII:
            fit = shell_fit(p, radius)
            rec = reconstruct(t['unit'], fit)
            for i in range(15):
                rows.append(dict(state=name, source=source, kink=t['kink'], radius=radius,
                    target_kind=t['kinds'][i], knob=t['knobs'][i], rank=fit['rank'],
                    single_error=float(rec['single_error'][i]), five_error=float(rec['five_error'][i]),
                    full_error=float(rec['full_error'][i]),
                    raw_target_norm=float(t['norms'][i]),
                    best_single_index=int(rec['one_indices'][i]+1)))
            if radius == 1e-4:
                key = name
                arrays[key+'_settings'] = p
                arrays[key+'_targets'] = t['unit']
                arrays[key+'_basis'] = fit['basis'][:fit['rank']]
                arrays[key+'_coefficients'] = rec['coeff'][:, :fit['rank']]
                arrays[key+'_singular'] = fit['singular']
                arrays[key+'_single'] = rec['single']; arrays[key+'_five'] = rec['five']; arrays[key+'_full'] = rec['full']
            if name == 'root':
                arrays[f'root_radius_{radius}_basis'] = fit['basis'][:fit['rank']]
                arrays[f'root_radius_{radius}_coefficients'] = rec['coeff'][:, :fit['rank']]
            if radius == 1e-4:
                summaries.append(dict(name=name, settings=p.tolist(), source=source, kink=t['kink'],
                                      one_sided_separation=t['separation'].tolist(), rank=fit['rank'],
                                      worst_single=float(rec['single_error'].max()), worst_five=float(rec['five_error'].max()),
                                      worst_full=float(rec['full_error'].max()), shell_error=fit['shell_error']))
        # Infinitesimal SVD is a known-derivative control, not sampled inference.
        velocities = directions(N, SEED)*SCALE
        dv = np.array([directional_derivative(p, v) for v in velocities])
        dv /= np.linalg.norm(dv, axis=1)[:, None]
        limit = full_svd(dv)
        lr = reconstruct(t['unit'], limit)
        for i in range(15):
            rows.append(dict(state=name, source=source, kink=t['kink'], radius=0., target_kind=t['kinds'][i],
                             knob=t['knobs'][i], rank=limit['rank'], single_error=float(lr['single_error'][i]),
                             five_error=float(lr['five_error'][i]), full_error=float(lr['full_error'][i]),
                             raw_target_norm=float(t['norms'][i]), best_single_index=int(lr['one_indices'][i]+1)))
        if len(summaries)%10 == 0:
            print(f'{len(summaries)}/{len(states)} starts verified.', flush=True)
    seed_checks = []
    for name in ['root','spread_02','lean_30']:
        p = next(p for n,p,_ in states if n == name)
        t = targets(p)
        first = shell_fit(p, 1e-4, SEED); second = shell_fit(p, 1e-4, 20261013)
        a = reconstruct(t['unit'], first); b = reconstruct(t['unit'], second)
        seed_checks.append(dict(name=name, first_rank=first['rank'], second_rank=second['rank'],
                               max_reconstructed_vector_difference=float(np.linalg.norm(a['full']-b['full'],axis=1).max()),
                               first_worst_full=float(a['full_error'].max()), second_worst_full=float(b['full_error'].max())))
    precision = []
    for name in ['root','spread_02','lean_30']:
        p = next(p for n,p,_ in states if n == name)
        t = targets(p); fit = shell_fit(p, 1e-4)
        native = render(fit['settings']).reshape(-1,784).astype(float)-render(p).reshape(-1).astype(float)
        nf = full_svd(native); rec = reconstruct(t['unit'], nf)
        precision.append(dict(name=name, float32_rank_at_same_cutoff=nf['rank'],
                              float32_worst_five_error=float(rec['five_error'].max()),
                              float32_worst_first_float64_rank_error=float(np.linalg.norm(t['unit']-(t['unit']@nf['basis'][:fit['rank']].T)@nf['basis'][:fit['rank']],axis=1).max()),
                              float32_worst_full_error=float(rec['full_error'].max())))
    aggregate = []
    for radius in [*RADII, 0.]:
        for kink in [False, True]:
            selected = [r for r in rows if r['radius'] == radius and r['kink'] == kink and r['target_kind'] in ['plus','minus']]
            if selected:
                aggregate.append(dict(radius=radius,kink=kink,states=len(set(r['state'] for r in selected)),
                    targets=len(selected), min_rank=min(r['rank'] for r in selected),max_rank=max(r['rank'] for r in selected),
                    worst_single=max(r['single_error'] for r in selected),worst_five=max(r['five_error'] for r in selected),
                    worst_full=max(r['full_error'] for r in selected),
                    median_five=float(np.median([r['five_error'] for r in selected])),
                    median_full=float(np.median([r['full_error'] for r in selected]))))
    result = dict(method=dict(radius=RADII,neighbors=N,seed=SEED,svd='uncentered seed-relative image displacements',
                    rank_cutoff_relative=RANK_RTOL,precision='float64 coverage computation and storage',
                    targets='analytic forward and backward derivatives of the actual min/max/clip renderer; also their average',
                    direction_units='each target normalized to length one; coefficients for physical-unit derivatives obtained by multiplying by its stored norm',
                    reconstruction='orthogonal projection; coefficients are target dot basis vector; no regularization or physical-knob matching',
                    single='best possible scalar multiple of any one of the five leading SVD vectors',
                    full='all sampled singular directions above the stated signal cutoff, not a completed ambient basis',
                    limit='SVD of exact infinitesimal directional derivatives; known-renderer control, not a data-only estimator',
                    oracle='true target vectors supplied to the coefficient solve; this tests representability, not discovery of weights',
                    bounds='two-sided root neighborhoods and walk states can extend beyond default ranges; stroke remains positive and unclipped'),
                  state_count=len(states), kink_states=[r['name'] for r in summaries if r['kink']],
                  aggregates=aggregate, states=summaries, derivative_checks=derivative_checks,
                  seed_checks=seed_checks, float32_controls=precision, extended_precision_controls=extended_precision_checks(states), seconds=time.monotonic()-started)
    (OUT/'results.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item())+'\n')
    with (OUT/'reconstruction_errors.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(OUT/'vectors_and_weights.npz',**arrays)
    make_figures()
    print(json.dumps(dict(aggregates=aggregate,seed_checks=seed_checks,precision=precision),indent=2),flush=True)


if __name__ == '__main__':
    main()
