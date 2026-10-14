"""Extrapolate finite-radius SVD second moments to radius zero.

Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/extrapolate_local_svd_limit.py
Analytic zero-radius data is used only for evaluation, never for fitting.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/svd-limit-mpl')
from pathlib import Path
import sys
import json
import csv
import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from grey_ones import KNOBS
from analyze_root_one_local_svd import directions,SCALE
from experiments.verify_true_knobs_in_svd import directional_derivative,targets,BASE,pixels,SEED,N
from analyze_generated_one_manifold_transport import jet

OUT=ROOT/'figures'/'local_svd_limit_extrapolation'
WINDOWS={'larger':[1e-3,3e-4,1e-4], 'smaller':[1e-4,3e-5,1e-5], 'finest':[1e-5,3e-6,1e-6]}
HELDOUT={'larger':3e-5, 'smaller':3e-6, 'finest':3e-7}


def weights(radii,at=0.):
    radii=np.asarray(radii,float)
    return np.array([np.prod([(at-radii[j])/(radii[i]-radii[j]) for j in range(len(radii)) if j!=i]) for i in range(len(radii))])


def finite_sample(p,radius):
    base=pixels(p)[0]
    velocity=directions(N,SEED)*SCALE
    amount=radius/np.linalg.norm(velocity@jet(p)[3].T,axis=1)
    # Fixed refinements avoid a radius-dependent stopping tolerance changing
    # the effective ray sampling law during extrapolation.
    for _ in range(4):
        x=pixels(p+amount[:,None]*velocity)
        amount*=radius/np.linalg.norm(x-base,axis=1)
    delta=pixels(p+amount[:,None]*velocity)-base
    norms=np.linalg.norm(delta,axis=1)
    error=float(np.max(np.abs(norms/radius-1)))
    assert error<5e-8
    return delta/norms[:,None],error


def covariance(vectors):
    return vectors.T@vectors/len(vectors)


def eigensystem(c):
    active=np.flatnonzero(np.any(c!=0,axis=0)|np.any(c!=0,axis=1))
    values,vectors=np.linalg.eigh(c[np.ix_(active,active)])
    order=np.argsort(values)[::-1]
    values=values[order]
    b=np.zeros((len(values),784));b[:,active]=vectors[:,order].T
    return values,b


def axis_error(basis,limit,k):
    a=basis[:k];ref=limit[:k]
    row,col=linear_sum_assignment(-np.abs(ref@a.T))
    order=col[np.argsort(row)]
    aligned=a[order].copy()
    signs=np.sign(np.sum(aligned*ref,axis=1));signs[signs==0]=1
    aligned*=signs[:,None]
    return np.linalg.norm(aligned-ref,axis=1),aligned


def projection_error(unit,basis,k):
    return np.linalg.norm(unit-(unit@basis[:k].T)@basis[:k],axis=1)


def run():
    old=np.load(ROOT/'figures/true_knobs_in_svd/vectors_and_weights.npz')
    state_list=[(name,old[name+'_settings']) for name in ['root','spread_02','spread_03','lean_30']]
    state_list.append(('smooth_generic',np.array([14.43,14.62,19.73,3.17,17.3])))
    rows=[];limit_records=[];archives={}
    for name,p in state_list:
        t=targets(p)
        velocities=directions(N,SEED)*SCALE
        dv=np.array([directional_derivative(p,v) for v in velocities])
        dv/=np.linalg.norm(dv,axis=1)[:,None]
        c0=covariance(dv)
        ev0,b0=eigensystem(c0)
        rank=int(np.sum(ev0>ev0[0]*1e-10))
        k=rank
        positive_gaps=[min(ev0[i]-ev0[i+1],ev0[i-1]-ev0[i] if i>0 else np.inf) for i in range(k)]
        products=t['unit'][:5]@b0[:5].T
        row,col=linear_sum_assignment(-np.abs(products));assign=col[np.argsort(row)]
        cosine=np.abs(products[np.arange(5),assign])
        limit_records.append(dict(name=name,settings=p.tolist(),limiting_rank=rank,kink=t['kink'],
            top_eigenvalues=ev0[:10].tolist(),smallest_axis_eigenvalue_gap=float(min(positive_gaps)),
            knob_to_assigned_axis_cosines=dict(zip(KNOBS,cosine.tolist())),
            best_single_knob_errors=dict(zip(KNOBS,np.sqrt(np.maximum(0,1-np.max(products**2,axis=1))).tolist())),
            worst_five_span_true_error=float(projection_error(t['unit'][5:],b0,5).max()),
            worst_full_limit_span_true_error=float(projection_error(t['unit'][5:],b0,k).max())))
        cache={}
        for r in sorted(set(sum(WINDOWS.values(),[])+list(HELDOUT.values())),reverse=True):
            v,shell_error=finite_sample(p,r)
            c=covariance(v)
            cache[r]=(c,shell_error)
            ev,b=eigensystem(c)
            err,_=axis_error(b,b0,k)
            leading_err,_=axis_error(b,b0,5)
            rows.append(dict(state=name,kind='measured',window='shared',radius=r,
                             max_axis_error=float(err.max()),max_leading_five_axis_error=float(leading_err.max()),shell_relative_error=shell_error,covariance_relative_error=float(np.linalg.norm(c-c0)/np.linalg.norm(c0)),
                             worst_five_span_true_error=float(projection_error(t['unit'][5:],b,5).max()),
                             worst_limit_rank_span_true_error=float(projection_error(t['unit'][5:],b,k).max())))
            archives[f'{name}_r_{r}_basis']=b[:k]
        for window,radii in WINDOWS.items():
            w=weights(radii)
            estimate=sum(wi*cache[r][0] for wi,r in zip(w,radii))
            ev,b=eigensystem(estimate)
            err,aligned=axis_error(b,b0,k)
            leading_err,_=axis_error(b,b0,5)
            held=HELDOUT[window]
            predicted=sum(wi*cache[r][0] for wi,r in zip(weights(radii,held),radii))
            actual=cache[held][0]
            rows.append(dict(state=name,kind='extrapolated_zero',window=window,radius=0.,
                             max_axis_error=float(err.max()),max_leading_five_axis_error=float(leading_err.max()),
                             covariance_relative_error=float(np.linalg.norm(estimate-c0)/np.linalg.norm(c0)),
                             worst_five_span_true_error=float(projection_error(t['unit'][5:],b,5).max()),
                             worst_limit_rank_span_true_error=float(projection_error(t['unit'][5:],b,k).max()),
                             heldout_radius=held,
                             heldout_covariance_prediction_relative_error=float(np.linalg.norm(predicted-actual)/np.linalg.norm(actual)),
                             lagrange_weights=w.tolist(),min_eigenvalue=float(ev.min())))
            archives[f'{name}_{window}_basis']=aligned
        archives[name+'_limit_basis']=b0[:k]
        archives[name+'_unit_knobs']=t['unit'][:5]
        archives[name+'_settings']=p
        print(f'{name}: limiting rank {rank}; finite windows extrapolated.',flush=True)
    # Sampling-seed sensitivity of individual limiting axes at the root.
    p=BASE
    dv=np.array([directional_derivative(p,v) for v in directions(N,20261013)*SCALE])
    dv/=np.linalg.norm(dv,axis=1)[:,None]
    _,other=eigensystem(covariance(dv))
    ref=archives['root_limit_basis']
    ae,_=axis_error(other,ref,5)
    seed_sensitivity=dict(other_seed=20261013,maximum_axis_difference=float(ae.max()),
                          maximum_other_axis_outside_reference_span=float(projection_error(other[:5],ref,5).max()))
    result=dict(method=dict(root=BASE.tolist(),states=5,neighbors=N,seed=SEED,windows=WINDOWS,heldout_radii=HELDOUT,
                data_matrix='seed-relative displacements normalized by their measured image length; fixed four radial refinements at each scale',
                second_moment='C(r) = normalized_displacements.T @ normalized_displacements / 256; uncentered',
                extrapolation='quadratic in radius for each second-moment entry; evaluate at zero; diagonalize',
                fitting='only the three finite-radius matrices in each window',
                analytic_limit='known exact directional derivatives on same fixed sampling rays; evaluation only',
                neighborhood='same default-range-shaped rays, scaled to equal pixel-L2 shells; root position bounds extended on both sides',
                axis_metric='maximum unit-vector difference to analytic limiting axes after sign/permutation alignment',
                subspace_metric='worst relative projection error of true unit knob derivatives, checking both one-sided directions',
                caveats=['finite extrapolation requires a stable small-radius expansion; no unconditional proof from finite measurements',
                         'individual axes require eigenvalue separation; a convergent subspace need not have uniquely convergent axes',
                         'analytic evaluation does not make limiting SVD axes physical knobs',
                         'finite fixed ray sampling and its population limit are different objects']),
                analytic_limits=limit_records,measurements=rows,root_seed_sensitivity=seed_sensitivity)
    return result,archives


def make_figure(result,archives):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['root','spread_02','spread_03','lean_30','smooth_generic']
    titles=['Root','Smooth spread 02','Near-boundary spread 03','Tilt 30° boundary','Smooth generic']
    colors=['#267c95','#329b75','#bd6e36','#9a5ca8','#6173b3']
    fig=plt.figure(figsize=(15,10.5),facecolor='#fafaf7')
    fig.text(.035,.97,'Can finite neighborhoods predict the zero-radius SVD directions?',fontsize=21,weight='bold',va='top')
    fig.text(.035,.926,'Generated 28 × 28 coverage: white = no ink, black = full ink. Same 256 sampling rays at every radius; float64 coverage.',fontsize=10.8)
    for i,(name,title,color) in enumerate(zip(names,titles,colors)):
        ax=fig.add_axes([.065+i*.185,.765,.08,.11])
        ax.imshow(pixels(archives[name+'_settings'])[0].reshape(28,28),cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
        ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(title,fontsize=9.5,pad=10,color=color)
    ax=fig.add_axes([.09,.30,.39,.37])
    for name,title,color in zip(names,titles,colors):
        measured=sorted([r for r in result['measurements'] if r['state']==name and r['kind']=='measured'],key=lambda r:r['radius'])
        ax.loglog([r['radius'] for r in measured],[r['max_leading_five_axis_error'] for r in measured],color=color,lw=1.5,marker='.',label=title)
        for j,window in enumerate(['larger','smaller','finest']):
            e=next(r for r in result['measurements'] if r['state']==name and r['window']==window and r['kind']=='extrapolated_zero')
            plotr=min(WINDOWS[window])
            ax.scatter(plotr,e['max_leading_five_axis_error'],color=color,marker=['s','*','^'][j],s=[45,100,55][j],zorder=4)
    ax.set_xlabel('Measured neighborhood radius; extrapolation markers sit at smallest fitting radius',fontsize=9)
    ax.set_ylabel('Maximum error versus analytic limiting five SVD axes',fontsize=10)
    ax.set_title('Predicting the SVD limit',fontsize=12,pad=12)
    ax.grid(alpha=.2);ax.tick_params(labelsize=9)
    ax.legend(fontsize=8,frameon=False,loc='upper left')
    ax=fig.add_axes([.60,.30,.35,.37])
    root=next(r for r in result['analytic_limits'] if r['name']=='root')
    ks=['cx','cy','height','width','lean']
    vals=[root['best_single_knob_errors'][k]*100 for k in ks]
    ax.bar(range(5),vals,color=['#5d81a1','#5d81a1','#b96836','#5d81a1','#5d81a1'])
    for i,v in enumerate(vals):ax.text(i,v+.35,f'{v:.2f}%',ha='center',fontsize=10)
    ax.set_xticks(range(5),['X position','Y position','Height','Width','Lean'],rotation=15)
    ax.set_ylim(0,max(vals)*1.22);ax.set_ylabel('Best scaled single limiting-axis error (%)',fontsize=10)
    ax.set_title('The limiting axes remain mixed at the root',fontsize=12,pad=12)
    ax.grid(axis='y',alpha=.2);ax.tick_params(labelsize=9)
    fig.text(.035,.205,'Lines/dots = measured finite-radius axes. Squares = zero-radius estimate from radii 0.001, 0.0003, 0.0001.',fontsize=10.5)
    fig.text(.035,.174,'Stars = zero-radius estimate from radii 0.0001, 0.00003, 0.00001. Marker positions are not actual measurements at radius zero.',fontsize=10.5)
    fig.text(.035,.130,'Triangles = zero-radius estimate from 0.00001, 0.000003, 0.000001. Analytic limits and true knobs are evaluation only.',fontsize=10.5)
    fig.text(.035,.090,'At regular starts the limiting span contains all five true directions. At the tilted boundary, seven directions span both sides.',fontsize=10.5)
    fig.text(.035,.049,'Axes can converge accurately without becoming pure knobs. Crossing a renderer boundary can spoil an extrapolation window.',fontsize=10.5)
    fig.savefig(OUT/'limit_extrapolation.png',dpi=155,facecolor=fig.get_facecolor());plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    result,archives=run()
    (OUT/'results.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item())+'\n')
    np.savez_compressed(OUT/'extrapolated_bases.npz',**archives)
    make_figure(result,archives)
    print(json.dumps(result,indent=2,default=lambda x:x.item()),flush=True)


if __name__=='__main__':main()
