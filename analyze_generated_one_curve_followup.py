"""All-point curve fitting, missing-error analysis, and knob-aligned constructions.

Uses a pooled covering cloud, with no train/validation/test split in the new fits.
Original stored results are preserved and diagnosed separately.
"""
from pathlib import Path
import json
import itertools
import csv
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import numpy as np
from scipy.linalg import solve, eigh
from scipy.stats import spearmanr
from analyze_generated_one_principal_curves import OUT, pixels, settings, project, error_stats, linear_start
from grey_ones import KNOBS, RANGES, render

FOLLOW = OUT/'followup'
BASE=RANGES.mean(axis=1)
SPAN=np.ptp(RANGES,axis=1)


def hat_basis(t,n=65):
    u=np.clip(t,0,1)*(n-1)
    j=np.minimum(u.astype(int),n-2);f=u-j
    b=np.zeros((len(t),n));b[np.arange(len(t)),j]=1-f;b[np.arange(len(t)),j+1]=f
    return b


def smooth_n(x,t,penalty,n):
    b=hat_basis(t,n);d2=np.diff(np.eye(n),n=2,axis=0)
    return solve(b.T@b+penalty*(d2.T@d2)+1e-8*np.eye(n),b.T@x,assume_a='pos')


def resample_n(c,n):
    arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(c,axis=0),axis=1))]
    arc/=max(arc[-1],1e-20)
    return np.column_stack([np.interp(np.linspace(0,1,n),arc,col) for col in c.T])


def fit_all(x,initial,n=65,penalty=1.,max_iter=45,name='curve'):
    c=resample_n(initial,n);best=c.copy();best_sse=project(x,c)[2].mean();history=[];stale=0
    for it in range(1,max_iter+1):
        _,t,_=project(x,c)
        c=resample_n(smooth_n(x,t,penalty,n),n)
        loss=float(project(x,c)[2].mean())
        history.append(dict(iteration=it,mean_squared_image_error=loss,
                            length=float(np.linalg.norm(np.diff(c,axis=0),axis=1).sum())))
        if loss<best_sse-1e-7:best_sse=loss;best=c.copy();stale=0
        else:stale+=1
        if it%15==0:print(f'{name}: iteration {it}, SSE/image {loss:.6f}',flush=True)
        if stale>=7:break
    return best,history


def axis_paths(p,n=129):
    # Each sample gets the exact renderer value along each fixed-midpoint axis.
    paths=[]
    for k in range(5):
        pp=np.tile(BASE,(len(p),1));pp[:,k]=p[:,k]
        paths.append(pixels(pp))
    return np.array(paths)


def additive_design(p,n):
    u=(p-RANGES[:,0])/SPAN
    # Gauge: every learned curve is zero at the common midpoint.
    pieces=[np.ones((len(p),1))]
    for k in range(5):
        b=hat_basis(u[:,k],n);b-=hat_basis(np.array([.5]),n)
        pieces.append(np.delete(b,n//2,axis=1))
    return np.column_stack(pieces)


def additive_fit(p,x,n=65):
    d=additive_design(p,n)
    # Unique midpoint gauge and a very small numerical ridge.
    coef=solve(d.T@d+1e-8*np.eye(d.shape[1]),d.T@x,assume_a='pos')
    pred=d@coef
    nodes=np.zeros((5,n,784))
    for k in range(5):
        nodes[k,np.arange(n)!=n//2]=coef[1+k*(n-1):1+(k+1)*(n-1)]
    return coef[0],nodes,pred


def stateful_path(p,order):
    # Curve k uses current settings of the already moved knobs.
    current=np.tile(BASE,(len(p),1));baseline=pixels(current)
    total=baseline.copy()
    for k in order:
        before=pixels(current)
        current[:,k]=p[:,k]
        after=pixels(current)
        total += after-before
    return total


def original_error_diagnosis(s):
    p=s['test_knobs'];x=s['test_images'];rec=s['reconstructed'];e=x-rec
    e2=e**2;mass=e2.sum()
    rows=np.arange(28)[None,:,None];cols=np.arange(28)[None,None,:]+.5
    top=(p[:,1]-p[:,2]/2)[:,None,None];bottom=(p[:,1]+p[:,2]/2)[:,None,None]
    cap=(abs(rows-np.floor(top))<=1)|(abs(rows-np.floor(bottom))<=1)
    cap=np.broadcast_to(cap,(len(p),28,28))
    center=p[:,0,None,None]-np.tan(np.radians(p[:,4,None,None]))*(rows+.5-p[:,1,None,None])
    left=center-p[:,3,None,None]/2;right=center+p[:,3,None,None]/2
    sides=((abs(cols-left)<=1.5)|(abs(cols-right)<=1.5))&((rows+.5>=top)&(rows+.5<=bottom))&~cap
    masks=[cap,sides,~(cap|sides)]
    regions={name:float(100*np.sum(e2.reshape(-1,28,28)*mask)/mass)
             for name,mask in zip(['top_bottom_bands','side_edge_bands','remaining_pixels'],masks)}
    tangent=[]
    for eps in [2e-4,7e-4]:
        jac=[]
        for k in range(5):
            delta=np.zeros(5);delta[k]=SPAN[k]*eps
            jac.append((pixels(p+delta)-pixels(p-delta))/(2*eps))
        j=np.stack(jac,axis=2)
        u,svals,_=np.linalg.svd(j,full_matrices=False)
        coords=np.einsum('nik,ni->nk',u,e)
        tang2=np.sum(coords**2,axis=1)
        tangent.append(dict(normalized_difference_step=eps,
            tangent_fraction_of_error_percent=float(100*tang2.sum()/mass),
            normal_fraction_of_error_percent=float(100*(mass-tang2.sum())/mass),
            smallest_local_singular_value=float(svals[:,-1].min())))
    knob_predict=[]
    for k in range(5):
        b=hat_basis((p[:,k]-RANGES[k,0])/SPAN[k],65)
        co=solve(b.T@b+1e-8*np.eye(65),b.T@e,assume_a='pos')
        knob_predict.append(float(100*(1-np.sum((e-b@co)**2)/mass)))
    worst=np.argsort(np.linalg.norm(e,axis=1))[-6:][::-1]
    result=dict(missing_variation_percent=100-error_stats(e,x,s['mean'])['explained_percent'],
                error_regions_percent=regions,tangent_decompositions=tangent,
                error_predictable_from_one_knob_percent=knob_predict,
                rms_pixel_error=float(np.sqrt(np.mean(e2))),
                rms_image_error=float(np.sqrt(np.mean(e2.sum(axis=1)))),
                worst_examples=[dict(index=int(i),knobs=p[i].tolist(),image_L2=float(np.linalg.norm(e[i])),
                                     max_pixel_error=float(np.abs(e[i]).max())) for i in worst])
    np.savez_compressed(FOLLOW/'original_error_examples.npz',indices=worst,images=x[worst],reconstructions=rec[worst],
        errors=e[worst],mean_squared_error_map=e2.mean(axis=0),mean_signed_error_map=e.mean(axis=0))
    return result


def export_equations(s):
    # Plain CSV for all original pixel-valued curve definitions.
    path=FOLLOW/'original_curve_nodes.csv'
    with path.open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['curve','node','normalized_arc_position']+[f'pixel_{r}_{c}' for r in range(28) for c in range(28)])
        for k,c in enumerate(s['curves']):
            arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(c,axis=0),axis=1))];arc/=arc[-1]
            for i in range(len(c)):writer.writerow([k+1,i,float(arc[i])]+c[i].tolist())


def main():
    FOLLOW.mkdir(exist_ok=True)
    s=np.load(OUT/'model_and_samples.npz')
    export_equations(s)
    original=original_error_diagnosis(s)
    print('Original missing-error diagnosis:',json.dumps(original),flush=True)
    u=np.array(list(itertools.product(np.linspace(0,1,5),repeat=5)))
    grid=RANGES[:,0]+u*SPAN
    rng=np.random.default_rng(734);faces=rng.random((1024,5))
    for i in range(len(faces)):faces[i,i%5]=1e-5 if i%2 else 1-1e-5
    p=np.vstack([s['train_knobs'],settings(2048,732),s['test_knobs'],grid,RANGES[:,0]+faces*SPAN])
    x=pixels(p);mean=x.mean(0);z=x-mean
    _,v=eigh(z.T@z);basis=v[:,::-1].T
    fixed=pixels(BASE)[0]+np.sum(axis_paths(p)-pixels(BASE)[0],axis=0)
    fixed_stats=error_stats(x-fixed,x,mean)
    additive_models=[]
    for n in [33,65,129]:
        a,c,pred=additive_fit(p,x,n)
        stat=error_stats(x-pred,x,mean)
        additive_models.append(dict(nodes=n,metrics=stat))
        print('Best knob-aligned additive model:',n,stat,flush=True)
        if n==65:
            additive_mean=a;additive_curves=c;additive_pred=pred
    # The state-dependent physical curve family reconstructs joint moves exactly.
    forward=stateful_path(p,[0,1,2,3,4]);reverse=stateful_path(p,[4,3,2,1,0])
    exact=dict(forward_max_pixel_error=float(np.abs(x-forward).max()),reverse_max_pixel_error=float(np.abs(x-reverse).max()),
        forward_reverse_max_pixel_error=float(np.abs(forward-reverse).max()),
        metrics=error_stats(x-forward,x,mean))
    # All coverage points participate in every fit. No validation set is reserved.
    fit_curves=[];history=[];res=z.copy();all_stats=[]
    for k in range(5):
        c,h=fit_all(res,linear_start(res),65,1.,45,f'all-point curve {k+1}')
        _,t,_=project(res,c);corr=[spearmanr(t,p[:,j]).statistic for j in range(5)]
        if corr[np.nanargmax(np.abs(corr))]<0:c=c[::-1]
        fit_curves.append(c);history.append(h);res-=project(res,c)[0]
        all_stats.append(error_stats(res,x,mean))
        print('All-point cumulative:',k+1,all_stats[-1],flush=True)
    pc_stats=[]
    for k in range(1,6):pc_stats.append(error_stats(z-z@basis[:k].T@basis[:k],x,mean))
    # Start from exact physical lean and width slices, then allow the fit to bend freely.
    starts=[]
    for knob in [4,3]:
        pp=np.tile(BASE,(65,1));pp[:,knob]=np.linspace(*RANGES[knob],65)
        c,h=fit_all(z,pixels(pp)-mean,65,1.,45,f'{KNOBS[knob]}-initialized first curve')
        fitted,t,_=project(z,c)
        corr=[float(spearmanr(t,p[:,j]).statistic) for j in range(5)]
        if corr[np.argmax(abs(np.array(corr)))]<0:c=c[::-1];corr=[-q for q in corr]
        starts.append(dict(initialization=KNOBS[knob],metrics=error_stats(z-fitted,x,mean),knob_spearman=corr))
        if knob==4:lean_started=c
        else:width_started=c
    # Structural examples: strongest fixed-axis failures, plus known seed and corners.
    worst=np.argsort(np.linalg.norm(x-fixed,axis=1))[-6:][::-1]
    example_p=np.vstack([BASE,p[worst],RANGES[:,0],RANGES[:,1]])
    true=pixels(example_p);fixed_ex=pixels(BASE)[0]+np.sum(axis_paths(example_p)-pixels(BASE)[0],axis=0)
    learned_ex=additive_design(example_p,65)
    # Recover coefficient layout for deterministic evaluation of the saved additive model.
    coef=np.vstack([additive_mean]+[np.delete(c,32,axis=0) for c in additive_curves])
    learned_ex=learned_ex@coef
    final_curves=np.array(fit_curves)
    np.savez_compressed(FOLLOW/'models_and_examples.npz',mean=mean,curves=final_curves,coverage_knobs=p,
         additive_mean=additive_mean,additive_curves=additive_curves,lean_initialized=lean_started,
         width_initialized=width_started,example_knobs=example_p,example_true=true,example_fixed=fixed_ex,
         example_additive=learned_ex,example_stateful=stateful_path(example_p,[0,1,2,3,4]),
         baseline_knobs=BASE,baseline_image=pixels(BASE)[0],pca_basis=basis[:5])
    results=dict(coverage_cloud=dict(total=len(p),uniform_samples=10240,full_lattice=3125,near_faces=1024,
        weighting='Equal weights for all pooled points; changed from the earlier uniform-only score'),
        original_missing_error=original,fixed_midpoint_knob_curves=fixed_stats,
        optimized_knob_aligned_additive_models=additive_models,state_dependent_knob_curves=exact,
        all_point_curves_cumulative=all_stats,all_point_pca_cumulative=pc_stats,
        initialized_from_physical_knob=starts,
        fit_settings=dict(nodes=65,penalty=1.,max_iterations=45,selection='Lowest reconstruction error on the entire covering cloud'),
        determinism='PCA, projection, and smoothing have no random draws; sample generation and initialization specify the result')
    (FOLLOW/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    (FOLLOW/'fit_history.json').write_text(json.dumps(history,indent=2)+'\n')
    print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
