"""Grouped physical knob directions across random generated ones.

Run from the repository root: .venv/bin/python analyze_generated_one_knob_direction_families.py
No learned representation is claimed to identify the manifold or its controls.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/generated-one-direction-families-mpl')
from pathlib import Path
import itertools
import json
import csv
import hashlib
import numpy as np
from scipy.linalg import eigh, solve
from scipy.spatial.distance import cdist
from scipy.cluster.vq import kmeans2
from scipy.stats import qmc
from grey_ones import render, KNOBS, RANGES
from analyze_generated_one_manifold_transport import jet

OUT = Path(__file__).resolve().parent / 'figures/generated_one_knob_direction_families'
SEED = 20261017
NAMES = ['Horizontal center', 'Vertical center', 'Height', 'Width', 'Lean']
COLORS = ['#007aaf', '#db7a12', '#7e51a0', '#16896c', '#c8476b']
DIMS = [1, 2, 3, 5, 8, 12, 20, 32, 48, 64, 96, 128, 192, 256, 320, 384]


def directions(p):
    return np.stack([jet(x)[3].T for x in p])  # sample, knob, pixel


def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-300)


def error(truth, prediction):
    absolute = np.linalg.norm(truth-prediction, axis=1)
    relative = absolute / np.linalg.norm(truth, axis=1)
    return dict(median=float(np.median(relative)), p95=float(np.quantile(relative, .95)),
                max=float(relative.max()), rms=float(np.sqrt(np.mean(relative**2))),
                max_pixel=float(np.abs(truth-prediction).max()), max_L2=float(absolute.max()))


def basis_fit(x, centered):
    mean = x.mean(0) if centered else np.zeros(x.shape[1])
    centered_x = x-mean
    active = np.any(centered_x != 0, axis=0)
    _, s, small_v = np.linalg.svd(centered_x[:,active], full_matrices=False)
    # Null singular vectors are arbitrary completions, not learned directions.
    rank = int(np.count_nonzero(s > s[0]*1e-10)) if s[0] else 0
    v = np.zeros((rank,x.shape[1])); v[:,active] = small_v[:rank]
    return mean, s, v


def project(x, mean, v, k):
    return mean + ((x-mean) @ v[:k].T) @ v[:k]


def curve(x, mean, v, dims):
    # Direct residuals avoid cancellation near floating-point precision.
    return [dict(dim=k, **error(x, project(x, mean, v, k))) for k in sorted(set([k for k in dims if k <= len(v)] + [len(v)]))]


def ridge_predict(features, y, query, alpha):
    scale = np.linalg.norm(features, axis=0)
    scale[scale < 1e-12] = 1
    a = features / scale
    b = query / scale
    return b @ solve(a.T@a + alpha*np.eye(a.shape[1]), a.T@y, assume_a='pos')


def poly_features(p, degree):
    z = (p-RANGES[:,0]) / np.ptp(RANGES, axis=1) * 2 - 1
    cols = [np.ones(len(z))] + [z[:,i] for i in range(5)]
    if degree == 2:
        cols += [z[:,i]*z[:,j] for i in range(5) for j in range(i,5)]
    return np.column_stack(cols)


def kernel_fit(x, y, valx, valy, query, gammas=(.125,.5,2,8), alphas=(1e-8,1e-5,.001,.1)):
    best = None
    for gamma in gammas:
        gram = np.exp(-gamma*cdist(x,x,'sqeuclidean'))
        vals, vecs = eigh(gram)
        crossval = np.exp(-gamma*cdist(valx,x,'sqeuclidean'))
        projected = vecs.T@y
        for alpha in alphas:
            weights = vecs @ (projected / (np.maximum(vals,0)+alpha)[:,None])
            pred = crossval@weights
            score = error(valy,pred)['rms']
            if best is None or score < best[0]:
                best = score, gamma, alpha, weights
    score,gamma,alpha,weights = best
    pred = np.exp(-gamma*cdist(query,x,'sqeuclidean'))@weights
    return pred, dict(gamma=gamma, alpha=alpha, validation_rms=score)


def kpca_fit(x):
    # RBF kernel, median pairwise distance. Test vectors only enter transform.
    ds = cdist(x,x,'sqeuclidean')
    gamma = 1/np.median(ds[ds>0])
    kernel = np.exp(-gamma*ds)
    km = kernel.mean(0); allmean = float(km.mean())
    centered = kernel-km[None,:]-km[:,None]+allmean
    w,v = eigh(centered)
    keep = np.flatnonzero(w > w.max()*1e-12)[::-1]
    w=w[keep];v=v[:,keep]
    def transform(q,k):
        cross = np.exp(-gamma*cdist(q,x,'sqeuclidean'))
        cross = cross-km[None,:]-cross.mean(1,keepdims=True)+allmean
        return cross@(v[:,:k]/np.sqrt(w[:k]))
    return transform, dict(gamma=float(gamma), retained_eigenvectors=len(w))


def finite_difference_checks(p):
    rng=np.random.default_rng(SEED+1)
    points=p[rng.choice(len(p),24,replace=False)]
    checks=[]
    for knob in range(5):
        maximum=[];central=[];rounding=[]
        for x in points:
            j=jet(x)[3][:,knob]
            step=1e-7 if knob<4 else 1e-6
            a=x.copy();b=x.copy();a[knob]+=step;b[knob]-=step
            fd=(render(a,dtype=np.float64)[0]-render(b,dtype=np.float64)[0]).ravel()/(2*step)
            maximum.append(np.linalg.norm(fd-j)/np.linalg.norm(j))
            central.append(np.array_equal(jet(a)[4],jet(b)[4]))
            f32=render(x)[0].ravel().astype(float)
            f64=render(x,dtype=np.float64)[0].ravel()
            rounding.append(np.max(np.abs(f32-f64)))
        checks.append(dict(knob=KNOBS[knob],max_relative_fd_error=float(max(maximum)),
                           all_checks_in_single_smooth_cell=bool(all(central)),
                           max_float32_image_rounding=float(max(rounding))))
    assert max(x['max_relative_fd_error'] for x in checks)<1e-6
    return checks


def possible_pixel_support(caps_only=False):
    """Envelope of possibly nonzero pixels, directly from the renderer box.

    For a fixed subrow, feasible cy is an interval. The extrema of the
    center offset -slope*(mid-cy) occur at the interval/angle corners.
    For cap derivatives, cy must permit top or bottom in that subrow.
    """
    from grey_ones import SUB
    support=np.zeros((28,28),bool)
    cxlo,cxhi=RANGES[0];cylo,cyhi=RANGES[1];hlo,hhi=RANGES[2]
    widthmax=RANGES[3,1];slopes=np.tan(np.deg2rad(RANGES[4]))
    for i in range(28*SUB):
        lo=i/SUB;hi=lo+1/SUB;mid=(lo+hi)/2
        intervals=[(lo-hhi/2,hi+hhi/2)]
        if caps_only:intervals=[(lo+hlo/2,hi+hhi/2),(lo-hhi/2,hi-hlo/2)]
        for low,high in intervals:
            low=max(cylo,low);high=min(cyhi,high)
            if low>=high:continue
            shifts=[-slope*(mid-y) for slope in slopes for y in [low,high]]
            left=cxlo+min(shifts)-widthmax/2
            right=cxhi+max(shifts)+widthmax/2
            support[i//SUB]|=(np.arange(28)+1>left)&(np.arange(28)<right)
    return support


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(SEED)
    p=RANGES[:,0]+rng.random((768,5))*np.ptp(RANGES,axis=1)
    # Edge probes avoid equality kinks. Exact corners are treated separately.
    levels=[1e-7,.5+1.234e-7,1-1e-7]
    grid=RANGES[:,0]+np.array(list(itertools.product(levels,repeat=5)))*np.ptp(RANGES,axis=1)
    train, val, test = p[:384],p[384:512],p[512:]
    allp=np.concatenate([p,grid])
    print('Computing analytic directions at',len(allp),'states',flush=True)
    d=directions(allp); images=render(allp,dtype=np.float64).reshape(-1,784)
    tr=d[:384];va=d[384:512];te=d[512:768];ed=d[768:]
    checks=finite_difference_checks(p)
    # Independent coverage enrichment: low-discrepancy interior plus edge-biased
    # states, generated without looking at the held-out vectors.
    broad=qmc.Sobol(5,scramble=True,seed=SEED+99).random_base2(12)
    boundary=np.random.default_rng(SEED+100).beta(.18,.18,(2048,5))
    boundary=np.clip(boundary,1e-9,1-1e-9)
    coarse=np.array(list(itertools.product([1e-6,.23,.47,.71,1-1e-6],repeat=5)))
    extra_p=RANGES[:,0]+np.concatenate([broad,boundary,coarse])*np.ptp(RANGES,axis=1)
    fresh=np.random.default_rng(SEED+200).random((512,5))
    final_grid=np.array(list(itertools.product([1e-5,.37,.68,1-1e-5],repeat=5)))
    audit_p=RANGES[:,0]+np.concatenate([fresh,final_grid])*np.ptp(RANGES,axis=1)
    audit_d=directions(audit_p)
    print('Coverage audit:',len(extra_p),'additional training states',flush=True)
    extra_d=directions(extra_p)
    assert np.max(np.abs(images[:24]-np.stack([jet(x)[0] for x in p[:24]])))<1e-13
    arrays={'settings':allp,'images':images,'directions':d,'train_indices':np.arange(384),
            'validation_indices':np.arange(384,512),'test_indices':np.arange(512,768),
            'edge_indices':np.arange(768,len(allp)), 'coverage_training_settings':extra_p,'final_audit_settings':audit_p,'final_audit_directions':audit_d}
    result=dict(seed=SEED,training_states=384,validation_states=128,test_states=256,
                near_edge_grid_states=len(grid),parameter_order=KNOBS,ranges=RANGES.tolist(),
                direction_definition='analytic derivative of coverage image per +1 pixel (cx, cy, height, width) or +1 degree (lean)',
                dtype='float64 coverage calculation, before default float32 storage',
                derivative_checks=checks,knobs=[],source_sha256={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ['grey_ones.py','analyze_generated_one_manifold_transport.py']})
    for j,name in enumerate(KNOBS):
        print('Analyzing',name,flush=True)
        x,y,z=tr[:,j],va[:,j],te[:,j]
        e=ed[:,j]
        mean,s,v=basis_fit(x,False)
        pm,ps,pv=basis_fit(x,True)
        um,us,uv=basis_fit(unit(x),False)
        arrays[name+'_svd_basis']=v;arrays[name+'_pca_basis']=pv;arrays[name+'_pca_mean']=pm
        arrays[name+'_unit_basis']=uv
        enriched=np.concatenate([x,extra_d[:,j]])
        em,es,ev=basis_fit(enriched,False)
        arrays[name+'_coverage_basis']=ev
        rank=int(np.count_nonzero(s>s[0]*1e-10))
        support=possible_pixel_support(caps_only=(j==2))
        active_rows=int(np.count_nonzero(support.any(1)))
        row_sum_rank=[0,5,5,7,0][j]
        row_sums=ev.reshape(len(ev),28,28).sum(2)
        observed_row_rank=int(np.count_nonzero(np.linalg.svd(row_sums,compute_uv=False)>1e-10))
        assert observed_row_rank==row_sum_rank
        assert int(support.sum())-active_rows+row_sum_rank==len(ev)
        assert np.array_equal(support.ravel(),np.any(np.abs(ev)>1e-10,axis=0))
        u=unit(d[:768,j]);pair=(u@u.T)[np.triu_indices(768,1)]
        record=dict(knob=name,rank_rtol_1e10=rank,singular_values=s.tolist(),
            svd_test=curve(z,mean,v,DIMS),svd_edge=curve(e,mean,v,DIMS),
            pca_test=curve(z,pm,pv,DIMS),pca_edge=curve(e,pm,pv,DIMS),
            unit_svd_test=curve(unit(z),um,uv,DIMS),
            mean_direction_length=float(np.linalg.norm(u.mean(0))),
            pairwise_cosine=dict(p05=float(np.quantile(pair,.05)),median=float(np.median(pair)),p95=float(np.quantile(pair,.95)),negative_fraction=float(np.mean(pair<0))),
            raw_speed_quantiles=np.quantile(np.linalg.norm(d[:768,j],axis=1),[0,.05,.5,.95,1]).tolist(),
            mean_template_error=error(z,np.tile(x.mean(0),(len(z),1))),
            row_sum_structure=dict(possible_pixels=int(support.sum()),active_rows=active_rows,
                row_sum_space_dimension=row_sum_rank,observed_row_sum_space_dimension=observed_row_rank,
                rank_bound=int(support.sum())-active_rows+row_sum_rank),
            coverage_audit=dict(training_states=len(enriched),rank=len(ev),test=error(z,project(z,em,ev,len(ev))),
                edge=error(e,project(e,em,ev,len(ev))),
                test_curve=curve(z,em,ev,DIMS),edge_curve=curve(e,em,ev,DIMS),
                active_training_pixels=int(np.count_nonzero(np.any(x!=0,axis=0))),
                active_enriched_pixels=int(np.count_nonzero(np.any(enriched!=0,axis=0))),
                newly_active_test_pixels=int(np.count_nonzero(np.any(z!=0,axis=0)&~np.any(x!=0,axis=0))),
                final_random_test=error(audit_d[:512,j],project(audit_d[:512,j],em,ev,len(ev))),
                final_edge_test=error(audit_d[512:,j],project(audit_d[512:,j],em,ev,len(ev))),
                final_random_curve=curve(audit_d[:512,j],em,ev,DIMS),
                final_edge_curve=curve(audit_d[512:,j],em,ev,DIMS)))
        arrays[name+'_pairwise_cosines']=pair[::32]
        for threshold in [.1,.01,1e-3,1e-6,1e-10]:
            key=str(threshold)
            record.setdefault('svd_dimensions_for_worst_test_error',{})[key]=next((c['dim'] for c in record['svd_test'] if c['max']<=threshold),None)
        record['span_learning_curve']=[]
        for n in [48,96,192,384]:
            _,ss,vv=basis_fit(x[:n],False)
            rr=int(np.count_nonzero(ss>ss[0]*1e-10))
            record['span_learning_curve'].append(dict(training_states=n,rank=rr,test=error(z,project(z,np.zeros(784),vv,rr)),edge=error(e,project(e,np.zeros(784),vv,rr))))
        # A union of 12 affine PCA patches, with nearest-centroid encoding.
        centers,labels=kmeans2(unit(x),12,minit='++',seed=SEED,iter=50)
        local=[]
        for cluster in range(12):
            group=x[labels==cluster]
            lm,ls,lv=basis_fit(group,True)
            lr=int(np.count_nonzero(ls>max(ls[0],1e-300)*1e-10))
            local.append((lm,lv[:lr]))
        choices=np.argmin(cdist(unit(z),centers),axis=1)
        echoices=np.argmin(cdist(unit(e),centers),axis=1)
        record['local_pca_test']=[];record['local_pca_edge']=[]
        for k in [1,2,3,5,8,12,20,32]:
            def reconstruct(q,ch):
                return np.stack([project(q[t:t+1],local[c][0],local[c][1],min(k,len(local[c][1])))[0] for t,c in enumerate(ch)])
            record['local_pca_test'].append(dict(dim=k,**error(z,reconstruct(z,choices))))
            record['local_pca_edge'].append(dict(dim=k,**error(e,reconstruct(e,echoices))))
        # Kernel PCA plus a nonlinear inverse, with inverse regularization chosen on validation.
        transform,kmeta=kpca_fit(unit(x));record['kernel_pca_meta']=kmeta
        record['kernel_pca_test']=[];record['kernel_pca_edge']=[]
        for k in [2,5,12,32]:
            tx=transform(unit(x),k);vy=transform(unit(y),k);tz=transform(unit(z),k);ee=transform(unit(e),k)
            scale=np.std(tx,axis=0);scale[scale<1e-12]=1
            pred,meta=kernel_fit(tx/scale,x,vy/scale,y,np.concatenate([tz,ee])/scale,
                                  gammas=(.01,.05,.2),alphas=(1e-7,1e-4,.01))
            record['kernel_pca_test'].append(dict(dim=k,**error(z,pred[:len(z)]),inverse=meta))
            record['kernel_pca_edge'].append(dict(dim=k,**error(e,pred[len(z):])))
            if k==2: arrays[name+'_kernel_scores']=np.concatenate([tx,vy,tz])
        # Test whether the starting state predicts the entire vector, without SVD truncation.
        state_models={}
        for degree in [1,2]:
            a=poly_features(train,degree);b=poly_features(val,degree);c=poly_features(np.concatenate([test,grid]),degree)
            best=None
            for alpha in [1e-8,1e-4,.01,1.]:
                score=error(y,ridge_predict(a,x,b,alpha))['rms']
                if best is None or score<best[0]:best=(score,alpha)
            pred=ridge_predict(a,x,c,best[1])
            state_models['linear_settings' if degree==1 else 'quadratic_settings']=dict(test=error(z,pred[:len(z)]),edge=error(e,pred[len(z):]),alpha=best[1])
        normp=(allp-RANGES[:,0])/np.ptp(RANGES,axis=1)
        pred,meta=kernel_fit(normp[:384],x,normp[384:512],y,normp[512:])
        state_models['rbf_settings']=dict(test=error(z,pred[:256]),edge=error(e,pred[256:]),hyperparameters=meta)
        # Image-based prediction uses a dual ridge model; settings are not supplied.
        ix=images[:384];iv=images[384:512];iz=images[512:];im=ix.mean(0);ym=x.mean(0)
        gram=(ix-im)@(ix-im).T;ew,ev=eigh(gram);iy=ev.T@(x-ym)
        best=None
        for alpha in [1e-8,1e-5,.001,.1,10.]:
            w=ev@(iy/(np.maximum(ew,0)+alpha)[:,None])
            score=error(y,ym+(iv-im)@(ix-im).T@w)['rms']
            if best is None or score<best[0]:best=(score,alpha,w)
        pred=ym+(iz-im)@(ix-im).T@best[2]
        state_models['ridge_pixels']=dict(test=error(z,pred[:256]),edge=error(e,pred[256:]),alpha=best[1])
        record['state_prediction']=state_models
        # Fit the closest common target to training tangent lines, then test on new sources.
        uu=unit(x);xx=images[:384]
        a=np.eye(784)-uu.T@uu/len(uu)
        b=(xx-uu*np.einsum('ij,ij->i',xx,uu)[:,None]).mean(0)
        target=solve(a,b,assume_a='pos')
        testx=images[512:768];testu=unit(z);delta=target-testx
        ticks=np.einsum('ij,ij->i',delta,testu)
        misses=np.linalg.norm(delta-ticks[:,None]*testu,axis=1)
        spread=np.sqrt(np.mean(np.sum((testx-testx.mean(0))**2,axis=1)))
        # Common length moves are a different question from intersection of tangent rays.
        xc=testx-testx.mean(0);uc=testu-testu.mean(0)
        besttick=max(0.,-np.sum(xc*uc)/np.sum(uc*uc))
        ratio=np.sqrt(np.mean(np.sum((xc+besttick*uc)**2,axis=1)))/spread
        record['endpoint_probe']=dict(closest_training_line_target_test_rms_miss=float(np.sqrt(np.mean(misses**2))),
            source_image_rms_spread=float(spread),miss_divided_by_source_spread=float(np.sqrt(np.mean(misses**2))/spread),
            forward_fraction=float(np.mean(ticks>=0)),best_shared_positive_unit_tick=float(besttick),
            shared_tick_endpoint_spread_ratio=float(ratio),
            caveat='Tangent lines extrapolate outside the curved family; this tests a common straight-line target, not a nonlinear future endpoint.')
        arrays[name+'_line_target']=target
        # Centered PCA scatter preserves sign; each dot is one held-out direction.
        arrays[name+'_pca_scores']=(d[:768,j]-pm)@pv[:2].T
        record['centered_pca_first_two_variance_fraction']=float(np.sum(ps[:2]**2)/np.sum(ps**2))
        result['knobs'].append(record)
    # Paired interventions: change exactly one setting at each starting state.
    base=p[:96];base_d=d[:96]
    sensitivity=np.empty((5,5));turning=np.empty((5,5))
    for k in range(5):
        changed=base.copy();changed[:,k]=RANGES[k,0]+rng.random(96)*np.ptp(RANGES,axis=1)[k]
        nd=directions(changed)
        sensitivity[:,k]=np.median(np.linalg.norm(nd-base_d,axis=2)/np.linalg.norm(base_d,axis=2),axis=0)
        turning[:,k]=np.median(np.einsum('nkp,nkp->nk',unit(nd),unit(base_d)),axis=0)
    result['paired_setting_sensitivity_median_relative_change']=sensitivity.tolist()
    result['paired_setting_cosine_median']=turning.tolist()
    # Renderer-specific identities. No learned fitting is used here.
    slope=np.tan(np.deg2rad(allp[:,4]))
    width_pred=.5*np.abs(d[:,0])
    cap=d[:,1]-slope[:,None]*d[:,0]
    height_pred=.5*np.abs(cap)
    result['identities']=dict(width_from_horizontal=error(d[:,3],width_pred),height_from_centers=error(d[:,2],height_pred),
        min_width_minus_abs_slope=float(RANGES[3,0]-np.max(np.abs(np.tan(np.deg2rad(RANGES[4]))))),
        max_abs_mass_derivative_errors={
            'cx_zero':float(np.abs(d[:,0].sum(1)).max()),'cy_zero':float(np.abs(d[:,1].sum(1)).max()),
            'height_equals_width':float(np.abs(d[:,2].sum(1)-allp[:,3]).max()),
            'width_equals_height':float(np.abs(d[:,3].sum(1)-allp[:,2]).max()),
            'lean_zero':float(np.abs(d[:,4].sum(1)).max())})
    # An exact renderer kink: symmetric derivatives can hide disagreeing sides.
    root=np.array([14.,14.,19.75,3.2,0.]);eps=1e-7;f=render(root,dtype=np.float64).ravel()
    kink=[]
    for k in range(5):
        a=root.copy();b=root.copy();a[k]+=eps;b[k]-=eps
        plus=(render(a,dtype=np.float64).ravel()-f)/eps
        minus=(f-render(b,dtype=np.float64).ravel())/eps
        kink.append(dict(knob=KNOBS[k],one_sided_difference_L2=float(np.linalg.norm(plus-minus)),
                         relative_side_difference=float(np.linalg.norm(plus-minus)/max(np.linalg.norm(plus),np.linalg.norm(minus),1e-300))))
    result['root_kink_check']=kink
    # A genuine lateral pixel event: upright edges exactly on integer columns.
    event=np.array([14.,14.37,19.73,4.,0.]);f=render(event,dtype=np.float64).ravel()
    event_checks=[]
    for k in [0,3]:
        left=event.copy();right=event.copy();left[k]-=eps;right[k]+=eps
        plus=(render(right,dtype=np.float64).ravel()-f)/eps
        minus=(f-render(left,dtype=np.float64).ravel())/eps
        event_checks.append(dict(knob=KNOBS[k],settings=event.tolist(),one_sided_difference_L2=float(np.linalg.norm(plus-minus)),
            relative_side_difference=float(np.linalg.norm(plus-minus)/max(np.linalg.norm(plus),np.linalg.norm(minus)))))
    result['lateral_pixel_event_checks']=event_checks
    np.savez_compressed(OUT/'arrays.npz',**arrays)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'reconstruction_errors.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['knob','method','split','dim','median','p95','max','rms','max_pixel','max_L2']);writer.writeheader()
        for rec in result['knobs']:
            for method in ['svd','pca','local_pca','kernel_pca']:
                for split in ['test','edge']:
                    for row in rec[method+'_'+split]:writer.writerow({k:v for k,v in dict(knob=rec['knob'],method=method,split=split,**row).items() if k in writer.fieldnames})
    # Supplemental curves are kept separately from the original held-out tests.
    with (OUT/'coverage_audit_errors.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['knob','split','dim','median','p95','max','rms','max_pixel','max_L2']);writer.writeheader()
        for rec in result['knobs']:
            for split in ['random','edge']:
                for row in rec['coverage_audit']['final_'+split+'_curve']:writer.writerow(dict(knob=rec['knob'],split=split,**row))
    make_figures(result,arrays)
    make_page(result,arrays)
    make_readme(result)
    print(json.dumps(dict(identities=result['identities'],summary=[dict(knob=r['knob'],rank=r['rank_rtol_1e10'],dimensions=r['svd_dimensions_for_worst_test_error'],alignment=r['mean_direction_length'],endpoint=r['endpoint_probe']) for r in result['knobs']]),indent=2),flush=True)


def figure_base(title,subtitle,size=(15,8)):
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=size,facecolor='#faf9f6')
    fig.text(.055,.965,title,fontsize=19,weight='bold',va='top')
    fig.text(.055,.92,subtitle,fontsize=10.5,va='top',linespacing=1.5)
    return fig


def make_figures(result,a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    # Familiar source images come first. Directions stay full-resolution pixel maps.
    fig=figure_base('What does each knob change at different starting images?',
        'Read each row left to right: original coverage image, then the five true local direction maps. All five knobs vary between rows.\n'
        'Black source pixels = ink (coverage 1); white = no ink (0). Red direction pixels gain ink; blue pixels lose ink as the named knob increases.',(16,11))
    indices=[512,513,514]
    maplimit=float(np.max(np.abs(unit(a['directions'][indices]))))
    grid=fig.add_gridspec(3,6,left=.045,right=.975,bottom=.20,top=.83,hspace=.75,wspace=.25)
    for row,i in enumerate(indices):
        ax=fig.add_subplot(grid[row,0]);ax.imshow(a['images'][i].reshape(28,28),cmap='gray_r',vmin=0,vmax=1)
        ax.set_title('Starting image' if row==0 else '')
        ax.set_xticks([0,14,27]);ax.set_yticks([0,14,27]);ax.tick_params(labelsize=7)
        ax.set_ylabel('pixel row ↓',fontsize=8)
        p=a['settings'][i]
        ax.text(.5,-.23,f'cx {p[0]:.2f}, cy {p[1]:.2f}\nh {p[2]:.2f}, w {p[3]:.2f}, lean {p[4]:.1f}°',ha='center',va='top',transform=ax.transAxes,fontsize=8)
        for j in range(5):
            ax=fig.add_subplot(grid[row,j+1]);v=a['directions'][i,j];speed=np.linalg.norm(v)
            ax.imshow((v/speed).reshape(28,28),cmap='RdBu_r',vmin=-maplimit,vmax=maplimit)
            ax.set_title(NAMES[j] if row==0 else '')
            ax.set_xticks([]);ax.set_yticks([])
            ax.text(.5,-.12,f'raw length {speed:.3f}\nper '+('degree' if j==4 else 'pixel'),ha='center',va='top',transform=ax.transAxes,fontsize=8.5)
    cax=fig.add_axes([.36,.075,.38,.017]);fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-maplimit,maplimit),cmap='RdBu_r'),cax=cax,orientation='horizontal').set_label('Coefficient in unit-length direction (shared scale across all 15 maps)',fontsize=9)
    fig.text(.055,.015,f'Each vector contains all 784 pixels; no projection. Each map is divided by its own raw length to compare shape. All 784 coefficients are shown; no clipping.',fontsize=9)
    fig.savefig(OUT/'01_direction_examples.png',dpi=155);plt.close(fig)

    fig=figure_base('How many shared pixel patterns reconstruct a new direction?',
        'Dotted: original 384-image fit. Solid: coverage-enriched fit from 9,653 images. Error = reconstructed-vector L2 error / true-vector L2 length.\n'
        'Solid curves use a fresh final audit: 512 random images (left), 1,024 near-edge grid images (right). Lower is better; dashed horizontal line = 10⁻¹⁰ target.',(15,7.5))
    grid=fig.add_gridspec(1,2,left=.08,right=.97,bottom=.23,top=.78,wspace=.23)
    for panel,split in enumerate(['test','edge']):
        ax=fig.add_subplot(grid[0,panel])
        for j,rec in enumerate(result['knobs']):
            rows=rec['svd_'+split];ax.plot([r['dim'] for r in rows],[max(r['max'],1e-15) for r in rows],':',lw=1.2,alpha=.6,color=COLORS[j])
            rows=rec['coverage_audit']['final_random_curve' if split=='test' else 'final_edge_curve']
            ax.plot([r['dim'] for r in rows],[max(r['max'],1e-15) for r in rows],'-o',ms=3,color=COLORS[j],label=f'{NAMES[j]} (enriched rank {rec["coverage_audit"]["rank"]})')
        ax.axhline(1e-10,color='#555',ls='--',lw=1);ax.text(1.3,1.9e-10,'near-exact target',fontsize=8)
        ax.set_xscale('log');ax.set_yscale('log');ax.set_ylim(1e-15,3);ax.set_xlim(1,400)
        ax.set_xlabel('Number of shared SVD patterns used (log scale)');ax.set_ylabel('Worst relative direction error (log scale)')
        ax.set_title('Random held-out states' if panel==0 else 'Near-edge grid: additional stress test')
        ax.grid(alpha=.16)
    handles,labels=fig.axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.075),ncol=3,frameon=False,fontsize=9)
    fig.text(.055,.025,'Uncentered SVD uses a basis through zero. Rank uses singular values > largest × 10⁻¹⁰; arbitrary null-space completions are excluded.\nDotted curves use the original 256 / 243 test states. Solid curves use fresh audit states. These are sampled direction tests, not a proof about every image.',fontsize=9)
    fig.savefig(OUT/'02_reconstruction.png',dpi=155);plt.close(fig)

    fig=figure_base('Are directions for the same knob all pointing the same way?',
        'Each distribution compares pairs of unit-length true directions at 768 independently random starting images. No signs are flipped to force agreement.\n'
        'Cosine +1 = identical direction; 0 = perpendicular; −1 = opposite. Vertical lines show each knob’s median; the displayed distribution is a sampled summary.',(15,7))
    ax=fig.add_axes([.09,.22,.86,.55])
    for j,rec in enumerate(result['knobs']):
        values=a[rec['knob']+'_pairwise_cosines'];hist,bins=np.histogram(values,bins=np.linspace(-1,1,71),density=True)
        ax.plot((bins[1:]+bins[:-1])/2,hist,color=COLORS[j],lw=2,label=f'{NAMES[j]}: median {rec["pairwise_cosine"]["median"]:.2f}')
        ax.axvline(rec['pairwise_cosine']['median'],color=COLORS[j],alpha=.45,ls='--')
    ax.set_xlim(-1,1);ax.set_xlabel('Cosine between two directions for the same knob');ax.set_ylabel('Relative frequency density')
    ax.axvline(0,color='#666',lw=.8);ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.12)
    fig.text(.055,.065,'Height and width directions only add ink, so their pairwise cosines cannot be negative. Translation and lean can gain ink in some pixels and lose it in others.\nThe full 784-pixel vectors define these comparisons; no 2D projection is involved. Each vector’s magnitude is discarded only for this angle comparison.',fontsize=10)
    fig.savefig(OUT/'03_alignment.png',dpi=155);plt.close(fig)

    # Exact structural relationships illustrated at one new random state.
    i=512;p=a['settings'][i];d=a['directions'][i];s=np.tan(np.deg2rad(p[4]))
    fig=figure_base('Two exact relationships between knob direction families',
        'One held-out starting image, followed by two calculations on its true pixel-direction maps. The same calculations were checked at all 1,011 sampled states.\n'
        'Top: width direction = half the absolute horizontal-center direction. Bottom: height direction = half |vertical-center direction − tan(lean) × horizontal-center direction|.',(15,9))
    grid=fig.add_gridspec(2,5,left=.06,right=.97,bottom=.18,top=.80,hspace=.5,wspace=.3)
    rows=[[(a['images'][i],'Source: coverage 0–1'),(d[0],'Horizontal-center direction'),(np.abs(d[0])/2,'Half the absolute value'),(d[3],'Actual width direction'),(np.abs(d[0])/2-d[3],'Prediction − actual')],
          [(a['images'][i],'Same starting image'),(d[1]-s*d[0],'Vertical − tan(lean) × horizontal'),(np.abs(d[1]-s*d[0])/2,'Half the absolute value'),(d[2],'Actual height direction'),(np.abs(d[1]-s*d[0])/2-d[2],'Prediction − actual')]]
    limit=max(np.abs(v).max() for row in rows for v,label in row[1:4])
    for r,row in enumerate(rows):
        for c,(v,label) in enumerate(row):
            ax=fig.add_subplot(grid[r,c]);ax.set_title(label,fontsize=9);ax.set_xticks([]);ax.set_yticks([])
            if c==0: ax.imshow(v.reshape(28,28),cmap='gray_r',vmin=0,vmax=1)
            elif c==4:
                ax.imshow(v.reshape(28,28),cmap='RdBu_r',vmin=-limit,vmax=limit)
                ax.text(.5,-.13,f'max pixel error {np.abs(v).max():.2e}',transform=ax.transAxes,ha='center',fontsize=9,color='#16896c')
            else:ax.imshow(v.reshape(28,28),cmap='RdBu_r',vmin=-limit,vmax=limit)
    cax=fig.add_axes([.30,.12,.4,.018]);fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-limit,limit),cmap='RdBu_r'),cax=cax,orientation='horizontal').set_label('Raw derivative coefficient: red = positive; blue = negative (shared scale; no clipping)',fontsize=9)
    fig.text(.055,.02,'These identities use the actual direction maps, not fitted SVD components. The left/right edges occupy distinct pixels within each row throughout the default box;\nthe top and bottom caps also occupy distinct rows. They do not imply that one fixed direction works everywhere, or recover the direction from the image alone.',fontsize=9)
    fig.savefig(OUT/'04_exact_links.png',dpi=155);plt.close(fig)


def make_readme(r):
    lines=['# Five knob direction families','',
        'A reproducible exploration of the known physical derivative for each knob across the generated straight-one family. Real MNIST is not used.','',
        'Run from the repository root: `.venv/bin/python analyze_generated_one_knob_direction_families.py`.','',
        'Open `index.html` for the illustrated report and interactive reconstruction inspector. `results.json` holds all measured summaries, `arrays.npz` holds settings, images, full raw derivatives and fitted bases, and `reconstruction_errors.csv` holds reconstruction curves.','',
        '## Sampling and precision','',
        'Seed 20261017. All five controls vary independently and uniformly across `grey_ones.RANGES`. There are 384 training images, 128 validation images for nonlinear-model choices, and 256 untouched random test images. An additional 243-point grid uses normalized levels 0.0000001, 0.5000001234, and 0.9999999 on every knob. It probes near edges without landing exactly on known cap kinks. This is finite sampled coverage, not a guarantee for the whole range.','',
        'A direction is the analytic derivative per positive unit of its knob: one pixel for centers, height and width; one degree for lean. These are instantaneous directions, not finite-step differences. Derivatives use the float64 coverage calculation before default float32 storage. At nonsmooth clipping/subrow events a unique two-sided derivative may not exist; exact-event one-sided derivatives must be treated separately. Twenty-four random states per knob were checked against tiny central rendered moves.','',
        '## Global SVD','',
        'Rows are direction vectors, columns are 784 pixel coefficients. Uncentered SVD uses a linear space through zero; centered PCA uses the training mean plus a basis. Unit-vector SVD separates orientation from speed. All reconstruction errors use true held-out vectors, without clipping. Relative error = vector reconstruction L2 error / true vector L2 length. Near-exact target = worst relative error at most 1e-10. Training rank counts singular values above the largest times 1e-10. Neither rank nor variance retention is called intrinsic dimension.','',
        '| Knob | Training rank | Patterns for worst test error ≤1% | Patterns for ≤1e-10 | Median pairwise cosine |',
        '| --- | ---: | ---: | ---: | ---: |']
    for q in r['knobs']:
        ds=q['svd_dimensions_for_worst_test_error'];lines.append(f'| {q["knob"]} | {q["rank_rtol_1e10"]} | {ds["0.01"] or "not reached"} | {ds["1e-10"] or "not reached"} | {q["pairwise_cosine"]["median"]:.3f} |')
    lines += ['', '## Coverage refinement and fresh audit','',
        'The original 384 fitted states miss rare configurations: even their full nonzero SVD spans do not reconstruct every test vector. Arbitrary zero-singular-value basis completions are excluded. We added 4096 independent scrambled Sobol states, 2048 beta(0.18, 0.18) edge-biased states, and a 3125-point grid at normalized levels 0.000001, 0.23, 0.47, 0.71, 0.999999. The total enriched training set has 9653 states. After this refinement we ran a fresh audit of 512 uniform random states and 1024 grid states at normalized levels 0.00001, 0.37, 0.68, 0.99999. This fresh audit was not used to fit or choose the basis.','',
        '| Knob | Enriched rank | Worst fresh random error | Worst fresh edge error |',
        '| --- | ---: | ---: | ---: |']
    for q in r['knobs']:
        c=q['coverage_audit'];lines.append(f'| {q["knob"]} | {c["rank"]} | {c["final_random_test"]["max"]:.3e} | {c["final_edge_test"]["max"]:.3e} |')
    lines += ['', 'The rank counts match simple support and row-total constraints. A renderer-box envelope gives 283 possible pixels across 23 rows, and 96 cap pixels across 6 rows. Horizontal and lean changes preserve ink in every row: 283 - 23 = 260. Height changes add equal total ink at the top and bottom: 96 - 1 = 95. Vertical changes have zero row totals except at the six cap rows, whose combined total is zero: 283 - (17 + 1) = 265. Width changes add the same ink amount in each of the 17 fully covered interior rows: 283 - 16 = 267. These bounds match the observed ranks. The envelope is computed directly from parameter-box extrema for each renderer subrow; the independence/rank lower bound is numerical.','', 'All five enriched spans reconstruct the fresh audit to approximately float64 precision. This is a sampled result, not a proof of full-box completeness or minimum rank. The initial test set was used diagnostically during the coverage refinement; the fresh audit is reported separately. The second figure compares the initial and enriched fits.','', '## Other reductions and prediction','',
        'The comparison includes centered PCA, SVD on unit vectors, a union of 12 local affine PCA patches, and RBF kernel PCA with a nonlinear RBF inverse. Kernel PCA is fitted on unit directions; its inverse predicts full raw directions, so changing magnitude can remain a source of error. The kernel uses the median training pairwise squared distance; inverse bandwidth and regularization are chosen using only validation states. A local PCA patch is selected by the closest training unit-vector centroid; the stated dimension is per patch and excludes its discrete label and the stored collection of bases. These methods consume an observed vector, not just its source image. They have explicit decoders and held-out errors.','',
        'Starting-state predictors are a separate supervised test: linear/quadratic settings regression, RBF settings regression, and ridge regression directly from all source-image pixels. The first three receive the five true settings. Hyperparameters use only validation images. Their errors are in `results.json`; they do not count as discovery of unknown knobs.','',
        '## Exact relationships','',
        'Across the default box, the stroke is wide enough that its left and right edge contributions cannot share a pixel within one image row: minimum width minus maximum absolute slope is greater than one pixel. Horizontal shift changes those edge contributions with opposite signs; increasing width changes each by half as much with positive signs. Therefore, pixel by pixel, `width direction = abs(horizontal-center direction) / 2`.','',
        'Vertical center motion consists of `tan(lean) * horizontal-center direction` plus signed top/bottom cap motion. Removing the horizontal part leaves the caps. They occupy separate rows, so `height direction = abs(vertical-center direction - tan(lean) * horizontal-center direction) / 2`. These statements follow the renderer formulas away from events; they were checked numerically across all sampled states. They link actual physical derivatives, not individual SVD components.','',
        f'Largest relative width-identity error: {r["identities"]["width_from_horizontal"]["max"]:.3e}. Largest relative height-identity error: {r["identities"]["height_from_centers"]["max"]:.3e}.','',
        '## Other questions','',
        'Pairwise cosines retain the positive-knob sign. The mean length of unit directions measures alignment: 1 means all identical; near 0 means cancellation. The figure histogram uses a deterministic subsample of pairs, while reported quantiles use every unordered pair among 768 random states.','',
        'Paired setting interventions replace one knob at a time at 96 starting states, recomputing all five directions. They measure which settings influence each direction; the sensitivity entry is the median relative vector change. This is a change to a direction pattern, not a direct ink change.','',
        'The common-target probe fits a point closest in least-squares distance to training tangent lines, and measures misses at new random images. It also finds the common positive unit-length step that most reduces source-image spread. Neither tangent-line extrapolation nor small projected distances prove an endpoint on the curved family. A common nonlinear future endpoint is a different question.','',
        'At the canonical root, one-sided direction checks are recorded separately in `root_kink_check`. A separate upright integer-column event has disagreeing one-sided horizontal/width directions, recorded in `lateral_pixel_event_checks`. The analytic routine averages cap-side derivatives at exact cap events; interpreting that average as a unique derivative would be misleading.','',
        '## Figures','',
        '- `01_direction_examples.png`: source images and five physical directions, with unit-normalized maps and raw speeds.',
        '- `02_reconstruction.png`: worst held-out and near-edge error versus SVD basis size.',
        '- `03_alignment.png`: cosine distributions without low-dimensional projection.',
        '- `04_exact_links.png`: direct pixel-map evidence for the two exact family relationships.','',
        'Limit: the renderer and known control labels are supplied. This experiment studies their direction families; it does not infer the physical knobs from unlabeled images or build a global learned image decoder.','']
    (OUT/'README.md').write_text('\n'.join(lines))


def make_page(r,a):
    # An offline inspection tool. Stored bases and source vectors give exact same decoders as the report.
    examples=np.arange(512,536)
    payload=dict(knobs=KNOBS,names=NAMES,settings=a['settings'][examples].round(8).tolist(),
        images=a['images'][examples].round(9).tolist(),directions=a['directions'][examples].tolist(),
        full_predictions=np.stack([project(a['directions'][examples,j],np.zeros(784),a[n+'_coverage_basis'],len(a[n+'_coverage_basis'])) for j,n in enumerate(KNOBS)],axis=1).tolist(),
        ranks=[len(a[n+'_coverage_basis']) for n in KNOBS],
        bases=[a[n+'_svd_basis'][:64].round(12).tolist() for n in KNOBS],
        pcabases=[a[n+'_pca_basis'][:64].round(12).tolist() for n in KNOBS],
        means=[a[n+'_pca_mean'].round(12).tolist() for n in KNOBS],
        coveragebases=[a[n+'_coverage_basis'][:64].round(12).tolist() for n in KNOBS],
        singular_values=[q['singular_values'] for q in r['knobs']],
        scatter=[a[n+'_pca_scores'].round(8).tolist() for n in KNOBS],
        scatter_settings=a['settings'][:768].round(8).tolist(),
        scatter_variance=[q['centered_pca_first_two_variance_fraction'] for q in r['knobs']])
    cards=''
    for q,name in zip(r['knobs'],NAMES):
        ds=q['svd_dimensions_for_worst_test_error']
        cards+=f'<tr><td>{name}</td><td>{q["rank_rtol_1e10"]}</td><td>{ds["0.01"] or "not reached"}</td><td>{ds["1e-10"] or "not reached"}</td><td>{q["pairwise_cosine"]["median"]:.3f}</td></tr>'
    coverage_rows=''
    for q,name in zip(r['knobs'],NAMES):
        c=q['coverage_audit']
        coverage_rows+=f'<tr><td>{name}</td><td>{c["rank"]}</td><td>{c["final_random_test"]["max"]:.2e}</td><td>{c["final_edge_test"]["max"]:.2e}</td></tr>'
    predictions=''
    for q,name in zip(r['knobs'],NAMES):
        models=q['state_prediction']
        predictions+='<tr><td>'+name+'</td>'+''.join(f'<td>{100*models[m]["test"]["median"]:.1f}% / {100*models[m]["test"]["max"]:.1f}%</td>' for m in ['linear_settings','quadratic_settings','rbf_settings','ridge_pixels'])+'</tr>'
    endpoint=''
    for q,name in zip(r['knobs'],NAMES):
        e=q['endpoint_probe'];endpoint+=f'<tr><td>{name}</td><td>{e["miss_divided_by_source_spread"]:.3f}</td><td>{100*e["forward_fraction"]:.0f}%</td><td>{e["shared_tick_endpoint_spread_ratio"]:.3f}</td></tr>'
    comparisons=''
    for q,name in zip(r['knobs'],NAMES):
        comparisons+='<tr><td>'+name+'</td>'+''.join(f'<td>{100*next(z for z in q[m+"_test"] if z["dim"]==5)["median"]:.1f}% / {100*next(z for z in q[m+"_test"] if z["dim"]==5)["max"]:.1f}%</td>' for m in ['svd','pca','local_pca','kernel_pca'])+'</tr>'
    sensitivity=''.join('<tr><td>'+name+'</td>'+''.join(f'<td>{v:.2f}</td>' for v in row)+'</tr>' for name,row in zip(NAMES,r['paired_setting_sensitivity_median_relative_change']))
    scatters=''.join(f'<option value="{i}">{n}</option>' for i,n in enumerate(NAMES))
    html='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Five knob direction families</title>
<style>:root{font:16px/1.6 system-ui,sans-serif;color:#23313a;background:#faf9f6}body{max-width:1200px;margin:auto;padding:30px 24px 70px}h1{font-size:40px;line-height:1.15;max-width:900px}h2{margin-top:44px;font-size:25px}p{max-width:920px}a{color:#007aaf}.label{color:#16896c;text-transform:uppercase;letter-spacing:.12em;font-size:12px;font-weight:700}.lead{font-size:20px}.note{border-left:4px solid #16896c;padding:12px 18px;background:#f0f5f2}.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:10px;border-bottom:1px solid #d6dddc}th{background:#eef1ef}figure{margin:26px 0}figure img{width:100%;height:auto}figcaption{color:#57646b;font-size:14px}.controls{display:flex;flex-wrap:wrap;gap:20px;align-items:center;background:#edf1ef;padding:18px;border-radius:9px}label{display:block;font-size:14px}select{padding:7px;font:inherit;max-width:100%}.maps{display:grid;grid-template-columns:repeat(4,1fr);gap:20px;margin-top:20px}.map{text-align:center}.map canvas{width:100%;image-rendering:pixelated;background:white}.map h3{font-size:16px;margin:0 0 8px}.map p{font-size:13px;line-height:1.4}output{font-weight:700}code{font-size:.9em}.small{font-size:14px;color:#57646b}@media(max-width:720px){h1{font-size:30px}.maps{grid-template-columns:repeat(2,1fr)}body{padding:20px 12px}.controls{gap:10px}}</style>
<body><p class="label">Generated straight 1s · measured exploration · seed 20261017</p><h1>One knob has a family of directions.<br>What structure does that family share?</h1>
<p class="lead">We generated 768 random images, calculated the five true local direction vectors at each one, and grouped them by physical knob. We fitted 384 vectors per group, tuned nonlinear models on 128, and tested on 256 unseen random states plus 243 near-edge grid states.</p>
<div class="note"><b>The most concrete pattern:</b> the width direction is half the absolute horizontal-center direction, pixel by pixel. A similar operation recovers the height direction from the two center directions and the lean. These identities hold to floating-point precision in this experiment. A small shared linear basis is a separate, harder question.</div>
<h2>1 · Begin with images and real pixel changes</h2><p>A source image has 784 coverage values: 0 means no ink and 1 means full ink. A direction has 784 signed values: it describes what changes under an infinitesimal positive turn of one knob. Red gains ink; blue loses ink.</p>
<figure><img src="01_direction_examples.png" alt="Three starting images followed by each of their five signed physical direction maps"><figcaption>All controls vary between rows. Unit normalization makes direction shapes comparable; the raw lengths below the maps retain the speed information. A direction map is not itself a valid stroke image.</figcaption></figure>
<h2>2 · Can a small shared basis reconstruct the whole direction?</h2><p>Uncentered SVD asks for shared pixel patterns that add together through zero. Centered PCA adds a shared mean first. We measure each error directly in all 784 pixels, including the worst new image. The near-exact target here is a worst relative error of 10⁻¹⁰.</p>
<figure><img src="02_reconstruction.png" alt="Worst direction reconstruction errors versus shared SVD basis size on random and near-edge test sets"></figure>
<div class="scroll"><table><thead><tr><th>Knob</th><th>Training rank</th><th>Patterns for ≤1% worst test error</th><th>Patterns for ≤10⁻¹⁰</th><th>Median pair cosine</th></tr></thead><tbody>__CARDS__</tbody></table></div>
<p><b>The original few hundred states missed rare configurations.</b> We then fitted a separate basis using 9,653 training images: the original 384 plus 4,096 low-discrepancy interior states, 2,048 edge-biased states and an independent 3,125-point coarse grid. After this coverage refinement, we tested on another 512 random images and a fresh 1,024-point near-edge grid.</p>
<div class="scroll"><table><thead><tr><th>Knob</th><th>Enriched SVD rank</th><th>Worst fresh random error</th><th>Worst fresh near-edge error</th></tr></thead><tbody>__COVERAGE__</tbody></table></div>
<p class="note">With enough shared patterns, all five direction groups reconstruct the fresh audit to about 10⁻¹⁵ relative error. Height needs 95 patterns; the other groups need 260–267. The nonlinear family can depend on few controls while spanning many fixed pixel patterns. This is an observed linear-span result, not a certified minimum or proof about every state.</p>
<p class="small">Counts use the tested basis sizes, not an exhaustive search for the smallest count. “Not reached” means none of those sizes passed. Training rank uses relative singular-value threshold 10⁻¹⁰. Neither that rank nor a projection’s appearance identifies intrinsic dimension.</p>
<p><b>The large ranks have a simple explanation.</b> Only 283 pixels can change anywhere in the default box. They occupy 23 rows. Horizontal shifts and lean preserve ink in every row, giving 283 − 23 = <b>260</b> available linear directions. Width adds equal ink to each of 17 always-full interior rows, leaving 283 − 16 = <b>267</b>. Height acts only on 96 cap pixels, and top/bottom added ink must balance, leaving <b>95</b>. Vertical shifts have 17 zero-sum interior rows and one total-cap constraint, leaving 283 − 18 = <b>265</b>. These row-total bounds match every measured rank above.</p><p class="small">Possible-pixel support comes from an envelope of the actual renderer formulas over the default parameter box. The numerical SVD establishes the observed rank; this does not identify unique physical axes within that span.</p>
<h2>3 · Inspect a new image yourself</h2><p>Choose one of 24 held-out starting images. Compare the true direction, the decoded direction, and the residual. All direction panels share one color scale, calculated from the true and reconstructed vectors and their difference; nothing is clipped. The residual is prediction minus truth.</p>
<div class="controls"><label>Knob<select id="knob">__OPTIONS__</select></label><label>Held-out starting image<select id="sample"></select></label><label>Decoder<select id="method"><option value="svd">Uncentered SVD</option><option value="pca">Centered PCA</option><option value="coverage">Coverage-enriched SVD, partial</option><option value="full">Coverage-enriched SVD, full span</option></select></label><label>Patterns: <output id="count">5</output><br><input id="rank" type="range" min="1" max="64" value="5"></label></div>
<p id="settings" class="small"></p><div class="maps"><div class="map"><h3>Starting image</h3><canvas id="source" width="280" height="280"></canvas><p>Coverage: white 0 → black 1</p></div><div class="map"><h3>True local direction</h3><canvas id="truth" width="280" height="280"></canvas><p id="speed"></p></div><div class="map"><h3>Reconstructed direction</h3><canvas id="prediction" width="280" height="280"></canvas><p id="err"></p></div><div class="map"><h3>Residual</h3><canvas id="residual" width="280" height="280"></canvas><p id="maxerr"></p></div></div><p id="scale" class="small"></p>
<h2>The leading shared patterns</h2><p>Choose a knob to see its training mean and the first five uncentered SVD patterns. These are full 28 × 28 pixel maps, not projections. Every map below is normalized to unit length, on one common color scale. An SVD pattern’s sign is arbitrary: positive and negative coefficients can both be used when reconstructing a vector.</p><label>Direction family<select id="modeknob">__OPTIONS__</select></label><div class="maps" id="modegallery" style="grid-template-columns:repeat(3,1fr)"></div><p id="modescale" class="small"></p>
<h2>A projected view: do nearby directions share a starting lean?</h2><p>Each dot below is one true direction among the original 768 random images, projected onto the first two centered PCA axes for the selected knob. Color shows the source image’s lean. The percentage states how much training variation those two axes retain; the other directions are hidden. This view cannot establish reconstruction accuracy.</p><label>Projected family<select id="cloudknob">__OPTIONS__</select></label><canvas id="cloud" width="1000" height="430" style="width:100%;background:white"></canvas><p id="cloudcaption" class="small"></p>
<h2>4 · Do nonlinear reductions do better?</h2><p>The entries below are median / worst relative error at five coordinates. Local PCA uses 12 patches, so it stores more information than one global basis: the patch label, its mean, and its basis. Kernel PCA has a fitted nonlinear inverse. Both fit training vectors only and reconstruct new vectors; they are exploratory models with no guarantee of numerical accuracy.</p>
<div class="scroll"><table><thead><tr><th>Knob</th><th>SVD, 5</th><th>PCA, 5</th><th>Local PCA, 5 per patch</th><th>Kernel PCA + inverse, 5</th></tr></thead><tbody>__COMPARISONS__</tbody></table></div>
<h2>5 · Which directions point together?</h2><figure><img src="03_alignment.png" alt="Full-pixel pairwise cosine distributions for the five knob direction families"></figure>
<p>Increasing width and height only adds ink. Their directions lie in the positive pixel orthant. Center and lean changes include both signs, and their directions can disagree more strongly across states. Averages can hide this spread.</p>
<h2>6 · Two exact relationships across families</h2><figure><img src="04_exact_links.png" alt="Original image and pixel-map calculations that exactly recover width and height directions"></figure>
<p><code>width direction = abs(horizontal-center direction) / 2</code></p><p><code>height direction = abs(vertical-center direction − tan(lean) × horizontal-center direction) / 2</code></p>
<p>Here tan uses lean in degrees after conversion to radians. Why it works: shifting the stroke moves its two side edges in opposite signs; widening moves both outward at half the speed. Removing the horizontal part of vertical-center motion leaves the top and bottom caps; taking their absolute values turns sliding into growing. The side edges occupy distinct pixels within each row throughout the default box, and the caps occupy distinct rows.</p>
<p class="note">These rules link <em>actual physical direction maps at the same image</em>. They do not say that arbitrary SVD components have these identities, or that a constant direction works at every image.</p>
<h2>7 · Does the starting state tell us the direction?</h2><p>Separate supervised predictors try to produce the entire direction. The first three receive the true five settings; the last receives only the source image’s pixels. Each entry is median / worst relative error on 256 new random states. Hyperparameters use validation data only.</p>
<div class="scroll"><table><thead><tr><th>Knob</th><th>Linear settings</th><th>Quadratic settings</th><th>RBF settings</th><th>Ridge from pixels</th></tr></thead><tbody>__PREDICTIONS__</tbody></table></div>
<p>Changing just one setting can also change the direction associated with another knob. The table below replaces one setting at a time at 96 random starts. An entry is median vector-change length divided by the original vector length; 0 means no change, 1 means a change as large as the original direction. This measures direction dependence, not a direct change of ink.</p>
<div class="scroll"><table><thead><tr><th>Direction being measured ↓ / setting changed →</th><th>Horizontal center</th><th>Vertical center</th><th>Height</th><th>Width</th><th>Lean</th></tr></thead><tbody>__SENSITIVITY__</tbody></table></div>
<h2>8 · Do their arrows aim at a common endpoint?</h2><p>We fitted the point closest to all training tangent lines, then measured how far new tangent lines miss it. A ratio near zero would support a common straight-line target; a ratio near one means the miss is as large as the original image spread. We also asked whether one shared positive-length move along the unit directions shrinks that spread.</p>
<div class="scroll"><table><thead><tr><th>Knob</th><th>Target miss / starting spread</th><th>Target ahead of arrow</th><th>Best shared positive-step spread / starting spread</th></tr></thead><tbody>__ENDPOINT__</tbody></table></div>
<p class="small">These are tangent-line extrapolations, which may leave the image family and valid coverage range. A common nonlinear future endpoint is a different question. The target was fitted on training images only.</p>
<h2>What this experiment establishes</h2><p>These are measured properties of the supplied renderer and physical knob labels. A derivative is piecewise defined: at pixel or subrow events the two one-sided directions may disagree. We checked smooth random derivatives against rendered moves and recorded separate one-sided checks at the canonical root and at an upright stroke whose side edges sit on integer pixel columns. The latter has genuinely different one-sided horizontal and width directions. No learned model here identifies unknown knobs or guarantees coverage of the entire family.</p>
<p><a href="README.md">Method and limitations</a> · <a href="results.json">All measured summaries</a> · <a href="reconstruction_errors.csv">Error curves as CSV</a> · <a href="arrays.npz">Full vectors, settings, and bases</a></p>
<script id="data" type="application/json">__DATA__</script><script>
'use strict';const data=JSON.parse(document.getElementById('data').textContent);const el=id=>document.getElementById(id);for(let i=0;i<data.images.length;i++){const o=document.createElement('option');o.value=i;o.textContent=String(i+1);el('sample').append(o)}
function paint(id,values,lim,coverage=false){const ctx=el(id).getContext('2d');for(let r=0;r<28;r++)for(let c=0;c<28;c++){let v=values[r*28+c];if(coverage){const g=Math.round(255*(1-v));ctx.fillStyle=`rgb(${g},${g},${g})`}else{v=Math.max(-1,Math.min(1,v/lim));const m=Math.abs(v);const white=Math.round(250*(1-m));ctx.fillStyle=v>=0?`rgb(${Math.round(250-50*m)},${white},${white})`:`rgb(${white},${Math.round(250-120*m)},${Math.round(250-45*m)})`}ctx.fillRect(c*10,r*10,10,10)}}
function update(){const j=+el('knob').value,i=+el('sample').value,full=el('method').value==='full',k=full?data.ranks[j]:+el('rank').value,pca=el('method').value==='pca',coverage=el('method').value==='coverage';el('count').textContent=k;el('rank').disabled=full;const truth=data.directions[i][j],mean=pca?data.means[j]:new Array(784).fill(0),basis=(pca?data.pcabases:(coverage?data.coveragebases:data.bases))[j],prediction=full?data.full_predictions[i][j].slice():mean.slice();if(!full)for(let b=0;b<k;b++){let score=0;for(let p=0;p<784;p++)score+=(truth[p]-mean[p])*basis[b][p];for(let p=0;p<784;p++)prediction[p]+=score*basis[b][p]}const residual=prediction.map((v,p)=>v-truth[p]);const norm=Math.hypot(...truth),err=Math.hypot(...residual)/norm;const lim=Math.max(...truth.map(Math.abs),...prediction.map(Math.abs),...residual.map(Math.abs),1e-30);paint('source',data.images[i],1,true);paint('truth',truth,lim);paint('prediction',prediction,lim);paint('residual',residual,lim);el('settings').textContent=data.knobs.map((n,t)=>`${n} = ${data.settings[i][t].toFixed(t===4?2:3)}${t===4?'°':''}`).join('  ·  ');el('speed').textContent=`Raw vector length ${norm.toFixed(4)} per ${j===4?'degree':'pixel'}`;el('err').textContent=`Relative error ${(100*err).toFixed(4)}% (${err.toExponential(2)})`;el('maxerr').textContent=`Largest pixel error ${Math.max(...residual.map(Math.abs)).toExponential(3)}`;el('scale').textContent=`Shared direction scale: blue −${lim.toFixed(4)} → white 0 → red +${lim.toFixed(4)}. All 784 coefficients shown. Partial decoders show 1–64 patterns. The full-span option uses all ${data.ranks[j]} enriched patterns; these decodes are precomputed in float64 for the selected held-out images. Basis coefficients for the partial interactive decoders are stored to 12 decimal places.`}
function modes(){const j=+el('modeknob').value,gallery=el('modegallery');gallery.replaceChildren();const raw=data.means[j],rawnorm=Math.hypot(...raw),maps=[raw.map(v=>v/rawnorm),...data.bases[j].slice(0,5)],ss=data.singular_values[j],total=ss.reduce((s,v)=>s+v*v,0);const lim=Math.max(...maps.flat().map(Math.abs));maps.forEach((map,m)=>{const box=document.createElement('div');box.className='map';const heading=document.createElement('h3');heading.textContent=m===0?'Mean true direction':`SVD pattern ${m}`;const canvas=document.createElement('canvas');canvas.id=`mode${m}`;canvas.width=canvas.height=280;const caption=document.createElement('p');caption.textContent=m===0?`Raw mean length ${rawnorm.toFixed(3)}`:`${(100*ss[m-1]**2/total).toFixed(2)}% of training squared-vector energy`;box.append(heading,canvas,caption);gallery.append(box);paint(canvas.id,map,lim)});el('modescale').textContent=`Shared unit-map scale: blue −${lim.toFixed(3)} to red +${lim.toFixed(3)}; no clipping. Pattern percentages refer to uncentered SVD energy, not intrinsic dimension.`}
function cloud(){const j=+el('cloudknob').value,ctx=el('cloud').getContext('2d'),points=data.scatter[j],xs=points.map(p=>p[0]),ys=points.map(p=>p[1]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);ctx.clearRect(0,0,1000,430);ctx.fillStyle='#23313a';ctx.font='14px system-ui';ctx.fillText('Centered PCA score 2 (raw derivative units)',20,25);ctx.strokeStyle='#7b898e';ctx.beginPath();ctx.moveTo(80,45);ctx.lineTo(80,375);ctx.lineTo(970,375);ctx.stroke();points.forEach((p,i)=>{const t=(data.scatter_settings[i][4]+10)/45;ctx.fillStyle=`hsl(${240-230*t},68%,47%)`;ctx.globalAlpha=.5;ctx.beginPath();ctx.arc(85+(p[0]-xmin)/(xmax-xmin)*875,370-(p[1]-ymin)/(ymax-ymin)*310,3,0,2*Math.PI);ctx.fill()});ctx.globalAlpha=1;ctx.fillStyle='#23313a';ctx.fillText(`Score 1: ${xmin.toFixed(2)} → ${xmax.toFixed(2)} (raw derivative units)`,290,412);ctx.fillText(ymax.toFixed(2),25,65);ctx.fillText(ymin.toFixed(2),25,370);el('cloudcaption').textContent=`First two axes retain ${(100*data.scatter_variance[j]).toFixed(1)}% of training variation. Color: blue = lean −10°, green ≈ 12.5°, red = +35°. Second-axis range ${ymin.toFixed(2)} to ${ymax.toFixed(2)}. Dots can overlap; this is a 2D projection of 784-pixel direction vectors.`}
el('modeknob').addEventListener('input',modes);el('cloudknob').addEventListener('input',cloud);modes();cloud();
for(const id of ['knob','sample','method','rank'])el(id).addEventListener('input',update);update();
</script></body></html>'''
    for key,val in {'CARDS':cards,'OPTIONS':scatters,'COMPARISONS':comparisons,'PREDICTIONS':predictions,'SENSITIVITY':sensitivity,'ENDPOINT':endpoint,'COVERAGE':coverage_rows,'DATA':json.dumps(payload,separators=(',',':'))}.items():html=html.replace('__'+key+'__',val)
    (OUT/'index.html').write_text(html)

if __name__=='__main__':main()
