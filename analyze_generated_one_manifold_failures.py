"""Bounds, connectivity, tangent, and finite-motion diagnostics of learned 1s."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from scipy.ndimage import label
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.stats import qmc
from grey_ones import RANGES, KNOBS
from fit_generated_one_principal_manifold import OUT as MODEL_DIR, decode, basis, encode_pixels, project, score
from optimize_generated_one_joint_curves import evaluate
from analyze_generated_one_principal_curves import pixels

OUT=MODEL_DIR/'failures'
JOINT=MODEL_DIR.parent/'joint'
THRESHOLDS=[1e-6,1e-3,.01,.05]


def full(x,active):
    a=np.zeros((len(x),784));a[:,active]=x
    return a.reshape(-1,28,28)


def partition(x,y):
    clipped=np.clip(y,0,1);raw=(y-x)**2;remaining=(clipped-x)**2
    total=float(raw.sum());left=float(remaining.sum());outside=(y<0)|(y>1)
    classes={'empty':x==0,'partially_covered':(x>0)&(x<1),'full_ink':x==1}
    parts={name:dict(pixel_count=int(mask.sum()),raw_sse=float(raw[mask].sum()),
        clipped_sse=float(remaining[mask].sum()),clipped_error_percent=100*float(remaining[mask].sum())/left)
        for name,mask in classes.items()}
    return dict(raw=score(x,y),clipped=score(x,clipped),raw_sse=total,clipped_sse=left,
        clipping_removed_error_percent=100*(1-left/total),
        error_at_out_of_range_pixels_percent=100*float(raw[outside].sum())/total,
        negative_count=int(np.sum(y< -1e-6)),above_one_count=int(np.sum(y>1+1e-6)),
        maximum_negative_violation=float(max(0,-y.min())),maximum_above_one_violation=float(max(0,y.max()-1)),
        source_pixel_classes=parts,
        images_with_out_of_range_above_0p01_percent=100*float(np.mean(np.any((y<-.01)|(y>1.01),axis=1))))


def connectivity(x,y,active):
    xx=full(x,active);yy=full(y,active);clipped=np.clip(yy,0,1)
    true_ink=xx>1e-6;structure=np.ones((3,3),int)
    result={};cleaned_at_default=None
    raw_total=float(np.sum((yy-xx)**2));clip_total=float(np.sum((clipped-xx)**2))
    for threshold in THRESHOLDS:
        counters=dict(negative_pixels=0,above_one_pixels=0,bounds_connected_pixels=0,bounds_disconnected_pixels=0,
            foreground_pixels=0,foreground_disconnected_pixels=0,foreground_disconnected_mass=0.,
            extra_ink_connected_pixels=0,extra_ink_disconnected_pixels=0,
            extra_ink_connected_sse=0.,extra_ink_disconnected_sse=0.,
            bounds_connected_raw_sse=0.,bounds_disconnected_raw_sse=0.,
            bounds_connected_clipping_removed_sse=0.,bounds_disconnected_clipping_removed_sse=0.,
            images_with_disconnected_ink=0)
        bound_counts=[];island_counts=[];clean=np.zeros_like(clipped)
        for i,(truth,raw,pred,ref) in enumerate(zip(xx,yy,clipped,true_ink)):
            anomaly=(raw < -threshold)|(raw>1+threshold)
            lab,_=label(ref|anomaly,structure)
            connected=anomaly & np.isin(lab,np.unique(lab[ref]));disconnected=anomaly & ~connected
            raw_error=(raw-truth)**2;clip_error=(pred-truth)**2
            fg=pred>threshold;component,n=label(fg,structure)
            if n:
                overlap=np.bincount(component.ravel(),weights=truth.ravel(),minlength=n+1);overlap[0]=-1
                main=int(np.argmax(overlap));main_mask=component==main
                off=fg & ~main_mask
                clean[i]=np.where(main_mask,pred,0)
            else:main_mask=np.zeros_like(fg);off=fg
            extra=fg & (truth==0)
            counters['negative_pixels']+=int(np.sum(raw < -threshold))
            counters['above_one_pixels']+=int(np.sum(raw>1+threshold))
            counters['bounds_connected_pixels']+=int(connected.sum());counters['bounds_disconnected_pixels']+=int(disconnected.sum())
            counters['foreground_pixels']+=int(fg.sum());counters['foreground_disconnected_pixels']+=int(off.sum())
            counters['foreground_disconnected_mass']+=float(pred[off].sum())
            counters['extra_ink_connected_pixels']+=int(np.sum(extra & main_mask));counters['extra_ink_disconnected_pixels']+=int(np.sum(extra & off))
            counters['extra_ink_connected_sse']+=float(clip_error[extra & main_mask].sum());counters['extra_ink_disconnected_sse']+=float(clip_error[extra & off].sum())
            counters['images_with_disconnected_ink']+=int(np.any(off))
            for name,mask in [('connected',connected),('disconnected',disconnected)]:
                counters[f'bounds_{name}_raw_sse']+=float(raw_error[mask].sum())
                counters[f'bounds_{name}_clipping_removed_sse']+=float((raw_error-clip_error)[mask].sum())
            bound_counts.append(int(anomaly.sum()));island_counts.append(int(off.sum()))
        counters['bounds_connected_count_percent']=100*counters['bounds_connected_pixels']/max(1,counters['bounds_connected_pixels']+counters['bounds_disconnected_pixels'])
        counters['bounds_connected_raw_error_percent']=100*counters['bounds_connected_raw_sse']/raw_total
        counters['bounds_disconnected_raw_error_percent']=100*counters['bounds_disconnected_raw_sse']/raw_total
        counters['disconnected_ghost_share_of_clipped_error_percent']=100*counters['extra_ink_disconnected_sse']/clip_total
        counters['connected_ghost_share_of_clipped_error_percent']=100*counters['extra_ink_connected_sse']/clip_total
        counters['mean_bounds_pixels_per_image']=float(np.mean(bound_counts));counters['p95_bounds_pixels_per_image']=float(np.quantile(bound_counts,.95))
        counters['mean_disconnected_foreground_pixels_per_image']=float(np.mean(island_counts))
        counters['images_with_disconnected_ink_percent']=100*counters['images_with_disconnected_ink']/len(x)
        counters['clip_threshold_and_keep_main_component']=score(x,clean.reshape(len(x),784)[:,active])
        result[str(threshold)]=counters
        if threshold==1e-3:cleaned_at_default=clean.reshape(len(x),784)[:,active]
    return result,cleaned_at_default


def effort():
    jr=json.loads((JOINT/'results.json').read_text());mr=json.loads((MODEL_DIR/'results.json').read_text())
    curve_path=['lean_seeded_65','joint_129','joint_257','joint_513']
    nh={n:len(json.loads((JOINT/f'{n}_history.json').read_text())) for n in curve_path}
    gh={n:len(json.loads((MODEL_DIR/f'{n}_history.json').read_text())) for n in ['uniform','guided','fine']}
    def runtime(filename,prefix=None):
        p=Path('/tmp')/filename
        if not p.exists():return None
        lines=p.read_text().splitlines()
        values=[float(m.group(1)) for line in lines if prefix is None or line.startswith(prefix)
                if (m:=re.search(r'elapsed ([\d.]+)s',line))]
        return max(values) if values else None
    times={n:runtime(f'generated_one_joint{size}.log',prefix) for n,size,prefix in
           [('lean_seeded_65','65','lean_seeded:'),('joint_129','129',None),('joint_257','257',None),('joint_513_first','513',None),('joint_513_polish','_polish',None)]}
    mt={n:runtime(f'principal_manifold_{n}.log') for n in gh}
    target=jr['candidates']['joint_513']['metrics']['explained_percent']
    h=json.loads((MODEL_DIR/'guided_history.json').read_text())
    cross=next(i+1 for i,v in enumerate(h) if v['metrics']['explained_percent']>=target)
    return dict(curve_path_iterations=nh,curve_winning_path_total_iterations=sum(nh.values()),
        grid_iterations=gh,grid_winning_path_total_iterations=gh['guided']+gh['fine'],
        curve_logged_optimization_seconds=times,grid_logged_optimization_seconds=mt,
        curve_winning_path_logged_seconds=sum(v for v in times.values() if v is not None),
        grid_winning_path_logged_seconds=sum(mt[n] for n in ['guided','fine'] if mt[n] is not None),
        guided_cycles_to_exceed_best_five_curves=cross,grid_nodes=mr['candidates']['fine']['node_count'],
        curve_nodes=5*513,notes='Logged loop times omit sampling, encoding checks, initialization overhead, and discarded search runs. One grid cycle includes a sparse node solve with many conjugate-gradient iterations; curve and grid cycles are not equivalent units.')


def pixel_study():
    OUT.mkdir(exist_ok=True);m=np.load(MODEL_DIR/'fine.npz');d=np.load(MODEL_DIR/'dense_uniform_examples.npz')
    old=np.load(JOINT/'joint_513_dense.npz');assert np.allclose(old['knobs'],d['knobs'])
    active=d['active'];x=d['images'];preds={'manifold':d['reconstructed'],'curves':old['reconstructed']}
    r=dict(count=len(x),connectivity='8-neighbor pixels. Bounds anomaly connection uses reference ink (>1e-6) union anomaly pixels. Clipped foreground uses a stated positive threshold; main component is the one with greatest overlap with true ink.',
           thresholds=THRESHOLDS,effort=effort(),models={})
    cleaned={}
    for name,y in preds.items():
        stats=partition(x,y);conn,clean=connectivity(x,y,active);stats['connectivity']=conn
        r['models'][name]=stats;cleaned[name]=clean
        print(name,json.dumps({k:stats[k] for k in ['clipping_removed_error_percent','raw','clipped']}),flush=True)
    nodes_clipped=decode(d['coordinates'],np.clip(m['nodes'],0,1),m['shape'])
    r['node_clipping_ablation']=dict(metrics=score(x,nodes_clipped),meaning='Clamp stored full-image grid nodes to [0,1], then interpolate at unchanged fitted coordinates. No refit or reprojection. Convex interpolation guarantees bounds, but it can change in-range pixels through lost cancellation.')
    # Same clipping comparison on every covering point.
    ref=m['reference_images'];mp=decode(m['coordinates'],m['nodes'],m['shape']);c=np.load(JOINT/'joint_513.npz')
    cp=(c['mean']+sum(evaluate(v,c['coordinates'][:,k]) for k,v in enumerate(c['curves'])))[:,active]
    r['covering_cloud']={n:partition(ref,y) for n,y in [('manifold',mp),('curves',cp)]}
    err=np.linalg.norm(np.clip(preds['manifold'],0,1)-x,axis=1)
    benefit=np.sum((preds['manifold']-x)**2-(np.clip(preds['manifold'],0,1)-x)**2,axis=1)
    ghost=(np.clip(preds['manifold'],0,1)**2*(x==0)).sum(1)
    ii=[int(np.argsort(err)[len(err)//2]),int(np.argmax(benefit)),int(np.argmax(err)),int(np.argmax(ghost))]+list(range(12))
    labels=['Typical error after clipping','Largest clipping benefit','Largest error after clipping','Largest extra-ink error']+[f'Uniform point {i+1}' for i in range(12)]
    np.savez_compressed(OUT/'pixel_examples.npz',indices=ii,labels=labels,knobs=d['knobs'][ii],images=x[ii],
        manifold=preds['manifold'][ii],curves=preds['curves'][ii],manifold_cleaned=cleaned['manifold'][ii],
        curves_cleaned=cleaned['curves'][ii],nodes_clipped=nodes_clipped[ii],active=active)
    # Is error concentrated at particular settings or positions inside interpolation cells?
    for axis,name in enumerate(KNOBS):
        u=(d['knobs'][:,axis]-RANGES[axis,0])/np.ptp(RANGES[axis])
        index=np.minimum((u*8).astype(int),7)
        r.setdefault('clipped_error_by_knob',{})[name]=[
            dict(count=int(np.sum(index==k)),rms_image_L2=float(np.sqrt(np.mean(err[index==k]**2)))) for k in range(8)]
    phase=np.mod(d['coordinates']*(m['shape']-1),1);phase_index=np.minimum((phase*5).astype(int),4)
    r['clipped_error_by_cell_phase']={name:[dict(count=int(np.sum(phase_index[:,axis]==k)),rms_image_L2=float(np.sqrt(np.mean(err[phase_index[:,axis]==k]**2)))) for k in range(5)] for axis,name in enumerate(KNOBS)}
    r['notes']='Clipping fixes scalar bounds only. A bounded connected image may still have incorrect width, lean, edge coverage, or cap geometry. Bounds violations are not the same as total distance from the renderer family.'
    (OUT/'pixels.json').write_text(json.dumps(r,indent=2)+'\n')
    return r


def projected_images(x,m):
    pred,u=encode_pixels(x,m['nodes'],m['shape'],m['reference_images'],m['coordinates'],starts=2,steps=20)
    difficult=np.flatnonzero(np.linalg.norm(x-pred,axis=1)>.35)
    if len(difficult):
        pp,uu=encode_pixels(x[difficult],m['nodes'],m['shape'],m['reference_images'],m['coordinates'],starts=8,steps=60)
        keep=np.sum((x[difficult]-pp)**2,axis=1)<np.sum((x[difficult]-pred[difficult])**2,axis=1)
        pred[difficult[keep]]=pp[keep];u[difficult[keep]]=uu[keep]
    return pred,u


def vector_metrics(actual,predicted):
    error=predicted-actual;true2=np.sum(actual**2,axis=1);pred2=np.sum(predicted**2,axis=1)
    cosine=np.sum(actual*predicted,axis=1)/np.maximum(np.sqrt(true2*pred2),1e-20)
    gain=np.sum(actual*predicted,axis=1)/np.maximum(true2,1e-20)
    parallel=(gain-1)[:,None]*actual;orthogonal=error-parallel
    return dict(relative_L2_error=float(np.sqrt(np.sum(error**2)/np.sum(actual**2))),
        median_cosine=float(np.median(cosine)),p10_cosine=float(np.quantile(cosine,.1)),
        median_change_gain=float(np.median(gain)),p10_change_gain=float(np.quantile(gain,.1)),
        p90_change_gain=float(np.quantile(gain,.9)),pooled_change_gain=float(np.sum(actual*predicted)/np.sum(actual**2)),
        error_parallel_to_actual_change_percent=100*float(np.sum(parallel**2)/np.sum(error**2)),
        error_orthogonal_to_actual_change_percent=100*float(np.sum(orthogonal**2)/np.sum(error**2)),
        rms_pixel_error=float(np.sqrt(np.sum(error**2)/(len(actual)*784))),
        error_on_unchanged_pixels_percent=100*float(np.sum(error**2*(np.abs(actual)<1e-6))/np.sum(error**2)))


def motion_study():
    m=np.load(MODEL_DIR/'fine.npz');active=m['active'];n=128
    un=.15+.6*qmc.Sobol(5,scramble=True,seed=948).random_base2(7)
    p=RANGES[:,0]+un*np.ptp(RANGES,axis=1);x=pixels(p)[:,active]
    base,u=projected_images(x,m);jf=np.stack([basis(u,m['shape'],k)@m['nodes'] for k in range(5)],axis=2)
    h=.001;stencils=[]
    for k in range(5):
        minus=p.copy();plus=p.copy();minus[:,k]-=h*np.ptp(RANGES[k]);plus[:,k]+=h*np.ptp(RANGES[k]);stencils.extend([minus,plus])
    images=pixels(np.concatenate(stencils))[:,active].reshape(5,2,n,-1)
    jt=np.transpose((images[:,1]-images[:,0])/(2*h),(1,2,0))
    pinv=np.linalg.pinv(jf,rcond=1e-8);g=pinv@jt
    result=dict(points=n,base_range_fractions=[.15,.75],seed=948,true_tangent_difference_step=h,
        meanings='True tangents use central finite differences of rendered images. Learned tangents use exact within-cell decoder derivatives. Calibration G is an oracle local diagnostic, not a learned physical knob controller.',
        median_coordinate_calibration=g.mean(axis=0).tolist(),median_abs_coordinate_calibration=np.median(abs(g),axis=0).tolist(),
        local_tangents={},finite_moves={})
    for clipped in [False,True]:
        a=jf*((base>0)&(base<1))[:,:,None] if clipped else jf
        gg=np.linalg.pinv(a,rcond=1e-8)@jt;fit=a@gg
        diag=np.sum(a*jt,axis=1)/np.maximum(np.sum(a*a,axis=1),1e-20)
        naive=a*diag[:,None,:]
        residual=(np.clip(base,0,1) if clipped else base)-x
        legal=jt@(np.linalg.pinv(jt,rcond=1e-8)@residual[:,:,None])
        result['local_tangents']['clipped' if clipped else 'raw']=dict(
            per_knob={name:dict(best_single_corresponding_axis=vector_metrics(jt[:,:,k],naive[:,:,k]),
                        best_combination_of_five_axes=vector_metrics(jt[:,:,k],fit[:,:,k])) for k,name in enumerate(KNOBS)},
            baseline_reconstruction_error_explained_by_true_tangent_percent=100*float(np.sum(legal**2)/np.sum(residual**2)),
            baseline_reconstruction_error_orthogonal_to_true_tangent_percent=100*float(np.sum((residual-legal[:,:,0])**2)/np.sum(residual**2)))
    examples=[]
    for step in [.025,.1]:
        pp=[];uu=[];bounded=[]
        for k in range(5):
            q=p.copy();q[:,k]+=step*np.ptp(RANGES[k]);pp.append(q)
            desired=u+step*g[:,:,k];bounded.append(np.any((desired<0)|(desired>1),axis=1));uu.append(np.clip(desired,0,1))
        truth=pixels(np.concatenate(pp))[:,active].reshape(5,n,-1)
        transported=decode(np.concatenate(uu),m['nodes'],m['shape']).reshape(5,n,-1)
        projected,positions=projected_images(truth.reshape(5*n,-1),m);projected=projected.reshape(5,n,-1)
        per={}
        for k,name in enumerate(KNOBS):
            v=truth[k]-x;rv=transported[k]-base;cv=np.clip(transported[k],0,1)-np.clip(base,0,1)
            per[name]=dict(physical_step=float(step*np.ptp(RANGES[k])),coordinate_box_clipped_cases=int(np.sum(bounded[k])),
                calibrated_local_direction_raw=vector_metrics(v,rv),calibrated_local_direction_clipped=vector_metrics(v,cv),
                separately_reconstructed_endpoints_raw=vector_metrics(v,projected[k]-base),
                separately_reconstructed_endpoints_clipped=vector_metrics(v,np.clip(projected[k],0,1)-np.clip(base,0,1)))
            # Typical example per knob, plus the worst lean case for a larger move.
            ids=[int(np.argsort(np.linalg.norm(cv-v,axis=1))[n//2])]
            if name=='lean' and step==.1:ids.append(int(np.argmax(np.linalg.norm(cv-v,axis=1))))
            for i in ids:
                examples.append(dict(knob=name,step=step,physical_step=step*np.ptp(RANGES[k]),seed_index=i,
                    label=f'{name}: {step*100:g}% range, '+('worst lean example' if len(ids)>1 and i==ids[-1] else 'typical error'),
                    source=x[i],true_after=truth[k,i],learned_before=base[i],learned_after=transported[k,i],
                    endpoint_reconstructed=projected[k,i],knobs=p[i],coordinates=u[i],calibration=g[i,:,k]))
        result['finite_moves'][str(step)]=per
        print('Motion step',step,json.dumps({k:v['calibrated_local_direction_clipped']['relative_L2_error'] for k,v in per.items()}),flush=True)
    result['interpretation_limits']='Local calibration uses the true tangent at the starting image only. Larger transported moves also test curvature and chart variation. Independently reconstructed endpoints are a best-case representation diagnostic; they are not a prediction of the new image from the knob alone. Tangent-normal residuals can include curvature, not only invalid pixels.'
    # Correct label: store both signed median and absolute median, not an average.
    result['median_coordinate_calibration']=np.median(g,axis=0).tolist()
    (OUT/'motion.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(OUT/'motion_examples.npz',active=active,**{key:np.array([e[key] for e in examples]) for key in examples[0]})
    return result


def family_distance_study():
    """Use the generator solely as a validity diagnostic, never as a decoder."""
    m=np.load(MODEL_DIR/'fine.npz');active=m['active']
    un=.15+.6*qmc.Sobol(5,scramble=True,seed=948).random_base2(5)
    p=RANGES[:,0]+un*np.ptp(RANGES,axis=1);x=pixels(p)[:,active];pred,u=projected_images(x,m)
    examples=np.load(OUT/'pixel_examples.npz')
    p=np.r_[p,examples['knobs'][:4]];x=np.r_[x,examples['images'][:4]]
    target=np.clip(np.r_[pred,examples['manifold'][:4]],0,1)
    original=(p-RANGES[:,0])/np.ptp(RANGES,axis=1);h=2e-4
    nearest=[];settings=[];records=[]
    def render_fraction(q):return pixels(RANGES[:,0]+np.atleast_2d(q)*np.ptp(RANGES,axis=1))[:,active]
    def jac(q):
        low=np.tile(q,(5,1));high=low.copy()
        for k in range(5):low[k,k]=max(0,q[k]-h);high[k,k]=min(1,q[k]+h)
        xx=render_fraction(np.r_[low,high]);denom=np.diag(high-low)
        return ((xx[5:]-xx[:5])/denom[:,None]).T
    for i,(start,y,source) in enumerate(zip(original,target,x)):
        delta=np.linalg.lstsq(jac(start),y-source,rcond=1e-8)[0]
        best_q=start.copy();best_image=source.copy();loss=float(np.sum((source-y)**2));evals=0
        for guess in [start,np.clip(start+delta,0,1)]:
            opt=least_squares(lambda q:render_fraction(q)[0]-y,guess,jac=jac,bounds=(np.zeros(5),np.ones(5)),
                              max_nfev=80,ftol=1e-9,xtol=1e-9,gtol=1e-7)
            candidate=render_fraction(opt.x)[0];cost=float(np.sum((candidate-y)**2));evals+=opt.nfev
            if cost<loss:loss=cost;best_q=opt.x;best_image=candidate
        nearest.append(best_image);settings.append(best_q)
        records.append(dict(index=i,group='32 interior points' if i<32 else 'four displayed error cases',
            original_image_error_L2=float(np.linalg.norm(y-source)),distance_to_best_valid_image_L2=float(np.sqrt(loss)),
            normalized_knob_shift=(best_q-start).tolist(),function_evaluations=evals))
    nearest=np.array(nearest);settings=np.array(settings);r=target-nearest;drift=nearest-x
    groups={}
    for name,sl in [('interior_32',slice(0,32)),('displayed_error_cases_4',slice(32,None))]:
        total=float(np.sum((target[sl]-x[sl])**2))
        groups[name]=dict(count=len(x[sl]),original_clipped_sse=total,
            remaining_distance_to_valid_family_sse=float(np.sum(r[sl]**2)),
            remaining_error_after_best_valid_knob_adjustment_percent=100*float(np.sum(r[sl]**2))/total,
            valid_image_drift_sse=float(np.sum(drift[sl]**2)),cross_term=2*float(np.sum(r[sl]*drift[sl])),
            mean_absolute_normalized_knob_shift=np.mean(abs(settings[sl]-original[sl]),axis=0).tolist())
    result=dict(groups=groups,images=records,
        method='Bounded nonlinear least squares fits the original renderer to each clipped reconstruction. Two starts: actual source knobs and a linearized true-Jacobian adjustment. This uses the renderer only to diagnose validity; no learned model or decoder changes.',
        limit='Local multistart search provides an upper bound on exact nearest-family distance, not a global minimum certificate. The original source is always retained as a feasible candidate. Only 32 interior points and four selected error examples are measured.')
    (OUT/'family_distance.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(OUT/'family_distance_examples.npz',images=x,target=target,nearest_valid=nearest,
        actual_knobs=p,nearest_valid_knobs=RANGES[:,0]+settings*np.ptp(RANGES,axis=1),active=active)
    print('Distance to valid family:',json.dumps(groups),flush=True)
    return result


def projection_search_study():
    """Test whether selected large failures are just local projection minima."""
    m=np.load(MODEL_DIR/'fine.npz');e=np.load(OUT/'pixel_examples.npz');x=e['images'][:4]
    _,neighbor=cKDTree(m['reference_images']).query(x,k=32)
    physical=(e['knobs'][:4]-RANGES[:,0])/np.ptp(RANGES,axis=1)
    random=qmc.Sobol(5,scramble=True,seed=950).random_base2(5)
    starts=np.concatenate([m['coordinates'][neighbor],physical[:,None,:],np.broadcast_to(random,(4,32,5))],axis=1)
    xx=np.repeat(x,starts.shape[1],axis=0)
    pred,u,history=project(xx,m['nodes'],m['shape'],starts.reshape(-1,5),steps=100)
    errors=np.linalg.norm(xx-pred,axis=1).reshape(4,-1);best=errors.argmin(1);records=[]
    for i,k in enumerate(best):
        records.append(dict(example=str(e['labels'][i]),original_raw_L2=float(np.linalg.norm(x[i]-e['manifold'][i])),
            wider_search_raw_L2=float(errors[i,k]),best_start='nearest fitted image' if k<32 else ('true normalized knobs' if k==32 else 'Sobol coordinate-box start'),
            best_coordinates=u.reshape(4,-1,5)[i,k].tolist()))
    result=dict(images=records,starts_per_image=65,steps_per_start=100,
        notes='Focused diagnostic on four selected cases. No model or stored projection result is replaced. The true-knob start is diagnostic initialization; the other 64 starts use pixels or the coordinate box. This still does not certify global nearest points.')
    (OUT/'projection_search.json').write_text(json.dumps(result,indent=2)+'\n');print('Wider projection search:',json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['all','pixels','motion','family','projection'],default='all');args=parser.parse_args()
    if args.phase in ['all','pixels']:pixel_study()
    if args.phase in ['all','motion']:motion_study()
    if args.phase in ['all','family']:family_distance_study()
    if args.phase in ['all','projection']:projection_search_study()
