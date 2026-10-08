"""Lean directions: width transport, separate-axis blending, and joint states.

Reproduce from the repository root:
    .venv/bin/python experiments/lean_transport_20261006/run.py
"""
from pathlib import Path
import csv
import itertools
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from grey_ones import render, RANGES, SUB, N_PIX
from lean_axis import geometry_from_image

OUT = Path(__file__).resolve().parent / 'results'
BASE = np.array([14.5, 14.5, 19.75, 3.2, 0.0])
YLO = np.arange(N_PIX * SUB) / SUB
Y = YLO + .5 / SUB
C = np.arange(N_PIX)


def edges(p):
    """Transported continuous edge locations and vertical coverage weights."""
    p = np.atleast_2d(np.asarray(p, float))
    cx, cy, h, w, theta = [p[:, k, None] for k in range(5)]
    a = np.clip((np.minimum(YLO + 1/SUB, cy+h/2) -
                 np.maximum(YLO, cy-h/2))*SUB, 0, 1)
    d = cy-Y
    center = cx+np.tan(np.deg2rad(theta))*d
    return center-w/2, center+w/2, a, d


def edge_template(b):
    """One universal ramp: ink left of a moving boundary in each column."""
    return np.clip(b[:, :, None]-C, 0, 1)


def field(p, side='center', degrees=True):
    """Exact lean tangent; center is the average of one-sided limits at events.

    'right' / 'left' mean increasing / decreasing lean, not stroke sides.
    Strict inside tests are used except for exact integer edge coincidences.
    """
    p = np.atleast_2d(np.asarray(p, float))
    chunks = []
    for q in np.array_split(p, max(1, (len(p)+31)//32)):
        l, r, a, d = edges(q)
        def inside(b):
            t = b[:, :, None]-C
            value = ((t > 0) & (t < 1)).astype(float)
            at_lo, at_hi = t == 0, t == 1
            if side == 'center':
                value += .5*(at_lo | at_hi)
            else:
                v = d[:, :, None]*(1 if side == 'right' else -1)
                value += (at_lo & (v > 0)) | (at_hi & (v < 0))
            return value
        v = (inside(r)-inside(l))*(a*d)[:, :, None]
        v = v.reshape(len(q), N_PIX, SUB, N_PIX).mean(2).reshape(len(q), -1)
        if degrees:
            v *= (np.pi/180/np.cos(np.deg2rad(q[:,4]))**2)[:,None]
        chunks.append(v)
    return np.concatenate(chunks)


def coverage(p):
    p = np.atleast_2d(np.asarray(p, float))
    parts=[]
    for q in np.array_split(p, max(1,(len(p)+31)//32)):
        l,r,a,_=edges(q)
        # A difference of translated instances of ONE ramp template.
        v = (edge_template(r)-edge_template(l))*a[:,:,None]
        parts.append(v.reshape(len(q),N_PIX,SUB,N_PIX).mean(2).reshape(len(q),-1))
    return np.concatenate(parts)


def move(p, target_angles, starting_images=None):
    """Analytic integral of the changing tangent, without calling render."""
    p=np.atleast_2d(np.asarray(p,float))
    target=p.copy(); target[:,4]=np.broadcast_to(target_angles,(len(p),))
    if starting_images is None:
        starting_images=coverage(p)
    return np.asarray(starting_images).reshape(len(p),-1)+coverage(target)-coverage(p)


def axis_blend(p, average=False, axes=(0,1,2,3)):
    """Separate-axis arrows are queried at the SAME current lean angle.

    This gives additive blending the advantage of perfect single-axis data.
    """
    p=np.atleast_2d(p)
    b=np.tile(BASE,(len(p),1)); b[:,4]=p[:,4]
    v0=field(b)
    vals=[]
    for k in axes:
        q=b.copy(); q[:,k]=p[:,k]
        vals.append(field(q))
    return np.mean(vals,axis=0) if average else sum(vals)-(len(axes)-1)*v0


def score(pred, truth):
    denom=np.linalg.norm(truth,axis=1)
    rel=np.linalg.norm(pred-truth,axis=1)/denom
    pnorm=np.linalg.norm(pred,axis=1)
    defined=(pnorm>1e-12)&(denom>1e-12)
    cos=np.sum(pred[defined]*truth[defined],axis=1)/(pnorm[defined]*denom[defined])
    return {'mean_relative_error':float(rel.mean()), 'p95_relative_error':float(np.quantile(rel,.95)),
            'max_relative_error':float(rel.max()),
            'mean_angle_degrees':float(np.rad2deg(np.arccos(np.clip(cos,-1,1))).mean()) if defined.any() else None,
            'undefined_angle_count':int((~defined).sum())}


def image_score(pred, truth):
    e=np.abs(pred-truth)
    return {'mean_l2_error':float(np.linalg.norm(e,axis=1).mean()),
            'max_l2_error':float(np.linalg.norm(e,axis=1).max()),
            'max_absolute_pixel_error':float(e.max())}


def event_integral(p, angles):
    """Integrate dF/du exactly on every interval between edge/pixel events."""
    p=np.asarray(p,float)
    uframes=np.tan(np.deg2rad(angles)); lo,hi=uframes.min(),uframes.max()
    upright=p.copy(); upright[4]=0
    l,r,a,d=edges(upright)
    active=(a[0]>0)&(np.abs(d[0])>1e-14)
    ev=np.concatenate([((C[:,None]-b[0,active])/d[0,active]).ravel()
                       for b in (l,r)])
    ev=ev[(ev>lo)&(ev<hi)]
    stops=np.unique(np.r_[uframes,ev])
    mids=(stops[:-1]+stops[1:])/2
    q=np.tile(p,(len(mids),1)); q[:,4]=np.rad2deg(np.arctan(mids))
    arrows=field(q,degrees=False)
    start=p.copy(); start[4]=angles[0]
    path=np.vstack([render(start).reshape(1,-1).astype(float),
         render(start).ravel().astype(float)+np.cumsum(arrows*np.diff(stops)[:,None],axis=0)])
    return path[np.searchsorted(stops,uframes)],len(stops)-1


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261006)
    result={'baseline':BASE.tolist(),'source':'grey_ones.render',
            'scope':'Controlled straight generated ones; all 784 pixels; float32 source rendering',
            'field_units':'coverage change per degree; unnormalized magnitude retained',
            'construction':'Analytic edge transport, using known renderer geometry; no learned model or per-state lookup table'}
    # Fine width-only slice, plus explicit pixel-event and outside-box probes.
    widths=np.unique(np.r_[np.linspace(1.8,4.6,1401),[1.,2.,3.,3-1e-9,3+1e-9]])
    wp=np.tile(BASE,(len(widths),1)); wp[:,3]=widths
    truth=field(wp); v0=field(BASE)[0]
    scale=(truth@v0)/(v0@v0)
    result['width_slice']={'n':len(wp),'range':[1.8,4.6],
        'extra_outside_box_width':1.0,
        'frozen_base_arrow':score(np.tile(v0,(len(wp),1))[widths>=1.8],truth[widths>=1.8]),
        'best_scalar_times_base_arrow':score((scale[:,None]*v0)[widths>=1.8],truth[widths>=1.8]),
        'event_width_at_upright':3.0,
        'distinct_upright_open_interval_patterns':2}
    with (OUT/'width_sweep.csv').open('w',newline='') as f:
        wr=csv.writer(f); wr.writerow(['width_px','inside_default_box','speed_per_degree','cosine_to_base','best_scalar_relative_error'])
        for w,v,s in zip(widths,truth,scale):
            wr.writerow([w,1.8<=w<=4.6,np.linalg.norm(v),np.dot(v,v0)/(np.linalg.norm(v)*np.linalg.norm(v0)),np.linalg.norm(v-s*v0)/np.linalg.norm(v)])
    probes=np.tile(BASE,(2,1)); probes[:,3]=[3.1,3.3]
    collision={'widths':[3.1,3.3], 'upright_arrow_l2_difference':float(np.linalg.norm(np.diff(field(probes),axis=0)))}
    probes[:,4]=2
    collision['arrow_l2_difference_at_2_degrees']=float(np.linalg.norm(np.diff(field(probes),axis=0)))
    result['raster_arrow_information_loss']=collision
    # Never fit joint examples: every axis component comes from a one-knob slice.
    joint_cases={}
    for name,ks in [('width_height',[2,3]),('width_cx',[0,3]),('all_other_knobs_upright',[0,1,2,3]),('full_five_knob_box',[0,1,2,3,4])]:
        p=np.tile(BASE,(512,1))
        for k in ks:p[:,k]=rng.uniform(*RANGES[k],len(p))
        exact=field(p)
        joint_cases[name]={'count':len(p),'add_separate_axis_changes':score(axis_blend(p,axes=[k for k in ks if k<4]),exact),
                          'average_separate_axis_arrows':score(axis_blend(p,True,axes=[k for k in ks if k<4]),exact)}
        if name=='full_five_knob_box':
            np.savez_compressed(OUT/'joint_states.npz',settings=p,truth=exact,additive=axis_blend(p),average=axis_blend(p,True))
    result['independent_axis_combinations']=joint_cases
    # Validate the universal edge rule on random, lattice, and near-event states.
    random=RANGES[:,0]+rng.random((1024,5))*np.diff(RANGES,axis=1).ravel()
    axes=[np.linspace(*r,3) for r in RANGES]
    lattice=np.array(list(itertools.product(*axes)))
    stress=[]
    for w in [1.8,3-1e-8,3,3+1e-8,4.6]:
        for theta in [-1e-8,0,1e-8]:
            p=BASE.copy();p[3]=w;p[4]=theta;stress.append(p)
    p=np.vstack([random,lattice,stress])
    actual=render(p).reshape(len(p),-1).astype(float)
    result['forward_transport_validation']={'random':len(random),'lattice':len(lattice),'near_events':len(stress),**image_score(coverage(p),actual)}
    angles=rng.uniform(*RANGES[4],len(p)); target=p.copy();target[:,4]=angles
    targets=render(target).reshape(len(p),-1).astype(float)
    pred=move(p,angles,actual)
    result['finite_lean_moves_known_settings']={'n':len(p),**image_score(pred,targets)}
    assert np.abs(pred-targets).max()<2e-7
    # A real image is sufficient: recover geometry, then transport edges.
    recovered=np.array([[g.cx,g.cy,g.height,g.width,g.lean]
        for g in (geometry_from_image(im.reshape(28,28)) for im in actual)])
    recovered_move=move(recovered,angles,actual)
    result['finite_lean_moves_image_only']={'n':len(p),**image_score(recovered_move,targets),
        'max_absolute_recovered_knob_errors':np.abs(recovered-p).max(axis=0).tolist()}
    assert np.abs(recovered_move-targets).max()<1e-5
    # Learn independent transformations in continuous geometry, not pixel arrows.
    # Image geometry recovery and the universal edge template are supplied assumptions.
    # Only one knob varies in each of these 95 training images.
    def descriptors(settings):
        z=np.asarray(settings,float).copy()
        z[:,4]=np.tan(np.deg2rad(z[:,4]))
        return z
    base_g=geometry_from_image(render(BASE)[0])
    base_read=np.array([base_g.cx,base_g.cy,base_g.height,base_g.width,base_g.lean])
    base_z=descriptors(base_read[None])[0]
    slopes=[];training_settings=[]
    for k in range(5):
        train=np.tile(BASE,(19,1))
        train[:,k]=np.linspace(RANGES[k,0],RANGES[k,1],21)[1:-1]
        training_settings.append(train.copy())
        train_images=render(train)
        read=np.array([[g.cx,g.cy,g.height,g.width,g.lean]
            for g in (geometry_from_image(im) for im in train_images)])
        dz=descriptors(train)[:,k]-descriptors(BASE[None])[0,k]
        response=descriptors(read)-base_z
        slopes.append(dz@response/(dz@dz))
    slopes=np.array(slopes)
    # Composition uses the inferred shifts before applying nonlinear pixel coverage.
    learned_z=base_z+(descriptors(p)-descriptors(BASE[None]))@slopes
    learned_p=learned_z.copy();learned_p[:,4]=np.rad2deg(np.arctan(learned_p[:,4]))
    learned_pred=move(learned_p,angles,actual)
    training_settings=np.vstack([BASE[None],*training_settings])
    learned_heldout=~np.any(np.all(np.isclose(p[:,None,:],training_settings[None,:,:],atol=1e-10,rtol=0),axis=2),axis=1)
    result['learn_separate_geometry_transforms']={
        'one_axis_training_images':95,'baseline_images':1,
        'stored_scalars':30,'descriptor_order':['cx','cy','height','width','tan(lean)'],
        'baseline_descriptor':base_z.tolist(),'transformation_coefficients':slopes.tolist(),
        'supplied_structure':'Image geometry recovery, tan(angle) coordinate, and continuous edge-to-pixel template',
        'heldout_count':int(learned_heldout.sum()),'excluded_training_overlaps':int((~learned_heldout).sum()),
        **image_score(learned_pred[learned_heldout],targets[learned_heldout]),
        'random_state_arrow_prediction':score(field(learned_p[:len(random)]),field(random))}
    assert np.abs(learned_pred-targets).max()<1e-5
    # One-sided derivatives at deliberate upright edge coincidences.
    eps=1e-3; boundary=BASE.copy();boundary[3]=3
    for side,sign in [('left',-1),('right',1)]:
        end=boundary.copy();end[4]+=sign*eps
        observed=(render(end).ravel().astype(float)-render(boundary).ravel())/(sign*eps)
        expected=field(boundary,side)[0]
        result['boundary_'+side]={'fd_step_degrees':eps,**score(expected[None],observed[None])}
    result['boundary_turn_degrees']=float(np.rad2deg(np.arccos(np.clip(
        np.dot(field(boundary,'left')[0],field(boundary,'right')[0])/
        (np.linalg.norm(field(boundary,'left'))*np.linalg.norm(field(boundary,'right'))),-1,1))))
    # Differencing at several scales: event smoothing versus float32 cancellation.
    fd={}
    for step in [.1,.025,.00625]:
        low=random[:256].copy();high=low.copy()
        low[:,4]-=step;high[:,4]+=step
        observed=(render(high).reshape(256,-1).astype(float)-render(low).reshape(256,-1))/(2*step)
        fd[str(step)]=score(field(random[:256]),observed)
    result['finite_difference_checks']=fd
    # Whole paths, including mixed geometry and fixed-vector controls.
    path_geometry=np.vstack([BASE, random[:5]])
    frames=np.linspace(*RANGES[4],181)
    paths=[]
    for i,g in enumerate(path_geometry):
        states=np.tile(g,(len(frames),1));states[:,4]=frames
        actual_path=render(states).reshape(len(frames),-1).astype(float)
        integrated,nsegments=event_integral(g,frames)
        fixed=actual_path[0]+(frames-frames[0])[:,None]*field(states[:1])
        paths.append({'geometry':g.tolist(),'frames':len(frames),'event_intervals':nsegments,
                      'transport_event_integral':image_score(integrated,actual_path),
                      'frozen_start_arrow':image_score(fixed,actual_path)})
        np.savez_compressed(OUT/f'path_{i}.npz',settings=states,actual=actual_path,integrated=integrated,frozen=fixed)
    result['paths']=paths
    assert max(x['transport_event_integral']['max_absolute_pixel_error'] for x in paths)<2e-7
    # Arrays for direct, lossless signed-pixel visual comparisons.
    figure_widths=[3.2,1.8,2.6,3.0,4.6]
    pfig=np.tile(BASE,(len(figure_widths),1));pfig[:,3]=figure_widths
    np.savez_compressed(OUT/'width_panels.npz',settings=pfig,images=render(pfig),arrows=field(pfig),
                        left=field(pfig,'left'),right=field(pfig,'right'))
    # Width responses at nonzero lean; variation is no longer just two plateaus.
    offzero=[]
    for theta in [0.,5.,12.5,30.]:
        q=wp.copy();q[:,4]=theta
        v=field(q); b=BASE.copy();b[4]=theta; vb=field(b)[0]
        s=(v@vb)/(vb@vb)
        offzero.append({'lean_degrees':theta,'best_scalar_times_base':score((s[:,None]*vb)[widths>=1.8],v[widths>=1.8])})
    result['width_slice_at_other_leans']=offzero
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
