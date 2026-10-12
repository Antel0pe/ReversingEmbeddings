"""Test direction-change history, calibrated counts, and a learned bilinear chart.

The matcher sees pixel vectors and their past values only. Known controls
generate test paths and evaluate associations, not predictions or matches.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/generated-one-mpl')
from pathlib import Path
import json
import gzip
import time
import numpy as np
from scipy.optimize import brentq,linear_sum_assignment
from scipy.stats import qmc
import analyze_generated_one_nearest_direction as previous
from analyze_generated_one_manifold_transport import jet,normalized
from grey_ones import KNOBS,render

OUT=Path(__file__).resolve().parent/'figures/generated_one_direction_history'
STEPS=[.01,.03,.1,.3]
HOPS=8
RADIUS=1e-5


def next_state(p,axis,sign,distance):
    """Local monotone single-control move, stopped at the endpoint bounds."""
    speed=np.linalg.norm(jet(p)[3][:,axis])
    limit=(previous.HIGH[axis]-p[axis] if sign>0 else p[axis]-previous.LOW[axis])*.999
    if limit<=0:return None
    root=previous.pixels(p)[0]
    def objective(amount):
        end=p.copy();end[axis]+=sign*amount
        return np.linalg.norm(previous.pixels(end)[0]-root)-distance
    hi=min(limit,1.1*distance/speed)
    while objective(hi)<0 and hi<limit:hi=min(limit,2*hi)
    if objective(hi)<0:return None
    amount=brentq(objective,0,hi,xtol=1e-13)
    end=p.copy();end[axis]+=sign*amount
    assert abs(np.linalg.norm(previous.pixels(end)[0]-root)-distance)<1e-10
    return end


def predict(history,trend):
    if not trend or len(history)<2:return history[-1]
    prediction=2*history[-1]-history[-2]
    return prediction/np.linalg.norm(prediction,axis=1)[:,None]


def choose(prediction,candidates,one_to_one):
    scores=np.abs(prediction@candidates.T)
    if one_to_one:
        row,col=linear_sum_assignment(-scores)
        selected=np.empty(5,int);selected[row]=col
    else:selected=scores.argmax(axis=1)
    return selected,scores


def record_match(common,mode,method,selected,scores,expected,labels):
    expected_index=np.array([np.flatnonzero(labels==i)[0] for i in expected])
    correct=labels[selected]==expected
    margins=[];unique=[]
    for i,index in enumerate(expected_index):
        margins.append(float(scores[i,index]-np.delete(scores[i],index).max()))
        unique.append(bool(correct[i] and (method.endswith('_joint') or margins[-1]>1e-10)))
    return dict(**common,mode=mode,method=method,count=5,correct=sum(unique),
                chosen_labels=labels[selected].tolist(),expected_labels=expected.tolist(),
                correct_flags=unique,margins=margins,
                duplicate_choices=5-len(np.unique(selected)))


def history_benchmark():
    with gzip.open(previous.OUT/'results.json.gz','rt') as f:old=json.load(f)
    all_starts=previous.starts()
    for name in ['additional_76','additional_128']:
        all_starts.append((name,np.array(next(s['settings'] for s in old['starts'] if s['name']==name))))
    del old
    methods={'nearest':(False,False),'trend':(True,False),
             'nearest_joint':(False,True),'trend_joint':(True,True)}
    rows=[];paths=[];started=time.monotonic();path_id=0;fit_count=0
    for name,start in all_starts:
        initial_fit=previous.local_fit(start,RADIUS,20261031)
        initial={'true':(normalized(jet(start)[3]).T,np.arange(5)),
                 'svd':(initial_fit['basis'],initial_fit['labels'])}
        for step in STEPS:
            for axis in range(5):
                for sign in [-1,1]:
                    path_id+=1;p=start.copy();states=[p.tolist()]
                    histories={kind:{method:[b.copy()] for method in methods} for kind,(b,_) in initial.items()}
                    truth_history={kind:[b.copy()] for kind,(b,_) in initial.items()}
                    for hop in range(1,HOPS+1):
                        end=next_state(p,axis,sign,step)
                        if end is None:break
                        fit=previous.local_fit(end,RADIUS,20261031+path_id*100+hop);fit_count+=1
                        frames={'true':(normalized(jet(end)[3]).T,np.arange(5)),
                                'svd':(fit['basis'],fit['labels'])}
                        states.append(end.tolist())
                        for kind,(candidate,labels) in frames.items():
                            expected=initial[kind][1]
                            common=dict(path=path_id,name=name,step=step,moved_knob=KNOBS[axis],
                                        sign=sign,hop=hop,field=kind)
                            for method,(trend,joint) in methods.items():
                                history=histories[kind][method]
                                prediction=predict(history,trend)
                                selected,scores=choose(prediction,candidate,joint)
                                rows.append(record_match(common,'rollout',method,selected,scores,expected,labels))
                                aligned=candidate[selected].copy()
                                signs=np.sign(np.sum(aligned*history[-1],axis=1));signs[signs==0]=1
                                history.append(aligned*signs[:,None])
                                if len(history)>2:history.pop(0)
                            # Diagnostic only: past associations are supplied correctly,
                            # to distinguish prediction quality from propagated mistakes.
                            history=truth_history[kind]
                            for method,trend in [('nearest',False),('trend',True)]:
                                selected,scores=choose(predict(history,trend),candidate,False)
                                rows.append(record_match(common,'correct_past_diagnostic',method,
                                                         selected,scores,expected,labels))
                            indices=[np.flatnonzero(labels==i)[0] for i in expected]
                            aligned=candidate[indices].copy()
                            signs=np.sign(np.sum(aligned*history[-1],axis=1));signs[signs==0]=1
                            history.append(aligned*signs[:,None])
                            if len(history)>2:history.pop(0)
                        p=end
                    paths.append(dict(path=path_id,name=name,step=step,moved_knob=KNOBS[axis],
                                      sign=sign,states=states,completed_hops=len(states)-1))
        print(f'History benchmark {name}: {fit_count} endpoint fits; {time.monotonic()-started:.1f}s',flush=True)
    aggregate=[]
    for field in ['true','svd']:
        for mode in ['rollout','correct_past_diagnostic']:
            for method in methods if mode=='rollout' else ['nearest','trend']:
                for step in STEPS:
                    for scope in ['after_bootstrap','hop8']:
                        subset=[r for r in rows if r['field']==field and r['mode']==mode and
                                r['method']==method and r['step']==step and
                                (r['hop']>=2 if scope=='after_bootstrap' else r['hop']==HOPS)]
                        aggregate.append(dict(field=field,mode=mode,method=method,step=step,scope=scope,
                                              count=sum(r['count'] for r in subset),
                                              correct=sum(r['correct'] for r in subset),
                                              duplicate_choices=sum(r['duplicate_choices'] for r in subset)))
    return dict(starts=[dict(name=n,settings=p.tolist()) for n,p in all_starts],
                local_radius=RADIUS,neighbors=previous.TRAIN,steps=STEPS,max_hops=HOPS,
                endpoint_sampling='independent seed at every hop; same samples for all matching methods',
                predictor='normalize(2*last_unit_vector-previous_unit_vector); equal image-distance hops',
                paths=paths,comparisons=rows,aggregate=aggregate)


def bilinear_control():
    """Fit four pixel arrays from four images; predict unseen joint settings."""
    train=np.array([[0,0],[1,0],[0,1],[1,1]],float)
    def settings(uv):
        return np.column_stack([np.full(len(uv),14.),np.full(len(uv),14.),
                                19.1+.4*uv[:,0],2.2+uv[:,1],np.zeros(len(uv))])
    images=previous.pixels(settings(train))
    design=lambda uv:np.column_stack([np.ones(len(uv)),uv,uv[:,0]*uv[:,1]])
    coefficients=np.linalg.solve(design(train),images)
    unit=qmc.Sobol(2,scramble=True,seed=20261032).random_base2(10)
    test=np.column_stack([(18.2+1.7*unit[:,0]-19.1)/.4,2.05+1.9*unit[:,1]-2.2])
    truth=previous.pixels(settings(test));prediction=design(test)@coefficients
    error=np.linalg.norm(prediction-truth,axis=1)
    return dict(training_images=4,heldout_joint_images=len(test),
                height_range=[18.2,19.9],width_range=[2.05,3.95],
                coordinates='u=(height-19.1)/0.4; v=width-2.2; fixed cx=cy=14, lean=0',
                equation='F(u,v)=A+B*u+C*v+D*u*v',
                max_heldout_L2_error=float(error.max()),
                mixed_derivative_norm=float(np.linalg.norm(coefficients[3])),
                maximum_pixel_error=float(np.abs(prediction-truth).max())),coefficients


def count_order_control():
    """Exact pixel-arc-length flows in an upright bilinear patch, no Euler step."""
    start=np.array([14.,14.,19.3,3.2,0.]);rows=[]
    for step in [.003,.01,.03,.1,.3]:
        endpoints=[];trajectories=[];matching_checks=[]
        for order in [[2,3],[3,2]]:
            p=start.copy();path=[p.copy()]
            for axis in order:
                speed=np.linalg.norm(jet(p)[3][:,axis])
                end=p.copy();end[axis]+=step/speed
                # Each direction is constant along its own physical-control
                # line in this patch, so this is an exact unit-speed flow.
                assert np.max(np.abs(jet(end)[3][:,axis]-jet(p)[3][:,axis]))<1e-12
                assert abs(np.linalg.norm(previous.pixels(end)[0]-previous.pixels(p)[0])-step)<1e-12
                scores=np.abs(normalized(jet(p)[3]).T@normalized(jet(end)[3]))
                matching_checks.append(bool(np.array_equal(scores.argmax(axis=1),np.arange(5))))
                p=end;path.append(p.copy())
            endpoints.append(p);trajectories.append(np.array(path))
        image_error=float(np.linalg.norm(previous.pixels(endpoints[0])[0]-previous.pixels(endpoints[1])[0]))
        # A physical-calibration control, with fixed height/width increments.
        physical_a=start.copy();physical_a[2]+=.05;physical_a[3]+=.05
        physical_b=start.copy();physical_b[3]+=.05;physical_b[2]+=.05
        physical_error=float(np.linalg.norm(previous.pixels(physical_a)[0]-previous.pixels(physical_b)[0]))
        rows.append(dict(step=step,counts=[1,1],height_then_width=endpoints[0].tolist(),
                         width_then_height=endpoints[1].tolist(),image_order_difference=image_error,
                         fixed_physical_increment_order_difference=physical_error,
                         all_direction_matches_correct=all(matching_checks),
                         trajectories=[p.tolist() for p in trajectories]))
    assert all(r['image_order_difference']>0 for r in rows)
    return dict(start_settings=start.tolist(),description='One count = step units of image arc length along the current true knob.',
                rows=rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    history=history_benchmark()
    bilinear,coefficients=bilinear_control();order=count_order_control()
    assert bilinear['max_heldout_L2_error']<1e-12
    with gzip.open(OUT/'history_trials.json.gz','wt') as f:json.dump(history,f,separators=(',',':'))
    summary=dict(method={k:v for k,v in history.items() if k not in ['comparisons','paths','aggregate']},
                 aggregate=history['aggregate'],bilinear_control=bilinear,count_order_control=order)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    np.savez_compressed(OUT/'bilinear_coefficients.npz',coefficients=coefficients)
    print('Direction-history experiment complete.',flush=True)


if __name__=='__main__':main()
