"""Empirically test nearest-direction continuation across generated-one states.

Known controls generate manifold moves and evaluate matches; matching uses
absolute pixel dot products only. Local fits are ordinary equal-weight SVD.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/generated-one-mpl')
from pathlib import Path
import json
import csv
import gzip
import time
import numpy as np
from scipy.optimize import brentq,linear_sum_assignment
from scipy.stats import qmc
from grey_ones import KNOBS,RANGES,render
from analyze_root_one_local_svd import directions,SCALE
from analyze_generated_one_local_bases_across_starts import starts,LOW,HIGH
from analyze_generated_one_manifold_transport import jet,normalized

HERE=Path(__file__).resolve().parent
OUT=HERE/'figures/generated_one_nearest_direction'
DISTANCES=[1e-5,1e-3,.01,.1,.3,1.,3.]
LOCAL_RADII=[1e-3,1e-5]
TRAIN=256
SOURCE_SEED=20261010
TARGET_SEED=20261021


def pixels(p):
    return render(p,dtype=np.float64).reshape(-1,784)


def assign(b,j):
    scores=np.abs(b@normalized(j))
    row,col=linear_sum_assignment(-scores)
    result=np.empty(5,int);result[row]=col
    return result,scores


def is_cap_kink(p):
    top=p[1]-p[2]/2;bottom=p[1]+p[2]/2
    return bool(abs(p[4])>1e-12 and any(abs(x*16-round(x*16))<1e-9 for x in [top,bottom]))


def local_fit(p,radius,seed):
    """Same normalized parameter rays as prior shells; faster radial solver.

    The analytic Jacobian initializes ray lengths only. Repeatedly measured
    image distances determine the actual equal-distance sample. No parameter
    derivatives enter the SVD matrix. Fallback uses scalar bracketing.
    """
    root=pixels(p)[0];j=jet(p)[3]
    velocity=directions(TRAIN,seed)*SCALE
    t=radius/np.linalg.norm(velocity@j.T,axis=1)
    for iteration in range(8):
        points=p+t[:,None]*velocity
        x=pixels(points)
        distance=np.linalg.norm(x-root,axis=1)
        if np.max(np.abs(distance/radius-1))<2e-6:
            break
        t*=radius/distance
    else:
        for i in range(TRAIN):
            def objective(amount):
                return np.linalg.norm(pixels(p+amount*velocity[i])[0]-root)-radius
            lo=0.;hi=t[i]*2
            while objective(hi)<0:hi*=2
            t[i]=brentq(objective,lo,hi,xtol=1e-14)
        points=p+t[:,None]*velocity;x=pixels(points)
        distance=np.linalg.norm(x-root,axis=1)
    relative_shell_error=float(np.max(np.abs(distance/radius-1)))
    assert relative_shell_error<3e-6
    v=x-root
    active=np.any(v!=0,axis=0)
    # Removing exactly zero columns changes no dot products or singular values.
    _,s,small=np.linalg.svd(v[:,active],full_matrices=False)
    basis=np.zeros((5,784));basis[:,active]=small[:5]
    assert np.max(np.abs(basis@basis.T-np.eye(5)))<1e-12
    labels,score=assign(basis,j)
    total=(s*s).sum()
    return dict(basis=basis,singular=s[:12],labels=labels,knob_scores=score,
                tail_energy=float(np.sum(s[5:]**2)/total),shell_error=relative_shell_error,
                cap_kink=is_cap_kink(p))


def endpoints(p,velocity,distances):
    """First sampled distance crossing along a straight parameter ray."""
    root=pixels(p)[0]
    with np.errstate(divide='ignore',invalid='ignore'):
        limit=np.where(velocity>0,(HIGH-p)/velocity,(LOW-p)/velocity)
    limit[velocity==0]=np.inf
    maximum=limit.min()*.999
    if maximum<=0:return {}
    grid=np.linspace(0,maximum,65)
    d=np.linalg.norm(pixels(p+grid[:,None]*velocity)-root,axis=1)
    result={}
    for target in distances:
        reached=np.flatnonzero(d>=target)
        if not len(reached):continue
        index=reached[0]
        def objective(t):
            return np.linalg.norm(pixels(p+t*velocity)[0]-root)-target
        amount=brentq(objective,grid[index-1],grid[index],xtol=1e-13)
        endpoint=p+amount*velocity
        measured=np.linalg.norm(pixels(endpoint)[0]-root)
        assert abs(measured-target)<1e-10
        result[target]=endpoint
    return result


def match_rows(query,candidate,expected_labels,candidate_labels):
    scores=np.abs(query@candidate.T)
    chosen=scores.argmax(axis=1)
    row,col=linear_sum_assignment(-scores)
    global_choice=np.empty(len(query),int);global_choice[row]=col
    rows=[]
    for i in range(len(query)):
        expected=int(expected_labels[i]);picked=int(chosen[i])
        expected_index=int(np.flatnonzero(candidate_labels==expected)[0])
        other=np.delete(scores[i],expected_index).max()
        tie_count=int(np.sum(scores[i]>=scores[i].max()-1e-10))
        expected_in_best_set=bool(scores[i].max()-scores[i,expected_index]<=1e-10)
        rows.append(dict(query_index=i,expected_knob=KNOBS[expected],
                         chosen_index=picked,chosen_knob=KNOBS[int(candidate_labels[picked])],
                         correct=bool(candidate_labels[picked]==expected),
                         global_assignment_correct=bool(candidate_labels[global_choice[i]]==expected),
                         same_knob_score=float(scores[i,expected_index]),best_score=float(scores[i,picked]),
                         margin=float(scores[i,expected_index]-other),scores=scores[i].tolist(),
                         tie_count=tie_count,expected_in_best_set=expected_in_best_set,
                         strict_correct=bool(expected_in_best_set and tie_count==1)))
    return rows


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    started=time.monotonic();all_starts=starts()
    extra=qmc.Sobol(5,scramble=True,seed=20261016).random_base2(8)
    extra=RANGES[:,0]+(.1+.8*extra)*(RANGES[:,1]-RANGES[:,0])
    oracle_starts=all_starts+[(f'additional_{i+1:02d}',p) for i,p in enumerate(extra)]
    oracle=[];svd=[];basis_follow=[];controls=[];fits=[];arrays={};endpoint_count=0
    sources={};endpoint_cache={}
    for name,p in all_starts:
        for radius in LOCAL_RADII:
            fit=local_fit(p,radius,SOURCE_SEED);sources[name,radius]=fit
            arrays[f'{name}_{radius}_source_basis']=fit['basis']
            fit2=local_fit(p,radius,TARGET_SEED)
            for row in match_rows(fit['basis'],fit2['basis'],fit['labels'],fit2['labels']):
                controls.append(dict(name=name,local_radius=radius,**row))
    print('Source fits and independent-seed zero-move controls ready.',flush=True)
    for start_index,(name,p) in enumerate(oracle_starts):
        j0=jet(p)[3];true0=normalized(j0).T
        for move in range(5):
            for sign in [-1,1]:
                velocity=np.eye(5)[move]*sign
                for distance,end in endpoints(p,velocity,DISTANCES).items():
                    j1=jet(end)[3];true1=normalized(j1).T
                    common=dict(name=name,start_settings=p.tolist(),end_settings=end.tolist(),
                                moved_knob=KNOBS[move],sign=sign,distance=distance,
                                source_cap_kink=is_cap_kink(p),target_cap_kink=is_cap_kink(end))
                    for row in match_rows(true0,true1,np.arange(5),np.arange(5)):
                        oracle.append(dict(**common,**row,followed_self=row['expected_knob']==KNOBS[move]))
                    if start_index>=len(all_starts):continue
                    endpoint_count+=1
                    for radius in LOCAL_RADII:
                        fit=local_fit(end,radius,TARGET_SEED)
                        source=sources[name,radius]
                        key=f'endpoint_{endpoint_count}_{radius}'
                        arrays[key+'_basis']=fit['basis']
                        endpoint_cache[name,move,sign,distance,radius]=fit
                        fits.append(dict(key=key,**common,local_radius=radius,
                                         labels=fit['labels'].tolist(),singular_values=fit['singular'].tolist(),
                                         tail_energy=fit['tail_energy'],shell_error=fit['shell_error']))
                        for row in match_rows(source['basis'],fit['basis'],source['labels'],fit['labels']):
                            svd.append(dict(**common,local_radius=radius,endpoint_key=key,**row,
                                followed_self=row['expected_knob']==KNOBS[move],
                                source_label_score=float(source['knob_scores'][row['query_index'],
                                    list(KNOBS).index(row['expected_knob'])]),
                                chosen_label_score=float(fit['knob_scores'][row['chosen_index'],
                                    list(KNOBS).index(row['chosen_knob'])])))
        print(f'Pure-knob paths: {start_index+1}/{len(oracle_starts)} starts; {endpoint_count} fitted endpoints; {time.monotonic()-started:.1f}s',flush=True)
    # Follow each estimated SVD vector itself by its least-squares parameter
    # velocity, rather than silently replacing it with a pure knob move.
    for index,(name,p) in enumerate(all_starts):
        radius=1e-5;source=sources[name,radius];j=jet(p)[3]
        for vector in range(5):
            b=source['basis'][vector]
            velocity=np.linalg.lstsq(j,b,rcond=None)[0]
            projection_error=float(np.linalg.norm(j@velocity-b))
            for sign in [-1,1]:
                for distance,end in endpoints(p,sign*velocity,[1e-5,1e-3,.01,.1,.3,1.]).items():
                    fit=local_fit(end,radius,TARGET_SEED)
                    matches=match_rows(source['basis'],fit['basis'],source['labels'],fit['labels'])
                    row=matches[vector]
                    key=f'basis_follow_{len(basis_follow)}'
                    arrays[key+'_source_vector']=b;arrays[key+'_target_basis']=fit['basis']
                    basis_follow.append(dict(name=name,start_settings=p.tolist(),end_settings=end.tolist(),
                        local_radius=radius,distance=distance,sign=sign,endpoint_key=key,
                        source_cap_kink=is_cap_kink(p),target_cap_kink=is_cap_kink(end),
                        parameter_velocity=(sign*velocity).tolist(),source_tangent_projection_error=projection_error,
                        target_labels=fit['labels'].tolist(),**row))
        print(f'Following SVD vectors: {index+1}/{len(all_starts)} starts; {time.monotonic()-started:.1f}s',flush=True)
    result=dict(method=dict(training_images=TRAIN,local_radii=LOCAL_RADII,step_distances=DISTANCES,
                            source_seed=SOURCE_SEED,target_seed=TARGET_SEED,knob_order=KNOBS,
                            ordinary_svd=True,weights='all one',matching='independent maximum absolute pixel cosine',
                            label_evaluation='one-to-one maximum absolute cosine to true generator derivatives; labels not used by matcher',
                            oracle_starts=len(oracle_starts),svd_starts=len(all_starts),
                            svd_follow_retraction='p_new=p+t*pinv(J(p))*source_vector; t chosen by measured image distance',
                            cap_kinks='analytic derivatives average the two cap sides; separately flagged, not smooth tangent claims'),
                oracle_comparisons=oracle,svd_comparisons=svd,svd_follow_comparisons=basis_follow,
                zero_move_seed_controls=controls,endpoint_fits=fits,
                starts=[dict(name=name,settings=p.tolist()) for name,p in oracle_starts])
    with gzip.open(OUT/'results.json.gz','wt') as f:
        json.dump(result,f,separators=(',',':'))
    np.savez_compressed(OUT/'basis_vectors.npz',**arrays)
    for filename,rows in [('oracle_comparisons.csv',oracle),('svd_comparisons.csv',svd),('svd_follow_comparisons.csv',basis_follow)]:
        fields=list(rows[0])
        with gzip.open(OUT/(filename+'.gz'),'wt',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    print(f'Complete: {len(oracle)} oracle, {len(svd)} fitted comparisons, {len(basis_follow)} followed-vector trials; {time.monotonic()-started:.1f}s',flush=True)


if __name__=='__main__':
    main()
