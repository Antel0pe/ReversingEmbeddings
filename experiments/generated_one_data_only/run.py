"""Data preparation and independent oracle evaluation; not discovery code."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import hashlib
import itertools
import json
import sys
import time
import numpy as np
from scipy.stats import qmc
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from grey_ones import render,RANGES
from learn import (moments,LocalPolynomial,global_polynomial,error_stats,design,
                   choose_measurements,choose_hyperparameters,NAMES)
OUT=ROOT/'results/generated_one_data_only'
SPAN=np.ptp(RANGES,axis=1)
LOW=RANGES[:,0]
COUNTS=[512,4096,32768]


def sample(power,seed):
    return LOW+qmc.Sobol(5,scramble=True,seed=seed).random_base2(power)*SPAN


def threshold(m):
    return bool(m['max_pixel']<=2e-6 and m['max_image_l2']<=1e-5)


def eval_points():
    points=[];tags=[]
    def add(p,tag):
        p=np.atleast_2d(p)
        p=p[np.all((p>=LOW)&(p<=LOW+SPAN),axis=1)]
        points.extend(p);tags.extend([tag]*len(p))
    add(sample(10,2601),'unseen_joint_states')
    add(LOW+np.array(list(itertools.product([0,.5,1],repeat=5)))*SPAN,'lattice_with_corners')
    p=sample(6,2602)
    for i in range(5):
        for edge in (1e-10,1-1e-10):
            z=p.copy();z[:,i]=LOW[i]+edge*SPAN[i];add(z,'near_faces')
    p=sample(6,2603)
    for sign in (-1,1):
        cap=p[:,1]+sign*p[:,2]/2
        for eps in (-1e-9,0,1e-9):
            z=p.copy();z[:,1]=np.round(cap*16)/16-sign*z[:,2]/2+eps
            add(z,'cap_transitions')
    for side in (-1,1):
        for y in (5+1/32,18+31/32):
            edge=p[:,0]-np.tan(np.deg2rad(p[:,4]))*(y-p[:,1])+side*p[:,3]/2
            for eps in (-1e-9,0,1e-9):
                z=p.copy();z[:,0]+=np.round(edge)-edge+eps
                add(z,'side_transitions')
    a,b=sample(6,2604),sample(6,2605)
    for fraction in (.125,.375,.625,.875):
        add((1-fraction)*a+fraction*b,'unseen_joint_paths')
    return np.array(points),np.array(tags)


def compressed_bytes(*arrays):
    return int(sum(x.nbytes for x in arrays))


def coordinate_diagnostics(p,keep,training_moments):
    # Known parameters and renderer appear only in the evaluator.
    scale=np.std(training_moments[:,keep],axis=0)
    scale=np.maximum(scale,1e-12)
    rows=[]
    for point in p:
        eps=1e-5
        steps=np.eye(5)*SPAN*eps
        high=moments(render(point+steps,dtype=np.float64))[:,keep]
        low=moments(render(point-steps,dtype=np.float64))[:,keep]
        jac=((high-low)/(2*eps)/scale).T
        singular=np.linalg.svd(jac,compute_uv=False)
        rows.append(singular[-1]/singular[0])
    return dict(states=len(p),normalized_step=1e-5,
                minimum_smallest_to_largest=float(np.min(rows)),
                median_smallest_to_largest=float(np.median(rows)),
                worst_state_index=int(np.argmin(rows)),
                zero_rank_count_at_ratio_1e_6=int(np.sum(np.array(rows)<1e-6)))


def domain_and_collision_checks(keep,training_p,training_x):
    rng=np.random.default_rng(2701)
    tm=moments(training_x)[:,keep]
    scale=np.std(tm,axis=0)
    mean=tm.mean(0)
    tree=cKDTree((tm-mean)/scale)
    def solve(target,start):
        def residual(unit):
            image=render(LOW+SPAN*unit,dtype=np.float64)
            return (moments(image)[0,keep]-target)/scale
        result=least_squares(residual,np.clip((start-LOW)/SPAN,1e-10,1-1e-10),
            bounds=(np.zeros(5),np.ones(5)),xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=160)
        return LOW+SPAN*result.x,float(np.linalg.norm(result.fun))
    # Look for a second pre-storage state with the same five image measurements.
    points=sample(4,2702)
    found=[];recover=[]
    for point in points:
        target=moments(render(point,dtype=np.float64))[0,keep]
        best=[]
        for start in (training_p[rng.integers(len(training_p))],LOW+.5*SPAN):
            solution,error=solve(target,start)
            image_error=float(np.linalg.norm(render(solution,dtype=np.float64)-render(point,dtype=np.float64)))
            best.append(dict(moment_residual=error,image_l2=image_error,
                parameter_difference_normalized=float(np.linalg.norm((solution-point)/SPAN))))
            if error<1e-9 and image_error>1e-5:
                found.append(dict(first=point.tolist(),second=solution.tolist(),**best[-1]))
        recover.append(min(best,key=lambda z:z['moment_residual']))
    # Probe the apparent bounding box and convex combinations. No theorem says
    # that these coordinates have a rectangular or even convex image domain.
    probes=[]
    for i in range(24):
        if i<12:
            target=tm.min(0)+rng.random(5)*np.ptp(tm,axis=0)
            kind='measurement_bounding_box'
        else:
            a,b=rng.integers(len(tm),size=2);alpha=rng.uniform(.2,.8)
            target=(1-alpha)*tm[a]+alpha*tm[b]
            kind='measurement_chord'
        _,nearest=tree.query((target-mean)/scale,k=3)
        solved=[solve(target,training_p[j]) for j in nearest]
        solution,error=min(solved,key=lambda x:x[1])
        probes.append(dict(kind=kind,requested=target.tolist(),best_moment_residual=error,
                           found_valid_representative=bool(error<1e-8)))
    return dict(collision_search=dict(states=16,starts_per_state=2,
        found_distinct_image_collisions=found, recoveries=recover,
        caveat='No found collision is not proof of injectivity.'),domain_probes=probes)


def local_patch_checks(keep):
    rng=np.random.default_rng(2801)
    smooth=np.array([14.43,14.62,19.73,3.17,17.3])
    cap=smooth.copy();cap[1]=4.75+cap[2]/2
    rows=[]
    for name,center in [('generic',smooth),('cap_transition',cap)]:
        for radius in (1e-2,1e-3,1e-4,1e-5):
            train=center+rng.uniform(-1,1,(256,5))*SPAN*radius
            valid=center+rng.uniform(-1,1,(128,5))*SPAN*radius
            for dtype in (np.float64,np.float32):
                x=render(train,dtype=dtype);truth=render(valid,dtype=dtype)
                mt,mv=moments(x)[:,keep],moments(truth)[:,keep]
                a,coefficient=global_polynomial(mt,x,mv,degree=3)
                b,coefb=global_polynomial(train,x,valid,degree=3)
                # Shared low-rankness of a local polynomial coefficient table
                # is a diagnostic only, never an explanation claim.
                singular=np.linalg.svd(coefb,compute_uv=False)
                rows.append(dict(start=name,radius_fraction=radius,storage=np.dtype(dtype).name,
                    images_only=error_stats(truth,a),supplied_coordinates=error_stats(truth,b),
                    polynomial_terms=len(coefb),coefficient_rank_at_1e_10=int(np.sum(singular>singular[0]*1e-10))))
    return rows


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    train_p=sample(15,2501)
    train_x=render(train_p)
    val_p=sample(7,2502);val_x=render(val_p)
    np.savez_compressed(OUT/'training_images.npz',images=train_x)
    np.savez_compressed(OUT/'training_with_coordinates.npz',images=train_x,coordinates=train_p)
    np.savez_compressed(OUT/'validation.npz',images=val_x,coordinates=val_p)
    # Everything up to the freeze operates on these public arrays alone.
    selected,selection=choose_measurements(train_x[:4096],val_x)
    measured=moments(train_x);val_measured=moments(val_x)
    print('Image measurements selected:',[NAMES[i] for i in selected],flush=True)
    best_a,tuning_a=choose_hyperparameters(measured[:4096,selected],train_x[:4096],val_measured[:,selected],val_x)
    best_b,tuning_b=choose_hyperparameters(train_p[:4096],train_x[:4096],val_p,val_x)
    frozen=dict(schema=1,selected_measurements=selected,measurement_names=[NAMES[i] for i in selected],
        image_only_hyperparameters={k:best_a[k] for k in ['degree','neighbors']},
        supplied_coordinate_hyperparameters={k:best_b[k] for k in ['degree','neighbors']},
        measurement_selection=selection,hyperparameter_selection_images_only=tuning_a,
        hyperparameter_selection_supplied_coordinates=tuning_b,
        training_counts=COUNTS,validation_states=len(val_x),source_sha256={
            str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),Path(__file__).with_name('learn.py'),Path(__file__).with_name('PROTOCOL.md')]})
    model_path=OUT/'frozen_model.json'
    model_path.write_text(json.dumps(frozen,indent=2)+'\n')
    model_sha=hashlib.sha256(model_path.read_bytes()).hexdigest()
    print('Frozen hyperparameters:',frozen['image_only_hyperparameters'],frozen['supplied_coordinate_hyperparameters'],flush=True)
    # Test arrays and their oracle-only settings are created after this point.
    test_p,tags=eval_points();test_x=render(test_p)
    test_m=moments(test_x)
    np.savez_compressed(OUT/'evaluation_public.npz',images=test_x,coordinates=test_p,tags=tags)
    results=dict(n=len(test_x),model_sha256=model_sha,counts={},global_controls={},
        goal_thresholds=dict(max_pixel=2e-6,max_image_l2=1e-5),
        coordinate_mean=measured[:,selected].mean(0).tolist(),coordinate_scale=measured[:,selected].std(0).tolist())
    final_predictions={}
    for n in COUNTS:
        result={}
        for condition,train_coords,test_coords,params in [
            ('images_only',measured[:n,selected],test_m[:,selected],frozen['image_only_hyperparameters']),
            ('supplied_coordinates',train_p[:n],test_p,frozen['supplied_coordinate_hyperparameters'])]:
            model=LocalPolynomial(train_coords,train_x[:n],**params)
            pred,details=model.predict(test_coords,details=True)
            record=error_stats(test_x,pred)
            record['passes_accuracy']=threshold(record)
            record['minimum_decoded_pixel']=float(np.min(pred))
            record['maximum_decoded_pixel']=float(np.max(pred))
            record['images_outside_coverage_by_over_2e_6']=int(np.count_nonzero(np.any((pred<-2e-6)|(pred>1+2e-6),axis=(1,2))))
            record['worst_l2_coordinates']=test_coords[record['worst_l2_index']].tolist()
            record['groups']={str(tag):dict(n=int(np.sum(tags==tag)),**error_stats(test_x[tags==tag],pred[tags==tag])) for tag in np.unique(tags)}
            record['worst_l2_settings']=test_p[record['worst_l2_index']].tolist()
            record['worst_pixel_settings']=test_p[record['worst_pixel_index']].tolist()
            record['retained_numeric_values']=int(train_coords.size+train_x[:n].size)
            record['retained_bytes']=compressed_bytes(train_coords,train_x[:n])
            record['maximum_evaluation_weight_l1']=max(d['weight_l1'] for d in details)
            result[condition]=record
            if n==COUNTS[-1]:final_predictions[condition]=pred
            print('n',n,condition,record['max_pixel'],record['max_image_l2'],flush=True)
        results['counts'][str(n)]=result
    for condition,x,query,params in [('images_only',measured[:,selected],test_m[:,selected],frozen['image_only_hyperparameters']),
                                   ('supplied_coordinates',train_p,test_p,frozen['supplied_coordinate_hyperparameters'])]:
        pred,coefficient=global_polynomial(x,train_x,query,params['degree'])
        results['global_controls'][condition]=dict(**error_stats(test_x,pred),
            polynomial_terms=len(coefficient),coefficient_count=int(coefficient.size),passes_accuracy=threshold(error_stats(test_x,pred)))
    # Supplied-coordinate encoder is learned separately from six generic moments.
    inverse=LocalPolynomial(measured,train_p,**frozen['supplied_coordinate_hyperparameters'])
    inferred=inverse.predict(test_m)
    decoder=LocalPolynomial(train_p,train_x,**frozen['supplied_coordinate_hyperparameters'])
    reconstructed=decoder.predict(inferred)
    results['supplied_coordinate_encoder']=dict(
        max_normalized_coordinate_error=np.max(np.abs((inferred-test_p)/SPAN),axis=0).tolist(),
        image_roundtrip=error_stats(test_x,reconstructed))
    np.savez_compressed(OUT/'predictions.npz',**final_predictions,encoded_supplied_coordinates=inferred,
                        roundtrip_with_learned_encoder=reconstructed)
    results['coordinate_diagnostics']=coordinate_diagnostics(test_p[:256],selected,measured)
    results['local_patches']=local_patch_checks(selected)
    results['domain_checks']=domain_and_collision_checks(selected,train_p,train_x)
    results['elapsed_seconds']=time.monotonic()-start
    results['success']=dict(numerical_accuracy={k:threshold(results['counts'][str(COUNTS[-1])][k])
        for k in ('images_only','supplied_coordinates')},
        compact_mechanism=False,global_domain_certified=False)
    (OUT/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False)+'\n')
    print('Finished in',results['elapsed_seconds'],'seconds. Scientific success:',results['success'],flush=True)

if __name__=='__main__':main()
