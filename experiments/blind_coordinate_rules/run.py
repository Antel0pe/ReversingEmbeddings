"""Blinded numeric-rule experiment; this adapter alone knows the renderer.

Run from repository root:
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/blind_coordinate_rules/run.py
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/blind-coordinate-rule-mpl')
from pathlib import Path
import sys, json, csv, time
from itertools import product
import numpy as np
from scipy.stats import qmc
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from grey_ones import render, RANGES
from coordinate_rule import CoordinateRule

OUT=ROOT/'figures'/'blind_coordinate_rules'
OUT.mkdir(parents=True,exist_ok=True)
BASE=np.array([14.,14.,19.75,3.2,0.])
SEED=20261015
STEP=.1
SUPPORT_TOL=1e-7
FIT_TOL=1e-9
PERM=np.random.default_rng(SEED).permutation(784)
UNDO=np.argsort(PERM)
CASES=[('height_width',[2,3],129),('width_lean',[3,4],65),
       ('all_five',list(range(5)),5)]


def sobol(n,d,seed):
    return qmc.Sobol(d,scramble=True,seed=seed).random_base2(int(np.log2(n)))


class Reference:
    def __init__(self,axes,permutation):
        self.axes=axes
        self.permutation=permutation
        self.lo=RANGES[axes,0]
        self.span=np.diff(RANGES[axes],axis=1).ravel()
    def physical(self,u):
        points=np.repeat(BASE[None,:],len(np.atleast_2d(u)),axis=0)
        points[:,self.axes]=self.lo+np.atleast_2d(u)*self.span
        return points
    def __call__(self,u):
        return render(self.physical(u),dtype=np.float64).reshape(-1,784)[:,self.permutation]


def score(pred_delta,true_delta,pred_new,true_new):
    actual=np.abs(true_delta)>SUPPORT_TOL
    guessed=np.abs(pred_delta)>SUPPORT_TOL
    fp=np.sum(guessed & ~actual,axis=1)
    fn=np.sum(~guessed & actual,axis=1)
    err=pred_delta-true_delta
    relative=np.linalg.norm(err,axis=1)/np.maximum(np.linalg.norm(true_delta,axis=1),1e-15)
    return dict(updates=len(err),false_indices=int(fp.sum()),missed_indices=int(fn.sum()),
        actual_indices=int(actual.sum()),predicted_indices=int(guessed.sum()),
        exact_support_fraction=float(np.mean((fp+fn)==0)),
        mean_false_indices=float(fp.mean()),mean_missed_indices=float(fn.mean()),
        max_change_pixel_error=float(np.abs(err).max()),
        mean_relative_change_error=float(relative.mean()),
        median_relative_change_error=float(np.median(relative)),
        p95_relative_change_error=float(np.quantile(relative,.95)),
        aggregate_relative_change_error=float(np.linalg.norm(err)/np.linalg.norm(true_delta)),
        max_endpoint_pixel_error=float(np.max(np.abs(pred_new-true_new))),
        median_endpoint_pixel_error=float(np.median(np.max(np.abs(pred_new-true_new),axis=1))))


def edge_validation(rule, reference):
    """Unseen probes near the discovered 2D breaks, including crossing updates."""
    points=[]
    for axis,knots in enumerate(rule.knots):
        for knot in knots[1:-1]:
            for shift in [-STEP,-.5*STEP,0.]:
                for epsilon in [-1e-8,-1e-10,1e-10,1e-8]:
                    value=knot+shift+epsilon
                    if not 0<=value<=1-STEP:continue
                    for other in [.071,.293,.517,.839]:
                        point=np.full(2,other);point[axis]=value
                        points.append(point)
    starts=np.tile(np.asarray(points),(2,1))
    targets=starts.copy()
    targets[:len(points),0]+=STEP;targets[len(points):,1]+=STEP
    truth_start=reference(starts);truth_new=reference(targets)
    pred_new=rule.interpolate(targets)
    return score(pred_new-rule.interpolate(starts),truth_new-truth_start,pred_new,truth_new)


def export_local_equations(rule,reference):
    """Readable numeric polynomials; emitted AFTER discovery, for 2D only."""
    cells=[]
    for i,j in product(range(len(rule.knots[0])-1),range(len(rule.knots[1])-1)):
        f00=rule.table_tensor[i,j,UNDO]
        f10=rule.table_tensor[i+1,j,UNDO]
        f01=rule.table_tensor[i,j+1,UNDO]
        f11=rule.table_tensor[i+1,j+1,UNDO]
        vectors=[f00,f10-f00,f01-f00,f11-f10-f01+f00]
        terms=[]
        for name,vector in zip(['base','first_control','second_control','interaction'],vectors):
            active=np.flatnonzero(np.abs(vector)>1e-12)
            terms.append(dict(name=name,output_indices=active.tolist(),coefficients=vector[active].tolist()))
        bounds=[[float(reference.lo[k]+reference.span[k]*rule.knots[k][cell]),
                 float(reference.lo[k]+reference.span[k]*rule.knots[k][cell+1])]
                for k,cell in enumerate([i,j])]
        cells.append(dict(cell=[i,j],bounds=bounds,terms=terms))
    return dict(equation='output = base + u*first_control + v*second_control + u*v*interaction',
                definition='u and v are fractions across the two listed cell intervals',
                output_indices='original flat pixel indices, restored only after fitting',
                coefficient_export_threshold=1e-12,cells=cells)


def run_case(name,axes,cap):
    start=time.monotonic()
    d=len(axes)
    # Contexts expose no control names or image semantics to the rule.
    contexts=np.vstack([np.full(d,.5),sobol(8,d,SEED+d)])
    reference=Reference(axes,PERM)
    rule=CoordinateRule(reference,d,contexts,FIT_TOL,cap)
    probe=sobol(256,d,SEED+40+d)
    sources=probe*(1-STEP)
    targets=[]; starts=[]; labels=[]
    for axis in range(d):
        target=sources.copy();target[:,axis]+=STEP
        targets.append(target);starts.append(sources);labels.extend([axis]*len(sources))
    starts=np.concatenate(starts);targets=np.concatenate(targets);labels=np.asarray(labels)
    truth_start=reference(starts);truth_new=reference(targets)
    truth_delta=truth_new-truth_start
    predicted_start=rule.interpolate(starts);predicted_new=rule.interpolate(targets)
    predicted_delta=predicted_new-predicted_start
    metrics=score(predicted_delta,truth_delta,predicted_new,truth_new)
    root=(BASE[axes]-reference.lo)/reference.span
    root_delta=[]
    for axis in range(d):
        target=root.copy();target[axis]+=STEP
        root_delta.append((reference(target)-reference(root))[0])
    fixed=np.asarray(root_delta)[labels]
    baseline=score(fixed,truth_delta,truth_start+fixed,truth_new)
    per_axis={chr(65+axes[axis]):score(predicted_delta[labels==axis],truth_delta[labels==axis],
             predicted_new[labels==axis],truth_new[labels==axis]) for axis in range(d)}
    support_sensitivity={}
    for threshold in [1e-7,1e-4,1e-2]:
        true=np.abs(truth_delta)>threshold;pred=np.abs(predicted_delta)>threshold
        support_sensitivity[str(threshold)]=dict(false_indices=int(np.sum(pred & ~true)),
            missed_indices=int(np.sum(true & ~pred)),
            exact_support_fraction=float(np.mean(np.all(true==pred,axis=1))))
    # Re-run the entire discovery with a different output order, including knots.
    audit_reference=Reference(axes,np.arange(784))
    audit=CoordinateRule(audit_reference,d,contexts,FIT_TOL,cap)
    same_knots=all(np.array_equal(a,b) for a,b in zip(rule.knots,audit.knots))
    same_queries=set(rule.observations.cache)==set(audit.observations.cache)
    audit_error=float(np.max(np.abs(rule.interpolate(probe)[:,UNDO]-audit.interpolate(probe))))
    assert same_knots and same_queries and audit_error<1e-10
    # SVD factorization must not materially change the interpolation table.
    direct_error=float(np.max(np.abs(rule.interpolate(probe)-rule.interpolate(probe,factored=False))))
    assert direct_error<1e-10
    # Explicit sparse output is exactly the rule's thresholded dense difference.
    sample_indices,new_values,sample_delta=rule.sparse_edits(starts[0],0,STEP,SUPPORT_TOL)
    assert np.array_equal(sample_indices,np.flatnonzero(np.abs(predicted_delta[0])>SUPPORT_TOL))
    assert np.max(np.abs(new_values-predicted_new[0,sample_indices]),initial=0)<1e-12
    # No evaluation source or destination may be an observed training coordinate.
    assert not any(tuple(x) in rule.observations.cache for x in np.r_[starts,targets])
    rule.save(OUT/f'{name}_rule.npz')
    if name=='height_width':
        (OUT/'height_width_explicit_cells.json').write_text(json.dumps(export_local_equations(rule,reference),indent=2))
    np.savez_compressed(OUT/f'{name}_evaluation.npz',
        starts=starts,targets=targets,labels=labels,
        source=truth_start[:,UNDO],truth_new=truth_new[:,UNDO],
        truth_delta=truth_delta[:,UNDO],predicted_delta=predicted_delta[:,UNDO],
        predicted_new=predicted_new[:,UNDO],baseline_delta=fixed[:,UNDO])
    result=dict(axes=[chr(65+i) for i in axes],knots=[k.tolist() for k in rule.knots],
        physical_knots=[(reference.lo[j]+reference.span[j]*k).tolist() for j,k in enumerate(rule.knots)],
        knot_diagnostics=rule.knot_diagnostics,knots_per_axis=[len(k) for k in rule.knots],
        coordinate_cells=int(np.prod([len(k)-1 for k in rule.knots])),
        table_states=len(rule.grid),oracle_training_states=len(rule.observations.cache),
        output_pattern_count=rule.rank,active_output_channels=len(rule.active),
        svd_table_max_error=rule.factor_error,
        raw_table_scalars=int(rule.table.size),
        stored_rule_scalars=int(rule.weights.size+rule.patterns.size+rule.anchor.size+sum(map(len,rule.knots))),
        scalar_entries_above_1e_10=int(np.sum(np.abs(rule.weights)>1e-10)+np.sum(np.abs(rule.patterns)>1e-10)),
        metrics=metrics,baseline=baseline,per_axis=per_axis,support_sensitivity=support_sensitivity,
        permutation_audit=dict(same_knots=same_knots,same_observed_coordinates=same_queries,
            same_pattern_count=rule.rank==audit.rank,max_restored_prediction_difference=audit_error),
        interpolation_vs_svd_max_difference=direct_error,seconds=time.monotonic()-start)
    if name=='height_width':result['near_discovered_breakpoints']=edge_validation(rule,reference)
    print(json.dumps({name:{k:result[k] for k in ['knots_per_axis','table_states','oracle_training_states','output_pattern_count','metrics','seconds']}}),flush=True)
    return result,rule


def main():
    results={};rules={}
    for name,axes,cap in CASES:
        results[name],rules[name]=run_case(name,axes,cap)
        (OUT/'results.json').write_text(json.dumps(results,indent=2))
    np.savez_compressed(OUT/'blinding.npz',output_permutation=PERM,inverse_permutation=UNDO)
    with (OUT/'metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['case','rule']+list(results['all_five']['metrics']))
        writer.writeheader()
        for name,result in results.items():
            for method in ['metrics','baseline']:
                writer.writerow(dict(case=name,rule='adaptive rule' if method=='metrics' else 'fixed root pattern',**result[method]))
    print('Finished numerical experiment.',flush=True)

if __name__=='__main__':main()
