"""Construct knob-indexed additive curves and test their representational limit.

On a complete Cartesian grid with equal weights, conditional means give the
global least-squares optimum over all additive knob-indexed functions.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
from pathlib import Path
import itertools
import json
import time
import numpy as np
from grey_ones import RANGES, KNOBS, render
from optimize_generated_one_joint_curves import SOURCE, node_fit, evaluate, encode_multistart
from analyze_generated_one_principal_curves import pixels, error_stats

OUT=Path(__file__).resolve().parent/'figures/generated_one_principal_curves/direct_construction'


def product_optimum(q,support):
    """Exact global additive optimum on a q^5 equal-weight product grid."""
    levels=(np.arange(q)+.5)/q
    indices=np.indices((q,)*5,dtype=np.int16).reshape(5,-1).T
    p=RANGES[:,0]+levels[indices]*np.ptp(RANGES,axis=1)
    # Use the full covering cloud's support, not just its corners: interior
    # lean values can put ink in pixels absent at every corner.
    x=np.empty((len(p),int(support.sum())),dtype=np.float32)
    for i in range(0,len(p),4096):
        full=render(p[i:i+4096]).reshape(-1,784)
        assert not np.any(full[:,~support])
        x[i:i+4096]=full[:,support]
    tensor=x.reshape((q,)*5+(x.shape[1],))
    mean=x.mean(0,dtype=np.float64)
    curves=[]
    for k in range(5):
        other_axes=tuple(j for j in range(5) if j!=k)
        curves.append(tensor.mean(axis=other_axes,dtype=np.float64)-mean)
    total=float(np.mean(np.sum(x.astype(np.float64)**2,axis=1))-np.sum(mean**2))
    captured=float(sum(np.mean(np.sum(c*c,axis=1)) for c in curves))
    # A product design makes the five centered effects mutually orthogonal.
    # Pythagoras gives the exact globally optimal additive squared error here.
    m=np.zeros(784);m[support]=mean
    c=np.zeros((5,q,784));c[:,:,support]=curves
    np.savez_compressed(OUT/f'product_{q}.npz',mean=m,curves=c,levels=levels)
    # Check the orthogonality identity against direct residuals on the smallest grid.
    if q==5:
        pred=mean+sum(curves[k][indices[:,k]] for k in range(5))
        observed=np.mean(np.sum((x-pred)**2,axis=1))
        assert abs(observed-(total-captured))<1e-7
    return dict(levels_per_knob=q,images=len(p),globally_optimal_knob_additive_capture_percent=100*captured/total,
        residual_sse_per_image=total-captured,variance_per_image=total,
        scope='Exact optimum on this equal-weight Cartesian grid; convergence is a numerical estimate for the continuous uniform box')


def main():
    OUT.mkdir(exist_ok=True)
    source=np.load(SOURCE);p=source['coverage_knobs'];x=pixels(p)
    active=np.any(x!=0,axis=0);xx=x[:,active];u=(p-RANGES[:,0])/np.ptp(RANGES,axis=1)
    r=dict(method='Direct knob coordinates; additive pixel-space curves; global conditional-mean construction on complete product grids',
        knob_order=KNOBS,original_covering_points=len(p),finite_cloud_fits=[],product_grid_optima=[])
    started=time.monotonic()
    for n in [65,129,257,513]:
        mean,curves=node_fit(xx,u,n,0.)
        pred=mean+sum(evaluate(curves[k],u[:,k]) for k in range(5))
        stat=error_stats(xx-pred,xx,xx.mean(0))
        r['finite_cloud_fits'].append(dict(nodes_per_curve=n,metrics=stat,
            note='Global least squares in this fixed-coordinate linear-spline basis, up to the tiny numerical ridge'))
        print('Direct fit',n,stat['explained_percent'],flush=True)
        if n==513:
            full_pred=np.zeros_like(x);full_pred[:,active]=pred
            np.savez_compressed(OUT/'direct_513_examples.npz',knobs=p,images=x,predictions=full_pred)
        (OUT/'results.json').write_text(json.dumps(r,indent=2)+'\n')
    for q in [5,9,13]:
        row=product_optimum(q,active);r['product_grid_optima'].append(row)
        print('Product grid:',row,'elapsed',time.monotonic()-started,flush=True)
        (OUT/'results.json').write_text(json.dumps(r,indent=2)+'\n')
    base=RANGES.mean(1);p4=np.tile(base,(4,1));p4[1,3]+=1.;p4[2,4]+=15.;p4[3,3]+=1.;p4[3,4]+=15.
    im=pixels(p4);interaction=im[3]-im[1]-im[2]+im[0]
    u4=(p4-RANGES[:,0])/np.ptp(RANGES,axis=1)
    direct=mean+sum(evaluate(curves[k],u4[:,k]) for k in range(5))
    assert np.linalg.norm(direct[3]-direct[1]-direct[2]+direct[0])<1e-10
    learned=np.load(OUT.parent/'joint/joint_513.npz')
    free,_,_=encode_multistart(im[:,active],learned['mean'][active],learned['curves'][:,:,active],
                             xx,learned['coordinates'])
    full_direct=np.zeros_like(im);full_direct[:,active]=direct
    full_free=np.zeros_like(im);full_free[:,active]=free
    r['interaction_witness']=dict(knobs=p4.tolist(),mixed_difference_L2=float(np.linalg.norm(interaction)),
        max_pixel_interaction=float(abs(interaction).max()),
        actual_direct_model_image_errors=np.linalg.norm(im-full_direct,axis=1).tolist(),
        actual_free_model_image_errors=np.linalg.norm(im-full_free,axis=1).tolist(),
        any_additive_model_four_corner_max_image_error_lower_bound=float(np.linalg.norm(interaction)/4),
        proof='Every sum of five single-knob functions has zero mixed difference; the renderer does not. Triangle inequality bounds at least one corner error.')
    np.savez_compressed(OUT/'interaction_witness.npz',images=im,interaction=interaction,direct=full_direct,free=full_free)
    (OUT/'results.json').write_text(json.dumps(r,indent=2)+'\n')
    print('Completed direct construction and structural witness.',flush=True)


if __name__=='__main__':main()
