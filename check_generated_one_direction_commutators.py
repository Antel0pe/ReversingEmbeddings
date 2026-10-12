"""Do ordinary SVD direction families admit independent coordinate counts?

Use fixed sampling rays for a reproducible smooth local frame. A component
of a pair's Lie bracket outside that pair obstructs coordinate axes even
after independently rescaling each axis. Known J only supplies valid
off-manifold-to-parameter lifts and evaluates the ambient bracket.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from analyze_generated_one_manifold_transport import jet
from analyze_generated_one_nearest_direction import local_fit
from analyze_root_one_local_svd import directions,SCALE
from analyze_generated_one_direction_history import OUT


def tangent_limit_basis(p,seed):
    """Radius-zero limit of the same ray-sampled shell SVD at a smooth point."""
    q,r=np.linalg.qr(jet(p)[3],mode='reduced')
    small=(directions(256,seed)*SCALE)@r.T
    small/=np.linalg.norm(small,axis=1)[:,None]
    _,_,right=np.linalg.svd(small,full_matrices=False)
    return right@q.T


def frame_velocity(p,reference,seed,radius,version):
    j=jet(p)[3]
    basis=(local_fit(p,radius,seed)['basis'] if version=='sampled_shell' else tangent_limit_basis(p,seed))
    row,col=linear_sum_assignment(-np.abs(reference@basis.T))
    indices=col[np.argsort(row)];basis=basis[indices]
    signs=np.sign(np.sum(reference*basis,axis=1));signs[signs==0]=1
    return np.linalg.lstsq(j,(basis*signs[:,None]).T,rcond=None)[0]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    starts=[np.array(p) for p in [[14.43,14.62,19.73,3.17,17.3],
                                  [14.27,14.58,19.61,2.87,11.7],
                                  [14.68,14.29,19.84,3.71,24.3]]]
    rows=[]
    for start_index,p in enumerate(starts):
        j=jet(p)[3];signature=jet(p)[4]
        for seed in [20261031,20261041,20261051]:
            for version in ['sampled_shell','tangent_limit']:
                basis=(local_fit(p,1e-6,seed)['basis'] if version=='sampled_shell' else tangent_limit_basis(p,seed))
                velocity=frame_velocity(p,basis,seed,1e-6,version)
                for epsilon in [1e-4,3e-5,1e-5]:
                    derivative=np.zeros((5,5,5));all_same_cell=True
                    for i in range(5):
                        plus=p+epsilon*velocity[:,i];minus=p-epsilon*velocity[:,i]
                        all_same_cell &= np.array_equal(jet(plus)[4],signature) and np.array_equal(jet(minus)[4],signature)
                        derivative[i]=(frame_velocity(plus,basis,seed,1e-6,version)-
                                       frame_velocity(minus,basis,seed,1e-6,version))/(2*epsilon)
                    for i in range(5):
                        for k in range(i+1,5):
                            bracket=j@(derivative[i,:,k]-derivative[k,:,i])
                            pair=j@velocity[:,[i,k]]
                            outside=bracket-pair@np.linalg.lstsq(pair,bracket,rcond=None)[0]
                            rows.append(dict(start=start_index,settings=p.tolist(),seed=seed,
                                             version=version,epsilon=epsilon,axis_pair=[i+1,k+1],
                                             same_renderer_formula_cell=bool(all_same_cell),
                                             bracket_L2=float(np.linalg.norm(bracket)),
                                             outside_pair_L2=float(np.linalg.norm(outside))))
    largest=max(rows,key=lambda r:r['outside_pair_L2'])
    report=dict(neighbors=256,local_radius=1e-6,
                sampling='fixed normalized parameter rays at source and perturbations; three seed repeats',
                method='central differences of parameter lifts of source-aligned ordinary SVD fields',
                controls='Constant physical parameter fields have exactly zero brackets.',
                rows=rows,largest=largest)
    (OUT/'commutator_check.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(largest,indent=2))
    print('All differences stay in a common renderer formula cell:',all(r['same_renderer_formula_cell'] for r in rows))


if __name__=='__main__':main()
