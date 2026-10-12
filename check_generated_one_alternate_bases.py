"""Construct replacements for the local basis using the same saved root data.

These are deliberately rotated bases, not SVD on an unchanged covariance.
Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python check_generated_one_alternate_bases.py
"""
from pathlib import Path
import json
import numpy as np


def main():
    root=Path(__file__).resolve().parent
    out=root/'figures/generated_one_manifold_transport'
    a=np.load(out/'analytic_geometry.npz')
    samples=np.load(root/'figures/generated_one_local_bases_across_starts/bases_and_samples.npz')
    tangent=a['root_tangent'];normal=a['root_normal_mixed_basis'];j=a['root_jacobian']
    base=np.column_stack([tangent,normal])
    test=samples['root_r_0p001_images'][512:]-samples['root_start_image']
    rng=np.random.default_rng(20261011)
    rotation5=np.eye(9)
    rotation5[:5,:5]=np.linalg.qr(rng.normal(size=(5,5)))[0]
    rotation9=np.linalg.qr(rng.normal(size=(9,9)))[0]
    result={};saved={}
    for name,rotation in [('five_tangent_directions_mixed',rotation5),('all_nine_directions_mixed',rotation9)]:
        basis=rotation@base.T
        coefficients=test@basis.T
        error=float(np.linalg.norm(test-coefficients@basis,axis=1).max())
        assert error<1e-13
        assert np.linalg.norm(basis@basis.T-np.eye(9))<1e-12
        knob_coefficients=np.linalg.lstsq(j,basis[:5].T,rcond=None)[0]
        normal_lengths=np.linalg.norm(basis@normal,axis=1)
        if name=='five_tangent_directions_mixed':
            # Every one of these five tangent vectors requires all five
            # physical derivatives: none is a pure original knob direction.
            assert np.min(np.abs(knob_coefficients))>1e-10
            assert np.max(normal_lengths[:5])<1e-12
        else:
            # Every vector has a nonzero normal component, so no individual
            # vector can be any of the five true tangent knob directions.
            assert np.min(normal_lengths)>.1
        saved[name+'_basis']=basis
        saved[name+'_heldout_coefficients']=coefficients
        saved[name+'_rotation']=rotation
        result[name]=dict(heldout_max_absolute_L2_error=error,
                         normal_component_lengths=normal_lengths.tolist(),
                         first_five_coefficients_in_true_knob_derivatives=knob_coefficients.tolist())
    result['interpretation']=('Constructed alternate bases for the same saved root neighborhood. '
        'The first mixes all five tangent knobs and retains four separate normal directions. '
        'The second mixes all nine so no individual vector is tangent. Both span the original '
        'full local space. These are valid bases, not reruns of unweighted SVD on unchanged data.')
    (out/'alternate_basis_check.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(out/'alternate_basis_check.npz',**saved)
    print(json.dumps({name:row['heldout_max_absolute_L2_error'] for name,row in result.items()
                      if isinstance(row,dict)}))


if __name__=='__main__':
    main()
