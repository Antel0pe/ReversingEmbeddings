"""Saved-function, response, read-isolation, and physical-evaluation hash checks."""
from pathlib import Path
import json
import hashlib
import numpy as np
from common import align_frames
from saved import feature_from_archive,reading
from discover import audit

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'figures/pooled_family_coordinates'


def main():
    for name in ['FREEZE.json','BALANCED_FREEZE.json']:
        for relative,digest in json.loads((OUT/name).read_text())['sha256'].items():
            assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest,relative
    blocked=[]
    for path in [ROOT/'grey_ones.py',ROOT/'experiments/derived_state_coordinates.py',Path('/tmp/pooled_family_coordinates_labels.npz')]:
        try:audit('open',(str(path),'r',0))
        except PermissionError:blocked.append(path.name)
        else:raise AssertionError('Forbidden read was accepted')
    old=np.load(ROOT/'figures/svd_blind_intermediates/observations.npz')
    new=np.load(OUT/'patch_observations.npz');balanced=np.load(OUT/'balanced_frames.npz')
    records=[];worst_value=0.;worst_spectrum=0.
    for run,filename in [('primary','models.npz'),('balanced','balanced_models.npz')]:
        models=np.load(OUT/filename)
        for key in models.files:
            if not key.endswith('_coefficient'):continue
            prefix=key.removesuffix('_coefficient')
            scope=next(s for s in ['local','broad']+[f'patch_{i}_{j}' for i in range(2) for j in range(4)] if prefix.startswith(s+'_'))
            data=old if scope in ['local','broad'] else new
            method='pca' if '_pca_' in prefix else 'svd'
            x=data[scope+'_test_x'][:8]
            reconstructed=reading(models,prefix,x)
            expected=models[prefix+'_test_q'][:8]
            value_error=np.linalg.norm(reconstructed-expected)/max(np.linalg.norm(expected),1e-20)
            assert value_error<1e-8,(prefix,value_error)
            worst_value=max(worst_value,value_error)
            f=feature_from_archive(models,prefix);c=models[key]
            fx=data[scope+'_test_frame_x'][:4]
            frame_source=balanced if run=='balanced' else data
            b=frame_source[scope+'_test_'+method][:4]
            analytic=f.derivatives(fx,b)@c
            saved=models[prefix+'_test_response'][:4]
            # Permutation/sign alignment changes the rows, but not singular values.
            sa=np.linalg.svd(analytic,compute_uv=False);sb=np.linalg.svd(saved,compute_uv=False)
            spectrum=np.linalg.norm(sa-sb)/max(np.linalg.norm(sa),1e-20)
            assert spectrum<1e-7,(prefix,spectrum)
            worst_spectrum=max(worst_spectrum,spectrum)
            errors=[]
            for h in [1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10]:
                fd=np.stack([((f.values(fx+h*b[:,j])-f.values(fx-h*b[:,j]))@c)/(2*h) for j in range(5)],axis=1)
                errors.append(float(np.linalg.norm(fd-analytic)/max(np.linalg.norm(analytic),1e-20)))
            best=min(errors)
            # A nonuniformly stable function is kept as a failure, not silently repaired.
            records.append(dict(run=run,model=prefix,best_finite_difference_relative_error=best,
                                derivative_check_pass=bool(best<1e-4)))
    output=dict(frozen_hashes_match=True,forbidden_reads_blocked=blocked,
        maximum_saved_value_relative_error=worst_value,maximum_response_spectrum_relative_error=worst_spectrum,
        finite_difference_checks=records,
        finite_difference_failures=[a for a in records if not a['derivative_check_pass']])
    (OUT/'verification.json').write_text(json.dumps(output,indent=2))
    print('Saved-function and spectrum checks passed; finite difference failures:',len(output['finite_difference_failures']))


if __name__=='__main__':main()
