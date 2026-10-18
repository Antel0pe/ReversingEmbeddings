"""Independent saved-artifact and gradient checks; never refits discovery."""
from pathlib import Path
import json
import hashlib
import numpy as np
from discover import audit

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'figures/svd_blind_intermediates'


def main():
    frozen=json.loads((OUT/'FREEZE.json').read_text())
    for p,digest in frozen['sha256'].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest
    denied=[]
    for p in [ROOT/'grey_ones.py',ROOT/'experiments/derived_state_coordinates.py',
              Path('/tmp/svd_blind_intermediates_labels.npz')]:
        try:audit('open',(str(p),'r',0))
        except PermissionError:denied.append(p.name)
        else:raise AssertionError('A forbidden source was allowed')
    x=np.load(OUT/'observations.npz');m=np.load(OUT/'discovered_models.npz')
    e=np.load(OUT/'evaluation_examples.npz')
    assert not any('setting' in k or k.endswith('_p') or 'derivative' in k for k in x.files)
    largest_fd=0.;largest_sv=0.
    for scope in ['local','broad']:
        for method in ['svd','pca','shuffled','global_pc']:
            key=scope+'_'+method
            ma=e[key+'_mask_a'];mb=e[key+'_mask_b'];c=e[key+'_constant']
            def reading(image):
                return image@ma+(image@mb)/image.sum(1)[:,None]+c
            image=x[f'{scope}_test_frame_x'][:12]
            b=x[f'{scope}_test_'+('pca' if method=='pca' else 'svd')][:12]
            mass=image.sum(1)
            analytic=b@ma+(b@mb)/mass[:,None,None]-(image@mb)[:,None,:]*b.sum(2)[:,:,None]/mass[:,None,None]**2
            eps=1e-5
            fd=np.stack([(reading(image+eps*b[:,j])-reading(image-eps*b[:,j]))/(2*eps) for j in range(5)],axis=1)
            relative=np.linalg.norm(fd-analytic)/max(np.linalg.norm(analytic),1e-20)
            assert relative<2e-6,(key,relative)
            largest_fd=max(largest_fd,relative)
            saved=m[key+'_test_response'][:12]
            sa=np.linalg.svd(analytic,compute_uv=False);ss=np.linalg.svd(saved,compute_uv=False)
            sr=np.linalg.norm(sa-ss)/max(np.linalg.norm(sa),1e-20)
            assert sr<1e-8,(key,sr)
            largest_sv=max(largest_sv,sr)
            assert np.max(abs(reading(x[f'{scope}_test_x'])-m[key+'_test_q']))<1e-8
    result=dict(all_frozen_hashes_match=True,forbidden_read_checks=denied,
        approved_archive_contains_no_settings_or_derivatives=True,
        maximum_relative_finite_difference_error=largest_fd,
        maximum_relative_response_spectrum_error=largest_sv,
        explicit_pixel_mask_formula_matches_saved_coordinates=True)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
