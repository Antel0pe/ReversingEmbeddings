"""Blind post-fit diagnosis: rotation needed for exactly separated readings."""
from pathlib import Path
import json
import hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'figures/pooled_family_coordinates'


def verify_frozen():
    manifest=json.loads((OUT/'FREEZE.json').read_text())
    for p,digest in manifest['sha256'].items():
        assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest,p
    return manifest


def main():
    verify_frozen()
    models=np.load(OUT/'models.npz');result={}
    for key in models.files:
        if not key.endswith('_test_response'):continue
        response=models[key]
        coefficient=np.linalg.pinv(np.swapaxes(response,1,2),rcond=1e-8)
        lengths=np.linalg.norm(coefficient,axis=1)
        cosine=np.diagonal(coefficient,axis1=1,axis2=2)/np.maximum(lengths,1e-30)
        angles=np.rad2deg(np.arccos(np.clip(cosine,-1,1)))
        residual=np.swapaxes(response,1,2)@coefficient-np.eye(5)
        result[key.removesuffix('_test_response')]=dict(
            median_rotation_degrees=float(np.median(angles)),p95_rotation_degrees=float(np.quantile(angles,.95)),
            maximum_rotation_degrees=float(angles.max()),
            fraction_within_5_degrees=float(np.mean(angles<=5)),
            fraction_within_10_degrees=float(np.mean(angles<=10)),
            fraction_within_20_degrees=float(np.mean(angles<=20)),
            median_speed_multiplier=float(np.median(lengths)),
            p95_speed_multiplier=float(np.quantile(lengths,.95)),
            exact_separation_residual=float(np.max(abs(residual))))
    (OUT/'blind_rotation_diagnostic.json').write_text(json.dumps(dict(
        meaning='Unique correction within the supplied five-vector span for unit coordinate responses, when the response matrix is invertible. Angles include sign reversals.',
        not_used_to_refit_or_select=True,models=result),indent=2))
    verify_frozen()
    for key in ['patch_0_0_svd_fixed_joint','patch_1_0_svd_fixed_joint','broad_svd_fixed_joint','broad_svd_quadratic']:
        print(key,result.get(key),flush=True)


if __name__=='__main__':main()
