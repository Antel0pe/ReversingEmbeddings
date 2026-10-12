"""Measure smaller-radius separation using ordinary, equally weighted SVD.

Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python analyze_generated_one_scale_separation.py
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json
import numpy as np
from grey_ones import render
from analyze_root_one_local_svd import directions, pixels, shell, residual_curve
from analyze_generated_one_local_bases_across_starts import LOW,HIGH

HERE=Path(__file__).resolve().parent
OUT=HERE/'figures/generated_one_scale_separation'
SOURCE=HERE/'figures/generated_one_local_bases_across_starts'
NAMES=['root','spread_02','spread_03','spread_04','lean_30']
RADII=[.0001,.00001]


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    source=np.load(SOURCE/'bases_and_samples.npz')
    existing=json.loads((SOURCE/'results.json').read_text())['results']
    rays=np.vstack([directions(512,20261010),directions(256,20261011)])
    rows=[];arrays={}
    for name in NAMES:
        center=source[name+'_start_settings'];root=pixels(center)[0]
        for radius in RADII:
            settings,x,native,shellerr=shell(rays,radius,root,center=center,low=LOW,high=HIGH)
            v=x-root
            _,s,b=np.linalg.svd(v[:512],full_matrices=False)
            rank=int(np.sum(s>s[0]*1e-10))
            rms,worst=residual_curve(v[512:],b[:rank])
            error=np.linalg.norm(v[512:]-(v[512:]@b[:rank].T)@b[:rank],axis=1).max()
            assert error<1e-12
            native_v=native.astype(float)-render(center)[0].ravel().astype(float)
            _,ns,nb=np.linalg.svd(native_v[:512],full_matrices=False)
            _,native_worst=residual_curve(native_v[512:],nb[:5])
            energy=s*s
            row=dict(name=name,settings=center.tolist(),radius=radius,
                     numerical_rank_relative_1e_10=rank,singular_values=s[:15].tolist(),
                     five_vectors_test_worst_percent=float(100*worst[5]),
                     five_vectors_tail_fitting_energy_fraction=float(energy[5:].sum()/energy.sum()),
                     full_basis_test_max_absolute_L2_error=float(error),
                     fifth_over_first_singular=float(s[4]/s[0]),
                     sixth_over_first_singular=float(s[5]/s[0]),
                     native_float32_five_vectors_test_worst_percent=float(100*native_worst[5]),
                     native_float32_sixth_over_first_singular=float(ns[5]/ns[0]),
                     max_shell_distance_error=float(shellerr))
            rows.append(row)
            key=name+'_r_'+str(radius)
            arrays[key+'_settings']=settings;arrays[key+'_images']=x
            arrays[key+'_basis']=b[:rank];arrays[key+'_singular_values']=s[:20]
            print(json.dumps(row),flush=True)
    slopes=[]
    for name in NAMES:
        selected=sorted([r for r in rows if r['name']==name],key=lambda r:r['radius'])
        rr=np.array([r['radius'] for r in selected])
        spec=np.array([r['singular_values'] for r in selected])
        # Both fits contain 512 equal-radius rows. sigma_j/r is the
        # first-order normalized amplitude; positive power means it vanishes.
        powers=np.log((spec[1]/rr[1])/(spec[0]/rr[0]))/np.log(rr[1]/rr[0])
        slopes.append(dict(name=name,normalized_singular_value_powers=powers.tolist(),
                           caveat='Two-scale slopes are diagnostics, not calibrated confidence intervals.'))
    result=dict(method='Ordinary equal-weight uncentered SVD; same direction sampling design at new radii',
                new_results=rows,normalized_spectrum_slopes=slopes,
                prior_results=[{k:v for k,v in row.items() if k in ['name','radius','five_vectors_test_worst_percent',
                    'numerical_rank_relative_1e_10','singular_values']} for row in existing if row['name'] in NAMES])
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(OUT/'smaller_radius_samples.npz',**arrays)


if __name__=='__main__':
    main()
