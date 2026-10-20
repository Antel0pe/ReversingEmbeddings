"""Acquire independent small patches; the learner never imports this file."""
from pathlib import Path
import sys
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from grey_ones import render,RANGES
from experiments.svd_blind_intermediates.export import fit_frames

OUT=ROOT/'figures/pooled_family_coordinates'
SPAN=np.ptp(RANGES,axis=1)


def pixels(p):return render(p,dtype=np.float64).reshape(-1,784)


def ball(p,r,n,rng):
    ray=rng.normal(size=(n,5))*SPAN
    ray/=np.linalg.norm(ray/SPAN,axis=1)[:,None]
    x=pixels(p)[0]
    speed=np.linalg.norm(pixels(p+1e-5*ray)-x,axis=1)/1e-5
    target=r*rng.uniform(.05,1,size=n)**.2
    amount=target/np.maximum(speed,1e-12)
    for _ in range(12):
        result=p+amount[:,None]*ray
        distance=np.linalg.norm(pixels(result)-x,axis=1)
        amount*=target/np.maximum(distance,1e-14)
    result=p+amount[:,None]*ray
    error=np.max(abs(np.linalg.norm(pixels(result)-x,axis=1)-target))
    return result,float(error)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    old=np.load('/tmp/svd_blind_intermediates_labels.npz')
    sources=[np.array([14.,14.,19.75,3.2,0.]),old['broad_train_frame_p'][0]]
    rng=np.random.default_rng(2026101901)
    public={};labels={};records=[]
    for si,source in enumerate(sources):
        for ri,r in enumerate([.005,.05,.25,1.]):
            key=f'patch_{si}_{ri}';public[key+'_source_x']=pixels(source)
            labels[key+'_source_p']=source
            errors=[]
            for split,n in [('train',1536),('val',256),('test',256)]:
                p,err=ball(source,r,n,rng);errors.append(err)
                public[key+'_'+split+'_x']=pixels(p);labels[key+'_'+split+'_p']=p
            for split,n in [('train',96),('val',32),('test',32)]:
                p,err=ball(source,r,n,rng);errors.append(err)
                public[key+'_'+split+'_frame_x']=pixels(p);labels[key+'_'+split+'_frame_p']=p
                fits=[fit_frames(point,2026101950+si*10000+ri*1000+i) for i,point in enumerate(p)]
                public[key+'_'+split+'_svd']=np.array([row[0][0] for row in fits])
                public[key+'_'+split+'_pca']=np.array([row[0][1] for row in fits])
            records.append(dict(scope=key,source_index=si,radius=r,max_ball_distance_error=max(errors)))
            print('acquired',key,'radius',r,flush=True)
    np.savez_compressed(OUT/'patch_observations.npz',**public)
    np.savez_compressed('/tmp/pooled_family_coordinates_labels.npz',**labels)
    (OUT/'acquisition.json').write_text(json.dumps(dict(patches=records,
        previous_input='figures/svd_blind_intermediates/observations.npz',
        neighborhood='192 observations per frame at image distance 0.0001; no analytic derivatives',
        physical_settings='Private evaluation archive, not a discovery input'),indent=2))


if __name__=='__main__':main()
