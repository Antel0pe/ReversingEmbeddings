"""Stationary and balanced-neighborhood acquisition; no learner imports this."""
from pathlib import Path
import sys
import json
import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from grey_ones import render,RANGES
OUT=ROOT/'figures/pooled_family_coordinates'
SPAN=np.ptp(RANGES,axis=1)


def pixels(p):return render(p,dtype=np.float64).reshape(-1,784)


def fit(p,n,seed,sobol):
    if sobol:
        u=qmc.Sobol(5,scramble=True,seed=seed).random_base2(int(np.log2(n)))
        ray=ndtri(np.clip(u,1e-10,1-1e-10))
    else:ray=np.random.default_rng(seed).normal(size=(n,5))
    ray/=np.linalg.norm(ray,axis=1)[:,None];ray*=SPAN
    x=pixels(p)[0]
    t=.0001/(np.linalg.norm(pixels(p+1e-5*ray)-x,axis=1)/1e-5)
    for _ in range(4):t*=.0001/np.linalg.norm(pixels(p+t[:,None]*ray)-x,axis=1)
    dx=pixels(p+t[:,None]*ray)-x;active=np.flatnonzero(np.any(dx!=0,axis=0))
    result=[];cov=[]
    for centered in [False,True]:
        a=dx[:,active]-dx[:,active].mean(0) if centered else dx[:,active]
        _,_,vt=np.linalg.svd(a,full_matrices=False)
        b=np.zeros((5,784));b[:,active]=vt[:5];result.append(b)
        c=np.zeros((784,784));c[np.ix_(active,active)]=a.T@a/len(a);cov.append(c)
    return result,cov


def main():
    p=np.load('/tmp/pooled_family_coordinates_labels.npz')
    public={};records={}
    for si in range(2):
        source=p[f'patch_{si}_0_source_p'];allfits={};pooled=[np.zeros((784,784)),np.zeros((784,784))]
        for name,n,count,sobol in [('gaussian_192',192,12,False),('sobol_512',512,12,True),('sobol_2048',2048,6,True)]:
            rows=[]
            for seed in range(count):
                fits,cov=fit(source,n,20261019700+si*100+seed,sobol);rows.append(fits)
                if n==2048:
                    for j in range(2):pooled[j]+=cov[j]/count
            allfits[name]=np.array(rows)
        records[str(si)]={}
        for j,method in enumerate(['svd','pca']):
            active=np.flatnonzero(np.diag(pooled[j])>0)
            ev,vec=eigh(pooled[j][np.ix_(active,active)])
            reference=np.zeros((5,784));reference[:,active]=vec[:,-5:][:,::-1].T
            records[str(si)][method]={}
            for name,frames in allfits.items():
                aligned=[]
                for b in frames[:,j]:
                    dot=reference@b.T;row,col=linear_sum_assignment(-abs(dot));order=col[np.argsort(row)]
                    aligned.append(b[order]*np.where(dot[np.arange(5),order]<0,-1.,1.)[:,None])
                aligned=np.array(aligned);response=aligned@reference.T
                gain=np.diagonal(response,axis1=1,axis2=2).mean(0)
                response/=gain[None,None,:]
                mask=~np.eye(5,dtype=bool)
                angle=np.rad2deg(np.arccos(np.clip(np.diagonal(aligned@reference.T,axis1=1,axis2=2),-1,1)))
                residual=aligned-(aligned@reference.T)@reference
                records[str(si)][method][name]=dict(off_rms=float(np.sqrt(np.mean(response[:,mask]**2))),
                    median_angle_degrees=float(np.median(angle)),p95_angle_degrees=float(np.quantile(angle,.95)),
                    mean_subspace_residual=float(np.mean(np.linalg.norm(residual,axis=2))))
        print('stationary source',si,records[str(si)]['svd'],flush=True)
    (OUT/'stationary_quality.json').write_text(json.dumps(records,indent=2))
    for si in range(2):
        for ri in [0,2]:
            key=f'patch_{si}_{ri}'
            for split in ['train','val','test']:
                points=p[key+'_'+split+'_frame_p'];rows=[]
                for i,point in enumerate(points):
                    fits,_=fit(point,512,20261019900+si*10000+ri*1000+['train','val','test'].index(split)*100+i,True)
                    rows.append(fits)
                rows=np.array(rows)
                public[key+'_'+split+'_svd']=rows[:,0];public[key+'_'+split+'_pca']=rows[:,1]
            print('balanced patch',key,flush=True)
    np.savez_compressed(OUT/'balanced_frames.npz',**public)


if __name__=='__main__':main()
