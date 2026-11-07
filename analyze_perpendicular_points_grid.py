"""Why the five-knob ink-position fit stalls: pixel grid or top and bottom ends? Run after analyze_perpendicular_points_fit.py."""
import numpy as np
from grey_ones import render, RANGES
import coverage_grids as cg
# 2. is the 11% floor the fit or the pixel grid?
LO,HI=np.asarray(RANGES,float).T
def feats(p):
    z=(p-LO)/(HI-LO)*2-1
    c=[np.ones(len(z))]+[z[:,i] for i in range(5)]
    c+=[z[:,i]*z[:,j] for i in range(5) for j in range(i,5)]
    c+=[z[:,i]*z[:,j]*z[:,k] for i in range(5) for j in range(i,5) for k in range(j,5)]
    return np.stack(c,1)
U=np.linspace(0,1,65)
def enc(imgs):
    n=imgs.shape[1]; E=np.arange(n+1.0)
    out=np.zeros((len(imgs),n,66))
    for a,im in enumerate(imgs):
        filled=[]
        for r,row in enumerate(im):
            m=row.sum(); out[a,r,0]=m
            if m<=1e-12: continue
            cdf=np.r_[0.,np.cumsum(row)]/m; idx=np.where(np.diff(cdf)>0)[0]
            out[a,r,1:]=np.interp(U,np.r_[cdf[idx],cdf[idx[-1]+1]],np.r_[E[idx],E[idx[-1]+1]]); filled.append(r)
        f=np.array(filled)
        for r in range(n):
            if r not in filled: out[a,r,1:]=out[a,f[np.argmin(np.abs(f-r))],1:]
    return out
def dec(c):
    n=c.shape[1]; E=np.arange(n+1.0); im=np.zeros((len(c),n,n))
    for a in range(len(c)):
        for r in range(n):
            q=np.maximum.accumulate(c[a,r,1:])
            im[a,r]=max(c[a,r,0],0)*np.diff(np.interp(E,q,np.linspace(0,1,len(q)),left=0,right=1))
    return np.clip(im,0,1)
rel=lambda p,t:100*np.linalg.norm(p.reshape(len(t),-1)-t.reshape(len(t),-1),axis=1)/np.linalg.norm(t.reshape(len(t),-1),axis=1)
rng=np.random.default_rng(2)
Pq=LO+(HI-LO)*rng.random((200,5)); Pt=LO+(HI-LO)*rng.random((600,5))
for n_pix,K in [(28,64),(112,64),(112,256)]:
    U=np.linspace(0,1,K+1)
    def enc2(imgs):
        n=imgs.shape[1]; E=np.arange(n+1.0); out=np.zeros((len(imgs),n,K+2))
        for a,im in enumerate(imgs):
            filled=[]
            for r,row in enumerate(im):
                m=row.sum(); out[a,r,0]=m
                if m<=1e-12: continue
                cdf=np.r_[0.,np.cumsum(row)]/m; idx=np.where(np.diff(cdf)>0)[0]
                out[a,r,1:]=np.interp(U,np.r_[cdf[idx],cdf[idx[-1]+1]],np.r_[E[idx],E[idx[-1]+1]]); filled.append(r)
            f=np.array(filled)
            for r in range(n):
                if r not in filled: out[a,r,1:]=out[a,f[np.argmin(np.abs(f-r))],1:]
        return out
    It=np.asarray(cg.render_coverage(Pt,n_pix=n_pix),float); Iq=np.asarray(cg.render_coverage(Pq,n_pix=n_pix),float)
    X,Xq=feats(Pt),feats(Pq)
    Ct,Cq=enc2(It),enc2(Iq)
    Ch=(Xq@np.linalg.lstsq(X,Ct.reshape(len(Pt),-1),rcond=None)[0]).reshape(Cq.shape)
    both=Ch; true_amount=Ch.copy(); true_amount[:,:,0]=Cq[:,:,0]; true_pos=Cq.copy(); true_pos[:,:,0]=Ch[:,:,0]
    print(f"grid {n_pix:3d}, {K} marks: both fitted {rel(dec(both),Iq).mean():5.2f}% | true ink-per-row, fitted positions {rel(dec(true_amount),Iq).mean():5.2f}% | fitted ink-per-row, true positions {rel(dec(true_pos),Iq).mean():5.2f}% | no fit {rel(dec(Cq),Iq).mean():5.2f}%")
