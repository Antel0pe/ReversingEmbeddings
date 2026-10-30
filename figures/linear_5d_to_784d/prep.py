import numpy as np
from grey_ones import render, RANGES, KNOBS
rng = np.random.default_rng(0)
lo,hi = RANGES[:,0],RANGES[:,1]
root = np.array([14.0,14.0,19.75,3.2,0.0])
S = "/tmp/claude-1000/-home-dmmsp-Projects-ReversingEmbeddings/6425139a-2f92-49b6-a2a7-3d48ffbf0cae/scratchpad/"

# fit on full box
n=20000
P = lo + rng.random((n,5))*(hi-lo)
X = render(P, dtype=np.float64).reshape(n,-1)
A = np.hstack([P, np.ones((n,1))])
W,*_ = np.linalg.lstsq(A, X, rcond=None)      # (6,784): rows 0-4 knob columns, row 5 offset

# held-out examples: typical / extreme lean / thin
ex = np.array([[14.5,14.5,19.75,3.2,12.5],
               [14.2,14.8,20.3,2.0,34.0],
               [14.9,14.1,19.1,4.5,-9.0]])
Xex = render(ex, dtype=np.float64)
Pex = (np.hstack([ex,np.ones((3,1))])@W).reshape(3,28,28)

# test metrics
nt=5000
Pt = lo + rng.random((nt,5))*(hi-lo)
Xt = render(Pt, dtype=np.float64).reshape(nt,-1)
Rt = np.hstack([Pt,np.ones((nt,1))])@W - Xt
r2 = 1-(Rt**2).sum()/((Xt-X.mean(0))**2).sum()
pred_t = np.hstack([Pt,np.ones((nt,1))])@W

# spectrum ceiling
s = np.linalg.svd(X-X.mean(0), compute_uv=False)
frac = np.cumsum(s**2)/(s**2).sum()

# accuracy vs neighbourhood scale
half=(hi-lo)/2
scales = np.array([1.0,0.3,0.1,0.03,0.01,0.003,0.001])
r2s=[]; maxe=[]
for f in scales:
    l,h = root-half*f, root+half*f
    Pa = l + rng.random((6000,5))*(h-l); Pb = l + rng.random((2000,5))*(h-l)
    Xa = render(Pa,dtype=np.float64).reshape(-1,784); Xb=render(Pb,dtype=np.float64).reshape(-1,784)
    Wl,*_ = np.linalg.lstsq(np.hstack([Pa,np.ones((len(Pa),1))]),Xa,rcond=None)
    R = np.hstack([Pb,np.ones((len(Pb),1))])@Wl - Xb
    r2s.append(1-(R**2).sum()/((Xb-Xa.mean(0))**2).sum()); maxe.append(np.abs(R).max())

# superposition example
a = np.array([14.1,14.2,19.2,2.0,30.0]); b = np.array([14.9,14.8,20.4,4.4,5.0])
c = a+b-root
fa,fb,fp,fc = render([a,b,root,c],dtype=np.float64)

# state-dependent knob column: +11.25 deg lean at two widths
base1 = np.array([14.5,14.5,19.75,1.9,0.0]); base2=np.array([14.5,14.5,19.75,4.5,0.0])
d1 = render([base1+[0,0,0,0,11.25]],dtype=np.float64)[0]-render([base1],dtype=np.float64)[0]
d2 = render([base2+[0,0,0,0,11.25]],dtype=np.float64)[0]-render([base2],dtype=np.float64)[0]
cos12 = (d1.ravel()@d2.ravel())/np.linalg.norm(d1)/np.linalg.norm(d2)

# single pixel coverage vs lean
leans = np.linspace(-10,35,400)
Pl = np.tile(np.array([14.5,14.5,19.75,3.2,0.0]),(400,1)); Pl[:,4]=leans
Xl = render(Pl,dtype=np.float64)
pixs = [(8,17),(5,18),(20,12)]
curves = np.array([Xl[:,r,c] for r,c in pixs])

np.savez(S+"fig.npz", W=W, ex=ex, Xex=Xex, Pex=Pex, r2=r2,
         max_abs=np.abs(Rt).max(), rms=np.sqrt((Rt**2).mean()),
         pred_min=pred_t.min(), pred_max=pred_t.max(),
         frac_neg=(pred_t<-0.01).mean(), frac_over=(pred_t>1.01).mean(),
         s=s, frac=frac, scales=scales, r2s=np.array(r2s), maxe=np.array(maxe),
         fa=fa,fb=fb,fp=fp,fc=fc,a=a,b=b,c=c,
         d1=d1,d2=d2,cos12=cos12,base1=base1,base2=base2,
         leans=leans,curves=curves,pixs=np.array(pixs), root=root,
         rootimg=render([root],dtype=np.float64)[0])
print("r2",r2,"max",np.abs(Rt).max(),"rms",np.sqrt((Rt**2).mean()))
print("pred range",pred_t.min(),pred_t.max(),"frac<-0.01",(pred_t<-0.01).mean(),"frac>1.01",(pred_t>1.01).mean())
print("frac5",frac[4],"need",int(np.searchsorted(frac,0.999999)+1))
print("scale r2",list(zip(scales,r2s,maxe)))
print("cos12",cos12, "norms",np.linalg.norm(d1),np.linalg.norm(d2))
print("W col norms",np.linalg.norm(W,axis=1))
