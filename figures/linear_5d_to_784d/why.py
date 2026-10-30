import numpy as np
from grey_ones import render, RANGES, KNOBS
rng = np.random.default_rng(1)
root = np.array([14.0,14.0,19.75,3.2,0.0])

# --- 1. superposition / parallelogram law. Affine f: f(a)+f(b)-f(p0) == f(a+b-p0)
print("superposition test: ||f(a)+f(b)-f(p0) - f(a+b-p0)|| relative to ||f(a+b-p0)-f(p0)||")
lo,hi = RANGES[:,0],RANGES[:,1]
errs=[]
for _ in range(2000):
    a = lo + rng.random(5)*(hi-lo); b = lo + rng.random(5)*(hi-lo)
    c = a+b-root
    fa,fb,fp,fc = render([a,b,root,c], dtype=np.float64)
    pred = fa+fb-fp
    e = np.linalg.norm(pred-fc)/max(np.linalg.norm(fc-fp),1e-12)
    errs.append(e)
errs=np.array(errs)
print(f"  median {np.median(errs)*100:.1f}%  mean {errs.mean()*100:.1f}%  max {errs.max()*100:.1f}%  (affine would be 0%)")

# --- 2. one knob's effect depends on the other knobs (no fixed column exists)
print("\nper-knob column test: pattern of a fixed knob step, measured at different base settings")
for k in range(5):
    step = (hi[k]-lo[k])*0.25
    cols=[]
    for _ in range(300):
        p = lo + rng.random(5)*(hi-lo)
        p[k] = lo[k] + rng.random()*(hi[k]-lo[k]-step)
        q = p.copy(); q[k]+=step
        d = (render([q],dtype=np.float64)[0]-render([p],dtype=np.float64)[0]).ravel()
        cols.append(d)
    C=np.array(cols); n=np.linalg.norm(C,axis=1,keepdims=True)
    U=C/n
    cos = U@U.T
    iu = np.triu_indices(len(U),1)
    print(f"  {KNOBS[k]:>6}: step {step:g}  cosine between pairs of its own effect patterns:"
          f" min {cos[iu].min():.3f} median {np.median(cos[iu]):.3f}   length ratio max/min {n.max()/n.min():.2f}")

# --- 3. how good can ANY 5-number -> image map be if the last stage is linear
P = lo + rng.random((20000,5))*(hi-lo)
X = render(P, dtype=np.float64).reshape(20000,-1)
mu = X.mean(0); s = np.linalg.svd(X-mu, compute_uv=False)
frac = np.cumsum(s**2)/(s**2).sum()
print("\nceiling for any map that ends in '5 numbers -> weights on 5 fixed images':")
print(f"  best 5-dim affine subspace keeps {frac[4]*100:.4f}% of image variance")
print(f"  -> residual is {np.sqrt(1-frac[4])*100:.2f}% of the signal, no matter how clever the 5 numbers are")

# --- 4. contrast: polynomial features (still a linear fit, nonlinear features)
from itertools import combinations_with_replacement as cwr
def poly(P,deg):
    cols=[np.ones(len(P))]
    for d in range(1,deg+1):
        for idx in cwr(range(5),d):
            c=np.ones(len(P))
            for i in idx: c=c*P[:,i]
            cols.append(c)
    return np.stack(cols,1)
Pte = lo + rng.random((5000,5))*(hi-lo)
Xte = render(Pte, dtype=np.float64).reshape(5000,-1)
print("\nsame fit with polynomial features of the 5 knobs (test set):")
for deg in (1,2,3,4,6):
    A=poly(P,deg); W,*_=np.linalg.lstsq(A,X,rcond=None)
    R=poly(Pte,deg)@W-Xte
    r2=1-(R**2).sum()/((Xte-mu)**2).sum()
    print(f"  degree {deg}: {A.shape[1]:4d} features  R2={r2:.6f}  max|pix err|={np.abs(R).max():.4f}")
