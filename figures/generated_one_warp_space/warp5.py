import sys, pathlib
_HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE.parents[1]), str(_HERE)]
import numpy as np, itertools
d=np.load(str(_HERE / "w4.npz"))
TH,TX,P=d["TH"],d["TX"],d["P"]

print("A) closed-form knobs from the 5 affine coefficients -- EXACT coefficients (sanity)")
def inv(T):
    rx,rxs,x1 = T[:,1],T[:,2],T[:,0]
    ry,y1     = T[:,8],T[:,6]
    s = rxs/rx; w = 3.2/rx; h = 19.75/ry
    cyn = -y1/ry; cy = 14+14*cyn
    cxn = -x1/rx - s*cyn; cx = 14+14*cxn
    return np.stack([cx,cy,h,w,np.degrees(np.arctan(s))],1)
for tag,T in [("exact coeffs",TX),("FITTED coeffs (data only)",TH)]:
    e=np.abs(inv(T)-P)
    print(f"   {tag:26s} median " + " ".join(f"{a}={b:.4f}" for a,b in
          zip(["cx","cy","h","w","lean"], np.median(e,0))))
    print(f"   {'':26s} max    " + " ".join(f"{a}={b:.4f}" for a,b in
          zip(["cx","cy","h","w","lean"], e.max(0))))

print("\nB) PCA scores on RAW fitted coefficients -> knobs (degree-2, ridge, held out)")
mu=TH.mean(0); A=TH-mu
U,S,Vt=np.linalg.svd(A,full_matrices=False)
tr,te=slice(0,60),slice(60,120)
for k in (4,5,6,7):
    L=A@Vt[:k].T
    cols=[np.ones(len(L))]+[np.prod(L[:,c],1) for dg in (1,2) for c in
          itertools.combinations_with_replacement(range(k),dg)]
    B=np.stack(cols,1); B=B/np.abs(B).max(0)
    lam=1e-6*len(B[tr])
    G=B[tr].T@B[tr]+lam*np.eye(B.shape[1])
    co=np.linalg.solve(G,B[tr].T@P[tr])
    e=np.abs(B[te]@co-P[te]).max(0)
    print(f"   k={k}: held-out max err " + " ".join(f"{a}={b:.3f}" for a,b in
          zip(["cx","cy","h","w","lean"], e)))

print("\nC) the vertical degeneracy, directly")
print("   corr(y<-1 fit error, y<-y2 fit value) =",
      f"{np.corrcoef(TH[:,6]-TX[:,6], TH[:,11])[0,1]:+.3f}")
print("   corr(y<-y fit error, y<-y2 fit value) =",
      f"{np.corrcoef(TH[:,8]-TX[:,8], TH[:,11])[0,1]:+.3f}")
print("   x-component coeff errors (mean):", f"{np.abs(TH[:,:3]-TX[:,:3]).mean():.5f}")
print("   y-component coeff errors (mean):", f"{np.abs(TH[:,[6,8]]-TX[:,[6,8]]).mean():.5f}")
