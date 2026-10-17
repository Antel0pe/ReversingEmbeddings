"""Generic data-fitted continuous piecewise polynomial, with one crease."""
import numpy as np
from scipy.optimize import least_squares
from learn import design


def fit_crease(x,y,starts=16):
    x=np.asarray(x,float);y=np.asarray(y,float)
    center=x.mean(0);scale=np.maximum(np.std(x,axis=0),1e-12)
    u=(x-center)/scale
    flat=y.reshape(len(y),-1)
    base=design(u,3);quad=design(u,2)
    coefficients=np.linalg.lstsq(base,flat,rcond=1e-11)[0]
    residual=flat-base@coefficients
    _,sv,vt=np.linalg.svd(residual,full_matrices=False)
    pattern=vt[0]
    scalar=flat@pattern
    scalar_scale=max(float(np.linalg.norm(residual@pattern)/np.sqrt(len(y))),1e-14)
    # Remove the polynomial subspace once; variable projection then requires
    # only a 21-column least-squares solve, independent of pixel count.
    orth=np.linalg.qr(base,mode='reduced')[0]
    response=(scalar-orth@(orth.T@scalar))/scalar_scale
    def normalize(theta):
        norm=np.linalg.norm(theta[:-1])
        return theta[:-1]/max(norm,1e-12),theta[-1]/max(norm,1e-12)
    def basis(theta):
        v,b=normalize(theta)
        return np.maximum(0,u@v-b)[:,None]*quad
    def objective(theta):
        h=basis(theta)
        h-=orth@(orth.T@h)
        beta=np.linalg.lstsq(h,response,rcond=1e-10)[0]
        return h@beta-response
    rng=np.random.default_rng(2911)
    initial=[np.r_[row,0.] for row in np.eye(x.shape[1])]
    initial.extend(np.r_[v/np.linalg.norm(v),0.] for v in rng.normal(size=(starts,x.shape[1])))
    best=None;history=[]
    for theta in initial:
        result=least_squares(objective,theta,xtol=1e-10,ftol=1e-10,gtol=1e-10,max_nfev=120,
                             diff_step=1e-5)
        loss=float(np.linalg.norm(result.fun))
        history.append(dict(loss=loss,evaluations=int(result.nfev)))
        if best is None or loss<best[0]:best=(loss,result.x.copy())
    v,b=normalize(best[1])
    a=np.column_stack([base,np.maximum(0,u@v-b)[:,None]*quad])
    all_coefficients=np.linalg.lstsq(a,flat,rcond=1e-11)[0]
    return dict(center=center,scale=scale,normal=v,offset=float(b),coefficient=all_coefficients,
        output_shape=y.shape[1:],training_plane_loss=best[0],optimization=history,
        leading_residual_singular=float(sv[0]),residual_rank=int(np.sum(sv>sv[0]*1e-8)))


def predict_crease(x,model):
    u=(np.atleast_2d(x)-model['center'])/model['scale']
    hinge=np.maximum(0,u@model['normal']-model['offset'])
    a=np.column_stack([design(u,3),hinge[:,None]*design(u,2)])
    return (a@model['coefficient']).reshape((len(u),)+model['output_shape'])
