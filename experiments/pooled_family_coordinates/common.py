"""Numerical methods on images and anonymous fields only."""
import itertools
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


def align_one(b,reference):
    dot=reference@b.T
    row,col=linear_sum_assignment(-abs(dot));order=col[np.argsort(row)]
    return b[order]*np.where(dot[np.arange(5),order]<0,-1.,1.)[:,None]


def align_frames(initial_x,initial_b,train_x,train_b,queries):
    x=np.concatenate([initial_x,train_x]);b=np.concatenate([initial_b.copy(),train_b.copy()])
    m=len(initial_x);n=len(x);distance=cdist(x,x)
    done=np.arange(n)<m;parent=np.argmin(distance[:m],axis=0)
    best=distance[:m].min(0);best[done]=np.inf
    while not np.all(done):
        i=int(np.argmin(best));b[i]=align_one(b[i],b[parent[i]])
        done[i]=True;best[i]=np.inf
        improve=(~done)&(distance[i]<best)
        parent[improve]=i;best[improve]=distance[i,improve]
    outputs=[b[m:]]
    for qx,qb in queries:
        near=np.argmin(cdist(qx,x),axis=1)
        outputs.append(np.array([align_one(bb,b[j]) for bb,j in zip(qb,near)]))
    _,nearest=cKDTree(x).query(train_x,k=7)
    good=total=0
    for i,nn in enumerate(nearest,start=m):
        for j in nn[1:]:
            if j==parent[i]:continue
            row,col=linear_sum_assignment(-abs(b[i]@b[j].T))
            good+=int(np.sum(col[np.argsort(row)]==np.arange(5)));total+=5
    return outputs,dict(agreement=good/max(total,1),comparisons=total)


def powers(k,degree,minimum=0):
    return [t for d in range(minimum,degree+1) for t in itertools.combinations_with_replacement(range(k),d)]


def design(x,degree):
    return np.column_stack([np.prod(x[:,t],axis=1) if t else np.ones(len(x)) for t in powers(x.shape[1],degree)])


class LocalPolynomial:
    def __init__(self,x,y,degree=2,neighbors=128):
        self.mean=x.mean(0);self.scale=np.maximum(x.std(0),1e-12)
        self.x=(x-self.mean)/self.scale;self.y=y
        self.tree=cKDTree(self.x);self.degree=degree;self.neighbors=min(neighbors,len(x))
        self.active=np.flatnonzero(np.any(y!=y[0],axis=0))

    def predict(self,q):
        u=(q-self.mean)/self.scale;dist,idx=self.tree.query(u,k=self.neighbors)
        result=np.tile(self.y[0],(len(q),1))
        for i,(point,near,d) in enumerate(zip(u,idx,dist)):
            h=max(d[-1],1e-12);a=design((self.x[near]-point)/h,self.degree)
            w=np.exp(-2*(d/h)**2)
            target=np.zeros(a.shape[1]);target[0]=1
            weights=np.linalg.lstsq((a*w[:,None]).T,target,rcond=1e-10)[0]*w
            result[i,self.active]=weights@self.y[near][:,self.active]
        return result


def image_metrics(truth,pred):
    delta=pred-truth;l2=np.linalg.norm(delta,axis=1)
    return dict(mean_l2=float(l2.mean()),median_l2=float(np.median(l2)),
        p95_l2=float(np.quantile(l2,.95)),max_l2=float(l2.max()),max_pixel=float(abs(delta).max()))


def response_metrics(r,gradient_norm=None):
    diag=np.diagonal(r,axis1=1,axis2=2)
    mask=~np.eye(5,dtype=bool)
    s=np.linalg.svd(r,compute_uv=False);ratio=s[:,-1]/np.maximum(s[:,0],1e-30)
    result=dict(off_rms=float(np.sqrt(np.mean(r[:,mask]**2))),
        own_rms_variation=float(np.sqrt(np.mean((diag-1)**2))),
        mean_own=diag.mean(0).tolist(),own_sign_reversal_fraction=float(np.mean(diag<=0)),
        median_rank_ratio=float(np.median(ratio)),minimum_rank_ratio=float(ratio.min()),
        rank_ratio_below_1e_3=float(np.mean(ratio<1e-3)),
        total_rms=float(np.sqrt(np.mean((r-np.eye(5))**2))),
        per_coordinate_off_rms=[float(np.sqrt(np.mean(np.delete(r[:,:,i],i,axis=1)**2))) for i in range(5)])
    if gradient_norm is not None:
        result['gradient_rms']=float(np.sqrt(np.mean(gradient_norm**2)))
        result['gradient_max']=float(np.max(gradient_norm))
    return result


class Features:
    def __init__(self,x,frame_x,frame_b,kind,bandwidth=1.):
        self.kind=kind;self.bandwidth_multiplier=bandwidth
        self.active=np.flatnonzero((x.std(0)>1e-10)|np.any(abs(frame_b)>1e-12,axis=(0,1)))
        self.mean=x[:,self.active].mean(0)
        raw=x[:,self.active]-self.mean
        ev,u=eigh(raw.T@raw/len(raw));order=np.argsort(ev)[::-1]
        ev=np.maximum(ev[order],0);u=u[:,order]
        self.pc=u;self.pc_variance=ev
        k=8 if kind=='cubic' else 12
        k=min(k,int(np.sum(ev>max(ev[0],1e-30)*1e-12)))
        self.projection=u[:,:k]/np.sqrt(np.maximum(ev[:k],1e-20))[None,:]
        self.terms=[];self.centers=np.zeros((0,k));self.h=1.
        if kind in ['quadratic','cubic']:
            self.terms=powers(k,3 if kind=='cubic' else 2,minimum=2)
        if kind=='kernel':
            z=(frame_x[:,self.active]-self.mean)@self.projection
            selected=[0];distance=np.sum((z-z[0])**2,axis=1)
            for _ in range(min(64,len(z))-1):
                j=int(np.argmax(distance));selected.append(j)
                distance=np.minimum(distance,np.sum((z-z[j])**2,axis=1))
            self.centers=z[selected]
            nearest=cdist(z,self.centers).min(1)
            self.h=max(float(np.median(nearest[nearest>1e-8])),.1)*bandwidth

    @property
    def raw_count(self):return len(self.active)

    def nonlinear(self,x,derivative=True):
        z=(x[:,self.active]-self.mean)@self.projection
        if self.kind=='kernel':
            difference=z[:,None,:]-self.centers
            phi=np.exp(-np.sum(difference**2,axis=2)/(2*self.h*self.h))
            if not derivative:return phi,None
            derivative=-phi[:,:,None]*difference/(self.h*self.h)
            return phi,derivative
        if self.terms:
            phi=np.column_stack([np.prod(z[:,t],axis=1) for t in self.terms])
            if not derivative:return phi,None
            derivative=np.zeros((len(x),len(self.terms),z.shape[1]))
            for j,t in enumerate(self.terms):
                for index,a in enumerate(t):
                    other=t[:index]+t[index+1:]
                    derivative[:,j,a]+=np.prod(z[:,other],axis=1)
            return phi,derivative
        return np.zeros((len(x),0)),np.zeros((len(x),0,z.shape[1]))

    def values(self,x):
        phi,_=self.nonlinear(x,derivative=False)
        return np.column_stack([x[:,self.active]-self.mean,phi])

    def derivatives(self,x,b):
        _,d=self.nonlinear(x)
        raw=b[:,:,self.active]
        extra=np.einsum('nfk,nmk->nfm',raw@self.projection,d)
        return np.concatenate([raw,extra],axis=2)

    def penalty(self,x):
        _,d=self.nonlinear(x);p=self.raw_count
        cross=self.projection@d.mean(0).T
        metric=self.projection.T@self.projection
        tail=np.einsum('nmk,njk->mj',d@metric,d,optimize=True)/len(x)
        return np.block([[np.eye(p),cross],[cross.T,tail]])

    def gradient(self,x,c):
        _,d=self.nonlinear(x);p=self.raw_count
        # n x active-pixel x output-coordinate
        return c[:p][None,:,:]+np.einsum('pk,nmk,mj->npj',self.projection,d,c[p:],optimize=True)

    def save(self,prefix,out):
        out[prefix+'_active']=self.active;out[prefix+'_mean']=self.mean
        out[prefix+'_projection']=self.projection;out[prefix+'_centers']=self.centers
        out[prefix+'_h']=np.array(self.h);out[prefix+'_kind']=np.array(self.kind)
        maxdegree=3 if self.kind=='cubic' else 2
        # Explicit multi-indices permit reconstruction without fitting anything.
        terms=np.full((len(self.terms),maxdegree),-1,dtype=int)
        for i,t in enumerate(self.terms):terms[i,:len(t)]=t
        out[prefix+'_terms']=terms


def physical_whitener(penalty):
    scale=np.sqrt(np.maximum(np.diag(penalty),1e-24))
    corr=penalty/scale[:,None]/scale[None,:]
    val,vec=eigh(corr)
    keep=val>max(val.max(),1e-30)*1e-10
    return (vec[:,keep]/np.sqrt(val[keep])[None,:])/scale[:,None]


def ridge_path(d,penalty,lambdas):
    n=len(d);k=d.shape[-1]
    transform=physical_whitener(penalty)
    a=d.reshape(-1,k)@transform/np.sqrt(n)
    target=np.tile(np.eye(5),(n,1))/np.sqrt(n)
    u,s,vt=np.linalg.svd(a,full_matrices=False)
    output=[]
    for lam in lambdas:
        c=transform@((vt.T*(s/(s*s+lam))[None,:])@(u.T@target))
        gain=np.diagonal(d@c,axis1=1,axis2=2).mean(0)
        if np.min(abs(gain))<1e-10:continue
        c/=gain
        output.append((dict(ridge=lam),c))
    return output


def constrained_path(d,penalty,tolerances):
    transform=physical_whitener(penalty);n=len(d);models=[];details=[]
    for i in range(5):
        other=np.delete(d,i,axis=1).reshape(-1,d.shape[-1])@transform
        covariance=other.T@other/len(other)
        ev,u=eigh(covariance);ev=np.maximum(ev,0)
        basis=transform@u;m=basis.T@d[:,i].mean(0)
        # A finite large multiplier approximates the attainable floor; report it.
        def solve(alpha):
            weights=m/(1+alpha*ev)
            denominator=float(m@weights)
            if denominator<1e-30:return None,np.inf
            weights/=denominator
            off=float(np.sqrt(np.sum(ev*weights*weights)))
            return weights,off
        limit=max(float(ev.max()),1e-15)
        _,floor=solve(1e16/limit)
        entries=[]
        for tol in tolerances:
            weights,off=solve(0.)
            if off>tol:
                if floor>tol*(1+1e-6):entries.append((None,dict(tolerance=tol,feasible=False,floor=floor)));continue
                lo=0.;hi=1./limit
                while solve(hi)[1]>tol and hi<1e17/limit:hi*=10
                for _ in range(60):
                    mid=(lo+hi)/2
                    if solve(mid)[1]>tol:lo=mid
                    else:hi=mid
                weights,off=solve(hi)
            c=basis@weights
            off=float(np.sqrt(np.mean((np.delete(d,i,axis=1)@c)**2)))
            entries.append((c,dict(tolerance=tol,feasible=True,train_off=off,floor=floor)))
        models.append(entries);details.append(dict(family=i+1,minimum_train_off=floor))
    for j,tol in enumerate(tolerances):
        entries=[models[i][j] for i in range(5)]
        metadata=dict(tolerance=tol,coordinates=[e[1] for e in entries],feasible=all(e[0] is not None for e in entries))
        coefficient=np.column_stack([e[0] for e in entries]) if metadata['feasible'] else None
        yield metadata,coefficient


def decoder_selection(q,x,qv,xv,queries):
    records=[]
    for degree,neighbors in [(1,48),(2,64),(2,128),(3,128),(3,192)]:
        model=LocalPolynomial(q,x,degree,neighbors)
        records.append(dict(degree=degree,neighbors=neighbors,
                            validation=image_metrics(xv,model.predict(qv))))
    best=min(records,key=lambda row:row['validation']['mean_l2'])
    model=LocalPolynomial(q,x,best['degree'],best['neighbors'])
    return best,records,{key:model.predict(value) for key,value in queries.items()}
