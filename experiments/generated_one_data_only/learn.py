"""Generic raw-image measurements and local polynomial regression.

No renderer, physical coordinate labels, parameter ranges, inverse program,
subrow count, stroke models, or supplied shape equations occur in this file.
"""
import itertools
import numpy as np
from scipy.spatial import cKDTree

NAMES = ['ink_amount', 'horizontal_ink_center', 'vertical_ink_center',
         'horizontal_spread_squared', 'vertical_spread_squared', 'cross_spread']


def moments(images):
    x = np.asarray(images, dtype=np.float64)
    n, h, w = x.shape
    xx = np.arange(w, dtype=float)[None, None, :] + .5
    yy = np.arange(h, dtype=float)[None, :, None] + .5
    mass = np.sum(x, axis=(1,2))
    if np.any(mass <= 0) or np.any(x < 0):
        raise ValueError('moments require nonnegative nonempty images')
    cx = np.sum(x*xx,axis=(1,2))/mass
    cy = np.sum(x*yy,axis=(1,2))/mass
    vx = np.sum(x*(xx-cx[:,None,None])**2,axis=(1,2))/mass
    vy = np.sum(x*(yy-cy[:,None,None])**2,axis=(1,2))/mass
    cv = np.sum(x*(xx-cx[:,None,None])*(yy-cy[:,None,None]),axis=(1,2))/mass
    return np.column_stack([mass,cx,cy,vx,vy,cv])


def powers(dim, degree):
    result = [()]
    for d in range(1,degree+1):
        result.extend(itertools.combinations_with_replacement(range(dim),d))
    return result


def design(x, degree):
    x = np.atleast_2d(x)
    return np.column_stack([np.prod(x[:,term],axis=1) if term else np.ones(len(x))
                            for term in powers(x.shape[1],degree)])


def error_stats(true,pred):
    delta=np.asarray(pred,dtype=float)-np.asarray(true,dtype=float)
    flat=delta.reshape(len(delta),-1)
    l2=np.linalg.norm(flat,axis=1)
    pixel=np.max(abs(flat),axis=1)
    return dict(max_pixel=float(pixel.max()),max_image_l2=float(l2.max()),
                median_image_l2=float(np.median(l2)),mean_image_l2=float(l2.mean()),
                worst_pixel_index=int(np.argmax(pixel)),worst_l2_index=int(np.argmax(l2)),
                out_of_coverage_pixels=int(np.count_nonzero((pred<0)|(pred>1))))


class LocalPolynomial:
    def __init__(self, x, y, degree=2, neighbors=96):
        self.x=np.asarray(x,dtype=float)
        self.y=np.asarray(y,dtype=float)
        self.degree=degree
        self.neighbors=min(neighbors,len(x))
        self.mean=self.x.mean(0)
        self.scale=np.std(self.x,axis=0)
        self.scale=np.maximum(self.scale,1e-12)
        self.u=(self.x-self.mean)/self.scale
        self.tree=cKDTree(self.u)
        self.output_shape=self.y.shape[1:]
        self.flat=self.y.reshape(len(y),-1)
        # Keep exact data-constant channels; no energy or variance truncation.
        self.constant=np.all(self.flat==self.flat[0],axis=0)
        self.active=np.flatnonzero(~self.constant)

    def predict(self,query,details=False):
        u=(np.atleast_2d(query)-self.mean)/self.scale
        distance,indices=self.tree.query(u,k=self.neighbors)
        result=np.empty((len(u),self.flat.shape[1]))
        result[:,self.constant]=self.flat[0,self.constant]
        conditions=[]
        for j,(point,idx,dist) in enumerate(zip(u,indices,distance)):
            delta=self.u[idx]-point
            bandwidth=max(float(dist[-1]),1e-12)
            a=design(delta/bandwidth,self.degree)
            root_weight=np.exp(-2*(dist/bandwidth)**2)
            weighted=a*root_weight[:,None]
            target=np.zeros(a.shape[1]);target[0]=1
            # A^T w=e0: the minimum-norm evaluation weights of a weighted
            # polynomial least-squares fit. All outputs share those weights.
            weights,_,rank,sv=np.linalg.lstsq(weighted.T,target,rcond=1e-10)
            weights*=root_weight
            result[j,self.active]=weights@self.flat[idx][:,self.active]
            if details:
                conditions.append(dict(rank=int(rank),terms=len(target),
                    farthest_neighbor=bandwidth,weight_l1=float(np.sum(abs(weights))),
                    relative_smallest=float(sv[-1]/sv[0])))
        result=result.reshape((len(u),)+self.output_shape)
        return (result,conditions) if details else result


def global_polynomial(x,y,query,degree):
    center=np.mean(x,axis=0);scale=np.maximum(np.std(x,axis=0),1e-12)
    a=design((x-center)/scale,degree)
    coefficient=np.linalg.lstsq(a,np.asarray(y).reshape(len(y),-1),rcond=1e-10)[0]
    prediction=design((query-center)/scale,degree)@coefficient
    return prediction.reshape((len(query),)+y.shape[1:]),coefficient


def choose_measurements(train_images,validation_images):
    train=moments(train_images);valid=moments(validation_images)
    records=[]
    for omitted in range(6):
        keep=[i for i in range(6) if i!=omitted]
        model=LocalPolynomial(train[:,keep],train_images,degree=2,neighbors=96)
        pred=model.predict(valid[:,keep])
        records.append(dict(omitted=omitted,kept=keep,metrics=error_stats(validation_images,pred)))
    best=min(records,key=lambda r:r['metrics']['mean_image_l2'])
    return best['kept'],records


def choose_hyperparameters(x,y,x_valid,y_valid):
    records=[]
    for degree,neighbors in [(1,48),(2,64),(2,128),(3,128),(3,192)]:
        model=LocalPolynomial(x,y,degree=degree,neighbors=neighbors)
        pred=model.predict(x_valid)
        records.append(dict(degree=degree,neighbors=neighbors,metrics=error_stats(y_valid,pred)))
    return min(records,key=lambda r:r['metrics']['mean_image_l2']),records
