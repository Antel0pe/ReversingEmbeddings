"""One five-dimensional elastic principal manifold with a learned pixel grid.

Initial coordinates and grid allocation use known knobs. Subsequent fitting
alternates learned grid values and pixel-only continuous-coordinate projection.
The decoder is five-dimensional cell interpolation, without renderer equations.
"""
import argparse
import itertools
import json
import os
import time
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from grey_ones import KNOBS, RANGES
from analyze_generated_one_principal_curves import pixels, settings, error_stats

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'figures/generated_one_principal_curves/followup/models_and_examples.npz'
OUT=ROOT/'figures/generated_one_principal_curves/manifold'
BITS=np.array(list(itertools.product([0,1],repeat=5)))
GRIDS={'uniform':(5,5,5,5,5),'guided':(3,3,3,5,33),'fine':(3,3,3,9,65),
       'caps':(5,5,3,5,33)}


def basis(u,shape,derivative=None):
    shape=np.array(shape,dtype=int)
    s=np.clip(u,0,1)*(shape-1)
    j=np.minimum(s.astype(int),shape-2); f=s-j
    stride=np.r_[np.cumprod(shape[:0:-1])[::-1],1]
    index=np.sum((j[:,None,:]+BITS[None,:,:])*stride,axis=2)
    factors=np.where(BITS[None,:,:],f[:,None,:],1-f[:,None,:])
    if derivative is not None:
        factors[:,:,derivative]=(2*BITS[:,derivative]-1)*(shape[derivative]-1)
    weight=np.prod(factors,axis=2)
    rows=np.repeat(np.arange(len(u)),32)
    b=sparse.coo_matrix((weight.ravel(),(rows,index.ravel())),shape=(len(u),np.prod(shape))).tocsr()
    b.eliminate_zeros()
    return b


def decode(u,nodes,shape):
    return basis(u,shape)@nodes


def regularizer(shape):
    """Discrete bending in each coordinate and a weak stretching term."""
    total=int(np.prod(shape)); bend=sparse.csr_matrix((total,total)); stretch=bend.copy()
    for axis,n in enumerate(shape):
        pieces=[]; edges=[]
        for k,m in enumerate(shape):
            if axis==k:
                d2=sparse.diags([np.ones(m-2),-2*np.ones(m-2),np.ones(m-2)],[0,1,2],shape=(m-2,m),format='csr')
                d1=sparse.diags([-np.ones(m-1),np.ones(m-1)],[0,1],shape=(m-1,m),format='csr')
                pieces.append(d2);edges.append(d1)
            else:
                pieces.append(sparse.eye(m,format='csr'));edges.append(sparse.eye(m,format='csr'))
        a=pieces[0]; e=edges[0]
        for p,q in zip(pieces[1:],edges[1:]):a=sparse.kron(a,p,format='csr');e=sparse.kron(e,q,format='csr')
        # Approximate normalized coordinate-space derivative energy per grid node.
        bend += ((n-1)**4/total)*(a.T@a)
        stretch += ((n-1)**2/total)*(e.T@e)
    return bend+1e-3*stretch


def batched_cg(a,rhs,initial=None,maxiter=200,tol=3e-6):
    """Independent Jacobi-preconditioned CG solves for all pixel columns."""
    x=np.zeros_like(rhs) if initial is None else initial.copy()
    r=rhs-a@x
    diagonal=np.maximum(a.diagonal(),1e-12)[:,None]
    z=r/diagonal; p=z.copy(); rz=np.sum(r*z,axis=0)
    norm=np.maximum(np.linalg.norm(rhs,axis=0),1e-12)
    for iteration in range(maxiter):
        ap=a@p
        denominator=np.sum(p*ap,axis=0)
        alpha=np.divide(rz,denominator,out=np.zeros_like(rz),where=denominator>1e-25)
        x+=p*alpha; r-=ap*alpha
        relative=np.linalg.norm(r,axis=0)/norm
        if np.max(relative)<tol:break
        z=r/diagonal; next_rz=np.sum(r*z,axis=0)
        beta=np.divide(next_rz,rz,out=np.zeros_like(rz),where=rz>1e-30)
        p=z+p*beta
        done=relative<tol
        p[:,done]=0; next_rz[done]=0
        rz=next_rz
    return x,dict(iterations=iteration+1,max_relative_normal_equation_residual=float(relative.max()),
                  pixel_columns_not_converged=int(np.sum(relative>=tol)))


def fit_nodes(x,u,shape,reg,penalty,initial=None,maxiter=200):
    b=basis(u,shape)
    lhs=(b.T@b+penalty*reg+1e-7*sparse.eye(b.shape[1])).tocsr()
    rhs=b.T@x
    if initial is None:
        initial=rhs/np.maximum(np.asarray(b.sum(0)).ravel(),1e-12)[:,None]
    return batched_cg(lhs,rhs,initial,maxiter=maxiter)


def project(x,nodes,shape,u,steps=12):
    """Approximate closest-point search with bounded damped Gauss–Newton steps."""
    u=np.clip(u.copy(),0,1); pred=decode(u,nodes,shape)
    loss=np.sum((x-pred)**2,axis=1); history=[float(loss.mean())]
    for iteration in range(steps):
        jac=np.stack([basis(u,shape,k)@nodes for k in range(5)],axis=2)
        gram=np.einsum('nik,nil->nkl',jac,jac)
        damping=np.maximum(np.trace(gram,axis1=1,axis2=2)/5,1e-10)*1e-5
        gram+=damping[:,None,None]*np.eye(5)
        rhs=np.einsum('nik,ni->nk',jac,x-pred)
        delta=np.linalg.solve(gram,rhs[:,:,None])[:,:,0]
        # Limit one step to one grid cell in each direction.
        ratio=np.max(np.abs(delta)*(np.array(shape)-1),axis=1)
        delta/=np.maximum(ratio,1)[:,None]
        start=u.copy();before=float(loss.sum())
        for factor in [1.,.5,.25,.1]:
            proposal=np.clip(start+factor*delta,0,1)
            pp=decode(proposal,nodes,shape);ll=np.sum((x-pp)**2,axis=1)
            keep=ll<loss-1e-12
            u[keep]=proposal[keep];pred[keep]=pp[keep];loss[keep]=ll[keep]
        history.append(float(loss.mean()))
        assert history[-1]<=history[-2]+1e-10
        if before-loss.sum()<1e-7*max(before,1):break
    return pred,u,history


def score(x,pred):
    result=error_stats(x-pred,x,x.mean(0))
    result['rms_pixel_error']=float(np.sqrt(np.sum((x-pred)**2)/(len(x)*784)))
    result['outside_coverage_range_fraction']=float(np.sum((pred<0)|(pred>1))/(len(x)*784))
    return result


def optimize(x,u,shape,stages,initial=None,name='manifold',cg_iterations=180):
    reg=regularizer(shape); nodes=initial; best=None;best_loss=np.inf;history=[]
    if initial is not None:
        best_loss=float(np.mean(np.sum((x-decode(u,nodes,shape))**2,axis=1)))
        best=(nodes.copy(),u.copy())
    started=time.monotonic()
    for penalty,cycles in stages:
        stale=0
        for cycle in range(cycles):
            nodes,cg=fit_nodes(x,u,shape,reg,penalty,nodes,maxiter=cg_iterations)
            direct=decode(u,nodes,shape)
            pred,new_u,ph=project(x,nodes,shape,u,steps=5)
            loss=float(np.mean(np.sum((x-pred)**2,axis=1)))
            stat=score(x,pred)
            row=dict(penalty=penalty,cycle=cycle+1,metrics=stat,solver=cg,
                before_coordinate_update_sse=float(np.mean(np.sum((x-direct)**2,axis=1))),
                projection_history=ph)
            history.append(row);u=new_u
            if loss<best_loss-1e-9:
                best_loss=loss;best=(nodes.copy(),u.copy());stale=0
            else:stale+=1
            print(f'{name} shape {shape}: penalty {penalty:g}, cycle {cycle+1}, '
                  f'capture {stat["explained_percent"]:.6f}%, CG {cg["iterations"]} '
                  f'residual {cg["max_relative_normal_equation_residual"]:.2g}, '
                  f'elapsed {time.monotonic()-started:.1f}s',flush=True)
            if stale>=4:break
        nodes,u=(v.copy() for v in best)
    pred,u,ph=project(x,nodes,shape,u,steps=20)
    return nodes,u,pred,history


def encode_pixels(x,nodes,shape,reference_x,reference_u,starts=2,steps=20):
    _,nearest=cKDTree(reference_x).query(x,k=starts,workers=4)
    if starts==1:nearest=nearest[:,None]
    best_pred=None;best_u=None;best_loss=np.full(len(x),np.inf)
    for k in range(starts):
        pred,u,h=project(x,nodes,shape,reference_u[nearest[:,k]],steps=steps)
        loss=np.sum((x-pred)**2,axis=1)
        keep=loss<best_loss
        if best_pred is None:best_pred=pred.copy();best_u=u.copy()
        else:best_pred[keep]=pred[keep];best_u[keep]=u[keep]
        best_loss[keep]=loss[keep]
    return best_pred,best_u


def local_rank(nodes,shape,u):
    jac=np.stack([basis(u,shape,k)@nodes for k in range(5)],axis=2)
    sv=np.linalg.svd(jac,compute_uv=False)
    relative=sv[:,-1]/np.maximum(sv[:,0],1e-20)
    return dict(points=len(u),smallest_singular_value_min=float(sv[:,-1].min()),
        smallest_singular_value_median=float(np.median(sv[:,-1])),
        full_local_rank_fraction_at_relative_tolerance_1e_6=float(np.mean(relative>1e-6)))


def refine_projection(results,x,u0,active):
    """Broaden search for difficult images without changing any learned node."""
    label=results['best_candidate'];model=np.load(OUT/f'{label}.npz')
    nodes=model['nodes'];shape=model['shape'];ref_u=model['coordinates'].copy()
    threshold=.35
    def improve(xx,pred,u,reference_u):
        loss=np.sum((xx-pred)**2,axis=1);ii=np.flatnonzero(loss>threshold**2)
        if len(ii):
            pp,uu=encode_pixels(xx[ii],nodes,shape,x,reference_u,starts=8,steps=60)
            keep=np.sum((xx[ii]-pp)**2,axis=1)<loss[ii]
            pred[ii[keep]]=pp[keep];u[ii[keep]]=uu[keep]
        return pred,u,dict(difficult_images=len(ii),improved_images=int(np.sum(keep)) if len(ii) else 0)
    pred=decode(ref_u,nodes,shape)
    pred,ref_u,diagnostic=improve(x,pred,ref_u,ref_u.copy())
    stored={k:model[k] for k in model.files};stored['coordinates']=ref_u
    np.savez_compressed(OUT/f'{label}.npz',**stored)
    stat=results['candidates'][label];stat['metrics']=score(x,pred)
    stat['projection_refinement']=diagnostic
    stat['coordinate_rms_shift_from_initial']=np.sqrt(np.mean((ref_u-u0)**2,axis=0)).tolist()
    stat['coordinate_knob_spearman']=[[float(spearmanr(ref_u[:,k],u0[:,j]).statistic) for j in range(5)] for k in range(5)]
    stat['local_rank']=local_rank(nodes,shape,ref_u[::max(1,len(ref_u)//512)])
    for name,result in results['sampling_checks'].items():
        path=OUT/f'{name}_examples.npz';data=np.load(path);stored={k:data[k] for k in data.files}
        pp,uu,diagnostic=improve(stored['images'],stored['reconstructed'].copy(),stored['coordinates'].copy(),ref_u)
        stored['reconstructed']=pp;stored['coordinates']=uu;np.savez_compressed(path,**stored)
        result['pixel_only_projection']=score(stored['images'],pp);result['projection_refinement']=diagnostic
        result['local_rank']=local_rank(nodes,shape,uu[::max(1,len(uu)//512)])
        print('Refined projection:',name,json.dumps(result),flush=True)
    results['encoder']='Two nearest pixel-image starts, twenty steps each; if image L2 residual exceeds 0.35, eight nearest-image starts and sixty steps, retaining only improvement'
    (OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=list(GRIDS)+['evaluate','refine'],default='guided')
    parser.add_argument('--cycles',type=int,default=7);parser.add_argument('--cg-iterations',type=int,default=180)
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    s=np.load(SOURCE);p=s['coverage_knobs'];images=pixels(p);active=np.any(images!=0,axis=0);x=images[:,active]
    u0=(p-RANGES[:,0])/np.ptp(RANGES,axis=1)
    path=OUT/'results.json'
    results=json.loads(path.read_text()) if path.exists() else dict(
        method='One continuous five-dimensional elastic principal grid with multilinear cell interpolation',
        reference_method='Elastic principal manifolds, Gorban and Zinovyev; continuous-cell projection variant',
        reference_url='https://arxiv.org/abs/cond-mat/0405648',
        coordinate_dimension=5,manifold_count=1,coverage_points=len(x),knob_order=KNOBS,
        ranges=RANGES.tolist(),metric='Squared Euclidean distance over all 784 coverage pixels',
        generator_use='Images and initial coordinate labels only; absent from learned decoder and projection',
        candidates={})
    if args.phase=='refine':
        refine_projection(results,x,u0,active)
    elif args.phase in GRIDS:
        shape=GRIDS[args.phase];u=u0.copy();initial=None
        if args.phase=='fine' and (OUT/'guided.npz').exists():
            prior=np.load(OUT/'guided.npz')
            grid_u=np.array(list(itertools.product(*[np.linspace(0,1,n) for n in shape])))
            initial=decode(grid_u,prior['nodes'],prior['shape']);u=prior['coordinates'].copy()
        stages=[(.001,2),(.0001,args.cycles)] if initial is not None else [(.01,3),(.001,args.cycles),(.0001,args.cycles)]
        nodes,u,pred,h=optimize(x,u,shape,stages,
                              initial,args.phase,args.cg_iterations)
        np.savez_compressed(OUT/f'{args.phase}.npz',nodes=nodes,shape=shape,coordinates=u,active=active,
                            coverage_knobs=p,reference_images=x)
        direct=decode(u0,nodes,shape)
        results['candidates'][args.phase]=dict(shape=shape,node_count=int(np.prod(shape)),
            metrics=score(x,pred),known_coordinate_metrics=score(x,direct),
            coordinate_knob_spearman=[[float(spearmanr(u[:,k],p[:,j]).statistic) for j in range(5)] for k in range(5)],
            coordinate_rms_shift_from_initial=np.sqrt(np.mean((u-u0)**2,axis=0)).tolist(),
            local_rank=local_rank(nodes,shape,u[::max(1,len(u)//512)]),
            iterations=len(h),max_cg_iterations=args.cg_iterations)
        results['candidates'][args.phase]['penalty_schedule']=stages
        (OUT/f'{args.phase}_history.json').write_text(json.dumps(h,indent=2)+'\n')
        path.write_text(json.dumps(results,indent=2)+'\n')
        print('Fitted candidate:',json.dumps(results['candidates'][args.phase]),flush=True)
    else:
        label=max(results['candidates'],key=lambda k:results['candidates'][k]['metrics']['explained_percent'])
        model=np.load(OUT/f'{label}.npz');nodes=model['nodes'];shape=model['shape'];ref_u=model['coordinates']
        rng=np.random.default_rng(741);mid=(rng.random((2048,5))+rng.random((2048,5)))/2
        corner=np.array(list(itertools.product([0.,1.],repeat=5)))
        probes={'dense_uniform':settings(16384,740),'all_32_corners':RANGES[:,0]+corner*np.ptp(RANGES,axis=1),
                'joint_midpoints':RANGES[:,0]+mid*np.ptp(RANGES,axis=1)}
        results['best_candidate']=label;results['sampling_checks']={}
        for name,pp in probes.items():
            full=pixels(pp);assert not np.any(full[:,~active]);xx=full[:,active]
            pred,uu=encode_pixels(xx,nodes,shape,x,ref_u,starts=2,steps=20)
            direct=decode((pp-RANGES[:,0])/np.ptp(RANGES,axis=1),nodes,shape)
            result=dict(count=len(xx),pixel_only_projection=score(xx,pred),
                        known_coordinates_without_projection=score(xx,direct),
                        local_rank=local_rank(nodes,shape,uu[::max(1,len(uu)//512)]))
            results['sampling_checks'][name]=result
            np.savez_compressed(OUT/f'{name}_examples.npz',knobs=pp,images=xx,reconstructed=pred,coordinates=uu,active=active)
            print('Sampling check:',name,json.dumps(result),flush=True)
        path.write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
