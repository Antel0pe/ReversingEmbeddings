"""Independent checker: NumPy + saved raw arrays; no discovery imports.

Run: .venv/bin/python experiments/generated_one_explanation/check.py
Add --oracle for fresh comparisons to the repository's actual renderer.
The decoder and image encoder below are separately implemented.
"""
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import sys
import numpy as np
BASE = Path(__file__).resolve().parents[2]
DEFAULT = BASE/'results/generated_one_explanation'
MONOMIALS = list(itertools.combinations_with_replacement(range(5),2))


def read_geometry(images):
    result=[]
    for image in images:
        im=np.asarray(image,dtype=float)
        row=np.sum(im,axis=-1)
        support=np.flatnonzero(row>0)
        first,last=int(support[0]),int(support[-1])
        width=float(np.median(row[first+1:last]))
        t=first+1-row[first]/width
        b=last+row[last]/width
        c=[]
        for r in [first+1,last-1]:
            prefix=np.array([np.sum(im[r,:j]) for j in range(29)])
            k=int(np.argmin(abs(prefix-width/2)))
            c.append(k-prefix[k]+width/2)
        slope=(c[0]-c[1])/(last-first-2)
        result.append([t,b,c[0]+slope*(first+1.5),width,slope])
    return np.array(result)


def poly(q):
    return np.column_stack([q]+[q[:,i]*q[:,j] for i,j in MONOMIALS])


def coordinates(images,coeff):
    return poly(read_geometry(images))@coeff.T


def inverse(z,coeff):
    matrix=coeff[:,:5]
    temp=np.linalg.solve(matrix,np.asarray(z).T).T
    extra=poly(temp)[:,5:]@coeff[:,5:].T
    return np.linalg.solve(matrix,(z-extra).T).T


def raster(geometry,count):
    """Independent, one-image implementation, using rowwise reductions."""
    result=[]
    leftcol=np.arange(28)[None,:]
    y=np.arange(28*count)/count
    for top,bottom,intercept,width,slope in geometry:
        vertical=np.maximum(0,np.minimum(y+1/count,bottom)-np.maximum(y,top))
        center=intercept-slope*(y+.5/count)
        left=center-width/2; right=center+width/2
        # Subtract the two cumulative coverages rather than clipping R-L.
        cumulative_right=np.clip(right[:,None]-leftcol,0,1)
        cumulative_left=np.clip(left[:,None]-leftcol,0,1)
        strips=(cumulative_right-cumulative_left)*vertical[:,None]
        result.append(strips.reshape(28,count,28).sum(axis=1))
    return np.array(result)


def errors(true,pred):
    d=np.asarray(pred,float)-np.asarray(true,float)
    norms=np.linalg.norm(d.reshape(len(d),-1),axis=1)
    return dict(max_pixel=float(np.max(abs(d))),max_image_l2=float(np.max(norms)),
                worst_l2_index=int(np.argmax(norms)))


def require(m,dtype):
    a,b=(5e-11,5e-10) if dtype=='float64' else (2e-6,1e-5)
    assert m['max_pixel']<=a and m['max_image_l2']<=b,m


def verify_interventions(folder,coeff,count,dtype):
    with np.load(folder/f'interventions_{dtype}_public.npz') as data:
        z=coordinates(data['source'],coeff); donor=coordinates(data['donor'],coeff)
        records=[]
        for i,index in enumerate(data['action_indices']):
            requested=z.copy(); requested[:,index]=donor[:,index]
            pred=raster(inverse(requested,coeff),count)
            m=errors(data['target'][i],pred);require(m,dtype);records.append(m)
        return records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--results',type=Path,default=DEFAULT)
    parser.add_argument('--oracle',action='store_true')
    args=parser.parse_args();folder=args.results
    raw=(folder/'model.json').read_bytes(); model=json.loads(raw)
    summary=json.loads((folder/'results.json').read_text())
    assert hashlib.sha256(raw).hexdigest()==summary['model_sha256']
    coeff=np.array(model['coefficients']);count=int(model['subrows'])
    assert count>0 and coeff.shape==(5,20)
    assert np.linalg.matrix_rank(coeff[:,:5])==5
    records={}
    for dtype in ('float64','float32'):
        with np.load(folder/f'test_{dtype}_public.npz') as data:
            images=data['images']
            z=coordinates(images,coeff)
            q=inverse(z,coeff)
            pred=raster(q,count)
            m=errors(images,pred);require(m,dtype)
            assert np.max(abs(inverse(coordinates(pred,coeff),coeff)-q))<1e-5
            records[dtype]=dict(n=len(images),**m)
            # Ground-truth comparison is evaluation only, never a fitted mixing.
            with np.load(folder/'test_oracle.npz') as oracle:
                p=oracle['settings'].copy()
            p[:,4]=np.tan(np.deg2rad(p[:,4]))
            alignment=[]
            for cal in summary['controls']:
                value=(z[:,cal['z_index']]-cal['offset'])/cal['scale']
                e=float(np.max(abs(value-p[:,cal['physical_index']])))
                assert e < (1e-10 if dtype=='float64' else 2e-6)
                alignment.append(e)
            records[dtype]['coordinate_errors_in_evaluation_order']=alignment
        records[dtype]['intervention_copies']=verify_interventions(folder,coeff,count,dtype)
    # Reject a known out-of-family image with straight average row geometry.
    with np.load(folder/'counterexamples.npz') as data:
        for key in ('collision_before','collision_after'):
            assert data[key].shape==(2,28,28)
        assert np.array_equal(data['collision_before'][0],data['collision_before'][1])
        assert not np.array_equal(data['collision_after'][0],data['collision_after'][1])
        mix=data['invalid_average']
        invalid=errors(mix,raster(inverse(coordinates(mix,coeff),coeff),count))
        assert invalid['max_pixel']>2e-6
        records['invalid_average_residual']=invalid
    if args.oracle:
        sys.path.insert(0,str(BASE))
        from grey_ones import render,RANGES
        rng=np.random.default_rng(20261017)
        unit=np.r_[rng.random((1024,5)),np.array(list(itertools.product([0.,1.],repeat=5)))]
        p=RANGES[:,0]+unit*np.diff(RANGES,axis=1).ravel()
        x=render(p,dtype=np.float64)
        pred=raster(inverse(coordinates(x,coeff),coeff),count)
        records['fresh_oracle']=errors(x,pred);require(records['fresh_oracle'],'float64')
        # Every fresh coordinate chosen inside the evaluator's support box.
        target=p.copy();target[:,4]=np.tan(np.deg2rad(target[:,4]))
        z=np.empty_like(target)
        for cal in summary['controls']:
            z[:,cal['z_index']]=cal['scale']*target[:,cal['physical_index']]+cal['offset']
        m=errors(x,raster(inverse(z,coeff),count));require(m,'float64')
        records['fresh_decoded_domain']=m
    output=dict(status='PASS',model_sha256=summary['model_sha256'],checks=records)
    (folder/'checker_results.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))

if __name__=='__main__':
    main()
