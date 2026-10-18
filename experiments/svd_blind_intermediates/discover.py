"""Image/vector-only discovery. No simulator or previous coordinate imports."""
from pathlib import Path
import sys
import os
import json
import hashlib
import itertools
import time
import numpy as np
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'figures/svd_blind_intermediates'
SOURCE = Path(__file__).resolve()
INPUT = OUT / 'observations.npz'
APPROVED = {SOURCE, INPUT, SOURCE.with_name('PROTOCOL.md')}
ACCESS = set()


def audit(event, args):
    if event != 'open' or not isinstance(args[0], (str, bytes, Path)): return
    p = Path(os.fsdecode(args[0])).resolve()
    mode, flags = args[1:3]
    writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or bool(flags & (os.O_WRONLY | os.O_RDWR))
    if writing: return
    if 'svd_blind_intermediates_labels' in p.name:
        raise PermissionError('Ground-truth archive forbidden during discovery')
    if p.is_relative_to(ROOT) and not p.is_relative_to(ROOT / '.venv'):
        if p not in APPROVED and not (p.parent == OUT and p.name in ['discovered_models.npz', 'blind_results.json']):
            raise PermissionError('Unapproved repository read: ' + p.name)
        ACCESS.add(str(p.relative_to(ROOT)))


def align_one(b, reference):
    dots = reference @ b.T
    row, col = linear_sum_assignment(-abs(dots))
    order = col[np.argsort(row)]
    return b[order] * np.where(dots[np.arange(5), order] < 0, -1., 1.)[:, None]


def align_frames(initial_x, initial_b, train_x, train_b, query):
    x = np.concatenate([initial_x, train_x])
    b = np.concatenate([initial_b.copy(), train_b.copy()])
    m = len(initial_x); n = len(x)
    d = cdist(x, x)
    visited = np.arange(n) < m
    parent = np.argmin(d[:m], axis=0)
    best = np.min(d[:m], axis=0); best[visited] = np.inf
    while not np.all(visited):
        j = int(np.argmin(best)); p = int(parent[j])
        b[j] = align_one(b[j], b[p]); visited[j] = True; best[j] = np.inf
        improve = (~visited) & (d[j] < best)
        parent[improve] = j; best[improve] = d[j, improve]
    outputs = [b[m:]]
    for qx, qb in query:
        nearest = np.argmin(cdist(qx, x), axis=1)
        outputs.append(np.array([align_one(v, b[k]) for v,k in zip(qb, nearest)]))
    _, idx = cKDTree(x).query(x[m:], k=7)
    correct = 0; total = 0
    for i, near in enumerate(idx, start=m):
        for j in near[1:]:
            if j == parent[i]: continue
            row,col = linear_sum_assignment(-abs(b[i] @ b[j].T))
            correct += int(np.sum(col[np.argsort(row)] == np.arange(5))); total += 5
    return outputs, dict(non_tree_family_agreement=correct/max(total,1), comparisons=total)


class Dictionary:
    def __init__(self, x, normalized, standardize=True):
        self.active = np.flatnonzero(np.std(x,axis=0) > 1e-8)
        self.normalized = normalized
        f = self.features(x)
        self.mean = f.mean(0)
        self.scale = np.maximum(f.std(0), f.std(0).max()*1e-6)
        if not standardize: self.scale=np.ones(f.shape[1])
        a = (f-self.mean)/self.scale
        ev,u = eigh(a.T @ a / len(a))
        order = np.argsort(ev)[::-1]
        ev=ev[order];u=u[:,order]
        k=min(96,int(np.sum(ev > ev[0]*1e-12)))
        self.project = u[:,:k] / np.sqrt(ev[:k])[None,:]

    def features(self,x):
        a=np.atleast_2d(x)[:,self.active]
        return np.concatenate([a,a/np.sum(x,axis=1)[:,None]],axis=1) if self.normalized else a

    def transform(self,x,k=None):
        return ((self.features(x)-self.mean)/self.scale) @ self.project[:,:k]

    def directions(self,x,b,k=None):
        a=b[:,:,self.active]
        if self.normalized:
            mass=x.sum(1)
            da=a/mass[:,None,None]-x[:,None,self.active]*b.sum(2)[:,:,None]/mass[:,None,None]**2
            a=np.concatenate([a,da],axis=2)
        return (a/self.scale) @ self.project[:,:k]


def sensitivity(d):
    energy=np.mean(d*d,axis=0) # family x coordinate
    return energy/np.maximum(energy.sum(0,keepdims=True),1e-30)


def rank_metrics(d):
    sv=np.linalg.svd(d,compute_uv=False)
    ratio=sv[:,-1]/np.maximum(sv[:,0],1e-30)
    return dict(minimum_ratio=float(ratio.min()),median_ratio=float(np.median(ratio)),
                fraction_ratio_below_1e_4=float(np.mean(ratio<1e-4)))


def choose_functions(u, uv, d, dv, ridge):
    energies=np.einsum('nfi,nfj->fij',d,d)/len(d)
    total=energies.sum(0); size=d.shape[2]
    regularizer=ridge*np.trace(total)/size*np.eye(size)
    candidates=[]; leaks=[]
    for f in range(5):
        _,v=eigh(total-energies[f]+regularizer,
                 energies[f]+np.eye(size)*np.trace(total)/size*1e-10,
                 subset_by_index=[0,2])
        v/=np.maximum(np.linalg.norm(v,axis=0),1e-30)
        response=dv@v
        e=np.mean(response*response,axis=0)
        candidates.append(v)
        leaks.append(1-e[f]/np.maximum(e.sum(0),1e-30))
    best=None
    for choices in itertools.product(range(3),repeat=5):
        c=np.column_stack([candidates[f][:,choices[f]] for f in range(5)])
        qv=uv@c
        covariance=np.corrcoef(qv,rowvar=False)
        sign,ld=np.linalg.slogdet(covariance)
        dependence=-ld/5 if sign>0 else 100.
        leak=float(np.mean([leaks[f][choices[f]] for f in range(5)]))
        score=leak+.04*dependence
        if best is None or score<best[0]: best=(score,c,leak,dependence,choices)
    score,c,leak,dependence,choices=best
    # Deterministic sign only; labels are still anonymous family IDs.
    for j in range(5):
        if c[np.argmax(abs(c[:,j])),j]<0: c[:,j]*=-1
    return c,dict(score=score,validation_leakage=leak,dependence=dependence,
                  eigenfunction_indices=list(choices))


def terms(dim, degree):
    return [()] + [t for d in range(1,degree+1) for t in itertools.combinations_with_replacement(range(dim),d)]


def design(x,degree):
    return np.column_stack([np.prod(x[:,t],axis=1) if t else np.ones(len(x)) for t in terms(x.shape[1],degree)])


class Polynomial:
    def __init__(self,x,y,degree=2,neighbors=128):
        self.mean=x.mean(0);self.scale=np.maximum(x.std(0),1e-10)
        self.x=(x-self.mean)/self.scale;self.y=y
        self.tree=cKDTree(self.x);self.degree=degree;self.neighbors=neighbors
        self.constant=np.all(y==y[0],axis=0);self.active=np.flatnonzero(~self.constant)

    def predict(self,query):
        u=(query-self.mean)/self.scale
        distance,indices=self.tree.query(u,k=min(self.neighbors,len(self.x)))
        prediction=np.tile(self.y[0],(len(query),1))
        for i,(q,idx,dist) in enumerate(zip(u,indices,distance)):
            h=max(dist[-1],1e-12)
            a=design((self.x[idx]-q)/h,self.degree)
            w=np.exp(-2*(dist/h)**2)
            target=np.zeros(a.shape[1]);target[0]=1
            weights=np.linalg.lstsq((a*w[:,None]).T,target,rcond=1e-10)[0]*w
            prediction[i,self.active]=weights@self.y[idx][:,self.active]
        return prediction


def metrics(truth,pred):
    delta=pred-truth;l2=np.linalg.norm(delta,axis=1)
    return dict(mean_l2=float(l2.mean()),median_l2=float(np.median(l2)),
        p95_l2=float(np.quantile(l2,.95)),max_l2=float(l2.max()),max_pixel=float(abs(delta).max()),
        exact_threshold_pass=bool(l2.max()<=1e-5 and abs(delta).max()<=2e-6))


def decode(x,q,qv,yv,queries):
    records=[]
    for degree,neighbors in [(1,48),(2,64),(2,128),(3,128),(3,192)]:
        model=Polynomial(q,x,degree,neighbors)
        record=dict(degree=degree,neighbors=neighbors,validation=metrics(yv,model.predict(qv)))
        records.append(record)
    choice=min(records,key=lambda r:r['validation']['mean_l2'])
    model=Polynomial(q,x,choice['degree'],choice['neighbors'])
    pred={k:model.predict(v) for k,v in queries.items()}
    mean=q.mean(0);scale=np.maximum(q.std(0),1e-10)
    coefficient=np.linalg.lstsq(design((q-mean)/scale,3),x,rcond=1e-10)[0]
    global_predictions={k:design((v-mean)/scale,3)@coefficient for k,v in queries.items()}
    return choice,records,pred,global_predictions,coefficient


def vector_compression(b,bt):
    rows=[]
    for f in range(5):
        _,s,vt=np.linalg.svd(b[:,f],full_matrices=False)
        cumulative=np.cumsum(s*s)/np.sum(s*s)
        k95=int(np.searchsorted(cumulative,.95)+1);k99=int(np.searchsorted(cumulative,.99)+1)
        residual=bt[:,f]-(bt[:,f]@vt[:k99].T)@vt[:k99]
        rows.append(dict(patterns_95=k95,patterns_99=k99,
            heldout_mean_vector_error=float(np.mean(np.linalg.norm(residual,axis=1))),
            heldout_max_vector_error=float(np.max(np.linalg.norm(residual,axis=1)))))
    return rows


def main():
    sys.addaudithook(audit)
    data=np.load(INPUT)
    archive={};result=dict(protocol='Image/vector-only discovery; physical evaluator has not run',scopes={})
    start=time.monotonic()
    for scope in ['local','broad']:
        x=data[f'{scope}_train_x'];xv=data[f'{scope}_val_x']
        fx=[data[f'{scope}_{s}_frame_x'] for s in ['train','val','test']]
        dictionaries={False:Dictionary(x,False),True:Dictionary(x,True)}
        scopes={};frames={}
        for method in ['svd','pca']:
            b=[data[f'{scope}_{s}_{method}'] for s in ['train','val','test']]
            aligned,diagnostic=align_frames(data['initial_x'],data['initial_b'],fx[0],b[0],list(zip(fx[1:],b[1:])))
            frames[method]=aligned
            records=[];models=[]
            for normalized,dictionary in dictionaries.items():
                for requested in [48,96]:
                    k=min(requested,dictionary.project.shape[1])
                    u=dictionary.transform(x,k);uv=dictionary.transform(xv,k)
                    d=dictionary.directions(fx[0],aligned[0],k)
                    dv=dictionary.directions(fx[1],aligned[1],k)
                    for ridge in [.001,.01,.1]:
                        c,r=choose_functions(u,uv,d,dv,ridge)
                        r.update(normalized=normalized,components=k,ridge=ridge,
                                 validation_rank=rank_metrics(dv@c))
                        records.append(r);models.append((dictionary,c,k))
            best=int(np.argmin([r['score'] for r in records]))
            dictionary,c,k=models[best]
            scopes[method]=dict(selection=records[best],all_attempts=records,
                               family_graph=diagnostic,
                               vector_compression=vector_compression(aligned[0],aligned[2]))
            frames[method+'_model']=(dictionary,c,k)
            print(scope,method,'encoder selected',records[best],flush=True)
        # Fixed negative control: destroy family consistency independently at each training state.
        dictionary,c,k=frames['svd_model']
        d=dictionary.directions(fx[0],frames['svd'][0],k)
        dv=dictionary.directions(fx[1],frames['svd'][1],k)
        rng=np.random.default_rng(2026101822)
        shuffled=np.array([row[rng.permutation(5)] for row in d])
        # Selection cannot use unshuffled validation families in this control.
        shuffled_v=np.array([row[rng.permutation(5)] for row in dv])
        cs,rs=choose_functions(dictionary.transform(x,k),dictionary.transform(xv,k),shuffled,shuffled_v,
                               scopes['svd']['selection']['ridge'])
        frames['shuffled_model']=(dictionary,cs,k)
        scopes['shuffled']=dict(selection=rs)
        # Trivial baseline: first five global raw-pixel PCs (whitened only for scale).
        raw=Dictionary(x,False,standardize=False);k0=min(96,raw.project.shape[1]);cp=np.zeros((k0,5));cp[:5]=np.eye(5)
        frames['global_pc_model']=(raw,cp,k0)
        scopes['global_pc']=dict(selection=dict(description='First five global image PCs'))
        for name in ['svd','pca','shuffled','global_pc']:
            dictionary,c,k=frames[name+'_model']
            prefix=f'{scope}_{name}'
            qs={s:dictionary.transform(data[f'{scope}_{s}_x'],k)@c for s in ['train','val','test','path','edge']}
            qf=[dictionary.transform(xx,k)@c for xx in fx]
            ref=frames['pca'] if name=='pca' else frames['svd']
            response=[dictionary.directions(xx,bb,k)@c for xx,bb in zip(fx,ref)]
            # For baselines find an anonymous-family permutation on validation only.
            perm=np.arange(5)
            if name in ['shuffled','global_pc']:
                row,col=linear_sum_assignment(-sensitivity(response[1]))
                perm=col[np.argsort(row)]
                c=c[:,perm];qs={s:v[:,perm] for s,v in qs.items()}
                qf=[v[:,perm] for v in qf];response=[v[:,:,perm] for v in response]
            scopes[name]['test_sensitivity']=sensitivity(response[2]).tolist()
            scopes[name]['test_mean_leakage']=float(1-np.trace(sensitivity(response[2]))/5)
            scopes[name]['test_rank']=rank_metrics(response[2])
            choices,attempts,pred,gpred,gcoef=decode(x,qs['train'],qs['val'],xv,{s:qs[s] for s in ['test','path','edge']})
            scopes[name]['decoder_selection']=choices
            scopes[name]['decoder_attempts']=attempts
            scopes[name]['reconstruction']={s:metrics(data[f'{scope}_{s}_x'],v) for s,v in pred.items()}
            scopes[name]['global_cubic']={s:metrics(data[f'{scope}_{s}_x'],v) for s,v in gpred.items()}
            # Affine, five-coordinate prediction of signed vector fields on new frames.
            a=np.column_stack([np.ones(len(qf[0])),qf[0]])
            field_coef=np.linalg.lstsq(a,ref[0].reshape(len(a),-1),rcond=1e-10)[0]
            field_pred=np.column_stack([np.ones(len(qf[2])),qf[2]])@field_coef
            ferr=np.linalg.norm((field_pred-ref[2].reshape(len(qf[2]),-1)).reshape(-1,5,784),axis=2)
            scopes[name]['affine_field_prediction']=dict(mean_relative=float(ferr.mean()),max_relative=float(ferr.max()))
            scopes[name]['stored_decoder_values']=int(x.size+qs['train'].size)
            archive.update({prefix+'_active':dictionary.active,prefix+'_feature_mean':dictionary.mean,
                prefix+'_feature_scale':dictionary.scale,prefix+'_project':dictionary.project[:,:k],
                prefix+'_coordinate_coefficients':c,prefix+'_normalized':np.array(dictionary.normalized),
                prefix+'_global_cubic_coefficients':gcoef})
            archive.update({prefix+'_'+s+'_q':v for s,v in qs.items()})
            for s,v in pred.items(): archive[prefix+'_'+s+'_prediction']=v
            archive[prefix+'_test_response']=response[2]
            # The local encoder's broad transfer is checked without refitting.
            if scope=='local':
                qb=dictionary.transform(data['broad_test_x'],k)@c
                model=Polynomial(qs['train'],x,choices['degree'],choices['neighbors'])
                transfer=model.predict(qb)
                scopes[name]['broad_transfer']=metrics(data['broad_test_x'],transfer)
            print(scope,name,'heldout',scopes[name]['reconstruction']['test'],'leak',scopes[name]['test_mean_leakage'],flush=True)
        result['scopes'][scope]=scopes
    result['elapsed_seconds']=time.monotonic()-start
    result['approved_repository_reads']=sorted(ACCESS)
    np.savez_compressed(OUT/'discovered_models.npz',**archive)
    (OUT/'blind_results.json').write_text(json.dumps(result,indent=2))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [SOURCE,SOURCE.with_name('PROTOCOL.md'),INPUT,OUT/'discovered_models.npz',OUT/'blind_results.json']}
    (OUT/'FREEZE.json').write_text(json.dumps(dict(physical_evaluation_has_run=False,sha256=hashes),indent=2))
    print('DISCOVERY FROZEN. No physical labels read.',flush=True)


if __name__=='__main__':main()
