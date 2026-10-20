"""Post-freeze evaluation of actual coordinate paths, mixing, and family PCA."""
from pathlib import Path
import sys
import json
import hashlib
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import numpy as np
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments/pooled_family_coordinates'))
from grey_ones import render, RANGES, SUB
from common import align_frames, design, powers
from saved import feature_from_archive

OLD = ROOT / 'figures/pooled_family_coordinates'
OUT = ROOT / 'figures/pooled_reading_paths'
SPAN = np.ptp(RANGES, axis=1)
SCALES = [.005, .05, .25, 1.]
NAMES = ['horizontal_center', 'vertical_center', 'height', 'width', 'lean_degrees']


def verify_freezes():
    hashes = {}
    for filename in ['FREEZE.json', 'BALANCED_FREEZE.json']:
        a = json.loads((OLD / filename).read_text())
        for path, digest in a['sha256'].items():
            assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == digest, path
        hashes[filename] = hashlib.sha256((OLD/filename).read_bytes()).hexdigest()
    return hashes


def image_jacobian(points):
    """Renderer derivative for evaluation-only inverse solving, batched."""
    p = np.atleast_2d(points)
    cx, cy, height, width, angle = [p[:, i, None] for i in range(5)]
    slope = np.tan(np.deg2rad(angle))
    top = cy-height/2; bottom = cy+height/2
    lo = np.arange(28*SUB)[None, :]/SUB; hi = lo+1/SUB
    mid = (lo+hi)/2
    v = np.clip(np.minimum(hi, bottom)-np.maximum(lo, top), 0, 1/SUB)
    def indicator(z):
        return ((lo < z)&(z < hi)).astype(float)+.5*((lo == z)|(hi == z))
    dt = -indicator(top); db = indicator(bottom)
    center = cx-slope*(mid-cy)
    left = center-width/2; right = center+width/2
    cols = np.arange(28)[None, None, :]
    overlap = np.clip(np.minimum(right[:, :, None], cols+1)-np.maximum(left[:, :, None], cols), 0, 1)
    def side(z):
        z = z[:, :, None]
        return ((cols < z)&(z < cols+1)).astype(float)+.5*((cols == z)|(z == cols+1))
    da = side(right)-side(left)
    dw = (side(right)+side(left))/2
    def integrate(a, b):
        return (a[:, :, None]*b).reshape(len(p), 28, SUB, 28).sum(2).reshape(len(p), 784)
    x = integrate(v, overlap)
    jx = integrate(v, da)
    jy = integrate(dt+db, overlap)+slope*jx
    jh = integrate((db-dt)/2, overlap)
    jw = integrate(v, dw)
    ja = integrate(v, da*(-(mid-cy)*np.pi/180*(1+slope*slope))[:, :, None])
    return x, np.stack([jx, jy, jh, jw, ja], axis=2)


def restore(archive, prefix):
    f = feature_from_archive(archive, prefix)
    c = archive[prefix+'_coefficient']
    assert f.kind == 'linear'
    matrix = np.zeros((5, 784)); matrix[:, f.active] = c.T
    offset = archive[prefix+'_center']-f.mean@c
    return matrix, offset


def original_maps(archive, prefix, scope, data, labels):
    q = [archive[prefix+'_'+split+'_q'] for split in ['train','val','test']]
    p = [labels[scope+'_'+split+'_p'] for split in ['train','val','test']]
    qm = q[0].mean(0); qs = q[0].std(0)
    pm = p[0].mean(0); ps = p[0].std(0)
    rows = {}
    for degree in [1, 2, 3]:
        coef = np.linalg.lstsq(design((q[0]-qm)/qs, degree), p[0], rcond=1e-10)[0]
        errors = {}
        for split, i in [('validation',1),('test',2)]:
            delta = design((q[i]-qm)/qs, degree)@coef-p[i]
            errors[split] = dict(rmse=np.sqrt(np.mean(delta*delta,0)).tolist(),
                               max_abs=np.max(abs(delta),0).tolist())
        rows[str(degree)] = dict(**errors, coefficients=coef.tolist(),
                                terms_zero_based=[list(t) for t in powers(5,degree)])
    stdcoef = np.linalg.lstsq(design((q[0]-qm)/qs,1), (p[0]-pm)/ps, rcond=1e-10)[0]
    return dict(names=NAMES, reading_mean=qm.tolist(), reading_scale=qs.tolist(),
                original_mean=pm.tolist(), original_scale=ps.tolist(),
                standardized_linear_matrix=stdcoef[1:].T.tolist(),
                standardized_linear_offset=stdcoef[0].tolist(), degrees=rows,
                labels_used_only_after_original_discovery_freeze=True)


def family_pca(train, test):
    out=[]
    for i in range(5):
        a=train[:,i];b=test[:,i];mean=a.mean(0)
        _,s,vt=np.linalg.svd(a-mean,full_matrices=False)
        total=np.sum(s*s);ratio=np.cumsum(s*s)/max(total,1e-30)
        errors={}
        for k in [0,1,2,5,10]:
            z=b-mean
            residual=z-(z@vt[:k].T)@vt[:k] if k else z
            errors[str(k)]=float(np.mean(np.linalg.norm(residual,axis=1)))
        out.append(dict(family=i+1,variation_rms=float(np.sqrt(total/len(a))),
                        pcs_for_95=int(np.searchsorted(ratio,.95)+1),
                        pcs_for_99=int(np.searchsorted(ratio,.99)+1),
                        test_mean_vector_error=errors,variance_fractions=(s*s/max(total,1e-30)).tolist(),
                        mean=mean.tolist(),components=vt[:10].tolist()))
    return out


def solve_targets(matrix, offset, initial, targets, qscale):
    p=initial.copy()
    for _ in range(18):
        x,j=image_jacobian(p);r=(x@matrix.T+offset-targets)/qscale
        error=np.max(abs(r),axis=1)
        if np.max(error)<2e-10:break
        a=np.einsum('op,npk->nok',matrix,j)*SPAN[None,None,:]/qscale[None,:,None]
        delta=(np.linalg.pinv(a,rcond=1e-10)@r[:,:,None])[:,:,0]*SPAN
        delta*=np.minimum(1.,.5/np.maximum(np.linalg.norm(delta/SPAN,axis=1),1e-30))[:,None]
        best=p.copy();besterror=error.copy()
        for factor in [1.,.5,.25,.0625]:
            trial=p-factor*delta
            trial_x=render(trial,dtype=np.float64).reshape(-1,784)
            e=np.max(abs((trial_x@matrix.T+offset-targets)/qscale),axis=1)
            good=e<besterror
            best[good]=trial[good];besterror[good]=e[good]
        if np.max(np.linalg.norm(best-p,axis=1))<1e-13:break
        p=best
    x=render(p,dtype=np.float64).reshape(-1,784)
    residual=np.max(abs((x@matrix.T+offset-targets)/qscale),axis=1)
    return p,x,residual


def in_canvas(p):
    slope=np.tan(np.deg2rad(p[...,4]));h=p[...,2];w=p[...,3]
    extent=abs(slope)*h/2+w/2
    return ((h>0)&(w>0)&(p[...,1]-h/2>=0)&(p[...,1]+h/2<=28)&
            (p[...,0]-extent>=0)&(p[...,0]+extent<=28))


def paths(matrix,offset,anchors,source_x,qscale):
    # Each group contains all five readings and nine start images.
    anchor_x,j=image_jacobian(anchors)
    q0=anchor_x@matrix.T+offset
    a=np.einsum('op,npk->nok',matrix,j)
    inv=np.linalg.pinv(a,rcond=1e-11)
    tangent=np.einsum('npk,nki->npi',j,inv)
    speed=np.linalg.norm(tangent,axis=1)
    records=[];saved={}
    ticks=np.linspace(-1,1,17)
    for mode in ['shared_reading_ticks','unit_initial_speed']:
        for halfspan in SCALES:
            n=len(anchors)
            base=np.tile(anchors,(5,1))
            baseq=np.tile(q0,(5,1))
            axis=np.repeat(np.arange(5),n)
            factor=(np.tile(1/speed[0],(n,1)) if mode=='shared_reading_ticks' else 1/speed).T.ravel()
            history=np.zeros((5*n,17,784));settings=np.zeros((5*n,17,5))
            errors=np.zeros((5*n,17))
            history[:,8]=np.tile(anchor_x,(5,1));settings[:,8]=base
            for sign in [-1,1]:
                current=base.copy()
                for step in range(1,9):
                    tick_index=8+sign*step
                    target=baseq.copy()
                    target[np.arange(5*n),axis]+=ticks[tick_index]*halfspan*factor
                    current,x,err=solve_targets(matrix,offset,current,target,qscale)
                    history[:,tick_index]=x;settings[:,tick_index]=current;errors[:,tick_index]=err
            all_ok=(errors.max(1)<1e-7)&np.all(in_canvas(settings),axis=1)
            within_box=np.all((settings>=RANGES[:,0])&(settings<=RANGES[:,1]),axis=(1,2))
            radius=np.max(np.linalg.norm(history-source_x,axis=2),axis=1)
            for i in range(5):
                images=history[i*n:(i+1)*n]
                ok=all_ok[i*n:(i+1)*n]
                ds=np.array([pdist(row) for row in images]);norm=np.linalg.norm(ds,axis=1)
                relative=np.linalg.norm(ds-ds[0],axis=1)/max(norm[0],1e-30)
                shape=np.linalg.norm(ds/np.maximum(norm[:,None],1e-30)-ds[0]/max(norm[0],1e-30),axis=1)
                # Comparing every family at its source is handled below.
                singular=np.array([np.linalg.svd(row-row[8],compute_uv=False) for row in images])
                one_dim_fraction=singular[:,0]**2/np.maximum(np.sum(singular*singular,axis=1),1e-30)
                valid=ok[1:]&ok[0]
                row=dict(mode=mode,halfspan=halfspan,reading=i+1,
                         paths=int(n),complete_paths=int(ok.sum()),
                         comparisons=int(valid.sum()),reference_complete=bool(ok[0]),
                         median_distance_matrix_error=float(np.median(relative[1:][valid])) if valid.any() else None,
                         max_distance_matrix_error=float(np.max(relative[1:][valid])) if valid.any() else None,
                         median_shape_only_error=float(np.median(shape[1:][valid])) if valid.any() else None,
                         max_shape_only_error=float(np.max(shape[1:][valid])) if valid.any() else None,
                         median_first_pc_energy=float(np.median(one_dim_fraction[ok])) if ok.any() else None,
                         max_normalized_reading_residual=float(errors[i*n:(i+1)*n].max()),
                         max_distance_from_chart_source=float(radius[i*n:(i+1)*n].max()),
                         paths_inside_radius_0p25=int(np.sum(radius[i*n:(i+1)*n]<=.25+1e-9)),
                         paths_inside_default_box=int(np.sum(within_box[i*n:(i+1)*n])),
                         each_distance_matrix_error=relative.tolist(),each_shape_only_error=shape.tolist(),
                         each_path_complete=ok.tolist())
                records.append(row)
            stem=mode+'_'+str(halfspan)
            saved[stem+'_images']=history.reshape(5,n,17,784)
            saved[stem+'_settings']=settings.reshape(5,n,17,5)
            saved[stem+'_residuals']=errors.reshape(5,n,17)
            src=history.reshape(5,n,17,784)[:,0]
            d=np.array([pdist(row) for row in src]);dn=d/np.maximum(np.linalg.norm(d,axis=1)[:,None],1e-30)
            saved[stem+'_source_shape_pair_errors']=squareform(pdist(dn))
            print(mode,halfspan,'complete',all_ok.sum(),'/',len(all_ok),flush=True)
    return records,saved


def repeated_pca(x,frame):
    centered=x-x.mean(0)
    active=np.flatnonzero(np.any(centered!=0,axis=0))
    u,s,vt=np.linalg.svd(centered[:,active],full_matrices=False)
    scores=centered[:,active]@vt.T
    _,s2,vt2=np.linalg.svd(scores,full_matrices=False)
    covariance=scores.T@scores
    off=covariance-np.diag(np.diag(covariance))
    rank=int(np.sum(s>s[0]*1e-10))
    centered_basis=frame-frame.mean(0)
    sb=np.linalg.svd(centered_basis,compute_uv=False)
    return dict(score_covariance_relative_off_diagonal=float(np.linalg.norm(off)/np.linalg.norm(covariance)),
                repeated_singular_value_relative_error=float(np.linalg.norm(s-s2)/np.linalg.norm(s)),
                first_pca_rank=rank,second_pca_rank=int(np.sum(s2>s2[0]*1e-10)),
                centered_five_orthonormal_vectors_singular_values=sb.tolist(),
                explanation='Full scores retain the original PCA spectrum. Centering five orthonormal basis rows gives four equal nonzero singular values.')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    original_hashes=verify_freezes()
    data=np.load(OLD/'patch_observations.npz')
    balanced=np.load(OLD/'balanced_frames.npz')
    initial=np.load(ROOT/'figures/svd_blind_intermediates/observations.npz')
    labels=np.load('/tmp/pooled_family_coordinates_labels.npz')
    results=dict(protocol='experiments/pooled_reading_paths/PROTOCOL.md',
                 frozen_manifest_hashes=original_hashes,charts={},names=NAMES,
                 posthoc_evaluation_uses_known_renderer_and_control_settings=True)
    arrays={}
    for run,filename in [('192_random','models.npz'),('512_balanced','balanced_models.npz')]:
        archive=np.load(OLD/filename)
        for si in range(2):
            scope=f'patch_{si}_2';prefix=scope+'_svd_fixed_joint';key=run+'_source_'+str(si)
            matrix,offset=restore(archive,prefix)
            anchors=np.concatenate([labels[scope+'_source_p'][None,:],labels[scope+'_test_frame_p'][:8]])
            for split in ['train','val','test']:
                actual=data[scope+'_'+split+'_x']@matrix.T+offset
                assert np.max(abs(actual-archive[prefix+'_'+split+'_q']))<1e-8
            mapping=original_maps(archive,prefix,scope,data,labels)
            fxs=[data[scope+'_'+s+'_frame_x'] for s in ['train','val','test']]
            bs=[(balanced if run=='512_balanced' else data)[scope+'_'+s+'_svd'] for s in ['train','val','test']]
            aligned,graph=align_frames(initial['initial_x'],initial['initial_b'],fxs[0],bs[0],list(zip(fxs[1:],bs[1:])))
            pc=family_pca(aligned[0],aligned[2])
            source_x=data[scope+'_source_x'][0]
            rows,saved=paths(matrix,offset,anchors,source_x,np.asarray(mapping['reading_scale']))
            for name,value in saved.items():arrays[key+'_'+name]=value
            arrays[key+'_matrix']=matrix;arrays[key+'_offset']=offset;arrays[key+'_anchor_settings']=anchors
            results['charts'][key]=dict(source_index=si,acquisition=run,mapping=mapping,
                                       pca=pc,family_graph=graph,paths=rows)
            print('FINISHED',key,flush=True)
    # Independent renderer/Jacobian verification, not a mirror-only check.
    checkpoints=np.concatenate([labels[f'patch_{si}_2_test_p'][:12] for si in range(2)])
    x,j=image_jacobian(checkpoints)
    checkx=render(checkpoints,dtype=np.float64).reshape(-1,784)
    derivative_errors=[]
    for i in range(5):
        eps=np.zeros(5);eps[i]=SPAN[i]*1e-7
        finite=(render(checkpoints+eps,dtype=np.float64).reshape(-1,784)-render(checkpoints-eps,dtype=np.float64).reshape(-1,784))/(2*eps[i])
        derivative_errors.append(float(np.linalg.norm(finite-j[:,:,i])/np.maximum(np.linalg.norm(j[:,:,i]),1e-30)))
    results['verification']=dict(max_renderer_difference=float(np.max(abs(x-checkx))),
                                 jacobian_relative_errors=derivative_errors,
                                 frozen_files_unchanged=verify_freezes()==original_hashes)
    assert results['verification']['max_renderer_difference']<1e-12
    assert max(derivative_errors)<2e-5
    assert results['verification']['frozen_files_unchanged']
    results['repeated_pca']=repeated_pca(data['patch_0_2_train_x'],balanced['patch_0_2_train_svd'][0])
    np.savez_compressed(OUT/'paths.npz',**arrays)
    (OUT/'results.json').write_text(json.dumps(results,indent=2))
    print('VERIFIED',results['verification'],flush=True)


if __name__=='__main__':main()
