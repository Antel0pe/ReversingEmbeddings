"""Post-freeze physical readouts and independently supplied geometric benchmarks."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
from scipy.optimize import linear_sum_assignment
from common import design,LocalPolynomial,align_frames,response_metrics,image_metrics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'figures/pooled_family_coordinates'
NAMES=['horizontal_center','vertical_center','height','width','lean_degrees']


def verify():
    manifests=[]
    for name in ['FREEZE.json','BALANCED_FREEZE.json']:
        a=json.loads((OUT/name).read_text());manifests.append(a)
        for p,digest in a['sha256'].items():
            assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==digest,p
    return manifests


def score(truth,pred,scale):
    error=pred-truth
    return dict(r2=(1-np.sum(error**2,axis=0)/np.maximum(np.sum((truth-truth.mean(0))**2,axis=0),1e-30)).tolist(),
        rmse=np.sqrt(np.mean(error**2,axis=0)).tolist(),max_error=np.max(abs(error),axis=0).tolist(),
        normalized_rmse=float(np.mean(np.sqrt(np.mean(error**2,axis=0))/np.maximum(scale,1e-14))))


def mapping(q,p):
    mu=q[0].mean(0);scale=np.maximum(q[0].std(0),1e-14);outscale=p[0].std(0)
    records=[]
    for degree in [1,2,3]:
        coefficient=np.linalg.lstsq(design((q[0]-mu)/scale,degree),p[0],rcond=1e-10)[0]
        v=design((q[1]-mu)/scale,degree)@coefficient
        t=design((q[2]-mu)/scale,degree)@coefficient
        records.append(dict(degree=degree,validation=score(p[1],v,outscale),test=score(p[2],t,outscale)))
    selected=min(records,key=lambda row:row['validation']['normalized_rmse'])
    local=LocalPolynomial(q[0],p[0],2,128).predict(q[2])
    return dict(selected=selected,attempts=records,local_quadratic=score(p[2],local,outscale))


def benchmark_response(train_r,test_r):
    energy=np.mean(train_r*train_r,axis=0)
    energy/=np.maximum(energy.sum(0),1e-30)
    row,col=linear_sum_assignment(-energy);order=col[np.argsort(row)]
    gain=np.diagonal(train_r[:,:,order],axis1=1,axis2=2).mean(0)
    gain=np.where(abs(gain)>1e-8,gain,np.ones(5))
    response=test_r[:,:,order]/gain[None,None,:]
    return dict(physical_columns_per_family=order.tolist(),response=response_metrics(response))


def main():
    manifests=verify() # No physical labels are opened before both freezes verify.
    labels_old=np.load('/tmp/svd_blind_intermediates_labels.npz')
    labels_new=np.load('/tmp/pooled_family_coordinates_labels.npz')
    old=np.load(ROOT/'figures/svd_blind_intermediates/observations.npz')
    new=np.load(OUT/'patch_observations.npz');balanced=np.load(OUT/'balanced_frames.npz')
    sys.path.insert(0,str(ROOT))
    from experiments.derived_state_coordinates import recover,coverage
    from experiments.svd_state_coordinate_bridge import readers
    from analyze_generated_one_manifold_transport import jet
    result=dict(physical_names=NAMES,discovery_frozen_before_evaluation=True,runs={},benchmarks={})
    for run,filename in [('primary','models.npz'),('balanced','balanced_models.npz')]:
        models=np.load(OUT/filename);rows={}
        for key in models.files:
            if not key.endswith('_coefficient'):continue
            prefix=key.removesuffix('_coefficient')
            scope=next(s for s in ['local','broad']+[f'patch_{i}_{j}' for i in range(2) for j in range(4)] if prefix.startswith(s+'_'))
            labels=labels_old if scope in ['local','broad'] else labels_new
            q=[models[prefix+'_'+s+'_q'] for s in ['train','val','test']]
            p=[labels[scope+'_'+s+'_p'] for s in ['train','val','test']]
            rows[prefix]=mapping(q,p)
        result['runs'][run]=rows
    scopes=['local','broad']+[f'patch_{i}_{j}' for i in range(2) for j in range(4)]
    for scope in scopes:
        data=old if scope in ['local','broad'] else new
        labels=labels_old if scope in ['local','broad'] else labels_new
        x=data[scope+'_test_x'];p,q,_,_=recover(x)
        row=dict(recovered_physical_max_error=np.max(abs(p-labels[scope+'_test_p']),axis=0).tolist(),
                 reconstruction=image_metrics(x,coverage(q).reshape(-1,784)),responses={})
        for method in ['svd','pca']:
            fxs=[data[scope+'_'+s+'_frame_x'] for s in ['train','val','test']]
            bs=[data[scope+'_'+s+'_'+method] for s in ['train','val','test']]
            aligned,_=align_frames(old['initial_x'],old['initial_b'],fxs[0],bs[0],list(zip(fxs[1:],bs[1:])))
            physical=[];derived=[]
            for index in [0,2]:
                gradients=[readers(image) for image in fxs[index]]
                dp=np.array([a[3] for a in gradients]);dq=np.array([a[2] for a in gradients])
                physical.append(np.einsum('nfp,ncp->nfc',aligned[index],dp))
                derived.append(np.einsum('nfp,ncp->nfc',aligned[index],dq))
            row['responses'][method]=dict(original_readers=benchmark_response(*physical),
                                          reverse_derived_readers=benchmark_response(*derived))
        # Check the original readers on actual physical directions, only as an oracle diagnostic.
        errors=[]
        points=labels[scope+'_test_frame_p'][:16]
        for image,point in zip(data[scope+'_test_frame_x'][:16],points):
            dp=readers(image)[3];errors.append(np.max(abs(dp@jet(point)[3]-np.eye(5))))
        row['oracle_original_response_identity_error']=float(max(errors))
        result['benchmarks'][scope]=row
        print(scope,'geometry max pixel',row['reconstruction']['max_pixel'],
              'original readers off to SVD',row['responses']['svd']['original_readers']['response']['off_rms'],flush=True)
    verify()
    (OUT/'physical_evaluation.json').write_text(json.dumps(result,indent=2))
    (OUT/'EVALUATION_ORDER.json').write_text(json.dumps(dict(
        sequence=['protocol','primary blind fit','primary freeze','blind rotation and acquisition-quality diagnostics',
                  'balanced blind fit','balanced freeze','both freezes verified','physical labels opened',
                  'physical readouts and geometric benchmarks evaluated','both freezes verified again'],
        physical_results_used_to_refit_discovery=False),indent=2))
    for scope in ['local','broad','patch_0_0','patch_1_0']:
        for group in ['fixed_joint','quadratic']:
            key=scope+'_svd_'+group
            if key in result['runs']['primary']:
                print(key,'postfreeze R2',result['runs']['primary'][key]['selected']['test']['r2'],flush=True)


if __name__=='__main__':main()
