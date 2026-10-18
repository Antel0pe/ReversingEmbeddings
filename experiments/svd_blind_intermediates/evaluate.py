"""Physical evaluation runs only after hash-frozen image/vector discovery."""
from pathlib import Path
import json
import hashlib
import numpy as np
from scipy.optimize import linear_sum_assignment
from discover import design, Polynomial

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'figures/svd_blind_intermediates'
NAMES=['horizontal_center','vertical_center','height','width','lean_degrees']


def verify():
    manifest=json.loads((OUT/'FREEZE.json').read_text())
    for relative,digest in manifest['sha256'].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()==digest, relative
    return manifest


def scores(truth,pred,scale):
    residual=pred-truth
    sse=np.sum(residual**2,axis=0)
    sst=np.sum((truth-truth.mean(0))**2,axis=0)
    return dict(r2=(1-sse/np.maximum(sst,1e-30)).tolist(),
        rmse=np.sqrt(np.mean(residual**2,axis=0)).tolist(),
        max_absolute=np.max(abs(residual),axis=0).tolist(),
        mean_normalized_rmse=float(np.mean(np.sqrt(np.mean(residual**2,axis=0))/scale)))


def mapping(train_x,train_y,val_x,val_y,test_x,test_y):
    mean=train_x.mean(0);scale=np.maximum(train_x.std(0),1e-12)
    outputscale=np.maximum(train_y.std(0),1e-12)
    records=[];predictions=[]
    for degree in [1,2,3]:
        coef=np.linalg.lstsq(design((train_x-mean)/scale,degree),train_y,rcond=1e-10)[0]
        pv=design((val_x-mean)/scale,degree)@coef
        pt=design((test_x-mean)/scale,degree)@coef
        records.append(dict(degree=degree,validation=scores(val_y,pv,outputscale),test=scores(test_y,pt,outputscale)))
        predictions.append(pt)
    best=int(np.argmin([r['validation']['mean_normalized_rmse'] for r in records]))
    local=Polynomial(train_x,train_y,2,128)
    plocal=local.predict(test_x)
    return dict(global_selected_degree=records[best]['degree'],global_attempts=records,
                global_test=records[best]['test'],local_quadratic_test=scores(test_y,plocal,outputscale)), predictions[best],plocal


def main():
    manifest=verify() # Must succeed before the first read of labels.
    truth=np.load('/tmp/svd_blind_intermediates_labels.npz')
    models=np.load(OUT/'discovered_models.npz')
    observations=np.load(OUT/'observations.npz')
    result=dict(discovery_hashes_verified_before_labels=True,physical_names=NAMES,scopes={})
    examples={}
    for scope in ['local','broad']:
        result['scopes'][scope]={}
        p=[truth[f'{scope}_{s}_p'] for s in ['train','val','test']]
        for method in ['svd','pca','shuffled','global_pc']:
            key=scope+'_'+method
            q=[models[key+'_'+s+'_q'] for s in ['train','val','test']]
            forward,pred,local=mapping(q[0],p[0],q[1],p[1],q[2],p[2])
            reverse,_,_=mapping(p[0],q[0],p[1],q[1],p[2],q[2])
            corr=np.corrcoef(q[2],p[2],rowvar=False)[:5,5:]
            row,col=linear_sum_assignment(-abs(corr))
            assignment=col[np.argsort(row)]
            rowresult=dict(discovered_to_original=forward,original_to_discovered=reverse,
                one_to_one_correlations=corr.tolist(),best_one_to_one_assignment=assignment.tolist(),
                assigned_absolute_correlations=abs(corr[np.arange(5),assignment]).tolist())
            result['scopes'][scope][method]=rowresult
            examples[key+'_physical_prediction']=pred
            examples[key+'_local_physical_prediction']=local
            # Export the actual fixed encoder as two signed pixel masks plus constants.
            a=models[key+'_active'];proj=models[key+'_project'];coef=models[key+'_coordinate_coefficients']
            weights=(proj@coef)/models[key+'_feature_scale'][:,None]
            constant=-(models[key+'_feature_mean']/models[key+'_feature_scale'])@proj@coef
            mask_a=np.zeros((784,5));mask_b=np.zeros((784,5))
            mask_a[a]=weights[:len(a)]
            if bool(models[key+'_normalized']):mask_b[a]=weights[len(a):]
            examples[key+'_mask_a']=mask_a;examples[key+'_mask_b']=mask_b;examples[key+'_constant']=constant
            x=observations[f'{scope}_test_x'];mass=x.sum(1)
            reconstructed=x@mask_a+(x@mask_b)/mass[:,None]+constant
            assert np.max(abs(reconstructed-q[2]))<1e-8
            print(scope,method,'post-freeze global R2',forward['global_test']['r2'],
                  'local R2',forward['local_quadratic_test']['r2'],flush=True)
    verify() # Physical results cannot modify discovery files unnoticed.
    (OUT/'physical_evaluation.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(OUT/'evaluation_examples.npz',**examples)
    (OUT/'EVALUATION_ORDER.json').write_text(json.dumps(dict(
        sequence=['protocol written','observations exported','image/vector-only discovery',
                  'discovery files hash-frozen','hashes verified','physical labels first opened',
                  'post-hoc maps evaluated','discovery hashes verified again'],
        frozen_discovery_sha256=manifest['sha256'],discovery_changed_after_physical_evaluation=False),indent=2))


if __name__=='__main__':main()
