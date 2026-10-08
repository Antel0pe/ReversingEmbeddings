"""Learn lean-vector variations from paired pixel observations only.

The renderer supplies controlled observations. Predictors never evaluate its
geometry, recover its settings, or call it to make a prediction.
"""
from pathlib import Path
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from grey_ones import render, RANGES

OUT=Path(__file__).resolve().parent/'results'
BASE=RANGES.mean(axis=1);BASE[4]=0
EPS=.02

def images(p):
    p=np.atleast_2d(p)
    return np.concatenate([render(q).reshape(len(q),-1).astype(float)
        for q in np.array_split(p,max(1,(len(p)+63)//64))])

def observed_arrows(p,step=EPS):
    p=np.atleast_2d(p).astype(float)
    a=p.copy();b=p.copy()
    a[:,4]=np.maximum(RANGES[4,0],p[:,4]-step)
    b[:,4]=np.minimum(RANGES[4,1],p[:,4]+step)
    return (images(b)-images(a))/(b[:,4]-a[:,4])[:,None]

def scores(pred,target):
    n=np.linalg.norm(target,axis=1)
    rel=np.linalg.norm(pred-target,axis=1)/n
    return {'mean_relative_error':float(rel.mean()),'p95_relative_error':float(np.quantile(rel,.95)),
            'max_relative_error':float(rel.max())}

def width_predict(x,origin,a,base_v,b,knots,values):
    q=(x-origin)@a
    alpha=np.interp(q,knots,values)
    return base_v+alpha[:,None]*b

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(20261007)
    x0=images(BASE)[0];v0=observed_arrows(BASE)[0]
    # Observe the width curve, then refine the interval with largest arrow change.
    coarse=np.linspace(*RANGES[3],65)
    p=np.tile(BASE,(len(coarse),1));p[:,3]=coarse
    coarse_v=observed_arrows(p)
    change=np.linalg.norm(np.diff(coarse_v,axis=0),axis=1)
    k=int(change.argmax());lo,hi=coarse[k],coarse[k+1]
    train_w=np.unique(np.r_[coarse,np.linspace(lo-.01,hi+.01,257)])
    train_p=np.tile(BASE,(len(train_w),1));train_p[:,3]=train_w
    train_x=images(train_p);train_v=observed_arrows(train_p)
    # A scalar coordinate learned from IMAGE displacement, not known image width.
    span=train_x[-1]-train_x[0]
    a=span/(span@span)
    q=(train_x-x0)@a
    b=train_v[0]-v0
    alpha=(train_v-v0)@b/(b@b)
    assert np.all(np.diff(q)>0)
    test_w=np.r_[rng.uniform(*RANGES[3],1000),np.linspace(lo,hi,301)+1.137e-7]
    assert not np.isclose(test_w[:,None],train_w[None,:],atol=1e-10,rtol=0).any()
    test_p=np.tile(BASE,(len(test_w),1));test_p[:,3]=test_w
    test_x=images(test_p);test_v=observed_arrows(test_p)
    pred=width_predict(test_x,x0,a,v0,b,q,alpha)
    single={'training_images':len(train_w),'heldout_uniform_random':1000,'heldout_transition_stress':301,
        'refinement_interval_discovered_from_vectors':[float(lo),float(hi)],
        'equation':'V_hat(x) = v0 + alpha(<a, x-x0>) * b',
        'random':scores(pred[:1000],test_v[:1000]),'transition_stress':scores(pred[1000:],test_v[1000:]),
        'all':scores(pred,test_v),
        'best_possible_one_change_vector':scores(v0+(((test_v-v0)@b/(b@b))[:,None]*b),test_v)}
    # Additional variation vectors: learn from data, keep coefficients as functions
    # of the single image-space coordinate. Measure prediction error, not variance.
    _,singular,vt=np.linalg.svd(train_v-v0,full_matrices=False)
    basis_results={}
    for rank in [1,2,4,8,16,32]:
        basis=vt[:rank].T
        coeff=(train_v-v0)@basis
        test_q=(test_x-x0)@a
        coef_pred=np.column_stack([np.interp(test_q,q,coeff[:,j]) for j in range(rank)])
        decoded=v0+coef_pred@basis.T
        if rank==16:pred16=decoded.copy()
        basis_results[str(rank)]={'random':scores(decoded[:1000],test_v[:1000]),
             'transition_stress':scores(decoded[1000:],test_v[1000:])}
    np.savez_compressed(OUT/'width_model.npz',origin=x0,coordinate_vector=a,base_lean_vector=v0,
        change_vector=b,coefficient_knots=q,coefficient_values=alpha,
        variation_basis=vt[:32].T,variation_coefficients=(train_v-v0)@vt[:32].T)
    np.savez_compressed(OUT/'width_evaluation.npz',settings=test_p,images=test_x,observed=test_v,
        predicted=pred,predicted16=pred16,train_q=q,train_alpha=alpha,test_q=(test_x-x0)@a)
    # Five independent observation curves; progress labels describe applied moves.
    # Learn a LINEAR image encoder from these observations, not a renderer inverse.
    grid=[];labels=[];curve_p=[];curve_v=[];axis_progress=[]
    spans=np.diff(RANGES,axis=1).ravel()
    for axis in range(5):
        vals=train_w if axis==3 else np.unique(np.r_[np.linspace(*RANGES[axis],65),BASE[axis]])
        pp=np.tile(BASE,(len(vals),1));pp[:,axis]=vals
        vv=observed_arrows(pp)
        xx=images(pp)
        yy=np.zeros((len(vals),5));yy[:,axis]=(vals-BASE[axis])/spans[axis]
        grid.append(xx-x0);labels.append(yy);curve_p.append(pp);curve_v.append(vv)
        axis_progress.append(yy[:,axis])
    design=np.vstack(grid);response=np.vstack(labels)
    # Regularized least-squares encoder, all columns derived from pixel data.
    u,s,vt_img=np.linalg.svd(design,full_matrices=False)
    ridge=1e-6*s[0]**2
    encoder=vt_img.T@((s/(s*s+ridge))[:,None]*(u.T@response))
    def additive(progress):
        v=np.tile(v0,(len(progress),1))
        for axis in range(5):
            delta=curve_v[axis]-v0
            v+=np.column_stack([np.interp(progress[:,axis],axis_progress[axis],delta[:,j]) for j in range(784)])
        return v
    # Shared dictionary of variation vectors across independent curves.
    all_delta=np.vstack([v-v0 for v in curve_v])
    _,sv,shared_vt=np.linalg.svd(all_delta,full_matrices=False)
    joint={}
    for name,axes in [('width_height',[2,3]),('width_horizontal',[0,3]),('all_five',[0,1,2,3,4])]:
        pp=np.tile(BASE,(512,1))
        for axis in axes:pp[:,axis]=rng.uniform(*RANGES[axis],len(pp))
        xx=images(pp);actual=observed_arrows(pp)
        progress=(pp-BASE)/spans
        ideal=additive(progress);encoded=(xx-x0)@encoder;encoded_pred=additive(encoded)
        joint[name]={'n':512,'additive_with_known_applied_move_labels':scores(ideal,actual),
          'additive_from_image_only_learned_coordinates':scores(encoded_pred,actual),
          'mean_absolute_coordinate_error':np.abs(encoded-progress).mean(axis=0).tolist()}
        if name=='all_five':
            basis_errors={}
            for rank in [8,32,64,128]:
                B=shared_vt[:rank].T
                projected=v0+(ideal-v0)@B@B.T
                basis_errors[str(rank)]=scores(projected,actual)
            joint[name]['variation_dictionary_predictions']=basis_errors
            np.savez_compressed(OUT/'joint_evaluation.npz',settings=pp,images=xx,observed=actual,
                additive=ideal,image_only=encoded_pred,encoder=encoder,variation_basis=shared_vt[:128].T)
    # Full-space check for the WIDTH-ONLY model: detect and expose extrapolation.
    pp=RANGES[:,0]+rng.random((512,5))*spans
    xx=images(pp);actual=observed_arrows(pp)
    full=scores(width_predict(xx,x0,a,v0,b,q,alpha),actual)
    B16=np.linalg.svd(train_v-v0,full_matrices=False)[2][:16].T
    coef16=(train_v-v0)@B16
    fullq=(xx-x0)@a
    decoded16=v0+np.column_stack([np.interp(fullq,q,coef16[:,j]) for j in range(16)])@B16.T
    full16=scores(decoded16,actual)
    # Finite lean steps directly test following this local predicted vector.
    endpoint=test_p.copy();endpoint[:,4]=.1
    actual_move=images(endpoint)-test_x
    finite=scores(.1*pred,actual_move)
    results={'scope':'Pixel vector observations only; no subrow, edge, or inverse geometry in predictor',
        'base_settings_for_observation':BASE.tolist(),'measured_arrow_half_step_degrees':EPS,
        'width_model':single,'width_variation_dictionary':basis_results,
        'single_axis_training_observations':len(design),'joint_tests':joint,
        'width_only_model_full_space_control':{'n':512,**full},
        'width_sixteen_vector_model_full_space_control':{'n':512,**full16},
        'width_model_frozen_vector_point_one_degree_move':finite,
        'limitations':['Finite differences estimate a local field and smooth switching events.',
            'One-axis observations do not determine arbitrary unseen joint interactions.',
            'The learned linear image encoder is approximate and its failures are scored separately.',
            'No assertion of exact all-state flow integration or exact manifold coverage is made.']}
    (OUT/'metrics.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))

if __name__=='__main__':main()
