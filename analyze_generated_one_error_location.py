"""Where does a fixed learned 5D manifold err, and does local bending predict it?"""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import json
import itertools
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from grey_ones import RANGES, KNOBS
from analyze_generated_one_principal_curves import pixels
from fit_generated_one_principal_manifold import OUT as MODEL_DIR, decode

OUT=MODEL_DIR/'locations'


def stats(sse,mask=None):
    if mask is None:mask=np.ones(len(sse),bool)
    v=sse[mask]
    if not len(v):return dict(count=0)
    return dict(count=len(v),point_share_percent=100*float(np.mean(mask)),
        squared_error_share_percent=100*float(v.sum()/sse.sum()),
        rms_image_L2=float(np.sqrt(v.mean())),mean_image_L2=float(np.sqrt(v).mean()),
        p95_image_L2=float(np.quantile(np.sqrt(v),.95)),max_image_L2=float(np.sqrt(v).max()))


def relation(sse,feature):
    finite=np.isfinite(feature);e=sse[finite];f=feature[finite]
    top=f>=np.quantile(f,.9);low=f<=np.quantile(f,.1)
    return dict(count=len(e),spearman_error_correlation=float(spearmanr(f,np.sqrt(e)).statistic),
        highest_feature_decile=stats(e,top),lowest_feature_decile=stats(e,low),
        high_to_low_rms_ratio=float(np.sqrt(e[top].mean()/max(e[low].mean(),1e-30))),
        feature_median=float(np.median(f)),feature_p90=float(np.quantile(f,.9)))


def binned(sse,values,edges):
    return [dict(low=float(a),high=float(b),**stats(sse,(values>=a)&(values<b if i<len(edges)-2 else values<=b)))
            for i,(a,b) in enumerate(zip(edges[:-1],edges[1:]))]


def main():
    OUT.mkdir(exist_ok=True);d=np.load(MODEL_DIR/'dense_uniform_examples.npz');m=np.load(MODEL_DIR/'fine.npz')
    p=d['knobs'];x=d['images'];raw=d['reconstructed'];clipped=np.clip(raw,0,1)
    u=(p-RANGES[:,0])/np.ptp(RANGES,axis=1);chart=d['coordinates'];shape=m['shape'];active=d['active']
    residuals={'raw':raw-x,'clipped':clipped-x};sse={k:np.sum(e*e,axis=1) for k,e in residuals.items()}
    face=np.min(np.minimum(u,1-u),axis=1);near_count=np.sum((u<.05)|(u>.95),axis=1)
    chart_bound=np.any((chart<1e-8)|(chart>1-1e-8),axis=1)
    # Actual caps and their proximity to a horizontal pixel-row boundary.
    top=p[:,1]-p[:,2]/2;bottom=p[:,1]+p[:,2]/2
    cap_row_dist=np.minimum(abs(top-np.round(top)),abs(bottom-np.round(bottom)))
    cap_phase=np.column_stack([top%1,bottom%1])
    tree=cKDTree((m['coverage_knobs']-RANGES[:,0])/np.ptp(RANGES,axis=1))
    setting_gap=tree.query(u)[0]
    r=dict(points=len(x),seed=1008,knob_order=KNOBS,ranges=RANGES.tolist(),
        model='Unchanged fine five-dimensional principal manifold; cached pixel-only projections on 16,384 uniform Sobol images (seed 740).',
        metric='Image L2 over all 784 ink-coverage pixels; concentration uses summed squared errors, and group RMS is sqrt(mean image squared error).',
        models={},curvature={},limits='Associations do not causally separate grid resolution, smoothing, sample spacing, boundary coverage and coordinate search. Local bending proxies are finite-scale diagnostics of the known renderer, not a computed global curvature tensor or a proof about every manifold point.')
    for name,e2 in sse.items():
        order=np.argsort(e2)[::-1]
        concentrated={}
        for frac in [.01,.05,.1]:
            mask=np.zeros(len(e2),bool);mask[order[:int(np.ceil(frac*len(e2)))]]=True
            concentrated[str(frac)]=stats(e2,mask)
        r['models'][name]=dict(overall=stats(e2),largest_error_fractions=concentrated,
            physical_face_distance=binned(e2,face,np.array([0,.01,.025,.05,.10,.20,.35,.5])),
            within_5_percent_any_physical_face=stats(e2,near_count>0),
            away_from_all_5_percent_faces=stats(e2,near_count==0),
            near_physical_face_count={str(i):stats(e2,near_count==i) for i in range(6)},
            fitted_chart_boundary=stats(e2,chart_bound),fitted_chart_interior=stats(e2,~chart_bound),
            knob_bins={k:binned(e2,u[:,j],np.linspace(0,1,11)) for j,k in enumerate(KNOBS)},
            cap_row_boundary_distance=binned(e2,cap_row_dist,np.linspace(0,.5,11)),
            nearest_reference_setting_gap=relation(e2,setting_gap))
        pairs={}
        for a,b in itertools.combinations(range(5),2):
            count=np.histogram2d(u[:,a],u[:,b],bins=10,range=[[0,1],[0,1]])[0]
            energy=np.histogram2d(u[:,a],u[:,b],bins=10,range=[[0,1],[0,1]],weights=e2)[0]
            pairs[KNOBS[a]+'_'+KNOBS[b]]=dict(axes=[KNOBS[a],KNOBS[b]],counts=count.tolist(),
                rms_image_L2=np.sqrt(energy/np.maximum(count,1)).tolist(),squared_error=energy.tolist())
        r['models'][name]['knob_pair_bins']=pairs
    print('Full-cloud concentration',json.dumps({k:v['largest_error_fractions'] for k,v in r['models'].items()}),flush=True)
    # Common interior cohort: all requested symmetric knob probes remain inside
    # the legal box; physical-boundary results above still use the full cloud.
    rng=np.random.default_rng(1008);eligible=np.flatnonzero(face>.025)
    ids=rng.choice(eligible,1024,replace=False);base=x[ids];pp=p[ids];err={k:v[ids] for k,v in sse.items()}
    jac=[];h0=.001
    for k in range(5):
        delta=np.zeros(5);delta[k]=h0*np.ptp(RANGES[k])
        minus=pixels(pp-delta)[:,active];plus=pixels(pp+delta)[:,active]
        jac.append((plus-minus)/(2*h0))
    jac=np.stack(jac,axis=2);q,s,_=np.linalg.svd(jac,full_matrices=False)
    keep=s>s[:,0,None]*1e-6;q=q*keep[:,None,:]
    diag=rng.normal(size=(5,5));diag/=np.linalg.norm(diag,axis=1,keepdims=True)
    directions=np.concatenate([np.eye(5),diag]);names=KNOBS+[f'joint direction {i+1}' for i in range(5)]
    curvature_data={};pixel_gap=cKDTree(m['reference_images']).query(base,workers=4)[0]
    r['curvature']=dict(points=len(ids),eligible_interior_points=len(eligible),
        minimum_physical_face_fraction=.025,indices_seed=1008,
        direction_names=names,joint_directions_normalized_setting_space=diag.tolist(),
        definition='Each path changes normalized knobs along a fixed direction. Turn angle compares before-to-center and center-to-after pixel vectors. Midpoint chord error measures their average minus the exact center. Normal chord error removes the local five-direction tangent span (central differences at 0.001 range fractions); normal bending proxy = 2 * normal chord error / mean endpoint-to-center pixel distance squared. It discounts acceleration inside the tangent span. These are scale-dependent proxies, not an invariant tensor.',
        scales={},nearest_reference_pixel_gap={k:relation(v,pixel_gap) for k,v in err.items()})
    for h in [.005,.02]:
        angle=[];bend=[];normal=[];curvature=[];change=[]
        for k,direction in enumerate(directions):
            delta=h*direction*np.ptp(RANGES,axis=1)
            minus=pixels(pp-delta)[:,active];plus=pixels(pp+delta)[:,active]
            left=base-minus;right=plus-base
            l=np.linalg.norm(left,axis=1);rr=np.linalg.norm(right,axis=1)
            cos=np.sum(left*right,axis=1)/np.maximum(l*rr,1e-20)
            angle.append(np.degrees(np.arccos(np.clip(cos,-1,1))))
            chord=(minus+plus)/2-base
            projection=np.einsum('nik,ni->nk',q,chord)
            orthogonal=chord-np.einsum('nik,nk->ni',q,projection)
            nb=np.linalg.norm(orthogonal,axis=1);step=(l+rr)/2
            bend.append(np.linalg.norm(chord,axis=1));normal.append(nb)
            curvature.append(2*nb/np.maximum(step*step,1e-20));change.append(step)
        angle=np.array(angle).T;bend=np.array(bend).T;normal=np.array(normal).T
        curvature=np.array(curvature).T;change=np.array(change).T
        features={'mean_normal_bending':np.mean(curvature,axis=1),
            'max_normal_bending':np.max(curvature,axis=1),'max_turn_angle_degrees':np.max(angle,axis=1),
            'max_midpoint_chord_error':np.max(bend,axis=1)}
        analyses={name:{k:relation(v,feature) for k,v in err.items()} for name,feature in features.items()}
        per_direction={name:dict(normal_bending={k:relation(v,curvature[:,j]) for k,v in err.items()},
            turn_angle={k:relation(v,angle[:,j]) for k,v in err.items()}) for j,name in enumerate(names)}
        r['curvature']['scales'][str(h)]=dict(range_fraction_step=h,feature_associations=analyses,per_direction=per_direction)
        curvature_data[str(h)]=dict(angle=angle,bend=bend,normal=normal,curvature=curvature,change=change)
        print('Curvature scale',h,json.dumps(analyses),flush=True)
    examples=[];labels=[]
    e2=sse['clipped'];high=np.argsort(e2)[::-1]
    examples.extend(high[:3]);labels.extend(['Largest clipped error','Second-largest clipped error','Third-largest clipped error'])
    examples.append(int(np.argmin(abs(np.sqrt(e2)-np.median(np.sqrt(e2))))));labels.append('Typical clipped error')
    hdata=curvature_data['0.02'];feature=np.mean(hdata['curvature'],axis=1)
    high_bend=np.flatnonzero(feature>=np.quantile(feature,.9));low_bend=np.flatnonzero(feature<=np.quantile(feature,.1))
    for label,candidates,extreme in [('Strong bending, small error',high_bend,False),('Weak bending, large error',low_bend,True)]:
        e=err['clipped'][candidates];j=candidates[np.argmax(e) if extreme else np.argmin(e)]
        examples.append(int(ids[j]));labels.append(label)
    r['examples']=[dict(index=int(i),label=label,knobs=p[i].tolist(),raw_L2=float(np.sqrt(sse['raw'][i])),
        clipped_L2=float(np.sqrt(e2[i])),physical_face_fraction=float(face[i]),
        chart_boundary=bool(chart_bound[i])) for i,label in zip(examples,labels)]
    # Pixel-space map: where in an image is squared error accumulated?
    pixel_maps={}
    for name,res in residuals.items():
        arr=np.zeros(784);arr[active]=np.sqrt(np.mean(res*res,axis=0));pixel_maps[name]=arr.reshape(28,28).tolist()
    r['pixel_RMS_maps']=pixel_maps
    squared=(clipped-x)**2;rows=np.flatnonzero(active)//28
    cap=(abs(rows[None,:]-np.floor(top)[:,None])<=1)|(abs(rows[None,:]-np.floor(bottom)[:,None])<=1)
    r['pixel_regions']=dict(definition='Cap bands include the pixel row containing each cap plus its preceding and following row (six rows in total for this family). Other rows are the stroke-body region. Pixel classes use exact source coverage.',
        cap_band_squared_error_percent=100*float(squared[cap].sum()/squared.sum()),
        **{name+'_squared_error_percent':100*float(squared[mask].sum()/squared.sum())
           for name,mask in [('empty',x==0),('fractional',(x>0)&(x<1)),('full',x==1)]})
    violation=np.sum((raw-clipped)**2,axis=1)
    r['bounds_violation_energy_definition']='Sum of squared distances from raw predicted pixels to [0,1]. This measures scalar overshoot independently of target error. It is different from the total error removed by clipping.'
    r['bounds_violation_energy']=stats(violation)
    for h,data in curvature_data.items():
        r['curvature']['scales'][h]['bounds_violation_association']=relation(violation[ids],np.mean(data['curvature'],axis=1))
    reference_pred=decode(m['coordinates'],m['nodes'],m['shape']);ref_u=(m['coverage_knobs']-RANGES[:,0])/np.ptp(RANGES,axis=1)
    reference_near=np.min(np.minimum(ref_u,1-ref_u),axis=1)<.05
    r['covering_cloud_check']=dict(count=len(reference_pred),
        method='Decode at the final fitted reference coordinates; no additional projection. The cloud includes lattice and near-face oversampling, unlike the uniform 16,384-point diagnostic.',models={})
    for name,y in [('raw',reference_pred),('clipped',np.clip(reference_pred,0,1))]:
        e2=np.sum((y-m['reference_images'])**2,axis=1)
        r['covering_cloud_check']['models'][name]=dict(overall=stats(e2),near_5_percent_limits=stats(e2,reference_near),away_from_5_percent_limits=stats(e2,~reference_near))
    arrays=dict(indices=ids,knobs=pp,truth=base,raw=raw[ids],clipped=clipped[ids],
        nearest_reference_pixel_gap=pixel_gap,nearest_reference_setting_gap=setting_gap[ids],
        physical_face_fraction=face[ids],active=active,examples_indices=examples,example_labels=labels)
    for h,data in curvature_data.items():
        for name,v in data.items():arrays[f'{h}_{name}']=v
    np.savez_compressed(OUT/'curvature_samples.npz',**arrays)
    np.savez_compressed(OUT/'image_examples.npz',indices=examples,labels=labels,knobs=p[examples],
        truth=x[examples],raw=raw[examples],clipped=clipped[examples],active=active)
    (OUT/'results.json').write_text(json.dumps(r,indent=2)+'\n')


if __name__=='__main__':main()
