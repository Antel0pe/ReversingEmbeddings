"""Compare true reference-image chords with exact knob midpoints and learned fits."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import json
import numpy as np
from scipy.spatial import cKDTree
from grey_ones import RANGES, KNOBS
from analyze_generated_one_principal_curves import pixels
from analyze_generated_one_manifold_failures import projected_images
from fit_generated_one_principal_manifold import OUT as MODEL_DIR, decode

OUT=MODEL_DIR/'midpoints'


def summary(values):
    return dict(mean=float(np.mean(values)),median=float(np.median(values)),
                p10=float(np.quantile(values,.1)),p95=float(np.quantile(values,.95)),max=float(np.max(values)))


def errors(truth,pred):
    e=pred-truth;sq=e*e;l2=np.linalg.norm(e,axis=1);total=float(sq.sum())
    masks={'empty':truth==0,'fractional':(truth>0)&(truth<1),'full':truth==1}
    return dict(image_L2=summary(l2),rms_image_L2=float(np.sqrt(np.mean(l2*l2))),
                rms_pixel=float(np.sqrt(np.mean(l2*l2)/784)),
                maximum_pixel_error=float(np.max(abs(e))),
                error_share_percent={k:100*float(sq[v].sum())/max(total,1e-30) for k,v in masks.items()},
                pixels_outside_0_1=int(np.sum((pred<0)|(pred>1))),
                images_outside_0_1_above_1e_6=int(np.sum(np.any((pred<-1e-6)|(pred>1+1e-6),axis=1))))


def main():
    OUT.mkdir(exist_ok=True);rng=np.random.default_rng(1007)
    m=np.load(MODEL_DIR/'fine.npz');p=m['coverage_knobs'];x=m['reference_images'];n=len(x)
    tree=cKDTree(x);probe=rng.choice(n,1024,replace=False)
    nn,ni=tree.query(x[probe],k=2,workers=4)
    pairs=[];groups=[]
    def add(label,ab):
        pairs.extend(ab);groups.extend([label]*len(ab))
    unique=np.unique(np.sort(np.column_stack([probe,ni[:,1]]),axis=1),axis=0)
    add('Nearest reference pairs',unique[rng.choice(len(unique),192,replace=False)])
    un=(p-RANGES[:,0])/np.ptp(RANGES,axis=1)
    lattice=np.flatnonzero(np.max(abs(un*4-np.round(un*4)),axis=1)<1e-9)
    ptree=cKDTree(un)
    for k,name in enumerate(KNOBS):
        a=lattice[un[lattice,k]<.99];a=rng.choice(a,24,replace=False)
        target=un[a].copy();target[:,k]+=.25
        distance,b=ptree.query(target);assert distance.max()<1e-9
        add('Adjacent lattice: '+name,np.column_stack([a,b]))
    bands=[('Distance 1',.5,1.5),('Distance 2',1.5,2.5),('Distance 3',2.5,3.5),
           ('Distance 4',3.75,4.25),('Distance 5',4.75,5.25)]
    selected={name:[] for name,_,_ in bands}
    for batch in range(150):
        ab=rng.integers(n,size=(5000,2));distance=np.linalg.norm(x[ab[:,0]]-x[ab[:,1]],axis=1)
        for name,lo,hi in bands:
            wanted=128-len(selected[name])
            if wanted>0:selected[name].extend(ab[(distance>=lo)&(distance<hi)][:wanted])
        if all(len(v)==128 for v in selected.values()):break
    assert all(len(v)==128 for v in selected.values())
    for name,_,_ in bands:add(name,selected[name])
    ab=np.array(pairs);groups=np.array(groups);pa=p[ab[:,0]];pb=p[ab[:,1]];mid=(pa+pb)/2
    truth=pixels(mid)[:,m['active']];average=(x[ab[:,0]]+x[ab[:,1]])/2
    print('Projecting',len(truth),'true midpoint images onto the unchanged learned manifold',flush=True)
    learned,u=projected_images(truth,m);clipped=np.clip(learned,0,1)
    chart_midpoint=decode((m['coordinates'][ab[:,0]]+m['coordinates'][ab[:,1]])/2,m['nodes'],m['shape'])
    distance=np.linalg.norm(x[ab[:,0]]-x[ab[:,1]],axis=1)
    near_dist,_=tree.query(truth,k=1,workers=4)
    result=dict(reference_count=n,seed=1007,pair_count=len(ab),knob_order=KNOBS,
        reference_sampling='10,240 uniform samples + 3,125 lattice points + 1,024 near-face points; no distance-4 selection rule.',
        midpoint_definition='Arithmetic mean of the five physical knob settings. This selects a definite valid image; pixel space alone has no unique true midpoint.',
        metric='Image L2 = sqrt(sum squared coverage errors over all 784 pixels). RMS image L2 aggregates squared error before averaging. Pixel RMS = RMS image L2 / 28.',
        nearest_reference_distance=summary(nn[:,1]),nearest_reference_probe_count=len(probe),
        projection='Same unchanged fine model and pixel-only projection as failure analysis: two nearest reference starts, 20 steps; residual > 0.35 gets eight starts and 60 steps, retaining lower error.',
        groups={})
    for name in dict.fromkeys(groups):
        take=groups==name;t=truth[take];a=average[take];c=clipped[take]
        ea=a-t;ec=c-t;norm=np.linalg.norm(ea,axis=1)*np.linalg.norm(ec,axis=1)
        nonzero=norm>1e-12
        cosine=np.sum(ea*ec,axis=1)[nonzero]/norm[nonzero]
        row=dict(count=int(take.sum()),endpoint_distance=summary(distance[take]),
                 midpoint_to_nearest_reference_distance=summary(near_dist[take]),
                 pixel_average=errors(t,a),learned_raw=errors(t,learned[take]),learned_clipped=errors(t,c),
                 mean_fitted_chart_coordinates=errors(t,chart_midpoint[take]),
                 ratio_average_to_clipped_manifold_rms=float(np.linalg.norm(ea)/np.linalg.norm(ec)),
                 average_larger_error_percent=100*float(np.mean(np.linalg.norm(ea,axis=1)>np.linalg.norm(ec,axis=1))),
                 signed_residual_cosine_median=float(np.median(cosine)) if len(cosine) else None)
        result['groups'][str(name)]=row
        print(name,'distance',row['endpoint_distance']['mean'],'average RMS',row['pixel_average']['rms_image_L2'],
              'manifold clipped RMS',row['learned_clipped']['rms_image_L2'],flush=True)
    # Both cohorts are comparisons, not a causal attribution of training error.
    old=json.loads((MODEL_DIR/'failures/pixels.json').read_text())['models']['manifold']
    result['previous_uniform_failure_analysis']={k:old[k] for k in ['raw','clipped','clipping_removed_error_percent']}
    # Hold the 128 distance-4 targets and motion directions fixed while shrinking
    # both endpoint knob offsets. These extra exact renders are diagnostic only.
    target_ids=np.flatnonzero(groups=='Distance 4');center=mid[target_ids]
    half=(pb[target_ids]-pa[target_ids])/2
    scales=[1.,.5,.25,.125,.0625];av=[];aa=[];bb=[]
    result['controlled_same_midpoints']=dict(count=len(target_ids),
        meaning='Same distance-4 midpoint images and same physical-knob path directions. Shrink endpoint offsets symmetrically by each fraction, render exact new endpoints, and average their pixels. Except fraction 1, endpoints are additional diagnostic renders, not original fitting references. No refit.',
        learned_raw=errors(truth[target_ids],learned[target_ids]),
        learned_clipped=errors(truth[target_ids],clipped[target_ids]),fractions={})
    for fraction in scales:
        a=pixels(center-fraction*half)[:,m['active']];b=pixels(center+fraction*half)[:,m['active']]
        blend=(a+b)/2;aa.append(a);bb.append(b);av.append(blend)
        result['controlled_same_midpoints']['fractions'][str(fraction)]=dict(
            endpoint_offset_fraction=fraction,endpoint_distance=summary(np.linalg.norm(a-b,axis=1)),
            pixel_average=errors(truth[target_ids],blend))
        print('Fixed-target shrink',fraction,'average RMS',errors(truth[target_ids],blend)['rms_image_L2'],flush=True)
    np.savez_compressed(OUT/'controlled_spacing.npz',pair_indices=target_ids,fractions=scales,
        endpoint_A=np.array(aa),endpoint_B=np.array(bb),average=np.array(av),truth=truth[target_ids],
        learned_raw=learned[target_ids],active=m['active'])
    ae=np.linalg.norm(average-truth,axis=1);ce=np.linalg.norm(clipped-truth,axis=1)
    examples=[];labels=[]
    def choose(label,candidates,criterion):
        i=int(candidates[np.argmin(abs(ae[candidates]-criterion))]);examples.append(i);labels.append(label)
    local=np.flatnonzero(groups=='Nearest reference pairs');d4=np.flatnonzero(groups=='Distance 4')
    choose('Typical nearest-reference pair',local,np.median(ae[local]))
    choose('Typical distance-4 pair',d4,np.median(ae[d4]))
    choose('Smallest averaging error at distance 4',d4,np.min(ae[d4]))
    choose('Largest averaging error at distance 4',d4,np.max(ae[d4]))
    for name in KNOBS:
        q=np.flatnonzero(groups=='Adjacent lattice: '+name)
        choose('Lattice step: '+name,q,np.median(ae[q]))
    result['display_examples']=[dict(label=label,pair_index=i,reference_indices=ab[i].tolist(),
        endpoint_distance=float(distance[i]),average_L2=float(ae[i]),learned_clipped_L2=float(ce[i]),
        endpoint_A_knobs=pa[i].tolist(),endpoint_B_knobs=pb[i].tolist(),midpoint_knobs=mid[i].tolist())
        for i,label in zip(examples,labels)]
    result['limits']='Conditional pair samples and nearest-neighbor probes are finite diagnostics. Mean physical knobs is an intended midpoint, not a geodesic or closest valid image to the chord. The reference cloud and fitted grid are unchanged. The comparisons do not apportion learned error causally to sample spacing, grid resolution, regularization, or projection.'
    np.savez_compressed(OUT/'pairs.npz',reference_indices=ab,groups=groups,endpoint_A_knobs=pa,endpoint_B_knobs=pb,
        midpoint_knobs=mid,endpoint_A=x[ab[:,0]],endpoint_B=x[ab[:,1]],truth=truth,average=average,
        learned_raw=learned,learned_clipped=clipped,chart_midpoint=chart_midpoint,coordinates=u,
        endpoint_distance=distance,nearest_reference_distance=near_dist,active=m['active'],
        example_indices=examples,example_labels=labels)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
