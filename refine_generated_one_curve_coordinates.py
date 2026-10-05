"""Improve coordinate selection while keeping every fitted curve node fixed."""
import json
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import numpy as np
from analyze_generated_one_principal_curves import OUT, project, error_stats, pixels

FOLLOW=OUT/'followup'


def refine(x,mean,curves,max_cycles=30):
    residual=x-mean;parts=[]
    for c in curves:
        f,_,_=project(residual,c);parts.append(f);residual-=f
    original=float(np.mean(np.sum(residual**2,axis=1)))
    history=[original];coordinates=[]
    for cycle in range(max_cycles):
        coordinates=[]
        for k,c in enumerate(curves):
            target=residual+parts[k]
            f,t,_=project(target,c)
            residual=target-f;parts[k]=f;coordinates.append(t)
        loss=float(np.mean(np.sum(residual**2,axis=1)))
        assert loss<=history[-1]+1e-10
        history.append(loss)
        if cycle%5==4:print('Coordinate refinement cycle',cycle+1,'SSE/image',loss,flush=True)
        if history[-2]-history[-1]<1e-6:break
    return x-residual,np.column_stack(coordinates),history


def main():
    s=np.load(OUT/'model_and_samples.npz')
    x=s['test_images'];pred,t,h=refine(x,s['mean'],s['curves'])
    old=float(np.sum((x-s['reconstructed'])**2));new=float(np.sum((x-pred)**2))
    result=dict(original_fixed_curves=dict(metrics=error_stats(x-pred,x,s['mean']),
        fraction_of_original_missing_error_removed_percent=float(100*(1-new/old)),
        mean_squared_error_by_cycle=h,curve_nodes_unchanged=True))
    np.savez_compressed(FOLLOW/'refined_original_coordinates.npz',reconstructed=pred,coordinates=t)
    if (FOLLOW/'models_and_examples.npz').exists():
        a=np.load(FOLLOW/'models_and_examples.npz');xx=pixels(a['coverage_knobs'])
        pp,tt,hh=refine(xx,a['mean'],a['curves'],20)
        result['all_point_fixed_curves']=dict(metrics=error_stats(xx-pp,xx,a['mean']),mean_squared_error_by_cycle=hh)
        np.savez_compressed(FOLLOW/'refined_all_point_coordinates.npz',coordinates=tt)
    (FOLLOW/'coordinate_refinement.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
