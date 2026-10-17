"""Local branch-discovery tests. Oracle restricted to this driver."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json,hashlib,sys,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from grey_ones import render,RANGES
from learn import global_polynomial,error_stats,moments
from crease import fit_crease,predict_crease
OUT=ROOT/'results/generated_one_data_only'


def run():
    start=time.monotonic();rng=np.random.default_rng(2903)
    center=np.array([14.451,14.615,19.73,3.17,17.3])
    radius=1e-4;span=np.ptp(RANGES,axis=1)
    p=center+rng.uniform(-1,1,(512,5))*span*radius
    keep=json.loads((OUT/'frozen_model.json').read_text())['selected_measurements']
    records=dict(local_radius_fraction=radius,sampling_note='Evaluator isolates one transition; learner sees arrays only.',conditions={})
    for dtype in (np.float64,np.float32):
        storage=np.dtype(dtype).name
        x=render(p,dtype=dtype)
        np.savez_compressed(OUT/f'crease_training_{storage}_public.npz',coordinates=p,images=x)
        for condition in ('supplied_coordinates','images_only'):
            train=p if condition=='supplied_coordinates' else moments(x)[:,keep]
            model=fit_crease(train,x)
            label=condition+'_'+storage
            saved={k:v for k,v in model.items() if isinstance(v,(np.ndarray,float,int))}
            np.savez_compressed(OUT/f'crease_model_{label}.npz',**saved)
            frozen_hash=hashlib.sha256((OUT/f'crease_model_{label}.npz').read_bytes()).hexdigest()
            # Fresh independent states, created after freezing this model.
            test=center+rng.uniform(-1,1,(1024,5))*span*radius
            truth=render(test,dtype=dtype)
            query=test if condition=='supplied_coordinates' else moments(truth)[:,keep]
            predicted=predict_crease(query,model)
            polynomial,_=global_polynomial(train,x,query,3)
            normal=model['normal']/model['scale']
            offset=float(model['offset']+model['normal']@(model['center']/model['scale']))
            denominator=np.max(np.abs(normal))
            row=dict(training_images=len(x),test_images=len(test),crease=error_stats(truth,predicted),
                ordinary_cubic=error_stats(truth,polynomial),
                plane_normal_in_input_units=(normal/denominator).tolist(),plane_offset_in_input_units=offset/denominator,
                training_plane_loss=model['training_plane_loss'],optimization=model['optimization'],
                model_sha256=frozen_hash,coefficient_count=int(model['coefficient'].size))
            records['conditions'][label]=row
            np.savez_compressed(OUT/f'crease_test_{label}_public.npz',coordinates=query,images=truth,
                predicted=predicted,ordinary_cubic=polynomial)
            print(label,row['crease']['max_pixel'],row['crease']['max_image_l2'],
                  'plane',row['plane_normal_in_input_units'],row['plane_offset_in_input_units'],flush=True)
    records['source_sha256']={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in
        [Path(__file__),Path(__file__).with_name('crease.py'),Path(__file__).with_name('HINGE_PROTOCOL.md')]}
    records['elapsed_seconds']=time.monotonic()-start
    (OUT/'crease_results.json').write_text(json.dumps(records,indent=2,allow_nan=False)+'\n')
    print('Finished in',records['elapsed_seconds'],'seconds.',flush=True)

if __name__=='__main__':run()
