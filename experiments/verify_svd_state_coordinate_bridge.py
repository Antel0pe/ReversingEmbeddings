"""Replay exported models without passing hidden query geometry to inference."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.svd_state_coordinate_bridge import (
    OUT,invert_five_scores,local_samples,pixels,frame,fromq,features,chart,SUB)


def main():
    r=json.loads((OUT/'results.json').read_text());a=dict(np.load(OUT/'models_and_examples.npz'))
    checks=[]
    for model in r['algebraic_calibrated_charts']:
        prefix='algebraic_'+model['name']+'_'+model['method']
        b=a[prefix+'_basis'];x0=a[prefix+'_source_image'];q0=a[prefix+'_q_origin'];half=a[prefix+'_q_scale']
        parts={tuple(k):a[prefix+f'_piece_{int(k[0])}_{int(k[1])}'] for k in a[prefix+'_piece_keys']}
        x=a[prefix+'_true_images']
        q,pred,count=invert_five_scores((x-x0)@b.T,b,x0,q0,half,parts)
        assert np.all(count==1) and np.max(np.abs(pred-x))<1e-12
        rot=np.linalg.qr(np.random.default_rng(2026102501).normal(size=(5,5)))[0]
        rb=rot@b
        qr,pr,cr=invert_five_scores((x-x0)@rb.T,rb,x0,q0,half,parts)
        assert np.all(cr==1) and np.max(np.abs(pr-x))<1e-12
        p=np.array(model['reference']);_,neighbors=local_samples(p,.001,64,2026102502,False)
        nb,_,_=frame(neighbors-pixels(p)[0],model['method']=='pca')
        qn,pn,cn=invert_five_scores((x-x0)@nb.T,nb,x0,q0,half,parts)
        assert np.all(cn==1) and np.max(np.abs(pn-x))<1e-12
        # All five original knob changes in two application orders. Queries
        # remain in this small local box, including across its cap boundaries.
        rng=np.random.default_rng(2026102503)
        endpoint_q=q0+(rng.random((64,5))*2-1)*half*.01
        endpoint_p=fromq(endpoint_q);commands=endpoint_p-fromq(q0)
        target=pixels(endpoint_p)
        order_errors=[]
        final_images=[]
        for order in [list(range(5)),list(reversed(range(5)))]:
            current=np.tile(x0,(64,1))
            for axis in order:
                cq,_,counts=invert_five_scores((current-x0)@b.T,b,x0,q0,half,parts)
                assert np.all(counts==1)
                cp=fromq(cq);cp[:,axis]+=commands[:,axis]
                nq=np.array([chart(v) for v in cp]);z=(nq-q0)/half
                assert np.max(np.abs(z))<1
                keys=np.floor(nq[:,:2]*SUB+1e-10).astype(int)
                new=np.zeros_like(current);matched=np.zeros(64,int)
                for key,beta in parts.items():
                    use=np.all(keys==key,axis=1)
                    new[use]=x0+features(z[use],2,True)@beta;matched[use]+=1
                assert np.all(matched==1)
                current=new
            order_errors.append(float(np.max(np.abs(current-target))))
            final_images.append(current)
        assert max(order_errors)<1e-12
        checks.append(dict(name=model['name'],method=model['method'],saved_model_replay_max_pixel=float(np.max(np.abs(pred-x))),
            random_basis_rotation_max_pixel=float(np.max(np.abs(pr-x))),fresh_64_neighbor_frame_max_pixel=float(np.max(np.abs(pn-x))),
            five_original_knob_orders_max_pixel_errors=order_errors,
            between_orders_max_pixel=float(np.max(np.abs(final_images[0]-final_images[1])))))
    r['independent_saved_model_checks']=checks
    (OUT/'results.json').write_text(json.dumps(r,indent=2,allow_nan=False))
    print(json.dumps(checks,indent=2))


if __name__=='__main__':main()
