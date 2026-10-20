"""Pooled constraints and scalar-function gradient fits, without physical labels."""
from pathlib import Path
import os
import sys
import json
import hashlib
import time
import numpy as np
from scipy.optimize import linear_sum_assignment
from common import (Features,align_frames,constrained_path,ridge_path,
    LocalPolynomial,decoder_selection,image_metrics,response_metrics)

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'figures/pooled_family_coordinates'
OLD=ROOT/'figures/svd_blind_intermediates/observations.npz'
NEW=OUT/'patch_observations.npz'
HERE=Path(__file__).resolve().parent
APPROVED={OLD,NEW,*HERE.glob('*.py'),HERE/'PROTOCOL.md'}
READS=set()


def audit(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes,Path)):return
    p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
    write=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR))
    if write:return
    if p.name.endswith('_labels.npz'):raise PermissionError('Physical evaluation data forbidden')
    if p.is_relative_to(ROOT) and not p.is_relative_to(ROOT/'.venv'):
        own_output=p.parent==OUT and p.name in ['models.npz','blind_results.json']
        if p not in APPROVED and not own_output:raise PermissionError('Unapproved repository read: '+p.name)
        READS.add(str(p.relative_to(ROOT)))


def triangle_check(feature,c,points):
    nodes,weights=np.polynomial.legendre.leggauss(24)
    t=(nodes+1)/2;weights=weights/2
    integral=np.zeros(5)
    for a,b in zip(points,np.roll(points,-1,axis=0)):
        x=a+t[:,None]*(b-a)
        derivative=np.einsum('npq,p->nq',feature.gradient(x,c),(b-a)[feature.active])
        integral+=weights@derivative
    return float(np.max(abs(integral)))


def curved_control():
    u=np.linspace(-1,1,101);v=np.sin(u*3)*.2
    x=np.column_stack([u,v+u*u])
    directions=np.stack([np.column_stack([np.ones(len(u)),2*u]),np.tile([0.,1.],(len(u),1))],axis=1)
    results={}
    for quadratic in [False,True]:
        features=np.column_stack([x,x[:,0]**2]) if quadratic else x
        d=np.concatenate([directions,(2*x[:,0,None,None]*directions[:,:,:1])],axis=2) if quadratic else directions
        target=np.tile(np.eye(2),(len(x),1))
        c=np.linalg.lstsq(d.reshape(-1,d.shape[-1]),target,rcond=None)[0]
        c/=np.diagonal(d@c,axis1=1,axis2=2).mean(0)
        response=d@c
        off=response[:,0,1]
        results['quadratic' if quadratic else 'fixed_linear']=dict(
            other_response_rms=float(np.sqrt(np.mean(off**2))),
            own_response_rms_error=float(np.sqrt(np.mean((response[:,1,1]-1)**2))),
            coefficients=c.tolist())
    results['meaning']='State=(u,v+u squared); v=y-x squared is an exact scalar reading with a changing gradient.'
    return results


def main():
    sys.addaudithook(audit)
    old=np.load(OLD);new=np.load(NEW)
    archive={};result=dict(scopes={},curved_control=curved_control(),selection='Image validation only')
    begin=time.monotonic()
    scopes=['local','broad']+[f'patch_{i}_{j}' for i in range(2) for j in range(4)]
    for scope in scopes:
        data=old if scope in ['local','broad'] else new
        x=data[scope+'_train_x'];xv=data[scope+'_val_x'];xt=data[scope+'_test_x']
        fxs=[data[scope+'_'+s+'_frame_x'] for s in ['train','val','test']]
        extra=['path','edge'] if scope in ['local','broad'] else []
        scope_result={}
        for method in ['svd','pca']:
            bs=[data[scope+'_'+s+'_'+method] for s in ['train','val','test']]
            aligned,graph=align_frames(old['initial_x'],old['initial_b'],fxs[0],bs[0],list(zip(fxs[1:],bs[1:])))
            raw=Features(x,fxs[0],aligned[0],'linear')
            entries={};candidates={};feature_map={}
            for kind,band in [('linear',1.),('quadratic',1.),('cubic',1.),('kernel',.75),('kernel',1.5)]:
                feature=raw if kind=='linear' else Features(x,fxs[0],aligned[0],kind,band)
                d=feature.derivatives(fxs[0],aligned[0]);dv=feature.derivatives(fxs[1],aligned[1])
                penalty=feature.penalty(fxs[0])
                group='fixed_joint' if kind=='linear' else kind
                for meta,c in ridge_path(d,penalty,[1e-5,1e-4,.001,.01,.1,1.]):
                    meta.update(kind=kind,bandwidth=band)
                    candidates.setdefault(group,[]).append((meta,feature,c,dv))
                if kind=='linear':
                    for meta,c in constrained_path(d,penalty,[.01,.03,.05,.1,.2,.3,.5,1.]):
                        if c is None:
                            entries.setdefault('pooled_constrained',dict(attempts=[]))['attempts'].append(meta)
                            continue
                        meta.update(kind=kind,bandwidth=band)
                        candidates.setdefault('pooled_constrained',[]).append((meta,feature,c,dv))
            for group,options in candidates.items():
                records=entries.setdefault(group,dict(attempts=[]))['attempts']
                evaluated=[]
                for meta,feature,c,dv in options:
                    q=feature.values(x)@c;qv=feature.values(xv)@c
                    # Fixed inexpensive validation decoder for encoder selection.
                    pred=LocalPolynomial(q,x,1,48).predict(qv)
                    record=dict(**meta,validation_image=image_metrics(xv,pred),
                        validation_response=response_metrics(dv@c),feature_count=c.shape[0])
                    records.append(record);evaluated.append((record,feature,c))
                best,feature,c=min(evaluated,key=lambda e:e[0]['validation_image']['mean_l2'])
                entries[group]['selected']=best;feature_map[group]=(feature,c)
            # Actual global image PC baseline, aligned to anonymous families on training only.
            c=raw.pc[:,:5].copy();response=raw.derivatives(fxs[0],aligned[0])@c
            energy=np.mean(response*response,axis=0)
            row,col=linear_sum_assignment(-energy/np.maximum(energy.sum(0),1e-30))
            c=c[:,col[np.argsort(row)]]
            gain=np.diagonal(raw.derivatives(fxs[0],aligned[0])@c,axis1=1,axis2=2).mean(0)
            gain=np.where(abs(gain)>1e-8,gain,np.ones(5));c/=gain
            entries['global_pc']=dict(selected=dict(description='Five ordinary global image PCs'))
            feature_map['global_pc']=(raw,c)
            for group,(feature,c) in feature_map.items():
                prefix=scope+'_'+method+'_'+group
                center=-(feature.values(x)@c).mean(0)
                qs={s:feature.values(data[scope+'_'+s+'_x'])@c+center for s in ['train','val','test']+extra}
                responses=[feature.derivatives(xx,bb)@c for xx,bb in zip(fxs,aligned)]
                gradient=feature.gradient(fxs[2],c)
                entries[group]['test_response']=response_metrics(responses[2],np.linalg.norm(gradient,axis=1))
                selected,attempts,pred=decoder_selection(qs['train'],x,qs['val'],xv,{s:qs[s] for s in ['test']+extra})
                entries[group]['decoder_selected']=selected;entries[group]['decoder_attempts']=attempts
                entries[group]['reconstruction']={s:image_metrics(data[scope+'_'+s+'_x'],v) for s,v in pred.items()}
                entries[group]['closed_triangle_integral']=triangle_check(feature,c,fxs[2][:3])
                entries[group]['retained_decoder_training_values']=int(x.size+qs['train'].size)
                feature.save(prefix,archive);archive[prefix+'_coefficient']=c;archive[prefix+'_center']=center
                archive[prefix+'_test_response']=responses[2]
                archive[prefix+'_test_example_prediction']=pred['test'][:32]
                for s,q in qs.items():archive[prefix+'_'+s+'_q']=q
                print(scope,method,group,'off',entries[group]['test_response']['off_rms'],
                      'own variation',entries[group]['test_response']['own_rms_variation'],
                      'image',entries[group]['reconstruction']['test']['mean_l2'],flush=True)
            scope_result[method]=dict(methods=entries,family_graph=graph,
                pointwise_control=dict(off_rms=0.,own_variation=0.,meaning='At each state the orthonormal basis is its own dual. This is not a learned scalar readout.'))
        result['scopes'][scope]=scope_result
    result['elapsed_seconds']=time.monotonic()-begin
    result['approved_repository_reads']=sorted(READS)
    np.savez_compressed(OUT/'models.npz',**archive)
    (OUT/'blind_results.json').write_text(json.dumps(result,indent=2))
    inputs=[OLD,NEW,HERE/'PROTOCOL.md',HERE/'common.py',HERE/'discover.py',OUT/'models.npz',OUT/'blind_results.json']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (OUT/'FREEZE.json').write_text(json.dumps(dict(physical_labels_opened=False,sha256=hashes),indent=2))
    print('DISCOVERY FROZEN',flush=True)


if __name__=='__main__':main()
