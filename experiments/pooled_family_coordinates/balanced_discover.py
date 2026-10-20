"""Acquisition-quality comparison; no physical coordinate inputs or targets."""
from pathlib import Path
import sys
import os
import json
import hashlib
import numpy as np
from common import Features,align_frames,ridge_path,LocalPolynomial,decoder_selection,image_metrics,response_metrics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'figures/pooled_family_coordinates';HERE=Path(__file__).resolve().parent
INPUTS=[ROOT/'figures/svd_blind_intermediates/observations.npz',OUT/'patch_observations.npz',OUT/'balanced_frames.npz']
APPROVED={*INPUTS,HERE/'common.py',HERE/'balanced_discover.py',HERE/'QUALITY_CONTROL.md'}


def audit(event,args):
    if event!='open' or not isinstance(args[0],(str,bytes,Path)):return
    p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
    if (isinstance(mode,str) and any(c in mode for c in 'wax+')) or flags&(os.O_WRONLY|os.O_RDWR):return
    if p.name.endswith('_labels.npz'):raise PermissionError('Physical coordinates forbidden')
    if p.is_relative_to(ROOT) and not p.is_relative_to(ROOT/'.venv'):
        if p not in APPROVED and not (p.parent==OUT and p.name in ['balanced_models.npz','balanced_results.json']):
            raise PermissionError('Unapproved read: '+p.name)


def main():
    sys.addaudithook(audit)
    initial=np.load(INPUTS[0]);data=np.load(INPUTS[1]);frames=np.load(INPUTS[2])
    result={};archive={}
    for si in range(2):
        for ri in [0,2]:
            scope=f'patch_{si}_{ri}';result[scope]={}
            x=data[scope+'_train_x'];xv=data[scope+'_val_x']
            fxs=[data[scope+'_'+s+'_frame_x'] for s in ['train','val','test']]
            for method in ['svd','pca']:
                bs=[frames[scope+'_'+s+'_'+method] for s in ['train','val','test']]
                aligned,graph=align_frames(initial['initial_x'],initial['initial_b'],fxs[0],bs[0],list(zip(fxs[1:],bs[1:])))
                result[scope][method]={}
                for kind in ['linear','quadratic']:
                    group='fixed_joint' if kind=='linear' else kind
                    feature=Features(x,fxs[0],aligned[0],kind)
                    d=feature.derivatives(fxs[0],aligned[0]);dv=feature.derivatives(fxs[1],aligned[1])
                    records=[];models=[]
                    for meta,c in ridge_path(d,feature.penalty(fxs[0]),[1e-5,1e-4,.001,.01,.1,1.]):
                        q=feature.values(x)@c;qv=feature.values(xv)@c
                        records.append(dict(**meta,validation_image=image_metrics(xv,LocalPolynomial(q,x,1,48).predict(qv)),
                            validation_response=response_metrics(dv@c)))
                        models.append(c)
                    j=int(np.argmin([a['validation_image']['mean_l2'] for a in records]));c=models[j]
                    prefix=scope+'_'+method+'_'+group
                    center=-(feature.values(x)@c).mean(0)
                    qs={s:feature.values(data[scope+'_'+s+'_x'])@c+center for s in ['train','val','test']}
                    response=feature.derivatives(fxs[2],aligned[2])@c
                    selected,decoders,pred=decoder_selection(qs['train'],x,qs['val'],xv,{'test':qs['test']})
                    row=dict(selected=records[j],attempts=records,decoder_selected=selected,decoder_attempts=decoders,
                        test_response=response_metrics(response,np.linalg.norm(feature.gradient(fxs[2],c),axis=1)),
                        reconstruction={'test':image_metrics(data[scope+'_test_x'],pred['test'])},family_graph=graph)
                    result[scope][method][group]=row
                    feature.save(prefix,archive);archive[prefix+'_coefficient']=c;archive[prefix+'_center']=center
                    archive[prefix+'_test_response']=response
                    for s,q in qs.items():archive[prefix+'_'+s+'_q']=q
                    print(scope,method,group,'off',row['test_response']['off_rms'],'image',row['reconstruction']['test']['mean_l2'],flush=True)
    np.savez_compressed(OUT/'balanced_models.npz',**archive)
    (OUT/'balanced_results.json').write_text(json.dumps(result,indent=2))
    paths=[*INPUTS,HERE/'common.py',HERE/'balanced_discover.py',HERE/'QUALITY_CONTROL.md',OUT/'balanced_models.npz',OUT/'balanced_results.json']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT/'BALANCED_FREEZE.json').write_text(json.dumps(dict(physical_evaluation_started=False,sha256=hashes),indent=2))
    print('BALANCED DISCOVERY FROZEN',flush=True)


if __name__=='__main__':main()
