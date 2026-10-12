"""Audit saved trials and isolate fixed-sampling versus changing-sample effects."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
import gzip
import numpy as np
from analyze_generated_one_direction_history import OUT,predict,choose,record_match
from check_generated_one_direction_commutators import tangent_limit_basis
from analyze_generated_one_manifold_transport import jet
from analyze_generated_one_nearest_direction import assign,local_fit,pixels,is_cap_kink


def track_frames(frames):
    basis,expected=frames[0]
    histories={m:[basis.copy()] for m in ['nearest','trend']};truth=[basis.copy()]
    records=[]
    for hop,(candidate,labels) in enumerate(frames[1:],1):
        for mode in ['rollout','correct_past_diagnostic']:
            for method in histories:
                history=histories[method] if mode=='rollout' else truth
                selected,scores=choose(predict(history,method=='trend'),candidate,False)
                row=record_match(dict(hop=hop),mode,method,selected,scores,expected,labels)
                records.append(row)
                if mode=='rollout':
                    aligned=candidate[selected].copy()
                    signs=np.sign(np.sum(aligned*history[-1],axis=1));signs[signs==0]=1
                    history.append(aligned*signs[:,None])
                    if len(history)>2:history.pop(0)
        indices=[np.flatnonzero(labels==i)[0] for i in expected]
        aligned=candidate[indices].copy()
        signs=np.sign(np.sum(aligned*truth[-1],axis=1));signs[signs==0]=1
        truth.append(aligned*signs[:,None])
        if len(truth)>2:truth.pop(0)
    return records


def aggregate(records):
    output=[]
    for mode in ['rollout','correct_past_diagnostic']:
        for method in ['nearest','trend']:
            for step in sorted(set(r['step'] for r in records)):
                for scope in ['after_bootstrap','hop8']:
                    selected=[r for r in records if r['mode']==mode and r['method']==method and r['step']==step and
                              (r['hop']>=2 if scope=='after_bootstrap' else r['hop']==8)]
                    output.append(dict(mode=mode,method=method,step=step,scope=scope,
                                       correct=sum(r['correct'] for r in selected),count=5*len(selected)))
    return output


def main():
    with gzip.open(OUT/'history_trials.json.gz','rt') as f:result=json.load(f)
    for row in result['comparisons']:
        assert row['correct']==sum(row['correct_flags'])
        assert row['duplicate_choices']==5-len(set(row['chosen_labels']))
    # Check every generated single-control step, batching rendered distances.
    maximum_distance_error=0.
    for path in result['paths']:
        states=np.array(path['states'])
        if len(states)<2:continue
        differences=np.diff(states,axis=0)
        axis=['cx','cy','height','width','lean'].index(path['moved_knob'])
        assert np.all(np.count_nonzero(differences,axis=1)==1)
        assert np.all(differences[:,axis]*path['sign']>0)
        rendered=pixels(states);distances=np.linalg.norm(np.diff(rendered,axis=0),axis=1)
        maximum_distance_error=max(maximum_distance_error,float(np.max(np.abs(distances-path['step']))))
    assert maximum_distance_error<1e-10
    # Analytic radius-zero SVD, using the same fixed 256 rays at every state.
    # This is a diagnostic control, not the pixel-only fitted experiment.
    records=[];included_paths=set();excluded_paths=0
    for path in result['paths']:
        if any(is_cap_kink(np.array(settings)) for settings in path['states']):
            excluded_paths+=1
            continue
        included_paths.add(path['path'])
        frames=[]
        for settings in path['states']:
            p=np.array(settings);basis=tangent_limit_basis(p,20261031)
            labels=assign(basis,jet(p)[3])[0]
            frames.append((basis,labels))
        for row in track_frames(frames):records.append(dict(path=path['path'],step=path['step'],**row))
    sampled=[row for row in result['comparisons'] if row['path'] in included_paths and
             row['field']=='svd' and row['method'] in ['nearest','trend']]
    control=dict(method='Analytic tangent-limit SVD of 256 fixed parameter rays; known J, no finite-shell/sample changes',
                 excluded_paths_with_tilted_cap_kink=excluded_paths,
                 aggregate=aggregate(records),sampled_SVD_on_same_paths=aggregate(sampled))
    (OUT/'history_tangent_limit_control.json').write_text(json.dumps(control,indent=2)+'\n')
    # Fresh SVD neighborhoods at a stationary image isolate sample variation.
    stationary=[]
    for start in result['starts']:
        p=np.array(start['settings']);frames=[]
        for hop in range(9):
            fit=local_fit(p,1e-5,20261031+hop*1009)
            frames.append((fit['basis'],fit['labels']))
        stationary.extend(dict(name=start['name'],step=0,**r) for r in track_frames(frames))
    report=dict(maximum_step_distance_error=maximum_distance_error,
                stationary_fresh_sample_control=aggregate(stationary))
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Saved trial counts and all rendered step distances verified; maximum error:',maximum_distance_error)
    print('Tangent-limit and stationary controls complete.')


if __name__=='__main__':main()
