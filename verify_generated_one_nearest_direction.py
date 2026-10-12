"""Repeat representative ordinary-SVD trials with 1,024 sample neighbors."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
import gzip
import numpy as np
import analyze_generated_one_nearest_direction as experiment
from analyze_generated_one_manifold_transport import jet


def verify_rare_true_field_failures():
    """Check the oracle's earliest failures with rendered differences and SVD."""
    experiment.TRAIN=256
    with gzip.open(experiment.OUT/'results.json.gz','rt') as f:
        result=json.load(f)
    rare=[r for r in result['oracle_comparisons'] if r['distance']==.1 and not r['strict_correct']]
    cache={};rows=[];maximum_difference_error=0.
    for row in rare:
        for settings in [row['start_settings'],row['end_settings']]:
            p=np.array(settings);jacobian=jet(p)[3]
            for knob in [1,2]:
                h=1e-6;step=np.eye(5)[knob]*h
                difference=(experiment.pixels(p+step)[0]-experiment.pixels(p-step)[0])/(2*h)
                error=float(np.linalg.norm(difference-jacobian[:,knob]))
                maximum_difference_error=max(maximum_difference_error,error)
                assert error<1e-7
        for radius in experiment.LOCAL_RADII:
            fits=[]
            for settings,seed in [(row['start_settings'],experiment.SOURCE_SEED),
                                  (row['end_settings'],experiment.TARGET_SEED)]:
                key=(tuple(settings),radius,seed)
                if key not in cache:
                    cache[key]=experiment.local_fit(np.array(settings),radius,seed)
                fits.append(cache[key])
            source,target=fits
            index=int(np.flatnonzero(source['labels']==experiment.KNOBS.index(row['expected_knob']))[0])
            match=experiment.match_rows(source['basis'],target['basis'],source['labels'],target['labels'])[index]
            rows.append(dict(name=row['name'],moved_knob=row['moved_knob'],sign=row['sign'],
                             expected_knob=row['expected_knob'],local_radius=radius,
                             true_field_correct=row['strict_correct'],svd_match_correct=match['strict_correct'],
                             svd_chosen_knob=match['chosen_knob'],svd_margin=match['margin']))
    report=dict(rare_oracle_comparisons=len(rare),finite_difference_step=1e-6,
                maximum_derivative_error=maximum_difference_error,svd_followup=rows)
    (experiment.OUT/'rare_true_field_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'Finite differences confirm {len(rare)} earliest true-field failures; '
          f'maximum derivative error {maximum_difference_error:.3g}.')


def main():
    experiment.TRAIN=1024
    starts=dict(experiment.starts());rows=[]
    for source_seed,target_seed in [(20261010,20261021),(20261022,20261023),(20261024,20261025)]:
        for name,knob,mode,distance in [('root','width','true_knob',3.),
                                         ('spread_07','cy','svd_vector',1.),
                                         ('spread_07','cy','svd_vector',.1)]:
            p=starts[name];radius=1e-5
            source=experiment.local_fit(p,radius,source_seed)
            index=int(np.flatnonzero(source['labels']==experiment.KNOBS.index(knob))[0])
            if mode=='true_knob':
                velocity=np.eye(5)[experiment.KNOBS.index(knob)]
            else:
                velocity=np.linalg.lstsq(jet(p)[3],source['basis'][index],rcond=None)[0]
                # Follow the same negative-cy physical orientation across
                # otherwise arbitrary SVD signs.
                if velocity[1]>0:velocity*=-1
            end=experiment.endpoints(p,velocity,[distance])[distance]
            target=experiment.local_fit(end,radius,target_seed)
            match=experiment.match_rows(source['basis'],target['basis'],source['labels'],target['labels'])[index]
            assert match['strict_correct']==(distance==.1)
            assert match['chosen_knob']==('height' if name=='root' else 'cy' if distance==.1 else 'cx')
            rows.append(dict(name=name,mode=mode,distance=distance,neighbors=experiment.TRAIN,
                             source_seed=source_seed,target_seed=target_seed,
                             start_settings=p.tolist(),end_settings=end.tolist(),**match))
    out=experiment.OUT/'larger_sample_verification.json'
    out.write_text(json.dumps(rows,indent=2)+'\n')
    print(json.dumps([dict(name=r['name'],distance=r['distance'],correct=r['strict_correct'],
                           chosen=r['chosen_knob'],margin=r['margin']) for r in rows],indent=2))
    verify_rare_true_field_failures()


if __name__=='__main__':main()
