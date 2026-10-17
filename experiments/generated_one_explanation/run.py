"""Oracle-isolated experiment driver. Run from the repository root.

The renderer and its private settings exist only in this evaluation driver.
Discovery receives separately serialized observations and anonymous links.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from pathlib import Path
import argparse
import hashlib
import itertools
import json
import sys
import time
import scipy
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import qmc
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from grey_ones import render, RANGES, KNOBS
from discover import (fit, save_model, encode, decode, encode_geometry,
                      decode_geometry, features, metrics, geometry_from_coordinates)
OUT = ROOT / 'results/generated_one_explanation'
SPAN = np.diff(RANGES, axis=1).ravel()
LO = RANGES[:, 0]
ROOT_ONE = np.array([14., 14., 19.75, 3.2, 0.])
# Hidden action correspondence. Discovery sees integers, never these names.
IDS = np.array([3, 0, 4, 1, 2])
LIMITS = { 'float64': (5e-11, 5e-10), 'float32': (2e-6, 1e-5) }


def sample(power, seed):
    return LO + qmc.Sobol(5, scramble=True, seed=seed).random_base2(power)*SPAN


def truth_coordinates(p):
    # Evaluator's allowed independent scalar conversion, not discovery input.
    v = np.array(p, dtype=np.float64, copy=True)
    v[:, 4] = np.tan(np.deg2rad(v[:, 4]))
    return v


def assert_metric(record, storage):
    a, b = LIMITS[storage]
    if record['max_pixel'] > a or record['max_image_l2'] > b:
        raise AssertionError(f'{storage} reconstruction threshold failed: {record}')


def make_training():
    p = sample(7, 701)
    source = np.repeat(sample(10, 702), 5, axis=0)
    target = source.copy()
    other = np.repeat(sample(10, 703), 5, axis=0)
    physical_axis = np.tile(np.arange(5), len(source)//5)
    target[np.arange(len(target)), physical_axis] = other[np.arange(len(target)), physical_axis]
    np.savez_compressed(OUT/'train_public.npz', images=render(p),
                        before=render(source), after=render(target), ids=IDS[physical_axis])


def test_points():
    points, tags = [], []
    def add(p, label):
        p = np.atleast_2d(p)
        valid = np.all((p >= LO-1e-14) & (p <= LO+SPAN+1e-14), axis=1)
        p = p[valid]
        points.extend(p); tags.extend([label]*len(p))
    add(sample(12, 801), 'held_out_sobol')
    add(LO+np.array(list(itertools.product(np.linspace(0,1,4), repeat=5)))*SPAN,
        'lattice_and_corners')
    p = sample(8, 802)
    for i in range(5):
        for side in (1e-10, 1-1e-10):
            z = p.copy(); z[:, i] = LO[i]+SPAN[i]*side
            add(z, 'near_faces')
    p = sample(8, 803)
    for sign in (-1, 1):
        cap = p[:, 1]+sign*p[:, 2]/2
        for eps in (-1e-9, 0., 1e-9):
            z = p.copy()
            z[:, 1] = np.round(cap*16)/16-sign*z[:, 2]/2+eps
            add(z, 'cap_subrow_transition')
    for side in (-1, 1):
        for y in (5+1/32, 18+31/32):
            edge = p[:, 0]-np.tan(np.deg2rad(p[:, 4]))*(y-p[:, 1])+side*p[:, 3]/2
            for eps in (-1e-9, 0., 1e-9):
                z = p.copy(); z[:, 0] += np.round(edge)-edge+eps
                add(z, 'side_column_transition')
    for row in (5, 23):
        center = p[:, 0]-np.tan(np.deg2rad(p[:, 4]))*(row+.5-p[:, 1])
        z = p.copy(); z[:, 0] += np.floor(center)+.5-center
        add(z, 'encoder_boundary_tie')
    add(ROOT_ONE, 'canonical_root')
    return np.array(points), np.array(tags)


def calibrate(model):
    p = sample(8, 901)
    z = encode(render(p, dtype=np.float64), model)
    v = truth_coordinates(p)
    # Select only a permutation and five independent affine calibrations.
    corr = np.corrcoef(z.T, v.T)[:5, 5:]
    zi, vi = linear_sum_assignment(-np.abs(corr))
    calibration = []
    for j, k in zip(zi, vi):
        a, b = np.linalg.lstsq(np.c_[v[:, k], np.ones(len(p))], z[:, j], rcond=None)[0]
        calibration.append(dict(z_index=int(j), physical_index=int(k),
                                physical_name=KNOBS[k], scale=float(a), offset=float(b)))
    return calibration


def scales():
    result = []
    generic = np.array([14.43, 14.62, 19.73, 3.17, 17.3])
    kink = generic.copy(); kink[1] = 4.75+kink[2]/2
    interior_upright = np.array([14.5,14.5,19.75,3.2,0.])
    rng = np.random.default_rng(1001)
    directions = rng.normal(size=(96,5)); directions /= np.linalg.norm(directions,axis=1)[:,None]
    for name, base in [('smooth', generic), ('tilted_cap_kink', kink),
                       ('upright_cap_join', interior_upright)]:
        for radius in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9):
            points = base+directions*SPAN*radius
            for dtype in (np.float64, np.float32):
                delta = (render(points,dtype=dtype).astype(float)-render(base,dtype=dtype).astype(float)).reshape(96,-1)
                sv = np.linalg.svd(delta, compute_uv=False)
                result.append(dict(start=name, normalized_radius=radius, storage=np.dtype(dtype).name,
                    first_singular=float(sv[0]), sixth_over_first=float(sv[5]/sv[0]),
                    residual_after_five=float(np.linalg.norm(sv[5:])/np.linalg.norm(sv)),
                    changed_images=int(np.sum(np.any(delta != 0,axis=1)))))
    return result


def continuous_area(q):
    """Exact area of the ideal strip, a deliberately different sensor model."""
    q=np.atleast_2d(q); result=[]
    column=np.arange(28)[None,:]
    row=np.arange(28)[:,None]
    for t,b,a,w,slope in q:
        low=np.maximum(row,t);high=np.minimum(row+1,b)
        length=np.maximum(0,high-low)
        def integral_ramp(offset):
            f0=offset-slope*low;f1=offset-slope*high
            if abs(slope)<1e-14:
                return np.maximum(0,f0)*length
            both=(f0>=0)&(f1>=0)
            cross=(f0>0)!=(f1>0)
            return np.where(both,(f0+f1)*length/2,
                np.where(cross,np.maximum(f0,f1)**2/(2*abs(slope)),0))* (length>0)
        area=(integral_ramp(a+w/2-column)-integral_ramp(a+w/2-column-1)
              -integral_ramp(a-w/2-column)+integral_ramp(a-w/2-column-1))
        result.append(area)
    return np.array(result)


def one_sided_diagnostics():
    bases=[('smooth',np.array([14.43,14.62,19.73,3.17,17.3])),
           ('tilted_cap_kink',np.array([14.43,14.615,19.73,3.17,17.3])),
           ('upright_cap_join',np.array([14.5,14.5,19.75,3.2,0.]))]
    result=[]
    for name,p in bases:
        h=1e-7
        steps=np.eye(5)*SPAN*h
        base=render(p,dtype=np.float64).reshape(784)
        plus=(render(p+steps,dtype=np.float64).reshape(5,784)-base)/h
        minus=(base-render(p-steps,dtype=np.float64).reshape(5,784))/h
        union=np.linalg.svd(np.r_[plus,minus],compute_uv=False)
        result.append(dict(start=name,normalized_step=h,
            relative_one_sided_disagreement=float(np.linalg.norm(plus-minus)/np.linalg.norm(plus)),
            union_singular_ratios=(union/union[0]).tolist()))
    return result


def collision():
    p = ROOT_ONE.copy(); other = p.copy(); other[1] += 1e-9
    before = render([p, other])
    step = np.array([0., .125, 0., 0., 0.])
    after = render([p+step, other+step])
    return dict(settings_first=p.tolist(), settings_second=other.tolist(),
                bitwise_same_observation=bool(np.array_equal(before[0], before[1])),
                physical_increment=step.tolist(),
                bitwise_same_future=bool(np.array_equal(after[0],after[1])),
                future_pixel_difference=float(np.max(np.abs(after[0]-after[1]))),
                future_image_l2=float(np.linalg.norm(after[0].astype(float)-after[1].astype(float))))


def mixed_counterexample(p):
    # Box-to-itself smooth mixing; the equation, not testing, gives invertibility.
    u = (p-LO)/SPAN
    v = u.copy(); k = 4*u[:,0]**2*(1-u[:,0])**2
    v[:,3] = u[:,3]+k*u[:,3]*(1-u[:,3])
    undo = v.copy()
    undo[:,3] = 2*v[:,3]/(1+k+np.sqrt((1+k)**2-4*k*v[:,3]))
    back = LO+undo*SPAN
    return dict(max_parameter_roundtrip=float(np.max(np.abs(back-p))),
                image_roundtrip=metrics(render(p,dtype=np.float64), render(back,dtype=np.float64)),
                largest_width_coordinate_mixing=float(np.max(np.abs(v[:,3]-u[:,3]))))


def interventions(model, dtype=np.float32):
    storage=np.dtype(dtype).name
    source, donor = sample(9, 1101), sample(9, 1102)
    x, d = render(source,dtype=dtype), render(donor,dtype=dtype)
    z, zd = encode(x, model), encode(d, model)
    records, all_targets, all_predictions = {}, [], []
    for axis in range(5):
        index = model['action_ids'].index(int(IDS[axis]))
        target = source.copy(); target[:,axis] = donor[:,axis]
        requested = z.copy(); requested[:,index] = zd[:,index]
        predicted = decode(requested, model)
        m = metrics(render(target,dtype=dtype), predicted); assert_metric(m, storage)
        records[KNOBS[axis]] = m
        all_targets.append(render(target,dtype=dtype)); all_predictions.append(predicted)
    order_records = []
    for order in ([0,1,2,3,4], [4,3,2,1,0], [2,4,1,0,3]):
        state = x.copy()
        for axis in order:
            current = encode(state, model)
            index = model['action_ids'].index(int(IDS[axis]))
            current[:,index] = zd[:,index]
            state = decode(current, model)
        m = metrics(render(donor,dtype=dtype), state); assert_metric(m, storage)
        order_records.append(dict(order=list(order), **m))
    # Joint independent control interpolation, using only scalar-allowed units.
    y0, y1 = truth_coordinates(source), truth_coordinates(donor)
    interpolation = []
    for fraction in (.125, .5, .875):
        zz = (1-fraction)*z+fraction*zd
        v = (1-fraction)*y0+fraction*y1
        v[:,4] = np.rad2deg(np.arctan(v[:,4]))
        m = metrics(render(v,dtype=dtype), decode(zz,model)); assert_metric(m,storage)
        interpolation.append(dict(fraction=fraction, **m))
    # File contains no target settings; independent checker recombines coordinates.
    np.savez_compressed(OUT/f'interventions_{storage}_public.npz', source=x, donor=d,
        target=np.array(all_targets), action_indices=np.array([model['action_ids'].index(int(k)) for k in IDS]),
        donor_target=render(donor,dtype=dtype))
    return dict(single_factor_copies=records, five_factor_orders=order_records,
                unseen_joint_interpolations=interpolation)


def pair_leakage(model):
    p, d = sample(8,1201), sample(8,1202)
    result=[]
    for axis in range(5):
        target=p.copy(); target[:,axis]=d[:,axis]
        q0,q1=encode_geometry(render(p)),encode_geometry(render(target))
        z0,z1=encode(render(p),model),encode(render(target),model)
        qscale=np.std(q0,axis=0);zscale=np.std(z0,axis=0)
        dq=np.max(np.abs(q1-q0)/qscale,axis=0)
        dz=np.max(np.abs(z1-z0)/zscale,axis=0)
        index=model['action_ids'].index(int(IDS[axis]))
        result.append(dict(physical_factor=KNOBS[axis], expected_z_index=index,
            mixed_geometry_chart_response=dq.tolist(), discovered_chart_response=dz.tolist(),
            discovered_off_axis_max=float(np.max(np.delete(dz,index)))))
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args()
    if args.out != OUT:
        raise ValueError('fixed relative output directory makes checker reproducible')
    OUT.mkdir(parents=True,exist_ok=True)
    started=time.monotonic()
    make_training()
    with np.load(OUT/'train_public.npz') as public:
        model=fit(public['images'], public['before'], public['after'], public['ids'])
    save_model(model,OUT/'model.json')
    model_hash=hashlib.sha256((OUT/'model.json').read_bytes()).hexdigest()
    print('Frozen model:',model['subrows'],'subrows; invariant degrees',model['degrees'],flush=True)
    p,tags=test_points()
    records={}; calibration=calibrate(model)
    for dtype in (np.float64,np.float32):
        storage=np.dtype(dtype).name
        observations=render(p,dtype=dtype)
        z=encode(observations,model);prediction=decode(z,model)
        m=metrics(observations,prediction);assert_metric(m,storage)
        by_group={str(tag):dict(n=int(np.sum(tags==tag)),**metrics(observations[tags==tag],prediction[tags==tag])) for tag in np.unique(tags)}
        alignment=[];v=truth_coordinates(p)
        for c in calibration:
            recovered=(z[:,c['z_index']]-c['offset'])/c['scale']
            alignment.append(dict(physical_name=c['physical_name'],
                max_abs_error=float(np.max(np.abs(recovered-v[:,c['physical_index']])))))
        # Both directions: decode a fresh permitted control point, encode, decode.
        fresh=sample(10,1301)
        tf=truth_coordinates(fresh)
        independent_z=np.empty((len(fresh),5))
        for c in calibration:
            independent_z[:,c['z_index']]=c['scale']*tf[:,c['physical_index']]+c['offset']
        reverse=metrics(render(fresh,dtype=dtype),decode(independent_z,model,dtype=dtype))
        assert_metric(reverse,storage)
        records[storage]=dict(n=len(p),**m,groups=by_group,alignment=alignment,
            worst_l2_settings=p[m['worst_l2_index']].tolist(),
            worst_pixel_settings=p[m['worst_pixel_index']].tolist(),
            decoded_domain_check=reverse)
        np.savez_compressed(OUT/f'test_{storage}_public.npz',images=observations,tags=tags)
        print(storage,records[storage]['max_pixel'],records[storage]['max_image_l2'],flush=True)
    np.savez_compressed(OUT/'test_oracle.npz',settings=p)
    # Controlled failure of the naive centroid and of a wrong raster count.
    x=render(p)
    rows=x.astype(float).sum(2)
    naive=(rows*(np.arange(28)+.5)).sum(1)/rows.sum(1)
    q=encode_geometry(x)
    negative=dict(ideal_continuous_area=metrics(x,continuous_area(q)),
                  storage_rounding=metrics(render(p,dtype=np.float64),x),
                  naive_vertical_centroid_max_error=float(np.max(np.abs(naive-p[:,1]))),
                  wrong_raster_8=metrics(x,decode_geometry(q,8)),
                  wrong_raster_64=metrics(x,decode_geometry(q,64)))
    # A deliberately invalid mixture can share row totals and straight centroids.
    pp=np.array([[14.5,14.5,19.75,3.,-10.],[14.5,14.5,19.75,3.,10.]])
    invalid=render(pp).mean(0,keepdims=True)
    negative['averaged_strokes_rejected']=metrics(invalid,decode(encode(invalid,model),model))
    with np.load(OUT/'train_public.npz') as public:
        shuffled=public['ids'].copy();np.random.default_rng(1701).shuffle(shuffled)
        try:
            fit(public['images'],public['before'],public['after'],shuffled)
        except ValueError as failure:
            negative['shuffled_action_ids_rejected']=str(failure)
        else:
            raise AssertionError('corrupted links unexpectedly passed')
    p0=ROOT_ONE.copy();p1=ROOT_ONE.copy();p1[1]+=1e-9
    step=np.array([0.,.125,0.,0.,0.])
    np.savez_compressed(OUT/'counterexamples.npz',collision_before=render([p0,p1]),
        collision_after=render([p0+step,p1+step]),invalid_average=invalid)
    result=dict(provenance=dict(python=sys.version.split()[0],numpy=np.__version__,scipy=scipy.__version__,
        seed_contract='fixed integer seeds in run.py',
        source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in [ROOT/'grey_ones.py',Path(__file__),Path(__file__).with_name('discover.py'),
                      Path(__file__).with_name('PROTOCOL.md')]}),
        protocol='experiments/generated_one_explanation/PROTOCOL.md',
        model_sha256=model_hash, controls=calibration, reconstructions=records,
        interventions={np.dtype(dt).name:interventions(model,dt) for dt in (np.float64,np.float32)}, linkage_test=pair_leakage(model),
        box_preserving_mixing=mixed_counterexample(sample(9,1401)),
        scale_diagnostics=scales(), one_sided_diagnostics=one_sided_diagnostics(), stored_collision=collision(), negative_controls=negative,
        elapsed_seconds=time.monotonic()-started)
    if not result['stored_collision']['bitwise_same_observation']:
        raise AssertionError('collision witness no longer reproduces')
    if result['stored_collision']['bitwise_same_future']:
        raise AssertionError('collision future separation no longer reproduces')
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('PASS:',len(p),'held-out states; full results saved.',flush=True)

if __name__=='__main__':
    main()
