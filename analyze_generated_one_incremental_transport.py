"""Incremental transport of five physical knob vectors; known-renderer controls.

All artifact paths are repository-relative. Run with the repository Python.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/generated-one-mpl')
from pathlib import Path
import csv
import json
import numpy as np
from scipy.ndimage import map_coordinates
from grey_ones import render, SUB, KNOBS, RANGES
from invert_grey import invert
from analyze_generated_one_manifold_transport import jet, chart, normalized, transport_path

OUT = Path(__file__).resolve().parent / 'figures/generated_one_incremental_transport'
ROOT = np.array([14., 14., 19.75, 3.2, 0.])
COUNTS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
SPANS = RANGES[:, 1] - RANGES[:, 0]


def image(p, dtype=np.float64):
    return render(p, dtype=dtype).reshape(-1, 784)


def positive_jacobian(p):
    """Five positive one-sided derivatives, including cap/pixel events.

    At smooth states these are the ordinary Jacobian. At kink states these
    columns are five directional probes, not a differentiable chart.
    """
    x, y, h, w, angle = p
    lo = np.arange(28 * SUB) / SUB
    hi = lo + 1 / SUB
    mid = (lo + hi) / 2
    cols = np.arange(28)
    s = np.tan(np.deg2rad(angle))
    prime = np.pi / 180 * (1 + s * s)
    top, bot = y - h / 2, y + h / 2
    vertical = np.clip(np.minimum(hi, bot) - np.maximum(lo, top), 0, 1 / SUB)
    centre = x - s * (mid - y)
    left, right = centre - w / 2, centre + w / 2
    ov = np.clip(np.minimum(right[:, None], cols + 1) - np.maximum(left[:, None], cols), 0, 1)

    def vactive(v, direction):
        return (((lo < v) & (v < hi)) | ((lo == v) & (direction > 0)) |
                ((hi == v) & (direction < 0))).astype(float)

    def xactive(v, direction):
        v, direction = v[:, None], direction[:, None]
        return (((cols < v) & (v < cols + 1)) | ((cols == v) & (direction > 0)) |
                ((cols + 1 == v) & (direction < 0))).astype(float)

    answer = []
    for k in range(5):
        d = np.eye(5)[k]
        dt, db = d[1] - d[2] / 2, d[1] + d[2] / 2
        dv = db * vactive(bot, db) - dt * vactive(top, dt)
        dc = d[0] + s * d[1] - prime * (mid - y) * d[4]
        dl, dr = dc - d[3] / 2, dc + d[3] / 2
        dx = dr[:, None] * xactive(right, dr) - dl[:, None] * xactive(left, dl)
        derivative = dv[:, None] * ov + vertical[:, None] * dx
        answer.append(derivative.reshape(28, SUB, 28).sum(1).ravel())
    return np.column_stack(answer)


def errors(carried, truth):
    relative = np.linalg.norm(carried - truth, axis=0) / np.linalg.norm(truth, axis=0)
    cosine = np.sum(carried * truth, axis=0) / np.maximum(
        np.linalg.norm(carried, axis=0) * np.linalg.norm(truth, axis=0), 1e-300)
    return dict(relative_errors=relative.tolist(), cosine=cosine.tolist(),
                max_relative_error=float(relative.max()), mean_relative_error=float(relative.mean()))


def paths(start=ROOT):
    changes = [np.eye(5)[i] * a for i, a in enumerate([.7, .7, .5, .6, 35.])]
    return {**{k: (start.copy(), start + d) for k, d in zip(KNOBS, changes)},
            'joint': (start.copy(), start + np.array([.65, .55, .45, .6, 25.])),
            'lean10': (start.copy(), start + np.array([0., 0., 0., 0., 10.]))}


def coordinate_derivative(p):
    _, y, _, _, angle = p
    s = np.tan(np.deg2rad(angle)); prime = np.pi / 180 * (1 + s * s)
    return np.array([[0, 1, -.5, 0, 0], [0, 1, .5, 0, 0],
                     [1, s, 0, 0, y * prime], [0, 0, 0, 1, 0], [0, 0, 0, 0, prime]])


def frame_and_taylor():
    exact_rows, taylor_rows, history = [], [], []
    cache = {}
    for name, (start, end) in paths().items():
        truth = positive_jacobian(end)
        for n in COUNTS:
            pp = start + np.linspace(0, 1, n + 1)[:, None] * (end - start)
            transported = positive_jacobian(start)
            initial_q = jet(start)[1].copy()
            qcarried = initial_q.copy()
            event_count = 0
            for step, (old, new) in enumerate(zip(pp[:-1], pp[1:]), 1):
                # This control reads BOTH exact frames: it is not discovery.
                a, b = positive_jacobian(old), positive_jacobian(new)
                transported = b @ (np.linalg.pinv(a) @ transported)
                oldjet, newjet = jet(old), jet(new)
                dq = chart(new) - chart(old)
                qcarried += np.einsum('aij,j->ai', oldjet[2], dq)
                if not np.array_equal(oldjet[4], newjet[4]):
                    event_count += 1
                if name == 'lean10' and n == 512:
                    history.append(dict(step=step, lean=float(new[4]),
                                        exact_error=errors(transported, b)['max_relative_error'],
                                        taylor_error=errors(qcarried @ coordinate_derivative(new), b)['max_relative_error']))
            exact_rows.append(dict(path=name, steps=n, **errors(transported, truth)))
            taylor_rows.append(dict(path=name, steps=n, formula_signature_changes=event_count,
                                    **errors(qcarried @ coordinate_derivative(end), truth)))
            if name == 'lean10' and n in [1, 16, 512]:
                cache[f'coordinate_{n}'] = transported
                cache[f'taylor_{n}'] = qcarried @ coordinate_derivative(end)
    # Additional regular starts, with endpoints in the allowed box.
    rng = np.random.default_rng(20261016)
    samples = RANGES[:, 0] + rng.random((64, 5)) * SPANS
    heldout = []
    max_derivative_error = 0.; max_inverse_error = 0.
    for p in samples:
        j = positive_jacobian(p)
        eps = 1e-7
        fd = (image(p[None, :] + np.eye(5) * eps) - image(p)) .T / eps
        max_derivative_error = max(max_derivative_error, errors(fd, j)['max_relative_error'])
        max_inverse_error = max(max_inverse_error, float(np.max(np.abs(invert(image(p)[0]) - p))))
        carried = j @ (np.linalg.pinv(positive_jacobian(ROOT)) @ positive_jacobian(ROOT))
        heldout.append(errors(carried, j)['max_relative_error'])
    return dict(coordinate=exact_rows, local_taylor=taylor_rows, history=history,
                heldout_states=64, heldout_coordinate_max_error=max(heldout),
                analytic_derivative_max_relative_validation_error=max_derivative_error,
                inverse_max_absolute_parameter_error=max_inverse_error), cache


def pixel_shear(values, slope_change, pivot):
    shape = values.shape
    v = values.reshape(28, 28, -1)
    rows, cols = np.indices((28, 28), dtype=float)
    coordinates = [rows, cols + slope_change * (rows + .5 - pivot)]
    return np.stack([map_coordinates(v[:, :, k], coordinates, order=1, mode='constant',
                                    cval=0., prefilter=False) for k in range(v.shape[2])], axis=-1).reshape(shape)


def lean_warps():
    output, cache, histories = [], {}, []
    initial = positive_jacobian(ROOT)
    for target in [1., 5., 10., 35.]:
        end = ROOT.copy(); end[4] = target
        truth = positive_jacobian(end)
        slope = np.tan(np.deg2rad(target))
        # Exact frozen geometric shear on a neighborhood has a different
        # action on cy and angle from the pure physical lean knob flow.
        geometric = truth.copy()
        geometric[:, 1] -= slope * truth[:, 0]
        geometric[:, 4] *= 1 / (1 + slope * slope)
        geometric_errors = errors(geometric, truth)
        for n in COUNTS:
            carried = initial.copy(); base = image(ROOT).T
            for step in range(1, n + 1):
                oldangle, newangle = target * (step - 1) / n, target * step / n
                ds = np.tan(np.deg2rad(newangle)) - np.tan(np.deg2rad(oldangle))
                carried = pixel_shear(carried, ds, ROOT[1])
                base = pixel_shear(base, ds, ROOT[1])
                if target == 10. and n == 512:
                    now = ROOT.copy();now[4] = newangle
                    histories.append(dict(step=step, lean=newangle,
                                          width_error=errors(carried, positive_jacobian(now))['relative_errors'][3]))
            corrected = carried.copy()
            corrected[:, 1] += slope * carried[:, 0]
            corrected[:, 4] *= 1 + slope * slope
            output.append(dict(lean=target, steps=n, step_degrees=target / n,
                               state_relative_error=float(np.linalg.norm(base[:, 0] - image(end)[0]) /
                                                          np.linalg.norm(image(end)[0])),
                               **errors(carried, truth),
                               corrected_relative_errors=errors(corrected, truth)['relative_errors'],
                               exact_geometric_relative_errors=geometric_errors['relative_errors']))
            if target == 10. and n in [1, 16, 512]:
                cache[f'pixel_{n}'] = carried
        cache[f'geometric_{target:g}'] = geometric
    slope_group = []
    for n in COUNTS:
        slope_group.append(dict(steps=n, requested_degrees=35.,
                                naive_equal_shear_final_degrees=float(np.rad2deg(np.arctan(n * np.tan(np.deg2rad(35. / n)))))))
    return dict(pixel_warps=output, accumulation_history=histories,
                fixed_small_angle_shear=slope_group), cache


def companion_test():
    results = []
    # Six VALID images travel through the same image -> inverse -> knob
    # increment -> render map. No off-manifold image+epsilon*vector calls.
    for dtype in [np.float64, np.float32]:
        for probe_fraction in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
            eps = SPANS * probe_fraction
            initial = np.vstack([ROOT, ROOT + np.eye(5) * eps])
            for target in [10., 35.]:
                end = ROOT.copy(); end[4] = target
                truth = positive_jacobian(end)
                for n in [1, 4, 16, 64, 256]:
                    stream = render(initial, dtype=dtype)
                    for _ in range(n):
                        recovered = np.array([invert(a) for a in stream])
                        recovered[:, 4] += target / n
                        stream = render(recovered, dtype=dtype)
                    v = ((stream[1:].astype(float) - stream[0].astype(float)) / eps[:, None, None]).reshape(5, 784).T
                    results.append(dict(dtype=np.dtype(dtype).name, probe_fraction=probe_fraction,
                                        lean=target, steps=n, state_relative_error=float(
                                            np.linalg.norm(stream[0].ravel() - image(end)[0]) / np.linalg.norm(image(end)[0])),
                                        **errors(v, truth)))
    all_knobs=[]
    for dtype, fraction in [(np.float64,1e-5),(np.float32,1e-4)]:
        eps=SPANS*fraction
        for name,(start,end) in paths().items():
            initial=np.vstack([start,start+np.eye(5)*eps])
            for n in [1,16,256]:
                stream=render(initial,dtype=dtype)
                for _ in range(n):
                    recovered=np.array([invert(a) for a in stream])
                    recovered+=(end-start)/n
                    stream=render(recovered,dtype=dtype)
                v=((stream[1:].astype(float)-stream[0].astype(float))/eps[:,None,None]).reshape(5,784).T
                all_knobs.append(dict(path=name,dtype=np.dtype(dtype).name,probe_fraction=fraction,steps=n,
                                      **errors(v,positive_jacobian(end))))
    return results,all_knobs


def shear_round_trips():
    rows=[]
    for n in COUNTS:
        v=positive_jacobian(ROOT);f=image(ROOT).T
        for old,new in list(zip(np.linspace(0,10,n+1)[:-1],np.linspace(0,10,n+1)[1:]))+list(zip(np.linspace(10,0,n+1)[:-1],np.linspace(10,0,n+1)[1:])):
            ds=np.tan(np.deg2rad(new))-np.tan(np.deg2rad(old))
            v=pixel_shear(v,ds,ROOT[1]);f=pixel_shear(f,ds,ROOT[1])
        rows.append(dict(steps_per_leg=n,**errors(v,positive_jacobian(ROOT)),
                         state_relative_error=float(np.linalg.norm(f[:,0]-image(ROOT)[0])/np.linalg.norm(image(ROOT)[0]))))
    return rows


def smooth_euler():
    # Same physical angular increments, first-order update of all five raw
    # fields. Current local derivatives are supplied, not the future frame.
    start=np.array([14.,14.013,19.73,3.2,0.])
    end=start.copy();end[4]=1.
    sig=jet(start)[4]
    assert all(np.array_equal(jet(p)[4],sig) for p in start+np.linspace(0,1,101)[:,None]*(end-start))
    records=[]
    for n in COUNTS:
        estimate=positive_jacobian(start)
        pp=start+np.linspace(0,1,n+1)[:,None]*(end-start)
        for p,new in zip(pp[:-1],pp[1:]):
            d=new-p
            jq,hq=jet(p)[1:3]
            c=coordinate_derivative(p)
            slope=np.tan(np.deg2rad(p[4]))
            prime=np.pi/180*(1+slope*slope)
            second=2*(np.pi/180)**2*slope*(1+slope*slope)
            dc=np.zeros((5,5))
            dc[2,1]=prime*d[4]
            dc[2,4]=d[1]*prime+p[1]*second*d[4]
            dc[4,4]=second*d[4]
            estimate+=np.einsum('aij,j->ai',hq,c@d)@c+jq@dc
        records.append(dict(steps=n,step_degrees=1/n,**errors(estimate,positive_jacobian(end))))
    assert records[-1]['max_relative_error']<records[0]['max_relative_error']/100
    return dict(start=start.tolist(),end=end.tolist(),single_formula_cell=True,results=records)


def smooth_and_kinks():
    smooth = np.array([14., 14.013, 19.73, 3.2, 0.])
    end = smooth.copy(); end[3] += .6
    signature = jet(smooth)[4]
    single_cell = all(np.array_equal(jet(p)[4], signature) for p in smooth + np.linspace(0, 1, 101)[:, None] * (end - smooth))
    assert single_cell
    startj, endj = positive_jacobian(smooth), positive_jacobian(end)
    metric = []; taylor = []; last = None
    for n in COUNTS:
        pp = smooth + np.linspace(0, 1, n + 1)[:, None] * (end - smooth)
        transported = transport_path(pp, startj)
        metric.append(dict(steps=n, **errors(transported, endj),
                           unit_direction_errors=np.linalg.norm(normalized(transported)-normalized(endj), axis=0).tolist(),
                           max_gram_preservation_error=float(np.max(np.abs(transported.T @ transported-startj.T @ startj))),
                           difference_from_previous=None if last is None else float(np.linalg.norm(transported-last))))
        last = transported
        qcarried = jet(smooth)[1].copy()
        for a, b in zip(pp[:-1], pp[1:]):
            qcarried += np.einsum('aij,j->ai', jet(a)[2], chart(b)-chart(a))
        taylor.append(dict(steps=n, **errors(qcarried @ coordinate_derivative(end), endj)))
    kinks = []
    for angle in [0., 1., 10., 35.]:
        p = ROOT.copy(); p[4] = angle
        for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
            f = image(p)[0]
            right = (image(p + np.eye(5)*eps)-f).T/eps
            left = (f-image(p - np.eye(5)*eps)).T/eps
            gap = np.linalg.norm(right-left, axis=0)/np.linalg.norm(right, axis=0)
            kinks.append(dict(lean=angle, probe=eps, side_disagreement=gap.tolist()))
    # A general counterexample on a flat 2D manifold. Both fields have unit
    # speed, and remain independent for |x|<pi/2, but do not commute:
    # X=d/dx; Y=(sin x,cos x).
    arbitrary_fields = []
    for n in COUNTS:
        initial_y = np.array([0., 1.])
        final_y = np.array([np.sin(.5), np.cos(.5)])
        # x-translation differential is identity at every increment.
        carried = initial_y.copy()
        for _ in range(n):
            carried = np.eye(2) @ carried
        arbitrary_fields.append(dict(steps=n, relative_error=float(np.linalg.norm(carried-final_y)),
                                     cosine=float(carried@final_y)))
    return dict(smooth_start=smooth.tolist(), smooth_end=end.tolist(), single_formula_cell=single_cell,
                metric=metric, local_taylor=taylor,
                true_unit_gram_change=float(np.max(np.abs(normalized(startj).T@normalized(startj)-normalized(endj).T@normalized(endj)))),
                kink_tests=kinks, arbitrary_noncommuting_fields=arbitrary_fields)


def write_csv(name, records):
    expanded = []
    for r in records:
        rr = {}
        for key, value in r.items():
            if isinstance(value, list) and len(value) == 5:
                rr.update({f'{key}_{k}': v for k, v in zip(KNOBS, value)})
            else:
                rr[key] = value
        expanded.append(rr)
    with (OUT/name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(expanded[0]))
        writer.writeheader(); writer.writerows(expanded)


def figures(data, cache):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#2f6ebc', '#be5b14', '#25835b', '#9a429f', '#4b5967']
    fig = plt.figure(figsize=(14, 9), dpi=150)
    fig.text(.05,.965,'Do smaller lean steps carry the five knob vectors correctly?',fontsize=19,weight='bold',va='top')
    fig.text(.05,.925,'Known generated-1 renderer; root (14, 14, 19.75, 3.2, 0°) → lean 10°. All 784 pixels are measured.',fontsize=11,va='top')
    for loc, p, title in [([.05,.68,.13,.19], ROOT,'Start: root 1'),([.24,.68,.13,.19], ROOT+np.array([0,0,0,0,10]),'Target: lean 10°')]:
        ax=fig.add_axes(loc);ax.imshow(image(p)[0].reshape(28,28),cmap='gray_r',vmin=0,vmax=1);ax.set_axis_off();ax.set_title(title,fontsize=11)
    fig.text(.43,.855,'One pixel: 0 = no ink; 1 = full ink.\nA knob vector: pixel change per pixel of position/size,\nor per degree of lean. Vectors are not unit-normalized.\nError = ||carried − true|| / ||true||. Lower is better.\nStep count increases rightward; total lean stays 10°.',fontsize=11,va='top',linespacing=1.6)
    ax=fig.add_axes([.07,.14,.40,.43])
    warp=[r for r in data['lean']['pixel_warps'] if r['lean']==10.]
    for k,(name,color) in enumerate(zip(KNOBS,colors)):
        ax.plot([r['steps'] for r in warp],[100*r['relative_errors'][k] for r in warp],'-o',label={'cx':'Horizontal position','cy':'Vertical position'}.get(name,name.title()),color=color,markersize=3)
    ax.set_xscale('log',base=2);ax.set_xlabel('Number of repeated pixel shears');ax.set_ylabel('Vector error (%)');ax.set_title('Repeated pixel interpolation does not converge to truth',fontsize=12)
    ax.set_ylim(bottom=0);ax.grid(alpha=.2);ax.legend(ncol=2,fontsize=9)
    ax2=fig.add_axes([.57,.14,.38,.43])
    history=data['lean']['accumulation_history']
    ax2.plot([r['lean'] for r in history],[100*r['width_error'] for r in history],color=colors[3],label='512 small pixel shears')
    ax2.set_xlabel('Accumulated lean (degrees)');ax2.set_ylabel('Width-vector error (%)');ax2.set_title('Error builds while taking 0.0195° steps',fontsize=12);ax2.grid(alpha=.2)
    ax2.legend(fontsize=9)
    fig.text(.05,.045,'Scope: positive one-sided knob derivatives at root cap events; float64 coverage. Pixel warp uses linear horizontal interpolation.\nThis tests one specified warp, not every possible vector-transport rule. Exact known-coordinate frame control: see report.',fontsize=10,va='bottom')
    fig.savefig(OUT/'increment_size_and_accumulation.png');plt.close(fig)

    end=ROOT.copy();end[4]=10.; truth=positive_jacobian(end)
    names=['Horizontal position','Vertical position','Height','Width','Lean']
    arrays=[positive_jacobian(ROOT),truth,cache['pixel_1'],cache['pixel_512']]
    rownames=['Starting vectors\nat root 1','True vectors\nat lean 10°','Carried by\n1 pixel shear','Carried by\n512 pixel shears']
    norms=np.linalg.norm(truth,axis=0)
    scaled=[a/norms for a in arrays];limit=max(np.max(np.abs(a)) for a in scaled)
    fig=plt.figure(figsize=(14,11),dpi=140)
    fig.text(.04,.975,'What happens to all five pixel-change patterns?',fontsize=20,weight='bold',va='top')
    fig.text(.04,.937,'Read down: root vectors → true target → carried estimates.\nRoot → lean 10°; other knobs fixed. Ink coverage: 0 = white, 1 = black.\nRed gains ink; blue loses ink. Panels below show pixel derivatives.',fontsize=11,va='top',linespacing=1.5)
    for x,p,label in [(.80,ROOT,'Root 1'),(.90,end,'Lean 10°')]:
        ax=fig.add_axes([x,.885,.075,.075]);ax.imshow(image(p)[0].reshape(28,28),cmap='gray_r',vmin=0,vmax=1);ax.set_axis_off();ax.text(.5,-.04,label,transform=ax.transAxes,ha='center',va='top',fontsize=10)
    for r,a in enumerate(scaled):
        for k in range(5):
            ax=fig.add_axes([.18+k*.155,.69-r*.175,.135,.15])
            shown=ax.imshow(a[:,k].reshape(28,28),cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest');ax.set_axis_off()
            if r==0:ax.set_title(names[k],fontsize=11,pad=9)
            if r>=2:
                err=errors(arrays[r],truth)['relative_errors'][k]
                ax.text(.5,-.05,f'Error {100*err:.1f}%',transform=ax.transAxes,ha='center',va='top',fontsize=10)
        fig.text(.025,.765-r*.175,rownames[r],fontsize=11,va='center')
    cax=fig.add_axes([.36,.10,.38,.016]);fig.colorbar(shown,cax=cax,orientation='horizontal').set_label('Signed coefficient / length of the true target vector',fontsize=10)
    fig.text(.04,.018,'Each column uses its true target vector length as a fixed divisor; one common signed color scale, no clipping.\nThe rows are positive one-sided physical derivatives (pixels/degrees), with no unit-vector renormalization after transport.',fontsize=10,va='bottom')
    fig.savefig(OUT/'five_vectors_after_lean.png');plt.close(fig)


def report(data):
    lines=['# Incremental transport of generated-1 knob vectors','',
           'The experiments use the actual `grey_ones.py` renderer, all 784 coverage pixels, and the canonical root `(14, 14, 19.75, 3.2, 0)`.',
           'A vector is the raw pixel derivative per pixel of position/size or per degree of lean. Error means Euclidean vector error divided by the true vector length. No per-step renormalization is applied.', '',
           '## Mathematical question','',
           'For a smooth decoder `F(p)` with independent coordinate derivatives, define the physical knob flow by `T_i^d(F(p)) = F(p + d e_i)`. Its differential, called the pushforward, satisfies `D T_i^d [J_j(p)] = J_j(p + d e_i)`. This follows by differentiating `T_i^d(F(p+t e_j)) = F(p+d e_i+t e_j)` with respect to `t`.',
           'This is exact on regular smooth chart regions, for finite permitted moves as well as infinitesimal ones. It relies on independent commuting knob coordinates, not just having five vectors. An arbitrary moving frame, normalized knob field, or fixed spatial warp need not satisfy it.',
           'Applying `T` directly to a vector is generally wrong; the vector operation is `D T`. For a linear image operator `A`, these coincide because `D A = A`.', '',
           'The known-coordinate frame map is `J(new) pinv(J(old))`. It uses both measured/analytic frames, so its success is a calibration control, not a discovery algorithm. Repeated maps compose exactly in exact arithmetic. At kink states we use explicitly defined positive one-sided columns; this algebraic map is then a convention, not a unique manifold differential.', '',
           '## Repeated pixel shear','',
           'The pixel baseline horizontally interpolates every row at its center, with constant-zero values outside the canvas. Both the base image and the five vectors receive the same linear operator. Slope increments are `tan(new lean)-tan(old lean)`, so the prescribed endpoint geometry is held fixed while step counts vary.',
           'At a frozen pivot, an exact geometric shear preserves cx/height/width coordinate derivatives, but maps `J_cy` to `J_cy(new)-delta_slope*J_cx(new)` and maps `J_lean` to `sec(old lean)^2/sec(new lean)^2 * J_lean(new)`. Thus even an exact frozen geometric shear is not the pure five-coordinate lean flow in a neighborhood. The spatial operator agrees on the base path but has different derivatives on nearby states.', '',
           '| Steps to 10° | Step (degrees) | Width error | Maximum of five errors | State image error |',
           '|---:|---:|---:|---:|---:|']
    for r in data['lean']['pixel_warps']:
        if r['lean']==10. and r['steps'] in [1,4,16,64,256,512]:
            lines.append(f"| {r['steps']} | {r['step_degrees']:.6g} | {100*r['relative_errors'][3]:.3f}% | {100*r['max_relative_error']:.3f}% | {100*r['state_relative_error']:.3f}% |")
    lines += ['', 'Every step is smaller, but repeated linear interpolation diffuses pixel information. The limiting error can increase rather than vanish. The frozen-shear mixing/scaling errors are separate from interpolation errors.',
              'Using the same `tan(delta_angle)` at every step is also not the exact angle flow: shear parameters add as slopes, not as angles. The corresponding experiment is in `results.json`.', '',
              '## Exact frame control and local curvature-only integration','',
              'Seven root paths move cx, cy, height, width, lean, joint knobs, or lean to 10°. Each is tested with 1–512 steps. Exact target frames are supplied to the coordinate control.',
              f"Worst repeated known-frame error: **{max(r['max_relative_error'] for r in data['frames']['coordinate']):.6g}**; 64 held-out states: **{data['frames']['heldout_coordinate_max_error']:.6g}**.",
              'For a separate local integrator, change coordinates to `(top, bottom, intercept, width, slope)`, integrate `J_q += H_q(current) delta_q`, then convert back to physical coordinates. It uses current exact local curvature but omits jumps at formula boundaries. It is exact in a single renderer formula cell because the decoder there is quadratic, while it can fail to converge over raster boundary crossings. This tests that specific update rule, not all possible integrators.', '',
              '| Root path | Curvature-only error, 1 step | 512 steps |', '|---|---:|---:|']
    for path in paths():
        rr=[r for r in data['frames']['local_taylor'] if r['path']==path]
        lines.append(f"| {path} | {100*rr[0]['max_relative_error']:.4f}% | {100*rr[-1]['max_relative_error']:.4f}% |")
    lines += ['', '## Correct transformation with valid companion images','',
              'Carry six valid images: the base and five nearby examples made by tiny positive knob probes. At every step apply the same image operation to all six: `render(invert(image) + knob_increment)`. At the endpoint, subtract the transformed base from each transformed companion and divide by its original knob probe. This approximates the correct differential without treating a signed derivative vector as an image.',
              'This method uses the existing known-family inverse and renderer, which contain the nonlinear family information. It is not a method based only on the original five vectors. Probes can cross events; finite probe bias is intentional and is tested by refinement. Float32 storage introduces subtraction and repeated inverse/render errors.', '',
              '| Storage | Fraction of knob range used for probes | Steps to 10° | Maximum of five errors |', '|---|---:|---:|---:|']
    for dtype,frac in [('float64',1e-5),('float32',1e-3),('float32',1e-6)]:
        for r in data['companions']:
            if r['dtype']==dtype and r['probe_fraction']==frac and r['lean']==10. and r['steps'] in [1,16,256]:
                lines.append(f"| {dtype} | {frac:g} | {r['steps']} | {100*r['max_relative_error']:.6g}% |")
    lines += ['', 'The same companion method also tests all seven root paths (each individual knob and two joint/lean controls) at 1, 16, and 256 increments; see `companion_all_knobs.csv`.', '',
              '| Move applied to all six images | Float64 max vector error after 256 steps | Float32 max vector error after 256 steps |',
              '|---|---:|---:|']
    for name in paths():
        rr=[r for r in data['companion_all_knobs'] if r['path']==name and r['steps']==256]
        lines.append(f"| {name} | {100*rr[0]['max_relative_error']:.6g}% | {100*rr[1]['max_relative_error']:.6g}% |")
    lines += ['', 'The float64 companion probe is 1e-5 of each knob range; the float32 probe is 1e-4. These probes differ, so this table is not a controlled precision-only comparison. The full probe sweep above compares equal probes.', '',
              'A forward 10-degree shear followed by its reverse should recover the source geometry. Repeated pixel warps instead leave irreversible blur; see `shear_round_trips.csv`.']
    lines += ['', '## Small steps do help a correct local approximation', '',
              'A separate nearby smooth start `(14,14.013,19.73,3.2,0)` moves to lean 1 degree without crossing any renderer formula boundary. A first-order update integrates the current analytic rate of change of all five physical-coordinate vectors. It uses current local derivatives, not the future target vectors. Unlike pixel interpolation, its error decreases with smaller increments.', '',
              '| Steps to 1 degree | Maximum raw-vector error |', '|---:|---:|']
    for r in data['smooth_euler']['results']:
        if r['steps'] in [1,4,16,64,256,512]:
            lines.append(f"| {r['steps']} | {100*r['max_relative_error']:.6g}% |")
    sm=data['smooth_and_kinks']
    lines += ['', '## Metric parallel transport has a different goal','',
              f"A smooth control starts at `{sm['smooth_start']}` and increases width by 0.6 pixels. All 101 audited points have the same renderer formula signature. Successive orthogonal frame maps approximate induced-metric parallel transport, which preserves lengths and angles, while coordinate knob vectors need not preserve them.",
              f"After 512 steps the largest raw-vector discrepancy is **{100*sm['metric'][-1]['max_relative_error']:.6g}%**. The height unit-direction discrepancy is **{sm['metric'][-1]['unit_direction_errors'][2]:.6g}**, and the true normalized frame Gram matrix changes by **{sm['true_unit_gram_change']:.6g}**. Thus this is a failure of same-knob preservation even without renderer kinks; it is not a failure of parallel transport itself.",
              f"On this same smooth path, the local quadratic-curvature rule has maximum error **{max(r['max_relative_error'] for r in sm['local_taylor']):.6g}** over all tested step counts.", '',
              '## Renderer events and general counterexample','',
              'The root caps lie on 1/16-pixel subrow boundaries. At zero lean the positive and negative derivatives agree in the limit. After lean changes, cy and height probes can have different left and right limits. Then there is no unique ordinary pixel-space Jacobian at that state, even though knob paths remain valid. These are renderer kinks, not evidence for extra independent knobs. See `kink_tests` in the JSON.',
              'The smooth flat-plane counterexample uses the independent unit fields `X = d/dx` and `Y = (sin(x), cos(x))` for `|x| < pi/2`. Transport by the x-translation differential leaves Y unchanged, while the local Y field rotates. After moving x by 0.5, the vector error stays 0.494808 for 1–512 increments. Small steps cannot repair a mismatch between noncommuting fields.', '',
              '## Scope and reproduction','',
              'Derivative probes can extend beyond the default sampling box (for example, a positive lean probe at 35 degrees); the renderer permits this. These are known-control diagnostics with deterministic paths and 64 held-out random states, not exhaustive validation of every point or discovery of knobs from unlabeled pixels. The theorem requires a regular differentiable chart. The raster experiments intentionally include one-sided conventions at events.',
              f"Analytic positive-derivative validation against forward renderer probes: maximum relative error **{data['frames']['analytic_derivative_max_relative_validation_error']:.6g}** over 64 held-out states. Maximum inverse parameter round-trip absolute error **{data['frames']['inverse_max_absolute_parameter_error']:.6g}**.",
              'Run `.venv/bin/python analyze_generated_one_incremental_transport.py` from the repository root.',
              'Files: `results.json`, `frame_control.csv`, `local_curvature.csv`, `pixel_shear.csv`, `companion_transport.csv`, and the two figures.', '',
              'Mathematical background: [MIT notes on commuting coordinate fields and flows](https://math.mit.edu/classes/18.966/2014SP/965/class4.pdf), [metrics and connections](https://www.wim.uni-mannheim.de/media/Lehrstuehle/wim/schmidt/FSS2024/Riemannian_Geometry/Web/RGch3.html).']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    print('Frame controls and local-curvature integration', flush=True)
    frames,cache=frame_and_taylor()
    print('Repeated pixel shears and accumulation histories', flush=True)
    lean,more=lean_warps();cache.update(more)
    print('Valid companion image pushforwards, float64 and float32', flush=True)
    companions,all_knobs=companion_test()
    round_trips=shear_round_trips()
    print('Smooth metric control and one-sided boundary probes', flush=True)
    smooth=smooth_and_kinks()
    euler=smooth_euler()
    data=dict(root=ROOT.tolist(),parameter_order=KNOBS,counts=COUNTS,
              method='Known-renderer diagnostics, raw positive one-sided derivatives; regular smooth-patch theorem separated from kinks',
              frames=frames,lean=lean,companions=companions,companion_all_knobs=all_knobs,shear_round_trips=round_trips,smooth_and_kinks=smooth,smooth_euler=euler)
    assert frames['analytic_derivative_max_relative_validation_error'] < 2e-5
    assert frames['inverse_max_absolute_parameter_error'] < 1e-10
    assert max(r['max_relative_error'] for r in frames['coordinate']) < 1e-8
    assert max(r['max_relative_error'] for r in smooth['local_taylor']) < 1e-10
    assert all(np.isfinite(r['max_relative_error']) for r in companions)
    (OUT/'results.json').write_text(json.dumps(data,indent=2)+'\n')
    write_csv('frame_control.csv',frames['coordinate'])
    write_csv('local_curvature.csv',frames['local_taylor'])
    write_csv('pixel_shear.csv',lean['pixel_warps'])
    write_csv('companion_transport.csv',companions)
    write_csv('companion_all_knobs.csv',all_knobs)
    write_csv('shear_round_trips.csv',round_trips)
    write_csv('smooth_euler.csv',euler['results'])
    report(data);figures(data,cache)
    print('Finished: figures/generated_one_incremental_transport',flush=True)


if __name__=='__main__':
    main()
