"""Smoothing sensitivity and independent checks for the principal-curve experiment."""
import json
import numpy as np
from analyze_generated_one_principal_curves import OUT, settings, pixels, fit, linear_start, project, error_stats
from grey_ones import RANGES


def main():
    saved = np.load(OUT/'model_and_samples.npz')
    r = json.loads((OUT/'results.json').read_text())
    train = pixels(saved['train_knobs'])
    val = pixels(settings(2048, 732))
    test, mean = saved['test_images'], saved['mean']
    curves = [saved['curves'][0]]
    trace0 = json.loads((OUT/'fit_history.json').read_text())[0][0]
    traces = [trace0]
    init = linear_start(train-mean)
    for penalty in [100., 1000., 10000., 100000.]:
        c, trace = fit(train-mean, val-mean, penalty, init, 'smoothing sensitivity')
        if np.dot(c[-1]-c[0], curves[0][-1]-curves[0][0]) < 0:
            c = c[::-1]
        curves.append(c); traces.append(trace)
    sensitivity = []
    for c, trace in zip(curves, traces):
        pred, t, _ = project(test-mean, c)
        length = float(np.linalg.norm(np.diff(c,axis=0),axis=1).sum())
        sensitivity.append(dict(penalty=trace['penalty'], selected_iteration=trace['selected_iteration'],
            validation_mse=trace['validation_mse'], length=length,
            length_to_chord=float(length/np.linalg.norm(c[-1]-c[0])),
            test=error_stats(test-mean-pred, test, mean)))
    # Exact known one-knob lean sweep, with all other knobs at range midpoints.
    p = np.tile(RANGES.mean(axis=1),(901,1)); p[:,4] = np.linspace(*RANGES[4],901)
    lean_path = pixels(p)
    lean_length = float(np.linalg.norm(np.diff(lean_path,axis=0),axis=1).sum())
    fitted, t, _ = project(test, lean_path)
    r['exact_mid_knob_lean_path'] = dict(length=lean_length,test=error_stats(test-fitted,test,mean))
    r['smoothing_sensitivity'] = sensitivity
    # Continuous projection implementation: compare with a direct segment loop.
    x = test[:17]-mean; c = saved['curves'][0]
    pred, _, dist = project(x,c)
    direct = np.full(len(x),np.inf)
    for a,b in zip(c[:-1],c[1:]):
        d=b-a; alpha=np.clip((x-a)@d/(d@d),0,1)
        direct = np.minimum(direct, np.sum((x-a-alpha[:,None]*d)**2,axis=1))
    assert np.allclose(dist,direct,atol=1e-10)
    # Check that stored sequential reconstructions equal the stated decoder.
    residual=test-mean; components=[]
    for c in saved['curves']:
        f, _, _ = project(residual,c); components.append(f); residual -= f
    assert np.allclose(mean+np.sum(components,axis=0),saved['reconstructed'],atol=1e-10)
    assert all(b['explained_percent'] >= a['explained_percent'] for a,b in zip(r['curves_cumulative'],r['curves_cumulative'][1:]))
    r['checks'] = dict(all_784_pixels_scored=True, exact_segment_projection_verified=True,
                       sequential_decoder_verified=True, heldout_errors_monotone=True)
    (OUT/'results.json').write_text(json.dumps(r,indent=2)+'\n')
    np.savez_compressed(OUT/'smoothing_sensitivity.npz', curves=np.array(curves), lean_path=lean_path)
    print(json.dumps(sensitivity,indent=2),flush=True)
    print('Exact lean path:',r['exact_mid_knob_lean_path'],flush=True)


if __name__=='__main__':
    main()
