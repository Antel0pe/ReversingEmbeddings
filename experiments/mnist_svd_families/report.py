"""Build the shareable report and rerender figures from measured arrays."""
from pathlib import Path
import sys
import json
import html
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment import OUT, DIMS, FEATURES, plot_all


def main():
    report = json.loads((OUT/'results.json').read_text())
    results = report['datasets']
    arrays = [dict(np.load(OUT/f)) for f in ['generated_arrays.npz', 'mnist_arrays.npz']]
    old = np.load(REPO/'figures/root_one_local_svd/bases_and_samples.npz')
    from experiment import fit
    singular, _ = fit(old['r_0p001_images'][:512]-old['root_image'].reshape(784))
    plot_all(results, arrays, dict(image=old['root_image'], basis=old['r_0p001_basis'][:5], singular=singular))
    extra = {}
    for d, a in zip(results, arrays):
        assigned = a['atlas_assignment']
        radii = np.array([r['radius'] for r in d['records'] if r['k'] == 128])
        distances = np.linalg.norm(a['test_images']-a['anchor_images'][assigned], axis=1)
        inside = distances <= radii[assigned]
        strict = dict(inside_count=int(inside.sum()), outside_count=int((~inside).sum()),
                      inside_median_relative=np.median(a['atlas_errors_128'][:, inside], axis=1).tolist(),
                      outside_median_relative=np.median(a['atlas_errors_128'][:, ~inside], axis=1).tolist())
        g = d['graphs']['pixel cosine']
        edges = {(e['i'], e['j']):e for e in g['edges']}
        filtered = []
        for minimum in [.6, .7, .8]:
            valid = {key for key,e in edges.items() if min(e['pixel_cosine']) >= minimum}
            triangles = [t for t in g['triangles'] if all(edge in valid for edge in
                        [(t['i'],t['j']), (t['i'],t['k']), (t['j'],t['k'])])]
            filtered.append(dict(minimum_all_axis_cosine=minimum, edges=len(valid), triangles=len(triangles),
                                 axis_closure=float(np.mean([t['fraction'] for t in triangles])) if triangles else None))
        strict['confidence_filter'] = filtered
        strict['entire_frame_triangle_closures'] = sum(t['fraction'] == 1 for t in g['triangles'])
        strict['descriptor_anchor_count'] = int(np.isfinite(a['descriptor_correlations']).all((1,2)).sum())
        extra[d['name']] = strict
    (OUT/'coverage_diagnostics.json').write_text(json.dumps(extra, indent=2)+'\n')
    controls = json.loads((OUT/'fitting_controls.json').read_text())
    verification = json.loads((OUT/'verification.json').read_text())
    lines = [
        '# MNIST 1s: local SVD vectors, families, and held-out reconstruction',
        '',
        'The local bases reveal useful handwriting variation, especially lean, thickness/ink, and horizontal position. '
        'Five vectors do not capture MNIST nearly as well as the controlled family, and nearest-direction labels do not form consistent global coordinates. '
        'Importantly, a finite generated cloud also loses the exceptionally clean behavior of generator-assisted tiny neighborhoods.',
        '',
        '## Exactly how the earlier generated experiment grouped vectors',
        '',
        'These were different operations, with different meanings of distance:',
        '',
        '1. **Selecting neighbors for SVD:** render points on a shell at equal Euclidean distance from the current starting image. '
        'The earlier 17-start survey used radii 0.1, 0.01, and 0.001, with 512 fitting and 256 independent checking neighbors. '
        'Fit the uncentered matrix whose rows are `neighbor image − starting image`; keep the leading right singular vectors. '
        'This is displacement energy about the start, not mean-centered PCA variance.',
        '2. **Matching the 17 frames to the root:** jointly assign the five unit vectors to the root\'s five vectors to maximize '
        '`sum(abs(dot(root vector, candidate vector)))`, using Hungarian assignment. Align signs afterward. '
        'This is closest direction up to sign, not closest image to the root. For unit vectors, the equivalent squared distance is '
        '`min(||a−b||², ||a+b||²) = 2 − 2 abs(dot(a,b))`. Low-quality matches were still assigned and shown.',
        '3. **Nearest-direction continuation:** after a move, independently select the endpoint vector with the largest absolute cosine to each source vector. '
        'The tested neighborhood radii were 0.001 and 0.00001. This does not enforce a one-to-one frame assignment in its main reported rule.',
        '4. **Later multi-path pattern grouping:** use 256 shell neighbors at radius 0.0001, keep five axes, and render candidate rays at 33 ticks '
        'from −2.4 to +2.4. Context offsets 0.2, 0.4, 0.6, 0.8 provide profiles; 0.3 and 0.7 check fresh offsets. '
        'Compare corresponding-path distance, displacement-vector mismatch, and displacement angle; normalize profiles by the start-image gap '
        '(angles first divided by 90 degrees). Choose signs by whole-curve displacement agreement. Enumerate 120 assignments at the nearest '
        'offset and use Hungarian assignments at the others to minimize profile discrepancy. The nearest offset anchors the profile comparison; '
        'it is not a nearest-distance rule for the vectors themselves.',
        '',
        '**Distance from the root was a diagnostic plot. Neither linearly increasing root distance nor constant physical knob increments defined these families.** '
        'Travel and probe scales were calibrated by initial pixel speed, and the generator/Jacobian supplied valid moves. '
        'The later multi-path run obtained 45/45 correctly signed executed steps across five nine-hop walks around one root, with mixed directions still present. '
        'That result is not evidence of global recovery, and its generator-assisted probes are unavailable on raw MNIST.',
        '',
        'Sources: [17-start comparison](../generated_one_local_bases_across_starts/README.md), '
        '[nearest continuation](../generated_one_nearest_direction/README.md), '
        '[first branch rule](../generated_one_svd_family_branches/README.md), '
        '[later multi-path rule](../generated_one_cross_path_patterns/README.md).',
        '',
        '## Adaptation to actual MNIST',
        '',
        '- Use all **6,742 training 1s** and the **1,135 official test 1s**, with intensities divided by 255. '
        'No deskewing, recentering, smoothing, augmentation, interpolation, or generated MNIST neighbors. '
        'Exact train/test duplicate checking removed zero test images. Archive checksums are in `results.json`.',
        '- Generate a separate 6,742/1,135 cloud from the canonical renderer, uniformly sampling its default five-parameter box. '
        'Both finite clouds receive exactly the same analysis. This controls sample count and method, not density, handwriting distribution, or pixel noise.',
        '- Choose the root minimizing the training 128th-neighbor distance. It is an actual image. '
        'MNIST root: original training index **5520**; nearest-to-mean comparison root: **45926**. '
        'Generated density root: cloud index **499**. No SVD, descriptor, or test score chooses these roots.',
        '- For the broad survey, use 64 anchors: density root, nearest-to-training-mean image, then farthest-point coverage of the denser 80% of training images. '
        'Fit 32, 64, 128, 256, and 512 observed nearest neighbors at each anchor. **128 is the main preselected setting**, not a test-optimized winner.',
        '- For family continuation, use a **separate 64-anchor exploration** starting at the same density root. '
        'Expand a frontier of 16 nearest observed training neighbors, choosing the point farthest from existing anchors within the denser 80%. '
        'Retain exploration edges plus six-nearest-anchor edges, requiring at least 16 shared SVD training neighbors. '
        'This uses pixels and distances only. The original broad-coverage graph left the root isolated, which is why it is unsuitable for continuation.',
        '- Group adjacent frames by joint maximum absolute pixel cosine. Also test an alternative: match correlations of basis responses to shared training images. '
        'Carry labels and signs along shortest pixel-distance graph routes. Different routes can disagree; the report measures that disagreement rather than forcing it away.',
        '- Hold out the official test set. For each test image, choose the nearest survey anchor using image distance, then project its displacement onto that anchor\'s fitted basis. '
        'Every test image is scored once in the whole-atlas result. Local descriptor correlations use test images inside each training-derived radius; those neighborhood observations overlap.',
        '',
        'There are **no equal-radius shells or constant path increments** in the MNIST fits. Normalizing observed displacement rows is tested only as an angular-weighting control; it does not create new images. '
        'The raw data has no mechanism for supplying infinitesimal neighbors, making new valid samples along a chosen axis, or labeling five true controls.',
        '',
        '## Reconstruction: how close do five vectors get?',
        '',
        'Remaining change is `||test − reconstruction|| / ||test − chosen anchor||`. '
        'Thus 0% is exact and 100% is the anchor alone. These percentages are **L2 residual fractions, not unexplained variance fractions or classification accuracy**. '
        'The decoder is a local affine projection; coefficients come from the target image. This checks representational capacity, not prediction from learned physical knob values.',
        '',
        '| Measured quantity, 128 neighbors | Generated finite cloud | Real MNIST |',
        '|---|---:|---:|',
    ]
    def row(label, values):
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    for dimension in [5,20]:
        index = DIMS.index(dimension)
        row(f'Median held-out remaining change, {dimension} vectors', [f'{100*d["atlas"]["128"]["median_relative"][index]:.1f}%' for d in results])
        row(f'90th percentile remaining change, {dimension} vectors', [f'{100*d["atlas"]["128"]["p90_relative"][index]:.1f}%' for d in results])
    row('Five-vector pixel RMSE, intensities 0–1', [f'{d["atlas"]["128"]["pixel_rmse"][3]:.4f}' for d in results])
    row('Median training axes for 95% displacement energy', [str(int(np.median([r['training_dimension95'] for r in d['records'] if r['k']==128]))) for d in results])
    row('Median observed 128-neighbor radius, pixel L2', [f'{np.median([r["radius"] for r in d["records"] if r["k"]==128]):.2f}' for d in results])
    row('Fraction of test images inside chosen anchor training radius', [f'{100*d["atlas"]["128"]["within_training_radius"]:.1f}%' for d in results])
    for n in [5,20]:
        row(f'Median remaining change INSIDE fitted radius, {n} vectors', [f'{100*extra[d["name"]]["inside_median_relative"][DIMS.index(n)]:.1f}%' for d in results])
    lines += [
        '',
        'The whole-atlas results include extrapolation for images outside their selected anchor radius; the last two rows isolate the supported-radius subset. '
        'The remaining unsupported fraction is itself a coverage limitation of this 64-anchor survey. The denser-80% anchor policy favors common handwriting, '
        'but every test 1, including atypical examples, remains in the whole-atlas score.',
        '',
        'The historical controlled root shell at radius 0.001 has median five-vector held-out relative error **0.002014%** and worst **0.011884%**. '
        'This is a useful idealized calibration, **not** a fair finite-data competitor: it uses the known renderer to supply arbitrarily close independent samples, '
        'with 512 fitting neighbors instead of 128. The new finite generated cloud shows that losing tiny local sampling already causes considerable deterioration.',
        '',
        'Increasing neighborhood size gives more stable covariance estimates but a less local fit. At 32/64/128/256/512 neighbors, median 95%-energy counts '
        'are generated **6/6/7/7/8**, MNIST **15/20/25/29/30**. These are finite-neighborhood linear counts, **not intrinsic dimensions**. '
        'Using 95% training energy alone would miss the held-out errors and inconsistent labels.',
        '',
        '## Do the assigned families remain consistent?',
        '',
        '| Pixel-cosine matching on supported graph | Generated finite cloud | Real MNIST |',
        '|---|---:|---:|',
    ]
    row('Connected root component', [f'{d["graphs"]["pixel cosine"]["root_component_size"]}/64 anchors' for d in results])
    row('Supported graph edges / tested triangles', [f'{len(d["graphs"]["pixel cosine"]["edges"])} / {len(d["graphs"]["pixel cosine"]["triangles"])}' for d in results])
    row('Median matched edge cosine', [f'{d["graphs"]["pixel cosine"]["median_edge_cosine"]:.3f}' for d in results])
    row('Axis labels closing a triangle', [f'{100*d["graphs"]["pixel cosine"]["triangle_axis_agreement"]:.1f}%' for d in results])
    row('All five labels closing the entire triangle', [f'{extra[d["name"]]["entire_frame_triangle_closures"]}/{len(d["graphs"]["pixel cosine"]["triangles"])}' for d in results])
    row('Shared-response alternative: axis triangle agreement', [f'{100*d["graphs"]["shared responses"]["triangle_axis_agreement"]:.1f}%' for d in results])
    row('Root-direct versus routed label agreement', [f'{100*d["graphs"]["pixel cosine"]["root_direct_route_agreement"]:.1f}%' for d in results])
    row('Median bootstrap matched-axis cosine', [f'{np.median([s["cosine"] for s in d["stability"]]):.3f}' for d in results])
    row('Median bootstrap five-space overlap', [f'{np.median([s["space_overlap"] for s in d["stability"]]):.3f}' for d in results])
    lines += [
        '',
        'A triangle check compares the direct A→C label with A→B→C. The chance expectation for independently random five-way permutations is 20% per axis. '
        'The measured ~54% is structured but far from a coherent global family assignment. These triangles share edges and anchors; they are not independent accuracy trials. '
        'Sign-inclusive closure is 53.3% generated and 53.5% MNIST.',
        '',
        'Rejecting weak matches leaves little scope: requiring **every** matched axis cosine ≥0.70 retains **11 generated edges and zero MNIST edges**, '
        'with zero complete triangles in either set. At ≥0.60, MNIST retains 11 edges and only 2 triangles, with 80% axis closure. '
        'That tiny selected subset cannot support a broad recovery claim. The full threshold tradeoff is saved in `coverage_diagnostics.json`.',
        '',
        'The generated cloud supplies a category check that MNIST cannot: assign its fitted axes to known derivatives by maximum total absolute cosine, '
        'then check whether adjacent matches preserve those categories. The pixel matcher scores **54.0%**, the response matcher **51.0%**. '
        'Median matched derivative cosine is **0.713**. These are approximate categories of mixed SVD axes, not pure-knob accuracy. '
        'In particular, internally consistent labels alone would not guarantee physical meaning.',
        '',
        'Bootstrap uses 12 resamples of 128 training displacements at each of the first 20 survey anchors. '
        'Five-space overlap is the mean squared principal cosine: it allows basis rotation. '
        'MNIST local spaces are reasonably stable while individual labels vary across states. The much harder continuation problem survives good local fits.',
        '',
        '## What the vectors correlate with',
        '',
        'No physical descriptors enter SVD, root selection, graph construction, or matching. After all fits and graph assignments, '
        'measure ink centers, vertical extent, thickness after linear lean removal, lean, row-center bend, and total ink on held-out images. '
        'The heatmap reports median absolute within-neighborhood Pearson correlations, not correlation of arbitrary global family counts.',
        '',
        '- **MNIST family 3 is most clearly lean-like:** median held-out |correlation| **0.82**; the analogous generated family is **0.79**. '
        'The same family number across datasets is incidental: each starts from its own root axes.',
        '- **MNIST family 2 mixes thickness and ink:** correlations **0.50** with thickness and **0.60** with total ink. '
        'Family 5 has a weaker horizontal-center association, **0.46**. The remaining columns mix side-edge, endpoint, and bend changes.',
        '- MNIST has no equally clear height/vertical-position family in this test. Its five-coefficient local descriptor readout has median held-out '
        'R² about **0.97 for lean**, **0.78 thickness**, **0.75 horizontal center**, **0.17 vertical center**, and **0.046 vertical extent**. '
        'These scores evaluate all five local coefficients jointly; they do not assign a single coefficient to each descriptor.',
        '- Linear projection of new coefficients can leave the valid intensity cube and produce ghosts or malformed strokes. '
        'The reconstruction figure marks all pixels outside [0,1] in amber, and scores the unclipped output.',
        '',
        'Definitions: center = intensity-weighted pixel-center mean; extent = √12 times vertical standard deviation; '
        'lean = atan(−covariance(x,y)/variance(y)) in degrees; thickness = √12 times horizontal standard deviation after linear regression on y; '
        'bend = intensity-weighted RMS row-center residual from its best line, using rows with >5% of the maximum row ink; total ink = sum of intensities. '
        'These are continuous descriptive measurements, not ground-truth handwriting controls.',
        '',
        '## Fitting controls and a global linear reference',
        '',
        '| Local fitting rule, 128 neighbors | Generated mean L2 error, 5 vectors | MNIST mean L2 error, 5 vectors |',
        '|---|---:|---:|',
    ]
    for variant in controls['MNIST']:
        row(variant, [f'{controls[d["name"]][variant]["5"]["mean_pixel_l2"]:.3f}' for d in results])
    row('One global five-component mean-centered PCA', [f'{d["atlas"]["128"]["global_mean_absolute"][3]:.3f}' for d in results])
    lines += [
        '',
        'Angular weighting divides each displacement row by its L2 norm before SVD. This gives nearby and distant directions equal length in the fitting matrix, '
        'without claiming any new valid image. Mean-centered local PCA uses the training-neighborhood mean as its affine offset. '
        'It improves MNIST mean five-vector L2 error from **1.773 to 1.706**, a modest change; it does not erase the generated/MNIST gap. '
        'A training-derived local atlas is much more effective than one global five-dimensional linear space, but this is still a set of local projections.',
        '',
        '## What this establishes and leaves open',
        '',
        'The experiment supports local low-dimensional approximations and a strong lean-related change on real 1s. '
        'It does **not** establish five globally identifiable handwriting knobs, a route-independent coordinate system, a decoder covering every possible 1, '
        'or a real-data analogue of the assisted 45/45 generated walk result. The finite generated control is essential: much of the clean original behavior '
        'relied on arbitrarily small, controlled sampling, not just on the existence of five latent controls.',
        '',
        'The next discriminating experiment would learn transitions between overlapping local subspaces, allowing mixtures instead of restricting each family to a single SVD axis, '
        'then evaluate independent chart overlap and held-out decoding. Denoising/alignment would be separate experiments because they change the pixel geometry. '
        'Neither is silently included here.',
        '',
        '## Figures',
        '',
    ]
    images = [('01_root_basis_vectors.png', 'Starting images and their five leading vectors'),
              ('02_vector_families.png', 'Vectors assigned to the same root families across starts'),
              ('03_heldout_reconstruction.png', 'Held-out errors and median-error image examples'),
              ('04_family_correlations.png', 'Independent held-out descriptor correlations'),
              ('05_stability_and_scale.png', 'Bootstrap stability, route consistency, sampling scale, and spectra')]
    for file, label in images:
        lines += [f'![{label}]({file})', '']
    lines += [
        '## Reproduce and audit', '',
        '```sh',
        'OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/mnist_svd_families/experiment.py',
        'OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/mnist_svd_families/controls_and_check.py',
        'OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/mnist_svd_families/report.py',
        '```', '',
        'Seed: 20261018. Uses the existing NumPy, SciPy, Matplotlib environment and local MNIST archives. '
        'The historical tiny-shell archive is an explicit input to calibration, not silently regenerated.', '',
        '- `results.json`: method, checksums, all 640 survey neighborhood fits, graph edges and triangles, bootstrap results, atlas errors, and descriptor summaries.',
        '- `mnist_arrays.npz`, `generated_arrays.npz`: exact training source indices, test images, bases, assigned families, descriptor correlations, per-image errors, and coefficients\' source data.',
        '- `test_indices.npz`: original official MNIST test indices.',
        '- `neighborhood_metrics.csv`: per-anchor and per-scale radii, energy counts, and held-out neighborhood counts.',
        '- `fitting_controls.json`, `coverage_diagnostics.json`: centering/weighting alternatives, supported-radius checks, confidence rejection tradeoffs.',
        '- `verification.json`: independent reconstruction of all 1,135 test images per cloud, nearest-anchor and orthonormality checks, and five original-pixel SVD refits.',
        '',
        f'Independent saved-array audit: maximum metric discrepancy '
        f'{max(v["maximum_metric_discrepancy"] for v in verification.values()):.2g}; minimum refitted five-space cosine '
        f'{min(v["minimum_refit_subspace_cosine"] for v in verification.values()):.15f}. '
        'All five final figures were read back and checked for image meaning, baseline, units, sign, sampling scope, clipping, and layout.',
        '',
    ]
    (OUT/'README.md').write_text('\n'.join(lines))
    # A local, portable visual entry point; no machine-specific filesystem paths.
    cards = ''.join(f'<section><h2>{html.escape(label)}</h2><a href="{file}"><img src="{file}" alt="{html.escape(label)}"></a></section>' for file,label in images)
    (OUT/'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>MNIST and generated 1s — local SVD</title><style>body{font:17px/1.6 system-ui;background:#f3f1ec;color:#212b32;margin:0 auto;padding:36px;max-width:1400px}'
        'h1{font-size:36px;line-height:1.2}h2{font-size:23px}section{background:white;padding:18px;margin:30px 0;border-radius:8px}img{width:100%;height:auto}a{color:#245875}</style>'
        '<h1>Real and generated 1s: local SVD families</h1><p>Five vectors leave median held-out change of <b>53.6% for MNIST</b> versus <b>27.6% for a size-matched generated cloud</b>. '
        'A lean-like family is visible, but labels agree around graph triangles only about 54% of the time. This is local structure, not recovered global handwriting knobs.</p>'
        '<p><a href="README.md">Full methods and interpretation</a> · <a href="results.json">Measured results</a> · <a href="verification.json">Independent verification</a></p>'+cards+'</html>')
    print(json.dumps(extra, indent=2))


if __name__ == '__main__':
    main()
