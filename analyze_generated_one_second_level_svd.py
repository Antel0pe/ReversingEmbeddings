"""Second-level SVD of generated-one direction families and their changes.

Known parameters are supplied. Ordinary local fits use only rendered pixel
changes; analytic derivatives provide an explicitly separate oracle control.
All files and reproduction instructions use repository-relative paths.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/generated-one-mpl')
from pathlib import Path
from itertools import product, combinations_with_replacement
import json
import time
import numpy as np
from scipy.stats import qmc
from scipy.optimize import linear_sum_assignment
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from grey_ones import render, KNOBS, RANGES
from analyze_generated_one_manifold_transport import jet, chart, unchart
import analyze_generated_one_nearest_direction as nearest

OUT = Path(__file__).resolve().parent / 'figures/generated_one_second_level_svd'
ROOT = np.array([14., 14., 19.75, 3.2, 0.])
SMOOTH = np.array([14.43, 14.62, 19.73, 3.17, 17.3])
SCALE = np.diff(RANGES, axis=1).ravel()
RANKS = [0, 1, 2, 3, 5, 10, 20, 40, 80, 160, 267]
PAIRS = list(combinations_with_replacement(range(5), 2))


def sobol(power, seed, d=5):
    return qmc.Sobol(d, scramble=True, seed=seed).random_base2(power)


def pixels(p):
    return render(p, dtype=np.float64).reshape(-1, 784)


def stats(predicted, truth):
    error = np.linalg.norm(predicted-truth, axis=-1)
    norms = np.linalg.norm(truth, axis=-1)
    relative = error / np.maximum(norms, 1e-30)
    return dict(median_relative=float(np.median(relative)),
                worst_relative=float(relative.max()),
                rms_relative=float(np.linalg.norm(error)/np.linalg.norm(norms)),
                worst_absolute_L2=float(error.max()),
                worst_absolute_pixel=float(np.abs(predicted-truth).max()))


def features(x, degree):
    terms = [np.ones(len(x))]
    terms.extend(x.T)
    if degree >= 2:
        terms.extend(x[:, i]*x[:, j] for i, j in combinations_with_replacement(range(x.shape[1]), 2))
    return np.column_stack(terms)


def compress(train, test, xtrain=None, xtest=None):
    """Centered SVD across states, preserving the mean as an extra vector.

    Projection is an optimistic compression test using the target vector.
    Polynomial coefficient prediction uses only supplied state coordinates.
    """
    mean = train.mean(0)
    _, singular, basis = np.linalg.svd(train-mean, full_matrices=False)
    numerical_rank = int(np.sum(singular > max(singular[0], 1e-30)*1e-10))
    basis = basis[:numerical_rank]
    rows = []
    for k in sorted(set(RANKS + [numerical_rank])):
        k = min(k, len(basis))
        if any(r['modes'] == k for r in rows):
            continue
        b = basis[:k]
        coeff_train = (train-mean)@b.T
        coeff_test = (test-mean)@b.T
        row = dict(modes=k, projection=stats(mean+coeff_test@b, test))
        if xtrain is not None and k in [0, 1, 2, 3, 5, 10, 20, 40, 80]:
            for degree in [1, 2]:
                fitted = np.linalg.lstsq(features(xtrain, degree), coeff_train, rcond=None)[0]
                predicted = mean + features(xtest, degree)@fitted@b
                row[f'coefficient_polynomial_degree_{degree}'] = stats(predicted, test)
        rows.append(row)
    # A cumulative direct residual gives the smallest rank meeting 1% worst
    # held-out relative vector error, rather than a training-energy cutoff.
    residual = test-mean
    norms = np.linalg.norm(test, axis=1)
    k_for_one_percent = 0 if np.max(np.linalg.norm(residual, axis=1)/norms) <= .01 else None
    for k, b in enumerate(basis, 1):
        residual -= np.outer(residual@b, b)
        if k_for_one_percent is None and np.max(np.linalg.norm(residual, axis=1)/norms) <= .01:
            k_for_one_percent = k
    return dict(centered_numerical_rank_rtol_1e_10=numerical_rank,
                modes_for_worst_heldout_relative_error_1_percent=k_for_one_percent,
                singular_values=singular.tolist(), rows=rows), (mean, basis)


def fields(points):
    values = np.array([jet(p)[3].T for p in points])
    return values


def family_experiments(arrays):
    results = {}
    for name in ['near_root', 'full_default_box']:
        train_u = sobol(9, 2026101701)
        test_u = sobol(8, 2026101702)
        if name == 'near_root':
            # Two-sided cx/cy box explicitly extends below their default 14.
            center = ROOT
            half = SCALE*.03
            train = center+(2*train_u-1)*half
            test = center+(2*test_u-1)*half
            lower, upper = center-half, center+half
        else:
            lower, upper = RANGES[:, 0], RANGES[:, 1]
            train = lower+train_u*(upper-lower)
            test = lower+test_u*(upper-lower)
        # Include all 3^5 lattice settings, not only random interior states.
        grid_u = np.array(list(product([0., .5, 1.], repeat=5)))
        grid = lower+grid_u*(upper-lower)
        # A central finite difference is defined even at renderer kinks; use
        # that explicit convention for lattice checks, rather than jet's
        # strict horizontal clipping indicator at exact integer boundaries.
        ftrain, ftest = fields(train), fields(test)
        fgrid = []
        step = 1e-6
        for p in grid:
            delta = np.eye(5)*step
            fgrid.append((pixels(p+delta)-pixels(p-delta))/(2*step))
        fgrid = np.array(fgrid)
        coordinate_center = (lower+upper)/2
        coordinate_half = (upper-lower)/2
        xtrain = (train-coordinate_center)/coordinate_half
        xtest = (test-coordinate_center)/coordinate_half
        region = dict(training_states=len(train), heldout_states=len(test),
                      lattice_states=len(grid), lower=lower.tolist(), upper=upper.tolist(),
                      families={})
        for i, knob in enumerate(KNOBS):
            report, (mean, basis) = compress(ftrain[:, i], ftest[:, i], xtrain, xtest)
            lattice_rows=[]
            for k in [0, 2, 5, 10, 20, 40, 80, 160, 267]:
                k=min(k, len(basis)); b=basis[:k]
                lattice_rows.append(dict(modes=k, projection=stats(mean+((fgrid[:, i]-mean)@b.T)@b, fgrid[:, i])))
            report['lattice_projection']=lattice_rows
            region['families'][knob]=report
            arrays[f'{name}_{knob}_mean']=mean
            arrays[f'{name}_{knob}_modes']=basis[:80]
        arrays[name+'_train_settings']=train
        arrays[name+'_heldout_settings']=test
        results[name]=region
        print('True direction-family compression:', name, flush=True)
    # Verify existing analytic helper against central differences at new
    # smooth full-domain settings (this is separate from compression fitting).
    errors=[]
    for p in (RANGES[:,0]+sobol(4,2026101710)*SCALE):
        delta=np.eye(5)*1e-6
        actual=(pixels(p+delta)-pixels(p-delta))/(2e-6)
        errors.append(stats(actual, jet(p)[3].T))
    results['analytic_derivative_control']=dict(states=16, finite_difference_step=1e-6,
        worst_relative=max(r['worst_relative'] for r in errors),
        worst_absolute_L2=max(r['worst_absolute_L2'] for r in errors))
    assert results['analytic_derivative_control']['worst_relative'] < 2e-7
    # Repeat with four times as many new fitting states. Keep the exact same
    # held-out full-box and lattice states to isolate sampling density.
    bigger=RANGES[:,0]+sobol(11,2026101711)*SCALE
    bigger_fields=fields(bigger)
    control={}
    for i,knob in enumerate(KNOBS):
        report,(mean,basis)=compress(bigger_fields[:,i],ftest[:,i])
        control[knob]=dict(training_states=len(bigger),
            centered_numerical_rank=report['centered_numerical_rank_rtol_1e_10'],
            modes_for_one_percent=report['modes_for_worst_heldout_relative_error_1_percent'],
            five_modes=next(r['projection'] for r in report['rows'] if r['modes']==5),
            fitted_span_heldout=report['rows'][-1]['projection'],
            fitted_span_lattice=stats(mean+((fgrid[:,i]-mean)@basis.T)@basis,fgrid[:,i]))
    results['larger_full_box_sampling_control']=control
    print('Larger full-box sampling control: 2048 fitting states',flush=True)
    return results


def align(reference, candidate):
    dots=reference@candidate.T
    r,c=linear_sum_assignment(-np.abs(dots))
    order=c[np.argsort(r)]
    aligned=candidate[order].copy()
    signs=np.sign(np.sum(reference*aligned,axis=1)); signs[signs==0]=1
    return aligned*signs[:,None]


def estimated_svd_experiments(arrays):
    nearest.TRAIN=128
    results={}
    for name, bounds in [('upright_width_patch', (2.2,3.8)),
                         ('upright_width_crossing_events', (1.8,4.6))]:
        widths=np.linspace(*bounds,129)
        points=np.tile(ROOT,(len(widths),1)); points[:,3]=widths
        raw=[]; fits=[]
        for index,p in enumerate(points):
            f=nearest.local_fit(p,1e-5,20261010)
            raw.append(f['basis']); fits.append(f)
            if index%32==0:
                print('Estimated SVD width path:',name,index,'/',len(points),flush=True)
        raw=np.array(raw)
        train_indices=np.arange(0,len(widths),2)
        test_indices=np.arange(1,len(widths),2)
        matched=np.empty_like(raw)
        matched[0]=raw[0]
        for index in train_indices[1:]:
            matched[index]=align(matched[index-2],raw[index])
        # Hold-out orientations are assigned to the closest training frame.
        # Target pixels choose only a permutation/sign, never compression modes.
        for index in test_indices:
            matched[index]=align(matched[index-1],raw[index])
        initial_labels,_=nearest.assign(matched[0],jet(points[0])[3])
        region=dict(width_bounds=list(bounds),training_states=len(train_indices),
                    heldout_states=len(test_indices),neighborhood_radius=1e-5,
                    neighbors_per_fit=128,initial_evaluation_labels=[KNOBS[i] for i in initial_labels],
                    families={})
        repeats=[]
        for index in train_indices[::4]:
            f=nearest.local_fit(points[index],1e-5,20261021)
            alternative=align(matched[index],f['basis'])
            repeats.extend(np.linalg.norm(alternative-matched[index],axis=1).tolist())
        region['independent_neighborhood_unit_vector_difference']=dict(
            median=float(np.median(repeats)),worst=float(np.max(repeats)),
            description='Independent 128-ray sample at identical states, matched by sign/permutation.')
        region['maximum_shell_relative_error']=max(f['shell_error'] for f in fits)
        region['families_initial_evaluation_label_not_pure_knob']=True
        x=((widths-widths.mean())/np.ptp(widths))[:,None]
        for i in range(5):
            key=f'axis_{i+1}_{KNOBS[initial_labels[i]]}_associated'
            result,_=compress(matched[train_indices,i],matched[test_indices,i],x[train_indices],x[test_indices])
            region['families'][key]=result
        arrays[name+'_widths']=widths
        arrays[name+'_matched_vectors']=matched
        results[name]=region
    return results


def cell_experiments(arrays):
    q0=chart(SMOOTH)
    f,j,h,jp,sig=jet(SMOOTH)
    corners=np.array(list(product([-1.,1.],repeat=5)))
    half=np.array([1.,1.,1.,1.,.1])*1e-3
    while not all(np.array_equal(jet(unchart(q0+d))[4],sig) for d in corners*half):
        half/=2
    ztrain=2*sobol(8,2026101703)-1
    ztest=2*sobol(9,2026101704)-1
    ptr=np.array([unchart(q0+z*half) for z in ztrain])
    pte=np.array([unchart(q0+z*half) for z in ztest])
    truth_train=pixels(ptr); truth_test=pixels(pte)
    design=features(ztrain,2)
    # Subtract the source image to avoid solving against a large DC term.
    beta=np.linalg.lstsq(design,truth_train-f,rcond=None)[0]
    fitted=f+features(ztest,2)@beta
    linear_beta=np.linalg.lstsq(features(ztrain,1),truth_train-f,rcond=None)[0]
    linear=f+features(ztest,1)@linear_beta
    # Derivatives are learned from those images, not copied from jet.
    jlearn=beta[1:6].T/half
    hlearn=np.zeros((784,5,5))
    for k,(i,l) in enumerate(PAIRS):
        value=beta[6+k]/(half[i]*half[l])
        hlearn[:,i,l]=hlearn[:,l,i]=value*(2 if i==l else 1)
    true_j=np.array([jet(p)[1].T for p in pte])
    pred_j=np.array([(jlearn+np.einsum('pij,j->pi',hlearn,z*half)).T for z in ztest])
    families={}
    for i,name in enumerate(['top','bottom','intercept','width','slope']):
        delta_train=np.array([jet(p)[1][:,i]-j[:,i] for p in ptr])
        delta_test=np.array([jet(p)[1][:,i]-j[:,i] for p in pte])
        # Uncentered SVD of vector changes relative to the reference.
        _,s,b=np.linalg.svd(delta_train,full_matrices=False)
        k=int(np.sum(s>max(s[0],1e-30)*1e-9))
        predicted=(delta_test@b[:k].T)@b[:k]
        families[name]=dict(direction_change_rank=k,singular_values=s[:10].tolist(),
            direction_change_reconstruction=stats(predicted,delta_test))
    interaction=np.array([h[:,i,l] for i,l in PAIRS])
    interaction_sv=np.linalg.svd(interaction,compute_uv=False)
    interaction_rank=int(np.sum(interaction_sv>interaction_sv[0]*1e-10))
    # Recover the physical width rule from the observed quadratic coefficients.
    cy_pattern=hlearn[:,0,3]+hlearn[:,1,3]
    height_pattern=(-hlearn[:,0,3]+hlearn[:,1,3])/2
    width_base=jlearn[:,3]
    width_prediction=np.array([width_base+(p[1]-SMOOTH[1])*cy_pattern+
        (p[2]-SMOOTH[2])*height_pattern for p in pte])
    true_width=np.array([jet(p)[3][:,3] for p in pte])
    expansion=[]
    for factor in [0,1,2,5,10,25,50,100,250,500]:
        z=ztest*factor
        ps=np.array([unchart(q0+v*half) for v in z])
        truth=pixels(ps)
        pred=f+features(z,2)@beta
        actual_j=np.array([jet(p)[3][:,3] for p in ps])
        pred_jw=np.array([width_base+(p[1]-SMOOTH[1])*cy_pattern+
            (p[2]-SMOOTH[2])*height_pattern for p in ps])
        same=sum(np.array_equal(jet(p)[4],sig) for p in ps)
        change=truth-f
        abs_error=np.linalg.norm(pred-truth,axis=1)
        norms=np.linalg.norm(change,axis=1)
        expansion.append(dict(box_expansion_factor=factor,same_formula_cell_count=same,
            tested_states=len(ps),image=stats(pred,truth),
            worst_error_relative_to_image_change=float(np.max(abs_error/np.maximum(norms,1e-30))) if factor else None,
            width_direction=stats(pred_jw,actual_j)))
    results=dict(reference_settings=SMOOTH.tolist(),chart_reference=q0.tolist(),
        coordinate_names=['top','bottom','intercept','width','slope'],
        certified_cell_half_spans=half.tolist(),training_images=len(ptr),heldout_images=len(pte),
        quadratic_design_columns=design.shape[1],design_condition_number=float(np.linalg.cond(design)),
        all_32_corners_same_formula=True,quadratic_image=stats(fitted,truth_test),
        affine_image=stats(linear,truth_test),learned_direction_fields=stats(pred_j,true_j),
        learned_physical_width_rule=stats(width_prediction,true_width),
        joint_interaction_rank=interaction_rank,second_level_families=families,
        extrapolation=expansion)
    assert np.max(np.abs(f-pixels(SMOOTH)[0]))<1e-13
    assert results['quadratic_image']['worst_absolute_L2']<1e-11
    assert results['learned_physical_width_rule']['worst_relative']<1e-7
    arrays.update(cell_reference=SMOOTH,cell_half_spans=half,cell_polynomial_coefficients=beta,
        cell_learned_jacobian=jlearn,cell_learned_hessian=hlearn,
        width_reference=width_base,width_cy_pattern=cy_pattern,width_height_pattern=height_pattern,
        cell_heldout_settings=pte,cell_heldout_images=truth_test)
    return results


def figure(results,arrays):
    cell=results['cell']
    fig=plt.figure(figsize=(14,10),facecolor='white')
    fig.text(.04,.97,'Can a changing knob direction have a short equation?',fontsize=20,weight='bold',va='top')
    fig.text(.04,.927,'Controlled 28 × 28 generated 1s. Each pixel stores ink coverage: 0 = white, 1 = black.\n'
        'Pixel columns increase rightward; rows downward. Top: one smooth formula cell around (14.43, 14.62, 19.73, 3.17, 17.3°).\n'
        'Bottom left: the full five-knob box. Bottom right: expanding the local box.',fontsize=10.5,va='top',linespacing=1.5)
    grid=fig.add_gridspec(1,4,left=.06,right=.92,bottom=.59,top=.82,wspace=.25)
    vectors=[pixels(SMOOTH)[0],arrays['width_reference'],arrays['width_cy_pattern'],arrays['width_height_pattern']]
    titles=['1. Reference image','2. Width direction here','3. Change when cy moves','4. Change when height moves']
    labels=['Ink coverage [0, 1]','Ink change per +1 px width','Direction change per +1 px cy','Direction change per +1 px height']
    limit=max(np.abs(v).max() for v in vectors[1:])
    for i in range(4):
        ax=fig.add_subplot(grid[i])
        if i==0: ax.imshow(vectors[i].reshape(28,28),cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
        else: im=ax.imshow(vectors[i].reshape(28,28),cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
        ax.set_title(titles[i],fontsize=10,pad=9)
        ax.set_xticks([]);ax.set_yticks([])
        ax.text(.5,-.09,labels[i],ha='center',va='top',transform=ax.transAxes,fontsize=9)
    ca=fig.add_axes([.94,.63,.012,.18]); fig.colorbar(im,cax=ca)
    fig.text(.06,.535,'width direction = reference pattern + (cy − 14.62) × pattern 3 + (height − 19.73) × pattern 4',
        fontsize=12,weight='bold')
    fig.text(.06,.496,f"Patterns recovered from 256 rendered images by a 21-column quadratic linear solve. "
        f"Worst width-direction error on 512 new states: {cell['learned_physical_width_rule']['worst_relative']:.2e}.\n"
        'Signed panels share one scale. Panel 2: red gains ink, blue loses ink. '
        'Panels 3–4 change the direction pattern itself; colors are not direct ink changes.',fontsize=9.5,linespacing=1.5)
    ax=fig.add_axes([.07,.13,.40,.27]); bx=fig.add_axes([.57,.13,.36,.27])
    for knob in KNOBS:
        rows=results['families']['full_default_box']['families'][knob]['rows']
        ax.plot([r['modes'] for r in rows],[100*r['projection']['worst_relative'] for r in rows],marker='.',label=knob)
    ax.set_xlim(0,245);ax.set_ylim(0,125)
    ax.set_xlabel('Fixed variation patterns retained, plus one mean vector',fontsize=9)
    ax.set_ylabel('Worst held-out direction error (%)',fontsize=9)
    ax.set_title('A few fixed patterns do not cover the whole box',fontsize=11)
    ax.axhline(1,color='black',ls=':',lw=1,label='1% target')
    ax.legend(fontsize=8,ncol=3,loc='upper right')
    ax.grid(alpha=.2)
    expansion=cell['extrapolation'][1:]
    bx.plot([r['box_expansion_factor'] for r in expansion],
            [max(1e-12,100*r['width_direction']['worst_relative']) for r in expansion],marker='o')
    bx.set_xscale('log');bx.set_yscale('log')
    bx.axhline(1,color='black',ls=':',lw=1)
    bx.set_xlabel('Expansion of the certified five-coordinate box',fontsize=9)
    bx.set_ylabel('Worst width-direction error (%)',fontsize=9)
    bx.set_title('The local equation fails after formula changes',fontsize=11)
    bx.grid(alpha=.2)
    for r in [expansion[0],expansion[4],expansion[-1]]:
        bx.annotate(f"{r['same_formula_cell_count']}/512 same cell",(r['box_expansion_factor'],
            max(1e-12,100*r['width_direction']['worst_relative'])),
            xytext=(-6,-15) if r is expansion[-1] else (4,8),
            ha='right' if r is expansion[-1] else 'left',textcoords='offset points',fontsize=7)
    fig.text(.06,.045,'Left: SVD modes fit to 512 states; tested on 256 independent Sobol states. Projection uses the true held-out direction to choose its coefficients.\n'
        'Right: the explicit local width equation predicts without target vectors. All 784 pixels are evaluated; these are sampled tests, not a global coverage proof.',
        fontsize=9,va='top',linespacing=1.5)
    fig.savefig(OUT/'second_level_svd.png',dpi=155)
    plt.close(fig)


def write_report(results):
    fam=results['families'];cell=results['cell']
    lines=['# Second-level SVD: compressing how generated-one directions vary','',
        '## What the earlier SVD actually used','',
        'The root/local-bases experiments used 512 rendered neighbors for fitting and 256 independent checking neighbors. '
        'Each row was the 784-pixel vector `neighbor image - starting image`; all five knobs varied together. '
        'The neighborhood mean was not subtracted. The five leading right singular vectors approximate a local space, '
        'and need not be five pure physical knobs. At root and radius 0.001 their derivative-assignment cosines were '
        'approximately 1, 1, 1, 0.9987, 0.9884. The worst held-out displacement error with five vectors was 0.0119%. '
        'These are distinct metrics; a cosine of 0.9884 is not 98.84% knob recovery accuracy. '
        'See [root experiment](../root_one_local_svd/README.md).','',
        'The later nearest-direction experiment used 256 shell neighbors per SVD, kept five vectors, '
        'and matched by largest absolute cosine at neighboring states. It succeeded in the original small-step sample '
        'and failed at larger moves and some boundary crossings. '
        'See [nearest-direction experiment](../generated_one_nearest_direction/README.md).','',
        '## Objects at the second level','',
        '1. A local displacement matrix has rows `F(p + small joint move) - F(p)`. Its SVD yields axes `b_i(p)`.\n'
        '2. Fix one matched axis label, stack `b_i(p)` from many states, subtract their mean, then run SVD. '
        'This gives fixed pixel patterns describing variation of that selected axis.\n'
        '3. A calibrated physical direction is `V_i(p) = derivative of F with respect to knob i`, with its speed retained. '
        'Stacking these is a separate oracle control; physical families are supplied here.\n'
        '4. Stack `V_i(p) - V_i(reference)` to isolate changes in the direction. '
        'At smooth points their rate of change is a second derivative, or mixed derivative when another knob moves.','',
        '`V_i(p) = mean direction + sum(coefficient_j(p) * fixed pattern_j)` is a legitimate compact representation. '
        'Repeating SVD on the same matrix or its orthonormal basis does not expose a new level of geometry. The new input must be variation across states. SVD supplies the patterns and coefficients at observed states; it does not automatically supply the functions '
        '`coefficient_j(p)`. We separately test explicit degree-one/two coefficient equations.','',
        'All rendering retains float64 coverage; root is `(14,14,19.75,3.2,0)`. No neural network is used. '
        'This is compression conditional on known coordinates and/or supplied family associations, not discovery of unique knobs.','',
        '## Test 1: true knob directions across states','',
        '512 fitting Sobol states and 256 independent Sobol states per region. '
        'Directions have their calibrated pixel/degrees speeds, not unit normalization. '
        'Second-level SVD is mean-centered. One mean vector is stored in addition to the counted modes. '
        'Projection errors are optimistic: the true target vector supplies the coefficients.','',
        '| Family | Near root: modes for ≤1% worst test error | Full box: modes for ≤1% worst test error | Full box: worst error with 5 modes |',
        '| --- | ---: | ---: | ---: |']
    for knob in KNOBS:
        a=fam['near_root']['families'][knob]
        b=fam['full_default_box']['families'][knob]
        e=next(r for r in b['rows'] if r['modes']==5)['projection']['worst_relative']
        lines.append(f"| {knob} | {a['modes_for_worst_heldout_relative_error_1_percent']} | {b['modes_for_worst_heldout_relative_error_1_percent'] if b['modes_for_worst_heldout_relative_error_1_percent'] is not None else 'Not reached in fitted span'} | {e:.2%} |")
    lines += ['', 'Near-root half-spans are 3% of each default knob range; cx and cy therefore explicitly extend left/up below 14. '
        'The full box is exactly the five default ranges. A separate 243-state three-level lattice checks edges and joint settings; '
        'its central differences use step 1e-6 and average two sides at nonsmooth settings. Those lattice errors are saved separately. '
        'A learned span may miss patterns occurring only at a boundary. Only modes above relative singular cutoff 1e-10 are retained; arbitrary null-space vectors do not count as learned modes. No continuum-wide guarantee follows from these sampled tests.','',
        'A four-times-larger control uses 2,048 independent fitting states and the same held-out/lattice targets. '
        'The table below reports the numerical fitted span and residuals; low-rank failures survive denser sampling.','',
        '| Family | Fitted centered rank | Worst error with 5 modes | Worst error with full fitted span, random targets | Worst error with full fitted span, lattice targets |',
        '| --- | ---: | ---: | ---: | ---: |']
    for knob,row in fam['larger_full_box_sampling_control'].items():
        lines.append(f"| {knob} | {row['centered_numerical_rank']} | {row['five_modes']['worst_relative']:.2%} | "
            f"{row['fitted_span_heldout']['worst_relative']:.2%} | {row['fitted_span_lattice']['worst_relative']:.2%} |")
    lines += ['',
        '## Test 2: actual first-level SVD vectors','',
        'Two width paths keep cx=cy=14, height=19.75, lean=0: widths 2.2–3.8 within an upright patch and 1.8–4.6 across events. '
        'Every first-level SVD uses 128 equally weighted rendered shell neighbors at image radius 1e-5. '
        '65 training widths are matched sequentially by one-to-one absolute pixel cosine, allowing signs; '
        '64 held-out widths are matched to their nearest preceding training state. No physical labels choose matches. '
        'One set of parameter rays is reused along the path to reduce sample jitter. '
        'Initial knob names in the table are evaluation associations only; they do not assert purity or persistent physical identity.','',
        '| Width path | Initial association | Modes for ≤1% worst test vector error | Worst error, 2 modes | Worst error, 2 modes + quadratic coefficient equation |',
        '| --- | --- | ---: | ---: | ---: |']
    for path,region in results['estimated'].items():
        for label,row in region['families'].items():
            r=next(r for r in row['rows'] if r['modes']==2)
            lines.append(f"| {path} | {label} | {row['modes_for_worst_heldout_relative_error_1_percent']} | "
                f"{r['projection']['worst_relative']:.2%} | {r['coefficient_polynomial_degree_2']['worst_relative']:.2%} |")
    lines += ['', 'Independent 128-ray repeat fits at 17 identical training states quantify sample sensitivity:']
    for path,region in results['estimated'].items():
        r=region['independent_neighborhood_unit_vector_difference']
        lines.append(f"- {path}: median unit-vector difference {r['median']:.4g}; maximum {r['worst']:.4g}.")
    lines += ['', 'Ordinary orthogonal SVD axes can rotate as neighborhood covariance changes, even when the underlying physical fields have simple rules. '
        'Independent-sample jitter is real, not physical curvature. SVD sign/permutation matching also cannot remove arbitrary rotations in nearly degenerate singular spaces.','',
        '## Test 3: SVD of direction changes in a certified formula cell','',
        'Use known coordinates `q=(top,bottom,intercept,width,slope)`, where:', '',
        '```text\ntop = cy - height/2\nbottom = cy + height/2\nslope = tan(lean in radians)\nintercept = cx + slope*cy\n```','',
        f"Reference is `{cell['reference_settings']}`. Certified coordinate half-spans are `{cell['certified_cell_half_spans']}`. "
        'All 32 corners share the same affine renderer signature. The signature conditions are linear inequalities in these coordinates, '
        'so their convex box stays in this formula cell. This certification uses the known generator.','',
        'Inside a cell the image is a quadratic polynomial and each q-direction is affine. '
        'SVD of the direction changes gives the following calibrated ranks (relative singular cutoff 1e-9):','',
        '| Direction | Rank of its changes |','| --- | ---: |']
    for name,row in cell['second_level_families'].items():
        lines.append(f"| {name} | {row['direction_change_rank']} |")
    lines += ['', f"The joint matrix of all 15 symmetric analytic second derivatives has rank **{cell['joint_interaction_rank']}**. "
        'The per-family numbers and this joint count differ because the families share variation patterns.','',
        'We then recover the polynomial from observations: 256 rendered images, a 21-column design matrix '
        '(constant, five linear terms, fifteen quadratic terms), and an ordinary least-squares solve per pixel. '
        'Analytic derivatives are used only for independent checks and formula-cell certification; they are not copied into the fit.','',
        f"- Worst image L2 error on 512 held-out joint settings: **{cell['quadratic_image']['worst_absolute_L2']:.3g}**.\n"
        f"- Linear-image baseline worst L2 error: **{cell['affine_image']['worst_absolute_L2']:.3g}**.\n"
        f"- Worst relative error across recovered q-direction fields: **{cell['learned_direction_fields']['worst_relative']:.3g}**.\n"
        f"- Worst relative error for the physical-width equation below: **{cell['learned_physical_width_rule']['worst_relative']:.3g}**.", '',
        'The physical-width equation is particularly short:', '',
        '```text\nwidth direction at current state =\n    width direction at reference\n    + (cy - reference cy) * vertical-position/width interaction pattern\n    + (height - reference height) * height/width interaction pattern\n```','',
        'Each term is a visible 784-pixel array. Within this cell, changing current width, cx, or lean does not change the width derivative. '
        'This statement stops at formula boundaries; those knobs can move side edges into other pixel/subrow regimes. '
        'A direction defined as unit length instead has an extra, state-dependent normalization factor.','',
        'The full local update uses the learned polynomial:', '',
        '```text\nimage(q0 + d) = reference image\n    + sum_i d_i * reference direction_i\n    + (1/2) * sum_i,j d_i*d_j * interaction_pattern_i,j\n\ndirection_i(q0 + d) = reference direction_i + sum_j d_j * interaction_pattern_i,j\n```','',
        'The interaction patterns are constant inside the cell. Integrating these calibrated direction fields includes their shared mixed terms '
        'and reconstructs joint changes; freezing five vectors would omit those terms. '
        'Exact cell polynomial structure was established in an earlier analytic experiment; the new result is second-level rank testing '
        'and recovery of the local equation from rendered observations.','',
        '## Expanding past the local rule','',
        '| Expansion of certified coordinate box | States retaining original cell / 512 | Worst width-direction relative error | Worst image L2 error |',
        '| --- | ---: | ---: | ---: |']
    for r in cell['extrapolation']:
        lines.append(f"| {r['box_expansion_factor']} | {r['same_formula_cell_count']} | {r['width_direction']['worst_relative']:.3g} | {r['image']['worst_absolute_L2']:.3g} |")
    lines += ['', '![Second-level SVD and local equation](second_level_svd.png)','',
        '## Interpretation and limits','',
        'The proposed second level is mathematically meaningful. It compresses a vector field by finding shared patterns in how its vectors vary. '
        'There is strong local structure and an explicit affine law for calibrated direction fields inside suitable cells. '
        'Across the full default box, edges move among pixel columns/rows and raw pixel-space directions need many more fixed patterns. '
        'Second-level SVD alone therefore does not yield a tiny global knob equation. '
        'A useful global representation must include state-dependent coefficient rules and the boundaries where those rules change, '
        'or an explicit transformation that follows moving edges. The experiments do not learn such a global switching rule.','',
        'The actual-SVD test covers two one-dimensional paths and must not be read as full-family validation. '
        'The physical-field compression test covers all five controls but has oracle family labels. '
        'The exact local equation uses known q-coordinates and renderer-cell certification. '
        'These are complementary tests, not a complete image-only identification procedure.','',
        '## Reproduce','',
        '```bash\nOPENBLAS_NUM_THREADS=1 .venv/bin/python analyze_generated_one_second_level_svd.py\n```','',
        '- `results.json`: all spectra, held-out errors, coefficient-rule tests, lattice controls, and box expansion results.\n'
        '- `arrays.npz`: sampled settings, direction modes, matched first-level vectors, and fitted polynomial/interaction arrays.\n'
        '- `second_level_svd.png`: source image, physical-width rule, full-box compression, and boundary failure.','']
    (OUT/'README.md').write_text('\n'.join(lines))


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    arrays={};start=time.monotonic()
    results=dict(method=dict(precision='float64 coverage',seed_base=20261017,
        root=ROOT.tolist(),coordinate_order=KNOBS,second_level_centering=True,
        parameter_labels_supplied=True,neural_network=False))
    results['families']=family_experiments(arrays)
    results['estimated']=estimated_svd_experiments(arrays)
    results['cell']=cell_experiments(arrays)
    results['elapsed_seconds']=time.monotonic()-start
    (OUT/'results.json').write_text(json.dumps(results,indent=2,allow_nan=False))
    np.savez_compressed(OUT/'arrays.npz',**arrays)
    figure(results,arrays)
    write_report(results)
    print('Complete in',round(time.monotonic()-start,1),'seconds',flush=True)


if __name__=='__main__':
    main()
