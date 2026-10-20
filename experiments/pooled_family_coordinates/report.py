"""Explain pooled perpendicular constraints, their tolerance, and held-out results."""
from pathlib import Path
import os
import json
import csv
os.environ.setdefault('MPLCONFIGDIR','/tmp/pooled-family-figures')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import LocalPolynomial,design,powers
from saved import feature_from_archive

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'figures/pooled_family_coordinates'


def rotations(response):
    coefficient=np.linalg.pinv(np.swapaxes(response,1,2),rcond=1e-8)
    length=np.linalg.norm(coefficient,axis=1)
    angle=np.rad2deg(np.arccos(np.clip(np.diagonal(coefficient,axis1=1,axis2=2)/np.maximum(length,1e-30),-1,1)))
    return dict(median_degrees=float(np.median(angle)),p95_degrees=float(np.quantile(angle,.95)),
                within_10_degrees=float(np.mean(angle<=10)),within_20_degrees=float(np.mean(angle<=20)))


def main():
    r=json.loads((OUT/'blind_results.json').read_text())
    b=json.loads((OUT/'balanced_results.json').read_text())
    p=json.loads((OUT/'physical_evaluation.json').read_text())
    quality=json.loads((OUT/'stationary_quality.json').read_text())
    verification=json.loads((OUT/'verification.json').read_text())
    old=np.load(ROOT/'figures/svd_blind_intermediates/observations.npz')
    new=np.load(OUT/'patch_observations.npz');models=np.load(OUT/'models.npz');balanced=np.load(OUT/'balanced_models.npz')
    rotate={key.removesuffix('_test_response'):rotations(balanced[key]) for key in balanced.files if key.endswith('_test_response')}
    (OUT/'balanced_rotation_diagnostic.json').write_text(json.dumps(rotate,indent=2))
    # Export mathematical coefficients after freeze; this does not refit discovery.
    function_dir=OUT/'functions';function_dir.mkdir(exist_ok=True)
    physical_labels=np.load('/tmp/pooled_family_coordinates_labels.npz')
    for si in range(2):
        for ri in [0,2]:
            scope=f'patch_{si}_{ri}';prefix=scope+'_svd_fixed_joint'
            feature=feature_from_archive(balanced,prefix);c=balanced[prefix+'_coefficient']
            matrix=np.zeros((5,784));matrix[:,feature.active]=c.T
            offset=balanced[prefix+'_center']-feature.mean@c
            with (function_dir/(scope+'_readings.csv')).open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['reading']+[f'pixel_{j}' for j in range(784)]+['constant'])
                for j in range(5):writer.writerow([j+1]+matrix[j].tolist()+[float(offset[j])])
            q=balanced[prefix+'_train_q'];mu=q.mean(0);scale=q.std(0)
            degree=p['runs']['balanced'][prefix]['selected']['degree']
            coefficient=np.linalg.lstsq(design((q-mu)/scale,degree),physical_labels[scope+'_train_p'],rcond=1e-10)[0]
            (function_dir/(scope+'_post_freeze_map.json')).write_text(json.dumps(dict(
                formula='Normalize the five readings, then evaluate the polynomial terms with these coefficients.',
                reading_mean=mu.tolist(),reading_scale=scale.tolist(),degree=degree,
                terms_zero_based=[list(t) for t in powers(5,degree)],
                coefficient_columns=p['physical_names'],coefficients=coefficient.tolist(),
                fitted_only_after_discovery_freeze=True),indent=2))
            check=new[scope+'_test_x']@matrix.T+offset
            assert np.max(abs(check-balanced[prefix+'_test_q']))<1e-8
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    radii=np.array([.005,.05,.25,1.])
    fig=plt.figure(figsize=(14,10.5),facecolor='#fafaf7')
    fig.text(.04,.965,'Pooling whole neighborhoods: how exclusive can a measurement be?',fontsize=19,weight='bold',va='top')
    fig.text(.04,.918,'Generated 28 × 28 coverage images: white = no ink; black = full ink. Five anonymous SVD families, no physical targets.',fontsize=10.5)
    grid=fig.add_gridspec(2,2,left=.06,right=.965,bottom=.225,top=.865,width_ratios=[1,4],wspace=.30,hspace=.46)
    maximum=0.
    for si in range(2):
        for group in ['fixed_joint','quadratic']:
            maximum=max(maximum,max(r['scopes'][f'patch_{si}_{j}']['svd']['methods'][group]['test_response']['off_rms'] for j in range(4)))
            maximum=max(maximum,max(b[f'patch_{si}_{j}']['svd'][group]['test_response']['off_rms'] for j in [0,2]))
    for si in range(2):
        ax=fig.add_subplot(grid[si,0])
        ax.imshow(new[f'patch_{si}_0_source_x'][0].reshape(28,28),cmap='gray_r',vmin=0,vmax=1)
        ax.set_xticks([]);ax.set_yticks([]);ax.set_title('Root start' if si==0 else 'Second start',fontsize=12,pad=10)
        ax.text(.5,-.12,'Same source at every radius',transform=ax.transAxes,ha='center',fontsize=9)
        ax=fig.add_subplot(grid[si,1])
        for group,color,label in [('fixed_joint','#2974a4','Fixed pixel weights'),('quadratic','#c46b35','State-dependent quadratic')]:
            y=np.array([r['scopes'][f'patch_{si}_{j}']['svd']['methods'][group]['test_response']['off_rms'] for j in range(4)])*100
            ax.plot(radii,y,'o--',color=color,alpha=.55,label=label+' · 192 random neighbors',lw=1.5)
            y2=np.array([b[f'patch_{si}_{j}']['svd'][group]['test_response']['off_rms'] for j in [0,2]])*100
            ax.plot(radii[[0,2]],y2,'D-',color=color,label=label+' · 512 balanced neighbors',lw=2.5)
        ax.axhline(5,color='#8e969f',lw=1,ls=':');ax.axhline(10,color='#535f68',lw=1,ls='--')
        ax.text(.97,5,'5%',transform=ax.get_yaxis_transform(),ha='right',va='bottom',fontsize=8)
        ax.text(.97,10,'10%',transform=ax.get_yaxis_transform(),ha='right',va='bottom',fontsize=8)
        ax.set_xscale('log');ax.set_xticks(radii,[str(a) for a in radii]);ax.set_ylim(0,max(40,maximum*115))
        ax.set_ylabel('Unwanted response (% of own unit)')
        ax.set_xlabel('Maximum image distance from source (pixel-value L2)')
        ax.set_title('32 new frames per mark; average over five readings and the other four families',fontsize=10,pad=12)
        if si==0:handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.54,.095),ncol=2,fontsize=9.5,frameon=False)
    fig.text(.34,.165,'The fixed and quadratic curves with 512 balanced neighbors nearly overlap.',fontsize=9.5,color='#515860')
    fig.text(.04,.068,'Smaller is better. Percentages are RMS directional responses, not variance retained or a bound on every individual case.',fontsize=10)
    fig.text(.04,.034,'Input quality matters: even an unchanged image had noisy axes. The 512-neighbor comparison changes both sample count and sampling design.',fontsize=9.5)
    fig.savefig(OUT/'pooled_tolerance.png',dpi=160);plt.close(fig)

    groups=['global_pc','fixed_joint','quadratic']
    labels=['Global PCA baseline','Pooled fixed weights','State-dependent quadratic']
    colors=['#9099a3','#2974a4','#c46b35']
    truth=old['broad_test_x']
    errors=np.linalg.norm(models['broad_svd_quadratic_test_example_prediction']-truth[:32],axis=1)
    example=int(np.argmax(errors));image=truth[example]
    predictions=[];examples=[]
    for group in groups:
        row=r['scopes']['broad']['svd']['methods'][group]
        choice=row['decoder_selected'];prefix='broad_svd_'+group
        decoder=LocalPolynomial(models[prefix+'_train_q'],old['broad_train_x'],choice['degree'],choice['neighbors'])
        pred=decoder.predict(models[prefix+'_test_q'][example:example+1])[0]
        predictions.append(pred);examples.append(float(np.linalg.norm(pred-image)))
    from experiments.derived_state_coordinates import recover,coverage
    reversed_image=coverage(recover(image.reshape(1,28,28))[1]).ravel()
    shown=[image,*predictions,reversed_image]
    shown_labels=['Observed image',*labels,'Known reverse-derived rule']
    fig=plt.figure(figsize=(15,11.3),facecolor='#fafaf7')
    fig.text(.035,.965,'Do the five learned readings actually describe the images?',fontsize=19,weight='bold',va='top')
    fig.text(.035,.922,'Broad held-out sample. Top example chosen as the largest quadratic error among the first 32 checking images.',fontsize=10.5)
    top=fig.add_gridspec(2,5,left=.045,right=.965,bottom=.48,top=.865,hspace=.45,wspace=.28)
    limit=max(float(np.max(abs(a-image))) for a in shown)
    for j,(a,label) in enumerate(zip(shown,shown_labels)):
        ax=fig.add_subplot(top[0,j]);ax.imshow(a.reshape(28,28),cmap='gray_r',vmin=0,vmax=1)
        ax.set_title(label,fontsize=10,pad=10);ax.set_xticks([]);ax.set_yticks([])
        ax.text(.5,-.13,f'Image L2 error {np.linalg.norm(a-image):.3g}',transform=ax.transAxes,ha='center',fontsize=9)
        ax=fig.add_subplot(top[1,j]);ax.imshow((a-image).reshape(28,28),cmap='RdBu_r',vmin=-limit,vmax=limit)
        ax.set_xticks([]);ax.set_yticks([])
        ax.text(.5,-.10,f'Max pixel error {np.max(abs(a-image)):.3g}',transform=ax.transAxes,ha='center',fontsize=9)
    fig.text(.045,.442,'Top: black = full ink, white = none; predicted values outside [0,1] saturate visually. All errors use unclipped predictions.',fontsize=9.5)
    fig.text(.045,.417,f'Bottom: prediction minus observed image. Red = extra predicted ink, blue = missing ink; common scale ±{limit:.3g}.',fontsize=9.5)
    bottom=fig.add_gridspec(1,2,left=.085,right=.965,bottom=.13,top=.345,wspace=.27)
    ax=fig.add_subplot(bottom[0,0])
    values=[r['scopes']['broad']['svd']['methods'][g]['reconstruction']['test']['mean_l2'] for g in groups]
    bars=ax.bar(np.arange(3),values,color=colors)
    for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v+.01,f'{v:.3f}',ha='center')
    ax.set_xticks(range(3),['Global PCA','Pooled fixed','Quadratic']);ax.set_ylim(0,max(values)*1.25)
    ax.set_ylabel('Mean image L2 error');ax.set_title('All 512 broad checking images · lower is better',fontsize=11,pad=12)
    ax=fig.add_subplot(bottom[0,1]);position=np.arange(5);width=.24
    allvalues=[]
    for j,(group,color,label) in enumerate(zip(groups,colors,labels)):
        values=p['runs']['primary']['broad_svd_'+group]['selected']['test']['r2'];allvalues+=values
        ax.bar(position+(j-1)*width,values,width,color=color,label=label)
    ax.axhline(1,color='#466151',ls='--',lw=1)
    ax.set_xticks(position,['Horizontal\ncenter','Vertical\ncenter','Height','Width','Lean'])
    ax.set_ylim(min(-.05,min(allvalues)-.05),1.12);ax.set_ylabel('Original-control readout R²')
    ax.set_title('Original labels used only after both discovery freezes',fontsize=11,pad=12)
    handles,legend_labels=ax.get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='center',bbox_to_anchor=(.53,.38),ncol=3,frameon=False,fontsize=9)
    fig.text(.045,.055,'R² is prediction quality, not an exact-answer percentage. Reverse-derived functions reconstruct exactly here but were never fitting targets.',fontsize=10)
    fig.savefig(OUT/'coordinate_usefulness.png',dpi=160);plt.close(fig)

    lines=['# Pooled perpendicular readings with explicit tolerance','',
    '**Result: pooling is straightforward linear algebra and works locally with sufficiently stable inputs. A fixed perpendicular vector is not guaranteed to exist even when a nonlinear coordinate does. The broad five-knob family remains unsolved by this discovery procedure.**','',
    '## What was tested','',
    'The measurement is a scalar function of the 784 image values. Its gradient, not the movement vector itself, should be perpendicular to the other families. Stack all fitting-state constraints together: one calibrated mean response to the target family and small responses to the other four. This solves the whole area jointly, not one vector at a time. Zero or unresponsive readings are excluded.','',
    'Two fixed-weight methods were compared: a minimum-gradient-norm common-nullspace solve with explicit RMS bounds; and a pooled derivative least-squares fit that also controls variation of the own-family response. State-dependent alternatives fit scalar quadratic, cubic, and Gaussian-kernel functions to the same derivative constraints. These are gradient-informed fits with explicit scalar potentials; we do not assign unrelated gradients independently at every image.','',
    'All data-varying raw pixel columns are kept in the linear part. Nonlinear features are generic functions of 8 or 12 image PCs. Penalties measure gradient size in original pixel units. No old edge/center/width readers, original coordinate values, analytic physical directions, or rendering formulas enter fitting or selection. The original controls still generate observations, so acquisition has a simulator sampling prior.','',
    'Method references: [least-squares mathematics](https://www.seas.ucla.edu/~vandenbe/133A/lectures/ls.pdf) and [gradient-augmented approximation](https://arxiv.org/abs/1712.06645). Our implementation uses regularized derivative least squares, not the latter paper\'s sparse recovery algorithm.','',
    '## What tolerance means','',
    'The explicit requested bounds are 1%, 3%, 5%, 10%, 20%, 30%, 50%, and 100% RMS unwanted response, relative to a mean own-family response normalized to one. The bound is imposed on the fitting frames. Validation and new-frame errors are measured independently; an infeasible bound is reported as infeasible. An average bound is not a worst-state guarantee. This is response tolerance, not retained variance or an image reconstruction tolerance.','',
    'Relaxation is encoded directly in the spectral/constrained solve. No manual nudging of pixels or directions is needed. A separate diagnostic computes the state-specific mixing of the five supplied vectors needed for perfectly separated responses, and records its rotation angles and speed changes. That algebraic correction alone is not a global-coordinate certificate.','',
    '## Curved-coordinate control','',
    'Consider a two-number state with x = u and y = v + u squared. The coordinate v equals y minus x squared. Its measurement gradient is (-2x, 1), which changes with the state. No nonzero fixed vector is perpendicular to all u-directions (1, 2u) while also measuring v. The fitted quadratic reading recovers this rule to numerical precision; the fixed readout retains 116.6% RMS unwanted response after unit calibration.','',
    'This is a direct example where the coordinate exists but a search for one constant perpendicular vector cannot find it. It is a separate mathematical control, not a feature supplied to the image learner.','',
    '## The input directions themselves have sampling noise','',
    'At each of two unchanged source images, compare repeated independent neighborhood fits. Pool the largest balanced-neighborhood covariances for reference axes, without physical labels.','',
    '| Acquisition | Root median axis rotation | Second-start median axis rotation | Root unwanted response | Second-start unwanted response |','| --- | ---: | ---: | ---: | ---: |']
    for k,label in [('gaussian_192','192 random neighbors'),('sobol_512','512 balanced neighbors'),('sobol_2048','2048 balanced neighbors')]:
        a=quality['0']['svd'][k];c=quality['1']['svd'][k]
        lines.append(f"| {label} | {a['median_angle_degrees']:.2f}° | {c['median_angle_degrees']:.2f}° | {100*a['off_rms']:.2f}% | {100*c['off_rms']:.2f}% |")
    lines += ['',
    'The fitted five-dimensional spaces are much more stable than individual axes (mean projection residuals about 1e-6 or less). Thus family-axis wobble is present even when the state does not change. The first comparison changes both sample count and sampling design; 512 versus 2048 keeps the balanced design. These are finite-sample controls, not universal noise bounds.','',
    'The follow-up re-estimates frames at the exact same patch states using 512 balanced neighbors and repeats fitting without changing the image sets. It was triggered by blind response scatter, before physical evaluation. Both runs are kept and separately frozen.','',
    '## Patch results with cleaner inputs','',
    '| Source | Maximum image distance | Method | Unwanted response | Own-response variation | Mean image L2 error | Median correction angle |','| --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for si in range(2):
        for ri in [0,2]:
            scope=f'patch_{si}_{ri}'
            for group,label in [('fixed_joint','Fixed pooled'),('quadratic','State-dependent quadratic')]:
                row=b[scope]['svd'][group];key=scope+'_svd_'+group
                lines.append(f"| {si} | {radii[ri]} | {label} | {100*row['test_response']['off_rms']:.2f}% | {100*row['test_response']['own_rms_variation']:.2f}% | {row['reconstruction']['test']['mean_l2']:.3g} | {rotate[key]['median_degrees']:.2f}° |")
    lines += ['',
    'Each mark uses 32 new frames, with all five readings checked against the four non-target families. Image reconstruction is checked on 256 different images. Mean image error and family selectivity are separate criteria. Small response error does not itself prove a one-to-one global representation.','',
    '## Broad-family results','',
    '| Method | Unwanted response | Own-response variation | Mean held-out image L2 error | Worst pixel error |','| --- | ---: | ---: | ---: | ---: |']
    for group,label in [('global_pc','Five global PCs'),('pooled_constrained','Common-nullspace constraint'),('fixed_joint','Fixed derivative fit'),('quadratic','Quadratic derivative fit'),('cubic','Cubic derivative fit'),('kernel','Kernel derivative fit')]:
        row=r['scopes']['broad']['svd']['methods'][group]
        lines.append(f"| {label} | {100*row['test_response']['off_rms']:.1f}% | {100*row['test_response']['own_rms_variation']:.1f}% | {row['reconstruction']['test']['mean_l2']:.4f} | {row['reconstruction']['test']['max_pixel']:.4f} |")
    lines += ['',
    'The broad quadratic readout improves mean image reconstruction over the global-PC baseline, but remains an approximation. Own-family response and direction identity vary substantially. The kernel method has a severe instability in the branch scope: its held-out gradient and response norms explode despite good-looking image interpolation. This failure is retained; smooth scalar functions and a training gradient penalty alone do not guarantee stable extrapolation.','',
    'Ordinary PCA and SVD were both tested throughout. The files retain every result, threshold, regularization candidate and validation choice. No general PCA advantage is established.','',
    '## Comparison to the reverse-derived coordinates','',
    'Original settings and known image readers were evaluated only after both discovery freezes verified. Maps from discovered readings to original controls used training labels and validation degree selection, then untouched checking labels.','',
    '| Broad readout | Horizontal center R² | Vertical center R² | Height R² | Width R² | Lean R² |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for group,label in [('global_pc','Global PCs'),('fixed_joint','Fixed pooled'),('quadratic','Quadratic pooled')]:
        scores=p['runs']['primary']['broad_svd_'+group]['selected']['test']['r2']
        lines.append('| '+label+' | '+' | '.join(f'{v:.4f}' for v in scores)+' |')
    lines += ['',
    'R² is explained variation on new labels, not a percentage of exact answers. Known reverse-derived readers reconstruct this family to numerical precision. Their responses to the supplied SVD axes need not be exclusive: those axes can mix physical controls. On actual original physical directions, the known readers have the identity response matrix. This distinction prevents treating imperfect raw-family exclusivity as proof that a valid intermediate coordinate does not exist.','',
    'The cleaner local fixed readings also map very accurately to all five original controls. These maps were fitted only after the readings were frozen:','',
    '| Source | Radius | Lowest R² among five controls | Largest pixel-control RMS error | Lean RMS error (degrees) |','| --- | ---: | ---: | ---: | ---: |']
    for si in range(2):
        for ri in [0,2]:
            scope=f'patch_{si}_{ri}';a=p['runs']['balanced'][scope+'_svd_fixed_joint']['selected']['test']
            lines.append(f"| {si} | {radii[ri]} | {min(a['r2']):.10f} | {max(a['rmse'][:4]):.3g} | {a['rmse'][4]:.3g} |")
    lines += ['',
    'The four pixel controls are horizontal center, vertical center, height and width. The local discovered readings are explicit fixed linear functions: a learned 5-by-784 matrix applied to the state plus five constants. The original settings are then approximated by a selected cubic polynomial of those five readings. They are useful local intermediate coordinates, not individually pure original knobs.','',
    'Actual coefficients are in `functions/`: each `patch_*_readings.csv` contains five pixel-weight rows plus offsets; each `patch_*_post_freeze_map.json` specifies the normalization, polynomial terms and original-control coefficients. No previous shape-reader formula enters those measurements.','',
    '## Scope, integrity, and limits','',
    'The primary comparison uses two existing scopes and eight new image-distance balls around two starts. Each new patch has 1,536 training images, 256 validation images, 256 checking images, and 96/32/32 separate frame states. The primary frame fit has 192 neighbors at radius 0.0001. New frame states are independent; the primary acquisition reuses its random-ray design by frame index across splits. The balanced comparison uses independent scrambled designs across splits. Root patches are two-sided and can extend beyond the default position bounds.','',
    'Discovery selection uses validation image reconstruction only; directional errors and conditioning are reported separately, including cases that fail. Each final decoder is selected among five fixed local-polynomial choices and retains its training examples. It is an interpolation diagnostic, not a compact exact image equation.','',
    'The scalar readings have route-independent values by construction. Closed-triangle gradient quadrature is a numerical consistency check in ambient image space, not a test that the original vector-family flows commute. The rotation diagnostic remains pointwise. No globally covering atlas or globally exact discovery is claimed.','',
    'The broad-family comparison retains the 192-random-neighbor inputs. The cleaner 512-neighbor repeat covers four patches only. A broad repeat with cleaner frames remains untested, so this run does not separate all global geometric limitations from acquisition error.','',
    f"Frozen hashes, blocked-read checks and saved-function response spectra pass. Finite-difference derivative checks report {len(verification['finite_difference_failures'])} numerical failures out of {len(verification['finite_difference_checks'])}; the individual results remain in verification.json.",'',
    'The useful next target is stable local scalar coordinates with explicit overlap transitions, cleaner direction estimates, and reconstruction constraints. The tests do not support requiring one fixed global perpendicular pixel vector per raw SVD family.','',
    '![Pooled tolerance across neighborhoods](pooled_tolerance.png)','',
    '![Coordinate usefulness](coordinate_usefulness.png)','',
    '## Saved evidence','',
    '- `PROTOCOL.md` and `QUALITY_CONTROL.md` in the experiment source directory: the pre-fit contracts.','- `FREEZE.json` and `BALANCED_FREEZE.json`: source/input/model hashes before physical evaluation.','- `blind_results.json` and `balanced_results.json`: every blind attempt and selected model.','- `stationary_quality.json`: repeated fits at unchanged images.','- `blind_rotation_diagnostic.json` and `balanced_rotation_diagnostic.json`: direction mixing required for exclusive readings.','- `physical_evaluation.json`: post-freeze physical maps and known-reader controls.','- `models.npz` and `balanced_models.npz`: fixed coefficients for the scalar functions.','- `verification.json`: saved-function and finite-difference checks.','',
    'Sources are under `experiments/pooled_family_coordinates/`. All document and artifact paths are repository relative.','']
    (OUT/'README.md').write_text('\n'.join(lines))
    print('Saved two explanatory figures and English report.')


if __name__=='__main__':
    import sys
    sys.path.insert(0,str(ROOT))
    main()
