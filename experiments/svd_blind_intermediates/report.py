"""Plain-English report and explanatory figures from frozen results only."""
from pathlib import Path
import json
import os
import csv
os.environ.setdefault('MPLCONFIGDIR','/tmp/svd-blind-figures')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'figures/svd_blind_intermediates'


def main():
    r=json.loads((OUT/'blind_results.json').read_text())
    p=json.loads((OUT/'physical_evaluation.json').read_text())
    o=np.load(OUT/'observations.npz');m=np.load(OUT/'discovered_models.npz');e=np.load(OUT/'evaluation_examples.npz')
    # Descriptive post-freeze diagnostics only; never used to choose/refit a model.
    frequency=np.mean(o['broad_train_x']>0,axis=0)
    rare=frequency<.01
    a=e['broad_svd_mask_a'];constant=e['broad_svd_constant']
    share=np.sum(a[rare]**2,axis=0)/np.sum(a*a,axis=0)
    (OUT/'post_freeze_weight_diagnostic.json').write_text(json.dumps(dict(
        criterion='Pixel has positive ink in fewer than 1 percent of training images',
        squared_pixel_weight_share=share.tolist(),not_used_for_discovery=True),indent=2))
    with (OUT/'broad_svd_functions.csv').open('w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['reading']+[f'pixel_{j}' for j in range(784)]+['constant'])
        for j in range(5):writer.writerow([j+1]+a[:,j].tolist()+[float(constant[j])])
    private=np.load('/tmp/svd_blind_intermediates_labels.npz')
    simple=[]
    for coordinate,target in [(1,3),(2,0)]:
        qt=m['broad_svd_train_q'][:,coordinate]
        coefficient=np.linalg.lstsq(np.column_stack([np.ones(len(qt)),qt]),private['broad_train_p'][:,target],rcond=None)[0]
        pred=coefficient[0]+coefficient[1]*m['broad_svd_test_q'][:,coordinate]
        err=pred-private['broad_test_p'][:,target]
        simple.append(dict(reading=int(coordinate+1),physical_column=int(target),
            constant=float(coefficient[0]),slope=float(coefficient[1]),
            test_rmse=float(np.sqrt(np.mean(err**2))),test_max_error=float(abs(err).max())))
    (OUT/'post_freeze_simple_maps.json').write_text(json.dumps(simple,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    names=['Global PCA baseline','Shuffled families','SVD families','PCA families']
    keys=['global_pc','shuffled','svd','pca']
    colors=['#7f8793','#bab2a4','#2574a9','#c56b35']
    fig=plt.figure(figsize=(15,10.6),facecolor='#fafaf7')
    fig.text(.035,.965,'Can anonymous direction families reveal useful image measurements?',fontsize=19,weight='bold',va='top')
    fig.text(.035,.921,'Discovery stage only: generated 28 × 28 coverage images; 0 = white / no ink, 1 = black / full ink.',fontsize=11)
    top=fig.add_gridspec(1,6,left=.045,right=.965,bottom=.63,top=.87,wspace=.28)
    image=o['broad_test_x'][0]
    ax=fig.add_subplot(top[0,0]);ax.imshow(image.reshape(28,28),cmap='gray_r',vmin=0,vmax=1)
    ax.set_title('Observed image',fontsize=11);ax.set_xticks([]);ax.set_yticks([])
    key='broad_svd';mass=image.sum();ma=e[key+'_mask_a'];mb=e[key+'_mask_b']
    gradients=ma+mb/mass-(image@mb)[None,:]/mass**2
    gradients/=np.maximum(np.linalg.norm(gradients,axis=0),1e-20)
    limit=np.max(abs(gradients))
    for j in range(5):
        ax=fig.add_subplot(top[0,j+1])
        ax.imshow(gradients[:,j].reshape(28,28),cmap='RdBu_r',vmin=-limit,vmax=limit)
        ax.set_title(f'Learned reading {j+1}',fontsize=11);ax.set_xticks([]);ax.set_yticks([])
    fig.text(.045,.593,'Signed measurement weights: adding ink at red pixels increases the reading; blue decreases it.',fontsize=10.5)
    fig.text(.045,.565,'These are reading sensitivities, not image-change vectors. Each map has unit length; all share one color scale.',fontsize=10,color='#515860')
    fig.text(.045,.538,'Large weights concentrate on rarely illuminated pixels. Smaller weights can be invisible at this shared scale.',fontsize=9.5,color='#515860')
    bottom=fig.add_gridspec(1,2,left=.075,right=.965,bottom=.16,top=.465,wspace=.35)
    ax=fig.add_subplot(bottom[0,0]);mtrx=np.array(r['scopes']['broad']['svd']['test_sensitivity'])*100
    im=ax.imshow(mtrx,cmap='Blues',vmin=0,vmax=100,aspect='auto')
    ax.set_xticks(range(5),[f'Reading {j+1}' for j in range(5)])
    ax.set_yticks(range(5),[f'Family {j+1}' for j in range(5)])
    ax.set_title('A · Which family changes each reading?\nShare of squared response on 96 new frames',fontsize=11,pad=12)
    for i in range(5):
        for j in range(5):ax.text(j,i,f'{mtrx[i,j]:.0f}%',ha='center',va='center',color='white' if mtrx[i,j]>55 else '#202630')
    ax.set_xlabel('Ideal separation would put 100% on each diagonal entry',fontsize=9,labelpad=12)
    ax=fig.add_subplot(bottom[0,1]);positions=np.arange(4);width=.34
    for offset,scope,label in [(-width/2,'local','Near existing branches'),(width/2,'broad','Across default range')]:
        vals=[r['scopes'][scope][k]['reconstruction']['test']['mean_l2'] for k in keys]
        bars=ax.bar(positions+offset,vals,width,label=label,color=colors,alpha=1 if scope=='broad' else .48)
        for bar,v in zip(bars,vals):ax.text(bar.get_x()+bar.get_width()/2,v+.024,f'{v:.3f}',ha='center',va='bottom',fontsize=8)
    ax.set_xticks(positions,['Global PCA','Shuffled','SVD families','PCA families'],rotation=12)
    ax.set_ylabel('Mean full-image reconstruction error (L2)')
    ax.set_ylim(0,1.78)
    ax.set_title('B · Do the five readings reconstruct new images?\n512 held-out images per scope; lower is better',fontsize=11,pad=12)
    ax.legend(loc='upper left',fontsize=9,frameon=False)
    fig.text(.045,.075,'Result: family selectivity improves, but reconstruction gets worse than the ordinary global PCA baseline.',fontsize=12,weight='bold')
    fig.text(.045,.042,'No original controls, old shape measurements, or renderer equations entered discovery. No method passed the exact-reconstruction thresholds.',fontsize=9.5)
    fig.savefig(OUT/'blind_discovery.png',dpi=160);plt.close(fig)

    fig=plt.figure(figsize=(14,8.7),facecolor='#fafaf7')
    fig.text(.04,.96,'What did the frozen measurements correspond to?',fontsize=20,weight='bold',va='top')
    fig.text(.04,.904,'Physical evaluation only: original labels were opened after discovery and decoder selection were hash-frozen.',fontsize=11)
    grid=fig.add_gridspec(1,2,left=.075,right=.965,bottom=.26,top=.80,wspace=.34)
    ax=fig.add_subplot(grid[0,0]);q=m['broad_svd_test_q'][:,1]
    # The width values are evaluation labels; discovery has already finished.
    truth=np.load('/tmp/svd_blind_intermediates_labels.npz')['broad_test_p']
    ax.scatter(truth[:,3],q,s=12,alpha=.5,c='#2574a9',edgecolors='none')
    corr=p['scopes']['broad']['svd']['one_to_one_correlations'][1][3]
    ax.set_title(f'A · One reading is strongly width-like\n512 new images; correlation = {corr:.4f}',fontsize=12,pad=15)
    ax.set_xlabel('Original width (pixels; evaluation only)');ax.set_ylabel('Blindly learned reading 2 (arbitrary units)')
    ax=fig.add_subplot(grid[0,1]);x=np.arange(5)
    r2=p['scopes']['broad']['svd']['discovered_to_original']['global_test']['r2']
    bars=ax.bar(x,r2,color=['#9cabb7','#9cabb7','#9cabb7','#2574a9','#9cabb7'])
    for bar,v in zip(bars,r2):ax.text(bar.get_x()+bar.get_width()/2,v+.025,f'{v:.3f}',ha='center')
    ax.axhline(1,color='#627564',ls='--',lw=1)
    ax.set_ylim(0,1.14);ax.set_xticks(x,['Horizontal\ncenter','Vertical\ncenter','Height','Width','Lean'])
    ax.set_ylabel('Held-out R²: 1 = perfect; 0 = predicting the test mean')
    ax.set_title('B · Can all five readings recover the controls?\nOne global quadratic map, selected after freeze',fontsize=12,pad=15)
    fig.text(.045,.145,'Partial discovery, not five recovered knobs: width is strong, while several other controls remain poorly recovered.',fontsize=12,weight='bold')
    fig.text(.045,.096,'R² is a prediction score, not a percentage of exact answers. The five-coordinate image decoder also failed its accuracy tests.',fontsize=10)
    fig.text(.045,.055,'No physical result changed the learned measurements, their weights, their family assignments, or the selected image decoder.',fontsize=10)
    fig.savefig(OUT/'post_freeze_mapping.png',dpi=160);plt.close(fig)

    lines=['# Blind discovery from estimated SVD/PCA families','',
    '**Outcome: partial structure was recovered, but this process did not recover five reliable intermediate controls.**',
    '','One broad-sample SVD measurement was strongly width-like (correlation 0.9949), another had correlation 0.8953 with horizontal position. These meanings were checked only after all discovery outputs were frozen. The five measurements together failed image reconstruction and full original-control recovery.','',
    '## What the learner received','',
    '- Pixel arrays and five estimated direction vectors per state.','- Anonymous initial family assignments copied from the earlier distance/angle branch experiment.','- Five as the requested coordinate count.','',
    'It received no original settings, names, parameter ranges, analytic derivatives, prior edge/center readers, or renderer equations. The acquisition program used simulator controls to generate observations; its ray-sampling distribution is a prior. The designer knows this family, so this is computational information isolation, not human-level blind design.','',
    '## The discovery process','',
    'For each family, search for an image measurement whose response is strong along that family and weak along the other four. Responses come from applying the candidate measurement to the estimated pixel vectors. A generalized eigenproblem chooses the pixel weights. The criterion uses no physical target.','',
    'Candidate functions are weighted pixel readings, or a weighted pixel reading plus another weighted reading divided by total image intensity, plus a constant. This generic intensity normalization is computed from the state. The two signed masks and constant completely specify each function. No row positions, shape landmarks, centers or edge formulas are supplied.','',
    'For the selected broad SVD model, the five readings equal one fixed 5-by-784 matrix applied to the state, plus five constants. The actual coefficients are saved in [broad_svd_functions.csv](broad_svd_functions.csv): each row is one reading, pixels are flattened row by row, and the last column is its constant. These are explicit mathematical functions, although their many coefficients do not constitute a short geometric explanation.','',
    'Generic feature reduction retains 48 or 96 directions, with three regularization levels and three eigenfunction candidates per family. Validation chooses a combination of five using family selectivity and low dependence between readings. The selected broad SVD model uses only weighted pixels; the normalization option was not needed for that result.','',
    'A separately selected local polynomial decoder predicts pixels from these five readings. It retains 8,192 training images, so it is an interpolation control, not a short discovered manifold equation. A global cubic is also tested as a compactness control and fails.','',
    '## Scope','',
    'Two scopes: small fresh joint perturbations around the 66 original branch states; and the broad default family range. Each has 8,192 training images, 512 validation images, 512 new checking images, 256 training frames, 96 validation frames and 96 new checking frames. Each frame uses 192 equal-image-distance observations at radius 0.0001. Both SVD and centered PCA see identical observations. Also check 12 joint paths of 17 images and 132 extra cases per scope, including all 32 corners and 100 face cases in the broad scope. All failures are saved.','',
    '## Did the families help?','',
    '| Scope | Method | Matched-family response share | Mean held-out image L2 error | Worst pixel error |','| --- | --- | ---: | ---: | ---: |']
    for scope in ['local','broad']:
        for name in keys:
            row=r['scopes'][scope][name];err=row['reconstruction']['test']
            lines.append(f"| {scope} | {name} | {100*(1-row['test_mean_leakage']):.1f}% | {err['mean_l2']:.5f} | {err['max_pixel']:.5f} |")
    lines += ['',
    'Response share is the average share of squared directional response attributable to the assigned anonymous family, not physical-knob accuracy. The shuffled control has one fixed randomization; no statistical significance claim is made. Lower image error is better. Errors use actual coverage values and are not clipped.','',
    'The SVD-family readings are more selective than the shuffled control, but less successful at reconstructing states than the five global image PCs. Selectivity alone is therefore an inadequate discovery objective. It can favor measurements that have useful local responses but awkward scale, poor conditioning, or an insufficiently simple relationship to the full state. The observed decoder failure alone does not prove exact global coordinate collisions.','',
    'A post-freeze diagnostic identifies one concrete failure mode: for every broad SVD reading, more than 99.9% of squared pixel-weight magnitude lies on pixels illuminated in fewer than 1% of training images. The pixel maps show these isolated large weights. This is weight concentration, not the fraction of prediction caused by those pixels. Regularization in whitened feature units did not control the resulting raw pixel sensitivities. These are not yet clean, robust intermediate measurements. No refitting followed this diagnosis.','',
    '## What stays consistent in the vector families?','',
    'Applying SVD again to the vectors within each family found shared patterns. Near the original branches, 7–9 patterns per family capture 99% of training vector energy; across the broad sample this rises to 48–90. The associated mean held-out vector projection errors are about 8–9% locally and 13–16% broadly. This is compression, not a discovered state-to-direction law or proof of manifold coordinates.','',
    f"The stored family labels agree with alternative nearby matches on {100*r['scopes']['local']['svd']['family_graph']['non_tree_family_agreement']:.1f}% of comparisons near the branches and {100*r['scopes']['broad']['svd']['family_graph']['non_tree_family_agreement']:.1f}% broadly. This is a self-consistency check, not correctness against original controls. Thus the initial family concept itself is imperfect when extended. Centered PCA did not solve this problem.",
    '', '## Mapping to original controls: checked after freeze','',
    'The evaluator verified hashes before first opening the private labels and verified them again afterward. It fitted linear, quadratic and cubic maps using training labels, selected degree on validation labels, and checked new labels. It also tested a less compact local quadratic mapping. None changed discovery.','',
    f"A simple post-freeze one-reading map is: width in pixels approximately equals {simple[0]['constant']:.6f} plus {simple[0]['slope']:.6f} times reading 2. Its held-out RMS error is {simple[0]['test_rmse']:.4f} pixels, with worst error {simple[0]['test_max_error']:.4f} pixels. This is a genuine approximate mapping, not exact recovery.",'',
    '| Broad SVD readings → original control | Global quadratic R² | Local quadratic R² | Global RMSE |','| --- | ---: | ---: | ---: |']
    row=p['scopes']['broad']['svd']['discovered_to_original']
    labels=['Horizontal center','Vertical center','Height','Width','Lean (degrees)']
    for i,label in enumerate(labels):
        lines.append(f"| {label} | {row['global_test']['r2'][i]:.4f} | {row['local_quadratic_test']['r2'][i]:.4f} | {row['global_test']['rmse'][i]:.5f} |")
    lines += ['',
    'R² is explained variation on new labels, not an exact-answer percentage. Width is strongly recoverable, but five complete intermediate controls were not found. The ordinary global PCA baseline performs substantially better within the branch region, which prevents attributing every successful mapping to family grouping.','',
    '## What this establishes and the next change to test','',
    'There is a general, implementable starting process: use each family to learn functions that remain comparatively unchanged under the other families. It produced a genuine partial discovery without supplied geometric readers. It is not yet a general method for recovering five independent controls.','',
    'The next justified change is to make preservation of all state information and local conditioning mandatory during coordinate selection, penalize large sensitivities in raw pixel units, and use family selectivity as a secondary preference. In parallel, test family consistency more strongly before requiring one scalar per family. This follow-up has not been run or tuned using the revealed physical labels.','',
    'No method passed the fixed numerical reconstruction thresholds (maximum pixel error 2e-6 and maximum image L2 error 1e-5). No globally exact five-function description, unique physical-knob discovery, or real-data success rate is claimed.','',
    '![Blind discovery and reconstruction](blind_discovery.png)','',
    '![Original-control evaluation after freezing](post_freeze_mapping.png)','',
    '## Evidence and reproduction','',
    '- `FREEZE.json`: hashes of the discovery source, protocol, observations, models and blind results.','- `EVALUATION_ORDER.json`: separation of discovery and physical evaluation.','- `verification.json`: hash, forbidden-read, gradient and saved-function checks.','- `blind_results.json`: every candidate selection and all blind held-out checks.','- `physical_evaluation.json`: mappings evaluated only after freeze.','- `discovered_models.npz`: actual coordinate functions and saved predictions.','- `evaluation_examples.npz`: explicit pixel masks and post-freeze predictions.','',
    'Sources are in `experiments/svd_blind_intermediates/`: run `export.py`, `discover.py`, `evaluate.py`, `verify.py`, then `report.py` with the existing Python environment. The physical-label archive is deliberately not part of the discovery inputs. All artifact paths are repository relative.','']
    (OUT/'README.md').write_text('\n'.join(lines))
    print('Saved English report and two figures.')


if __name__=='__main__':main()
