"""Centered PCA of fixed-width lean paths and their pooled width/lean slice.
Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python analyze_generated_lean_width_pca.py
"""
import inspect
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import grey_ones
from make_generated_lean_width_comparison import BASE, LO, HI, sample

OUT = Path('figures/generated_lean_width_comparison/width_sweep/pca')
# Same arithmetic and subrows as the actual renderer; preserve its intermediate
# float64 image instead of rounding to float32 to diagnose tiny PCA components.
ns = dict(vars(grey_ones))
exec(inspect.getsource(grey_ones.render).replace('np.float32)', 'np.float64)'), ns)
render64 = ns['render']

def imgs(width, lean):
    p = np.tile(BASE, (len(np.atleast_1d(lean)), 1))
    p[:, 3], p[:, 4] = width, lean
    x = render64(p).reshape(-1, 784)
    assert np.array_equal(x.astype(np.float32), grey_ones.render(p).reshape(-1, 784))
    return x

def pca(x, label):
    active = np.ptp(x, axis=0) > 0
    z = x[:, active] - x[:, active].mean(axis=0)
    _, s, vt = np.linalg.svd(z, full_matrices=False)
    energy = s*s
    cumulative = np.cumsum(energy) / energy.sum()
    ranks = {str(t): int(np.sum(s > s[0]*t)) for t in [1e-6,1e-8,1e-10,1e-12]}
    r = ranks['1e-10']
    def count(frac):
        return int(np.searchsorted(cumulative, frac)+1)
    sf = np.linalg.svd(x.astype(np.float32).astype(float)[:, active] - x.astype(np.float32).astype(float)[:, active].mean(axis=0), compute_uv=False)
    result = dict(label=label, points=len(x), varying_pixels=int(active.sum()), components_95=count(.95), components_99=count(.99), components_9999=count(.9999), rank_tolerance_sweep=ranks, float32_default_numerical_rank=int(np.sum(sf>sf[0]*max(z.shape)*np.finfo(float).eps)), singular_values=s.tolist(), cumulative_variance=cumulative.tolist(), residual_fraction_at_rank=float(np.sum(energy[r:])/energy.sum()))
    return result, (active, x[:,active].mean(axis=0), vt)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    widths=np.round(BASE[3]+np.arange(11)*.1,10)
    lean=np.linspace(LO,HI,901)
    dense=[imgs(w,lean) for w in widths]
    rows=[]; sparse=[]
    for w,x in zip(widths,dense):
        a,_=pca(x,f'width {w:.1f}'); a['width']=float(w); rows.append(a)
        knots,_,_=sample(w,3.)
        a,_=pca(imgs(w,knots),f'3-unit samples, width {w:.1f}'); a['width']=float(w); sparse.append(a)
    pooled,basis=pca(np.concatenate(dense),'All widths pooled: original images')
    aligned,_=pca(np.concatenate([x-imgs(w,[0])[0] for w,x in zip(widths,dense)]),'All widths pooled: subtract each upright')
    # Held-out lean AND width values, reconstructed using pooled original-image PCs.
    test=np.concatenate([imgs(w,np.arange(LO+.025,HI,.05)) for w in np.arange(3.25,4.2,.1)])
    active,mean,vt=basis
    z=test[:,active]-mean
    checks={}
    for k in sorted(set([pooled['components_95'],pooled['components_9999'],pooled['rank_tolerance_sweep']['1e-10']])):
        error=np.linalg.norm(z-(z@vt[:k].T)@vt[:k],axis=1)
        checks[str(k)]={'median_image_error':float(np.median(error)),'max_image_error':float(error.max()),'fraction_of_test_centered_energy_lost':float(np.sum(error**2)/np.sum(z**2))}
    # Density check for sampled rank stability at representative widths and pooled.
    finer=[imgs(w,np.linspace(LO,HI,1801)) for w in widths]
    stability=[]
    for i in [0,5,10]:
        a,_=pca(finer[i],f'double lean density width {widths[i]:.1f}');stability.append(a)
    a,_=pca(np.concatenate(finer),'double lean density pooled');stability.append(a)
    results=dict(fixed=dict(center=BASE[:2].tolist(),height=float(BASE[2]),lean_range=[float(LO),float(HI)],widths=widths.tolist(),dense_lean_step_degrees=.05),method='Mean-centered, unstandardized 784 coverage pixels; float64 renderer arithmetic with unchanged 16 subrows. Exact cast agreement with native float32 renderer asserted.',rank_definition='Numerical affine rank: singular value > largest singular value times 1e-10. Not a proof of exact continuum rank.',per_width=rows,three_unit_samples=sparse,pooled=pooled,pooled_upright_aligned=aligned,density_check=stability,held_out_pooled_reconstruction=checks)
    (OUT/'results.json').write_text(json.dumps(results,indent=2))
    lines=['# PCA of width-dependent lean paths','',results['method'],'','Each dense path has 901 equally weighted lean samples from -10 to +35 degrees. Eleven widths from 3.2 to 4.2 in steps of 0.1 are pooled with equal weight. Center (14.5,14.5) and height 19.75 are fixed.','', '“100%” below means numerical affine rank at relative singular-value tolerance 1e-10, not an exact continuum proof. See JSON for tolerance sweeps, native float32 rounding ranks, and doubled-density checks.','', '| Data | Points | 95% | 99.99% | Numerical 100% |','| --- | ---: | ---: | ---: | ---: |']
    for a in rows+[pooled,aligned]:
        lines.append(f"| {a['label']} | {a['points']} | {a['components_95']} | {a['components_9999']} | {a['rank_tolerance_sweep']['1e-10']} |")
    lines += ['', '## Original 3-unit chord samples', '', '| Width | Points | 95% | Numerical 100% |','| --- | ---: | ---: | ---: |']
    for a in sparse:lines.append(f"| {a['width']:.1f} | {a['points']} | {a['components_95']} | {a['rank_tolerance_sweep']['1e-10']} |")
    lines += ['', 'Sparse ranks describe only the listed points, not the continuous lean path. Subtracting a single upright image before centering an individual path leaves its PCA unchanged; per-width subtraction changes the pooled data.', '', '## Held-out pooled reconstruction', '', 'These images use intermediate widths and lean values absent from training. Errors are L2 in all 784 pixel coverage values.', '', '| Components | Median error | Maximum error | Test energy lost |','| --- | ---: | ---: | ---: |']
    for k,v in checks.items():lines.append(f"| {k} | {v['median_image_error']:.6g} | {v['max_image_error']:.6g} | {100*v['fraction_of_test_centered_energy_lost']:.6g}% |")
    lines += ['', 'PCA dimension measures a global flat subspace. A fixed-width path still has one varying knob; the pooled width/lean slice has two. No full five-knob-family claim is made.', '', '![PCA results](pca_summary.png)', '', 'Reproduce: `OPENBLAS_NUM_THREADS=1 .venv/bin/python analyze_generated_lean_width_pca.py`.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(14,8))
    fig.text(.06,.95,'How many flat PCA directions describe the lean paths?',fontsize=19,weight='bold')
    fig.text(.06,.90,'Generated 28 × 28 images: coverage 0 = white, 1 = black. Width changes from 3.2 to 4.2; lean runs −10° to +35°.\nCenter and height stay fixed. PCA uses all 784 pixels, subtracts the mean, and does not rescale individual pixels.',fontsize=11)
    for j,(w,l) in enumerate([(3.2,0),(3.2,25),(4.2,25)]):
        ax=fig.add_axes([.08+j*.14,.72,.09,.12]);ax.imshow(imgs(w,[l])[0].reshape(28,28),cmap='gray_r',vmin=0,vmax=1);ax.axis('off');ax.set_title(f'width {w}, lean {l}°',fontsize=10)
    fig.text(.55,.765,f"Each path: 901 samples; pooled: {pooled['points']:,} images.\nPooled: {pooled['components_95']} PCs for 95%; {pooled['rank_tolerance_sweep']['1e-10']} for numerical 100%.\nThis is a sampled two-knob slice, not the full five-knob family.",fontsize=12)
    ax=fig.add_axes([.07,.23,.39,.39])
    ax.plot(widths,[a['components_95'] for a in rows],'-o',label='95% variance',color='#23689c')
    ax.plot(widths,[a['components_9999'] for a in rows],'-s',label='99.99% variance',color='#b57920')
    ax.plot(widths,[a['rank_tolerance_sweep']['1e-10'] for a in rows],'-^',label='Numerical 100% (rank)',color='#733da0')
    ax.set(xlabel='Width knob value',ylabel='Number of PCA components',title='Separate PCA for each fixed-width lean path');ax.grid(alpha=.2);ax.legend(fontsize=10)
    ax=fig.add_axes([.57,.23,.36,.39])
    for a,col in [(rows[0],'#23689c'),(rows[-1],'#b57920'),(pooled,'#24282c'),(aligned,'#43865b')]:
        cum=np.array(a['cumulative_variance']);r=a['rank_tolerance_sweep']['1e-10'];res=1-cum[:r];ax.semilogy(np.arange(1,r+1),np.maximum(res,1e-16),label=a['label'],color=col)
    ax.axhline(.05,color='#888',ls=':',label='95% retained: 5% left')
    ax.set(xlabel='Number of PCA components',ylabel='Fraction of variance left out',title='Pooled paths need a larger flat subspace',ylim=(1e-14,1));ax.grid(alpha=.2);ax.legend(fontsize=9)
    fig.text(.06,.075,'Numerical 100%: keep singular values > 10⁻¹⁰ × the largest; smaller values treated as rounding noise.\nFloat64 preserves the renderer’s intermediate coverage; native float32 rounding otherwise adds tiny spurious components.\nRight: the plotted residual is clipped below 10⁻¹⁶. Variance retention does not guarantee every image or direction is accurate.',fontsize=10)
    fig.savefig(OUT/'pca_summary.png',dpi=160);plt.close(fig)
    print(json.dumps({'per_width':[(a['width'],a['components_95'],a['components_9999'],a['rank_tolerance_sweep']) for a in rows], 'pooled':{k:pooled[k] for k in ['components_95','components_9999','rank_tolerance_sweep']},'aligned':{k:aligned[k] for k in ['components_95','components_9999','rank_tolerance_sweep']},'density_check':[(a['label'],a['components_95'],a['rank_tolerance_sweep']) for a in stability],'heldout':checks},indent=2))

if __name__=='__main__':main()
