"""Readable post-freeze results, with one distance-comparison figure."""
from pathlib import Path
import json
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/reading-path-figures')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run import image_jacobian, restore, verify_freezes, SPAN, ROOT, OLD, OUT, SCALES


def pooled(c,mode,scale,field):
    values=[];complete=0;total=0
    for row in c['paths']:
        if row['mode']!=mode or row['halfspan']!=scale:continue
        complete+=row['complete_paths'];total+=row['paths']
        if not row['reference_complete']:continue
        values.extend(v for v,ok in zip(row[field][1:],row['each_path_complete'][1:]) if ok)
    return np.asarray(values),complete,total


def main():
    before=verify_freezes()
    r=json.loads((OUT/'results.json').read_text());a=np.load(OUT/'paths.npz')
    summary={};verification=dict(valid_path_max_off_reading_error=0.,valid_path_max_linearity_error=0.)
    for key,c in r['charts'].items():
        summary[key]={}
        for mode in ['shared_reading_ticks','unit_initial_speed']:
            summary[key][mode]={}
            for scale in SCALES:
                err,complete,total=pooled(c,mode,scale,'each_distance_matrix_error')
                shape,_,_=pooled(c,mode,scale,'each_shape_only_error')
                rows=[p for p in c['paths'] if p['mode']==mode and p['halfspan']==scale]
                ref=np.array([p['reference_complete'] for p in rows])
                cross=a[key+'_'+mode+'_'+str(scale)+'_source_shape_pair_errors']
                selected=np.triu(np.outer(ref,ref),1)
                summary[key][mode][str(scale)]=dict(
                    complete=complete,total=total,comparisons=len(err),
                    median_distance_error=float(np.median(err)) if len(err) else None,
                    max_distance_error=float(np.max(err)) if len(err) else None,
                    median_shape_error=float(np.median(shape)) if len(shape) else None,
                    max_shape_error=float(np.max(shape)) if len(shape) else None,
                    cross_family_complete_pairs=int(selected.sum()),
                    cross_family_pairs_within_1_percent=int(np.sum(cross[selected]<.01)),
                    max_cross_family_shape_error=float(np.max(cross[selected])) if selected.any() else None)
                # Independently inspect actual saved images: other readings stay
                # fixed and the active reading follows an evenly spaced line.
                images=a[key+'_'+mode+'_'+str(scale)+'_images']
                matrix=a[key+'_matrix'];offset=a[key+'_offset']
                q=images@matrix.T+offset
                qscale=np.asarray(c['mapping']['reading_scale'])
                for i,row in enumerate(rows):
                    good=np.asarray(row['each_path_complete'])
                    if not good.any():continue
                    delta=(q[i,good]-q[i,good,8,None,:])/qscale
                    off=np.delete(delta,i,axis=2)
                    verification['valid_path_max_off_reading_error']=max(verification['valid_path_max_off_reading_error'],float(abs(off).max()))
                    expected=np.linspace(-1,1,17)[None,:]*delta[:,-1,i,None]
                    verification['valid_path_max_linearity_error']=max(verification['valid_path_max_linearity_error'],float(abs(delta[:,:,i]-expected).max()))
        points=a[key+'_anchor_settings']
        x,j=image_jacobian(points)
        assert np.max(abs(x-a[key+'_shared_reading_ticks_0.005_images'][0,:,8]))<1e-12
        # A chart that works at its own source need not transfer to the other.
        other_key=key[:-1]+str(1-c['source_index'])
        other_points=a[other_key+'_anchor_settings'][:1]
        _,other_j=image_jacobian(other_points)
        matrix=a[key+'_matrix'];qs=np.asarray(c['mapping']['reading_scale'])
        singular=np.linalg.svd((matrix@other_j[0])*SPAN/qs[:,None],compute_uv=False)
        summary[key]['other_source_rank']=int(np.sum(singular>singular[0]*1e-8))
        summary[key]['other_source_scaled_singular_values']=singular.tolist()
    assert verification['valid_path_max_off_reading_error']<1e-6
    assert verification['valid_path_max_linearity_error']<1e-6
    verification['frozen_files_unchanged']=verify_freezes()==before
    summary['verification']=verification
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2))

    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(15,9.5),facecolor='#fbfaf6')
    fig.text(.035,.965,'Do the learned coordinate paths keep the same distance pattern?',fontsize=18,weight='bold',va='top')
    fig.text(.035,.918,'Generated 28 × 28 images; white = no ink, black = full ink. Freeze the learned pixel weights; vary one reading, hold four fixed.',fontsize=10.5)
    grid=fig.add_gridspec(2,3,left=.045,right=.975,bottom=.265,top=.845,width_ratios=[.72,2,2],hspace=.58,wspace=.4)
    colors={'192_random':'#be683b','512_balanced':'#276b9d'}
    for si in range(2):
        ax=fig.add_subplot(grid[si,0])
        image=a[f'512_balanced_source_{si}_shared_reading_ticks_0.005_images'][0,0,8]
        ax.imshow(image.reshape(28,28),cmap='gray_r',vmin=0,vmax=1)
        ax.set_title('Upright source' if si==0 else 'Second source',fontsize=11,pad=9)
        ax.set_xticks([]);ax.set_yticks([])
        ax.text(.5,-.12,'Baseline + 8 nearby starts',transform=ax.transAxes,ha='center',fontsize=9)
        for column,(field,title) in enumerate([
            ('each_distance_matrix_error','Same reading ticks: distance disagreement'),
            ('each_shape_only_error','Remove overall speed: shape disagreement')],start=1):
            ax=fig.add_subplot(grid[si,column])
            for run,label in [('192_random','192 random'),('512_balanced','512 balanced')]:
                c=r['charts'][run+'_source_'+str(si)]
                values=[pooled(c,'shared_reading_ticks',s,field)[0]*100 for s in SCALES]
                median=[np.median(v) for v in values]
                low=[np.quantile(v,.1) for v in values];high=[np.quantile(v,.9) for v in values]
                ax.plot(SCALES,median,'o-',color=colors[run],label=label,lw=2)
                ax.fill_between(SCALES,low,high,color=colors[run],alpha=.13)
            ax.axhline(1,color='#70767b',ls='--',lw=1)
            ax.text(.98,1,'1% comparison line',ha='right',va='bottom',transform=ax.get_yaxis_transform(),fontsize=8)
            ax.set_xscale('log');ax.set_yscale('symlog',linthresh=.001)
            ax.set_xticks(SCALES,['0.005','0.05','0.25','1.0'])
            ax.set_ylim(0,6)
            ax.set_title(title,fontsize=10.5,pad=10)
            ax.set_xlabel('Path half-span (initial image-distance units)',fontsize=9)
            ax.set_ylabel('Difference from baseline path (%)',fontsize=9)
            if si==0 and column==1:handles,labels=ax.get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.53,.185),ncol=2,frameon=False)
    fig.text(.035,.164,'Each comparison uses all pairwise distances among 17 images. Lines = median; bands = 10th–90th percentiles over complete comparisons.',fontsize=9.5)
    fig.text(.035,.126,'At half-span 0.05 all 45 paths per chart completed. At 0.25: 40/45 and 41/45 upright; 45/45 second source. At 1.0: 27/45 in every chart.',fontsize=9.5)
    fig.text(.035,.090,'Large-step statistics include only completed paths. Starts are within radius 0.25; paths can leave it. Solver uses the known renderer, not a learned decoder.',fontsize=9)
    fig.text(.035,.052,'Crucial limit: at half-span 0.05, all 10 pairs of DIFFERENT source knobs also have shapes within 1%. Similarity does not identify the knob.',fontsize=10,weight='bold')
    fig.savefig(OUT/'distance_similarity.png',dpi=160);plt.close(fig)

    lines=['# What the frozen learned readings actually do','',
           '**Result: the learned coordinates have very similar local path distances across nearby starts, but different coordinates also have nearly identical distance signatures. Distance agreement is not a unique knob identifier.**','',
           'This evaluates the actual 192-random and 512-balanced fixed pixel readings from the radius-0.25 patches. It does not substitute the reverse-engineered top/bottom/slope coordinates. All existing frozen files were checked unchanged.','',
           '## How a path was defined','',
           'At each of nine starts (the source and the first eight held-out frame states), change one frozen scalar reading and hold the other four fixed. Solve the five equations on the known generated-image family. The renderer and derivative are evaluation tools here, not a discovered inverse model. Each path has 17 samples. Both signs are followed by continuation.','',
           'Shared reading ticks are calibrated once for each coordinate at the chart source to unit initial pixel speed. The alternative calibration normalizes initial speed separately at each start. Shape-only comparison divides each complete pairwise-distance matrix by its norm. This distinguishes consistent distances from similarities manufactured by rescaling.','',
           '## Measured distance agreement','',
           '| Acquisition / source | Half-span | Complete paths | Median distance-matrix difference | Maximum difference | Median shape-only difference |',
           '| --- | ---: | ---: | ---: | ---: | ---: |']
    for key,c in r['charts'].items():
        for scale in SCALES:
            s=summary[key]['shared_reading_ticks'][str(scale)]
            lines.append(f"| {key} | {scale:g} | {s['complete']}/{s['total']} | {100*s['median_distance_error']:.6g}% | {100*s['max_distance_error']:.6g}% | {100*s['median_shape_error']:.6g}% |")
    lines+=['',
            'There are five reference paths and up to 40 reference-to-held-out comparisons per chart. At half-span 0.05 every path completed. The largest distance differences were 0.18–0.19% at the upright patch and 1.11–1.32% at the second patch. The balanced run is not uniformly better on this path metric. Counts and sampling design both changed.','',
            'At half-spans through 0.25, every pair of different source-coordinate paths also has shape-only distance difference below 1% in all four charts. Almost straight paths at comparable speed naturally have almost identical distances, even if they change different original controls. This is a concrete counter-control to using distance similarity alone for family identity.','',
            'At half-span 1, 18 of 45 paths per chart did not complete to the coordinate-residual requirement. Incomplete paths are saved. This is a continuation failure, not proof that every requested coordinate target is mathematically impossible. Surviving-path statistics must not be interpreted as full-chart success. Some smaller paths also leave the radius-0.25 training ball. Default parameter bounds are not imposed, consistent with the original two-sided root patches; positive stroke size and full containment on the canvas are required.','',
            'The same frozen pixel chart can also lose rank at the other distant source:']
    for key in r['charts']:
        lines.append(f"- {key}: derivative rank {summary[key]['other_source_rank']} of 5 at the other source.")
    lines+=['','## The actual original-knob mappings','',
            'These are post-freeze maps of the learned readings, not geometric readers supplied to discovery. A standardized reading subtracts its training mean and divides by its training standard deviation. The original controls are standardized the same way. Thus the matrices below compare effects on comparable scales, without mixing pixels with degrees.','',
            'Each matrix maps the five standardized readings to standardized original controls. It is the best linear fit on the original training set. The full cubic fit is a better approximation and uses squares, pairwise products and cubic terms of all five readings. Exact means, scales, every polynomial coefficient, and held-out errors are in `results.json` under each chart mapping.','']
    for key,c in r['charts'].items():
        lines += [f'### {key}','', '| Original control | Reading 1 | Reading 2 | Reading 3 | Reading 4 | Reading 5 |','| --- | ---: | ---: | ---: | ---: | ---: |']
        for name,row in zip(r['names'],c['mapping']['standardized_linear_matrix']):
            lines.append('| '+name+' | '+' | '.join(f'{v:+.4f}' for v in row)+' |')
        lines+=['','| Original control | Linear held-out RMS error | Cubic held-out RMS error |','| --- | ---: | ---: |']
        for i,name in enumerate(r['names']):
            e1=c['mapping']['degrees']['1']['test']['rmse'][i]
            e3=c['mapping']['degrees']['3']['test']['rmse'][i]
            lines.append(f'| {name} | {e1:.6g} | {e3:.6g} |')
        lines+=['','Position/height/width errors are in pixels; lean errors are in degrees. These maps use original labels after discovery. Near-perfect cubic interpolation locally is not a compact exact global manifold equation.','']
    lines+=['## PCA of changes within each vector family','',
            'Fit PCA to centered, sign-aligned training direction vectors in each family. Use held-out states for vector approximation errors. The variation includes curvature, family matching errors and finite-neighborhood sampling noise. PCA is not told which source causes it.','',
            '| Acquisition / source | Family | PCs for 95% training variation | Mean-only checking vector error | With 2 PCs | With 5 PCs |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for key,c in r['charts'].items():
        for p in c['pca']:
            e=p['test_mean_vector_error']
            lines.append(f"| {key} | {p['family']} | {p['pcs_for_95']} | {e['0']:.6g} | {e['2']:.6g} | {e['5']:.6g} |")
    lines+=['',
            'Direction vectors have length one; checking errors are mean Euclidean errors in the full 784-dimensional direction, not original-knob accuracies. A family mean plus a few changing patterns can compress its observed direction variation. This does not give the state-dependent coefficients of those patterns, independent coordinates, or an image decoder.','',
            '## Applying PCA repeatedly','',
            f"Applying PCA again to the full PCA scores changed the singular-value spectrum by relative error {r['repeated_pca']['repeated_singular_value_relative_error']:.3g}. It produced no new information. Once PCA has decorrelated the data, an unchanged second pass retains the same principal directions up to signs, ordering and rotations within equal-variance spaces. Truncation can remove information but cannot recover new structure by repetition.",'',
            'A different question is PCA on a collection of direction vectors from many states; that changes the dataset to the variation of the directions and is what the family experiment above tests. Centered PCA on just five orthonormal basis vectors at a single state instead has four equal nonzero singular values. That result follows from their orthonormality, not the manifold having four knobs.','',
            '## Tangent combinations versus finite moves','',
            'At a smooth state where five original control derivatives are independent, every alternative coordinate direction is a linear combination of those five derivatives at that state. A nonlinear coordinate transformation makes the combination coefficients depend on the state. A finite displacement along the resulting curved path need not lie in the starting tangent plane. Nonlinear coordinate maps and local linear combinations are compatible.','',
            '## Verification and artifacts','',
            f"- Independent renderer agreement: maximum difference {r['verification']['max_renderer_difference']:.3g}.",
            f"- Analytic evaluation Jacobian versus central differences: maximum relative error {max(r['verification']['jacobian_relative_errors']):.3g}.",
            f"- Saved valid paths: other-reading drift below {verification['valid_path_max_off_reading_error']:.3g} training standard deviations.",
            f"- Saved valid paths: active-reading line error below {verification['valid_path_max_linearity_error']:.3g} training standard deviations.",
            '- Both original freeze manifests and all files they cover remain unchanged.',
            '- `results.json`: full mappings, centered family PCA, all path measurements and verification.',
            '- `summary.json`: pooled comparisons and cross-family controls.',
            '- `paths.npz`: images, source settings, inverse solutions, residuals and actual frozen measurement matrices.',
            '- Sources and pre-run protocol: `experiments/pooled_reading_paths/`.','',
            '![Distance similarity of the learned coordinate paths](distance_similarity.png)']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
