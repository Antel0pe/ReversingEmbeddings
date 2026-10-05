"""Test angle ordering and simple image-vector rules between width curves."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from make_generated_lean_width_comparison import BASE, LO, HI, images, norm, angle

OUT=Path(__file__).resolve().parent/'figures/generated_lean_width_comparison/width_sweep/transforms'


def score(pred,target,source):
    error=norm(pred-target)
    return dict(median_image_error=float(np.median(error)),p95_image_error=float(np.percentile(error,95)),
        max_image_error=float(error.max()),error_fraction_of_true_width_change=float(np.linalg.norm(pred-target)/np.linalg.norm(target-source)),
        invalid_image_fraction=float(np.mean(np.any((pred<0)|(pred>1),axis=1))),
        maximum_pixel_error=float(np.max(abs(pred-target))))


def at_midpoints(x):
    return (x[:-1]+x[1:])/2


def angle_graph_rule_tests():
    theta=np.linspace(LO+.5,HI-.5,881)
    base=images(BASE[3],theta+.05)-images(BASE[3],theta-.05)
    increments=[.1,.15,.5,.6,1.]
    graph={off:angle(base,images(BASE[3]+off,theta+.05)-images(BASE[3]+off,theta-.05)) for off in increments}
    rows=[]
    training=np.arange(len(theta))%2==0
    testing=~training
    for source,target in [(.1,.15),(.5,.6),(.1,1.),(.5,1.)]:
        x,y=graph[source],graph[target]
        xt,yt=x[training],y[training]
        factor=float(xt@yt/(xt@xt))
        shift=float(np.mean(yt-xt))
        affine=np.linalg.lstsq(np.column_stack([xt,np.ones(len(xt))]),yt,rcond=None)[0]
        models={
            'add_one_angle_offset':(x+shift,dict(offset_degrees=shift)),
            'multiply_by_one_factor':(factor*x,dict(factor=factor)),
            'factor_and_offset':(affine[0]*x+affine[1],dict(factor=float(affine[0]),offset_degrees=float(affine[1]))),
            'multiply_by_width_increment_ratio':((target/source)*x,dict(factor=target/source))}
        row=dict(source_increase=source,target_increase=target,models={})
        for label,(pred,params) in models.items():
            err=pred[testing]-y[testing]
            row['models'][label]={**params,'held_out_lean_rmse_degrees':float(np.sqrt(np.mean(err**2))),
                'held_out_lean_max_error_degrees':float(np.max(abs(err)))}
        rows.append(row)
    return dict(training='alternating 0.1-degree lean samples',evaluation='intermediate 0.05-degree lean samples',
        note='Fits predict a scalar angle-to-reference graph only. They do not specify a 784D direction or reconstruct images. Target training samples calibrate each pair.',rows=rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    # Re-evaluate the angle graph at a finer lean grid, discarding float32-scale reversals.
    theta=np.linspace(LO+.5,HI-.5,4401)
    offsets=np.arange(1,11)/10
    angles={}
    for h in [.1,.02]:
        base=images(BASE[3],theta+h/2)-images(BASE[3],theta-h/2)
        aa=[]
        for off in offsets:
            tangent=images(BASE[3]+off,theta+h/2)-images(BASE[3]+off,theta-h/2)
            aa.append(angle(base,tangent))
        aa=np.array(aa)
        pairs=[]
        for i in range(10):
            for j in range(i+1,10):
                difference=aa[i]-aa[j]
                meaningful=(np.minimum(aa[i],aa[j])>.5)&(abs(difference)>.01)
                inds=np.flatnonzero(meaningful)
                flips=np.flatnonzero(difference[inds[:-1]]*difference[inds[1:]]<0)
                pairs.append(dict(increase_a=float(offsets[i]),increase_b=float(offsets[j]),
                    largest_order_reversal_degrees=float(max(0,difference.max())),
                    nontrivial_crossing_brackets=[[float(theta[inds[k]]),float(theta[inds[k+1]])] for k in flips]))
        angles[str(h)]=dict(grid_degrees=.01,pair_count=45,zero_overlap_excluded_below_angle=.5,
            sign_tolerance_degrees=.01,pairs_with_detected_crossings=sum(bool(p['nontrivial_crossing_brackets']) for p in pairs),
            maximum_order_reversal_degrees=max(p['largest_order_reversal_degrees'] for p in pairs),pairs=pairs)
    # Known curves are sampled at 0.1-degree lean intervals; every evaluation lean is between them.
    train=np.linspace(LO,HI,451)
    test=at_midpoints(train)
    baseline_train=images(BASE[3],train)
    baseline=images(BASE[3],test)
    baseline_est=at_midpoints(baseline_train)
    baseline0=images(BASE[3],0)[0]
    neighbor_train=images(BASE[3]+.1,train)
    neighbor_est=at_midpoints(neighbor_train)
    anchor_offsets=np.arange(11)/10
    bank=[at_midpoints(images(BASE[3]+off,train)) for off in anchor_offsets]
    evaluation_offsets=np.round(np.arange(.05,1,.1),10)
    rows=[]
    # Target samples fit only the best-global-affine diagnostic, never the response/interpolation models.
    for off in list(evaluation_offsets)+[1.]:
        target=images(BASE[3]+off,test)
        target0=images(BASE[3]+off,0)[0]
        target_train=images(BASE[3]+off,train)
        ac=baseline_train-baseline_train.mean(0)
        bc=target_train-target_train.mean(0)
        scale=float(np.sum(ac*bc)/np.sum(ac*ac))
        shift=target_train.mean(0)-scale*baseline_train.mean(0)
        predictions={
            'unchanged_curve':baseline_est,
            'constant_upright_offset':baseline_est+(target0-baseline0),
            'multiply_by_width_ratio':baseline_est*((BASE[3]+off)/BASE[3]),
            'best_scalar_plus_constant_offset':scale*baseline_est+shift,
            'scaled_neighbor_response':baseline_est+(off/.1)*(neighbor_est-baseline_est),
            'interpolate_endpoint_curves':bank[0]+off*(bank[-1]-bank[0])}
        if off<1:
            i=int(np.floor(off/.1))
            fraction=(off-anchor_offsets[i])/.1
            predictions['interpolate_adjacent_width_curves']=(1-fraction)*bank[i]+fraction*bank[i+1]
        predictions['scaled_neighbor_response_clipped']=np.clip(predictions['scaled_neighbor_response'],0,1)
        metrics={k:score(p,target,baseline) for k,p in predictions.items()}
        rows.append(dict(increase=float(off),width=float(BASE[3]+off),true_width_gap_median=float(np.median(norm(target-baseline))),
            best_scalar=scale,models=metrics,endpoint_interpolation_uses_target_itself=bool(off==1)))
    result=dict(normal_knobs=BASE.tolist(),angle_graph_ordering=angles,
        transformation_test=dict(training_lean_step=.1,evaluation_lean='midpoints of 0.1-degree training grid',
            held_out_width_increases=evaluation_offsets.tolist(),extra_extrapolation_test=1.,rows=rows,
            error_units='Euclidean image distance over 784 coverage values; no clipping before scoring',
            model_definitions={
                'constant_upright_offset':'normal_curve(t) + [target_upright - normal_upright]',
                'multiply_by_width_ratio':'normal_curve(t) * target_width/normal_width',
                'best_scalar_plus_constant_offset':'a * normal_curve(t) + b; one scalar and one fixed 784-vector, fitted using target training samples',
                'scaled_neighbor_response':'normal_curve(t) + increase/0.1 * [neighbor_3.3_curve(t) - normal_curve(t)]',
                'interpolate_endpoint_curves':'normal_curve(t) + increase * [width_4.2_curve(t) - normal_curve(t)]',
                'interpolate_adjacent_width_curves':'weighted mean of known curves bracketing target width at 0.1-width spacing'},
            limits='The reusable increment is a lean-dependent 784-vector curve, not one fixed offset or one angle. Neighbor interpolation requires two nearby known curves. Center and height remain fixed.'))
    result['angle_graph_rules']=angle_graph_rule_tests()
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(16,8.8))
    fig.text(.065,.95,'Can a scaled width increment predict another lean path?',fontsize=18,weight='bold')
    fig.text(.065,.89,'Learn the pixel-change curve from width 3.2 to 3.3. To predict +0.15, add 1.5 times that change at the SAME lean.\n'
        'All errors compare full 784-pixel images. Known curves use 0.1-degree samples; test lean values lie halfway between them.',fontsize=11)
    ax=fig.add_axes([.065,.275,.43,.47])
    labels=[('unchanged_curve','Ignore width change','#888888','--'),
        ('constant_upright_offset','Add one fixed upright offset','#ba5c24','-'),
        ('scaled_neighbor_response','Scale the +0.1 response at each lean','#276baf','-'),
        ('interpolate_adjacent_width_curves','Interpolate two adjacent known widths','#218453','-')]
    for key,label,color,style in labels:
        ax.plot(evaluation_offsets,[r['models'][key]['median_image_error'] for r in rows[:-1]],style,color=color,marker='o',ms=4,label=label)
    ax.set(xlabel='Width-knob increase from normal 3.2',ylabel='Median prediction error (image L2)',title='Prediction error over the full lean path')
    ax.grid(alpha=.15);ax.legend(loc='upper left',fontsize=9)
    # Image evidence at a fixed lean. Invalid extrapolated coverages are visibly marked.
    exlean=20.
    x=images(BASE[3],exlean)[0]
    neighbor=images(BASE[3]+.1,exlean)[0]
    fixed=[.15,1.]
    for rr,off in enumerate(fixed):
        target=images(BASE[3]+off,exlean)[0]
        pred=x+(off/.1)*(neighbor-x)
        err=pred-target
        for cc,(data,title) in enumerate([(target,f'True width {BASE[3]+off:.2f}'),(pred,f'Add {off/.1:g} x the +0.1 change'),(err,'Prediction minus true')]):
            ax=fig.add_axes([.57+cc*.135,.50-rr*.28,.105,.175])
            if cc<2:
                ax.imshow(data.reshape(28,28),cmap='gray_r',vmin=0,vmax=1)
                bad=((data<0)|(data>1)).reshape(28,28)
                overlay=np.zeros((28,28,4));overlay[bad]=[1,0,1,1]
                ax.imshow(overlay)
            else:
                # Same error scale across both examples, defined below for reproducibility.
                limit=max(.01,float(np.max(abs(x+10*(neighbor-x)-images(BASE[3]+1,exlean)[0]))))
                ax.imshow(data.reshape(28,28),cmap='RdBu_r',vmin=-limit,vmax=limit)
            ax.set_xticks([]);ax.set_yticks([]);ax.set_title(title,fontsize=9)
            if cc==0:ax.text(-.2,.5,f'Width\n+{off:g}\nLean 20 deg',ha='right',va='center',transform=ax.transAxes,fontsize=9)
            if cc==1:ax.text(.5,-.08,f'Error {norm(err):.3f}; invalid pixels {np.count_nonzero((pred<0)|(pred>1))}',ha='center',va='top',transform=ax.transAxes,fontsize=9)
    fig.text(.065,.07,'Black pixels are ink coverage 1, white pixels are 0. Magenta marks predicted coverage outside [0, 1]; errors are scored before clipping.\n'
        f'Error maps use shared limits +/- {limit:.3f} coverage: blue = too little predicted ink; red = too much. Center and height stay fixed.\n'
        'The blue rule needs the normal and +0.1 curves. The green rule needs two curves adjacent to each target width; its additional information is explicit.',fontsize=10)
    fig.savefig(OUT/'prediction_rules.png',dpi=150);plt.close(fig)
    # Report separates scalar-angle graph ordering from actual vector reconstruction.
    lines=['# Crossings and simple transformations between width-dependent lean curves','',
        'The attached graph is the previously generated direction-angle graph. Each plotted value is an angle to the normal-width lean tangent at the same lean. It is not an absolute orientation coordinate.','',
        '## Crossing questions','']
    for h,a in angles.items():
        lines.append(f"Centered lean step {h} degrees, sampled every 0.01 degree: {a['pairs_with_detected_crossings']} of 45 pairs show nontrivial order reversals. Largest numerical reversal is {a['maximum_order_reversal_degrees']:.6f} degrees. Near-zero overlaps below 0.5 degree and pair differences below 0.01 degree are excluded from crossing detection.")
    lines += ['', 'This is a dense numerical check on widths 3.3 to 4.2, not a proof of ordering at every possible width/lean. Curves can overlap in angle near upright and on plateaus without swapping order. Equal angles to a common reference generally do not determine identical 784D directions.',
        'Actual rendered curves and their connected polylines at different widths cannot intersect for these fixed unclipped strokes: total ink = height times width, different for every width. Coinciding images would require equal total ink.','',
        '## Transformation tests','',
        'All target images are scored at lean values between training samples. Fresh target width increases are +0.05, +0.15, ..., +0.95. One additional +1 test measures extrapolation from the +0.1 response.',
        'An error fraction below 1 means better than ignoring width. It divides pooled prediction error by the true width-change magnitude; it is not error divided by the norm of the entire image.',
        'The best scalar-plus-offset model uses target training samples to fit one scalar and one fixed 784-vector, so it is a favorable diagnostic of that model class, not a predictor learned without seeing the target width.','',
        '| Width increase | True gap, median | Fixed upright offset error | Best scalar + offset error | Scaled +0.1 response error | Adjacent-width interpolation error |',
        '| --- | --- | --- | --- | --- | --- |']
    for r in rows:
        def cell(key):return f"{r['models'][key]['median_image_error']:.3f} ({100*r['models'][key]['error_fraction_of_true_width_change']:.1f}% of width effect)" if key in r['models'] else 'known endpoint; not scored'
        lines.append(f"| +{r['increase']:.2f} | {r['true_width_gap_median']:.3f} | {cell('constant_upright_offset')} | {cell('best_scalar_plus_constant_offset')} | {cell('scaled_neighbor_response')} | {cell('interpolate_adjacent_width_curves')} |")
    lines += ['', '## A reusable nearby-width rule','',
        'At each lean, compute `width_step(lean) = curve_width_3.3(lean) - curve_width_3.2(lean)`. Then predict `curve_width_3.35(lean) approximately = curve_width_3.2(lean) + 1.5 * width_step(lean)`.',
        'The same scalar 1.5 applies to the entire response curve. However, the response itself depends on lean. This is not one fixed 784-vector added to every point, and the approximate images need not satisfy the exact generator equation.',
        'Extrapolating this +0.1 response by 10 times to +1 accumulates much larger error. Width effects are piecewise linear in raw coverage at a fixed lean, with slope changes when stroke edges cross pixel/subrow events. Interpolating neighboring known width curves handles those changes much better.','',
        '![Prediction rules and pixel evidence](prediction_rules.png)','',
        'Reproduce with `python analyze_generated_lean_width_transforms.py`. This uses the actual Python renderer. The entire five-knob family is outside the test scope.']
    lines += ['', 'A supplementary rule clips scaled-response predictions to [0, 1]. It improves extrapolation, but does not establish membership in the generated family. Clipped-response metrics are also saved in results.json.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(dict(angle_summary={h:{k:v for k,v in a.items() if k!='pairs'} for h,a in angles.items()},
        transform_summary=[r for r in rows if r['increase'] in [.15,.55,.95,1.]]),indent=2))

if __name__=='__main__':main()
