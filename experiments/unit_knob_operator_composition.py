"""Original-control one-step operations on a finite widening pattern.

Known geometry is supplied. This is a representation/composition test, not
learning geometry or discovering operators from isolated pixel vectors.
Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/unit_knob_operator_composition.py
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/unit-knob-operators-mpl')
from pathlib import Path
import sys
import itertools
import json
import csv
import numpy as np
from scipy.stats import qmc
from scipy.ndimage import affine_transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from grey_ones import render,RANGES,KNOBS,SUB
from experiments.width_pattern_state_transport import predict,update_pattern

OUT=ROOT/'figures'/'unit_knob_operator_composition'
BASE=np.array([14.,14.,19.75,3.2,0.])
UNIT=.05*np.ptp(RANGES,axis=1)
WIDTH_STEP=.28
SEED=20261015
COUNT=64
PERMS=list(itertools.permutations(range(5)))
YLO=np.arange(28*SUB)/SUB
YMID=YLO+.5/SUB
COLS=np.arange(28)


def images(p):
    return render(p,dtype=np.float64).reshape(-1,784)


def truth(p):
    p=np.atleast_2d(p);q=p.copy();q[:,3]+=WIDTH_STEP
    for settings in [p,q]:
        reach=settings[:,3]/2+settings[:,2]/2*np.abs(np.tan(np.deg2rad(settings[:,4])))
        assert np.all((settings[:,2]>0)&(settings[:,3]>0)&
                      (settings[:,0]-reach>0)&(settings[:,0]+reach<28)&
                      (settings[:,1]-settings[:,2]/2>0)&(settings[:,1]+settings[:,2]/2<28))
    return images(q)-images(p)


def root_bands():
    cx,cy,h,w,_=BASE
    bounds=[(cx-w/2-WIDTH_STEP/2,cx-w/2),
            (cx+w/2,cx+w/2+WIDTH_STEP/2)]
    return np.array([[[left,cy-h/2],[right,cy-h/2],
                      [right,cy+h/2],[left,cy+h/2]] for left,right in bounds])


def coverage(bands):
    """Rasterize transported polygons with the original 16-subrow convention.

    This reads polygon vertices, not current knob settings, and does not call
    grey_ones.render. It handles only the parallelogram band class used here.
    """
    bands=np.asarray(bands).reshape(-1,2,4,2)
    output=[]
    for start in range(0,len(bands),128):
        v=bands[start:start+128].reshape(-1,4,2)
        top=v[:,0,1,None];bottom=v[:,3,1,None]
        fraction=(YMID[None,:]-top)/(bottom-top)
        left=v[:,0,0,None]+fraction*(v[:,3,0,None]-v[:,0,0,None])
        right=v[:,1,0,None]+fraction*(v[:,2,0,None]-v[:,1,0,None])
        weight=np.clip((np.minimum(YLO[None,:]+1/SUB,bottom)-np.maximum(YLO[None,:],top))*SUB,0,1)
        overlap=np.clip(np.minimum(right[:,:,None],COLS[None,None,:]+1)-
                        np.maximum(left[:,:,None],COLS[None,None,:]),0,1)
        result=(overlap*weight[:,:,None]).reshape(-1,28,SUB,28).mean(2)
        output.append(result.reshape(-1,2,28,28))
    return np.concatenate(output)


def matrices(state,axis,amount):
    """One rule per knob; each acts on the whole CURRENT pattern.

    Matrices use physical x,y coordinates. Recompute the local pivot, height
    ratio, and tangent-slope increment from current state at every unit step.
    """
    cx,cy,h,w,theta=state
    a=np.tile(np.eye(3),(2,1,1))
    if axis==0:
        a[:,0,2]=amount
    elif axis==1:
        a[:,1,2]=amount
    elif axis==2:
        ratio=(h+amount)/h
        slope=np.tan(np.deg2rad(theta))
        a[:,1,1]=ratio;a[:,1,2]=(1-ratio)*cy
        a[:,0,1]=-slope*(ratio-1)
        a[:,0,2]=slope*(ratio-1)*cy
    elif axis==3:
        a[:,0,2]=[-amount/2,amount/2]
    else:
        ds=np.tan(np.deg2rad(theta+amount))-np.tan(np.deg2rad(theta))
        a[:,0,1]=-ds;a[:,0,2]=ds*cy
    return a


def move_vertices(bands,mat):
    return np.einsum('bij,bkj->bki',mat[:,:2,:2],bands)+mat[:,:2,2,None].transpose(0,2,1)


def move_raster(channels,mat):
    out=[]
    for channel,a in zip(channels,mat):
        inv=np.linalg.inv(a[:2,:2])
        # Image samples live at physical pixel centers, not at grid corners.
        offset=inv@(np.full(2,.5)-a[:2,2])-.5
        out.append(affine_transform(channel,inv[::-1,::-1],offset[::-1],
                                    output_shape=(28,28),order=1,mode='constant',cval=0.,prefilter=False))
    return np.array(out)


def schedule(counts,order,interleaved=False,subdivisions=1):
    counts=np.asarray(counts,dtype=int)
    if interleaved:
        for layer in range(np.abs(counts).max()):
            for axis in order:
                if layer < abs(counts[axis]):
                    for _ in range(subdivisions):
                        yield axis,np.sign(counts[axis])*UNIT[axis]/subdivisions
    else:
        for axis in order:
            for _ in range(abs(counts[axis])):
                for _ in range(subdivisions):
                    yield axis,np.sign(counts[axis])*UNIT[axis]/subdivisions


def follow(counts,order,interleaved=False,subdivisions=1,raster=False,keep=False):
    state=BASE.copy();bands=root_bands()
    channels=coverage(bands)[0] if raster else None
    history=[]
    if keep:
        history.append(dict(state=state.copy(),bands=bands.copy(),channels=channels.copy(),axis=-1))
    for axis,amount in schedule(counts,order,interleaved,subdivisions):
        mat=matrices(state,axis,amount)
        bands=move_vertices(bands,mat)
        if raster:
            channels=move_raster(channels,mat)
        state[axis]+=amount
        if keep:
            history.append(dict(state=state.copy(),bands=bands.copy(),channels=channels.copy(),axis=axis))
    return state,bands,channels,history


def score(predicted,actual):
    relative=np.linalg.norm(predicted-actual,axis=1)/np.linalg.norm(actual,axis=1)
    return dict(pooled_error_fraction=float(np.linalg.norm(predicted-actual)/np.linalg.norm(actual)),
                median_error_fraction=float(np.median(relative)),worst_error_fraction=float(relative.max()),
                maximum_pixel_absolute_error=float(np.abs(predicted-actual).max()))


def subdivide_old_model():
    source=ROOT/'figures'/'width_pattern_state_transport'/'patterns_and_model.npz'
    data=np.load(source)
    model=tuple(data[k] for k in ['origin','linear','diagonal','cross'])
    targets=data['states_1.0'];actual=data['truth_1.0']
    direct=predict(targets,model)['interaction']
    records=[]
    for subdivisions in [1,2,4,8,16]:
        result=[]
        for target in targets:
            state=np.zeros(5);pattern=model[0].copy()
            for axis in range(5):
                amount=target[axis]/subdivisions
                for _ in range(subdivisions):
                    pattern=update_pattern(pattern,state,axis,amount,model)
                    state[axis]+=amount
            result.append(pattern)
        result=np.array(result)
        records.append(dict(subdivisions=subdivisions,**score(result,actual),
                            maximum_difference_from_direct=float(np.linalg.norm(result-direct,axis=1).max())))
    return records


def experiment():
    u=qmc.Sobol(5,scramble=True,seed=SEED).random_base2(6)
    levels=np.array([-4,-3,-2,-1,1,2,3,4])
    counts=levels[(u*8).astype(int)]
    targets=BASE+counts*UNIT
    actual=truth(targets)
    all_bands=[];all_states=[]
    for count in counts:
        for order in PERMS:
            state,bands,_,_=follow(count,order)
            all_bands.append(bands);all_states.append(state)
    all_bands=np.array(all_bands);all_states=np.array(all_states)
    predicted=coverage(all_bands).sum(axis=1).reshape(COUNT,120,784)
    errors=predicted-actual[:,None,:]
    relative=np.linalg.norm(errors,axis=2)/np.linalg.norm(actual,axis=1)[:,None]
    operator_result=dict(targets=COUNT,block_orders_per_target=120,
                         compared_endpoints=COUNT*120,
                         maximum_relative_error=float(relative.max()),
                         maximum_pixel_absolute_error=float(np.abs(errors).max()),
                         maximum_state_difference=float(np.abs(all_states.reshape(COUNT,120,5)-targets[:,None,:]).max()),
                         maximum_order_disagreement_pixel_L2=float(np.linalg.norm(predicted-predicted[:,0,None,:],axis=2).max()))
    raster_results={};raster_arrays={}
    for name,order,interleaved in [('forward',range(5),False),('reverse',range(4,-1,-1),False),('interleaved',range(5),True)]:
        forecasts=[]
        for count in counts:
            _,_,channels,_=follow(count,order,interleaved,raster=True)
            forecasts.append(channels.sum(axis=0).ravel())
        raster_arrays[name]=np.array(forecasts)
        raster_results[name]=score(raster_arrays[name],actual)
    order_disagreement=float(np.linalg.norm(raster_arrays['forward']-raster_arrays['reverse'])/np.linalg.norm(actual))
    smaller=[]
    for subdivisions in [1,4,16]:
        forecast=[]
        for count in counts[:8]:
            _,_,channels,_=follow(count,range(5),subdivisions=subdivisions,raster=True)
            forecast.append(channels.sum(axis=0).ravel())
        smaller.append(dict(subdivisions=subdivisions,**score(np.array(forecast),actual[:8])))
    # A visible example: four height units, two lean units, and one of each rest.
    sequence=[2,2,4,0,4,1,3,2,2]
    state=BASE.copy();bands=root_bands();channels=coverage(bands)[0]
    history=[dict(state=state.copy(),bands=bands.copy(),channels=channels.copy(),axis=-1)]
    for axis in sequence:
        mat=matrices(state,axis,UNIT[axis]);bands=move_vertices(bands,mat);channels=move_raster(channels,mat)
        state[axis]+=UNIT[axis]
        history.append(dict(state=state.copy(),bands=bands.copy(),channels=channels.copy(),axis=axis))
    true_trace=truth(np.array([h['state'] for h in history]))
    geometric_trace=coverage(np.array([h['bands'] for h in history])).sum(axis=1).reshape(-1,784)
    raster_trace=np.array([h['channels'].sum(axis=0).ravel() for h in history])
    result=dict(method=dict(root=BASE.tolist(),knob_order=KNOBS,unit=UNIT.tolist(),
                            unit_definition='5 percent of each default knob range width; original control units, not equal image distances',
                            requested_width_increment=WIDTH_STEP,seed=SEED,
                            geometry='two finite widening parallelogram bands, four vertices each; current five control values retained',
                            operator_count=5,operator_class='two-channel homogeneous 3x3 affine maps, recomputed using the current state',
                            source='known renderer geometry supplied by hand; not learned from axis observations or pixels',
                            continuous_update='transform current band vertices and update only the changed state coordinate',
                            raster_comparison='same maps repeatedly bilinearly resample two 28x28 band maps; band labels and state supplied',
                            rasterizer='independent band coverage routine, matching the original 16-subrow convention; no render calls in updates',
                            targets='64 Sobol integer-count combinations; every knob changes by 1 to 4 signed units',
                            default_box='position bounds explicitly extended around canonical root; all sampled strokes positive and unclipped',
                            precision='float64'),
                geometric=operator_result,raster_only=raster_results,raster_order_disagreement_fraction=order_disagreement,
                raster_subdivision=smaller,previous_quadratic_subdivision=subdivide_old_model(),
                visible_trace=dict(axes=sequence,endpoint=history[-1]['state'].tolist(),
                                   geometric=score(geometric_trace,true_trace),raster=score(raster_trace,true_trace)))
    arrays=dict(counts=counts,target_settings=targets,actual_patterns=actual,geometric_patterns=predicted,
                endpoint_bands=all_bands,trace_states=np.array([h['state'] for h in history]),
                trace_actual=true_trace,trace_geometric=geometric_trace,trace_raster=raster_trace,
                **{'raster_'+k:v for k,v in raster_arrays.items()})
    assert operator_result['maximum_relative_error'] < 1e-10
    assert all(np.isfinite(v).all() for v in arrays.values())
    return result,arrays,history


def plot_chain(result,arrays,history):
    ticks=[0,2,4,7,9]
    starts=images(arrays['trace_states'])
    actual=arrays['trace_actual'];geo=arrays['trace_geometric'];raster=arrays['trace_raster'];error=raster-actual
    scale=float(max(np.abs(actual).max(),np.abs(geo).max(),np.abs(raster).max()))
    escale=float(np.abs(error[ticks]).max())
    fig=plt.figure(figsize=(14.5,13.7),facecolor='#fafaf7')
    fig.text(.035,.972,'1 · One-knob updates carry earlier changes forward',fontsize=21,weight='bold',va='top')
    fig.text(.035,.938,'Generated 28 × 28 coverage images: white = no ink, black = full ink. Every pattern predicts the same +0.28 px widening.',fontsize=10.5)
    fig.text(.035,.914,'Unit amounts: horizontal/vertical 0.05 px; height 0.075 px; width 0.14 px; lean 2.25°. Known geometry supplies all five update rules.',fontsize=10.5)
    fig.text(.035,.890,'Sequence: height, height, lean, horizontal, lean, vertical, width, height, height. Only one original knob changes at each step.',fontsize=10.5)
    grid=fig.add_gridspec(5,5,left=.14,right=.97,top=.83,bottom=.20,wspace=.28,hspace=.15)
    for j,tick in enumerate(ticks):
        for i,data in enumerate([starts,actual,geo,raster,error]):
            ax=fig.add_subplot(grid[i,j]);ax.set_xticks([]);ax.set_yticks([])
            if i==0:
                ax.imshow(data[tick].reshape(28,28),cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
                ax.set_title(f'Step {tick}'+(' · root' if tick==0 else ''),fontsize=11,pad=9)
            else:
                ax.imshow(data[tick].reshape(28,28),cmap='RdBu_r',vmin=-(escale if i==4 else scale),vmax=escale if i==4 else scale,interpolation='nearest')
                if i in [2,3]:
                    relative=np.linalg.norm(data[tick]-actual[tick])/np.linalg.norm(actual[tick])
                    ax.text(.5,-.045,f'Error {relative*100:.1f}%',transform=ax.transAxes,ha='center',va='top',fontsize=8)
    for y,label in zip([.775,.647,.519,.391,.263],['Current image\n(reference)','True width\npattern','Transported\nboundaries','Repeatedly\nresampled pixels','Pixel prediction\nminus truth']):
        fig.text(.025,y,label,fontsize=10.5,va='center')
    cbax=fig.add_axes([.24,.145,.30,.012]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-scale,scale),cmap='RdBu_r'),cax=cbax,orientation='horizontal')
    cb.set_label('Pattern scale: red gains ink; blue loses ink',fontsize=9);cb.ax.tick_params(labelsize=8)
    cbax=fig.add_axes([.64,.145,.30,.012]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(-escale,escale),cmap='RdBu_r'),cax=cbax,orientation='horizontal')
    cb.set_label('Error scale (separate): red overpredicts; blue underpredicts',fontsize=9);cb.ax.tick_params(labelsize=8)
    ge=result['visible_trace']['geometric']['worst_error_fraction']*100
    ra=result['visible_trace']['raster']['worst_error_fraction']*100
    fig.text(.035,.073,f'Largest error along this trace: transported boundaries {ge:.2g}% (roundoff); repeatedly resampled pixels {ra:.1f}%.',fontsize=11)
    fig.text(.035,.047,'Boundary coordinates retain subpixel positions. The pixel version is given the same state, side labels, and maps; it loses information when resampled.',fontsize=10)
    fig.text(.035,.023,'This is a supplied geometric model, not discovery from pixel vectors. All maps share a scale; no values are clipped.',fontsize=10)
    fig.savefig(OUT/'01_unit_chain.png',dpi=145,facecolor=fig.get_facecolor());plt.close(fig)


def plot_checks(result):
    fig=plt.figure(figsize=(15.5,7.5),facecolor='#fafaf7')
    fig.text(.035,.962,'2 · Do repeated single-knob updates preserve the combined pattern?',fontsize=20,weight='bold',va='top')
    fig.text(.035,.907,'Errors use all 784 pixel values and divide by the true +0.28 px widening effect, not the norm of the whole starting image.',fontsize=11)
    grid=fig.add_gridspec(1,3,left=.075,right=.97,top=.76,bottom=.29,wspace=.40)
    ax=fig.add_subplot(grid[0,0])
    values=[result['geometric']['maximum_relative_error']*100]+[result['raster_only'][key]['pooled_error_fraction']*100 for key in ['forward','reverse','interleaved']]
    ax.bar(range(4),values,color=['#299a86','#c17d44','#c17d44','#c17d44'])
    ax.set_xticks(range(4),['Boundary\nmodel','Pixels\nforward','Pixels\nreverse','Pixels\ninterleaved'],fontsize=9)
    for i,v in enumerate(values):ax.text(i,v+1,f'{v:.1e}%' if i==0 else f'{v:.1f}%',ha='center',va='bottom',fontsize=9)
    ax.set_ylim(0,max(values)*1.25);ax.set_ylabel('Prediction error (%)',fontsize=10)
    ax.set_title('64 joint targets; five knobs changed',fontsize=11,pad=14);ax.grid(axis='y',alpha=.2)
    ax.text(.01,-.23,'Boundary bar: worst of all 7,680 order endpoints.\nPixel bars: pooled error, one schedule each.',transform=ax.transAxes,fontsize=8.5,va='top')
    ax=fig.add_subplot(grid[0,1])
    records=result['previous_quadratic_subdivision'];x=[r['subdivisions'] for r in records];y=[100*r['pooled_error_fraction'] for r in records]
    ax.plot(x,y,'o-',color='#73529c');ax.set_xscale('log',base=2);ax.set_xticks(x,[str(v) for v in x]);ax.set_ylim(0,35)
    ax.set_xlabel('Pieces per coordinate update',fontsize=10);ax.set_ylabel('Prediction error (%)',fontsize=10)
    ax.set_title('Subdivide the previous quadratic rule',fontsize=11,pad=14);ax.grid(alpha=.2)
    ax.text(.05,.90,f'Error remains {y[0]:.2f}%',transform=ax.transAxes,fontsize=10)
    ax.text(.01,-.23,'Same previous 128 joint states at 10% variation.\nExact finite updates of the same fitted rule.',transform=ax.transAxes,fontsize=8.5,va='top')
    ax=fig.add_subplot(grid[0,2])
    r=result['raster_subdivision'];x=[v['subdivisions'] for v in r];y=[100*v['pooled_error_fraction'] for v in r]
    ax.plot(x,y,'o-',color='#c17d44');ax.set_xscale('log',base=2);ax.set_xticks(x,[str(v) for v in x]);ax.set_ylim(0,max(y)*1.18)
    ax.set_xlabel('Pieces per unit knob step',fontsize=10);ax.set_ylabel('Prediction error (%)',fontsize=10)
    ax.set_title('Repeated pixel resampling',fontsize=11,pad=14);ax.grid(alpha=.2)
    ax.text(.01,-.23,'First eight joint targets; forward schedule.\nBilinear interpolation at every smaller step.',transform=ax.transAxes,fontsize=8.5,va='top')
    fig.text(.035,.100,'Five reusable operations suffice for this known geometric pattern. Their current pivot, height ratio, and lean slope must be updated at every step.',fontsize=10.5)
    fig.text(.035,.067,'Small steps do not remove an incorrect model or restore lost pixel detail. No learning-complexity or general-manifold claim follows from these tests.',fontsize=10.5)
    fig.text(.035,.034,'Targets use 1–4 signed units per knob (5–20% of each default range span). Root position bounds are explicitly extended; strokes remain unclipped.',fontsize=10)
    fig.savefig(OUT/'02_composition_checks.png',dpi=150,facecolor=fig.get_facecolor());plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    result,arrays,history=experiment()
    plot_chain(result,arrays,history);plot_checks(result)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(OUT/'states_and_patterns.npz',**arrays)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
