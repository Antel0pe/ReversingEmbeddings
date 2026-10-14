"""Learn how a finite width-change pattern changes with five state values.

Run from the repository root:
OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/width_pattern_state_transport.py

Known state coordinates are supplied. Fit only root/axis/pair samples, then
measure errors at fresh joint states. No SVD, edge detection, or image fitting.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/width-pattern-state-mpl')
from pathlib import Path
import sys
import itertools
import json
import csv
import numpy as np
from scipy.stats import qmc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from grey_ones import render,RANGES,KNOBS

OUT=ROOT/'figures'/'width_pattern_state_transport'
BASE=np.array([14.,14.,19.75,3.2,0.])
SPANS=.1*np.ptp(RANGES,axis=1)
WIDTH_STEP=.28
SEED=20261014
SCALES=[.25,.5,1.,2.]
COUNT=128
PAIRS=list(itertools.combinations(range(5),2))


def image(p):
    return render(p,dtype=np.float64).reshape(-1,784)


def valid(p):
    p=np.atleast_2d(p)
    reach=p[:,3]/2+p[:,2]/2*np.abs(np.tan(np.deg2rad(p[:,4])))
    return bool(np.all((p[:,2]>0)&(p[:,3]>0)&(p[:,0]-reach>0)&(p[:,0]+reach<28)&
                       (p[:,1]-p[:,2]/2>0)&(p[:,1]+p[:,2]/2<28)))


def width_pattern(x):
    p=BASE+np.atleast_2d(x)*SPANS
    q=p.copy();q[:,3]+=WIDTH_STEP
    assert valid(p) and valid(q)
    return image(q)-image(p)


def learn():
    # A finite-difference interpolation stencil, not regression on joint data.
    origin=width_pattern(np.zeros(5))[0]
    linear=np.empty((5,784));diagonal=np.empty_like(linear)
    train=[np.zeros(5)]
    for i in range(5):
        e=np.eye(5)[i]
        plus=width_pattern(e)[0];minus=width_pattern(-e)[0]
        linear[i]=(plus-minus)/2
        diagonal[i]=(plus+minus-2*origin)/2
        train += [e,-e]
    cross=np.zeros((5,5,784))
    for i,j in PAIRS:
        ei=np.eye(5)[i];ej=np.eye(5)[j]
        corners=np.array([ei+ej,ei-ej,-ei+ej,-ei-ej])
        values=width_pattern(corners)
        cross[i,j]=(values[0]-values[1]-values[2]+values[3])/4
        cross[j,i]=cross[i,j]
        train.extend(corners)
    return origin,linear,diagonal,cross,np.array(train)


def predict(x,model):
    origin,linear,diagonal,cross=model
    x=np.atleast_2d(x)
    frozen=np.broadcast_to(origin,(len(x),784)).copy()
    additive=frozen+x@linear
    quadratic=additive+(x*x)@diagonal
    for i,j in PAIRS:
        quadratic += x[:,i,None]*x[:,j,None]*cross[i,j]
    return dict(frozen=frozen,additive=additive,interaction=quadratic)


def update_pattern(pattern,state,axis,amount,model):
    _,linear,diagonal,cross=model
    response=linear[axis]+2*state[axis]*diagonal[axis]+state@cross[axis]
    return pattern+amount*response+amount*amount*diagonal[axis]


def order_check(test,model):
    # The quadratic finite-update rule should agree with its direct decoder.
    errors=[];endpoints=[]
    for target in test[:16]:
        direct=predict(target,model)['interaction'][0]
        for order in itertools.permutations(range(5)):
            state=np.zeros(5);pattern=model[0].copy()
            for axis in order:
                pattern=update_pattern(pattern,state,axis,target[axis],model)
                state[axis]=target[axis]
            errors.append(np.linalg.norm(pattern-direct))
        endpoints.append(direct)
    return dict(states=16,orders_per_state=120,
                maximum_sequential_vs_direct_pixel_L2=float(max(errors)),
                interpretation='Internal order consistency of the learned quadratic rule; not evidence that its prediction matches the renderer.')


def metrics(truth,prediction):
    diff=prediction-truth
    errors=np.linalg.norm(diff,axis=1)
    norm=np.linalg.norm(truth,axis=1)
    rel=errors/norm
    return dict(pooled_error_fraction=float(np.linalg.norm(diff)/np.linalg.norm(truth)),
                median_state_error_fraction=float(np.median(rel)),
                worst_state_error_fraction=float(rel.max()),
                worst_pixel_absolute_error=float(np.abs(diff).max()),
                states_with_negative_predicted_ink_change=int(np.sum(np.any(prediction < -1e-12,axis=1))),
                negative_prediction_L2=float(np.linalg.norm(np.minimum(prediction,0))),
                minimum_prediction=float(prediction.min()), maximum_prediction=float(prediction.max()))


def plot_height():
    heights=np.array([19.75,19.,20.,20.25,21.])
    p=np.tile(BASE,(len(heights),1));p[:,2]=heights
    q=p.copy();q[:,3]+=WIDTH_STEP
    initial=image(p);changes=image(q)-initial
    fine=np.arange(18.75,21.251,.125)
    fp=np.tile(BASE,(len(fine),1));fp[:,2]=fine
    fq=fp.copy();fq[:,3]+=WIDTH_STEP
    vectors=image(fq)-image(fp)
    delta=np.diff(vectors,axis=0)
    unit=delta/np.linalg.norm(delta,axis=1)[:,None]
    angles=np.rad2deg(np.arccos(np.clip(np.sum(unit[1:]*unit[:-1],axis=1),-1,1)))
    fig=plt.figure(figsize=(13,9.2),facecolor='#fafaf7')
    fig.text(.04,.966,'1 · Straight edge motion can bend the pixel-space path',fontsize=20,weight='bold',va='top')
    fig.text(.04,.914,'Generated 28 × 28 ones: white = no ink, black = full ink. Center (14,14), width 3.2 px, lean 0°; only height varies.',fontsize=10.5)
    fig.text(.04,.883,'Every bottom image shows the pixel changes for the SAME +0.28 px width increase at that height.',fontsize=11)
    grid=fig.add_gridspec(2,5,left=.13,right=.96,top=.82,bottom=.43,wspace=.35,hspace=.30)
    for k,h in enumerate(heights):
        ax=fig.add_subplot(grid[0,k]);ax.imshow(initial[k].reshape(28,28),cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
        ax.set_title(('Root · ' if k==0 else '')+f'height {h:g} px',fontsize=11,pad=8);ax.set_xticks([]);ax.set_yticks([])
        ax=fig.add_subplot(grid[1,k]);im=ax.imshow(changes[k].reshape(28,28),cmap='RdBu_r',vmin=-.14,vmax=.14,interpolation='nearest')
        ax.set_xticks([]);ax.set_yticks([])
    fig.text(.025,.732,'Starting\nimage',fontsize=11,va='center')
    fig.text(.025,.524,'Width-change\npattern',fontsize=11,va='center')
    cbax=fig.add_axes([.30,.370,.42,.018]);cb=fig.colorbar(im,cax=cbax,orientation='horizontal')
    cb.set_label('Ink coverage change: blue loses ink; red gains ink. Common scale; no clipped values.',fontsize=9)
    ax=fig.add_axes([.13,.135,.83,.145])
    ax.plot(fine[1:-1],angles,'o-',color='#70509b',ms=4)
    ax.axvline(20,color='#888888',ls=':',lw=1)
    ax.set_xlim(fine[1],fine[-2]);ax.set_ylim(-3,96);ax.set_yticks([0,45,90])
    ax.set_xlabel('Intermediate height (pixels); equal height steps of 0.125 px',fontsize=10)
    ax.set_ylabel('Turn angle (degrees)',fontsize=10);ax.grid(alpha=.2)
    ax.set_title('Angle between consecutive changes of the WIDTH pattern as height increases',fontsize=10.5,pad=10)
    fig.text(.04,.038,'Within a fixed row region, the width pattern follows a straight segment. At height 20, active cap rows switch and the pixel-space path turns 90°.',fontsize=10)
    fig.text(.04,.015,'This is one upright width–height slice. Heights above 20.5 px extend the default range to show the next pixel-row region.',fontsize=9)
    fig.savefig(OUT/'01_height_pattern.png',dpi=155,facecolor=fig.get_facecolor());plt.close(fig)
    return dict(height_step=.125,maximum_turn_degrees=float(angles.max()),height_at_largest_turn=float(fine[1:-1][angles.argmax()]),
                minimum_turn_degrees=float(angles.min()))


def plot_joint(model,test,truth,predictions,all_metrics):
    error=np.linalg.norm(predictions['interaction']-truth,axis=1)/np.linalg.norm(truth,axis=1)
    median_id=np.argsort(error)[len(error)//2];worst_id=error.argmax()
    height_only=np.zeros(5);height_only[2]=1
    cases=np.array([np.zeros(5),height_only,test[median_id],test[worst_id]])
    p=BASE+cases*SPANS;start=image(p)
    actual=width_pattern(cases);pred=predict(cases,model)['interaction'];resid=pred-actual
    magnitude=float(max(np.abs(actual).max(),np.abs(pred).max()))
    errmag=float(np.abs(resid).max())
    fig=plt.figure(figsize=(15.4,12.8),facecolor='#fafaf7')
    fig.text(.035,.973,'2 · Do learned state-dependent updates predict widening at new joint states?',fontsize=19,weight='bold',va='top')
    fig.text(.035,.934,'Known coordinates supplied; no SVD or image analysis. Fit root + axis + pair samples, then test 128 fresh five-knob combinations per scale.',fontsize=10.5)
    fig.text(.035,.910,'Target at every state: the complete 784-pixel change for width +0.28 px. Predictions are shown without clipping or normalization.',fontsize=10.5)
    grid=fig.add_gridspec(4,4,left=.14,right=.76,top=.85,bottom=.30,wspace=.30,hspace=.18)
    titles=['Root · reference','Height only\ntraining axis endpoint','Fresh joint state\nmedian relative error','Fresh joint state\nworst relative error']
    for j in range(4):
        for i in range(4):
            ax=fig.add_subplot(grid[i,j]);ax.set_xticks([]);ax.set_yticks([])
            data=[start,actual,pred,resid][i][j].reshape(28,28)
            if i==0:
                ax.imshow(data,cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
                ax.set_title(titles[j],fontsize=10,pad=10)
            else:
                im=ax.imshow(data,cmap='RdBu_r',vmin=-(errmag if i==3 else magnitude),vmax=errmag if i==3 else magnitude,interpolation='nearest')
            if i==3:
                rel=np.linalg.norm(resid[j])/np.linalg.norm(actual[j])
                ax.text(.5,-.08,f'Error: {rel*100:.1f}%',transform=ax.transAxes,ha='center',va='top',fontsize=9)
    for y,label in zip([.788,.65,.514,.376],['Starting image','True width\nchange','Predicted width\nchange','Prediction minus\ntrue change']):
        fig.text(.025,y,label,fontsize=10.5,va='center')
    ax=fig.add_axes([.82,.45,.15,.36])
    for key,label,color in [('frozen','Root pattern','#878787'),('additive','Additive updates','#3669a2'),('interaction','With interactions','#ba612a')]:
        ax.plot([s*10 for s in SCALES],[all_metrics[str(s)][key]['pooled_error_fraction']*100 for s in SCALES],'o-',label=label,color=color,lw=1.7)
    ax.set_xticks([2.5,5,10,20]);ax.set_xlabel('State variation (% of\neach knob default span)',fontsize=9)
    ax.set_ylabel('Pooled prediction error (% of true width effect)',fontsize=9)
    ax.set_title('All joint tests',fontsize=11,pad=10);ax.grid(alpha=.2);ax.legend(fontsize=8,frameon=False,loc='upper left')
    cax=fig.add_axes([.82,.355,.15,.014]);cbar=fig.colorbar(plt.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-magnitude,vcenter=0,vmax=magnitude),cmap='RdBu_r'),cax=cax,orientation='horizontal')
    cbar.set_label('True / prediction scale',fontsize=8);cbar.ax.tick_params(labelsize=8)
    cax=fig.add_axes([.82,.292,.15,.014]);cbar=fig.colorbar(plt.cm.ScalarMappable(norm=TwoSlopeNorm(vmin=-errmag,vcenter=0,vmax=errmag),cmap='RdBu_r'),cax=cax,orientation='horizontal')
    cbar.set_label('Error scale (separate)',fontsize=8);cbar.ax.tick_params(labelsize=8)
    fig.text(.14,.222,'Coordinates below are relative to root: (horizontal px, vertical px, height px, width px, lean degrees).',fontsize=10)
    for j in range(4):
        delta=cases[j]*SPANS
        fig.text(.14+j*.21,.190, f'Case {j+1}: ('+', '.join(f'{v:+.3f}' for v in delta)+')',fontsize=8)
    fig.text(.035,.132,'At 10% variation: root pattern error 47.5%; additive rule 41.2%; interaction rule 26.4%. Worst joint interaction-rule error: 43.6%.',fontsize=11)
    fig.text(.035,.102,'A broad fitted rule can be worse near the root: at 2.5% variation, root error is 0.6% and interaction-rule error is 7.5%.',fontsize=10.5)
    fig.text(.035,.071,'Blue in a predicted width change is erroneous ink loss: increasing width alone never removes ink. Error colors describe prediction error, not direct ink change.',fontsize=9.5)
    fig.text(.035,.043,'One local quadratic model, fixed settings. Coefficients are fixed pixel arrays; mixed terms make each update depend on the other current coordinates.',fontsize=9.5)
    fig.savefig(OUT/'02_joint_prediction.png',dpi=150,facecolor=fig.get_facecolor());plt.close(fig)
    return dict(median_case_index=int(median_id),worst_case_index=int(worst_id),
                image_scale=magnitude,error_scale=errmag,
                pictured_coordinate_offsets=(cases*SPANS).tolist())


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    origin,linear,diagonal,cross,train=learn();model=(origin,linear,diagonal,cross)
    rays=2*qmc.Sobol(5,scramble=True,seed=SEED).random_base2(7)-1
    all_metrics={};arrays={};rows=[]
    for scale in SCALES:
        x=rays*scale;truth=width_pattern(x);predictions=predict(x,model)
        all_metrics[str(scale)]={key:metrics(truth,value) for key,value in predictions.items()}
        arrays[f'states_{scale}']=x;arrays[f'truth_{scale}']=truth
        for key,value in predictions.items():
            arrays[f'{key}_{scale}']=value
            norm=np.linalg.norm(truth,axis=1);errors=np.linalg.norm(value-truth,axis=1)/norm
            for i in range(COUNT):
                rows.append(dict(scale=scale,index=i,model=key,relative_error=float(errors[i])))
    consistency=order_check(rays,model)
    height=plot_height()
    joint=plot_joint(model,rays,arrays['truth_1.0'],predict(rays,model),all_metrics)
    result=dict(method=dict(root=BASE.tolist(),knob_order=KNOBS,state_spans=SPANS.tolist(),
                            state_span_definition='10 percent of each default range width, centered at canonical root; position bounds explicitly extended',
                            width_increment=WIDTH_STEP,train_states=len(train),training_maximum_changed_coordinates=2,
                            target='finite width effect render(state + 0.28 width) - render(state)',
                            model='quadratic function of five normalized current state coordinates, including current width',
                            coefficient_arrays=21,training_images=2*len(train),test_states_per_scale=COUNT,
                            test_scales=SCALES,seed=SEED,precision='float64 renderer coverage',
                            fit='symmetric axis and pair finite-difference stencil; no joint holdout data used for coefficients',
                            comparisons='one quadratic transport model, with its additive truncation and fixed-root ablations',
                            scope='known coordinates and true finite-width patterns; no learned coordinates, no SVD purity or tracking test',
                            clipping='none; signed and negative predictions are retained and measured'),
                metrics=all_metrics,order_consistency=consistency,height_example=height,pictured_cases=joint,
                verification=dict(train_shape=list(train.shape),all_test_arrays_finite=all(np.isfinite(v).all() for v in arrays.values()),
                                  true_width_change_minimum=float(min(arrays[f'truth_{s}'].min() for s in SCALES))))
    assert len(train)==51 and result['verification']['all_test_arrays_finite']
    assert result['verification']['true_width_change_minimum'] >= -1e-12
    assert consistency['maximum_sequential_vs_direct_pixel_L2'] < 1e-12
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'joint_errors.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    np.savez_compressed(OUT/'patterns_and_model.npz',origin=origin,linear=linear,diagonal=diagonal,cross=cross,
                        training_states=train,training_patterns=width_pattern(train),**arrays)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
