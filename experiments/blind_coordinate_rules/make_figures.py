"""Display/evaluation only: restore image layout after numeric rule discovery."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/blind-coordinate-rule-mpl')
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

OUT=Path(__file__).resolve().parents[2]/'figures'/'blind_coordinate_rules'
RESULTS=json.loads((OUT/'results.json').read_text())
NAMES=['height_width','width_lean','all_five']
LABELS=['Height + width slice','Width + lean slice','All five controls']
EXTRA='#d629a0'; MISSED='#00a6c7'; OOR='#e5a400'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12})


def examples():
    chosen=[]
    for name in NAMES:
        a=np.load(OUT/f'{name}_evaluation.npz')
        width_axis=1 if name=='height_width' else 0 if name=='width_lean' else 3
        candidates=np.flatnonzero(a['labels']==width_axis)
        relative=np.linalg.norm(a['predicted_delta']-a['truth_delta'],axis=1)/np.linalg.norm(a['truth_delta'],axis=1)
        index=candidates[np.argsort(relative[candidates])[len(candidates)//2]]
        chosen.append((a,int(index),float(relative[index])))
    return chosen


def make_updates():
    selected=examples()
    scale=max(np.max(np.abs(a[key][i])) for a,i,_ in selected for key in ['truth_delta','predicted_delta'])
    fig=plt.figure(figsize=(19,11),facecolor='white')
    fig.text(.04,.976,'Which pixels does the blind numeric rule change?',fontsize=20,weight='bold',va='top')
    fig.text(.04,.94,'Generated 1s: 28 × 28 ink coverage, 0 = white, 1 = black. Every row applies width +0.28 pixels.',fontsize=12)
    fig.text(.04,.915,'Each example is the median-error width update in its test set. Control names and image layout were revealed only after fitting.',fontsize=11)
    cols=['Source image','Actual pixel change','Rule’s pixel change','Actual next image','Source + rule change','Changing-index errors']
    x0=.04; step=.159; width=.135
    for j,title in enumerate(cols):
        fig.text(x0+j*step+width/2,.878,title,ha='center',weight='bold',fontsize=11)
    example_records=[]
    for row,((a,i,relative),name,label) in enumerate(zip(selected,NAMES,LABELS)):
        y=.655-row*.272
        source=a['source'][i].reshape(28,28)
        truth=a['truth_delta'][i].reshape(28,28)
        pred=a['predicted_delta'][i].reshape(28,28)
        actual=a['truth_new'][i].reshape(28,28)
        inferred=source+pred
        true_support=np.abs(truth)>1e-7;pred_support=np.abs(pred)>1e-7
        fp=pred_support & ~true_support;fn=true_support & ~pred_support
        errors=np.ones((28,28,3))
        errors[true_support & pred_support]=[.65,.65,.65]
        errors[fp]=matplotlib.colors.to_rgb(EXTRA)
        errors[fn]=matplotlib.colors.to_rgb(MISSED)
        panels=[source,truth,pred,actual,inferred,errors]
        outside=(inferred < -1e-7)|(inferred > 1+1e-7)
        for j,panel in enumerate(panels):
            ax=fig.add_axes([x0+j*step,y,width,.19])
            if j in [1,2]:ax.imshow(panel,cmap='RdBu_r',vmin=-scale,vmax=scale,interpolation='nearest')
            elif j==5:ax.imshow(panel,interpolation='nearest')
            else:ax.imshow(panel,cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
            if j==4 and outside.any():
                overlay=np.zeros((28,28,4));overlay[outside]=[*matplotlib.colors.to_rgb(OOR),1.]
                ax.imshow(overlay,interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_color('#888888')
        fig.text(x0,y+.204,label,weight='bold',fontsize=12)
        if name=='height_width':
            note='Centers (14,14), lean 0°; vary height and width.'
        elif name=='width_lean':
            note='Centers (14,14), height 19.75; vary width and lean.'
        else:note='All five values vary; this panel changes width only.'
        fig.text(x0,y-.016,note,fontsize=9,va='top')
        fig.text(x0+2*step+width/2,y-.018,f'Change error: {100*relative:.3g}%\nLargest pixel error: {np.abs(pred-truth).max():.3g}',ha='center',va='top',fontsize=10)
        fig.text(x0+4*step+width/2,y-.018,f'{int(outside.sum())} values outside [0,1]\nGold marks clipped display values',ha='center',va='top',fontsize=9)
        fig.text(x0+5*step+width/2,y-.018,f'{int(fp.sum())} extra; {int(fn.sum())} missed\nThreshold: |change| > 0.0000001',ha='center',va='top',fontsize=9)
        example_records.append(dict(case=name,evaluation_index=i,source_numeric_coordinates=a['starts'][i].tolist(),
            action_numeric_control=int(a['labels'][i]),relative_change_error=relative,
            false_indices=int(fp.sum()),missed_indices=int(fn.sum())))
    bar=fig.add_axes([.22,.041,.34,.011])
    cb=fig.colorbar(ScalarMappable(norm=Normalize(-scale,scale),cmap='RdBu_r'),cax=bar,orientation='horizontal')
    cb.set_ticks([-scale,0,scale]);cb.set_ticklabels([f'−{scale:.3g}','0',f'+{scale:.3g}']);cb.ax.tick_params(labelsize=9)
    fig.text(.04,.045,'All change panels share this scale:',fontsize=10)
    fig.text(.39,.008,'Blue = ink lost; red = ink gained',fontsize=10,ha='center')
    handles=[Patch(color='.65',label='Correct changed index'),Patch(color=EXTRA,label='Extra index'),Patch(color=MISSED,label='Missed index')]
    fig.legend(handles=handles,loc='lower right',bbox_to_anchor=(.985,.013),ncol=3,frameon=False,fontsize=10)
    fig.savefig(OUT/'01_pixel_updates.png',dpi=150)
    plt.close(fig)
    (OUT/'displayed_examples.json').write_text(json.dumps(example_records,indent=2))


def make_structure():
    fig=plt.figure(figsize=(17,10),facecolor='white')
    fig.text(.045,.977,'What structure was discovered without knowing the image geometry?',fontsize=19,weight='bold',va='top')
    fig.text(.045,.938,'The height/width slice yields six explicit bilinear regions. The harder cases require many more stored patterns.',fontsize=12)
    r=RESULTS['height_width']
    h,w=map(np.asarray,r['physical_knots'])
    ax=fig.add_axes([.07,.57,.28,.295])
    for x in w:ax.axvline(x,color='#8a9bab',lw=1)
    for y in h:ax.axhline(y,color='#8a9bab',lw=1)
    xx,yy=np.meshgrid(w,h);ax.scatter(xx,yy,s=38,c='#184969',zorder=3)
    ax.set_xlim(1.65,4.75);ax.set_ylim(18.9,20.62)
    ax.set_xticks(w);ax.set_yticks(h)
    ax.set_xlabel('Control D, revealed as width (pixels)')
    ax.set_ylabel('Control C, revealed as height (pixels)')
    ax.set_title('1. Numerically discovered breakpoints',loc='left',pad=16)
    fig.text(.07,.485,f'12 retained reference states (dots); {r["oracle_training_states"]:,} samples used to find them.\nBreaks at C = 20 and D = 2, 4 were not supplied.',fontsize=10)
    cells=json.loads((OUT/'height_width_explicit_cells.json').read_text())['cells']
    cell=next(c for c in cells if c['cell']==[0,1])
    vectors={}
    for term in cell['terms']:
        values=np.zeros(784);values[term['output_indices']]=term['coefficients'];vectors[term['name']]=values.reshape(28,28)
    scale=max(np.max(abs(vectors['second_control'])),np.max(abs(vectors['interaction'])))
    panels=[('second_control',.43,'2. D-effect at C = 19'),('interaction',.74,'3. Extra D-effect when C rises to 20')]
    for key,x,title in panels:
        ax=fig.add_axes([x,.57,.20,.295])
        ax.imshow(vectors[key],cmap='RdBu_r',vmin=-scale,vmax=scale,interpolation='nearest')
        ax.set_xticks([]);ax.set_yticks([]);ax.set_title(title,pad=16,fontsize=12)
    fig.text(.43,.505,'Change for D: 2 → 4, C held at 19.\nRed = ink gained; blue = ink lost.',fontsize=10)
    fig.text(.74,.505,'Change TO that pixel-change pattern.\nThis is an interaction, not direct ink gain.',fontsize=10)
    bar=fig.add_axes([.51,.462,.32,.012])
    cb=fig.colorbar(ScalarMappable(norm=Normalize(-scale,scale),cmap='RdBu_r'),cax=bar,orientation='horizontal')
    cb.set_ticks([-scale,0,scale]);cb.ax.tick_params(labelsize=8)
    fig.text(.67,.419,'Within this region: image = base + u × C-effect + v × D-effect + u × v × interaction',ha='center',fontsize=11,weight='bold')
    fig.text(.67,.393,'u = height − 19; v = (width − 2) / 2. Every coefficient is an explicit array of measured pixel values.',ha='center',fontsize=10)
    fig.text(.05,.347,'Held-out updates: the full five-control case checks all five actions, not just the displayed width example.',fontsize=11,weight='bold')
    table_ax=fig.add_axes([.04,.128,.92,.192]);table_ax.axis('off')
    rows=[]
    for name,label in zip(NAMES,LABELS):
        q=RESULTS[name];m=q['metrics']
        err=m['aggregate_relative_change_error']*100
        err_text='< 0.00000001%' if err<1e-8 else f'{err:.2f}%'
        rows.append([label,f'{m["updates"]:,}',
            f'{m["mean_false_indices"]:.2f} / {m["mean_missed_indices"]:.2f}',
            f'{100*m["exact_support_fraction"]:.1f}%',err_text,
            f'{q["table_states"]:,} / {q["output_pattern_count"]}'])
    columns=['Case','Test updates','Extra / missed indices\nper update','Exact index sets','Change error¹','Stored states /\nfixed patterns']
    table=table_ax.table(cellText=rows,colLabels=columns,loc='center',cellLoc='center',
                        colWidths=[.21,.115,.20,.14,.15,.185])
    table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,2.55)
    for (i,j),cell in table.get_celld().items():
        cell.set_edgecolor('#c7d0d8')
        if i==0:cell.set_facecolor('#e5edf3');cell.set_text_props(weight='bold')
        elif i==1:cell.set_facecolor('#eef8ef')
        else:cell.set_facecolor('#fff7ed')
    fig.text(.05,.078,'¹ Change error = combined Euclidean error / combined true change over the held-out updates. Index threshold: 0.0000001.',fontsize=10)
    fig.text(.05,.050,'All five controls used a coarse 5 × 5 × 5 × 5 × 5 grid (sampling budget). This failure does not establish impossibility.',fontsize=10)
    fig.text(.05,.022,'SVD compression error < 2 × 10⁻¹³ per pixel in every case. Reordering outputs preserved all discovered knots and sample choices.',fontsize=10)
    fig.savefig(OUT/'02_discovered_structure.png',dpi=150)
    plt.close(fig)


if __name__=='__main__':
    make_updates();make_structure()
    print('Saved both explanatory figures.')
