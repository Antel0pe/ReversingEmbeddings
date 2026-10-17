"""Explain the full-family failure without hiding invalid pixel predictions."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/data-only-experiment-mpl')
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/generated_one_data_only'


def main():
    summary=json.loads((OUT/'results.json').read_text())
    with np.load(OUT/'evaluation_public.npz') as data:
        images=data['images'];tags=data['tags']
    predictions=np.load(OUT/'predictions.npz')
    index=summary['counts']['32768']['images_only']['worst_l2_index']
    truth=images[index]
    a=predictions['images_only'][index];b=predictions['supplied_coordinates'][index]
    errors=[a-truth,b-truth]
    limit=max(float(np.max(abs(e))) for e in errors)
    fig=plt.figure(figsize=(15,9),facecolor='white')
    fig.text(.05,.965,'Do data-only equations reconstruct valid images?',fontsize=21,weight='bold',va='top')
    fig.text(.05,.913,'One generic local-polynomial method; no stroke equations. 32,768 training images; 3,114 held-out states.',fontsize=12)
    fig.text(.05,.879,f'Example: worst images-only reconstruction, held-out state {index} ({tags[index]}). All 784 pixels are shown.',fontsize=11)
    top=[]
    for j,(im,name) in enumerate([(truth,'Observed image'),(a,'Images only'),(b,'Five coordinates supplied')]):
        ax=fig.add_axes([.055+j*.30,.535,.235,.285])
        plotted=ax.imshow(im,cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
        invalid=(im<-2e-6)|(im>1+2e-6)
        overlay=np.zeros((28,28,4));overlay[invalid]=[.9,0,.7,1]
        ax.imshow(overlay,interpolation='nearest')
        if j==0:caption='Reference: raw ink coverage'
        else:
            difference=im-truth
            caption=f'L2 error {np.linalg.norm(difference):.3f}; max pixel {np.max(abs(difference)):.3f}'
        ax.set_title(name,fontsize=13,pad=12)
        ax.text(.5,-.29,caption,ha='center',transform=ax.transAxes,fontsize=10)
        ax.set_xticks([0,14,27]);ax.set_yticks([0,14,27])
        ax.set_xlabel('Pixel column',fontsize=9)
        if j==0:ax.set_ylabel('Pixel row (downward)',fontsize=9)
        top.append(ax)
    cax=fig.add_axes([.943,.55,.012,.25])
    cb=fig.colorbar(plotted,cax=cax);cb.set_ticks([0,.5,1]);cb.set_label('Ink coverage: 0 none, 1 full',fontsize=10)
    for j,(difference,name) in enumerate(zip(errors,['Images-only pixel error','Supplied-coordinate pixel error'])):
        ax=fig.add_axes([.055+j*.30,.145,.235,.255])
        signed=ax.imshow(difference,cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
        ax.set_title(name,fontsize=12,pad=10)
        ax.set_xticks([0,14,27]);ax.set_yticks([0,14,27]);ax.set_xlabel('Pixel column',fontsize=9)
        if j==0:ax.set_ylabel('Pixel row (downward)',fontsize=9)
    cax=fig.add_axes([.622,.155,.012,.23])
    cb=fig.colorbar(signed,cax=cax);cb.set_label('Red: excess ink; blue: missing ink',fontsize=8)
    cb.ax.yaxis.set_label_position('left')
    ax=fig.add_axes([.755,.17,.215,.215])
    for condition,color,label in [('images_only','#1476aa','Images only'),('supplied_coordinates','#cf7623','Coordinates supplied')]:
        n=[int(k) for k in summary['counts']]
        e=[summary['counts'][str(i)][condition]['max_image_l2'] for i in n]
        ax.plot(n,e,'o-',label=label,color=color,lw=2)
    ax.axhline(1e-5,color='#267b3c',ls='--',lw=1.5,label='Required: <= 0.00001')
    ax.set_xscale('log');ax.set_yscale('log');ax.set_ylim(1e-6,30)
    ax.set_xlabel('Number of training images',fontsize=10)
    ax.set_ylabel('Worst image L2 error',fontsize=10)
    ax.set_title('All 3,114 held-out states',fontsize=11,pad=12)
    ax.grid(alpha=.2);ax.legend(fontsize=8,loc='lower left')
    fig.text(.05,.064,'Magenta marks decoded coverage outside [0,1] by more than 0.000002; the grayscale background clips those values.',fontsize=10)
    fig.text(.05,.032,'Both signed-error panels use the same scale. Five readable measurements and optional coordinates did not yield a valid global decoder.',fontsize=10)
    fig.savefig(OUT/'whole_family_failure.png',dpi=150)
    plt.close(fig)

if __name__=='__main__':main()
