"""Explanatory figures for direction history and count calibration."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/generated-one-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_generated_one_direction_history import OUT
from grey_ones import render


def pixel_image(ax,img,title,cmap='gray_r',limit=None):
    if limit is None:shown=ax.imshow(img.reshape(28,28),cmap=cmap,vmin=0,vmax=1,interpolation='nearest')
    else:shown=ax.imshow(img.reshape(28,28),cmap=cmap,vmin=-limit,vmax=limit,interpolation='nearest')
    ax.set_xticks([]);ax.set_yticks([]);ax.set_title(title,fontsize=10,pad=9)
    return shown


def patterns(summary):
    coeff=np.load(OUT/'bilinear_coefficients.npz')['coefficients']
    p=np.array([14.,14.,19.3,2.4,0.]);next_p=p.copy();next_p[3]=3.6
    old=coeff[1]+.2*coeff[3];predicted=coeff[1]+1.4*coeff[3]
    tall=next_p.copy();tall[2]+=.4
    actual=(render(tall,dtype=np.float64)-render(next_p,dtype=np.float64)).ravel()
    limit=max(abs(old).max(),abs(predicted).max(),abs(actual).max())
    fig=plt.figure(figsize=(14,10),facecolor='white')
    fig.text(.035,.975,'Can a direction’s change pattern help keep its identity?',fontsize=21,weight='bold',va='top')
    fig.text(.035,.923,
             'Top: height/width slice, start height 19.3 px, fixed center (14,14), no lean. Bottom: separate history test across all five controls.\n'
             'Controlled generated 1s; white = no ink, black = full ink. Signed change maps use a common scale: red gains ink, blue loses ink.',
             fontsize=10.5,va='top',linespacing=1.5)
    values=[render(p)[0],render(next_p)[0],old,predicted,actual]
    titles=['Starting 1: width 2.4','After width → 3.6','Height effect at width 2.4',
            'Predicted height effect at 3.6','Measured height effect at 3.6']
    last=None
    for i,(value,title) in enumerate(zip(values,titles)):
        ax=fig.add_axes([.045+i*.195,.67,.13,.18])
        last=pixel_image(ax,value,title,limit=None if i<2 else limit,cmap='gray_r' if i<2 else 'RdBu_r')
    fig.text(.035,.60,
             'Four images fit F(u,v) = A + Bu + Cv + Duv; u = (height − 19.1)/0.4, v = width − 2.2.\n'
             'The height effect is B + Dv: maps show the image change for +0.4 px height. This law predicts the whole pixel pattern.\n'
             f"On 1,024 unseen joint height/width settings: maximum image L2 error {summary['bilinear_control']['max_heldout_L2_error']:.2e}.\n"
             'Exact upright patch: 18 < height < 20 px, 2 < width < 4 px. Crossing its boundaries requires another formula.',
             fontsize=10.5,va='top',linespacing=1.5)
    cax=fig.add_axes([.68,.645,.27,.012])
    cb=fig.colorbar(last,cax=cax,orientation='horizontal');cb.set_label('Pixel ink change for +0.4 px height',fontsize=8)
    colors={'nearest':'#087aa4','trend':'#cc6735'}
    for left,field,title in [(.08,'true','Actual five physical directions'),(.56,'svd','Five ordinary SVD directions')]:
        ax=fig.add_axes([left,.21,.38,.27])
        for method,label in [('nearest','Nearest to previous vector'),('trend','Nearest to predicted trend')]:
            rows=[x for x in summary['aggregate'] if x['field']==field and x['mode']=='rollout' and
                  x['scope']=='hop8' and x['method']==method]
            ax.plot([x['step'] for x in rows],[100*x['correct']/x['count'] for x in rows],
                    color=colors[method],marker='o',label=label)
        ax.set_xscale('log');ax.set_xticks([.01,.03,.1,.3],['0.01','0.03','0.1','0.3'])
        ax.set_ylim(90,100.5);ax.axhline(100,color='#777',lw=.7,ls=':');ax.grid(alpha=.15)
        ax.set_xlabel('Image distance per hop');ax.set_ylabel('Correct associations after hop 8 (%)')
        ax.set_title(title,fontsize=12);ax.legend(loc='lower left',fontsize=9)
        if field=='true':
            ax.text(.03,.43,'Both curves coincide',transform=ax.transAxes,fontsize=9,color='#555')
    fig.text(.035,.145,
             'Trend prediction = normalize(2 × last unit vector − previous unit vector); the first hop uses ordinary nearest matching.\n'
             '19 starts: original 17 plus two prior boundary failures. Each path moves one true control for up to eight equal-distance hops.\n'
             'All five families are tracked without correcting past mistakes. 950 / 950 / 885 / 710 associations reach hop 8 at the four step sizes.\n'
             'SVD: 256 equal-weight neighbors, radius 0.00001, fresh seed at each state; same fits for both rules. Labels only evaluate matches.\n'
             'Matching uses absolute pixel cosine; signs are aligned before extrapolation. SVD names are associations and can mix physical controls.',
             fontsize=9.2,va='top',linespacing=1.45)
    fig.savefig(OUT/'direction_patterns.png',dpi=150);plt.close(fig)


def counts(summary):
    result=summary['count_order_control'];row=next(x for x in result['rows'] if x['step']==.1)
    start=np.array(result['start_settings']);paths=[np.array(p) for p in row['trajectories']]
    truth=[render(p,dtype=np.float64).reshape(-1,784) for p in paths]
    difference=truth[0][-1]-truth[1][-1];limit=np.abs(difference).max()
    fig=plt.figure(figsize=(13.5,9),facecolor='white')
    fig.text(.035,.97,'Do identical knob counts determine the same image?',fontsize=22,weight='bold',va='top')
    fig.text(.035,.905,
             'Exact true height/width flows, fixed center (14,14), no lean. Start: height 19.3 px, width 3.2 px.\n'
             'One tick moves 0.1 units of image arc length along the current knob. Both routes count height +1 and width +1.\n'
             'Images: white = 0 ink, black = 1. Every nearest-direction association on both routes is correct.',
             fontsize=11,va='top',linespacing=1.5)
    titles=[['Start','Height +1','Then width +1'],['Start','Width +1','Then height +1']]
    for i,path in enumerate(paths):
        for k,p in enumerate(path):
            ax=fig.add_axes([.06+k*.215,.63-i*.27,.13,.17])
            pixel_image(ax,truth[i][k],titles[i][k])
            ax.text(.5,-.09,f'h = {p[2]:.7f}\nw = {p[3]:.7f}',transform=ax.transAxes,
                    ha='center',va='top',fontsize=9)
            if k<2:fig.text(.218+k*.215,.715-i*.27,'→',fontsize=23,ha='center')
    ax=fig.add_axes([.77,.50,.15,.20])
    shown=pixel_image(ax,difference,'Endpoint difference','RdBu_r',limit)
    ax.text(.5,-.13,f"Image L2 = {row['image_order_difference']:.7f}",transform=ax.transAxes,ha='center',fontsize=10)
    cax=fig.add_axes([.74,.435,.22,.015]);cb=fig.colorbar(shown,cax=cax,orientation='horizontal',
                                                     ticks=[-limit,-limit/2,0,limit/2,limit],format='%.1e')
    cb.ax.tick_params(labelsize=8)
    cb.set_label('First route − second route: pixel ink difference',fontsize=8)
    fig.text(.735,.37,'Difference colors magnify small values.\nRed = more ink after height then width.\nBoth routes stay inside the upright\nbilinear patch; no numerical stepping\napproximation or boundary crossing.',fontsize=9,va='top',linespacing=1.45)
    ax=fig.add_axes([.08,.09,.42,.18])
    rows=result['rows'];x=np.array([r['step'] for r in rows]);y=np.array([r['image_order_difference'] for r in rows])
    ax.loglog(x,y,marker='o',color='#8e4da2',label='Measured order difference')
    ax.loglog(x,y[0]*(x/x[0])**2,ls='--',color='#777',label='Quadratic reference, anchored at first point')
    ax.set_xlabel('Image arc length per tick');ax.set_ylabel('Endpoint image L2 difference')
    ax.legend(fontsize=7.7);ax.grid(alpha=.15)
    fig.text(.57,.23,
             'Height and width remain the correct families.\n'
             'The step units change with the other knob, so counts depend on order.\n'
             'Control: fixed physical height/width increments commute exactly\n'
             '(measured endpoint image difference = 0).\n'
             'A coordinate description therefore needs a consistent calibration\n'
             'and an order-consistency check, in addition to matching labels.',
             fontsize=10.2,va='top',linespacing=1.5)
    fig.savefig(OUT/'same_counts_different_states.png',dpi=150);plt.close(fig)


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    patterns(summary);counts(summary)


if __name__=='__main__':main()
