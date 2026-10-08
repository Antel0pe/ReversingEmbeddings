"""Explanatory figures and a self-contained viewer using Python experiment data."""
from pathlib import Path
import base64
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from run import BASE, OUT, field, axis_blend, move, render, score


def pixel_panel(ax, values, scale=None, image=False):
    a=np.asarray(values).reshape(28,28)
    if image:ax.imshow(a,cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
    else:ax.imshow(a,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-scale,vcenter=0,vmax=scale),interpolation='nearest')
    ax.set_xticks([0,14,27]);ax.set_yticks([0,14,27]);ax.tick_params(labelsize=7)
    ax.set_xlabel('pixel column',fontsize=8);ax.set_ylabel('pixel row (down)',fontsize=8)
    return ax.images[0]


def figures():
    m=json.loads((OUT/'metrics.json').read_text())
    d=np.load(OUT/'width_panels.npz');ar=d['arrows'];diff=ar-ar[0]
    fig=plt.figure(figsize=(15,11.3),facecolor='white')
    fig.suptitle('How does width change the lean direction?',fontsize=20,fontweight='bold',y=.978)
    fig.text(.5,.936,'Controlled generated 1: center (14.5, 14.5) px, height 19.75 px, starting lean 0°. All 784 pixels are shown.',ha='center',fontsize=10.5)
    fig.text(.5,.907,'Read top to bottom: ink coverage → local pixel changes per degree → change in that pattern from the reference.',ha='center',fontsize=10.5)
    gs=fig.add_gridspec(3,5,left=.065,right=.91,top=.855,bottom=.18,hspace=.53,wspace=.4)
    vm=np.abs(ar).max();dm=np.abs(diff).max()
    for j,p in enumerate(d['settings']):
        ax=fig.add_subplot(gs[0,j]);pixel_panel(ax,d['images'][j],image=True)
        ax.set_title(('REFERENCE\n' if j==0 else '')+f'Width {p[3]:g} px',fontsize=11)
        ax.text(.5,-.29,'0 = no ink; 1 = full ink',transform=ax.transAxes,ha='center',fontsize=8)
        ax=fig.add_subplot(gs[1,j]); im=pixel_panel(ax,ar[j],scale=vm)
        ax.set_title('Local lean pattern',fontsize=10)
        ax.text(.5,-.29,f'Vector length {np.linalg.norm(ar[j]):.3f} /degree',transform=ax.transAxes,ha='center',fontsize=8.5)
        ax=fig.add_subplot(gs[2,j]);di=pixel_panel(ax,diff[j],scale=dm)
        ax.set_title('Pattern minus reference',fontsize=10)
        ax.text(.5,-.29,f'Difference length {np.linalg.norm(diff[j]):.3f} /degree',transform=ax.transAxes,ha='center',fontsize=8.5)
    ca=fig.add_axes([.925,.44,.012,.18]);fig.colorbar(im,cax=ca).set_label('coverage change /degree',fontsize=9)
    ca=fig.add_axes([.925,.2,.012,.18]);fig.colorbar(di,cax=ca).set_label('change in the pattern /degree',fontsize=9)
    fig.text(.07,.105,'Lean-pattern row: red gains ink, blue loses ink. Difference row: red/blue changes the pattern, not ink directly.',fontsize=10)
    fig.text(.07,.073,'Widths 1.8 and 2.6 share one pattern; 3.2 and 4.6 share another. At width 3, the two one-sided patterns differ.',fontsize=10)
    fig.text(.07,.043,'Width 3 shows their average; a single two-sided derivative does not exist there. Widening moves each continuous edge by half the width change.',fontsize=9.7)
    fig.text(.07,.017,'Fixed upright slice. Full-space check: 1,282 random/lattice/event states; edge transport reproduced finite moves with maximum pixel error 5.95 × 10⁻⁸.',fontsize=9,color='#555')
    fig.savefig(OUT/'width_patterns.png',dpi=120);plt.close(fig)
    j=np.load(OUT/'joint_states.npz');settings=j['settings'];truth=j['truth'];add=j['additive']
    errors=np.linalg.norm(add-truth,axis=1)/np.linalg.norm(truth,axis=1)
    ix=int(np.argsort(np.abs(errors-np.median(errors)))[0]);p=settings[ix]
    example=[render(p)[0],truth[ix],add[ix],add[ix]-truth[ix]]
    fig=plt.figure(figsize=(14,8.5),facecolor='white')
    fig.suptitle('Can separate knob effects predict a joint lean direction?',fontsize=19,fontweight='bold',y=.975)
    fig.text(.5,.928,f'Example: center ({p[0]:.2f}, {p[1]:.2f}) px; height {p[2]:.2f} px; width {p[3]:.2f} px; lean {p[4]:.2f}°.',ha='center',fontsize=10.5)
    gs=fig.add_gridspec(2,4,left=.065,right=.9,top=.855,bottom=.19,hspace=.5,wspace=.42,height_ratios=[1,1])
    scale=max(np.abs(a).max() for a in example[1:])
    titles=['Starting generated image','Reference lean direction','Add separate axis changes','Prediction minus reference']
    for c,a in enumerate(example):
        ax=fig.add_subplot(gs[0,c]);im=pixel_panel(ax,a,scale=scale,image=c==0);ax.set_title(titles[c],fontsize=10)
        label='coverage: 0 no ink, 1 full ink' if c==0 else (f'{errors[ix]*100:.1f}% relative vector error' if c==3 else f'length {np.linalg.norm(a):.3f} /degree')
        ax.text(.5,-.28,label,transform=ax.transAxes,ha='center',fontsize=8.5)
    ca=fig.add_axes([.925,.58,.012,.23]);fig.colorbar(im,cax=ca).set_label('coverage change /degree',fontsize=9)
    ax=fig.add_subplot(gs[1,:2])
    ax.hist(errors*100,bins=24,color='#376fa0',edgecolor='white')
    ax.set(xlabel='relative vector error (%)',ylabel='number of joint states',title='512 held-out states across the five-knob box')
    ax.axvline(errors.mean()*100,color='#bc4a32',lw=2,label=f'mean {errors.mean()*100:.1f}%');ax.legend(fontsize=9);ax.grid(axis='y',alpha=.15)
    ax=fig.add_subplot(gs[1,2:]);ax.axis('off')
    txt=('PERFECT SINGLE-AXIS INPUTS\n'
         'All axis arrows use the same current lean angle.\n\n'
         f'Add separate changes: {errors.mean()*100:.1f}% mean vector error\n'
         f'Average separate arrows: {m["independent_axis_combinations"]["full_five_knob_box"]["average_separate_axis_arrows"]["mean_relative_error"]*100:.1f}%\n\n'
         'Compose continuous geometry first:\n'
         f'{m["learn_separate_geometry_transforms"]["heldout_count"]:,} held-out finite lean moves\n'
         f'largest pixel error {m["learn_separate_geometry_transforms"]["max_absolute_pixel_error"]:.2e}\n'
         'after calibration on 95 single-axis images.')
    ax.text(0,1,txt,va='top',fontsize=10.3,linespacing=1.35)
    fig.text(.07,.105,'Vector error = length of (predicted − reference) / length of reference, using all 784 coverage values.',fontsize=10)
    fig.text(.07,.075,'Red/blue in the direction panels means gain/loss of ink. In the residual panel it means over/underprediction of that pattern.',fontsize=9.5)
    fig.text(.07,.045,'Calibration supplies image-to-geometry recovery and the edge-to-pixel formula. This is a test of composition, not discovery of a new coordinate system.',fontsize=9.2)
    fig.text(.07,.017,'The example is closest to median error. The histogram shows the full sampled control. Shared direction/residual scale; no values clipped.',fontsize=9,color='#555')
    fig.savefig(OUT/'joint_composition.png',dpi=120);plt.close(fig)


def viewer():
    widths=[1.8,2.,2.6,3.,3.2,3.3,3.8,4.2,4.6]
    angles=[-10.,-5.,0.,.25,.5,1.,2.,5.,10.,12.5,20.,30.,34.,35.]
    presets=[('Reference',14.5,14.5,19.75),('Right, down, taller',14.9,14.8,20.4),
             ('Left, up, shorter',14.1,14.2,19.1),('Left, down, taller',14.1,14.8,20.4)]
    settings=np.array([[cx,cy,h,w,t] for _,cx,cy,h in presets for w in widths for t in angles])
    n=len(settings); actual=render(settings).reshape(n,-1).astype(float)
    truth=field(settings);add=axis_blend(settings)
    delta=np.where(settings[:,4]>=35,-1.,1.)
    target=settings.copy();target[:,4]+=delta
    endpoints=render(target).reshape(n,-1).astype(float)
    moved=move(settings,target[:,4],actual)
    directional=np.empty_like(truth)
    for sign,side in [(1,'right'),(-1,'left')]:
        sel=delta==sign;directional[sel]=field(settings[sel],side)
    arrays={k:base64.b64encode(np.asarray(v,'<f4').tobytes()).decode() for k,v in
            [('images',actual),('arrows',truth),('blends',add),('targets',endpoints),('moves',moved),('oneSided',directional)]}
    meta={'widths':widths,'angles':angles,'presets':presets,'settings':settings.tolist(),
          'vectorErrors':(np.linalg.norm(add-truth,axis=1)/np.linalg.norm(truth,axis=1)).tolist(),
          'moveErrors':np.abs(moved-endpoints).max(axis=1).tolist(),
          'directionScale':float(max(np.abs(truth).max(),np.abs(add).max(),np.abs(add-truth).max()))}
    template=(Path(__file__).parent/'viewer_template.html').read_text()
    html=template.replace('__META__',json.dumps(meta)).replace('__ARRAYS__',json.dumps(arrays))
    (OUT/'viewer.html').write_text(html)
    (OUT/'viewer_reference.json').write_text(json.dumps({'states':n,'initial_state_index':4*len(angles)+2,
        'initial_expected_vector_error':meta['vectorErrors'][4*len(angles)+2],
        'payload_bytes':len(html.encode())},indent=2)+'\n')

if __name__=='__main__':
    figures();viewer()
