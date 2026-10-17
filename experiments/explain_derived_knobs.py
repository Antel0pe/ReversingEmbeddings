"""Plain-language figures: physical versus edge knobs, and a curved image patch."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/derived-knob-explainer-mpl')
from pathlib import Path
import sys,json
import numpy as np
from scipy.stats import qmc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from grey_ones import render,RANGES
from analyze_generated_one_manifold_transport import chart,unchart
OUT=ROOT/'figures/derived_state_coordinates'
REFERENCE=np.array([14.5,14.5,19.75,3.2,20.])


def image(p):
    return render(p,dtype=np.float64).reshape(-1,784)


def knob_examples():
    originals=[]
    for index,amount,title,description in [
        (0,.4,'Horizontal center +0.40 px','Moves right 0.40 px\nHeight and tilt stay fixed'),
        (1,.4,'Vertical center +0.40 px','Moves down 0.40 px\nHeight and horizontal center stay fixed'),
        (2,.7,'Height +0.70 px','Top rises 0.35 px; bottom drops 0.35 px\nCenter stays fixed'),
        (3,1.2,'Width +1.20 px','Sides move apart by 0.60 px each\nCenter and tilt stay fixed'),
        (4,12.,'Lean +12 degrees','Tilts from 20 to 32 degrees\nCenter stays fixed')]:
        p=REFERENCE.copy();p[index]+=amount
        originals.append((p,title,description))
    edge=[]
    for index,amount,title,description in [
        (2,.4,'Line anchor +0.40 px','Moves right 0.40 px\nSame action as horizontal-center control'),
        (0,.7,'Top moves down 0.70 px','Shorter 0.70 px; center drops 0.35 px\nHorizontal center also shifts left 0.13 px'),
        (1,.7,'Bottom moves down 0.70 px','Taller 0.70 px; center drops 0.35 px\nHorizontal center also shifts left 0.13 px'),
        (3,1.2,'Width +1.20 px','Same widening as the original width control\nCenter and tilt stay fixed'),
        (4,.4/14.5,'Slope increases','Tilts from 20 to about 21.38 degrees\nCenter moves left 0.40 px; line anchor fixed')]:
        q=chart(REFERENCE);q[index]+=amount;p=unchart(q)
        edge.append((p,title,description))
    for p,_,_ in originals+edge:
        assert np.all(p>=RANGES[:,0]-1e-12) and np.all(p<=RANGES[:,1]+1e-12)
    return originals,edge


def make_knob_figure(originals,edge):
    fig=plt.figure(figsize=(19,11),facecolor='white')
    fig.text(.035,.972,'Same family of images; two ways to describe and control it',fontsize=21,weight='bold',va='top')
    fig.text(.035,.925,'Reference: horizontal center 14.5 px, vertical center 14.5 px, height 19.75 px, width 3.2 px, lean 20 degrees.\n'
        'Each panel starts from that reference. Only one control in its row changes; all other controls in that same set stay fixed.',
        fontsize=11,va='top',linespacing=1.5)
    row_data=[[(REFERENCE,'Reference','Starting tilted 1')]+originals,
              [(REFERENCE,'Reference','The identical starting 1')]+edge]
    row_y=[.59,.18]
    fig.text(.035,.861,'Original physical controls',fontsize=13,weight='bold')
    fig.text(.035,.451,'Derived edge-and-line controls',fontsize=13,weight='bold')
    records=[]
    for row,items in enumerate(row_data):
        for col,(p,title,description) in enumerate(items):
            left=.045+col*.158
            ax=fig.add_axes([left,row_y[row],.135,.225])
            ax.imshow(image(p)[0].reshape(28,28),cmap='gray_r',vmin=0,vmax=1,
                      extent=[0,28,28,0],interpolation='nearest')
            ax.scatter([REFERENCE[0]],[REFERENCE[1]],s=55,facecolors='none',edgecolors='#315b9c',linewidths=1.3,zorder=3)
            ax.scatter([p[0]],[p[1]],s=12,color='#315b9c',zorder=4)
            if np.linalg.norm(p[:2]-REFERENCE[:2])>.08:
                ax.annotate('',xy=p[:2],xytext=REFERENCE[:2],arrowprops=dict(arrowstyle='->',color='#315b9c',lw=1.0),zorder=5)
            ax.set_title(title,fontsize=10.2,pad=10)
            ax.set_xticks([0,14,28]);ax.tick_params(labelsize=8,length=2)
            ax.set_yticks([0,14,28] if col==0 else [])
            if col==0:
                ax.set_ylabel('Vertical pixel position',fontsize=9)
                ax.set_xlabel('Horizontal pixel position',fontsize=9)
            ax.text(.5,-.18,description,transform=ax.transAxes,ha='center',va='top',fontsize=9.2,linespacing=1.5)
            records.append(dict(row=['original','derived'][row],title=title,physical_settings=p.tolist(),
                                derived_settings=chart(p).tolist(),physical_change=(p-REFERENCE).tolist()))
    fig.text(.035,.052,'Every panel shows the whole 28 × 28 image: white = no ink, black = full ink. '
        'Open blue circle = reference center; small filled blue dot = resulting center.',fontsize=10)
    fig.text(.035,.025,'Line anchor means the horizontal position where the extended centerline meets the canvas top. '
        'Changing slope alone pivots around that anchor; the original lean control preserves the stroke center.',fontsize=10)
    fig.savefig(OUT/'knob_actions_english.png',dpi=150)
    plt.close(fig)
    return records


def saddle_experiment():
    # Entire declared upright patch is exactly bilinear, despite crossings
    # of vertical subrow boundaries: all upright subrow cross-sections agree.
    parameters=np.tile([14.,14.,19.1,2.2,0.],(4,1))
    parameters[:,2]+=.8*np.array([0,1,0,1])
    parameters[:,3]+=1.6*np.array([0,0,1,1])
    x=image(parameters)
    a=x[0];b=x[1]-a;c=x[2]-a;d=x[3]-x[1]-x[2]+a
    basis=np.column_stack([b,c,d])
    uv=qmc.Sobol(2,scramble=True,seed=2026101811).random_base2(10)
    settings=np.tile([14.,14.,19.1,2.2,0.],(len(uv),1))
    settings[:,2]+=.8*uv[:,0];settings[:,3]+=1.6*uv[:,1]
    truth=image(settings)
    prediction=a+uv[:,0,None]*b+uv[:,1,None]*c+(uv[:,0]*uv[:,1])[:,None]*d
    error=np.linalg.norm(prediction-truth,axis=1)
    singular=np.linalg.svd(basis,compute_uv=False)
    assert np.sum(singular>singular[0]*1e-12)==3
    assert np.max(error)<1e-12
    stats=dict(height_range=[19.1,19.9],width_range=[2.2,3.8],fixed_center=[14.,14.],fixed_lean=0.,
        test_states=len(uv),independent_pixel_patterns=3,singular_values=singular.tolist(),
        largest_image_L2_error=float(error.max()),largest_pixel_error=float(np.abs(prediction-truth).max()),
        missing_corner_pattern_norm=float(np.linalg.norm(d)),pattern_weight_equation='third weight = first weight times second weight')
    # Exact coordinates in the B,C,D basis: (u,v,u*v). This is not a
    # lossy projection, but its drawn Euclidean metric is not pixel distance.
    fig=plt.figure(figsize=(14,11),facecolor='white')
    fig.text(.045,.973,'Why adding contributions can describe a curved image family',fontsize=19,weight='bold',va='top')
    fig.text(.045,.929,'One exact two-knob slice: center fixed at (14,14), lean fixed at zero. Height 19.1–19.9 px; width 2.2–3.8 px.\n'
        'The four corner images determine three independent pixel-change patterns: height, width, and their corner interaction.',
        fontsize=10.6,va='top',linespacing=1.5)
    for i,(label,index) in enumerate([('Starting image',0),('Height only',1),('Width only',2),('Height and width together',3)]):
        ax=fig.add_axes([.07+.235*i,.67,.15,.19])
        ax.imshow(x[index].reshape(28,28),cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
        ax.set_xticks([]);ax.set_yticks([])
        ax.set_title(label,fontsize=10.5,pad=10)
        ax.text(.5,-.09,f'Height {parameters[index,2]:.1f}; width {parameters[index,3]:.1f}',
                transform=ax.transAxes,ha='center',va='top',fontsize=10)
    fig.text(.045,.603,'Corner-pattern weight = height-pattern weight × width-pattern weight.',fontsize=12,weight='bold')
    fig.text(.045,.568,'Exact coordinates for this slice: each plotted point represents one whole 784-pixel image, not one pixel.',fontsize=10.8)
    u,v=np.meshgrid(np.linspace(0,1,25),np.linspace(0,1,25))
    time=np.linspace(0,1,101)
    for i in range(2):
        ax=fig.add_axes([.015+.49*i,.16,.46,.355],projection='3d')
        z=np.zeros_like(u) if i==0 else u*v
        ax.plot_surface(u,v,z,color='#7eabb1',alpha=.32,edgecolor='none',shade=False)
        ax.plot_wireframe(u,v,z,rstride=6,cstride=6,color='#65868a',lw=.7,alpha=.8)
        ax.plot(time,time,np.zeros_like(time) if i==0 else time*time,color='#a33b4c',lw=3)
        ax.plot(time,np.full_like(time,.5),np.zeros_like(time) if i==0 else time*.5,color='#315b9c',lw=2,ls='--')
        ax.set(xlim=(0,1),ylim=(0,1),zlim=(0,1))
        ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1]);ax.set_zticks([0,.5,1]);ax.tick_params(labelsize=9,pad=1)
        ax.set_xlabel('Height-pattern weight',fontsize=10,labelpad=7)
        ax.set_ylabel('Width-pattern weight',fontsize=10,labelpad=7)
        ax.set_zlabel('Corner-pattern weight',fontsize=10,labelpad=6)
        ax.view_init(elev=23,azim=-55)
        ax.set_title('Flat approximation: omit the corner pattern' if i==0 else 'Exact image patch: include the corner pattern',fontsize=12,pad=14)
    fig.text(.06,.125,f'Largest missing-pattern error: {np.linalg.norm(d):.2f} image-distance units',fontsize=10.3)
    fig.text(.55,.125,f'Worst reconstruction error on 1,024 new images: {error.max():.2g}',fontsize=10.3)
    fig.legend(handles=[Line2D([0],[0],color='#a33b4c',lw=3,label='Both knobs increase: curved path on the exact patch'),
                        Line2D([0],[0],color='#315b9c',lw=2,ls='--',label='Width held fixed: straight path on this patch')],
               loc='lower center',bbox_to_anchor=(.5,.065),ncol=1,frameon=False,fontsize=10)
    fig.text(.045,.045,'No image information is discarded within this slice. Axes are weights of full pixel patterns; their drawn angles and distances are not the 784D pixel metric.\n'
        'White = 0 ink, black = 1. Three original controls are fixed. This illustrates a local two-dimensional patch, not the entire five-dimensional family.',
        fontsize=9.2,va='top',linespacing=1.5)
    fig.savefig(OUT/'curved_patch_english.png',dpi=150)
    plt.close(fig)
    np.savez_compressed(OUT/'curved_patch_patterns.npz',base=a,height_pattern=b,width_pattern=c,corner_pattern=d,
                        test_settings=settings,predicted_images=prediction)
    return stats


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    originals,edge=knob_examples()
    result=dict(reference=REFERENCE.tolist(),actions=make_knob_figure(originals,edge),saddle=saddle_experiment())
    (OUT/'plain_english_checks.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result['saddle'],indent=2))


if __name__=='__main__':main()
