"""Numerical inspection of an erroneous channel; uses no edge/shape detector."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/blind-coordinate-rule-mpl')
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from coordinate_rule import CoordinateRule
from run import Reference, PERM, UNDO, OUT

rule=CoordinateRule.load(OUT/'width_lean_rule.npz')
data=np.load(OUT/'width_lean_evaluation.npz')
false=(abs(data['truth_delta'])<=1e-7)&(abs(data['predicted_delta'])>1e-7)
update,channel=np.unravel_index(np.argmax(np.where(false,abs(data['predicted_delta']),0)),false.shape)
source=data['starts'][update];target=data['targets'][update]
ref=Reference([3,4],PERM)
points=np.array([source,target])
true=ref(points)[:,UNDO][:,channel]
pred=rule.interpolate(points)[:,UNDO][:,channel]
point=points[np.argmax(abs(pred-true))]
cell=[int(np.clip(np.searchsorted(k,x,side='right')-1,0,len(k)-2)) for k,x in zip(rule.knots,point)]
fractions=[np.linspace(k[i],k[i+1],129) for k,i in zip(rule.knots,cell)]
x,y=np.meshgrid(*fractions,indexing='xy')
u=np.c_[x.ravel(),y.ravel()]
actual=ref(u)[:,UNDO][:,channel].reshape(x.shape)
interpolated=rule.interpolate(u)[:,UNDO][:,channel].reshape(x.shape)
error=interpolated-actual
# All views are of ONE numeric output coordinate, chosen by maximum false change.
physical_x=ref.lo[0]+ref.span[0]*x
physical_y=ref.lo[1]+ref.span[1]*y
extent=[physical_x.min(),physical_x.max(),physical_y.min(),physical_y.max()]
fig=plt.figure(figsize=(16,6.8),facecolor='white')
fig.text(.055,.965,'Why does interpolation change a pixel that should stay unchanged?',fontsize=19,weight='bold',va='top')
fig.text(.055,.905,f'One output channel (flat pixel index {channel}), selected for the largest false change in the width/lean test.',fontsize=12)
fig.text(.055,.867,'One learned coordinate cell; gold marks full ink in the reference. Coverage colors are zoomed near 1 to reveal the discrepancy.',fontsize=11)
axes=[]
coverage_low=float(min(actual.min(),interpolated.min()))
for i,(values,title,cmap,lo,hi) in enumerate([
    (actual,'Actual pixel value','gray_r',coverage_low,1),
    (interpolated,'Rule’s pixel value','gray_r',coverage_low,1),
    (error,'Rule minus actual value','RdBu_r',-float(abs(error).max()),float(abs(error).max()))]):
    ax=fig.add_axes([.075+i*.305,.29,.245,.49])
    im=ax.imshow(values,origin='lower',extent=extent,aspect='auto',cmap=cmap,vmin=lo,vmax=hi)
    if actual.min()<1-1e-7<actual.max():
        ax.contour(physical_x,physical_y,actual,levels=[1-1e-7],colors='#efb92e',linewidths=1.8)
    ax.set_xlabel('Control D: width (pixels)')
    ax.set_ylabel('Control E: lean (degrees)')
    ax.set_title(title,pad=12)
    ax.tick_params(labelsize=9)
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(5))
    cax=fig.add_axes([.075+i*.305,.18,.245,.018])
    cb=fig.colorbar(im,cax=cax,orientation='horizontal');cb.ax.tick_params(labelsize=9)
    cb.set_ticks([lo,(lo+hi)/2,hi])
    axes.append(ax)
fig.text(.055,.105,f'In the selected update the true channel stays {true[0]:.6f} → {true[1]:.6f}; the rule invents a change of {pred[1]-pred[0]:+.6f}.',fontsize=11)
fig.text(.055,.058,'The true transition crosses the cell interior. A weighted average of its corners smooths across that transition.',fontsize=11)
fig.text(.055,.018,'Gold boundary estimated from 129 × 129 reference samples. No pixel-grid adjacency was used to find or plot it.',fontsize=10)
fig.savefig(OUT/'03_joint_coordinate_boundary.png',dpi=150)
plt.close(fig)
summary=dict(selection='largest absolute false pixel change over width/lean held-out updates',
    evaluation_index=int(update),original_output_channel=int(channel),
    true_source_value=float(true[0]),true_target_value=float(true[1]),
    predicted_false_change=float(pred[1]-pred[0]),cell_indices=cell,
    max_zoom_pixel_error=float(abs(error).max()),zoom_points=len(u))
(OUT/'failure_inspection.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary))
