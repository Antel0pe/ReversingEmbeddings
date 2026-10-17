"""Validate explicit state-derived coordinates and exact finite knob actions.

Public explanation is a mathematical note, with no programming examples.
The formula implementation accepts only images and requested increments.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/derived-state-coordinates-mpl')
from pathlib import Path
import sys,json,itertools,time
import numpy as np
from scipy.stats import qmc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from grey_ones import render,RANGES,KNOBS,SUB
OUT=ROOT/'figures/derived_state_coordinates'
SPAN=np.diff(RANGES,axis=1).ravel()
BASE=np.array([14.,14.,19.75,3.2,0.])


def unit(power,seed):
    return qmc.Sobol(5,scramble=True,seed=seed).random_base2(power)


def observe(points,dtype=np.float64):
    return render(points,dtype=dtype).reshape(-1,28,28)


def recover(images):
    """Return original physical coordinates using only image measurements.

    The fixed masks are valid over the entire declared default range box.
    Lean/horizontal center require two adaptive cumulative-row measurements.
    """
    x=np.asarray(images,dtype=np.float64).reshape(-1,28,28)
    rows=x.sum(2)
    w=rows[:,10]
    upper=rows[:,:6].sum(1)
    lower=rows[:,23:].sum(1)
    top=6-upper/w
    bottom=23+lower/w
    cy=14.5+(lower-upper)/(2*w)
    height=17+(upper+lower)/w
    centers=[];boundaries=[];clearances=[]
    for row in [10,18]:
        cumulative=np.column_stack([np.zeros(len(x)),np.cumsum(x[:,row],axis=1)])
        k=np.argmin(np.abs(cumulative-w[:,None]/2),axis=1)
        centers.append(k-cumulative[np.arange(len(x)),k]+w/2)
        boundaries.append(k)
    slope=(centers[0]-centers[1])/8
    intercept=centers[0]+10.5*slope
    cx=intercept-slope*cy
    lean=np.rad2deg(np.arctan(slope))
    p=np.column_stack([cx,cy,height,w,lean])
    q=np.column_stack([top,bottom,intercept,w,slope])
    raw=np.column_stack([w,upper,lower,centers[0],centers[1]])
    return p,q,raw,np.array(boundaries).T


def coverage(q):
    """Explicit interval-intersection equation, independent of render()."""
    q=np.atleast_2d(q)
    lo=np.arange(28*SUB)/SUB
    hi=lo+1/SUB
    y=(lo+hi)/2
    c=np.arange(28)
    output=np.empty((len(q),28,28))
    for start in range(0,len(q),128):
        t,b,a,w,s=[q[start:start+128,j,None] for j in range(5)]
        vertical=np.maximum(0,np.minimum(hi,b)-np.maximum(lo,t))
        middle=a-s*y
        left=middle-w/2;right=middle+w/2
        horizontal=np.maximum(0,np.minimum(right[:,:,None],c+1)-np.maximum(left[:,:,None],c))
        output[start:start+128]=(vertical[:,:,None]*horizontal).reshape(-1,28,SUB,28).sum(2)
    return output


def action(images,increments):
    """Only the 784-state and requested five increments enter this rule."""
    _,q,_,_=recover(images)
    d=np.broadcast_to(increments,(len(q),5))
    t,b,a,w,s=q.T
    cy=(t+b)/2
    sn=np.tan(np.arctan(s)+np.deg2rad(d[:,4]))
    qn=np.column_stack([t+d[:,1]-d[:,2]/2,b+d[:,1]+d[:,2]/2,
        a+d[:,0]+sn*(cy+d[:,1])-s*cy,w+d[:,3],sn])
    return coverage(qn)


def errors(truth,pred):
    delta=np.asarray(pred)-np.asarray(truth)
    return dict(worst_pixel=float(np.max(np.abs(delta))),
                worst_image_L2=float(np.max(np.linalg.norm(delta.reshape(len(delta),-1),axis=1))))


def settings():
    points=[];tags=[]
    def add(values,tag):
        values=np.asarray(values).reshape(-1,5)
        points.extend(values);tags.extend([tag]*len(values))
    add(RANGES[:,0]+unit(12,2026101717)*SPAN,'sobol')
    grid=np.array(list(itertools.product(np.linspace(0,1,5),repeat=5)))
    add(RANGES[:,0]+grid*SPAN,'five_level_lattice')
    edge=unit(10,2026101718)
    for i in range(5):
        edge[i::5,i]=np.where(np.arange(len(edge[i::5]))%2,1e-10,1-1e-10)
    add(RANGES[:,0]+edge*SPAN,'near_box_faces')
    # Exact and nearby subrow cap crossings.
    p=RANGES[:,0]+unit(10,2026101719)*SPAN
    for end_sign in [-1,1]:
        edge_y=p[:,1]+end_sign*p[:,2]/2
        cy=np.round(edge_y*16)/16-end_sign*p[:,2]/2
        for offset in [-1e-9,0.,1e-9]:
            z=p.copy();z[:,1]=cy+offset
            keep=(z[:,1]>=14)&(z[:,1]<=15)
            add(z[keep],f'cap_boundary_{offset}')
    # A stroke side exactly on an integer column at a selected subrow.
    for side in [-1,1]:
        for y in [10+1/32,18+31/32]:
            center=p[:,0]-np.tan(np.deg2rad(p[:,4]))*(y-p[:,1])
            edge_x=center+side*p[:,3]/2
            cx=p[:,0]+np.round(edge_x)-edge_x
            for offset in [-1e-9,0.,1e-9]:
                z=p.copy();z[:,0]=cx+offset
                keep=(z[:,0]>=14)&(z[:,0]<=15)
                add(z[keep],f'side_boundary_{offset}')
    # Half-ink-boundary ties; either minimizing integer must yield one center.
    for row in [10,18]:
        center=p[:,0]-np.tan(np.deg2rad(p[:,4]))*(row+.5-p[:,1])
        z=p.copy();z[:,0]+=np.floor(center)+.5-center
        keep=(z[:,0]>=14)&(z[:,0]<=15)
        add(z[keep],'cumulative_boundary_tie')
    add(BASE,'canonical_root')
    return np.array(points),np.array(tags)


def scalar_examples():
    examples=[]
    for name,p in [('root',BASE),('tilted',[14.8,14.63,20.2,2.1,33.]),
                   ('thin_corner',[15.,14.,19.,1.8,35.])]:
        x=observe(p)
        recovered,q,raw,k=recover(x)
        row=x.sum(2)
        centroid=(row*(np.arange(28)+.5)).sum(1)/row.sum(1)
        examples.append(dict(name=name,settings=np.asarray(p).tolist(),
            width=float(raw[0,0]),upper_ink=float(raw[0,1]),lower_ink=float(raw[0,2]),
            recovered_center=float(recovered[0,1]),ordinary_centroid=float(centroid[0]),
            row_centers=raw[0,3:].tolist(),chosen_boundaries=k[0].tolist()))
    return examples


def main():
    started=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    p,tags=settings()
    x=observe(p)
    recovered,q,raw,k=recover(x)
    restored=coverage(q)
    result=dict(states=len(p),counts={str(t):int(np.sum(tags==t)) for t in np.unique(tags)},
        knob_order=KNOBS,parameter_units=['px','px','px','px','degrees'],
        float64_recovery_worst_absolute=np.max(np.abs(recovered-p),axis=0).tolist(),
        float64_image_roundtrip=errors(x,restored),examples=scalar_examples())
    assert np.max(np.abs(recovered-p))<1e-10
    assert result['float64_image_roundtrip']['worst_pixel']<1e-11
    rows=x.sum(2)
    center_of_mass=(rows*(np.arange(28)+.5)).sum(1)/rows.sum(1)
    result['ordinary_vertical_centroid_worst_error']=float(np.max(np.abs(center_of_mass-p[:,1])))
    # Verify the chosen cumulative boundary is inside every selected row strip.
    margins=[]
    for j,row in enumerate([10,18]):
        ys=row+(np.arange(16)+.5)/16
        mid=p[:,0,None]-np.tan(np.deg2rad(p[:,4,None]))*(ys-p[:,1,None])
        clearance=p[:,3,None]/2-np.abs(k[:,j,None]-mid)
        margins.append(float(clearance.min()))
    result['measured_min_chosen_boundary_clearance']=min(margins)
    result['proven_common_interval_half_width_lower_bound']=float(1.8/2-15*np.tan(np.deg2rad(35))/32)
    assert result['measured_min_chosen_boundary_clearance']>0
    print('Coordinate and image round trips:',len(p),'states',flush=True)
    # Compare finite mathematical actions against independent renderer targets.
    chosen=np.unique(np.linspace(0,len(p)-1,2048,dtype=int))
    sources=p[chosen];current=x[chosen]
    target_u=unit(11,2026101720)
    destination=RANGES[:,0]+target_u*SPAN
    result['single_knob_updates']={}
    for i,name in enumerate(KNOBS):
        target=sources.copy();target[:,i]=destination[:,i]
        delta=target-sources
        predicted=action(current,delta)
        truth=observe(target)
        result['single_knob_updates'][name]=dict(states=len(sources),**errors(truth,predicted),
            recovered_other_knob_drift=float(np.max(np.abs(recover(predicted)[0][:,np.arange(5)!=i]-sources[:,np.arange(5)!=i]))))
        assert result['single_knob_updates'][name]['worst_pixel']<1e-11
    combined=action(current,destination-sources)
    result['joint_updates']=dict(states=len(sources),**errors(observe(destination),combined))
    print('All five single-knob actions and joint updates passed.',flush=True)
    # Repeated image-only composition: endpoints from multiple application orders.
    order_states=sources[:128];start=current[:128];dest=destination[:128]
    d=dest-order_states
    order_records=[]
    for order in [list(range(5)),list(reversed(range(5))),[4,2,0,3,1],[1,3,0,2,4]]:
        state=start.copy()
        for axis in order:
            step=np.zeros_like(d);step[:,axis]=d[:,axis]
            state=action(state,step)
        order_records.append(dict(order=[KNOBS[i] for i in order],**errors(observe(dest),state)))
    result['five_step_order_checks']=order_records
    result['identity']=errors(current,action(current,np.zeros(5)))
    result['inverse']=errors(current,action(combined,sources-destination))
    # Explicit original-knob alignment, not just invertibility of another chart.
    result['physical_coordinate_alignment']=dict(
        max_abs_error_after_joint_action=float(np.max(np.abs(recover(combined)[0]-destination))))
    # Float32 storage is a separate experiment, not mathematical exactness.
    x32=observe(p,dtype=np.float32)
    recovered32,q32,_,_=recover(x32)
    result['float32_recovery_worst_absolute']=np.max(np.abs(recovered32-p),axis=0).tolist()
    result['float32_image_roundtrip']=errors(x32,coverage(q32))
    result['float32_joint_updates']=errors(observe(destination,dtype=np.float32),
        action(observe(sources,dtype=np.float32),destination-sources))
    p0=BASE.copy();p1=BASE.copy();p1[1]+=1e-9
    same=bool(np.array_equal(observe(p0,dtype=np.float32),observe(p1,dtype=np.float32)))
    assert same
    result['float32_nonidentifiability_example']=dict(first_settings=p0.tolist(),
        second_settings=p1.tolist(),images_bitwise_identical=same,
        mathematical_vertical_center_difference=float(p1[1]-p0[1]))
    step=np.array([0.,.125,0.,0.,0.])
    target0=observe(p0+step,dtype=np.float32)
    target1=observe(p1+step,dtype=np.float32)
    assert not np.array_equal(target0,target1)
    result['float32_nonidentifiability_example'].update(
        common_vertical_increment=.125,
        target_images_bitwise_identical=bool(np.array_equal(target0,target1)),
        largest_target_pixel_difference=float(np.max(np.abs(target1-target0))))
    # Coordinate mixing is predicted exactly by the known chart differential.
    result['coordinate_response_columns']=dict(
        q_order=['top','bottom','intercept','width','slope'],
        physical_knob_order=KNOBS,
        root_jacobian=[[0,1,-.5,0,0],[0,1,.5,0,0],[1,0,0,0,14*np.pi/180],
                       [0,0,0,1,0],[0,0,0,0,np.pi/180]])
    result['elapsed_seconds']=time.monotonic()-started
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    # Store representative observations, not the entire multi-thousand-state run.
    np.savez_compressed(OUT/'examples_and_checks.npz',source_settings=sources[:128],
        source_images=current[:128],target_settings=destination[:128],
        predicted_joint_images=combined[:128],recovered_coordinates=recovered[chosen[:128]],
        float32_collision_first=observe(p0,dtype=np.float32),float32_collision_second=observe(p1,dtype=np.float32))
    print(json.dumps(result,indent=2),flush=True)
    make_figure(result)


def make_figure(result):
    fig=plt.figure(figsize=(13,8),facecolor='white')
    fig.text(.04,.96,'Vertical center is a function of the 784 pixel values',fontsize=19,weight='bold',va='top')
    fig.text(.04,.905,'Controlled 28 × 28 generated 1s: white = 0 ink, black = 1. All coordinates are in canvas pixels.\n'
        'One tilted example is shown; the equation holds throughout all five default parameter ranges.',fontsize=10.5,va='top',linespacing=1.5)
    p=np.array(result['examples'][1]['settings']);x=observe(p)[0]
    ax=fig.add_axes([.07,.27,.32,.55])
    ax.imshow(x,cmap='gray_r',vmin=0,vmax=1,origin='upper',extent=(0,28,28,0),interpolation='nearest')
    ax.add_patch(Rectangle((0,0),28,6,facecolor='#e6a33b',alpha=.22,edgecolor='#aa6b00',lw=1.5))
    ax.add_patch(Rectangle((0,23),28,5,facecolor='#6489c9',alpha=.22,edgecolor='#234c91',lw=1.5))
    ax.add_patch(Rectangle((0,10),28,1,facecolor='#52ab80',alpha=.35,edgecolor='#267c50',lw=1.5))
    ax.axhline(p[1],color='#a12670',ls='--',lw=1.5)
    ax.set_xlabel('Horizontal pixel position (increases rightward)',fontsize=9)
    ax.set_ylabel('Vertical pixel position (increases downward)',fontsize=9)
    ax.set_xticks([0,7,14,21,28]);ax.set_yticks([0,6,10,14.63,23,28])
    ax.set_title('Current state: tilted 1',fontsize=12,pad=10)
    ax.text(.5,-.17,'Dashed line: true vertical center = 14.63 px',transform=ax.transAxes,ha='center',fontsize=9,color='#a12670')
    example=result['examples'][1]
    fig.text(.45,.78,f"U = ink in upper six rows = {example['upper_ink']:.4f}",fontsize=12,color='#986000')
    fig.text(.45,.715,f"L = ink in lower five rows = {example['lower_ink']:.4f}",fontsize=12,color='#234c91')
    fig.text(.45,.65,f"W = ink in row 10 = {example['width']:.4f}",fontsize=12,color='#267c50')
    fig.text(.45,.53,r'$\mathrm{vertical\ center}(I)=14.5+\frac{L(I)-U(I)}{2W(I)}$',fontsize=22)
    fig.text(.45,.435,f"Recovered center: {example['recovered_center']:.8f} px\n"
        f"Ordinary ink centroid: {example['ordinary_centroid']:.8f} px",fontsize=12,linespacing=1.8)
    error=result['float64_recovery_worst_absolute'][1]
    fig.text(.45,.30,f"Checked on {result['states']:,} states across all five controls.\n"
        f"Worst center error: {error:.2g} px with float64 evaluation.\n"
        f"Ordinary ink centroid worst error: {result['ordinary_vertical_centroid_worst_error']:.4g} px.",fontsize=10.5,linespacing=1.7)
    fig.text(.045,.15,'Why it is exact: width × top-band height = U, and width × bottom-band height = L.\n'
        'The top is 6 − U/W, the bottom is 23 + L/W, and their midpoint gives the equation.\n'
        'The masks are fixed. They contain no supplied center, height, width, or lean values.',fontsize=10.5,va='top',linespacing=1.6)
    fig.text(.045,.05,'Equation: exact for unrounded renderer coverage in the default box. Numerical tests include lattice, near-edge and clipping-boundary states.\n'
        'Float32 storage loses information: distinct real centers can produce identical stored images.',fontsize=9,va='top',linespacing=1.4)
    fig.savefig(OUT/'vertical_center_equation.png',dpi=155)
    plt.close(fig)


if __name__=='__main__':
    main()
