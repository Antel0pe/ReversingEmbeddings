"""Explicit sparse pixel edits for width/height, then width/lean.

No learned model. Uses known straight-stroke interval geometry, with either
five coordinates or a current 784-value image as the only state input.
Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/sparse_pixel_change_rules.py
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/sparse-pixel-change-mpl')
from pathlib import Path
import sys,itertools,json,csv
import numpy as np
from scipy.stats import qmc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from grey_ones import render,SUB

OUT=ROOT/'figures'/'sparse_pixel_change_rules'
BASE=np.array([14.,14.,19.75,3.2,0.])
TOL64=1e-10
TOL32=1e-6
YLO=np.arange(28*SUB)/SUB
YMID=YLO+.5/SUB


def clip_overlap(start,end,cells):
    return np.maximum(0,np.minimum(end,cells+1)-np.maximum(start,cells))


def profiles(p):
    cx,cy,h,w,angle=p
    assert abs(angle)<1e-8
    return clip_overlap(cx-w/2,cx+w/2,np.arange(28)),clip_overlap(cy-h/2,cy+h/2,np.arange(28))


def recover(image):
    """Closed-form image input: sums, two cumulative boundaries, arithmetic.

    Assumes a valid positive unclipped member of the declared family with
    >=2 full interior rows and an internal integer boundary through each
    selected row's whole slanted stroke. This holds for the tested domains.
    """
    image=np.asarray(image,dtype=np.float64).reshape(28,28)
    mass=image.sum(axis=1)
    w=float(mass.max())
    ink=np.flatnonzero(mass>1e-12)
    first,last=int(ink[0]),int(ink[-1])
    top=first+1-mass[first]/w
    bottom=last+mass[last]/w
    h=bottom-top;cy=(top+bottom)/2
    rows=[first+1,last-1]
    if rows[0]>=rows[1]:raise ValueError('Need two full interior rows.')
    centers=[]
    for row in rows:
        cumulative=np.r_[0.,np.cumsum(image[row])]
        k=int(np.argmin(np.abs(cumulative-w/2)))
        centers.append(k-cumulative[k]+w/2)
    slope=(centers[0]-centers[1])/(rows[1]-rows[0])
    cx=centers[0]+slope*(rows[0]+.5-cy)
    return np.array([cx,cy,h,w,np.rad2deg(np.arctan(slope))])


def recover_wh(image):
    """Upright centered slice: just two sums; other controls are fixed."""
    mass=np.asarray(image,dtype=np.float64).reshape(28,28).sum(axis=1)
    width=float(mass.max())
    height=float(mass.sum()/width)
    return np.array([14.,14.,height,width,0.])


def pixel_values(p,indices):
    """Observe ONLY specified pixels, with the source's subrow convention."""
    indices=np.asarray(indices,int)
    if not len(indices):return np.array([])
    row=indices//28;col=indices%28
    ylo=row[:,None]+np.arange(SUB)[None,:]/SUB
    y=ylo+.5/SUB
    cx,cy,h,w,angle=p
    weight=np.clip((np.minimum(ylo+1/SUB,cy+h/2)-np.maximum(ylo,cy-h/2))*SUB,0,1)
    center=cx-np.tan(np.deg2rad(angle))*(y-cy)
    overlap=np.clip(np.minimum(center+w/2,col[:,None]+1)-np.maximum(center-w/2,col[:,None]),0,1)
    return (weight*overlap).mean(axis=1)


def finish(p,indices,delta,current,tol):
    indices=np.asarray(indices,int);delta=np.asarray(delta,float)
    keep=np.abs(delta)>tol
    indices=indices[keep];delta=delta[keep]
    old=pixel_values(p,indices) if current is None else np.asarray(current,float).ravel()[indices]
    return dict(indices=indices,delta=delta,old=old,new=old+delta)


def wh_edits(p,axis,amount=1.,current=None,tol=TOL64):
    """Two 28-element profiles; sparse Cartesian product, no full rendering."""
    a,b=profiles(p)
    q=np.array(p,float);q[axis]+=amount
    an,bn=profiles(q)
    if axis==3:
        horizontal=an-a;vertical=b
    elif axis==2:
        horizontal=a;vertical=bn-b
    else:raise ValueError('Width or height only.')
    rows=np.flatnonzero(vertical!=0);cols=np.flatnonzero(horizontal!=0)
    rr,cc=np.meshgrid(rows,cols,indexing='ij')
    idx=(28*rr+cc).ravel()
    delta=(vertical[rows,None]*horizontal[None,cols]).ravel()
    return finish(p,idx,delta,current,tol)


def accumulate_segment(acc,row,left,right,weight):
    """Enumerate exactly the columns intersected by one added/lost strip."""
    if right<=left or weight==0:return
    lo=max(0,int(np.floor(left)));hi=min(28,int(np.ceil(right)))
    for col in range(lo,hi):
        overlap=max(0.,min(right,col+1)-max(left,col))
        if overlap:
            idx=28*row+col
            acc[idx]=acc.get(idx,0.)+weight*overlap


def row_interval_edits(p,axis,amount=1.,current=None,tol=TOL64):
    """Sparse changes to subrow intervals. Emits strips, not a full next image."""
    cx,cy,h,w,angle=p
    weight=np.clip((np.minimum(YLO+1/SUB,cy+h/2)-np.maximum(YLO,cy-h/2))*SUB,0,1)/SUB
    center=cx-np.tan(np.deg2rad(angle))*(YMID-cy)
    left=center-w/2;right=center+w/2
    if axis==4:
        shift=-(YMID-cy)*(np.tan(np.deg2rad(angle+amount))-np.tan(np.deg2rad(angle)))
    elif axis==2:
        newweight=np.clip((np.minimum(YLO+1/SUB,cy+(h+amount)/2)-np.maximum(YLO,cy-(h+amount)/2))*SUB,0,1)/SUB
    acc={}
    for j in range(28*SUB):
        row=j//SUB
        l,r,t=left[j],right[j],weight[j]
        if axis==3:
            if t==0:continue
            if amount>=0:
                accumulate_segment(acc,row,l-amount/2,l,t)
                accumulate_segment(acc,row,r,r+amount/2,t)
            else:
                accumulate_segment(acc,row,l,l-amount/2,-t)
                accumulate_segment(acc,row,r+amount/2,r,-t)
        elif axis==2:
            accumulate_segment(acc,row,l,r,newweight[j]-t)
        elif axis==4:
            if t==0:continue
            d=shift[j]
            if d>=0:
                accumulate_segment(acc,row,l,min(r,l+d),-t)
                accumulate_segment(acc,row,max(r,l+d),r+d,t)
            else:
                accumulate_segment(acc,row,l+d,min(l,r+d),t)
                accumulate_segment(acc,row,max(l,r+d),r,-t)
        else:raise ValueError('Unsupported action.')
    idx=np.array(sorted(acc),int)
    return finish(p,idx,np.array([acc[i] for i in idx]),current,tol)


def apply_edits(image,edits):
    output=np.asarray(image,float).ravel().copy()
    output[edits['indices']]=edits['new']
    return output


def valid(p):
    p=np.atleast_2d(p)
    reach=p[:,3]/2+p[:,2]/2*np.abs(np.tan(np.deg2rad(p[:,4])))
    return bool(np.all((p[:,2]>0)&(p[:,3]>0)&(p[:,0]-reach>0)&(p[:,0]+reach<28)&
                       (p[:,1]-p[:,2]/2>0)&(p[:,1]+p[:,2]/2<28)))


def evaluate(sources,axes,name,rule):
    sources=np.asarray(sources)
    original64=render(sources,dtype=np.float64).reshape(-1,784)
    original32=render(sources).reshape(-1,784)
    records=[];max_coordinate_error=np.zeros(5)
    for axis in axes:
        targets=sources.copy();targets[:,axis]+=1
        assert valid(sources) and valid(targets)
        target64=render(targets,dtype=np.float64).reshape(-1,784)
        target32=render(targets).reshape(-1,784)
        for route in ['coordinates64','image64','image32']:
            low=route=='image32';tol=TOL32 if low else TOL64
            originals=original32 if low else original64
            targets_img=target32 if low else target64
            for i,p in enumerate(sources):
                if route=='coordinates64':
                    state=p;current=None
                else:
                    state=(recover_wh(originals[i]) if name=='width_height' else recover(originals[i]));current=originals[i]
                    if route=='image64':max_coordinate_error=np.maximum(max_coordinate_error,np.abs(state-p))
                # General row rule tolerates a recovered near-zero lean. The
                # upright route explicitly uses the known fixed upright slice.
                if name=='width_height':
                    state=state.copy();state[4]=0.
                edits=rule(state,axis,current=current,tol=tol)
                predicted=apply_edits(originals[i],edits)
                actual_delta=targets_img[i]-originals[i]
                actual_idx=np.flatnonzero(np.abs(actual_delta)>tol)
                false=np.setdiff1d(edits['indices'],actual_idx)
                missed=np.setdiff1d(actual_idx,edits['indices'])
                records.append(dict(slice=name,route=route,source=i,action=['cx','cy','height','width','lean'][axis],
                                    threshold=tol,actual_changed_pixels=len(actual_idx),returned_pixels=len(edits['indices']),
                                    false_indices=len(false),missed_indices=len(missed),
                                    maximum_pixel_error=float(np.abs(predicted-targets_img[i]).max()),
                                    pixel_L2_error=float(np.linalg.norm(predicted-targets_img[i]))))
    summary={}
    for route in ['coordinates64','image64','image32']:
        rows=[r for r in records if r['route']==route]
        summary[route]=dict(updates=len(rows),false_indices=sum(r['false_indices'] for r in rows),
                            missed_indices=sum(r['missed_indices'] for r in rows),
                            cases_with_wrong_indices=sum(r['false_indices']+r['missed_indices']>0 for r in rows),
                            maximum_pixel_error=max(r['maximum_pixel_error'] for r in rows),
                            maximum_pixel_L2_error=max(r['pixel_L2_error'] for r in rows),
                            support_threshold=rows[0]['threshold'])
    return dict(source_states=len(sources),routes=summary,
                maximum_image64_coordinate_recovery_error=max_coordinate_error.tolist()),records


class HiddenMasks:
    """Carry two 1D profiles and two prospective one-unit pixel-change masks."""
    def __init__(self,p):
        self.p=np.array(p,float);self.a,self.b=profiles(self.p)
        self.dw=self.width_columns();self.dh=self.height_rows()
        self.width_mask=np.outer(self.b,self.dw)
        self.height_mask=np.outer(self.dh,self.a)
    def width_columns(self):
        q=self.p.copy();q[3]+=1
        return profiles(q)[0]-self.a
    def height_rows(self):
        q=self.p.copy();q[2]+=1
        return profiles(q)[1]-self.b
    def move(self,axis):
        self.p[axis]+=1
        anew,bnew=profiles(self.p)
        if axis==2:
            # Update the OTHER mask by extending only its current side columns.
            self.width_mask+=np.outer(bnew-self.b,self.dw)
            self.b=bnew
            self.dh=self.height_rows()
            self.height_mask=np.outer(self.dh,self.a)
        else:
            # Update the OTHER mask by widening only its current end rows.
            self.height_mask+=np.outer(self.dh,anew-self.a)
            self.a=anew
            self.dw=self.width_columns()
            self.width_mask=np.outer(self.b,self.dw)


def hidden_check():
    sequence=[2,3,2,2,3,2]
    hidden=HiddenMasks(BASE);history=[];error=0.
    for step in range(len(sequence)+1):
        p=hidden.p.copy();im=render(p,dtype=np.float64).ravel()
        w=p.copy();w[3]+=1;h=p.copy();h[2]+=1
        true_w=render(w,dtype=np.float64).ravel()-im
        true_h=render(h,dtype=np.float64).ravel()-im
        error=max(error,np.abs(hidden.width_mask.ravel()-true_w).max(),np.abs(hidden.height_mask.ravel()-true_h).max())
        history.append(dict(p=p,image=im,width=hidden.width_mask.copy(),height=hidden.height_mask.copy()))
        if step<len(sequence):hidden.move(sequence[step])
    return history,dict(steps=len(sequence),maximum_hidden_mask_pixel_error=float(error),
                        method='Update the other prospective mask using an outer-product correction; refresh the changed knob own prospective edge profile.')


def chain_checks():
    result={}
    for name,actions,orders,rule in [
        ('width_height',[2,2,3,3],sorted(set(itertools.permutations([2,2,3,3]))),wh_edits),
        ('width_lean',[3,3]+[4]*10,[[3 if i in slots else 4 for i in range(12)] for slots in itertools.combinations(range(12),2)],row_interval_edits)]:
        error=0.;false=0;miss=0;count=0;ends=[]
        for order in orders:
            state=BASE.copy();image=render(state,dtype=np.float64).ravel()
            for axis in order:
                recovered=recover_wh(image) if name=='width_height' else recover(image)
                e=rule(recovered,axis,current=image)
                image=apply_edits(image,e);state[axis]+=1
                ref=render(state,dtype=np.float64).ravel()
                error=max(error,np.abs(image-ref).max())
                actual_delta=ref-render(state-np.eye(5)[axis],dtype=np.float64).ravel()
                expected=np.flatnonzero(np.abs(actual_delta)>TOL64)
                false+=len(np.setdiff1d(e['indices'],expected));miss+=len(np.setdiff1d(expected,e['indices']))
                count+=1
            ends.append(image)
        ends=np.array(ends)
        result[name]=dict(schedules=len(orders),total_updates=count,maximum_pixel_error=float(error),
                          false_indices=false,missed_indices=miss,
                          maximum_endpoint_order_disagreement=float(np.abs(ends-ends[0]).max()))
    return result


def mark_support(ax,indices):
    for idx in indices:
        r,c=divmod(int(idx),28)
        ax.add_patch(Rectangle((c-.5,r-.5),1,1,fill=False,edgecolor='#28754c',linewidth=.45))


def plot_wh(history,summary):
    selected=[0,1,2,4,6]
    fig=plt.figure(figsize=(14.5,10.5),facecolor='#fafaf7')
    fig.text(.035,.97,'1 · Height changes which rows belong to the next widening operation',fontsize=20,weight='bold',va='top')
    fig.text(.035,.924,'Generated 28 × 28 coverage images: white = no ink, black = full ink. Center (14,14), lean 0°; width and height vary.',fontsize=10.5)
    fig.text(.035,.897,'Each operation adds 1 pixel of total width or height. Bottom rows show prospective changes from EACH current state.',fontsize=10.5)
    fig.text(.035,.870,'Green square = a returned pixel index. Red gains ink; blue loses ink. All change maps share the scale −0.5 to +0.5.',fontsize=10.5)
    grid=fig.add_gridspec(3,5,left=.14,right=.97,top=.80,bottom=.22,wspace=.28,hspace=.35)
    for col,step in enumerate(selected):
        item=history[step];p=item['p']
        for row,key in enumerate(['image','width','height']):
            ax=fig.add_subplot(grid[row,col]);ax.set_xticks([]);ax.set_yticks([])
            data=item[key].reshape(28,28)
            if row==0:
                ax.imshow(data,cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
                ax.set_title(f'Step {step}'+(' · root' if step==0 else '')+f'\nw={p[3]:.2f}, h={p[2]:.2f} px',fontsize=10,pad=10)
            else:
                im=ax.imshow(data,cmap='RdBu_r',vmin=-.5,vmax=.5,interpolation='nearest')
                indices=np.flatnonzero(np.abs(data.ravel())>TOL64);mark_support(ax,indices)
                ax.text(.5,-.07,f'{len(indices)} changed indices',transform=ax.transAxes,ha='center',va='top',fontsize=8.5)
    for y,text in zip([.72,.51,.30],['Current image','Next width +1\npixel changes','Next height +1\npixel changes']):
        fig.text(.025,y,text,fontsize=11,va='center')
    cbax=fig.add_axes([.32,.157,.43,.015]);cb=fig.colorbar(im,cax=cbax,orientation='horizontal')
    cb.set_label('Change in ink coverage; indices use zero-based row, column and flat index = 28 × row + column',fontsize=9)
    r=summary['routes'];n=r['coordinates64']['updates']
    fig.text(.035,.087,f'Across {n} tested updates per input route: known coordinates and image-only inputs both returned zero false and zero missed indices (threshold 1e−10).',fontsize=10)
    fig.text(.035,.058,'Sequence: height, width, height, height, width, height. Hidden masks are updated by row/column products; no trained model or full-image predictor.',fontsize=10)
    fig.text(.035,.029,'The trace extends default width/height ranges to show pixel-boundary crossings. All strokes stay positive and unclipped. Unit height adds 0.5 px at each end.',fontsize=9.5)
    fig.savefig(OUT/'01_width_height_indices.png',dpi=150,facecolor=fig.get_facecolor());plt.close(fig)


def plot_wl(summary):
    angles=[0.,5.,15.,30.]
    fig=plt.figure(figsize=(13,10.4),facecolor='#fafaf7')
    fig.text(.035,.972,'2 · The same interval rule follows width and lean across pixel indices',fontsize=19,weight='bold',va='top')
    fig.text(.035,.925,'Generated 28 × 28 coverage images: white = no ink, black = full ink. Center (14,14), height 19.75 px, width 3.2 px.',fontsize=10.5)
    fig.text(.035,.897,'Only lean varies between columns. Green squares mark returned indices; change maps share −0.5 to +0.5 coverage units.',fontsize=10.5)
    fig.text(.035,.869,'Each pixel row uses the original 16 subrows: move its horizontal interval, then sum added and removed coverage.',fontsize=10.5)
    grid=fig.add_gridspec(3,4,left=.16,right=.97,top=.80,bottom=.22,wspace=.30,hspace=.35)
    for col,angle in enumerate(angles):
        p=BASE.copy();p[4]=angle;image=render(p,dtype=np.float64).ravel()
        g=recover(image)
        for row,axis in enumerate([None,3,4]):
            ax=fig.add_subplot(grid[row,col]);ax.set_xticks([]);ax.set_yticks([])
            if row==0:
                ax.imshow(image.reshape(28,28),cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
                ax.set_title(f'Lean {angle:g}°'+(' · root' if angle==0 else ''),fontsize=11,pad=10)
            else:
                e=row_interval_edits(g,axis,current=image)
                delta=np.zeros(784);delta[e['indices']]=e['delta']
                im=ax.imshow(delta.reshape(28,28),cmap='RdBu_r',vmin=-.5,vmax=.5,interpolation='nearest');mark_support(ax,e['indices'])
                ax.text(.5,-.07,f'{len(e["indices"])} indices; no false/missed',transform=ax.transAxes,ha='center',va='top',fontsize=8.5)
    for y,text in zip([.72,.51,.30],['Current image','Width +1 px\npixel changes','Lean +1°\npixel changes']):
        fig.text(.025,y,text,fontsize=11,va='center')
    cbax=fig.add_axes([.32,.157,.43,.015]);cb=fig.colorbar(im,cax=cbax,orientation='horizontal')
    cb.set_label('Red gains ink, blue loses ink. Signed subrow contributions are combined before returning pixel indices.',fontsize=9)
    r=summary['routes'];n=r['coordinates64']['updates']
    fig.text(.035,.087,f'Across {n} tested updates per input route: zero false/missed indices for both known coordinates and float64 images (threshold 1e−10).',fontsize=10)
    fig.text(.035,.058,'Widths 1.8–4.6 px and lean −10° to +35° are sampled, including event cases. A unit move may extend the default box; no canvas clipping occurs.',fontsize=9.5)
    fig.text(.035,.029,'The image-only rule reads row sums and two cumulative-ink boundaries. The geometric interval law is supplied explicitly, not learned from examples.',fontsize=9.5)
    fig.savefig(OUT/'02_width_lean_indices.png',dpi=150,facecolor=fig.get_facecolor());plt.close(fig)


def save_examples(history):
    cases=[('heightened_width',history[1]['p'],3,wh_edits),('heightened_height',history[1]['p'],2,wh_edits),
           ('lean30_width',np.array([14,14,19.75,3.2,30.]),3,row_interval_edits),
           ('lean30_lean',np.array([14,14,19.75,3.2,30.]),4,row_interval_edits)]
    rows=[];examples=[]
    for name,p,axis,rule in cases:
        image=render(p,dtype=np.float64).ravel();state=recover_wh(image) if rule is wh_edits else recover(image);e=rule(state,axis,current=image)
        for i,delta,old,new in zip(e['indices'],e['delta'],e['old'],e['new']):
            r,c=divmod(int(i),28)
            rows.append(dict(case=name,flat_index=int(i),row=r,column=c,old=float(old),change=float(delta),new=float(new)))
        examples.append(dict(case=name,input='current 784-value image only',action=['cx','cy','height','width','lean'][axis],amount=1,
                             changed_pixel_count=len(e['indices']),edits=[r for r in rows if r['case']==name]))
    with (OUT/'example_pixel_edits.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'example_pixel_edits.json').write_text(json.dumps(examples,indent=2)+'\n')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lattice=np.array([[14,14,h,w,0] for h,w in itertools.product([19,19.25,19.75,20,20.5,21,22],[1.8,2,2.4,3,3.2,4,4.6])])
    random=qmc.Sobol(2,scramble=True,seed=20261016).random_base2(7)
    joint=np.tile(BASE,(len(random),1));joint[:,2]=19+1.5*random[:,0];joint[:,3]=1.8+2.8*random[:,1]
    wh,records=evaluate(np.vstack([lattice,joint]),[2,3],'width_height',wh_edits)
    print('Width/height:',json.dumps(wh),flush=True)
    for route in ['coordinates64','image64']:
        assert wh['routes'][route]['false_indices']==wh['routes'][route]['missed_indices']==0
        assert wh['routes'][route]['maximum_pixel_error']<1e-10
    # Width-height first passed to numerical precision; now attempt width-lean.
    lattice=np.array([[14,14,19.75,w,l] for w,l in itertools.product([1.8,2,2.4,3,3.2,4,4.6],[-10,-5,0,.001,1,5,10,17.5,30,35])])
    random=qmc.Sobol(2,scramble=True,seed=20261017).random_base2(7)
    joint=np.tile(BASE,(len(random),1));joint[:,3]=1.8+2.8*random[:,0];joint[:,4]=-10+45*random[:,1]
    wl,new=evaluate(np.vstack([lattice,joint]),[3,4],'width_lean',row_interval_edits);records+=new
    print('Width/lean:',json.dumps(wl),flush=True)
    history,hidden=hidden_check();chains=chain_checks()
    for route in ['coordinates64','image64']:
        assert wl['routes'][route]['false_indices']==wl['routes'][route]['missed_indices']==0
        assert wl['routes'][route]['maximum_pixel_error']<1e-10
    for case in chains.values():
        assert case['false_indices']==case['missed_indices']==0
        assert case['maximum_pixel_error']<1e-10
    plot_wh(history,wh);plot_wl(wl);save_examples(history)
    result=dict(method=dict(root=BASE.tolist(),primary='sparse interval overlap edits; no learned model, no whole-image generation inside the update',
                            coordinate_input='five true coordinates; fixed other controls as declared in each slice',
                            image_input='784 values only; closed-form row/cumulative sums recover needed boundaries',
                            index_convention='zero-based row and column; flat index = 28*row + column',
                            units='width +1 horizontal pixel; height +1 total pixel; lean +1 degree',
                            support_threshold_float64=TOL64,support_threshold_float32=TOL32,
                            scope='positive unclipped parallelogram strokes, width-height and width-lean slices; original 16-subrow integration',
                            assumptions='known stroke/coverage structure and valid source images; no general-image or membership claim',
                            float32='separate default-storage sensitivity check against float32 renderer outputs'),
                width_height=wh,width_lean=wl,hidden_masks=hidden,image_only_chains=chains)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'all_update_metrics.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    np.savez_compressed(OUT/'width_height_trace.npz',states=np.array([x['p'] for x in history]),
                        images=np.array([x['image'] for x in history]),width_masks=np.array([x['width'] for x in history]),
                        height_masks=np.array([x['height'] for x in history]))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
