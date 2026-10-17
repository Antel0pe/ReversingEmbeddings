"""Reversible registration of physical direction maps; geometry-assisted, not blind discovery.

Run: .venv/bin/python analyze_generated_one_edge_alignment.py
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/edge-alignment-mpl')
from pathlib import Path
import json
import csv
import itertools
import numpy as np
from scipy.stats import qmc
from grey_ones import render,RANGES,KNOBS
from invert_grey import invert
from analyze_generated_one_manifold_transport import jet
from analyze_generated_one_knob_direction_families import basis_fit,project,error,kernel_fit

OUT=Path(__file__).resolve().parent/'figures/generated_one_edge_alignment'
OLD=Path(__file__).resolve().parent/'figures/generated_one_knob_direction_families'
SEED=20261018
NAMES=['Horizontal center','Vertical center','Height','Width','Lean']
COLORS=['#007aaf','#db7a12','#7e51a0','#16896c','#c8476b']
METHODS=['raw','center','boundary']
DIMS=[1,2,3,5,8,12,15,20,32,48,64,80,96,112,128,192,256,320]
R,C=np.indices((28,28));R=R.ravel();C=C.ravel()


def recover(images):
    """Existing geometry-based inverse, evaluated on image pixels only."""
    return np.stack([invert(x) for x in images])


def mapping(p,method,branch=None):
    x,y,h,w,angle=p;s=np.tan(np.deg2rad(angle));center=x-s*(R+.5-y)
    if method=='raw':return np.arange(784),np.ones(784,bool),784
    if method=='center':
        # Reversible circular pixel shifts, without interpolation or scaling.
        rr=(R-int(np.rint(y))+14)%28
        cc=(C-np.rint(center).astype(int)+14)%28
        return rr*28+cc,np.ones(784,bool),784
    upper=R+.5<y;left=C+.5<center
    floor_top=np.floor(y-h/2);ceil_bottom=np.ceil(y+h/2)
    floor_left=np.floor(center-w/2);floor_right=np.floor(center+w/2)
    if branch is not None:
        # A requested one-sided derivative chooses the limiting registration
        # chart. Five geometry scalars + known knob label + one side bit suffice.
        knob,side=branch;dp=np.zeros(5);dp[knob]=side
        prime=np.pi/180*(1+s*s)
        dcenter=dp[0]+s*dp[1]-prime*(R+.5-y)*dp[4]
        def limit_round(value,derivative,ceil=False):
            value=np.asarray(value);derivative=np.asarray(derivative)
            nearest=np.rint(value);tie=np.abs(value-nearest)<1e-10
            default=np.ceil(value) if ceil else np.floor(value)
            chosen=nearest+(derivative>0) if ceil else nearest-(derivative<0)
            return np.where(tie,chosen,default)
        upper=np.where(np.abs(R+.5-y)<1e-10,dp[1]>0,upper)
        left=np.where(np.abs(C+.5-center)<1e-10,dcenter>0,left)
        floor_top=limit_round(y-h/2,dp[1]-dp[2]/2)
        ceil_bottom=limit_round(y+h/2,dp[1]+dp[2]/2,ceil=True)
        floor_left=limit_round(center-w/2,dcenter-dp[3]/2)
        floor_right=limit_round(center+w/2,dcenter+dp[3]/2)
    quadrant=(~upper).astype(int)*2+(~left)
    # One-row halos retain one-sided cap directions on exact row boundaries.
    yy=np.where(upper,R-(int(floor_top)-1),int(ceil_bottom)-R)
    edge=np.where(left,center-w/2,center+w/2)
    xx=np.where(left,C-floor_left,floor_right-C).astype(int)+2
    valid=(yy>=0)&(yy<14)&(xx>=0)&(xx<8)
    return quadrant*112+yy*8+xx,valid,448


def encode(d,p,method,branches=None):
    # d: samples x families x pixels. No direction values are interpolated.
    size=448 if method=='boundary' else 784
    out=np.zeros((len(d),d.shape[1],size))
    lost=0.
    for i,(x,setting) in enumerate(zip(d,p)):
        index,ok,_=mapping(setting,method,None if branches is None else branches[i])
        lost=max(lost,float(np.max(np.abs(x[:,~ok]),initial=0)))
        assert len(np.unique(index[ok]))==np.count_nonzero(ok)
        out[i][:,index[ok]]=x[:,ok]
    assert lost<1e-12,(method,lost)
    return out


def decode(z,p,method,branches=None):
    output=np.zeros((len(z),784))
    for i,(x,setting) in enumerate(zip(z,p)):
        index,ok,_=mapping(setting,method,None if branches is None else branches[i])
        output[i,ok]=x[index[ok]]
    return output


def test_model(truth,encoded,p,method,basis,k):
    pred=project(encoded,np.zeros(encoded.shape[1]),basis,k)
    decoded=decode(pred,p,method)
    rec=error(truth,decoded)
    rec['encoded_max_relative']=error(encoded,pred)['max']
    # Some padded patch cells have no target pixel for a particular geometry.
    used=encode(decoded[:,None,:],p,method)[:,0]
    rec['max_discarded_patch_L2']=float(np.linalg.norm(pred-used,axis=1).max())
    return dict(dim=int(k),**rec)


def all_dirs(p):return np.stack([jet(x)[3].T for x in p])


def additional_event_audit(arrays):
    rng=np.random.default_rng(SEED+77)
    points=[];categories=[]
    for kind in ['upright_integer_edges','integer_cap','subrow_cap','center_partition_tie']:
        for i in range(8):
            p=RANGES[:,0]+rng.random(5)*np.ptp(RANGES,axis=1)
            if kind=='upright_integer_edges':
                p[4]=0;p[0]=[14.,14.5,15.][i%3];p[3]=3. if p[0]==14.5 else [2.,4.][i%2]
            elif kind=='integer_cap':
                top=[4.,5.][i%2]
                p[1]=(14.+rng.random()*.25) if top==4 else (14.5+rng.random()*.5)
                p[2]=2*(p[1]-top)
            elif kind=='subrow_cap':
                top=4.125;p[1]=14.+rng.random()*.25;p[2]=2*(p[1]-top)
            else:p[0]=14.5;p[1]=14.5;p[4]=0
            assert np.all((p>=RANGES[:,0])&(p<=RANGES[:,1]))
            points.append(p);categories.append(kind)
    image=render(points,dtype=np.float64);inferred=recover(image);inferred32=recover(render(points))
    truth=[];prediction=[];prediction32=[];fd_errors=[];records=[]
    for i,(p,ip) in enumerate(zip(points,inferred)):
        for j,name in enumerate(KNOBS):
            basis=arrays['boundary_'+name+'_basis']
            for side in [-1,1]:
                eps=1e-7 if j<4 else 1e-6
                far=p.copy();near=p.copy();far[j]+=side*eps;near[j]+=side*eps/2
                limit=2*jet(near)[3][:,j]-jet(far)[3][:,j]
                fd=(render(far,dtype=np.float64).ravel()-image[i].ravel())/(side*eps)
                branch=[(j,side)]
                enc=encode(limit[None,None,:],ip[None],'boundary',branches=branch)[:,0]
                pred=decode(project(enc,np.zeros(448),basis,len(basis)),ip[None],'boundary',branches=branch)[0]
                enc32=encode(limit[None,None,:],inferred32[i:i+1],'boundary',branches=branch)[:,0]
                pred32=decode(project(enc32,np.zeros(448),basis,len(basis)),inferred32[i:i+1],'boundary',branches=branch)[0]
                truth.append(limit);prediction.append(pred);prediction32.append(pred32);fd_errors.append(np.linalg.norm(fd-limit)/np.linalg.norm(limit))
                records.append(dict(category=categories[i],state=i,knob=name,side=side,relative_error=float(np.linalg.norm(pred-limit)/np.linalg.norm(limit))))
    return dict(states=np.asarray(points).tolist(),categories=categories,derivative_count=len(truth),records=records,
                summary=error(np.array(truth),np.array(prediction)),float32_geometry_summary=error(np.array(truth),np.array(prediction32)),max_fd_relative_error=float(max(fd_errors)),
                selection='32 new states generated independently after choosing side-aware registration, no basis refit or model tuning')


def run():
    OUT.mkdir(parents=True,exist_ok=True)
    old=np.load(OLD/'arrays.npz')
    original=old['settings'][:768];grid=old['settings'][768:]
    # Fresh evaluation independent of the earlier family experiment.
    rng=np.random.default_rng(SEED)
    random=RANGES[:,0]+rng.random((512,5))*np.ptp(RANGES,axis=1)
    levels=[1e-5,.319,.681,1-1e-5]
    edge=RANGES[:,0]+np.array(list(itertools.product(levels,repeat=5)))*np.ptp(RANGES,axis=1)
    states=np.concatenate([original,grid,random,edge])
    images=render(states,dtype=np.float64)
    inferred=recover(images)
    print('Recovering alignment from',len(states),'source images',flush=True)
    directions=np.concatenate([old['directions'],all_dirs(np.concatenate([random,edge]))])
    assert len(states)==len(directions)
    splits={'original_random':np.arange(512,768),'original_edge':np.arange(768,1011),
            'fresh_random':np.arange(1011,1523),'fresh_edge':np.arange(1523,len(states))}
    result=dict(seed=SEED,training_states=384,validation_states=128,
        splits={k:len(v) for k,v in splits.items()},source_inverse='existing invert_grey.invert; geometry supplied by known renderer assumptions, parameters recovered from image pixels',
        max_parameter_recovery_error=np.max(np.abs(inferred-states),axis=0).tolist(),
        parameter_order=KNOBS,geometry_scalar_count={'raw':0,'center':3,'boundary':5},
        encoded_dimensions={'raw':784,'center':784,'boundary':448},methods={},joint_checks={},
        limits=['The SVD encoder consumes an observed direction; its coefficients are not a learned state-to-direction law.',
                'Image alignment uses known straight-stroke geometry and an existing inverse; this is not blind coordinate discovery.',
                'Integer registration is reversible and uses no resampling; pixel/subrow phase remains.',
                'A generic unregistered boundary event may have two one-sided derivatives; these are tested separately.',
                'No spatial map, phase list, cluster label or parameter is stored beyond the counted alignment scalars.'])
    arrays={'settings':states,'inferred_settings':inferred,'images':images,'directions':directions}
    for method in METHODS:
        print('Fitting',method,flush=True)
        encoded=encode(directions,inferred,method)
        # Check that registration itself is exactly reversible, prior to SVD.
        for j in range(5):assert np.max(np.abs(decode(encoded[:,j],inferred,method)-directions[:,j]))==0
        rows=[]
        for j,name in enumerate(KNOBS):
            train=encoded[:384,j]
            _,s,b=basis_fit(train,False)
            arrays[method+'_'+name+'_basis']=b
            dims=sorted(set([k for k in DIMS if k<=len(b)]+[len(b)]))
            record=dict(knob=name,rank=len(b),singular_values=s.tolist(),curves={})
            for split,idx in splits.items():
                record['curves'][split]=[test_model(directions[idx,j],encoded[idx,j],inferred[idx],method,b,k) for k in dims]
            record['min_tested_basis_count_for_1pct_worst_fresh']=next((k for k in dims if all(next(v for v in record['curves'][sp] if v['dim']==k)['max']<=.01 for sp in ['fresh_random','fresh_edge'])),None)
            rows.append(record)
        result['methods'][method]=rows
        if method=='boundary':
            arrays['aligned_directions']=encoded
            # Prediction is a separate experiment: only the source image enters.
            normalized=(inferred-RANGES[:,0])/np.ptp(RANGES,axis=1)
            predictors=[]
            for j,name in enumerate(KNOBS):
                pred,meta=kernel_fit(normalized[:384],encoded[:384,j],normalized[384:512],encoded[384:512,j],normalized[1011:],
                    gammas=(.125,.5,2,8),alphas=(1e-8,1e-5,.001,.1))
                pd=decode(pred,inferred[1011:],method)
                predictors.append(dict(knob=name,hyperparameters=meta,
                    fresh_random=error(directions[1011:1523,j],pd[:512]),fresh_edge=error(directions[1523:,j],pd[512:])))
            result['source_only_rbf_prediction']=predictors
    # Compare full aligned and raw 5-pattern decoders on simultaneous knob changes.
    idx=splits['fresh_random'];delta=rng.normal(size=(len(idx),5))*np.ptp(RANGES,axis=1)
    truth=np.einsum('nj,njp->np',delta,directions[idx])
    for method in ['raw','boundary']:
        encoded=encode(directions[idx],inferred[idx],method)
        for count in [5,'full']:
            decoded=[]
            for j,name in enumerate(KNOBS):
                b=arrays[method+'_'+name+'_basis'];k=len(b) if count=='full' else 5
                decoded.append(decode(project(encoded[:,j],np.zeros(encoded.shape[2]),b,k),inferred[idx],method))
            mix=np.einsum('nj,njp->np',delta,np.stack(decoded,axis=1))
            result['joint_checks'][method+'_'+str(count)]=error(truth,mix)
    # Finite moves: compare both true and decoded first-order predictions, separating
    # integration/curvature error from direction reconstruction error.
    finite=[];start=states[1011:1043];start_im=images[1011:1043].reshape(32,784)
    direction=rng.normal(size=(32,5));direction/=np.linalg.norm(direction,axis=1)[:,None]
    decoded=[]
    encoded=arrays['aligned_directions'][1011:1043]
    for j,name in enumerate(KNOBS):
        b=arrays['boundary_'+name+'_basis'];decoded.append(decode(project(encoded[:,j],np.zeros(448),b,len(b)),inferred[1011:1043],'boundary'))
    decoded=np.stack(decoded,axis=1)
    for size in [1e-5,1e-4,1e-3,.01,.1]:
        endpoint=np.clip(start+size*direction*np.ptp(RANGES,axis=1),RANGES[:,0],RANGES[:,1]);step=endpoint-start
        change=render(endpoint,dtype=np.float64).reshape(32,784)-start_im
        true_linear=np.einsum('nj,njp->np',step,directions[1011:1043])
        fit_linear=np.einsum('nj,njp->np',step,decoded)
        finite.append(dict(normalized_step=size,true_first_order=error(change,true_linear),registered_first_order=error(change,fit_linear)))
    result['finite_joint_moves']=finite
    # The old fully covered raw spaces overlap substantially across knob labels.
    overlap=[]
    for source,target in [('height','width'),('cx','lean'),('lean','cx'),('cx','width'),('cx','cy')]:
        u=old[source+'_coverage_basis'];v=old[target+'_coverage_basis']
        overlap.append(dict(source=source,target=target,source_rank=len(u),target_rank=len(v),
            max_unit_basis_projection_residual=float(np.linalg.norm(u-(u@v.T)@v,axis=1).max())))
    result['earlier_raw_span_overlap']=overlap
    # Share three registered fields across all five physical knobs using the
    # independently verified nonlinear identities, rather than five libraries.
    fresh_idx=np.arange(1011,len(states));ed=arrays['aligned_directions'][fresh_idx]
    decoded_three={}
    for j in [0,2,4]:
        b=arrays['boundary_'+KNOBS[j]+'_basis']
        decoded_three[j]=decode(project(ed[:,j],np.zeros(448),b,len(b)),inferred[fresh_idx],'boundary')
    dc=decoded_three[0];dh=decoded_three[2];dl=decoded_three[4]
    slope=np.tan(np.deg2rad(inferred[fresh_idx,4]))[:,None]
    capsign=np.where(R[None,:]+.5<inferred[fresh_idx,1,None],-1.,1.)
    all_five=np.stack([dc,slope*dc+2*capsign*dh,dh,np.abs(dc)/2,dl],axis=1)
    result['shared_three_field_decoder']=dict(primitive_fields=['cx','height','lean'],
        basis_coefficients=sum(len(arrays['boundary_'+KNOBS[j]+'_basis']) for j in [0,2,4]),
        geometry_scalars=5,total_scalars=5+sum(len(arrays['boundary_'+KNOBS[j]+'_basis']) for j in [0,2,4]),
        families=[dict(knob=name,**error(directions[fresh_idx,j],all_five[:,j])) for j,name in enumerate(KNOBS)],
        note='Uses known nonlinear family identities; this is a shared compression decoder, not a newly learned state-only predictor.')
    # One-sided limits at exact lateral / cap events. The pixels at the exact source
    # choose registration; the limiting derivative is obtained from its own side.
    event_states=np.array([[14.,14.,20.,4.,0.],[14.5,14.,19.75,3.,0.],
        [14.43,14.,19.75,3.17,17.3],[15.,15.,19.,4.6,35.],[14.,14.,20.5,1.8,-10.]])
    event_images=render(event_states,dtype=np.float64);event_inferred=recover(event_images)
    event_records=[];evtrue=[];evpred=[];evside=[]
    for ii,(p,ip) in enumerate(zip(event_states,event_inferred)):
        for j,name in enumerate(KNOBS):
            for sign in [-1,1]:
                eps=1e-7 if j<4 else 1e-6
                a=p.copy();b=p.copy();a[j]+=sign*eps;b[j]+=sign*eps/2
                limit=2*jet(b)[3][:,j]-jet(a)[3][:,j]
                fd=(render(a,dtype=np.float64).ravel()-event_images[ii].ravel())/(sign*eps)
                ec=encode(limit[None,None,:],ip[None],'boundary')[:,0]
                basis=arrays['boundary_'+name+'_basis']
                fitted=decode(project(ec,np.zeros(448),basis,len(basis)),ip[None],'boundary')[0]
                branch=[(j,sign)]
                side_encoded=encode(limit[None,None,:],ip[None],'boundary',branches=branch)[:,0]
                side_fitted=decode(project(side_encoded,np.zeros(448),basis,len(basis)),ip[None],'boundary',branches=branch)[0]
                event_records.append(dict(state=ii,knob=name,side=int(sign),
                    side_aware_relative_error=float(np.linalg.norm(side_fitted-limit)/np.linalg.norm(limit)),
                    fd_relative_error=float(np.linalg.norm(fd-limit)/np.linalg.norm(limit)),
                    reconstruction_relative_error=float(np.linalg.norm(fitted-limit)/np.linalg.norm(limit))))
                evtrue.append(limit);evpred.append(fitted);evside.append(side_fitted)
    result['exact_event_checks']=dict(states=event_states.tolist(),records=event_records,
        summary=error(np.array(evtrue),np.array(evpred)),side_aware_summary=error(np.array(evtrue),np.array(evside)),
        side_aware_metadata='At events: five source-derived geometry scalars, known knob identity, plus one bit selecting the requested one-sided derivative. Rounding ties use an explicit tolerance of 1e-10.',
        note='At box boundaries a side can leave the sampling box; render supports those infinitesimal moves. Event tests may expose an unseen registration branch.')
    arrays['event_images']=event_images;arrays['event_true']=np.array(evtrue);arrays['event_single_chart']=np.array(evpred);arrays['event_side_chart']=np.array(evside)
    # Loss of source precision is a separate practical stress test.
    image32=render(states[1011:],dtype=np.float32);inferred32=recover(image32)
    z32=encode(directions[1011:],inferred32,'boundary')
    precision=[]
    for j,name in enumerate(KNOBS):
        b=arrays['boundary_'+name+'_basis'];pr=decode(project(z32[:,j],np.zeros(448),b,len(b)),inferred32,'boundary')
        precision.append(dict(knob=name,aligned_decode_from_float32_geometry=error(directions[1011:,j],pr)))
    result['additional_event_audit']=additional_event_audit(arrays)
    result['float32_alignment_stress']=dict(max_setting_errors=np.max(np.abs(inferred32-states[1011:]),axis=0).tolist(),families=precision)
    np.savez_compressed(OUT/'arrays.npz',**arrays)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'curves.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['method','knob','split','dim','median','p95','max','rms','max_pixel','max_L2','encoded_max_relative','max_discarded_patch_L2']);writer.writeheader()
        for method,rows in result['methods'].items():
            for row in rows:
                for split,points in row['curves'].items():
                    for point in points:writer.writerow(dict(method=method,knob=row['knob'],split=split,**point))
    figures(result,arrays)
    report(result,arrays)
    print(json.dumps(dict(ranks={m:[q['rank'] for q in rows] for m,rows in result['methods'].items()},
        fresh_errors={m:[q['curves']['fresh_edge'][-1]['max'] for q in rows] for m,rows in result['methods'].items()},
        event=result['exact_event_checks']['side_aware_summary'],additional_event=result['additional_event_audit']['summary'],prediction=[x['fresh_random']['median'] for x in result['source_only_rbf_prediction']]),indent=2),flush=True)


def figbase(title,subtitle,size):
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=size,facecolor='#faf9f6')
    fig.text(.045,.97,title,va='top',fontsize=18,weight='bold')
    fig.text(.045,.925,subtitle,va='top',fontsize=10,linespacing=1.5)
    return fig


def figures(r,a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    # One familiar vector shown before and after the reversible operation.
    fig=figbase('Does putting the edges in the same places reveal a shared pattern?',
        'Three random held-out source images; each row shows its horizontal-center direction before registration and four registered corner patches.\n'
        'Source coverage: white 0, black 1. Direction coefficients: red gains ink, blue loses ink. All direction panels use one scale; no clipping or interpolation.',(16,10))
    idx=[1011,1012,1013];limit=float(np.abs(a['directions'][idx,0]).max())
    g=fig.add_gridspec(3,6,left=.05,right=.98,top=.82,bottom=.18,hspace=.5,wspace=.3)
    titles=['Source image','Original direction','Upper left','Upper right','Lower left','Lower right']
    for row,i in enumerate(idx):
        for col in range(6):
            ax=fig.add_subplot(g[row,col]);ax.set_title(titles[col] if row==0 else '',fontsize=10)
            if col==0:ax.imshow(a['images'][i],cmap='gray_r',vmin=0,vmax=1)
            elif col==1:ax.imshow(a['directions'][i,0].reshape(28,28),cmap='RdBu_r',vmin=-limit,vmax=limit)
            else:ax.imshow(a['aligned_directions'][i,0].reshape(4,14,8)[col-2],cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='equal')
            ax.set_xticks([]);ax.set_yticks([])
            if col==0:
                p=a['settings'][i];ax.text(.5,-.11,f'width {p[3]:.2f}, height {p[2]:.2f}\nlean {p[4]:.1f}°',transform=ax.transAxes,ha='center',va='top',fontsize=9)
    cax=fig.add_axes([.30,.11,.4,.018]);fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(-limit,limit),cmap='RdBu_r'),cax=cax,orientation='horizontal').set_label('Raw pixel-change coefficient per +1 horizontal-center pixel (same scale throughout)',fontsize=9)
    fig.text(.045,.02,'Each patch: columns run inward in pixel offsets from its side edge; rows run inward in pixel offsets from its top or bottom cap. Each is 14 × 8 coefficients, including zero padding.\nThe operation only relocates existing coefficients. Five source-image-derived geometry numbers specify where every patch goes back.',fontsize=9)
    fig.savefig(OUT/'01_registration.png',dpi=150);plt.close(fig)
    fig=figbase('How many patterns remain after registering the boundaries?',
        'Every model fits the same 384 direction examples per knob. Bar height = nonzero SVD rank (relative singular-value threshold 10⁻¹⁰).\n'
        'Gray: original pixel coordinates. Blue: centerline registration. Green: four boundary patches. Alignment metadata is additional to these coefficient counts.',(14,7))
    ax=fig.add_axes([.09,.25,.86,.53]);xx=np.arange(5)
    for k,(method,label,color) in enumerate(zip(METHODS,['Raw pixels','Centerline registration (+3 scalars)','Boundary registration (+5 scalars)'],['#929b9d','#007aaf','#16896c'])):
        values=[z['rank'] for z in r['methods'][method]]
        bars=ax.bar(xx+(k-1)*.25,values,.23,color=color,label=label)
        ax.bar_label(bars,padding=4,fontsize=10)
    ax.set_xticks(xx,NAMES);ax.set_ylabel('Number of fitted direction coefficients');ax.set_ylim(0,285);ax.grid(axis='y',alpha=.13)
    fig.legend(*ax.get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.09),frameon=False,ncol=1,fontsize=10)
    fig.text(.045,.025,'These ranks describe observed direction vectors; their coefficients are computed from the vector being encoded. The model does not infer those coefficients\nfrom the source image. Full-rank held-out reconstruction is tested separately, including fresh near-edge states and exact-event one-sided directions.',fontsize=9)
    fig.savefig(OUT/'02_basis_counts.png',dpi=150);plt.close(fig)
    fig=figbase('Does the aligned basis actually reconstruct unseen directions?',
        'Solid: four-patch registration. Dotted: raw pixel coordinates. All models fit 384 training images. Error is measured after decoding into all 784 original pixels.\n'
        'Left: worst relative error among 512 fresh random states. Right: worst among 1,024 fresh near-edge grid states. Dashed horizontal line: 10⁻¹⁰ target.',(15,8))
    g=fig.add_gridspec(1,2,left=.08,right=.97,bottom=.25,top=.79,wspace=.23)
    for i,split in enumerate(['fresh_random','fresh_edge']):
        ax=fig.add_subplot(g[0,i])
        for j in range(5):
            for method,style in [('raw',':'),('boundary','-')]:
                row=r['methods'][method][j]['curves'][split]
                ax.plot([q['dim'] for q in row],[max(q['max'],1e-16) for q in row],style,color=COLORS[j],lw=1.6,label=NAMES[j] if method=='boundary' else None)
        ax.set_xscale('log');ax.set_yscale('log');ax.set_ylim(1e-16,3);ax.set_xlim(1,300);ax.axhline(1e-10,color='#666',ls='--',lw=.8)
        ax.set_xlabel('Number of direction coefficients (log scale)');ax.set_ylabel('Worst relative vector reconstruction error');ax.grid(alpha=.15)
        ax.set_title('Fresh random images' if i==0 else 'Fresh near-edge images')
    fig.legend(*fig.axes[0].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.09),ncol=3,frameon=False)
    fig.text(.045,.025,'Near-edge samples approach the range-box boundaries but avoid exact renderer kinks. Exact-event one-sided directions are reported separately.\nSVD null-space completions are excluded. Geometry is recovered from each source image using the existing known-shape inverse.',fontsize=9)
    fig.savefig(OUT/'03_reconstruction.png',dpi=150);plt.close(fig)

    fig=figbase('At an exact boundary, which side of the move matters',
        'Start: upright stroke, center (14, 14), height 20, width 4. Increase height from an exact pixel-row boundary.\n'
        'The outward one-sided direction gains ink in two previously empty rows. The chart must use that side of the boundary.',(15,5.5))
    # Event order: first state, height knob index 2, positive side -> vector 5.
    i=5;v=a['event_true'][i];wrong=a['event_single_chart'][i];right=a['event_side_chart'][i]
    g=fig.add_gridspec(1,4,left=.06,right=.97,top=.72,bottom=.22,wspace=.25)
    maps=[a['event_images'][0].ravel(),v,wrong,right]
    titles=['Source coverage (0–1)','True outward height direction','Decoded with source-side chart','Decoded with side-aware chart']
    lim=float(np.max(np.abs(np.stack([v,wrong,right]))))
    for c,(values,title) in enumerate(zip(maps,titles)):
        ax=fig.add_subplot(g[0,c]);ax.set_title(title,fontsize=10);ax.set_xticks([]);ax.set_yticks([])
        if c==0:ax.imshow(values.reshape(28,28),cmap='gray_r',vmin=0,vmax=1)
        else:ax.imshow(values.reshape(28,28),cmap='RdBu_r',vmin=-lim,vmax=lim)
        if c>=2:
            er=np.linalg.norm(values-v)/np.linalg.norm(v)
            ax.text(.5,-.12,f'relative error {er:.2e}',ha='center',transform=ax.transAxes,color='#16896c' if c==3 else '#b74343',fontsize=10)
    fig.text(.06,.085,f'Direction scale: blue = −{lim:.2f}, white = 0, red = +{lim:.2f} ink coefficient per height pixel; shared scale, no clipping.\nSide-aware chart: same five geometry scalars and same learned patterns, plus a bit choosing which side of the event is requested.',fontsize=10)
    fig.savefig(OUT/'04_event_charts.png',dpi=150);plt.close(fig)


def report(r,a):
    rows='';prediction=''
    for j,name in enumerate(NAMES):
        rows+='<tr><td>'+name+'</td>'+''.join(f'<td>{r["methods"][m][j]["rank"]}</td>' for m in METHODS)
        rows+=f'<td>{r["methods"]["boundary"][j]["curves"]["fresh_random"][-1]["max"]:.2e}</td><td>{r["methods"]["boundary"][j]["curves"]["fresh_edge"][-1]["max"]:.2e}</td></tr>'
        p=r['source_only_rbf_prediction'][j]
        prediction+=f'<tr><td>{name}</td><td>{100*p["fresh_random"]["median"]:.1f}%</td><td>{100*p["fresh_random"]["max"]:.1f}%</td><td>{100*p["fresh_edge"]["max"]:.1f}%</td></tr>'
    event=r['exact_event_checks']['side_aware_summary']
    budget=5+sum(x['rank'] for x in r['methods']['boundary'])
    shared=r['shared_three_field_decoder']['total_scalars']
    text=f'''# Edge alignment of the five direction families

Run `.venv/bin/python analyze_generated_one_edge_alignment.py` from the repository root.

## What the earlier experiment actually learned

The large SVD libraries were fixed linear pixel patterns, with coefficients obtained by projecting an already-observed direction. They were not newly discovered piecewise motion functions. Existing analytic renderer derivatives supplied the physical knob directions. Two pixelwise nonlinear links between families were established: width is half the absolute horizontal-center direction, and height is half the absolute vertical-center direction after removing tan(lean) times horizontal-center direction. These do not reduce the five-dimensional local tangent rank: absolute value is nonlinear, not a linear dependency between direction vectors.

## Alignment experiment

384 training images, 128 validation images for a separate state-only predictor, 256 original random tests, 243 original near-edge tests, 512 fresh random tests and 1024 fresh near-edge tests. The seed is {SEED}. All five knobs vary. Images use float64 coverage before float32 storage. Alignment settings are recovered from source pixels by the pre-existing `invert_grey.invert`, using the known straight-stroke geometry.

Raw pixels store zero alignment scalars. Centerline registration stores three: horizontal center, vertical center and lean. It applies reversible integer shifts to each pixel row and a reversible integer vertical shift. Boundary registration stores five: the two centers, height, width and lean. It splits the raster at the centerline and vertical center into four corner patches. For each half-row it aligns the corresponding side edge by an integer column shift; rows are measured from the corresponding cap. Each patch has 14 rows and 8 columns, with halos and zero padding. Values are neither interpolated nor scaled. The reverse mapping returns every original nonzero coefficient exactly before reduction.

For one direction, the code length is five geometry numbers plus k learned basis coefficients, with the physical knob label known. At an exact event, the requested one-sided derivative also supplies one discrete side bit per known knob. For all five directions together at a smooth state, the geometry is shared: five geometry numbers plus the sum of the five basis sizes = {budget} continuous numbers at the fitted ranks. The decoder also stores a fixed set of shared basis vectors. Floor/ceil decisions, row shifts and quadrant masks are deterministic functions of the five geometry numbers; no per-row displacement list, unknown cluster label or uncounted phase list is supplied.

## Sharing rules across knob labels

The earlier full raw height span lies within the width span, and the earlier full raw horizontal and lean spans coincide numerically. Unit basis-vector projection residuals are around 1e-14. These are spans pooled across different source images; they do not equate the knobs at a single image.

The two nonlinear identities also let us store only the aligned horizontal, height and lean fields. Width is half the absolute horizontal field; vertical is tan(lean) times horizontal plus the signed cap field obtained from height. This uses {shared} total scalars including geometry on smooth samples, versus {budget} for five separately encoded fields. The identities are supplied mathematical rules, not fitted state-to-coefficient functions. Results are in `shared_three_field_decoder`.

## Limits that matter

The SVD encoder is supplied the actual vector. This is compression, not a state-only motion predictor. We separately fit an RBF predictor from five image-derived settings to the aligned maps, selecting bandwidth and regularization on validation images only. Its held-out errors are reported separately and remain far above numerical precision.

The five geometry numbers are the existing physical coordinates. Together with the known renderer they already specify the image and its derivatives. We do not claim to have discovered a novel five-coordinate description. This experiment measures how much coordinate registration simplifies a generic shared pixel-pattern library.

Integer alignment leaves fractional pixel positions, the sixteen subrow crossings, cap placement within a row and changing angle-dependent speeds. A reduced rank cannot by itself establish which remaining effect causes each pattern. The result is specific to this tested registration, not a lower bound on every possible nonlinear alignment.

## Joint directions and finite moves

At smooth fresh states we combine all five reconstructed directions with random simultaneous knob increments and compare to the same linear combination of the true Jacobian columns. This checks instantaneous joint moves. A finite change is different: true first-order predictions themselves deviate from rendered endpoints as steps cross events or follow curvature. Both true and reconstructed first-order errors are recorded across normalized step sizes 0.00001 through 0.1. No near-exact finite-step decoder is claimed.

## Events and precision

Five exact boundary/intersection states are tested with both one-sided physical directions. Limits are obtained by extrapolating analytic derivatives from that side and checked against one-sided rendered moves. A single source-side registration chart fails for three of 50 vectors, with worst relative error 1.0. Choosing the chart on the requested side of each event reduces the worst error to {event['max']:.3e}, without adding learned patterns. This uses one additional side-selection bit for each requested one-sided direction; the knob identity is already known. Grid-rounding ties use tolerance 1e-10. This repair was motivated by the first five event states; a separate event audit is reported in `additional_event_audit`. These side derivatives are not averaged together or treated as a unique smooth Jacobian. Some sides leave the default range box; rendering allows these infinitesimal probes.

Float32 source images are tested separately for their effect on alignment decisions while keeping the target direction in float64. This isolates geometry/branch selection sensitivity, not the error of taking finite differences of rounded images.

## Artifacts

- `index.html`: illustrated report and interactive source/direction/registered/reconstructed comparison.
- `01_registration.png`, `02_basis_counts.png`, `03_reconstruction.png`, `04_event_charts.png`: explanatory figures.
- `results.json`: all measured summaries, predictor errors, event checks and finite-step checks.
- `curves.csv`: raw and registered reconstruction curves.
- `arrays.npz`: source images, directions, inferred geometry, aligned maps and learned bases.
'''
    (OUT/'README.md').write_text(text)
    idx=[1011,1012,1013,1014,1015,1016,1017,1018]
    payload=dict(names=NAMES,knobs=KNOBS,settings=a['settings'][idx].tolist(),images=a['images'][idx].reshape(-1,784).tolist(),
        directions=a['directions'][idx].tolist(),aligned=a['aligned_directions'][idx].tolist(),
        bases=[a['boundary_'+name+'_basis'].tolist() for name in KNOBS],inferred=a['inferred_settings'][idx].tolist())
    body='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Putting the edges in the same places</title><style>
:root{font:16px/1.65 system-ui,sans-serif;background:#faf9f6;color:#23313a}body{max-width:1180px;margin:auto;padding:28px 24px 70px}h1{font-size:38px;line-height:1.2}h2{margin-top:42px}p{max-width:960px}.note{background:#eef4f1;border-left:4px solid #16896c;padding:15px 20px}.small{font-size:14px;color:#57656b}.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:10px;text-align:left;border-bottom:1px solid #d5ddda}th{background:#edf1ef}img.figure{width:100%;height:auto}a{color:#007aaf}.controls{display:flex;flex-wrap:wrap;gap:20px;align-items:center;background:#edf1ef;padding:18px}select{padding:6px;font:inherit}label{font-size:14px}.maps{display:grid;grid-template-columns:repeat(4,1fr);gap:18px;margin:20px 0}.maps h3{font-size:15px}.maps canvas{width:100%;image-rendering:pixelated}.maps p{font-size:13px}code{font-size:.9em}@media(max-width:700px){body{padding:20px 12px}h1{font-size:29px}.maps{grid-template-columns:repeat(2,1fr)}}
</style><body><p class="small">Controlled generated 1s · geometry-assisted registration · no pixel resampling</p><h1>Is position disguising a simpler direction pattern?</h1>
<div class="note"><b>First, a correction about the previous experiment.</b> Its large libraries were linear SVD patterns. They reconstructed directions supplied by the existing analytic renderer. We did not learn hundreds of piecewise motion functions. The exact nonlinear rules we found link width to horizontal shift and height to center shifts; the five direction families are not isolated rule collections.</div>
<p>This experiment moves direction coefficients into a common boundary frame before fitting SVD. It then moves the decoded coefficients back into the original 28 × 28 image, so success is measured in the original pixels. Alignment must be stored too: three numbers for centerline registration, five for full boundary registration.</p>
<h2>1 · The same operation at different starting images</h2><p>Each source pixel is ink coverage: white means 0, black means 1. Its horizontal-center direction has signed pixel changes: red gains ink, blue loses ink. We split that map into upper-left, upper-right, lower-left and lower-right patches, aligning each side and cap by whole-pixel moves.</p><img class="figure" src="01_registration.png" alt="Starting images and their horizontal change maps followed by four registered corner patches">
<h2>2 · Position was hiding a substantial amount of simplicity</h2><img class="figure" src="02_basis_counts.png" alt="Ranks of the same 384 training direction vectors before and after registration">
<div class="scroll"><table><thead><tr><th>Knob</th><th>Raw rank</th><th>Centerline rank</th><th>Boundary rank</th><th>Worst fresh random error, full boundary basis</th><th>Worst fresh edge error</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<p>The raw ranks above use the original 384 training images. The earlier experiment used additional coverage to reach ranks 260, 265, 95, 267 and 260. Registered maps have fewer patterns and their training span covers unseen registered patterns more effectively. A large reduction is not the same as a collapse to one or two shared patterns.</p>
<p class="note"><b>What is counted:</b> for one direction, store five source-image-derived geometry numbers plus its k SVD coefficients. The five families can share those geometry numbers. Four patches contain 448 slots, including zeros; a fixed trained basis is stored by the decoder. Every integer shift and partition follows from those five numbers, with no per-row position list supplied.</p>
<h2>How much do the rules overlap?</h2><p>The earlier full height library lies within the width library’s linear span. The full horizontal and lean libraries span the same space to numerical precision. These libraries pool vectors from many source images; this does not make two knobs interchangeable at one image.</p><p>The exact nonlinear identities are more specific. With the five geometry numbers shared, the aligned horizontal, height and lean fields reconstruct all five direction maps: width is half the absolute horizontal map, and vertical is tan(lean) times horizontal plus the signed top/bottom caps from height. This shared decoder stores <b>__SHARED__ scalars including geometry</b> across the five fields. It is still an encoder of observed directions, and the shared mathematical rules are supplied.</p>
<h2>3 · Does it survive decoding back to the original pixels?</h2><img class="figure" src="03_reconstruction.png" alt="Held-out relative direction errors for raw and registered bases">
<p>The 512 random and 1,024 near-edge states here are fresh tests. Error is the reconstructed direction’s L2 error divided by the true direction’s L2 length. Near-edge sampling is separate from exact renderer events, where the two one-sided directions may differ.</p>
<h2>4 · Inspect a decoded direction</h2><p>The encoder sees the actual direction and its source image. Choose a small basis or the full learned basis. Red and blue have one common scale in the three original-pixel direction panels; the residual is prediction minus truth.</p>
<div class="controls"><label>Knob <select id="knob">__OPTIONS__</select></label><label>Fresh source <select id="sample"></select></label><label>Patterns <input id="count" type="range" min="1" max="98" value="5"> <output id="number">5</output></label><button id="full">Use full basis</button></div><p id="settings" class="small"></p>
<div class="maps"><div><h3>Source coverage</h3><canvas id="source" width="280" height="280"></canvas><p>White 0 → black 1</p></div><div><h3>True direction</h3><canvas id="truth" width="280" height="280"></canvas><p id="length"></p></div><div><h3>Decoded direction</h3><canvas id="decoded" width="280" height="280"></canvas><p id="error"></p></div><div><h3>Residual</h3><canvas id="residual" width="280" height="280"></canvas><p id="pixel"></p></div></div><p id="scale" class="small"></p>
<h2>5 · Compression is not yet prediction from the image alone</h2><p>For the near-exact reconstruction test, coefficients are computed from the true direction. To test a different claim, we fitted a separate RBF model that sees only the five geometry numbers recovered from the source image, predicts the aligned direction, and decodes it. Its bandwidth and regularization were chosen using 128 validation images.</p><div class="scroll"><table><thead><tr><th>Knob</th><th>Median fresh random error</th><th>Worst fresh random error</th><th>Worst fresh near-edge error</th></tr></thead><tbody>__PREDICTION__</tbody></table></div>
<p class="note">The source-only predictor remains imperfect. Near-exact compression does not establish a learned function that predicts the knob’s direction from the state. The five geometry numbers plus the known analytic renderer already provide that function as a supplied baseline; this experiment does not rediscover it.</p>
<h2>6 · Joint moves and exact boundaries</h2><p>We tested simultaneous changes by combining all five decoded directions at 512 fresh smooth states. We also compared true and decoded first-order predictions against rendered finite joint moves at normalized sizes from 10⁻⁵ to 0.1. Curvature and event crossings produce finite-step error even with exact local directions; those errors are listed separately in the results.</p><p>At five exact event/intersection states, we tested both one-sided directions for all five knobs. A single source-side chart failed on 3 of those 50 vectors, reaching 100% relative error. Choosing the registration chart on the requested side of the event reduced the worst error to <b>__EVENT__</b>, without adding learned patterns. This adds one side-selection bit for a requested one-sided direction; grid-rounding ties use tolerance 10⁻¹⁰. A separate audit of 320 one-sided directions at 32 new event states had worst relative error <b>__AUDIT__</b>, using the same basis and side-aware chart. The derivative limits were checked against one-sided rendered moves. These tests include sides just outside the default sampling box, supported by the renderer. Float32 source-image alignment is also tested separately. For the 320 exact-event audit directions its worst relative error was <b>__FLOAT32_EVENT__</b>; tiny recovery errors can affect an exact chart decision.</p>
<img class="figure" src="04_event_charts.png" alt="An exact boundary event where source-side registration fails and a side-aware chart recovers the direction"><h2>What remains after integer registration?</h2><p>The stroke’s position relative to fractional pixel boundaries still changes. So do the sixteen subrow edge crossings, cap placement within a row, and the lever-arm weights in the lean direction. Whole-pixel alignment removes location variation but preserves those effects. This result measures one reversible registration; it does not rule out a more effective nonlinear representation.</p>
<p><a href="README.md">Method, counts and limitations</a> · <a href="results.json">All measurements</a> · <a href="curves.csv">Reconstruction curves</a> · <a href="arrays.npz">Images, vectors and fitted bases</a> · <a href="../generated_one_knob_direction_families/index.html">Earlier experiment</a></p>
<script type="application/json" id="data">__DATA__</script><script>
'use strict';const d=JSON.parse(document.getElementById('data').textContent),el=id=>document.getElementById(id);for(let i=0;i<d.images.length;i++){const o=document.createElement('option');o.value=i;o.textContent=i+1;el('sample').append(o)}
function paint(id,v,lim,source=false){const c=el(id).getContext('2d');for(let r=0;r<28;r++)for(let k=0;k<28;k++){let x=v[r*28+k];if(source){const g=Math.round(255*(1-x));c.fillStyle=`rgb(${g},${g},${g})`}else{x=Math.max(-1,Math.min(1,x/lim));const m=Math.abs(x),white=Math.round(250*(1-m));c.fillStyle=x>=0?`rgb(${Math.round(250-50*m)},${white},${white})`:`rgb(${white},${Math.round(250-120*m)},${Math.round(250-45*m)})`}c.fillRect(k*10,r*10,10,10)}}
function update(){const j=+el('knob').value,i=+el('sample').value,b=d.bases[j];el('count').max=b.length;el('count').value=Math.min(+el('count').value,b.length);const k=+el('count').value;el('number').textContent=k;const enc=d.aligned[i][j],pred=new Array(448).fill(0);for(let m=0;m<k;m++){let s=0;for(let p=0;p<448;p++)s+=enc[p]*b[m][p];for(let p=0;p<448;p++)pred[p]+=s*b[m][p]}const [x,y,h,w,angle]=d.inferred[i],slope=Math.tan(angle*Math.PI/180),out=new Array(784).fill(0);for(let r=0;r<28;r++)for(let c=0;c<28;c++){const center=x-slope*(r+.5-y),up=r+.5<y,left=c+.5<center,q=(up?0:2)+(left?0:1),yy=up?r-(Math.floor(y-h/2)-1):Math.ceil(y+h/2)-r,edge=center+(left?-w/2:w/2),xx=(left?c-Math.floor(edge):Math.floor(edge)-c)+2;if(yy>=0&&yy<14&&xx>=0&&xx<8)out[r*28+c]=pred[q*112+yy*8+xx]}const truth=d.directions[i][j],res=out.map((v,p)=>v-truth[p]),norm=Math.hypot(...truth),err=Math.hypot(...res)/norm,lim=Math.max(...truth.map(Math.abs),...out.map(Math.abs),...res.map(Math.abs),1e-30);paint('source',d.images[i],1,true);paint('truth',truth,lim);paint('decoded',out,lim);paint('residual',res,lim);el('length').textContent=`True length ${norm.toFixed(4)} per ${j===4?'degree':'pixel'}`;el('error').textContent=`Relative error ${(err*100).toFixed(3)}% (${err.toExponential(2)})`;el('pixel').textContent=`Max pixel error ${Math.max(...res.map(Math.abs)).toExponential(3)}`;el('settings').textContent=d.knobs.map((n,p)=>`${n} = ${d.settings[i][p].toFixed(3)}`).join(' · ');el('scale').textContent=`Common scale: blue −${lim.toFixed(4)} to red +${lim.toFixed(4)}. No clipping. ${k} fitted coefficients + 5 source-derived geometry numbers.`}
for(const id of ['knob','sample','count'])el(id).addEventListener('input',update);el('full').addEventListener('click',()=>{el('count').value=d.bases[+el('knob').value].length;update()});update();
</script></body></html>'''
    for key,value in dict(ROWS=rows,PREDICTION=prediction,OPTIONS=''.join(f'<option value="{j}">{n}</option>' for j,n in enumerate(NAMES)),EVENT=f'{event["max"]:.2e}',AUDIT=f'{r["additional_event_audit"]["summary"]["max"]:.2e}',SHARED=str(shared),FLOAT32_EVENT=f'{r["additional_event_audit"]["float32_geometry_summary"]["max"]:.2e}',DATA=json.dumps(payload,separators=(',',':'))).items():body=body.replace('__'+key+'__',value)
    (OUT/'index.html').write_text(body)

if __name__=='__main__':run()
