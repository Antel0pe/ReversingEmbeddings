"""Can measured SVD/PCA frames support state-derived controls and local formulas?

Core reconstruction uses image displacements and five projected coordinates.
Observable-assisted interpretation uses the previously explicit image readers.
Known renderer derivatives are evaluation targets and sampling initializers only.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/svd-state-bridge-mpl')
from pathlib import Path
import sys,json,time,itertools
import numpy as np
from scipy.optimize import linear_sum_assignment,brentq
from scipy.special import ndtri
from scipy.stats import qmc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from grey_ones import render,RANGES,KNOBS,SUB
from analyze_generated_one_manifold_transport import jet,chart,unchart
from experiments.derived_state_coordinates import recover
OUT=ROOT/'figures/svd_state_coordinate_bridge'
SPAN=np.diff(RANGES,axis=1).ravel()
PAIRS=list(itertools.combinations_with_replacement(range(5),2))


def pixels(p):return render(p,dtype=np.float64).reshape(-1,784)


def features(z,degree=2,constant=False):
    cols=([np.ones(len(z))] if constant else [])+list(z.T)
    if degree==2:cols += [z[:,i]*z[:,j] for i,j in PAIRS]
    return np.column_stack(cols)


def fromq(q):
    q=np.atleast_2d(q);t,b,a,w,s=q.T;cy=(t+b)/2
    return np.column_stack([a-s*cy,cy,b-t,w,np.rad2deg(np.arctan(s))])


def image_metrics(pred,truth,origin):
    residual=pred-truth
    absolute=np.linalg.norm(residual,axis=1)
    target=np.linalg.norm(truth-origin,axis=1)
    relative=absolute/np.maximum(target,1e-30)
    return dict(worst_relative_change=float(relative.max()),median_relative_change=float(np.median(relative)),
                p95_relative_change=float(np.quantile(relative,.95)),worst_image_L2=float(absolute.max()),
                worst_pixel=float(np.abs(residual).max()))


def field_metrics(estimated,truth):
    e=np.linalg.norm(estimated-truth,axis=0)/np.linalg.norm(truth,axis=0)
    return dict(per_knob_relative=e.tolist(),worst_relative=float(e.max()),median_relative=float(np.median(e)))


def readers(x):
    """Exact derivatives of the explicit state observables, not renderer jets."""
    p,q,raw,k=recover(np.asarray(x).reshape(1,28,28));p=p[0];q=q[0];w,u,l=raw[0,:3]
    mw=np.zeros((28,28));mw[10]=1;mw=mw.ravel()
    mu=np.zeros((28,28));mu[:6]=1;mu=mu.ravel()
    ml=np.zeros((28,28));ml[23:]=1;ml=ml.ravel()
    gt=-mu/w+u*mw/w**2
    gb=ml/w-l*mw/w**2
    gm=[]
    for j,row in enumerate([10,18]):
        m=np.zeros((28,28));m[row,:k[0,j]]=1
        gm.append(-m.ravel()+mw/2)
    gs=(gm[0]-gm[1])/8
    ga=gm[0]+10.5*gs
    dq=np.stack([gt,gb,ga,mw,gs])
    s=q[4];cy=p[1]
    physical_from_q=np.array([[-s/2,-s/2,1,0,-cy],[.5,.5,0,0,0],[-1,1,0,0,0],
                              [0,0,0,1,0],[0,0,0,0,180/np.pi/(1+s*s)]])
    return p,q,dq,physical_from_q@dq,physical_from_q


def frame(dx,centered):
    mean=dx.mean(0) if centered else np.zeros(784)
    a=dx-mean;active=np.any(a!=0,axis=0)
    _,s,b=np.linalg.svd(a[:,active],full_matrices=False)
    result=np.zeros((5,784));result[:,active]=b[:5]
    return result,mean,s


def local_samples(p,radius,n,seed,ball=True):
    """Generate observations; parameter arrays never enter learned models."""
    u=qmc.Sobol(6,scramble=True,seed=seed).random_base2(int(np.log2(n)))
    ray=ndtri(u[:,:5]);ray/=np.linalg.norm(ray,axis=1,keepdims=True);velocity=ray*SPAN
    requested=radius*(.05+.95*u[:,5] if ball else np.ones(n))
    start=pixels(p)[0]
    amount=requested/np.linalg.norm(velocity@jet(p)[3].T,axis=1)
    for iteration in range(12):
        points=p+amount[:,None]*velocity;x=pixels(points)
        distances=np.linalg.norm(x-start,axis=1)
        bad=np.abs(distances/requested-1)>1e-7
        if not np.any(bad):break
        amount*=requested/distances
    if np.any(bad):
        for i in np.flatnonzero(bad):
            def objective(t):return np.linalg.norm(pixels(p+t*velocity[i])[0]-start)-requested[i]
            hi=amount[i]*2
            while objective(hi)<0:hi*=2
            amount[i]=brentq(objective,0,hi,xtol=1e-14)
        points=p+amount[:,None]*velocity;x=pixels(points)
    measured=np.linalg.norm(x-start,axis=1)
    assert np.max(np.abs(measured/requested-1))<3e-6
    return points,x


def calibrate(b,dp):
    response=(dp/SPAN[:,None])@b.T
    # Full inversion mixes all five estimated axes, without assigning labels.
    unit_range_fields=np.linalg.solve(response.T,b).T
    return unit_range_fields/SPAN[None,:],response,float(np.linalg.cond(response))


def loaded_cases():
    old=np.load(ROOT/'figures/generated_one_local_bases_across_starts/bases_and_samples.npz')
    fine=np.load(ROOT/'figures/generated_one_scale_separation/smaller_radius_samples.npz')
    names=[k.removesuffix('_start_settings') for k in old.files if k.endswith('_start_settings')]
    for name in names:
        for radius in [.1,.01,.001,.0001,.00001]:
            data=old if radius>=.001 else fine
            key=name+'_r_'+(str(radius).replace('.','p') if radius>=.001 else str(radius))
            if key+'_images' not in data.files:continue
            yield name,old[name+'_start_settings'],radius,data[key+'_images'][:512],data[key+'_images'][512:]
    # Fresh five-dimensional start control, not only the familiar 17 states.
    u=qmc.Sobol(5,scramble=True,seed=2026101801).random_base2(5)
    points=RANGES[:,0]+(.05+.9*u)*SPAN
    for i,p in enumerate(points):
        _,x=local_samples(p,.001,512,2026101802+i,False)
        _,test=local_samples(p,.001,256,2026101902+i,False)
        yield f'new_{i+1:02d}',p,.001,x,test


def graph_model(b,x0,xtrain,xtest,commands,qtarget):
    """No settings or renderer functions enter the fitting/prediction below.

    Core decoder uses only projected pixels. State-to-geometry maps are
    explicitly separate readout fits, used to interpret/control this chart.
    """
    p0,q0,dq,dp,k=readers(x0)
    train_p,train_q,_,_=recover(xtrain.reshape(-1,28,28))
    test_p,test_q,_,_=recover(xtest.reshape(-1,28,28))
    z=(xtrain-x0)@b.T
    zscale=np.maximum(np.std(z,axis=0),1e-30)
    z/=zscale
    testz=((xtest-x0)@b.T)/zscale
    qscale=np.maximum(np.std(train_q-q0,axis=0),1e-30)
    train_geom=(train_q-q0)/qscale
    target_geom=(qtarget-q0)/qscale
    results={};models={}
    for degree in [1,2]:
        design=features(z,degree)
        beta=np.linalg.lstsq(design,xtrain-x0,rcond=None)[0]
        qbeta=np.linalg.lstsq(design,train_geom,rcond=None)[0]
        inverse=np.linalg.lstsq(features(train_geom,degree),z,rcond=None)[0]
        decoded=x0+features(testz,degree)@beta
        mappedq=q0+(features(testz,degree)@qbeta)*qscale
        mappedp=fromq(mappedq)
        predictedz=features(target_geom,degree)@inverse
        controlled=x0+features(predictedz,degree)@beta
        # Observable response unmixes the local derivative of the image-only
        # polynomial, including its fitted directions outside the raw span.
        graph_j=beta[:5].T
        graph_q=(qbeta[:5]*qscale).T
        physical_response=(k@graph_q)/SPAN[:,None]
        jphysical=np.linalg.solve(physical_response.T,graph_j.T).T/SPAN[None,:]
        delta=(test_p-p0)/SPAN
        map_error=(mappedp-test_p)/SPAN
        semantic_relative=np.linalg.norm(map_error,axis=1)/np.maximum(np.linalg.norm(delta,axis=1),1e-30)
        results[str(degree)]=dict(
            decoder=image_metrics(decoded,xtest,x0),
            command_prediction=image_metrics(controlled,xtest,x0),
            mapping_worst_relative_normalized_change=float(semantic_relative.max()),
            mapping_worst_absolute_physical=np.max(np.abs(mappedp-test_p),axis=0).tolist(),
            graph_tangent_condition=float(np.linalg.cond(physical_response)),
            design_condition=float(np.linalg.cond(design)))
        models[str(degree)]=dict(beta=beta,qbeta=qbeta,inverse=inverse,zscale=zscale,qscale=qscale,
                               jphysical=jphysical,prediction=controlled,mappedq=mappedq)
    # Split only at derived top/bottom subrow boundaries. This uses supplied
    # geometry to choose pieces, but coefficients still come only from images.
    train_keys=np.floor(train_q[:,:2]*SUB+1e-10).astype(int)
    target_keys=np.floor(qtarget[:,:2]*SUB+1e-10).astype(int)
    piecewise=models['2']['prediction'].copy();miss=0;branches=[]
    for key in np.unique(target_keys,axis=0):
        tr=np.all(train_keys==key,axis=1);te=np.all(target_keys==key,axis=1)
        if tr.sum()<32:
            miss+=int(te.sum());continue
        design=features(z[tr],2,True)
        inv_design=features(train_geom[tr],2,True)
        if np.linalg.matrix_rank(design,tol=1e-10)<21 or np.linalg.matrix_rank(inv_design,tol=1e-10)<21:
            miss+=int(te.sum());continue
        coef=np.linalg.lstsq(design,xtrain[tr]-x0,rcond=None)[0]
        inv=np.linalg.lstsq(inv_design,z[tr],rcond=None)[0]
        pred_z=features(target_geom[te],2,True)@inv
        piecewise[te]=x0+features(pred_z,2,True)@coef
        branches.append(dict(key=key.tolist(),training=int(tr.sum()),checking=int(te.sum())))
    results['piecewise_degree_2']=dict(command_prediction=image_metrics(piecewise,xtest,x0),
        fallback_test_states=miss,branches=branches,
        geometry_assumption='Split by top and bottom 1/16-pixel subrow cells; not learned from vectors alone.')
    return results,models


def case_study():
    rows=[];arrays={};start=time.monotonic()
    for case_index,(name,p,radius,neighbors,old_test) in enumerate(loaded_cases()):
        x0=pixels(p)[0];p0,q0,dq,dp,k=readers(x0)
        _,jtrueq,_,jtrue,_=jet(p)
        # State-reader chain rule checked independently; no derivative target
        # is passed to frame calibration or polynomial fitting.
        assert np.max(np.abs(dp@jtrue-np.eye(5)))<1e-10
        train_p,xtrain=local_samples(p,radius,512,2026102000+case_index*2)
        test_p,xtest=local_samples(p,radius,256,2026102001+case_index*2)
        commands=test_p-p
        # Only source image-derived settings and requested increments enter.
        targetq=np.array([chart(t) for t in p0+commands])
        for method in ['svd','pca']:
            b,mean,s=frame(neighbors-x0,method=='pca')
            jcal,response,cond=calibrate(b,dp)
            unittrue=jtrue/np.linalg.norm(jtrue,axis=0)
            cosine=np.abs(unittrue.T@b.T)
            labels,indices=linear_sum_assignment(-cosine)
            order=indices[np.argsort(labels)]
            # Oracle assignment makes this single-axis control optimistic;
            # full five-axis calibration uses no labels or target vectors.
            single=np.column_stack([b[order[i]]/(dp[i]@b[order[i]]) for i in range(5)])
            poly,models=graph_model(b,x0,xtrain,xtest,commands,targetq)
            for degree in ['1','2']:
                poly[degree]['calibrated_graph_direction']=field_metrics(models[degree]['jphysical'],jtrue)
            row=dict(name=name,radius=radius,method=method,settings=p.tolist(),
                source_cap_kink=bool(abs(p[4])>1e-10 and any(abs(v*16-round(v*16))<1e-9 for v in q0[:2])),
                original_neighborhood_train=512,chart_train=512,chart_test=256,
                assigned_axis_cosines=cosine[np.arange(5),order].tolist(),
                best_assigned_single_axis=field_metrics(single,jtrue),
                calibrated_span=field_metrics(jcal,jtrue),condition=cond,
                five_axis_span_best_possible=field_metrics(b.T@(b@jtrue),jtrue),
                original_linear_reconstruction=image_metrics(x0+mean+((xtest-x0-mean)@b.T)@b,xtest,x0),
                polynomial=poly)
            rows.append(row)
            if (name,radius) in [('root',.001),('root',.00001),('lean_30',.001),('spread_03',.001),('new_01',.001)]:
                prefix=f'{name}_{radius}_{method}'
                arrays[prefix+'_basis']=b;arrays[prefix+'_physical_response']=dp@b.T
                arrays[prefix+'_unmixed_fields']=jcal;arrays[prefix+'_true_fields']=jtrue
                arrays[prefix+'_source_image']=x0;arrays[prefix+'_heldout_images']=xtest
                arrays[prefix+'_heldout_settings']=test_p
                for key,value in models['2'].items():arrays[prefix+'_'+key]=value
        if case_index%10==0:
            print('Neighborhood cases',case_index+1,'elapsed',round(time.monotonic()-start,1),'s',flush=True)
    return rows,arrays


def collision_control(arrays):
    p0=np.array([14.,14.,19.75,3.2,0.]);_,fit=local_samples(p0,.001,512,2026102201,False)
    p1=np.array([14.5,14.5,19.6,3.2,30.]);p2=p1.copy();p2[2]=19.8
    x1,x2=pixels([p1,p2]);result={}
    for method in ['svd','pca']:
        b,mean,_=frame(fit-pixels(p0)[0],method=='pca')
        difference=(x2-x1)@b.T
        assert np.max(np.abs(difference))<1e-14
        result[method]=dict(largest_five_coordinate_difference=float(np.max(np.abs(difference))),
            true_image_distance=float(np.linalg.norm(x2-x1)),height_difference=.2)
        arrays['collision_'+method+'_basis']=b
    arrays['collision_images']=np.stack([x1,x2]);arrays['collision_settings']=np.stack([p1,p2])
    return result


def same_cell_fields(arrays):
    reference=np.array([14.43,14.62,19.73,3.17,17.3]);q0=chart(reference)
    half=np.array([.000125,.000125,.000125,.000125,.0000125])
    u=qmc.Sobol(5,scramble=True,seed=2026102301).random_base2(6)
    qs=q0+(2*u-1)*half
    ps=np.array([unchart(q) for q in qs])
    signature=jet(reference)[4]
    assert all(np.array_equal(jet(p)[4],signature) for p in ps)
    records=[]
    gathered={m:[] for m in ['svd','pca']}
    truth=np.array([jet(p)[1].T for p in ps])
    for i,p in enumerate(ps):
        _,neighbors=local_samples(p,1e-5,256,2026102302+i,False)
        x0=pixels(p)[0];_,_,dq,_,_=readers(x0)
        for method in gathered:
            b,_,_=frame(neighbors-x0,method=='pca')
            response=dq@b.T
            gathered[method].append(np.linalg.solve(response.T,b).T.T)
    # Stored fields: states, coordinate families, pixels.
    observed_q=np.array([readers(pixels(p)[0])[1] for p in ps])
    design=np.column_stack([np.ones(len(qs)),(observed_q-q0)/half])
    for method,values in gathered.items():
        v=np.array(values)
        families=[]
        for family,label in enumerate(['top','bottom','intercept','width','slope']):
            train=v[:32,family];test=truth[32:,family]
            mean=train.mean(0);_,s,b=np.linalg.svd(train-mean,full_matrices=False)
            compact=mean+((test-mean)@b[:2].T)@b[:2]
            fitted=np.linalg.lstsq(design[:32],train,rcond=None)[0]
            predicted=design[32:]@fitted
            # Relative to full direction, not misleading tiny variation energy.
            error=np.linalg.norm(predicted-test,axis=1)/np.linalg.norm(test,axis=1)
            two_error=np.linalg.norm(compact-test,axis=1)/np.linalg.norm(test,axis=1)
            families.append(dict(family=label,affine_field_prediction_worst_relative=float(error.max()),
                two_pattern_projection_worst_relative=float(two_error.max()),singular_values=s[:8].tolist()))
        records.append(dict(method=method,training_states=32,heldout_states=32,
            family_results=families,reference=reference.tolist(),coordinate_half_spans=half.tolist(),
            explanation='Observable-assisted unmixing of independently sampled SVD/PCA frames, then explicit affine field fit.'))
        arrays['cell_'+method+'_estimated_fields']=v
    arrays['cell_settings']=ps;arrays['cell_q_coordinates']=qs;arrays['cell_true_fields']=truth
    return records




def polynomial_jacobian(z,coef):
    jac=np.broadcast_to(coef[1:6].T,(len(z),5,5)).copy()
    for index,(i,j) in enumerate(PAIRS):
        term=coef[6+index]
        jac[:,:,i]+=z[:,j,None]*term
        jac[:,:,j]+=z[:,i,None]*term
    return jac


def invert_five_scores(scores,basis,origin_image,qorigin,half,pieces):
    """Infer geometry/images from five measured scalars and saved coefficients.

    No query image, true query geometry, generator, or true derivative is used.
    """
    recovered=np.zeros((len(scores),5));predicted=np.zeros((len(scores),784))
    counts=np.zeros(len(scores),int)
    for key,beta in pieces.items():
        coef=beta@basis.T
        z=np.zeros((len(scores),5))
        for iteration in range(7):
            residual=features(z,2,True)@coef-scores
            step=np.linalg.solve(polynomial_jacobian(z,coef),residual[:,:,None])[:,:,0]
            z-=step
        candidate=qorigin+z*half
        candidate_keys=np.floor(candidate[:,:2]*SUB+1e-10).astype(int)
        valid=(np.max(np.abs(z),axis=1)<=1+1e-7)&np.all(candidate_keys==key,axis=1)
        recovered[valid]=candidate[valid]
        predicted[valid]=origin_image+features(z[valid],2,True)@beta
        counts[valid]+=1
    return recovered,predicted,counts


def algebraic_charts(arrays):
    """Identify exact piecewise quadratic charts from state observables.

    Held-out inference receives ONLY five SVD/PCA scores. Derived q labels
    belong to fitting images; test geometry is used strictly for evaluation.
    """
    answers=[]
    corners=np.array(list(itertools.product([-1.,1.],repeat=5)))
    for name,p in [('root',np.array([14.,14.,19.75,3.2,0.])),
                   ('smooth',np.array([14.43,14.62,19.73,3.17,17.3])),
                   ('tilted_cap_boundary',np.array([14.,14.,19.75,3.2,30.]))]:
        q0=chart(p);x0=pixels(p)[0]
        mid=(np.arange(28*SUB)+.5)/SUB
        def side_key(q):
            center=q[2]-q[4]*mid
            return np.r_[np.floor(center-q[3]/2),np.floor(center+q[3]/2)]
        half=np.array([1.,1.,1.,1.,.1])*1e-3
        # Keep all side formulas fixed. Caps may cross their known subrow
        # boundaries, so this tests multiple pieces for boundary references.
        while not all(np.array_equal(side_key(q0+c*half),side_key(q0)) for c in corners):half/=2
        train_q=q0+(2*qmc.Sobol(5,scramble=True,seed=2026102401).random_base2(10)-1)*half
        test_q=q0+(2*qmc.Sobol(5,scramble=True,seed=2026102402).random_base2(9)-1)*half
        train_p=fromq(train_q);test_p=fromq(test_q)
        train_images=pixels(train_p);test_images=pixels(test_p)
        _,observed_q,_,_=recover(train_images.reshape(-1,28,28))
        qorigin=readers(x0)[1]
        normalized_q=(observed_q-qorigin)/half
        keys=np.floor(observed_q[:,:2]*SUB+1e-10).astype(int)
        fits={}
        for key in np.unique(keys,axis=0):
            take=np.all(keys==key,axis=1)
            design=features(normalized_q[take],2,True)
            assert np.linalg.matrix_rank(design,tol=1e-10)==21
            beta=np.linalg.lstsq(design,train_images[take]-x0,rcond=None)[0]
            fits[tuple(key)]=dict(beta=beta,training=int(take.sum()),design_condition=float(np.linalg.cond(design)))
        _,neighbors=local_samples(p,.001,512,2026102403,False)
        for method in ['svd','pca']:
            b,_,_=frame(neighbors-x0,method=='pca')
            score=(test_images-x0)@b.T
            recovered=np.zeros((len(score),5));predicted=np.zeros_like(test_images)
            counts=np.zeros(len(score),int)
            projection_fits={key:fit['beta']@b.T for key,fit in fits.items()}
            recovered,predicted,counts=invert_five_scores(score,b,x0,qorigin,half,
                {key:fit['beta'] for key,fit in fits.items()})
            assert np.all(counts==1),(name,method,np.unique(counts,return_counts=True))
            recovered_p=fromq(recovered)
            # A numerical lower-bound certificate: after normalizing by a
            # reference derivative, every piece's Jacobian is within eps<1
            # of the identity throughout the convex box. The Jacobian is
            # affine, so its spectral norm bound is maximized at a corner.
            first=next(iter(projection_fits.values()))
            a0=first[1:6].T
            maximum=0.
            for coef in projection_fits.values():
                normalized=np.linalg.solve(a0,polynomial_jacobian(corners,coef))-np.eye(5)
                maximum=max(maximum,float(np.linalg.norm(normalized,ord=2,axis=(1,2)).max()))
            assert maximum<1
            # Closed-loop actions: infer source q from its five scores, then
            # change all original controls and predict another held-out image.
            source_index=np.arange(256);target_index=np.arange(256,512)
            command=test_p[target_index]-test_p[source_index]
            target_q=np.array([chart(v) for v in recovered_p[source_index]+command])
            target_keys=np.floor(target_q[:,:2]*SUB+1e-10).astype(int)
            output=np.zeros((256,784));matched=np.zeros(256,int)
            for key,fit in fits.items():
                take=np.all(target_keys==key,axis=1)
                z=(target_q[take]-qorigin)/half
                output[take]=x0+features(z,2,True)@fit['beta'];matched[take]+=1
            assert np.all(matched==1)
            assigned=np.abs((jet(p)[3]/np.linalg.norm(jet(p)[3],axis=0)).T@b.T)
            row,col=linear_sum_assignment(-assigned)
            info=dict(name=name,method=method,reference=p.tolist(),coordinate_half_spans=half.tolist(),
                train_images=1024,test_images=512,pieces=len(fits),
                piece_training_counts=[f['training'] for f in fits.values()],
                smallest_assigned_axis_cosine=float(assigned[row,col].min()),
                all_heldout_scores_have_unique_inverse=True,
                normalized_jacobian_perturbation_bound=maximum,
                numerical_injectivity_margin=1-maximum,
                reconstruction=image_metrics(predicted,test_images,x0),
                largest_recovered_physical_error=np.max(np.abs(recovered_p-test_p),axis=0).tolist(),
                closed_loop_joint_actions=image_metrics(output,test_images[target_index],test_images[source_index]))
            answers.append(info)
            prefix='algebraic_'+name+'_'+method
            arrays[prefix+'_basis']=b
            arrays[prefix+'_test_scores']=score
            arrays[prefix+'_recovered_q']=recovered
            arrays[prefix+'_predicted_images']=predicted
            arrays[prefix+'_true_images']=test_images
            arrays[prefix+'_true_settings']=test_p
            arrays[prefix+'_q_origin']=qorigin;arrays[prefix+'_q_scale']=half
            arrays[prefix+'_source_image']=x0
            arrays[prefix+'_piece_keys']=np.array(list(fits),int)
            for key,fit in fits.items():arrays[prefix+f'_piece_{int(key[0])}_{int(key[1])}']=fit['beta']
        print('Algebraic calibrated chart:',name,len(fits),'pieces',flush=True)
    return answers


def summarize(rows):
    result=[]
    for method in ['svd','pca']:
        for radius in sorted(set(r['radius'] for r in rows),reverse=True):
            rr=[r for r in rows if r['method']==method and r['radius']==radius]
            rec=dict(method=method,radius=radius,cases=len(rr))
            for key,fn in [('single_axis',lambda r:r['best_assigned_single_axis']['worst_relative']),
                ('unmixed_span',lambda r:r['calibrated_span']['worst_relative']),
                ('linear_graph',lambda r:r['polynomial']['1']['command_prediction']['worst_relative_change']),
                ('quadratic_graph',lambda r:r['polynomial']['2']['command_prediction']['worst_relative_change']),
                ('piecewise_graph',lambda r:r['polynomial']['piecewise_degree_2']['command_prediction']['worst_relative_change']),
                ('quadratic_reconstruction',lambda r:r['polynomial']['2']['decoder']['worst_relative_change']),
                ('quadratic_mapping',lambda r:r['polynomial']['2']['mapping_worst_relative_normalized_change'])]:
                values=np.array([fn(r) for r in rr]);rec[key]=dict(median=float(np.median(values)),worst=float(values.max()))
            result.append(rec)
    return result


def main():
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    rows,arrays=case_study()
    result=dict(method=dict(precision='float64 coverage; ordinary storage rounding is not a research target',
        raw_frames='SVD of image-minus-start; PCA subtracts neighborhood mean first; no whitening',
        latent_coordinates='Five dot products of image-minus-start with measured local basis',
        data_only_decoder='Ordinary least-squares explicit quadratic image formula in five measured coordinates; no renderer formula',
        geometry_assistance='State-derived top, bottom, intercept, width, slope and their explicit conversion to original controls',
        derivative_targets='Known derivatives enter evaluation and sampler initialization only, never calibration/fitting targets',
        sampling='512 new independently sampled filled-neighborhood images for chart fits; 256 new held-out joint moves',
        sampling_bounds='Two-sided local rays may extend position bounds below 14 as in prior root fits; full 32 new starts use interior default box',
        repeats='Original 61 saved neighborhoods plus 32 new five-dimensional source states'),
        cases=rows,summary=summarize(rows))
    result['fixed_frame_collision']=collision_control(arrays)
    result['same_cell_field_compression']=same_cell_fields(arrays)
    result['algebraic_calibrated_charts']=algebraic_charts(arrays)
    result['elapsed_seconds']=time.monotonic()-start
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    np.savez_compressed(OUT/'models_and_examples.npz',**arrays)
    print('COMPLETE',result['elapsed_seconds'],'seconds',flush=True)
    print(json.dumps(result['summary'],indent=2),flush=True)


if __name__=='__main__':main()
