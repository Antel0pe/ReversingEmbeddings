"""Exact patch derivatives, nonlinear residuals, and knob-family ambiguity.

This diagnostic deliberately uses the known renderer and five coordinates.
It tests geometric claims, not discovery from an unlabeled point cloud.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/generated-one-mpl")
from pathlib import Path
import json
import numpy as np
from scipy.stats import qmc
from grey_ones import render, SUB, KNOBS

HERE=Path(__file__).resolve().parent
OUT=HERE/"figures/generated_one_manifold_transport"
ROOT=np.array([14.,14.,19.75,3.2,0.])


def image(p):
    return render(p,dtype=np.float64).reshape(-1,784)


def chart(p):
    x,y,h,w,angle=np.asarray(p)
    slope=np.tan(np.deg2rad(angle))
    return np.array([y-h/2,y+h/2,x+slope*y,w,slope])


def unchart(q):
    top,bottom,a,w,s=q;cy=(top+bottom)/2
    return np.array([a-s*cy,cy,bottom-top,w,np.rad2deg(np.arctan(s))])


def jet(p):
    """Analytic image/Jacobian/Hessian within a renderer formula cell.

    q = (top, bottom, intercept, width, slope). At exact cap boundaries,
    use the average of the two one-sided cap derivatives. At the upright
    root the first derivative agrees across sides, but mixed second
    derivatives need not. No claim of a unique Hessian at those kinks.
    """
    top,bottom,a,w,s=chart(p)
    lo=np.arange(28*SUB)/SUB;hi=lo+1/SUB;mid=(lo+hi)/2
    cols=np.arange(28)
    vertical=np.clip(np.minimum(hi,bottom)-np.maximum(lo,top),0,1/SUB)
    def interval_indicator(v):
        return ((lo<v)&(v<hi)).astype(float)+.5*((lo==v)|(hi==v))
    dt=-interval_indicator(top);db=interval_indicator(bottom)
    left=a-s*mid-w/2;right=a-s*mid+w/2
    overlap=np.clip(np.minimum(right[:,None],cols+1)-np.maximum(left[:,None],cols),0,1)
    il=((cols<left[:,None])&(left[:,None]<cols+1)).astype(float)
    ir=((cols<right[:,None])&(right[:,None]<cols+1)).astype(float)
    da=ir-il;dw=(ir+il)/2;ds=-mid[:,None]*da
    def integrate(v,h):
        return (v[:,None]*h).reshape(28,SUB,28).sum(1).ravel()
    rendered=integrate(vertical,overlap)
    jq=np.column_stack([integrate(dt,overlap),integrate(db,overlap),
                        integrate(vertical,da),integrate(vertical,dw),integrate(vertical,ds)])
    hq=np.zeros((784,5,5))
    for i,v in [(0,dt),(1,db)]:
        for j,h in [(2,da),(3,dw),(4,ds)]:
            hq[:,i,j]=hq[:,j,i]=integrate(v,h)
    x,y,h,w,angle=p
    prime=np.pi/180*(1+s*s)
    dq=np.array([[0,1,-.5,0,0],[0,1,.5,0,0],[1,s,0,0,y*prime],
                 [0,0,0,1,0],[0,0,0,0,prime]])
    # Cell signature: all active affine formulas must remain unchanged.
    signature=np.concatenate([dt,db,il.ravel(),ir.ravel()])
    return rendered,jq,hq,jq@dq,signature


def span(matrix,rtol=1e-10):
    u,s,_=np.linalg.svd(matrix,full_matrices=False)
    return u[:,s>s[0]*rtol],s


def normalized(j):
    return j/np.linalg.norm(j,axis=0)


def geometric_evidence(archive):
    result=[]
    for name in ["root","lean_minus7","lean_12p5","lean_30"]:
        p=archive[name+"_start_settings"]
        f,jq,h,j,sig=jet(p);unit=normalized(j)
        assert np.max(np.abs(f-image(p)[0]))<3e-14
        gram=unit.T@unit
        result.append(dict(name=name,settings=p.tolist(),cosine_gram=gram.tolist(),
                           cx_cy_cosine=float(gram[0,1]),height_width_cosine=float(gram[2,3]),
                           physical_jacobian_rank=int(np.linalg.matrix_rank(j))))
    return result


def residual_evidence(archive):
    f,jq,h,j,_=jet(ROOT);tangent,_=span(j)
    mixed=h.reshape(784,25)
    normal=mixed-tangent@(tangent.T@mixed)
    nb,ns=span(normal)
    # Four concrete interactions explain the normal space more directly than
    # arbitrary singular vectors of the mixed-derivative matrix.
    specific=np.column_stack([-.5*h[:,0,3]+.5*h[:,1,3],h[:,0,3]+h[:,1,3],
                              -.5*h[:,0,2]+.5*h[:,1,2],h[:,0,2]+h[:,1,2]])
    specific-=tangent@(tangent.T@specific)
    assert span(specific)[0].shape[1]==4
    complete,cs=span(np.column_stack([j,mixed]))
    full=archive['root_r_0p001_basis']
    test=archive['root_r_0p001_images'][512:]-f
    reconstructed=(test@complete)@complete.T
    # SVD's trailing vectors are only approximately normal to the true tangent.
    extra_normal=full[5:]-(full[5:]@tangent)@tangent.T
    extra_normal/=np.linalg.norm(extra_normal,axis=1)[:,None]
    cosines=np.linalg.svd(extra_normal@nb,compute_uv=False)
    result=dict(tangent_rank=tangent.shape[1],mixed_normal_rank=nb.shape[1],
                tangent_plus_mixed_rank=complete.shape[1],
                heldout_max_absolute_L2_error=float(np.linalg.norm(test-reconstructed,axis=1).max()),
                trailing_vectors_distance_to_mixed_normal_span=np.linalg.norm(extra_normal-(extra_normal@nb)@nb.T,axis=1).tolist(),
                trailing_vectors_tangent_fraction=np.linalg.norm(full[5:]@tangent,axis=1).tolist(),
                mixed_normal_singular_values=ns.tolist())
    assert nb.shape[1]==4 and complete.shape[1]==9
    assert result['heldout_max_absolute_L2_error']<1e-12
    return result,dict(root_jacobian=j,root_mixed_hessian=h,root_tangent=tangent,
                       root_normal_mixed_basis=nb,root_specific_normal_interactions=specific,
                       root_complete_basis=complete)


def exact_cell_evidence():
    # Use a smooth generic point, avoiding the canonical root's cap kink.
    p=np.array([14.43,14.62,19.73,3.17,17.3])
    f,j,h,jp,sig=jet(p);q0=chart(p)
    unit=qmc.Sobol(5,scramble=True,seed=20261014).random_base2(10)*2-1
    scale=np.array([1.,1.,1.,1.,.1])*1e-3
    # Start with a larger box, then shrink until every corner is in this
    # affine formula cell. The cell inequalities are linear in q, so corner
    # checks certify the entire box, not just the sampled joint points.
    corners=np.array(np.meshgrid(*[[-1.,1.]]*5)).reshape(5,-1).T
    while not all(np.array_equal(jet(unchart(q0+d))[4],sig) for d in corners*scale):
        scale/=2
    points=np.array([unchart(q0+d) for d in unit*scale])
    signatures=np.array([jet(x)[4] for x in points])
    assert np.all(signatures==sig)
    d=np.array([chart(x)-q0 for x in points])
    predicted=f+d@j.T+.5*np.einsum('ni,pij,nj->np',d,h,d)
    linear=f+d@j.T
    truth=image(points)
    error=np.linalg.norm(predicted-truth,axis=1)
    # A finite set of quadratic coefficients exactly specifies the whole cell.
    return dict(settings=p.tolist(),coordinate_names=['top','bottom','intercept','width','slope'],
                heldout_joint_points=len(points),all_formula_signatures_identical=True,
                all_32_box_corners_share_formula_signature=True,
                maximum_quadratic_image_L2_error=float(error.max()),
                maximum_quadratic_pixel_error=float(np.abs(predicted-truth).max()),
                maximum_linear_image_L2_error=float(np.linalg.norm(linear-truth,axis=1).max()),
                cell_half_spans=scale.tolist()),dict(cell_base=p,cell_jacobian=j,cell_hessian=h,
                                                   cell_points=points,cell_predictions=predicted)


def affine_family_evidence():
    # Exact bilinear upright patch: same height move, varying widths 2<w<4.
    widths=np.linspace(2.1,3.9,37);heights=[19.0,19.3,19.75,19.9]
    jhs=[]
    for height in heights:
        row=[]
        for width in widths:
            p=ROOT.copy();p[2]=height;p[3]=width
            row.append(jet(p)[3][:,2])
        jhs.append(row)
    jhs=np.asarray(jhs)
    slopes=np.diff(jhs,axis=1)/np.diff(widths)[None,:,None]
    reference=slopes[0,0]
    # Same slope across heights and widths: J_h(w) = C + w*D in this patch.
    deviation=np.linalg.norm(slopes-reference,axis=2).max()
    dh=.03;dw=.02
    height_move=np.array([0,0,dh,0,0]);width_move=np.array([0,0,0,dw,0])
    interaction=image(ROOT+height_move+width_move)[0]-image(ROOT+height_move)[0]-image(ROOT+width_move)[0]+image(ROOT)[0]
    predicted=reference*dh*dw
    outside=ROOT.copy();outside[2]=20.1
    outside_next=outside.copy();outside_next[3]+=.05
    outside_slope=(jet(outside_next)[3][:,2]-jet(outside)[3][:,2])/.05
    return dict(width_range=[2.1,3.9],width_spacing=.05,heights=heights,
                max_height_field_width_slope_difference=float(deviation),
                slope_norm=float(np.linalg.norm(reference)),
                max_height_field_change_for_equal_width_step=float(np.linalg.norm(np.diff(jhs,axis=1),axis=2).max()),
                finite_interaction_steps=dict(height=dh,width=dw),
                finite_interaction_L2=float(np.linalg.norm(interaction)),
                finite_interaction_prediction_error_L2=float(np.linalg.norm(interaction-predicted)),
                slope_change_outside_patch_at_height_20p1=float(np.linalg.norm(outside_slope-reference)),
                statement='J_height(w+dw)-J_height(w) = dw*D, D constant while 2<w<4 and 18<h<20; fixed center, zero lean')


def coordinate_ambiguity():
    # z_x=x, z_w=w+beta*(x-14)^2, others unchanged. Identity Jacobian at root.
    beta=2.
    def forward(p):
        z=np.asarray(p).copy();z[...,3]+=beta*(z[...,0]-14)**2
        return z
    def inverse(z):
        p=np.asarray(z).copy();p[...,3]-=beta*(p[...,0]-14)**2
        return p
    p=ROOT.copy();p[0]=14.3
    j=jet(p)[3]
    mixed=j[:,0]-2*beta*(p[0]-14)*j[:,3]
    pure=j[:,0]
    cos=float(mixed@pure/np.linalg.norm(mixed)/np.linalg.norm(pure))
    u=qmc.Sobol(5,scramble=True,seed=20261015).random_base2(8)
    points=np.array([13.8,13.8,19.3,2.5,-5])+u*np.array([.7,.7,.8,1.,15])
    decoded=inverse(forward(points))
    original=image(points);new=image(decoded)
    # Both orders are coordinate translations in z and hence commute exactly.
    z=forward(p);dx=.02;dw=.03
    def step(z,i,amount):
        z=z.copy();z[i]+=amount;return z
    endpoint_a=inverse(step(step(z,0,dx),3,dw))
    endpoint_b=inverse(step(step(z,3,dw),0,dx))
    return dict(beta=beta,coordinates='z_cx=cx; z_width=width+2*(cx-14)^2; other coordinates unchanged',
                inverse='cx=z_cx; width=z_width-2*(z_cx-14)^2',
                root_coordinate_jacobian_is_identity=True,
                example_settings=p.tolist(),alternative_cx_field_cosine_to_true_cx=cos,
                image_roundtrip_max_L2=float(np.linalg.norm(new-original,axis=1).max()),
                coordinate_roundtrip_max_error=float(np.abs(decoded-points).max()),
                two_orders_endpoint_max_error=float(np.abs(endpoint_a-endpoint_b).max()),
                bracket_statement='All five alternative coordinate fields commute, but the cx-like field mixes cx and width away from the root')


def tangent_frame(p):
    return np.linalg.qr(jet(p)[3])[0]


def coordinate_transport_evidence(archive):
    initial=jet(ROOT)[3];inverse=np.linalg.pinv(initial)
    rows=[];jacobians=[];skipped=[]
    for name in [key.removesuffix('_start_settings') for key in archive.files if key.endswith('_start_settings')]:
        p=archive[name+'_start_settings'];q=chart(p)
        on_cap_boundary=any(abs(value*SUB-round(value*SUB))<1e-10 for value in q[:2])
        if on_cap_boundary and abs(q[4])>1e-12:
            skipped.append(name);continue
        j=jet(p)[3]
        # Apply T=J(target) pinv(J(root)) in factors rather than materializing
        # a dense 784-by-784 ambient matrix.
        carried=j@(inverse@initial)
        error=np.linalg.norm(carried-j,axis=0)/np.linalg.norm(j,axis=0)
        rows.append(dict(name=name,max_relative_family_transport_error=float(error.max())))
        jacobians.append(j)
    mid,end=jacobians[-2:]
    direct=end@(inverse@initial)
    via_mid=end@(np.linalg.pinv(mid)@(mid@(inverse@initial)))
    return dict(method='Known-coordinate transport J(target) pinv(J(source)); no image observables or SVD-axis assignment',
                starts_checked=len(rows),results=rows,excluded_cap_kinks=skipped,
                max_relative_family_transport_error=max(row['max_relative_family_transport_error'] for row in rows),
                two_routes_max_absolute_difference=float(np.abs(direct-via_mid).max()))


def transport_path(points,initial):
    previous=tangent_frame(points[0]);coeff=previous.T@initial
    for p in points[1:]:
        current=tangent_frame(p)
        u,s,vt=np.linalg.svd(current.T@previous)
        coeff=(u@vt)@coeff
        previous=current
    return previous@coeff


def transport_evidence():
    p=np.array([14.43,14.62,19.73,3.17,17.3])
    # Find a short fixed-formula width interval; the limit approximates
    # metric parallel transport via successive closest orthogonal frame maps.
    sig=jet(p)[4];delta=.005
    while not all(np.array_equal(jet(p+np.array([0,0,0,w,0]))[4],sig) for w in np.linspace(0,delta,21)):
        delta/=2
    endpoint=p+np.array([0,0,0,delta,0])
    initial=normalized(jet(p)[3]);true_end=normalized(jet(endpoint)[3])
    rows=[];last=None
    for count in [16,64,256]:
        path=p+np.linspace(0,1,count+1)[:,None]*(endpoint-p)
        moved=transport_path(path,initial)
        gramerr=np.abs(moved.T@moved-initial.T@initial).max()
        cos=np.sum(moved*true_end,axis=0)
        rows.append(dict(steps=count,cosine_to_true_knob=cos.tolist(),
                         max_gram_preservation_error=float(gramerr),
                         difference_from_previous_resolution=None if last is None else float(np.linalg.norm(moved-last))))
        last=moved
    return dict(start=p.tolist(),end=endpoint.tolist(),width_step=delta,all_in_one_formula_cell=True,
                method='Successive orthogonal Procrustes maps between exact tangent frames; converges to induced-metric parallel transport on smooth paths',
                true_knob_unit_gram_change=float(np.abs(initial.T@initial-true_end.T@true_end).max()),
                resolutions=rows)


def figures(result,arrays,archive):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=image(ROOT)[0].reshape(28,28)
    # One explanatory figure: source first, then five true effects and four
    # analytically identified normal interaction directions.
    first=normalized(arrays['root_jacobian']).T
    normal=normalized(arrays['root_specific_normal_interactions']).T
    values=np.vstack([first,normal]);limit=np.abs(values).max()
    fig=plt.figure(figsize=(15,8.2),facecolor='white')
    fig.text(.035,.96,'What are the four extra local linear directions?',fontsize=21,weight='bold',va='top')
    fig.text(.035,.90,'Generated 1s; root (cx, cy, height, width, lean) = (14, 14, 19.75, 3.2, 0°). White = no ink; black = full ink.\n'
             'Top: five true first derivatives. Bottom: four independent normal components of interactions between pairs of controls.\n'
             'Signed maps have unit length and share one scale: red = positive coefficient; blue = negative. Bottom colors describe second-order pixel changes.',
             fontsize=10.5,va='top',linespacing=1.5)
    source_ax=fig.add_axes([.035,.40,.13,.27]);source_ax.imshow(root,cmap='gray_r',vmin=0,vmax=1)
    source_ax.set_xticks([]);source_ax.set_yticks([]);source_ax.set_title('Root image',fontsize=11)
    grid=fig.add_gridspec(2,5,left=.22,right=.975,top=.75,bottom=.25,hspace=.47,wspace=.25)
    for i in range(10):
        ax=fig.add_subplot(grid[i//5,i%5]);ax.set_xticks([]);ax.set_yticks([])
        if i<5:
            value=first[i];title='True '+KNOBS[i]+' direction'
        elif i<9:
            value=normal[i-5];title=['Normal: height × width','Normal: cy × width',
                                   'Normal: height × cx','Normal: cy × cx'][i-5]
        else:
            ax.axis('off');ax.text(.0,.65,'These four span the\nnormal part of SVD 6–9.\nIndividual bases can rotate.',fontsize=10,va='top');continue
        shown=ax.imshow(value.reshape(28,28),cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
        ax.set_title(title,fontsize=10,pad=9)
    error=result['residuals']['heldout_max_absolute_L2_error']
    fig.text(.035,.17,f'5 tangent directions + 4 mixed normal directions reconstruct all 256 saved held-out root neighbors at radius 0.001: worst L2 error {error:.2g}.\n'
             'Normal means perpendicular to the true tangent space. Mixed derivatives describe how a direction changes when another coordinate changes.\n'
             'This is a known-renderer diagnostic, not a new knob-discovery method; boundary-dependent coefficients remain necessary.',fontsize=10.5,va='top',linespacing=1.5)
    cax=fig.add_axes([.63,.045,.32,.018]);fig.colorbar(shown,cax=cax,orientation='horizontal').set_label('Unit signed derivative-map coefficient',fontsize=9)
    fig.savefig(OUT/'extra_vectors_explained.png',dpi=150);plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True,parents=True)
    archive=np.load(HERE/'figures/generated_one_local_bases_across_starts/bases_and_samples.npz')
    residual,arrays=residual_evidence(archive)
    cell,extra=exact_cell_evidence();arrays.update(extra)
    result=dict(scope='Known-five-knob analytic geometry; no claim of unsupervised identification',
                overlap=geometric_evidence(archive),residuals=residual,exact_cell=cell,
                affine_height_width=affine_family_evidence(),ambiguity=coordinate_ambiguity(),
                metric_transport=transport_evidence(),coordinate_transport=coordinate_transport_evidence(archive))
    assert result['exact_cell']['maximum_quadratic_image_L2_error']<1e-12
    assert result['affine_height_width']['max_height_field_width_slope_difference']<1e-12
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    np.savez_compressed(OUT/'analytic_geometry.npz',**arrays)
    figures(result,arrays,archive)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
