"""Lean curves in 784D: python make_generated_lean_width_comparison.py --width-increase 1."""
import argparse, csv, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.spatial.distance import cdist
from grey_ones import RANGES, render

BASE = RANGES.mean(axis=1)
BASE[4] = 0
LO, HI = RANGES[4]
OUT = Path(__file__).resolve().parent / 'figures/generated_lean_width_comparison'

def images(w, t):
    t = np.asarray(t).ravel()
    p = np.tile(BASE, (len(t), 1))
    p[:, 3], p[:, 4] = w, t
    return render(p).reshape(-1, 784).astype(float)

def norm(x):
    return np.linalg.norm(x, axis=-1)

def angle(a, b):
    return np.degrees(np.arccos(np.clip(np.sum(a*b, axis=-1)/(norm(a)*norm(b)), -1, 1)))

def stats(x):
    return dict(min=float(np.min(x)), median=float(np.median(x)), max=float(np.max(x)))

def sample(w, step):
    knots = [float(LO)]
    for end in [0., float(HI)]:
        while knots[-1] < end-1e-8:
            start = knots[-1]
            x0 = images(w, start)[0]
            ts = np.linspace(start, end, max(2, int(np.ceil((end-start)/.025))+1))
            hits = np.flatnonzero(norm(images(w, ts)-x0) >= step)
            if not len(hits):
                knots.append(end)
                break
            k = hits[0]
            root = brentq(lambda t: float(norm(images(w,t)[0]-x0))-step, ts[k-1], ts[k], xtol=1e-9)
            knots.append(end if end-root < 1e-7 else float(root))
    x = images(w, knots)
    gaps = norm(np.diff(x, axis=0))
    forced = np.isclose(knots[1:], 0) | np.isclose(knots[1:], HI)
    assert np.all(abs(gaps[~forced]-step) < 2e-6)
    assert np.all(gaps[forced] <= step+2e-6)
    assert np.all(np.diff(knots) > 0)
    assert all(np.any(np.isclose(knots,t)) for t in [LO,0,HI])
    return np.array(knots), x, gaps

def closest_segments(a,b,c,d):
    u,v,z = b-a,d-c,a-c
    uu,vv,uv = u@u,v@v,u@v
    candidates = [(s,float(np.clip(v@(z+s*u)/vv,0,1))) for s in [0.,1.]]
    candidates += [(float(np.clip(u@(t*v-z)/uu,0,1)),t) for t in [0.,1.]]
    h = np.array([[uu,-uv],[-uv,vv]])
    if np.linalg.det(h) > 1e-12*uu*vv:
        st = np.linalg.solve(h,[-u@z,v@z])
        if np.all((st >= 0) & (st <= 1)):
            candidates.append(tuple(st))
    return min((float(norm(z+s*u-t*v)),s,t) for s,t in candidates)

def closest_paths(a,b):
    return min(((*closest_segments(a[i],a[i+1],b[j],b[j+1]),i,j))
               for i in range(len(a)-1) for j in range(len(b)-1))

def to_polyline(points,path):
    result = np.full(len(points), np.inf)
    for a,b in zip(path[:-1],path[1:]):
        u = b-a
        t = np.clip((points-a)@u/(u@u),0,1)
        result = np.minimum(result,norm(points-a-t[:,None]*u))
    return result

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width',type=float,default=None,help='Absolute width knob; otherwise increase normal width.')
    parser.add_argument('--width-increase',type=float,default=1.)
    parser.add_argument('--data-only',action='store_true',help='Save measurements without duplicate figures.')
    parser.add_argument('--step',type=float,default=3.)
    args = parser.parse_args()
    if args.width is None:
        args.width = float(BASE[3]+args.width_increase)
    widths = [float(BASE[3]),args.width]
    if args.width <= 0 or args.step <= 0 or np.isclose(*widths):
        parser.error('Use distinct positive widths and positive spacing.')
    if max(widths)/2+BASE[2]/2*np.tan(np.radians(max(abs(LO),abs(HI)))) >= min(BASE[0],28-BASE[0]):
        parser.error('This experiment requires no canvas clipping.')
    OUT.mkdir(parents=True,exist_ok=True)
    paths = [sample(w,args.step) for w in widths]
    ts = np.linspace(LO,HI,901)
    dense = [images(w,ts) for w in widths]
    separation = norm(dense[1]-dense[0])
    translated = norm((dense[1]-images(widths[1],0))-(dense[0]-images(widths[0],0)))
    matched = np.linspace(LO+.5,HI-.5,881)
    angles = {}
    for h in [.02,.1,.5,1.]:
        deltas = [images(w,matched+h/2)-images(w,matched-h/2) for w in widths]
        angles[str(h)] = angle(*deltas)
    coarse_directions = []
    for knots,x,gaps in paths:
        indices = np.clip(np.searchsorted(knots,ts,side='right')-1,0,len(knots)-2)
        coarse_directions.append(np.diff(x,axis=0)[indices])
    coarse_angles = angle(*coarse_directions)
    closest = closest_paths(paths[0][1],paths[1][1])
    pair = cdist(*dense)
    ij = np.unravel_index(np.argmin(pair),pair.shape)
    nearest = [pair.min(axis=1),pair.min(axis=0)]
    path_results = []
    for w,(knots,x,gaps),xx in zip(widths,paths,dense):
        length = {}
        for count in [451,901,1801]:
            length[str(45/(count-1))] = float(norm(np.diff(images(w,np.linspace(LO,HI,count)),axis=0)).sum())
        path_results.append(dict(width=w,point_count=len(knots),lean_degrees=knots.tolist(),gaps=gaps.tolist(),
            polyline_length=float(gaps.sum()),dense_arc_length_by_degree_spacing=length,
            endpoint_chord=float(norm(x[-1]-x[0])),
            consecutive_segment_turn_degrees=angle(np.diff(x,axis=0)[:-1],np.diff(x,axis=0)[1:]).tolist(),
            curve_to_polyline_error=stats(to_polyline(xx,x))))
    shape = [images(w,np.linspace(LO,HI,91)) for w in widths]
    dm = [cdist(x,x) for x in shape]
    scale = float(np.sum(dm[0]*dm[1])/np.sum(dm[0]**2))
    shape_error = float(norm((dm[1]-scale*dm[0]).ravel())/norm(dm[1].ravel()))
    masses = [x.sum(axis=1) for x in dense]
    assert all(np.max(abs(m-w*BASE[2])) < 2e-5 for m,w in zip(masses,widths))
    result = dict(baseline_knobs=BASE.tolist(),comparison_width=args.width,
        comparison_outside_default_width_range=not RANGES[3,0]<=args.width<=RANGES[3,1],
        distance_definition='Euclidean distance across all 784 ink coverage values',sampling_step=args.step,
        sampling_rule='Forward from minimum; force upright and restart; force maximum. First chord crossing.',
        paths=path_results,same_lean_separation=stats(separation),
        matched_lean_direction_angle_degrees={h:stats(a) for h,a in angles.items()},
        matched_lean_polyline_segment_angle_degrees=stats(coarse_angles),
        matched_lean_direction_angle_at_zero={h:float(a[np.argmin(abs(matched))]) for h,a in angles.items()},
        endpoint_direction_angle_degrees=float(angle(dense[0][-1]-dense[0][0],dense[1][-1]-dense[1][0])),
        upright_aligned_displacement_mismatch=stats(translated),
        polyline_closest=dict(distance=closest[0],fraction_normal=closest[1],fraction_comparison=closest[2],segment_normal=closest[3],segment_comparison=closest[4]),
        dense_rendered_curves_closest=dict(distance=float(pair[ij]),lean_normal=float(ts[ij[0]]),lean_comparison=float(ts[ij[1]]),grid_degrees=.05),
        dense_rendered_curve_nearest_distances=[stats(a) for a in nearest],
        dense_rendered_curve_hausdorff_estimate=float(max(a.max() for a in nearest)),
        total_ink_by_width=[stats(m) for m in masses],
        analytic_nonintersection_lower_bound=float(abs(widths[1]-widths[0])*BASE[2]/28),
        best_uniform_distance_matrix_scale=scale,distance_matrix_residual_fraction_after_scale=shape_error)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'sampled_points.csv').open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['path','cx','cy','height','width','lean_deg','distance_from_previous','forced_anchor'])
        for name,w,(knots,x,gaps) in zip(['normal','comparison'],widths,paths):
            for i,t in enumerate(knots):
                writer.writerow([name,*BASE[:3],w,t,'' if i==0 else gaps[i-1],any(np.isclose(t,z) for z in [LO,0,HI])])
    with (OUT/'matched_lean_metrics.csv').open('w',newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['lean_deg','separation','upright_aligned_mismatch','angle_01degree','polyline_segment_angle'])
        for t,sep,mis in zip(ts,separation,translated):
            a = float(np.interp(t,matched,angles['0.1'])) if matched[0]<=t<=matched[-1] else ''
            writer.writerow([t,sep,mis,a,float(np.interp(t,ts,coarse_angles))])
    np.savez_compressed(OUT/'sampled_images.npz',normal=paths[0][1].reshape(-1,28,28),comparison=paths[1][1].reshape(-1,28,28),normal_lean=paths[0][0],comparison_lean=paths[1][0])
    if args.data_only:
        print(json.dumps(result,indent=2))
        return
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':150})
    colors = ['#2368ad','#df7725']
    names = [f'Normal: width {widths[0]:g} px',f'Comparison: width {widths[1]:g} px']
    n = max(len(p[0]) for p in paths)
    fig = plt.figure(figsize=(max(12,n*1.35),6.4))
    fig.text(.035,.95,'Which images form each ordered lean path?',fontsize=18,weight='bold')
    fig.text(.035,.87,'Generated 28 x 28 images: white = no ink (0), black = full ink (1). Upright references have thick outlines.\nFixed center (14.5, 14.5) px, height 19.75 px. Lean increases left to right; distances use all 784 pixels.',fontsize=11)
    gs = fig.add_gridspec(2,n,left=.035,right=.985,bottom=.23,top=.71,hspace=.9,wspace=.16)
    for r,(name,color,(knots,x,gaps)) in enumerate(zip(names,colors,paths)):
        fig.text(.035,.765 if r==0 else .465,name,color=color,fontsize=12,weight='bold')
        for k in range(n):
            ax = fig.add_subplot(gs[r,k]); ax.set_xticks([]); ax.set_yticks([])
            if k>=len(knots): ax.axis('off'); continue
            ax.imshow(x[k].reshape(28,28),cmap='gray_r',vmin=0,vmax=1)
            for spine in ax.spines.values():
                spine.set_visible(True); spine.set_color(color); spine.set_linewidth(3 if np.isclose(knots[k],0) else .6)
            ax.set_title(f'{knots[k]:.2f} degrees',fontsize=9,color=color)
            ax.text(.5,-.1,'start' if k==0 else f'gap {gaps[k-1]:.2f}',ha='center',va='top',transform=ax.transAxes)
    fig.text(.035,.08,f'Gap = image distance from previous image; target {args.step:g}. Minimum lean, upright, and maximum lean are always included.\nShort anchor gaps are intentional. Default width range: {RANGES[3,0]:g}-{RANGES[3,1]:g} px. Straight connections can leave the generated family.',fontsize=11)
    fig.savefig(OUT/'sampled_paths.png'); plt.close(fig)
    fig,axs = plt.subplots(1,3,figsize=(15,5.8))
    fig.subplots_adjust(left=.065,right=.985,bottom=.29,top=.71,wspace=.36)
    fig.text(.065,.94,'Does changing width alter the lean path?',fontsize=18,weight='bold')
    fig.text(.065,.85,f'Normal width {widths[0]:g} px versus {widths[1]:g} px; only lean varies. Every measurement uses the original 784-dimensional images.\nRead left to right: separation, direction, then whether one fixed translation aligns the curves.',fontsize=11)
    axs[0].plot(ts,separation,color='#604493')
    axs[0].set(title=f'Same-lean gap: {separation.min():.2f} to {separation.max():.2f}',xlabel='Lean (degrees)',ylabel='Image distance (L2)')
    axs[1].plot(matched,angles['0.1'],color='#604493',label='0.1 degree centered chord')
    axs[1].plot(ts,coarse_angles,color='#999999',ls='--',label='3-unit polyline segments')
    axs[1].axhline(0,color='black',ls=':',label='0 = same direction')
    axs[1].set(title=f'Direction angle: median {np.median(angles["0.1"]):.1f} degrees',xlabel='Lean (degrees)',ylabel='Angle between pixel-change vectors (degrees)',ylim=(-3,max(100,float(coarse_angles.max())+3)))
    axs[1].legend(loc='lower right',fontsize=8)
    axs[2].plot(ts,translated,color='#604493'); axs[2].axhline(0,color='black',ls=':')
    axs[2].set(title=f'Upright-aligned mismatch: max {translated.max():.2f}',xlabel='Lean (degrees)',ylabel='Displacement mismatch (image L2)')
    for ax in axs: ax.axvline(0,color='#bbbbbb',ls=':'); ax.grid(alpha=.15)
    fig.text(.065,.115,'Angles compare forward directions at matched lean. Solid: small 0.1-degree chords; dashed: requested 3-unit segments. Normalization removes speed.\nAlignment subtracts each curve\'s upright image: zero error everywhere would mean exact translated copies.\n'+f'Closest requested straight-segment paths: {closest[0]:.3f} image units. No crossing: the paths have different constant total ink.',fontsize=10)
    fig.savefig(OUT/'comparison_metrics.png'); plt.close(fig)
    example = np.array([0.,10.,25.])
    source = [images(w,example) for w in widths]
    delta = [images(w,example+.05)-images(w,example-.05) for w in widths]
    unit = [d/norm(d)[:,None] for d in delta]
    difference = unit[1]-unit[0]
    common,second = float(np.max(abs(np.concatenate(unit)))),float(np.max(abs(difference)))
    fig = plt.figure(figsize=(11,11))
    fig.text(.22,.96,'Which pixels change when lean increases?',fontsize=17,weight='bold')
    fig.text(.22,.905,'Top: source images (black = ink). Below: unit-length change vectors for a centered 0.1 degree lean move.\nAll 784 coordinates shown. In direction rows blue = ink loss, red = ink gain. Only width differs.',fontsize=10)
    gs = fig.add_gridspec(5,3,left=.25,right=.82,bottom=.13,top=.84,hspace=.5,wspace=.38)
    rows = [*source,*unit,difference]
    labels = [f'Normal source\nwidth {widths[0]:g} px',f'Comparison source\nwidth {widths[1]:g} px','Normal direction\nL2 norm = 1','Comparison direction\nL2 norm = 1','Direction difference\ncomparison minus normal']
    for r,row in enumerate(rows):
        for c,x in enumerate(row):
            ax = fig.add_subplot(gs[r,c]); ax.set_xticks([]); ax.set_yticks([])
            if r<2: ax.imshow(x.reshape(28,28),cmap='gray_r',vmin=0,vmax=1)
            else:
                limit = common if r<4 else second
                im = ax.imshow(x.reshape(28,28),cmap='RdBu_r',vmin=-limit,vmax=limit)
            if r==0: ax.set_title(f'Lean {example[c]:g} degrees')
            if c==0: ax.text(-.3,.5,labels[r],ha='right',va='center',transform=ax.transAxes,fontsize=10)
            if r==4: ax.text(.5,-.1,f'angle {angle(unit[0][c],unit[1][c]):.1f} degrees',ha='center',va='top',transform=ax.transAxes,fontsize=9)
        if r in [3,4]:
            cbax = fig.add_axes([.87,.335 if r==3 else .16,.018,.15])
            fig.colorbar(im,cax=cbax); cbax.set_title('value',fontsize=9)
    fig.text(.22,.045,'Direction rows share a symmetric color scale. The bottom row has its own labeled scale: a change in the\npixel-change pattern, rather than direct ink gain/loss. No values are clipped. This is a fixed-width slice.',fontsize=10)
    fig.savefig(OUT/'pixel_directions.png'); plt.close(fig)
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
