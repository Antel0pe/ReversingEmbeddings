"""Summarize nearest-direction experiments and draw explanatory figures."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/generated-one-mpl')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import json
import csv
import gzip
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from grey_ones import render,KNOBS
from analyze_generated_one_manifold_transport import jet,normalized

OUT=Path(__file__).resolve().parent/'figures/generated_one_nearest_direction'


def summarize(rows):
    return dict(count=len(rows),strict_correct=sum(x['strict_correct'] for x in rows),
                expected_among_closest=sum(x['expected_in_best_set'] for x in rows),
                ties=sum(x['tie_count']>1 for x in rows),
                greedy_correct=sum(x['correct'] for x in rows),
                global_assignment_correct=sum(x['global_assignment_correct'] for x in rows))


def aggregates(result):
    output=[]
    for method,key in [('true_fields','oracle_comparisons'),('svd_fields','svd_comparisons'),('follow_svd','svd_follow_comparisons')]:
        rows=result[key]
        radii=sorted(set(x.get('local_radius') for x in rows),key=lambda x:0 if x is None else x)
        for radius in radii:
            for distance in sorted(set(x['distance'] for x in rows)):
                base=[x for x in rows if x['distance']==distance and x.get('local_radius')==radius]
                for scope in ['all','smooth_pairs']:
                    subset=base if scope=='all' else [x for x in base if not x['source_cap_kink'] and not x['target_cap_kink']]
                    for movement in ['self','context','both']:
                        chosen=subset if movement=='both' else [x for x in subset if x.get('followed_self',True)==(movement=='self')]
                        if chosen:output.append(dict(method=method,local_radius=radius,distance=distance,
                                                    scope=scope,movement=movement,**summarize(chosen)))
    return output


def summary_figure(result,aggregate):
    fig=plt.figure(figsize=(16,9.3),facecolor='white')
    fig.text(.035,.97,'Does the nearest direction at the next image keep the same knob?',fontsize=21,weight='bold',va='top')
    fig.text(.035,.91,
             'Controlled generated 1s. Each trial moves to another valid image, then selects the largest absolute pixel cosine.\n'
             'A success requires the same knob category to be uniquely nearest; ties within 10⁻¹⁰ count as ambiguous. Both movement signs are tested.',
             fontsize=11,va='top',linespacing=1.5)
    for left,name in [(.045,'root'),(.15,'spread_03'),(.255,'spread_07')]:
        p=next(s['settings'] for s in result['starts'] if s['name']==name)
        ax=fig.add_axes([left,.705,.075,.14])
        ax.imshow(render(p)[0],cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
        ax.set_xticks([]);ax.set_yticks([]);ax.set_title(name.replace('_',' '),fontsize=9)
    fig.text(.37,.82,
             f"Starting images shown as examples; white = no ink, black = full ink.\n"
             f"True-field test: {result['method']['oracle_starts']} starts. SVD tests: {result['method']['svd_starts']} starts; 256 equal-weight neighbors per fit.\n"
             'Root (cx, cy, height, width, lean) = (14, 14, 19.75, 3.2, 0°).\n'
             'SVD endpoint samples use a different sampling seed from the source.',fontsize=10.5,va='top',linespacing=1.5)
    axes=[]
    for left,title in [(.065,'True known knob directions'),(.385,'Ordinary SVD: move a true knob'),(.705,'Ordinary SVD: follow its vector')]:
        ax=fig.add_axes([left,.23,.26,.40]);axes.append(ax)
        ax.set_title(title,fontsize=12,pad=12)
        ax.set_xscale('log');ax.set_xlim(7e-6,4);ax.set_ylim(0,105)
        ax.set_xlabel('Step: distance between the two images',fontsize=10)
        ax.set_xticks([1e-5,.001,.01,.1,.3,1,3],['10⁻⁵','0.001','0.01','0.1','0.3','1','3'])
        ax.tick_params(axis='x',labelsize=8)
        ax.grid(alpha=.15);ax.axhline(100,color='#888',lw=.6,ls=':')
    axes[0].set_ylabel('Unique nearest match has same category (%)',fontsize=10)
    def curve(ax,method,movement,radius,label,color,linestyle):
        rows=[x for x in aggregate if x['method']==method and x['movement']==movement and x['scope']=='all' and x['local_radius']==radius]
        rows.sort(key=lambda x:x['distance'])
        ax.plot([x['distance'] for x in rows],[100*x['strict_correct']/x['count'] for x in rows],
                marker='o',markersize=4,color=color,ls=linestyle,label=label)
        return rows
    aa=curve(axes[0],'true_fields','self',None,'Move the tracked knob','#0077aa','-')
    bb=curve(axes[0],'true_fields','context',None,'Move another knob','#cc6633','--')
    at_point=[x for x in result['oracle_comparisons'] if x['distance']==.1]
    failures=sum(not x['strict_correct'] for x in at_point)
    axes[0].annotate(f'At 0.1: {failures} failures / {len(at_point):,}\nSmall dip is hidden at this scale',
                     xy=(.1,100*(1-failures/len(at_point))),xytext=(.003,77),fontsize=8,
                     arrowprops=dict(arrowstyle='->',color='#555'),color='#333')
    for radius,color in [(1e-3,'#0077aa'),(1e-5,'#cc6633')]:
        curve(axes[1],'svd_fields','self',radius,f'Self; neighborhood {radius:g}',color,'-')
        curve(axes[1],'svd_fields','context',radius,f'Other knob; neighborhood {radius:g}',color,'--')
    cc=curve(axes[2],'follow_svd','self',1e-5,'Neighborhood 0.00001','#228855','-')
    for ax in axes:ax.legend(loc='lower left',fontsize=8.5)
    control=summarize(result['zero_move_seed_controls'])
    fig.text(.065,.17,f"True-field self trials per distance: {min(x['count'] for x in aa)}–{max(x['count'] for x in aa)}.\n"
             'Bounds limit which large steps are reachable.',fontsize=9,va='top',linespacing=1.5)
    fig.text(.385,.17,'The two neighborhood sizes can overlap on the plot.\n'
             'Generator labels evaluate the match; they do not choose it.',fontsize=9,va='top',linespacing=1.5)
    fig.text(.705,.17,f"Zero-move, new-seed control: {control['strict_correct']}/{control['count']} correct.\n"
             'A known-renderer retraction keeps vector-following on the manifold.',fontsize=9,va='top',linespacing=1.5)
    fig.text(.035,.075,
             'Self/context tests compare all five source directions with all five endpoint directions. SVD categories use a one-to-one assignment to true derivatives.\n'
             'These categories need not be pure knob directions. True-field derivatives at exact renderer cap kinks use averaged sides; smooth-pair counts are in the data.\n'
             'Measured finite experiments; no claim that the same step works for every possible start. All comparisons use all 784 pixels; signs are reversible.',
             fontsize=9.5,va='top',linespacing=1.5)
    fig.savefig(OUT/'nearest_direction_success.png',dpi=150);plt.close(fig)


def counterexample_figure(result,archive):
    # Include one successful changing vector and two robust ordinary-SVD
    # failures, with cases specified independently of display orientation.
    success=next(x for x in result['svd_follow_comparisons'] if x['name']=='spread_07' and
                 x['distance']==.1 and x['expected_knob']=='cy' and x['parameter_velocity'][1]<0)
    fail=next(x for x in result['svd_follow_comparisons'] if x['name']=='spread_07' and
              x['distance']==1. and x['expected_knob']=='cy' and x['parameter_velocity'][1]<0)
    width=next(x for x in result['svd_comparisons'] if x['name']=='root' and x['local_radius']==1e-5 and
               x['distance']==3 and x['moved_knob']=='width' and x['sign']==1 and x['expected_knob']=='width')
    selected=[('Follow cy-associated vector\nSmall step succeeds',success,'follow'),
              ('Follow cy-associated vector\nLarger step switches to cx',fail,'follow'),
              ('Move true width\nOld width is closer to height',width,'pure')]
    fits={f['key']:f for f in result['endpoint_fits']}
    panels=[]
    for label,row,mode in selected:
        if mode=='follow':
            source=archive[row['endpoint_key']+'_source_vector'];target=archive[row['endpoint_key']+'_target_basis']
            labels=row['target_labels']
        else:
            source=archive[f"{row['name']}_{row['local_radius']}_source_basis"][row['query_index']]
            target=archive[row['endpoint_key']+'_basis'];labels=fits[row['endpoint_key']]['labels']
        expected=int(np.flatnonzero(np.array(labels)==KNOBS.index(row['expected_knob']))[0])
        same=target[expected].copy();chosen=target[row['chosen_index']].copy()
        if source@same<0:same*=-1
        if source@chosen<0:chosen*=-1
        panels.append((label,row,[source,same,chosen]))
    limit=max(np.abs(v).max() for _,_,values in panels for v in values)
    fig=plt.figure(figsize=(16,10.3),facecolor='white')
    fig.text(.035,.97,'Nearest-direction tracking: a small step succeeds, larger steps can change identity',fontsize=19,weight='bold',va='top')
    fig.text(.035,.91,
             'Ordinary equal-weight SVD, 256 neighbors per fit; neighborhood image radius 0.00001. Matching uses absolute pixel cosine.\n'
             'Images: white = 0 ink, black = 1. Unit vectors: red gains ink, blue loses ink; common color scale and signs aligned with the source.\n'
             'Knob names are evaluation categories assigned from known derivatives; a fitted vector can mix physical knobs.',fontsize=10.5,va='top',linespacing=1.5)
    grid=fig.add_gridspec(3,5,left=.25,right=.97,top=.79,bottom=.18,wspace=.35,hspace=.65)
    titles=['Starting 1','Next 1','Direction at start','Same category at next 1','Nearest direction at next 1']
    for i,(label,row,values) in enumerate(panels):
        for k in range(5):
            ax=fig.add_subplot(grid[i,k]);ax.set_xticks([]);ax.set_yticks([])
            if k<2:
                p=row['start_settings'] if k==0 else row['end_settings']
                ax.imshow(render(p)[0],cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
                caption=f'({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f},\n{p[3]:.3f}, {p[4]:.2f}°)'
            else:
                shown=ax.imshow(values[k-2].reshape(28,28),cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
                caption=(row['expected_knob'] if k==2 else
                         f"{row['expected_knob']} · cosine {row['same_knob_score']:.4f}" if k==3 else
                         f"{row['chosen_knob']} · cosine {row['best_score']:.4f}")
            if i==0:ax.set_title(titles[k],fontsize=9,pad=12)
            ax.text(.5,-.10,caption,transform=ax.transAxes,ha='center',va='top',fontsize=8.6)
        y=.18+.61*(1-(i+.5)/3)
        fig.text(.025,y,label+f"\nStep image distance {row['distance']:g}\nSame-minus-runner-up {row['margin']:+.4f}",
                 fontsize=10.5,va='center',linespacing=1.6)
    fig.text(.035,.105,'Settings below images: (cx, cy, height, width, lean in degrees). All pixels participate in matching.\n'
             'Vector-following uses the least-squares parameter velocity at the source, then renders the resulting state.\n'
             'All three outcomes repeat with 1,024 neighbors and three independent seed pairs; they are not post-fit basis rotations.',fontsize=9.5,va='top',linespacing=1.5)
    cax=fig.add_axes([.65,.045,.30,.016]);fig.colorbar(shown,cax=cax,orientation='horizontal').set_label('Pixel change per unit vector coefficient',fontsize=9)
    fig.savefig(OUT/'nearest_direction_examples.png',dpi=150);plt.close(fig)


def main():
    with gzip.open(OUT/'results.json.gz','rt') as f:
        result=json.load(f)
    aggregate=aggregates(result)
    summary=dict(method=result['method'],aggregate=aggregate,zero_move_controls=summarize(result['zero_move_seed_controls']))
    for name,key in [('true_fields','oracle_comparisons'),('svd_fields','svd_comparisons'),('follow_svd','svd_follow_comparisons')]:
        rows=result[key]
        failures=[x for x in rows if not x['expected_in_best_set']]
        strict_failures=[x for x in rows if not x['strict_correct']]
        summary[name]=dict(total=summarize(rows),minimum_step_with_wrong_match=min((x['distance'] for x in failures),default=None),
                           minimum_step_with_wrong_or_ambiguous_match=min((x['distance'] for x in strict_failures),default=None))
        subset=[x for x in rows if x['distance']==.1 and (name!='svd_fields' or x['local_radius']==1e-5)]
        angles=np.rad2deg(np.arccos(np.clip([x['same_knob_score'] for x in subset],0,1)))
        summary[name]['same_category_turn_degrees_at_step_0p1']=dict(median=float(np.median(angles)),maximum=float(angles.max()))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (OUT/'aggregate.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(aggregate[0]));w.writeheader();w.writerows(aggregate)
    archive=np.load(OUT/'basis_vectors.npz')
    summary_figure(result,aggregate);counterexample_figure(result,archive)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['aggregate','method']},indent=2))


if __name__=='__main__':main()
