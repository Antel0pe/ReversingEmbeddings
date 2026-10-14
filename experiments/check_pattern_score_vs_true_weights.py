"""Does the previous cross-path score favor true five-vector combinations?

Run: OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/check_pattern_score_vs_true_weights.py
A small diagnostic scan of mixtures, not a new blind knob-recovery method.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/pattern-score-weights-mpl')
from pathlib import Path
import sys
import json
import csv
import numpy as np
from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from grey_ones import KNOBS
from analyze_generated_one_manifold_transport import jet
from experiments.svd_cross_path_patterns import BASE,OFFSETS,UNSEEN,TICKS,MID,observe,profile,candidates,true_curves,pixels,valid

OUT=ROOT/'figures'/'pattern_score_vs_true_weights'
SOURCE=ROOT/'figures'/'generated_one_cross_path_patterns'
ALPHAS=np.array([-1.,-.5,0.,.25,.5,.75,1.,1.25,1.5,2.])
ZERO=int(np.flatnonzero(ALPHAS==0)[0])
ONE=int(np.flatnonzero(ALPHAS==1)[0])


def run():
    old=json.loads((SOURCE/'results.json').read_text())
    z=np.load(SOURCE/'paths_and_profiles.npz')
    root=observe(BASE)
    assert np.max(np.abs(root['curves']-z['root_curves']))<1e-12
    labels=np.array([KNOBS.index(k) for k in old['root']['labels']])
    root_signs=np.array(old['root']['signs'])
    pure_root,_=true_curves(root,labels,root_signs)
    references=[root['curves'],pure_root]
    feature=np.empty((2,5,5,len(ALPHAS),6,3*len(TICKS)-1))
    error=np.empty((5,5,len(ALPHAS),6))
    actual_directions=np.empty((5,5,len(ALPHAS),6,784))
    selected_curves={}
    for b in range(5):
        for j in range(6):
            training=j<4
            p=z['settings'][b,j] if training else z['unseen_settings'][b,j-4]
            node=observe(p)
            jmat=jet(p)[3]
            pinv=np.linalg.pinv(jmat)
            obs=candidates(root['curves'],node)
            orders=z['assignments'][b,j] if training else z['unseen_assignments'][b,j-4]
            for f in range(5):
                k=int(orders[f]);sign=obs['signs'][f,k]
                coeff_svd=np.zeros(5);coeff_svd[k]=sign
                true_unit=jmat[:,labels[f]]*root_signs[f]
                true_unit/=np.linalg.norm(true_unit)
                coeff_true=node['basis']@true_unit
                coeff=(1-ALPHAS[:,None])*coeff_svd+ALPHAS[:,None]*coeff_true
                requested=coeff@node['basis']
                lifts=requested@pinv.T
                speed=np.linalg.norm(lifts@jmat.T,axis=1)
                lifts/=speed[:,None]
                actual=lifts@jmat.T
                actual_directions[b,f,:,j]=actual
                error[b,f,:,j]=np.linalg.norm(actual-true_unit,axis=1)
                settings=p[None,None,:]+TICKS[None,:,None]*lifts[:,None,:]
                assert valid(settings.reshape(-1,5))
                curves=pixels(settings.reshape(-1,5)).reshape(len(ALPHAS),len(TICKS),784)
                stored=z['candidate_curves'][b,j,k][::sign] if training else z['unseen_curves'][b,j-4,k][::sign]
                assert np.max(np.abs(curves[ZERO]-stored))<5e-12
                for ref_index,reference in enumerate(references):
                    for a in range(len(ALPHAS)):
                        pp=profile(reference[f],curves[a])
                        # All tested mixtures retain the original orientation.
                        assert np.mean(pp['mismatch']**2) <= np.mean(profile(reference[f],curves[a,::-1])['mismatch']**2)+1e-10
                        feature[ref_index,b,f,a,j]=pp['feature']
                if b==0 and f==4 and j in [0,3]:
                    selected_curves[f'height_under_lean_offset_{j}']=curves
        print(f'Context branch {b+1}/5 scanned.',flush=True)
    train=np.sum((feature[:,:,:,:,1:4]-feature[:,:,:,:,0:1])**2,axis=(-1,-2))
    template=feature[:,:,:,:,:4].mean(axis=-2)
    fresh=np.sum((feature[:,:,:,:,4:]-template[:,:,:,:,None,:])**2,axis=(-1,-2))
    # Verify the original objective, including every context/family pair.
    for b in range(5):
        assert abs(train[0,b,:,ZERO].sum()-old['contexts'][b]['selected_loss'])<1e-9
        assert abs(train[1,b,:,ONE].sum()-old['contexts'][b]['pure_control_loss'])<1e-7
    cross=np.ones((5,5),bool);np.fill_diagonal(cross,False)
    scores=[];records=[]
    for r,refname in enumerate(['original_svd_root_reference','known_pure_root_reference']):
        total=np.array([train[r,:,f][np.arange(5)!=f].sum(axis=0) for f in range(5)])
        unseen_total=np.array([fresh[r,:,f][np.arange(5)!=f].sum(axis=0) for f in range(5)])
        best=np.argmin(total,axis=1)
        correlations=[]
        for b in range(5):
            for f in range(5):
                if b==f:continue
                rho=float(spearmanr(train[r,b,f],error[b,f,:,:4].mean(axis=1)).statistic)
                correlations.append(rho)
                for a,alpha in enumerate(ALPHAS):
                    records.append(dict(reference=refname,context=KNOBS[labels[b]],family=KNOBS[labels[f]],
                                        alpha=float(alpha),training_score=float(train[r,b,f,a]),
                                        fresh_score=float(fresh[r,b,f,a]),
                                        mean_direction_error=float(error[b,f,a,:4].mean())))
        families=[]
        for f in range(5):
            chosen=best[f]
            mask=np.arange(5)!=f
            families.append(dict(family=f+1,knob=KNOBS[labels[f]],best_tested_alpha=float(ALPHAS[chosen]),
                original_svd_score=float(total[f,ZERO]),true_combination_score=float(total[f,ONE]),
                best_tested_score=float(total[f,chosen]),
                improvement_over_true_percent=float(100*(1-total[f,chosen]/total[f,ONE])),
                original_svd_fresh_score=float(unseen_total[f,ZERO]),true_combination_fresh_score=float(unseen_total[f,ONE]),
                best_tested_fresh_score=float(unseen_total[f,chosen]),
                best_mean_direction_error=float(error[:,f,chosen,:4][mask].mean()),
                original_mean_direction_error=float(error[:,f,ZERO,:4][mask].mean()),
                true_combination_mean_direction_error=float(error[:,f,ONE,:4][mask].mean())))
        # Independent families cannot collapse to one repeated direction.
        smallest=1.
        for b in range(5):
            for j in range(6):
                frame=np.array([actual_directions[b,f,best[f],j] for f in range(5)])
                smallest=min(smallest,float(np.linalg.svd(frame,compute_uv=False)[-1]))
        true_better=sum(train[r,b,f,ONE]<train[r,b,f,ZERO] for b in range(5) for f in range(5) if b!=f)
        best_beats_true=sum(np.min(train[r,b,f])<train[r,b,f,ONE]*(1-1e-8) for b in range(5) for f in range(5) if b!=f)
        scores.append(dict(reference=refname,cross_knob_pairs=20,true_better_than_original_count=true_better,
                           some_tested_mixture_beats_true_count=best_beats_true,
                           median_within_pair_spearman_score_vs_true_error=float(np.median(correlations)),
                           minimum_joint_frame_singular_value=smallest,families=families))
    result=dict(method=dict(root=BASE.tolist(),radius=1e-4,neighbors=256,
                           alphas=ALPHAS.tolist(),training_offsets=OFFSETS.tolist(),fresh_offsets=UNSEEN.tolist(),ticks=TICKS.tolist(),
                           mixtures='five-dimensional coefficients interpolate from selected signed SVD axis (alpha 0) to known true-knob projection (alpha 1); extrapolations also tested',
                           valid_moves='same Jacobian-based valid-ray lift as the original experiment, with initial pixel speed normalized to one',
                           reference='original fixed SVD root paths for the primary comparison; known-pure root paths are a separate anchor diagnostic',
                           objective='original three normalized profiles, nearest-offset-anchored squared discrepancy; no changes to weights or normalization',
                           selection='one alpha per family, shared across its four OTHER-knob context branches; minimize summed training score on the stated finite grid',
                           holdout='score fresh offset curves against mean training profiles for that same alpha; alpha is chosen on training only',
                           scope='small root-local scan, not an arbitrary-mixture/global optimum or a blind recovery method',
                           oracle='true directions define the alpha=1 candidate endpoint and evaluate purity; the score does not receive knob identities'),
                original_score_reproduced=True,pure_reference_score_reproduced=True,
                results=scores)
    return result,records,dict(alphas=ALPHAS,training_scores=train,fresh_scores=fresh,
                               direction_errors=error,features=feature,root_svd_curves=root['curves'],
                               root_pure_curves=pure_root,labels=labels,**selected_curves)


def plot(result,arrays):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig=plt.figure(figsize=(15,10.5),facecolor='#fafaf7')
    fig.text(.035,.972,'Does the shared distance-and-angle score prefer the true knobs?',fontsize=21,weight='bold',va='top')
    ax0=fig.add_axes([.04,.823,.075,.102])
    ax0.imshow(pixels(BASE)[0].reshape(28,28),cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
    ax0.set_xticks([]);ax0.set_yticks([])
    ax0.text(.5,-.10,'Root 1',transform=ax0.transAxes,ha='center',fontsize=9)
    fig.text(.145,.909,'Generated 28 × 28 coverage: white = no ink, black = full ink. One root; the same 20 cross-knob branch pairs.',fontsize=10.5)
    fig.text(.145,.879,'Mixing parameter: 0 = the selected SVD axis; 1 = the known true-knob combination. Both use the same five-vector span.',fontsize=10.5)
    fig.text(.145,.849,'Only the reference path changes between panels. Score = agreement across four offset profiles; lower is more consistent.',fontsize=10.5)
    names=['Lean','Width','Horizontal position','Vertical position','Height']
    colors=['#9165ad','#21948b','#3676ad','#c18a2a','#c35152']
    handles=[Line2D([0],[0],color=c,lw=2) for c in colors]
    fig.legend(handles,names,loc='center',bbox_to_anchor=(.54,.805),ncol=5,frameon=False,fontsize=10)
    for r in range(2):
        ax=fig.add_axes([.08+r*.5,.47,.37,.27])
        for f,fam in enumerate(result['results'][r]['families']):
            score=arrays['training_scores'][r,:,f][np.arange(5)!=f].sum(axis=0)
            ax.plot(ALPHAS,score/fam['true_combination_score'],color=colors[f],marker='.',lw=1.6)
        ax.axhline(1,color='#777777',ls=':',lw=1)
        ax.axvline(0,color='#999999',ls='--',lw=.8)
        ax.axvline(1,color='#444444',ls='--',lw=.8)
        ax.set_yscale('log');ax.grid(alpha=.18)
        ax.set_xlabel('Mixing parameter (0 = SVD, 1 = true)',fontsize=9)
        ax.set_ylabel('Score / true-combination score',fontsize=9)
        ax.tick_params(labelsize=9)
        title='Original SVD root reference' if r==0 else 'Known-pure root reference (diagnostic)'
        ax.set_title(title,fontsize=12,pad=13)
    ax=fig.add_axes([.08,.14,.37,.23]);x=np.arange(5)
    for r,color,shift in [(0,'#b96532',-.16),(1,'#397ba6',.16)]:
        y=[f['best_tested_alpha'] for f in result['results'][r]['families']]
        ax.bar(x+shift,y,.3,color=color,label=['SVD root reference','Pure root reference'][r])
    ax.axhline(1,color='#555555',ls='--',lw=1)
    ax.axhline(0,color='#999999',lw=.8)
    ax.set_xticks(x,['Lean','Width','X position','Y position','Height'],rotation=15)
    ax.set_ylabel('Best tested mixing parameter',fontsize=9)
    ax.set_title('Which weights minimize the sampled score?',fontsize=11,pad=12)
    ax.legend(fontsize=8,frameon=False);ax.tick_params(labelsize=8)
    ax=fig.add_axes([.58,.14,.37,.23])
    for r,color,shift in [(0,'#b96532',-.16),(1,'#397ba6',.16)]:
        y=np.array([f['best_mean_direction_error'] for f in result['results'][r]['families']])*100
        ax.bar(x+shift,y,.3,color=color)
    ax.set_xticks(x,['Lean','Width','X position','Y position','Height'],rotation=15)
    ax.set_ylabel('Mean error versus true unit direction (%)',fontsize=9)
    ax.set_title('A better score can still leave a mixed direction',fontsize=11,pad=12)
    ax.tick_params(labelsize=8);ax.grid(axis='y',alpha=.15)
    fig.text(.035,.055,'Each family uses one mixing parameter across all four other context branches. Directions remain independent; no family-collapse solution.',fontsize=10)
    fig.text(.035,.025,'Known true directions construct the comparison candidates. This finite scan does not prove a global minimum or discover physical knobs blindly.',fontsize=10)
    fig.savefig(OUT/'score_vs_weights.png',dpi=155,facecolor=fig.get_facecolor());plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    result,records,arrays=run()
    (OUT/'results.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item())+'\n')
    with (OUT/'scores.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    np.savez_compressed(OUT/'scan_data.npz',**arrays)
    plot(result,arrays)
    print(json.dumps(result['results'],indent=2,default=lambda x:x.item()),flush=True)


if __name__=='__main__':
    main()
