"""Audit fixed five-curve decoder: reconstruction artifacts and local knob moves."""
import os
os.environ['OPENBLAS_NUM_THREADS']='4'
os.environ['OMP_NUM_THREADS']='4'
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import label
from scipy.stats import qmc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from grey_ones import render, RANGES, KNOBS
from optimize_generated_one_joint_curves import evaluate, encode_multistart, encode

OUT=Path('figures/generated_one_principal_curves/quality')
MODEL=OUT.parent/'joint'


def capture(x,y):
    return float(100*(1-np.sum((x-y)**2)/np.sum((x-x.mean(0))**2)))


def detached(a,threshold):
    lab,n=label(a.reshape(28,28)>=threshold,np.ones((3,3)))
    if n<=1:return 0,0.,0.
    masses=np.bincount(lab.ravel(),weights=np.maximum(a,0),minlength=n+1);masses[0]=0
    largest=int(np.argmax(masses));mask=(lab.ravel()!=0)&(lab.ravel()!=largest)
    return int(mask.sum()),float(a[mask].sum()),float(a[mask].max(initial=0))


def main():
    OUT.mkdir(exist_ok=True)
    m=np.load(MODEL/'joint_513.npz');d=np.load(MODEL/'joint_513_dense.npz')
    exact=render(d['knobs']).reshape(-1,784).astype(float)
    pred=np.zeros_like(exact);pred[:,d['active']]=d['reconstructed']
    e=pred-exact;zero=exact==0
    err=np.linalg.norm(e,axis=1)
    r=dict(scope='Audit of saved joint_513 model without refitting; scores use unclipped pixels',
           points=len(exact),capture_percent=capture(exact,pred),
           pixel_RMSE=float(np.sqrt(np.mean(e**2))),max_pixel_error=float(abs(e).max()),
           image_L2_quantiles={str(q):float(np.quantile(err,q)) for q in [.5,.95,1]},
           artifact_thresholds={})
    for threshold in [.01,.05,.1]:
        components=np.array([detached(a,threshold) for a in pred])
        r['artifact_thresholds'][str(threshold)]=dict(
            mean_extra_pixels_on_exact_white=float(np.mean(np.sum(zero&(pred>=threshold),axis=1))),
            p95_extra_pixels_on_exact_white=float(np.quantile(np.sum(zero&(pred>=threshold),axis=1),.95)),
            percent_images_with_detached_ink=float(100*np.mean(components[:,0]>0)),
            mean_detached_pixels=float(components[:,0].mean()),
            max_detached_pixels=int(components[:,0].max()),
            mean_detached_ink_percent_of_source=float(100*np.mean(components[:,1]/exact.sum(1))),
            percent_pixels_outside_range_by_at_least_threshold=float(100*np.mean((pred < -threshold)|(pred > 1+threshold))))
        if threshold==.05: detached_mass=components[:,1]
    r['out_of_range_pixel_percent']=float(100*np.mean((pred<0)|(pred>1)))
    print('Artifact audit:',json.dumps(r),flush=True)
    # Small moves around nine different image states; no refitting.
    anchors=np.vstack([np.full(5,.5),.12+.76*qmc.Sobol(5,scramble=True,seed=862).random_base2(3)])
    local=np.repeat(anchors[:,None,None,:],5,axis=1)
    local=np.repeat(local,9,axis=2)
    for k in range(5):local[:,k,:,k]+=np.linspace(-.1,.1,9)
    pp=RANGES[:,0]+local.reshape(-1,5)*np.ptp(RANGES,axis=1)
    xx=render(pp).reshape(-1,784).astype(float)
    support=d['active'];reference=render(m['coverage_knobs']).reshape(-1,784)[:,support]
    assert not np.any(xx[:,~support])
    yy,tt,simple=encode_multistart(xx[:,support],m['mean'][support],m['curves'][:,:,support],reference,m['coordinates'])
    full=np.zeros_like(xx);full[:,support]=yy
    x=xx.reshape(9,5,9,784);y=full.reshape(9,5,9,784)
    # Track each sweep outward from its middle instead of independently restarting.
    continuous=full.reshape(45,9,784).copy()
    coord=tt.reshape(45,9,5)
    target=xx.reshape(45,9,784)
    for order in [range(5,9),range(3,-1,-1)]:
        previous=coord[:,4].copy()
        for step in order:
            part,previous=encode(target[:,step,support],m['mean'][support],m['curves'][:,:,support],previous,cycles=30)
            continuous[:,step,support]=part
    continuous=continuous.reshape(9,5,9,784)
    rows=[]
    for a in range(9):
        for k in range(5):
            dx=x[a,k]-x[a,k,4];dy=y[a,k]-y[a,k,4]
            rows.append(dict(anchor=a,knob=KNOBS[k],
                anchored_change_capture_percent=float(100*(1-np.sum((dx-dy)**2)/np.sum(dx**2))),
                continuation_change_capture_percent=float(100*(1-np.sum((dx-(continuous[a,k]-continuous[a,k,4]))**2)/np.sum(dx**2))),
                rms_change_L2=float(np.sqrt(np.mean(np.sum((dx-dy)**2,axis=1)))),
                rms_actual_change_L2=float(np.sqrt(np.mean(np.sum(dx**2,axis=1))))))
    r['local_moves']=dict(anchors_normalized=anchors.tolist(),
        definition='Move one knob from -10% to +10% of its allowed range around each of nine starts; subtract each reconstructed starting image before measuring change error. 9 samples per sweep, 45 sweeps; coordinates inferred from pixels.',
        global_image_capture_percent=capture(xx,full),
        sequential_start_global_capture_percent=capture(xx[:,support],simple),
        sweeps=rows,continuation_global_image_capture_percent=capture(xx,continuous.reshape(-1,784)))
    print('Local capture ranges:',{k:[min(v['anchored_change_capture_percent'] for v in rows if v['knob']==k),max(v['anchored_change_capture_percent'] for v in rows if v['knob']==k)] for k in KNOBS},flush=True)
    np.savez_compressed(OUT/'local_moves.npz',knobs=pp,images=xx,predictions=full,coordinates=tt,continuation=continuous.reshape(-1,784))
    (OUT/'results.json').write_text(json.dumps(r,indent=2)+'\n')
    selected=[int(np.argmin(abs(err-np.quantile(err,.5)))),int(np.argmin(abs(err-np.quantile(err,.95)))),int(np.argmax(err)),int(np.argmax(detached_mass))]
    labels=['Median image error','95th-percentile error','Largest image error','Most detached ink (at 5%)']
    fig,axs=plt.subplots(4,3,figsize=(10.4,12.1))
    fig.subplots_adjust(left=.12,right=.88,top=.87,bottom=.12,wspace=.3,hspace=.45)
    fig.text(.07,.974,'Are the reconstructed strokes speckled?',fontsize=19,weight='bold',va='top')
    fig.text(.07,.932,'Fixed five-curve decoder; 16,384 additional generated images. Each pixel is ink coverage: 0 to 1.',fontsize=10)
    titles=['Exact source\nwhite = no ink; black = full ink','Reconstruction at normal contrast\ndisplay clips to [0, 1]; scores do not','Signed pixel error\nred = excess ink; blue = missing ink']
    for row,(idx,name) in enumerate(zip(selected,labels)):
        for col,img in enumerate([exact[idx],pred[idx],e[idx]]):
            ax=axs[row,col]
            if col<2:ax.imshow(img.reshape(28,28),cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
            else:im=ax.imshow(img.reshape(28,28),cmap='RdBu_r',vmin=-.35,vmax=.35,interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            if row==0:ax.set_title(titles[col],fontsize=10,pad=12)
        axs[row,0].set_ylabel(name,fontsize=10,labelpad=13)
        axs[row,1].set_xlabel(f'Image L2 error {err[idx]:.3f}',fontsize=10)
        axs[row,2].set_xlabel(f'Largest pixel error {abs(e[idx]).max():.3f}',fontsize=10)
    ca=fig.add_axes([.23,.085,.54,.015]);fig.colorbar(im,cax=ca,orientation='horizontal',label='Prediction minus source: coverage difference; common scale across all rows')
    fig.text(.07,.012,'Examples selected by error or detached ink, not appearance. Entire 28 x 28 images shown.\nDetached ink = thresholded pixels outside the largest 8-connected component; threshold 0.05.',fontsize=9.5)
    fig.savefig(OUT/'reconstruction_audit.png',dpi=145);plt.close(fig)
    # Local-change heatmap: unlike global capture, small moves have their own denominator.
    fig,axs=plt.subplots(1,2,figsize=(14,7.4));fig.subplots_adjust(left=.09,right=.88,top=.73,bottom=.28,wspace=.28)
    fig.text(.055,.97,'Do small knob changes reconstruct correctly from different starts?',fontsize=18,weight='bold',va='top')
    fig.text(.055,.915,'Nine starts x five knobs; each sweep spans +/-10% of its knob range (9 images). Fixed decoder; no refitting.',fontsize=11)
    fig.text(.055,.869,'Compare predicted changes with exact changes after subtracting the corresponding starting image.',fontsize=11)
    for ax,key,title in zip(axs,['anchored_change_capture_percent','continuation_change_capture_percent'],['Infer positions separately for every image','Start each search from the previous image']):
        table=np.array([v[key] for v in rows]).reshape(9,5)
        im=ax.imshow(table,cmap='RdYlGn',vmin=0,vmax=100,aspect='auto')
        ax.set_title(title,fontsize=12,pad=15)
        ax.set_xticks(range(5),['horizontal\ncenter','vertical\ncenter','height','width','lean'],fontsize=10)
        ax.set_yticks(range(9),['center']+[f'start {i}' for i in range(1,9)])
        for a in range(9):
            for k in range(5):ax.text(k,a,f'{table[a,k]:.0f}%',ha='center',va='center',fontsize=9,color='black')
    ca=fig.add_axes([.91,.29,.014,.43]);fig.colorbar(im,cax=ca,label='Local change captured (%)')
    fig.text(.055,.105,'100% = exact changes; 0% = as much squared error as the change itself; negative = worse.\nColors saturate at 0 and 100; printed scores round to whole percentages. All settings and scores are in results.json.\nFollowing the previous image improves several small moves but worsens overall source-image fitting.\nThis checks sampled local moves, not every starting image or every possible move.',fontsize=10)
    fig.savefig(OUT/'local_moves.png',dpi=145);plt.close(fig)
    summaries=''.join(f'<tr><td>{k}</td><td>{min(v["anchored_change_capture_percent"] for v in rows if v["knob"]==k):.1f}%</td><td>{max(v["anchored_change_capture_percent"] for v in rows if v["knob"]==k):.1f}%</td></tr>' for k in KNOBS)
    html=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Five-curve image quality audit</title><style>body{{max-width:1050px;margin:32px auto;padding:0 22px;font:17px/1.6 system-ui;color:#172b3a;background:#fafafa}}img{{width:100%;height:auto}}td,th{{padding:7px 18px;text-align:left}}code{{background:#edf0f3}}a{{color:#2563a6}}</style><h1>Five-curve reconstruction quality</h1><p><a href="../joint/">Joint fit</a> · <a href="../direct_construction/">Direct knob construction</a> · <a href="results.json">Full measurements</a></p><p>The model stays fixed throughout this audit. Coordinates are inferred from each source image's pixels; no stroke-rendering equation appears in the decoder. Scores use raw predictions without clipping.</p><h2>Why it can handle interactions</h2><p>Prediction = intercept + C1(t1) + C2(t2) + C3(t3) + C4(t4) + C5(t5). Each position t can depend on all five knobs through the source pixels. Changing width from another lean angle can select different positions on several curves. The additive restriction in curve coordinates does not imply additivity in physical knob values.</p><h2>Global accuracy and visible artifacts</h2><p>The original fitting cloud captures 99.7336%; these 16,384 additional images capture {r['capture_percent']:.4f}%. Mean pixel RMS error is {r['pixel_RMSE']:.4f} coverage; maximum pixel error is {r['max_pixel_error']:.3f}. The global variance denominator is distance from the overall image mean; no zero-pixel dilution is used in this score.</p><img src="reconstruction_audit.png" alt="Exact sources, reconstructions at normal contrast, and signed error maps"><p>There are faint artifacts. At a 0.05 coverage threshold, {r['artifact_thresholds']['0.05']['percent_images_with_detached_ink']:.2f}% of images contain ink detached from the largest component, averaging {r['artifact_thresholds']['0.05']['mean_detached_pixels']:.3f} detached pixels per image. At 0.01 coverage, the corresponding fraction is {r['artifact_thresholds']['0.01']['percent_images_with_detached_ink']:.2f}%. Pixel errors and out-of-range excursions remain real even when a normal-contrast display makes them hard to see.</p><h2>Different starting strokes</h2><p>Previous checks included 16,384 additional joint states, 2,048 joint midpoints, and all 32 corners. This new check sweeps each physical knob around nine starting strokes; each local move is compared with its own reconstructed starting image. All five curve coordinates are re-inferred for each image.</p><img src="local_moves.png" alt="Local change capture across nine starting strokes and five knob sweeps"><table><tr><th>Knob</th><th>Lowest local change capture</th><th>Highest</th></tr>{summaries}</table><p>A high global score does not establish accurate small moves: local changes have much smaller variance. This audit does not certify global optimality, continuity of inferred coordinates, exact validity of reconstructed strokes, or validity of arbitrary curve-coordinate combinations.</p><p>Additional continuation check: initialize each next image from the previous image's coordinates along each sweep. Its results are stored per sweep as continuation_change_capture_percent in results.json. This improves the small width, height, and position changes substantially. Local height change capture still ranges from -52% to 51%, and vertical position from -7% to 62%. Overall source-image capture drops from 99.4133% to 98.9750%. This reduces restarts but does not establish a globally optimal inverse.</p><h2>Coordinate-search initialization</h2><p>On the 16,384 additional images, sequential projection alone captures 99.1184%; selecting the better of that and a nearest-covering-image coordinate initialization captures 99.4661%. The 99.7336% fitting-cloud score uses stored fitted coordinates. Curve positions are optimized separately for each source image; this is reconstruction from known pixels, not direct generation from requested knobs.</p></html>'''
    (OUT/'index.html').write_text(html)
    (OUT/'README.md').write_text('# Five-curve reconstruction audit\n\nRun `analyze_generated_one_curve_quality.py` to reproduce the audit of the saved model. See `results.json` for thresholds, local moves and scope. Figure scores use raw, unclipped predictions; only grayscale displays clip. No model refitting.\n')
    for page in ['joint','direct_construction']:
        path=OUT.parent/page/'index.html';text=path.read_text();marker='<!-- reconstruction-quality-link -->'
        if marker not in text:text=text.replace('</body>',marker+'<p><a href="../quality/">Image quality and different starting strokes</a></p></body>') if '</body>' in text else text+marker+'<p><a href="../quality/">Image quality and different starting strokes</a></p>'
        path.write_text(text)
    print('Saved quality report and figures.',flush=True)

if __name__=='__main__':main()
