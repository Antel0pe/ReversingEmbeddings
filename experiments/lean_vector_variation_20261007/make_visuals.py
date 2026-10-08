"""Explain the measured vector model without importing renderer geometry."""
from pathlib import Path
import base64
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

OUT=Path(__file__).resolve().parent/'results'

def main():
    m=json.loads((OUT/'metrics.json').read_text())
    model=np.load(OUT/'width_model.npz');d=np.load(OUT/'width_evaluation.npz')
    err=np.linalg.norm(d['predicted']-d['observed'],axis=1)/np.linalg.norm(d['observed'],axis=1)
    ix=int(err.argmax());p=d['settings'][ix]
    reference=model['origin'];v0=model['base_lean_vector'];b=model['change_vector']
    q=d['test_q'][ix]
    alpha=np.interp(q,model['coefficient_knots'],model['coefficient_values'])
    err16=np.linalg.norm(d['predicted16'][ix]-d['observed'][ix])/np.linalg.norm(d['observed'][ix])
    fig=plt.figure(figsize=(14.5,9.3),facecolor='white')
    fig.suptitle('Can a lean vector be predicted by adding a learned change-vector?',fontsize=18,fontweight='bold',y=.975)
    fig.text(.5,.932,'Width-only upright slice: center and height fixed during observation. Prediction receives the 784-pixel image only.',ha='center',fontsize=10.6)
    fig.text(.5,.902,'Measured arrow = [image after +0.02° lean − image after −0.02° lean] / 0.04°. No geometry or inverse is used.',ha='center',fontsize=10.2)
    gs=fig.add_gridspec(2,4,left=.065,right=.89,top=.83,bottom=.25,hspace=.65,wspace=.45)
    arrays=[reference,d['images'][ix],v0,b]
    scale=max(np.abs(v0).max(),np.abs(b).max(),np.abs(d['observed'][ix]).max(),np.abs(d['predicted'][ix]).max())
    titles=['Reference image x₀',f'Unseen image x (width {p[3]:.5f})','Reference lean vector v₀','Change-vector b']
    def panel(ax,a,signed):
        if signed:im=ax.imshow(a.reshape(28,28),cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-scale,vcenter=0,vmax=scale),interpolation='nearest')
        else:im=ax.imshow(a.reshape(28,28),cmap='gray_r',vmin=0,vmax=1,interpolation='nearest')
        ax.set_xticks([0,14,27]);ax.set_yticks([0,14,27]);ax.tick_params(labelsize=7)
        ax.set_xlabel('pixel column',fontsize=8);ax.set_ylabel('pixel row (down)',fontsize=8)
        return im
    for j,a in enumerate(arrays):
        ax=fig.add_subplot(gs[0,j]);im=panel(ax,a,j>=2);ax.set_title(titles[j],fontsize=10)
        txt='coverage: 0 no ink, 1 full ink' if j<2 else f'length {np.linalg.norm(a):.4f} /degree'
        ax.text(.5,-.31,txt,ha='center',transform=ax.transAxes,fontsize=8)
    ax=fig.add_subplot(gs[1,:2]);ax.plot(model['coefficient_knots'],model['coefficient_values'],color='#2d729e',lw=2)
    ax.scatter([q],[alpha],s=45,color='#be4934',zorder=3,label='unseen example')
    ax.set(xlabel='q = image displacement projected onto a learned vector',ylabel='α(q): amount of change-vector added',title='One scalar function learned from width observations')
    ax.legend(fontsize=8);ax.grid(alpha=.2);ax.tick_params(labelsize=8)
    for j,a,title in [(2,d['observed'][ix],'Observed lean vector'),(3,d['predicted'][ix],'Prediction v₀ + α(q)b')]:
        ax=fig.add_subplot(gs[1,j]);im=panel(ax,a,True);ax.set_title(title,fontsize=10)
        text='Unseen transition stress case' if j==2 else f'{err[ix]*100:.2f}% vector error'
        ax.text(.5,-.31,text,ha='center',transform=ax.transAxes,fontsize=8.5)
    ca=fig.add_axes([.925,.39,.012,.34]);fig.colorbar(im,cax=ca).set_label('signed coverage change /degree',fontsize=9)
    fig.text(.065,.135,'v₀ and b are actual 784-component pixel vectors. Red/blue in v₀ and the lean vectors means ink gain/loss.',fontsize=10)
    fig.text(.065,.105,'In b, red/blue means increasing/decreasing the lean response itself. It is a change to a vector, not an image move.',fontsize=9.7)
    fig.text(.065,.075,f'Worst one-vector width test: {err.max()*100:.2f}%. Adding 16 learned variation vectors: worst confirmed stress error 0.376%; this example {err16*100:.3f}%.',fontsize=9.5)
    fig.text(.065,.045,'Not a whole-manifold result: the same width-only 16-vector model has >100% mean error on 512 full five-knob states.',fontsize=9.5)
    fig.text(.065,.015,'All pixels shown. Shared signed scale; nothing clipped. Baseline width 3.2 px, height 19.75 px, center (14.5, 14.5) px, lean 0°.',fontsize=9,color='#555')
    fig.savefig(OUT/'vector_variation.png',dpi=120);plt.close(fig)
    png=base64.b64encode((OUT/'vector_variation.png').read_bytes()).decode()
    html='''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>A vector describing change in lean</title><style>body{max-width:1080px;margin:0 auto;padding:28px;color:#253746;background:#fbfcfa;font:17px/1.6 system-ui}h1{font-size:32px;line-height:1.2}.eq{padding:22px;margin:24px 0;background:#edf2f3;border-left:4px solid #31758e}math{font-size:25px}img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%}td,th{padding:10px;text-align:left;border-bottom:1px solid #cfd9dc}.limit{background:#f8eedf;padding:15px}code{font-size:16px}a{color:#236a91}</style></head><body>
<h1>A vector describing how the lean vector changes</h1>
<p>This experiment predicts pixel vectors from pixel observations. It does not use stroke edges, recover horizontal or vertical centers, or reconstruct the knob code.</p>
<div class="eq"><math display="block"><mrow><mover><mi>V</mi><mo>^</mo></mover><mo>(</mo><mi>x</mi><mo>)</mo><mo>=</mo><msub><mi>v</mi><mn>0</mn></msub><mo>+</mo><mi>α</mi><mo>(</mo><mi>q</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>)</mo><mi>b</mi></mrow></math><math display="block"><mrow><mi>q</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>=</mo><mo>⟨</mo><mi>a</mi><mo>,</mo><mi>x</mi><mo>−</mo><msub><mi>x</mi><mn>0</mn></msub><mo>⟩</mo></mrow></math></div>
<p><strong>x</strong> is the input image as a vector of 784 intensities. <strong>x₀</strong> is the reference image. <strong>v₀</strong> is its observed lean vector. <strong>b</strong> is the difference between two observed lean vectors at different widths. <strong>a</strong> is learned from image differences; its dot product supplies a scalar position along the width observation curve. <strong>α</strong> is learned from measured lean-vector changes.</p>
<p>This is a line in the space of lean vectors: start at v₀ and move along b. The coefficient depends on the image. A nonlinear curve can require several such vectors:</p>
<div class="eq"><math display="block"><mrow><mover><mi>V</mi><mo>^</mo></mover><mo>(</mo><mi>x</mi><mo>)</mo><mo>=</mo><msub><mi>v</mi><mn>0</mn></msub><mo>+</mo><munderover><mo>∑</mo><mrow><mi>j</mi><mo>=</mo><mn>1</mn></mrow><mi>r</mi></munderover><msub><mi>α</mi><mi>j</mi></msub><mo>(</mo><mi>q</mi><mo>(</mo><mi>x</mi><mo>)</mo><mo>)</mo><msub><mi>b</mi><mi>j</mi></msub></mrow></math></div>
<img alt="Images, reference lean vector, its change-vector, and a prediction at an unseen width" src="data:image/png;base64,''' + png + '''">
<p>At 1,000 unseen random widths, one change-vector gives 0.039% mean vector error. Testing the transition more densely exposes a 14.70% worst error. Sixteen learned variation vectors reduce worst transition error to 0.376% in an independent confirmation. These scores compare observed finite-difference arrows, not exact derivatives.</p>
<p class="limit"><strong>Joint prediction remains unsolved.</strong> With independent observation curves only, adding their vector changes gives 154.5% mean error across new five-knob states even when supplied the known applied-change labels. Learning coordinates from the image does not remove those interactions.</p>
<p>Mathematically, a width/height interaction can contribute <code>s × t × b_wh</code>. That vector is invisible on both single-axis curves because either s or t is zero there. Joint measurements or an additional structural assumption are needed to identify it.</p>
<p><a href="../README.md">Full experiment report</a>. This width-slice model is empirical and approximate. It is not a decoded whole manifold and is not verified as a field that can be integrated through arbitrary lean paths.</p></body></html>'''
    # The self-contained artifact remains usable without a server.
    (OUT/'viewer.html').write_text(html)

if __name__=='__main__':main()
