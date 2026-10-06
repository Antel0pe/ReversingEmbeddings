"""Show direct knob-curve constructions and their additive restriction."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_direct_knob_curve_limits import OUT
from make_generated_one_joint_curve_view import ink


def main():
    r=json.loads((OUT/'results.json').read_text())
    learned=json.loads((OUT.parent/'joint/results.json').read_text())
    best=learned['candidates']['joint_513']['metrics']['explained_percent']
    s=np.load(OUT/'interaction_witness.npz')
    fig=plt.figure(figsize=(13.4,12),facecolor='white')
    fig.text(.055,.97,'Five exact knob coordinates do not imply five additive curves',fontsize=19,weight='bold',va='top')
    fig.text(.055,.925,'Controlled generated 1s: every image is 28 × 28 ink coverage. White = no ink; black = full ink.',fontsize=10.5)
    ax=fig.add_axes([.09,.68,.83,.18])
    scores=[v['metrics']['explained_percent'] for v in r['finite_cloud_fits']]+[best]
    ax.bar(range(5),scores,color=['#bd5a22']*4+['#2563a6'],width=.6)
    ax.set_ylim(0,109);ax.set_ylabel('Variation captured (%)',fontsize=10)
    ax.set_xticks(range(5),['Knob-indexed\n65 nodes','Knob-indexed\n129 nodes','Knob-indexed\n257 nodes','Knob-indexed\n513 nodes','Free positions\n513 nodes'],fontsize=10)
    for i,v in enumerate(scores):ax.text(i,v+2,f'{v:.2f}%',ha='center',fontsize=11,weight='bold')
    ax.spines[['top','right']].set_visible(False)
    fig.text(.09,.626,'All bars: same 14,389 images and five curves. Orange: each true knob sets one curve position. Blue: positions may mix knobs.',fontsize=9.8)
    gs=fig.add_gridspec(3,4,left=.19,right=.94,top=.57,bottom=.13,hspace=.34,wspace=.30)
    titles=['Starting image','Width +1 px','Lean +15°','Both changes together']
    for row,(key,label) in enumerate([('images','Exact source'),('direct','Knob-indexed\n513-node sum'),('free','Free positions\n513-node sum')]):
        fig.text(.045,.504-row*.149,label,fontsize=10.5,weight='bold',va='center',linespacing=1.5)
        for j,im in enumerate(s[key]):
            error=np.linalg.norm(im-s['images'][j])
            ink(fig.add_subplot(gs[row,j]),im,titles[j] if row==0 else '',f'Image L2 error {error:.3f}' if row else '')
    fig.text(.055,.072,'Other knobs stay at their midpoints. Start: width 3.2 px, lean 12.5°. Magenta tint marks fitted values outside [0, 1].',fontsize=10)
    fig.text(.055,.041,'Proof witness: image(both) − image(width only) − image(lean only) + image(start) has L2 = 3.460, not zero.\n'
             'Every additive knob-indexed model gives zero for that difference. At least one of these four images must have L2 error ≥ 0.865.',fontsize=10,linespacing=1.6)
    fig.savefig(OUT/'direct_comparison.png',dpi=140);plt.close(fig)
    grids=r['product_grid_optima']
    lines=['# Directly constructed knob curves','',
        'Assumption: the position on curve k is the actual value of knob k, normalized to its allowed range. The decoder is a fitted base plus five independently indexed pixel-space curves. Curve positions are not allowed to mix knobs.','',
        '## Same-cloud comparison','',
        '| Nodes per knob curve | Globally fitted fixed-coordinate capture |','| ---: | ---: |']
    lines += [f'| {v["nodes_per_curve"]} | {v["metrics"]["explained_percent"]:.6f}% |' for v in r['finite_cloud_fits']]
    lines += ['',f'The freely fitted five-curve model with 513 nodes captures {best:.6f}% on the same 14,389 images. Direct 513-node curves trail it by {best-scores[3]:.6f} percentage points. The direct solve is linear least squares with a tiny numerical ridge; it does not suffer from curve-initialization local minima.','',
        '## Global additive optimum on complete grids','',
        'For every value of each knob, average the images across all combinations of the other four knobs. Subtract the overall mean. Those five lists of pixel-valued effects are the globally best additive curves on an equally weighted Cartesian grid. No iterative curve search or imposed smoothness is needed. Interpolating between those lists produces ordinary polylines.','',
        'Because the grid is a product distribution, the five centered effects are mutually orthogonal. The unexplained residual is orthogonal to every single-knob function. This proves optimality over all additive knob functions on the grid, not just the chosen spline basis.','',
        '| Values per knob | All joint images | Global optimal capture |','| ---: | ---: | ---: |']
    lines += [f'| {v["levels_per_knob"]} | {v["images"]:,} | {v["globally_optimal_knob_additive_capture_percent"]:.6f}% |' for v in grids]
    lines += ['', 'Levels are the midpoints of equal intervals across each knob range. The grid optima are exact for those discrete distributions. Approximately 81.4% is a converging numerical estimate for the continuous independent-uniform box, not an exact universal percentage ceiling. Grid and pooled-cloud percentages use different distributions.','',
        '## Why infinitely fine knob curves cannot make the model exact','',
        'For any additive model, changing width and lean has zero mixed difference:','',
        '```text\nmodel(both) - model(width only) - model(lean only) + model(start) = 0\n```','',
        'The actual renderer gives mixed-difference L2 3.460099700 and maximum pixel interaction 0.500000030 for width +1 px and lean +15 degrees from the midpoint settings. The triangle inequality forces at least one of these four images to have image L2 reconstruction error at least 0.865024925 in every additive knob-indexed model. This conclusion is independent of node count, curve smoothness, or fitting algorithm.','',
        '## Scope of the conclusion','',
        'This impossibility result does not cover the learned free-coordinate curves: their positions can change together when a physical knob changes. Their observed scores remain 99.733620% on the fitting cloud and 99.466107% on additional uniform points. Their distance from a perfect score is 0.266380 and 0.533893 percentage points respectively. There is no certified global optimum for the free-coordinate five-curve model.','',
        'A finite set can also be memorized by a polyline visiting every image, with the other four contributions zero. That establishes only finite-data feasibility and does not establish a useful smooth five-coordinate representation of the continuous manifold.','',
        'Raw results: results.json. Constructed product-grid nodes: product_5.npz, product_9.npz, product_13.npz. Comparison: direct_comparison.png.','']
    (OUT/'README.md').write_text('\n'.join(lines))
    rows=''.join(f'<tr><td>{v["levels_per_knob"]}</td><td>{v["images"]:,}</td><td>{v["globally_optimal_knob_additive_capture_percent"]:.3f}%</td></tr>' for v in grids)
    html=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Direct knob curves: what is possible?</title><style>body{{font:17px/1.65 system-ui;color:#182a3b;background:#f5f7fa;margin:0}}main{{max-width:1050px;margin:auto;padding:28px 24px}}section{{background:white;padding:24px;border-radius:14px;margin:24px 0}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #ddd}}a{{color:#2563a6}}code{{font-size:15px}}</style></head><body><main><a href="../joint/">← Free five-curve fit</a><h1>Construct curves directly from the knobs</h1><p>Each true knob value supplies the position on its corresponding curve. That makes a stricter model than the free fit.</p><section><h2>What happens when we add detail?</h2><img src="direct_comparison.png" alt="Same-cloud direct knob-curve fits compared with free curves, and four images proving the additive restriction"><p>The best direct 513-node fit captures {scores[3]:.3f}%; the free five-curve fit captures {best:.3f}%. These scores use the same 14,389-point cloud.</p></section><section><h2>Can we construct the global best?</h2><p>Yes, for independent knob-indexed curves on a complete grid. Average over the other four knobs at each value, then subtract the overall mean. This directly gives the optimal additive curves.</p><table><thead><tr><th>Values per knob</th><th>All joint images</th><th>Global best capture</th></tr></thead><tbody>{rows}</tbody></table><p>These are exact optima on the stated discrete grids. Around 81.4% is the observed limit as the uniform grid gets denser; it is not a proven exact percentage for the continuous box.</p></section><section><h2>Why finer curves cannot reach perfection</h2><p>Width changes the pixel pattern differently at different lean values. A single width-only contribution cannot express that dependence. Every additive knob-indexed model has zero mixed difference; the four rendered images above have mixed-difference L2 3.460.</p><p>This proves exact reconstruction is impossible under this restriction, even with infinitely many nodes. Free mixed coordinates avoid that restriction; the proof does not set their ceiling.</p></section><p><a href="README.md">Detailed method, proof, and limits</a> · <a href="results.json">Measured results</a></p></main></body></html>'''
    (OUT/'index.html').write_text(html)
    parent=OUT.parent/'joint/index.html';text=parent.read_text();marker='<!-- direct-construction-link -->'
    if marker not in text:
        text=text.replace('</main>',marker+'<section><h2>Construct curves from the exact knob values</h2><p><a href="../direct_construction/">Compare direct knob curves with the free fit, and see the representational limit</a>.</p></section></main>')
        parent.write_text(text)
    print('Created direct-construction figure and report.',flush=True)


if __name__=='__main__':main()
