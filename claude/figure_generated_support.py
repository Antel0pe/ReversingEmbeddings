"""Explanatory image of exact pixel support for the generated-one family."""

from itertools import product
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_global_span import coverage64, possible_ink_mask
from grey_ones import RANGES


def main():
    sample = np.array([14.43, 14.62, 19.77, 3.2, 17.3])
    baseline = coverage64(sample)[0].reshape(28, 28)
    rng = np.random.default_rng(20260930)
    unit = np.concatenate([
        rng.random((4000, 5)),
        np.array(list(product(np.linspace(0, 1, 5), repeat=5))),
        np.array(list(product([0.001, 0.5, 0.999], repeat=5))),
    ])
    parameters = RANGES[:, 0] + unit * (RANGES[:, 1] - RANGES[:, 0])
    sampled = (np.max(coverage64(parameters), axis=0) > 0).reshape(28, 28)
    exact = possible_ink_mask().reshape(28, 28)
    assert np.array_equal(sampled, exact)

    fig = plt.figure(figsize=(13.2, 5.6), dpi=170, facecolor="white")
    fig.text(.5, .955, "Which pixels can a generated 1 ever occupy?", ha="center", va="top",
             fontsize=17, weight="bold", color="#20303a")
    fig.text(.5, .895, "Controlled five-knob family · 28 × 28 coverage pixels · all five settings vary in the two right panels",
             ha="center", va="top", fontsize=10, color="#52616b")
    positions=[(.065,.245,.245,.55),(.38,.245,.245,.55),(.695,.245,.245,.55)]
    titles=["One starting image", "Observed across 7,368 settings", "Exact allowed-family support"]
    notes=["center 14.43, 14.62 · height 19.77\nwidth 3.20 · lean 17.3°",
           "283 pixels receive ink in this sample",
           "283 possible · 501 always zero"]
    for i,(x,y,w,h) in enumerate(positions):
        ax=fig.add_axes([x,y,w,h])
        if i==0:
            ax.imshow(baseline,cmap="Greys",vmin=0,vmax=1,interpolation="nearest")
        else:
            ax.imshow([sampled,exact][i-1],cmap=ListedColormap(["#f1f3f4","#278b91"]),vmin=0,vmax=1,
                      interpolation="nearest")
        ax.set_xlim(-.5,27.5);ax.set_ylim(27.5,-.5)
        ax.set_xticks([0,7,14,21,27]);ax.set_yticks([0,7,14,21,27])
        ax.set_xlabel("pixel column",fontsize=9);ax.set_ylabel("pixel row",fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_title(titles[i],fontsize=11.3,pad=9,color="#20303a")
        fig.text(x+w/2,.177,notes[i],ha="center",va="top",fontsize=9,color="#31444f")
    fig.text(.5,.07,"Left: black = full ink coverage, white = none. Right: teal = possible ink, white = always zero.",
             ha="center",va="bottom",fontsize=9.8,color="#20303a")
    fig.text(.5,.035,"The exact mask is computed from stroke and cap bounds; agreement with the sampled mask is a check, not the proof.",
             ha="center",va="bottom",fontsize=8.6,color="#52616b")
    destination=Path(__file__).with_name("generated_support.png")
    fig.savefig(destination,facecolor="white")
    plt.close(fig)
    print(destination)


if __name__=="__main__":
    main()
