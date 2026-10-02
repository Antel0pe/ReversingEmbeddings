"""Explain a height detour during a lean change, with a 100-case control."""

import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from explore_global_span import coverage64


def main():
    local = Path(__file__).resolve().parent
    example = json.loads((local / "lean_shortcut_results.json").read_text())[1]
    control = json.loads((local / "shortcut_refined_results.json").read_text())
    start = np.asarray(example["start"])
    end = np.asarray(example["end"])
    direct_mid = (start + end) / 2
    detour_mid = direct_mid.copy()
    detour_mid[2] -= example["sine_height_and_width"]["height_depth"]
    detour_mid[3] -= example["sine_height_and_width"]["width_depth"]
    examples = [start, direct_mid, detour_mid, end]
    images = coverage64(examples).reshape(4, 28, 28)
    gains = np.array([r["gain_percent_4096"] for r in control["records"]])

    fig = plt.figure(figsize=(15, 9), dpi=150, facecolor="white")
    fig.text(.5,.968,"Shortening the stroke makes this lean route shorter in image space",
             ha="center",va="top",fontsize=19,weight="bold",color="#213039")
    fig.text(.5,.922,"One exact generated-1 example above · 100 varied generated-1 controls below · all images use the same 0–1 coverage scale",
             ha="center",va="top",fontsize=10.5,color="#53636a")
    positions=[.045,.235,.425,.615]
    titles=["Start", "Direct route · halfway", "Shorten-and-lean · halfway", "Same endpoint"]
    notes=["height 20.50 · lean −10°", "height 20.50 · lean 12.5°",
           f"height {detour_mid[2]:.2f} · lean 12.5°", "height 20.50 · lean 35°"]
    for i,x in enumerate(positions):
        ax=fig.add_axes([x,.47,.16,.36])
        ax.imshow(images[i],cmap="Greys",vmin=0,vmax=1,interpolation="nearest")
        ax.set_xticks([0,14,27]);ax.set_yticks([0,14,27]);ax.tick_params(labelsize=8)
        ax.set(xlabel="pixel column",ylabel="pixel row")
        ax.set_title(titles[i],fontsize=10.3,pad=8,color="#213039")
        fig.text(x+.08,.445,notes[i],ha="center",va="top",fontsize=8.6,color="#384a53")
    ax=fig.add_axes([.825,.50,.13,.30])
    direct=example["lean_only_length_8192"]
    detour=example["sine_height_and_width"]["length_8192"]
    ax.bar([0,1],[direct,detour],color=["#667b86","#258b91"],width=.65)
    ax.set_ylim(0,34);ax.set_xticks([0,1],["direct","detour"]);ax.tick_params(labelsize=8.5)
    ax.set_ylabel("accumulated pixel L2",fontsize=8.5)
    for i,value in enumerate([direct,detour]):ax.text(i,value+.6,f"{value:.2f}",ha="center",fontsize=9)
    ax.set_title("Route length",fontsize=10.5,pad=8)
    fig.text(.89,.445,"5.05% shorter",ha="center",va="top",fontsize=9,color="#147178",weight="bold")

    ax=fig.add_axes([.15,.16,.70,.19])
    bins=np.arange(0,6.25,.5)
    ax.hist(gains,bins=bins,color="#a4cdd0",edgecolor="white")
    ax.axvline(0,color="#a74a41",linestyle="--",linewidth=1.2,label="no gain")
    ax.axvline(np.median(gains),color="#1e6871",linewidth=2,label=f"median {np.median(gains):.2f}%")
    ax.axvline(example["sine_height_and_width"]["gain_percent"],color="#1e6871",linestyle=":",linewidth=1.7,label="example 5.05%")
    ax.set(xlim=(-.25,6.25),xlabel="shortening of pixel-space path (%)",ylabel="number of controls")
    ax.set_title("Control: 100 other large lean changes, each allowed a simple height dip",fontsize=11,pad=7)
    ax.legend(loc="upper left",fontsize=8,ncol=3,frameon=False)
    fig.text(.5,.075,"Controls vary center, height, width and lean endpoints; starting lean is −10° to 0°, ending lean 25° to 35°.",
             ha="center",fontsize=9,color="#43545c")
    fig.text(.5,.046,"Height follows one smooth down-and-up sine wave. Gains use 4,096 rendered steps; direct and detour share endpoints.",
             ha="center",fontsize=9,color="#43545c")
    fig.text(.5,.017,"The bars compare lengths along valid image paths; neither bar is an endpoint distance or a proven geodesic.",
             ha="center",fontsize=8.4,color="#65727a")
    destination=local/"generated_lean_shortcut.png"
    fig.savefig(destination,facecolor="white")
    plt.close(fig)
    print(destination)


if __name__=="__main__":
    main()
