"""Attach every original pixel coordinate to each vertex of the unwrapped sheet.

Run from the repository root:
    OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/inspect_original_sheet_coordinates.py
"""
import base64
import gzip
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/original-sheet-coordinate-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Rectangle
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from grey_ones import render

OUT=ROOT/"figures/generated_one_width_vectors_by_lean/original_coordinates"
SOURCE=OUT.parent/"sheet_views/sheet_data.npz"


def triangles(nt,nw):
    cells=[]
    for t in range(nt-1):
        for w in range(nw-1):
            a=t*nw+w;b=a+1;c=a+nw;d=c+1
            cells.extend([[a,b,d],[a,d,c]])
    return np.array(cells,dtype=np.int32)


def figure(images,coords,faces,root):
    fig=plt.figure(figsize=(15,7.6),facecolor="white")
    fig.text(.04,.95,"Unwrap the sheet, keep every original pixel coordinate attached",fontsize=19,weight="bold")
    fig.text(.04,.887,"The same 2D positions appear in both maps. Color reads a different original coordinate at each image on the sheet.\n"
             "Each 784D coordinate is a pixel's ink coverage (0–1). Map axes locate the image on the unwrapped sheet.",fontsize=11,linespacing=1.5)
    ax=fig.add_axes([.045,.36,.16,.36])
    ax.imshow(images[root].reshape(28,28),cmap="gray_r",vmin=0,vmax=1,interpolation="nearest")
    chosen=[155,376]
    for k,c in zip(chosen,["#c94732","#d19b17"]):
        r,col=divmod(k,28)
        ax.add_patch(Rectangle((col-.5,r-.5),1,1,fill=False,edgecolor=c,lw=2))
    ax.set_title("Selected original image\nRoot: width 3.2, lean 0°",fontsize=11,pad=12)
    ax.set_xticks([]);ax.set_yticks([])
    ax.text(.5,-.13,"White = 0; black = 1 ink.\nIts full 784 values stay stored.",ha="center",va="top",transform=ax.transAxes,fontsize=10,linespacing=1.5)
    mesh=mtri.Triangulation(coords[:,0],coords[:,1],faces)
    for left,k,c in zip([.28,.62],chosen,["#c94732","#d19b17"]):
        ax=fig.add_axes([left,.36,.27,.36])
        image=ax.tripcolor(mesh,images[:,k],shading="gouraud",cmap="viridis",vmin=0,vmax=1)
        contour=ax.tricontour(mesh,images[:,k],levels=[.25,.5,.75],colors="white",linewidths=.8)
        ax.clabel(contour,fmt="%.2f",fontsize=8)
        ax.scatter(*coords[root],s=55,facecolor="none",edgecolor="#d94b3e",lw=2,zorder=5)
        ax.set_aspect("equal",adjustable="box")
        ax.set(xlabel="Unwrapped coordinate 1",ylabel="Unwrapped coordinate 2")
        r,col=divmod(k,28)
        ax.set_title(f"Original coordinate #{k}\npixel row {r}, column {col}",fontsize=11,color=c,pad=12)
        fig.text(left,.275,f"At the marked root: coordinate #{k} = {images[root,k]:.6f}",fontsize=10)
    cax=fig.add_axes([.923,.36,.013,.36])
    fig.colorbar(image,cax=cax,label="Original pixel coverage (0–1)")
    fig.text(.04,.17,"The address has two map numbers. The data attached to that address still has all 784 original numbers.\n"
             "Two selected coordinate fields are shown here; the interactive inspector can select any pixel and read the complete vector.\n"
             "A coordinate can stay at 0 or 1 over a whole region while other coordinates change. Matching one value does not mean matching images.",fontsize=10.5,linespacing=1.6)
    fig.text(.04,.055,"1,334 exact sampled images: 29 widths × 46 leans; center (14,14), height 19.75 px fixed. Indices start at zero; index = 28 × row + column.\n"
             "Positions reuse the graph-distance unfolding. Positions and interior color interpolation are approximate; stored vertex values are unchanged float32.\n"
             "The layout can distort original pixel distances. It does not replace the original coordinates or provide a decoder between sampled vertices.",fontsize=10,linespacing=1.5)
    fig.savefig(OUT/"09_original_coordinates_on_sheet.png",dpi=150)
    fig.savefig(OUT/"09_original_coordinates_on_sheet.svg")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    a=np.load(SOURCE)
    widths,leans=a["widths"],a["leans"]
    images=a["images"].reshape(-1,784).astype("<f4")
    coords=a["unfold2"]
    root=int(a["root_index"])
    nt,nw=len(leans),len(widths)
    faces=triangles(nt,nw)
    # Check every attached image against its actual renderer settings.
    p=np.tile([14.,14.,19.75,3.2,0.],(nt,nw,1))
    p[:,:,3],p[:,:,4]=widths,leans[:,None]
    direct=render(p.reshape(-1,5)).reshape(-1,784)
    assert np.array_equal(images,direct)
    raw=images.tobytes()
    compressed=gzip.compress(raw)
    assert gzip.decompress(compressed)==raw
    assert len(np.unique(coords,axis=0))==len(images)
    variation=np.ptp(images.astype(float),axis=0)
    summary=dict(vertices=len(images),original_coordinates_per_vertex=784,
                 original_scalar_values_retained=int(images.size),
                 width_count=nw,lean_count=nt,fixed=dict(cx=14,cy=14,height=19.75),
                 varying_coordinates=int(np.sum(variation>0)),
                 constant_coordinates=int(np.sum(variation==0)),
                 coordinate_meaning="coverage at one original pixel; values from 0 to 1",
                 indexing="zero-based row-major: coordinate index = 28*row + column",
                 geometry="Existing graph-distance unfolding positions; axes are addresses on the map, not individual original pixel axes.",
                 checks=dict(all_stored_images_bit_identical_to_renderer=True,
                             transport_roundtrip_bit_identical=True,
                             distinct_map_addresses_for_all_vertices=True),
                 limitations=["A finite sampled sheet, not the full continuous surface.",
                              "Unwrapped distances can differ from original image distances.",
                              "Between sampled vertices the coordinate-field colors and contours are interpolated.",
                              "All original values are retained at vertices. This is data attachment, not a claim of a continuous learned decoder."],
                 understanding_check=dict(object="One unwrapped sheet with actual original pixel values attached at every sampled vertex",
                                          axes="Map addresses versus original pixel-coordinate values are distinguished explicitly",
                                          baseline="Root image and its two selected coordinate values are marked",
                                          evidence="Two original coordinate fields on the same addresses; interactive full-vector inspection",
                                          scope="Bit-exact source values at finite vertices; approximate map and between-vertex interpolation"))
    payload=dict(widths=widths.tolist(),leans=leans.tolist(),coords=coords.tolist(),faces=faces.tolist(),
                 root=root,summary=summary,images=base64.b64encode(compressed).decode())
    (OUT/"results.json").write_text(json.dumps(summary,indent=2)+"\n")
    (OUT/"index.html").write_text((ROOT/"experiments/inspect_original_sheet_coordinates.html").read_text().replace("__ORIGINAL_COORDINATES__",json.dumps(payload,separators=(",",":"))))
    figure(images,coords,faces,root)
    (OUT/"README.md").write_text("""# Original 784D values attached to the unwrapped sheet

Run `OPENBLAS_NUM_THREADS=1 .venv/bin/python experiments/inspect_original_sheet_coordinates.py` from the repository root, after generating `sheet_views/sheet_data.npz` with the preceding sheet experiment.

Each of 1,334 map vertices stores its full original 784-value float32 coverage image, without projecting or rounding those values.
The two map numbers are an address for a point on the sheet. The 784 original numbers are the vector attached to that address.
All arrays are checked bit-for-bit against `grey_ones.render` at their corresponding width/lean settings.
The gzip/base64 transfer is checked for a byte-identical round trip.
The displayed unwrapped positions come from the earlier local-graph-distance MDS map. Their distance distortion and discretization checks are in `sheet_views/results.json`.

The inspector shows the full coordinate array as a 28 by 28 numeric table, supports selecting any of the 784 pixel coordinates, and exports the complete unrounded vector for any selected point as JSON.
An original coordinate means a pixel's coverage value in [0,1], not the row or column number of that pixel.
Indices start at zero, with `index = 28 * row + column`.
The map color and contour lines show the selected original coordinate as a function over the sheet.
Between sampled vertices those fields are linearly interpolated; this is not an exact continuous decoder.
The full-vector display can be rounded for readability. The selected-coordinate readout and export use the original unrounded numbers.

Nothing about storing the 784 values makes the layout distance preserving. Distinct images can share the same value of one original coordinate; the full vector distinguishes them.
These statements apply to this fixed-center, fixed-height finite sample of the width/lean family.
""")
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    main()
