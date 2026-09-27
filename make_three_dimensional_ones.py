"""A three-control image manifold with static and interactive 3D views.

Run: python make_three_dimensional_ones.py
Outputs: figures/three_dimensional_ones.png and figures/three_dimensional_ones.html
"""

from base64 import b64encode
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import Isomap, MDS
import plotly.graph_objects as go

from three_dimensional_ones_volume import (
    FACE_COLORS, SLICE_COLOR, model_interpolators, parameter_sheet,
    sheet_xyz, write_volume_html,
)


OUT = Path("figures")
OUT.mkdir(exist_ok=True)
SUB = 16
N = 28
CX = CY = 14.5
HEIGHT = 19.75
LEAN = np.linspace(-3.0, 3.0, 9)
BOW = np.linspace(-2.0, 2.0, 9)
WIDTH = np.linspace(1.8, 4.2, 5)
COLOR = {"ink": "#26333c", "blue": "#22658f", "orange": "#cb7927"}


def render(lean, bow, width):
    """Coverage image of a vertical stroke whose center bows sideways."""
    y_lo = np.arange(N * SUB, dtype=float) / SUB
    y_mid = y_lo + .5 / SUB
    z = (y_mid - CY) / (HEIGHT / 2)
    vertical = np.clip(
        (np.minimum(y_lo + 1 / SUB, CY + HEIGHT / 2)
         - np.maximum(y_lo, CY - HEIGHT / 2)) * SUB, 0, 1
    )
    center = CX - lean * z + bow * (1 - z * z)
    left, right = center - width / 2, center + width / 2
    columns = np.arange(N, dtype=float)
    overlap = np.clip(
        np.minimum(right[:, None], columns[None, :] + 1)
        - np.maximum(left[:, None], columns[None, :]), 0, 1
    )
    return (overlap * vertical[:, None]).reshape(N, SUB, N).mean(axis=1)


def sample_family():
    settings = np.array([(a, b, w) for w in WIDTH for b in BOW for a in LEAN])
    images = np.array([render(*q) for q in settings])
    return settings, images


def grid_edges():
    edges = []
    for wi in range(len(WIDTH)):
        for bi in range(len(BOW)):
            for ai in range(len(LEAN)):
                i = (wi * len(BOW) + bi) * len(LEAN) + ai
                if ai + 1 < len(LEAN):
                    edges.append((i, i + 1))
                if bi + 1 < len(BOW):
                    edges.append((i, i + len(LEAN)))
                if wi + 1 < len(WIDTH):
                    edges.append((i, i + len(BOW) * len(LEAN)))
    return np.array(edges)


def grid_curves():
    curves = []
    for wi in (0, 2, 4):
        for bi in (0, 4, 8):
            curves.append([(wi * 9 + bi) * 9 + ai for ai in range(9)])
        for ai in (0, 4, 8):
            curves.append([(wi * 9 + bi) * 9 + ai for bi in range(9)])
    for bi, ai in ((0, 0), (0, 8), (8, 0), (8, 8), (4, 4)):
        curves.append([(wi * 9 + bi) * 9 + ai for wi in range(5)])
    return curves


def manifold_view(settings, images):
    flat = images.reshape(len(images), -1)
    original = squareform(pdist(flat))
    raw_views = {
        "global": MDS(n_components=3, metric=True,
                      dissimilarity="precomputed", n_init=4,
                      random_state=9, max_iter=250,
                      eps=1e-5).fit_transform(original),
        "local": Isomap(n_neighbors=20, n_components=3).fit_transform(flat),
    }
    edges = grid_edges()
    local_true = original[edges[:, 0], edges[:, 1]]
    pairs = np.triu_indices(len(settings), 1)
    scaled_q = (settings - settings.mean(axis=0)) / settings.std(axis=0)
    views, errors = {}, {}
    for name, view in raw_views.items():
        # Rigid rotation only: align display axes loosely with the controls.
        rotation, _ = orthogonal_procrustes(view, scaled_q)
        views[name] = view @ rotation
        displayed = squareform(pdist(views[name]))
        local_shown = displayed[edges[:, 0], edges[:, 1]]
        local_error = np.abs(local_shown - local_true) / local_true
        all_error = np.abs(displayed[pairs] - original[pairs]) / original[pairs]
        errors[name] = {
            "local_median": float(np.median(local_error)),
            "local_p90": float(np.percentile(local_error, 90)),
            "all_median": float(np.median(all_error)),
            "all_p90": float(np.percentile(all_error, 90)),
        }
    return views, errors


def sample_index(lean, bow, width):
    ai = int(np.argmin(np.abs(LEAN - lean)))
    bi = int(np.argmin(np.abs(BOW - bow)))
    wi = int(np.argmin(np.abs(WIDTH - width)))
    return (wi * 9 + bi) * 9 + ai


def examples(images):
    landmark = {
        "A": sample_index(0, 0, 3.0),
        "B": sample_index(3, -2, 4.2),
        "C": sample_index(-3, 2, 1.8),
    }
    y_mid = (np.arange(N * SUB, dtype=float) + .5) / SUB
    z = (y_mid - CY) / (HEIGHT / 2)
    off = render(0, 0, 3 + .8 * z)  # Plausible taper, impossible with fixed width.
    return landmark, off


def local_singular_values():
    q = np.array([.37, .21, 3.07])
    step = 1e-4
    changes = []
    for j in range(3):
        plus, minus = q.copy(), q.copy()
        plus[j] += step
        minus[j] -= step
        changes.append(((render(*plus) - render(*minus)) / (2 * step)).ravel())
    return np.linalg.svd(np.array(changes), compute_uv=False)


def static_figure(settings, images, views, errors, landmark, off):
    fig = plt.figure(figsize=(17.5, 11.8), facecolor="white")
    fig.suptitle("A 3D test manifold made from bowed, single-stroke 1s",
                 fontsize=19, fontweight="bold", y=.989, color=COLOR["ink"])
    fig.text(.5, .958,
             "A continuous three-control generator defines every 28×28 image. "
             "These 3D sheets show its six parameter boundaries and a middle cut.",
             ha="center", fontsize=10.4, color="#50606a")
    gs = fig.add_gridspec(2, 4, left=.055, right=.97, top=.9, bottom=.13,
                          height_ratios=[.69, 1.46], hspace=.28, wspace=.21)
    imgs = [("A · straight center\nlean 0, bow 0, width 3.0", images[landmark["A"]]),
            ("B · thick, tilted, bowed\nlean 3, bow −2, width 4.2", images[landmark["B"]]),
            ("C · thin, oppositely bowed\nlean −3, bow 2, width 1.8", images[landmark["C"]]),
            ("OFF manifold · tapered stroke\nrow-by-row width changes", off)]
    for col, (title, pixels) in enumerate(imgs):
        ax = fig.add_subplot(gs[0, col])
        ax.imshow(pixels, cmap="gray_r", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_title(title, fontsize=10.3, pad=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.text(.5, .652,
             "On manifold = every pixel matches the stroke-coverage equation for one allowed "
             "(lean, bow, width). The tapered stroke looks like a 1 but cannot have one fixed width.",
             ha="center", fontsize=10, color=COLOR["ink"])
    interpolators = model_interpolators(views, LEAN, BOW, WIDTH)
    faces = [("lean", LEAN[0], "lean low"),
             ("lean", LEAN[-1], "lean high"),
             ("bow", BOW[0], "bow low"),
             ("bow", BOW[-1], "bow high"),
             ("width", WIDTH[0], "width low"),
             ("width", WIDTH[-1], "width high")]
    for j, (layout_name, angle) in enumerate((("global", -65), ("local", 34))):
        view = views[layout_name]
        interpolator = interpolators[layout_name]
        ax = fig.add_subplot(gs[1, j * 2:(j + 1) * 2], projection="3d")
        for kind, value, color_name in faces:
            xyz = sheet_xyz(interpolator, parameter_sheet(kind, value))
            ax.plot_surface(xyz[..., 0], xyz[..., 1], xyz[..., 2],
                            color=FACE_COLORS[color_name], alpha=.23,
                            linewidth=0, shade=False)
        xyz = sheet_xyz(interpolator, parameter_sheet("width", 3.0))
        ax.plot_surface(xyz[..., 0], xyz[..., 1], xyz[..., 2],
                        color=SLICE_COLOR, alpha=.65, linewidth=0,
                        shade=False)
        for landmark_name, index in landmark.items():
            pt = view[index]
            ax.scatter(*pt, s=110, color=COLOR["orange"],
                       edgecolor="white", linewidth=.9, depthshade=False)
            ax.text(*(pt + np.array([.22, .22, .22])), landmark_name,
                    fontsize=12, fontweight="bold", color=COLOR["ink"])
        ax.view_init(elev=20, azim=angle)
        ax.set_box_aspect(np.ptp(view, axis=0))
        ax.set(xlabel="3D view x", ylabel="3D view y", zlabel="3D view z")
        ax.tick_params(labelsize=7)
        title = ("Global pixel-distance fit" if layout_name == "global"
                 else "Local neighbor-distance unfolding")
        ax.set_title(title + " · continuous sheets\n"
                     f"neighbor error {errors[layout_name]['local_median']:.0%}, "
                     f"all-pair error {errors[layout_name]['all_median']:.0%} (median)",
                     fontsize=11.3, pad=13)
    fig.text(.5, .078,
             "Blue sheets: lean limits · teal: bow limits · gold: width limits · red: interior width = 3. "
             "Left fits all distances; right unfolds local neighbors.",
             ha="center", fontsize=9.7, color=COLOR["ink"])
    fig.text(.5, .046,
             "An interior image has three independent local knob directions (Jacobian rank 3). "
             "Both plots are approximate; neither is a unique true 3D shape.",
             ha="center", fontsize=9.5, color="#50606a")
    fig.savefig(OUT / "three_dimensional_ones.png", dpi=160,
                facecolor="white")
    plt.close(fig)


def png_url(pixels):
    bytes_io = BytesIO()
    Image.fromarray(np.uint8(np.round((1 - pixels) * 255)), mode="L").save(
        bytes_io, format="PNG"
    )
    return "data:image/png;base64," + b64encode(bytes_io.getvalue()).decode("ascii")


def interactive_figure(settings, images, views, errors, landmark, off):
    fig = go.Figure()
    traces_per_view = len(WIDTH) + len(grid_curves())
    for name in ("global", "local"):
        view = views[name]
        shown = name == "global"
        for wi, width in enumerate(WIDTH):
            ids = np.arange(wi * 81, (wi + 1) * 81)
            color = plt.cm.viridis(wi / 4)
            rgb = (f"rgb({int(color[0]*255)},"
                   f"{int(color[1]*255)},{int(color[2]*255)})")
            fig.add_trace(go.Scatter3d(
                x=view[ids, 0], y=view[ids, 1], z=view[ids, 2],
                mode="markers", name=f"width {width:.1f}",
                marker=dict(size=4, color=rgb, opacity=.82),
                customdata=np.column_stack((ids, settings[ids])),
                hovertemplate=("lean %{customdata[1]:.2f} px"
                               "<br>bow %{customdata[2]:.2f} px"
                               "<br>width %{customdata[3]:.2f} px<extra></extra>"),
                visible=shown,
            ))
        for curve in grid_curves():
            xyz = view[curve]
            fig.add_trace(go.Scatter3d(
                x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                mode="lines", line=dict(color="rgba(70,84,97,.30)", width=2),
                hoverinfo="skip", showlegend=False, visible=shown,
            ))
    global_visible = [True] * traces_per_view + [False] * traces_per_view
    local_visible = [False] * traces_per_view + [True] * traces_per_view
    fig.update_layout(
        scene=dict(xaxis_title="display x", yaxis_title="display y",
                   zaxis_title="display z", aspectmode="data"),
        margin=dict(l=0, r=0, t=55, b=0),
        legend=dict(orientation="h", y=1.01, x=.02),
        updatemenus=[dict(type="buttons", x=.5, y=1.11, xanchor="center",
                          buttons=[dict(label="Fit all pixel distances",
                                        method="update",
                                        args=[{"visible": global_visible}]),
                                   dict(label="Unfold local neighbors",
                                        method="update",
                                        args=[{"visible": local_visible}])])],
        paper_bgcolor="white",
    )
    plot_html = fig.to_html(full_html=False, include_plotlyjs=True,
                            div_id="manifold-plot")
    image_urls = [png_url(img) for img in images]
    off_url = png_url(off)
    state_data = [[float(v) for v in row] for row in settings]
    import json

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Three-control bowed 1 manifold</title>
<style>
body {{margin:0;font:16px system-ui,sans-serif;color:#26333c;background:#f7f8fa}}
header {{padding:18px 25px;background:white;border-bottom:1px solid #d9e0e5}}
h1 {{font-size:23px;margin:0 0 7px}}p {{line-height:1.43;margin:7px 0}}
main {{display:grid;grid-template-columns:minmax(500px,3fr) minmax(260px,1fr);gap:14px;padding:15px}}
#manifold-plot {{height:74vh;min-height:540px;background:white;border:1px solid #d9e0e5}}
aside {{background:white;border:1px solid #d9e0e5;padding:17px}}
aside img {{width:210px;height:210px;image-rendering:pixelated;border:1px solid #e2e6e9}}
.muted {{color:#55636c}}.metric {{background:#eef3f6;padding:12px;margin-top:15px}}
@media(max-width:850px){{main{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>What does a three-control image family look like in 3D?</h1>
<p>405 exact generated images. Lean, bow, and stroke width change; center and height stay fixed.
Switch between two 3D placements, rotate the plot, click a dot to see its image,
and click a width in the legend to hide/show its layer.</p></header>
<main><div>{plot_html}</div><aside>
<h2>Selected generated 1</h2><img id="selected-image" alt="selected generated 1">
<p id="selected-coordinates"></p>
<p class="muted">Black is full ink coverage; white is zero. Every dot is on the manifold by the same coverage equation. Gray lines connect nearby knob settings.</p>
<div class="metric"><strong>What the 3D positions mean</strong>
<p>The first placement fits all pairwise 784-pixel distances. The second unfolds nearby images. The display axes have no direct physical meaning.</p>
<p><strong>All-distance fit:</strong> median neighbor error {errors['global']['local_median']:.0%}; all-pair error {errors['global']['all_median']:.0%}.</p>
<p><strong>Local unfolding:</strong> median neighbor error {errors['local']['local_median']:.0%}; all-pair error {errors['local']['all_median']:.0%}.</p>
<p>Both 3D views distort distances, so neither is an exact embedding.</p></div>
<h3>Outside the manifold</h3><img src="{off_url}" alt="plausible tapered 1">
<p class="muted">This stroke is tapered: its row ink mass changes with height. Every allowed generated image has one constant width, so no control setting produces it.</p>
</aside></main><script>
const imageUrls = {json.dumps(image_urls)};
const states = {json.dumps(state_data)};
function showState(i) {{
  document.getElementById('selected-image').src = imageUrls[i];
  const q = states[i];
  document.getElementById('selected-coordinates').textContent =
    `lean ${{q[0].toFixed(2)}} px · bow ${{q[1].toFixed(2)}} px · width ${{q[2].toFixed(2)}} px`;
}}
showState({landmark['A']});
document.getElementById('manifold-plot').on('plotly_click', event => {{
  const data = event.points[0].customdata;
  const i = data && data[0];
  if (Number.isInteger(i)) showState(i);
}});
</script></body></html>"""
    (OUT / "three_dimensional_ones.html").write_text(html, encoding="utf-8")


def main():
    settings, images = sample_family()
    views, errors = manifold_view(settings, images)
    landmark, off = examples(images)
    static_figure(settings, images, views, errors, landmark, off)
    write_volume_html(OUT / "three_dimensional_ones.html", views, errors,
                      LEAN, BOW, WIDTH)
    print("states", len(settings), "local Jacobian singular values",
          np.round(local_singular_values(), 4))
    print("3D embedding relative distance errors:", errors)
    print("outputs: figures/three_dimensional_ones.png and .html")


if __name__ == "__main__":
    main()
