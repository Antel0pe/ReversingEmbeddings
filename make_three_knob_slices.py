"""Show three exact three-knob slices and their approximate pixel-distance layouts.

Run: python make_three_knob_slices.py
Outputs: figures/three_knob_slices.png and figures/three_knob_slices.html
"""

from base64 import b64encode
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from PIL import Image
from plotly.subplots import make_subplots
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist, squareform

from grey_ones import RANGES, render


OUT = Path("figures")
OUT.mkdir(exist_ok=True)
LEVELS = np.linspace(0, 1, 7)
CASES = [
    ("Position and thickness", (0, 1, 3), "Move the 1 across the image and widen it."),
    ("Position, thickness, and tilt", (0, 3, 4), "Move, widen, and lean the 1."),
    ("Size, thickness, and tilt", (2, 3, 4), "Grow, widen, and lean the 1."),
]
LABELS = ["Horizontal position", "Vertical position", "Stroke height", "Stroke width", "Lean angle"]
UNITS = ["px", "px", "px", "px", "°"]
COLORS = ["#1885a5", "#d57832", "#794baa"]


def sample_case(triplet):
    coordinates = np.array(np.meshgrid(LEVELS, LEVELS, LEVELS, indexing="ij"))
    coordinates = coordinates.reshape(3, -1).T
    knobs = np.tile(RANGES.mean(axis=1), (len(coordinates), 1))
    for local_axis, knob_index in enumerate(triplet):
        knobs[:, knob_index] = RANGES[knob_index, 0] + coordinates[:, local_axis] * np.ptp(RANGES[knob_index])
    images = render(knobs)
    return coordinates, knobs, images


def pixel_layout(coordinates, images):
    pixels = images.reshape(len(images), -1).astype(float)
    distances = squareform(pdist(pixels))
    squared = distances ** 2
    gram = -.5 * (squared - squared.mean(axis=0)[None, :]
                  - squared.mean(axis=1)[:, None] + squared.mean())
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    layout = eigenvectors[:, -3:] @ np.diag(np.sqrt(np.maximum(eigenvalues[-3:], 0)))
    rotation, _ = orthogonal_procrustes(layout, coordinates - coordinates.mean(axis=0))
    layout = layout @ rotation
    shown = squareform(pdist(layout))
    upper = np.triu_indices(len(images), 1)
    pair_error = np.abs(shown[upper] - distances[upper]) / distances[upper]
    # Three families of nearest-grid neighbors, with each undirected edge once.
    grid = np.arange(7 ** 3).reshape(7, 7, 7)
    edges = np.concatenate([
        np.stack((grid[:-1, :, :].ravel(), grid[1:, :, :].ravel()), axis=1),
        np.stack((grid[:, :-1, :].ravel(), grid[:, 1:, :].ravel()), axis=1),
        np.stack((grid[:, :, :-1].ravel(), grid[:, :, 1:].ravel()), axis=1),
    ])
    edge_error = np.abs(shown[edges[:, 0], edges[:, 1]] - distances[edges[:, 0], edges[:, 1]]) / distances[edges[:, 0], edges[:, 1]]
    return layout, {
        "pair_median": float(np.median(pair_error)),
        "pair_p90": float(np.percentile(pair_error, 90)),
        "neighbor_median": float(np.median(edge_error)),
        "neighbor_p90": float(np.percentile(edge_error, 90)),
    }


def paths():
    grid = np.arange(7 ** 3).reshape(7, 7, 7)
    for axis in range(3):
        other = [a for a in range(3) if a != axis]
        for p in (0, 3, 6):
            for q in (0, 3, 6):
                # Nine lines per knob: boundary, middle, and opposite boundary.
                selection = [slice(None)] * 3
                selection[other[0]] = p
                selection[other[1]] = q
                yield axis, grid[tuple(selection)].ravel()


def boundary_faces(points):
    cube = points.reshape(7, 7, 7, 3)
    for axis in range(3):
        for side in (0, -1):
            face = np.take(cube, side, axis=axis)
            vertices = face.reshape(-1, 3)
            triangles = []
            for i in range(6):
                for j in range(6):
                    a = i * 7 + j
                    triangles.extend(((a, a + 1, a + 7), (a + 1, a + 8, a + 7)))
            yield axis, side, vertices, np.asarray(triangles)


def image_url(image):
    buffer = BytesIO()
    Image.fromarray(np.uint8(np.round((1 - image) * 255)), mode="L").resize((140, 140), Image.Resampling.NEAREST).save(buffer, format="PNG")
    return "data:image/png;base64," + b64encode(buffer.getvalue()).decode("ascii")


def plotly_panel(fig, points, knobs, triplet, column, show_legend):
    for axis, side, vertices, triangles in boundary_faces(points):
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
            i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
            color=COLORS[axis], opacity=.15, hoverinfo="skip", showlegend=False,
            flatshading=True,
        ), row=1, col=column)
    for axis, ids in paths():
        x = points[ids]
        hover = ["<br>".join(f"{LABELS[k]}: {knobs[i, k]:.2f} {UNITS[k]}" for k in triplet)
                 for i in ids]
        fig.add_trace(go.Scatter3d(
            x=x[:, 0], y=x[:, 1], z=x[:, 2], mode="lines+markers",
            line=dict(color=COLORS[axis], width=5),
            marker=dict(color=COLORS[axis], size=2.5),
            name=LABELS[triplet[axis]], legendgroup=f"axis{axis}",
            showlegend=show_legend and np.array_equal(ids, next(ids0 for a, ids0 in paths() if a == axis)),
            text=hover, hovertemplate="%{text}<extra></extra>",
        ), row=1, col=column)


def interactive_case(title, description, triplet, exact, knobs, images, layout, errors):
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=("Exact 3D knob address", "Approximate 3D pixel-distance layout"),
                        horizontal_spacing=.02)
    plotly_panel(fig, exact, knobs, triplet, 1, True)
    plotly_panel(fig, layout, knobs, triplet, 2, False)
    for scene, is_exact in (("scene", True), ("scene2", False)):
        if is_exact:
            axes = [f"{LABELS[k]} (fraction of range)" for k in triplet]
            axis_settings = dict(range=[-.1, 1.1], tickvals=[0, .5, 1], ticktext=["low", "mid", "high"])
        else:
            axes = ["layout x (pixel L2)", "layout y (pixel L2)", "layout z (pixel L2)"]
            axis_settings = {}
        fig.update_layout(**{scene: dict(
            xaxis=dict(title=axes[0], **axis_settings),
            yaxis=dict(title=axes[1], **axis_settings),
            zaxis=dict(title=axes[2], **axis_settings),
            aspectmode="data", camera=dict(eye=dict(x=1.55, y=1.55, z=1.25)),
        )})
    fig.update_layout(height=650, width=1450, margin=dict(l=0, r=0, t=55, b=0),
                      paper_bgcolor="white", font=dict(color="#24323c"),
                      legend=dict(orientation="h", x=.5, xanchor="center", y=-.04))
    landmarks = [0, 171, 342]
    thumbnails = "".join(
        f'<div class="thumb"><img src="{image_url(images[i])}" alt="Generated 1 at sampled knob setting">'
        f'<strong>{label}</strong><span>{" · ".join(f"{LABELS[k]} {knobs[i, k]:.2f} {UNITS[k]}" for k in triplet)}</span></div>'
        for i, label in zip(landmarks, ("All low", "All middle", "All high"))
    )
    status = (f"343 sampled generated images; other two knobs fixed at their midpoints. "
              f"Neighbor distance error: median {errors['neighbor_median']:.0%}, 90th percentile {errors['neighbor_p90']:.0%}. "
              f"All-pair distance error: median {errors['pair_median']:.0%}, 90th percentile {errors['pair_p90']:.0%}.")
    return (f'<h2>{title}</h2><p>{description} {status}</p><div class="thumbs">{thumbnails}</div>'
            + fig.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True, "displaylogo": False}))


def static_case(fig, gs, row, title, triplet, exact, knobs, images, layout, errors):
    header = fig.add_subplot(gs[2 * row, :])
    header.axis("off")
    header.text(0, .88, title, fontsize=13, weight="bold", color="#24323c", va="top")
    for j, k in enumerate(triplet):
        header.text(j * .255, .62,
                    f"■ {LABELS[k]} ({RANGES[k, 0]:g}–{RANGES[k, 1]:g} {UNITS[k]})",
                    fontsize=9.2, color=COLORS[j], va="top")
    fixed = ", ".join(f"{LABELS[k]} {RANGES[k].mean():.2f} {UNITS[k]}" for k in range(5) if k not in triplet)
    header.text(0, .35,
                f"Held fixed: {fixed}. 343 sampled images. Neighbor error in 3D fit: median {errors['neighbor_median']:.0%}; all-pair 90th percentile {errors['pair_p90']:.0%}.",
                fontsize=8.8, color="#455762", va="top")
    img_ax = fig.add_subplot(gs[2 * row + 1, 0])
    # Three concrete examples give meaning to the colored grid.
    montage = np.concatenate([images[i] for i in (0, 171, 342)], axis=1)
    img_ax.imshow(montage, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    img_ax.set_title("Generated 1s: all low · middle · all high", fontsize=9, pad=6)
    img_ax.set_xticks([])
    img_ax.set_yticks([])
    for spine in img_ax.spines.values():
        spine.set_visible(False)
    for col, points in ((1, exact), (2, layout)):
        ax = fig.add_subplot(gs[2 * row + 1, col], projection="3d")
        for axis, side, face, triangles in boundary_faces(points):
            ax.plot_trisurf(face[:, 0], face[:, 1], face[:, 2], triangles=triangles,
                            color=COLORS[axis], alpha=.15, linewidth=0, shade=False)
        for axis, ids in paths():
            xyz = points[ids]
            ax.plot(*xyz.T, color=COLORS[axis], lw=1.3, alpha=.75)
            ax.scatter(*xyz[[0, -1]].T, color=COLORS[axis], s=5, depthshade=False)
        ax.view_init(elev=21, azim=-61)
        ax.set_box_aspect(np.maximum(np.ptp(points, axis=0), .01), zoom=.84)
        ax.tick_params(labelsize=6)
        if col == 1:
            ax.set(xlabel=LABELS[triplet[0]], ylabel=LABELS[triplet[1]], zlabel=LABELS[triplet[2]])
            ax.set_xlim(-.1, 1.1)
            ax.set_ylim(-.1, 1.1)
            ax.set_zlim(-.1, 1.1)
            ax.set_title("Exact knob coordinates", fontsize=10, pad=6)
        else:
            ax.set(xlabel="layout x", ylabel="layout y", zlabel="layout z")
            ax.set_title(f"3D fit of 784-pixel distances\nmedian pair error {errors['pair_median']:.0%}", fontsize=10, pad=6)


def main():
    results = []
    for title, triplet, description in CASES:
        exact, knobs, images = sample_case(triplet)
        layout, errors = pixel_layout(exact, images)
        print(title, errors)
        results.append((title, triplet, description, exact, knobs, images, layout, errors))
    fig = plt.figure(figsize=(18, 17), facecolor="white")
    fig.suptitle("Three 3D slices through the five-knob generated-1 family", x=.5, y=.987,
                 fontsize=19, weight="bold", color="#24323c")
    fig.text(.5, .964, "Each image is 28×28 pixel coverage (0 = blank, 1 = filled). Two controls stay at their midpoint; three span their full ranges.",
             ha="center", fontsize=10, color="#455762")
    fig.text(.5, .945, "Colored lines follow one knob while the other two stay fixed. Left: example images. Center: exact parameter address. Right: approximate pixel-space distances.",
             ha="center", fontsize=10, color="#455762")
    gs = fig.add_gridspec(6, 3, left=.025, right=.985, top=.925, bottom=.075,
                          width_ratios=[.65, 1, 1], height_ratios=[.23, 1] * 3,
                          hspace=.10, wspace=.02)
    for row, (title, triplet, description, exact, knobs, images, layout, errors) in enumerate(results):
        static_case(fig, gs, row, title, triplet, exact, knobs, images, layout, errors)
    fig.text(.5, .042, "The exact chart keeps every knob setting but its straight box does not show pixel distance. The fitted chart bends, but changes distances as stated above.",
             ha="center", fontsize=9.2, color="#455762")
    fig.text(.5, .022, "This is the controlled generator, not real MNIST. Surfaces and lines connect sampled knob settings; neither display is a uniquely true 3D shape in 784D.",
             ha="center", fontsize=9.2, color="#455762")
    fig.savefig(OUT / "three_knob_slices.png", dpi=150, facecolor="white")
    plt.close(fig)
    parts = []
    for title, triplet, description, exact, knobs, images, layout, errors in results:
        parts.append(interactive_case(title, description, triplet, exact, knobs, images, layout, errors))
    import plotly.offline as offline
    plotly_js = offline.get_plotlyjs()
    sections = "".join(f'<section id="case{i}" class="case" style="display:{"block" if i == 0 else "none"}">{content}</section>' for i, content in enumerate(parts))
    buttons = "".join(f'<button onclick="showCase({i})" id="button{i}" class="{"active" if i == 0 else ""}">{title}</button>' for i, (title, *_rest) in enumerate(CASES))
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Three-knob generated 1 slices</title>
    <style>body{{font:16px/1.5 system-ui,sans-serif;color:#24323c;margin:20px auto;max-width:1500px;padding:0 18px}}
    h1{{font-size:29px;margin-bottom:4px}}h2{{margin:20px 0 2px}}p{{max-width:1100px}}.tabs{{display:flex;gap:8px;flex-wrap:wrap;margin:20px 0}}
    button{{border:1px solid #b8c8ce;border-radius:8px;padding:10px 16px;background:white;cursor:pointer;font-size:15px}}button.active{{background:#204b60;color:white}}
    .thumbs{{display:flex;gap:24px;flex-wrap:wrap;margin:22px 0 0}}.thumb{{display:flex;align-items:center;gap:10px;max-width:330px}}
    .thumb img{{width:100px;height:100px;image-rendering:pixelated;border:1px solid #bbc6cb}}.thumb strong,.thumb span{{display:block;font-size:12px}}
    .thumb span{{color:#52616a}}.note{{background:#eef4f6;padding:12px 16px;border-radius:8px}}</style><script>{plotly_js}</script></head><body>
    <h1>Three 3D slices through generated 1s</h1><p class="note">A generated 1 is a 28×28 image of pixel coverage: 0 is blank, 1 is fully inked. Two of its five controls stay at their midpoints. The other three vary across their full ranges. Blue, orange, and purple lines each vary one control. The <b>left plot gives exact knob addresses</b>; the <b>right plot bends the grid to approximate Euclidean distances between the 784 pixel values</b>. The listed distance errors show what the 3D fit loses.</p>
    <div class="tabs">{buttons}</div>{sections}
    <p class="note">These are 343 samples of each continuous three-knob slice. The line and surface connections show the known parameter topology. No 3D layout is claimed to exactly preserve every pixel-space distance; this is the controlled generator, not the real MNIST dataset.</p>
    <script>function showCase(i){{document.querySelectorAll('.case').forEach((el,j)=>el.style.display=j===i?'block':'none');document.querySelectorAll('.tabs button').forEach((el,j)=>el.classList.toggle('active',j===i));document.querySelectorAll('#case'+i+' .js-plotly-plot').forEach(el=>Plotly.Plots.resize(el));}}</script></body></html>'''
    (OUT / "three_knob_slices.html").write_text(html)


if __name__ == "__main__":
    main()
