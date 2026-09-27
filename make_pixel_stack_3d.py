"""3D pixel stacks for all five generated-1 knobs and five knob pairs.

Run: python make_pixel_stack_3d.py
Outputs an interactive HTML gallery and two contact sheets in figures/.
"""

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from plotly.subplots import make_subplots

from grey_ones import RANGES
from lean_axis import Geometry, coverage


OUT = Path("figures")
OUT.mkdir(exist_ok=True)
LEVELS = 25
PROGRESS = np.linspace(0, 1, LEVELS)
FRACTIONS = .15 + .70 * PROGRESS
BASE = RANGES.mean(axis=1)
PAIR_SEED = 123
DISPLAY_NAMES = ("Horizontal position", "Vertical position", "Stroke height",
                 "Stroke width", "Lean angle")
UNITS = ("px", "px", "px", "px", "°")
RED = "#bf443b"
BLUE = "#3671b0"


@dataclass
class Case:
    title: str
    knobs: tuple[int, ...]
    settings: np.ndarray
    images: np.ndarray
    differences: np.ndarray


@dataclass
class CellSurface:
    vertices: np.ndarray
    triangles: np.ndarray
    triangle_colors: list[str]
    quads: np.ndarray
    quad_colors: np.ndarray
    hover_data: np.ndarray
    edge_points: np.ndarray


def geometry(q):
    return Geometry(*map(float, q))


def cases():
    rng = np.random.default_rng(PAIR_SEED)
    possible_pairs = list(combinations(range(5), 2))
    chosen = [possible_pairs[i] for i in rng.choice(len(possible_pairs), 5,
                                                     replace=False)]
    result = []
    for knobs in [(i,) for i in range(5)] + chosen:
        settings = np.repeat(BASE[None, :], LEVELS, axis=0)
        for j in knobs:
            lo, hi = RANGES[j]
            settings[:, j] = lo + FRACTIONS * (hi - lo)
        images = np.stack([coverage(geometry(q)) for q in settings])
        title = " + ".join(DISPLAY_NAMES[j] for j in knobs)
        result.append(Case(title, knobs, settings, images,
                           np.diff(images, axis=0)))
    return result


def setting_text(case):
    return "; ".join(
        f"{case.settings[0,j]:.2f}→"
        f"{case.settings[-1,j]:.2f} {UNITS[j]}"
        for j in case.knobs
    )


def _tint(base, amount):
    """Keep small occupied cells visible; use saturation for magnitude."""
    strength = .32 + .68 * np.sqrt(np.clip(amount, 0, 1))
    rgb = 1 - strength * (1 - np.asarray(base[:3]))
    return tuple(float(x) for x in rgb)


def _hex(rgb):
    return "#" + "".join(f"{int(round(255 * x)):02x}" for x in rgb)


def cell_surface(values, z_edges, material, max_change=1):
    """Make filled exposed voxel faces and their pixel-grid outlines.

    A state cell is occupied when coverage is positive. A change cell is
    occupied for only its sign. Faces against empty cells or another sign are
    exposed; interior faces of the same material are omitted.
    """
    if material == "state":
        occupied = values > 0
    elif material == "gain":
        occupied = values > 0
    elif material == "loss":
        occupied = values < 0
    else:
        raise ValueError(material)
    directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0),
                  (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    face_corners = [(0, 1, 2, 3), (4, 7, 6, 5),
                    (0, 4, 5, 1), (3, 2, 6, 7),
                    (0, 3, 7, 4), (1, 5, 6, 2)]
    # Corner numbering is z-low square 0..3, z-high square 4..7.
    verts = []
    triangles = []
    triangle_colors = []
    quads = []
    quad_colors = []
    hover = []
    edges = set()
    for k, r, c in np.argwhere(occupied):
        z0, z1 = float(z_edges[k]), float(z_edges[k + 1])
        x0, x1 = float(c), float(c + 1)
        y0, y1 = float(r), float(r + 1)
        corners = ((x0, y0, z0), (x1, y0, z0),
                   (x1, y1, z0), (x0, y1, z0),
                   (x0, y0, z1), (x1, y0, z1),
                   (x1, y1, z1), (x0, y1, z1))
        value = float(values[k, r, c])
        if material == "state":
            base = plt.cm.viridis((z0 + z1) / 2)
            color = _tint(base, value)
        else:
            base = (191 / 255, 68 / 255, 59 / 255) if value > 0 else (
                54 / 255, 113 / 255, 176 / 255)
            color = _tint(base, abs(value) / max_change)
        color_hex = _hex(color)
        for (dk, dr, dc), indices in zip(directions, face_corners):
            nk, nr, nc = k + dk, r + dr, c + dc
            if (0 <= nk < occupied.shape[0] and
                    0 <= nr < occupied.shape[1] and
                    0 <= nc < occupied.shape[2] and
                    occupied[nk, nr, nc]):
                continue
            quad = tuple(corners[i] for i in indices)
            offset = len(verts)
            verts.extend(quad)
            quads.append(quad)
            quad_colors.append(color)
            triangles.extend(((offset, offset + 1, offset + 2),
                              (offset, offset + 2, offset + 3)))
            triangle_colors.extend((color_hex, color_hex))
            hover.extend(((r, c, value, (z0 + z1) / 2),) * 4)
            for a, b in zip(quad, quad[1:] + quad[:1]):
                edges.add(tuple(sorted((a, b))))
    edge_points = []
    for a, b in sorted(edges):
        edge_points.extend((a, b, (np.nan, np.nan, np.nan)))
    return CellSurface(
        vertices=np.asarray(verts, dtype=float).reshape(-1, 3),
        triangles=np.asarray(triangles, dtype=int).reshape(-1, 3),
        triangle_colors=triangle_colors,
        quads=np.asarray(quads, dtype=float).reshape(-1, 4, 3),
        quad_colors=np.asarray(quad_colors, dtype=float).reshape(-1, 3),
        hover_data=np.asarray(hover, dtype=float).reshape(-1, 4),
        edge_points=np.asarray(edge_points, dtype=float).reshape(-1, 3),
    )


def build_surfaces(all_cases):
    state_edges = np.r_[0, (PROGRESS[:-1] + PROGRESS[1:]) / 2, 1]
    change_edges = PROGRESS
    max_change = max(float(np.max(np.abs(case.differences)))
                     for case in all_cases)
    return [(
        cell_surface(case.images, state_edges, "state"),
        cell_surface(case.differences, change_edges, "gain", max_change),
        cell_surface(case.differences, change_edges, "loss", max_change),
    ) for case in all_cases], max_change


def mesh_trace(surface, name, kind):
    vertices = surface.vertices
    triangles = surface.triangles
    return go.Mesh3d(
        x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
        i=triangles[:, 0], j=triangles[:, 1], k=triangles[:, 2],
        facecolor=surface.triangle_colors, customdata=surface.hover_data,
        flatshading=True, opacity=1,
        lighting=dict(ambient=.82, diffuse=.22, specular=.03,
                      roughness=1, fresnel=0),
        name=name, showlegend=False,
        hovertemplate=(f"{name}<br>row %{{customdata[0]:.0f}}, "
                       "column %{customdata[1]:.0f}<br>"
                       + ("ink coverage " if kind == "state" else "pixel change ")
                       + "%{customdata[2]:+.4f}<br>path progress "
                       "%{customdata[3]:.2f}<extra></extra>"),
    )


def edge_trace(points, name):
    return go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode="lines", line=dict(color="rgba(28,39,48,.31)", width=1.2),
        name=name, hoverinfo="skip", showlegend=False,
    )


def plotly_traces(surfaces):
    state, gain, loss = surfaces
    change_edges = np.concatenate((gain.edge_points, loss.edge_points))
    return [mesh_trace(state, "Rendered ink", "state"),
            edge_trace(state.edge_points, "Image cell grid"),
            mesh_trace(gain, "Ink gained", "change"),
            mesh_trace(loss, "Ink lost", "change"),
            edge_trace(change_edges, "Change cell grid")]


def interactive_figure(all_cases, all_surfaces):
    fig = make_subplots(rows=1, cols=2,
                        specs=[[{"type": "scene"}, {"type": "scene"}]],
                        subplot_titles=("A · Filled cells of the rendered 1s",
                                        "B · Filled cells of pixel changes"),
                        horizontal_spacing=.08)
    traces_per_case = 5
    for index, surfaces in enumerate(all_surfaces):
        traces = plotly_traces(surfaces)
        for trace, column in zip(traces, (1, 1, 2, 2, 2)):
            trace.visible = index == 0
            fig.add_trace(trace, row=1, col=column)
    buttons = []
    for index, case in enumerate(all_cases):
        shown = [index * traces_per_case <= i < (index + 1) * traces_per_case
                 for i in range(traces_per_case * len(all_cases))]
        buttons.append(dict(
            label=("Single: " if len(case.knobs) == 1 else "Pair: ")
                  + case.title,
            method="update",
            args=[{"visible": shown},
                  {"title.text": f"{case.title} · {setting_text(case)}"}],
        ))
    camera = dict(eye=dict(x=1.65, y=-1.75, z=1.20))
    scene = dict(
        xaxis=dict(title="image column", range=[0, 28], tickvals=[0, 14, 28],
                   showgrid=True, gridcolor="#dae3e7"),
        yaxis=dict(title="image row", range=[28, 0], tickvals=[0, 14, 28],
                   showgrid=True, gridcolor="#dae3e7"),
        zaxis=dict(title="path progress", range=[0, 1], tickvals=[0, .5, 1]),
        aspectmode="manual", aspectratio=dict(x=1, y=1, z=1.55),
        camera=camera, bgcolor="#f9fbfc",
    )
    fig.update_layout(
        title=dict(text=f"{all_cases[0].title} · {setting_text(all_cases[0])}",
                   x=.5, y=.975, font=dict(size=18)),
        scene=scene, scene2=scene, height=790,
        margin=dict(l=10, r=10, t=112, b=10),
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="Arial, sans-serif", color="#26343b"),
        legend=dict(orientation="h", x=.62, y=.03),
        uirevision="pixel-stack-cameras",
        updatemenus=[dict(type="dropdown", direction="down", buttons=buttons,
                          x=.5, xanchor="center", y=1.075, yanchor="bottom",
                          bgcolor="white", bordercolor="#b9c5cb")],
    )
    return fig


def write_html(fig):
    plot = fig.to_html(full_html=False, include_plotlyjs=True,
                       div_id="pixel-stack-plots",
                       config={"displaylogo": False,
                               "responsive": True})
    html = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Generated 1 pixel stacks in 3D</title>
<style>body{margin:0;background:#f3f6f8;color:#26343b;font:15px/1.45 system-ui,sans-serif}
main{max-width:1600px;margin:auto;padding:18px}h1{margin:0 0 6px;font-size:25px}
p{max-width:1400px;margin:7px 0}section{background:white;border:1px solid #dbe4e8;
border-radius:9px;margin-top:16px;padding:8px}ul{margin:10px 0;padding-left:25px}
.note{color:#52636d;font-size:14px}</style></head><body><main>
<h1>Stacking generated “1” images and their pixel-change vectors</h1>
<p>Choose one of five single knobs or five seeded random knob pairs. Drag either 3D plot to rotate;
scroll to zoom. Both plots use the original 28 image columns and rows as horizontal axes.
Height is progress along the selected knob path, from 0 to 1.</p>
<p>Horizontal position moves the stroke left or right; vertical position moves it up or down.
Stroke height lengthens it, stroke width thickens it, and lean angle tilts it.</p>
<p><strong>A</strong> fills the image-grid cells that contain ink in each of 25 rendered images.
Purple cells are early in the path; yellow cells are late. Paler cells have less ink coverage.
<strong>B</strong> fills cells whose pixel value changes over each of 24 steps: red gains ink,
blue loses ink, and deeper color means a larger change. Thin dark lines mark visible cell
boundaries. Rotate the plots to see the filled structure from different sides.</p>
<p class="note">Other knobs stay at their range midpoints. Each moved knob runs from 15% to 85%
of its declared range; in a pair both move together, so the pair still traces a one-parameter
path. These are stacks over pixel position and path progress, not 3D distance-preserving
coordinates of the 784-dimensional image manifold. A cell is filled whenever its sampled
coverage or change is nonzero; paleness encodes magnitude, and the visible faces carry
the gridlines. Cells cover sampled path intervals, not the unsampled continuous path.</p>
<section>""" + plot + """</section>
<p class="note">The renderer uses continuous pixel coverage. The change layer records
<code>image at next level − image at current level</code>; it is a finite step rather than
an infinitesimal derivative. Generated images are shown here, not real MNIST samples.</p>
</main></body></html>"""
    (OUT / "pixel_stacks_3d.html").write_text(html, encoding="utf-8")


def setup_axis(ax):
    ax.view_init(elev=22, azim=-63)
    ax.set(xlim=(0, 28), ylim=(28, 0), zlim=(0, 1),
           xlabel="column", ylabel="row", zlabel="progress")
    ax.set_xticks([0, 14, 28])
    ax.set_yticks([0, 14, 28])
    ax.set_zticks([0, .5, 1])
    ax.set_box_aspect((1, 1, 1.45))
    ax.tick_params(labelsize=7, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("#f8fafb")
        axis.pane.set_alpha(.4)


def contact_sheet(all_cases, all_surfaces, pair=False):
    selected = list(zip(all_cases[5:], all_surfaces[5:])) if pair else list(
        zip(all_cases[:5], all_surfaces[:5]))
    fig = plt.figure(figsize=(14.5, 24), facecolor="white")
    title = ("Five simultaneous two-knob paths: image stacks and change stacks"
             if pair else "Five single-knob paths: image stacks and change stacks")
    fig.suptitle(title, y=.991, fontsize=17.5, fontweight="bold", color="#26343b")
    subtitle = ("Pairs sampled without replacement from all 10 possible pairs; seed 123. "
                "Both knobs move 15%→85% of their ranges together."
                if pair else "One knob moves 15%→85% of its range; the other four remain at their midpoints.")
    fig.text(.5, .972, subtitle, ha="center", fontsize=10.2, color="#52636d")
    fig.text(.5, .955,
             "Each cuboid fills one nonzero pixel cell over one sampled path interval. "
             "Left: 25 images, purple early → yellow late. Right: 24 changes, red gains ink, blue loses.",
             ha="center", fontsize=9.9, color="#26343b")
    inset = fig.add_axes([.885, .933, .065, .06])
    inset.imshow(coverage(geometry(BASE)), cmap="gray_r", vmin=0, vmax=1,
                 interpolation="nearest")
    inset.set_title("Midpoint 1", fontsize=8, pad=2)
    inset.set_xticks([])
    inset.set_yticks([])
    gs = fig.add_gridspec(5, 2, left=.04, right=.97, top=.93, bottom=.085,
                          hspace=.20, wspace=.04)
    for row_index, (case, (state, gain, loss)) in enumerate(selected):
        ax = fig.add_subplot(gs[row_index, 0], projection="3d")
        state_faces = Poly3DCollection(
            state.quads, facecolors=state.quad_colors,
            edgecolors=(.13, .17, .20, .35), linewidths=.13,
            rasterized=True)
        ax.add_collection3d(state_faces)
        setup_axis(ax)
        if pair:
            state_title = f"{case.title} · rendered 1s"
        else:
            state_title = f"{case.title} · rendered 1s\n{setting_text(case)}"
        ax.set_title(state_title, fontsize=10.0, pad=1)

        ax = fig.add_subplot(gs[row_index, 1], projection="3d")
        change_faces = Poly3DCollection(
            np.concatenate((gain.quads, loss.quads)),
            facecolors=np.concatenate((gain.quad_colors, loss.quad_colors)),
            edgecolors=(.13, .17, .20, .34), linewidths=.13,
            rasterized=True)
        ax.add_collection3d(change_faces)
        setup_axis(ax)
        ax.set_title(f"{case.title} · change to next image",
                     fontsize=10.0, pad=1)
    fig.text(.5, .056,
             "x = image column, y = image row (0 at image top), z = path progress 0→1. "
             "Visible cuboid faces have gridlines. Paler fills indicate smaller coverage or change.",
             ha="center", fontsize=9.4, color="#26343b")
    fig.text(.5, .039,
             "These are literal pixel stacks, not a distance-preserving 3D embedding of 784D images. "
             "Each row is a one-parameter path through the controlled five-knob generator.",
             ha="center", fontsize=9.2, color="#52636d")
    suffix = "pairs" if pair else "single"
    fig.savefig(OUT / f"pixel_stacks_3d_{suffix}.png", dpi=145,
                facecolor="white")
    plt.close(fig)


def main():
    all_cases = cases()
    all_surfaces, _ = build_surfaces(all_cases)
    write_html(interactive_figure(all_cases, all_surfaces))
    contact_sheet(all_cases, all_surfaces)
    contact_sheet(all_cases, all_surfaces, pair=True)
    for case in all_cases:
        print(case.title, setting_text(case),
              "start-end pixel distance",
              round(float(np.linalg.norm(case.images[-1] - case.images[0])), 3))


if __name__ == "__main__":
    main()
