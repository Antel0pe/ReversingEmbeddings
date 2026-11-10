"""Pictures: one scorecard for a run, and one explanation per failed test."""
import textwrap

import matplotlib
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection

from .diagnose import where_it_fails
from .scoring import align_to_square, signed_area, triangles

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, FAINT = "#0b0b0b", "#52514e", "#dddcd6"
GOOD_BG, BAD_BG = "#d8eeda", "#f6d6d3"
FOLD, HIT, MARK = "#d03b3b", "#eda100", "#0b0b0b"
BLUE, ORANGE = np.array(matplotlib.colors.to_rgb("#2a78d6")), np.array(matplotlib.colors.to_rgb("#eb6834"))


def box_shape(xyz):
    """A 3-D plot's proportions, with a minimum thickness: a perfectly flat sheet has none."""
    size = np.ptp(xyz, 0)
    return np.maximum(size, 0.05 * size.max())


def print_colours(uv):
    """Blue to orange left to right, pale to deep bottom to top: each spot has its own colour."""
    a = ((uv[:, 0] + 1) / 2)[:, None]
    shade = (0.30 + 0.70 * (uv[:, 1] + 1) / 2)[:, None]
    return ((1 - a) * BLUE + a * ORANGE) * shade + (1 - shade)


def scorecard(summary, methods, sheets, args, path):
    fig_h = 1.9 + 0.46 * len(sheets)
    fig = plt.figure(figsize=(3.0 + 2.1 * len(methods), fig_h), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, fig.get_figwidth())
    ax.set_ylim(fig_h, 0)
    ax.axis("off")
    ax.text(0.25, 0.42, "Unfolding test suite", fontsize=18, fontweight="bold", color=INK, va="center")
    ax.text(0.25, 0.80, f"{args.grid}×{args.grid} dots, {args.seeds} seed(s), must pass on every seed.", fontsize=10.5,
            color=MUTED, va="center")
    left = 2.9
    for j, m in enumerate(methods):
        ax.text(left + 2.1 * (j + 0.5), 1.28, m, ha="center", va="center", fontsize=11, color=INK, fontweight="bold")
    y = 1.55
    for family, title in [("bent", "Bent, not stretched (must return the square)"), ("stretched", "Stretched (no flat shape to return)")]:
        names = [s for s in sheets if summary[f"{s} / {methods[0]}"]["family"] == family]
        if not names:
            continue
        ax.text(0.25, y + 0.14, title, fontsize=10, color=MUTED, va="center", style="italic")
        y += 0.32
        for name in names:
            ax.text(0.25, y + 0.2, name, fontsize=11, va="center", color=INK)
            for j, m in enumerate(methods):
                row = summary[f"{name} / {m}"]
                ax.add_patch(plt.Rectangle((left + 2.1 * j + 0.04, y + 0.02), 2.02, 0.38,
                                           color=GOOD_BG if row["passed"] else BAD_BG, lw=0))
                ax.text(left + 2.1 * (j + 0.5), y + 0.21, row["words"], ha="center", va="center", fontsize=10.5, color=INK)
            y += 0.42
        y += 0.08
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)


def failure_figure(test, row, path):
    inst, flat = row["instance"], row["flat"]
    r, diagnosis, info = row["result"], row["diagnosis"], row["info"]
    uv, xyz, grid = inst.uv, inst.xyz, inst.grid
    T = triangles(grid)
    folded, collides, _ = where_it_fails(flat, uv, grid)
    example = diagnosis.get("example")
    colours = print_colours(uv)
    fig = plt.figure(figsize=(15, 8.1), facecolor="white")
    sheet_name, method_name = test.split(" / ")
    fig.text(0.025, 0.955, f"Why {method_name} fails on {sheet_name}", fontsize=19, fontweight="bold", color=INK)
    causes = diagnosis["causes"] or [{"cause": "no cause named", "plain": ""}]
    fig.text(0.025, 0.92, "Cause: " + "; ".join(c["cause"] for c in causes) + ".", fontsize=13, color=INK, va="top")
    fig.text(0.025, 0.885, textwrap.fill(" ".join(c["plain"] for c in causes), 150), fontsize=11, color=MUTED,
             va="top", linespacing=1.4)

    # 1. where on the paper
    ax = fig.add_axes([0.03, 0.14, 0.27, 0.56])
    ax.scatter(uv[:, 0], uv[:, 1], c=colours, s=5, linewidths=0, rasterized=True)
    if collides.any():
        ax.add_collection(PolyCollection(uv[T[collides]], facecolor=HIT, edgecolor="none", alpha=0.75))
    if folded.any():
        ax.add_collection(PolyCollection(uv[T[folded]], facecolor=FOLD, edgecolor="none", alpha=0.85))
    _style(ax)
    ax.set_title("Where on the paper it went wrong", fontsize=13, loc="left", color=INK)
    if example:
        _mark(ax, example["paper_a"], "A")
        _mark(ax, example["paper_b"], "B")
    ax.text(0, -0.03, f"Red: comes out mirror-flipped ({100 * diagnosis['folded_paper']:.1f}% of the paper).\n"
            f"Amber: lands on a far-away part ({100 * diagnosis['colliding_paper']:.1f}% of the paper).",
            transform=ax.transAxes, va="top", fontsize=10.5, color=MUTED, linespacing=1.4)

    # 2. the sheet in 3-D, with the cause drawn on it
    ax = fig.add_axes([0.33, 0.11, 0.32, 0.62], projection="3d")
    ax.scatter(*xyz.T, c=colours, s=3, linewidths=0, depthshade=False, rasterized=True, alpha=0.75)
    caption = "The sheet in 3D, as given to the method (before hiding)."
    if info.get("edges") is not None:
        jumps = info["edges"][np.linalg.norm(uv[info["edges"][:, 0]] - uv[info["edges"][:, 1]], axis=1) > 0.3]
        if len(jumps):
            for a, b in jumps[:: max(1, len(jumps) // 150)]:
                ax.plot(*np.c_[xyz[a], xyz[b]], color=MARK, lw=0.9)
            caption = f"Black: neighbour links that jump between layers ({len(jumps)})."
    if example:
        c3 = xyz[T].mean(1)
        for key, name in [("triangle_a", "A"), ("triangle_b", "B")]:
            ax.scatter(*c3[example[key]], s=70, c=MARK, depthshade=False)
            ax.text(*c3[example[key]], f"  {name}", fontsize=13, fontweight="bold", color=MARK)
    ax.set_box_aspect(box_shape(xyz))
    ax.set_axis_off()
    ax.view_init(elev=24, azim=-60)
    ax.set_title("The sheet", fontsize=13, loc="left", color=INK)
    fig.text(0.34, 0.075, caption, fontsize=10.5, color=MUTED)

    # 3. the result
    ax = fig.add_axes([0.69, 0.14, 0.27, 0.56])
    ax.scatter(flat[:, 0], flat[:, 1], c=colours, s=5, linewidths=0, rasterized=True)
    _style(ax)
    ax.set_aspect("equal", "datalim")
    ax.set_title("What the method produced", fontsize=13, loc="left", color=INK)
    if example:
        _mark(ax, example["flat_a"], "A", dx=-1)
        _mark(ax, example["flat_b"], "B", dx=1)
        text = (f"A and B land together here, but are {example['paper_distance']:.2f} apart on the paper\n"
                f"(the side is 2) and {example['distance_3d']:.2f} apart in 3D.")
    else:
        text = f"Off the true square by {r['square_error_median']:.1f}% of the side (typical)."
    ax.text(0, -0.03, text, transform=ax.transAxes, va="top", fontsize=10.5, color=MUTED, linespacing=1.4)
    fig.savefig(path, dpi=110, facecolor="white")
    plt.close(fig)


def _style(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ax.spines.values():
        side.set_color(FAINT)


def _mark(ax, p, label, dx=0):
    ax.plot(*p, "o", ms=8, color=MARK, mec="white", mew=1.2, zorder=6)
    ax.annotate(label, p, xytext=(10 * (dx or 1), 8), textcoords="offset points", fontsize=13, fontweight="bold",
                color=MARK, zorder=7)


# ------------------------------------------------------------------ the gallery
PLAIN_NAMES = {"check_k10": "Walking +\nlink check", "convex_edge": "Link check +\nconvex edge",
               "best_fit_plane": "Best-fit plane\n(projection)", "tangent_plane": "Plane at a\nrandom dot",
               "isomap_k6": "Walking,\n6 neighbours", "isomap_k10": "Walking,\n10 neighbours",
               "isomap_k20": "Walking,\n20 neighbours", "adaptive_tol05": "Flatness hops,\nstrict (0.05)",
               "adaptive_tol10": "Flatness hops,\nmedium (0.10)", "adaptive_tol20": "Flatness hops,\nloose (0.20)"}
FAMILY_TITLES = {"bent": "Sheets bent without stretching: a correct result is the original square",
                 "stretched": "Sheets stretched while being shaped: no flat result can match the square exactly"}
ERROR_MAX = 25.0           # % of the side at which the colour scale tops out
ERROR_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list("error", ["#eaf2fb", "#7fb2e6", "#2a78d6", "#123e78"])
GRID_EVERY = 8             # graph-paper lines every this many grid cells
VIEW = 1.75                # half-width of every output panel, in paper units


def _graph_paper(aligned, grid):
    """Lines of constant row and column on the paper, drawn where they landed."""
    nodes = aligned.reshape(grid, grid, 2)
    pick = np.arange(0, grid, GRID_EVERY)
    return [nodes[j] for j in pick] + [nodes[:, i] for i in pick]


def gallery(detail, summary, methods, sheets, family, path):
    """Rows are input sheets, columns are methods; every output is lined up with the square."""
    if not sheets:
        return
    label_w, input_w, cell_w, cell_h, top = 1.75, 2.15, 1.95, 2.45, 2.15
    width = max(label_w + input_w + cell_w * len(methods) + 0.2, 19.3)      # room for the legend
    height = top + cell_h * len(sheets) + 0.2
    fig = plt.figure(figsize=(width, height), facecolor="white")

    def box(x, y, w, h):                     # inches from the top-left corner
        return [x / width, 1 - (y + h) / height, w / width, h / height]

    fig.text(0.2 / width, 1 - 0.38 / height, FAMILY_TITLES[family], fontsize=19, fontweight="bold", color=INK, va="center")
    paper_colours = family == "stretched"     # the square is not the target here, so distance from it means little
    fig.text(0.2 / width, 1 - 0.76 / height, ("Each row is one input; each column is one method's flat result, lined up. \"Lengths\": short distances in the result "
             "against the 3D sheet (1× = unchanged). \"Folded N%\": paper flipped or overlapping."
             if paper_colours else
             "Each row is one input; each column is one method's flat result, turned, mirrored and resized to best "
             "match the original square, so orientation and size don't count. \"Folded N%\" = share of the paper flipped over or "
             "landing on another part."), fontsize=11, color=MUTED, va="center")

    # legend: what the colours mean
    y0 = 1.02
    ax = fig.add_axes(box(0.2, y0, 0.62, 0.62))
    legend_uv = np.stack(np.meshgrid(np.linspace(-1, 1, 60), np.linspace(1, -1, 60)), axis=2).reshape(-1, 2)
    ax.imshow(print_colours(legend_uv).reshape(60, 60, 3), extent=(-1, 1, -1, 1))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.text(0.92 / width, 1 - (y0 + 0.31) / height, ("INPUT AND OUTPUT colours: where\na dot sits on the original paper" if paper_colours
             else "INPUT colours: where a dot\nsits on the original paper"),
             fontsize=11, color=INK, va="center", linespacing=1.35)
    if paper_colours:
        fig.text(4.2 / width, 1 - (y0 + 0.31) / height, "A smooth blend means the sheet came back cleanly.\nSpeckle or broken colour means parts landed together.",
                 fontsize=11, color=INK, va="center", linespacing=1.35)
    else:
        ax = fig.add_axes(box(3.55, y0 + 0.22, 3.6, 0.2))
        ax.imshow(np.linspace(0, 1, 256)[None], cmap=ERROR_CMAP, aspect="auto", extent=(0, ERROR_MAX, 0, 1))
        ax.set_yticks([])
        ax.set_xticks([0, 5, 10, 15, 20, 25], ["0%", "5%", "10%", "15%", "20%", "25%+"], fontsize=10)
        ax.tick_params(length=2, colors=MUTED)
        fig.text(3.55 / width, 1 - (y0 + 0.08) / height, "OUTPUT colours: how far each patch landed from its true place "
                 "(% of the paper's side)", fontsize=11, color=INK, va="center")
    ax = fig.add_axes(box(11.0, y0 + 0.17, 0.42, 0.3))
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=FOLD))
    ax.axis("off")
    fig.text(11.52 / width, 1 - (y0 + 0.32) / height, "Red: folded over, so this part of\nthe sheet is mirror-flipped",
             fontsize=11, color=INK, va="center", linespacing=1.35)
    if not paper_colours:
        ax = fig.add_axes(box(15.0, y0 + 0.17, 0.42, 0.3))
        ax.plot([0, 1], [0.5, 0.5], color="#8a8983", lw=1.6)
        ax.axis("off")
        fig.text(15.52 / width, 1 - (y0 + 0.32) / height, "Grey outline: the original square.\nThin lines: graph paper printed on it.",
                 fontsize=11, color=INK, va="center", linespacing=1.35)

    fig.text((label_w + input_w / 2) / width, 1 - (top - 0.16) / height, "Crumpled input", fontsize=12.5, ha="center",
             va="center", color=INK, fontweight="bold")
    for j, m in enumerate(methods):
        fig.text((label_w + input_w + cell_w * (j + 0.5)) / width, 1 - (top - 0.2) / height, PLAIN_NAMES.get(m, m),
                 fontsize=12, ha="center", va="center", color=INK, fontweight="bold", linespacing=1.15)

    for i, name in enumerate(sheets):
        y = top + cell_h * i
        first = detail[f"{name} / {methods[0]}"]["instance"]
        fig.text(0.2 / width, 1 - (y + cell_h * 0.42) / height, name.replace("_", " ").capitalize(), fontsize=13,
                 fontweight="bold", color=INK, va="center")
        fig.text(0.2 / width, 1 - (y + cell_h * 0.42 + 0.22) / height, textwrap.fill(first.sheet.note, 21), fontsize=9.5,
                 color=MUTED, va="top", linespacing=1.3)
        ax = fig.add_axes(box(label_w, y, input_w, cell_h - 0.1), projection="3d")
        ax.scatter(*first.xyz.T, c=print_colours(first.uv), s=2.2, linewidths=0, depthshade=False, rasterized=True)
        ax.view_init(elev=26, azim=-58)
        ax.set_box_aspect(box_shape(first.xyz))
        ax.set_axis_off()
        for j, m in enumerate(methods):
            test = f"{name} / {m}"
            row, r = detail[test], summary[test]["worst"]
            ax = fig.add_axes(box(label_w + input_w + cell_w * j + 0.08, y + 0.02, cell_w - 0.16, cell_w - 0.16))
            ax.set_xlim(-VIEW, VIEW)
            ax.set_ylim(-VIEW, VIEW)
            ax.set_aspect("equal")
            _style(ax)
            if row["flat"] is None:
                ax.text(0, 0, "crashed", ha="center", va="center", fontsize=12, color=FOLD)
            else:
                inst, flat = row["instance"], row["flat"]
                aligned, _ = align_to_square(flat, inst.uv)
                miss = np.linalg.norm(aligned - inst.uv, axis=1) / 2 * 100
                T = triangles(inst.grid, 2)
                if paper_colours:
                    ax.add_collection(PolyCollection(aligned[T], facecolors=print_colours(inst.uv[T].mean(1)),
                                                     edgecolor="none", rasterized=True, zorder=2))
                else:
                    ax.add_collection(PolyCollection(aligned[T], array=miss[T].mean(1), cmap=ERROR_CMAP,
                                                     clim=(0, ERROR_MAX), edgecolor="none", rasterized=True, zorder=2))
                area = signed_area(aligned, T)
                paper = signed_area(inst.uv, T)
                flip = (area < 0) if paper[area > 0].sum() >= paper[area < 0].sum() else (area > 0)
                if flip.any():
                    ax.add_collection(PolyCollection(aligned[T[flip]], facecolor=FOLD, edgecolor=FOLD, lw=0.25,
                                                     rasterized=True, zorder=3))
                ax.add_collection(LineCollection(_graph_paper(aligned, inst.grid), colors="#0b0b0b", linewidths=0.35,
                                                 alpha=0.2 if paper_colours else 0.45, zorder=4))
                if not paper_colours:
                    ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color="#8a8983", lw=1.1, zorder=5)
            words = summary[test]["words"]
            passed = summary[test]["passed"]
            ax.text(0.5, -0.045, "clean" if passed else words, transform=ax.transAxes, ha="center", va="top", fontsize=12,
                    fontweight="bold", color="#008300" if passed else FOLD)
            if row["flat"] is not None:
                ax.text(0.5, -0.145, (f"lengths {r['length_ratio_p5']:.1f}× to {r['length_ratio_p95']:.1f}×" if paper_colours else
                                      f"off {r['square_error_median']:.1f}% typical, {r['square_error_worst']:.0f}% worst"),
                        transform=ax.transAxes, ha="center", va="top", fontsize=9.5, color=MUTED)
    fig.savefig(path, dpi=105, facecolor="white")
    plt.close(fig)
