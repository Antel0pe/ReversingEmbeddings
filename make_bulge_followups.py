"""Three things the first bulge figures showed but did not explain.

Looking at bulge_size.png and bulge_direction.png raises three questions, and all
three have measurable answers:

  1. BITES.     At high bulge the strokes look like something took bites out of
                their sides. Those are pixels the arithmetic sent NEGATIVE. The
                state space is not R^784, it is the unit cube [0,1]^784, and an
                arc of any real curvature walks straight out of it. The display
                clips, and the clipped region is the bite.

  2. THE STALL. The strokes barely rotate until the final frame. Progress ALONG
                the chord is exactly uniform for every bulge -- provably, because
                the control point sits exactly over the midpoint -- so the stall
                is not in the parameterisation. It is that the perpendicular
                excursion swamps the image, and only deflates at the very end.

  3. THE AXIS.  Walking the bulge direction once around the circle, only two
                narrow windows produce a connected image -- about +/-20 degrees
                around the manifold's own direction, and the same around its exact
                negation. One end of that axis is a 1. The other is a clean X,
                both strokes at once. Everything else shatters into 30-80 fragments
                of confetti, and over 200 random bearings drawn from the full
                782-dimensional perpendicular space, not one fell below 53 pieces.

Writes figures/bulge_bites.png, bulge_rotation.png, bulge_axis.png.
"""

import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import label

import paths
from make_bulge_figures import setup, tilt

OUT = "figures"
GREEN, RED, PURPLE, GREY = "#1a7f37", "#b3261e", "#6a1b9a", "#444"


def bite_overlay(ax, raw):
    """Draw the image as displayed, then mark in red the pixels clipped below 0."""
    ax.imshow(np.clip(raw, 0, 1).reshape(28, 28), cmap="gray", vmin=0, vmax=1)
    neg = (raw.reshape(28, 28) < -1e-9).astype(float)
    ax.imshow(np.ma.masked_where(neg == 0, neg), cmap=ListedColormap([RED]),
              vmin=0, vmax=1, alpha=0.85)
    ax.set_xticks([]); ax.set_yticks([])


def blobs(img):
    return int(label(np.clip(img, 0, 1).reshape(28, 28) > 0.2)[1])


def main():
    ones, D, theta, ai, bi, route, near = setup()
    A, B = ones[ai], ones[bi]
    s = paths.arclength_param(route)
    mid = (A + B) / 2
    gm = np.array([np.interp(0.5, s, route[:, j]) for j in range(784)])
    h0, u0 = paths.bulge_of(A, B, gm)
    e = (B - A) / np.linalg.norm(B - A)
    NC = 9
    ts = np.linspace(0, 1, NC)
    arc = lambda f: paths.arc_through(A, B, mid + f * h0 * u0, ts)
    real = np.array([[np.interp(v, s, route[:, j]) for j in range(784)] for v in ts])
    print(f"h0 = {h0:.2f}   chord = {D[ai,bi]:.2f}   THETA = {theta:.2f}")

    # =====================================================================
    # FIGURE 1 -- the bites are ink the arc drove negative
    # =====================================================================
    rows = [("REAL MANIFOLD ROUTE\nnever leaves the cube", real, GREEN, True),
            ("h = 0\nstraight line", arc(0.0), RED, True)]
    rows += [(f"h = {f:g} x h0", arc(f), "#333", False) for f in [1.0, 1.5, 2.5, 4.0, 6.0]]
    R = len(rows)
    FW, FH = 1.62 * NC + 3.2, 1.78 * R + 5.0
    fig = plt.figure(figsize=(FW, FH))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.78 * R, 4.6], hspace=0.14,
                             left=0.205, right=0.985, top=0.895, bottom=0.055)
    gs = outer[0].subgridspec(R, NC, wspace=0.06, hspace=0.32)
    for r, (name, F, colour, heavy) in enumerate(rows):
        for c in range(NC):
            ax = fig.add_subplot(gs[r, c])
            bite_overlay(ax, F[c])
            nm = -np.minimum(F[c], 0).sum()
            ax.text(0.5, -0.11, "--" if nm < 1e-6 else f"{nm:.1f}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=9, weight="bold",
                    color=GREEN if nm < 1e-6 else RED)
            for sp in ax.spines.values():
                sp.set_edgecolor(colour); sp.set_linewidth(1.6 if heavy else 0.9)
            if r == 0:
                ax.set_title(f"t = {ts[c]:.2f}", fontsize=10, pad=7)
            if c == 0:
                ax.text(-0.16, 0.5, name, transform=ax.transAxes, ha="right", va="center",
                        fontsize=10.5, color=colour, linespacing=1.5,
                        weight="bold" if heavy else "normal")

    ax = fig.add_subplot(outer[1])
    FS = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 4.0, 6.0, 10.0]
    tt = np.linspace(0, 1, 21)
    negm, overm, frac = [], [], []
    for f in FS:
        Fa = paths.arc_through(A, B, mid + f * h0 * u0, tt)
        lost = (-np.minimum(Fa, 0) + np.maximum(Fa - 1, 0)).sum(1)
        negm.append(-np.minimum(Fa, 0).sum(1).mean())
        overm.append(np.maximum(Fa - 1, 0).sum(1).mean())
        frac.append((lost / np.clip(Fa, 0, 1).sum(1)).mean())
    ax.plot(FS, negm, "o-", lw=2.6, ms=8, color=RED, label="ink driven below 0 (the bites)")
    ax.plot(FS, overm, "s-", lw=2.6, ms=8, color=PURPLE, label="ink driven above 1 (blown highlights)")
    ax.axhline(0, ls="--", lw=2, color=GREEN)
    ax.text(10.2, 0, "  real manifold route:\n  exactly zero, always",
            fontsize=10.5, color=GREEN, va="center")
    ax.set_xlabel("bulge, in multiples of h0", fontsize=12)
    ax.set_ylabel("mass outside [0, 1]\nper frame", fontsize=11.5)
    ax.set_xticks(FS); ax.set_xticklabels([f"{f:g}" for f in FS])
    ax.grid(alpha=.3); ax.legend(loc="upper left", fontsize=11)
    ax.set_title("How far outside the unit cube each arc goes "
                 f"(at h = 6 x h0 the display throws away {frac[7]:.1f}x more mass than it shows)",
                 fontsize=12.5, pad=10)
    fig.text(0.5, 0.968, "THE BITES ARE NEGATIVE INK:  the state space is a cube, and arcs leave it",
             ha="center", fontsize=16, weight="bold")
    fig.text(0.5, 0.940, "Every pixel must lie in [0, 1]. A curved arc through 784-D space does not "
                         "know that. Red marks pixels the arithmetic sent below zero.",
             ha="center", fontsize=11.5)
    fig.text(0.5, 0.916, "The number under each frame is how much ink was clipped away. "
                         "The real route never needs clipping at all.",
             ha="center", fontsize=10.5, style="italic", color=GREY)
    fig.savefig(f"{OUT}/bulge_bites.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote bulge_bites.png")
    print("  negative mass per frame:", " ".join(f"{f:g}x={m:.1f}" for f, m in zip(FS, negm)))

    # =====================================================================
    # FIGURE 2 -- the stall, and why it is not in the parameterisation
    # =====================================================================
    FS2 = [0.0, 1.0, 2.5, 6.0]
    cols = ["#b3261e", "#1565c0", "#ef6c00", "#6a1b9a"]
    FW, FH = 15.0, 11.6
    fig = plt.figure(figsize=(FW, FH))
    # explicit bands; the main column stops at 0.775 so the side panel cannot collide
    L, Rt = 0.085, 0.775
    gs_strip = fig.add_gridspec(1, 1, left=L, right=Rt, top=0.845, bottom=0.715)
    gs_tilt  = fig.add_gridspec(1, 1, left=L, right=Rt, top=0.625, bottom=0.385)
    gs_chord = fig.add_gridspec(1, 1, left=L, right=Rt, top=0.295, bottom=0.070)

    sub = gs_strip[0].subgridspec(1, NC, wspace=0.06)
    F6 = arc(6.0)
    for c in range(NC):
        ax = fig.add_subplot(sub[0, c])
        ax.imshow(np.clip(F6[c], 0, 1).reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t = {ts[c]:.2f}", fontsize=9.5, pad=5)
        tv = tilt(np.clip(F6[c], 0, 1).reshape(28, 28))
        ax.text(0.5, -0.12, f"{tv:+.0f}°", transform=ax.transAxes, ha="center", va="top",
                fontsize=10, weight="bold", color=PURPLE)
        if c == 0:
            ax.text(-0.14, 0.5, "h = 6 x h0\nthe stall, visually", transform=ax.transAxes,
                    ha="right", va="center", fontsize=10.5, color=PURPLE, weight="bold")

    ax = fig.add_subplot(gs_tilt[0])
    ax.plot(ts, [tilt(np.clip(x, 0, 1).reshape(28, 28)) for x in real], "o-", lw=3.2, ms=9,
            color=GREEN, label="REAL MANIFOLD ROUTE", zorder=5)
    for f, c in zip(FS2, cols):
        ax.plot(ts, [tilt(np.clip(x, 0, 1).reshape(28, 28)) for x in arc(f)], "o--", lw=2.2, ms=7,
                color=c, label=f"h = {f:g} x h0" + (" (straight line)" if f == 0 else ""))
    ax.axhline(tilt(A.reshape(28, 28)), ls=":", lw=1.5, color=GREY)
    ax.axhline(tilt(B.reshape(28, 28)), ls=":", lw=1.5, color=GREY)
    ax.text(1.015, tilt(A.reshape(28, 28)), " A", fontsize=11, color=GREY, va="center")
    ax.text(1.015, tilt(B.reshape(28, 28)), " B", fontsize=11, color=GREY, va="center")
    ax.set_ylabel("measured tilt of the frame\n(degrees)", fontsize=11.5)
    ax.set_xlabel("t", fontsize=11.5); ax.grid(alpha=.3); ax.set_xlim(-0.02, 1.05)
    ax.legend(loc="upper left", fontsize=10.5, ncol=2)
    ax.set_title("WHAT YOU SEE: the tilt stalls, then snaps. The bigger the bulge, the flatter "
                 "the stall and the harder the snap.", fontsize=12.5, pad=10)

    ax = fig.add_subplot(gs_chord[0])
    for f, c in zip(FS2, cols):
        F = arc(f)
        ax.plot(ts, [np.dot(x - A, e) / np.linalg.norm(B - A) for x in F], "o-", lw=2.4, ms=8,
                color=c, label=f"h = {f:g} x h0")
    ax.plot(ts, ts, "-", lw=6, color="#ccc", zorder=0, label="perfectly uniform")
    ax.set_ylabel("progress along the chord\nA -> B", fontsize=11.5)
    ax.set_xlabel("t", fontsize=11.5); ax.grid(alpha=.3); ax.set_xlim(-0.02, 1.05)
    ax.legend(loc="upper left", fontsize=10.5, ncol=2)
    ax.set_title("WHY IT IS NOT THE TIMING: every bulge advances along the chord at exactly the "
                 "same uniform rate. All five lines coincide.", fontsize=12.5, pad=10)

    axr = fig.add_axes([0.845, 0.385, 0.140, 0.240])
    for f, c in zip(FS2, cols):
        axr.plot(ts, [np.linalg.norm(x - B) for x in arc(f)], "o-", lw=2.2, ms=5,
                 color=c, label=f"{f:g}x")
    axr.plot([0.875], [np.linalg.norm(arc(6.0)[7] - B)], "o", ms=13, mfc="none",
             mec=PURPLE, mew=2.2)
    axr.plot([0.875], [np.linalg.norm(arc(0.0)[7] - B)], "o", ms=13, mfc="none",
             mec=RED, mew=2.2)
    axr.set_title("THE FOLLOW-UP:\ndistance to B", fontsize=11, pad=8, weight="bold")
    axr.set_xlabel("t", fontsize=10); axr.grid(alpha=.3); axr.tick_params(labelsize=9)
    axr.legend(fontsize=8.5, loc="upper right", title="bulge", title_fontsize=8.5)
    fig.text(0.915, 0.355, "The circled frame is the same\nstep (t = 0.88) on every route.\n"
             "It sits 1.1 from B at h = 0 and\n15.0 from B at h = 6 x h0 --\nso yes, the bulge "
             "changes how\nfar a given step has left to go.",
             ha="center", va="top", fontsize=9.5, color=GREY, linespacing=1.6)

    fig.text(0.5, 0.965, "THE STALL:  why a bulged arc barely rotates until the last frame",
             ha="center", fontsize=16, weight="bold")
    fig.text(0.5, 0.937, "The arc is not slowing down. It is being dragged so far sideways that the "
                         "bulge, not the stroke, is what you are looking at -- until it deflates.",
             ha="center", fontsize=11.5)
    fig.text(0.5, 0.913, "Which also answers the follow-up: yes, the same frame sits much further "
                         "from B under a harsher bulge (right-hand panel).",
             ha="center", fontsize=10.5, style="italic", color=GREY)
    fig.savefig(f"{OUT}/bulge_rotation.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote bulge_rotation.png")
    for f in FS2:
        print(f"  h={f:g}x tilt:", " ".join(f"{tilt(np.clip(x,0,1).reshape(28,28)):+6.1f}" for x in arc(f)))

    # =====================================================================
    # FIGURE 3 -- the +/-u0 axis: a 1 at one end, an X at the other
    # =====================================================================
    rng = np.random.default_rng(1)
    v = rng.normal(size=784); v -= np.dot(v, e) * e; v -= np.dot(v, u0) * u0; v /= np.linalg.norm(v)
    U = lambda p: np.cos(np.radians(p)) * u0 + np.sin(np.radians(p)) * v
    half = lambda p: paths.arc_through(A, B, mid + h0 * U(p), [0.5])[0]
    ANG = np.arange(0, 360, 15)
    nb = np.array([blobs(half(p)) for p in ANG])
    dq = np.array([near([half(p)])[0] for p in ANG])
    Q, _ = np.linalg.qr(np.stack([A, B], 1))
    sAB = np.where((A > 0.2) | (B > 0.2))[0]

    FW, FH = 15.0, 13.4
    fig = plt.figure(figsize=(FW, FH))
    # explicit bands, so nothing can drift into anything else
    gs_ref  = fig.add_gridspec(1, 1, left=0.075, right=0.975, top=0.838, bottom=0.700)
    gs_axis = fig.add_gridspec(1, 1, left=0.075, right=0.975, top=0.612, bottom=0.446)
    gs_bot  = fig.add_gridspec(1, 1, left=0.075, right=0.975, top=0.330, bottom=0.060)

    refs = [("A\n(real, -35°)", A, GREEN), ("midpoint\n(h = 0)", mid, RED),
            ("real manifold\nhalfway image", gm, GREEN), ("B\n(real, +10°)", B, GREEN)]
    sub = gs_ref[0].subgridspec(1, 9, wspace=0.10)
    for k, (nm, img, c) in enumerate(refs):
        ax = fig.add_subplot(sub[0, k + (1 if k < 2 else 2)])
        ax.imshow(np.clip(img, 0, 1).reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(nm, fontsize=10.5, color=c, pad=6)
        ax.text(0.5, -0.09, f"{blobs(img)} piece", transform=ax.transAxes, ha="center",
                va="top", fontsize=10, color=GREY)
        for sp in ax.spines.values():
            sp.set_edgecolor(c); sp.set_linewidth(1.8)
    fig.text(0.5, 0.872, "REFERENCE:  the four images every other panel is measured against. "
             "Note the midpoint is ALREADY a faint X -- a left-leaning 1 crossed with a "
             "right-leaning one.", ha="center", fontsize=12, weight="bold", color="#222")

    FF = [2.0, 1.0, 0.5, 0.0, -0.5, -1.0, -2.0]
    sub = gs_axis[0].subgridspec(1, len(FF), wspace=0.10)
    for k, f in enumerate(FF):
        img = paths.arc_through(A, B, mid + f * h0 * u0, [0.5])[0]
        ax = fig.add_subplot(sub[0, k])
        ax.imshow(np.clip(img, 0, 1).reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        lbl = {1.0: "0°\nthe manifold's\nown direction", 0.0: "the midpoint\n(no bulge)",
               -1.0: "180°\nexactly negated"}.get(f, f"{f:+g} x h0")
        c = GREEN if f == 1.0 else (PURPLE if f == -1.0 else GREY)
        ax.set_title(lbl, fontsize=10, color=c, pad=6,
                     weight="bold" if f in (1.0, -1.0) else "normal")
        ax.text(0.5, -0.09, f"d = {near([img])[0]:.1f}", transform=ax.transAxes, ha="center",
                va="top", fontsize=10, weight="bold",
                color=GREEN if near([img])[0] <= theta else RED)
        for sp in ax.spines.values():
            sp.set_edgecolor(c); sp.set_linewidth(2.4 if f in (1.0, -1.0) else 0.9)
    fig.text(0.5, 0.652, "ONE AXIS, WALKED END TO END:  a legible 1 at one end, a hard-edged X at "
             "the other, the faint ghost in the middle", ha="center", fontsize=13, weight="bold",
             color="#222")
    fig.text(0.5, 0.398, "Walking towards 180 degrees does not invent the X -- it sharpens the one "
             "the midpoint already had. That is why it looks clean and is still 5.7 from any real 1.",
             ha="center", fontsize=11, style="italic", color=GREY)

    bot = gs_bot[0].subgridspec(1, 2, width_ratios=[3.5, 1.0], wspace=0.34)
    ax = fig.add_subplot(bot[0, 0])
    for centre in (0, 180):
        ax.axvspan(centre - 21, centre + 21, color=GREEN if centre == 0 else PURPLE,
                   alpha=0.10, zorder=0)
    ax.bar(ANG, nb, width=11, color=[GREEN if p == 0 else (PURPLE if p == 180 else "#bbb")
                                            for p in ANG], zorder=3)
    ax.axhline(1.06, ls="--", lw=2, color=GREEN)
    ax.text(362, 1.06, "  real 1s: 1.06", fontsize=10.5, color=GREEN, va="bottom")
    for p, n in zip(ANG, nb):
        if p in (0, 180):
            ax.annotate(f"{n} piece\n{'a 1' if p == 0 else 'an X'}", xy=(p, n),
                        xytext=(p, 26 if p == 0 else 30), ha="center", fontsize=11,
                        weight="bold", color=GREEN if p == 0 else PURPLE,
                        arrowprops=dict(arrowstyle="->", lw=1.8,
                                        color=GREEN if p == 0 else PURPLE))
    ax.axvspan(339, 358, color=GREEN, alpha=0.10, zorder=0)
    ax.set_xticks(range(0, 360, 45)); ax.set_xlim(-12, 358)
    ax.set_xlabel("angle of the bulge direction, at constant bulge length "
                  f"{h0:.1f} (the same circle as bulge_direction.png)", fontsize=11.5)
    ax.set_ylabel("connected ink pieces\nin the halfway image", fontsize=11.5)
    ax.grid(alpha=.3, axis="y")
    ax.set_title("THE TEST: how many separate pieces of ink the halfway image breaks into "
                 "(shaded = the two connected windows, each about +/-20 degrees wide)",
                 fontsize=12, pad=10)
    ax2 = ax.twinx()
    ax2.plot(ANG, dq, "o-", lw=2.0, ms=6, color=RED, alpha=.75, zorder=4)
    ax2.set_ylabel("distance to nearest real 1", color=RED, fontsize=11)
    ax2.tick_params(axis="y", labelcolor=RED); ax2.set_ylim(0, 7.2)

    axh = fig.add_subplot(bot[0, 1])
    rb = []
    for _ in range(200):
        w = rng.normal(size=784); w -= np.dot(w, e) * e; w /= np.linalg.norm(w)
        rb.append(blobs(paths.arc_through(A, B, mid + h0 * w, [0.5])[0]))
    rb = np.array(rb)
    axh.hist(rb, bins=16, color="#bbb", edgecolor="#888", zorder=3)
    axh.axvline(1, lw=3, color=GREEN)
    axh.text(1, axh.get_ylim()[1] * 0.97, " the axis\n (1 piece)", fontsize=10, color=GREEN,
             va="top", weight="bold")
    axh.set_xlabel("connected pieces", fontsize=10.5)
    axh.set_ylabel("count", fontsize=10.5)
    axh.tick_params(labelsize=9); axh.grid(alpha=.3, axis="y")
    axh.set_title("CONTROL: 200 random bearings\nfrom the full 782-D perpendicular\n"
                  f"space. Best of them: {rb.min()} pieces.", fontsize=10.5, pad=8)

    fig.text(0.5, 0.968, "THE AXIS:  of every direction on the circle, only two narrow windows "
             "draw a connected stroke", ha="center", fontsize=15.5, weight="bold")
    fig.text(0.5, 0.940, "About +/-20 degrees around the manifold's own bulge direction, and the "
                         "same around its exact opposite. Everything between them is confetti.",
             ha="center", fontsize=11.5)
    fig.text(0.5, 0.916, "The X is not new structure: 98% of its ink sits on pixels A or B already "
                         "touched. It is the two strokes drawn at once.",
             ha="center", fontsize=10.5, style="italic", color=GREY)
    fig.savefig(f"{OUT}/bulge_axis.png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote bulge_axis.png")

    m180 = mid - h0 * u0
    off = lambda i: 1 - np.clip(i, 0, 1)[sAB].sum() / np.clip(i, 0, 1).sum()
    ins = lambda i: np.linalg.norm(Q @ (Q.T @ i)) / np.linalg.norm(i)
    print(f"\n  180 deg: {blobs(m180)} piece, {100*off(m180):.1f}% of its ink off the A/B strokes, "
          f"{100*ins(m180):.0f}% inside span(A,B), d = {near([m180])[0]:.2f}")
    print(f"  0   deg: {blobs(half(0))} piece, {100*off(half(0)):.1f}% off-stroke, "
          f"d = {near([half(0)])[0]:.2f}")
    print(f"  other angles: {nb[(ANG!=0)&(ANG!=180)].min()}-{nb[(ANG!=0)&(ANG!=180)].max()} pieces")


if __name__ == "__main__":
    main()
