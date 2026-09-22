"""Push a plain 1 outward in five directions, three different ways, and watch what survives.

  INTRINSIC  follow each knob: the image at step t is render(centre + t * knob_step).
             The path bends with the manifold.
  STRAIGHT   the same five neighbours (one knob-step away, distance theta = 1.89),
             but keep going along the STRAIGHT line centre -> neighbour:
             centre + t * (neighbour - centre). At t = 1 identical to INTRINSIC.
  LINEAR     straight along the top five of the 214 principal axes of the whole
             manifold: centre + t * 1.89 * PC_k.

Every image carries rho: the distance from what you see (pixels clipped to [0, 1])
to the nearest image the generator can draw with ANY knob values -- found by
reading the knobs off in closed form (invert_grey.py) and polishing with a few
least-squares steps. rho = 0 means it is still a stroke. Pixels pushed above 1 are
tinted red, below 0 blue: they cannot be displayed, and the render would otherwise
hide them.

  push_rings.gif  -- the animation
  push_rings.png  -- the same thing as a static table, steps 0..4

Run: python make_push_gif.py
"""
import matplotlib
matplotlib.use("Agg")
import numpy as np, matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import least_squares
import grey_ones as G
from invert_grey import invert

OUT = "figures"
INK, GOOD, BAD, REF = "#1a1a1a", "#1a7f37", "#c62828", "#1565c0"
THETA, EPS8 = 1.89, 0.11
CENTRE = np.array([14.5, 14.5, 19.75, 3.2, 0.0])
SIGN = np.array([+1, -1, -1, +1, +1])          # the direction with room on the canvas
T_MAX = 4.0
TS = np.round(np.arange(0, T_MAX + 1e-9, 0.1), 2)
LO_B = np.array([1.0, 1.0, 0.5, 0.2, -75.0]); HI_B = np.array([27.0, 27.0, 27.5, 14.0, 75.0])
ARROW = {1: "→", -1: "←"}
LABEL = ["cx →", "cy ↑", "height ↓", "width ↑", "lean ↗"]


def R(p):
    return G.render(np.atleast_2d(p)).reshape(-1, 784).astype(np.float64)


def knob_steps(x0):
    """Per-knob step that moves the image exactly THETA away from the centre."""
    out = np.zeros(5)
    for k in range(5):
        lo, hi = 0.0, 20.0
        for _ in range(60):
            m = (lo + hi) / 2; p = CENTRE.copy(); p[k] += SIGN[k] * m
            lo, hi = (m, hi) if np.linalg.norm(R(p)[0] - x0) < THETA else (lo, m)
        out[k] = SIGN[k] * m
    return out


def rho(x, starts):
    """Distance from image x to the nearest stroke the generator can draw."""
    best = np.linalg.norm(x)                      # the empty image is a lower bar
    try:
        starts = list(starts) + [invert(np.clip(x, 0, 1), full_tol=5e-2)]
    except Exception:
        pass
    for s in starts:
        s = np.clip(s, LO_B + 1e-6, HI_B - 1e-6)
        if not np.all(np.isfinite(s)): continue
        r = least_squares(lambda p: R(p)[0] - x, s, bounds=(LO_B, HI_B), diff_step=1e-3,
                          max_nfev=60)
        best = min(best, float(np.linalg.norm(r.fun)))
    return best


def main():
    x0 = R(CENTRE)[0]
    step = knob_steps(x0)
    rng = np.random.default_rng(0)
    S = R(G.sample(20000, rng)); Z = S - S.mean(0)
    V = np.linalg.svd(Z, full_matrices=False)[2][:5]
    nbr = np.array([R(CENTRE + np.eye(5)[k] * step[k])[0] for k in range(5)])

    rings = ["INTRINSIC — follow the knob", "STRAIGHT — follow the line to the neighbour",
             "LINEAR — follow a principal axis"]
    names = [LABEL, LABEL, [f"PC {k+1}" for k in range(5)]]
    imgs = np.zeros((3, 5, len(TS), 784)); knobs_at = np.zeros((5, len(TS), 5))
    for ti, t in enumerate(TS):
        for k in range(5):
            p = CENTRE + np.eye(5)[k] * step[k] * t; knobs_at[k, ti] = p
            imgs[0, k, ti] = R(p)[0]
            imgs[1, k, ti] = x0 + t * (nbr[k] - x0)
            imgs[2, k, ti] = x0 + t * THETA * V[k]
    RHO = np.zeros((3, 5, len(TS))); ESC = np.zeros((3, 5, len(TS))); MOV = np.zeros((3, 5, len(TS)))
    for r in range(3):
        for k in range(5):
            for ti in range(len(TS)):
                x = imgs[r, k, ti]; xc = np.clip(x, 0, 1)
                ESC[r, k, ti] = np.abs(x - xc).sum()
                MOV[r, k, ti] = np.linalg.norm(xc - x0)          # how far the VISIBLE image got
                starts = [CENTRE, knobs_at[k, ti]]
                RHO[r, k, ti] = rho(xc, starts) if TS[ti] > 0 else 0.0
    np.savez(f"{OUT}/push_rings_data.npz", imgs=imgs, rho=RHO, esc=ESC, ts=TS, step=step)

    # ------------------------------------------------------------------ layout
    W, H = 15.6, 6.9
    fig = plt.figure(figsize=(W, H))
    fig.text(.5, .955, "Push a plain 1 outward five ways — along the knobs, along straight lines, along principal axes",
             ha="center", va="center", fontsize=14.5, fontweight="bold")
    fig.text(.5, .905, "Every ring starts from the same upright 1. ρ under each image = its distance from the nearest stroke the "
                       "generator can draw (0 = still a stroke; 0.11 = invisible at 8-bit).\n"
                       "“moved” = how far the visible image has got from the start. Red tint = pushed above full black, blue = below white. "
                       "At step 1 the first two rings are identical: they only part ways after that.",
             ha="center", va="center", fontsize=9.2, color="#555", linespacing=1.5)
    step_txt = fig.text(.5, .052, "", ha="center", va="center", fontsize=12, fontweight="bold")
    fig.text(.5, .018, f"One step = the distance to a nearest neighbour, θ = {THETA}. "
                       f"Knob directions: cx +{step[0]:.2f} px, cy {step[1]:+.2f} px, height {step[2]:+.2f} px, "
                       f"width +{step[3]:.2f} px, lean +{step[4]:.2f}° per step.",
             ha="center", va="center", fontsize=8.6, color="#777")
    size = .066; size_y = size * W / H
    RX = .105; RY = RX * W / H
    handles = []
    for r in range(3):
        cxf, cyf = .175 + r * .325, .46
        fig.text(cxf, .815, rings[r], ha="center", va="center", fontsize=11.5, fontweight="bold",
                 color=GOOD if r == 0 else INK)
        slots = [(cxf, cyf)] + [(cxf + RX * np.sin(2 * np.pi * k / 5), cyf + RY * np.cos(2 * np.pi * k / 5) * .93)
                                for k in range(5)]
        row = []
        for s_i, (px, py) in enumerate(slots):
            ax = fig.add_axes([px - size / 2, py - size_y / 2, size, size_y])
            ax.set_xticks([]); ax.set_yticks([])
            im = ax.imshow(np.zeros((28, 28, 3)), interpolation="nearest")
            for sp in ax.spines.values():
                sp.set_edgecolor(REF if s_i == 0 else "#bbb"); sp.set_linewidth(2 if s_i == 0 else .7)
            if s_i == 0:
                ax.text(.5, -.08, "start", transform=ax.transAxes, ha="center", va="top",
                        fontsize=8, color=REF)
                row.append((im, None, None)); continue
            k = s_i - 1
            ax.text(.5, 1.05, names[r][k], transform=ax.transAxes, ha="center", va="bottom",
                    fontsize=8.4, fontweight="bold", color=INK)
            t1 = ax.text(.5, -.07, "", transform=ax.transAxes, ha="center", va="top",
                         fontsize=8.4, fontweight="bold")
            row.append((im, t1, k))
        handles.append(row)

    def rgb(x):
        xc = np.clip(x, 0, 1).reshape(28, 28)
        g = 1 - xc
        out = np.stack([g, g, g], -1)
        over = np.clip(x - 1, 0, None).reshape(28, 28); under = np.clip(-x, 0, None).reshape(28, 28)
        a = np.clip(over / .5, 0, 1)[..., None]; b = np.clip(under / .5, 0, 1)[..., None]
        out = out * (1 - a) + np.array([.85, .12, .12]) * a
        out = out * (1 - b) + np.array([.15, .35, .9]) * b
        return out

    def draw(ti):
        for r in range(3):
            for im, t1, k in handles[r]:
                if k is None:
                    im.set_data(rgb(x0)); continue
                im.set_data(rgb(imgs[r, k, ti]))
                v = RHO[r, k, ti]
                t1.set_text(f"ρ {v:.2f}\nmoved {MOV[r, k, ti]:.1f}"); t1.set_color(GOOD if v < EPS8 else BAD)
        step_txt.set_text(f"step {TS[ti]:.1f}   —   {TS[ti]*THETA:.2f} out along each straight line")

    frames, durs = [], []
    for ti in range(len(TS)):
        draw(ti); fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        hold = 1400 if TS[ti] in (0.0, 1.0) else 2600 if ti == len(TS) - 1 else 110
        durs.append(hold)
    pal = frames[-1].quantize(colors=96, method=Image.Quantize.MEDIANCUT)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    q[0].save(f"{OUT}/push_rings.gif", save_all=True, append_images=q[1:], duration=durs, loop=0,
              optimize=True)
    plt.close(fig)

    # ------------------------------------------------------------------ static table
    tsel = [0, 10, 20, 30, 40]
    fig = plt.figure(figsize=(12.8, 12.6))
    fig.text(.5, .975, "The same pushes as a table: which directions are still a stroke?",
             ha="center", va="center", fontsize=15, fontweight="bold")
    fig.text(.5, .946, "Columns: steps 0–4 (one step = θ = 1.89). ρ = distance to the nearest drawable stroke; "
                       "green under 0.11 (invisible), red over.\n“moved” = how far the VISIBLE image has got from the start. "
                       "Red/blue tint = ink above 1 / below 0, which the picture cannot show.",
             ha="center", va="center", fontsize=9.4, color="#555")
    L, T, B = .17, .925, .02
    rh = (T - B) / 15
    groups = ["INTRINSIC", "STRAIGHT", "LINEAR"]
    for r in range(3):
        for k in range(5):
            row_i = r * 5 + k; y0 = T - (row_i + 1) * rh
            for c, ti in enumerate(tsel):
                ax = fig.add_axes([L + c * .155, y0 + rh * .20, .05 * 12.6 / 12.8 * 1.0, rh * .74])
                ax.imshow(rgb(imgs[r, k, ti]), interpolation="nearest")
                ax.set_xticks([]); ax.set_yticks([])
                for sp in ax.spines.values(): sp.set_edgecolor("#bbb"); sp.set_linewidth(.6)
                v = RHO[r, k, ti]
                ax.text(1.12, .5, f"ρ {v:.2f}\nmoved {MOV[r, k, ti]:.1f}", transform=ax.transAxes,
                        ha="left", va="center", fontsize=8.4, fontweight="bold", linespacing=1.5,
                        color=GOOD if v < EPS8 else BAD)
                if row_i == 0:
                    ax.text(.5, 1.12, f"step {TS[ti]:.0f}", transform=ax.transAxes, ha="center",
                            va="bottom", fontsize=9.5, fontweight="bold")
            fig.text(L - .012, y0 + rh * .57, f"{groups[r]}  {names[r][k]}", ha="right", va="center",
                     fontsize=9.6, fontweight="bold", color=GOOD if r == 0 else INK)
        if r < 2:
            yline = T - (r + 1) * 5 * rh + rh * .08
            fig.add_artist(plt.Line2D([.03, .97], [yline, yline], color="#ccc", lw=1))
    fig.savefig(f"{OUT}/push_rings.png", dpi=120)
    plt.close(fig)

    print("rho at step 1 / 2 / 4   |  visible image moved at step 1 / 2 / 4")
    for r, g in enumerate(groups):
        for k in range(5):
            print(f"  {g:9s} {names[r][k]:10s}  " + "  ".join(f"{RHO[r,k,ti]:5.2f}" for ti in (10, 20, 40))
                  + "   |  " + "  ".join(f"{MOV[r,k,ti]:5.2f}" for ti in (10, 20, 40)))


if __name__ == "__main__":
    main()
