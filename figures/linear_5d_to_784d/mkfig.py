import numpy as np, matplotlib, os
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from grey_ones import KNOBS, render as _r

S = "/tmp/claude-1000/-home-dmmsp-Projects-ReversingEmbeddings/6425139a-2f92-49b6-a2a7-3d48ffbf0cae/scratchpad/"
OUT = "/home/dmmsp/Projects/ReversingEmbeddings/figures/linear_5d_to_784d/"
os.makedirs(OUT, exist_ok=True)
d = np.load(S+"fig.npz")
INK, DIV = "gray_r", "coolwarm"

def ink(fig, rect, img, title=None, ts=9):
    ax = fig.add_axes(rect); ax.imshow(img, cmap=INK, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=ts, pad=4)
    return ax

def diff(fig, rect, img, vmax=None, title=None, ts=9, cbar=True):
    ax = fig.add_axes(rect); v = vmax or np.abs(img).max()
    im = ax.imshow(img, cmap=DIV, vmin=-v, vmax=v, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    if title: ax.set_title(title, fontsize=ts, pad=4)
    if cbar:
        cax = fig.add_axes([rect[0]+rect[2]+0.004, rect[1], 0.007, rect[3]])
        fig.colorbar(im, cax=cax).ax.tick_params(labelsize=7)
    return ax, v

def mark_outside(ax, img):
    neg = np.argwhere(img < -0.02); over = np.argwhere(img > 1.02)
    if len(neg):  ax.plot(neg[:,1], neg[:,0], ".", ms=2.0, color="#00b4e0")
    if len(over): ax.plot(over[:,1], over[:,0], ".", ms=2.0, color="#e5179b")
    ax.set_xlim(-0.5,27.5); ax.set_ylim(27.5,-0.5)
    return len(neg), len(over)

def cap(ax, text, fs=7.6, dy=-0.07):
    ax.text(0.5, dy, text, transform=ax.transAxes, ha="center", va="top",
            fontsize=fs, linespacing=1.4)

# ================= FIGURE 1 =================
fig = plt.figure(figsize=(16.0, 13.0))
fig.text(0.5, 0.975, "Can one linear map send the 5 knobs to the 784 pixels of a generated 1?",
         ha="center", fontsize=16)
fig.text(0.5, 0.952, "Best-fit linear (affine) map, least squares on 20,000 renders spanning the full default knob box, scored on 5,000 held-out renders",
         ha="center", fontsize=11.5, color="#333")

W = d["W"]; H = 0.105; Y = 0.800
ax = ink(fig, [0.045, Y, 0.082, H], d["rootimg"], "root 1\n(14, 14, 19.75, 3.2, 0°)", 9)
cax = fig.add_axes([0.133, Y, 0.007, H])
cb = fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(0,1), cmap=INK), cax=cax)
cb.ax.tick_params(labelsize=7); cb.set_label("coverage", fontsize=7.5)
cap(ax, "one pixel = ink coverage\n0 = white (no ink), 1 = black (full ink)", 8)

axo, v = diff(fig, [0.195, Y, 0.082, H], W[5].reshape(28,28), None, "offset image  (constant term)")
cap(axo, f"own colour scale, ±{v:.1f}", 8)

for k in range(5):
    axk, v = diff(fig, [0.370 + k*0.122, Y, 0.082, H], W[k].reshape(28,28), None, f"× {KNOBS[k]}")
    cap(axk, f"±{v:.2f} per unit\n|column| = {np.linalg.norm(W[k]):.2f}", 7.6)

fig.text(0.045, 0.745,
 "What the fitted map is:   predicted image  =  offset  +  cx·(column cx)  +  cy·(column cy)  +  height·(column height)  +  width·(column width)  +  lean·(column lean)\n"
 "i.e. six fixed 28×28 images, mixed with weights read straight off the knobs. Red = that term adds ink, blue = removes ink; each panel has its own scale (printed under it)\n"
 "because the knobs carry different units (pixels, pixels, pixels, pixels, degrees). These six images are the best such set that exists for this data.",
 fontsize=10.2, va="top", linespacing=1.7)

# ---- held-out examples
Y2 = 0.525
labels = ["typical:  lean 12.5°", "extreme:  lean 34°, thin", "extreme:  wide, lean −9°"]
emax = max(np.abs(d["Xex"][i]-d["Pex"][i]).max() for i in range(3))
for i in range(3):
    t, p = d["Xex"][i], d["Pex"][i]
    x0 = 0.040 + i*0.300
    fig.text(x0, Y2+H+0.040, labels[i] + f"   ({', '.join(f'{x:g}' for x in d['ex'][i])})",
             fontsize=10.2, fontweight="bold")
    a1 = ink(fig, [x0, Y2, 0.078, H], t, "true render", 9.5); cap(a1, "what the renderer gives", 7.6)
    a2 = ink(fig, [x0+0.094, Y2, 0.078, H], p, "linear prediction", 9.5)
    nn, no = mark_outside(a2, p)
    cap(a2, f"{nn} px < 0 ink (cyan)\n{no} px > full ink (magenta)", 7.6)
    a3, _ = diff(fig, [x0+0.188, Y2, 0.078, H], p-t, emax, "error: pred − true", 9.5)
    cap(a3, f"scale ±{emax:.2f}; red = too much ink\nworst pixel {np.abs(p-t).max():.2f}", 7.6)

fig.text(0.045, 0.470,
 f"Held-out score of the best linear map over the whole knob box:   R² = {float(d['r2']):.4f}  —  it reproduces {float(d['r2'])*100:.1f}% of the image variation, not 99.9999%.\n"
 f"RMS pixel error {float(d['rms']):.3f} coverage units;  worst pixel error {float(d['max_abs']):.2f}, larger than the entire 0→1 ink scale.  {float(d['frac_neg'])*100:.1f}% of predicted pixels hold negative ink\n"
 f"and {float(d['frac_over'])*100:.1f}% hold more than full ink (predictions run from {float(d['pred_min']):.2f} to {float(d['pred_max']):.2f}); those are dotted above rather than hidden by the 0–1 grey display.",
 fontsize=10.2, va="top", linespacing=1.7, color="#9b0000")

# ---- bottom left: the ceiling
frac = d["frac"]; ks = np.arange(1, 785)
axA = fig.add_axes([0.055, 0.080, 0.385, 0.225])
axA.loglog(ks, np.maximum(1-frac, 1e-12), "-", lw=2, color="#1f4e79")
axA.axhline(1e-6, color="#d81b8c", ls="--", lw=1.4)
axA.text(1.15, 1.35e-6, "the asked-for 99.9999% accuracy", fontsize=9, color="#d81b8c", ha="left", va="bottom")
axA.plot([5],[1-frac[4]], "o", ms=12, mfc="none", mec="#c00", mew=2)
axA.annotate(f"k = 5, one number per knob:\nmisses {(1-frac[4])*100:.1f}% of the variation", xy=(5.4,1-frac[4]),
             xytext=(16, 0.30), fontsize=9.5, color="#c00",
             arrowprops=dict(arrowstyle="->", color="#c00", lw=1.3))
need = int(np.searchsorted(frac, 0.999999)+1)
axA.axvline(need, color="#2e7d32", ls=":", lw=1.8)
axA.annotate(f"k = {need} fixed images are needed\nbefore 99.9999% is reached", xy=(need, 1.1e-6),
             xytext=(1.4, 1.5e-5), fontsize=9.5, color="#2e7d32",
             arrowprops=dict(arrowstyle="->", color="#2e7d32", lw=1.3))
axA.set_xlabel("k = how many fixed 28×28 images the linear recipe may mix", fontsize=10)
axA.set_ylabel("share of image variation still missed\n(log scale; lower is better)", fontsize=10)
axA.set_title("The ceiling on every linear map, measured without fitting one\nSVD of the 20,000 renders: the best flat k-dimensional slab that could hold this family",
              fontsize=10.8, pad=9, linespacing=1.5)
axA.set_ylim(1e-7, 1.5); axA.set_xlim(1, 800); axA.grid(alpha=0.3, which="both")

# ---- bottom right: shrinking patch
axB = fig.add_axes([0.575, 0.080, 0.370, 0.225])
sc, r2s = d["scales"], d["r2s"]
axB.loglog(sc*100, 1-r2s, "o-", color="#1f4e79", ms=6)
axB.axhline(1e-6, color="#d81b8c", ls="--", lw=1.4)
axB.text(90, 1.4e-6, "99.9999% accuracy", fontsize=9, color="#d81b8c", va="bottom", ha="left")
axB.invert_xaxis()
for x,y in zip(sc*100, 1-r2s):
    lab = f"R²={1-y:.6f}" if y > 1e-5 else f"R²={1-y:.8f}"
    axB.annotate(lab, (x,y), textcoords="offset points",
                 xytext=(-9,-3) if x<=1 else (7,7), fontsize=7.8,
                 ha="right" if x<=1 else "left", va="center" if x<=1 else "bottom")
axB.set_xlabel("width of the knob patch the map is fitted and tested on\n(% of each knob's full default range; →  right = tiny patch around root 1)", fontsize=10)
axB.set_ylabel("1 − R² on held-out renders\n(log scale; lower is better)", fontsize=10)
axB.set_title("The same fit, on smaller and smaller patches of knob space\nA linear map becomes exact only as the patch shrinks to a point — that is a tangent plane",
              fontsize=10.8, pad=9, linespacing=1.5)
axB.grid(alpha=0.3, which="both"); axB.set_ylim(1e-8, 2)
axB.annotate("whole family", xy=(100, 1-r2s[0]), xytext=(60, 2e-3), fontsize=9, color="#555",
             arrowprops=dict(arrowstyle="->", color="#555", lw=1.1))

fig.savefig(OUT+"linear_map_attempt.png", dpi=125, facecolor="white")
print("saved 1")

# ================= FIGURE 2 =================
f2 = plt.figure(figsize=(16.0, 13.6))
f2.text(0.5, 0.977, "Why no linear map can do it: the renderer breaks both rules a linear map must obey",
        ha="center", fontsize=16)
f2.text(0.5, 0.957, "Every panel below comes straight from the generated-1 renderer — nothing is fitted or learned here",
        ha="center", fontsize=11.5, color="#333")

H2 = 0.098
# --- Rule 1
Y = 0.780
f2.text(0.045, 0.925, "Rule 1 — adding two knob moves must add their images.", fontsize=12.5, fontweight="bold")
f2.text(0.045, 0.902, "Start at root 1. Move the knobs to a; separately, move them to b. A linear map must place the combined move a + b − root exactly on A + B − P.",
        fontsize=10.2)
fa,fb,fp,fc = d["fa"],d["fb"],d["fp"],d["fc"]; pred = fa+fb-fp
names = ["A = render(a)", "B = render(b)", "P = render(root 1)",
         "A + B − P\nwhat a linear map must give", "render(a + b − root)\nwhat the renderer gives"]
caps = [f"a = ({', '.join(f'{x:g}' for x in d['a'])})", f"b = ({', '.join(f'{x:g}' for x in d['b'])})",
        "the shared starting image", "", f"a + b − root = ({', '.join(f'{x:g}' for x in d['c'])})"]
for i,(nm,img) in enumerate(zip(names,[fa,fb,fp,pred,fc])):
    a = ink(f2, [0.045 + i*0.148, Y, 0.078, H2], img, nm, 9.5)
    if i==3:
        nn,_ = mark_outside(a, img); caps[3] = f"{nn} pixels of negative ink (cyan)"
    cap(a, caps[i], 7.8)
a, v = diff(f2, [0.045+5*0.148, Y, 0.078, H2], pred-fc, None, "mismatch\nbetween the two", 9.5)
rel = np.linalg.norm(pred-fc)/np.linalg.norm(fc-fp)
cap(a, f"red = linear map predicts too much ink\nworst pixel {np.abs(pred-fc).max():.2f} coverage", 7.8)
f2.text(0.045, 0.728,
 f"Measured:  the mismatch is {rel*100:.0f}% as large as the move it is supposed to describe.  Over 2,000 random knob pairs drawn from the full box: median 85%, mean 90%, worst 208%.\n"
 "A linear map would score 0% on every single pair. This is the defining property of linearity, and the renderer fails it almost everywhere in the box.",
 fontsize=10.2, va="top", linespacing=1.7, color="#9b0000")

# --- Rule 2
Y = 0.495
f2.text(0.045, 0.655, "Rule 2 — one knob must always paint the same pattern.", fontsize=12.5, fontweight="bold")
f2.text(0.045, 0.637, "In the recipe on the first figure, turning 'lean' always adds one fixed image, just scaled. Here is the identical +11.25° lean step, taken from two different starting 1s.",
        fontsize=10.2)
d1, d2 = d["d1"], d["d2"]; vb = max(np.abs(d1).max(), np.abs(d2).max())
for i,(bp, dd, lab) in enumerate([(d["base1"], d1, "thin stroke, width 1.9"), (d["base2"], d2, "wide stroke, width 4.5")]):
    x0 = 0.045 + i*0.320
    f2.text(x0, 0.614, f"start {i+1}:  {lab}", fontsize=10.5, fontweight="bold", color="#1f4e79")
    a1 = ink(f2, [x0, Y, 0.078, H2], _r([bp],dtype=np.float64)[0], "before: lean 0°", 9.5)
    a2 = ink(f2, [x0+0.094, Y, 0.078, H2], _r([bp+[0,0,0,0,11.25]],dtype=np.float64)[0], "after: lean +11.25°", 9.5)
    a3,_ = diff(f2, [x0+0.188, Y, 0.078, H2], dd, vb, "what that one step did", 9.5)
    cap(a3, f"common scale ±{vb:.2f}\nred = ink gained", 7.6)
a,v = diff(f2, [0.045+2*0.320, Y, 0.078, H2], d1-d2, None, "start 1's pattern\nminus start 2's", 9.5)
cap(a, "a change in the pixel-change pattern itself,\nnot ink gained by any one image", 7.6)
f2.text(0.045, 0.452,
 f"Measured:  the cosine between the two effect patterns is {float(d['cos12']):.2f}  (1.00 would mean the same direction, which is what linearity requires).\n"
 "Across 300 random starting settings per knob, the median cosine between a knob's own effect patterns is 0.17 – 0.30 for all five knobs, and some pairs are negative:\n"
 "a knob's 'column' is not a fixed column at all — it is a function of where the other knobs already are.",
 fontsize=10.2, va="top", linespacing=1.7, color="#9b0000")

# --- bottom: curves + formula
axc = f2.add_axes([0.058, 0.065, 0.315, 0.215])
cols = ["#1f4e79", "#c0392b", "#2e7d32"]
for i,(r,c) in enumerate(d["pixs"]):
    axc.plot(d["leans"], d["curves"][i], color=cols[i], lw=2.2, label=f"pixel at row {r}, col {c}")
axc.set_xlabel("lean (degrees);  cx, cy, height, width held at 14.5, 14.5, 19.75, 3.2", fontsize=9.5)
axc.set_ylabel("that pixel's ink coverage", fontsize=10)
axc.set_title("Follow three single pixels as one knob turns\nBends and corners — a linear map forces all 784 of these curves to be straight lines",
              fontsize=10.8, pad=9, linespacing=1.5)
axc.grid(alpha=0.3); axc.legend(fontsize=9, loc="upper left"); axc.set_ylim(-0.04, 1.12)
axc.axhline(0, color="#999", lw=0.8)

f2.text(0.445, 0.305, "Where the bends come from, in the renderer's own formula", fontsize=12.5, fontweight="bold")
f2.text(0.445, 0.283,
 "Each pixel's coverage is\n"
 "    ink = mean over 16 sub-rows of   clip( min(R, col+1) − max(L, col), 0, 1 ) × vertical overlap\n"
 "    where   L, R  =  cx − tan(lean)·(y − cy)  ∓  width/2\n\n"
 "Three things in there can never be linear in (cx, cy, height, width, lean):\n"
 "   •  tan(lean)                     a curve, not a straight line\n"
 "   •  tan(lean)·cy ,  ∓ width/2     products of knobs; a linear recipe has no product terms\n"
 "   •  clip , min , max              corners: the slope jumps when a stroke edge crosses a pixel border\n\n"
 "The corners are why a merely curved fit does not rescue it either. Same data, same least squares, but\n"
 "feeding polynomial combinations of the knobs as the inputs (all scored on held-out renders):\n"
 "        degree 1   (6 inputs)     R² = 0.563           degree 3   (56 inputs)    R² = 0.884\n"
 "        degree 2   (21 inputs)    R² = 0.788           degree 6   (462 inputs)   R² = 0.968\n\n"
 "Still far from 99.9999%, because no single global polynomial has corners. An exact 5-knob decoder does\n"
 "exist for this renderer, but it is piecewise: a different quadratic formula in each region of knob space.",
 fontsize=9.9, va="top", family="monospace", linespacing=1.65)

f2.savefig(OUT+"linear_map_why_not.png", dpi=125, facecolor="white")
print("saved 2")
