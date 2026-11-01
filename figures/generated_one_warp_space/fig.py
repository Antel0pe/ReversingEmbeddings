import sys, pathlib
_HERE = pathlib.Path(__file__).resolve().parent
sys.path[:0] = [str(_HERE.parents[1]), str(_HERE)]
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import map_coordinates
from warp2 import render_at, make_basis, ROOT_P, IDENT
SC_DIR = str(_HERE)

N=112; SCL=N/28.0; B=make_basis(N)
def warp(src,th,order=1):
    x0=(14+14*np.einsum('k,kij->ij',th[:6],B))*SCL-0.5
    y0=(14+14*np.einsum('k,kij->ij',th[6:],B))*SCL-0.5
    return map_coordinates(src,[y0,x0],order=order,mode="constant",cval=0.0)

d=np.load(f"{SC_DIR}/w4.npz"); TH,TX,P=d["TH"],d["TX"],d["P"]
root=render_at(ROOT_P[None],N)[0]; I=render_at(P,N); F=I.reshape(len(P),-1)
tr,te=slice(0,60),slice(60,120)

def rankk(T,k):
    mu=T[tr].mean(0); A=T[tr]-mu
    Vt=np.linalg.svd(A,full_matrices=False)[2]
    Tk=mu+(T[te]-mu)@Vt[:k].T@Vt[:k]
    R=np.stack([warp(root,t) for t in Tk]).reshape(len(Tk),-1)
    return np.linalg.norm(R-F[te])/np.linalg.norm(F[te])
mp=F[tr].mean(0); Vtp=np.linalg.svd(F[tr]-mp,full_matrices=False)[2]
def rankk_pix(k):
    R=mp+(F[te]-mp)@Vtp[:k].T@Vtp[:k]
    return np.linalg.norm(R-F[te])/np.linalg.norm(F[te])
ks=np.arange(1,11)
e_ex=[rankk(TX,k) for k in ks]; e_fit=[rankk(TH,k) for k in ks]; e_pix=[rankk_pix(k) for k in ks]
floor=np.linalg.norm(np.stack([warp(root,t) for t in TX]).reshape(len(TX),-1)-F)/np.linalg.norm(F)

s_ex=np.linalg.svd(TX-TX.mean(0),compute_uv=False); s_ex/=s_ex[0]
s_fit=np.linalg.svd(TH-TH.mean(0),compute_uv=False); s_fit/=s_fit[0]
s_pix=np.linalg.svd(F-F.mean(0),compute_uv=False); s_pix/=s_pix[0]
EPS=1e-18

# ============================ FIGURE 1 ============================
fig=plt.figure(figsize=(17.5,13.0))
gs=GridSpec(3,6,figure=fig,height_ratios=[1.0,1.25,1.0],hspace=0.40,wspace=0.42,
            left=0.055,right=0.965,top=0.885,bottom=0.055)
fig.suptitle("Generated-1 family (synthetic parallelogram strokes, NOT real MNIST):\n"
             "the same five knobs are 5 flat directions when you describe WHERE ink moves,\n"
             "but need >50 directions when you describe HOW MUCH ink each pixel has",
             fontsize=15.5,fontweight="bold",y=0.975)
j=te.start+int(np.argmax(P[te,4])); tgt=I[j]   # strongest lean in the test half
ax=fig.add_subplot(gs[0,0]); ax.imshow(root,cmap="gray_r",vmin=0,vmax=1)
ax.set_title("A1. BASELINE: root 1\n(14, 14, 19.75, 3.2, 0°)",fontsize=10.5)
ax.set_xlabel("one square = one pixel\ndark = ink coverage 1, white = 0",fontsize=8.5)
ax.set_xticks([]); ax.set_yticks([])
ax=fig.add_subplot(gs[0,1]); ax.imshow(tgt,cmap="gray_r",vmin=0,vmax=1)
ax.set_title(f"A2. ONE TARGET 1\n({P[j,0]:.1f}, {P[j,1]:.1f}, {P[j,2]:.1f}, "
             f"{P[j,3]:.1f}, {P[j,4]:.0f}°)",fontsize=10.5)
ax.set_xlabel("all five knobs differ\nfrom the baseline",fontsize=8.5)
ax.set_xticks([]); ax.set_yticks([])
ax=fig.add_subplot(gs[0,2])
g=np.linspace(0,28,8); th=TX[j]
for gy in g:
    xs=np.linspace(0,28,40); Xn=(xs-14)/14; Yn=(gy-14)/14
    ax.plot(xs,np.full_like(xs,gy),color="0.72",lw=1.3)
for gx in g:
    ys=np.linspace(0,28,40)
    ax.plot(np.full_like(ys,gx),ys,color="0.72",lw=1.3)
inv=np.linalg.inv(np.array([[th[1],th[2]],[th[7],th[8]]]))
for gy in g:
    xs=np.linspace(0,28,40); XY=np.stack([(xs-14)/14,np.full_like(xs,(gy-14)/14)])
    out=inv@(XY-np.array([[th[0]],[th[6]]])); ax.plot(14+14*out[0],14+14*out[1],color="#1f77b4",lw=1.5)
for gx in g:
    ys=np.linspace(0,28,40); XY=np.stack([np.full_like(ys,(gx-14)/14),(ys-14)/14])
    out=inv@(XY-np.array([[th[0]],[th[6]]])); ax.plot(14+14*out[0],14+14*out[1],color="#1f77b4",lw=1.5)
ax.set_xlim(0,28); ax.set_ylim(28,0); ax.set_aspect("equal")
ax.set_title("A3. THE OPERATION\nA1 → A2 is one affine map\nof the PLANE",fontsize=10.5)
ax.set_xlabel("grey = original pixel grid\nblue = where it is sent",fontsize=8.5)
ax.set_xticks([]); ax.set_yticks([])
ax=fig.add_subplot(gs[0,3]); v=np.abs(tgt-root).max()
im=ax.imshow(tgt-root,cmap="RdBu_r",vmin=-v,vmax=v)
ax.set_title("A4. A2 minus A1,\nper pixel",fontsize=10.5)
ax.set_xlabel("red = target has MORE ink\nblue = target has LESS ink",fontsize=8.5)
ax.set_xticks([]); ax.set_yticks([])
plt.colorbar(im,ax=ax,fraction=0.046,label="coverage difference")
ax=fig.add_subplot(gs[0,4:6]); ax.axis("off")
ax.text(0.0,0.97,"Two ways to write down one image",fontsize=11.5,fontweight="bold",va="top")
ax.text(0.0,0.84,
 "PIXEL description (all earlier work)\n"
 "  784 numbers = how much ink sits\n"
 "  at each fixed place.  Panel A4 is\n"
 "  this description of the change.\n\n"
 "WARP description (tested here)\n"
 "  12 numbers = a smooth map saying\n"
 "  where each place came from:\n"
 "   x_src = a0+a1x+a2y+a3x\u00b2+a4xy+a5y\u00b2\n"
 "   y_src = b0+b1x+b2y+b3x\u00b2+b4xy+b5y\u00b2\n"
 "  Panel A3 is this description of\n"
 "  the SAME change.\n\n"
 "  Nothing affine was assumed: the 6\n"
 "  quadratic terms were free to be\n"
 "  used, and the fit had to discover\n"
 "  on its own that they are not needed.",
 fontsize=8.7,va="top",family="monospace",linespacing=1.45)
ax=fig.add_subplot(gs[1,0:3])
ax.semilogy(np.arange(1,13),np.maximum(s_ex,EPS),"o-",color="#2a9d4a",lw=2.2,ms=7,
            label="warp coefficients, exact (12 numbers/image)")
ax.semilogy(np.arange(1,13),np.maximum(s_fit,EPS),"s--",color="#1f77b4",lw=1.8,ms=6,
            label="warp coefficients, fitted from pixels only")
ax.semilogy(np.arange(1,61),s_pix[:60],"^-",color="#c23b22",lw=1.6,ms=4,
            label="pixel vectors (12544 numbers/image)")
ax.axvline(5,color="0.4",ls=":",lw=1.6)
ax.text(5.4,2e-6,"5 knobs",fontsize=10,color="0.3",rotation=90,va="bottom")
ax.annotate("exact warp coefficients fall to\nmachine zero (1e-17) after #5:\nthe family is a 5-dimensional\nFLAT subspace here",
            xy=(6,1e-16),xytext=(12,1e-13),fontsize=9.2,color="#2a9d4a",
            arrowprops=dict(arrowstyle="->",color="#2a9d4a",lw=1.4))
ax.annotate("pixel spectrum never falls:\n50 directions still leave 8% error",
            xy=(40,s_pix[39]),xytext=(26,2e-7),fontsize=9.2,color="#c23b22",
            arrowprops=dict(arrowstyle="->",color="#c23b22",lw=1.4))
ax.set_xlabel("component number (sorted largest first)",fontsize=10)
ax.set_ylabel("singular value ÷ largest singular value",fontsize=10)
ax.set_title("B. How many straight directions does each description need?\n"
             "120 settings drawn from the whole default knob box",fontsize=11.5)
ax.set_ylim(1e-18,3); ax.set_xlim(0,60); ax.legend(fontsize=9,loc="center right")
ax.grid(alpha=0.3,which="both")
ax=fig.add_subplot(gs[1,3:6])
ax.plot(ks,e_ex,"o-",color="#2a9d4a",lw=2.2,ms=7,label="warp space, exact coefficients")
ax.plot(ks,e_fit,"s--",color="#1f77b4",lw=1.8,ms=6,label="warp space, fitted from pixels")
ax.plot(ks,e_pix,"^-",color="#c23b22",lw=1.8,ms=6,label="pixel space (PCA)")
ax.axhline(floor,color="0.45",ls="-.",lw=1.5)
ax.text(1.15,0.015,f"floor = {floor:.3f}: cost of resampling a 28-px-wide\n"
        "picture through the warp, not model error",fontsize=8.8,color="0.35")
ax.axvline(5,color="0.4",ls=":",lw=1.6)
ax.annotate(f"at 5 directions:\nwarp {e_ex[4]:.3f}  vs  pixels {e_pix[4]:.3f}\n"
            f"({e_pix[4]/e_ex[4]:.1f}x better)",xy=(5,e_pix[4]),xytext=(5.9,0.56),
            fontsize=9.6,arrowprops=dict(arrowstyle="->",color="0.35",lw=1.3))
ax.set_xlabel("number of directions kept (k)",fontsize=10)
ax.set_ylabel("held-out image error\n(‖reconstruction − truth‖ ÷ ‖truth‖)",fontsize=10)
ax.set_title("C. Rebuilding 60 UNSEEN 1s from k directions\n"
             "fitted on the other 60 (same images, same metric, both curves)",fontsize=11.5)
ax.set_ylim(0,0.82); ax.set_xticks(ks); ax.legend(fontsize=9); ax.grid(alpha=0.3)
mu=TX[tr].mean(0); Vt5=np.linalg.svd(TX[tr]-mu,full_matrices=False)[2][:5]
t5=mu+(TX[j]-mu)@Vt5.T@Vt5; rec_w=warp(root,t5)
rec_p=(mp+(F[j]-mp)@Vtp[:5].T@Vtp[:5]).reshape(N,N)
panels=[(tgt,"D1. truth (unseen 1)","gray_r",None),
        (rec_w,"D2. warp space, 5 directions","gray_r",None),
        (rec_w-tgt,"D3. D2 error","RdBu_r",None),
        (rec_p,"D4. pixel space, 5 directions","gray_r",None),
        (rec_p-tgt,"D5. D4 error","RdBu_r",None)]
vd=max(np.abs(rec_w-tgt).max(),np.abs(rec_p-tgt).max())
for c,(img,ttl,cm,_) in enumerate(panels):
    ax=fig.add_subplot(gs[2,c])
    if cm=="gray_r": im=ax.imshow(img,cmap=cm,vmin=0,vmax=1)
    else: im=ax.imshow(img,cmap=cm,vmin=-vd,vmax=vd)
    err=np.linalg.norm(img-tgt)/np.linalg.norm(tgt) if c in (1,3) else None
    ax.set_title(ttl+(f"\nerror {err:.3f}" if err is not None else ""),fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    if c in (2,4):
        ax.set_xlabel("red = too much ink\nblue = too little",fontsize=8.2)
        plt.colorbar(im,ax=ax,fraction=0.046)
    elif c==0: ax.set_xlabel("this one image, not an average",fontsize=8.2)
ax=fig.add_subplot(gs[2,5]); ax.axis("off")
ax.text(0.0,0.95,"D3 vs D5 is the whole result.\n\n"
 "Both reconstructions use exactly\n5 numbers per image.\n\n"
 "The pixel version (D5) cannot put\nink in the right place, so it smears\n"
 "a faint average stroke and leaves\nlarge errors along both edges.\n\n"
 "The warp version (D3) leaves only\na thin resampling rim.\n\n"
 "Panels D2/D4 share the ink scale\n(0–1); D3/D5 share the error scale.",
 fontsize=9.2,va="top",linespacing=1.5)
fig.savefig(f"{SC_DIR}/warp_vs_pixels.png",dpi=135,facecolor="white")
print("fig1 saved")
print(f"e_ex5={e_ex[4]:.4f} e_fit5={e_fit[4]:.4f} e_pix5={e_pix[4]:.4f} floor={floor:.4f}")
